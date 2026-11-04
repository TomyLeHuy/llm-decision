"""Load splits and scores, add multi_turn, join labels. Read-only access to the paper package."""
import json
import os
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C  # noqa: E402


# the nine self-hosted services come from these two sources; agentharm, atbench, r_judge, tsbench, when2call are benchmarks
ENTERPRISE_SOURCES = {"internal_v2", "v3_internal"}


@lru_cache(maxsize=None)
def load_split(split):
    with open(C.SPLITS[split], encoding="utf-8") as f:
        return json.load(f)


def n_user_turns(sample):
    return sum(m.get("role") == "user" for m in sample.get("prompt", []))


def sample_table(split):
    """One row per sample: dataset_idx, label, service, risk_class, risk_family, multi_turn, n_user_turns,
    scored tool (first user message) and labelled tool (last user message), and the annotator reasoning."""
    rows = []
    for i, s in enumerate(load_split(split)):
        users = [m["content"] for m in s["prompt"] if m.get("role") == "user"]
        first, last = json.loads(users[0]), json.loads(users[-1])
        meta = s.get("meta", {})
        comp = s.get("completion") or [{}]
        rows.append(dict(
            dataset_idx=i, ground_truth=meta.get("decision"), service=meta.get("service", "unknown"),
            risk_class=meta.get("risk_class") or "", risk_family=meta.get("risk_family") or "",
            source=meta.get("source", ""),
            service_type="enterprise" if meta.get("source") in ENTERPRISE_SOURCES else "benchmark",
            n_user_turns=len(users), multi_turn=len(users) > 1,
            scored_tool=first.get("assistant_proposed_tool_call", {}).get("tool"),
            labelled_tool=last.get("assistant_proposed_tool_call", {}).get("tool"),
            annotator=comp[-1].get("content", "") if comp else "",
        ))
    return pd.DataFrame(rows)


def is_failed(r):
    """Same rule as score_split.is_failed: error, missing/negative probability, or sum not within 0.02 of 1."""
    if r.get("error"):
        return True
    v = [r.get(k) for k in C.PCOLS]
    if any(x is None or (isinstance(x, float) and x != x) for x in v):
        return True
    return min(v) < 0 or abs(sum(v) - 1) > 0.02


RUNS = C.XAI / "runs"
SOURCES = ("paper", "today", "today_fixed", "mt_truncated", "mt_full", "mt_users_only", "mt_final_only")


def default_source():
    """score source for analyses; set with the environment variable XAI_SOURCE (default 'paper')"""
    s = os.environ.get("XAI_SOURCE", "paper")
    assert s in SOURCES, s
    return s


def results_dir(source=None):
    source = source or default_source()
    d = C.RESULTS if source == "paper" else C.RESULTS / source
    d.mkdir(parents=True, exist_ok=True)
    return d


def _read_records(path):
    recs = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                recs[(r["dataset_idx"], r["model"])] = r
    return recs


@lru_cache(maxsize=None)
def load_scores(split, source=None):
    """Long table, last record per (dataset_idx, model), with 'failed', short model name and multi_turn.
    paper       : the stored scores of the paper package (October test run; 23/27 Oct train run)
    today       : test = runs/today/test_scores.jsonl (all calls, re-scored today, truncated input as in the paper);
                  train = stored single-turn scores + runs/today/train_multiturn_scores.jsonl (multi-turn, today, truncated)
    today_fixed : as 'today', but every multi-turn call scored with the full conversation (runs/fixed_multiturn/)"""
    source = source or default_source()
    recs = _read_records(C.SCORES[split])
    st = sample_table(split)[["dataset_idx", "multi_turn", "n_user_turns"]]
    mt_ids = set(st.dataset_idx[st.multi_turn])
    if source != "paper":
        if split == "test":
            recs = _read_records(RUNS / "today/test_scores.jsonl")
        else:
            recs = {k: v for k, v in recs.items() if k[0] not in mt_ids}
            recs.update(_read_records(RUNS / f"today/{split}_multiturn_scores.jsonl"))
        if source.startswith("mt_"):  # Step 5b: multi-turn calls from runs/multiturn_variants/<variant> (3-repeat means)
            fixed = _read_records(RUNS / f"multiturn_variants/{source[3:]}/{split}_scores.jsonl")
            assert {k[0] for k in fixed} == mt_ids, "variant run must cover exactly the multi-turn calls"
            recs = {k: v for k, v in recs.items() if k[0] not in mt_ids}
            recs.update(fixed)
        if source == "today_fixed":
            fixed = _read_records(RUNS / f"fixed_multiturn/{split}_scores.jsonl")
            assert {k[0] for k in fixed} == mt_ids, "fixed run must cover exactly the multi-turn calls"
            recs = {k: v for k, v in recs.items() if k[0] not in mt_ids}
            recs.update(fixed)
    d = pd.DataFrame(list(recs.values()))
    d["failed"] = [is_failed(r) for r in recs.values()]
    d["outcome"] = [("provider_refusal" if "refused to answer" in str(r.get("error") or "") else
                     "error" if is_failed(r) else "ok") for r in recs.values()]
    d["short"] = d.model.map(C.SHORT)
    return d.merge(st, on="dataset_idx", how="left")


def common_ids(split, source=None):
    """dataset_idx values scored without failure by all five models (test/paper: the paper's 1,433). For other
    sources the set is intersected with the paper's set, so that before/after comparisons use the same calls."""
    d = load_scores(split, source)
    ok = d[~d.failed]
    sets = [set(g.dataset_idx) for _, g in ok.groupby("model")]
    assert len(sets) == len(C.MODELS)
    ids = set.intersection(*sets)
    if (source or default_source()) != "paper":
        ids &= set(common_ids(split, "paper"))
    return np.array(sorted(ids))


def prob_matrices(split, ids=None, source=None):
    """{short: (N,3) array}, clipped at 0 and renormalised exactly as the paper's analysis (mat() in steps 02-06),
    plus 'mean' (jev + microsoft + gpt) and the sample table aligned to ids."""
    d = load_scores(split, source)
    d = d[~d.failed]
    ids = common_ids(split, source) if ids is None else np.asarray(ids)
    out = {}
    for m in C.MODELS:
        g = d[d.model == m].set_index("dataset_idx").loc[ids, C.PCOLS].to_numpy(float)
        g = np.clip(g, 0, None)
        out[C.SHORT[m]] = g / g.sum(1, keepdims=True)
    out["mean"] = np.mean([out["jev"], out["microsoft"], out["gpt"]], 0)
    meta = sample_table(split).set_index("dataset_idx").loc[ids].reset_index()
    return out, meta


def label_index(labels):
    return np.array([C.CLASSES.index(x) for x in labels])


def jsonable(o):
    """recursively turn tuple/other keys into strings ('a|b') so that json.dump never fails on groupby output"""
    if isinstance(o, dict):
        return {(k if isinstance(k, str) else "|".join(map(str, k)) if isinstance(k, tuple) else str(k)): jsonable(v)
                for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    return o


def dump_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(jsonable(obj), f, indent=1, default=str)
