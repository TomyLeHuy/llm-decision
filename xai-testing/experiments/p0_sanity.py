"""Phase 0 sanity checks.

  python experiments/p0_sanity.py s1            round trip of the state string (no API calls)
  python experiments/p0_sanity.py s2 [--dry]    drift: 30 stratified test calls x 5 models vs. stored scores
  python experiments/p0_sanity.py s3 [--dry]    noise floor: 10 calls x 5 models x 3 fresh repeats
  python experiments/p0_sanity.py s4            cost / runtime estimate for the Phase B pilot and Phase C (no API calls)
Outputs in results/p0_*.
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config as C  # noqa: E402
from src import io as xio  # noqa: E402
from src import units as U  # noqa: E402
from src import query as Q  # noqa: E402

R = C.RESULTS
R.mkdir(exist_ok=True)


def stratified(meta, n, keys, seed):
    """Draw n rows as evenly as possible over the cells of `keys` (fixed seed); small cells are taken whole and the
    remainder is spread over the larger cells."""
    rng = np.random.default_rng(seed)
    cells = {k: g.dataset_idx.to_numpy() for k, g in meta.groupby(keys)}
    quota = {k: 0 for k in cells}
    left = n
    while left > 0:
        open_cells = [k for k in sorted(cells) if quota[k] < len(cells[k])]
        if not open_cells:
            break
        for k in open_cells:
            if left == 0:
                break
            quota[k] += 1
            left -= 1
    out = []
    for k in sorted(cells):
        ids = cells[k].copy()
        rng.shuffle(ids)
        out += ids[: quota[k]].tolist()
    return sorted(out)


# ------------------------------------------------------------------------------------------------- S1
def s1():
    rows, summary = [], {}
    for split in ["test", "train", "mailu"]:
        data = xio.load_split(split)
        n_mis = n_nofmt = n_mask_bad = n_units = 0
        fmts, unit_counts, payload_diff = Counter(), Counter(), 0
        for i, s in enumerate(data):
            url0, p0 = Q.captured_payload(s)
            obj, fmt, units = U.parse(s)
            fmts[fmt] += 1
            if fmt is None:
                n_nofmt += 1
                continue
            url1, p1 = Q.captured_payload(U.variant(s))
            same_state = p1["state"] == p0["state"]
            same_rest = (url0 == url1) and {k: v for k, v in p0.items() if k != "state"} == \
                        {k: v for k, v in p1.items() if k != "state"}
            n_mis += not same_state
            payload_diff += not same_rest
            unit_counts[len(units)] += 1
            # every M1 (remove) variant must be valid JSON and differ from the baseline only in that unit
            for u in units:
                n_units += 1
                c = U.scored_content(U.variant(s, u, "remove"))
                try:
                    ok = json.loads(c) == U.apply_mask(obj, u, "remove") and c != U.scored_content(s)
                except Exception:
                    ok = False
                n_mask_bad += not ok
            if split == "test":
                rows.append(dict(dataset_idx=i, fmt=fmt, n_units=len(units), state_identical=same_state,
                                 state_len=len(p0["state"])))
        summary[split] = dict(n=len(data), state_mismatches=n_mis, other_payload_mismatches=payload_diff,
                              no_exact_serialiser=n_nofmt, formats=dict(fmts), remove_variants=n_units,
                              remove_variants_invalid=n_mask_bad,
                              units_per_sample=dict(sorted(unit_counts.items())))
        print(split, json.dumps(summary[split]))

    # truncation facts on test: scored call vs labelled call, shared scored inputs
    data = xio.load_split("test")
    st = xio.sample_table("test")
    first_last_same = 0
    for i in st[st.multi_turn].dataset_idx:
        users = [json.loads(m["content"]) for m in data[i]["prompt"] if m["role"] == "user"]
        first_last_same += users[0].get(U.CALL_KEY) == users[-1].get(U.CALL_KEY)
    states = {i: Q.state_string(s) for i, s in enumerate(data)}
    groups = Counter(states.values())
    st["state_shared"] = [groups[states[i]] > 1 for i in st.dataset_idx]
    # conflicting labels for an identical scored input
    lab = {}
    for i in st.dataset_idx:
        lab.setdefault(states[i], set()).add(st.ground_truth[i])
    st["state_label_conflict"] = [len(lab[states[i]]) > 1 for i in st.dataset_idx]
    summary["truncation_test"] = dict(
        multi_turn=int(st.multi_turn.sum()),
        multi_turn_first_call_equals_last_call=int(first_last_same),
        multi_turn_scored_tool_differs=int((st.multi_turn & (st.scored_tool != st.labelled_tool)).sum()),
        distinct_scored_states=len(groups), samples_sharing_scored_state=int(st.state_shared.sum()),
        samples_sharing_by_multi_turn=st.groupby("multi_turn").state_shared.sum().astype(int).to_dict(),
        samples_with_conflicting_label_for_same_scored_state=int(st.state_label_conflict.sum()),
        conflicting_by_multi_turn=st.groupby("multi_turn").state_label_conflict.sum().astype(int).to_dict(),
    )
    summary["truncation_test"] = {k: ({str(a): b for a, b in v.items()} if isinstance(v, dict) else v)
                                  for k, v in summary["truncation_test"].items()}
    print("truncation", json.dumps(summary["truncation_test"]))
    pd.DataFrame(rows).to_csv(R / "p0_s1_roundtrip_test.csv", index=False)
    st[["dataset_idx", "multi_turn", "state_shared", "state_label_conflict"]].to_csv(R / "p0_s1_shared_states_test.csv", index=False)
    json.dump(summary, open(R / "p0_s1_summary.json", "w"), indent=1)


# ------------------------------------------------------------------------------------------------- S2
def s2_ids():
    _, meta = xio.prob_matrices("test")
    return stratified(meta, C.S2_PER_MODEL, ["ground_truth", "multi_turn"], C.SEED)


def s2(dry):
    ids = s2_ids()
    data = xio.load_split("test")
    jobs = [dict(model=m, sample=U.variant(data[i]), tag="p0_s2", meta=dict(split="test", dataset_idx=int(i)))
            for i in ids for m in C.MODELS]
    res = Q.Querier().run(jobs, dry_run=dry)
    if dry:
        return
    old = xio.load_scores("test").set_index(["dataset_idx", "model"])
    st = xio.sample_table("test").set_index("dataset_idx")
    rows = []
    for j, r in zip(jobs, res):
        i, m = j["meta"]["dataset_idx"], j["model"]
        o = old.loc[(i, m)]
        p = r["probabilities"] if r else {}
        rows.append(dict(dataset_idx=i, model=C.SHORT[m], ground_truth=st.ground_truth[i], multi_turn=st.multi_turn[i],
                         new_choice=r and r["choice"], old_choice=o.pred_choice,
                         **{f"new_{c}": p.get(c) for c in C.CLASSES}, **{f"old_{c}": o[f"p_{c}"] for c in C.CLASSES},
                         new_response_model=r and r["response_model"], old_response_model=o.get("response_model"),
                         new_cost=r and r["cost_usd"], old_cost=o.cost_usd, failed=r is None))
    d = pd.DataFrame(rows)
    for c in C.CLASSES:
        d[f"absdiff_{c}"] = (d[f"new_{c}"] - d[f"old_{c}"]).abs()
    d["same_choice"] = d.new_choice == d.old_choice
    d["max_absdiff"] = d[[f"absdiff_{c}" for c in C.CLASSES]].max(axis=1)
    d.to_csv(R / "p0_s2_drift.csv", index=False)
    agg = dict(mad_execute=("absdiff_direct_execute", "mean"), mad_ask=("absdiff_ask_human", "mean"),
               mad_refuse=("absdiff_refuse", "mean"), max_absdiff=("max_absdiff", "max"),
               share_exact_same_vector=("max_absdiff", lambda x: float(np.mean(x < 1e-9))),
               same_choice=("same_choice", "mean"), n=("dataset_idx", "size"))
    out = {"by_model": d.groupby("model").agg(**agg).round(4).reset_index().to_dict("records"),
           "by_model_multi_turn": d.groupby(["model", "multi_turn"]).agg(**agg).round(4).reset_index().to_dict("records"),
           "response_models": d.groupby("model").apply(
               lambda g: dict(new=sorted(set(map(str, g.new_response_model))), old=sorted(set(map(str, g.old_response_model)))),
               include_groups=False).to_dict(),
           "cost": dict(new=float(d.new_cost.sum()), old_same_calls=float(d.old_cost.sum())),
           "ids": [int(i) for i in ids], "seed": C.SEED, "strata": d.drop_duplicates("dataset_idx").groupby(
               ["ground_truth", "multi_turn"]).size().rename("n").reset_index().astype(str).to_dict("records")}
    # train-run versions of the three original models (the October test run did not record versions)
    tr = xio.load_scores("train")
    out["train_run_response_models"] = tr.groupby("short").response_model.agg(lambda x: sorted(set(map(str, x.dropna())))).to_dict()
    json.dump(out, open(R / "p0_s2_summary.json", "w"), indent=1, default=str)
    print(pd.DataFrame(out["by_model"]).to_string())
    print(pd.DataFrame(out["by_model_multi_turn"]).to_string())
    print(json.dumps(out["response_models"], indent=1))


# ------------------------------------------------------------------------------------------------- S3
def s3_ids():
    _, meta = xio.prob_matrices("test")
    m = meta[meta.dataset_idx.isin(s2_ids())]  # a subset of the S2 calls, so S2 gives a 4th draw
    return stratified(m, C.S3_N_CALLS, ["ground_truth", "multi_turn"], C.SEED)


def s3(dry):
    ids = s3_ids()
    data = xio.load_split("test")
    jobs = [dict(model=m, sample=U.variant(data[i]), tag="p0_s3", repeat=r, meta=dict(split="test", dataset_idx=int(i)))
            for r in range(C.S3_REPEATS) for i in ids for m in C.MODELS]
    res = Q.Querier().run(jobs, dry_run=dry)
    if dry:
        return
    st = xio.sample_table("test").set_index("dataset_idx")
    rows = [dict(dataset_idx=j["meta"]["dataset_idx"], model=C.SHORT[j["model"]], repeat=j["repeat"], choice=r["choice"],
                 multi_turn=st.multi_turn[j["meta"]["dataset_idx"]], ground_truth=st.ground_truth[j["meta"]["dataset_idx"]],
                 response_model=r["response_model"], **{f"p_{c}": r["probabilities"][c] for c in C.CLASSES})
            for j, r in zip(jobs, res) if r]
    d = pd.DataFrame(rows)
    d.to_csv(R / "p0_s3_repeats.csv", index=False)
    g = d.groupby(["model", "dataset_idx"])
    sd = g[C.PCOLS].std(ddof=1)
    rng_ = g[C.PCOLS].max() - g[C.PCOLS].min()
    flips = g.choice.nunique().gt(1)
    per = sd.join(rng_, rsuffix="_range").join(flips.rename("choice_varies")).reset_index()
    per = per.merge(st[["multi_turn"]], left_on="dataset_idx", right_index=True)
    per.to_csv(R / "p0_s3_per_call.csv", index=False)
    agg = {f"sd_{c}_{s}": (f"p_{c}", s) for c in C.CLASSES for s in ("mean", "max")}
    agg.update(max_range_any=("p_refuse_range", "max"), calls_with_any_change=("p_refuse_range", lambda x: 0),
               choice_varies=("choice_varies", "sum"), n_calls=("dataset_idx", "size"))
    tab = per.groupby("model").agg(**agg)
    tab["calls_with_any_change"] = per.assign(
        ch=per[[f"p_{c}_range" for c in C.CLASSES]].max(axis=1) > 1e-12).groupby("model").ch.sum()
    tab["max_range_any"] = per[[f"p_{c}_range" for c in C.CLASSES]].max(axis=1).groupby(per.model).max()
    print(tab.round(5).to_string())
    pooled = {c: float(np.sqrt((per[f"p_{c}"] ** 2).mean())) for c in C.CLASSES}  # RMS of within-call SD
    pooled_by_model = {m: {c: float(np.sqrt((g_[f"p_{c}"] ** 2).mean())) for c in C.CLASSES} for m, g_ in per.groupby("model")}
    out = dict(ids=[int(i) for i in ids], seed=C.SEED, repeats=C.S3_REPEATS, table=tab.round(6).reset_index().to_dict("records"),
               rms_within_sd_all=pooled, rms_within_sd_by_model=pooled_by_model,
               by_multi_turn=per.groupby(["model", "multi_turn"])[C.PCOLS].mean().round(6).reset_index().to_dict("records"))
    # 4th draw: the S2 result for the same calls (a different run minutes earlier)
    if (R / "p0_s2_drift.csv").exists():
        s2d = pd.read_csv(R / "p0_s2_drift.csv")
        s2d = s2d[s2d.dataset_idx.isin(ids)][["dataset_idx", "model"] + [f"new_{c}" for c in C.CLASSES]]
        m4 = d.merge(s2d, on=["dataset_idx", "model"])
        out["max_absdiff_vs_s2_run"] = {m: float(max((g_[f"p_{c}"] - g_[f"new_{c}"]).abs().max() for c in C.CLASSES))
                                        for m, g_ in m4.groupby("model")}
        print("max |repeat - S2| per model:", out["max_absdiff_vs_s2_run"])
    json.dump(out, open(R / "p0_s3_summary.json", "w"), indent=1, default=str)
    print("RMS within-call SD:", {k: round(v, 5) for k, v in pooled.items()})


# ------------------------------------------------------------------------------------------------- S4
def pilot_ids():
    """Phase B pilot sample (drawn here only to estimate the cost; NOT queried in Phase 0)."""
    _, meta = xio.prob_matrices("test")
    return stratified(meta, 50, ["ground_truth", "multi_turn", "service_type"], C.SEED)


def s4():
    data = xio.load_split("test")
    old = xio.load_scores("test")
    oldc = old.set_index(["dataset_idx", "model"]).cost_usd
    oldlat = old.groupby("model").latency_ms.median()
    st = xio.sample_table("test").set_index("dataset_idx")
    # observed cost ratio new/old from S2 (price change check)
    s2d = pd.read_csv(R / "p0_s2_drift.csv")
    ratio = (s2d.groupby("model").new_cost.sum() / s2d.groupby("model").old_cost.sum()).to_dict()
    inv = {v: k for k, v in C.SHORT.items()}

    def est(ids, n_strat):
        calls = cost = 0.0
        per_group = {}
        for i in ids:
            n_u = len(U.parse(data[i])[2])
            n_calls = 1 + n_strat * n_u
            mt = bool(st.multi_turn[i])
            for m in C.MODELS:
                c = oldc[(i, m)] * ratio.get(C.SHORT[m], 1.0) * n_calls  # masked inputs are shorter: an upper bound for M1
                calls += n_calls
                cost += c
                pg = per_group.setdefault("multi_turn" if mt else "single_turn", [0, 0.0, 0])
                pg[0] += n_calls
                pg[1] += c
            per_group["multi_turn" if mt else "single_turn"][2] += 1
        lat = float(oldlat.max()) / 1000  # seconds per call, slowest model
        return dict(n_samples=len(ids), calls=int(calls), cost_usd=round(cost, 4),
                    runtime_min_4_workers=round(calls * lat / 4 / 60, 1),
                    by_group={k: dict(samples=v[2], calls=int(v[0]), cost_usd=round(v[1], 4)) for k, v in per_group.items()})

    pids = pilot_ids()
    _, meta = xio.prob_matrices("test")
    out = dict(cost_ratio_new_over_old=ratio, pilot_ids=pids, pilot_seed=C.SEED,
               pilot_strata=meta[meta.dataset_idx.isin(pids)].groupby(["ground_truth", "multi_turn", "service_type"]).size()
               .rename("n").reset_index().astype(str).to_dict("records"),
               pilot_both_strategies=est(pids, 2), full_one_strategy=est(meta.dataset_idx.tolist(), 1),
               full_both_strategies=est(meta.dataset_idx.tolist(), 2), spent_so_far=Q.spent_usd(),
               spent_by_tag={k: dict(calls=v[0], usd=round(v[1], 5)) for k, v in Q.spent_by_tag().items()})
    json.dump(out, open(R / "p0_s4_estimate.json", "w"), indent=1, default=str)
    print(json.dumps({k: v for k, v in out.items() if k != "pilot_ids"}, indent=1, default=str))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["s1", "s2", "s3", "s4"])
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    {"s1": s1, "s4": s4}.get(a.step, lambda: None)() if a.step in ("s1", "s4") else {"s2": s2, "s3": s3}[a.step](a.dry)
