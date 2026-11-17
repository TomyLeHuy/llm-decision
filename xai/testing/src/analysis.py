"""Checks, the Shapley table and plots, all from the cache (no API calls).

  .venv/bin/python src/analysis.py            # checks + results/shapley_values.csv + results/checks.json
  .venv/bin/python src/analysis.py --plots    # also the PNGs in plots/
  .venv/bin/python src/analysis.py --model pplx --fixed tool --plots   # other models: results/<slug>/, plots/<slug>/

Checks: round trip, determinism (results/determinism.json), agreement of the full coalition with the October scores,
efficiency of llmSHAP's values, an independent exact Shapley computation from the cached 2^n coalition values,
leave-one-out from the same cache and its rank correlation with Shapley.
"""
import argparse
import itertools
import json
from collections import Counter
from math import factorial

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from adapter import aggregate, cache_key, failure_type, load_cache, payload_for
from common import (CLASSES, DATA_PATH, MODEL, SUBSET_PATH, ModelPaths, make_state, parse_fixed, resolve_model, ss,
                    variant_suffix)
from fields import ALL_FIELDS, APTC_FIELDS, CallFields


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def exact_shapley(players, v):
    """v: dict frozenset -> float. Returns {player: phi}."""
    n = len(players)
    w = {k: factorial(k) * factorial(n - k - 1) / factorial(n) for k in range(n)}
    phi = {}
    for p in players:
        others = [q for q in players if q != p]
        phi[p] = sum(w[k] * (v[frozenset(S) | {p}] - v[frozenset(S)])
                     for k in range(n) for S in itertools.combinations(others, k))
    return phi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plots", action="store_true")
    ap.add_argument("--fixed", help="same as run_shapley.py --fixed (fields kept in every coalition)")
    ap.add_argument("--model", default=MODEL, help="model id or slug (gpt, jev, microsoft, liquid, pplx)")
    a = ap.parse_args()
    fixed = parse_fixed(a.fixed)
    sfx = variant_suffix(fixed)
    mp = ModelPaths(resolve_model(a.model))
    model, RESULTS, repeats = mp.model, mp.results, mp.repeats()

    data = json.load(open(DATA_PATH, encoding="utf-8"))
    sub = json.load(open(SUBSET_PATH))
    label_of = {c["dataset_idx"]: c["label"] for c in sub["calls"]}
    answers = load_cache(mp.cache)                                  # key -> raw answers (file order)
    cache = {k: agg for k, v in answers.items() if (agg := aggregate(v, repeats)) is not None}
    raw = {}
    for r in load_jsonl(mp.raw(sfx)):
        raw[(r["dataset_idx"], r["class"])] = r                      # latest record wins
    ids = sorted({i for i, _ in raw}, key=lambda i: [c["dataset_idx"] for c in sub["calls"]].index(i))

    oct_ = pd.read_csv(mp.oct)
    oct_ = oct_[oct_.model == model].set_index("dataset_idx")
    oct_failed = sorted(int(i) for i in ids if i not in oct_.index or pd.notna(oct_.loc[i, "error"]))

    rows, per_call, refusal_sizes, refused_calls, refusal_pattern = [], [], Counter(), [], Counter()
    eff_err, indep_err, vend_err, oct_diff, oct_diff_mean = [], [], [], [], []
    noise = {c: [] for c in CLASSES}
    for i in ids:
        c = CallFields(data[i])                                       # asserts the round trip again
        assert ss.build_user_content(data[i]["prompt"]) == c.user_content(c.players), "scorer input differs"
        players = [p for p in c.players if p not in fixed]
        fx = tuple(p for p in c.players if p in fixed)
        n = len(players)
        recs = {}
        for k in range(n + 1):
            for S in itertools.combinations(players, k):
                key = cache_key(payload_for(make_state(c.system, c.user_content(S + fx)), model))
                assert key in cache, f"call {i}: coalition {S} not in cache"
                recs[frozenset(S)] = cache[key]
        refused = {S: r for S, r in recs.items() if r["status"] != "ok"}
        undefined = bool(refused) or i in oct_failed                 # October failure: undefined for this model
        full = recs[frozenset(players)]
        if full["status"] == "ok":
            oct_diff.append(max(abs(full["probabilities"][k] - oct_.loc[i, f"p_{k}"]) for k in CLASSES))
            oct_diff_mean.extend(abs(full["probabilities"][k] - oct_.loc[i, f"p_{k}"]) for k in CLASSES)
        info = {"dataset_idx": i, "label": label_of[i], "n_fields": len(c.players), "players": players, "fixed": fx,
                "n_refused_coalitions": len(refused),
                "refused_coalitions": [sorted(S, key=players.index) for S in refused],
                "full_probs": full.get("probabilities"), "pred_full": full.get("choice")}
        per_call.append(info)
        if refused:
            refused_calls.append(i)
            refusal_sizes.update(len(S) for S in refused)
            for S in refused:   # does the refused request still contain any tool-call field (incl. fixed ones)?
                refusal_pattern["with_tool_call_field" if (set(S) | set(fx)) & set(APTC_FIELDS)
                                else "without_any_tool_call_field"] += 1
        for cls in CLASSES:
            r = raw[(i, cls)]
            v = None if undefined else {S: rec["probabilities"][cls] for S, rec in recs.items()}
            if v is not None and repeats > 1:
                noise[cls].extend(rec["std"][cls] for rec in recs.values())
            if v is not None:
                phi_ind = exact_shapley(players, v)
                full_set = frozenset(players)
                loo = {p: v[full_set] - v[full_set - {p}] for p in players}
                eff_err.append(abs(r["v_empty"] + sum(r["phi"].values()) - r["v_full"]))
                indep_err.append(max(abs(phi_ind[p] - r["phi"][p]) for p in players))
                vend_err.append(max(abs(r["v_empty"] - v[frozenset()]), abs(r["v_full"] - v[full_set])))
            for p in ALL_FIELDS:
                if p not in players:
                    continue
                rows.append({"dataset_idx": i, "label": label_of[i], "n_fields": len(c.players), "field": p, "class": cls,
                             "phi": np.nan if undefined else r["phi"][p], "loo": np.nan if undefined else loo[p],
                             "v_empty": np.nan if undefined else r["v_empty"],
                             "v_full": np.nan if undefined else r["v_full"],
                             "status": "undefined_provider_refusal" if refused else
                                       "undefined_october_failure" if undefined else "ok"})

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / f"shapley_values{sfx}.csv", index=False)
    ok = df[df.status == "ok"]

    rank = {}
    for cls in CLASSES:
        d = ok[ok["class"] == cls]
        rho = spearmanr(d.phi, d.loo).statistic if len(d) > 2 else None
        per = [spearmanr(g.phi, g.loo).statistic for _, g in d.groupby("dataset_idx")
               if g.phi.nunique() > 1 and g.loo.nunique() > 1]
        same_top = np.mean([g.loc[g.phi.abs().idxmax(), "field"] == g.loc[g.loo.abs().idxmax(), "field"]
                            for _, g in d.groupby("dataset_idx")]) if len(d) else None
        rank[cls] = {"pooled_spearman": rho, "median_per_call_spearman": float(np.median(per)) if per else None,
                     "n_calls_per_call_rho": len(per), "share_same_top_field_by_abs": same_top}

    det = json.load(open(mp.determinism)) if mp.determinism.exists() else []
    probe = det if isinstance(det, dict) else None                  # new models: probe_determinism.py summary
    if probe is not None:
        det = []
    statuses = Counter(r["status"] for r in cache.values())
    checks = {
        "model": model, "fixed_fields": fixed,
        "n_calls": len(ids), "n_complete": len(ids) - len(set(refused_calls) | set(oct_failed)),
        "round_trip": {"byte_identical_full_coalition": True, "equals_score_split_build_user_content": True,
                       "n_calls": len(ids)},
        "determinism": {"n_calls": len(det), "max_abs_diff": max((d["max_abs_diff"] for d in det
                                                                  if d["max_abs_diff"] is not None), default=None),
                        "detail": det},
        "agreement_with_october": {"n_calls": len(oct_diff), "max_abs_diff": max(oct_diff, default=None)},
        "efficiency_max_abs_error": max(eff_err, default=None),
        "independent_shapley_max_abs_diff": max(indep_err, default=None),
        "llmshap_v_empty_v_full_vs_cache_max_abs_diff": max(vend_err, default=None),
        "leave_one_out_vs_shapley": rank,
        "provider_refusals": {"n_calls_affected": len(refused_calls), "calls": refused_calls,
                              "refused_coalitions_by_size": dict(sorted(refusal_sizes.items())),
                              "refused_coalitions_by_tool_call_content": dict(refusal_pattern),
                              "rule": "no imputation: a call with any refused coalition has undefined phi",
                              "cache_status_counts": dict(statuses)},
        "per_call": per_call,
    }
    # ---- additions for the five-model comparison (appended, so earlier keys keep their content) ----
    ftypes = {}
    for v in answers.values():
        for x in v:
            ft = failure_type(x)
            if ft:
                ftypes.setdefault(ft, {"count": 0, "example": x.get("example_body") or x.get("error")})
                ftypes[ft]["count"] += 1
    checks["repeats"] = repeats
    checks["agreement_with_october_mean_abs_diff"] = float(np.mean(oct_diff_mean)) if oct_diff_mean else None
    checks["october_failed_calls_undefined"] = oct_failed
    checks["failure_types_in_cache"] = ftypes
    if probe is not None:
        checks["determinism_probe"] = {k: probe[k] for k in ("max_abs_diff", "mean_std_per_class_and_input",
                                                              "deterministic", "n_queries", "failure_types")}
    if repeats > 1:
        ok_ = df[df.status == "ok"]
        nz = {}
        for cls in CLASSES:
            lvl = float(np.mean(noise[cls])) if noise[cls] else None
            ph = ok_[ok_["class"] == cls].phi.abs()
            nz[cls] = {"mean_std_single_answer": lvl,
                       "mean_std_of_payoff_mean": lvl / np.sqrt(repeats) if lvl is not None else None,
                       "share_abs_phi_below_mean_std": float((ph < lvl).mean()) if lvl is not None and len(ph) else None,
                       "share_abs_phi_below_std_of_mean": float((ph < lvl / np.sqrt(repeats)).mean())
                       if lvl is not None and len(ph) else None}
        allstd = [x for c in CLASSES for x in noise[c]]
        checks["noise"] = {"repeats": repeats, "mean_std_all_classes": float(np.mean(allstd)) if allstd else None,
                           "per_class": nz}
    if not fixed:
        empties, notool = [], Counter()
        for pc in per_call:
            c = CallFields(data[pc["dataset_idx"]])
            for k in range(len(c.players) + 1):
                for S in itertools.combinations(c.players, k):
                    if set(S) & set(APTC_FIELDS):
                        continue
                    st = cache[cache_key(payload_for(make_state(c.system, c.user_content(S)), model))]["status"]
                    notool[st] += 1
                    if not S:
                        empties.append({"dataset_idx": pc["dataset_idx"], "status": st})
        checks["seven_player_pilot"] = {
            "empty_coalition": empties, "coalitions_without_tool_call_field": dict(notool),
            "defined_for_all_calls": all(pc["n_refused_coalitions"] == 0 for pc in per_call)}
    if not fixed and model == MODEL:
        checks["finding_7_player_design"] = (
            "With all 7 fields as players, gpt refuses (HTTP 502 'OpenAI refused to answer question') the empty "
            "coalition of every pilot call and other coalitions that contain no tool-call field (tool, tool_args, "
            "description); v(empty) is therefore undefined and no call has Shapley values. Evaluated on the 3 pilot "
            "calls only; the designs with tool / the whole tool call fixed are the ones run on all 30 calls.")
    if "finding_7_player_design" in checks:          # keep the gpt pilot's original key order: added keys go last
        first = list(checks)[:list(checks).index("per_call") + 1] + ["finding_7_player_design"]
        checks = {**{k: checks[k] for k in first}, **{k: v for k, v in checks.items() if k not in first}}
    json.dump(checks, open(RESULTS / f"checks{sfx}.json", "w"), indent=2, default=float)
    print(f"model: {model} (repeats {repeats}) | variant: {'7-player design' if not fixed else 'fixed ' + str(fixed)}")

    print(f"calls: {len(ids)} | complete: {checks['n_complete']} | undefined (provider refusal): {refused_calls}")
    print(f"refused coalitions by size: {dict(sorted(refusal_sizes.items()))} | by tool-call content: "
          f"{dict(refusal_pattern)} | cache statuses: {dict(statuses)}")
    print(f"round trip: OK for {len(ids)} calls (byte-identical; equals score_split.build_user_content)")
    if probe is not None:
        print(f"determinism probe: {checks['determinism_probe']}")
    else:
        print(f"determinism: {[(d['dataset_idx'], d['max_abs_diff']) for d in det]}")
    print(f"agreement with October scores: max abs diff {checks['agreement_with_october']['max_abs_diff']} | mean "
          f"{checks['agreement_with_october_mean_abs_diff']} over {len(oct_diff)} calls | October failures {oct_failed}")
    print(f"failure types in cache: {({k: v['count'] for k, v in ftypes.items()})}")
    if "noise" in checks:
        print(f"noise: {json.dumps(checks['noise'])}")
    if "seven_player_pilot" in checks:
        print(f"7-player: {json.dumps(checks['seven_player_pilot'])}")
    print(f"efficiency max abs error: {checks['efficiency_max_abs_error']}")
    print(f"independent exact Shapley vs llmSHAP: max abs diff {checks['independent_shapley_max_abs_diff']}")
    print(f"llmSHAP v_empty/v_full vs cache: max abs diff {checks['llmshap_v_empty_v_full_vs_cache_max_abs_diff']}")
    for cls, r in rank.items():
        print(f"LOO vs Shapley [{cls}]: pooled rho {r['pooled_spearman']} | median per-call rho "
              f"{r['median_per_call_spearman']} (n={r['n_calls_per_call_rho']}) | same top |.| field "
              f"{r['share_same_top_field_by_abs']}")
    for pc in per_call:
        fp = pc["full_probs"] or {}
        print(f"  {pc['dataset_idx']:>5} {pc['label']:<15} n={pc['n_fields']} full=" +
              " ".join(f"{k[:3]}={fp.get(k, float('nan')):.3f}" for k in CLASSES) +
              f" refused={pc['n_refused_coalitions']}")
    if a.plots:
        import plots
        plots.make_all(df, sfx, mp.plots, mp.slug)


if __name__ == "__main__":
    main()
