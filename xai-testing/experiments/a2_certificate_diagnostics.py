"""A2 certificate diagnostics per model x service, on stored scores; no API calls.

For each (scorer, service) with at least one true refusal, separately for the test pool and the train pool:
  n_rf, n_ask, n_exec; shares of refusals at p_refuse = 0 and <= 0.01; AUROC of p_refuse (refusals vs all other calls
  of the service); and three certification checks that use ALL labelled calls of the pool as calibration data:
  - g5_sep   : the paper's G5 per-service harm test at level delta/(2m) (= 'per service separately', needs >= 102
               refusals), inside the ask-certified region of the pooled data (G5 step A on all calls of the pool).
  - g5_sim4  : the same at delta/(2m*4) (the paper's four simultaneous groups, needs >= 129).
  - g4_pool  : the paper's G4 on the whole pool (one boundary for all services; needs >= 88 refusals in total);
               reported per service as the harm rate inside the selected region.
  binding lever (first that applies): ask_floor (no region passes the pooled ask test), sample_size
  (n_rf < n_min), resolution_floor (the strictest admissible region already holds > k_max refusals), certifies.
Everything is also reported for single-turn calls only and for multi-turn calls only (shares and AUROC; the
certification check on single-turn calls only is a what-if, NOT a fixed pipeline).
The paper's own G5 groups (atbench, agentharm, tsbench, other) are evaluated the same way.
Outputs: results/a2_*.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binom, mannwhitneyu

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config as C  # noqa: E402
from src import io as xio  # noqa: E402
from src import gate as G  # noqa: E402

R = xio.results_dir()  # results/ for the paper scores, results/<XAI_SOURCE>/ otherwise
DE, AH, RF = G.DE, G.AH, G.RF
SCORERS = ["jev", "microsoft", "gpt", "liquid", "pplx", "mean"]
M = len(G.LA)
LV = {"g5_sep": C.DELTA / 2 / M, "g5_sim4": C.DELTA / 2 / M / 4}
NMIN = {k: G.n_min(v) for k, v in LV.items()} | {"g4": G.n_min(C.DELTA / M)}


def auroc(pos, neg):
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    return float(mannwhitneyu(pos, neg, alternative="two-sided").statistic / (len(pos) * len(neg)))


def ask_region(P, y):
    """G5 step A on the pool: for each la the largest lr whose ask risk is certified (fixed sequence)."""
    n_ah = int((y == AH).sum())
    out = {}
    for la in G.LA:
        ok = None
        for lr in G.LR:
            ex = (P[:, RF] <= lr) & (P[:, AH] <= la)
            if binom.cdf(int(ex[y == AH].sum()), n_ah, C.A_ASK) <= C.DELTA / 2 / M:
                ok = lr
            else:
                break
        out[la] = ok
    return out


def harm_check(P, y, mask, region, level):
    """Best certified (la, lr) for the service's refusals within the ask-certified region; plus diagnostics."""
    n_rf = int(((y == RF) & mask).sum())
    km = G.kmax(n_rf, level)
    best = None
    for la, lr_max in region.items():
        if lr_max is None:
            continue
        for lr in G.LR:
            if lr > lr_max:
                break
            ex = mask & (P[:, RF] <= lr) & (P[:, AH] <= la)
            k = int(ex[y == RF].sum())
            if binom.cdf(k, n_rf, C.A_HARM) <= level:
                safe = float(np.mean(ex[mask & (y == DE)])) if (mask & (y == DE)).any() else np.nan
                if best is None or (not np.isnan(safe) and safe > best[2]):
                    best = (la, lr, safe, k)
            else:
                break
    if all(v is None for v in region.values()):
        lever = "ask_floor"
    elif n_rf < G.n_min(level):
        lever = "sample_size"
    elif best is None:
        lever = "resolution_floor"
    else:
        lever = "certifies"
    # strictest admissible region: lr = 0 and the smallest la with a certified ask region
    la0 = next((la for la, v in region.items() if v is not None), None)
    k_strict = int((mask & (P[:, RF] <= 0) & (P[:, AH] <= la0))[y == RF].sum()) if la0 is not None else None
    extra = G.extra_needed(k_strict, n_rf, level) if (k_strict is not None and lever != "certifies") else 0
    return dict(certifies=best is not None, lever=lever, k_max=km, k_strict=k_strict, extra_clean_refusals=extra,
                safe_autonomy=best[2] if best else 0.0, la=best[0] if best else None, lr=best[1] if best else None)


def diagnose(split):
    P, meta = xio.prob_matrices(split)
    y = xio.label_index(meta.ground_truth)
    svc = meta.service.to_numpy()
    mt = meta.multi_turn.to_numpy()
    tmeta = xio.prob_matrices("test", source="paper")[1]  # the paper defines the G5 groups on the test pool; reused for train
    _, groups = G.paper_groups(tmeta.service.to_numpy(), xio.label_index(tmeta.ground_truth))
    big = [g for g in groups if g != "other"]
    pgrp = np.array([s if s in big else "other" for s in svc])
    rows = []
    units = [(s, svc == s) for s in sorted(set(svc[y == RF]))] + [(f"G5:{g}", pgrp == g) for g in groups]
    for name in SCORERS:
        Pm = P[name]
        for subset, sub in [("all", np.ones(len(y), bool)), ("single_turn", ~mt), ("multi_turn", mt)]:
            region = ask_region(Pm[sub], y[sub]) if subset != "multi_turn" else None
            for s, msk in units:
                m = msk & sub
                n_rf = int(((y == RF) & m).sum())
                if n_rf == 0:
                    continue
                prf = Pm[(y == RF) & m, RF]
                r = dict(split=split, scorer=name, subset=subset, service=s, n_rf=n_rf, n_ask=int(((y == AH) & m).sum()),
                         n_exec=int(((y == DE) & m).sum()), rf_p0=float(np.mean(prf == 0)),
                         rf_le001=float(np.mean(prf <= 0.01)), auroc=auroc(prf, Pm[(y != RF) & m, RF]),
                         auroc_vs_exec=auroc(prf, Pm[(y == DE) & m, RF]))
                if region is not None:
                    for k, lv in LV.items():
                        hc = harm_check(Pm[sub], y[sub], m[sub], region, lv)
                        r.update({f"{k}_{a}": b for a, b in hc.items()})
                rows.append(r)
    return pd.DataFrame(rows)


def g4_pool(split):
    """G4 on the whole pool and the per-service harm rate inside its region (one boundary for all services)."""
    P, meta = xio.prob_matrices(split)
    y = xio.label_index(meta.ground_truth)
    svc = meta.service.to_numpy()
    rows = []
    for name in SCORERS:
        b = G.g4_select(P[name], y, G.g4_candidates(P[name], y))
        if b is None:
            rows.append(dict(split=split, scorer=name, certifies=False))
            continue
        a = G.pol(P[name], b[1], b[2], b[3])
        r = dict(split=split, scorer=name, certifies=True, lr=b[1], la=b[2], tr=b[3],
                 safe=float(np.mean(a[y == DE] == DE)), harm=float(np.mean(a[y == RF] == DE)))
        for s in sorted(set(svc[y == RF])):
            m = (svc == s) & (y == RF)
            if m.sum() >= 10:
                r[f"harm_{s}"] = float(np.mean(a[m] == DE))
        rows.append(r)
    return pd.DataFrame(rows)


def verbal(r, split):
    if r.g5_sep_lever == "certifies":
        return f"certifiable per service ({r.n_rf} refusals; safe autonomy {r.g5_sep_safe_autonomy:.0%})"
    if r.g5_sep_lever == "ask_floor":
        return "no region passes the pooled ask test: the ask-budget floor binds before the service is examined"
    if r.g5_sep_lever == "sample_size":
        sep = f"{r.n_rf} labelled refusals, {NMIN['g5_sep']} needed: sample size binds"
        if r.rf_p0 > 0 or r.rf_le001 > 0.05:
            sep += f"; with more data the floor would bind next ({r.rf_le001:.0%} of refusals at p <= 0.01)"
        return sep
    return (f"enough refusals ({r.n_rf}), but the strictest region already executes {r.g5_sep_k_strict} "
            f"(max {r.g5_sep_k_max}): resolution floor binds ({r.rf_p0:.0%} at p = 0, {r.rf_le001:.0%} at p <= 0.01, "
            f"AUROC {r.auroc:.2f}); {r.g5_sep_extra_clean_refusals} more clean refusals would certify the strictest region")


def main():
    d = pd.concat([diagnose("test"), diagnose("train")], ignore_index=True)
    d["why"] = [verbal(r, r.split) if r.subset != "multi_turn" else "" for r in d.itertuples()]
    d.to_csv(R / "a2_model_service.csv", index=False)
    pool = pd.concat([g4_pool("test"), g4_pool("train")], ignore_index=True)
    pool.to_csv(R / "a2_g4_pool.csv", index=False)
    # compact tables
    a = d[d.subset == "all"]
    piv = a.pivot_table(index=["service"], columns=["split", "scorer"], values="g5_sep_lever", aggfunc="first")
    piv.to_csv(R / "a2_lever_table.csv")
    lever_counts = a.groupby(["split", "g5_sep_lever"]).size().unstack(fill_value=0)
    print("binding lever counts (per-service, delta/(2m)):\n", lever_counts)
    print("\nlevers by service (test | train):\n", piv.to_string())
    turn = d[d.subset != "all"].pivot_table(index=["split", "service"], columns=["subset", "scorer"],
                                            values=["rf_le001", "auroc"], aggfunc="first")
    turn.round(3).to_csv(R / "a2_turn_split.csv")
    sel = d[(d.split == "test") & d.service.isin(["atbench", "dh_app"]) & (d.subset != "all")]
    print("\natbench / dh_app by turn (test):\n", sel[["service", "subset", "scorer", "n_rf", "rf_p0", "rf_le001", "auroc"]]
          .round(3).to_string())
    print("\nG4 on whole pool:\n", pool.round(3).to_string())
    res = dict(n_min=NMIN, lever_counts={f"{a_}|{b_}": int(v) for (a_, b_), v in lever_counts.stack().items()},
               groups=a[a.service.str.startswith("G5:")][["split", "scorer", "service", "n_rf", "g5_sim4_lever",
                                                           "g5_sim4_certifies", "g5_sim4_safe_autonomy",
                                                           "g5_sim4_extra_clean_refusals"]].to_dict("records"))
    json.dump(res, open(R / "a2_summary.json", "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
