"""A3 influence of single calibration calls on the G4 certificate (leave-one-out), on stored scores; no API calls.

Setting: the paper's G4 default (alpha_harm 5%, alpha_ask 15%, delta 10%, m = 9 ask thresholds, harm cost 10), test
pool, 50/50 split with the paper's seed 0 (first of the 60 G4 splits), scorers jev, liquid, pplx.
Utilities of the selected policy:
  U_test  safe autonomy on the fixed test half (the "certified safe-autonomy rate" a deployment would get)
  U_cal   safe autonomy on the calibration half
  slack   log10(delta/m) - log10(p), p = LTT p-value of the selected (lr, la); > 0 means certified, larger is safer
LOO: every calibration call is removed in turn (all refusals, and ALL other calibration calls; the plan's random 200
are a subset, flagged in the output), G4 is re-certified and re-selected, and the change in each utility is recorded.
Influence = utility(full) - utility(without the call): positive = the call helps the certificate.
Boundaries: for each la, the first lr at which the fixed sequence stopped; the number of extra clean calibration
refusals (outside the region) or asks needed to pass that test, and the test-half safe autonomy if it passed.
Outputs: results/a3_*.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binom

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config as C  # noqa: E402
from src import io as xio  # noqa: E402
from src import gate as G  # noqa: E402

R = xio.results_dir()  # results/ for the paper scores, results/<XAI_SOURCE>/ otherwise
DE, AH, RF = G.DE, G.AH, G.RF
LV = C.DELTA / len(G.LA)
# a change of the test-half safe autonomy below 1 point is at most two of the 270 safe test calls; for liquid such
# changes come from an exact cost tie (up to float rounding) between la = 0.2 and la = 0.5 that any removal flips
MATERIAL = 0.01


def fit(Pc, yc, Pt, yt):
    cand = G.g4_candidates(Pc, yc)
    b = G.g4_select(Pc, yc, cand)
    if b is None:
        return dict(found=False, U_test=0.0, U_cal=0.0, slack=-np.inf, n_cand=0, lr=None, la=None, tr=None)
    _, lr, la, tr = b
    pv = G.ltt_pvalue(Pc, yc, lr, la)
    a_t, a_c = G.pol(Pt, lr, la, tr), G.pol(Pc, lr, la, tr)
    return dict(found=True, U_test=float(np.mean(a_t[yt == DE] == DE)), U_cal=float(np.mean(a_c[yc == DE] == DE)),
                slack=math.log10(LV) - math.log10(max(pv["p"], 1e-300)), n_cand=len(cand), lr=lr, la=la, tr=tr,
                harm_test=float(np.mean(a_t[yt == RF] == DE)), unconf_test=float(np.mean(a_t[yt == AH] == DE)),
                k_rf=pv["k_rf"], k_ah=pv["k_ah"], p_rf=pv["p_rf"], p_ah=pv["p_ah"])


def boundaries(Pc, yc, Pt, yt):
    rows = []
    n_rf, n_ah = int((yc == RF).sum()), int((yc == AH).sum())
    for la in G.LA:
        last = None
        for lr in G.LR:
            pv = G.ltt_pvalue(Pc, yc, lr, la)
            if pv["p_rf"] <= LV and pv["p_ah"] <= LV:
                last = lr
                continue
            ex_t = (Pt[:, RF] <= lr) & (Pt[:, AH] <= la)
            rows.append(dict(la=la, last_certified_lr=last, failing_lr=lr, k_rf=pv["k_rf"], n_rf=n_rf, k_ah=pv["k_ah"],
                             n_ah=n_ah, harm_fails=pv["p_rf"] > LV, ask_fails=pv["p_ah"] > LV,
                             extra_clean_refusals=G.extra_needed(pv["k_rf"], n_rf, LV, C.A_HARM) if pv["p_rf"] > LV else 0,
                             extra_clean_asks=G.extra_needed(pv["k_ah"], n_ah, LV, C.A_ASK) if pv["p_ah"] > LV else 0,
                             safe_test_if_passed=float(np.mean(ex_t[yt == DE])),
                             safe_test_at_last=float(np.mean(((Pt[:, RF] <= last) & (Pt[:, AH] <= la))[yt == DE]))
                             if last is not None else 0.0))
            break
    return rows


def main():
    P, meta = xio.prob_matrices("test")
    y = xio.label_index(meta.ground_truth)
    cal = G.split(y, C.A3_SPLIT_SEED)
    ci = np.flatnonzero(cal)
    a1 = pd.read_csv(R / "a1_flags_test.csv").set_index("dataset_idx")
    rng = np.random.default_rng(C.SEED)
    others = ci[y[ci] != RF]
    rand200 = set(rng.choice(others, size=min(C.A3_N_OTHER, len(others)), replace=False).tolist())
    rows, base_rows, bnd_rows = [], [], []
    for name in C.A3_MODELS:
        Pm = P[name]
        Pt, yt = Pm[~cal], y[~cal]
        base = fit(Pm[cal], y[cal], Pt, yt)
        base_rows.append(dict(model=name, **base))
        bnd_rows += [dict(model=name, **b) for b in boundaries(Pm[cal], y[cal], Pt, yt)]
        for j in ci:
            keep = cal.copy(); keep[j] = False
            r = fit(Pm[keep], y[keep], Pt, yt)
            di = int(meta.dataset_idx[j])
            ex = (Pm[j, RF] <= base["lr"]) & (Pm[j, AH] <= base["la"]) if base["found"] else False
            rows.append(dict(model=name, dataset_idx=di, label=C.CLASSES[y[j]], service=meta.service[j],
                             multi_turn=bool(meta.multi_turn[j]), in_random200=j in rand200,
                             in_selected_region=bool(ex), p_refuse=Pm[j, RF], p_ask=Pm[j, AH],
                             a1_n_flags=int(a1.n_flags[di]), a1_suggested=a1.suggested[di],
                             dU_test=base["U_test"] - r["U_test"], dU_cal=base["U_cal"] - r["U_cal"],
                             dslack=base["slack"] - r["slack"], policy_changed=(r["lr"], r["la"], r["tr"]) != (base["lr"], base["la"], base["tr"]),
                             new_lr=r["lr"], new_la=r["la"], new_tr=r["tr"], loo_U_test=r["U_test"]))
    d = pd.DataFrame(rows)
    d.to_csv(R / "a3_loo.csv", index=False)
    base = pd.DataFrame(base_rows); base.to_csv(R / "a3_baseline.csv", index=False)
    bnd = pd.DataFrame(bnd_rows); bnd.to_csv(R / "a3_boundaries.csv", index=False)
    print("baseline (split seed 0):\n", base.round(4).to_string())
    print("\nfailing boundaries:\n", bnd.round(3).to_string())

    res = dict(baseline=base.to_dict("records"), split_seed=C.A3_SPLIT_SEED, sample_seed=C.SEED,
               n_cal=int(cal.sum()), n_cal_refusals=int((y[cal] == RF).sum()))
    summ = []
    for (m, lab), g in d.assign(grp=np.where(d.label == "refuse", "refuse", "other")).groupby(["model", "grp"]):
        for mt, h in [("all", g), ("single_turn", g[~g.multi_turn]), ("multi_turn", g[g.multi_turn])]:
            if lab == "other" and mt == "all":
                h = h[h.in_random200]  # plan's random 200
            summ.append(dict(model=m, group=lab + ("" if lab == "refuse" or mt != "all" else " (random 200)"), subset=mt,
                             n=len(h), policy_changed=int(h.policy_changed.sum()),
                             material_change=int((h.dU_test.abs() >= MATERIAL).sum()),
                             boundary_holders=int((h.dU_test <= -MATERIAL).sum()),
                             dU_test_nonzero=int((h.dU_test.abs() > 1e-12).sum()),
                             dU_test_min=h.dU_test.min(), dU_test_max=h.dU_test.max(),
                             dslack_mean=h.dslack.mean(), dslack_min=h.dslack.min(), dslack_max=h.dslack.max(),
                             in_region=int(h.in_selected_region.sum())))
    summ = pd.DataFrame(summ)
    summ.to_csv(R / "a3_summary_table.csv", index=False)
    print("\nLOO distribution:\n", summ.round(4).to_string())
    # top 20 most influential per model: |dU_test| first, then |dslack|
    tops = []
    for m, g in d.groupby("model"):
        t = g.assign(a=g.dU_test.abs(), b=g.dslack.abs()).sort_values(["a", "b"], ascending=False).head(20)
        tops.append(t.drop(columns=["a", "b"]))
    top = pd.concat(tops)
    top.to_csv(R / "a3_top20.csv", index=False)
    holders = d[d.dU_test <= -MATERIAL]
    holders.to_csv(R / "a3_boundary_holders.csv", index=False)
    res["boundary_holders"] = holders[["model", "dataset_idx", "service", "label", "multi_turn", "p_refuse", "p_ask",
                                       "a1_n_flags", "dU_test", "new_lr", "new_la", "loo_U_test"]].to_dict("records")
    print("\nboundary holders (removing the call raises test safe autonomy by >= 1 point):\n",
          holders[["model", "dataset_idx", "service", "multi_turn", "p_refuse", "p_ask", "a1_n_flags", "dU_test", "new_lr",
                   "new_la"]].round(4).to_string())
    res["top20_overlap"] = {m: dict(n=len(t), refusals=int((t.label == "refuse").sum()), multi_turn=int(t.multi_turn.sum()),
                                    a1_flag_ge3=int((t.a1_n_flags >= 3).sum()), in_region=int(t.in_selected_region.sum()),
                                    changes_policy=int(t.policy_changed.sum()))
                            for m, t in top.groupby("model")}
    # base rates for comparison
    res["base_rates_cal"] = {m: dict(refuse_share=float((g.label == "refuse").mean()), multi_turn=float(g.multi_turn.mean()),
                                     a1_flag_ge3=float((g.a1_n_flags >= 3).mean())) for m, g in d.groupby("model")}
    print("\ntop-20 overlap:", json.dumps(res["top20_overlap"]))
    print("base rates:", json.dumps(res["base_rates_cal"]))
    res["summary"] = summ.to_dict("records")
    res["boundaries"] = bnd.to_dict("records")
    json.dump(res, open(R / "a3_summary.json", "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
