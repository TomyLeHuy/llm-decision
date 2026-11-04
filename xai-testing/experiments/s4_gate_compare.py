"""Step 4 analysis (no API calls): G4/G5 on the October scores (paper) vs. today's re-score of the same 1,433 calls.

  G4: 60 paper splits (50/50), harm cost 10 -> certified in how many splits, safe autonomy, harm, unconf, budget exceedance
  G5: simultaneous groups, 50/50 (60 splits) and 90% calibration (40 repetitions, seeds 1000+s as in the paper)
  Deployment: G4 calibrated on all 1,433 calls (the frozen policy of the confirmatory run) for each source, and the
  OCTOBER deployment policy applied to TODAY's scores (what a frozen gate would do after the drift).
  Drift on the full split: per model and turn group, mean |today - October| per class, share of calls whose change
  exceeds 2*sqrt(2)*SD_S3, Spearman of p_refuse, agreement of argmax choices.
Output: results/s4_gate_compare.json, s4_drift_full.csv, s4_g4_splits.csv
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binom, spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config as C  # noqa: E402
from src import io as xio  # noqa: E402
from src import gate as G  # noqa: E402

R = C.RESULTS
DE, AH, RF = G.DE, G.AH, G.RF
SCORERS = ["jev", "microsoft", "gpt", "liquid", "pplx", "mean"]
LV = C.DELTA / len(G.LA)


def rates(a, y):
    f = lambda k: float(np.mean(a[y == k] == DE))  # noqa: E731
    return dict(safe=f(DE), harm=f(RF), unconf=f(AH))


def upper(k, n, conf=0.95):
    from scipy.stats import beta
    return float(beta.ppf(conf, k + 1, n - k)) if k < n else 1.0


def main():
    ids_p = xio.common_ids("test", "paper")
    ids_t = xio.common_ids("test", "today")
    ids = np.intersect1d(ids_p, ids_t)
    Pp, meta = xio.prob_matrices("test", ids, "paper")
    Pt, _ = xio.prob_matrices("test", ids, "today")
    y = xio.label_index(meta.ground_truth)
    mt = meta.multi_turn.to_numpy()
    src = meta.service.to_numpy()
    grp, groups = G.paper_groups(src, y)
    out = dict(n_calls=len(ids), n_paper=len(ids_p), n_today=len(ids_t),
               today_outcomes=xio.load_scores("test", "today").groupby(["short", "outcome"]).size().astype(int)
               .rename(lambda x: x if isinstance(x, str) else "|".join(x)).to_dict())

    # ---------- drift on the full split
    s3 = json.load(open(R / "p0_s3_summary.json"))["rms_within_sd_by_model"]
    drows = []
    for m in ["jev", "microsoft", "gpt", "liquid", "pplx"]:
        for g, sel in [("single_turn", ~mt), ("multi_turn", mt)]:
            a, b = Pp[m][sel], Pt[m][sel]
            r = dict(model=m, group=g, n=int(sel.sum()))
            for k, c in enumerate(C.CLASSES):
                r[f"mad_{c}"] = float(np.abs(a[:, k] - b[:, k]).mean())
            thr = np.array([2 * np.sqrt(2) * s3[m][c] for c in C.CLASSES])
            r["share_beyond_noise"] = float(np.mean((np.abs(a - b) > np.maximum(thr, 1e-6)[None, :]).any(1)))
            r["spearman_refuse"] = float(spearmanr(a[:, RF], b[:, RF]).statistic)
            r["same_choice"] = float(np.mean(a.argmax(1) == b.argmax(1)))
            r["refusals_p_le_001_oct"] = float(np.mean(a[y[sel] == RF, RF] <= 0.01))
            r["refusals_p_le_001_today"] = float(np.mean(b[y[sel] == RF, RF] <= 0.01))
            drows.append(r)
    drift = pd.DataFrame(drows); drift.to_csv(R / "s4_drift_full.csv", index=False)
    out["drift"] = drift.round(4).to_dict("records")

    # ---------- G4 over 60 splits, G5 50/50 and 90%
    rows, g5rows = [], []
    for src_name, P in [("paper", Pp), ("today", Pt)]:
        for m in SCORERS:
            for s in range(60):
                cal = G.split(y, s)
                b = G.g4_select(P[m][cal], y[cal], G.g4_candidates(P[m][cal], y[cal]))
                r = dict(source=src_name, model=m, seed=s, found=b is not None)
                if b is not None:
                    a = G.pol(P[m], b[1], b[2], b[3]); te = ~cal
                    r.update(lr=b[1], la=b[2], tr=b[3], **rates(a[te], y[te]),
                             **{f"{k}_{g}": v for g, sel in [("single", te & ~mt), ("multi", te & mt)] for k, v in rates(a[sel], y[sel]).items()})
                rows.append(r)
                fit, cg = G.g5_fit(P[m][cal], y[cal], grp[cal], groups)
                a = G.g5_apply(P[m][~cal], grp[~cal], fit)
                g5rows.append(dict(source=src_name, model=m, frac=0.5, seed=s, n_cert=len(cg), **rates(a, y[~cal])))
            for s in range(40):
                cal = G.split(y, 1000 + s, 0.9)
                fit, cg = G.g5_fit(P[m][cal], y[cal], grp[cal], groups)
                a = G.g5_apply(P[m][~cal], grp[~cal], fit)
                g5rows.append(dict(source=src_name, model=m, frac=0.9, seed=s, n_cert=len(cg), cert=",".join(sorted(cg)), **rates(a, y[~cal])))
    g4 = pd.DataFrame(rows); g4.to_csv(R / "s4_g4_splits.csv", index=False)
    gg = g4.groupby(["model", "source"])
    t = gg[["safe", "harm", "unconf", "harm_single", "harm_multi"]].mean()
    t["found"] = gg.found.mean()
    t["over_harm"] = gg.harm.apply(lambda x: float(np.mean(x.dropna() > C.A_HARM)))
    t["over_ask"] = gg.unconf.apply(lambda x: float(np.mean(x.dropna() > C.A_ASK)))
    out["g4"] = t.round(4).reset_index().to_dict("records")
    g5 = pd.DataFrame(g5rows)
    t5 = g5.groupby(["model", "source", "frac"])[["safe", "harm", "unconf", "n_cert"]].mean()
    out["g5"] = t5.round(4).reset_index().to_dict("records")

    # ---------- deployment policies on all calls; October policy applied to today's scores
    dep = []
    for m in SCORERS:
        pols = {}
        for src_name, P in [("paper", Pp), ("today", Pt)]:
            b = G.g4_select(P[m], y, G.g4_candidates(P[m], y))
            pols[src_name] = b
            r = dict(model=m, calibrated_on=src_name, applied_to=src_name, certified=b is not None)
            if b is not None:
                pv = G.ltt_pvalue(P[m], y, b[1], b[2]); a = G.pol(P[m], b[1], b[2], b[3])
                r.update(lr=b[1], la=b[2], tr=b[3], p=pv["p"], slack_log10=float(np.log10(LV) - np.log10(max(pv["p"], 1e-300))), **rates(a, y))
            dep.append(r)
        b = pols["paper"]
        if b is not None:
            a = G.pol(Pt[m], b[1], b[2], b[3])
            k_rf, n_rf = int(np.sum(a[y == RF] == DE)), int(np.sum(y == RF)); k_ah, n_ah = int(np.sum(a[y == AH] == DE)), int(np.sum(y == AH))
            dep.append(dict(model=m, calibrated_on="paper", applied_to="today", certified=None, lr=b[1], la=b[2], tr=b[3],
                            **rates(a, y), harm_upper95=upper(k_rf, n_rf), unconf_upper95=upper(k_ah, n_ah),
                            ltt_p_today=float(max(binom.cdf(k_rf, n_rf, C.A_HARM), binom.cdf(k_ah, n_ah, C.A_ASK))),
                            still_passes_ltt_today=bool(max(binom.cdf(k_rf, n_rf, C.A_HARM), binom.cdf(k_ah, n_ah, C.A_ASK)) <= LV),
                            harm_single=rates(a[~mt], y[~mt])["harm"], harm_multi=rates(a[mt], y[mt])["harm"]))
    dep = pd.DataFrame(dep)
    out["deployment"] = dep.round(4).to_dict("records")
    xio.dump_json(out, R / "s4_gate_compare.json")
    pd.set_option("display.width", 250)
    print("n", len(ids), out["today_outcomes"])
    print(drift.round(3).to_string())
    print(t.round(3).to_string())
    print(t5.round(3).to_string())
    print(dep.round(4).to_string())


if __name__ == "__main__":
    main()
