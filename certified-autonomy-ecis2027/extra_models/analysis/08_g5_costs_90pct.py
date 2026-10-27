"""Step 08. Cost sweep for G4 and G5 with 90% of the calls for calibration (40 repetitions). Paper: Table 4, last two rows.
Usage: python analysis/08_g5_costs_90pct.py results/scores_test_1436.csv results/r08_g5_costs.json"""
import json, os, sys
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); SRC, OUT = sys.argv[1], sys.argv[2]
code = open(os.path.join(HERE, "06_revision_g5_costs_frontier.py")).read().split("RES = {}")[0]
sys.argv = ["x", SRC, os.devnull]; ns = {}; exec(code, ns)
M, y, grp, split, g4_candidates, g4_select, pol, metrics, g5_fit, g5_apply, DELTA, TRS, cost = [ns[k] for k in ["M", "y", "grp", "split", "g4_candidates", "g4_select", "pol", "metrics", "g5_fit", "g5_apply", "DELTA", "TRS", "cost"]]
rows = []
for name in ["mean", "microsoft", "liquid", "pplx"]:
    P = M[name]
    for s in range(40):
        cal = split(1000 + s, 0.9); Pc, yc, Pt, yt, gc, gt = P[cal], y[cal], P[~cal], y[~cal], grp[cal], grp[~cal]
        cand = g4_candidates(Pc, yc, DELTA)
        for ch in [10, 50, 200]:
            b = g4_select(Pc, yc, cand, ch); nb = min((cost(pol(Pc, None, 0, tr), yc, ch), None, 0, tr) for tr in TRS)
            pick = b if (b is not None and b[0] <= nb[0]) else nb
            rows.append(dict(model=name, gate="G4", ch=ch, **metrics(pol(Pt, pick[1], pick[2], pick[3]), yt)))
            fit, cg = g5_fit(Pc, yc, gc, DELTA, ch, True); rows.append(dict(model=name, gate="G5", ch=ch, **metrics(g5_apply(Pt, gt, fit), yt)))
S = pd.DataFrame(rows).groupby(["model", "gate", "ch"]).mean(numeric_only=True)
print(S[["safe", "harm", "unconf", "review", "blocked", "wrong_block", "cost10", "cost50", "cost200"]].round(3).to_string())
json.dump(S.round(4).reset_index().to_dict("records"), open(OUT, "w"), indent=1)
