"""Step 07. Does adding the trivially certified 'execute nothing' policy (Table 1, Step 4) to G4's candidates change the
cost-optimal choice? Paper: none at a harm cost of 10 (all reported G4 results), only jev at 50 and 200.
Usage: python analysis/07_empty_policy_check.py results/scores_test_1436.csv results/r07_empty_policy_check.json"""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); SRC, OUT = sys.argv[1], sys.argv[2]
code = open(os.path.join(HERE, "06_revision_g5_costs_frontier.py")).read().split("RES = {}")[0]   # definitions only
sys.argv = ["x", SRC, os.devnull]; ns = {}; exec(code, ns)
M, y, SPLITS, g4_candidates, g4_select, pol, cost, metrics, TRS, DELTA = [ns[k] for k in ["M", "y", "SPLITS", "g4_candidates", "g4_select", "pol", "cost", "metrics", "TRS", "DELTA"]]
res = []
for name in ["mean", "microsoft", "jev", "liquid", "pplx"]:
    P = M[name]
    for ch in [10, 50, 200]:
        changed = 0; nothing = 0; safe = []; harm = []
        for s, cal in enumerate(SPLITS[:60]):
            Pc, yc, Pt, yt = P[cal], y[cal], P[~cal], y[~cal]
            b = g4_select(Pc, yc, g4_candidates(Pc, yc, DELTA), ch)
            nb = min((cost(pol(Pc, None, 0, tr), yc, ch), None, 0, tr) for tr in TRS)
            pick = b if (b is not None and b[0] <= nb[0]) else nb
            nothing += pick is nb; changed += (b is None or pick[1:] != b[1:])
            m = metrics(pol(Pt, pick[1], pick[2], pick[3]), yt); safe.append(m["safe"]); harm.append(m["harm"])
        r = dict(model=name, harm_cost=ch, splits_changed=changed, empty_chosen=nothing, safe=float(np.mean(safe)), harm=float(np.mean(harm)))
        res.append(r); print(r)
json.dump(res, open(OUT, "w"), indent=1)
