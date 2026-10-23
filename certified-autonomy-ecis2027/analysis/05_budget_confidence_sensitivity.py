"""Step 05. Usage: python analysis/05_budget_confidence_sensitivity.py results/scores_test_1436.csv results/r05_sensitivity.json
Paper: Table 5 (80/90/95% confidence columns).
S1 sensitivity of the certified policy (G4) to the harm budget and the confidence level.
Same data, policy family, cost (harm cost 10) and 60 stratified 50/50 splits as E6b; ask budget 15%."""
import json, sys
import numpy as np, pandas as pd
from scipy.stats import binom
SRC, OUT = sys.argv[1], sys.argv[2]
ORDER = ["direct_execute", "ask_human", "refuse"]; DE, AH, RF = 0, 1, 2; PC = [f"p_{c}" for c in ORDER]
d = pd.read_csv(SRC); d = d[d.error.isna()].copy(); d["short"] = d.model.str.split("/").str[-1]
common = set.intersection(*[set(g.dataset_idx) for _, g in d.groupby("short")]); d = d[d.dataset_idx.isin(common)]
meta = d.drop_duplicates("dataset_idx").set_index("dataset_idx").sort_index(); ids = meta.index.to_numpy()
y = meta.ground_truth.map({c: i for i, c in enumerate(ORDER)}).to_numpy(); N = len(ids)
def mat(m):
    g = d[d.short == m].set_index("dataset_idx").loc[ids, PC].to_numpy(float); g = np.clip(g, 0, None); return g / g.sum(1, keepdims=True)
M = {m: mat(m) for m in ["jev-1.14", "microsoft-decision-1", "gpt-6-luna-decisions"]}
M["ensemble (mean of 3)"] = np.mean(list(M.values()), 0)
def split(seed):
    rng = np.random.default_rng(seed); cal = np.zeros(N, bool)
    for k in range(3):
        idx = np.flatnonzero(y == k).copy(); rng.shuffle(idx); cal[idx[: len(idx) // 2]] = True
    return cal
SPLITS = [split(s) for s in range(60)]
LR = [0.0, 1e-4, 3e-4, 1e-3, 2e-3, 5e-3, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30]
LA = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]; TRS = [0.5, 0.7, 0.9, 1.01]; A_AH = 0.15; HARM = 10
def pol(P, lr, la, tr):
    a = np.full(len(P), AH); a[(P[:, RF] <= lr) & (P[:, AH] <= la)] = DE; a[P[:, RF] >= tr] = RF; return a
def cost(a, yy):
    return np.mean(a == AH) + HARM * np.mean((a == DE) & (yy == RF)) + HARM / 5 * np.mean((a == DE) & (yy == AH)) + 2 * np.mean((a == RF) & (yy != RF))
rows = []
for name, P in M.items():
    for a_rf in [0.02, 0.05, 0.10]:
        for delta in [0.05, 0.10, 0.20]:
            for s, cal in enumerate(SPLITS):
                Pc, yc, Pt, yt = P[cal], y[cal], P[~cal], y[~cal]; n_rf, n_ah = (yc == RF).sum(), (yc == AH).sum(); cand = []
                for la in LA:
                    for lr in LR:
                        ex = (Pc[:, RF] <= lr) & (Pc[:, AH] <= la)
                        if binom.cdf(int(ex[yc == RF].sum()), n_rf, a_rf) <= delta / len(LA) and binom.cdf(int(ex[yc == AH].sum()), n_ah, A_AH) <= delta / len(LA): cand.append((lr, la))
                        else: break
                if not cand: rows.append(dict(model=name, a_rf=a_rf, delta=delta, seed=s, found=0)); continue
                b = min(((cost(pol(Pc, lr, la, tr), yc), lr, la, tr) for (lr, la) in cand for tr in TRS)); a = pol(Pt, b[1], b[2], b[3])
                rows.append(dict(model=name, a_rf=a_rf, delta=delta, seed=s, found=1, de=np.mean(a[yt == DE] == DE), rf=np.mean(a[yt == RF] == DE),
                                 ah=np.mean(a[yt == AH] == DE), human=np.mean(a == AH), viol=float(np.mean(a[yt == RF] == DE) > a_rf or np.mean(a[yt == AH] == DE) > A_AH)))
S = pd.DataFrame(rows); S.to_csv(OUT.replace(".json", ".csv"), index=False)
g = S.groupby(["model", "a_rf", "delta"])
T = pd.DataFrame({"certifiable": g.found.mean(), "safe": g.de.mean(), "harm": g.rf.mean(), "ask": g.ah.mean(), "human": g.human.mean(), "viol": g.viol.mean()})
print(T.round(3).to_string()); json.dump(T.round(4).reset_index().to_dict("records"), open(OUT, "w"), indent=1, default=float)
