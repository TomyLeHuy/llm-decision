"""Step 03. Usage: python analysis/03_policies_and_groups.py results/scores_test_1436.csv results/r03_policies_groups.json
Paper: Table 3 rows G4 (E6b), tuned policy, Figure 5 (E10), Section 5.5 coarse-vendor mean (E11).
E6b certified (Learn-then-Test) vs tuned vs Mondrian-in-expectation, with a refuse-probability policy family
E10 group-conditional calibration of the refusal class (by source) vs class-conditional only
E11 ensemble of the two coarse-grid vendors (jev + gpt)
"""
import json, sys
import numpy as np, pandas as pd
from scipy.stats import binom
pd.set_option("display.width", 230); pd.set_option("display.max_columns", 40)
SRC, OUT = sys.argv[1], sys.argv[2]
ORDER = ["direct_execute", "ask_human", "refuse"]; DE, AH, RF = 0, 1, 2; PC = [f"p_{c}" for c in ORDER]
d = pd.read_csv(SRC); d = d[d.error.isna()].copy(); d["short"] = d.model.str.split("/").str[-1]
common = set.intersection(*[set(g.dataset_idx) for _, g in d.groupby("short")]); d = d[d.dataset_idx.isin(common)]
meta = d.drop_duplicates("dataset_idx").set_index("dataset_idx").sort_index(); ids = meta.index.to_numpy()
y = meta.ground_truth.map({c: i for i, c in enumerate(ORDER)}).to_numpy(); src = meta.service.to_numpy(); N = len(ids)
def mat(m):
    g = d[d.short == m].set_index("dataset_idx").loc[ids, PC].to_numpy(float); g = np.clip(g, 0, None); return g / g.sum(1, keepdims=True)
def quant(P, grid):
    Q = np.round(P / grid) * grid; return Q / Q.sum(1, keepdims=True)
M = {m: mat(m) for m in ["jev-1.14", "microsoft-decision-1", "gpt-6-luna-decisions", "d1-ultra", "pplx-decider-v1.2"]}
M["ensemble (mean of 3)"] = np.mean([M["jev-1.14"], M["microsoft-decision-1"], M["gpt-6-luna-decisions"]], 0)
M["ensemble jev+gpt (both 0.01 grid)"] = quant(np.mean([M["jev-1.14"], M["gpt-6-luna-decisions"]], 0), 0.01)
M["microsoft, rounded to 0.01"] = quant(M["microsoft-decision-1"], 0.01)
def split(seed):
    rng = np.random.default_rng(seed); cal = np.zeros(N, bool)
    for k in range(3):
        idx = np.flatnonzero(y == k).copy(); rng.shuffle(idx); cal[idx[: len(idx) // 2]] = True
    return cal
SPLITS = [split(s) for s in range(100)]
def qhat(s, a):
    n = len(s); lv = np.ceil((n + 1) * (1 - a)) / n
    return np.inf if (n == 0 or lv > 1) else float(np.quantile(s, lv, method="higher"))
def rates(a, yy):
    return dict(rf_exec=np.mean(a[yy == RF] == DE), ah_exec=np.mean(a[yy == AH] == DE), de_exec=np.mean(a[yy == DE] == DE),
                de_block=np.mean(a[yy == DE] == RF), auto_execute=np.mean(a == DE), auto_refuse=np.mean(a == RF), to_human=np.mean(a == AH))
R = {}
# co-occurrence for E11
pt = {m: M[m][np.arange(N), y] for m in M}
R["E11_floor"] = {m: dict(zero=float(np.mean(pt[m][y == RF] == 0)), le001=float(np.mean(pt[m][y == RF] <= 0.01))) for m in M}
print("E11 floor on true refusals:", {k: {a: round(b, 3) for a, b in v.items()} for k, v in R["E11_floor"].items()})

# ---------------- E6b policy family: execute if p_RF <= lr and p_AH <= la ; refuse if p_RF >= tr ; else ask
LR = [0.0, 1e-4, 3e-4, 1e-3, 2e-3, 5e-3, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30]   # increasing risk
LA = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
TRS = [0.5, 0.7, 0.9, 1.01]
def pol(P, lr, la, tr):
    a = np.full(len(P), AH); a[(P[:, RF] <= lr) & (P[:, AH] <= la)] = DE; a[P[:, RF] >= tr] = RF; return a
def cost(a, yy, ch):
    return np.mean(a == AH) + ch * np.mean((a == DE) & (yy == RF)) + ch / 5 * np.mean((a == DE) & (yy == AH)) + 2 * np.mean((a == RF) & (yy != RF))
def candidates(P, yy, a_rf, a_ah, delta, mode):
    """returns list of (lr, la) admissible. certified: LTT with Bonferroni over LA, fixed sequence over LR"""
    n_rf, n_ah = (yy == RF).sum(), (yy == AH).sum(); out = []
    for la in LA:
        for lr in LR:
            ex = (P[:, RF] <= lr) & (P[:, AH] <= la); k_rf = int(np.sum(ex[yy == RF])); k_ah = int(np.sum(ex[yy == AH]))
            if mode == "certified":
                if binom.cdf(k_rf, n_rf, a_rf) <= delta / len(LA) and binom.cdf(k_ah, n_ah, a_ah) <= delta / len(LA): out.append((lr, la))
                else: break
            else:
                if k_rf / n_rf <= a_rf and k_ah / n_ah <= a_ah: out.append((lr, la))
    return out
rows = []; A_RF, A_AH, DELTA = 0.05, 0.15, 0.10
for name, P in M.items():
    for s, cal in enumerate(SPLITS[:60]):
        Pc, yc, Pt, yt = P[cal], y[cal], P[~cal], y[~cal]
        for mode in ["tuned", "certified"]:
            cand = candidates(Pc, yc, A_RF, A_AH, DELTA, mode)
            for ch in [10, 50, 200]:
                if not cand: rows.append(dict(model=name, mode=mode, harm=ch, seed=s, found=False)); continue
                best = min(((cost(pol(Pc, lr, la, tr), yc, ch), lr, la, tr) for (lr, la) in cand for tr in TRS))
                a = pol(Pt, best[1], best[2], best[3])
                rows.append(dict(model=name, mode=mode, harm=ch, seed=s, found=True, lr=best[1], la=best[2], tr=best[3], **rates(a, yt)))
        # Mondrian in expectation, same budgets, set rule "execute if refuse excluded and DE in set"
        Sc = 1 - Pc; tc = Sc[np.arange(len(yc)), yc]; q = np.array([qhat(tc[yc == k], {DE: .15, AH: A_AH, RF: A_RF}[k]) for k in range(3)])
        S = (1 - Pt) <= q[None, :]; a = np.full(len(Pt), AH); a[(~S[:, RF]) & S[:, DE]] = DE; a[S[:, RF] & ~S[:, DE] & ~S[:, AH]] = RF
        rows.append(dict(model=name, mode="mondrian (in expectation)", harm=0, seed=s, found=True, **rates(a, yt)))
E6 = pd.DataFrame(rows); g = E6[E6.found].groupby(["model", "mode", "harm"])
E6s = g[["rf_exec", "ah_exec", "de_exec", "de_block", "auto_execute", "auto_refuse", "to_human"]].mean()
E6s["viol_rf"] = g.rf_exec.apply(lambda x: np.mean(x > A_RF)); E6s["viol_ah"] = g.ah_exec.apply(lambda x: np.mean(x > A_AH))
E6s["found"] = E6.groupby(["model", "mode", "harm"]).found.mean(); print("\nE6b\n", E6s.round(3)); R["E6b"] = E6s.round(4).reset_index().to_dict("records")
E6.to_csv(OUT.replace(".json", "_E6b.csv"), index=False)

# ---------------- E10 group-conditional refusal calibration by source
# group for refusals = source if it has >= 30 refusals in calibration-sized pool, else "other"; known at deployment (the target service)
big = [s for s in np.unique(src) if ((src == s) & (y == RF)).sum() >= 40]
grp = np.array([s if s in big else "other" for s in src]); print("\nE10 refusal groups:", big, "+ other")
rows = []
for name in ["microsoft-decision-1", "gpt-6-luna-decisions", "d1-ultra", "pplx-decider-v1.2", "ensemble (mean of 3)"]:
    P = M[name]
    for s, cal in enumerate(SPLITS):
        Pc, yc, gc, Pt, yt, gt = P[cal], y[cal], grp[cal], P[~cal], y[~cal], grp[~cal]
        Sc = 1 - Pc; tc = Sc[np.arange(len(yc)), yc]
        q_de, q_ah = qhat(tc[yc == DE], .15), qhat(tc[yc == AH], .15)
        for variant in ["class-conditional", "class x source"]:
            if variant == "class-conditional": q_rf = np.full(len(Pt), qhat(tc[yc == RF], .05))
            else:
                qg = {gname: qhat(tc[(yc == RF) & (gc == gname)], .05) for gname in np.unique(grp)}; q_rf = np.array([qg[gg] for gg in gt])
            S = np.stack([(1 - Pt[:, DE]) <= q_de, (1 - Pt[:, AH]) <= q_ah, (1 - Pt[:, RF]) <= q_rf], 1)
            a = np.full(len(Pt), AH); a[(~S[:, RF]) & S[:, DE]] = DE; a[S[:, RF] & ~S[:, DE] & ~S[:, AH]] = RF
            r = rates(a, yt)
            for gname in np.unique(grp):
                m_rf = (gt == gname) & (yt == RF); m_all = gt == gname
                r[f"cat_{gname}"] = np.mean(a[m_rf] == DE) if m_rf.any() else np.nan
                r[f"auto_{gname}"] = np.mean(a[m_all & (yt == DE)] == DE) if (m_all & (yt == DE)).any() else np.nan
            rows.append(dict(model=name, variant=variant, seed=s, **r))
E10 = pd.DataFrame(rows); cols = ["rf_exec", "ah_exec", "de_exec", "auto_execute"] + [c for c in E10.columns if c.startswith("cat_")] + [c for c in E10.columns if c.startswith("auto_") and c != "auto_execute" and c != "auto_refuse"]
E10s = E10.groupby(["model", "variant"])[cols].mean(); print("\nE10\n", E10s.round(3).T); R["E10"] = E10s.round(4).reset_index().to_dict("records")
json.dump(R, open(OUT, "w"), indent=1, default=float); print("done")
