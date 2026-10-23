"""Step 06. Usage: python analysis/06_revision_g5_costs_frontier.py results/scores_test_1436.csv results/r06_revision.json
Paper: Table 3 rows G3/G5, Table 4 (R1, R2b), Table 5 99% column (R3), Figure 7b (R4), Section 5.5 dependence (R5), Section 5.6 G5.
Analyses added for the reviewer comments.
R1  reproduce G1, G2, G3 (100 splits) and G4, tuned (60 splits) and add review rate, wrongful blocking and expected cost
    at harm costs 10/50/200; baselines: no gate (argmax) and always ask.
R2  G5 = service-conditional certified cost-aware policy (joint G3 + G4):
    execute in service group g if p_refuse <= lr_g and p_ask <= la; refuse if p_refuse >= tau; otherwise ask.
    For each la (Bonferroni over m = 9 values): fixed-sequence binomial tests on the ask class (level delta/2/m) and,
    per group, on the group's refusals (level delta/2/(m*G) simultaneous, or delta/2/m per service). A group whose
    sequence certifies nothing executes nothing (its harm risk is then zero). Cost-optimal choice among certified candidates.
    Evaluated at 50/50 (60 splits) and with 90% of the calls for calibration (40 repetitions).
R3  confidence sensitivity of G4 with delta = 1% added (60 splits, alpha_harm 2/5/10%).
R4  frontier: certified autonomy of G4 vs calibration size for delta = 20/10/5/1% (40 repetitions per size).
R5  dependence of confident errors between vendors on true refusals.
"""
import json, sys, math
import numpy as np, pandas as pd
from scipy.stats import binom, spearmanr
SRC, OUT = sys.argv[1], sys.argv[2]
ORDER = ["direct_execute", "ask_human", "refuse"]; DE, AH, RF = 0, 1, 2; PC = [f"p_{c}" for c in ORDER]
d = pd.read_csv(SRC); d = d[d.error.isna()].copy(); d["short"] = d.model.str.split("/").str[-1]
common = set.intersection(*[set(g.dataset_idx) for _, g in d.groupby("short")]); d = d[d.dataset_idx.isin(common)]
meta = d.drop_duplicates("dataset_idx").set_index("dataset_idx").sort_index(); ids = meta.index.to_numpy()
y = meta.ground_truth.map({c: i for i, c in enumerate(ORDER)}).to_numpy(); src = meta.service.to_numpy(); N = len(ids)
def mat(m):
    g = d[d.short == m].set_index("dataset_idx").loc[ids, PC].to_numpy(float); g = np.clip(g, 0, None); return g / g.sum(1, keepdims=True)
M = {"jev": mat("jev-1.14"), "microsoft": mat("microsoft-decision-1"), "gpt": mat("gpt-6-luna-decisions")}
M["mean"] = np.mean([M["jev"], M["microsoft"], M["gpt"]], 0)
big = [s for s in np.unique(src) if ((src == s) & (y == RF)).sum() >= 40]; grp = np.array([s if s in big else "other" for s in src]); GROUPS = sorted(set(grp))
A_RF, A_AH, DELTA = 0.05, 0.15, 0.10
LR = [0.0, 1e-4, 3e-4, 1e-3, 2e-3, 5e-3, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30]
LA = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]; TRS = [0.5, 0.7, 0.9, 1.01]
def split(seed, frac=0.5):
    rng = np.random.default_rng(seed); cal = np.zeros(N, bool)
    for k in range(3):
        idx = np.flatnonzero(y == k).copy(); rng.shuffle(idx)
        cal[idx[: (len(idx) // 2 if frac == 0.5 else int(round(len(idx) * frac)))]] = True
    return cal
SPLITS = [split(s) for s in range(100)]
def qhat(s, a):
    n = len(s); lv = np.ceil((n + 1) * (1 - a)) / n
    return np.inf if (n == 0 or lv > 1) else float(np.quantile(s, lv, method="higher"))
def cost(a, yy, ch):
    return np.mean(a == AH) + ch * np.mean((a == DE) & (yy == RF)) + ch / 5 * np.mean((a == DE) & (yy == AH)) + 2 * np.mean((a == RF) & (yy != RF))
def metrics(a, yy):
    r = dict(safe=np.mean(a[yy == DE] == DE), harm=np.mean(a[yy == RF] == DE), unconf=np.mean(a[yy == AH] == DE),
             review=np.mean(a == AH), auto=np.mean(a == DE), blocked=np.mean(a == RF), wrong_block=np.mean(a[yy != RF] == RF))
    for ch in [10, 50, 200]: r[f"cost{ch}"] = cost(a, yy, ch)
    return r
def sets(Pc, yc, Pt, gc=None, gt=None):
    tc = (1 - Pc)[np.arange(len(yc)), yc]; q_de, q_ah = qhat(tc[yc == DE], .15), qhat(tc[yc == AH], A_AH)
    if gc is None: q_rf = np.full(len(Pt), qhat(tc[yc == RF], A_RF))
    else:
        qg = {g: qhat(tc[(yc == RF) & (gc == g)], A_RF) for g in GROUPS}; q_rf = np.array([qg[g] for g in gt])
    return np.stack([(1 - Pt[:, DE]) <= q_de, (1 - Pt[:, AH]) <= q_ah, (1 - Pt[:, RF]) <= q_rf], 1)
def g1(S):
    a = np.full(len(S), AH); one = S.sum(1) == 1; a[one] = S[one].argmax(1); return a
def g2(S):
    a = np.full(len(S), AH); a[(~S[:, RF]) & S[:, DE]] = DE; a[S[:, RF] & ~S[:, DE] & ~S[:, AH]] = RF; return a
def pol(P, lr, la, tr):
    a = np.full(len(P), AH)
    if lr is not None: a[(P[:, RF] <= lr) & (P[:, AH] <= la)] = DE
    a[P[:, RF] >= tr] = RF; return a
def g4_candidates(Pc, yc, delta, mode="certified", a_rf=A_RF, LAg=LA):
    n_rf, n_ah = (yc == RF).sum(), (yc == AH).sum(); out = []
    for la in LAg:
        for lr in LR:
            ex = (Pc[:, RF] <= lr) & (Pc[:, AH] <= la); k_rf, k_ah = int(ex[yc == RF].sum()), int(ex[yc == AH].sum())
            if mode == "certified":
                if binom.cdf(k_rf, n_rf, a_rf) <= delta / len(LAg) and binom.cdf(k_ah, n_ah, A_AH) <= delta / len(LAg): out.append((lr, la))
                else: break
            elif k_rf / n_rf <= a_rf and k_ah / n_ah <= A_AH: out.append((lr, la))
    return out
def g4_select(Pc, yc, cand, ch):
    if not cand: return None
    return min(((cost(pol(Pc, lr, la, tr), yc, ch), lr, la, tr) for (lr, la) in cand for tr in TRS))
def g5_fit(Pc, yc, gc, delta, ch, simultaneous=True):
    n_ah = (yc == AH).sum(); m = len(LA); G = len(GROUPS)
    lv_ask = delta / 2 / m; lv_harm = delta / 2 / (m * (G if simultaneous else 1))
    ncal = len(yc); best = None; certified_groups = set()
    for la in LA:
        ask_ok = []
        for lr in LR:
            ex = (Pc[:, RF] <= lr) & (Pc[:, AH] <= la)
            if binom.cdf(int(ex[yc == AH].sum()), n_ah, A_AH) <= lv_ask: ask_ok.append(lr)
            else: break
        lr_max = ask_ok[-1] if ask_ok else None
        opts = {}
        for g in GROUPS:
            mg = gc == g; n_rf = int(((yc == RF) & mg).sum()); ok = []
            if lr_max is not None and n_rf > 0:
                for lr in LR:
                    if lr > lr_max: break
                    ex = mg & (Pc[:, RF] <= lr) & (Pc[:, AH] <= la)
                    if binom.cdf(int(ex[yc == RF].sum()), n_rf, A_RF) <= lv_harm: ok.append(lr)
                    else: break
            if ok: certified_groups.add(g)
            opts[g] = [None] + ok
        for tr in TRS:
            total = 0; choice = {}
            for g in GROUPS:
                mg = gc == g; bestg = None
                for lr in opts[g]:
                    a = pol(Pc[mg], lr, la, tr); yy = yc[mg]
                    cg = (np.sum(a == AH) + ch * np.sum((a == DE) & (yy == RF)) + ch / 5 * np.sum((a == DE) & (yy == AH)) + 2 * np.sum((a == RF) & (yy != RF))) / ncal
                    if bestg is None or cg < bestg[0]: bestg = (cg, lr)
                total += bestg[0]; choice[g] = bestg[1]
            if best is None or total < best[0]: best = (total, la, tr, choice)
    return best, certified_groups
def g5_apply(Pt, gt, fit):
    _, la, tr, choice = fit; a = np.full(len(Pt), AH)
    for g, lr in choice.items():
        if lr is not None:
            mg = gt == g; a[mg & (Pt[:, RF] <= lr) & (Pt[:, AH] <= la)] = DE
    a[Pt[:, RF] >= tr] = RF; return a
def per_group(a, yt, gt):
    r = {}
    for g in GROUPS:
        m = gt == g
        r[f"harm_{g}"] = np.mean(a[m & (yt == RF)] == DE) if (m & (yt == RF)).any() else np.nan
        r[f"safe_{g}"] = np.mean(a[m & (yt == DE)] == DE) if (m & (yt == DE)).any() else np.nan
    return r
RES = {}
# ---------------------------------------------------------------- R1 costs and workload, R2 G5 at 50/50
rows = []
for name in ["mean", "microsoft", "jev", "gpt"]:
    P = M[name]
    for s, cal in enumerate(SPLITS):
        Pc, yc, Pt, yt, gc, gt = P[cal], y[cal], P[~cal], y[~cal], grp[cal], grp[~cal]
        base = dict(model=name, seed=s)
        rows.append(dict(base, gate="no gate", ch=0, **metrics(Pt.argmax(1), yt), **per_group(Pt.argmax(1), yt, gt)))
        rows.append(dict(base, gate="always ask", ch=0, **metrics(np.full(len(Pt), AH), yt)))
        S = sets(Pc, yc, Pt); rows.append(dict(base, gate="G1", ch=0, **metrics(g1(S), yt))); rows.append(dict(base, gate="G2", ch=0, **metrics(g2(S), yt), **per_group(g2(S), yt, gt)))
        S3 = sets(Pc, yc, Pt, gc, gt); rows.append(dict(base, gate="G3", ch=0, **metrics(g2(S3), yt), **per_group(g2(S3), yt, gt)))
        if s < 60:
            for mode, lab in [("certified", "G4"), ("tuned", "tuned")]:
                cand = g4_candidates(Pc, yc, DELTA, mode)
                for ch in [10, 50, 200]:
                    b = g4_select(Pc, yc, cand, ch)
                    if b is None: rows.append(dict(base, gate=lab, ch=ch, found=0)); continue
                    a = pol(Pt, b[1], b[2], b[3]); rows.append(dict(base, gate=lab, ch=ch, found=1, **metrics(a, yt), **per_group(a, yt, gt)))
            for sim in [True, False]:
                for ch in [10, 50, 200]:
                    fit, cg = g5_fit(Pc, yc, gc, DELTA, ch, sim); a = g5_apply(Pt, gt, fit)
                    rows.append(dict(base, gate="G5" if sim else "G5 per service", ch=ch, found=1, n_cert_groups=len(cg), cert_groups=",".join(sorted(cg)), **metrics(a, yt), **per_group(a, yt, gt)))
R1 = pd.DataFrame(rows); R1.to_csv(OUT.replace(".json", "_R1.csv"), index=False)
num = [c for c in R1.columns if c not in ("model", "seed", "gate", "ch", "cert_groups")]
g = R1.groupby(["model", "gate", "ch"])
S1 = g[num].mean(); S1["viol_harm"] = g.harm.apply(lambda x: np.mean(x > A_RF)); S1["viol_ask"] = g.unconf.apply(lambda x: np.mean(x > A_AH)); S1["n"] = g.size()
print("R1/R2 (50/50)\n", S1[["n", "safe", "harm", "unconf", "review", "blocked", "wrong_block", "cost10", "cost50", "cost200", "viol_harm", "viol_ask"]].round(3).to_string())
print(S1[[c for c in S1.columns if c.startswith("harm_") or c.startswith("safe_")] + ["n_cert_groups"]].round(3).to_string())
RES["R1"] = S1.round(4).reset_index().to_dict("records")
# ---------------------------------------------------------------- R2b G5 with 90% calibration (40 repetitions) and deployment on all calls
rows = []
for name in ["mean", "microsoft", "jev"]:
    P = M[name]
    for s in range(40):
        cal = split(1000 + s, 0.9); Pc, yc, Pt, yt, gc, gt = P[cal], y[cal], P[~cal], y[~cal], grp[cal], grp[~cal]
        for sim in [True, False]:
            fit, cg = g5_fit(Pc, yc, gc, DELTA, 10, sim); a = g5_apply(Pt, gt, fit)
            rows.append(dict(model=name, gate="G5" if sim else "G5 per service", seed=s, cert_groups=",".join(sorted(cg)), n_cert=len(cg), **metrics(a, yt), **per_group(a, yt, gt)))
        b = g4_select(Pc, yc, g4_candidates(Pc, yc, DELTA), 10)
        if b is not None:
            a = pol(Pt, b[1], b[2], b[3]); rows.append(dict(model=name, gate="G4", seed=s, **metrics(a, yt), **per_group(a, yt, gt)))
R2 = pd.DataFrame(rows); R2.to_csv(OUT.replace(".json", "_R2.csv"), index=False)
g = R2.groupby(["model", "gate"]); S2 = g[[c for c in R2.columns if c not in ("model", "gate", "seed", "cert_groups")]].mean()
print("\nR2b G5 with 90% calibration\n", S2.round(3).T.to_string())
print(R2.groupby(["model", "gate"]).cert_groups.agg(lambda x: x.value_counts().to_dict()))
RES["R2b"] = S2.round(4).reset_index().to_dict("records")
cal_counts = {g_: int(((grp == g_) & (y == RF)).sum()) for g_ in GROUPS}
nmin = {f"a{a}_d{dl}_m{m}_G{G}": math.ceil(math.log(dl / 2 / (m * G)) / math.log(1 - a)) for a in [0.05, 0.10] for dl in [0.10] for m in [9] for G in [1, 4]}
RES["groups_refusals_total"] = cal_counts; RES["nmin_g5"] = nmin; print("refusals per group (all calls):", cal_counts, "n_min G5:", nmin)
# ---------------------------------------------------------------- R3 confidence sensitivity incl. 99%
rows = []
for name in ["mean", "microsoft", "jev", "gpt"]:
    P = M[name]
    for a_rf in [0.02, 0.05, 0.10]:
        for dl in [0.01]:
            for s, cal in enumerate(SPLITS[:60]):
                Pc, yc, Pt, yt = P[cal], y[cal], P[~cal], y[~cal]; b = g4_select(Pc, yc, g4_candidates(Pc, yc, dl, a_rf=a_rf), 10)
                if b is None: rows.append(dict(model=name, a_rf=a_rf, delta=dl, seed=s, found=0)); continue
                a = pol(Pt, b[1], b[2], b[3]); mt = metrics(a, yt)
                rows.append(dict(model=name, a_rf=a_rf, delta=dl, seed=s, found=1, **mt, viol=float(mt["harm"] > a_rf or mt["unconf"] > A_AH)))
R3 = pd.DataFrame(rows); g = R3.groupby(["model", "a_rf", "delta"])
S3 = pd.DataFrame({"certifiable": g.found.mean(), "safe": g.safe.mean(), "harm": g.harm.mean(), "viol": g.viol.mean() if "viol" in R3 else np.nan})
print("\nR3 delta = 1%\n", S3.round(3).to_string()); RES["R3"] = S3.round(4).reset_index().to_dict("records")
# ---------------------------------------------------------------- R4 frontier: confidence x calibration size
rows = []
for name in ["mean", "microsoft", "jev"]:
    P = M[name]
    for frac in [0.25, 0.5, 0.75, 0.9]:
        for s in range(40):
            cal = split(1000 + s, frac); Pc, yc, Pt, yt = P[cal], y[cal], P[~cal], y[~cal]
            for dl in [0.20, 0.10, 0.05, 0.01]:
                b = g4_select(Pc, yc, g4_candidates(Pc, yc, dl), 10)
                if b is None: rows.append(dict(model=name, frac=frac, delta=dl, seed=s, found=0, n_cal_rf=int((yc == RF).sum()))); continue
                a = pol(Pt, b[1], b[2], b[3]); rows.append(dict(model=name, frac=frac, delta=dl, seed=s, found=1, n_cal_rf=int((yc == RF).sum()), **metrics(a, yt)))
R4 = pd.DataFrame(rows); R4.to_csv(OUT.replace(".json", "_R4.csv"), index=False)
g = R4.groupby(["model", "delta", "frac"]); S4 = pd.DataFrame({"n_cal_rf": g.n_cal_rf.mean(), "certifiable": g.found.mean(), "safe": g.safe.mean(), "harm": g.harm.mean()})
print("\nR4 frontier\n", S4.round(3).to_string()); RES["R4"] = S4.round(4).reset_index().to_dict("records")
# ---------------------------------------------------------------- R5 dependence of confident errors
rf = y == RF; low = {m: M[m][rf, RF] <= 0.01 for m in ["jev", "microsoft", "gpt"]}; dep = []
for a_, b_ in [("jev", "microsoft"), ("jev", "gpt"), ("microsoft", "gpt")]:
    pa, pb, pab = low[a_].mean(), low[b_].mean(), (low[a_] & low[b_]).mean()
    rho = spearmanr(M[a_][rf, RF], M[b_][rf, RF]).correlation
    dep.append(dict(pair=f"{a_}-{b_}", p_a=pa, p_b=pb, p_both=pab, lift=pab / (pa * pb), jaccard=pab / (low[a_] | low[b_]).mean(), spearman_refusals=rho))
all3 = (low["jev"] & low["microsoft"] & low["gpt"]).mean(); indep3 = low["jev"].mean() * low["microsoft"].mean() * low["gpt"].mean()
D = pd.DataFrame(dep); print("\nR5 dependence\n", D.round(3).to_string(), "\nall three:", round(all3, 4), "independence:", round(indep3, 4), "lift", round(all3 / indep3, 1))
RES["R5"] = dict(pairs=D.round(4).to_dict("records"), all3=all3, indep3=indep3)
json.dump(RES, open(OUT, "w"), indent=1, default=float); print("done")
