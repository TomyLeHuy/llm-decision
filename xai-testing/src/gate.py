"""Thin copies of the paper's gate code (G4 and G5), for certificate-level analyses.

The analysis steps in certified-autonomy-ecis2027/extra_models/analysis/ are scripts with side effects at import time,
so the functions below are COPIED from them, unchanged in logic:
  LR, LA, TRS, pol, cost, g4_candidates, g4_select  <- 06_revision_g5_costs_frontier.py (identical to 03 E6b)
  g5_fit, g5_apply                                   <- 06_revision_g5_costs_frontier.py (GROUPS made a parameter)
  split                                              <- 03/06 (stratified 50/50 calibration-test split by label)
`check_reproduction()` re-runs G4 for every model on the paper's 60 splits and compares with the stored per-split
results (results/extra_models/r03_policies_groups_E6b.csv); it must report 0 mismatches.
Additions (not in the paper): ltt_pvalue, candidate_trace, extra_needed.
"""
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import binom

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C  # noqa: E402

DE, AH, RF = 0, 1, 2
A_RF, A_AH, DELTA = C.A_HARM, C.A_ASK, C.DELTA
LR = [0.0, 1e-4, 3e-4, 1e-3, 2e-3, 5e-3, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30]
LA = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
TRS = [0.5, 0.7, 0.9, 1.01]


def split(y, seed, frac=0.5):
    N = len(y)
    rng = np.random.default_rng(seed); cal = np.zeros(N, bool)
    for k in range(3):
        idx = np.flatnonzero(y == k).copy(); rng.shuffle(idx)
        cal[idx[: (len(idx) // 2 if frac == 0.5 else int(round(len(idx) * frac)))]] = True
    return cal


def cost(a, yy, ch):
    return np.mean(a == AH) + ch * np.mean((a == DE) & (yy == RF)) + ch / 5 * np.mean((a == DE) & (yy == AH)) + 2 * np.mean((a == RF) & (yy != RF))


def pol(P, lr, la, tr):
    a = np.full(len(P), AH)
    if lr is not None: a[(P[:, RF] <= lr) & (P[:, AH] <= la)] = DE
    a[P[:, RF] >= tr] = RF; return a


def g4_candidates(Pc, yc, delta=DELTA, mode="certified", a_rf=A_RF, LAg=LA):
    n_rf, n_ah = (yc == RF).sum(), (yc == AH).sum(); out = []
    for la in LAg:
        for lr in LR:
            ex = (Pc[:, RF] <= lr) & (Pc[:, AH] <= la); k_rf, k_ah = int(ex[yc == RF].sum()), int(ex[yc == AH].sum())
            if mode == "certified":
                if binom.cdf(k_rf, n_rf, a_rf) <= delta / len(LAg) and binom.cdf(k_ah, n_ah, A_AH) <= delta / len(LAg): out.append((lr, la))
                else: break
            elif k_rf / n_rf <= a_rf and k_ah / n_ah <= A_AH: out.append((lr, la))
    return out


def g4_select(Pc, yc, cand, ch=C.HARM_COST):
    if not cand: return None
    return min(((cost(pol(Pc, lr, la, tr), yc, ch), lr, la, tr) for (lr, la) in cand for tr in TRS))


def g5_fit(Pc, yc, gc, groups, delta=DELTA, ch=C.HARM_COST, simultaneous=True):
    n_ah = (yc == AH).sum(); m = len(LA); G = len(groups)
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
        for g in groups:
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
            for g in groups:
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


def paper_groups(src, y):
    """G5 service groups of the paper: services with >= 40 refusals in the test pool, rest pooled as 'other'."""
    big = [s for s in np.unique(src) if ((src == s) & (y == RF)).sum() >= 40]
    return np.array([s if s in big else "other" for s in src]), sorted(set(big) | {"other"})


# ---------------------------------------------------------------- additions
def ltt_pvalue(Pc, yc, lr, la, a_rf=A_RF, a_ah=A_AH):
    ex = (Pc[:, RF] <= lr) & (Pc[:, AH] <= la)
    k_rf, k_ah = int(ex[yc == RF].sum()), int(ex[yc == AH].sum())
    n_rf, n_ah = int((yc == RF).sum()), int((yc == AH).sum())
    p_rf, p_ah = binom.cdf(k_rf, n_rf, a_rf), binom.cdf(k_ah, n_ah, a_ah)
    return dict(k_rf=k_rf, n_rf=n_rf, k_ah=k_ah, n_ah=n_ah, p_rf=float(p_rf), p_ah=float(p_ah), p=float(max(p_rf, p_ah)))


def kmax(n, level, a=A_RF):
    """largest k with F(k; n, a) <= level, or -1 if none (Proposition 2)."""
    if n <= 0: return -1
    k = int(binom.ppf(level, n, a))
    while k >= 0 and binom.cdf(k, n, a) > level: k -= 1
    while binom.cdf(k + 1, n, a) <= level: k += 1
    return k


def n_min(level, a=A_RF):
    return math.ceil(math.log(level) / math.log(1 - a))


def extra_needed(k, n, level, a=A_RF, cap=5000):
    """number x of additional calibration cases of the class, all OUTSIDE the execution region ('clean'), so that
    F(k; n + x, a) <= level. 0 if already certified; None if more than cap."""
    for x in range(cap + 1):
        if binom.cdf(k, n + x, a) <= level: return x
    return None


def check_reproduction(P, y, name_in_csv, n_splits=60):
    ref = pd.read_csv(C.PKG / "results/extra_models/r03_policies_groups_E6b.csv")
    ref = ref[(ref.model == name_in_csv) & (ref["mode"] == "certified") & (ref.harm == C.HARM_COST)].set_index("seed")
    mism = 0
    for s in range(n_splits):
        cal = split(y, s)
        b = g4_select(P[cal], y[cal], g4_candidates(P[cal], y[cal]))
        r = ref.loc[s]
        if b is None:
            mism += bool(r.found)
            continue
        a = pol(P[~cal], b[1], b[2], b[3]); de = np.mean(a[y[~cal] == DE] == DE)
        mism += not (r.found and abs(r.lr - b[1]) < 1e-12 and abs(r.la - b[2]) < 1e-12 and abs(r.tr - b[3]) < 1e-12
                     and abs(r.de_exec - de) < 1e-9)
    return mism
