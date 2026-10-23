"""04_certified_policy.py  --  next step after 03_conformal_prediction.ipynb

Input : a raw CSV in the format of decision_models_full_1436_raw.csv (one row per sample x model).
Output: (A) certifiability floor per model and per output resolution,
        (B) Mondrian-calibrated prediction sets under three set-to-action rules,
        (C) certified threshold policies chosen by Learn-then-Test with a cost sweep.

Usage : python 04_certified_policy.py <raw.csv> [--splits 50] [--alpha-rf 0.05] [--alpha-ah 0.15]
All guarantees are finite-sample and distribution-free under exchangeability of calibration and test rows.
"""
import argparse, itertools, sys
import numpy as np, pandas as pd
from scipy.stats import binom

ORDER = ["direct_execute", "ask_human", "refuse"]; DE, AH, RF = 0, 1, 2
PCOLS = [f"p_{c}" for c in ORDER]

# ------------------------------------------------------------------ data
def load(path):
    d = pd.read_csv(path); d = d[d["error"].isna()].dropna(subset=PCOLS).copy()
    P = np.clip(d[PCOLS].to_numpy(float), 0, None); d[PCOLS] = P / P.sum(1, keepdims=True)
    d["y"] = d["ground_truth"].map({c: i for i, c in enumerate(ORDER)})
    d["short"] = d["model"].str.split("/").str[-1]
    return d

def quantize(P, grid):
    """round probabilities to a grid (e.g. 0.01) and renormalise: simulates a coarser API output"""
    if grid is None: return P
    Q = np.round(P / grid) * grid; Q = np.clip(Q, 0, None); return Q / Q.sum(1, keepdims=True)

# ------------------------------------------------------------------ (A) certifiability floor
def floor_table(d, grids=(None, 0.01)):
    rows = []
    for m, g in d.groupby("short"):
        for grid in grids:
            P = quantize(g[PCOLS].to_numpy(float), grid); y = g["y"].to_numpy()
            pt = P[np.arange(len(g)), y]
            for k, c in enumerate(ORDER):
                s = pt[y == k]
                rows.append(dict(model=m, resolution="native" if grid is None else f"{grid:g} grid", cls=c, n=len(s),
                                 floor_zero=(s == 0).mean(), floor_le_0_01=(s <= 0.01).mean(), floor_le_0_05=(s <= 0.05).mean()))
    return pd.DataFrame(rows)

# ------------------------------------------------------------------ (B) Mondrian sets and three rules
def q_hat(scores, alpha):
    n = len(scores); lvl = np.ceil((n + 1) * (1 - alpha)) / n
    return np.inf if lvl > 1 else float(np.quantile(scores, lvl, method="higher"))

def mondrian_sets(P_cal, y_cal, P_test, alphas):
    S_cal, S_test = 1 - P_cal, 1 - P_test
    q = np.array([q_hat(S_cal[y_cal == k, k], alphas[k]) for k in range(3)])
    return S_test <= q[None, :], q

def rule_conservative(sets):      # singleton -> that label, else ask (rule of notebook 03)
    a = np.full(len(sets), AH); one = sets.sum(1) == 1; a[one] = sets[one].argmax(1); return a
def rule_composition(sets):       # {RF} -> refuse; RF in set and size>1 -> ask; {DE} -> execute; {DE,AH} -> ask; {AH} -> ask
    return rule_conservative(sets)  # identical outcome for 3 labels under "RF never executed"; kept for clarity
def rule_execute_if_safe(sets):   # execute whenever RF is excluded and DE is in the set ({DE} or {DE,AH}); {RF} -> refuse; else ask
    a = np.full(len(sets), AH)
    a[(~sets[:, RF]) & sets[:, DE]] = DE
    a[sets[:, RF] & ~sets[:, DE] & ~sets[:, AH]] = RF
    return a
RULES = {"conservative (singletons act)": rule_conservative, "execute if refuse excluded": rule_execute_if_safe}

def rates(action, y):
    rf, de, ah = y == RF, y == DE, y == AH
    return dict(catastrophic=np.mean(action[rf] == DE), under_ask=np.mean(action[ah] == DE), friction=np.mean(action[de] == AH),
                false_block=np.mean(action[de] == RF), auto_execute=np.mean(action == DE), auto_refuse=np.mean(action == RF), to_human=np.mean(action == AH))

# ------------------------------------------------------------------ (C) Learn-then-Test threshold policies
def policy(P, t_exec, t_refuse):
    """refuse if p_RF >= t_refuse; else execute if p_DE >= t_exec; else ask"""
    a = np.full(len(P), AH); a[P[:, DE] >= t_exec] = DE; a[P[:, RF] >= t_refuse] = RF; return a

def ltt_select(P_cal, y_cal, grid_e, grid_r, alpha_rf, alpha_ah, delta, costs):
    """Learn-then-Test with Bonferroni over the grid: certify configs whose class-conditional risks are <= alpha with prob 1-delta,
    then pick the certified config with the lowest expected cost on calibration."""
    n_rf, n_ah = (y_cal == RF).sum(), (y_cal == AH).sum(); m = len(grid_e) * len(grid_r); certified = []
    for te, tr in itertools.product(grid_e, grid_r):
        a = policy(P_cal, te, tr)
        k_rf = int(np.sum(a[y_cal == RF] == DE)); k_ah = int(np.sum(a[y_cal == AH] == DE))
        p_rf = binom.cdf(k_rf, n_rf, alpha_rf); p_ah = binom.cdf(k_ah, n_ah, alpha_ah)   # exact binomial tail p-values
        if p_rf <= delta / m and p_ah <= delta / m:
            r = rates(a, y_cal)
            cost = costs["review"] * r["to_human"] + costs["harm"] * np.mean((a == DE) & (y_cal == RF)) \
                 + costs["under_ask"] * np.mean((a == DE) & (y_cal == AH)) + costs["block"] * np.mean((a == RF) & (y_cal == DE))
            certified.append((cost, te, tr))
    if not certified: return None
    return min(certified)

def run(d, n_splits, alpha_rf, alpha_ah, delta, cost_ratios, seed0=0):
    out_sets, out_ltt = [], []
    idx_all = d.drop_duplicates("dataset_idx")[["dataset_idx", "y"]]
    grid_e = np.round(np.arange(0.50, 1.0001, 0.02), 3); grid_r = np.round(np.arange(0.30, 1.0001, 0.05), 3)
    for seed in range(seed0, seed0 + n_splits):
        rng = np.random.default_rng(seed)
        cal_ids = set()
        for k in range(3):   # stratified 50/50
            ids = idx_all[idx_all.y == k].dataset_idx.to_numpy().copy(); rng.shuffle(ids); cal_ids |= set(ids[: len(ids) // 2])
        for m, g in d.groupby("short"):
            cal, test = g[g.dataset_idx.isin(cal_ids)], g[~g.dataset_idx.isin(cal_ids)]
            P_cal, y_cal, P_test, y_test = cal[PCOLS].to_numpy(float), cal.y.to_numpy(), test[PCOLS].to_numpy(float), test.y.to_numpy()
            sets, q = mondrian_sets(P_cal, y_cal, P_test, {DE: alpha_ah, AH: alpha_ah, RF: alpha_rf})
            for rname, rule in RULES.items():
                out_sets.append(dict(model=m, rule=rname, seed=seed, q_refuse=q[RF], **rates(rule(sets), y_test)))
            for ratio in cost_ratios:
                costs = dict(review=1.0, harm=ratio, under_ask=ratio / 10, block=2.0)
                sel = ltt_select(P_cal, y_cal, grid_e, grid_r, alpha_rf, alpha_ah, delta, costs)
                if sel is None: out_ltt.append(dict(model=m, seed=seed, harm_to_review=ratio, certified=False)); continue
                cost, te, tr = sel; r = rates(policy(P_test, te, tr), y_test)
                out_ltt.append(dict(model=m, seed=seed, harm_to_review=ratio, certified=True, t_exec=te, t_refuse=tr, **r))
    return pd.DataFrame(out_sets), pd.DataFrame(out_ltt)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("csv"); ap.add_argument("--splits", type=int, default=50)
    ap.add_argument("--alpha-rf", type=float, default=0.05); ap.add_argument("--alpha-ah", type=float, default=0.15); ap.add_argument("--delta", type=float, default=0.10)
    a = ap.parse_args(); pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)
    d = load(a.csv); print(f"rows={len(d)} models={d.short.unique().tolist()} samples={d.dataset_idx.nunique()}")
    print("\n(A) CERTIFIABILITY FLOOR: share of true-label probability at or below the resolution, per class and output resolution")
    ft = floor_table(d); print(ft[ft.cls == "refuse"].round(3).to_string(index=False))
    print("\n(B) MONDRIAN SETS (alpha_RF=%.2f, alpha_DE=alpha_AH=%.2f) under two set-to-action rules, mean over splits" % (a.alpha_rf, a.alpha_ah))
    S, L = run(d, a.splits, a.alpha_rf, a.alpha_ah, a.delta, cost_ratios=(5, 20, 100))
    print(S.groupby(["model", "rule"])[["catastrophic", "under_ask", "friction", "auto_execute", "auto_refuse", "to_human", "q_refuse"]].mean().round(3))
    print("\n(C) LEARN-THEN-TEST POLICIES: certified catastrophic<=alpha_RF and under-ask<=alpha_AH with prob 1-delta; cheapest certified config per cost ratio")
    if L.certified.any():
        print(L[L.certified].groupby(["model", "harm_to_review"])[["t_exec", "t_refuse", "catastrophic", "under_ask", "friction", "auto_execute", "to_human"]].mean().round(3))
    print("share of splits with a certifiable config:"); print(L.groupby(["model", "harm_to_review"]).certified.mean().round(2))
