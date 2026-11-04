"""Exact field-level Shapley values for calls with <= 8 units, gpt only (Tomy, 3 Nov: pplx turned non-deterministic on
3 Nov, see det_check_evening1/nov4; gpt re-verified deterministic on 4 Nov 02:29 UTC, det_check_nov4: 60/60 identical
on repeat and to Step 4), so one query per coalition suffices.

  python experiments/e_shapley.py query      every subset S of the units: the scored message with ONLY the units in S
                                             present (the others removed as in M1); chunks of 50 calls, 8 workers
  python experiments/e_shapley.py analyze    -> results/e_shapley_values.csv.gz, e_shapley_pairs.csv.gz, e_shapley_summary.json

Value function v_c(S) = p_c(input with units S). Shapley phi_i and the pairwise Shapley interaction index I_ij are
computed exactly from all 2^n values. Occlusion (leave-one-out) in the same convention: LOO_i = v(N) - v(N minus i),
i.e. Phase C's attribution. A call is excluded for a model if any coalition returned provider_refusal (counted).
"""
import argparse
import json
import math
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config as C  # noqa: E402
from src import io as xio  # noqa: E402
from src import units as U  # noqa: E402
from src import query as Q  # noqa: E402

R = C.RESULTS
MODELS = ["openai/gpt-6-luna-decisions"]
MAXU = 8
TAG = "shapley"
EPS = 0.005  # half a 0.01 step: the resolution of a deterministic model's change


def calls():
    data = xio.load_split("test")
    ids = [int(i) for i in xio.common_ids("test", "paper") if len(U.parse(data[i])[2]) <= MAXU]
    return data, ids


def coalition_sample(sample, obj, fmt, units, mask):
    o = obj
    for k, u in enumerate(units):
        if not (mask >> k) & 1:
            o = U.apply_mask(o, u, "remove")
    return U.with_content(sample, U.render(o, fmt))


def query():
    data, ids = calls()
    q = Q.Querier()
    t0 = time.time()
    print(f"calls {len(ids)} | coalitions {sum(2 ** len(U.parse(data[i])[2]) for i in ids)} x {len(MODELS)} models", flush=True)
    for c0 in range(0, len(ids), 50):
        jobs = []
        for i in ids[c0:c0 + 50]:
            s = data[i]; obj, fmt, units = U.parse(s)
            for mask in range(2 ** len(units)):
                smp = coalition_sample(s, obj, fmt, units, mask)
                for m in MODELS:
                    jobs.append(dict(model=m, sample=smp, tag=TAG, meta=dict(dataset_idx=i, mask=mask)))
        q.run(jobs, workers=8, verbose=False)  # deterministic models: any cached answer of the same input is valid
        print(f"[shapley] calls {min(c0 + 50, len(ids))}/{len(ids)} | {(time.time() - t0) / 60:.0f} min | spent USD {q.spent:.3f}", flush=True)


def shapley(V, n):
    """V: (2^n, k) values indexed by bitmask. Returns phi (n, k) and interaction I (n, n, k)."""
    fact = [math.factorial(x) for x in range(n + 1)]
    pc = np.array([bin(m).count("1") for m in range(2 ** n)])
    phi = np.zeros((n, V.shape[1]))
    for i in range(n):
        b = 1 << i
        for S in range(2 ** n):
            if S & b: continue
            s = pc[S]
            phi[i] += fact[s] * fact[n - s - 1] / fact[n] * (V[S | b] - V[S])
    I = np.zeros((n, n, V.shape[1]))
    for i, j in combinations(range(n), 2):
        bi, bj = 1 << i, 1 << j
        for S in range(2 ** n):
            if S & bi or S & bj: continue
            s = pc[S]
            w = fact[s] * fact[n - s - 2] / fact[n - 1]
            I[i, j] += w * (V[S | bi | bj] - V[S | bi] - V[S | bj] + V[S])
        I[j, i] = I[i, j]
    return phi, I


def analyze():
    data, ids = calls()
    q = Q.Querier()
    st = xio.sample_table("test").set_index("dataset_idx")
    vals, pairs, summ, excluded, prrows = [], [], [], {}, []
    for i in ids:
        s = data[i]; obj, fmt, units = U.parse(s); n = len(units)
        for m in MODELS:
            V = np.full((2 ** n, 3), np.nan); bad = 0
            for mask in range(2 ** n):
                r = q.lookup(m, coalition_sample(s, obj, fmt, units, mask))
                o = Q.outcome(r)
                prrows.append(dict(dataset_idx=i, mask=mask, n_units=n, n_present=bin(mask).count("1"), outcome=o,
                                   label=st.ground_truth[i], multi_turn=bool(st.multi_turn[i]),
                                   **{f"has_{t}": any(((mask >> k) & 1) and U.unit_type(u) == t for k, u in enumerate(units))
                                      for t in ["user_task", "user_memory", "prior_steps", "service", "tool", "description", "tool_args"]}))
                if o != "ok":
                    bad += 1; continue
                V[mask] = [r["probabilities"][c] for c in C.CLASSES]
            ms = C.SHORT[m]
            if bad:
                excluded.setdefault(ms, []).append(dict(dataset_idx=i, n_bad=bad)); continue
            phi, I = shapley(V, n)
            full = 2 ** n - 1
            loo = np.array([V[full] - V[full & ~(1 << k)] for k in range(n)])
            eff_err = float(np.abs(phi.sum(0) - (V[full] - V[0])).max())
            for k, u in enumerate(units):
                vals.append(dict(dataset_idx=i, model=ms, unit=u, unit_type=U.unit_type(u), n_units=n,
                                 **{f"phi_{c}": phi[k, j] for j, c in enumerate(C.CLASSES)},
                                 **{f"loo_{c}": loo[k, j] for j, c in enumerate(C.CLASSES)}))
            for a, b in combinations(range(n), 2):
                pairs.append(dict(dataset_idx=i, model=ms, unit_a=units[a], unit_b=units[b],
                                  types="|".join(sorted([U.unit_type(units[a]), U.unit_type(units[b])])),
                                  **{f"I_{c}": I[a, b, j] for j, c in enumerate(C.CLASSES)}))
            # per-call comparison, on p_refuse and on the class with the largest total movement
            r = dict(dataset_idx=i, model=ms, n_units=n, multi_turn=bool(st.multi_turn[i]), label=st.ground_truth[i], eff_err=eff_err,
                     v_full_refuse=V[full, 2], v_empty_refuse=V[0, 2])
            for j, c in enumerate(C.CLASSES):
                p_, l_ = phi[:, j], loo[:, j]
                r[f"rho_{c}"] = spearmanr(p_, l_).statistic if np.ptp(p_) > 0 and np.ptp(l_) > 0 else np.nan
                r[f"top_same_{c}"] = int(np.argmax(np.abs(p_)) == np.argmax(np.abs(l_))) if np.abs(p_).max() > EPS else np.nan
                r[f"sign_conflict_{c}"] = int(np.sum((np.abs(p_) > EPS) & (np.abs(l_) > EPS) & (np.sign(p_) != np.sign(l_))))
                r[f"hidden_{c}"] = int(np.sum((np.abs(l_) <= EPS) & (np.abs(p_) > 0.05)))   # redundant: LOO ~ 0, Shapley sizeable
                r[f"interaction_ratio_{c}"] = float(np.abs(np.triu(I[:, :, j], 1)).sum() / max(np.abs(p_).sum(), 1e-12))
                r[f"loo_sum_{c}"] = float(l_.sum()); r[f"phi_sum_{c}"] = float(p_.sum())
            summ.append(r)
    pr = pd.DataFrame(prrows); pr.to_csv(R / "e_provider_refusal_coalitions.csv.gz", index=False)
    v = pd.DataFrame(vals); p = pd.DataFrame(pairs); s = pd.DataFrame(summ)
    v.to_csv(R / "e_shapley_values.csv.gz", index=False); p.to_csv(R / "e_shapley_pairs.csv.gz", index=False)
    s.to_csv(R / "e_shapley_calls.csv", index=False)
    s["turn"] = s.multi_turn.map({False: "single_turn", True: "multi_turn"})
    agg = {}
    for c in C.CLASSES:
        agg.update({f"rho_{c}": (f"rho_{c}", "median"), f"top_same_{c}": (f"top_same_{c}", "mean"),
                    f"calls_with_sign_conflict_{c}": (f"sign_conflict_{c}", lambda x: float(np.mean(x > 0))),
                    f"calls_with_hidden_{c}": (f"hidden_{c}", lambda x: float(np.mean(x > 0))),
                    f"interaction_ratio_{c}": (f"interaction_ratio_{c}", "median")})
    t = s.groupby(["model", "turn"]).agg(n=("dataset_idx", "size"), max_eff_err=("eff_err", "max"), **agg).reset_index()
    # which pairs interact most (mean |I_refuse| by unit-type pair)
    p["absI"] = p[[f"I_{c}" for c in C.CLASSES]].abs().max(axis=1)
    tp = p.groupby(["model", "types"]).agg(n=("absI", "size"), mean_absI=("absI", "mean"),
                                           share_gt_005=("absI", lambda x: float(np.mean(x > 0.05)))).reset_index()
    tp = tp.sort_values(["model", "mean_absI"], ascending=[True, False])
    tp.to_csv(R / "e_shapley_pair_types.csv", index=False)
    # hidden units by type (LOO ~ 0 but |phi| > 0.05 on any class)
    v["hidden"] = ((v[[f"loo_{c}" for c in C.CLASSES]].abs().to_numpy() <= EPS) & (v[[f"phi_{c}" for c in C.CLASSES]].abs().to_numpy() > 0.05)).any(axis=1)
    hid = v.groupby(["model", "unit_type"]).agg(n=("hidden", "size"), hidden=("hidden", "mean")).reset_index()
    out = dict(n_calls=len(ids), excluded={k: len(x) for k, x in excluded.items()}, per_call=t.round(4).to_dict("records"),
               pair_types_top=tp.groupby("model").head(6).round(4).to_dict("records"), hidden_by_type=hid.round(4).to_dict("records"))
    xio.dump_json(out, R / "e_shapley_summary.json")
    pd.set_option("display.width", 250)
    print(t.round(3).to_string()); print(tp.groupby("model").head(6).round(3).to_string()); print(hid.round(3).to_string())
    print("excluded", out["excluded"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("step", choices=["query", "analyze"]); a = ap.parse_args()
    query() if a.step == "query" else analyze()
