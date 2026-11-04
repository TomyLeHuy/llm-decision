"""Phase C: field-level occlusion by REMOVAL (M1) on all 1,433 paper calls x 5 models, 8 workers.

  python experiments/c_occlusion.py query [--dry]   baseline + one variant per unit; jev, microsoft, liquid: 3 repeats
                                                    (repeat keys c0, c1, c2); gpt, pplx: 1 run. Tag 'phase_c'.
  python experiments/c_occlusion.py collect         -> results/c_long.csv.gz (one row per call x model x variant) and
                                                    results/c_noise.json (empirical null for the noisy models)
Outcomes per row: ok | provider_refusal (vendor declined to answer; kept as its own outcome, never as a missing value)
| missing. For the noisy models p_* are the MEAN over the 3 repeats and sd_* the SD; an attribution is the difference
of two such means. Above-noise rule: TV(baseline mean, masked mean) > q95 of the same statistic under the empirical null
(residual bootstrap, see null_tv); for gpt and pplx, which are deterministic (S3, pilot check), TV > 0.005.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config as C  # noqa: E402
from src import io as xio  # noqa: E402
from src import units as U  # noqa: E402
from src import query as Q  # noqa: E402

R = C.RESULTS
TAG = "phase_c"
NOISY = ["typesafe/jev-1.14", "microsoft/microsoft-decision-1", "liquid/d1-ultra"]
DET = ["openai/gpt-6-luna-decisions"]
# pplx was deterministic until ~15:00 UTC on 3 Nov and stochastic from ~21:00 UTC (det_check_evening1): it is
# queried separately once its design (1 run or 3 repeats) is decided; --pplx-reps 1|3 adds it to DET or NOISY
PPLX = "perplexity/pplx-decider-v1.2"
REPS = ["c0", "c1", "c2"]
LABEL = {"microsoft": "microsoft (post-drift)"}


def variants(ids):
    data = xio.load_split("test")
    out = []
    for i in ids:
        s = data[i]
        _, _, units = U.parse(s)
        out.append(dict(dataset_idx=int(i), unit=None, unit_type="baseline", n_units=len(units), sample=U.variant(s)))
        for u in units:
            out.append(dict(dataset_idx=int(i), unit=u, unit_type=U.unit_type(u), n_units=len(units),
                            sample=U.variant(s, u, "remove")))
    return out


def jobs_for(vs):
    det = [dict(model=m, sample=v["sample"], tag=TAG, meta=dict(dataset_idx=v["dataset_idx"], unit=v["unit"]))
           for v in vs for m in DET]
    noisy = [dict(model=m, sample=v["sample"], tag=TAG, repeat=r, meta=dict(dataset_idx=v["dataset_idx"], unit=v["unit"]))
             for r in REPS for v in vs for m in NOISY]
    return det, noisy


def set_pplx(reps):
    if reps == 1 and PPLX not in DET: DET.append(PPLX)
    if reps == 3 and PPLX not in NOISY: NOISY.append(PPLX)


def query(dry, only=None):
    ids = xio.common_ids("test", "paper")
    vs = variants(ids)
    det, noisy = jobs_for(vs)
    if only:
        det = [j for j in det if j["model"] in only]; noisy = [j for j in noisy if j["model"] in only]
    print(f"calls {len(ids)} | variants {len(vs)} | deterministic jobs {len(det)} | noisy jobs {len(noisy)}", flush=True)
    q = Q.Querier()
    t0 = time.time()
    q.run(det, workers=8, dry_run=dry, require_tag=TAG)
    q.run(noisy, workers=8, dry_run=dry, require_tag=TAG)
    print(f"done in {(time.time() - t0) / 60:.1f} min | spent USD {q.spent:.4f}", flush=True)


def collect():
    ids = xio.common_ids("test", "paper")
    vs = variants(ids)
    q = Q.Querier()
    st = xio.sample_table("test").set_index("dataset_idx")
    rows = []
    for v in vs:
        st_str = Q.state_string(v["sample"])
        for m in DET + NOISY:
            reps = [None] if m in DET else REPS
            recs = [q.cache.get(Q.cache_key(m, st_str, r)) for r in reps]
            outs = [Q.outcome(r) for r in recs]
            ok = [r for r, o in zip(recs, outs) if o == "ok"]
            P = np.array([[r["probabilities"][c] for c in C.CLASSES] for r in ok]) if ok else np.full((0, 3), np.nan)
            o = "ok" if ok else ("provider_refusal" if "provider_refusal" in outs else "missing")
            row = dict(dataset_idx=v["dataset_idx"], model=C.SHORT[m], unit=v["unit"], unit_type=v["unit_type"],
                       n_units=v["n_units"], outcome=o, n_reps=len(ok),
                       n_provider_refusal=sum(x == "provider_refusal" for x in outs),
                       cost_usd=sum(float(r.get("cost_usd") or 0) for r in recs if r))
            for k, c in enumerate(C.CLASSES):
                row[f"p_{c}"] = float(P[:, k].mean()) if len(P) else np.nan
                row[f"sd_{c}"] = float(P[:, k].std(ddof=1)) if len(P) > 1 else (0.0 if len(P) == 1 and m in DET else np.nan)
                for j, r in enumerate(reps):
                    if r is not None:
                        row[f"p_{c}_{r}"] = float(P[j, k]) if j < len(P) and outs[j] == "ok" else np.nan
            row["choice"] = C.CLASSES[int(np.argmax(P.mean(0)))] if len(P) else None
            rows.append(row)
    d = pd.DataFrame(rows)
    d["model_label"] = d.model.map(lambda m: LABEL.get(m, m))
    for col in ["ground_truth", "multi_turn", "service", "service_type", "risk_class", "risk_family", "scored_tool", "labelled_tool"]:
        d[col] = d.dataset_idx.map(st[col])
    base = d[d.unit_type == "baseline"].set_index(["dataset_idx", "model"])
    key = list(zip(d.dataset_idx, d.model))
    for c in C.CLASSES:
        d[f"base_{c}"] = base[f"p_{c}"].reindex(key).to_numpy()
        d[f"attr_{c}"] = d[f"base_{c}"] - d[f"p_{c}"]
    d["base_outcome"] = base["outcome"].reindex(key).to_numpy()
    d["base_choice"] = base["choice"].reindex(key).to_numpy()
    d["flip"] = (d.choice != d.base_choice) & d.choice.notna() & d.base_choice.notna()
    d["tv"] = d[[f"attr_{c}" for c in C.CLASSES]].abs().sum(axis=1, min_count=3) / 2
    nul = null_tv(d)
    d["thr_tv"] = d.model.map(lambda m: nul[m]["q95_tv"])
    d["above"] = d.tv > d.thr_tv
    d.to_csv(R / "c_long.csv.gz", index=False)
    xio.dump_json(nul, R / "c_noise.json")
    print("rows", len(d), "| outcomes", d.groupby(["model", "outcome"]).size().to_dict())
    print("null", json.dumps(nul, indent=1))


def null_tv(d, B=20000, seed=C.SEED):
    """Empirical null of TV(mean of 3, mean of 3) for two inputs with NO true difference: residual vectors
    (p_rep - mean over reps) are pooled per model, inflated by sqrt(3/2) (3-sample residuals are shrunk), and two
    sets of 3 are drawn with replacement. Keeps the heavy tails and the coupling of the three classes."""
    rng = np.random.default_rng(seed)
    out = {}
    for m in [C.SHORT[x] for x in DET]:
        out[m] = dict(q95_tv=0.005, rule="deterministic: any change of at least half a 0.01 step")
    for m in [C.SHORT[x] for x in NOISY]:
        g = d[(d.model == m) & (d.n_reps == 3)]
        X = np.stack([g[[f"p_{c}_{r}" for c in C.CLASSES]].to_numpy() for r in REPS], 1)  # (n, 3 reps, 3 classes)
        res = (X - X.mean(1, keepdims=True)).reshape(-1, 3) * np.sqrt(3 / 2)
        a = res[rng.integers(len(res), size=(B, 3))].mean(1)
        b = res[rng.integers(len(res), size=(B, 3))].mean(1)
        tv = np.abs(a - b).sum(1) / 2
        single = np.abs(res[rng.integers(len(res), size=B)] - res[rng.integers(len(res), size=B)]).sum(1) / 2
        out[m] = dict(q95_tv=float(np.quantile(tv, 0.95)), q99_tv=float(np.quantile(tv, 0.99)),
                      q95_tv_single_query=float(np.quantile(single, 0.95)),
                      residual_sd={c: float(res[:, k].std()) for k, c in enumerate(C.CLASSES)}, n_variants=int(len(g)),
                      rule="TV of two 3-repeat means above the 95th percentile of the residual-bootstrap null")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["query", "collect"])
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--pplx-reps", type=int, choices=[0, 1, 3], default=0, help="0 = leave pplx out (pending)")
    ap.add_argument("--only", nargs="*", help="query only these full model ids")
    a = ap.parse_args()
    set_pplx(a.pplx_reps)
    query(a.dry, a.only) if a.step == "query" else collect()
