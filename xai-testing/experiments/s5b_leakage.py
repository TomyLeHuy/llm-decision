"""Step 5b: leakage check of the corrected multi-turn input (Tomy, 3 Nov).

Four inputs for the 123 test + 733 train multi-turn calls, all queried in the same run with the same design:
  truncated   the paper's input (first user message only)
  full        full conversation incl. reviewer turns (decisions on the earlier steps = labels of those steps; upper bound)
  users_only  (a) all user turns, no reviewer turns
  final_only  (b) only the final user message (labelled call + prior_steps; no user_task in any of these calls)
jev, microsoft, liquid, pplx: 3 fresh repeats each (repeat keys mt0..mt2), mean used, SD reported; gpt: 1 run.

  python experiments/s5b_leakage.py query      -> runs/multiturn_variants/<variant>/<split>_scores.jsonl
  python experiments/s5b_leakage.py compare    -> results/s5b_leakage.csv / .json (side-by-side table)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config as C  # noqa: E402
from src import io as xio  # noqa: E402
from src import units as U  # noqa: E402
from src import query as Q  # noqa: E402
from src import gate as G  # noqa: E402
from src.scoring_fixed import VARIANTS  # noqa: E402

OUTDIR = C.XAI / "runs/multiturn_variants"
NOISY = ["typesafe/jev-1.14", "microsoft/microsoft-decision-1", "liquid/d1-ultra", "perplexity/pplx-decider-v1.2"]
DET = ["openai/gpt-6-luna-decisions"]
REPS = ["mt0", "mt1", "mt2"]
TAG = "leak"


def make(v, s):
    return U.variant(s) if VARIANTS[v] is None else VARIANTS[v](s)


def query():
    q = Q.Querier()
    for split in ["test", "train"]:
        data = xio.load_split(split)
        st = xio.sample_table(split)
        ids = st.dataset_idx[st.multi_turn].tolist()
        for v in VARIANTS:
            smp = {i: make(v, data[i]) for i in ids}
            det = [dict(model=m, sample=smp[i], tag=TAG, meta=dict(split=split, dataset_idx=i, variant=v)) for i in ids for m in DET]
            noisy = [dict(model=m, sample=smp[i], tag=TAG, repeat=r, meta=dict(split=split, dataset_idx=i, variant=v))
                     for r in REPS for i in ids for m in NOISY]
            q.run(det, workers=8)          # gpt is deterministic: a cached answer of the identical input is valid
            q.run(noisy, workers=8, require_tag=TAG)
            write(q, split, v, data, ids, smp)
            print(f"[leak] {split} {v} done | spent USD {q.spent:.3f}", flush=True)


def write(q, split, v, data, ids, smp):
    path = OUTDIR / v / f"{split}_scores.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for i in ids:
            meta = data[i]["meta"]
            stt = Q.state_string(smp[i])
            for m in DET + NOISY:
                recs = [q.cache.get(Q.cache_key(m, stt, r)) for r in ([None] if m in DET else REPS)]
                outs = [Q.outcome(r) for r in recs]
                ok = [r for r, o in zip(recs, outs) if o == "ok"]
                P = np.array([[r["probabilities"][c] for c in C.CLASSES] for r in ok]) if ok else None
                o = "ok" if ok else ("provider_refusal" if "provider_refusal" in outs else "missing")
                mean = P.mean(0) if ok else [float("nan")] * 3
                pred = C.CLASSES[int(np.argmax(mean))] if ok else None
                rec = dict(dataset_idx=i, service=meta.get("service"), risk_class=meta.get("risk_class"), risk_family=meta.get("risk_family"),
                           ground_truth=meta.get("decision"), model=m, pred_choice=pred,
                           **{f"p_{c}": float(mean[k]) for k, c in enumerate(C.CLASSES)},
                           **{f"sd_{c}": (float(P[:, k].std(ddof=1)) if ok and len(P) > 1 else 0.0) for k, c in enumerate(C.CLASSES)},
                           prob_sum=float(np.sum(mean)) if ok else float("nan"), n_reps=len(ok), n_provider_refusal=outs.count("provider_refusal"),
                           cost_usd=sum(float(r.get("cost_usd") or 0) for r in recs if r), error=None if ok else o,
                           split=split, input=v, outcome=o, response_model=ok[0]["response_model"] if ok else None)
                f.write(json.dumps(rec) + "\n")


def auroc(pos, neg):
    if len(pos) == 0 or len(neg) == 0: return np.nan
    return float(mannwhitneyu(pos, neg, alternative="two-sided").statistic / (len(pos) * len(neg)))


def compare():
    rows, g4rows = [], []
    for v in VARIANTS:
        src = f"mt_{v}"
        for split in ["test", "train"]:
            P, meta = xio.prob_matrices(split, source=src)
            y = xio.label_index(meta.ground_truth); mt = meta.multi_turn.to_numpy(); svc = meta.service.to_numpy()
            for m in ["jev", "microsoft", "gpt", "liquid", "pplx", "mean"]:
                p = P[m]
                r = dict(variant=v, split=split, model=m)
                # dh_app / atbench as in A2: refusals of the service vs its non-refusals, all turns (every dh_app refusal is
                # multi-turn, every dh_app non-refusal single-turn, so only the refusals' scores differ between inputs)
                for g, sel in [("multi", mt), ("atbench_multi", mt & (svc == "atbench")), ("atbench", svc == "atbench"), ("dh_app", svc == "dh_app")]:
                    yy, pp = y[sel], p[sel]
                    r[f"auroc_{g}"] = auroc(pp[yy == 2, 2], pp[yy != 2, 2])
                    r[f"refusals_executed_{g}"] = float(np.mean(pp[yy == 2].argmax(1) == 0)) if (yy == 2).any() else np.nan
                    r[f"n_refusals_{g}"] = int((yy == 2).sum())
                r["refusals_p_le_001_multi"] = float(np.mean(p[mt & (y == 2), 2] <= 0.01))
                # G4 on the whole pool: per-service harm (one boundary for all services)
                b = G.g4_select(p, y, G.g4_candidates(p, y))
                if b is not None:
                    a = G.pol(p, b[1], b[2], b[3])
                    r["pool_safe"] = float(np.mean(a[y == 0] == 0))
                    for g, sel in [("multi", mt), ("atbench", svc == "atbench"), ("dh_app", svc == "dh_app")]:
                        r[f"pool_harm_{g}"] = float(np.mean(a[sel & (y == 2)] == 0))
                rows.append(r)
                if split == "test":  # A3b-style: G4 over the paper's 60 splits
                    for s in range(60):
                        cal = G.split(y, s)
                        b = G.g4_select(p[cal], y[cal], G.g4_candidates(p[cal], y[cal]))
                        if b is None:
                            g4rows.append(dict(variant=v, model=m, seed=s, found=False)); continue
                        a = G.pol(p, b[1], b[2], b[3]); te = ~cal
                        f = lambda sel, k: float(np.mean(a[sel & (y == k)] == 0))  # noqa: E731
                        g4rows.append(dict(variant=v, model=m, seed=s, found=True, safe=f(te, 0), harm=f(te, 2),
                                           harm_single=f(te & ~mt, 2), harm_multi=f(te & mt, 2), unconf=f(te, 1)))
    t = pd.DataFrame(rows)
    g4 = pd.DataFrame(g4rows)
    gs = g4[g4.found].groupby(["variant", "model"]).agg(safe=("safe", "mean"), harm=("harm", "mean"), harm_single=("harm_single", "mean"),
                                                      harm_multi=("harm_multi", "mean"),
                                                      over_harm=("harm", lambda x: float(np.mean(x > C.A_HARM))),
                                                      over_harm_multi=("harm_multi", lambda x: float(np.mean(x > C.A_HARM)))).reset_index()
    cs = g4.groupby(["variant", "model"]).found.mean().rename("certified_share").reset_index()
    gs = cs.merge(gs, on=["variant", "model"], how="left")  # pairs that never certify keep NaN autonomy
    t.to_csv(C.RESULTS / "s5b_leakage_scorers.csv", index=False); gs.to_csv(C.RESULTS / "s5b_leakage_g4.csv", index=False)
    g4.to_csv(C.RESULTS / "s5b_leakage_g4_splits.csv", index=False)
    # noise of the 3 repeats per variant (mean SD over multi-turn calls)
    sd = []
    for v in VARIANTS:
        for split in ["test", "train"]:
            d = pd.read_json(OUTDIR / v / f"{split}_scores.jsonl", lines=True)
            d = d[d.n_reps == 3]
            sd.append(dict(variant=v, split=split, **d.groupby("model")[["sd_refuse"]].mean().sd_refuse.round(4).to_dict()))
    out = dict(scorers=t.round(4).to_dict("records"), g4=gs.round(4).to_dict("records"), repeat_sd_refuse=sd)
    xio.dump_json(out, C.RESULTS / "s5b_leakage.json")
    pd.set_option("display.width", 250)
    order = list(VARIANTS)
    for col in ["auroc_multi", "refusals_executed_multi", "auroc_atbench_multi", "auroc_atbench", "auroc_dh_app", "refusals_executed_dh_app", "pool_harm_dh_app", "pool_harm_atbench"]:
        print(col); print(t.pivot_table(index=["split", "model"], columns="variant", values=col)[order].round(3).to_string())
    for col in ["certified_share", "safe", "harm_multi", "harm_single", "over_harm_multi", "over_harm"]:
        print("G4 60 splits", col); print(gs.pivot_table(index="model", columns="variant", values=col)[order].round(3).to_string())
    print(pd.DataFrame(sd).to_string())


if __name__ == "__main__":
    {"query": query, "compare": compare}[sys.argv[1]]()
