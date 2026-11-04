"""Determinism re-check for gpt and pplx (exploratory, added after pplx was found to differ between afternoon and
evening of 3 Nov): 60 test calls (seed 1) x 2 fresh repeats, compared with each other and with tonight's Step-4 run.
Usage: python experiments/det_check.py <label>   (label goes into the repeat key, e.g. evening1)"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import config as C
from src import io as xio, units as U, query as Q

lab = sys.argv[1]
ids = np.random.default_rng(1).choice(xio.common_ids("test", "paper"), 60, replace=False)
data = xio.load_split("test")
M = ["openai/gpt-6-luna-decisions", "perplexity/pplx-decider-v1.2"]
jobs = [dict(model=m, sample=U.variant(data[i]), tag=f"detcheck_{lab}", repeat=f"{lab}_{r}", meta=dict(dataset_idx=int(i)))
        for r in range(2) for i in ids for m in M]
res = Q.Querier().run(jobs, workers=8)
today = pd.read_json(C.XAI / "runs/today/test_scores.jsonl", lines=True).set_index(["dataset_idx", "model"])
rows = []
for j, r in zip(jobs, res):
    if Q.outcome(r) != "ok": continue
    t = today.loc[(j["meta"]["dataset_idx"], j["model"])]
    rows.append(dict(model=C.SHORT[j["model"]], idx=j["meta"]["dataset_idx"], rep=j["repeat"], queried_at=r["queried_at"],
                     **{c: r["probabilities"][c] for c in C.CLASSES}, d_step4=max(abs(r["probabilities"][c] - t[f"p_{c}"]) for c in C.CLASSES)))
d = pd.DataFrame(rows)
w = d.pivot_table(index=["model", "idx"], columns="rep", values=C.CLASSES)
d01 = np.abs(w.xs(f"{lab}_0", axis=1, level=1) - w.xs(f"{lab}_1", axis=1, level=1)).max(axis=1)
out = pd.DataFrame({"rep0_vs_rep1_identical": d01.lt(1e-6).groupby(level=0).mean(), "rep0_vs_rep1_max": d01.groupby(level=0).max(),
                    "vs_step4_identical": d.assign(z=d.d_step4 < 1e-6).groupby("model").z.mean(), "vs_step4_max": d.groupby("model").d_step4.max()})
d.to_csv(C.RESULTS / f"det_check_{lab}.csv", index=False)
print(out.round(4).to_string()); print("window", d.queried_at.min(), d.queried_at.max())
