"""Noise floor on MASKED inputs (Phase B pilot check): 40 random pilot variants (20 M1, 20 M2; seed 0) x 5 models,
one fresh repeat each (repeat key 'b_pilot_rep1'), compared with the pilot response. Output: results/b_pilot_noise_check.*"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config as C  # noqa: E402
from src import query as Q  # noqa: E402
from experiments.b_occlusion import build_variants  # noqa: E402
from experiments.p0_sanity import pilot_ids  # noqa: E402


def main():
    v = [x for x in build_variants(pilot_ids()) if x["sample"] is not None]
    rng = np.random.default_rng(C.SEED)
    pick = []
    for strat in ["M1_remove", "M2_replace"]:
        c = [x for x in v if x["strategy"] == strat]
        pick += [c[k] for k in rng.choice(len(c), 20, replace=False)]
    q = Q.Querier()
    jobs = [dict(model=m, sample=x["sample"], tag="b_pilot_noise", repeat="b_pilot_rep1",
                 meta=dict(dataset_idx=int(x["dataset_idx"]), unit=x["unit"], strategy=x["strategy"])) for x in pick for m in C.MODELS]
    res = q.run(jobs)
    rows = []
    for j, r in zip(jobs, res):
        o = q.lookup(j["model"], j["sample"])  # pilot response
        if r is None or o is None:
            continue
        rows.append(dict(model=C.SHORT[j["model"]], **j["meta"],
                         **{f"d_{c}": abs(r["probabilities"][c] - o["probabilities"][c]) for c in C.CLASSES},
                         choice_same=r["choice"] == o["choice"]))
    d = pd.DataFrame(rows)
    d.to_csv(C.RESULTS / "b_pilot_noise_check.csv", index=False)
    d["d_max"] = d[[f"d_{c}" for c in C.CLASSES]].max(axis=1)
    s3 = json.load(open(C.RESULTS / "p0_s3_summary.json"))["rms_within_sd_by_model"]
    t = d.groupby("model").agg(n=("d_max", "size"), identical=("d_max", lambda x: float(np.mean(x < 1e-6))),
                               mean_abs=("d_max", "mean"), max_abs=("d_max", "max"), choice_same=("choice_same", "mean"))
    # implied SD of one query from |difference of two queries|: E|X-Y| = 2*SD/sqrt(pi) for normal noise
    for c in C.CLASSES:
        t[f"implied_sd_{c}"] = d.groupby("model")[f"d_{c}"].mean() * np.sqrt(np.pi) / 2
        t[f"s3_sd_{c}"] = t.index.map(lambda m: s3[m][c])
    pd.set_option("display.width", 250)
    print(t.round(4).to_string())
    json.dump(t.round(5).reset_index().to_dict("records"), open(C.RESULTS / "b_pilot_noise_check.json", "w"), indent=1)


if __name__ == "__main__":
    main()
