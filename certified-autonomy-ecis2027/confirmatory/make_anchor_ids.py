"""Draw the 150 anchor calls (50 per label) from the 1,433 test-split calls scored by all three models.
They are re-scored to detect vendor model drift since the original run. Already run once; output: confirmatory/anchor_ids.json.
Usage: python confirmatory/make_anchor_ids.py results/scores_test_1436.csv confirmatory/anchor_ids.json"""
import json, sys
import numpy as np, pandas as pd
d = pd.read_csv(sys.argv[1]); d = d[d.error.isna()]
ok = d.groupby("dataset_idx").model.nunique(); ok = ok[ok == 3].index
lab = d[d.dataset_idx.isin(ok)].drop_duplicates("dataset_idx").set_index("dataset_idx").ground_truth.sort_index()
rng = np.random.default_rng(20261023)
ids = sorted(int(i) for c in ["direct_execute", "ask_human", "refuse"] for i in rng.choice(lab[lab == c].index.to_numpy(), 50, replace=False))
json.dump(ids, open(sys.argv[2], "w")); print(len(ids), "anchor ids ->", sys.argv[2])
