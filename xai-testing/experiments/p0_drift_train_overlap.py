"""S2 add-on (no API calls): for S2 test calls whose scored state string is identical to a train sample's, compare the
fresh S2 probabilities with the 23 Oct train-run scores (which record response_model) and with the October test run.
Localises any drift in time: October test run -> 23 Oct train run -> today."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import config as C
from src import io as xio, query as Q

s2 = pd.read_csv(C.RESULTS / "p0_s2_drift.csv")
test, train = xio.load_split("test"), xio.load_split("train")
tstate = {i: Q.state_string(test[i]) for i in s2.dataset_idx.unique()}
idx = {}
for j, s in enumerate(train):
    idx.setdefault(Q.state_string(s), j)
tr = xio.load_scores("train"); tr = tr[~tr.failed].set_index(["dataset_idx", "short"])
rows = []
for _, r in s2.iterrows():
    j = idx.get(tstate[r.dataset_idx])
    if j is None or (j, r.model) not in tr.index: continue
    t = tr.loc[(j, r.model)]
    rows.append(dict(dataset_idx=r.dataset_idx, train_idx=j, model=r.model, multi_turn=r.multi_turn,
                     today_vs_train=max(abs(r[f"new_{c}"] - t[f"p_{c}"]) for c in C.CLASSES),
                     oct_vs_train=max(abs(r[f"old_{c}"] - t[f"p_{c}"]) for c in C.CLASSES),
                     today_vs_oct=r.max_absdiff, train_version=t.response_model))
d = pd.DataFrame(rows); d.to_csv(C.RESULTS / "p0_s2_train_overlap.csv", index=False)
tab = d.groupby("model").agg(n=("dataset_idx", "size"), today_vs_train_mean=("today_vs_train", "mean"),
                             today_vs_train_max=("today_vs_train", "max"), oct_vs_train_mean=("oct_vs_train", "mean"),
                             oct_vs_train_max=("oct_vs_train", "max"), today_vs_oct_mean=("today_vs_oct", "mean"))
print(tab.round(4).to_string())
json.dump(tab.round(5).reset_index().to_dict("records"), open(C.RESULTS / "p0_s2_train_overlap.json", "w"), indent=1)
