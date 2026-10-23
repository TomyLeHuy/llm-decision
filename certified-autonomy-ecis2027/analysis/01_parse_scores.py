"""Step 01: convert the scoring checkpoint (one JSON line per sample x model call) into the flat table all analyses read.
Usage: python analysis/01_parse_scores.py data/scores/test_1436_scores.jsonl results/scores_test_1436.csv
Keeps the last record per (dataset_idx, model), as a resumed run may contain retries."""
import json, sys
import pandas as pd
src, out = sys.argv[1], sys.argv[2]
rows = {}
with open(src, encoding="utf-8") as f:
    for line in f:
        if line.strip():
            r = json.loads(line); rows[(r["dataset_idx"], r["model"])] = r
cols = ["dataset_idx", "service", "risk_class", "risk_family", "ground_truth", "model", "pred_choice", "is_match",
        "p_direct_execute", "p_ask_human", "p_refuse", "prob_sum", "latency_ms", "cost_usd", "error"]
d = pd.DataFrame(list(rows.values()))[cols]
d.to_csv(out, index=False)
print(f"{len(d)} calls, {d.error.notna().sum()} failed, {d.dataset_idx.nunique()} samples -> {out}")
