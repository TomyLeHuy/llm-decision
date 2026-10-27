"""Check a scoring run before sending it on.

Usage
  python check_run.py --data data/train_7767.json --scores runs/train_scores.jsonl
  python check_run.py --data data/test_1436.json --scores runs/anchor_scores.jsonl --ids anchor_ids.json

Prints, per model: calls done, calls with errors (latest record), probability vectors that do not sum to about 1,
total cost; then the number of samples scored by all models. Ends with OK or a list of problems.
"""
import argparse, json, math
from collections import Counter, defaultdict

MODELS = ["liquid/d1-ultra", "perplexity/pplx-decider-v1.2"]

ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); ap.add_argument("--scores", required=True); ap.add_argument("--ids")
a = ap.parse_args()
n_file = len(json.load(open(a.data, encoding="utf-8")))
ids = set(json.load(open(a.ids))) if a.ids else set(range(n_file))
latest = {}
with open(a.scores, encoding="utf-8") as f:
    for line in f:
        if line.strip():
            r = json.loads(line); latest[(r["dataset_idx"], r["model"])] = r
problems = []
ok_items = defaultdict(set); cost = 0.0
for m in MODELS:
    recs = [latest[(i, m)] for i in ids if (i, m) in latest]
    err = [r for r in recs if r.get("error")]
    good = [r for r in recs if not r.get("error")]
    def invalid(r):
        v = [r.get(k) for k in ("p_direct_execute", "p_ask_human", "p_refuse")]
        return any(x is None or (isinstance(x, float) and math.isnan(x)) for x in v) or min(v) < 0 or abs(sum(v) - 1) > 0.02
    badsum = [r for r in good if invalid(r)]
    cost += sum(r.get("cost_usd") or 0 for r in recs)
    for r in good:
        if r not in badsum: ok_items[r["dataset_idx"]].add(m)
    print(f"{m:34s} done {len(recs):5d}/{len(ids)} | errors {len(err):3d} | bad probability vectors {len(badsum):3d}"
          + (f" | example error: {err[0]['error'][:60]}" if err else ""))
    if len(recs) < len(ids): problems.append(f"{m}: {len(ids) - len(recs)} calls missing (rerun score_split.py)")
    if len(err) > 0.01 * len(ids): problems.append(f"{m}: {len(err)} errors, more than 1% (run score_split.py --retry-failed)")
    if badsum: problems.append(f"{m}: {len(badsum)} invalid probability vectors (run score_split.py --retry-failed)")
complete = [i for i in ids if len(ok_items[i]) == len(MODELS)]
print(f"samples scored by all models: {len(complete)} of {len(ids)} | total cost USD {cost:.2f}")
print("labels of complete samples:", dict(Counter(latest[(i, MODELS[0])]["ground_truth"] for i in complete)))
print("response_model values:", dict(Counter(r.get("response_model") for r in latest.values())))
print("OK" if not problems else "PROBLEMS:\n  " + "\n  ".join(problems))
