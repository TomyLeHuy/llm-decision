"""Step A: determinism probe for one model (fresh answers, bypassing and not filling the cache).

For each pilot call (541, 444, 1075):
  - the full input, queried FULL_REPEATS = 5 times;
  - 2 random partial coalitions, queried PARTIAL_REPEATS = 3 times each. They are drawn (seed 20261111) from the
    proper subsets that contain `tool`, i.e. from the main design's coalitions, so they are inputs the Shapley run
    actually uses.
Deterministic = every repeated input gives identical probabilities (max diff 0) and the same status.
Output: results/<slug>/determinism.json (gpt: not used; its 2-query check stays in results/determinism.json).

  .venv/bin/python src/probe_determinism.py --model jev --dry-run
  .venv/bin/python src/probe_determinism.py --model jev
"""
import argparse
import json
import random
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from statistics import mean, pstdev

from adapter import DecisionsLLM, failure_type
from common import CLASSES, DATA_PATH, MAX_WORKERS, SUBSET_PATH, ModelPaths, make_state, resolve_model
from fields import CallFields

FULL_REPEATS, PARTIAL_REPEATS, N_PARTIAL, SEED = 5, 3, 2, 20261111


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--max-cost", type=float, default=2.0)
    a = ap.parse_args()
    mp = ModelPaths(resolve_model(a.model))
    data = json.load(open(DATA_PATH, encoding="utf-8"))
    sub = json.load(open(SUBSET_PATH))
    rng = random.Random(SEED)

    items = []      # (dataset_idx, coalition tuple, repeats)
    for i in sub["pilot_ids"]:
        c = CallFields(data[i])
        items.append((i, tuple(c.players), FULL_REPEATS))
        others = [p for p in c.players if p != "tool"]
        cands = [tuple(p for p in c.players if p == "tool" or (m >> others.index(p)) & 1)
                 for m in range(2 ** len(others) - 1)]          # proper subsets containing tool (excl. full input)
        for S in rng.sample(cands, N_PARTIAL):
            items.append((i, S, PARTIAL_REPEATS))
    n_q = sum(r for *_, r in items)
    est = n_q * mp.cost
    print(f"model {mp.model}: {len(items)} inputs, {n_q} queries x {mp.cost:.2e} = USD {est:.4f} (cap {a.max_cost})")
    if a.dry_run:
        return
    assert est <= a.max_cost, "estimate exceeds the cap"

    llm = DecisionsLLM(model=mp.model, cache_path=mp.cache, use_disk_cache=False)
    jobs = [(k, i, S) for k, (i, S, r) in enumerate(items) for _ in range(r)]
    states = {k: make_state(CallFields(data[i]).system, CallFields(data[i]).user_content(S)) for k, (i, S, _) in
              enumerate(items)}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        answers = list(ex.map(lambda j: (j[0], llm.query(states[j[0]], bypass_cache=True)), jobs))

    out, failures = [], {}
    for k, (i, S, r) in enumerate(items):
        ans = [x for kk, x in answers if kk == k]
        st = [x["status"] for x in ans]
        ok = [x["probabilities"] for x in ans if x["status"] == "ok"]
        for x in ans:
            ft = failure_type(x)
            if ft:
                failures.setdefault(ft, {"count": 0, "example": x.get("example_body") or x.get("error")})
                failures[ft]["count"] += 1
        rng_ = {c: (max(p[c] for p in ok) - min(p[c] for p in ok)) if len(ok) > 1 else None for c in CLASSES}
        out.append({"dataset_idx": i, "coalition": list(S), "kind": "full" if len(S) == len(CallFields(data[i]).players)
                    else "partial", "statuses": st, "probabilities": ok,
                    "max_abs_diff": max(rng_.values()) if len(ok) > 1 else None,
                    "std": {c: pstdev([p[c] for p in ok]) for c in CLASSES} if len(ok) > 1 else None})
    diffs = [o["max_abs_diff"] for o in out if o["max_abs_diff"] is not None]
    mixed = [o for o in out if len(set(o["statuses"])) > 1]
    stds = [v for o in out if o["std"] for v in o["std"].values()]
    summary = {"model": mp.model, "queried_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "n_inputs": len(items), "n_queries": n_q, "n_http": llm.n_http,
               "billed_cost_usd": llm.cost_usd, "max_abs_diff": max(diffs, default=None),
               "mean_std_per_class_and_input": mean(stds) if stds else None,
               "n_inputs_with_mixed_status": len(mixed),
               "deterministic": bool(diffs) and max(diffs) == 0 and not mixed,
               "failure_types": failures, "inputs": out}
    mp.results.mkdir(parents=True, exist_ok=True)
    json.dump(summary, open(mp.determinism, "w"), indent=2)
    print(f"  max abs diff {summary['max_abs_diff']} | mean std {summary['mean_std_per_class_and_input']} | mixed "
          f"status {len(mixed)} | deterministic {summary['deterministic']} | cost USD {llm.cost_usd:.4f} | "
          f"failures {({k: v['count'] for k, v in failures.items()})}")
    for o in out:
        print(f"   {o['dataset_idx']:>5} {o['kind']:<7} |S|={len(o['coalition'])} statuses={sorted(set(o['statuses']))} "
              f"max_diff={o['max_abs_diff']}")


if __name__ == "__main__":
    main()
