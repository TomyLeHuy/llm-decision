"""Run exact field-level Shapley (llmSHAP, FullEnumerationSampler) on gpt for the selected calls.

Usage (from xai/testing):
  .venv/bin/python src/run_shapley.py --which pilot --dry-run
  .venv/bin/python src/run_shapley.py --which pilot --determinism
  .venv/bin/python src/run_shapley.py --which all --dry-run
  .venv/bin/python src/run_shapley.py --which all

Every distinct request goes to the network at most once (results/coalitions.jsonl); the three per-class Shapley runs
and any rerun are served from that cache, so the run is resumable. The dry run makes no API calls.
"""
import argparse
import itertools
import json
import time
from datetime import datetime, timezone

from llmSHAP.attribution_methods.coalition_sampler import FullEnumerationSampler
from llmSHAP.attribution_methods.shapley_attribution import ShapleyAttribution

from adapter import DecisionsLLM, cache_key, payload_for
from codec import ClassProbability, FieldCodec, make_data_handler
from common import (CLASSES, COST_PER_CALL, DATA_PATH, DETERMINISM_PATH, LLMSHAP_RAW, MAX_WORKERS, MODEL, RUN_LOG,
                    SUBSET_PATH, make_state, parse_fixed, variant_suffix)
from fields import CallFields

MAX_COST_USD = 2.0


def all_coalitions(players):
    for k in range(len(players) + 1):
        yield from itertools.combinations(players, k)


def log(line):
    print(line)
    with open(RUN_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["pilot", "all"], required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--determinism", action="store_true", help="re-query the full coalition of the 3 pilot calls")
    ap.add_argument("--max-cost", type=float, default=MAX_COST_USD)
    ap.add_argument("--fixed", help="comma-separated fields kept in every coalition (not players), e.g. tool")
    a = ap.parse_args()
    fixed = parse_fixed(a.fixed)
    raw_path = LLMSHAP_RAW.with_name(LLMSHAP_RAW.stem + variant_suffix(fixed) + ".jsonl")

    data = json.load(open(DATA_PATH, encoding="utf-8"))
    sub = json.load(open(SUBSET_PATH))
    ids = sub["pilot_ids"] if a.which == "pilot" else [c["dataset_idx"] for c in sub["calls"]]
    calls = {i: CallFields(data[i]) for i in ids}               # asserts byte-identical round trip

    llm = DecisionsLLM()
    needed = set()
    n_coal = 0
    for i, c in calls.items():
        fx = [p for p in c.players if p in fixed]
        for S in all_coalitions([p for p in c.players if p not in fixed]):
            n_coal += 1
            needed.add(cache_key(payload_for(make_state(c.system, c.user_content(S + tuple(fx))))))
    todo = needed - set(llm.cache)
    n_det = len(sub["pilot_ids"]) if a.determinism else 0
    est = (len(todo) + n_det) * COST_PER_CALL
    log(f"\n[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] which={a.which} calls={len(ids)} "
        f"dry_run={a.dry_run} determinism={a.determinism} fixed={fixed}")
    log(f"  round trip: byte-identical full coalition for all {len(ids)} calls")
    log(f"  coalitions (sum of 2^n): {n_coal} | distinct requests: {len(needed)} | already cached: {len(needed) - len(todo)}"
        f" | to query: {len(todo)} (+{n_det} determinism re-queries)")
    log(f"  cost estimate: {len(todo) + n_det} x {COST_PER_CALL:.2e} = USD {est:.4f} (cap USD {a.max_cost:.2f})")
    if a.dry_run:
        return
    if est > a.max_cost:
        log("  ABORT: estimate exceeds the cap"); raise SystemExit(1)

    t0 = time.time()
    with open(raw_path, "a", encoding="utf-8") as raw:
        for i, c in calls.items():
            dh = make_data_handler(c, fixed)
            vars_ = [p for p in c.players if p not in fixed]
            codec = FieldCodec(c)
            for cls in CLASSES:
                attr = ShapleyAttribution(model=llm, data_handler=dh, prompt_codec=codec,
                                          sampler=FullEnumerationSampler(len(vars_)), use_cache=True,
                                          verbose=False, num_threads=MAX_WORKERS, value_function=ClassProbability(cls))
                res = attr.attribution()
                rec = {"dataset_idx": i, "class": cls, "players": vars_, "fixed": [p for p in c.players if p in fixed],
                       "phi": {f: v["score"] for f, v in res.attribution.items() if f in vars_},
                       "v_empty": res.empty_baseline, "v_full": res.grand_coalition_value}
                raw.write(json.dumps(rec) + "\n"); raw.flush()
            print(f"  call {i}: n={len(c.players)} (players {len(vars_)}) done | new requests so far {llm.n_new} | {time.time() - t0:.0f}s",
                  flush=True)

    if a.determinism:
        det = []
        for i in sub["pilot_ids"]:
            c = CallFields(data[i])
            state = make_state(c.system, c.user_content(c.players))
            first = llm.cache[cache_key(payload_for(state))]
            second = llm.query(state, bypass_cache=True)
            diff = (max(abs(first["probabilities"][k] - second["probabilities"][k]) for k in CLASSES)
                    if first["status"] == second["status"] == "ok" else None)
            det.append({"dataset_idx": i, "first": first.get("probabilities"), "second": second.get("probabilities"),
                        "status": [first["status"], second["status"]], "max_abs_diff": diff})
        json.dump(det, open(DETERMINISM_PATH, "w"), indent=2)
        log(f"  determinism re-queries: {[(d['dataset_idx'], d['max_abs_diff']) for d in det]}")

    status = {}
    for r in llm.cache.values():
        status[r["status"]] = status.get(r["status"], 0) + 1
    log(f"  done in {time.time() - t0:.0f}s | model {MODEL} | distinct requests answered now: {llm.n_new} | "
        f"HTTP requests incl. retries: {llm.n_http} | cache hits: {llm.n_hits} | billed cost now (API usage.cost): "
        f"USD {llm.cost_usd:.4f} | cache size {len(llm.cache)} {status}")


if __name__ == "__main__":
    main()
