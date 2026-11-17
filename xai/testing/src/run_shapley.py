"""Run exact field-level Shapley (llmSHAP, FullEnumerationSampler) on one decision model for the selected calls.

--model takes a full id or a slug (gpt [default], jev, microsoft, liquid, pplx). gpt keeps its files in results/ and
plots/; the other models write to results/<slug>/ and plots/<slug>/. --repeats r (non-deterministic models): the
payoff of a coalition is the mean of r answers (see adapter.py); the value is stored per model in run_config.json and
must not change between runs of the same model.

Usage (from xai/testing):
  .venv/bin/python src/run_shapley.py --which pilot --dry-run
  .venv/bin/python src/run_shapley.py --which pilot --determinism
  .venv/bin/python src/run_shapley.py --which all --dry-run
  .venv/bin/python src/run_shapley.py --which all
  .venv/bin/python src/run_shapley.py --model pplx --repeats 3 --which all --fixed tool --dry-run

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
from common import (CLASSES, DATA_PATH, MAX_WORKERS, MODEL, SUBSET_PATH, ModelPaths, make_state, parse_fixed,
                    resolve_model, variant_suffix)
from fields import CallFields

MAX_COST_USD = 2.0


def all_coalitions(players):
    for k in range(len(players) + 1):
        yield from itertools.combinations(players, k)


def log(line, path):
    print(line)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["pilot", "all"], required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--determinism", action="store_true", help="re-query the full coalition of the 3 pilot calls")
    ap.add_argument("--max-cost", type=float, default=MAX_COST_USD)
    ap.add_argument("--fixed", help="comma-separated fields kept in every coalition (not players), e.g. tool")
    ap.add_argument("--model", default=MODEL, help="model id or slug (gpt, jev, microsoft, liquid, pplx)")
    ap.add_argument("--repeats", type=int, help="answers per coalition, payoff = their mean (default: the model's "
                                                 "run_config.json, else 1)")
    a = ap.parse_args()
    fixed = parse_fixed(a.fixed)
    mp = ModelPaths(resolve_model(a.model))
    model, COST_PER_CALL = mp.model, mp.cost
    repeats = a.repeats or mp.repeats()
    if mp.run_config.exists() or repeats != 1:
        assert not mp.run_config.exists() or mp.repeats() == repeats, (
            f"{mp.slug} was run with repeats={mp.repeats()}; mixing repeats would change cached payoffs")
    raw_path = mp.raw(variant_suffix(fixed))
    RUN_LOG = mp.run_log
    log_ = lambda line: log(line, RUN_LOG)

    data = json.load(open(DATA_PATH, encoding="utf-8"))
    sub = json.load(open(SUBSET_PATH))
    ids = sub["pilot_ids"] if a.which == "pilot" else [c["dataset_idx"] for c in sub["calls"]]
    calls = {i: CallFields(data[i]) for i in ids}               # asserts byte-identical round trip

    llm = DecisionsLLM(model=model, cache_path=mp.cache, repeats=repeats)
    needed = set()
    n_coal = 0
    for i, c in calls.items():
        fx = [p for p in c.players if p in fixed]
        for S in all_coalitions([p for p in c.players if p not in fixed]):
            n_coal += 1
            needed.add(cache_key(payload_for(make_state(c.system, c.user_content(S + tuple(fx))), model)))
    todo = {k for k in needed if llm.missing_answers(k)}
    n_answers = sum(llm.missing_answers(k) for k in todo)
    n_det = len(sub["pilot_ids"]) if a.determinism else 0
    est = (n_answers + n_det) * COST_PER_CALL
    log_(f"\n[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] model={model} repeats={repeats} "
         f"which={a.which} calls={len(ids)} dry_run={a.dry_run} determinism={a.determinism} fixed={fixed}")
    log_(f"  round trip: byte-identical full coalition for all {len(ids)} calls")
    log_(f"  coalitions (sum of 2^n): {n_coal} | distinct requests: {len(needed)} | complete in cache: "
         f"{len(needed) - len(todo)} | to query: {len(todo)} payloads, {n_answers} answers (+{n_det} determinism re-queries)")
    log_(f"  cost estimate: {n_answers + n_det} x {COST_PER_CALL:.2e} = USD {est:.4f} (cap USD {a.max_cost:.2f})")
    if a.dry_run:
        return
    if est > a.max_cost:
        log_("  ABORT: estimate exceeds the cap"); raise SystemExit(1)
    if model != MODEL or repeats != 1:
        mp.run_config.parent.mkdir(parents=True, exist_ok=True)
        json.dump({"model": model, "repeats": repeats}, open(mp.run_config, "w"), indent=2)

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
            first = llm.cache[cache_key(payload_for(state, model))]
            second = llm.query(state, bypass_cache=True)
            diff = (max(abs(first["probabilities"][k] - second["probabilities"][k]) for k in CLASSES)
                    if first["status"] == second["status"] == "ok" else None)
            det.append({"dataset_idx": i, "first": first.get("probabilities"), "second": second.get("probabilities"),
                        "status": [first["status"], second["status"]], "max_abs_diff": diff})
        json.dump(det, open(mp.determinism, "w"), indent=2)
        log_(f"  determinism re-queries: {[(d['dataset_idx'], d['max_abs_diff']) for d in det]}")

    status = {}
    for r in llm.cache.values():
        status[r["status"]] = status.get(r["status"], 0) + 1
    log_(f"  done in {time.time() - t0:.0f}s | model {model} | answers received now: {llm.n_new} | "
        f"HTTP requests incl. retries: {llm.n_http} | cache hits: {llm.n_hits} | billed cost now (API usage.cost): "
        f"USD {llm.cost_usd:.4f} | cache size {len(llm.cache)} {status}")


if __name__ == "__main__":
    main()
