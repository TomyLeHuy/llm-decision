# Field-level Shapley values for the five decision models (pilot)

Exact Shapley values with **llmSHAP 1.5.2** (Naudot, Sundqvist & Kampik 2025, arXiv 2511.01311). The players are the
**input fields** of a proposed tool call, not tokens. The payoff v(S) is a model's probability for one class
(direct_execute, ask_human, refuse) when only the fields in S are sent. Scope: 30 single-step calls from
`test_1436.json` and five decision models, each called through the OpenRouter decisions endpoint exactly as in the
paper's scorer:

| slug | model id | cost/call | deterministic | repeats r |
|---|---|---|---|---|
| gpt | `openai/gpt-6-luna-decisions` | 8.8e-5 | yes | 1 |
| jev | `typesafe/jev-1.14` | 4.8e-5 | no | 3 |
| microsoft | `microsoft/microsoft-decision-1` | 3.6e-5 | no | 3 |
| liquid | `liquid/d1-ultra` | 9.2e-5 | no | 3 |
| pplx | `perplexity/pplx-decider-v1.2` | 4.6e-5 | no | 3 |

gpt was run first (10 Nov 2026); the other four followed on 11 Nov 2026. Not in scope yet: entropy as payoff,
conformal uncertainty profiles, calibration, counterfactuals.

## Result in one paragraph

The 7-player design (all fields as players) **does not work** for gpt. The model refuses the empty coalition and
other coalitions without any tool-call field (HTTP 502 "OpenAI refused to answer question"), so v(∅) is undefined for
every call. Two designs with a fixed part were therefore run on all 30 calls, both from one shared cache:

| design | players | fixed in every coalition | complete calls (gpt) | role |
|---|---|---|---|---|
| `__fixed-tool` | user_task, service, user_memory, prior_steps, tool_args, description (≤6) | `tool` | 29 / 30 (519 undefined) | **main design** |
| `__fixed-description+tool+tool_args` | user_task, service, user_memory, prior_steps (≤4) | whole `assistant_proposed_tool_call` | 30 / 30 | second view |

v(∅) therefore means "the reviewer sees only the tool name" (main design) or "only the proposed call" (second view).
It **differs per call**, so it is reported per call (`v_empty`) next to φ.

The other four models answer every input, including the empty one. For them the 7-player design is defined and was
added as a third view (no suffix); the two fixed designs were run as well, for comparability with gpt.

## Findings

**F1: the 7-player design is undefined for gpt.** On the 3 pilot calls (541, 444, 1075), 10 of 222 distinct requests
were refused. Counted per call, these are 12 coalitions (the empty coalition is the same request for all three calls),
by size {0: 3, 1: 5, 2: 3, 3: 1}. All of them contained **no** tool-call field, and the empty coalition was refused
for every call. See `results/checks.json` → `finding_7_player_design` and `provider_refusals`.

**F2: refusals with a fixed tool name are rare but not zero.** Main design: 1 refused coalition out of 1,188 distinct
requests, in call 519 (label refuse, 6 fields). The refused coalition is {tool, user_memory, description}, so "no
tool-call field → refusal" is not the whole rule. Following the no-imputation rule, call 519 has no φ in the main
design. Second view: 0 refusals.

**F3: only gpt refuses.** jev, microsoft, liquid and pplx answered all 2,317 distinct requests of the 7-player design
on all 30 calls (3 answers each), including the empty input and the 300 coalition slots without any tool-call field.
No new failure type occurred: no HTTP 200 without a valid probability vector, no other error body.

**F4: the four other models are not deterministic.** In the probe (3 pilot calls; full input 5×, 2 partial inputs 3×
each), the max difference between repeated answers was 0.03 (jev), 0.032 (microsoft), 0.27 (liquid) and 0.051 (pplx).
Their payoffs are therefore the mean of r = 3 answers.

**F5: microsoft has drifted since October.** Its full-input probabilities differ from the October scores by up to
0.74 (mean 0.11), far above its noise level (mean std 0.008). The API still reports the same version string
(`microsoft-decision-1-20261009`). The 23 Oct anchor re-score in the frozen package showed only a mean diff of 0.015
for microsoft (computed ad hoc from `runs/anchor_scores.jsonl`, not by a script in this folder).

The cross-model results are in `experiment.md`, section 6.

## How to run (from `xai/testing/`)

```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python src/select_subset.py                                  # -> results/subset_ids.json (no API)

# gpt (default model; files in results/ and plots/)
.venv/bin/python src/run_shapley.py --which all --fixed tool --dry-run # queries + cost, no API
.venv/bin/python src/run_shapley.py --which all --fixed tool           # paid; resumable, cached
.venv/bin/python src/run_shapley.py --which all --fixed tool,tool_args,description   # served from the cache
.venv/bin/python src/analysis.py --fixed tool --plots                  # checks, CSV, plots (no API)
.venv/bin/python src/analysis.py --fixed tool,tool_args,description --plots
.venv/bin/python src/analysis.py                                       # 7-player design (pilot calls only)

# other models (--model takes a slug or the full id; files in results/<slug>/ and plots/<slug>/)
.venv/bin/python src/probe_determinism.py --model pplx --dry-run       # Step A: determinism probe
.venv/bin/python src/probe_determinism.py --model pplx
.venv/bin/python src/run_shapley.py --model pplx --repeats 3 --which pilot          # Step B: 7-player pilot
.venv/bin/python src/analysis.py --model pplx                                       #   -> seven_player_pilot
.venv/bin/python src/run_shapley.py --model pplx --which all --dry-run              # 7-player, all 30 calls
.venv/bin/python src/run_shapley.py --model pplx --which all                        # (repeats from run_config.json)
.venv/bin/python src/run_shapley.py --model pplx --which all --fixed tool           # from the cache
.venv/bin/python src/run_shapley.py --model pplx --which all --fixed tool,tool_args,description
.venv/bin/python src/analysis.py --model pplx [--fixed ...] --plots

.venv/bin/python src/compare.py                                        # Step D: cross-model comparison (no API)
```

`OPENROUTER_API_KEY` is read from the environment or the repo's `.env`. `run_shapley.py` and `probe_determinism.py`
stop if the estimate (missing answers × cost per call) exceeds USD 2 (`--max-cost`). At most 4 requests are in flight:
llmSHAP uses `num_threads=4`, the adapter has a semaphore of 4, and the models were run one after another.

## Files

- `src/common.py`: paths, the model table and per-model paths (`ModelPaths`). It imports `score_split.py` from the
  frozen package read-only (no `__pycache__` is written there) for the endpoint, `QUESTIONS_PAYLOAD`,
  `APPROX_COST_PER_CALL` and `build_user_content`. The liquid/pplx ids and costs are read from
  `extra_models/score_extra.py` with `ast`, because importing it would overwrite `score_split.MODELS` in this process.
  The state template is copied verbatim from `call_openrouter_decision`.
- `src/fields.py`: players and coalition → user JSON.
- `src/adapter.py`: llmSHAP `LLMInterface` with a persistent cache, repeats and failure classification.
- `src/codec.py`: `PromptCodec`, `ValueFunction`, `DataHandler` factory.
- `src/select_subset.py`, `src/run_shapley.py`, `src/analysis.py`, `src/plots.py`.
- `src/probe_determinism.py`: Step A. `src/compare.py`: Step D.

Per model, gpt in `results/` and `plots/`, the others in `results/<slug>/` and `plots/<slug>/`:

- `coalitions.jsonl`: the cache, one file per model. One line per raw answer, keyed by sha256 of the canonical
  payload `{model, state, questions}`. Each line holds the status, probabilities, cost and response model. A key holds
  r lines for a model run with r repeats.
- `shapley_values<suffix>.csv`: dataset_idx, label, n_fields, field, class, phi (llmSHAP), loo, v_empty, v_full,
  status. Undefined calls keep their rows with NaN values and `status=undefined_provider_refusal` (or
  `undefined_october_failure`). `n_fields` counts all present fields, including the fixed ones.
- `checks<suffix>.json`: all checks plus per-call details (players, refused coalitions, full probabilities). Added
  for the five-model step: `repeats`, `agreement_with_october_mean_abs_diff`, `october_failed_calls_undefined`,
  `failure_types_in_cache` (with an example body), `determinism_probe`, `noise` (r > 1) and `seven_player_pilot`
  (7-player design).
- `llmshap_raw<suffix>.jsonl`: llmSHAP's raw output per call and class.
- `determinism.json` (gpt: 3 re-queries; others: the probe), `run_log.txt` (queries and cost per run),
  `run_config.json` (repeats; not for gpt).
- `plots/[<slug>/]beeswarm_<class><suffix>.png`: one row per field, one dot per call at φ, colour and shape = true
  label. Fields are text, so dots are not coloured by feature value.
- `plots/[<slug>/]mean_abs_phi<suffix>.png`: mean |φ| per field, one panel per class.
- `results/subset_ids.json`: the 30 calls, shared by all models.
- `results/compare/`, `plots/compare/`: see `src/compare.py` (mean |φ| and own-class φ per field × model, baselines,
  pairwise agreement, `model_table.csv/.md`, `summary.json`).
- Suffixes: none = 7-player design (gpt: pilot only). `__fixed-tool` = main design.
  `__fixed-description+tool+tool_args` = second view.

## Design decisions

1. **Removing a field = deleting its key.** tool / tool_args / description live inside
   `assistant_proposed_tool_call`, which is dropped once all three are removed. Key order is kept.
2. **Serialisation:** the 1,311 single-step candidates use three formats: `json.dumps(indent=2, ensure_ascii=False)`
   (829), compact `ensure_ascii=True` (440) and compact `ensure_ascii=False` (42). The format is detected per record
   and reused for every coalition. A byte-identical round trip of the full coalition is asserted for every candidate,
   along with equality to `score_split.build_user_content`.
3. **Subset:** single-step records only. 10 per label, split 3/3/4 over 5/6/7 present fields, seed 20261110. The 3
   records whose October gpt score was already a provider refusal (474, 1299, 1333) are excluded, because their
   full-coalition value is undefined. Pilot calls: 541 (direct_execute, 5 fields), 444 (ask_human, 6), 1075 (refuse, 7).
   The same 30 calls are used for every model. None of the 30 failed in October for any of the five models.
4. **Refusals:** an HTTP 502 whose body contains "refused to answer" is recorded at once as `provider_refusal`, with
   no 5 retries. This is the one deviation from score_split's retry policy: the model is deterministic, so retries
   cannot change the answer. A 200 without a valid probability vector (same rule as `is_failed`) and other
   non-retryable 4xx are also `provider_refusal`. 401/402 stop the run. Network or rate-limit failures after all
   retries count as `transient_error`. These are not cached, so a rerun retries them. In this run there were none.
5. **No imputation:** a call with any refused coalition gets no φ. This is counted per design.
6. **Fixed fields** are llmSHAP `permanent_keys`. They are in every coalition, including the "empty" one, and get no φ.
7. **Cache:** llmSHAP's `use_cache=True` plus our disk cache. The three per-class runs and both designs share it.
   `num_coalition_evaluations` is left at 1 (repeats would be no-ops).
8. **Repeats for non-deterministic models (`--repeats r`):** the payoff of a coalition is the mean of its **first r
   answers in the cache file**. Once r answers exist the value is fixed, so all designs and reruns use the same v(S),
   and efficiency, symmetry and the null-player property hold for these values. If any of the r answers is a refusal,
   the coalition counts as refused. r is stored in `run_config.json`, and `run_shapley.py` refuses to mix values of r
   for one model. With r = 1 (gpt) the payoff is the single cached answer, exactly as before.
9. **Noise check (r > 1):** for every coalition the sample std of its r answers is kept. `checks.json` → `noise`
   reports the mean std per class and the share of |φ| below it (and below std/√r, the std of the mean).
10. **One cache file per model.** The key contains the model id as well, so an answer of one model can never be
    returned for another. The adapter also asserts that a cache file holds only its own model.
11. **Determinism probe answers** bypass the cache and are not used as payoffs.

## Checks, gpt (30 calls; run `analysis.py` to reproduce)

| check | main (`tool` fixed) | second view (tool call fixed) |
|---|---|---|
| round trip, byte-identical + equal to `build_user_content` | 30/30 | 30/30 |
| determinism: full coalition queried twice (pilot calls 541, 444, 1075) | max diff 0 | same queries |
| full-coalition probabilities vs October `scores_test_1436.csv` | max diff 0.0 (30 calls) | 0.0 |
| efficiency, max \|v(∅)+Σφ − v(N)\| | 1.2e-16 | 1.1e-16 |
| independent exact Shapley from cached 2^n values vs llmSHAP | max diff 0.0 | 0.0 |
| llmSHAP v(∅), v(N) vs cache | 0.0 | 0.0 |
| complete calls | 29 (519 undefined) | 30 |
| LOO vs φ, pooled Spearman (exec / ask / refuse) | 0.42 / 0.58 / 0.69 | 0.66 / 0.75 / 0.79 |
| LOO vs φ, median per-call Spearman | 0.53 / 0.75 / 0.78 | 1.00 / 1.00 / 0.97 |
| same top-\|·\| field, LOO vs φ | 55% / 62% / 59% | 73% / 77% / 67% |

With 6 players, leave-one-out often names a different top field than Shapley (interactions matter). With 4 players
the two agree much more.

## Checks, five models (main design; from `results/compare/model_table.csv` and `checks*.json`)

| | gpt | jev | microsoft | liquid | pplx |
|---|---|---|---|---|---|
| complete calls: main / second / 7-player | 29 / 30 / 0 of 3 | 30 / 30 / 30 | 30 / 30 / 30 | 30 / 30 / 30 | 30 / 30 / 30 |
| refused coalitions (main) | 1 | 0 | 0 | 0 | 0 |
| round trip | 30/30 | 30/30 | 30/30 | 30/30 | 30/30 |
| efficiency, max error (all designs) | ≤1.2e-16 | ≤1.2e-16 | ≤2.2e-16 | ≤1.1e-16 | ≤1.1e-16 |
| independent Shapley vs llmSHAP; v(∅)/v(N) vs cache | 0.0; 0.0 | 0.0; 0.0 | 0.0; 0.0 | 0.0; 0.0 | 0.0; 0.0 |
| full input vs October: max / mean abs diff | 0.0 / 0.0 | 0.060 / 0.010 | 0.745 / 0.113 | 0.124 / 0.019 | 0.061 / 0.006 |
| noise: mean std of one answer | – | 0.011 | 0.008 | 0.030 | 0.004 |
| share of \|φ\| below that std (exec / ask / refuse) | – | 42% / 37% / 57% | 24% / 18% / 39% | 45% / 46% / 59% | 8% / 12% / 22% |
| LOO vs φ, median per-call Spearman | 0.53 / 0.75 / 0.78 | 0.71 / 0.74 / 0.79 | 0.81 / 0.80 / 0.75 | 0.73 / 0.70 / 0.65 | 0.71 / 0.40 / 0.54 |

The October diff is not asserted for non-deterministic or drifted models; it is reported only.

## Cost

gpt: distinct requests answered 1,301 (1,298 cached + 3 determinism re-queries); HTTP requests including retries
1,308; billed (API `usage.cost`) USD 0.0209 (pilot) + USD 0.1066 (remaining 27 calls) = **USD 0.1275**.

Other models (11 Nov 2026), billed: probe + 7-player pilot + 7-player on all 30 calls. Both fixed designs were served
entirely from the cache.

| model | answers in cache | probe answers | billed USD |
|---|---|---|---|
| jev | 6,951 | 33 | 0.3709 |
| microsoft | 6,951 | 33 | 0.3038 |
| liquid | 6,951 | 33 | 0.2816 |
| pplx | 6,951 | 33 | 0.1405 |
| **total, 4 models** | 27,804 | 132 | **1.0968** |

## Main pattern, gpt (main design, 29 calls; descriptive, n is small)

- **Refusals are driven by the task text and the call description, not the arguments.** For refuse-labelled calls,
  the mean φ on P(refuse) is +0.34 for user_task and +0.19 for description, and −0.02 for tool_args. Mean |φ| on
  P(refuse): user_task 0.152, description 0.079, user_memory 0.062.
- **direct_execute is pushed up by tool_args, user_memory and prior_steps.** For direct_execute-labelled calls,
  the mean φ on P(direct_execute) is +0.07 to +0.09 for each of these.
- **service barely matters.** Mean |φ| is 0.02–0.03 for every class.
- **The tool name alone (v(∅)) already carries much of the decision for execute/ask, but not for refuse.** With only
  the tool name, refuse-labelled calls get mean P(refuse) = 0.18 and P(ask_human) = 0.68. The context fields are
  what move them toward refuse.
- φ explains the **model**, not the label. Several calls are misclassified by gpt (e.g. 1365, labelled ask_human,
  gets P(direct_execute) = 1.0), and their φ describe that wrong decision.

## Not verified / limitations

- gpt: determinism was re-checked on 3 calls only. The other 27 rely on gpt being deterministic and on the exact match
  of all 30 full coalitions with the October scores.
- Other models: determinism was probed on 3 calls (9 inputs). r = 3 was fixed by the task, not tuned; for jev and
  liquid a large share of the small φ values lies below the single-answer noise.
- 30 calls, single-step only. All patterns are descriptive, with no uncertainty estimates.
- The 7-player finding for gpt comes from 3 calls. The exact refusal rule is unknown (519 shows a refusal *with* a
  tool-call field present).
- Excluding the 3 October gpt refusals and dropping 519 (a refuse-labelled call) bias the subset slightly toward
  inputs gpt accepts.
- Removing a field is one choice of "absence". φ measures the effect of deletion, not of a neutral replacement.
- The microsoft drift was not investigated further; its φ describe the current model, not the one scored in October.
- The gpt `checks*.json` files gained the new keys listed under Files; all earlier keys and their values are unchanged.
  All other gpt files reproduce bit-identically from the cache.
- `.env` produces harmless python-dotenv parse warnings for its lines 2–3. The key itself loads.
