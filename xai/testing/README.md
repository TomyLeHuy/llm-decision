# Field-level Shapley values for the gpt decision model (pilot)

Exact Shapley values with **llmSHAP 1.5.2** (Naudot, Sundqvist & Kampik 2025, arXiv 2511.01311). The players are the
**input fields** of a proposed tool call, not tokens. The payoff v(S) is gpt's probability for one class
(direct_execute, ask_human, refuse) when only the fields in S are sent. Model: `openai/gpt-6-luna-decisions`
(deterministic). Scope: 30 single-step calls from `test_1436.json`. Not in scope yet: entropy as payoff, conformal
uncertainty profiles, the other vendor models.

## Result in one paragraph

The 7-player design (all fields as players) **does not work** for gpt. The model refuses the empty coalition and
other coalitions without any tool-call field (HTTP 502 "OpenAI refused to answer question"), so v(∅) is undefined for
every call. Two designs with a fixed part were therefore run on all 30 calls, both from one shared cache:

| design | players | fixed in every coalition | complete calls | role |
|---|---|---|---|---|
| `__fixed-tool` | user_task, service, user_memory, prior_steps, tool_args, description (≤6) | `tool` | 29 / 30 (519 undefined) | **main design** |
| `__fixed-description+tool+tool_args` | user_task, service, user_memory, prior_steps (≤4) | whole `assistant_proposed_tool_call` | 30 / 30 | second view |

v(∅) therefore means "the reviewer sees only the tool name" (main design) or "only the proposed call" (second view).
It **differs per call**, so it is reported per call (`v_empty`) next to φ.

## Findings

**F1: the 7-player design is undefined for gpt.** On the 3 pilot calls (541, 444, 1075), 10 of 222 coalitions were
refused, by size {0: 3, 1: 5, 2: 3, 3: 1}. All 10 contained **no** tool-call field, and the empty coalition was
refused for every call. See `results/checks.json` → `finding_7_player_design` and `provider_refusals`.

**F2: refusals with a fixed tool name are rare but not zero.** Main design: 1 refused coalition out of 1,188 distinct
requests, in call 519 (label refuse, 6 fields). The refused coalition is {tool, user_memory, description}, so "no
tool-call field → refusal" is not the whole rule. Following the no-imputation rule, call 519 has no φ in the main
design. Second view: 0 refusals.

## How to run (from `xai/testing/`)

```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python src/select_subset.py                                  # -> results/subset_ids.json (no API)
.venv/bin/python src/run_shapley.py --which all --fixed tool --dry-run # queries + cost, no API
.venv/bin/python src/run_shapley.py --which all --fixed tool           # paid; resumable, cached
.venv/bin/python src/run_shapley.py --which all --fixed tool,tool_args,description   # served from the cache
.venv/bin/python src/analysis.py --fixed tool --plots                  # checks, CSV, plots (no API)
.venv/bin/python src/analysis.py --fixed tool,tool_args,description --plots
.venv/bin/python src/analysis.py                                       # 7-player design (pilot calls only)
```

`OPENROUTER_API_KEY` is read from the environment or the repo's `.env`. `run_shapley.py` stops if the estimate
(distinct uncached requests × `APPROX_COST_PER_CALL`) exceeds USD 2 (`--max-cost`). It sends at most 4 requests at a
time: llmSHAP uses `num_threads=4`, and the adapter also has a semaphore of 4.

## Files

- `src/common.py`: paths. It imports `score_split.py` from the frozen package read-only (no `__pycache__` is
  written there) for the endpoint, `QUESTIONS_PAYLOAD`, `APPROX_COST_PER_CALL` and `build_user_content`. The state
  template is copied verbatim from `call_openrouter_decision`.
- `src/fields.py`: players and coalition → user JSON.
- `src/adapter.py`: llmSHAP `LLMInterface` with a persistent cache.
- `src/codec.py`: `PromptCodec`, `ValueFunction`, `DataHandler` factory.
- `src/select_subset.py`, `src/run_shapley.py`, `src/analysis.py`, `src/plots.py`.
- `results/coalitions.jsonl`: the cache. One line per distinct request, keyed by sha256 of the canonical payload
  `{model, state, questions}`. Each line holds the status, probabilities, cost and response model.
- `results/shapley_values<suffix>.csv`: dataset_idx, label, n_fields, field, class, phi (llmSHAP), loo, v_empty,
  v_full, status. Undefined calls keep their rows with NaN values and `status=undefined_provider_refusal`.
  `n_fields` counts all present fields, including the fixed ones.
- `results/checks<suffix>.json`: all checks plus per-call details (players, refused coalitions, full probabilities).
- `results/llmshap_raw<suffix>.jsonl`: llmSHAP's raw output per call and class.
- `results/subset_ids.json`, `results/determinism.json`, `results/run_log.txt` (queries and cost per run).
- `plots/beeswarm_<class><suffix>.png`: one row per field, one dot per call at φ, colour and shape = true label.
  Fields are text, so dots are not coloured by feature value.
- `plots/mean_abs_phi<suffix>.png`: mean |φ| per field, one panel per class.
- No suffix = 7-player design (pilot only). `__fixed-tool` = main design.
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
4. **Refusals:** an HTTP 502 whose body contains "refused to answer" is recorded at once as `provider_refusal`, with
   no 5 retries. This is the one deviation from score_split's retry policy: the model is deterministic, so retries
   cannot change the answer. A 200 without a valid probability vector (same rule as `is_failed`) and other
   non-retryable 4xx are also `provider_refusal`. 401/402 stop the run. Network or rate-limit failures after all
   retries count as `transient_error`. These are not cached, so a rerun retries them. In this run there were none.
5. **No imputation:** a call with any refused coalition gets no φ. This is counted per design.
6. **Fixed fields** are llmSHAP `permanent_keys`. They are in every coalition, including the "empty" one, and get no φ.
7. **Cache:** llmSHAP's `use_cache=True` plus our disk cache. The three per-class runs and both designs share it.
   `num_coalition_evaluations` is left at 1 (repeats would be no-ops).

## Checks (30 calls; run `analysis.py` to reproduce)

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

## Cost

Distinct requests answered: 1,301 (1,298 cached + 3 determinism re-queries). HTTP requests including retries: 1,308.
Billed (API `usage.cost`): USD 0.0209 (pilot) + USD 0.1066 (remaining 27 calls, main design) = **USD 0.1275**. The
second view and all reruns were served from the cache.

## Main pattern (main design, 29 calls; descriptive, n is small)

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

- Determinism was re-checked on 3 calls only. The other 27 rely on gpt being deterministic and on the exact match of
  all 30 full coalitions with the October scores.
- 30 calls, single-step only, gpt only. The patterns above are descriptive, with no uncertainty estimates.
- The 7-player finding comes from 3 calls. The exact refusal rule is unknown (519 shows a refusal *with* a tool-call
  field present).
- Excluding the 3 October refusals and dropping 519 (a refuse-labelled call) bias the subset slightly toward inputs
  gpt accepts.
- Removing a field is one choice of "absence". φ measures the effect of deletion, not of a neutral replacement.
- `.env` produces harmless python-dotenv parse warnings for its lines 2–3. The key itself loads.
