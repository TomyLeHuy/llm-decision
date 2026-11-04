# Decisions log

One line of reason per choice. Phase 0 and Phase A (3 Nov 2026).

## Infrastructure

| # | Choice | Reason |
|---|---|---|
| D1 | `score_split.py` is loaded twice via `importlib` from its file path, with `sys.dont_write_bytecode = True` during import. One instance makes the real calls, unchanged. The other gets a stub transport (`requests.post` replaced on that instance only) and is used to read the exact payload/state string. | The paper's function stays the single source of the state string; the stub never touches the live instance; no `__pycache__` is written into the frozen package (verified: SHA256SUMS 9/9 OK, `git status` clean). |
| D2 | Masking by **re-serialising with the sample's own JSON format**, not by string surgery. `units.detect_format` finds the serialiser that reproduces the original exactly: indent 2 or compact, ASCII-escaped or not (4 variants). | S1 shows all 9,401 samples (test, train, mailu) match one variant byte for byte, so re-serialisation is exact; string surgery would be more fragile. |
| D3 | Units are taken from the **first** user message only (what is scored). | Plan rule 2; the multi-turn group is a test case, not re-scored. |
| D4 | Cache stores failed calls too, but treats them as misses (retry). | Failed calls cost 0 in `call_openrouter_decision`; keeping them documents errors. |
| D5 | Budget check before each batch: spent (sum of cached `cost_usd`) + 1.5 × per-model mean cost × calls must stay below USD 10; refuse at ≥ 80% spent. | Cost is per token and varies by call; the 1.5 margin covers long inputs. |
| D6 | S3 repeats go to a separate file and key (`#repeat=r`); 3 fresh repeats, the S2 call serves as a 4th draw minutes earlier. | Plan: bypass the cache for repeats. |
| D7 | Workers = 4. | Same limit as `score_split.py`. |
| D8 | `.env` is read with `dotenv_values` (not exported to `os.environ` by our code). python-dotenv warns that `.env` lines 2–3 cannot be parsed; they were not read or printed. The key on line 1 loads correctly. | Never print or log the key. |

## Phase 0

| # | Choice | Reason |
|---|---|---|
| P1 | S2 uses the **same 30 test calls for all 5 models** (150 calls). They are stratified by label × single/multi-turn, 5 per cell, seed 0, drawn from the paper's 1,433 calls. | Comparable across models; equal cells give 15 single- and 15 multi-turn calls. |
| P2 | S3: 10 calls drawn from the S2 calls, stratified the same way (seed 0; 5 single, 5 multi-turn) × 5 models × 3 repeats. | A subset of S2 gives a 4th draw for free. |
| P3 | Noise floor = RMS of the within-call SD over calls, per model and class; the max range is also reported. | RMS is the right pooled SD; the range shows the tails. |
| P4 | The October test run for jev/microsoft/gpt has **no `response_model`**. Versions are compared with the 23 Oct train run instead, and drift is localised with test calls whose scored input is byte-identical to a train sample (`p0_drift_train_overlap.py`, 16 calls per model). | It is the only way to date the drift. |
| P5 | Pilot cost estimate (S4) uses the stored per-call October cost of each pilot sample × (1 + 2 × units). This is an upper bound for M1 (masked inputs are shorter); M2 may be slightly longer or shorter. | Costs are per token; the observed today/October cost ratio in S2 is exactly 1.0 for all models. |
| P6 | Pilot sample drawn already, for the estimate only (not queried). 50 calls stratified by label × single/multi-turn × enterprise/benchmark, seed 0. Small cells are taken whole and the remainder is spread. Enterprise = `meta.source` in {internal_v2, v3_internal}; the five benchmarks are agentharm, atbench, r_judge (incl. dh_app), tsbench and when2call. | Matches the paper's "nine self-hosted services and five benchmarks". |

## Phase A

| # | Choice | Reason |
|---|---|---|
| A1a | Confident Learning implemented directly (cleanlab is not installed): per-class threshold = mean self-confidence; a sample is flagged if its confident class (the largest p among classes above threshold) differs from the label. | This is the standard confident-joint off-diagonal rule; no extra dependency. |
| A1b | Main audit on the paper's 1,433 test calls; train (7,755) is a replication with its own thresholds. | Plan. |
| A1c | The review export holds the 40 single-turn calls with the most model agreement (all have 5/5 flags), sorted by the lowest mean probability of the given label. Multi-turn flags are exported separately. | Multi-turn flags are mostly pipeline effects (plan caveat). |
| A1d | Added an independent label-noise signal: test calls whose **full prompt** is byte-identical to a train call with a different label. | Free, and it does not depend on any model. |
| A2a | "Certifies" per model × service = the paper's G5 per-group harm test, run on **all labelled calls of the pool** as calibration (test pool: 1,433; train pool: 7,755). The test is restricted to the region whose ask risk is certified on the whole pool (G5 step A). Two levels: δ/(2m), per service separately (n ≥ 102), and δ/(2m·4), the paper's 4 simultaneous groups (n ≥ 129). | It answers "could this service ever be certified with all the data we have", which isolates the levers. G4 on the pool is reported as well (`a2_g4_pool.csv`). |
| A2b | Binding lever, first that applies: ask_floor (no region passes the pooled ask test), then sample_size (n_refusals < n_min), then resolution_floor (the strictest admissible region already holds more than k_max refusals), otherwise certifies. "Extra clean refusals" is computed for the strictest admissible region. | The order follows the logical dependency in G5. |
| A2c | AUROC of p_refuse: refusals vs all other calls of the same service (also vs execute only, in the CSV). Not defined for tsbench (refusals only) or for dh_app multi-turn (no non-refusals in that subset). | Separability within the service is what a per-service threshold needs. |
| A2d | The single-turn-only certification check is reported as a **what-if** (it removes calls), not as a fix. | Plan Section 4: no re-scoring. |
| A3a | Split = the paper's seed 0 (first G4 split). LOO is run over **all** 716 calibration calls (202 refusals + 514 others), a superset of the plan's "random 200 others"; the plan's random 200 (seed 0) are flagged and summarised separately. | It costs 14 s in total. |
| A3b | Utilities: test-half safe autonomy (primary), calibration-half safe autonomy, and log10 LTT slack of the selected policy. A change counts as material at ≥ 1 point of test safe autonomy. | liquid has an exact cost tie (up to float rounding) between λ_a = 0.2 and 0.5. Removing *any* call flips it, by 0.37 points (one safe test call). Without the threshold, every LOO for liquid would count as a "policy change". |
| A3c | The top-20 ranking is kept in `a3_top20.csv`, but the findings report the **boundary holders** (material changes) instead. | Beyond the few holders, the top 20 consists of ties in slack (every refusal outside the region shifts slack by the same 0.02). |
| A3d | Truncated Monte-Carlo Data Shapley is **not** run. | LOO is informative only for 6 (model, call) pairs; each holder alone moves the boundary, so Shapley would mostly split the same effect among them. |
| A3e | Added A3b (exploratory): G4 over the paper's 60 splits, calibrated with and without the multi-turn calibration calls. Test harm is reported separately for single- and multi-turn calls. | It tests the plan's Section 4 hypothesis at the certificate level, at no cost. |

## Changes after Tomy's instructions of 3 Nov (afternoon)

| # | Choice | Reason |
|---|---|---|
| T1 | `BUDGET_USD` raised from 10 to 50; the 80% stop now triggers at USD 40. | Tomy: cost is no longer a constraint, so the cap only guards against runaway loops. |
| T2 | microsoft is kept and labelled "microsoft (post-drift)" in all Phase B outputs. | Tomy. |

## Phase B pilot

| # | Choice | Reason |
|---|---|---|
| B1 | Pilot = the 50 calls already drawn in S4: stratified by label × single/multi-turn × enterprise/benchmark, seed 0. 28 single-turn, 22 multi-turn. | Plan. |
| B2 | M2 donors: direct_execute calls of the **test split** from a **different service**, scored (first) message. | Plan; the test split keeps donor content in-distribution. |
| B3 | M2 for `tool_args.<k>` (Tomy approved): use the same key from a donor if available, else a value of the same JSON type, else any argument. | Argument names are tool-specific. Pilot: 49 same_key, 94 type_matched, 0 any_arg. |
| B4 | Donor values must be non-empty (not null, "", [] or {}) and must differ from the original. | A null donor would duplicate M1. A bug where a null donor was mistaken for "no donor" was found in the dry run and fixed before any call. |
| B5 | Same-run baselines: `Querier.run(..., require_tag="b_pilot")` treats cache entries from other runs as misses. 35 jobs overlapped with S2 (4 pilot calls are S2 calls) and are re-queried; the newer record supersedes the older one. | Plan rule 5. |
| B6 | Identical scored inputs inside the batch are queried once: 15 duplicate states (multi-turn calls sharing a first message), 75 jobs. | They are byte-identical inputs. |
| B7 | Noise threshold per model and class = max(2·√2·SD_S3, 0.005). | The SD of a difference of two noisy queries is √2·SD, and 2 of those give ≈ 95%. The 0.005 floor is for the deterministic gpt/pplx; for gpt it effectively means one 0.01 grid step. |
| B8 | Effect size per unit = total-variation distance between the baseline and masked probability vectors (½ Σ_c \|Δp_c\|), plus the signed Δ per class. A flip = argmax choice changes. | One scalar per unit for rankings; the signed per-class values are kept. |
| B9 | M1/M2 agreement per (call, model): Spearman ρ over units (TV, and signed Δp_refuse), only with ≥ 3 units and non-constant vectors; top-unit agreement; counts of units above noise in both / only one. | Plan: rank correlation per sample. |
| B10 | Worked examples: for each category (single-turn refusal, single-turn ask, multi-turn refusal), the pilot call with the most above-noise M1 units summed over models. | A deterministic rule, not cherry-picked by hand. |

## Round 3 (Tomy's decisions of 3 Nov, evening)

| # | Choice | Reason |
|---|---|---|
| R1 | `provider_refusal` is its own outcome everywhere: `query.outcome()` returns ok / provider_refusal / error. Provider refusals are cached as final (never retried, never a missing value) and appear in every output table. | Tomy. A refusal costs 5 retries (~31 s), and it is deterministic in the pilot. |
| R2 | Phase C uses removal only (M2 dropped); the pilot comparison stays in FINDINGS. | Tomy. |
| R3 | 8 workers for Steps 4/5, Phase C and Shapley; the budget is re-checked every 4,000 calls (spend tracked in memory). | Tomy (8 workers); the cache file is too large to re-read before every batch. |
| R4 | Step 4 "today" = all 1,436 test calls × 5 models with the paper input, one run (tag `today_test`, fresh). Records in score_split format + `outcome` + `input`, in `runs/today/test_scores.jsonl`. | Tomy. |
| R5 | Corrected input (Step 5): one user message holding the whole conversation, turn by turn, with the last user turn marked "CURRENT PROPOSAL TO AUDIT"; it is passed to the **unchanged** `call_openrouter_decision` (`src/scoring_fixed.py`). Only the text after "### Current Context, Tool Proposal & Ambience" differs. | Keeps endpoint, schema, retries and the state template identical; score_split.py is untouched. |
| R6 | The assistant turns are kept in the corrected input. They are the reviewer decisions on the EARLIER proposals of the session, not the label of the final call, but they tell the scorer how the earlier steps were judged. | Tomy asked for the full conversation; flagged as a possible source of optimism in the after-scores. |
| R7 | Before/after design. **Before** = today's paper-input scores (test: Step 4; train: the 733 multi-turn calls re-scored today with the paper input, `runs/today/train_multiturn_scores.jsonl`). **After** = the same, with the multi-turn calls replaced by the full-conversation scores. Train single-turn calls keep their stored 23/27 Oct scores in both. | Before and after then differ only in the input of the multi-turn calls, not in the date (microsoft drifted after 23 Oct). |
| R8 | Score sources are selected with `XAI_SOURCE` = paper / today / today_fixed. Outputs go to `results/` (paper), `results/today/` and `results/today_fixed/`. Non-paper sources are restricted to the paper's 1,433 / 7,755 calls. | Reproducible before/after with unchanged scripts. |
| R9 | Phase C noise rule for jev, microsoft and liquid: attribution = difference of two 3-repeat means. "Above noise" if its TV exceeds the 95th percentile of an empirical null: pooled within-variant residuals (× √(3/2)), two means of 3 drawn with replacement, B = 20,000, seed 0. gpt and pplx: TV > 0.005. | The pilot showed that the per-class Gaussian threshold fires on 15–30% of pure noise. The bootstrap keeps the heavy tails and the coupling of the three classes, and gives one 5% test per unit. |
| R10 | D2 uses the mapping `results/riskclass_unit_map.csv`, written from TASK_AUTHORING §6 and marked **UNREVIEWED**; D2 results are provisional. The annotator-reasoning reference uses simple string matching (field names, tool name, argument keys ≥ 4 characters, argument values ≥ 4 characters). | Tomy. |
| R11 | Exact Shapley: calls with ≤ 8 units, gpt and pplx; v_c(S) = p_c with only the units in S present (others removed as in M1). Exact φ and the pairwise Shapley interaction index from all 2^n values; validated on toy games. Cached answers of identical inputs are reused regardless of run (deterministic models). | Tomy. Determinism makes repeats unnecessary. |

## Round 4 (Tomy's decisions of 3 Nov, late evening)

| # | Choice | Reason |
|---|---|---|
| L1 | pplx: 3 repeats in Phase C (queried after the four other models, same tag `phase_c`); exact Shapley for gpt only. | Tomy; pplx became non-deterministic on 3 Nov (`det_check_evening1.csv`). |
| L2 | Leakage check: four multi-turn inputs (truncated, full, users_only = (a), final_only = (b)) queried in one run with an identical design: 3 fresh repeats (mt0–mt2) for jev, microsoft, liquid and pplx, 1 run for gpt. The Step 5 single-run "full" result stays in FINDINGS but is superseded by this run. | Equal noise treatment; averaging 3 repeats would otherwise favour the new variants. |
| L3 | (a) = all user turns verbatim under a header, without the reviewer turns; (b) = the final user message verbatim. (b) never contains `user_task` (0 of 123 test calls); it holds the labelled call and its `prior_steps`. | Tomy. |
| L4 | Main figure in FINDINGS = variant (a); full conversation = upper bound; (b) is reported alongside. | (a) keeps all information the agent proposed and the user said, and removes only the reviewer's earlier verdicts. (b) also removes the user's task, so it tests a different question. |
| L5 | Single-turn calls in every variant = Step 4's today scores (1 run); only the multi-turn calls differ. | Before/after differs only in the multi-turn input. |
| L6 | Bug found and fixed: waiting loops that used `pgrep -f "c_occlusion.py query"` matched their own command line and would never have ended. They were stopped before doing anything and replaced by waiting on the PID. No duplicate runs happened. | Process hygiene. |

## Round 5 (4 Nov, analysis of the overnight runs)

| # | Choice | Reason |
|---|---|---|
| N1 | Leakage table: dh_app and atbench AUROC are computed as in A2, i.e. the service's refusals vs its non-refusals across all turns. Every dh_app refusal is multi-turn and every dh_app non-refusal single-turn, so a multi-turn-only AUROC is undefined; only the refusals' scores change between inputs. atbench is also reported multi-turn-only. | Comparable to the Step 5 / A2 figures. |
| N2 | In the leakage G4 table, scorers that never certify (gpt, 0/60 under every input) are kept with a certified share of 0, not dropped. | The first `compare` crashed on the mismatch; fixed by a merge. |
| N3 | Phase C collect with pplx as a noisy model (3 repeats, residual-bootstrap null like jev, microsoft and liquid). pplx is added to the D5 test-retest analysis. | L1. |
| N4 | gpt is kept under the deterministic 0.005 rule after a check: the Phase C baseline equals Step 4 on all 1,433 calls up to float rounding (218 differ by TV < 0.0005, 0 choice changes); det_check_nov4 gives 60/60 identical. Because the rule counts 0.01 jumps, gpt's share above TV 0.04 is reported too. | gpt's above-noise share (84–94%) looked suspicious; it is real, but coarse. |
| N5 | Exact Shapley for gpt only (1,016 calls ≤ 8 units, 123,056 coalitions), after re-verifying determinism on 4 Nov 02:29 UTC. Cached Phase C answers for identical inputs are reused. | Tomy (L1). |
| N6 | `Querier.run` now raises on the first HTTP 402 (account out of credits) and cancels the pending futures. | The Shapley run kept submitting for ~20 min after the OpenRouter balance ran out (24,683 × 402, no cost). Errors were never cached, so re-running fills exactly the gaps. |
| N7 | Shapley analysis uses only calls whose 2^n coalitions all returned ok. Calls with any provider_refusal (433) or missing coalition (274, credits) are excluded and counted, never imputed. The refusal pattern over all coalitions is saved (`e_provider_refusal_coalitions.csv.gz`). Bug fixed: the per-unit "hidden" flag compared DataFrames with misaligned column names and was always False. | Honest reporting; Shapley is undefined with missing values. |
| N8 | Shapley completed after the credit top-up: gpt re-verified deterministic at 09:04 UTC on 4 Nov (`det_check_nov4b.csv`, 60/60); only the 24,922 missing coalitions were queried (0 errors). The provisional summary is kept as `e_shapley_summary_provisional.json`. | Same design as N5; the answers to identical inputs are reused. |
