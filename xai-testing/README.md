# xai-testing

XAI experiments on the certified delegation gate (plan: `xai_plan.md`; results: `FINDINGS.md`; choices: `DECISIONS.md`).
Reads from `../certified-autonomy-ecis2027/` and never writes there. Run everything from this folder with the
repo's Python environment (numpy, pandas, scipy, matplotlib, requests, python-dotenv).

## Layout

```
config.py        paths, models, seeds, budget cap (USD 50, stop at 80%), Phase 0/A settings
src/io.py        load splits and scores (last record per call, paper's failure rule), multi_turn, service_type
src/units.py     parse the scored (first) user message into units; exact re-serialisation; masking (remove/replace)
src/query.py     cached (cache/responses.jsonl), resumable, budget-checked calls through score_split.call_openrouter_decision
src/gate.py      copies of the paper's G4/G5 code + reproduction check + LTT helpers
experiments/     one script per step (below)
results/         outputs; never edited by hand
cache/           API responses (append-only JSONL); repeats in cache/repeats.jsonl
```

## Steps

| step | command | API calls | output |
|---|---|---|---|
| S1 round trip | `python experiments/p0_sanity.py s1` | none | `results/p0_s1_*` |
| S2 drift | `python experiments/p0_sanity.py s2` (`--dry` to preview) | 150 | `results/p0_s2_*` |
| S2 add-on | `python experiments/p0_drift_train_overlap.py` | none | `results/p0_s2_train_overlap.*` |
| S3 noise | `python experiments/p0_sanity.py s3` | 150 (repeats) | `results/p0_s3_*` |
| S4 estimate | `python experiments/p0_sanity.py s4` | none | `results/p0_s4_estimate.json` |
| A1 label audit | `python experiments/a1_label_audit.py` | none | `results/a1_*` |
| A2 diagnostics | `python experiments/a2_certificate_diagnostics.py` | none | `results/a2_*` |
| A3 LOO influence | `python experiments/a3_calibration_influence.py` (needs A1) | none | `results/a3_*` |
| A3b multi-turn what-if | `python experiments/a3b_multiturn_whatif.py` | none | `results/a3b_*` |
| figures | `python experiments/figures_p0_a.py` | none | `results/fig_*.png` |
| A1b duplicates export | `python experiments/a1b_export_duplicates.py` | none | `results/a1b_train_test_label_conflicts.csv` |
| B pilot | `python experiments/b_occlusion.py --pilot` (`--dry` to preview) | 3,875 | cache, tag `b_pilot` |
| B pilot analysis | `python experiments/b_occlusion.py --pilot --analyze` | none | `results/b_pilot_*` |
| B noise check | `python experiments/b_noise_check.py` (after the pilot) | 200 (repeats) | `results/b_pilot_noise_check.*` |
| Step 4 today re-score | `python experiments/s45_rescore.py today` | 7,180 | `runs/today/` |
| Step 4 gate compare | `python experiments/s4_gate_compare.py` | none | `results/s4_*` |
| Step 5 full conversation | `python experiments/s45_rescore.py fixed`, then A-scripts with `XAI_SOURCE=today` / `today_fixed`, then `python experiments/s5_before_after.py` | 4,280 | `runs/fixed_multiturn/`, `results/s5_*` |
| Step 5b leakage check | `python experiments/s5b_leakage.py query` / `compare` | ≈ 44.5k (4 inputs × 3 repeats) | `runs/multiturn_variants/`, `results/s5b_*` |
| determinism check | `python experiments/det_check.py <label>` | 240 | `results/det_check_<label>.csv` |
| Phase C occlusion | `python experiments/c_occlusion.py query` then `query --pplx-reps 3 --only perplexity/pplx-decider-v1.2`; `collect --pplx-reps 3` | ≈ 160k | `results/c_long.csv.gz`, `c_noise.json` |
| Phase D (D0–D5) | `python experiments/d_analysis.py` | none | `results/d_*` |
| Shapley (gpt) | `python experiments/e_shapley.py query` / `analyze` | 123,056 coalitions (complete) | `results/e_*` |

Re-running any paid step costs nothing: cached responses are reused (the cache key is
sha256(model + "\n" + state string)). Check spending with
`python -c "from src import query as Q; print(Q.spent_usd(), Q.spent_by_tag())"`.

Gate reproduction check: `python -c "from src import io as xio, gate as G; P,m=xio.prob_matrices('test'); y=xio.label_index(m.ground_truth); print(G.check_reproduction(P['jev'], y, 'jev-1.14'))"` prints 0.
