# Data

## Included

| File | Content |
|---|---|
| `scores/test_1436_scores.jsonl` | The original scoring run (notebook `02_full_evalrun`): one line per call, 1,436 test-split samples x 3 decision models = 4,308 calls; 3 gpt calls failed. All paper results are computed from this file. |
| `scores/pilot_batch100_scores.csv` | Pilot run of notebook `01_testing_openrouter` on 100 samples; not used in the paper. |

Fields: `dataset_idx` (position in the split file), `service`, `risk_class`, `risk_family`, `ground_truth` (corpus label), `model`, `pred_choice`, `is_match`, `p_direct_execute`, `p_ask_human`, `p_refuse`, `prob_sum`, `latency_ms`, `cost_usd`, `error`. Runs made with `confirmatory/score_split.py` add `split`, `scored_at` and `response_model`.

## Not included

The SAFETY SENTRY corpus itself (Chen et al., 2026, release `full_data_20260530`): `test_1436.json`, the train split (7,767 samples) and the Mailu out-of-distribution split (198 samples). Obtain them from the corpus release and place them in `data/raw/` (ignored by git). Each sample has a `prompt` (system and user message) and a `meta` block with `decision`, `service`, `risk_class` and `risk_family`. Labels were produced by two language-model annotators and arbitrated by the corpus authors; a human audit of the test split found 92% agreement.
