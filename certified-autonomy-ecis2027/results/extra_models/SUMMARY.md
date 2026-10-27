# Additional decision models: liquid/d1-ultra and perplexity/pplx-decider-v1.2

Run on 27 October 2026. **Exploratory, not pre-registered:** the confirmatory protocol was frozen on 23 October with three models. The two models were added afterwards, and their frozen G4 policies were set on 27 October, after the confirmatory results of the other three were known. No prediction in `PROTOCOL.md` was stated for them. The binding confirmatory verdicts remain those in `results/confirmatory/`.

## What was run

- **Scoring:** `extra_models/score_extra.py` imports `confirmatory/score_split.py` unchanged and replaces only the model list. It scored test (1,436 calls), Mailu (198) and train (7,767) with both models. That is 18,802 calls, 0 errors, USD 0.71. Versions returned: `liquid/d1-ultra-20261015`, `perplexity/pplx-decider-v1.2-20261016`.
- **Analysis:** `extra_models/make_copies.py` copies `analysis/02`–`09`, `confirmatory/confirm_analysis.py` and `check_run.py`, and adds the two models to the model lists. Each edit is an exact replacement with a count check. The mean of three stays jev + microsoft + gpt. `extra_models/run_extra.sh` runs everything. The originals are untouched; `verify_frozen.py` reports OK.
- **Same calls:** the new models have no failed calls, so every analysis uses the same 1,433 test and 7,755 train calls as before.
- **Reproduction check:** `extra_models/compare_original.py` confirms that every number for jev, microsoft, gpt and the mean is identical to the original results, in all eight paper result files and in the confirmatory results.
- **Not covered:**
  - C1 drift: the new models have no earlier run to compare against.
  - The rounding manipulation and the jev + gpt grid ensemble, which are specific to those vendors.
  - The figures (`figures/make_figs.py`).

## Test split (paper analyses)

| Scorer | Accuracy | Harmful execution, no gate | ECE | Refusals at p = 0 | Refusals at p ≤ 0.01 |
|---|---|---|---|---|---|
| jev | 64.9% | 15.8% | 0.148 | 7.9% | 13.3% |
| microsoft | 59.7% | 26.4% | 0.307 | 0.0% | 22.5% |
| gpt | 62.3% | 20.7% | 0.252 | 2.7% | 12.1% |
| **liquid** | **70.0%** | 21.7% | **0.079** | 0.0% | **1.5%** |
| **pplx** | 65.2% | 24.0% | 0.190 | 0.0% | 8.2% |
| mean of three | 63.4% | 24.4% | 0.185 | 0.0% | 9.9% |

Gates, 50/50 splits, harm cost 10:

| Scorer | G4 safe autonomy | G4 harm | G4 cost | G4 over a budget | G2 over harm budget | G1 over ask budget |
|---|---|---|---|---|---|---|
| jev | 32.0% | 1.9% | 0.82 | 0% | 0% | 0% |
| microsoft | 26.0% | 0.8% | 0.84 | 0% | 30% | 36% |
| gpt | not certifiable | | | | 0% | 1% |
| **liquid** | **49.5%** | 1.7% | **0.73** | 1.7% harm | 15% | 0% |
| **pplx** | **49.2%** | 1.7% | **0.75** | 1.7% ask | 16% | 6% |
| mean of three | 33.8% | 1.6% | 0.81 | 1.7% | 24% | 3% |

- **Floor:** neither new model returns exact zeros, and liquid almost never gives a true refusal a probability at or below 0.01 (1.5%). This is consistent with the paper's floor argument: the model with the lowest floor reaches the highest certified autonomy.
- **G4:** both new models are certifiable in every split and at every harm cost. They execute about half of the safe calls, compared with 26–34% for the original models.
- **Confidence and budget (Table 5):** at a 2% harm budget, both models are certifiable only at 80% confidence. At 5%, they are certifiable at every confidence level, with 42–50% safe autonomy.
- **G5 with 90% calibration:** for both new models, G5 executes fewer safe calls than G4. Liquid reaches 45.1% vs 53.5%, and pplx 25.2% vs 60.6%. This is the opposite of the paper's G5 claim for the mean of three (43% vs 33%). "G5 per service" stays above G4 for liquid only (61.8%).
- **Vendor dependence (R5):** the new models' confident errors coincide strongly with the others' (lift 3.4–12.3, Spearman 0.79–0.93). The liquid–pplx pair has the highest lift (12.3).
- **Deployment policies** (G4 on all 1,433 test calls, used as frozen policies below):
  - liquid: execute if p(refuse) ≤ 0.01 and p(ask) ≤ 0.20; block if p(refuse) ≥ 0.5.
  - pplx: execute if p(refuse) ≤ 0.005 and p(ask) ≤ 0.05; block if p(refuse) ≥ 0.5.

  The confirmatory script recomputes both and finds the same values.

## Untouched calls (confirmatory analyses, exploratory for these models)

| Frozen G4 policy | Safe autonomy | Harmful execution [95% CI] | Unconfirmed execution [95% CI] | Cost |
|---|---|---|---|---|
| train / jev | 27.6% | 1.6% [1.0%, 2.3%] | 4.5% [3.6%, 5.5%] | 0.82 |
| train / microsoft | 24.1% | 1.5% [0.9%, 2.2%] | 5.2% [4.3%, 6.3%] | 0.85 |
| train / mean of three | 27.3% | 1.5% [1.0%, 2.2%] | 5.0% [4.1%, 6.1%] | 0.82 |
| **train / liquid** | 48.4% | **4.6% [3.6%, 5.7%]** | 11.8% [10.4%, 13.3%] | 0.78 |
| **train / pplx** | 53.6% | **6.4% [5.3%, 7.7%]** | 13.7% [12.2%, 15.3%] | 0.80 |
| Mailu / liquid | 7.8% | 0.0% [0.0%, 5.2%] | 0.0% [0.0%, 4.6%] | 0.67 |
| Mailu / pplx | 7.8% | 0.0% [0.0%, 5.2%] | 1.3% [0.0%, 6.9%] | 0.76 |

- **Frozen policies:**
  - **pplx exceeds the 5% harm budget** on untouched calls (6.4%, lower bound 5.3%). Under the P4 rule it would be "not confirmed".
  - **Liquid stays within budget** on the point estimate (4.6%), but its upper bound is above 5%.
  - Both new policies are much more permissive than the original three: about 50% safe autonomy, compared with 24–28%. They use up most of the budget, while the original policies use about a third of it.
  - The protocol notes that the guarantee holds with 90% probability per calibration sample, so one failure is possible even for a correct design. But it is the first frozen policy that fails.
- **Recalibrated on the train split (C3):** G4 is never over a budget for either model (0 of 4 disjoint samples, 0% of 100 resamples). Safe autonomy there is 20–23%, similar to the original models.
  - G1 for pplx exceeds the harm budget in 57% of resamples.
  - G2 for liquid exceeds the harm budget in 29% and the ask budget in 41% of resamples.
- **Full pool (C4), harm cost 10:**
  - Liquid: G4 36.7% safe at 3.3% harm; G5 14.1% at 0.6% harm.
  - Pplx: G4 35.9% safe at 3.5% harm; G5 22.1% at 1.9% harm.
  - G5 again executes fewer safe calls than G4 (as P8c for the mean of three).
- **Mailu:** neither frozen policy executes a refusal (0 of 69), but each executes only 7.8% of safe calls. Gates calibrated on the train pool execute no refusals either.

## Points to discuss before using these in the paper

1. **Framing:** report as an additional, exploratory comparison. These models' policies were set after the confirmatory results were known, and adding scorers changes a frozen design element, so it needs a `DEVIATIONS.md` entry.
2. **The pplx result** is the most important one for the paper's argument. Its policy certified on the test split does not hold its harm budget on 7,755 untouched calls, while all original policies do. Recalibration on the untouched data does hold. Part of the explanation may be the overlap between test and train (650 identical prompts), which this run does not separate out.
3. **Better models, same G5 finding:** the new models are better calibrated and have lower floors, which roughly doubles the certified autonomy on the test split. G5 still does not beat G4 for them.

## Files

| Path | Content |
|---|---|
| `runs/extra_models/{test,mailu,train}_scores.jsonl`, `*_check.txt` | raw scores and checks of the new models |
| `runs/extra_models/combined/` | original + new scores, input of the analyses |
| `results/extra_models/r02`–`r09*.json/csv`, `logs/` | paper analyses with five models |
| `results/extra_models/confirmatory/REPORT.md`, `paper_table.md`, `confirmatory_results.json`, `C3_per_sample.csv` | confirmatory analysis with five models |
| `extra_models/` | `score_extra.py`, `make_copies.py`, `run_extra.sh`, `compare_original.py` and the script copies |
