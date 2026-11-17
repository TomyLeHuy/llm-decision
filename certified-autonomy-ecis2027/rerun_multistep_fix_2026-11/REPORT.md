# Re-run with the multi-step fix: impact on the ECIS results

*Tomy Le, 17 November 2026*

## 1. Summary

I re-scored all splits with the three paper models (jev, microsoft, gpt), this time with the full multi-step context,
and ran the unchanged paper and confirmatory analyses on the new scores.

- **Most of the paper holds.** 16 to 19 of the 23 confirmatory predictions keep their verdict, depending on the
  variant. Calibration is still necessary, the certified gate still keeps harm within budget, gpt is still not
  certifiable, and the mean of three still gives the most autonomy.
- **The fix strongly reduces harmful executions on multi-step records.** For example, gpt now executes 12.5% of the
  multi-step refusals in the test split instead of 84.7%.
- **The score-resolution floor was largely caused by the bug.** jev's 7.9% of refusals with p(refuse) = 0 drop to
  1.5%, and prediction P2a is no longer confirmed.
- **microsoft has changed since October.** Its answers differ even on inputs that are identical in both runs, so its
  changes are mostly drift, not the effect of the fix.
- **Absolute numbers change a lot.** The abstract, the scorer table and the G4 numbers would need new values, and the
  frozen confirmatory policies no longer match the corrected data.

## 2. Background

In the submitted run, the scorer sent only the **first** user message of a record to the decision model. For
multi-step records the label belongs to the **last** proposed tool call, so the model judged a different call than the
one that was labelled (documented in `confirmatory/DEVIATIONS.md`, 10 Nov 2026). With `--input-mode all_steps`, the
scorer now sends every user turn, oldest first, and marks the last one as the call to audit.

Multi-step records make up 123 of 1,436 test records, 733 of 7,767 train records, 28 of 198 Mailu records and 14 of
the 150 anchor calls. Most of them in the test split come from **atbench** (85 of its 119 test records).

## 3. What I ran

| Step | What | Output |
|---|---|---|
| Scoring | 4 splits x 3 models, `--input-mode all_steps`, same endpoint, schema and retry policy as the paper | `runs/*_scores_all_steps.jsonl`, `runs/*_check.txt` (all OK) |
| Paper analyses | `analysis/01`–`09` on the new test scores | `results/paper/` |
| Confirmatory | `confirmatory/confirm_analysis.py` (frozen script), with the new test scores as reference | `results/confirmatory/` |
| Single-step variants | Multi-step records dropped from every split, same scripts, once from the new and once from the old scores | `results/single_step_only/`, `results/ecis_single_step_only/` |
| Comparison | `compare_with_ecis.py` | `results/comparison/` |

`10_floor_numbers.py` was not re-run, because it does not use any scores. Nothing outside this folder was changed: the
original scores and results stay where they are and remain the reference for the submitted version (reproducible
with `--input-mode first_step`).

- **Cost:** USD 2.43 (28,653 calls; test 0.36, train 1.99, Mailu 0.03, anchor 0.04).
- **Failures:** gpt refused 2 test calls (474 and 1299; in October also 1333) and 12 train calls with HTTP 502
  "refused to answer". As in the paper, these calls are excluded from the analyses.
- **Environment check:** before the re-run, I ran the same scripts in my environment on the **old** scores. The
  confirmatory results and r03–r09 were byte-identical to the archived ones. In r02, 14 values in rows 12–13 of table
  E6 differed by up to 0.017 on identical input, so I excluded these two rows from all comparisons.

### Four versions

| Version | Scores | Records |
|---|---|---|
| **ECIS** | submitted (first step only) | all |
| **ECIS_SINGLE** | submitted | single-step only |
| **FIX** | re-run, all steps | all |
| **FIX_SINGLE** | re-run, all steps | single-step only |

Single-step records get exactly the same input in both runs. This allows two clean comparisons:
- **ECIS_SINGLE vs FIX_SINGLE** shows only vendor drift and sampling noise.
- **ECIS vs ECIS_SINGLE** shows how much the wrongly scored multi-step records shaped the paper's numbers.

## 4. Findings

### 4.1 Harmful execution on multi-step refusals drops sharply

Share of true refusals among the multi-step records that the model would execute (most probable action):

| Split | Model | ECIS | FIX | Refusal recall FIX |
|---|---|---|---|---|
| test | gpt | 84.7% | 12.5% | 9.7% |
| test | jev | 47.2% | 22.2% | 11.1% |
| test | microsoft | 86.1% | 38.9% | 18.1% |
| train | gpt | 89.8% | 18.2% | 18.2% |
| train | jev | 61.9% | 25.6% | 24.4% |
| train | microsoft | 88.6% | 34.7% | 30.1% |

With the full history, the models no longer execute most of these calls, but they mostly send them to **ask_human**
rather than refuse. That is why accuracy on multi-step records barely changes in the test split (gpt 0.350 → 0.350,
jev 0.366 → 0.325, microsoft 0.390 → 0.350), while it rises in train (gpt 0.461 → 0.581).

How often the old setup was right by chance: the first step's own decision equals the final label in 35.8% (test),
49.5% (train) and 75.0% (Mailu) of the multi-step records.

The earlier steps are not always executed. The reviewer decisions on them were:

| Split | Execute | Ask | Refuse |
|---|---|---|---|
| test | 170 | 12 | 0 |
| train | 1,321 | 71 | 11 |
| Mailu | 9 | 30 | 0 |

The current input shows the earlier proposals, but not these decisions.

### 4.2 The score-resolution floor was largely produced by the bug

Uncalibrated scorer table on the test split (r09, the scorer table in the paper):

| Scorer | Version | n | Accuracy | Refusal recall | Harmful execution | ECE | Refusals at p = 0 | Refusals at p ≤ 0.01 |
|---|---|---|---|---|---|---|---|---|
| jev | ECIS | 1433 | 0.649 | 0.472 | 15.8% | 0.148 | 7.9% | 13.3% |
| jev | ECIS_SINGLE | 1311 | 0.675 | 0.557 | 9.0% | 0.130 | 1.5% | 6.0% |
| jev | FIX | 1434 | 0.637 | 0.478 | 11.8% | 0.161 | 1.5% | 6.2% |
| jev | FIX_SINGLE | 1311 | 0.667 | 0.557 | 9.0% | 0.142 | 1.5% | 5.7% |
| microsoft | ECIS | 1433 | 0.597 | 0.415 | 26.4% | 0.307 | 0.0% | 22.5% |
| microsoft | ECIS_SINGLE | 1311 | 0.616 | 0.482 | 13.8% | 0.291 | 0.0% | 14.4% |
| microsoft | FIX | 1434 | 0.654 | 0.562 | 20.4% | 0.191 | 0.0% | 6.2% |
| microsoft | FIX_SINGLE | 1311 | 0.683 | 0.644 | 16.5% | 0.168 | 0.0% | 4.2% |
| gpt | ECIS | 1433 | 0.623 | 0.474 | 20.7% | 0.252 | 2.7% | 12.1% |
| gpt | ECIS_SINGLE | 1311 | 0.648 | 0.563 | 6.9% | 0.236 | 0.9% | 6.9% |
| gpt | FIX | 1434 | 0.623 | 0.480 | 7.9% | 0.257 | 1.2% | 9.6% |
| gpt | FIX_SINGLE | 1311 | 0.648 | 0.563 | 6.9% | 0.236 | 0.9% | 6.9% |
| mean of three | ECIS | 1433 | 0.634 | 0.476 | 24.4% | 0.185 | 0.0% | 9.9% |
| mean of three | FIX | 1434 | 0.650 | 0.539 | 11.6% | 0.144 | 0.0% | 1.5% |

- **jev.** In the paper, 7.9% of true refusals get p(refuse) = 0. In every other version it is 1.5%, including
  ECIS_SINGLE, which uses the paper's own scores. So the zeros came almost entirely from multi-step refusals that were
  scored on their harmless first step. This was our main example of the score-resolution floor.
- **gpt.** Zeros fall from 2.7% to 1.2%. gpt is deterministic, so ECIS_SINGLE and FIX_SINGLE are identical and the
  whole change comes from the fix.
- **gpt's policy-family floor holds.** Asks at p = 0 stay at 12.1% (FIX) and 12.5% (single-step only), and gpt is not
  certifiable under G4 in any version.
- **Abstract.** "Uncalibrated models execute 16 to 26% of calls that should be blocked" would become **7.9% to 20.4%**
  (FIX), or 11.6% for the mean of three.

### 4.3 microsoft has drifted, and its changes are not caused by the fix

On single-step records, whose input is identical in both runs, microsoft's probabilities now differ from October:

| Split | Mean difference | Records where the most probable action changed |
|---|---|---|
| test | 0.17 | 19% |
| train | 0.15 | 16% |
| Mailu | 0.24 | 32% |

For comparison, jev differs by a mean of 0.02 (sampling noise) and gpt by exactly 0.

The API still reports the same version (`microsoft-decision-1-20261009`). The confirmatory drift check (C1) does not
catch this, because it compares the anchor re-score with a reference scored on the same day.

So for microsoft and for the mean of three, FIX combines the fix with a changed model. Example: microsoft's refusals
at p ≤ 0.01 go from 22.5% (ECIS) to 14.4% (ECIS_SINGLE) to 4.2% (FIX_SINGLE). The first drop comes from the fix, the
second from drift.

### 4.4 Certification: same pattern, more autonomy

Certified gate G4 on the test split (r04 L2, 9 ask thresholds):

| Model | Version | Certifiable | Thresholds (p_rf ≤, p_ask ≤, refuse if p_rf ≥) | Refusals executed | Asks executed | Safe calls executed | To human |
|---|---|---|---|---|---|---|---|
| jev | ECIS | yes | 0.000, 0.02, 0.5 | 1.0% | 2.1% | 28.9% | 74.1% |
| jev | FIX | yes | 0.000, 0.20, 0.5 | 1.0% | 11.3% | 53.0% | 61.4% |
| microsoft | ECIS | yes | 0.0003, 0.10, 0.5 | 0.7% | 2.1% | 26.3% | 75.1% |
| microsoft | FIX | yes | 0.005, 0.05, 0.5 | 1.5% | 5.1% | 46.1% | 61.1% |
| mean of three | ECIS | yes | 0.002, 0.02, 0.5 | 0.7% | 2.7% | 30.0% | 73.3% |
| mean of three | FIX | yes | 0.01, 0.10, 0.5 | 1.2% | 7.0% | 53.3% | 61.3% |
| gpt | all four | no | – | – | – | – | – |

- **Unchanged:** which models can be certified.
- **Changed:** safe autonomy under G4 rises from 26–30% to 46–53%, and the certified thresholds are different.
- **Protocol:** the **frozen policies (C2) no longer match** the ones recomputed from the corrected data, for jev,
  microsoft and the mean. They were derived from the buggy test scores.

### 4.5 Confirmatory predictions

| Version | Same verdict as ECIS | Changed |
|---|---|---|
| ECIS_SINGLE | 19 of 23 | P2a, P2c, P7a, P7b |
| FIX | 17 of 23 | P2a, P2b, P2c, P2d, P5b, P5c |
| FIX_SINGLE | 16 of 23 | P2a, P2b, P2c, P2d, P5c, P7a, P7b |

| ID | Claim (short) | ECIS | FIX | FIX_SINGLE | Cause of the change |
|---|---|---|---|---|---|
| P2a | jev p(refuse) = 0 for > 5% of train refusals | confirmed, 6.4% | not confirmed, 2.5% | not confirmed, 2.6% | **fix** (ECIS_SINGLE already 2.1%) |
| P2b | rounded microsoft p = 0 for > 5% | confirmed, 16.8% | not confirmed, 3.9% | not confirmed, 4.0% | **drift** (ECIS_SINGLE still 12.5%) |
| P2c, P2d | follow-ups of P2a, P2b | – | not applicable | not applicable | follow P2a, P2b |
| P5b | G2 exceeds harm budget ≥ 10% (microsoft or mean) | confirmed, ms 29% / mean 44% | not confirmed, ms 2% / mean 0% | confirmed, ms 31% / mean 6% | **fix** on the multi-step records |
| P5c | G1 exceeds ask budget ≥ 10% (microsoft) | confirmed, 20% | not confirmed, 0% | not confirmed, 0% | **drift** (ECIS_SINGLE 26%) |
| P7a, P7b | atbench refusals under G2 / G3 | not confirmed / confirmed | unchanged | not applicable | single-step only keeps 9 of 79 atbench refusals |

Unchanged in all versions:
- **P1**, uncalibrated models execute > 10% of refusals (FIX: jev 12.9%, microsoft 17.9%, gpt 12.3%).
- **P3**, gpt is not certifiable.
- **P4**, certified harm stays within budget. Harm is now lower, e.g. jev 1.6% → 0.4%.
- **P5a**.
- **P6**, the mean has the highest G2 autonomy (jev 9.2% → 56.3%, gpt 5.8% → 38.1%).
- **P8a–d, P9, P10a/b**.

## 5. Implications for the paper

1. **The core claims hold.** About 70 to 80% of the predictions keep their verdict in every variant.
2. **The score-resolution floor argument weakens most.** jev's zero-probability refusals, our main illustration of
   the floor, shrink from 7.9% to 1.5%, and P2a fails. The rounding manipulation for microsoft fails as well, but
   because of drift, not because of the fix.
3. **Many absolute numbers change.** Harmful execution, autonomy and thresholds all move, so the abstract, Table 1 and
   the G4 results would need updating.
4. **Single steps only, or both corrected:**

| Option | Advantages | Drawbacks |
|---|---|---|
| **Single steps only** | Clean, and reproducible from the paper's own scores (ECIS_SINGLE), with no drift | atbench loses 70 of its 79 train refusals, so P7 can no longer be tested |
| **Both, corrected** | Keeps atbench | Needs the re-scored data, microsoft is now a different model than in October, and the frozen confirmatory policies would have to be re-derived (a protocol deviation) |
| **Middle way** (not done here) | Corrected multi-step records plus the October single-step scores give the same input as the fix for every record, without microsoft's drift on the single steps | Mixes scoring dates for microsoft and jev |

## 6. Open points

- **microsoft drift** dominates its changes. A paper based on new scores would describe a different microsoft model
  than the one we evaluated in October.
- **Earlier reviewer decisions are not part of the input.** Some chains continue after an ask_human, and 11 train
  records after a refuse, but the model does not see this. Adding those decisions to the input would be a separate run.
- Two rows of r02 table E6 vary from run to run on identical input (see section 3). The other analysis steps are
  deterministic.
- jev is not deterministic (single-step mean difference 0.02), so small jev differences between versions are noise.
- The 14 gpt refusals are excluded as in the paper, so n differs slightly between versions (1,433 vs 1,434 on test).
- SAFETY SENTRY already contains calls derived from When2Call (service `when2call`: 271 test and 1,546 train
  records), which is relevant for the When2Call analysis.

## 7. How to reproduce

From `certified-autonomy-ecis2027/`:

```bash
DRY=1 bash rerun_multistep_fix_2026-11/run_rerun.sh          # counts and cost, no API calls
bash rerun_multistep_fix_2026-11/run_rerun.sh                # scoring (resumable) + paper + confirmatory analyses
bash rerun_multistep_fix_2026-11/run_single_step_only.sh             # FIX_SINGLE  (no API calls)
SOURCE=ecis bash rerun_multistep_fix_2026-11/run_single_step_only.sh # ECIS_SINGLE (no API calls)
python rerun_multistep_fix_2026-11/compare_with_ecis.py      # results/comparison/
```

`PY=<python>` sets the interpreter. I used Python 3.12 with numpy 2.5, pandas 3.1 and scipy 1.18. All tables in this
report come from `results/comparison/tables.md`; the raw values are in the CSV files next to it.
