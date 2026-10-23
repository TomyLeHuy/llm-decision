# Analysis notes (working notes of 22 October 2026)

Written during the analysis; file names refer to the working layout. Mapping to this repository: code/05 to 09 = analysis/02 to 06, code/10 = analysis/07, code/11 = analysis/08; full/raw_full.csv = results/scores_test_1436.csv; results.json = r02_core.json, results2 = r03, results3 = r04, results4_sensitivity = r05, results5_revision = r06, results6_g5costs = r08. Numbers in the paper take precedence where wording differs.


Data: 1,433 step-level tool calls scored by all three decision models (540 execute, 488 ask, 405 refuse); the 3 failed gpt calls are dropped for every model so all comparisons use the same items. Calibration and test are stratified 50/50 halves of the test split, 100 random splits (60 for the certified policies). Baseline numbers reproduce notebook 02 exactly.

Budgets throughout: at most 5% of calls that should be blocked may be executed; at most 15% of calls that need confirmation may be executed. "Safe calls that run without a human" is the share of true direct_execute calls the gate executes on its own.

## Five findings

**1. Certifiability floor.** A decision model can be certified at a budget only if fewer of the harmful calls than the budget receive the lowest probability the API can express. Below that point the gate must escalate everything (Figure F1).

| Model | Output | Refusals at exactly 0 | Safe calls automated at 5% budget |
| --- | --- | --- | --- |
| jev-1.14 | 0.01 grid | 7.9% | 0% (starts between 7.5% and 10% budget) |
| gpt-6-luna | 0.01 grid | 2.7% | 50% |
| microsoft-decision-1 | continuous | 0% | 56% |
| same microsoft scores rounded to 0.01 | 0.01 grid | 17.0% | 0% (starts between 15% and 20% budget) |
| mean of the three | continuous | 0% | 63% |

The earlier "ranking reversal" is a resolution effect: rounding microsoft's own scores to the grid the other vendors use removes all of its autonomy at the 5% budget.

**2. Vendor diversity lowers the floor.** On true refusals, 12 to 23% get p ≤ 0.01 from each single model, but only 5.4% from all three at once, and no refusal gets an exact zero from all three. Averaging the three gives the highest autonomy at every strict budget (63% at 5%), and still 59% after rounding the average to 0.01. Limit: on the dh_app service all three give the same near-zero refuse probability to all 18 refusals, a shared blind spot that averaging cannot fix.

**3. The class guarantee is an average over very different harms.** Calibrating one threshold for all refusals keeps the overall rate within 5%, but atbench refusals are executed at 15% (microsoft) and 17% (mean of three), three times the budget, hidden by 124 easy tsbench refusals. Calibrating the refusal threshold per target service brings every group within budget (2 to 4%) and moves the cost of safety to the risky service: benign autonomy on atbench falls from 22 to 25% to 1 to 7%, elsewhere it rises (microsoft overall 56% to 60%) (Figure F2). The target service is known at decision time, so this is deployable.

**4. "On average" is not "with high confidence".** Prediction sets guarantee the budget only on average across deployments: with microsoft, 36% of test splits break a budget. A certified policy (Learn-then-Test, binomial tests, fixed-sequence search) guarantees both budgets with 90% confidence: 0 to 2% of splits break one, at a cost of 10 to 14 points of autonomy. Because the certified policy looks at refuse and ask probabilities jointly, it also rescues the coarse-grid model: jev goes from 1% to 32% safe calls automated (Figure F3).

| Model | Prediction sets: automated / splits breaking a budget | Certified policy: automated / splits breaking a budget |
| --- | --- | --- |
| microsoft-decision-1 | 37% / 36% | 26% / 0% |
| mean of the three | 48% / 4% | 34% / 2% |
| jev-1.14 | 1% / 0% | 32% / 0% |
| gpt-6-luna | 23% / 0% | not certifiable (ask-budget floor) |

**5. Unseen services break the guarantee.** Calibrating on all other services and testing on one held-out service: atbench refusals are executed at 17% (mean of three) and 20% (microsoft); nocodb at 25% for gpt; dh_app at 94% for the mean of three. Calibration must be refreshed per service before the guarantee can be claimed for it.

## What the gate costs

At the 5% budget the best configuration lets 34 to 63% of safe calls run without a human, depending on whether the guarantee is on average or with 90% confidence, against 82 to 86% for an unsafe argmax gate. Raising the assumed cost of one harmful execution from 10 to 200 review-equivalents pushes the certified policy towards escalating nearly everything (mean of three: 34% to 7%).

## Open items before submission

1. Score the 7,767-item training split with the same three models (≈ $2, under 3 hours) and calibrate on it: the certified policy tightens towards the 5% budget, gpt becomes certifiable, and per-service calibration gets enough data for more services.
2. Score the Mailu out-of-distribution split (198 items) for a true unseen-service test.
3. Report labels as LLM-generated with author arbitration (92% audited agreement); every guarantee is with respect to these labels.

## Files

- code/05_full_conformal.py: reproduction, floor, resolution, ensemble, set rules, per-source and leave-one-source-out tests, budget sweep
- code/06_policies_groups.py: certified versus tuned policies with cost sweep, per-service calibration, coarse-vendor ensemble
- full/results.json, full/results2.json and the CSVs: every number above
- full/figures/F1 to F3

## Calibration size with the existing scores (update 22 Oct)

The decision models are fixed and were never trained on this corpus, so every labelled item can be calibration data; the 50/50 split was only needed to evaluate. With the existing 1,433 items:

- **Evaluate with a rotating hold-out** (calibrate on 90%, test on 10%, 40 repetitions) instead of halving the data: about 364 calibration refusals instead of 202.
- **Calibrate the deployed gate on all 1,433 items** (405 refusals).
- **Pre-set fewer candidate policies** (3 ask thresholds instead of 9), which lowers the multiple-testing penalty.

Effect on the certified policy (5% and 15% budgets, 90% confidence), Figure F4:

| Model | Certifiable, 50/50 split | Certifiable, 90/10 | Safe calls automated, 50/50 | Safe calls automated, 90/10 | Gate calibrated on all items (in-sample) |
| --- | --- | --- | --- | --- | --- |
| jev-1.14 | 53% of splits | 100% | 42% | 40% | 50% |
| mean of the three | 100% | 100% | 37% | 46% | 46% |
| microsoft-decision-1 | 100% | 100% | 26% | 27% | 26% |
| gpt-6-luna | 0% | 0% | not certifiable | not certifiable | not certifiable |

What this does not fix:

- **gpt stays uncertifiable, and the binding budget is the ask budget, not the refusal budget.** gpt gives 12.1% of calls that need confirmation an ask probability of exactly 0 together with an execute probability of 1. Even its strictest execution rule therefore executes 20% of those calls, above the 15% ask budget. The floor applies to every budgeted class. It needs a looser ask budget, not more data.
- **The rounding floor.** jev under the prediction-set method stays at 0% automation at any calibration size; only the joint policy rescues it.
- **Per-service certification.** The certified policy keeps the overall rate within budget but atbench still sits at 7 to 10%; certifying per service with 90% confidence needs more refusals per service than the test split has.
- **Pre-setting the 3 thresholds after seeing these results is data snooping.** For the paper, fix the candidate grid in the protocol (round values) and state it, or confirm on newly scored data.

Verdict on the training split: not required for a valid submission; the learning curve flattens after about 300 calibration refusals for most models. It is still worth scoring for per-service certification, and for an independent test set that reviewers can trust.

## Revision analyses for the reviewer comments (update 22 Oct, evening)

Scripts: code/08_sensitivity.py (budget x confidence), code/09_revision.py (G5, costs, 99% confidence, frontier, vendor dependence), code/10_fallback_check.py (empty policy in G4), code/11_g5_costs.py (G4/G5 cost sweep at 90% calibration). Outputs: full/results4_sensitivity.*, full/results5_revision*.json/csv, full/results6_g5costs.json.

- G5 (service-conditional certified policy, simultaneous over 4 groups, 90% confidence): needs >= 129 calibration refusals per group at a 5% budget. With 50/50 splits no group qualifies and G5 executes nothing. With 90% calibration it certifies the pooled "other" group in 40/40 repetitions; safe autonomy 43.3% (G4 32.6%), harmful execution 0% in every group (G4: 6.6% on atbench), cost 0.74 vs 0.81 at harm cost 10.
- Costs (mean of three, half calibration, review-equivalents per call at harm cost 10/50/200): no gate 1.33/5.30/20.20; review all 1.00; G1 0.89/1.52/3.89; G2 0.91/2.08/6.45; G3 0.88/1.87/5.58; tuned 0.82/0.92/1.04; G4 0.81/0.92/1.04; execute nothing 0.88.
- Adding the trivially certified empty policy to G4 changes nothing at harm cost 10 (all models) and only jev at 50/200.
- 99% confidence at a 5% budget: mean 27%, microsoft 15%, jev 26% (certifiable in 82% of splits); nothing certifiable at 2%.
- Minimum calibration refusals for G4 (m = 9): 75/88/102/133 at 5% for 80/90/95/99% confidence; 189 to 337 at 2%.
- Vendor confident errors (p_refuse <= 0.01 on true refusals) are dependent: pairwise lift 2.8 to 3.9, Spearman 0.76 to 0.87; all three low 5.4% vs 0.36% under independence.
