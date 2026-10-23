# Internal review rounds and responses

The review texts are in `internal_reviews/`. Version numbers refer to `CHANGELOG.md`.

## Round 1 (on version 2)

| # | Comment | Change in version 3 |
|---|---|---|
| 1 | G3 and G4 not evaluated jointly | New variant G5 (service-conditional certified policy). It needs at least 129 calibration refusals per group; with half the data it executes nothing, with 90% it certifies the pooled services and beats G4 (Section 5.6, Table 4). |
| 2 | "36% of deployments" too strong | "repeated calibration-test splits" throughout; Protocol explains why splits are not deployments. |
| 3 | Novelty claim on conformal prediction too strong | Contribution reframed as organisational operationalisation of known methods. |
| 4 | Certifiability floor imprecise | Two floors: score-resolution (Proposition 1) and policy-family (Proposition 2). |
| 5 | G4 method too compressed | Procedure box (Table 1): hypotheses, joint test, multiplicity, fixed sequence, empty policy, why selection keeps the guarantee. |
| 6 | Development and evaluation on the same corpus | Three roles of evidence separated (Section 3); confirmatory test stated as pending. |
| 7 | IS contribution | Delegation framed as action-level, dynamic, machine-mediated, risk-budgeted and recalibrated; title changed. |
| 8 | Label description | Exact description of the label process; "expert-arbitrated" removed. |
| 9 | Cost objective not evaluated | Table 4: review rate, wrongful blocking and cost at three harm costs against no gate and reviewing every call. |
| 10 | 90% confidence too weak | 80/90/95/99% (Table 5) and the confidence and sample-size frontier (Figure 7). |
| 11 | "Independent vendors" | Wording changed; dependence measured (lift 2.8 to 3.9, rank correlation 0.76 to 0.87). |
| 12 | Budgets not justified | Stated as illustrative; harm budget varied at 2, 5 and 10%. |
| 13 | AI Act claim | "auditable evidence supporting" oversight, not compliance. |
| 14 | "Certification" ambiguous | Defined as a statistical guarantee in the Introduction. |

## Round 2 (on version 3)

| # | Comment | Change in version 4 |
|---|---|---|
| 1 | No confirmatory evaluation | Pending; limitation explicit; generality claim removed. Plan and code: `confirmatory/`. |
| 2 | G5 logic hard to follow | G5 as steps A to C in Table 1, with the argument why group thresholds cannot raise the certified ask risk. |
| 3 | DP3 sounds universal | Conditional on demonstrated error diversity and on verifying a lower floor. |
| 4 | G5 result reads as a failure | Framed as the data cost of granular delegation boundaries. |
| 5 | Floors deserve prominence | In abstract, contributions and conclusion. |
| 6 | "Cost-aware certified" ambiguous | "certified for its risks and selected by estimated cost". |
| 7 to 9 | AI Act, 36% wording, IS framing | Kept and checked. |
| new | Opening figure requested by the authors | Figure 1: calls in, three actions out, with the organisation's parameters. |

## Round 3 (on version 4)

| # | Comment | Change in version 5 |
|---|---|---|
| 1 | Page count (17 in the reviewer's rendering) | The authors checked 16 pages in Word; layout kept unchanged. A pre-tested shortened text is kept in `paper/manuscript_shortened_option.md`. |
| 2 | Confirmatory evaluation | `confirmatory/` (frozen protocol, scoring and analysis code). |
| 3 | Separate G5 steps in Table 1 | G5 steps set in italics under a header naming the additions. |
| 5 | Granularity trade-off | Sentence added to the theoretical contribution. |
| 9 | DP5 wording incomplete | Rewritten to match Section 5.7. |
| 4, 6 to 8, 10 to 12 | Resolved items | No change. |
