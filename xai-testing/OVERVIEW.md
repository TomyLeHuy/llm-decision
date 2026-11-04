# XAI experiments on the certified delegation gate: what was done and what it means


**Related files:**
- `framing_memo.md`: the proposed core message, a paper structure and open decisions
- `xai_methods.md`: method details, results per method and references
- `FINDINGS.md`: all numbers, with n and the noise floors
- `DECISIONS.md`: every design choice with its reason
- `xai_plan.md`: the original plan

All experiments ran in `xai-testing/`. The paper package `certified-autonomy-ecis2027/` was never modified (its frozen checksums still match). Total API cost was USD 19.36. All results are exploratory, not pre-registered.

---

## 1. Starting point

The ECIS paper's gate takes three probabilities from a decision model (execute, ask, refuse) for every tool call. From labelled calls it chooses thresholds so that a guarantee holds: "with 90% confidence, at most 5% of the calls that should be refused run autonomously." The gate gives this promise but explains nothing:

- why a call is routed the way it is
- why a service cannot be certified
- what the certificate depends on

The goal was to find out where XAI can play a real role here. Two levels were tested separately:

- **Prediction level:** what in a single call drives the model's decision? This is the classic XAI question that SHAP and LIME answer.
- **Certificate level:** what does the certificate as a whole depend on: which data, which labels, which subgroups, which model state?

**Constraint throughout:** only the models' output probabilities are visible, with no weights or gradients. Every method had to work under this constraint.

### Models tested

Five commercial decision models were used, all through the OpenRouter decisions endpoint (`/api/alpha/decisions`). Every model received the same system policy, the same criteria for the three actions and the same input format as in the ECIS paper. The first three are the paper's models. liquid and pplx were added on 27 October as an exploratory extension, not covered by the confirmatory protocol.

| Short name | Model ID | Version string returned | In ECIS paper |
|---|---|---|---|
| jev | `typesafe/jev-1.14` | `jev-1.14-20261014` | yes |
| microsoft | `microsoft/microsoft-decision-1` | `microsoft-decision-1-20261009` | yes |
| gpt | `openai/gpt-6-luna-decisions` | `gpt-6-luna-decisions-20261006` | yes |
| liquid | `liquid/d1-ultra` | `d1-ultra-20261015` | no (added 27 Oct) |
| pplx | `perplexity/pplx-decider-v1.2` | `pplx-decider-v1.2-20261016` | no (added 27 Oct) |

**Without the gate** (test split, 1,433 calls, October scores, paper input):

| Model | Accuracy | Refusals executed without gate | ECE | Refusals at p_refuse = 0 | Refusals at p_refuse ≤ 0.01 | G4 safe autonomy (5% / 90%) |
|---|---|---|---|---|---|---|
| jev | 64.9% | 15.8% | 0.148 | 7.9% | 13.3% | 32.0% |
| microsoft | 59.7% | 26.4% | 0.307 | 0.0% | 22.5% | 26.0% |
| gpt | 62.3% | 20.7% | 0.252 | 2.7% | 12.1% | not certifiable (ask budget) |
| liquid | 70.0% | 21.7% | 0.079 | 0.0% | 1.5% | 49.5% |
| pplx | 65.2% | 24.0% | 0.190 | 0.0% | 8.2% | 49.2% |

*Source: `certified-autonomy-ecis2027/results/extra_models/SUMMARY.md`. These are the paper-input values; the corrected multi-turn input raises G4 safe autonomy by 8–21 points (Section 4).*

**Behaviour relevant to the XAI experiments:**

| Model | Determinism | Noise threshold (TV, 3-repeat mean) | Drift observed | Used in |
|---|---|---|---|---|
| jev | stochastic | 0.044 | within noise | occlusion (3 repeats) |
| microsoft | stochastic | 0.038 | **scores shifted after 23 Oct**, version unchanged; results labelled "post-drift" | occlusion (3 repeats) |
| gpt | deterministic (verified 3×, 60/60) | 0.005 (any change counts) | none; refuses to answer some edited or partial inputs (`provider_refusal`) | occlusion (1 run), **exact Shapley** |
| liquid | stochastic (noisiest) | 0.071 | within noise | occlusion (3 repeats) |
| pplx | deterministic until 3 Nov ~15:00 UTC, **stochastic afterwards**, version unchanged | 0.018 | determinism lost | occlusion (3 repeats) |

The "mean of three" scorer from the paper (the average of jev, microsoft and gpt) is reported alongside where the paper reports it.

---

## 2. Step 0: are the measurements trustworthy?

Three checks came before any explanation.

- **Round trip.** If removing a field changes the probability, the change must come from the field and not from the JSON being rebuilt slightly differently. Rebuilding the input without masking gives a byte-identical string for all 9,401 calls.
- **Noise floor.** If a model returns different numbers for the same input, a small effect may be pure noise.
  - gpt is fully deterministic.
  - jev, microsoft and liquid vary by roughly 0.02–0.07.
  - pplx was deterministic at first and then stopped being so (Section 5).
- **Drift.** Is the model explained today the one certified in October? microsoft had changed noticeably, under the same version string.

These checks later became findings in their own right:
- Whether a model can be explained at all depends on how it delivers its output, in parallel to the resolution floor for certifiability in the ECIS paper.
- Vendors change their models silently.

---

## 3. Phase A: explaining the certificate (no API calls)

These analyses used only the existing scores. They are simple, but they produced the strongest results.

### A1: Label audit (Confident Learning)

**Intuition.** If five models all confidently say "this call is execute" while the label says "ask", the label may be wrong. Confident Learning formalises this with per-class confidence thresholds. The models were not trained on this data, so their probabilities count as out-of-sample, which the method requires.

**Result:**

- 298 single-turn calls (22.7%) are flagged by at least 3 of 5 models; about 109 would be expected if the models flagged independently.
- The flags concentrate in the adapted benchmarks (tsbench, when2call, agentharm).
- Model-free evidence: 50 identical prompts carry a different label in train and test.

**Caution.** Many flags follow the models' shared bias towards execute. Five models agreeing is no proof, because they make similar mistakes. A manual review of the 40 strongest flags and the 50 conflicts is in progress.

### A2: Lever decomposition (why is a service not certifiable?)

**Intuition.** The ECIS paper identified three levers:

- too few labelled refusals (history lever)
- too many probabilities at exactly zero (output lever)
- refusals that cannot be separated from safe calls (AUROC)

For each model × service pair, the analysis determines which lever binds and states it in one sentence, for example "54 refusals, 102 needed".

**Result:**

- On the test data, almost everything fails for lack of refusals.
- On the larger train data, five services become certifiable.
- gpt always fails because of its ask budget.
- atbench and dh_app looked genuinely inseparable (AUROC ≈ 0.55). This turned out to be the key clue.

### A3: Leave-one-out influence on the certificate

**Intuition.** Remove each calibration call in turn, re-certify, and see how much the certified autonomy changes. This is influence analysis, but the "model" is the calibration procedure, not a neural network, so it costs almost nothing.

**Result:**

- Almost every single call is irrelevant to the certificate.
- For liquid, however, removing any one of five refusals raises certified autonomy from 38.9% to 55.9%.
- Four of those five are multi-turn calls from atbench, and all five are flagged by all five models in A1.
- Two independent methods pointing at the same few calls was the signal that something was wrong with them.

### A3b: Where is the risk budget spent?

**Intuition.** The guarantee is marginal: it holds across all refusals together and does not promise that every subgroup stays below 5%. The analysis therefore splits the harm actually executed by subgroup.

**Result:**

- About 1% of single-turn refusals run autonomously.
- Of multi-turn refusals, 5–8% do; this subgroup exceeds 5% in 47–73% of splits.
- The guarantee holds formally, but almost the entire budget is spent on one subgroup.

---

## 4. The discovery: the multi-turn truncation

All three clues pointed at the multi-turn calls:

- the inseparable services
- the five boundary-holding calls
- the subgroup harm

The cause is in the scoring code (`score_split.py`, identical to notebook 02). It sends only the **first** user message to the model, while in multi-turn sessions the label belongs to the **last** step. The model therefore judged, for example, "read the student record" (harmless) while the label referred to "upload with a leaked access token" (refuse). The model was not blind; it was looking at the wrong call.

**Before/after test (Steps 5 and 5b).** The affected calls were re-scored with the correct input, in a separate folder; the paper's scores stay untouched. There was a leakage risk: the full conversation contains the reviewer's earlier decisions, which could reveal labels. The main variant (a) therefore uses only the user turns, without those decisions.

**Result:**

- Refusal AUROC on multi-turn calls rises from about 0.5 to 0.68–0.86.
- dh_app becomes almost perfectly separable.
- Certified safe autonomy rises by 8 to 21 points (mean of three: 35.4% → 51.7%).
- Subgroup harm falls from 5–8% to 0–4.7%.
- No leakage: variant (a) is as good as or better than the full conversation.
- **What stays:** atbench and dh_app are still not certifiable on their own, but now only because they have too few labelled refusals.

This is the core of the whole exercise. **Explanations of the certificate found an error the certificate itself could not show, and fixing it measurably improved the certificate.**

---

## 5. Drift (Step 4)

The full test split was re-scored on 3 November and the certificate recomputed.

- **microsoft** changed its scores. The frozen October policy remains safe, but it now lets through only 1.7% of safe calls instead of 26.3%. The gate became useless without anyone noticing.
- **pplx** went from deterministic to random during 3 November.
- Both changes happened under unchanged version strings.

**Lesson:**

- A certificate is tied to a model state that the vendor's label does not identify.
- Regular re-checks on a fixed set of anchor calls are needed; version numbers cannot be trusted.
- Drift broke the usefulness, not the guarantee: a quiet kind of failure.

---

## 6. Phases B–D: explaining individual decisions

### Field occlusion (M1)

**Intuition.** Remove one field at a time and measure how much the probability changes; a large change means the field mattered. The fields are user_task, tool_args, prior_steps, user_memory, tool, description and service. This is the simplest form of the idea behind SHAP and LIME (removal-based explanations).

We used fields instead of tokens for three reasons:

- removing tokens breaks the JSON
- token-level masking costs hundreds of queries per call
- reviewers read a call field by field

**Pilot: removing vs replacing a field.** The two masking strategies gave very different explanations. Replacement creates inconsistent calls (for example, email addresses as a train destination), so it mainly measures how the model reacts to nonsense. This is the known out-of-distribution problem from the XAI literature, now shown on tool calls. The full run therefore used removal only.

**Full run** (1,433 calls × 5 models, 3 repeats for the stochastic models):

- **What drives the decisions:** the task, the arguments and the prior steps. The tool name, the description and the service barely matter on their own.
- **No manipulation through the description:** a benign-sounding agent description pushes the decision for ≥ 3 models in only 3 of 405 refusals.
- **Stable within a model, different between models:** repeats agree well (0.80–0.97), but gpt explains entirely differently from the others (0.09–0.27).
- **Mixed plausibility (provisional; the risk-class map is not yet reviewed).**
  - For ask cases such as hearsay, the top field often matches the expected one.
  - For exfiltration refusals it often does not: the models refuse because of what the user asked, not because of where the data goes.
- **Occlusion did not detect the truncation.** The model relies on the call it sees, and occlusion reports exactly that, correctly. It cannot know that the call is the wrong one.

### Exact Shapley (M2)

**Intuition.** Occlusion misses interactions: if two fields carry the same information, removing one changes nothing because the other still covers it. Shapley averages each field's contribution over all combinations of the other fields and catches this. With 8 fields that is 256 queries per call, so it was run only for deterministic gpt.

**Result:**

- **Undefined for 51% of the calls.** gpt refuses to answer some heavily stripped-down inputs, and a single missing combination makes exact Shapley impossible. With a vendor-guarded model, it often cannot be computed at all.
- **Disagrees with occlusion.** Where it can be computed, it picks the same most important field as occlusion in only 35–48% of calls.
- **Interactions dominate:** they are about twice the size of the single-field effects.
- **Tool and description stand in for each other,** so occlusion systematically understates both. The statement "tool and description barely matter" is partly an artefact of the method.

**Summary for the prediction level:**

- The explanations are not random; they are reproducible within a model.
- But the "reason" shown for a decision depends on the method, the masking and the model.
- They found none of the real problems.

---

## 7. What the results say

| | Certificate/data level | Prediction level |
|---|---|---|
| Found the pipeline error | yes | no |
| Found label problems | yes | no |
| Found the drift | yes | no (drift undermines it instead) |
| Unique and robust | yes | no (depends on method and model) |
| Cost | ~0 | high, often undefined |

**Possible framings, with strengths and weaknesses:**

**(b) "Explain the certificate, not the prediction" (audit)** *(currently preferred)*
- **Claim:** certified gates need an audit layer that explains the certificate; per-call attributions are unsuitable for this.
- **Strengths:**
  - There is a complete end-to-end case: detect → fix → re-certify, with a measurable gain.
  - The negative result on attributions strengthens the claim instead of weakening it.
  - It fits IS well, as a second design-science cycle with design principles.
- **Weaknesses:**
  - The individual methods (leave-one-out, Confident Learning, subgroup decomposition) are not new; their use on certificates is.
  - A reviewer could say "you found a bug in your own code". The counter-argument: the formally valid certificate did not show the bug, which can happen in any real pipeline. It is still a single case.

**(a) "Explaining certifiability"**
- **Claim:** a method that explains to managers why something is or is not certifiable, and what would change that.
- **Strengths:** builds directly on the ECIS levers; easy to understand ("54 refusals, 102 needed"); the XAI expertise enters as counterfactual explanations of the certificate.
- **Weakness:** on this data, the per-service explanation is usually trivial (too few refusals), and the interesting cases came from the pipeline, not the model. Probably too thin on its own; it works better as the content of (b).

**(c) Explanations for reviewers**
- **Claim:** help the people who review two thirds of the calls.
- **Strengths:** addresses the ECIS paper's limitation (no human evaluation); high practical relevance.
- **Weakness:** the current data argue against it. If the "reason" shown changes with the method, a user study would mainly measure which method happened to be picked. This becomes viable only with a justified, stable explanation method, as a later, separate paper.

**(d) Drift: "certificates of black-box models expire silently"**
- **Strength:** a strong, practical finding.
- **Weakness:** on its own too thin (two observations). Better as part of (b), backed by a small anchor-monitoring experiment.

---

## 8. Open questions for the framing decision

1. **Artefact or insight?** Design science on (b) delivers an artefact (the audit layer) plus design principles. An empirical paper would deliver an insight: prediction-level XAI fails here, certificate-level XAI works.
2. **How general is one case?** The truncation is a single defect. Is it representative of a class (input-pipeline errors), or is a second case needed, such as a deliberately injected defect, to show that the audit catches those too? A small injection experiment could close this gap cheaply.
3. **How should the negative attribution results be used?** As the main contribution (risky, because negative results are a harder sell), or as the contrast that justifies the certificate approach (more robust)?
4. **Who is the audience?** Auditors and managers fit (a) and (b) and the EU AI Act; reviewers fit (c).
5. **How should the ECIS correction be handled?** The follow-up builds on the defect. That is honest and strong if the ECIS correction is visibly made first.

**Tomy's current assessment:** (b), with (a) as its content, is the most robust option and the one the data support best. Its decisive weakness is the single case, so adding a second, deliberately injected defect as evidence should be discussed.

---

## 9. Consequences for the submitted ECIS paper

- The finding that no model separates refusals on two benchmark services (AUROC 0.48–0.59) is largely a truncation artefact.
- The empirical size of jev's resolution floor is overstated: 27 of its 32 zero-probability refusals are truncated multi-turn calls.
- The floor mechanism itself, the propositions and the guarantee are unaffected.
- The co-authors should be informed, and a correction prepared for the revision or camera-ready version.

## 10. Still open

- **Manual reviews by Tomy:** the risk-class → field map (makes the plausibility results final), and the 40 label flags plus the 50 train/test conflicts (gives a verified label-error rate).
- **Optional experiments, depending on the framing:**
  - anchor-based drift monitoring
  - an injected-defect test
  - sampled Shapley for one stochastic model
  - an expert evaluation with IT risk managers or auditors
