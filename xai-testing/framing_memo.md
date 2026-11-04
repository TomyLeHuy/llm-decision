# Framing memo: XAI follow-up to "Statistically Certified Autonomy"

*For a narrative overview of what was done and why, read `OVERVIEW.md` first.*

*4 November 2026. Based on `FINDINGS.md`; methods in `xai_methods.md`.*

## Core message (proposal)

> **To audit a certified delegation gate, explain the certificate, not the individual prediction.**
>
> A statistical certificate can stay formally valid while the system behind it is broken. Explanations at the level of the certificate and the data, which need no access to the model and almost no queries, revealed three such failures in our gate:
>
> - a pipeline defect
> - a failing subgroup hidden inside the guarantee
> - silent vendor drift
>
> Fixing the defect raised certified autonomy by 8–21 points. Per-call attributions (occlusion and exact Shapley over tool-call fields) were reproducible within a model but depended on method and model, were undefined for half the calls of a vendor-guarded model, and did not detect any of the three failures.

This is framing (b), explanations as an audit tool, with (a), explaining certifiability, as the design content. It also argues, with evidence, against the obvious move of adding SHAP to the reviewer interface.

## The five strongest findings (n and details in `FINDINGS.md`)

1. **The scorer never saw the labelled call in multi-turn sessions.** On 123 test calls (733 train), only the first step was scored. Fixed without label leakage:
   - refusal AUROC 0.5 → 0.68–0.86
   - certified safe autonomy +8 to +21 points (mean of three: 35.4% → 51.7%)
   - multi-turn harm 5.1–8.1% → 0–4.7%
2. **The guarantee held while a subgroup failed.** Overall harm stayed within budget, but harm on multi-turn refusals exceeded 5% in 47–73% of splits. Leave-one-out influence showed that liquid's certified boundary was held by five faulty refusals. Per-call occlusion missed it (detection AUROC 0.34–0.45).
3. **Two vendors changed behaviour within two weeks under unchanged version strings.**
   - microsoft's frozen policy stayed safe but fell from 26.3% to 1.7% autonomy.
   - pplx lost its determinism.
4. **The labels have a large, consistent disagreement set.** 298 single-turn calls are flagged by ≥ 3 of 5 models (109 expected by chance), and 50 identical prompts carry conflicting labels. The main pattern is refuse → ask. Manual verification is pending.
5. **Per-call field attributions are not unique.**
   - Masking strategies agree at ρ 0.0–0.6.
   - Shapley and occlusion pick the same top field in 35–48% of calls.
   - gpt and the other models agree at ρ 0.09–0.27.
   - Interactions are about 2× the main effects.
   - Shapley is undefined for 51% of gpt calls, because the provider refuses partial inputs.

## Possible paper structure (design science, second cycle on the gate)

1. **Problem:** certified delegation guarantees an error rate but says nothing about whether the certificate rests on sound data, a correct input pipeline and a stable model.
2. **Objectives:** an audit layer that explains the certificate, works with black-box access, and is understandable for operators and auditors.
3. **Artefact:** an "explained certificate" consisting of five audit components:
   - lever decomposition (why not certifiable)
   - influence of calibration calls on the certificate
   - risk-budget decomposition by subgroup
   - label audit
   - anchor-based drift check
4. **Demonstration:** the five decision models and SAFETY SENTRY, including the truncation case as an end-to-end example: detect → fix → re-certify.
5. **Evaluation:**
   - Technical: did the audit find real defects, and does fixing them improve the certificate?
   - Contrast: per-call attributions on the same cases (negative result).
   - Small expert evaluation: about 8–12 IT risk managers or auditors, using think-aloud on the explained certificate.
6. **Design principles,** for example:
   - audit the certificate, not just the scores
   - decompose the risk budget by subgroup
   - re-certify on anchor drift, not on version changes
   - treat per-call attributions as model-specific and method-dependent
7. **Implications:** for the EU AI Act Art. 13/14 (documentation and oversight), procurement and the human-review interface.

**Target:** ICIS 2027 (design science track), or ECIS 2028. Possibly an extended journal version (BISE or JAIS) later. The negative attribution results alone could be a short CS paper (for example an XAI workshop).

## What is still missing

- **Tomy's manual reviews** (in progress):
  - the risk-class → field map: makes the plausibility results final
  - 40 label flags + 50 conflicts: turns "flags" into a verified label-error rate
- **Drift monitoring experiment** (cheap): would a daily anchor set of about 100 calls have caught the microsoft and pplx changes immediately? This gives the drift principle an evaluated mechanism.
- **Expert evaluation:** recruitment and ethics approval. This is the main effort and covers the ECIS paper's limitation that it had no human evaluation.
- **Optional:** sampled Shapley for one stochastic model, to show that finding 5 is not gpt-specific.

## Consequence for the submitted ECIS paper

- The finding that no model separates refusals on two benchmark services (AUROC 0.48–0.59) is largely an artefact of the truncation.
- Most of jev's zero-probability refusals (27 of 32) are truncated multi-turn calls, so the empirical size of the resolution floor is overstated. The mechanism itself is unaffected: rounding microsoft's probabilities still turns a large share of its refusal probabilities into exact zeros.
- The propositions and the guarantee are unaffected.
- **Recommendation:** inform the co-authors now and prepare a correction for the revision or camera-ready version. The follow-up paper can then report the defect openly as the motivating audit case.

## Questions for the supervisor

1. Is the core message (explain the certificate, not the prediction) the right centre, or should the follow-up stay closer to classic XAI (attributions for reviewers)?
2. Design-science paper with an expert evaluation, or a quantitative audit study without humans first?
3. How should we handle the ECIS correction, given that the follow-up builds on the defect?
