# XAI methods tested on the certified delegation gate

*For a narrative overview of what was done and why, read `OVERVIEW.md` first.*

*Status as of 4 November 2026: all planned runs are complete (Phases 0, A, B, C, D, the before/after re-scoring and exact Shapley for gpt). Phase E (counterfactuals, precedents) was not started. All results are exploratory, not pre-registered. Full numbers are in `FINDINGS.md`, design choices in `DECISIONS.md`. Total API cost: USD 19.36.*

## 1. Setting and constraint

The delegation gate from our ECIS 2027 paper routes each tool call that an AI agent proposes to one of three outcomes: **execute** autonomously, **ask** a human, or **refuse**. A decision model supplies the probabilities for these three outcomes. The gate wraps the model with class-conditional conformal calibration and Learn-then-Test (LTT) certification (Angelopoulos et al., 2025). The result is a guarantee such as: with 90% confidence, at most 5% of the calls that should be refused run autonomously.

**Models.** Five commercial decision models, accessed through the OpenRouter decisions endpoint: `typesafe/jev-1.14`, `microsoft/microsoft-decision-1`, `openai/gpt-6-luna-decisions`, `liquid/d1-ultra` and `perplexity/pplx-decider-v1.2`.

**Data.** The SAFETY SENTRY corpus (Chen et al., 2026): 1,436 test, 7,767 train and 198 out-of-distribution calls. The labels come from LLM annotators, with 92% human agreement. Each call also carries a trigger class (`risk_class`) and the annotator's written reasoning.

**Design constraint.** The gate is model-agnostic and sees only the output probability vector: no weights, gradients or internals. Every method here therefore works with black-box queries only, or with no queries at all.

**Explanation units.** The model input is a JSON object, so we explain at the level of its fields rather than tokens. The units are `user_task`, `user_memory`, `prior_steps`, `tool`, each entry in `tool_args`, `description` (free text written by the agent) and `service`. Calls have 4–15 such units, 7.6 on average.

**Two levels of explanation.**

- **Prediction-level:** what in a tool call drives the routing probabilities.
- **Certificate-level:** which data, labels, subgroups and model states decide whether the certificate exists and how much autonomy it grants.

## 2. Overview

| # | Method | Level | Question | Queries | Status | Verdict |
|---|---|---|---|---|---|---|
| M0 | Validity checks: round trip, noise floor, determinism | prerequisite | Are perturbation results interpretable at all? | few hundred | done | essential; found non-determinism |
| M1 | Field-level occlusion (leave-one-field-out) | prediction | Which fields drive the routing? | F+1 per call, × 3 repeats for noisy models | done, 1,433 calls × 5 models | reproducible within a model, but not unique |
| M2 | Exact field-level Shapley values, with interactions | prediction | Same, accounting for interactions | 2^F per call | done, gpt, 1,016 calls with ≤ 8 fields | undefined for 51% of calls; disagrees with M1 |
| M3 | Faithfulness vs plausibility | prediction | Do attributions match the trigger class and annotator reasoning? | 0 extra | done, provisional (risk map unreviewed) | mixed |
| M4 | Confident Learning (label audit) | data | Which labels are probably wrong? | 0 | done | large, consistent disagreement set |
| M5 | Leave-one-out influence on the certificate | certificate | Which calibration calls does the certificate depend on? | 0 | done | found boundary-holding faulty calls |
| M6 | Certificate diagnostics (lever decomposition) | certificate | Why is a model × service pair not certifiable? | 0 | done | works; usually "too few refusals" |
| M7 | Risk-budget decomposition by subgroup | certificate | Where is the harm budget spent? | 0 | done | found the hidden failing subgroup |
| M8 | Anchor re-scoring (drift audit) | certificate | Is the certified model still the deployed model? | 1 per anchor call | done | found two silent vendor changes |
| M9 | Before/after re-scoring with leakage check | audit | Does fixing the defect found by M5–M7 change the certificate? | ≈ 45k | done | +8 to +21 points of certified autonomy |
| — | Counterfactual edits, labelled precedents | prediction / review | What would make a call executable? Which similar cases exist? | tens per call | not started | deferred |

## 3. Prerequisite: validity checks (M0)

Perturbation-based explanations are meaningful only if (a) an unmodified input reproduces the original pipeline exactly, (b) perturbation effects exceed the variation between identical queries, and (c) the explained model is the certified one. Explanation methods can look plausible while being insensitive to the model (Adebayo et al., 2018), and they can be unstable under small input changes (Alvarez-Melis & Jaakkola, 2018).

**Results:**

- **Round trip:** the rebuilt input is byte-identical for all 9,401 samples.
- **Noise:** gpt is deterministic (verified three times, 60/60 identical each time). The others vary: the noise threshold on the total-variation (TV) distance between two 3-repeat means is 0.018 for pplx, 0.038 for microsoft, 0.044 for jev and 0.071 for liquid. A single query would need about twice these thresholds.
- **Determinism changes:** pplx was deterministic until about 15:00 UTC on 3 November and stochastic from about 21:00 UTC (two repeats agree on only 42–43% of calls), with its version string unchanged (see M8).

**Implications:**

- Explanations of stochastic black boxes need repeated queries. Three repeats raise the share of effects that are distinguishable from noise by 9–11 points.
- Whether a model can be explained at all depends on how it delivers its output. This parallels the score-resolution floor for certifiability in the ECIS paper.

## 4. Prediction-level methods

### 4.1 Field-level occlusion (M1)

**Idea.** Remove one unit at a time and measure the change in each class probability, a_c(u) = p_c(x) − p_c(x without u). Removal-based attribution goes back to occlusion in vision (Zeiler & Fergus, 2014) and to representation erasure in NLP (Li, Monroe & Jurafsky, 2016). Covert, Lundberg & Lee (2021) show that SHAP, LIME, permutation tests and occlusion are all removal-based explanations. They differ in three choices: how features are removed, which model behaviour is explained, and how the influence is summarised.

**Design:**

- **Fields, not tokens.** Field-level masking keeps the JSON valid, matches how a reviewer reads a tool call, and costs F+1 queries per call instead of hundreds.
- **Fresh baselines.** The unmasked input is re-queried in the same run, so vendor drift is not mistaken for a field effect.
- **Repeats.** Noisy models are queried 3 times per variant and averaged.
- **Masking strategy.** The pilot compared removing a field with replacing it by the same field from another call. The full run (12,371 inputs per model) uses removal only, because of the pilot result below.

**Results:**

- **Masking strategy changes the explanation.** Removal and replacement agree poorly: the median rank correlation is 0.0–0.6. Replacement mainly measures how the models react to an inconsistent call: it flips the action in 44% of argument swaps. This is the out-of-distribution problem of removal and replacement functions (Hooker et al., 2019; Hase, Xie & Bansal, 2021), consistent with work showing that baseline choice drives attributions (Sturmfels, Lundberg & Lee, 2020; Sundararajan & Najmi, 2020).
- **What drives the decisions.** `user_task`, `tool_args` and `prior_steps` drive them (mean TV 0.13–0.20 for the non-gpt models). Alone, `tool`, `description` and `service` move little (TV 0.02–0.06). gpt reacts to every field, in coarse 0.01 steps.
- **Stability.** Attributions are reproducible within a model when 3 repeats are averaged (rank correlation 0.80–0.97), but they differ between models: median 0.42–0.45, and gpt against the others only 0.09–0.27.
- **Shortcut test.** There is no systematic shortcut through benign-sounding agent descriptions: on only 3 of 405 refusals does removing the description raise p_refuse for ≥ 3 models.
- **Provider refusals.** gpt refused 7 of 12,371 masked inputs. These are kept as a separate outcome (`provider_refusal`), not as missing values.
- **The truncation was not detected.** On the 123 multi-turn calls, the scorer only saw the first step of the session (Section 5.6). Occlusion did not flag them as unusual; a label-free detection test has AUROC 0.34–0.45, below chance. Occlusion faithfully reports that the scorer relies on the call it sees; it cannot tell that the call it sees is the wrong one.

### 4.2 Exact field-level Shapley values (M2)

**Idea.** The Shapley value (Shapley, 1953) averages a unit's marginal contribution over all subsets of the other units, using the same removal operation as M1. For model explanation this follows Štrumbelj & Kononenko (2014) and Lundberg & Lee (2017). Treating fields as players is a grouped Shapley value (Jullum, Redelmeier & Aas, 2021), which keeps exact computation feasible: 2^F queries, 123,056 for 1,016 gpt calls with ≤ 8 fields. We also computed pairwise Shapley interaction indices. For larger F, permutation sampling (Castro, Gómez & Tejada, 2009) would be needed.

**Results:**

- **Undefined for half of the calls.** gpt refused 1.3% of the reduced inputs, mostly the heavily stripped-down ones: 4.5% of inputs with ≤ 25% of the fields, none for the full input. As a result, exact Shapley is undefined for **51% of the eligible calls** (74% of 8-field calls). For a vendor-guarded model, interaction-aware attributions exist only where the vendor's own filter does not fire on partial inputs.
- **Disagreement with occlusion.** On the 499 complete calls, the median rank correlation with occlusion is 0.41–0.63, and the top field matches in only 35–48% of calls.
- **Interactions dominate.** The summed pairwise interactions are about 2.0–2.4 times the summed main effects. The strongest pair is description × tool.
- **Redundant fields.** In 40% of cases, `tool` and `description` stand in for each other: removing one alone changes almost nothing, removing both changes a lot. Occlusion therefore understates them (mean |φ| 0.15–0.16 vs 0.06–0.07 under leave-one-out) and overstates `prior_steps` and `user_memory`.
- **Scope:** shown for gpt only.

### 4.3 Faithfulness vs plausibility (M3)

An explanation can be faithful to the model without being plausible to people, and the reverse (Jacovi & Goldberg, 2020). Following the ERASER benchmark's comparison of model rationales with human rationales (DeYoung et al., 2020), we compared the top-attributed field with two references:

- **Trigger-class mapping.** A map from each `risk_class` to the field that should drive the decision. It is provisional, because Tomy has not yet reviewed it.
- **Annotator reasoning.** Which fields the annotator's reasoning mentions.

**Results** (provisional):

- **Against the trigger-class map:** the top field matches in 47–52% of calls for the four non-gpt models, against 38–39% expected by chance. gpt is at chance.
- **Per class:** above chance for missing-authorisation and hearsay classes. Below chance for data-exfiltration (15% vs 48%) and audit-tampering refusals: the models refuse exfiltration because of what the user asked, not because of the call's target.
- **Against the annotator reasoning:** the top field appears there in 72–85% of single-turn calls, against 53% by chance. This test is weak, because nearly every reasoning mentions some field.

### 4.4 Methods deliberately not used

- **Token-level LIME and SHAP** (Ribeiro, Singh & Guestrin, 2016; Lundberg & Lee, 2017). On JSON inputs they produce malformed calls and need 10²–10³ queries per call. Perturbation explainers can also be fooled by a model that detects perturbed inputs (Slack et al., 2020), and the vendors control the models.
- **Gradient methods such as Integrated Gradients** (Sundararajan, Taly & Yan, 2017). No gradients are available.
- **LLM-generated rationales.** The decision models produce none, and rationales from a separate LLM are frequently unfaithful (Turpin et al., 2023).

## 5. Data- and certificate-level methods

### 5.1 Confident Learning label audit (M4)

**Idea.** Confident Learning (Northcutt, Jiang & Chuang, 2021) estimates likely label errors from out-of-sample predicted probabilities and per-class confidence thresholds. The decision models were not trained on this corpus, so their probabilities are out-of-sample by construction. Label errors are common even in widely used benchmark test sets (Northcutt, Athalye & Mueller, 2021).

**Results:**

- **Volume.** 298 single-turn test calls are flagged by ≥ 3 of the 5 models, against 109 expected if the models flagged independently; 56 are flagged by all 5. Flags concentrate in the adapted benchmarks: tsbench 43%, when2call 33%, agentharm 32%, against 3–24% for the enterprise services.
- **Independent evidence.** 50 test calls have a byte-identical prompt in train with a different label. These are mostly execute vs ask; no test refusal has a conflicting train copy.
- **Direction.** After the multi-turn fix, the dominant disagreement is refuse → ask: the models see the risk but prefer to escalate rather than refuse.
- **Caveat.** Shared model bias produces the same pattern as real label errors. The 40 strongest flags and the 50 conflicts are under manual review.

### 5.2 Leave-one-out influence on the certificate (M5)

**Idea.** The influence of a training point is the change in a model's output when the point is removed (Cook, 1977). Koh & Liang (2017) approximated this for deep models, and Data Shapley averages it over subsets (Ghorbani & Zou, 2019; Jia et al., 2019). Our "model" is the LTT calibration procedure, so removal is a cheap recomputation. We value each calibration call by its effect on the certified safe-autonomy rate.

**Results:**

- **Overall.** The certificate is robust to almost every single call.
- **The exceptions.** For liquid, removing any one of five refusals raises certified safe autonomy from 38.9% to 55.9%. Four of these five are truncated multi-turn calls, and all five are flagged by all five models in M4.
- **After the fix (M9),** these calls no longer hold the boundary.
- **Data Shapley was not run,** because leave-one-out was too coarse to rank the remaining calls.

### 5.3 Certificate diagnostics: lever decomposition (M6)

**Idea.** For each model × service pair, attribute non-certifiability to the levers identified in the ECIS paper, building on LTT and conformal prediction (Vovk, Gammerman & Shafer, 2005; Angelopoulos & Bates, 2023):

- **History lever:** labelled refusals against the minimum sample size (88–129).
- **Output lever:** the share of refusals at or near p = 0 (resolution floor).
- **Separability:** AUROC of p_refuse for refusals versus the rest.

The output is a verbal counterfactual, such as "54 labelled refusals, 102 needed". We found no prior work that explains why a risk-control certificate cannot be issued.

**Results:**

- **gpt** fails everywhere, because no policy keeps it within the ask budget.
- **Test data:** every service fails on sample size, with 12–54 refusals against the 102 needed.
- **Train data:**
  - Five services certify: agentharm, erpnext, owncloud, rocketchat and vaultwarden.
  - gitea, openemr and zammad are 7–9 refusals short.
  - nocodb under jev is the only case where the resolution floor binds.
- **atbench and dh_app** appeared inseparable (AUROC ≈ 0.55). After the fix (M9) they are separable, but they stay uncertifiable for lack of labelled refusals.

### 5.4 Risk-budget decomposition (M7)

**Idea.** The LTT guarantee is marginal over all refusals. We decompose the harm actually executed by subgroup, to show where the budget is spent. The guarantee itself does not report this.

**Results** (truncated input, 60 splits):

- **Subgroups.** Overall harm stays within budget (exceeded in ≤ 5% of splits). Yet harm on multi-turn refusals exceeds 5% in 47–73% of splits: about 1% of single-turn refusals run autonomously, against 5–8% of multi-turn ones.
- **Interpretation.** The guarantee holds, but most of its budget is spent on a single subgroup.
- **Dropping those calls is not a fix:** without them in calibration, liquid would execute 32% of multi-turn refusals.

### 5.5 Anchor re-scoring as a drift audit (M8)

**Idea.** A certificate holds only for the model state it was computed on. Commercial LLM services change behaviour over time without notice (Chen, Zaharia & Zou, 2024), and conformal guarantees assume that future data resemble the calibration data (Barber et al., 2023). We re-scored the full test split and recomputed the certificate.

**Results.** Two vendors changed behaviour within two weeks, both under unchanged version strings:

- **microsoft: the scores shifted.** Mean change per class was 0.07–0.15, and the choice changed on 18% of calls.
  - The frozen October policy still keeps both risk budgets, but now executes **1.7% instead of 26.3%** of safe calls.
  - Recalibrated on today's scores, it reaches 32.6%.
- **pplx: the model lost its determinism.**
  - Its effect on the certificate was small.
  - It invalidates explanation methods that assume a deterministic black box: exact Shapley was restricted to gpt.

**Implication.** Drift broke the gate's usefulness, not its guarantee. Certificates and explanations need anchor-based monitoring and a timestamp; version strings do not identify a model's behaviour.

### 5.6 Before/after re-scoring with leakage check (M9)

**Background.** The audit (M5–M7) pointed at a pipeline defect. For the 123 multi-turn test samples (733 in train), the scoring sends only the **first** user message, while the label refers to the **last** proposed call. The scorer therefore judged a different, usually harmless, call.

**What we did.** Re-scored these calls in separate folders, leaving the paper's frozen scores untouched, with three inputs:

- the full conversation
- **(a)** all user turns without the reviewer's earlier decisions (main figure)
- **(b)** the final user message only

Variant (a) checks whether the earlier decisions leak labels.

**Results with (a):**

- **Separability.** Refusal AUROC on multi-turn calls rises from about 0.5 to 0.68–0.79 (test) and 0.73–0.86 (train). dh_app becomes almost perfectly separable.
- **Certified autonomy.** G4 safe autonomy rises by **8 to 21 points** (mean of three: 35.4% → 51.7%).
- **Subgroup harm.** Harm on multi-turn refusals falls from 5.1–8.1% to 0–4.7%.
- **No measurable leakage.** (a) matches or beats the full conversation on almost every metric.

**What this shows.** Certificate-level explanations found a defect the certificate could not show; fixing it improved the certificate measurably.

## 6. Planned but deferred

- **Counterfactual edits.** The minimal change that would alter the routing (Wachter, Mittelstadt & Russell, 2018; Ross, Marasović & Peters, 2021), verified by re-querying, because LLM-proposed counterfactuals are often invalid (Mayne et al., 2025).
- **Labelled precedents.** Example-based explanations for reviewers.

Both serve reviewer-facing explanations, which the results above currently do not support. They are deferred until the framing is decided.

## 7. Overall picture

| | Certificate- and data-level explanations (M4–M9) | Prediction-level explanations (M1–M3) |
|---|---|---|
| Black-box feasibility | full; 0 or few queries | needs repeats for stochastic models; Shapley undefined for 51% of gpt calls |
| Uniqueness | deterministic given the data | depends on masking choice, method (occlusion vs Shapley) and model |
| Found the pipeline defect? | yes (M5, M6, M7) | no (M1, D4) |
| Found the label problems? | yes (M4) | no |
| Found the vendor drift? | yes (M8) | no; drift instead undermines them |
| Readable for managers or auditors | yes ("54 refusals, 102 needed"; "budget spent on one subgroup") | partly; the "reason" changes with the method |

**Emerging core message:** *to audit a certified delegation gate, explain the certificate, not the individual prediction.*

## 8. Open items

1. **Tomy's reviews:** the risk-class → field map (M3), the 40 strongest label flags and the 50 train/test conflicts (M4).
2. **Shapley for the other four models.** Not run: it would need repeats, and a large share of reduced inputs might be refused.
3. **Drift monitoring.** Could a small daily anchor set have detected the microsoft and pplx changes immediately? Cheap to test.
4. **Consequences for the ECIS submission.** The finding that no model separates refusals on two benchmark services is largely a truncation artefact. The size of jev's resolution floor is also affected: 27 of its 32 zero-probability refusals are truncated multi-turn calls.

## 8. References

Adebayo, J., Gilmer, J., Muelly, M., Goodfellow, I., Hardt, M., & Kim, B. (2018). Sanity checks for saliency maps. *Advances in Neural Information Processing Systems 31*. https://arxiv.org/abs/1810.03292

Alvarez-Melis, D., & Jaakkola, T. S. (2018). On the robustness of interpretability methods. *ICML Workshop on Human Interpretability in Machine Learning*. https://arxiv.org/abs/1806.08049

Angelopoulos, A. N., & Bates, S. (2023). Conformal prediction: A gentle introduction. *Foundations and Trends in Machine Learning, 16*(4), 494–591. https://arxiv.org/abs/2107.07511

Angelopoulos, A. N., Bates, S., Candès, E. J., Jordan, M. I., & Lei, L. (2025). Learn then test: Calibrating predictive algorithms to achieve risk control. *The Annals of Applied Statistics, 19*(2), 1641–1662. https://doi.org/10.1214/24-AOAS1998

Barber, R. F., Candès, E. J., Ramdas, A., & Tibshirani, R. J. (2023). Conformal prediction beyond exchangeability. *The Annals of Statistics, 51*(2), 816–845. https://doi.org/10.1214/23-AOS2276

Castro, J., Gómez, D., & Tejada, J. (2009). Polynomial calculation of the Shapley value based on sampling. *Computers & Operations Research, 36*(5), 1726–1730. https://doi.org/10.1016/j.cor.2008.04.004

Chen, L., Zaharia, M., & Zou, J. (2024). How is ChatGPT's behavior changing over time? *Harvard Data Science Review, 6*(2). https://doi.org/10.1162/99608f92.5317da47

Cook, R. D. (1977). Detection of influential observation in linear regression. *Technometrics, 19*(1), 15–18. https://doi.org/10.1080/00401706.1977.10489493

Covert, I., Lundberg, S., & Lee, S.-I. (2021). Explaining by removing: A unified framework for model explanation. *Journal of Machine Learning Research, 22*(209), 1–90. http://jmlr.org/papers/v22/20-1316.html

DeYoung, J., Jain, S., Rajani, N. F., Lehman, E., Xiong, C., Socher, R., & Wallace, B. C. (2020). ERASER: A benchmark to evaluate rationalized NLP models. *Proceedings of ACL 2020*, 4443–4458. https://doi.org/10.18653/v1/2020.acl-main.408

Ghorbani, A., & Zou, J. (2019). Data Shapley: Equitable valuation of data for machine learning. *Proceedings of ICML 2019*, PMLR 97, 2242–2251. https://arxiv.org/abs/1904.02868

Hase, P., Xie, H., & Bansal, M. (2021). The out-of-distribution problem in explainability and search methods for feature importance explanations. *Advances in Neural Information Processing Systems 34*. https://arxiv.org/abs/2106.00786

Hooker, S., Erhan, D., Kindermans, P.-J., & Kim, B. (2019). A benchmark for interpretability methods in deep neural networks. *Advances in Neural Information Processing Systems 32*. https://arxiv.org/abs/1806.10758

Jacovi, A., & Goldberg, Y. (2020). Towards faithfully interpretable NLP systems: How should we define and evaluate faithfulness? *Proceedings of ACL 2020*, 4198–4205. https://doi.org/10.18653/v1/2020.acl-main.386

Jia, R., Dao, D., Wang, B., Hubis, F. A., Gürel, N. M., Li, B., Zhang, C., Spanos, C. J., & Song, D. (2019). Efficient task-specific data valuation for nearest neighbor algorithms. *Proceedings of the VLDB Endowment, 12*(11), 1610–1623. https://doi.org/10.14778/3342263.3342637

Jullum, M., Redelmeier, A., & Aas, K. (2021). groupShapley: Efficient prediction explanation with Shapley values for feature groups. arXiv:2106.12228. https://arxiv.org/abs/2106.12228

Koh, P. W., & Liang, P. (2017). Understanding black-box predictions via influence functions. *Proceedings of ICML 2017*, PMLR 70, 1885–1894. https://arxiv.org/abs/1703.04730

Li, J., Monroe, W., & Jurafsky, D. (2016). Understanding neural networks through representation erasure. arXiv:1612.08220. https://arxiv.org/abs/1612.08220

Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems 30*. https://arxiv.org/abs/1705.07874

Mayne, H., et al. (2025). LLMs don't know their own decision boundaries: The unreliability of self-generated counterfactual explanations. *Proceedings of EMNLP 2025*. *(details taken from our literature review; check before citing)*

Northcutt, C. G., Athalye, A., & Mueller, J. (2021). Pervasive label errors in test sets destabilize machine learning benchmarks. *NeurIPS 2021 Datasets and Benchmarks Track*. https://arxiv.org/abs/2103.14749

Northcutt, C. G., Jiang, L., & Chuang, I. L. (2021). Confident learning: Estimating uncertainty in dataset labels. *Journal of Artificial Intelligence Research, 70*, 1373–1411. https://doi.org/10.1613/jair.1.12125

Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). "Why should I trust you?" Explaining the predictions of any classifier. *Proceedings of KDD 2016*, 1135–1144. https://doi.org/10.1145/2939672.2939778

Ross, A., Marasović, A., & Peters, M. E. (2021). Explaining NLP models via minimal contrastive editing (MiCE). *Findings of ACL-IJCNLP 2021*, 3840–3852. https://doi.org/10.18653/v1/2021.findings-acl.336

Shapley, L. S. (1953). A value for n-person games. In H. W. Kuhn & A. W. Tucker (Eds.), *Contributions to the Theory of Games II* (pp. 307–317). Princeton University Press.

Slack, D., Hilgard, S., Jia, E., Singh, S., & Lakkaraju, H. (2020). Fooling LIME and SHAP: Adversarial attacks on post hoc explanation methods. *Proceedings of AIES 2020*, 180–186. https://doi.org/10.1145/3375627.3375830

Štrumbelj, E., & Kononenko, I. (2014). Explaining prediction models and individual predictions with feature contributions. *Knowledge and Information Systems, 41*(3), 647–665. https://doi.org/10.1007/s10115-013-0679-x

Sturmfels, P., Lundberg, S., & Lee, S.-I. (2020). Visualizing the impact of feature attribution baselines. *Distill*. https://doi.org/10.23915/distill.00022

Sundararajan, M., & Najmi, A. (2020). The many Shapley values for model explanation. *Proceedings of ICML 2020*, PMLR 119, 9269–9278. https://arxiv.org/abs/1908.08474

Sundararajan, M., Taly, A., & Yan, Q. (2017). Axiomatic attribution for deep networks. *Proceedings of ICML 2017*, PMLR 70, 3319–3328. https://arxiv.org/abs/1703.01365

Turpin, M., Michael, J., Perez, E., & Bowman, S. R. (2023). Language models don't always say what they think: Unfaithful explanations in chain-of-thought prompting. *Advances in Neural Information Processing Systems 36*. https://arxiv.org/abs/2305.04388

Vovk, V., Gammerman, A., & Shafer, G. (2005). *Algorithmic Learning in a Random World*. Springer. https://doi.org/10.1007/b106715

Wachter, S., Mittelstadt, B., & Russell, C. (2018). Counterfactual explanations without opening the black box: Automated decisions and the GDPR. *Harvard Journal of Law & Technology, 31*(2), 841–887. https://arxiv.org/abs/1711.00399

Zeiler, M. D., & Fergus, R. (2014). Visualizing and understanding convolutional networks. In *Computer Vision – ECCV 2014*, LNCS 8689, 818–833. Springer. https://doi.org/10.1007/978-3-319-10590-1_53
