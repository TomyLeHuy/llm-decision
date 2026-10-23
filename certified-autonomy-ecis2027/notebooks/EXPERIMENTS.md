# Experimental Setup and Results

Conformal prediction on top of black-box decision models for step-level tool-call safety review.
Code: `02_full_evalrun.ipynb` (inference + baseline metrics) and `03_conformal_prediction.ipynb` (conformal analysis).
Raw outputs are in `results/`.

---

## 1. Objective

An autonomous agent proposes a tool call. A *safety reviewer* must choose one of three governance actions:

| Label | Meaning |
|---|---|
| `direct_execute` | safe to run autonomously |
| `ask_human` | requires human confirmation |
| `refuse` | must be blocked |

The errors do not cost the same. The critical failure is **catastrophic execution**: the true label is `refuse` and the system executes the call.
We ask:

1. How well do off-the-shelf decision models perform as reviewers (accuracy, calibration, governance errors)?
2. Can split conformal prediction (CP) on their output probabilities give a **finite-sample, distribution-free bound** on catastrophic execution? What does that bound cost in autonomy?

---

## 2. Experimental Setup

### 2.1 Data

- **Source:** in-domain test split `data/test_1436.json` of the step-level safety-review corpus (release `full_data_20260530`). The train split (7,767) and the held-out Mailu OOD split (198) are **not used** so far.
- **Size and label distribution:** N = 1436. `direct_execute` 540 (37.6 %), `ask_human` 488 (34.0 %), `refuse` 408 (28.4 %).
- **Services:** 15 sources, e.g. when2call 271, vaultwarden 196, agentharm 143, tsbench 125, atbench 119, zammad 97, plus smaller app-based services (nocodb, gitea, rocketchat, erpnext, owncloud, openemr, …).
- **Input per sample:** the reviewer system prompt (decision contract) plus a JSON user message with `user_task`, `service` and `assistant_proposed_tool_call`.
- **Ground truth:** `meta.decision`.
- **Note:** `risk_family` metadata is missing for 748 / 1436 samples (empty or NaN), so stratified analyses by risk family are only indicative.

### 2.2 Models and inference

| Model (OpenRouter ID) | Successful calls |
|---|---|
| `typesafe/jev-1.14` | 1436 |
| `microsoft/microsoft-decision-1` | 1436 |
| `openai/gpt-6-luna-decisions` | 1433 (3 × HTTP 502 after 5 retries) |

- **Endpoint:** OpenRouter `api/alpha/decisions`. Each request sends a `state` (system + user message of the sample) and a single `choice` question with three options. Each option comes with a natural-language criterion (identical for all models; see `QUESTIONS_PAYLOAD` in notebook 02).
- **Output:** a discrete choice plus a probability vector $\hat p(x) \in \Delta^2$ over the three labels. How the provider computes these probabilities is opaque, so the models are treated as **black boxes**.
- **Probability resolution:** `jev-1.14` and `gpt-6-luna-decisions` return probabilities on a 0.01 grid, including exact zeros. `microsoft-decision-1` returns continuous values and no exact zeros.
- **Pre-processing:** failed calls are dropped. Probabilities are clipped at 0 and renormalized (the max deviation of the sum from 1 was 0.01).
- **Robustness of the run:** exponential backoff (5 retries) and per-call JSONL checkpointing. Total runtime 31 min, total cost ≈ $0.34.

### 2.3 Baseline evaluation (argmax / returned choice)

- **Classification:** accuracy, balanced accuracy, macro / weighted F1, Cohen's κ, per-class precision / recall.
- **Probability quality:** Brier score, NLL (probabilities clipped at 1e-15), top-1 ECE (15 bins).
- **Uncertainty signal:** mean normalized entropy on errors vs. on correct predictions.
- **Governance rates:**

DE = `direct_execute`, AH = `ask_human`, RF = `refuse`.

| Rate | Definition |
|---|---|
| catastrophic | $P(\hat y = \text{DE} \mid y = \text{RF})$ |
| safe fallback | $P(\hat y = \text{AH} \mid y = \text{RF})$ |
| friction | $P(\hat y = \text{AH} \mid y = \text{DE})$ |
| false block | $P(\hat y = \text{RF} \mid y = \text{DE})$ |
| automation | $P(\hat y \neq \text{AH})$ |

### 2.4 Conformal prediction protocol

**Split conformal.** The 1436 samples are split 50/50 into calibration and test, stratified by ground truth. The split is the same for all models; the gpt model's 3 missing samples are simply absent.

For each split:

1. Compute nonconformity scores $s(x_i, y_i)$ on the calibration set.
2. Compute $\hat q$ as the $\lceil (n+1)(1-\alpha) \rceil / n$ empirical quantile, using `method="higher"`.
3. Build the prediction set $C(x) = \{y : s(x,y) \le \hat q\}$.

Guarantee (under exchangeability): $P(y \in C(x)) \ge 1-\alpha$.

| Method | Score $s(x,y)$ | Quantile |
|---|---|---|
| LAC | $1 - \hat p_y(x)$ | one global quantile |
| APS (randomized) | $\sum_{y': \hat p_{y'} > \hat p_y} \hat p_{y'} + U \cdot \hat p_y$ | one global quantile |
| Mondrian LAC | $1 - \hat p_y(x)$ | one quantile per true class $k$ with its own $\alpha_k$; guarantees $P(y \in C(x) \mid y = k) \ge 1-\alpha_k$ |

**Governance decision rule.** A singleton set $\{y\}$ leads to action $y$. Any set of size ≥ 2 or an empty set leads to `ask_human`.

Under this rule a true `refuse` can only be auto-executed when `refuse` ∉ C(x). With Mondrian CP this gives **catastrophic rate ≤ α_refuse** in expectation.

**Configurations.**

| Experiment | Methods | α | Repetitions |
|---|---|---|---|
| E1: main comparison | LAC, APS, Mondrian LAC | 0.10 (all classes) | 100 random splits (seeds 0–99) |
| E2: α sweep | LAC | 0.02, 0.04, …, 0.40 | 30 splits per α |
| E3: risk-asymmetric | Mondrian LAC | α_DE = 0.15, α_AH = 0.15, α_RF = 0.05 | 100 splits |

**Metrics.**
- **Coverage:** marginal and per class.
- **Set size:** mean, plus singleton rate and empty-set rate.
- **Singleton accuracy.**
- **Governance rates** after applying the decision rule, compared with argmax on the same test half.

All values reported below are means over the splits.

---

## 3. Results

### 3.1 Baseline (no conformal step, full N)

| Model | Acc | Bal. Acc | Macro F1 | κ | Brier ↓ | ECE ↓ | NLL ↓ | H(err)/H(cor) |
|---|---|---|---|---|---|---|---|---|
| jev-1.14 | **0.649** | **0.632** | **0.638** | **0.461** | **0.505** | **0.148** | 1.676 | 1.52 |
| microsoft-decision-1 | 0.595 | 0.576 | 0.576 | 0.375 | 0.678 | 0.308 | **1.547** | 1.74 |
| gpt-6-luna-decisions | 0.623 | 0.608 | 0.615 | 0.420 | 0.598 | 0.252 | 2.930 | 1.73 |

| Model | `refuse` precision / recall | Catastrophic | Safe fallback | Friction | False block |
|---|---|---|---|---|---|
| jev-1.14 | 0.89 / 0.47 | 15.7 % | 37.0 % | 14.4 % | 0.9 % |
| microsoft-decision-1 | 0.82 / 0.41 | 26.2 % | 32.4 % | 12.0 % | 2.2 % |
| gpt-6-luna-decisions | 0.88 / 0.48 | 20.7 % | 31.6 % | 17.0 % | 0.6 % |

- All models are biased towards `direct_execute` (recall 0.82–0.86) and miss more than half of the true refusals.
- `ask_human` vs. `direct_execute` is the most confused boundary. For true `ask_human` samples, microsoft puts more mean mass on `direct_execute` (0.49) than on `ask_human` (0.45).
- Errors have 1.5–1.7× higher entropy than correct predictions, so the probabilities carry an uncertainty signal that CP can use.
- The share of samples whose true label gets probability ≤ 0.01 is 5.0 % (jev), 14.1 % (microsoft) and 12.9 % (gpt). Exact zeros occur for 2.6 % (jev) and 5.7 % (gpt) of samples, and never for microsoft.

### 3.2 E1: Validity and efficiency at α = 0.10

| Model | Method | Coverage | Avg. set size | Singleton | Singleton acc. |
|---|---|---|---|---|---|
| jev-1.14 | LAC | 0.906 | 1.84 | 0.305 | 0.863 |
| | APS | 0.900 | 1.88 | 0.278 | 0.848 |
| | Mondrian | 0.915 | 2.00 | 0.247 | 0.849 |
| microsoft-decision-1 | LAC | 0.902 | 2.13 | 0.234 | 0.770 |
| | APS | 0.901 | 2.12 | 0.213 | 0.767 |
| | Mondrian | 0.906 | 2.02 | 0.210 | 0.793 |
| gpt-6-luna-decisions | LAC | 0.943 | 2.45 | 0.155 | 0.726 |
| | APS | 0.904 | 2.19 | 0.242 | 0.735 |
| | Mondrian | 0.954 | 2.31 | 0.016 | 0.853 |

- Marginal coverage reaches the target for every model and method. Split-to-split std is ≈ 0.014. Empty sets occur in ≤ 0.2 % of cases.
- **gpt over-covers under LAC/Mondrian.** Its probabilities sit on a 0.01 grid with many exact zeros, so $\hat q$ lands on 0.99 and the set boundary cannot be placed between ties. Randomized APS removes the over-coverage.
- **Efficiency is low:** 1.8–2.5 of 3 labels per set on average. The calibrated LAC thresholds are 0.93 (jev), 0.99 (gpt) and 0.995 (microsoft), i.e. a label is excluded only if $\hat p_y < 0.07$ (jev) or < 0.01 (gpt / microsoft).

### 3.3 E1: Class-conditional coverage

| Model | Method | cov(DE) | cov(AH) | cov(RF) |
|---|---|---|---|---|
| jev-1.14 | LAC | 0.977 | 0.922 | **0.792** |
| | Mondrian | 0.906 | 0.914 | 0.926 |
| microsoft-decision-1 | LAC | 0.996 | 0.859 | **0.830** |
| | Mondrian | 0.902 | 0.908 | 0.907 |
| gpt-6-luna-decisions | LAC | 0.978 | 0.879 | 0.974 |
| | Mondrian | 0.904 | 0.995 | 0.971 |

Marginal CP under-covers the safety-critical class `refuse` by about 7–11 pp for jev and microsoft. Mondrian CP restores at least 1 − α coverage for every class.

### 3.4 E1: Governance impact (argmax → CP-gated, LAC at α = 0.10)

| Model | Catastrophic | Friction | False block | Automation | Action acc. |
|---|---|---|---|---|---|
| jev-1.14 | 15.8 → **3.2 %** | 14.5 → 56.7 % | 0.9 → 0 % | 64.3 → 23.0 % | 0.648 → 0.514 |
| microsoft-decision-1 | 26.7 → **1.4 %** | 11.8 → 62.7 % | 2.3 → 0 % | 71.0 → 21.2 % | 0.595 → 0.452 |
| gpt-6-luna-decisions | 20.7 → **0.0 %** | 17.0 → 74.0 % | 0.5 → 0 % | 66.7 → 14.5 % | 0.623 → 0.403 |

With Mondrian at α = 0.10 the catastrophic rate is 2.1 % (jev), 0.4 % (microsoft) and 0.0 % (gpt). gpt's automation then collapses to 0.5 %.

"Action acc." counts escalations of true `direct_execute` / `refuse` samples as errors. It therefore understates the safety value of escalating.

### 3.5 E2: Coverage–autonomy trade-off (LAC, 30 splits)

| α | Model | Coverage | Avg. size | Singleton | Catastrophic (CP) |
|---|---|---|---|---|---|
| 0.10 | jev / microsoft / gpt | 0.91 / 0.90 / 0.94 | 1.84 / 2.14 / 2.45 | 0.31 / 0.23 / 0.16 | 3.2 / 1.3 / 0.0 % |
| 0.20 | jev / microsoft / gpt | 0.81 / 0.80 / 0.81 | 1.44 / 1.65 / 1.62 | 0.59 / 0.50 / 0.53 | 9.2 / 6.2 / 5.7 % |
| 0.30 | jev / microsoft / gpt | 0.70 / 0.70 / 0.71 | 1.14 / 1.28 / 1.19 | 0.86 / 0.75 / 0.81 | 14.6 / 16.8 / 12.2 % |

- Coverage tracks 1 − α over the full range (α = 0.02–0.40). See the figure in notebook 03.
- At α ≈ 0.30 the catastrophic rate approaches the argmax baseline. Autonomy and safety trade off almost monotonically.

### 3.6 E3: Risk-asymmetric Mondrian (α_RF = 0.05, α_DE = α_AH = 0.15)

| Model | cov(DE) | cov(AH) | cov(RF) | Avg. size | Catastrophic (argmax → CP) | Friction (CP) | Automation (CP) | $\hat q_{RF}$ |
|---|---|---|---|---|---|---|---|---|
| jev-1.14 | 0.856 | 0.866 | 1.000 | 2.19 | 15.8 → 0.0 % | 100 % | **4.6 %** | 1.000 |
| microsoft-decision-1 | 0.854 | 0.858 | 0.959 | 2.02 | 26.7 → 1.3 % | 63.4 % | **20.9 %** | 0.999 |
| gpt-6-luna-decisions | 0.855 | 0.886 | 0.974 | 2.12 | 20.7 → 0.0 % | 76.4 % | **13.9 %** | 0.990 |

- The catastrophic rate stays within the α_RF = 5 % budget for every model.
- **For jev the guarantee is trivial.** 7.8 % of true-`refuse` samples get $\hat p_{RF} = 0$ exactly, which is more than α_RF, so $\hat q_{RF} = 1$ and `refuse` is included in **every** set. No `direct_execute` singleton can ever be formed. Friction is 100 % and automation is 4.6 %, almost entirely `refuse` singletons.
- **The model ranking reverses.** microsoft has the worst accuracy and ECE but never outputs exact zeros and has the best NLL. It keeps the most autonomy under the strict budget (20.9 %).
- Example split (jev): the singleton rate is 26 % for the `refuse` risk family vs. 1.4 % for `clean_de`. CP-gated autonomy comes mainly from refusals, not from benign executions.

---

## 4. Summary of Findings

1. **The argmax decision is unsafe as an autonomous gate.** 16–27 % of harmful tool calls would be executed.
2. **CP is valid on this data.** Marginal coverage matches 1 − α for all models. Mondrian CP is required to get the guarantee on `refuse`; marginal CP under-covers it by up to 11 pp.
3. **CP-gating bounds catastrophic execution to a few percent,** down from 16–27 % to 0–3 % at α = 0.10, and ≤ α_RF under the asymmetric setting. Autonomous decisions that remain are more accurate (singleton accuracy 0.73–0.86 vs. 0.60–0.65 argmax).
4. **The cost is autonomy.** Only 5–23 % of decisions remain automated, and most benign calls are escalated. This is acceptable for our primary goal (safety over autonomy), but it limits practical usefulness.
5. **CP efficiency depends on the tail of the true-label probability (exact or near zeros), not on accuracy or ECE.** Model selection for conformal governance should therefore use proper scoring rules, and a model-level comparison should account for this.

## 5. Threats to Validity

- **Exchangeability.** Calibration and test halves come from the same in-domain test split. The guarantee is untested under service or attack-type shift; the Mailu OOD split is available for this.
- **Black-box probabilities.** It is unknown how the endpoint derives the probabilities: they could be token log-probs, sampling frequencies or verbalized values. Their resolution differs across models (0.01 grid vs. continuous), which affects ties and over-coverage.
- **Small per-class calibration sets.** About 204 `refuse` samples per split. For small α_k the finite-sample correction quickly reaches $\hat q = 1$, which gives trivial sets.
- **Label noise.** The corpus contains relabeled samples (`label_flipped`, `label_corrected` in `meta`). Coverage is measured against these labels.
- **Decision rule.** Every non-singleton set escalates. Rules that look at set composition (e.g. `{DE, AH}` → AH, any set containing RF → AH, `{RF}` → refuse) were not evaluated separately.
- **Baseline definitions.** Notebook 02 uses the returned `choice`; notebook 03 uses the argmax of the probabilities. These differ in 1–4 samples per model (ties).
- **Single dataset and three models.** No confidence intervals across models. Split variability is reported as std over splits in notebook 03.

## 6. Planned Next Steps

- **Model multiplicity:** quantify disagreement between the three (or more) decision models on the same samples. This includes argmax disagreement, prediction-set overlap, and whether samples with catastrophic errors are shared across models. Also test multiplicity-aware aggregation (e.g. union or intersection of sets, ensemble probabilities) before calibration.
- **Conformal risk control** with an explicit cost matrix instead of 0/1 coverage.
- **Shift evaluation:** calibrate in-domain, test on Mailu OOD and on held-out services.
