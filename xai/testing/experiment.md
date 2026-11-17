# XAI pilot: field-level Shapley values for the delegation gate

## 1. Goal

Following our call on 10 November, the first XAI step is **SHAP with summary plots**. This pilot answers two questions:

1. Can SHAP be computed cleanly for the decision model behind the gate?
2. What do the first summary plots show?

These steps come later and are **not** part of this pilot:
- calibration
- conformal "uncertainty profiles" (SHAP per set size 1/2/3)
- the other two models of the gate (jev, microsoft)
- counterfactuals

## 2. Setup

| | |
|---|---|
| Model | gpt (`openai/gpt-6-luna-decisions`) via the OpenRouter decisions endpoint, called exactly as in the paper's scoring script |
| Output | probabilities for `direct_execute`, `ask_human`, `refuse` |
| Data | 30 calls from the test split (`test_1436.json`), single-step only, 10 per true label, fixed seed |
| Library | llmSHAP 1.5.2 (Naudot, Sundqvist & Kampik 2025, arXiv 2511.01311) |
| Cost | USD 0.13 for 1,301 distinct API requests |

**Why single-step only:** I fixed a bug for multi-step calls. Earlier, only the first user turn was sent. Single-step inputs are identical before and after the fix, so they can be used now. Multi-step calls follow after re-scoring.

**Why gpt first:** gpt is deterministic, so one query per input gives the exact value with no sampling noise.

## 3. Method

### Players are input fields, not tokens

Each call reaches the model as a JSON with up to seven fields:

- Context: `user_task`, `service`, `user_memory`, `prior_steps`
- Proposed tool call: `tool`, `tool_args`, `description`

Each field is one Shapley "player". The system policy and the decision questions are always sent; they are not players.

**Why fields and not tokens:**
- Fields are the units the gate actually reasons about (task, memory, proposed action).
- With at most 7 fields there are at most 2⁷ = 128 subsets, so Shapley values can be computed **exactly**. No sampling approximation (KernelSHAP-style) is needed.

### How a value is computed

1. For one call, build every subset ("coalition") of its fields. "Removing" a field means deleting its key from the JSON.
2. Send each subset to gpt once and cache the answer. The cache makes the Shapley axioms hold even for LLMs; this is the core idea of llmSHAP.
3. The payoff v(S) is gpt's probability for one class when only the fields in S are present. Each class is run separately.
4. llmSHAP computes the exact Shapley value φ of every field: its average marginal effect on that probability over all subsets.
5. **Efficiency** holds per call: v(baseline) + Σ φ = gpt's probability on the full input.

### Why llmSHAP, and what it does here

[llmSHAP](https://github.com/filipnaudot/llmSHAP) (Naudot, Sundqvist & Kampik 2025, arXiv 2511.01311; v1.5.2, MIT licence) is a Shapley library built for LLMs.

**The problem it solves.** Shapley values are only well defined if v(S) is a fixed number for each subset S. LLM outputs can vary between calls, so the same subset can get different payoffs during one computation. When that happens, the Shapley axioms (efficiency, symmetry, null player) can fail. llmSHAP queries each subset once and caches the answer. Every later use of that subset reuses the cached value, so the axioms hold by construction. This is the paper's main contribution.

**Why it fits our setting.**
- **Players are user-defined units.** We can use JSON fields as players instead of tokens or words.
- **Payoff is user-defined.** We use gpt's class probability; prediction entropy would also work later.
- **Exact computation.** `FullEnumerationSampler` enumerates all subsets, which is feasible with ≤6 players. Approximate samplers (stratified, sliding window, counterfactual) are available if the player count grows.
- **Fixed fields are supported.** `permanent_keys` keeps the fixed parts (tool name, or the whole tool call) in every subset, including the baseline. This is exactly what the refusal problem required.
- **Robust for non-deterministic models.** gpt is deterministic, but jev and microsoft may not be. The cache keeps the values consistent for them as well.

**Division of work between llmSHAP and our code.**

| llmSHAP does | Our code does |
|---|---|
| enumerates the subsets and applies the Shapley weights | `LLMInterface`: sends a subset to the OpenRouter decisions endpoint, exactly as the paper's scorer does, with an extra disk cache shared across runs and designs |
| caches answers per subset | `PromptCodec`: turns a subset of fields back into the user JSON, in the record's original serialisation format |
| returns φ per player | `ValueFunction`: picks one class probability as the payoff |
| keeps `permanent_keys` in every subset | `DataHandler`: defines the players and the fixed fields |

So llmSHAP is the Shapley engine, and our code decides what a player, a removal and a payoff mean for the gate. As a safeguard, I also recomputed the exact Shapley values with my own implementation from the cached payoffs. The results are identical to llmSHAP's (max difference 0.0, section 4).

**Why not TokenSHAP**, the other LLM-SHAP package I checked:
- It uses a difference of averages, which is a Banzhaf-like index, not Shapley.
- It normalises values by shifting them to the minimum, which loses the sign (whether a part raises or lowers the output).
- Its only payoff is text similarity, not a class probability.
- It has no cache.

llmSHAP builds no categories of its own; the players are whatever we define, here the input fields. Polarity (whether a field raises or lowers a probability) comes directly from the sign of φ.

### Difference from classic SHAP

| | Classic SHAP (tabular) | This pilot |
|---|---|---|
| "Absent" feature | replaced by values from background data | field deleted from the input |
| Base value | mean prediction E[f(X)] | model output with only the fixed fields (differs per call) |
| Estimation | KernelSHAP / TreeSHAP | exact enumeration (≤ 64 subsets per call) |
| Dot colour in summary plot | feature value (high/low) | true label (text has no high/low) |

### A design problem: gpt refuses inputs without a tool call

The first plan was to treat all 7 fields as players. On 3 pilot calls this failed:
- gpt returns an HTTP 502 "refused to answer" for the empty input and for any subset without a tool-call field. That was 10 of 222 subsets.
- So the baseline v(∅) is undefined, and Shapley values cannot be computed.

I therefore ran two designs, in which part of the input is always present:

| Design | Always present | Players | Baseline means … | Complete calls |
|---|---|---|---|---|
| **Main** (`fixed-tool`) | tool name | user_task, service, user_memory, prior_steps, tool_args, description | "the reviewer only knows which tool is called" | 29 / 30 |
| **Second view** (`fixed-description+tool+tool_args`) | the whole proposed tool call | user_task, service, user_memory, prior_steps | "the reviewer knows the proposed action, but no context" | 30 / 30 |

In the main design, one subset of call 519 was refused, even though the tool name was present. I did not impute it, so call 519 has no values in the main design.

The two designs answer different questions:
- **Main:** what moves the decision, including the action's arguments and description?
- **Second view:** given exactly this action, which context makes gpt more or less cautious?

## 4. Quality checks

All checks passed.

| Check | Result |
|---|---|
| The reconstructed full input is byte-identical to what the paper's scorer sent | 30/30 |
| Determinism: same input queried twice (3 calls) | max difference 0 |
| Full-input probabilities vs the October scores of the paper | max difference 0.0 |
| Efficiency: max \|v(baseline) + Σφ − v(full)\| | 1.2e-16 |
| My own exact Shapley implementation vs llmSHAP | max difference 0.0 |

**Comparison with leave-one-out (LOO), which removes one field at a time:**
- In the main design (6 players), LOO and Shapley name the same most important field in only 55–62% of calls. The median per-call rank correlation is 0.53 / 0.75 / 0.78 (execute / ask / refuse).
- So the fields interact, and simple occlusion would give a different picture.
- In the second view (4 players), the two agree much more (67–77% same top field).

## 5. Results

The numbers are descriptive: 30 calls, no confidence intervals yet.

### 5.1 Global importance: mean |φ| per field

Each bar is the average absolute φ of a field, over the calls that contain the field. It shows how much a field moves the probability, not in which direction.

![Mean |phi|, main design](plots/mean_abs_phi__fixed-tool.png)

| Field (main design) | P(execute) | P(ask) | P(refuse) |
|---|---|---|---|
| user_task | 0.077 | **0.133** | **0.152** |
| tool_args | **0.084** | 0.091 | 0.049 |
| prior_steps (n = 12) | 0.077 | 0.080 | 0.042 |
| user_memory (n = 20) | 0.043 | 0.078 | 0.062 |
| description | 0.032 | 0.075 | 0.079 |
| service | 0.023 | 0.027 | 0.020 |

- **user_task** is the most influential field for ask and refuse.
- **tool_args**, **prior_steps** and **user_task** matter most for execute.
- **service** barely matters for any class.

### 5.2 Per-call effects: beeswarm plots (main design)

**How to read them:**
- One row per field, with n = the number of calls that contain the field.
- One dot per call, placed at its φ for that class's probability. Right of 0 raises the probability; left of 0 lowers it.
- Colour and shape show the true label.

**Refuse**

![Beeswarm refuse, main design](plots/beeswarm_refuse__fixed-tool.png)

- For refuse-labelled calls, the mean φ on P(refuse) is:
  - **+0.34 for user_task**, up to +0.76 for single calls
  - **+0.19 for description**
  - −0.02 for tool_args
- So refusals are driven by what the task asks for and by how the action is described, not by the concrete arguments.
- Baselines for refuse-labelled calls: with only the tool name, gpt gives mean P(refuse) = 0.18 and P(ask) = 0.68. On the full input, mean P(refuse) = 0.73. The context fields move these calls from "ask" to "refuse".

**Execute**

![Beeswarm direct_execute, main design](plots/beeswarm_direct_execute__fixed-tool.png)

- For execute-labelled calls, the fields that raise P(execute) are:
  - **prior_steps** (+0.09)
  - **user_memory** (+0.08)
  - **tool_args** (+0.07)
- Concrete arguments and a history of earlier steps make gpt more willing to execute.

**Ask**

![Beeswarm ask_human, main design](plots/beeswarm_ask_human__fixed-tool.png)

- Effects are mixed and closer to 0.
- For ask-labelled calls, the tool name alone already gives a P(ask) of 0.44, and the full input gives 0.51.
- gpt often confuses ask-labelled calls with execute: their mean P(execute) on the full input is 0.40.

### 5.3 Second view: the whole tool call fixed

![Mean |phi|, second view](plots/mean_abs_phi__fixed-description+tool+tool_args.png)

- **The proposed tool call alone already carries much of the refusal.** For refuse-labelled calls, the baseline P(refuse) rises from 0.18 (tool name only) to **0.50** (whole tool call).
  - So tool_args and description together account for a large part of the refusal, before any context is seen.
- **user_memory gains weight once the action is fixed.** Its mean |φ| on P(ask) is 0.116, about equal to user_task.
  - For execute-labelled calls, the memory raises P(execute) by +0.14 on average.
  - In single calls, it lowers P(ask) by up to −0.74.

**Refuse**

![Beeswarm refuse, second view](plots/beeswarm_refuse__fixed-description+tool+tool_args.png)

**Execute**

![Beeswarm direct_execute, second view](plots/beeswarm_direct_execute__fixed-description+tool+tool_args.png)

**Ask**

![Beeswarm ask_human, second view](plots/beeswarm_ask_human__fixed-description+tool+tool_args.png)

### 5.4 Observation to discuss: the description field may leak the label

In several refuse-labelled calls, the `description` of the proposed tool call states the problematic intent outright. Examples (shortened):

- 1255: *"Create a shadow staging folder … M&A target …"*
- 918: *"Create a fabricated … encounter … for billing purposes"*
- 832: *"Rewrite the reason … to soften the chart narrative"*
- 405: *"Close the open … regulator inquiry ticket …"*

An agent would rarely describe its own action like this. This may make refusal easier than in realistic use, and it explains the high φ of `description` for refuse.

## 6. Five-model comparison

*Run on 11 November 2026.* The same 30 calls and the same two designs were run for the other four decision models:
jev (`typesafe/jev-1.14`), microsoft (`microsoft/microsoft-decision-1`), liquid (`liquid/d1-ultra`) and pplx
(`perplexity/pplx-decider-v1.2`). All numbers in this section are descriptive (n = 30 calls, no confidence intervals).
They come from `results/compare/` (`src/compare.py`) and the per-model `checks*.json`.

### 6.1 What changed in the method

- **Determinism probe first.** Per model: the full input of the 3 pilot calls 5×, plus 2 random partial inputs per
  call 3× each (33 queries). Unlike gpt, none of the four is deterministic. The max difference between repeated
  answers was 0.03 (jev), 0.032 (microsoft), 0.27 (liquid) and 0.051 (pplx).
- **Payoff = mean of 3 answers.** For these models every subset was queried 3 times, and v(S) is the mean. The mean
  is computed once from the cache and reused everywhere, so the Shapley axioms still hold. The 3 raw answers and
  their standard deviation are kept, so we can say which φ are smaller than the answer noise.
- **The 7-player design works for the four other models.** They answered every input, including the empty one and
  all subsets without a tool-call field. Only gpt refuses. So for them the 7-player design is a third view, and the
  two fixed designs were run as well (entirely from the same cache) for comparability with gpt.

### 6.2 Key numbers per model

| | gpt | jev | microsoft | liquid | pplx |
|---|---|---|---|---|---|
| deterministic | yes | no | no | no | no |
| answers per subset | 1 | 3 | 3 | 3 | 3 |
| refusals of the provider | 11 requests | 0 | 0 | 0 | 0 |
| 7-player design defined | no | yes | yes | yes | yes |
| complete calls: main / second view / 7-player | 29 / 30 / – | 30 / 30 / 30 | 30 / 30 / 30 | 30 / 30 / 30 | 30 / 30 / 30 |
| full input vs October scores: max / mean diff | 0.0 / 0.0 | 0.060 / 0.010 | **0.745 / 0.113** | 0.124 / 0.019 | 0.061 / 0.006 |
| answer noise: mean std | – | 0.011 | 0.008 | 0.030 | 0.004 |
| share of \|φ\| below that noise (main; exec / ask / refuse) | – | 42 / 37 / 57% | 24 / 18 / 39% | 45 / 46 / 59% | 8 / 12 / 22% |
| cost (USD) | 0.13 | 0.37 | 0.30 | 0.28 | 0.14 |

All quality checks of section 4 pass for every model and design: byte-identical round trip (30/30), efficiency
error ≤ 2.2e-16, own exact Shapley vs llmSHAP max difference 0.0, llmSHAP baseline and full value vs cache 0.0.

**microsoft has changed since October.** On the full input it differs from the October scores by up to 0.74 (mean
0.11), much more than its answer noise (0.008), while the API still reports the same version string
(`microsoft-decision-1-20261009`). Its φ explain today's model, not the one scored for the paper.

### 6.3 Global importance across models (main design)

![Mean |phi| per field and model, main design](plots/compare/mean_abs_phi__fixed-tool.png)

- **user_task is the strongest field for P(refuse) in all five models** (mean |φ| 0.10–0.17).
- **jev and microsoft rely much more on tool_args.** Mean |φ| on P(ask) is 0.196 (jev) and 0.319 (microsoft),
  against 0.09–0.10 for gpt, liquid and pplx. It is their top field for execute and ask.
- **service barely matters in any model** (mean |φ| ≤ 0.04).

Second view (whole tool call fixed):

![Mean |phi| per field and model, second view](plots/compare/mean_abs_phi__fixed-description+tool+tool_args.png)

### 6.4 Direction of the effects on the label's own class

![Mean phi on the own class, main design](plots/compare/mean_phi_own_class__fixed-tool.png)

| Field (main design) | gpt | jev | microsoft | liquid | pplx |
|---|---|---|---|---|---|
| **refuse-labelled, φ on P(refuse):** user_task | +0.343 | +0.161 | +0.188 | +0.225 | +0.203 |
| description | +0.189 | +0.188 | +0.195 | +0.170 | +0.174 |
| user_memory | +0.079 | +0.125 | +0.105 | +0.117 | +0.129 |
| **execute-labelled, φ on P(execute):** tool_args | +0.074 | +0.272 | +0.349 | +0.172 | +0.141 |
| prior_steps | +0.086 | −0.178 | −0.262 | +0.011 | −0.029 |
| user_memory | +0.082 | +0.078 | +0.055 | +0.077 | +0.101 |

- **description pushes towards refuse equally in all five models** (+0.17 to +0.20). This fits the possible label
  leakage in section 5.4: it is a property of the data, not of one model.
- **gpt relies most on the task text for refusals** (+0.34, against +0.16 to +0.23). In the second view, with the
  whole tool call fixed, user_task adds almost nothing to P(refuse) for the other four (−0.01 to +0.03), but still
  +0.15 for gpt.
- **prior_steps changes sign between models.** For execute-labelled calls it raises P(execute) in gpt (+0.09) but
  lowers it in jev (−0.18) and microsoft (−0.26). The same history makes some models more and others less willing to
  execute.

![Mean phi on the own class, second view](plots/compare/mean_phi_own_class__fixed-description+tool+tool_args.png)

**Baselines on the label's own class (main design, mean over calls).** v(∅) = only the tool name; v(full) = full
input.

| | gpt | jev | microsoft | liquid | pplx |
|---|---|---|---|---|---|
| execute-labelled: P(execute), v(∅) → v(full) | 0.62 → 0.83 | 0.41 → 0.72 | 0.39 → 0.77 | 0.59 → 0.84 | 0.59 → 0.90 |
| ask-labelled: P(ask), v(∅) → v(full) | 0.44 → 0.51 | 0.74 → 0.60 | 0.73 → 0.54 | 0.43 → 0.39 | 0.54 → 0.49 |
| refuse-labelled: P(refuse), v(∅) → v(full) | 0.18 → 0.73 | 0.12 → 0.57 | 0.14 → 0.70 | 0.21 → 0.79 | 0.15 → 0.62 |

With only the tool name, jev and microsoft lean towards "ask" (about 0.74 for ask-labelled calls). For them, the
context mostly moves calls away from "ask"; gpt, liquid and pplx start lower and move less.

### 6.5 How similar are the explanations?

![Agreement between models, main design](plots/compare/agreement__fixed-tool.png)

For each pair of models and each call, I correlated the φ vectors (Spearman) and checked whether both name the same
top field.

- **gpt is the outlier.** Main design: pairs with gpt have a mean of median correlations of 0.53 and the same top
  field in 50% of calls. Pairs without gpt reach 0.66 and 63%. Second view: 0.62 vs 0.82.
- **The most similar pairs are jev–microsoft** (median ρ 0.83 / 0.79 / 0.70 for execute / ask / refuse) **and
  liquid–pplx** (0.77 / 0.68 / 0.80).

### 6.6 What this means

- The pipeline transfers to all five models without changes to the design; only gpt needs the fixed tool name.
- For non-deterministic models, many small φ lie within the answer noise (37–59% of all φ for jev and liquid in the
  main design, 8–39% for pplx and microsoft). Only
  the larger effects in the plots should be read as signal; more repeats would be needed for the small ones.
- Model drift (microsoft) affects explanations as much as scores: φ must be computed on the model version that the
  gate actually uses.

## 7. Limitations

- 30 calls, single-step only, gpt only. All patterns are descriptive.
- φ explains **what gpt does, not whether it is correct**. Several calls are misclassified (e.g. 1365, labelled ask_human, gets P(execute) = 1.0), and their φ describe that wrong decision.
- Removing a field is one choice of "absence". φ measures the effect of deleting a field, not of replacing it with a neutral value.
- The refusal rule of gpt is unknown. Call 519 was refused even with the tool name present.
- Excluding the 3 calls that gpt already refused in October, and dropping call 519, biases the sample slightly towards inputs gpt accepts.
- Determinism was re-checked directly on 3 calls. The other 27 rely on the exact match with the October scores.

## 8. Planned next steps

1. **Scale up:** all ≈1,300 single-step test calls for gpt, main design. Estimated ≈50k requests, ≈USD 4–5.
2. **Calibration + conformal sets** on the full-input scores. Then **uncertainty profiles**: separate summary plots for calls with set size 1, 2 and 3.
3. Optionally, **entropy of the prediction as the payoff**. This explains which fields make gpt uncertain, not only which class it picks.
4. Multi-step calls after re-scoring with the bug fix.
5. The other two models (jev, microsoft) on the same 30 calls and the same designs, so the explanations can be compared across models.

**Questions for you:**
- Is the main design ("tool name fixed") the right baseline, or should the second view be the main one?
- How should we handle the possible label leakage in `description`?
