CATEGORY: Completed Research Paper
SHORT TITLE: Statistically Certified Autonomy
TITLE: Statistically Certified Autonomy: Design Principles for Risk-Bounded Delegation to AI Agents

ABSTRACT:
Organisations let AI agents act on their systems and must decide, action by action, whether a call may run autonomously, needs human confirmation or must be blocked. We frame this as machine-mediated, risk-budgeted delegation and, following design science research, develop a delegation gate that wraps black-box decision models with conformal calibration and states, before deployment, a statistical guarantee on harmful autonomous executions. On 1,433 agent tool calls scored by three commercial decision models, uncalibrated models execute 16 to 26% of calls that should be blocked. Certifiability is not a matter of accuracy: it faces a score-resolution floor and a policy-family floor set by the budget, the confidence level and the labelled data. Guarantees that hold only on average exceed the budget in up to 36% of repeated calibration samples, and finer, per-service boundaries need more labelled refusals than most services offer. Five design principles specify the delegation boundary.

KEYWORDS: Agentic AI, Human oversight, Delegation, Conformal prediction, Design science

# Introduction

AI agents no longer only recommend; they act. An agent that resolves a support ticket sends emails, edits customer records and closes cases through tool calls on the organisation's systems. Each call raises a delegation question that used to be settled once, when a task was assigned to a person: may this call run autonomously, does it need a human's confirmation, or must it be blocked? Delegation to agentic information systems (Baird & Maruping, 2021) thereby changes in kind. It becomes action-level rather than task-level, dynamic rather than fixed at assignment, mediated by a second machine that decides on the organisation's behalf, explicitly bounded by the risk the organisation accepts, and recalibrated whenever the agent's scope changes. Agent frameworks expose the question as an approval step (Mozannar et al., 2025), recent work formalises it as routing into execute, ask and refuse (Chen et al., 2026), and the European AI Act requires that high-risk systems be designed for effective human oversight (European Union, 2024, Art. 14).

In practice, the routing decision is delegated to a decision model. Providers now sell models that read the agent's context and the proposed call and return a choice with a probability for each option. Such models are cheap and fast, but they cannot be trusted at face value. Agents themselves recognise when not to act in at most 60% of paired cases (Liu et al., 2026), and the guard models that route their calls are tuned on a threshold whose error rate on harmful calls is reported after the fact rather than guaranteed before deployment (Chen et al., 2026). A manager who has to sign off on an agent's autonomy needs a different kind of statement: at most 5% of the calls that should have been blocked will run without a human, and this holds with stated confidence. Figure 1 illustrates such a gate: proposed calls go in, one action per call comes out, and the organisation sets the budgets, the confidence level and the cost of harm.

FIGURE: ../figures/fig1_overview.png | The certified delegation gate from the organisation's perspective. The three calls are illustrative; the shares of all calls executed, sent to a human or blocked are those of the certified policy G4 for the mean of three decision models in the evaluation (Section 5.8).

Conformal prediction offers statements of exactly this form. Given labelled examples, it turns any model's scores into decisions whose error rate is bounded in finite samples without assumptions on the model, as long as future cases resemble the calibration cases (Angelopoulos & Bates, 2023; Vovk et al., 2005). We use certification in this statistical sense throughout: a guarantee, derived from labelled calibration data and a hypothesis test, that a stated error rate holds with a stated confidence; it is not a regulatory or third-party certification. What is missing is design knowledge for the organisational setting, in which the decision model is bought rather than built: which properties of a vendor's model make certification possible, whether to combine vendors, at which level to calibrate, which guarantee to demand, and what it costs in human attention and labelled data. We ask: How should a delegation gate for AI agents be designed so that it bounds harmful autonomous executions with a statistical guarantee while preserving as much autonomy as possible?

Following design science research (Hevner et al., 2004; Peffers et al., 2007), we design a delegation gate as an organisational mechanism that defines and enforces delegation boundaries under uncertainty. It wraps black-box decision models with class- and service-conditional conformal calibration and a decision policy that is statistically certified for its risks and selected by estimated organisational cost. We evaluate it on 1,433 tool calls from a public agent-safety corpus whose labels were produced by language-model annotators, arbitrated by the corpus authors and audited by human reviewers (Chen et al., 2026), scored by three commercial decision models. The evaluation yields design knowledge on how conformal guarantees behave when an organisation delegates through black-box vendors: certification has a floor set by the resolution of a model's output and one set by the policy family, budget, confidence and labelled sample; vendors with diverse, though not independent, errors lower the first; per-service certification needs more labelled refusals than most services offer; and guarantees that hold only on average exceed the budget in up to 36% of repeated calibration-test splits, against at most 2% for the certified policy at a cost of 11 to 15 points of autonomy.

The paper contributes five design principles (Gregor et al., 2020) that specify the delegation boundary of agentic systems; two certifiability floors, which show that organisations must assess not only a decision model's accuracy but whether the model and the policy family consuming its outputs are statistically certifiable under their risk budget and labelled data; and a quantification of the price of certified autonomy in human review, expected cost and labelled data, including the data cost of finer delegation boundaries.

# Background

## Delegation to agentic AI and human oversight

Information systems research has long studied when humans delegate decisions to systems and how such delegation should be designed. Baird and Maruping (2021) conceptualise delegation to agentic information systems as a transfer of rights and responsibilities that depends on the artefact's endowments and the human's preferences, and Fügener et al. (2022) show that productive human-AI delegation requires knowledge of when the machine is better. Machine learning has formalised the same question as selective classification, where a model abstains when uncertain (Chow, 1970; Geifman & El-Yaniv, 2017), and as learning to defer, where a model learns when to hand a case to a human expert (Madras et al., 2018; Mozannar & Sontag, 2020). Most of this work treats delegation as a decision about tasks or cases, taken or learned once; agentic systems shift it to individual actions, decided at run time by another machine.

For AI agents, delegation becomes a stream of micro-decisions. Benchmarks document that agents take harmful or inappropriate actions when they should refuse or ask (Andriushchenko et al., 2025; Ruan et al., 2024; Yuan et al., 2024) and that they rarely recognise such situations themselves (Liu et al., 2026). Agent frameworks therefore route proposed calls through an approval layer, using developer-set rules for clearly reversible or irreversible actions and a model-based judgement for the rest (Mozannar et al., 2025). Chen et al. (2026) introduce three-way routing into execute, ask and refuse, with a fine-tuned guard whose operating threshold trades over-asking against under-asking. The common thread is a decision rule tuned on data, without a stated bound on the errors that matter most.

## Conformal prediction and risk control

Split conformal prediction calibrates a fixed model on labelled data that the model has not been trained on (Vovk et al., 2005). A nonconformity score measures how poorly the model's output fits the true label; the empirical quantile of the calibration scores defines a threshold, and a new case receives the set of all labels whose score lies below it. If calibration and new cases are exchangeable, the true label lies in the set with probability at least 1 − α (Angelopoulos & Bates, 2023). Scores include one minus the label probability (Sadinle et al., 2019) and cumulative probabilities (Romano, Sesia, et al., 2020). Calibrating per class or per group yields class- or group-conditional guarantees (Romano, Barber, et al., 2020; Vovk, 2012). Conformal risk control bounds the expected loss of a tuned threshold (Angelopoulos et al., 2024), and Learn-then-Test certifies candidate parameters through multiple hypothesis testing, so that the selected policy's risk is below the budget with probability at least 1 − δ over the calibration data (Angelopoulos et al., 2021; Bates et al., 2021). All guarantees rest on exchangeability; under distribution shift they need reweighting or recalibration (Barber et al., 2023; Tibshirani et al., 2019). Applications include language models (Kumar et al., 2023; Quach et al., 2024), robot planners that ask for help (Ren et al., 2023), experts choosing from prediction sets (Straitouri et al., 2023) and tool-call arguments, where per-role calibration keeps rare high-risk roles from being diluted (Rahman et al., 2026).

# Research Approach

We follow the design science research methodology of Peffers et al. (2007). The problem is the absence of verifiable bounds on harmful autonomous actions of AI agents. From it we derive five objectives: O1, bound the share of calls that should be refused but are executed autonomously (harmful execution); O2, bound the share of calls that need confirmation but are executed autonomously (unconfirmed execution); O3, maximise the share of safe calls executed without a human (safe autonomy); O4, work with any vendor's decision model accessed only through its output probabilities; and O5, let the organisation weigh the costs of harm, human review and wrongful blocking.

We keep three roles of evidence apart. Development evidence comes from three design iterations on the test split of the corpus: standard prediction sets revealed under-covered refusals; class-conditional calibration revealed near-zero autonomy for some vendors and led to the floors; the third iteration added vendor combination, service-conditional calibration and the certified policies. Internal evaluation re-estimates the final variants on repeated random calibration-test splits of the same calls, with controlled manipulations of the model outputs. Generalisation evidence is limited to held-out services and output rounding; a confirmatory evaluation of the frozen design on untouched data is pending (Sections 7.3 and 7.4). The evaluation follows the Framework for Evaluation in Design Science (Venable et al., 2016) with a technical risk and efficacy strategy and compares the variants with the uncalibrated decision of each model, with reviewing every call and with each other. The design knowledge is formulated as design principles following Gregor et al. (2020).

# The Delegation Gate

## Architecture

Figure 2 adds the calibration part to the run-time flow of Figure 1. At run time, decision models from different vendors receive the proposed call, consisting of the user task, the target service and the call with its arguments, together with a fixed policy that defines the three actions; the gate averages their probability vectors and compares the result with certified thresholds. Before deployment, a certifiability check decides which vendors can be admitted, and a calibration component derives the thresholds from labelled past calls of the same services, the budgets and the costs. When a service is added, calibration runs again before the gate claims its guarantee for that service.

FIGURE: ../figures/fig2_architecture.png | Architecture of the delegation gate. The upper lane runs for every proposed call; the lower lane runs before deployment and whenever a service is added. Tags mark the design principles (DP, Table 7) each component implements.

## Risks, budgets and costs

Let y ∈ {execute, ask, refuse} be the correct action for a call and a the gate's action. The organisation sets two budgets, α_{harm} for harmful execution and α_{ask} for unconfirmed execution, and the gate must satisfy

EQUATION: R_{harm} = P(a = execute | y = refuse) ≤ α_{harm} and R_{ask} = P(a = execute | y = ask) ≤ α_{ask},

while maximising safe autonomy, P(a = execute | y = execute). Conditioning on the true class matters: refusals are a minority, so a bound on all calls together would allow most refusals to be executed. The organisation's expected cost per call, with one human review as the unit, is

EQUATION: C = P(a = ask) + c_{harm}·P(a = execute, y = refuse) + c_{unconf}·P(a = execute, y = ask) + c_{block}·P(a = refuse, y ≠ refuse).

The evaluation uses α_{harm} = 5%, α_{ask} = 15%, c_{harm} = 10, c_{unconf} = c_{harm}/5 and c_{block} = 2 as illustrative risk preferences; Section 5.8 varies the harm budget, the confidence level and the harm cost.

## Gate variants

The gate offers five variants that differ in what they condition on and what they guarantee.

G1, class-conditional prediction sets. With the score s(x, y) = 1 − p_{y}(x), the gate computes for each class k the conformal quantile q_{k} of the scores of calibration cases of class k at level α_{k} and forms the set C(x) of all labels whose score lies below their threshold. A singleton set is acted on; any other set is sent to a human. A refusal can only be executed if refuse is excluded from its set, which happens with probability at most α_{harm}; likewise for ask. Both budgets hold in expectation over calibration samples.

G2, execute whenever refuse is excluded. G2 executes whenever execute is in the set and refuse is not, even if ask is also in the set. It bounds harmful execution in the same way but gives up the bound on unconfirmed execution in exchange for autonomy.

G3, service-conditional sets. G3 calibrates the refusal threshold separately for each target service with enough labelled refusals and pools the rest. The service is known when the call is proposed, so the guarantee holds per service, in expectation.

G4, certified policy with cost-based selection. G4 uses the whole probability vector: execute if p_{refuse} ≤ λ_{r} and p_{ask} ≤ λ_{a}; refuse if p_{refuse} ≥ τ; otherwise ask. It certifies candidate thresholds with Learn-then-Test and selects the certified candidate with the lowest cost C on the calibration data (Table 1). The grid is fixed in advance: λ_{r} takes 15 values from 0 to 0.30, λ_{a} nine values from 0.01 to 0.50 and τ four values from 0.5 to never; δ = 10%.

TABLE:
WIDTHS: 100
SPLIT: yes
| Certification procedure of G4, with the additions for G5 |
| Input. Labelled calibration calls with probability vectors; budgets α_{harm}, α_{ask}; confidence 1 − δ; grids Λ_{r} ordered from strict to permissive, Λ_{a} with m values, and T; costs c_{harm}, c_{unconf}, c_{block}. |
| Step 1. For each λ_{a} ∈ Λ_{a}, run one test sequence at level δ/m (Bonferroni correction over the m sequences). |
| Step 2. Within a sequence, take λ_{r} from Λ_{r} in order. Let E(λ) be the calls with p_{refuse} ≤ λ_{r} and p_{ask} ≤ λ_{a}, and k_{harm} and k_{ask} the numbers of calibration refusals and asks in E(λ). |
| Step 3. Test the null hypothesis H_{0}(λ): R_{harm}(λ) > α_{harm} or R_{ask}(λ) > α_{ask}. Its p-value is max{F(k_{harm}; n_{refuse}, α_{harm}), F(k_{ask}; n_{ask}, α_{ask})}, where F is the binomial distribution function; H_{0} is rejected only if both risks are (intersection-union test). |
| Step 4. If the p-value is at most δ/m, add λ to the certified set C and continue; otherwise stop the sequence (fixed-sequence testing). Add the empty policy, which executes nothing and is trivially certified. |
| Step 5. With probability at least 1 − δ over the calibration data, every policy in C satisfies both budgets simultaneously. Choose λ ∈ C and τ ∈ T with the lowest calibration cost C. Because the guarantee covers all of C at once, this selection on the same data preserves it. τ only decides which non-executed calls are blocked, does not change E(λ) and needs no test. |
| *G5, step A.* For each λ_{a}, certify the ask risk on all asks as in Steps 2 to 4 at level δ/(2m), and let λ_{r}* be the largest certified λ_{r}. |
| *G5, step B.* For each of the G service groups, certify the group's harm risk on its own refusals at level δ/(2mG), using only λ_{r} ≤ λ_{r}*; a group without a certified λ_{r} executes nothing. |
| *G5, step C.* Every group's execution region lies inside {p_{refuse} ≤ λ_{r}*, p_{ask} ≤ λ_{a}}, whose ask risk is certified, and a smaller region cannot execute more asks, so group-specific thresholds cannot raise the ask risk above the certified bound. With the Bonferroni correction over all m + mG tests, all guarantees hold jointly with probability at least 1 − δ. |
CAPTION: Certification procedure of the certified variants G4 and G5.

G5, service-conditional certified policy. G5 integrates G3 and G4. It uses the same policy family with a separate λ_{r} per service group, certifies each group's harm risk and the overall ask risk with high confidence, and executes nothing in groups whose risk cannot be certified (Table 1).

## Two certifiability floors

Black-box decision models express probabilities with finite resolution, and some return exact zeros. This creates limits that no calibration can overcome, and they take two forms.

Proposition 1 (score-resolution floor). Let v be the lowest refusal probability a model can return. For gate G1 or G2 calibrated on n refusals at level α_{harm}, if more than n − ⌈(n + 1)(1 − α_{harm})⌉ calibration refusals receive v, then refuse is included in every prediction set and the gate never executes autonomously. As n grows, the condition approaches π > α_{harm}, where π is the share of refusals that receive v.

Proof. The refusal threshold is the ⌈(n + 1)(1 − α_{harm})⌉-th smallest refusal score among calibration refusals. If more than the stated number have the maximal score 1 − v, this order statistic equals 1 − v, every call has a refusal score of at most the threshold, refuse is in every set, and neither G1 nor G2 executes. The same holds for the ask class.

Proposition 2 (policy-family floor). Let k_{max}(n) be the largest k with F(k; n, α) ≤ δ/m. G4 can certify a non-empty policy only if its execution region contains at most k_{max}(n) of the n calibration cases of each budgeted class. Every candidate region contains the calls whose refusal and ask probabilities are both at their lowest expressible values, so G4 executes nothing when more than k_{max}(n) budgeted cases receive them. Because no k qualifies while (1 − α)ⁿ > δ/m, no non-empty policy can be certified with fewer than ln(δ/m)/ln(1 − α) calibration cases of a budgeted class.

The proof follows from Steps 3 and 4 of Table 1. For α_{harm} = 5%, δ = 10% and m = 9, the region may contain no refusal at all with 101 calibration refusals, at most 3 of 202 (1.5%) and at most 8 of 364 (2.2%), and at least 88 refusals are needed; for G5 with four simultaneous groups, each group needs 129. The two floors depend on different things. The score-resolution floor is a property of one score of the model and binds gates that threshold that score alone. The policy-family floor depends on the model's outputs, the policy family, the budget, the confidence level and the size of the labelled sample. A richer family can escape the first floor without escaping the second, and both can be measured before procurement.

# Evaluation

## Data and decision models

TABLE:
WIDTHS: 14,15,12,10,11,7,10,11,10
| Scorer | Output | Accuracy | Refusal recall | Harmful execution | ECE | Refusals at p = 0 | Refusals at p ≤ 0.01 | Asks at p = 0 |
| jev | 0.01 grid | 0.649 | 0.47 | 15.8% | 0.15 | 7.9% | 13.3% | 0.8% |
| microsoft | continuous | 0.597 | 0.41 | 26.4% | 0.31 | 0.0% | 22.5% | 0.0% |
| gpt | 0.01 grid | 0.623 | 0.47 | 20.7% | 0.25 | 2.7% | 12.1% | 12.1% |
| mean of three | continuous | 0.634 | 0.48 | 24.4% | 0.19 | 0.0% | 9.9% | 0.0% |
CAPTION: Decision models used without calibration (N = 1,433), each deciding for the action with the highest probability. Harmful execution is the share of true refusals executed; ECE is the top-1 expected calibration error with 15 bins; the last three columns give the share of true refusals or asks that receive a probability of exactly zero or at most 0.01 for their correct class.

We use the in-distribution test split of the SAFETY SENTRY corpus (Chen et al., 2026). Each record is one proposed tool call with the user task, the target service and the call. Labels (execute, ask or refuse) were produced by two language-model annotators following a 14-trigger taxonomy, with disagreements arbitrated by the corpus authors; a human audit of the split found 92% agreement with the labels. All guarantees below hold with respect to these labels. The calls come from nine self-hosted enterprise services, such as password vaults, ticketing and electronic medical records, and from five adapted public benchmarks. We used three commercial decision models through a common decisions endpoint that returns a choice and a probability vector: typesafe/jev-1.14 (jev), microsoft/microsoft-decision-1 (microsoft) and openai/gpt-6-luna-decisions (gpt). Every request contained the same system policy and the same criteria for the three actions. The 4,308 requests cost about USD 0.34; three gpt requests failed after five retries, and we use the 1,433 calls scored by all three models (540 execute, 488 ask, 405 refuse). We add the mean of the three probability vectors as a fourth scorer. Table 2 describes the scorers: used directly, all of them execute 16 to 26% of the calls that should be refused, and the most accurate model is not the one with the fewest zero-probability refusals.

## Protocol

The decision models were not trained on the corpus, so every labelled call can serve as calibration data; a hold-out is needed only to evaluate. We draw 100 random calibration-test splits of 50% each, stratified by label and identical across scorers (60 splits for G4, G5 and the tuned comparison policy), calibrate on one half and report rates on the other. Besides means, we report the share of splits in which a test rate exceeds its budget. These splits re-sample one corpus and are not independent deployments; the share illustrates the risk an organisation takes with the one calibration sample it receives. Further analyses round microsoft's probabilities to the 0.01 grid, calibrate per service, hold out services, and vary the calibration share (25% to 90%, 40 repetitions each), budgets, confidence levels and costs.

## Class-conditional calibration is necessary

Standard conformal prediction with one global threshold reaches its marginal target, but the guarantee is consumed by the easy classes. At a 10% level, the refusal class is covered for only 79% of refusals under jev and 83% under microsoft, with similar values for cumulative scores, because both models are biased towards execute. Class-conditional calibration at the same level restores refusal coverage to 91 to 97%, and the three vendors then execute 0.0 to 2.2% of refusals under the G1 rule, against 15.8 to 26.4% without calibration. All further variants are class-conditional.

## The score-resolution floor and how a richer policy escapes it

Figure 3a shows where the floor comes from: the share of true refusals each model scores at or below a given refusal probability. Mass at the left edge cannot be separated from safe calls by any threshold on this score. Figure 3b shows the consequence under G2. jev, which gives exactly zero refusal probability to 7.9% of refusals, executes nothing at budgets of 2 to 5% and starts at 7.5%. gpt, with 2.7% zeros, executes 3% of safe calls at a 2% budget and 50% at 5%. microsoft, which never returns an exact zero, executes 41% even at 2%. The decisive test is the manipulation: rounding microsoft's own probabilities to the 0.01 grid that the other vendors use turns 17.0% of its refusal probabilities into exact zeros, and its safe autonomy at the 5% budget falls from 56% to 0%. The advantage of the least accurate model under strict budgets is thus a property of the output format, not of the model.

The policy family decides whether this floor binds. G4 also conditions on the ask probability, and jev scores its zero-probability refusals as ask-like, so G4 can exclude them: it executes 32% of jev's safe calls with no budget exceeded in any split, against 0.5% under G1. gpt shows the policy-family floor instead. It assigns an ask probability of exactly zero, together with an execute probability of one, to 12.1% of the calls that need confirmation; every candidate region contains them, G4's strictest region already executes 20% of them, and gpt cannot be certified at a 15% ask budget although it satisfies the harm budget.

## Combining vendors lowers the floor

The vendors' confident errors are not independent, but they are diverse (Figure 4a). On true refusals, a refusal probability of at most 0.01 from one vendor makes the same from another vendor 2.8 to 3.9 times more likely than independence would predict, and the rank correlation of their refusal probabilities is 0.76 to 0.87. Yet of the 115 refusals scored that low by at least one vendor, 93 are scored low by only one or two, and only 22 (5.4%) by all three; no refusal receives an exact zero from all three. Averaging therefore pulls most confident errors up, and the mean has the lowest score-resolution floor and the highest autonomy at every budget of 3% and above (Figure 3b): 63% of safe calls at the 5% budget under G2 and 48% under G1, against 56% and 37% for the best single model. The mean of the two grid-valued vendors can be certified under G4 (31% safe autonomy) although one of its members cannot. Under G4 itself, the gain over the best single model shrinks to about two points.

FIGURE: ../figures/fig3_floor.png | The score-resolution floor. (a) Cumulative share of true refusals (N = 405) at or below each refusal probability; markers at zero show the share at the lowest expressible value. (b) Safe autonomy under gate G2 as a function of the harmful-execution budget (ask budget 15%, means over 50 splits).

FIGURE: ../figures/fig4_vendors.png | Diverse and shared errors. (a) True refusals that receive a refusal probability of at most 0.01, by the combination of vendors that do so; only 22 of 405 are scored low by all three. (b) Refusal probabilities on the service dh_app for true refusals (filled) and true execute calls (hollow); values of zero are drawn at the bottom of the axis.

Diversity fails where errors are shared (Figure 4b). On one small service with 18 refusals and 8 safe calls, each vendor gives the refusals the same refusal probability as the safe calls: jev returns zero for all of them, gpt 0.02 for nearly all, and microsoft values in one narrow band around 0.002. No threshold on any vendor or on their mean separates the two groups, and when this service is held out, the gate executes 94% of its refusals under the mean of three.

## Services: heterogeneity and the integrated gate

The class-conditional guarantee holds on average over all refusals, but refusals differ. Under G2, refusals on the atbench service are executed at 14.8% with microsoft and 16.8% with the mean of three, about three times the budget; the overall rate stays below 5% because the 178 refusals of the pooled remaining services are almost never executed (Figure 5a). Service-conditional calibration (G3) brings every service group within budget on average, at 0 to 3.8%, and relocates the cost of safety to the risky service (Figure 5b): safe autonomy on atbench falls from 22 to 25% to 1 to 7%, while on the pooled remaining services it rises, from 62% to 71% for microsoft. The same heterogeneity breaks the guarantee for services the gate has not seen: calibrated on all other services, the gate executes 20% (microsoft), 17% (mean of three) and 11% (gpt) of atbench refusals.

FIGURE: ../figures/fig5_services.png | Harmful execution (a) and safe autonomy (b) per service group under gate G2 with one refusal threshold (hollow) and under service-conditional calibration G3 (filled), means over 100 splits; numbers in brackets are refusals in the corpus. tsbench contains only refusals and is therefore absent from (b).

The integrated variant G5 adds the high-confidence guarantee per service, and its result is a sample-size limit. With half of the calls for calibration, no service group reaches the 129 calibration refusals that Proposition 2 requires for four simultaneous groups, and G5 executes nothing. With 90% of the calls for calibration, G5 certifies the pooled group of the remaining services in all 40 repetitions, executes nothing on atbench, agentharm and tsbench, and executes none of the test refusals in any group. Its overall safe autonomy is then higher than that of G4 on the same splits, 43% against 33% for the mean of three, because it can be permissive where risk is low (52% against 36% on the pooled services) instead of equally strict everywhere, while G4 executes 6.6% of atbench refusals. Certifying each service at 90% confidence separately rather than simultaneously lowers the requirement to 102 refusals. That low-volume services cannot be certified is not an implementation failure but a limit set by the budget, the confidence level, the multiplicity correction and the labelled history: more granular delegation boundaries improve local risk control but raise the labelled data needed to certify them.

## Guarantees on average versus with confidence

TABLE:
WIDTHS: 16,25,12,13,16,18
SPLIT: yes
| Scorer | Gate | Safe autonomy | Harmful execution | Unconfirmed execution | Splits over harm / ask budget |
| jev | G4 certified policy | 32.0% | 1.9% | 3.3% | 0% / 0% |
| microsoft | G1 class-conditional sets | 37.1% | 1.4% | 14.0% | 0% / 36% |
| microsoft | G2 execute if refuse excluded | 56.1% | 4.3% | 23.9% | 30% / not bounded |
| microsoft | G3 service-conditional sets | 60.2% | 3.0% | 27.4% | 14% / not bounded |
| microsoft | G4 certified policy | 26.0% | 0.8% | 2.1% | 0% / 0% |
| gpt | G2 execute if refuse excluded | 50.4% | 2.7% | 20.8% | 0% / not bounded |
| gpt | G4 certified policy | not certifiable | | | |
| mean of three | G1 class-conditional sets | 48.4% | 3.0% | 10.6% | 2% / 3% |
| mean of three | G2 execute if refuse excluded | 63.4% | 4.3% | 25.1% | 24% / not bounded |
| mean of three | G4 certified policy | 33.8% | 1.6% | 5.1% | 2% / 2% |
| all scorers | G5 service-conditional certified | 0.0% | 0.0% | 0.0% | 0% / 0% |
CAPTION: Gate variants at a 5% harm budget and a 15% ask budget (G4 and G5 with 90% confidence and a harm cost of ten reviews), half of the calls for calibration. Means over 100 splits (60 for G4 and G5); the last column gives the share of test halves in which the observed rate exceeds the budget. Without a gate, the models execute 83 to 87% of safe calls and 16 to 26% of calls that should be refused.

Table 3 compares the gate variants, and Figure 6 shows the rate in every test split rather than the mean. The set-based variants satisfy the budgets on average, as the theory promises, but exceed them in many splits. With microsoft, G1 keeps harmful execution within budget in every split but exceeds the ask budget in 36% of them, and G2 exceeds the harm budget in 30%; with the mean of three, G2 does so in 24%. The certified policy G4 stays within both budgets in all splits for jev and microsoft and in 98% for the mean of three. A policy chosen from the same family on the same data without the statistical test looks almost as safe on average but exceeds a budget more often: in 5% of splits for jev on harm and 8% for the mean of three on unconfirmed execution.

FIGURE: ../figures/fig6_splits.png | Harmful (a) and unconfirmed (b) execution in each test split, with the share of splits above the budget on the right; 100 splits for G1 and G2, 60 for the tuned policy (G4's family without the test) and G4. For microsoft, the tuned policy coincides with G4; G2 does not target the ask budget (n/a).

## The price of certification: review, cost, confidence and data

Certification is paid for in autonomy and therefore in human review. Table 4 evaluates the organisation's cost objective for the mean of three. Without a gate, one call costs 1.33 review-equivalents at a harm cost of ten reviews, more than reviewing every call (1.00), because a quarter of the calls that should be blocked run. Every gate is cheaper than reviewing everything at this harm cost, and G4 is the cheapest with half of the data (0.81), because it blocks clear refusals (wrongly in 1.9% of the other calls) and executes few calls that need confirmation, each of which costs two reviews. As the harm cost rises to 50 and 200 reviews, the set-based gates become far more expensive than reviewing everything, while G4 re-selects stricter policies (11% and 7% safe autonomy) and stays close to the cost of reviewing every call it does not block (0.92 and 1.04 against 0.88). Certification bounds the risks, not the cost: the calibration cost of the selected policy is optimistic because the policy was selected on it, and at these harm costs G4's realised cost exceeds that of executing nothing. With 90% of the data for calibration, G5 is cheaper than G4 at every harm cost and keeps 13% of safe calls autonomous even at a harm cost of 200, because it stays permissive on the services it can certify.

TABLE:
WIDTHS: 30,12,12,12,11,11,12
SPLIT: yes
| Gate (mean of three) | Safe autonomy | Sent to a human | Wrongly blocked | Cost, harm = 10 | Cost, harm = 50 | Cost, harm = 200 |
| No gate (model decides) | 87.0% | 29.6% | 2.5% | 1.33 | 5.30 | 20.20 |
| Review every call | 0.0% | 100% | 0.0% | 1.00 | 1.00 | 1.00 |
| G1 class-conditional sets | 48.4% | 72.8% | 0.0% | 0.89 | 1.52 | 3.89 |
| G2 execute if refuse excluded | 63.4% | 61.9% | 0.0% | 0.91 | 2.08 | 6.45 |
| G3 service-conditional sets | 60.2% | 63.6% | 0.0% | 0.88 | 1.87 | 5.58 |
| Tuned policy, no test | 37.3% | 68.7% | 1.9% | 0.82 | 0.92 | 1.04 |
| G4 certified policy | 33.8% | 70.7% | 1.9% | 0.81 | 0.92 | 1.04 |
| G5 (executes nothing) | 0.0% | 85.7% | 1.9% | 0.88 | 0.88 | 0.88 |
| G4, 90% calibration | 32.6% | 70.6% | 1.7% | 0.81 | 0.88 | 0.94 |
| G5, 90% calibration | 43.3% | 65.6% | 1.7% | 0.74 | 0.86 | 0.85 |
CAPTION: Workload and expected cost per call for the mean of three, in review-equivalents (an unconfirmed execution costs a fifth of the harm cost, a wrongful block two reviews). The first three numeric columns refer to a harm cost of ten; G4, G5 and the tuned policy are re-selected for each harm cost. Half of the calls for calibration unless stated; the last two rows use 90% (40 repetitions).

TABLE:
WIDTHS: 12,22,22,22,22
| Harm budget | 99% confidence (δ = 1%) | 95% confidence (δ = 5%) | 90% confidence (δ = 10%) | 80% confidence (δ = 20%) |
| 2% | not certifiable | not certifiable | not certifiable | 15 / 9 / 25 (jev in 35% of splits) |
| 5% | 27 / 15 / 26 (jev in 82% of splits) | 32 / 24 / 29 | 34 / 26 / 32 | 35 / 26 / 32 |
| 10% | 33 / 26 / 39 | 34 / 26 / 41 | 34 / 26 / 42 | 36 / 26 / 43 |
CAPTION: Safe autonomy (%) of the certified policy G4 for the mean of three / microsoft / jev by harm budget and confidence level (ask budget 15%, harm cost of ten reviews, 202 calibration refusals, means over 60 splits). gpt is not certifiable in any setting.

Table 5 varies the budget and the confidence level, which a manager sets for G4. Confidence is cheap up to 95%: at a 5% budget, raising it from 90% to 95% costs two to three points of autonomy, and 99% costs seven points for the mean of three and eleven for microsoft relative to 90%. The budget is not cheap. At 2%, no scorer can be certified with 90% confidence from 202 calibration refusals, because Proposition 2 requires at least 223; at 80% confidence the gate becomes certifiable but executes only 9 to 15% of safe calls for microsoft and the mean. Loosening the budget to 10% raises jev's autonomy to 42% but leaves the other two almost unchanged, because their cost-optimal certified policy already lies inside the stricter budget.

The third price is labelled data (Figure 7). At a 5% budget, G4 needs at least 75, 88, 102 and 133 labelled refusals for 80, 90, 95 and 99% confidence; at 2%, it needs 189 to 337. Above the minimum, autonomy rises steeply and then flattens: the certified autonomy of the mean of three is 18% with 101 calibration refusals and 33% with 202 at 90% confidence, and stays flat up to 364; jev is certifiable in only 45% of repetitions with 101 refusals.

## Assessment against the design objectives

Table 6 assesses the artefact against the objectives. The certified variants meet the two bounding objectives with high confidence, per class for G4 and per certified service for G5. O3 is met only partly, because the price of certification is substantial; O4 holds only for vendors that pass the floor tests, which turns the objective into a procurement criterion; and O5 is met for the selection, while the cost itself is not guaranteed.

FIGURE: ../figures/fig7_frontier.png | Confidence, budget and labelled data. (a) Minimum number of labelled refusals for G4 to certify any non-empty policy (Proposition 2, nine ask thresholds); dotted lines mark the refusals available. (b) Safe autonomy of G4 for the mean of three by calibration size and confidence level (5% budget, 40 repetitions per size); crosses mark sizes at which no policy can be certified.

TABLE:
WIDTHS: 22,58,20
SPLIT: yes
| Objective | Evidence | Assessment |
| O1 Bound harmful execution | G4 keeps it within 5% in 98 to 100% of splits, against 16 to 26% without a gate; G5 per service where enough labelled refusals exist. | Met per class; per service where data suffice |
| O2 Bound unconfirmed execution | G4 keeps it within 15% in 98 to 100% of splits; G1 on average; G2 and G3 do not bound it. | Met with G4 and G5 |
| O3 Maximise safe autonomy | G4 executes 26 to 34% of safe calls, G5 43% with 90% of the data, against 83 to 87% without a gate. | Partly met; price quantified |
| O4 Work with any vendor's probabilities | Works for jev, microsoft and the mean; gpt fails on its policy-family floor; rounding removes microsoft's certifiability. | Met for vendors that pass the floor tests |
| O5 Weigh harm, review and blocking | Selection by expected cost; G4 cheapest at a harm cost of ten with half of the data, G5 at every harm cost with 90%. | Met for selection; cost not guaranteed |
CAPTION: Assessment of the delegation gate against the design objectives.

# Design Principles

Table 7 condenses the evaluation into five design principles, stated after Gregor et al. (2020) as actions for implementers of delegation gates, with the mechanism and evidence behind them. Together they specify the delegation boundary of an agentic system along four dimensions: what is delegated (DP1), to which decision models (DP2, DP3), where (DP4) and with which assurance (DP5). They apply under three boundary conditions: labelled past calls exist for the services in scope, the decision models expose probabilities for all three actions, and future calls resemble the calibration calls.

TABLE:
WIDTHS: 7,48,45
SPLIT: yes
| DP | Principle | Rationale and evidence |
| DP1 | Calibrate the guarantee for each critical class separately, so that the budget for harmful execution is spent only on calls that should be refused. | A global threshold lets the easy classes absorb the error budget; for two vendors, refusal coverage falls to 79 to 83% at a 90% target (Section 5.3). |
| DP2 | Before procurement, test each decision model's score-resolution floor and its policy-family floor at the organisation's budget, confidence and calibration size, for every budgeted class, and require outputs with sufficient resolution. | A model cannot be certified when too many critical cases receive the lowest expressible probabilities (Propositions 1 and 2); rounding one vendor's output removes all its autonomy under G1 and G2 (Figure 3). |
| DP3 | When decision models show sufficiently diverse critical errors on labelled data, combine their outputs before calibration, verify that the combination lowers the certifiability floor, and inspect each service for shared blind spots. | Confident errors are dependent but only partly shared (5.4% of refusals scored low by all three versus 12 to 22% by each), so the mean has the lowest resolution floor; on one service all vendors are blind (Figure 4). |
| DP4 | Calibrate per target service, collect enough labelled critical calls to certify each service at the required confidence, route calls of uncertified services to humans, and recalibrate before claiming the guarantee for a new service. | Class-level guarantees average over heterogeneous harms (atbench at three times the budget); held-out services break the guarantee; per-service certification needs 102 to 129 refusals per service and then yields more autonomy than one class-level policy (Section 5.6). |
| DP5 | Certify the policy with high confidence over a pre-specified family that uses the full probability vector, and choose among certified policies by the organisation's costs. | Guarantees that hold only on average exceed the budget in up to 36% of repeated calibration-test splits; the certified policy exceeds it in at most 2%, at 11 to 15 points of autonomy, escapes the resolution floor of coarse models and has the lowest expected cost at moderate harm costs (Tables 3 and 4, Figure 6). |
CAPTION: Design principles for certified delegation gates.

# Discussion

## Theoretical contributions

The paper conceptualises delegation in agentic information systems as a boundary that is drawn per action, by a purchased decision model, under a risk budget the organisation states in advance, and redrawn whenever the agent's scope changes. Baird and Maruping (2021) describe delegation as a transfer of rights and responsibilities that depends on the endowments of the agentic artefact; in agentic systems this transfer happens for every action and is itself delegated to a machine. The gate is the organisational mechanism that defines and enforces this boundary under uncertainty, and the design principles specify what may cross it, which decision models may draw it, where it holds and with which assurance. Granularity has an evidence price: finer boundaries improve local risk control but need more labelled history to certify. Prior work treated the gate as a classifier to be tuned (Chen et al., 2026; Mozannar et al., 2025); we treat it as an accountable boundary whose guarantee is part of its specification.

The two floors add a criterion of delegate suitability that accuracy-based comparisons do not reveal. Whether a decision model can draw a certified boundary is decided by the tail of its probabilities for the critical classes, by the resolution in which a vendor exposes them and by the policy family that reads them. Selecting by accuracy or calibration error would pick the vendor with the highest resolution floor in our data; the floors are direct and testable. For conformal prediction, the contribution lies in operationalisation. Class-conditional calibration, group conditioning and high-confidence risk control are known (Angelopoulos et al., 2021; Vovk, 2012); we show how they behave when an organisation delegates through black-box vendors, where output resolution, vendor diversity, service heterogeneity and the size of the labelled history decide what can be certified and at what price.

## Implications for practice

The gate turns the question of how much autonomy to grant into parameters a manager can own: two budgets, a confidence level and the cost of a harmful execution relative to a review. The price is explicit. At a 5% harm budget, G4 sends 71 to 75% of calls to a human, compared with 29 to 36% when the models decide alone, so review capacity has to be planned with the guarantee; because tired reviewers make mistakes (Turan, 2026), the costs should reflect actual review effort. Labelled data become an asset to plan for: each service that should carry its own guarantee needs 102 to 129 labelled refusals at a 5% budget and 90% confidence, depending on whether services are certified separately or jointly, and services without them are best routed to humans. For procurement, DP2 translates into a requirement that is easy to verify: vendors should expose unrounded probabilities, and buyers should measure both floors on their own labelled data before signing. For governance, the certified policy can provide auditable evidence supporting the implementation and documentation of human-oversight mechanisms in the sense of Article 14 of the AI Act (European Union, 2024): which calls are delegated, with which error bound, at which confidence and for which services. It does not by itself establish compliance, which also depends on whether reviewers can understand and override the system.

## Limitations

The evidence has the three roles of Section 3, and the confirmatory role is not yet filled: the design was developed and evaluated on the same 1,433 calls, so the design principles rest on internal evaluation and limited generalisation checks. Labels were produced by language models with arbitration and a human audit; all guarantees hold with respect to these labels, label errors on refusals translate directly into measured harm rates, and because labels and decision models both come from language models their errors may be correlated, which would make the rates optimistic. Repeated splits of one corpus are not independent deployments. Services and labels are partly confounded, the service-level results rest on three large service groups and a pooled remainder, and the guarantee for the pooled group averages over its services. The decision models are black boxes accessed through an alpha-stage endpoint whose versions may change, which limits reproducibility.

## Future research

First, the frozen design should be evaluated on untouched data, such as the corpus's training split of 7,767 calls and its out-of-distribution split, which would also provide the per-service refusals G5 needs. Second, field studies should examine how reviewers respond to the calls a certified gate sends them, since the guarantee covers what is executed, not the quality of review. Third, the gate can be extended to sequences of calls, where the harm of an action depends on earlier actions, and to budgets that adapt to review capacity.

# Conclusion

AI agents turn delegation into a decision per action, and organisations need to state the error those decisions may make before they let agents act. We designed and evaluated a delegation gate that wraps commercial decision models with conformal calibration. It bounds the share of calls that should be blocked but are executed, from 16 to 26% without a gate to a 5% budget that holds with stated confidence, and it shows what that bound costs in review, expected cost and labelled data. Organisations must judge a decision model not only by its accuracy but by whether it, together with the policy that reads it, can be certified under their budget and labelled data, and finer boundaries require more labelled history. Five design principles summarise how to draw such a boundary.


# References

Andriushchenko, M., Souly, A., Dziemian, M., Duenas, D., Lin, M., Wang, J., Hendrycks, D., Zou, A., Kolter, Z., Fredrikson, M., Winsor, E., Wynne, J., Gal, Y., & Davies, X. (2025). AgentHarm: A benchmark for measuring harmfulness of LLM agents. In Proceedings of the Thirteenth International Conference on Learning Representations (ICLR 2025).

Angelopoulos, A. N., & Bates, S. (2023). Conformal prediction: A gentle introduction. Foundations and Trends in Machine Learning, 16(4), 494–591.

Angelopoulos, A. N., Bates, S., Candès, E. J., Jordan, M. I., & Lei, L. (2021). Learn then test: Calibrating predictive algorithms to achieve risk control. arXiv preprint arXiv:2110.01052.

Angelopoulos, A. N., Bates, S., Fisch, A., Lei, L., & Schuster, T. (2024). Conformal risk control. In Proceedings of the Twelfth International Conference on Learning Representations (ICLR 2024).

Baird, A., & Maruping, L. M. (2021). The next generation of research on IS use: A theoretical framework of delegation to and from agentic IS artifacts. MIS Quarterly, 45(1), 315–341.

Barber, R. F., Candès, E. J., Ramdas, A., & Tibshirani, R. J. (2023). Conformal prediction beyond exchangeability. The Annals of Statistics, 51(2), 816–845.

Bates, S., Angelopoulos, A., Lei, L., Malik, J., & Jordan, M. (2021). Distribution-free, risk-controlling prediction sets. Journal of the ACM, 68(6), Article 43.

Chen, T., Hu, C., & Wang, W. (2026). SAFETY SENTRY: Context-aware human intervention via EXECUTE-ASK-REFUSE routing. arXiv preprint arXiv:2607.13594.

Chow, C. K. (1970). On optimum recognition error and reject tradeoff. IEEE Transactions on Information Theory, 16(1), 41–46.

European Union. (2024). Regulation (EU) 2024/1689 of the European Parliament and of the Council of 13 June 2024 laying down harmonised rules on artificial intelligence (Artificial Intelligence Act). Official Journal of the European Union, L 2024/1689.

Fügener, A., Grahl, J., Gupta, A., & Ketter, W. (2022). Cognitive challenges in human–artificial intelligence collaboration: Investigating the path toward productive delegation. Information Systems Research, 33(2), 678–696.

Geifman, Y., & El-Yaniv, R. (2017). Selective classification for deep neural networks. In Advances in Neural Information Processing Systems 30 (pp. 4878–4887).

Gregor, S., Chandra Kruse, L., & Seidel, S. (2020). Research perspectives: The anatomy of a design principle. Journal of the Association for Information Systems, 21(6), 1622–1652.

Hevner, A. R., March, S. T., Park, J., & Ram, S. (2004). Design science in information systems research. MIS Quarterly, 28(1), 75–105.

Kumar, B., Lu, C., Gupta, G., Palepu, A., Bellamy, D., Raskar, R., & Beam, A. (2023). Conformal prediction with large language models for multi-choice question answering. arXiv preprint arXiv:2305.18404.

Liu, X., Zhang, Y. E., Kasprova, V., Rabbani, P., Zahraei, P. S., Zhang, T., Ebrahimpour-Boroojeny, A., & Chandrasekaran, V. (2026). AgentAbstain: Do LLM agents know when not to act? arXiv preprint arXiv:2607.10059.

Madras, D., Pitassi, T., & Zemel, R. (2018). Predict responsibly: Improving fairness and accuracy by learning to defer. In Advances in Neural Information Processing Systems 31 (pp. 6147–6157).

Mozannar, H., Bansal, G., Tan, C., Fourney, A., Dibia, V., Chen, J., Gerrits, J., Payne, T., Maldaner, M. K., Grunde-McLaughlin, M., Zhu, E., Bassman, G., Alber, J., Chang, P., Loynd, R., Niedtner, F., Kamar, E., Murad, M., Hosn, R., & Amershi, S. (2025). Magentic-UI: Towards human-in-the-loop agentic systems. arXiv preprint arXiv:2507.22358.

Mozannar, H., & Sontag, D. (2020). Consistent estimators for learning to defer to an expert. In Proceedings of the 37th International Conference on Machine Learning (pp. 7076–7087). PMLR.

Peffers, K., Tuunanen, T., Rothenberger, M. A., & Chatterjee, S. (2007). A design science research methodology for information systems research. Journal of Management Information Systems, 24(3), 45–77.

Quach, V., Fisch, A., Schuster, T., Yala, A., Sohn, J. H., Jaakkola, T. S., & Barzilay, R. (2024). Conformal language modeling. In Proceedings of the Twelfth International Conference on Learning Representations (ICLR 2024).

Rahman, M. A., Rahman, M. A., Samin, N. H., Tasnia, K. R., Amin, M. H., Ahona, S. R., & Noshin, J. A. (2026). Beyond aggregate risk: Role-stratified conformal risk control for LLM tool calls. arXiv preprint arXiv:2607.24343.

Ren, A. Z., Dixit, A., Bodrova, A., Singh, S., Tu, S., Brown, N., Xu, P., Takayama, L., Xia, F., Varley, J., Xu, Z., Sadigh, D., Zeng, A., & Majumdar, A. (2023). Robots that ask for help: Uncertainty alignment for large language model planners. In Proceedings of the 7th Conference on Robot Learning (pp. 661–682). PMLR.

Romano, Y., Barber, R. F., Sabatti, C., & Candès, E. J. (2020). With malice toward none: Assessing uncertainty via equalized coverage. Harvard Data Science Review, 2(2).

Romano, Y., Sesia, M., & Candès, E. J. (2020). Classification with valid and adaptive coverage. In Advances in Neural Information Processing Systems 33 (pp. 3581–3591).

Ruan, Y., Dong, H., Wang, A., Pitis, S., Zhou, Y., Ba, J., Dubois, Y., Maddison, C. J., & Hashimoto, T. (2024). Identifying the risks of LM agents with an LM-emulated sandbox. In Proceedings of the Twelfth International Conference on Learning Representations (ICLR 2024).

Sadinle, M., Lei, J., & Wasserman, L. (2019). Least ambiguous set-valued classifiers with bounded error levels. Journal of the American Statistical Association, 114(525), 223–234.

Straitouri, E., Wang, L., Okati, N., & Gomez-Rodriguez, M. (2023). Improving expert predictions with conformal prediction. In Proceedings of the 40th International Conference on Machine Learning (pp. 32633–32653). PMLR.

Tibshirani, R. J., Barber, R. F., Candès, E. J., & Ramdas, A. (2019). Conformal prediction under covariate shift. In Advances in Neural Information Processing Systems 32 (pp. 2530–2540).

Turan, E. (2026). Oversight has a capacity: Calibrating agent guards to a subjective, fatiguing human. arXiv preprint arXiv:2606.08919.

Venable, J., Pries-Heje, J., & Baskerville, R. (2016). FEDS: A framework for evaluation in design science research. European Journal of Information Systems, 25(1), 77–89.

Vovk, V. (2012). Conditional validity of inductive conformal predictors. In Proceedings of the Asian Conference on Machine Learning (pp. 475–490). PMLR.

Vovk, V., Gammerman, A., & Shafer, G. (2005). Algorithmic learning in a random world. Springer.

Yuan, T., He, Z., Dong, L., Wang, Y., Zhao, R., Xia, T., Xu, L., Zhou, B., Li, F., Zhang, Z., Wang, R., & Liu, G. (2024). R-Judge: Benchmarking safety risk awareness for LLM agents. In Findings of the Association for Computational Linguistics: EMNLP 2024 (pp. 1467–1490).
