"""Copy the paper and confirmatory scripts into extra_models/ and add the two extra decision models to their model lists.

The originals in analysis/ and confirmatory/ are not touched (confirmatory/ stays frozen, see verify_frozen.py). Each
change is an exact text replacement that must occur the stated number of times, so a copy fails loudly instead of
silently differing. The mean of three stays the mean of jev, microsoft and gpt; the extra models enter as single scorers.

Usage (from the repository root): python extra_models/make_copies.py
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "extra_models")
OLD3 = '["jev-1.14", "microsoft-decision-1", "gpt-6-luna-decisions"'
NEW5 = '["jev-1.14", "microsoft-decision-1", "gpt-6-luna-decisions", "d1-ultra", "pplx-decider-v1.2"'
M3 = '"gpt": mat("gpt-6-luna-decisions")}'
M5 = '"gpt": mat("gpt-6-luna-decisions"), "liquid": mat("d1-ultra"), "pplx": mat("pplx-decider-v1.2")}'

EDITS = {
    "analysis/02_conformal_core.py": [
        (OLD3, NEW5, 6),
        ('np.mean([MODELS[m] for m in list(MODELS)], axis=0)',
         'np.mean([MODELS[m] for m in ["jev-1.14", "microsoft-decision-1", "gpt-6-luna-decisions"]], axis=0)', 1),
        # vendor co-occurrence of near-zero refusal probabilities: keep the paper's three vendors
        ('both = np.stack([MODELS[m][np.arange(N), y] for m in ' + NEW5 + ']])',
         'both = np.stack([MODELS[m][np.arange(N), y] for m in ' + OLD3 + ']])', 1),
    ],
    "analysis/03_policies_and_groups.py": [
        (OLD3, NEW5, 1),
        ('["microsoft-decision-1", "gpt-6-luna-decisions", "ensemble (mean of 3)"]',
         '["microsoft-decision-1", "gpt-6-luna-decisions", "d1-ultra", "pplx-decider-v1.2", "ensemble (mean of 3)"]', 1),
    ],
    "analysis/04_calibration_size.py": [(OLD3, NEW5, 1)],
    "analysis/05_budget_confidence_sensitivity.py": [
        (OLD3, NEW5, 1),
        ('M["ensemble (mean of 3)"] = np.mean(list(M.values()), 0)',
         'M["ensemble (mean of 3)"] = np.mean([M["jev-1.14"], M["microsoft-decision-1"], M["gpt-6-luna-decisions"]], 0)', 1),
    ],
    "analysis/06_revision_g5_costs_frontier.py": [
        (M3, M5, 1),
        ('for name in ["mean", "microsoft", "jev", "gpt"]:', 'for name in ["mean", "microsoft", "jev", "gpt", "liquid", "pplx"]:', 2),
        ('for name in ["mean", "microsoft", "jev"]:', 'for name in ["mean", "microsoft", "jev", "liquid", "pplx"]:', 2),
        # R5 dependence: all vendor pairs; the "all three" statistic stays on the paper's three vendors
        ('low = {m: M[m][rf, RF] <= 0.01 for m in ["jev", "microsoft", "gpt"]}',
         'low = {m: M[m][rf, RF] <= 0.01 for m in ["jev", "microsoft", "gpt", "liquid", "pplx"]}', 1),
        ('for a_, b_ in [("jev", "microsoft"), ("jev", "gpt"), ("microsoft", "gpt")]:',
         'for a_, b_ in __import__("itertools").combinations(["jev", "microsoft", "gpt", "liquid", "pplx"], 2):', 1),
    ],
    "analysis/07_empty_policy_check.py": [('for name in ["mean", "microsoft", "jev"]:', 'for name in ["mean", "microsoft", "jev", "liquid", "pplx"]:', 1)],
    "analysis/08_g5_costs_90pct.py": [('for name in ["mean", "microsoft"]:', 'for name in ["mean", "microsoft", "liquid", "pplx"]:', 1)],
    "analysis/09_scorer_table.py": [(M3, M5, 1)],
    "confirmatory/check_run.py": [
        ('MODELS = ["typesafe/jev-1.14", "microsoft/microsoft-decision-1", "openai/gpt-6-luna-decisions"]',
         'MODELS = ["liquid/d1-ultra", "perplexity/pplx-decider-v1.2"]', 1),
        ('len(ok_items[i]) == 3]', 'len(ok_items[i]) == len(MODELS)]', 1),
        ('samples scored by all three models', 'samples scored by all models', 2),
    ],
    "confirmatory/confirm_analysis.py": [
        ('"openai/gpt-6-luna-decisions": "gpt"}',
         '"openai/gpt-6-luna-decisions": "gpt",\n              "liquid/d1-ultra": "liquid", "perplexity/pplx-decider-v1.2": "pplx"}', 1),
        ('SCORERS = ["jev", "microsoft", "gpt", "mean", "microsoft_r"]', 'SCORERS = ["jev", "microsoft", "gpt", "mean", "microsoft_r", "liquid", "pplx"]', 1),
        ('if ok.m.nunique() < 3:', 'if ok.m.nunique() < 5:', 1),
        ('valid scores for fewer than three models', 'valid scores for fewer than five models', 1),
        ('no call is scored by all three models', 'no call is scored by all five models', 1),
        ('    for m in ["jev", "microsoft", "gpt"]:\n        g = ok[ok.m == m]', '    for m in ["jev", "microsoft", "gpt", "liquid", "pplx"]:\n        g = ok[ok.m == m]', 1),
        ('for k in ["jev", "microsoft", "gpt", "mean"]:', 'for k in ["jev", "microsoft", "gpt", "mean", "liquid", "pplx"]:', 1),
        ('for k in ["mean", "microsoft", "jev"]:', 'for k in ["mean", "microsoft", "jev", "liquid", "pplx"]:', 1),
        # frozen G4 deployment policies of the extra models: read from step 04 (L2) of the extra-model paper analysis
        ('FROZEN = {"jev": (0.0, 0.02, 0.5), "microsoft": (0.0003, 0.10, 0.5), "mean": (0.002, 0.02, 0.5)}',
         'FROZEN = {"jev": (0.0, 0.02, 0.5), "microsoft": (0.0003, 0.10, 0.5), "mean": (0.002, 0.02, 0.5)}\n'
         'FROZEN.update({k: tuple(v) for k, v in json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "frozen_extra.json"))).items()})', 1),
    ],
}

for rel, edits in EDITS.items():
    src = open(os.path.join(ROOT, rel), encoding="utf-8").read()
    for old, new, n in edits:
        c = src.count(old)
        assert c == n, f"{rel}: expected {n} occurrence(s), found {c}: {old[:80]}"
        src = src.replace(old, new)
    dst = os.path.join(OUT, rel)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    open(dst, "w", encoding="utf-8").write(src)
    print(f"{rel} -> extra_models/{rel} ({len(edits)} edit(s))")
