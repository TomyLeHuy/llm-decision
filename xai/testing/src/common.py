"""Shared paths and the read-only import of the paper's scorer (certified-autonomy-ecis2027/confirmatory/score_split.py)."""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]          # xai/testing
REPO = ROOT.parents[1]                               # llm-decision
PKG = REPO / "certified-autonomy-ecis2027"           # frozen package, read only
RESULTS = ROOT / "results"
PLOTS = ROOT / "plots"

DATA_PATH = PKG / "data" / "test_1436.json"
OCT_SCORES = PKG / "results" / "scores_test_1436.csv"
CACHE_PATH = RESULTS / "coalitions.jsonl"
SUBSET_PATH = RESULTS / "subset_ids.json"
LLMSHAP_RAW = RESULTS / "llmshap_raw.jsonl"
RUN_LOG = RESULTS / "run_log.txt"
DETERMINISM_PATH = RESULTS / "determinism.json"

# Import without writing a __pycache__ into the frozen package.
sys.dont_write_bytecode = True
sys.path.insert(0, str(PKG / "confirmatory"))
import score_split as ss  # noqa: E402
sys.path.pop(0)

try:
    from dotenv import load_dotenv
    load_dotenv(REPO / ".env")
except ImportError:
    pass

MODEL = "openai/gpt-6-luna-decisions"                 # default model (the gpt pilot)
assert MODEL in ss.MODELS
COST_PER_CALL = ss.APPROX_COST_PER_CALL[MODEL]
CLASSES = list(ss.DECISION_SPACE)                   # direct_execute, ask_human, refuse
MAX_WORKERS = 4

# Verbatim copy of the state template inside ss.call_openrouter_decision (it is not exposed as a constant there).
STATE_TEMPLATE = ("### System Reviewer Contract & Governance Policy\n{system}\n\n"
                  "### Current Context, Tool Proposal & Ambience\n{user}")


def make_state(system: str, user: str) -> str:
    return STATE_TEMPLATE.format(system=system, user=user)


def api_key() -> str:
    k = os.environ.get("OPENROUTER_API_KEY")
    assert k and not k.endswith("DEIN_KEY_HIER"), "set OPENROUTER_API_KEY (environment or repo .env)"
    return k


def variant_suffix(fixed) -> str:
    """'' for the 7-player design; '__fixed-tool' etc. when some fields are kept in every coalition (not players)."""
    return "" if not fixed else "__fixed-" + "+".join(sorted(fixed))


def parse_fixed(arg: str | None) -> list:
    from fields import ALL_FIELDS
    fixed = [f for f in (arg or "").split(",") if f]
    assert set(fixed) <= set(ALL_FIELDS), f"unknown fields in --fixed: {set(fixed) - set(ALL_FIELDS)}"
    return fixed


# ---- the five decision models ---------------------------------------------------------------------------------------
EXTRA_SCORER = PKG / "extra_models" / "score_extra.py"
OCT_SCORES_EXTRA = PKG / "results" / "extra_models" / "scores_test_1436.csv"


def _extra_constants():
    """EXTRA_MODELS and the cost dict of score_extra.py, read with ast. Importing it would overwrite ss.MODELS and
    ss.APPROX_COST_PER_CALL in this process (module-level assignments)."""
    import ast
    tree = ast.parse(EXTRA_SCORER.read_text(encoding="utf-8"))
    models = costs = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            t = node.targets[0]
            if isinstance(t, ast.Name) and t.id == "EXTRA_MODELS":
                models = ast.literal_eval(node.value)
            elif isinstance(t, ast.Attribute) and t.attr == "APPROX_COST_PER_CALL":
                costs = ast.literal_eval(node.value)
    assert models and costs and set(models) == set(costs), "could not read score_extra.py constants"
    return models, costs


_EXTRA_MODELS, _EXTRA_COSTS = _extra_constants()
SLUGS = {"openai/gpt-6-luna-decisions": "gpt", "typesafe/jev-1.14": "jev", "microsoft/microsoft-decision-1": "microsoft",
         "liquid/d1-ultra": "liquid", "perplexity/pplx-decider-v1.2": "pplx"}
MODEL_COST = {**{m: ss.APPROX_COST_PER_CALL[m] for m in ss.MODELS}, **_EXTRA_COSTS}
MODEL_OCT = {**{m: OCT_SCORES for m in ss.MODELS}, **{m: OCT_SCORES_EXTRA for m in _EXTRA_MODELS}}
assert set(SLUGS) == set(MODEL_COST) == set(MODEL_OCT), "model tables disagree"
MODEL_OF_SLUG = {v: k for k, v in SLUGS.items()}


class ModelPaths:
    """Where one model's files live. gpt keeps the original layout (results/, plots/); the others use a sub-folder."""

    def __init__(self, model: str):
        assert model in SLUGS, f"unknown model {model}; known: {sorted(SLUGS)}"
        self.model, self.slug, self.cost, self.oct = model, SLUGS[model], MODEL_COST[model], MODEL_OCT[model]
        legacy = model == MODEL
        self.results = RESULTS if legacy else RESULTS / self.slug
        self.plots = PLOTS if legacy else PLOTS / self.slug
        self.cache = self.results / "coalitions.jsonl"         # one cache file per model
        self.raw_stem = self.results / "llmshap_raw"
        self.run_log = self.results / "run_log.txt"
        self.determinism = self.results / "determinism.json"
        self.run_config = self.results / "run_config.json"    # repeats used for this model (absent for gpt = 1)

    def raw(self, sfx: str):
        return self.raw_stem.with_name(self.raw_stem.name + sfx + ".jsonl")

    def repeats(self) -> int:
        if self.run_config.exists():
            return int(json.load(open(self.run_config))["repeats"])
        return 1


def resolve_model(arg: str) -> str:
    """Accept a full model id or a slug (gpt, jev, microsoft, liquid, pplx)."""
    return MODEL_OF_SLUG.get(arg, arg)
