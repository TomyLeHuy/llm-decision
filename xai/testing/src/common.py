"""Shared paths and the read-only import of the paper's scorer (certified-autonomy-ecis2027/confirmatory/score_split.py)."""
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

MODEL = "openai/gpt-6-luna-decisions"
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
