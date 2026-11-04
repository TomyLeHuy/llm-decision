"""Central settings for the XAI experiments. Paths are relative to the repo root (llm-decision/)."""
from pathlib import Path

XAI = Path(__file__).resolve().parent
ROOT = XAI.parent
PKG = ROOT / "certified-autonomy-ecis2027"          # read-only: never write here
ENV_FILE = ROOT / ".env"

# ---- data (read-only) -------------------------------------------------------------------------------------
SPLITS = {
    "test": PKG / "data/raw/test_1436.json",
    "train": PKG / "data/raw/train_7767.json",
    "mailu": PKG / "data/raw/mailu_ood_198.json",
}
SCORES = {k: PKG / f"runs/extra_models/combined/{k}_scores.jsonl" for k in SPLITS}
SCORE_SPLIT_PY = PKG / "confirmatory/score_split.py"
TASK_AUTHORING = PKG / "data/raw/split_documentation/TASK_AUTHORING.md"

# ---- outputs ----------------------------------------------------------------------------------------------
RESULTS = XAI / "results"
CACHE = XAI / "cache"
CACHE_FILE = CACHE / "responses.jsonl"           # append-only; key = sha256(model + "\n" + state_string)
REPEAT_CACHE_FILE = CACHE / "repeats.jsonl"      # S3 noise-floor repeats; key additionally includes repeat index

# ---- models -----------------------------------------------------------------------------------------------
MODELS = ["typesafe/jev-1.14", "microsoft/microsoft-decision-1", "openai/gpt-6-luna-decisions",
          "liquid/d1-ultra", "perplexity/pplx-decider-v1.2"]
SHORT = {"typesafe/jev-1.14": "jev", "microsoft/microsoft-decision-1": "microsoft",
         "openai/gpt-6-luna-decisions": "gpt", "liquid/d1-ultra": "liquid", "perplexity/pplx-decider-v1.2": "pplx"}
CLASSES = ["direct_execute", "ask_human", "refuse"]
PCOLS = [f"p_{c}" for c in CLASSES]

# ---- budget -----------------------------------------------------------------------------------------------
BUDGET_USD = 50.0          # hard cap (raised from 10 on 3 Nov by Tomy; protection against runaway loops only)
BUDGET_WARN_FRAC = 0.80    # stop and report at 80%
MAX_WORKERS = 4            # as in score_split.py ("do not exceed 4")

# ---- seeds ------------------------------------------------------------------------------------------------
SEED = 0                   # every sampling step in this plan uses SEED unless stated otherwise

# ---- Phase 0 ----------------------------------------------------------------------------------------------
S2_PER_MODEL = 30          # stratified by label x single/multi-turn
S3_N_CALLS = 10
S3_REPEATS = 3

# ---- Phase A ----------------------------------------------------------------------------------------------
# gate settings of the paper (Sections 4.2, 4.3; analysis/03 E6b)
A_HARM, A_ASK, DELTA, HARM_COST = 0.05, 0.15, 0.10, 10
A3_MODELS = ["jev", "liquid", "pplx"]
A3_SPLIT_SEED = 0          # paper's split seed 0 (first of the 60 G4 splits)
A3_N_OTHER = 200

# ---- Phase B (not run yet) --------------------------------------------------------------------------------
MASK_STRATEGIES = ["remove", "replace"]    # M1, M2
