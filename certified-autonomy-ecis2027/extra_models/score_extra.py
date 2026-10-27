"""Score a split with the two additional decision models, using confirmatory/score_split.py unchanged.

Only the model list is replaced; endpoint, decision schema, state string, headers, timeout, retries, checkpointing and
record format are those of score_split.py. The frozen confirmatory files are not modified, and these scores are not
part of the pre-registered protocol.

Usage (from the repository root): same arguments as score_split.py, for example
  python extra_models/score_extra.py --data data/train_7767.json --split train --out runs/extra_models/train_scores.jsonl --workers 4
"""
import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "confirmatory"))
import score_split as ss

EXTRA_MODELS = ["liquid/d1-ultra", "perplexity/pplx-decider-v1.2"]
ss.MODELS = EXTRA_MODELS
ss.APPROX_COST_PER_CALL = {"liquid/d1-ultra": 9.2e-5, "perplexity/pplx-decider-v1.2": 4.6e-5}

if __name__ == "__main__":
    # score_split.py accepts --split train|mailu|anchor; allow "test" (the full test split) and store it as given
    if "--split" in sys.argv and sys.argv[sys.argv.index("--split") + 1] == "test":
        sys.argv[sys.argv.index("--split") + 1] = "anchor"
        _parse = ss.argparse.ArgumentParser.parse_args
        def parse_args(self, *a, **k):
            ns = _parse(self, *a, **k); ns.split = "test"; return ns
        ss.argparse.ArgumentParser.parse_args = parse_args
    ss.main()
