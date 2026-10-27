#!/usr/bin/env bash
# All paper and confirmatory analyses with the two extra decision models (liquid/d1-ultra, perplexity/pplx-decider-v1.2).
# Run from the repository root after the extra-model scoring: bash extra_models/run_extra.sh
# Uses the script copies made by extra_models/make_copies.py; originals in analysis/ and confirmatory/ are not touched.
# Outputs: results/extra_models/ (paper analyses) and results/extra_models/confirmatory/ (confirmatory analysis).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
X=runs/extra_models; C=$X/combined; O=results/extra_models
mkdir -p "$C" "$O/logs" "$O/confirmatory"
echo "== 0. frozen originals unchanged?"; $PY confirmatory/verify_frozen.py
echo "== 1. check the extra-model runs"
$PY extra_models/confirmatory/check_run.py --data data/test_1436.json --scores $X/test_scores.jsonl | tee $X/test_check.txt
$PY extra_models/confirmatory/check_run.py --data data/mailu_ood_198.json --scores $X/mailu_scores.jsonl | tee $X/mailu_check.txt
$PY extra_models/confirmatory/check_run.py --data data/train_7767.json --scores $X/train_scores.jsonl | tee $X/train_check.txt
echo "== 2. combine original and extra scores"
cat data/scores/test_1436_scores.jsonl $X/test_scores.jsonl > $C/test_scores.jsonl
cat runs/mailu_scores.jsonl $X/mailu_scores.jsonl > $C/mailu_scores.jsonl
cat runs/train_scores.jsonl $X/train_scores.jsonl > $C/train_scores.jsonl
echo "== 3. paper analyses (test split)"
S=$O/scores_test_1436.csv
run() { echo ">> $*"; local log="$O/logs/$(basename "$2" .py).log"; "$@" > "$log" 2>&1 || { echo "failed, see $log"; exit 1; }; }
run $PY analysis/01_parse_scores.py $C/test_scores.jsonl $S
for step in 02_conformal_core:r02_core 03_policies_and_groups:r03_policies_groups 04_calibration_size:r04_calibration_size \
            05_budget_confidence_sensitivity:r05_sensitivity 06_revision_g5_costs_frontier:r06_revision 07_empty_policy_check:r07_empty_policy_check \
            08_g5_costs_90pct:r08_g5_costs 09_scorer_table:r09_scorer_table; do
  run $PY extra_models/analysis/${step%%:*}.py $S $O/${step##*:}.json
done
echo "== 4. G4 deployment policies of the extra models (step 04, L2, nine ask thresholds) -> frozen_extra.json"
$PY - "$O/r04_calibration_size.json" extra_models/confirmatory/frozen_extra.json <<'EOF'
import json, sys
L2 = json.load(open(sys.argv[1]))["L2"]; key = {"d1-ultra": "liquid", "pplx-decider-v1.2": "pplx"}; out = {}
for r in L2:
    if r["model"] in key and r["method"] == "certified, 9 ask thresholds":
        if r["certifiable"]: out[key[r["model"]]] = [r["execute_if_p_refuse_at_most"], r["and_p_ask_at_most"], r["refuse_if_p_refuse_at_least"]]
        else: print(key[r["model"]], "not certifiable on the test split: no frozen policy")
json.dump(out, open(sys.argv[2], "w")); print("frozen policies of the extra models:", out)
EOF
echo "== 5. confirmatory analysis with five models (train, Mailu; no anchor: the extra models have no earlier run)"
$PY extra_models/confirmatory/confirm_analysis.py --reference $C/test_scores.jsonl --train $C/train_scores.jsonl \
  --mailu $C/mailu_scores.jsonl --out $O/confirmatory | tee $O/confirmatory/analysis_log.txt
echo "== done: $O"
