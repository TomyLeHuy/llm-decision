#!/usr/bin/env bash
# Reproduce every number in the paper from the test-split scores. Run from the repository root: bash analysis/run_all.sh
# Runtime: about 30 to 60 minutes on a laptop (steps 05 and 06 dominate). Logs go to results/logs/.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p results/logs
S=results/scores_test_1436.csv
run() { echo ">> $*"; local log="results/logs/$(basename "$2" .py).log"; "$@" > "$log" 2>&1 || { echo "failed, see $log"; exit 1; }; }
run python analysis/01_parse_scores.py data/scores/test_1436_scores.jsonl "$S"
run python analysis/02_conformal_core.py "$S" results/r02_core.json
run python analysis/03_policies_and_groups.py "$S" results/r03_policies_groups.json
run python analysis/04_calibration_size.py "$S" results/r04_calibration_size.json
run python analysis/05_budget_confidence_sensitivity.py "$S" results/r05_sensitivity.json
run python analysis/06_revision_g5_costs_frontier.py "$S" results/r06_revision.json
run python analysis/07_empty_policy_check.py "$S" results/r07_empty_policy_check.json
run python analysis/08_g5_costs_90pct.py "$S" results/r08_g5_costs.json
run python analysis/09_scorer_table.py "$S" results/r09_scorer_table.json
run python analysis/10_floor_numbers.py results/r10_floor_numbers.json
echo "all analysis steps done"
