#!/usr/bin/env bash
# Re-run of the paper pipeline with the multi-step input fix (score_split.py --input-mode all_steps).
#
# Why: the submitted ECIS run sent only the FIRST user message of multi-step records, while the label belongs to the
# LAST proposed call (confirmatory/DEVIATIONS.md, 2026-11-10). This re-run scores every split again with the same three
# paper models (jev, microsoft, gpt), now with all user turns, and runs the unchanged analysis scripts on the new
# scores. Nothing outside this folder is written: the original scores and results stay where they are (= the archive of
# the submitted ECIS version, reproducible with --input-mode first_step).
#
# Run from the package root:  bash rerun_multistep_fix_2026-11/run_rerun.sh            (resumable)
#                              DRY=1 bash rerun_multistep_fix_2026-11/run_rerun.sh      (counts + cost, no API calls)
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
R=rerun_multistep_fix_2026-11
RUNS=$R/runs; RES=$R/results; LOG=$R/logs
MODE="--input-mode all_steps"
mkdir -p "$RUNS" "$RES/paper" "$RES/confirmatory" "$LOG"

echo "== 0. frozen files unchanged?"; $PY confirmatory/verify_frozen.py

# split name | data file | extra args | output
SPLITS=(
  "test|data/test_1436.json||$RUNS/test_scores_all_steps.jsonl"
  "mailu|data/mailu_ood_198.json||$RUNS/mailu_scores_all_steps.jsonl"
  "anchor|data/test_1436.json|--ids confirmatory/anchor_ids.json|$RUNS/anchor_scores_all_steps.jsonl"
  "train|data/train_7767.json||$RUNS/train_scores_all_steps.jsonl"
)
echo "== 1. dry runs (no API calls)"
for s in "${SPLITS[@]}"; do IFS="|" read -r name data extra out <<< "$s"
  $PY confirmatory/score_split.py --data "$data" --split "$name" $extra $MODE --out "$out" --dry-run
done
[ "${DRY:-0}" = "1" ] && { echo "dry run only"; exit 0; }

for s in "${SPLITS[@]}"; do IFS="|" read -r name data extra out <<< "$s"
  echo "== 2. scoring: $name"
  $PY confirmatory/score_split.py --data "$data" --split "$name" $extra $MODE --out "$out" --workers 4
  $PY confirmatory/score_split.py --data "$data" --split "$name" $extra $MODE --out "$out" --workers 2 --retry-failed
  $PY confirmatory/check_run.py --data "$data" --scores "$out" $extra | tee "$RUNS/${name}_check.txt"
done

echo "== 3. paper analyses (test split), unchanged scripts, new paths"
S=$RES/paper/scores_test_1436.csv
run() { echo ">> $*"; local log="$LOG/$(basename "$2" .py).log"; "$@" > "$log" 2>&1 || { echo "failed, see $log"; exit 1; }; }
run $PY analysis/01_parse_scores.py "$RUNS/test_scores_all_steps.jsonl" "$S"
for step in 02_conformal_core:r02_core 03_policies_and_groups:r03_policies_groups 04_calibration_size:r04_calibration_size \
            05_budget_confidence_sensitivity:r05_sensitivity 06_revision_g5_costs_frontier:r06_revision \
            07_empty_policy_check:r07_empty_policy_check 08_g5_costs_90pct:r08_g5_costs 09_scorer_table:r09_scorer_table; do
  run $PY analysis/${step%%:*}.py "$S" "$RES/paper/${step##*:}.json"
done

echo "== 4. confirmatory analysis (frozen script; reference = the new test scores)"
$PY confirmatory/confirm_analysis.py --reference "$RUNS/test_scores_all_steps.jsonl" \
  --train "$RUNS/train_scores_all_steps.jsonl" --mailu "$RUNS/mailu_scores_all_steps.jsonl" \
  --anchor "$RUNS/anchor_scores_all_steps.jsonl" --anchor-ids confirmatory/anchor_ids.json \
  --out "$RES/confirmatory" | tee "$RES/confirmatory/analysis_log.txt"
echo "== done: $R"
