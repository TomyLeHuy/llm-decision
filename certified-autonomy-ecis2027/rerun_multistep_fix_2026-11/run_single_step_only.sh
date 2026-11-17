#!/usr/bin/env bash
# Variant "single-step records only"; no API calls.
# Multi-step records are dropped from every split, then the unchanged paper and confirmatory scripts are run.
# Single-step records get the same input in first_step and all_steps mode, so this variant is free of the input bug.
#   SOURCE=new (default): from the re-run's all_steps scores   -> results/single_step_only/
#   SOURCE=ecis:          from the original ECIS scores         -> results/ecis_single_step_only/
# Comparing the two isolates vendor drift / sampling noise on identical inputs from the effect of the fix.
# Run from the package root after run_rerun.sh:  [SOURCE=ecis] bash rerun_multistep_fix_2026-11/run_single_step_only.sh
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
R=rerun_multistep_fix_2026-11
SOURCE=${SOURCE:-new}
if [ "$SOURCE" = "ecis" ]; then TAG=ecis_single_step_only; else TAG=single_step_only; fi
SS=$R/runs/$TAG; RES=$R/results/$TAG; LOG=$R/logs/$TAG
mkdir -p "$SS" "$RES/paper" "$RES/confirmatory" "$LOG"

echo "== 1. filter the $SOURCE score files to single-step records"
$PY - "$R/runs" "$SS" "$SOURCE" <<'EOF'
import json, sys
runs, out, source = sys.argv[1:4]
ecis = {"test": "data/scores/test_1436_scores.jsonl", "mailu": "runs/mailu_scores.jsonl",
        "anchor": "runs/anchor_scores.jsonl", "train": "runs/train_scores.jsonl"}
for split, data in [("test", "data/test_1436.json"), ("mailu", "data/mailu_ood_198.json"),
                    ("anchor", "data/test_1436.json"), ("train", "data/train_7767.json")]:
    d = json.load(open(data, encoding="utf-8"))
    single = {i for i, r in enumerate(d) if sum(m.get("role") == "user" for m in r["prompt"]) == 1}
    n_in = n_out = 0
    src = ecis[split] if source == "ecis" else f"{runs}/{split}_scores_all_steps.jsonl"
    with open(src, encoding="utf-8") as f, open(f"{out}/{split}_scores_single_only.jsonl", "w", encoding="utf-8") as g:
        for line in f:
            if line.strip():
                n_in += 1
                if json.loads(line)["dataset_idx"] in single:
                    g.write(line); n_out += 1
    print(f"{split}: kept {n_out} of {n_in} score lines (single-step records: {len(single)} of {len(d)})")
EOF

echo "== 2. paper analyses on single-step test records"
S=$RES/paper/scores_test_1436.csv
run() { echo ">> $*"; local log="$LOG/$(basename "$2" .py).log"; "$@" > "$log" 2>&1 || { echo "failed, see $log"; exit 1; }; }
run $PY analysis/01_parse_scores.py "$SS/test_scores_single_only.jsonl" "$S"
for step in 02_conformal_core:r02_core 03_policies_and_groups:r03_policies_groups 04_calibration_size:r04_calibration_size \
            05_budget_confidence_sensitivity:r05_sensitivity 06_revision_g5_costs_frontier:r06_revision \
            07_empty_policy_check:r07_empty_policy_check 08_g5_costs_90pct:r08_g5_costs 09_scorer_table:r09_scorer_table; do
  run $PY analysis/${step%%:*}.py "$S" "$RES/paper/${step##*:}.json"
done

echo "== 3. confirmatory analysis on single-step records"
$PY confirmatory/confirm_analysis.py --reference "$SS/test_scores_single_only.jsonl" \
  --train "$SS/train_scores_single_only.jsonl" --mailu "$SS/mailu_scores_single_only.jsonl" \
  --anchor "$SS/anchor_scores_single_only.jsonl" --anchor-ids confirmatory/anchor_ids.json \
  --out "$RES/confirmatory" | tee "$RES/confirmatory/analysis_log.txt"
echo "== done: $RES"
