#!/usr/bin/env bash
# Confirmatory evaluation, all steps. Run from the repository root:
#   bash confirmatory/run_confirmatory.sh <path to test_1436.json> <path to train split .json> <path to Mailu split .json>
# Safe to interrupt and rerun with the same arguments: finished API calls are skipped and never billed twice.
set -euo pipefail
cd "$(dirname "$0")/.."
TEST=${1:?path to test_1436.json}; TRAIN=${2:?path to train split}; MAILU=${3:?path to Mailu split}
mkdir -p runs results/confirmatory
echo "== 0. frozen files unchanged?"; python confirmatory/verify_frozen.py
echo "== 1. dry runs (no API calls)"
python confirmatory/score_split.py --data "$TEST" --split anchor --ids confirmatory/anchor_ids.json --out runs/anchor_scores.jsonl --dry-run
python confirmatory/score_split.py --data "$MAILU" --split mailu --out runs/mailu_scores.jsonl --dry-run
python confirmatory/score_split.py --data "$TRAIN" --split train --out runs/train_scores.jsonl --dry-run
for s in anchor mailu train; do
  case $s in anchor) D=$TEST; X="--ids confirmatory/anchor_ids.json";; mailu) D=$MAILU; X="";; train) D=$TRAIN; X="";; esac
  echo "== scoring: $s"
  python confirmatory/score_split.py --data "$D" --split $s $X --out runs/${s}_scores.jsonl --workers 4
  python confirmatory/score_split.py --data "$D" --split $s $X --out runs/${s}_scores.jsonl --workers 2 --retry-failed
  python confirmatory/check_run.py --data "$D" --scores runs/${s}_scores.jsonl $X | tee runs/${s}_check.txt
done
echo "== analysis"
python confirmatory/confirm_analysis.py --reference data/scores/test_1436_scores.jsonl \
  --train runs/train_scores.jsonl --mailu runs/mailu_scores.jsonl --anchor runs/anchor_scores.jsonl \
  --anchor-ids confirmatory/anchor_ids.json --out results/confirmatory | tee results/confirmatory/analysis_log.txt
echo "== done. Send: runs/*.jsonl, runs/*_check.txt, results/confirmatory/ (REPORT.md, paper_table.md, confirmatory_results.json, C3_per_sample.csv)"
