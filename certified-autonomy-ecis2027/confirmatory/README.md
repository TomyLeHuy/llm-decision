# Confirmatory evaluation: instructions for running it

You will score three sets of agent tool calls with the same three decision models as in the first run, then run a fixed analysis script. You do not design or tune anything: every choice is already frozen in `PROTOCOL.md`. Your job is to produce clean scores and to report anything unexpected instead of fixing it.

**Effort:** about 1 hour of hands-on work plus 1 to 3 hours of waiting. **API cost:** about USD 2.50. **Needed by:** 27 October 2026, so the results can go into the paper before the 30 October deadline.

## What you need

- The SAFETY SENTRY release `full_data_20260530`: `test_1436.json` (as in the first run), the train split (7,767 samples) and the Mailu out-of-distribution split (198 samples). Use the file names you have; the scripts take paths.
- An OpenRouter API key with at least USD 5 credit, in an `.env` file in the repository root (`OPENROUTER_API_KEY=sk-or-...`) or as an environment variable. Never commit the key.
- Python 3.10 or newer.

## Step 0: freeze (5 minutes, before any scoring)

1. Do not edit anything in `confirmatory/` except creating `DEVIATIONS.md` if needed.
2. Check the frozen files: `python confirmatory/verify_frozen.py`. Every line must say `OK`.
3. Commit and push the repository unchanged (or create a tag such as `confirmatory-frozen`). Send the commit hash and date to Nijat. This timestamp is the evidence that the analysis was fixed before the data existed.

## Step 1: set up (10 minutes)

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
mkdir -p data/raw   # copy the three split files here, for example data/raw/train_7767.json
```

## Step 2: dry runs (2 minutes, no API calls, no cost)

```bash
python confirmatory/score_split.py --data data/raw/test_1436.json --split anchor --ids confirmatory/anchor_ids.json --out runs/anchor_scores.jsonl --dry-run
python confirmatory/score_split.py --data data/raw/mailu_198.json --split mailu --out runs/mailu_scores.jsonl --dry-run
python confirmatory/score_split.py --data data/raw/train_7767.json --split train --out runs/train_scores.jsonl --dry-run
```

Each prints the number of samples, labels, services and the expected cost. Check: the anchor run shows 150 samples (50 per label), Mailu about 198, train about 7,767. If a file fails to load or a count is far off, stop and report.

## Step 3: score (one command, or step by step)

Everything at once (resumable):

```bash
bash confirmatory/run_confirmatory.sh data/raw/test_1436.json data/raw/train_7767.json data/raw/mailu_198.json
```

Or step by step, in this order:

| Order | What | Calls | Time (4 workers) | Cost |
|---|---|---|---|---|
| 1 | Anchor: 150 test-split calls, re-scored to detect model changes | 450 | 3 to 5 min | about USD 0.03 |
| 2 | Mailu split | 594 | 5 min | about USD 0.05 |
| 3 | Train split | 23,301 | 1 to 3 h | about USD 2.00 |

For each of the three runs:

```bash
python confirmatory/score_split.py --data <file> --split <anchor|mailu|train> [--ids confirmatory/anchor_ids.json] --out runs/<name>_scores.jsonl --workers 4
python confirmatory/score_split.py --data <file> --split <...> [--ids ...] --out runs/<name>_scores.jsonl --workers 2 --retry-failed
python confirmatory/check_run.py --data <file> --scores runs/<name>_scores.jsonl [--ids ...]
```

`check_run.py` must end with `OK`. It reports missing calls, errors above 1% and malformed probability vectors.

For the long train run, keep the computer awake: on macOS prefix the command with `caffeinate -i`; on Linux use `nohup ... &` or `tmux`. If the run stops for any reason (crash, sleep, Ctrl+C, rate limit), run the same command again. Finished calls are skipped and never billed twice.

## Step 4: analyse (2 minutes)

```bash
python confirmatory/confirm_analysis.py --reference data/scores/test_1436_scores.jsonl \
  --train runs/train_scores.jsonl --mailu runs/mailu_scores.jsonl --anchor runs/anchor_scores.jsonl \
  --anchor-ids confirmatory/anchor_ids.json --out results/confirmatory
```

This writes `results/confirmatory/REPORT.md` (all results and the verdict for each prediction), `paper_table.md` (the table for the paper), `confirmatory_results.json` and `C3_per_sample.csv`. Run it once. Do not rerun it with different settings, and do not interpret or filter the results before sending them.

## Step 5: send back

- `runs/anchor_scores.jsonl`, `runs/mailu_scores.jsonl`, `runs/train_scores.jsonl` and the three `runs/*_check.txt` files (or the output of `check_run.py`)
- the whole folder `results/confirmatory/`
- `DEVIATIONS.md`, if anything deviated
- then commit the `runs/` and `results/confirmatory/` files to the repository

## Rules

- Do not change the models, the decision schema, the prompt construction, the retry policy or the analysis script.
- Do not drop, re-pick or relabel any sample. Failed calls are handled by the one retry pass and the analysis.
- Do not rerun the scoring of a split from scratch after seeing results.
- If something is off (a model is unavailable or renamed, more than 1% of calls keep failing, `response_model` differs from the requested model, a split has a different format), stop and write it into `DEVIATIONS.md` with the date. Never substitute another model.

## Troubleshooting

| Message | Meaning and what to do |
|---|---|
| `HTTP 429` in errors | rate limit; the script already retries five times. Rerun with `--workers 2`, then `--retry-failed`. |
| `HTTP 402` | out of credit; top up and rerun the same command. |
| `HTTP 404` or `400` mentioning the model | the model ID is not available any more: stop and report (deviation). |
| `Max retries exceeded` for a few calls | network trouble; `--retry-failed` handles it. |
| `check_run.py` lists missing calls | the run was interrupted; rerun the scoring command. |
| `verify_frozen.py` says `CHANGED` | a frozen file was edited; restore it with `git checkout -- confirmatory/` before going on. |
