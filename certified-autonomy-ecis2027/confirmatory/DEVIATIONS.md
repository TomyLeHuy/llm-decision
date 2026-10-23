# Deviations from PROTOCOL.md

Recorded on 23 October 2026, before `confirm_analysis.py` was run.

| Date | Deviation | Reason | Consequence |
|---|---|---|---|
| 2026-10-23 | The Mailu split file is `data/mailu_ood_198.json` (README example: `mailu_198.json`). | File name as shipped in release `full_data_20260530` (see `data/manifest.json`). Content unchanged; SHA-256 fbda3b24…5ab7aa57. | None on results; the scripts take paths. |
| 2026-10-23 | Step 0 of the README (commit and push or tag the frozen repository before scoring) was not done: the working copy is not a git repository. | Repository received as an unzipped copy. `verify_frozen.py` reported OK for all nine frozen files before any scoring. | The timestamp evidence is the `verify_frozen.py` check and the file modification times, not a commit. Should be committed as soon as possible. |
| 2026-10-23 | The OpenRouter key was read from `../llm-decision/.env` and passed as an environment variable. | No `.env` in this repository. | None (same endpoint and models). |
| 2026-10-23 | Train scoring was interrupted once (Ctrl+C, laptop taken offline) after 3,660 calls and resumed with the same command. | Leaving the office. | None: resuming is allowed (README); no successful call was re-scored. |

## Recorded for transparency (within the protocol, not a deviation)

- Train split: 12 of 7,767 gpt calls (0.15%, below the 1% threshold) still failed after the one `--retry-failed` pass, all with `HTTP 502 ... OpenAI refused to answer question "safety_decision"`. Under the inclusion rule (Section 2) these 12 items are excluded for all three models; 7,755 items are analysed. Labels of the excluded items: refuse 8, ask_human 2, direct_execute 2 (services: agentharm 4, api_token_api 3, when2call 2, customdashboardsapi 1, atbench 1, rocketchat 1). dataset_idx: 597, 811, 1995, 3377, 3629, 4726, 5265, 5402, 6152, 7018, 7740, 7741. Because the excluded items are mostly refusals, the exclusion is not random; its effect on refusal rates is at most 8 of 1,718 refusals (0.5 percentage points).
- Anchor and Mailu runs: 0 failed calls.
