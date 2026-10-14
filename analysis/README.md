# analysis/ — independent recomputation of the findings

These scripts recompute every number in `docs/findings.md` directly from
the raw run rows in `results/` — deliberately importing nothing from
`src/context_eval` (not even `report.py`), so a bug in the harness's own
aggregation cannot silently back the write-up.

- `aggregate.py` — walks all `results/**/*.jsonl`; per-file per-arm
  per-budget accuracy, errors, blanks, overflow → `table.txt` /
  `all_rows.json` (both gitignored derived artifacts).
- `lme_deep.py` — LongMemEval deep dive: per-category breakdowns,
  seen/fresh split (via `data/longmemeval_dev100.jsonl` ordering),
  question-matched paired comparisons, matched base-vs-fixed M sets.
- `bykind.py` — probe-level (continuity/currency/needle/next-step)
  breakdowns for the R9/R10/R11/mega files.

Run: `python3 analysis/<script>.py` (stdlib only).
