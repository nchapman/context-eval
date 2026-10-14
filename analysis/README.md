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

**Testing choice (deliberate):** these scripts are not collected by
pytest. They are session-audit tools whose output is diffed by hand
against `docs/findings.md`; the green gate (`ruff check`) only lint-checks
them. Their input trees (`results/`, `data/`) are gitignored, so the
harness test-suite cannot exercise them without fixtures that would
duplicate the scripts' logic rather than check it. Guardrails that DO
apply: rows that fail to parse or are not JSON objects become
`_parse_error` and are skipped, unreadable or missing files warn and
continue, and every number they print must reproduce a findings figure
or the doc is wrong.
