# results/ — run archive map (this file is tracked; everything else is not)

~1.1 GB of run rows on disk, gitignored by design (reproducible via the
harness + `scripts/run_suite.sh`; per-run raw rows are the evidence behind
`docs/findings.md`). Layout by era — the findings table each directory
feeds is noted. Row schema: one JSON object per run (`strategy`,
`budget_tokens`, `answers[]` with per-answer `score`, `usage`,
`overhead_usage`, telemetry). Recompute any findings number with the
scripts in `analysis/`.

| directory | era / regime | arms | feeds |
|---|---|---|---|
| `early/` | pre-suite probes (LoCoMo supersession tests, first SWE runs, first reports) | mixed | lit-review history only |
| `smoke/`, `smoke2/`, `smoke3/` | harness bring-up + calibration smokes (incl. the reasoning-on/off fold probes) | all | R4/R8 calibration story |
| `suite/` | the 511-run battery: LoCoMo 8K/16K, SWE 64K/128K, pi 128K/256K, thinking variants | 7 | findings §1; old R1, R5–R7 |
| `continuity/` | first continuity grids (incl. the shared-cap confounded regime) | all | R2/R3 history |
| `prompting/`, `prompting2/`, `reasoning/`, `representation/` | TOPICS-instruction and calibration re-measurements | flat/lcm/retrieval/rep | R2/R3/R8 |
| `final/` | the calibrated grids: `g-*`/`t-*` (R2), `grid64*`/`saw-*` (R9/R10 sawtooth backs), `q-*` trace QA, `a-*` hidden-needle study, `p-*` pilots | flat/lcm/rep/conclusions/sawtooth family | findings §2; R9/R10 |
| `mega/` | 10×-depth sessions (~5.5M tok, one seed per back) | sawtooth none/v1/v4 | findings §6 baseline (R11) |
| `longmemeval/` | `dev25/` tuning sweeps 1–9, `dev100/` seen/fresh sweeps, `m25/` matched base-vs-fixed sets + `s25_*` S analogs, `v2/` 16K/128K extremes + cue-card pilot | flat/lcm/ledger5 (+flat-v2) | findings §3, §5 |
| `collection/`, `validation/`, `qual/`, `sweeps/`, `bench/` | ledger5 resident-memory dials: live collection recordings, sealed-tier validation, dial sweeps (offline), extraction-recall bench | sawtooth-ledger5 | findings §6; docs/resident-memory-theory.md |

`.recording.json` files next to shard jsonl are the per-run recordings
(fold snapshots, dial state, render blocks) — the forensic layer for
`flat_compact.recording()` and `sawtooth-ledger5` dials.

Costs: every run row carries `usage` (answering; already includes
per-answer usage) and `overhead_usage` (compaction/maintenance). Total
recorded spend across the archive: ~$222.
