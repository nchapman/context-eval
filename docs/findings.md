# Findings (rewritten from raw data, 2025-10-14)

This document supersedes both earlier write-ups (the README's R1–R11 and
the 10-13 "flat ties hierarchy" findings). Every number below was
recomputed directly from the run rows in `results/` by independent
scripts (`analysis/aggregate.py`, `analysis/lme_deep.py`,
`analysis/bykind.py`) — not by `report.py` and not by reading the old
documents first. Where a recomputation disagrees with an earlier claim,
this document says so. Judge behavior was spot-checked by hand on
discordant arm pairs (see §8). 406 tests green at time of writing.

## The instrument (one paragraph)

Replay harness, one model throughout (`deepseek/deepseek-v4.1-flash` via
OpenRouter, vetted fp8 providers). Arms are clean-room mechanisms over a
shared verbatim archive + search tools; budgets bound the assembled view
only; all token/cost accounting through one counter. Three regimes:

| corpus | horizon | question source | key files |
|---|---|---|---|
| SWE traces, LoCoMo | 8K–256K | seeded needles, corpus QA, continuity | `results/suite`, `results/final/g-*,t-*` |
| real pi sessions + sawtooth grid | 1–2M tokens | continuity/currency/needle/next-step probes | `results/final/grid64*,saw-*` |
| mega sessions (interleaved) | ~5.5M tokens, ~115–153 teeth | same probes | `results/mega`, `results/collection` |
| LongMemEval S (~120K tok) / M (~1.2M tok) | external conversational QA, official judge | 5 categories | `results/longmemeval/{dev100,m25,v2}` |

## The arm we built

The sawtooth arms are this project's own design, built to isolate one
variable: the compaction schedule (trigger, 12K-token verbatim head,
archive + search) is pinned bit-identical across arms, and only the
resident back varies — none / prose rewrite / append-only prose
segments / typed ledger (`src/context_eval/strategies/sawtooth.py`).
`sawtooth-ledger5` is the theory-driven endpoint
(`docs/resident-memory-theory.md`): typed topic-keyed records appended
once, mechanical latest-wins supersession in code, an activation-law
ladder and render-time demotion for the resident block. All three
headline arms (flat, lcm, ledger5) were tuned on the dev25 split;
dev100's first 25 samples are exactly those, so §5 reports seen/fresh
splits and fresh numbers only.

## Headline answers

1. **Cued single-fact recall is solved at every depth we could test.**
   Needles ≈1.0 for every archive-preserving arm at 8K through 5.5M
   tokens (verified in every file above). What the model can *name*, the
   archive + search returns. This is why architecture comparisons on
   ordinary QA keep tying.
2. **What separates memory designs is uncued inventory, supersession,
   and aggregation — and the differences are large exactly there**
   (0.0 vs 0.9 continuity; +26 pts supersession) while ordinary QA
   differences stay ≤6 pts.
3. **The prose hierarchy (LCM) buys nothing over flat /compact.** On
   fresh LongMemEval-S (dev100) it is slightly *worse* (0.71 vs 0.74);
   it edges flat on the s25 matched sets (0.86 vs 0.82) and on M
   post-fix (0.66 vs 0.64, n≈50/cell) — never by more than noise; on
   real-session probes it loses topic names like every other prose
   rewriter (continuity 0.02). Hierarchy-as-prose is not worth its
   complexity at any regime tested.
4. **The typed append-only ledger is the one structure that pays** —
   modestly on external QA (best-or-tied in 5/5 fresh S comparisons —
   4 wins, 1 tie; question-level paired discordants 16–8 vs flat,
   χ²≈2.7, p≈0.10, each question counted once across sweeps —
   consistent direction, still short of significance), decisively on inventory probes
   (0.77–0.91 vs 0.00–0.02 for all rewriters at 1–2M tokens),
   and it is the *only* design that has ever enumerated at 5.5M tokens
   (continuity 0.70–0.80, §6) — at cost parity with flat (§5).
5. **Flat /compact is genuinely robust on volume, but that is partly
   corpus luck.** An 8× budget squeeze (128K→16K, 12→126 folds/run)
   costs only 0.13 (0.80→0.67). But its failures are *shape* failures,
   not decay: multi-session aggregation sits at 0.50 at both budget
   extremes, and current-value tracking drops to 0.57 at 16K. LongMemEval
   rewards recency; a corpus with real drift would bite harder (§9).
6. **Everything decays with horizon** (−0.12 to −0.32 question-matched
   at 10×), and per-arm fixes claw back ~0.05–0.08 of it. The decay is
   universal; the *loss pattern* is architectural (§4).

## 1. Cued recall: solved, everywhere

- Suite @128K SWE (n=88/cell): lcm 0.955, flat 0.943, retrieval 0.943,
  arc 0.909, editing 0.75, window 0.773, full-context 0.806 with 4/22
  overflow. @64K: flat/retrieval 0.989, lcm 0.943.
- Needles 1.00 on every arm in the stress grid (`final/g-*, t-*`), the
  sawtooth grid, and the mega sessions (2 of 48 mega needle answers
  scored 0 — both read by hand and confirmed as refusals to parrot an
  obviously-injected line, a probe artifact, not retrieval misses).
- LongMemEval budget extremes, flat only (n=30/cell, unseen samples):
  16K → 0.67 at ~126 folds/run; 128K → 0.80 at ~12. Category detail:
  temporal 0.83→1.00, knowledge-update 0.57→0.86, multi-session
  **0.50 at both corners** (`results/longmemeval/v2/sweep0_extremes.jsonl`).

Archive + search makes *named* facts depth-robust. Every conclusion
about architecture must therefore come from the probes below — which is
also why earlier "tie" headlines on QA-heavy suites were structurally
guaranteed.

## 2. Inventory: where architecture separates — and when

Uncued "what has this session worked on?" after folding (LoCoMo stress
grid, calibrated calls, n=10/cell):

| arm | @8K | @16K |
|---|---|---|
| lcm (tree of small summaries) | 0.99 | 0.99 |
| flat-compact (whole-map rewrite) | 0.53 | 0.96 |
| representation (merge-churn) | 0.39 | 0.68 |
| flat + one TOPICS prompt line | 0.84 | 0.92 |
| lcm + same line | 1.00 | 1.00 |

Separation begins exactly when compaction begins. The same instruction
that rescues flat was *poison* under uncalibrated calls (shared
completion cap → 42–61% of folds returned empty; reasoning-off +
stated budget + require-content retry took it to zero — the single
most consequential engineering finding of the project, unchanged from
the earlier write-up).

At real-session scale (1–2M tokens, 64K view, 3 sessions × 3 seeds) the
prose arms lose entirely and the typed ledger takes over:

| arm | continuity | currency | needle / next-step |
|---|---|---|---|
| flat-compact | 0.00 | 0.52 | 1.00 / 1.00 |
| lcm | 0.02 | 0.64 | 1.00 / 1.00 |
| conclusions (typed ledger; post-extraction-fix grid64b runs) | 0.77 | 0.80 | 1.00 / 1.00 |

Under the pinned-schedule sawtooth (identical teeth, only the back
varies, 2 seeds × 3 sessions): no-back 0.05 continuity, prose rewrite
0.10, append-only prose segments 0.05, typed ledger 0.82 → 0.91 after
status-semantics fixes. **Append-only is not the active ingredient —
structure is**: frozen prose summaries tie evicting outright; typed,
topic-keyed records that code can dedupe (latest-wins) carry the
inventory.

## 3. Supersession and aggregation: the shape failures

LongMemEval M (10× horizon), post-fix, question-matched sets (n=49–50/arm):

| category | flat | ledger5 | lcm |
|---|---|---|---|
| knowledge-update (current value) | 0.62 | **0.88** | 0.75 |
| multi-session (aggregate/count) | 0.50 | 0.43 | 0.57 |
| temporal-reasoning | 0.57 | 0.54 \* | 0.50 |
| single-session-user / -assistant | 0.83–1.00 | 0.83–1.00 | 0.83–1.00 |

\* ledger5 temporal is n=13: two of its runs carry single-question
errors that were dropped rather than scored 0 (as zeros it would read
0.47). Flat and lcm are n=14. single-session-preference has n=2 and is
omitted.

- **Supersession separates**: flat keeps both the old and new value and
  the answerer picks wrong at depth (0.62); latest-wins dedupe holds
  0.88. This is the digest-shape mechanism, and it is fixable in flat
  (the sectioned "replace, never keep both" prompt was part of the +0.05
  fix).
- **Aggregation fails for everyone.** 0.43–0.57 on M; 0.50 for flat at
  both budget extremes. Ledger5 forensics: the facts exist in the store
  but no path guarantees *all* of a topic's rows reach the view when a
  count is asked — the model counts what is rendered and stops ("you
  serviced **1 bike**", gold 2). This is answer-time access, not
  storage, and it is the open problem no arm solved.

## 4. The loss-mode table (verified)

| loss mode | who fails | signature | measured |
|---|---|---|---|
| uncued inventory | all prose rewriters (flat, lcm, segments) | topic names vanish from summaries | 0.00–0.10 continuity at 1–2M; 0.53 @8K folding |
| supersession | flat | both values kept; old wins at depth | 0.57 @16K, 0.62 on M vs ledger5 0.88 |
| answer-time access (aggregation) | **everyone** | counts what's rendered, stops | 0.43–0.57 multi-session on M; 0.50 at both extremes |
| extraction recall | typed ledgers | what the extractor never records never exists | one prompt rule moved continuity 0.49→0.77 mid-grid |
| call truncation (fixed) | all summarizers, pre-calibration | 42–61% empty folds, silent | reasoning-off + budget + retry → 0 empties in 60/60 |

Every fix in the project's history lands in one of these rows — the
taxonomy is diagnostic, not decorative.

## 5. LongMemEval: the honest fresh numbers

The tuning split (`dev25`) inflated *everyone* — in the dev100 runs
(final configs) all three arms score 0.92–1.00 on those same 25 seen
samples (the dev25 sweeps themselves show mid-tuning configs and
lower pooled means: flat 0.83 / ledger5 0.92 / lcm 0.87). Fresh-sample
numbers only:

| comparison | flat | ledger5 | lcm | n |
|---|---|---|---|---|
| dev100 fresh-75 (3 sweeps) | 0.74 | 0.77 | 0.71 | 75×3/arm |
| s25 matched fresh sets | 0.82 | 0.88 | 0.86 | 50/arm |
| M base → fixed (matched sets A/B) | 0.59→**0.64** | 0.65→0.63 | 0.68→0.66 | 48–50/arm |

Convention throughout: errored answers (`score_method="error"`) are
excluded from denominators, matching the harness's own `report.py`; the
`analysis/` scripts implement the same rule and reproduce every row
above from raw rows.

- Ledger5 leads flat in 4 of 5 fresh sweep-level comparisons and ties
  the fifth (0.88 vs 0.88 on s25 set A); paired discordants, question
  level — each sample+question counted ONCE across the three dev100
  sweeps (net direction; only 2 questions flip sign between sweeps) —
  give 16–8 (McNemar χ²≈2.7, p≈0.10). The pool is fresh-only (the 25
  tuning samples excluded) and strictly paired (both arms answered);
  the run-level pool that repeats questions across sweeps reads 22–12
  (χ²≈2.9) — same inference. **Consistent direction, individually
  underpowered — call it a probable ~3–6 pt edge, not a tie, not a
  win.** Its wins concentrate in multi-session and temporal; its
  losses are extraction misses and one superseded-value slip
  (hand-checked, §8).
- LCM ≤ flat on fresh dev100 S in all 3 sweeps (0.71 vs 0.74) and
  edges it on the s25 matched sets (0.86 vs 0.82) and on M (0.66 vs
  0.64) — a wash at every n we have; the hierarchy does not pay.
- The flat fix is real (+0.05 pooled, positive in both matched batches);
  the ledger5 arity-N fixes were exactly neutral; lcm's tight caps cost
  it on set B (0.64→0.68 after loosening) though it settled at 0.66.
- S→M decay, question-matched (answered-question convention): flat
  −0.21/−0.24, lcm −0.12/−0.24, ledger5 −0.13/−0.32. Universal.
- A third M question set exists (sweep1: flat 0.64 / ledger5 0.88 /
  lcm 0.76) but was never re-run post-fix; it is excluded from the
  matched table above rather than mixed into it.
- **Costs are parity.** M overhead per 25 runs: flat $6.56–6.58,
  lcm $6.23–6.26, ledger5 $6.65–6.72; on dev100 flat's overhead is the
  *highest* ($2.18 vs ledger5 $1.27 per 100 runs). The earlier claims
  "lcm is the cheapest summarizer" and "flat costs less" were both
  overstatements from different regimes; treat them as equal-price.

## 6. Mega scale: the resident-memory result (new, previously buried)

At ~5.5M tokens / ~115–153 teeth every flat-grain back collapsed
(R11 verified: continuity 0.0–0.3 for none / ledger-v1 / ledger4; the
rewrite back was not run at mega scale).
The theory-driven ledger5 (activation-law ladder, append-only store,
render-time demotion; `docs/resident-memory-theory.md`) was then run on
the same mega sessions for dial collection (`results/collection/`,
4 runs):

| probe | ledger4 (R11) | ledger5 + dials |
|---|---|---|
| continuity | 0.00 / 0.00 | **0.80 / 0.70 / 0.70 / 0.70** |
| currency | 0.40 / 0.20 | 0.60 / 0.60 / 0.40 / 0.90 |
| needle / next-step | 1.0 / 1.0 | 1.0 / 1.0 |

This supports the theory's P1 prediction behaviorally — with the
strength caveat that it rests on 4 single runs from one session
family — a hierarchical, activation-managed index holds enumeration
at a depth where every flat structure fails, the mechanism the whole
project was searching for. It lives in the recordings and a
lit-review note, not in any prior headline.

**Probe-validity caveat (measured, important):** with seeded naming
moved to the first half of the stream, topics go quiet, seal, and the
continuity answer curates them out (continuity 0.0 while currency
reads 10/10; its one zero needle was another refusal —
`results/validation/`). Enumeration is *mass-gated*: topics
with recent mentions enumerate; sealed topics stay cued-status-
reachable but not enumerated. The 0.7–0.8 number is for
recently-active topics; treat "continuity" as two probes (active
enumeration, sealed recall) from here on.

## 7. What the earlier write-ups got wrong or oversold

1. "Flat is statistically indistinguishable from hierarchical
   compaction" — true for LCM, but the generalization oversold: the
   tie rested on the SWE suite and matched n≈50 sets while the
   instrument-sensitivity argument drew partly on tuning-era effects,
   and it read the ≤6-pt ledger5 edges as pure noise when the direction
   was consistent in every fresh comparison. It also buried the entire
   probe axis (0.0 vs 0.9 continuity), where architecture actually
   separates.
2. Cost ordering claims (both directions) — see §5; parity is the
   defensible statement.
3. The R11 conclusion "nobody's continuity survives 10× depth" was
   true when written and is now superseded by §6.
4. Sweep-level variance on M (flat 0.52–0.72 across fresh sets,
   answered-question convention) means any single n=25 sweep is noise;
   only matched-set and pooled comparisons are meaningful. The final
   table used matched sets — that part was right.

## 8. Judge and instrument validity

- Every discordant pair in dev100/sweep3 (n=6) hand-checked: scoring
  correct; differences are real memory behavior (ledger wins on dates,
  aggregation, honest abstention on under-determined golds; flat wins
  on ledger extraction misses). Official LongMemEval check used.
- Failure signatures audited across all files: no provider-death blanks
  in any table-cited run (the m25 key-cap incident was deleted and
  re-run; overflow counts as wrong by design in `acc*` cells). Five
  scattered single-question errors exist in the cited LongMemEval
  files (flat ×1, ledger5 ×4); each drops at most one question and is
  footnoted where it touches a cited cell (§3).
- Tuning-set inflation is real and measured (seen vs fresh, §5). Any
  future arm comparison must report seen/fresh separately.
- Sensitivity record of the instrument (why small gaps are interpretable
  at all): it resolved the flat fix (+0.05, positive in both matched
  batches), the lcm cap regression and recovery (±0.05 on set B), the
  instruction flip (+31 pts @8K), and every
  sawtooth back separation (0.05 vs 0.82). Gaps of ≤6 pts on fresh QA
  remain inside noise; gaps ≥20 pts on probes are real.

## 9. What this says to practitioners

1. **Tail + searchable archive is the floor.** It solves named-fact
   recall at any depth and costs ~$0 compaction. What it cannot do is
   inventory, supersession, or aggregation.
2. **Calibrate every utility call** (reasoning-off, stated budget,
   require-content retry). An uncalibrated call doesn't add noise — it
   silently no-ops while billing, and it re-ranks arms by call-count ×
   reasoning appetite. Biggest single lesson; unchanged.
3. **Don't ship a prose hierarchy.** LCM never beat flat by more than
   noise anywhere on fresh data and its summaries lose topic names
   like flat's do.
4. **If you need the agent to know what it's working on, ship a typed,
   append-only ledger assembled in code** (latest-wins dedupe,
   status-change lines, non-exhaustive-index footer). It is the only
   mechanism that held inventory at every scale tested through 5.5M
   tokens, it wins supersession by +26 pts on M, and it costs the same
   as flat. Audit extraction recall — it is the one failure surface.
5. **Budget the resident block by concurrent workstream count, not
   transcript length** (~30–40 tokens/workstream; saturation at
   ~90–120 topics was measured before the activation-law ladder; the
   ladder extends it but see §6's mass-gating caveat).
6. **Aggregation/counting is unsolved for every design** — access at
   answer time, not storage. Whoever fixes "count all of X" first wins
   the category no arm can currently touch.
7. **The discriminating corpus still does not exist.** LongMemEval's
   calendar span is ~10 days median and rewards recency; flat's
   shape-failures (supersession, aggregation) are exactly what sparse
   multi-month contact with real drift would amplify. That remains the
   highest-value dataset to build (see `docs/dataset-plan.md`).

## 10. Honest ledger

- ~$222 recorded API spend on disk across the project ($39 answering,
  $183 compaction/maintenance overhead; run-level usage already
  includes per-answer usage, so the two are summed once); ≈4,900 run
  rows in `results/` (merged and split copies of some legs overlap),
  one model.
- n per claim is stated inline; anything below n≈50/cell or inside
  ±6 pts should be read as directional.
- Raw rows for every table are in `results/` (paths cited inline);
  recomputation scripts in `analysis/` produce every number in this
  document from those rows.
- The continuity probe's meaning changed once mid-project (§6); probe
  configs are recorded per-run in the recordings.
