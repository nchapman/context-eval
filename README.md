# context-eval

**Does hierarchical compaction beat the `/compact` default — and what does
it actually take for an agent to remember what it was working on?**

**TL;DR — popular compaction strategies all do well at recall you can
name, but continuously struggle with recall you can't.**

"Find the staging database's auth token" is solved everywhere we looked:
archive + search returns needles ≈1.0 for every arm from 8K budgets
to 5.5M tokens, and ordinary QA gaps stay ≤6 pts. The unnamed
questions — *what have we worked on* (inventory), *which value is
current* (supersession), *how many* (aggregation) — are where flat,
LCM, and every prose design collapse, and where the one structure
that holds them pays.

- **Keep the deployed default.** The research hierarchy (LCM) never
  beat flat `/compact` by more than noise on any fresh comparison
  (0.71 vs 0.74 on fresh LongMemEval-S).
- **Ship a typed append-only ledger if the agent must know what it's
  working on** (our arm, [§4](#4-method-in-ten-lines)): inventory 0.77–0.91 vs 0.00–0.02 for
  prose rewriters at 1–2M tokens, supersession 0.88 vs 0.62 (n≈50) —
  at cost parity. Its ordinary-QA edge is direction, not significance
  (p≈0.10).
- **Counting is unsolved for everyone** (0.43–0.57 multi-session) —
  the open problem no arm touched.
- **Calibrate every utility call.** Uncalibrated calls silently
  returned empty on 42–61% of folds and inverted an arm ranking;
  calibrated → 0 empties in 60/60.

7,062 replay runs, one model, one cost counter (~$220 total); 18 arms
across SWE traces, LoCoMo, LongMemEval S/M, real 1–2M-token agent
sessions, and 5.5M-token mega-sessions at 8K–256K budgets. Full
evidence and error bars: [`docs/findings.md`](docs/findings.md).

## 1. The six answers

1. **Cued single-fact recall is solved at every depth tested.** Injected
   needles score ≈1.0 for every archive-preserving arm from 8K budgets to
   5.5M tokens. What the model can *name*, search returns.
2. **What separates designs is uncued inventory, supersession, and
   aggregation** — differences are large exactly there (0.0 vs 0.9
   continuity; +26 pts supersession) while ordinary QA differences stay
   ≤6 pts.
3. **The prose hierarchy (LCM) buys nothing over flat `/compact`.** Slightly
   worse on fresh LongMemEval-S (0.71 vs 0.74), edges flat elsewhere by
   less than noise, and loses topic names like every other prose rewriter
   (continuity 0.02 on real 1–2M-token sessions).
4. **The typed append-only ledger is the one structure that pays** — best or
   tied in 5/5 fresh QA comparisons (question-level paired discordants 16–8
   vs flat, χ²≈2.7, p≈0.10 — direction, still short of significance),
   decisive on inventory probes (0.77–0.91 vs 0.00–0.02), the only design
   to enumerate at 5.5M tokens — at cost parity with flat.
5. **Flat `/compact` is extraordinarily robust on volume — partly corpus
   luck.** An 8× budget squeeze (12→126 folds/run) costs only 0.13
   (0.80→0.67); but its failures are *shape* failures: multi-session
   aggregation sits at 0.50 at both budget extremes.
6. **Everything decays with horizon** (−0.12 to −0.32 question-matched at
   10×) and fixes claw back ~0.05–0.08. The decay is universal; the *loss
   pattern* is architectural.

## 2. The one table: how each design fails

Each design fails in a characteristic, greppable way — every fix in
the project's history lands in one of these rows:

| loss mode | who fails | signature | measured |
|---|---|---|---|
| uncued inventory | all prose rewriters (flat, lcm, segments) | topic names vanish from summaries | 0.00–0.10 continuity at 1–2M tokens; 0.53 @8K folding |
| supersession | flat | keeps old *and* new value; old wins at depth | 0.57 @16K, 0.62 on M vs ledger 0.88 |
| answer-time access (aggregation) | **everyone** | counts what's rendered, stops | 0.43–0.57 multi-session on M; 0.50 at both budget extremes |
| extraction recall | typed ledgers | what the extractor never records never exists | one prompt rule moved continuity 0.49→0.77 mid-grid |
| call truncation (fixed) | all summarizers, pre-calibration | 42–61% of folds return empty, silently | reasoning-off + budget + retry → 0 empties in 60/60 |

## 3. Why this exists

Every production coding agent ships context management — `/compact`,
in-place editing, condensers — and a growing literature proposes
hierarchies, citations, and memory tiers. Nobody had run the deployed
default against the research proposals under one harness, one model, and
one cost ledger. So we did, across:

| corpus | horizon | probes |
|---|---|---|
| SWE traces, LoCoMo | 8K–256K | needles, corpus QA, continuity |
| real pi agent sessions + sawtooth grid | 1–2M tokens | continuity, currency, needle, next-step |
| mega sessions (interleaved) | ~5.5M tokens, ~115–153 compaction teeth | same probes |
| LongMemEval S (~120K tok) / M (~1.2M tok) | external conversational QA | 5 categories, official judge |

## 4. Method in ten lines

Replay harness (`src/context_eval/`): append → strategy hook → assemble →
budget check, then question-time tool interleaving. Arms are clean-room
mechanisms over a shared verbatim archive + search tools: evict
([`sliding-window`](src/context_eval/strategies/sliding_window.py),
[`context-editing`](src/context_eval/strategies/context_editing.py)), search
([`retrieval`](src/context_eval/strategies/retrieval.py),
[`arc`](src/context_eval/strategies/arc.py)),
rewrite ([`flat-compact`](src/context_eval/strategies/flat_compact.py),
[`lcm`](src/context_eval/strategies/lcm.py),
[`sawtooth-rewrite/segments`](src/context_eval/strategies/sawtooth.py)),
maintain ([`representation`](src/context_eval/strategies/representation.py)),
and typed append-only ledgers
([`conclusions`](src/context_eval/strategies/conclusions.py),
[`sawtooth-ledger*`](src/context_eval/strategies/sawtooth.py)) — 18
registered arms. Budgets bound the assembled view
only; all token/cost accounting flows through one counter. One model
everywhere (`deepseek/deepseek-v4.1-flash`); LongMemEval scored with its
official check; matched-set and seen/fresh discipline throughout. Details:
[findings §8](docs/findings.md#8-judge-and-instrument-validity).

One arm family is ours. The `sawtooth-*` arms pin the compaction
schedule bit-identical — same trigger, same 12K-token verbatim head,
same archive + search — and vary only what each cut leaves resident
(no back / prose rewrite / append-only prose segments / typed ledger),
so arm differences isolate that one variable.
`sawtooth-ledger5` is the theory-driven endpoint of that line
([docs/resident-memory-theory.md](docs/resident-memory-theory.md)): a
typed, append-only ledger of topic-keyed records (status changes,
facts, ids and numbers verbatim) with latest-wins supersession applied
in code, an activation-law ladder that budgets the resident block by
workstream, and render-time demotion of quiet topics. All three
headline arms — flat, lcm, ledger5 — were tuned on the 25-sample dev25
split; dev100's first 25 samples are exactly those, which is why every
LongMemEval number below is seen/fresh split and why
[§5.5](#55-the-honest-fresh-longmemeval-numbers)'s fresh
numbers, not the 0.92–1.00 seen scores, are the real ones.

## 5. Findings

### 5.1 Cued recall: solved, everywhere

Suite @128K SWE (n=88/cell): lcm 0.955, flat 0.943, retrieval 0.943 —
window 0.773, editing 0.75, full-context 0.806 only because it overflows.
Needles 1.00 on every arm at every stress budget and at 5.5M tokens (2 of
48 mega needle answers scored 0 — both verified refusals to parrot an
obviously injected line). Flat at budget extremes: 0.67 @16K (~126
folds/run) vs 0.80 @128K (~12). Archive + search makes *named* facts
depth-robust — which is why QA-only comparisons keep tying and the real
signal lives in the probes below.

### 5.2 Inventory: where architecture separates — exactly when compaction begins

Uncued "what has this session worked on?" (LoCoMo stress grid, calibrated
calls, n=10/cell):

| arm | @8K | @16K |
|---|---|---|
| lcm (tree of small summaries) | 0.99 | 0.99 |
| flat-compact (whole-map rewrite) | 0.53 | 0.96 |
| representation (merge-churn) | 0.39 | 0.68 |
| flat + one TOPICS prompt line | 0.84 | 0.92 |

At real-session scale (1–2M tokens, 64K view) the prose arms lose entirely
and the typed ledger takes over: continuity flat 0.00 / lcm 0.02 /
**conclusions 0.77**; currency 0.52 / 0.64 / **0.80**. Under a pinned
sawtooth schedule (identical teeth, only the resident back varies):
no-back 0.05, prose rewrite 0.10, append-only prose 0.05, typed ledger
0.82→0.91 after status-semantics fixes. **Append-only is not the active
ingredient — structure is**: frozen prose summaries tie evicting outright;
typed, topic-keyed records that code can dedupe (latest-wins) carry the
inventory.

### 5.3 Supersession and aggregation: the shape failures

LongMemEval M (10× horizon), post-fix, matched sets (n≈50/arm):

| category | flat | ledger5 | lcm |
|---|---|---|---|
| knowledge-update (current value) | 0.62 | **0.88** | 0.75 |
| multi-session (aggregate/count) | 0.50 | 0.43 | 0.57 |
| temporal-reasoning | 0.57 | 0.54 * | 0.50 |

Supersession separates: latest-wins dedupe holds current values (+26 pts
over flat). **Aggregation fails for everyone** — the facts exist in the
store but nothing guarantees all of a topic's rows reach the view when a
count is asked ("you serviced **1 bike**", gold 2). Answer-time access,
not storage: the open problem no arm solved. (\* n=13; two dropped error
rows —
[findings §3](docs/findings.md#3-supersession-and-aggregation-the-shape-failures).)

### 5.4 Flat's robustness is real — and partly corpus luck

The 8× squeeze from 128K to 16K costs flat only 0.13 even as folds go 12 →
126 per run; a rolling rewrite is accidental re-consolidation and
LongMemEval (median calendar span ~10 days) rewards recency. But the
failures that remain are shape failures, not decay: multi-session 0.50 at
*both* budget extremes, current-value tracking 0.57 at 16K. A corpus with
sparse contact over months and real drift would bite harder — and it does
not exist yet ([§8](#8-what-this-doesnt-show)).

### 5.5 The honest fresh LongMemEval numbers

| comparison | flat | ledger5 | lcm | n |
|---|---|---|---|---|
| dev100 fresh-75 (3 sweeps) | 0.74 | 0.77 | 0.71 | 75×3/arm |
| s25 matched fresh sets | 0.82 | 0.88 | 0.86 | 50/arm |
| M base → fixed (matched sets) | 0.59→**0.64** | 0.65→0.63 | 0.68→0.66 | 48–50/arm |

Ledger5 leads flat in 4 of 5 fresh comparisons and ties the fifth
(question-level paired discordants 16–8, χ²≈2.7, p≈0.10 — each question
counted once across sweeps, fresh-only, strictly paired — a probable
~3–6 pt edge, not a tie,
not a win). LCM is a wash everywhere. The flat fix is real (+0.05 pooled,
both matched batches). S→M decay is universal: −0.12 to −0.32. **Costs
are parity** (M overhead $6.2–6.7 per 25 runs across all three arms) —
earlier "cheapest summarizer" claims in both directions were overstatements.

### 5.6 Mega scale: the ladder enumerates where everything collapsed

At ~5.5M tokens every flat-grain back collapsed (R11: continuity 0.0–0.3
for none / ledger-v1 / ledger4). The theory-driven ledger5 — activation-law
ladder, append-only store, render-time demotion
([docs/resident-memory-theory.md](docs/resident-memory-theory.md)) — holds **continuity 0.70–0.80, currency
0.4–0.9, needles 1.0** on the same sessions (n=4 runs; treat as one
session family's evidence). Caveat, measured: enumeration is *mass-gated* —
topics that went quiet early seal and get curated out of uncued answers
(continuity 0.0 with currency 10/10 in the validation run). The resident
index preserves findability and status; it cannot force enumeration of
long-dormant work.

## 6. What we'd tell an agent builder

1. **Tail + searchable archive is the floor.** It solves named-fact recall
   at any depth at ~$0 compaction cost. It cannot do inventory,
   supersession, or aggregation.
2. **Calibrate every utility call** (reasoning-off, stated budget,
   require-content retry). An uncalibrated call doesn't add noise — it
   silently no-ops while billing, and it re-ranks arms by call-count ×
   reasoning appetite. The project's single biggest engineering lesson.
3. **Don't ship a prose hierarchy.** LCM never beat flat by more than
   noise anywhere on fresh data, and its summaries lose topic names like
   flat's do.
4. **If the agent must know what it's working on, ship a typed,
   append-only ledger assembled in code** (latest-wins dedupe,
   status-change lines, non-exhaustive-index footer). The only mechanism
   that held inventory at every scale tested through 5.5M tokens; wins
   supersession by +26 pts; costs the same as flat. Audit extraction
   recall — its one failure surface.
5. **Budget the resident block by concurrent workstream count, not
   transcript length** (~30–40 tokens/workstream; saturation at ~90–120
   topics measured before the activation-law ladder).
6. **Aggregation/counting is unsolved for every design.** Access at answer
   time, not storage — whoever fixes "count all of X" wins a category no
   arm can touch.
7. **The discriminating corpus still does not exist** — sparse multi-month
   contact with real drift. Highest-value dataset to build next
   ([`docs/dataset-plan.md`](docs/dataset-plan.md)).

## 7. How the conclusions evolved

Four measurement lessons changed headlines mid-project — each caught by
the harness or an independent recheck, which is why we trust the rest:

- **Calibration physics.** Summarization calls sharing a completion cap
  with reasoning returned empty 42–61% of the time, silently, and
  *inverted* an early arm ordering before reasoning-off fixed it. Blank
  completions are the default failure mode of LLM utility calls on
  reasoning models.
- **Tuning-set inflation.** The tuning split scored all arms 0.92–1.00;
  only the seen/fresh split exposed the real fresh numbers — and shrank a
  10-pt "ledger5 wins" gap to a probable 3–6 pts.
- **The tie claim, corrected.** "Flat statistically indistinguishable from
  hierarchy" was true for LCM but overgeneralized, partly on
  tuning-era sensitivity arguments; the corrected reading is [§1](#1-the-six-answers).
- **Probe validity.** The continuity probe's meaning depends on where a
  topic's last mention lands (mass-gating); a stronger needle-detection
  effect surfaced as refusals. Probes are instruments — calibrate them
  like instruments.

## 8. What this doesn't show

One model class builds every arm's working context and answers every
question. Costs are replay-based (production pays the view every turn;
ratios transfer, absolutes don't). n is stated per claim in
findings.md; anything inside ±6 pts or below n≈50 is directional. The
corpus that would discriminate structure's value — months of sparse
contact with real drift — does not exist; LongMemEval's ~10-day median
span flatters recency-biased designs. And aggregation fails for
everyone ([§5.3](#53-supersession-and-aggregation-the-shape-failures)), so no arm "wins" conversational retention outright.

## 9. Reproducing

```bash
uv run pytest -q && uv run ruff check src tests scripts analysis  # 406 tests, the green gate
uv run python scripts/preflight.py --leg swe    # cost/wall-clock gate BEFORE any live leg
uv run python -m context_eval.run_eval --path data/longmemeval_dev100.jsonl \
  --strategies flat-compact,lcm,sawtooth-ledger5 --budgets 64000 \
  --samples 100 --out results/x.jsonl
python3 analysis/lme_deep.py                   # recompute [findings §5](docs/findings.md#5-longmemeval-the-honest-fresh-numbers) from raw rows
```

Corpora rebuild from loaders ([`data/README.md`](data/README.md)); ~$222
total measured API spend across the project's life; raw run rows live in
`results/` (gitignored, mapped in
[`results/README.md`](results/README.md)).

## 10. Repo map & further reading

- [`docs/findings.md`](docs/findings.md) — **the full evidence report**:
  every table, n, correction, and the loss-mode taxonomy, recomputable via
  [`analysis/`](analysis/) (imports nothing from the harness by design).
- [`docs/lit-review.md`](docs/lit-review.md) — per-arm sources, ports,
  deviations, and the complete run-by-run history (includes the old R1–R11
  narrative era).
- [`docs/resident-memory-theory.md`](docs/resident-memory-theory.md) — the
  activation-law resident memory behind ledger5: axioms, dials,
  falsifiable predictions.
- [`docs/dataset-plan.md`](docs/dataset-plan.md) — the missing-corpus gap
  and what would fill it. [`docs/budgets.md`](docs/budgets.md) — real
  deployment budget tiers.
- [`data/README.md`](data/README.md) / [`results/README.md`](results/README.md) —
  tracked manifests for the gitignored corpora and run archive.
- `src/context_eval/` — the harness; `tests/` — 406 tests; `AGENTS.md` —
  engineering instructions for working on this repo.

MIT licensed. 7,062 recorded runs, one model, ~$222.
