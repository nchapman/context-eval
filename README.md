# context-eval

**Does hierarchical compaction beat the `/compact` default? And what does it actually take for an agent to remember what it was working on?**

We built a clean-room benchmark of eight context-management strategies — the
truncation baselines, the deployed industry default, archive-search designs,
a hierarchical summarizer (LCM), and an incremental-representation memory
(a Honcho-mechanism port) — and ran all of them through the same harness on
the same model with the same cost accounting: 600+ runs across LoCoMo
dialogues, SWE-bench-derived traces, and real multi-day agent sessions up to
2M tokens. Four findings survive every budget tier we could afford:

1. **Hierarchy doesn't pay.** The deployed flat summary (`/compact`) matches
   or beats the LCM summary DAG everywhere (0.95 vs 0.94 @128K, 1.00 vs 0.92
   @256K), and plain retrieval over a verbatim archive ties both at zero
   summarization overhead.
2. **The default won while broken.** A reasoning-cap bug silently blanked
   flat's summaries for the entire main suite — and it *still* won, on
   verbatim tail + archive search alone. Summaries were never carrying the
   deployment-tier scores.
3. **Inventory is not promptable.** Instructing fold-based summarizers to
   enumerate open topics *degrades* continuity mechanically: the instruction
   adds output burden to length-capped calls, truncation explodes, the map
   blanks (up to 100% empty folds measured).
4. **Inventory is architectural.** A maintained memory document — merged
   incrementally, never re-derived — wins uncued topic recall at every
   budget (54%/68% vs flat's 34%/62%) with the lowest abstention, at QA
   parity, for a priced ~3× summarization premium. Maintained state carries
   what regenerated output cannot.

Everything here is reproducible from this repo: `scripts/run_suite.sh` with
a preflight cost gate, ~$25 of total API spend at time of writing.

---

## §1 Why this exists

Every production coding agent ships some flavor of context management:
Claude Code has `/compact` and auto-compact, Cline edits its history in
place, OpenHands condenses, and a growing literature proposes hierarchical
summarization trees with recall tools. They exist because sessions outgrow
windows — our pi corpus contains real agent sessions from 128K to 2M
tokens, and even a 1M-token window overflows on them.

Context Compaction Theory (arXiv:2608.01326) warns that all deployed
compaction shares a failure mode: re-summarizing summaries compounds loss
until "after 2 compactions, all reasoning is gone." The field's answer is
usually more structure — hierarchies, citations, memory tiers. But nobody
had run the deployed default against the research proposals under one
harness, one model, and one cost ledger. So we did, in three phases:

1. **The suite** (511 runs): seven strategies × four budget tiers × three
   corpora, needles and QA. *Does structure buy accuracy?*
2. **The continuity probe**: an uncued "what are we working on?" question
   after seeding 10 topic threads into a session. *Does any strategy
   preserve the session's own inventory?* This is the question users
   actually feel — an agent that forgets the task list mid-session.
3. **The mechanism experiments**: prompting vs structure for inventory
   persistence, and a clean-room port of the one architecture that
   explicitly claims to solve it.

## §2 The harness

**Clean-room arms.** Every strategy is implemented from its paper or docs,
not imported — the same discipline for LCM's DAG, ARC's citations, and the
Honcho-mechanism port. Arms differ in *mechanism only*: shared trigger
thresholds where the design permits, shared verbatim-tail fractions,
shared archive-search tools, same model, same seeding.

**One model everywhere.** `deepseek/deepseek-v4.1-flash` via OpenRouter
summarizes, answers, and judges in every arm. When the memory-builder and
the answerer are the same model, differences are architecture, not model
class. (The judge is also flash — see §6.)

**One cost authority.** `TokenCounter.count_message()` — content + wire
overhead + tool-call arguments — is the only token accounting. Budgets
bound the *assembled view* only, as in production tools; fixed harness
costs sit outside. Every run records per-arm overhead usage (tokens, USD,
cache hits) separately from answering usage.

**Corpora.** LoCoMo long-form dialogues (8K/16K budgets — the stress
tier), SWE-bench-derived traces (64K/128K — deployment tier), and real pi
agent sessions (128K/256K, up to 2M raw — uncapped tier), plus
thinking-drop variants of the pi sessions for the reasoning-representation
ablation.

**Probes.**
- *Needles*: verbatim facts injected at random depths; accuracy is reported
  by depth so "search recovered it" and "it was still in view" are
  distinguishable.
- *QA*: the corpus' own questions (LoCoMo), graded ≥0.5 by the judge.
- *Continuity*: 10 topic threads seeded through a session, then — with no
  cue — "list everything this session has worked on, with status." Scored
  as recall − 0.25×false-memories; every answer is split into **abst**
  (blank: honest miss) vs **confab** (confident wrong answer). Accuracy
  alone conflates the two; the split is the diagnosis.
- *Overflow honesty*: runs that blow the budget answer nothing by design;
  `acc*` counts their questions as wrong rather than dropping them.

**Discipline.** Preflight (`scripts/preflight.py`) estimates runs, calls,
minutes, and dollars per leg and refuses to launch over ceiling without an
explicit override. Prompt changes apply symmetrically to all summarizing
arms. Ablations are separate reports, never merged into shared cells.
`empty_summaries` telemetry counts every compaction whose completion
returned no content — nonzero means that arm is running with blank maps
and every downstream number for it is suspect. (That invariant is finding
F2's origin story.)

## §3 The eight arms

The design space, mapped:

```
                        what the agent SEES each turn
              verbatim archive-heavy  ──────►  derived map-heavy
            ┌─────────────────────────────────────────────────┐
            │  full-context    retrieval      flat-compact     │
   how the  │  sliding-window  arc           lcm               │
   state    │  context-editing                              ▓▓ │
   UPDATES  │                                       representation│
   ──────►  │  evict / prune   search on     re-derive or      │
            │                  demand        maintain          │
            └─────────────────────────────────────────────────┘
             lose everything   recoverable   summarize the past,
             past the edge     past          or *be* the past's state
```

Each arm isolates one design question.

### full-context — the control
Nothing is managed; the whole transcript is the view.
**Isolates:** the ceiling (and the cost of reaching for it). It overflows
in every stress cell, 8/25 deployment cells, and *all* uncapped cells —
even a 1M window cannot hold our real sessions. Where it fits, it wins;
where it doesn't, `acc*` collapses to 0.58.

```
view = [entire transcript]            (until it doesn't fit — then: nothing)
```

### sliding-window — the floor
Keep the newest N tokens verbatim, evict the rest forever.
**Isolates:** what *any* summarization buys over plain eviction. Answer:
a lot. Window is the only arm that collapses at deployment tier (0.69
@128K vs flat's 0.95) and dies outright on thinking-drop sessions (0.17).

```
view = [newest ──────────── N tokens]
        (everything older: gone)
```

### context-editing — surgical eviction
Rewrite history in place: strip old tool-call payloads first (Cline's
approach), keep the words around them.
**Isolates:** how far in-place pruning gets without any summarization
call. Respectable at mid tier (0.77 @64K), insufficient at deployment
(0.75 @128K) — payload stripping can't compress *reasoning*, only noise.

```
view = [user/assistant turns; old tool outputs → "[pruned]"]
```

### flat-compact — the deployed default
What Claude Code `/compact`, Cline, Codex, and OpenHands' condenser all
do: when the projected view crosses ~90% of budget, one summarization call
folds everything older than a kept 25% tail into a running summary; the
agent continues from `[summary] + tail`. Repeated triggers re-summarize
`PREVIOUS SUMMARY + new history` — the summary-of-summary compounding that
Compaction Theory warns about.

**Isolates:** the baseline every research proposal must beat. It also
carries our two cheapest lessons: F2 (it won the suite with blank
summaries — tail + archive search carried it) and F4 (its continuity
ceiling is structural).

```
turn n:   [ ────────── tail ────────── ]          view grows freely
trigger:  [SUMMARY v1] [ ── tail ── ]
later:    [SUMMARY v2 = f(SUMMARY v1, history)]   ← re-derives the whole map
              + [ ── tail ── ]                        each time it fires
```

### retrieval — no summary at all
The view is a small recent window plus an FTS index over the *verbatim*
archive; the model searches before answering. Zero summarization calls,
zero summarization loss.
**Isolates:** whether summaries matter at all when the archive is
searchable. Deployment-tier answer: no — 0.95 @128K, 1.00 @256K, tying
flat at $0.00 compaction overhead vs its $0.0065/run. Summaries' value
shows up only under continuity, not QA (F5).

### arc — addressable citations
Every message keeps an anchor (`§m000123`); the view shows citations with
snippets; `history_recall(id)` fetches full originals deterministically
(Cognition's blog design).
**Isolates:** whether *deterministic* recall tools get used. They don't:
**0 tool calls in 511 runs** (F3). Discoverability is the boundary — a
tool the model doesn't spontaneously call does not exist, no matter how
clean its contract.

### lcm — the hierarchical challenger
Segments summarize into leaf nodes (D1); sibling summaries consolidate into
parents (DN); escalation re-derives on demand; the view is the active
frontier plus pointers; recall tools traverse the DAG. The LCM paper's
clean-room port, tuned per its ablations (lean citations below ρ).

**Isolates:** the research-proposal answer to compounding — does a tree of
small summaries beat one flat fold? Deployment-tier answer: no. 0.94
@128K / 0.92 @256K against flat's 0.95/1.00, at 1.5× the compaction cost
($0.0097 vs $0.0065 @128K) and 21+ calls per stress run. Its continuity
collapse at small budgets (12% @8K) is the probe's founding observation.

### representation — maintained state (the Honcho-mechanism port)
Honcho (Plastic Labs) routes messages to a background deriver that merges
conclusions into per-entity representations incrementally — the map is
never re-derived from the transcript. Our port keeps the mechanism and
drops the deployment machinery (async queue, dialectic tiers, dreamer):

```
             flat-compact                    representation
        ┌──────────────────┐          ┌──────────────────────────┐
        │ fold EVERYTHING  │          │ merge ONLY THE DELTA     │
        │ since last fold  │          │ into maintained memory   │
        │                  │          │                          │
 t1:    │ S1 = f(h1)       │   t1:    │ M1 = merge(∅, h1)        │
 t2:    │ S2 = f(S1,h2)  ←─┼─ re-     │ M2 = merge(M1, h2) ←─┼─ input is
 t3:    │ S3 = f(S2,h3)    │  derive  │ M3 = merge(M2, h3)   │   O(memory
        │   ...grows with  │  whole   │   ...input stays      │   + delta)
        │   history        │  map     │   ~4.5K tokens        │
        └──────────────────┘          └──────────────────────────┘
```

The memory is a bounded doc (Goal & decisions | Entities & values | Open
threads `name|status`); updates fire per ~2K new tokens; the view is
`[memory] + verbatim tail`, same tools as flat. **Isolates:** whether
*maintaining* the map beats *regenerating* it — the exact capability F4
showed fold-based summarizers cannot carry.

## §4 Findings

### F1 — Hierarchy doesn't pay (and search is free)

| @128K SWE (n=22-25) | acc | acc* | overhead $/run |
|---|---|---|---|
| flat-compact | **0.95** | **0.95** | $0.0065 |
| retrieval | **0.95** | **0.95** | $0.0000 |
| lcm | 0.94 | 0.94 | $0.0097 |
| context-editing | 0.75 | 0.66 | $0.0000 |
| sliding-window | 0.69 | 0.69 | $0.0000 |
| full-context | 0.81 | 0.58 | — (7/25 overflow) |

At 256K on real agent sessions: flat 1.00, retrieval 1.00, lcm 0.92. The
DAG's extra calls buy nothing the flat fold doesn't already provide, and
retrieval demonstrates that at these budgets a verbatim archive + search
is worth a summary. Deep-needle splits agree: arms that keep/search
verbatim recover deep facts; summarizers' deep recall tracks whatever
their summaries happened to retain.

### F2 — The default won while broken

flash is a reasoning model, and `max_tokens` caps reasoning *and* content
together. At the original 2048 summary cap, big folds spent it all
thinking:

```
completion budget ▐ 2048 tokens ▌
reasoning        ██████████████████████ 2039
content          ▏ 18 chars, finish=length
```

`compactions=4, summary_chars=0`: flat ran the entire main suite
**summary-less** — and still won F1, on tail + archive search. Two
consequences. First, the honest headline: flat's deployment-tier wins never
depended on its summaries. Second, the invariant: `empty_summaries` now
counts blank compactions in every run, and a nonzero count invalidates the
arm's cells (this bug hid behind `or self._summary` guards for 511 runs).
At 4096 the summaries land — and empty-fold rates of 42–61% at stress
budgets remain (F4, F5), because the physics scales with input size.

### F3 — Tools that aren't called don't exist

ARC's deterministic `history_recall`: 0 invocations across 511 runs. The
model searches (flat's `history_search`: ~5-8 calls/run at stress) but
never escalates to targeted recall — not fixable by prompt description
(we tried; documented in docs/lit-review.md as a model-behavior boundary).
The chain is **discoverability → usage → recovery**: only the first link
is the tool designer's. Deployment lesson: inject, don't advertise.

### F4 — Inventory is not promptable

The continuity probe asks, uncued, for the session's topic inventory. We
compared flat/lcm with and without an explicit "end every summary with a
TOPICS line" instruction (same model, same cap, n=10/cell):

```
LoCoMo continuity recall (recall / % empty folds)
                      no instruction     + TOPICS instruction
flat @8K   34%              │  19%   ← instruction made it WORSE
lcm  @8K   12%              │   0%
flat @16K  62%              │  30%   ← 100% of folds truncated empty
lcm  @16K  52%              │  15%
```

The instruction doesn't inform the summarizer — it *loads* it. More output
target + more reasoning on a length-capped call means more truncation
(flat hit 100% empty folds at 16K), blanker maps, more abstention. On
reasoning models, prompt content and completion-cap physics are entangled:
you cannot buy state with words when the words are what gets truncated.

### F5 — Inventory is architectural

The representation arm, same grid, same cap (n=10/cell):

| LoCoMo continuity | recall | abst | QA | empty merges | overhead tok/run |
|---|---|---|---|---|---|
| representation @8K | **54%** | **20%** | 70% | 43% | ~55K |
| flat @8K | 34% | 40% | 68% | 61% | 34K |
| lcm @8K | 12% | 80% | 68% | 50% | 70K |
| representation @16K | **68%** | **10%** | 68% | 29% | ~55K |
| flat @16K | 62% | 30% | 70% | 42% | 18K |
| lcm @16K | 52% | 40% | 69% | 42% | 30K |

Best continuity at both budgets, lowest abstention everywhere, QA parity —
for ~3× flat's summarization overhead at 16K (the amortization premium:
~8-9 small merges per run instead of 1-2 big folds). And the failure modes
differ in *kind*: both arms lose folds/merges to the same truncation
physics, but a lost merge costs one delta (memory preserved, tail and
tools intact) while a lost fold wipes the arm's entire map. That asymmetry
— graceful vs catastrophic degradation under identical physics — is why
the recall gap doesn't close even with 29-43% of merges blank.

Together F4+F5 are the project's core claim: **the session inventory
persists only when it is maintained state, not when it is regenerated
output.**

### F6 — Honest misses ≠ confabulation

`acc` hides failure mode. At stress tier, flat abstains on 32% (blank,
scores 0 — the honest miss) while window splits 15% abst / 48% confab.
Overflow-penalized `acc*` exposes the strategies that only look cheap:
full-context 0.81 → 0.58 @128K, arc 0.91 → 0.80. An agent that
confidently hallucinates a superseded decision is a different product
failure than one that says nothing; the report splits them everywhere.

### F7 — Thinking-drop robustness

Real pi sessions with reasoning-turn representations dropped vs preserved
(thinking/ directory): summarizers hold up (flat 0.92, lcm 1.00), window
collapses to 0.17. Whatever the memory strategy, it must survive the
transcript *as the deployment actually persists it* — including dropped
or recapped reasoning — and summarization-based views are notably robust
to that specific mutation.

### F8 — Reasoning separation (in flight)

An 8192 summary cap across all three summarizing arms is running as this
is written; results amend this section. The question: with full reasoning
headroom, do empty folds/merges collapse — and does the F5 ordering
survive at full strength, or does flat close the continuity gap once its
summaries reliably land? Both outcomes are findings; the experiment
design guarantees attribution (cap is the only variable vs the F5 table).

## §5 What we'd tell an agent builder

Priced recipe, current evidence:

1. **Verbatim tail + searchable archive is the floor, not the strategy.**
   It ties full summarization on QA at $0 compaction cost (F1). Flat's
   real win was never its summary (F2).
2. **If you summarize, budget reasoning separately from content.** A
   shared cap is a silent killer: measured 42–100% empty folds, invisible
   without telemetry. Cap at 2× your content target minimum; count empty
   completions; fail loudly.
3. **Don't rely on the model calling memory tools.** 0/511 (F3). Inject
   the map into the view; search on the model's own initiative happens,
   targeted recall doesn't.
4. **If continuity matters, maintain the inventory — don't regenerate
   it.** A bounded memory doc merged per ~2K new tokens beats fold-based
   summarizers on uncued recall by +20/+6 pts with less abstention, at a
   measured ~3× summarization premium (F5). Prompting your summarizer to
   "keep a topic list" makes things worse, mechanically (F4).
5. **Budget for degradation.** Choose the strategy whose *failure* you can
   afford: lost-delta (representation) vs wiped-map (flat/lcm) vs
   lost-everything (window).

## §6 Threats to validity

- **One model class.** flash builds every arm's memory and answers every
  question. Stronger summarizer models could reorder F1/F5 — but the
  truncation physics (F2/F4) is a property of reasoning models as a class,
  and our single-model design is what makes the comparisons causal.
- **Judge = solver.** The judge is the same flash; ≥0.5 grading is coarse.
  Systematic bias would have to favor one arm's phrasing to reorder
  results.
- **Estimator gaps.** `CharEstimator` (≈4 chars/token) vs provider counts
  drifts a few percent; budgets are view-bounds and the same estimator
  prices every arm symmetrically. Calibration against recorded usage is on
  the backlog.
- **n at uncapped tier.** 3 pi sessions per cell; CIs are wide ([0.76,
  1.00] on flat's 256K 1.00). The flat-vs-lcm ordering there is
  consistent with the n=25 deployment tier, but treat magnitudes
  cautiously.
- **The port ≠ Honcho.** Our representation arm is single-document,
  synchronous, no dialectic tiers or dreamer. Results license claims
  about the incremental-merge mechanism, not about their product (their
  async deriver with big-model budgets is precisely the deployment
  answer to F2's physics).
- **Synthetic markers.** Continuity topics are seeded distinctive terms
  ("VELA-9", "amphora"); recall-of-real-inventory may differ in ways that
  flatter verbatim matching. The abst/confab split partially controls
  this.

## §7 Reproducing

```
src/context_eval/
  strategies/        the eight arms (representation.py is the newest)
  run_eval.py        one leg: corpus × strategies × budgets × samples
  questions.py       needle/continuity seeding; judge grading
  report.py          aggregation: acc/acc*, abst/confab, depth, sanity gates
  driver.py          replay loop, overflow policy, telemetry capture
  provider.py        OpenRouter client: SSE guards, retry ladder caps
scripts/
  run_suite.sh       the full battery (leg groups, 3 workers wide)
  preflight.py       cost/wall-clock gate per leg (refuses over ceiling)
tests/               220 tests: per-arm behavior, parity drift-catchers,
                     budget-arithmetic oracles, provider wallclock guards
docs/lit-review.md   per-strategy sources, ports, deviations, findings log
```

```bash
uv run pytest && uv run ruff check src tests scripts   # the green gate
uv run python scripts/preflight.py --leg locomo        # cost estimate first
./scripts/run_suite.sh --skip-swe --skip-pi            # then the battery
```

Total API spend to date: ~$25. A single LoCoMo leg (8 strategies × 2
budgets × 10 samples) is ~$1.70 and ~56 min at 3 workers. Adding a ninth
arm: implement the Strategy interface (~150-400 lines + tests), register
it, wire it into a suite group — the harness, telemetry, and report treat
it identically, which is the entire point.
