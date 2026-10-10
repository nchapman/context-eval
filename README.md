# context-eval

**Does hierarchical compaction beat the `/compact` default? And what does it actually take for an agent to remember what it was working on?**

We built a clean-room benchmark of eight context-management strategies — the
truncation baselines, the deployed industry default, archive-search designs,
a hierarchical summarizer (LCM), and an incremental-representation memory
(a Honcho-mechanism port) — and ran all of them through the same harness on
the same model with the same cost accounting: 600+ runs across LoCoMo
dialogues, SWE-bench-derived traces, and real multi-day agent sessions up to
2M tokens. Four findings survive every budget tier we could afford:

1. **On cued recall, every archive-preserving strategy ties — structure
   buys nothing.** Flat `/compact`, plain retrieval over a verbatim
   archive, and the LCM hierarchy are statistically indistinguishable at
   deployment budgets (0.94–0.95 @128K; 1.00/1.00/0.92 @256K), while
   eviction-only (0.69) and in-place editing (0.75) lose and full-context
   is infeasible (overflow). Retrieval matches the summarizers at $0
   compaction cost.
2. **On uncued inventory recall, strategies separate by how they maintain
   the map.** Maintained memory (incremental merge) > flat fold >
   hierarchical consolidation: 54%/34%/12% at 8K, 68%/62%/52% at 16K,
   with maintained memory also failing most honestly (10–20% abstention
   vs 30–80%).
3. **Regenerated maps cannot be instructed into carrying state; maintained
   maps carry it natively.** Adding a "keep a topic inventory" instruction
   to fold-based summarizers *degrades* them (30%/19% recall, vs 62%/34%
   uninstructed): instruction load increases truncation, which blanks the
   map. The inventory question is architectural, not promptable.
4. **Completion truncation is the shared failure physics — and the
   differentiator.** On reasoning models, 29–61% of summarization calls
   return empty across every summarizing strategy. Fold maps degrade
   catastrophically (one empty fold wipes the map); maintained memory
   degrades gracefully (one empty merge loses one delta). Under identical
   physics, recall diverges.

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
the origin of the empty-summary invariant.)

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
anchors R1 (its QA tie was achieved with tail + archive search even
when its summaries blanked — see the measurement note) and R2/R3
(its continuity ceiling is structural, and instruction-immune).

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
shows up only under continuity, not QA (R2).

### arc — addressable citations
Every message keeps an anchor (`§m000123`); the view shows citations with
snippets; `history_recall(id)` fetches full originals deterministically
(Cognition's blog design).
**Isolates:** whether *deterministic* recall tools get used. They don't:
**0 tool calls in 511 runs** (R5). Discoverability is the boundary — a
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
*maintaining* the map beats *regenerating* it — the exact capability R3
showed fold-based summarizers cannot carry.

## §4 Findings

### R1 — On cued recall, archive-preserving strategies tie; structure buys nothing

| @128K SWE (n=22-25) | acc | acc* | overhead $/run |
|---|---|---|---|
| flat-compact | **0.95** | **0.95** | $0.0065 |
| retrieval | **0.95** | **0.95** | $0.0000 |
| lcm | 0.94 | 0.94 | $0.0097 |
| context-editing | 0.75 | 0.66 | $0.0000 |
| sliding-window | 0.69 | 0.69 | $0.0000 |
| full-context | 0.81 | 0.58 | — (7/25 overflow) |

At 256K on real agent sessions: flat 1.00, retrieval 1.00, lcm 0.92. When
the question points at the information (needles, corpus QA), any strategy
that keeps or can find the verbatim past performs equally — the DAG's
extra calls buy nothing a flat fold doesn't provide, and a searchable
archive is worth a summary at zero summarization cost. What separates the
tiers is what happens to information outside the view: eviction loses it
(window 0.69), pruning loses reasoning (editing 0.75), and overflow loses
everything (full-context 0.81→0.58 penalized).

### R2 — On uncued recall, strategies separate by how they maintain the map

The continuity probe asks, with no cue, for the session's topic inventory.
Same grid, same model, same cap (n=10/cell):

| LoCoMo continuity | map regime | recall | abst | QA |
|---|---|---|---|---|
| representation @8K | maintained (merge) | **54%** | **20%** | 70% |
| flat-compact @8K | regenerated (fold) | 34% | 40% | 68% |
| lcm @8K | regenerated (tree) | 12% | 80% | 68% |
| representation @16K | maintained (merge) | **68%** | **10%** | 68% |
| flat-compact @16K | regenerated (fold) | 62% | 30% | 70% |
| lcm @16K | regenerated (tree) | 52% | 40% | 69% |

The ordering — maintained > flat-fold > hierarchical — holds at both
budgets, as does the abstention ordering (same direction). The
hierarchy's structure, built to preserve detail at scale, is the *worst*
uncued-inventory keeper at stress budgets: consolidation layers are
exactly where enumerated topics stop surviving. Overhead prices the
regimes: ~55K tokens/run for ~8-9 amortized merges (representation) vs
18-34K for 1-2 folds (flat) vs 30-70K for the D1+DN ladder (lcm).

### R3 — Regenerated maps can't be instructed into carrying state

If the R2 gap were a prompting deficiency, an explicit instruction should
close it. It does the opposite. Adding "end every summary with a TOPICS
line enumerating named projects and their status" to both fold-based
summarizers (same model, same cap, n=10/cell):

```
LoCoMo continuity recall (recall / % empty folds)
                      no instruction     + TOPICS instruction
flat @8K   34%              │  19%   ← instruction made it worse
lcm  @8K   12%              │   0%
flat @16K  62%              │  30%   ← 100% of folds truncated empty
lcm  @16K  52%              │  15%
```

The instruction doesn't inform the summarizer — it loads it. More output
target plus more reasoning on a length-capped call means more truncation
(flat: 100% of folds empty at 16K), blanker maps, more abstention. On
reasoning models, prompt content and completion-cap physics are
entangled: you cannot buy state with words when the words are what gets
truncated. Maintained memory needs no such instruction — the inventory is
its content, not an output format.

### R4 — Under shared truncation physics, map regime decides the failure shape

On this model class, 29–61% of summarization calls return empty
(measured per arm: flat 42–61%, lcm 42–50%, representation 29–43% of
folds/merges) — reasoning consumes the completion cap before content.
This is a property of *every summarizing strategy* on reasoning models,
and it is where the map regimes diverge structurally:

```
fold arm, one empty compaction:   [MAP WIPED] + tail + tools
merge arm, one empty merge:       [map − that delta] + tail + tools
```

R2's recall ordering persists *despite* representation losing 29–43% of
its merges: each loss costs one bounded delta while the maintained map
survives; a fold arm's empty compaction costs its entire map until the
next successful fold. Catastrophic vs graceful degradation under
identical external physics is a strategy property, not luck.

### R5 — Models search; they don't recall

Across 511 suite runs, the models spontaneously issue archive search
(flat: ~5-8 `history_search` calls/run at stress; retrieval's whole
design) but **never once** call ARC's deterministic `history_recall`
(0/511) — not fixable by tool description (documented as a
model-behavior boundary in docs/lit-review.md). Tool-mediated recovery
has a discoverability→usage→recovery chain, and only the first link
belongs to the tool designer. Strategies that inject their map into the
don't depend on it; strategies that gate recovery behind targeted tool
calls surrender it.

### R6 — Who fails honestly

`acc` conflates failure modes; the abst/confab split separates them.
At stress budgets, summarizers abstain (flat 32% blank, scoring 0 — the
honest miss) while window splits 15% abst / 48% confab — an agent that
confidently states a superseded decision is a different product failure
than one that says nothing. Overflow-penalized `acc*` exposes the same
honesty dimension across tiers: full-context 0.81→0.58, arc 0.91→0.80,
flat/lcm/retrieval unchanged (they never overflow).

### R7 — Thinking-drop robustness

Real pi sessions with reasoning-turn representations dropped vs
preserved: summarizer-backed strategies hold up (flat 0.92, lcm 1.00),
window collapses to 0.17. A memory strategy must survive the transcript
as deployments actually persist it — including dropped or recapped
reasoning — and derived-view strategies are notably robust to that
mutation while verbatim-window views are not.

### R8 — Reasoning separation (in flight)

An 8192 completion cap across all three summarizing arms is running as
this is written; results amend this section. The question: with full
headroom, does R4's truncation rate collapse — and does R2's ordering
survive when folds land as reliably as merges? Either answer sharpens
the map-regime claim.

### A note on measurement hazards

Three of the above numbers exist only because the harness caught us
being wrong: the main suite's flat arm ran with silently empty summaries
(caught later; R1's tie among summarizers was achieved by tail+search
alone, and `empty_summaries` telemetry now makes it an invariant);
a first port of the representation arm re-created fold physics by
tau-batching deltas; a delta trigger miscounted the kept tail and
thrashed merges. We report them here not as findings but as the reason
§2's discipline exists: on reasoning models, *blank completions are the
default failure mode of summarization*, and an eval that doesn't count
them will misattribute everything downstream.

## §5 What we'd tell an agent builder

Priced recipe, current evidence:

1. **Verbatim tail + searchable archive is the floor, not the strategy.**
   It ties full summarization on QA at $0 compaction cost (R1).
2. **If you summarize, budget reasoning separately from content.** A
   shared cap is a silent killer: measured 42–100% empty folds, invisible
   without telemetry. Cap at 2× your content target minimum; count empty
   completions; fail loudly.
3. **Don't rely on the model calling memory tools.** 0/511 (R5). Inject
   the map into the view; search on the model's own initiative happens,
   targeted recall doesn't.
4. **If continuity matters, maintain the inventory — don't regenerate
   it.** A bounded memory doc merged per ~2K new tokens beats fold-based
   summarizers on uncued recall by +20/+6 pts with less abstention, at a
   measured ~3× summarization premium (R2). Prompting your summarizer to
   "keep a topic list" makes things worse, mechanically (R3).
5. **Budget for degradation.** Choose the strategy whose *failure* you can
   afford: lost-delta (representation) vs wiped-map (flat/lcm) vs
   lost-everything (window).

## §6 Threats to validity

- **One model class.** flash builds every arm's memory and answers every
  question. Stronger summarizer models could reorder R1/R2 — but the
  truncation physics (R4) is a property of reasoning models as a class,
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
  answer to R4's physics).
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
