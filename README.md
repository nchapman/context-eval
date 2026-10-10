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
2. **On uncued inventory recall, architecture separates exactly when
   compaction begins.** With calibrated compaction calls (see 4), the
   hierarchy keeps 99% of the session's topic inventory at both stress
   budgets; the flat fold holds 96% @16K but drops to 53% @8K once
   folding starts — each summary-of-summary pass loses enumerated topics —
   and incremental-merge memory erodes to 39–68% through its own churn
   (9–10 whole-memory regenerations per run). Cued recall saturates
   everywhere: needles 100% on every arm, corpus QA parity.
3. **Regenerated maps cannot be instructed into carrying state.** Adding
   a "keep a topic inventory" instruction to fold-based summarizers
   *degrades* them (30%/19% recall, vs 62%/34% uninstructed, under the
   shared-cap regime): instruction load increases truncation, which
   blanks the map. Under calibrated calls the uninstructed gap persists
   structurally (hierarchy 99% vs fold 53% @8K) — inventory is
   architecture, not wording.
4. **Completion truncation is the shared failure physics — and
   calibration, not bigger caps, is the fix.** On reasoning models,
   29–61% of summarization calls return empty when reasoning shares the
   completion cap; 8192-token caps don't cure it (one probe: 8192
   reasoning tokens, zero content). Disabling reasoning on utility calls
   (`reasoning: {enabled: false}`) with a stated word budget and an
   empty-at-length retry took 60/60 grid runs to zero blanks at ~¼ the
   thinking-token bill. An uncalibrated utility call doesn't add noise —
   it *inverted* finding 2's ordering before we caught it.

Everything here is reproducible from this repo: `scripts/run_suite.sh` with
a preflight cost gate, ~$30 of total API spend at time of writing.

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
when its summaries blanked — see the measurement note) and R2 (its
continuity is intact until folding starts — 96% @16K at 1–2 folds,
53% @8K — and the TOPICS instruction halves it, R3).

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
shows up only under uncued continuity at stress budgets, not QA (R2).

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
($0.0097 vs $0.0065 @128K) and 21+ calls per stress run. Under
calibrated compaction it is the strongest uncued-inventory keeper at
stress budgets (99% at both, R2) — a ranking the shared-cap regime had
inverted (its 5-call ladder starved worst there: 12% @8K, the probe's
founding observation and the confound that started R4/R8).

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
*maintaining* the map beats *regenerating* it. Measured under calibrated
calls: merge-churn is its own loss channel — 9–10 whole-memory
regenerations per run erode inventory (39–68% recall, R2), and its
resilience under the shared-cap regime was the confound's cumulative
masking, not architectural superiority.

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

### R2 — On uncued recall, architecture separates exactly when compaction begins

The continuity probe asks, with no cue, for the session's topic inventory.
Same grid, same model, calibrated compaction calls (n=10/cell; the
confounded-regime history of this table is in the measurement note):

| LoCoMo continuity | map regime | @8K recall / abst | @16K recall / abst |
|---|---|---|---|
| lcm | regenerated (tree) | **99%** / 0% | **99%** / 0% |
| flat-compact | regenerated (fold) | 53% / 20% | 96% / 0% |
| representation | maintained (merge) | 39% / 30% | 68% / 20% |

Needle recovery is **100% on every arm at both budgets** — no strategy
loses injected specifics, even flat while folding at 8K — and corpus QA
is parity (24–32%, within noise). What separates the arms is *uncued*
inventory, and the ordering tracks how each map survives its own
compaction events:

- **lcm** keeps enumeration nearly perfect because every call is small:
D1 summarizes one bounded segment, DN consolidates four summaries with
`[§id]` references — under calibrated calls every node lands full-fat,
and nothing in the tree ever re-derives the whole map.
- **flat** only loses when it folds: at 16K it folds 1–2×/run and sits at
96%; at 8K folding starts earlier and each summary-of-summary pass drops
enumerated topics (53%, 20% abstention).
- **representation** merges 9–10×/run and every merge regenerates the
whole memory doc — the churn itself erodes inventory (39–68%), the worst
uncued keeper in this configuration: maintaining loses to both
regenerators here, and the cumulative structure that looked resilient
under the shared-cap regime was masking starvation, not preserving
inventory.

Overhead prices the regimes: 13–23K overhead tokens/run (flat, 1–2 folds)
vs 17–25K (lcm, ~5 D1+DN calls) vs 33–38K (representation, ~9 merges).
The hierarchy's continuity win costs ≈1.3× flat's compaction overhead —
not the 3× premium the confounded regime suggested.

### R3 — Regenerated maps can't be instructed into carrying state

If the R2 gap were a prompting deficiency, an explicit instruction should
close it. Adding "end every summary with a TOPICS line enumerating named
projects and their status" to both fold-based summarizers (same model,
n=10/cell) did the opposite — under the then-current 4096 shared cap,
where the instruction *loaded* the call (more target, more reasoning) and
truncation answered:

```
LoCoMo continuity recall (recall / % empty folds) — shared-cap regime
                      no instruction     + TOPICS instruction
flat @8K   34%              │  19%   ← instruction made it worse
lcm  @8K   12%              │   0%
flat @16K  62%              │  30%   ← 100% of folds truncated empty
lcm  @16K  52%              │  15%
```

Those magnitudes are regime-bound (see the measurement note); the
mechanism is not: on reasoning models, prompt content and
completion-cap physics are entangled, and you cannot buy state with
words when the words are what gets truncated. The structural half
survives calibration untouched: with compaction calls now reasoning-off
and budget-honoring, the no-instruction gap persists (flat 53% vs lcm
99% @8K, R2) — inventory is architecture, not wording. Whether a TOPICS
instruction *helps* flat once truncation no longer binds is measured in
the calibrated re-run below.

### R4 — Shared-cap truncation physics is real; calibration, not bigger caps, is the fix

On this model class, a summarization call that shares its completion cap
with reasoning returns empty 29–61% of the time (measured per arm: flat
42–61%, lcm 42–50%, representation 29–43%) — reasoning consumes the cap
before content. Doubling the cap does not fix it (8192 still blanks;
one calibrated probe: 8192 reasoning tokens, zero content — appetite,
not headroom). What fixes it is removing the appetite from the call:

```
calibrated compaction call = reasoning: {enabled: false}
                            + word budget stated in the prompt
                            + matched backstop cap (1024)
                            + require-content retry (escalate ×2)
```

Smoke calibration: 8/8 reasoning-off cells complete, honor the stated
~150-word budget, retain 100% of available inventory markers, and bill
zero discarded thinking tokens (previously 1.1–8.2K/call). Grid
validation: 60/60 runs, zero empty summaries, 6 escalations (all
recovered, billed honestly), 3 stray reasoning calls total — the profile
holds in production conditions, and `require_content` catches the rare
provider that ignores the flag.

The failure shapes this physics produced remain design-relevant: a fold
arm's empty compaction wipes its entire map until the next successful
fold; a merge arm loses one bounded delta. Catastrophic vs graceful under
identical external physics is a strategy property — but under calibration
neither shape fires, which is the point.

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
honest miss; suite-era number, partially summary-blank-driven — see R4)
while window splits 15% abst / 48% confab — an agent that
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

### R8 — Utility calls must be reasoning-off (outcome: calibration solved it)

R8 asked whether R4's truncation rate collapses at higher caps and
whether R2's ordering survives when folds land as reliably as merges.
Answer: bigger caps don't work (8192 still blanks — the model's reasoning
appetite scales to whatever cap it gets), and reasoning-off changes the
*findings*, not just the noise. Every summarizing arm's compaction call
now runs `reasoning: {enabled: false}` with a stated word budget and a
require-content retry (R4); the calibrated grid re-measured R2 end to
end, and the ordering inverted from the shared-cap regime — the old
ordering had measured call-count × reasoning appetite (lcm's 5-call
ladder starved worst; representation's cumulative memory masked the
starvation and looked best), not architecture. The engineering lesson
generalizes past this harness: any LLM utility call — summarization,
extraction, routing — must have its reasoning channel disabled or its
budget separated, or it silently no-ops while billing for thinking.

### A note on measurement hazards

Four of the above numbers exist only because the harness caught us
being wrong: the main suite's flat arm ran with silently empty summaries
(caught later; R1's tie among summarizers was achieved by tail+search
alone, and `empty_summaries` telemetry now makes it an invariant);
a first port of the representation arm re-created fold physics by
tau-batching deltas; a delta trigger miscounted the kept tail and
thrashed merges; and the shared-cap regime (R4) *inverted* the R2
ordering before calibration — an uncalibrated utility call doesn't add
noise to a comparison, it re-ranks the arms by call-count × reasoning
appetite while every number still looks plausible. `report.py` now
refuses to render any cell with blank compactions (`--allow-invalid`
exists for forensics). We report them here not as findings but as the
reason §2's discipline exists: on reasoning models, *blank completions
are the default failure mode of summarization*, and an eval that doesn't
count them will misattribute everything downstream.

## §5 What we'd tell an agent builder

Priced recipe, current evidence:

1. **Verbatim tail + searchable archive is the floor, not the strategy.**
   It ties full summarization on QA at $0 compaction cost (R1).
2. **Calibrate every utility call: reasoning-off, stated budget,
   require-content retry.** A shared completion cap is a silent killer —
   measured 42–100% empty folds, invisible without telemetry, and bigger
   caps don't fix it (8192 still blanks: the model's reasoning appetite
   scales to the cap). `reasoning: {enabled: false}` + a word budget
   stated in the prompt + an empty-at-length retry took 60/60 grid runs
   to zero blanks at ~¼ of the previous thinking-token bill (R4, R8).
3. **Don't rely on the model calling memory tools.** 0/511 (R5). Inject
   the map into the view; search on the model's own initiative happens,
   targeted recall doesn't.
4. **If continuity matters at stress budgets, structure the map as a
   hierarchy of small summaries.** lcm holds 99% uncued inventory at
   both budgets vs flat's 53% @8K (folding drops topics), at ≈1.3× flat's
   compaction overhead — each call is small, lands full-fat, and nothing
   re-derives the whole map (R2). Maintained-memory variants pay their
   own cost: merge-churn regenerates the whole doc 9–10×/run and erodes
   inventory. Prompting your summarizer to "keep a topic list" made it
   worse under truncation physics (R3) — the fix is architecture, not
   wording.
5. **Budget for degradation.** Choose the strategy whose *failure* you can
   afford: lost-delta (representation) vs wiped-map (flat) vs
   lost-everything (window) — and remember calibration (R4) prevents the
   first two from ever firing.

## §6 Threats to validity

- **One model class.** flash builds every arm's memory and answers every
  question. Stronger summarizer models could reorder R1/R2 — but the
  truncation physics (R4) is a property of reasoning models as a class
  (and so is its fix), and our single-model design is what makes the
  comparisons causal.
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
  answer to R4's physics)..
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
tests/               225 tests: per-arm behavior, parity drift-catchers,
                     budget-arithmetic oracles, provider wallclock guards
docs/lit-review.md   per-strategy sources, ports, deviations, findings log
```

```bash
uv run pytest && uv run ruff check src tests scripts   # the green gate
uv run python scripts/preflight.py --leg locomo        # cost estimate first
./scripts/run_suite.sh --skip-swe --skip-pi            # then the battery
```

Total API spend to date: ~$30. A single LoCoMo leg (8 strategies × 2
budgets × 10 samples) is ~$1.70 and ~56 min at 3 workers. Adding a ninth
arm: implement the Strategy interface (~150-400 lines + tests), register
it, wire it into a suite group — the harness, telemetry, and report treat
it identically, which is the entire point.
