# context-eval

**Does hierarchical compaction beat the `/compact` default? And what does it actually take for an agent to remember what it was working on?**

We built a clean-room benchmark of nine context-management strategies — the
truncation baselines, the deployed industry default, archive-search designs,
a hierarchical summarizer (LCM), and an incremental-representation memory
(a Honcho-mechanism port) — and ran all of them through the same harness on
the same model with the same cost accounting: 600+ runs across LoCoMo
dialogues, SWE-bench-derived traces, and real multi-day agent sessions up to
2M tokens. Four findings survive every budget tier we could afford:

1. **On cued recall, archive-preserving strategies near-tie — and the
   clean hierarchy drops nothing.** Flat `/compact`, plain retrieval,
   and the LCM hierarchy sit within 6 pts at deployment budgets
   (calibrated: 0.94 / 0.95 / 1.00 @128K), while eviction-only (0.69)
   and in-place editing (0.75) lose and full-context is infeasible
   (overflow). Every summarizer miss is a needle lost at a compaction
   event — folds and merges drop specifics occasionally, the tree's
   small calls dropped none in 176 runs — and calibration halved
   summarizer overhead, making the hierarchy the cheapest summarizer
   per run.
2. **On uncued inventory recall, architecture separates exactly when
   compaction begins.** With calibrated compaction calls (see 4), the
   hierarchy keeps 99% of the session's topic inventory at both stress
   budgets; the flat fold holds 96% @16K but drops to 53% @8K once
   folding starts — each summary-of-summary pass loses enumerated topics —
   and incremental-merge memory erodes to 39–68% through its own churn
   (9–10 whole-memory regenerations per run). Cued recall saturates
   everywhere: needles 100% on every arm, corpus QA parity.
3. **Instruction and call physics are entangled.** A "keep a topic
   inventory" instruction *degraded* fold-based summarizers under the
   shared cap (30%/19% vs 62%/34% — instruction load feeds the
   truncation that blanks the map) and *rescued* them under calibrated
   calls (flat @8K: 53%→84%, abstention 20%→0%). The same prompt is
   poison or the cheapest continuity win in the study depending entirely
   on the utility call's physics. Hierarchy still holds 99–100% with no
   instruction at all.
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

## §3 The nine arms

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
anchors R1 (94% @128K clean: its residual misses are all needles lost at
folds — and its suite-era 0.95 tie was achieved by tail + archive
search even while its summaries blanked, see the measurement note) and
R2/R3 (its
continuity is intact until folding starts — 96% @16K at 1–2 folds,
53% @8K — and a TOPICS instruction halves it under the shared cap but
rescues it to 84% under calibrated calls).

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
**0 tool calls in 511 suite runs** — but under forced occlusion the
tool fires freely (R5): the suite's silence was rational no-need, not
discoverability. What remains true: when the view reads as complete,
the model won't go looking — and can't tell a truncated preview from
a full one.

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

### conclusions — append-only observations, mechanically assembled
The Honcho-v3 mechanism done right: extract *typed observations*
(`TOPIC | status | fact [§id]`) from each aging segment and **append**
them to a store that is never rewritten — the working representation is
assembled in code at read time (latest-wins dedupe per topic), never
re-derived by an LLM. The view is the mechanically-deduped inventory
(always resident, grows with distinct topics only) + a bounded recent-
conclusions block + verbatim tail, with `conclusions_search` over the
bullets and `history_search`/`history_recall` over the verbatim archive.
Extraction calls are the only loss events — small, calibrated, and
capped per segment. **Isolates:** whether a *mechanical* assembly of
append-only observations beats LLM-maintained maps (representation) and
summary trees (lcm) on long horizons. It does — see R9 — and its single
failure surface is extraction recall: what the extractor never records,
the inventory never carries (measured and fixed mid-grid: one-line
status mentions losing the bullet-slot competition, 0.49→0.77
continuity from one prompt rule).

### sawtooth — one schedule, interchangeable backs (the R10 experiment)
A family of five registered arms sharing a *bit-identical* compaction
schedule — fill to budget, cut to a 12K verbatim head + a ≤4K back,
trigger priced at the back's cap so tooth boundaries cannot depend on
back content — with the back's content semantics as the only variable:
`none` (no back), `rewrite` (one incremental fold per tooth —
flat-compact's semantics), `segments` (per-chunk summaries appended
once, never re-derived — append-only but unstructured), and
`ledger`/`ledger2`/`ledger3` (typed TOPIC|status|fact observations +
mechanical latest-wins inventory; v2/v3 add naming continuity via a
read-only KNOWN TOPICS vocabulary, topic-balanced residency, citation
stripping, and phrase caps). All modes ship the same verbatim archive
and the same search tools. **Isolates:** what the back of the sawtooth
tooth contributes, with cadence, budget, and retrieval held constant —
and the answer (R10) is that only the typed ledger carries an
inventory; prose backs tie the empty back.

## §4 Findings

### R1 — On cued recall, archive-preserving strategies tie — and the
clean hierarchy is nominally perfect

| @128K SWE (n=22/cell) | acc | acc* | overhead $/run |
|---|---|---|---|
| lcm † | **1.00** | **1.00** | $0.0026 |
| flat-compact † | 0.94 | 0.94 | $0.0031 |
| retrieval | **0.95** | **0.95** | $0.0000 |
| representation † | 0.97 | 0.97 | $0.0093 |
| context-editing | 0.75 | 0.66 | $0.0000 |
| sliding-window | 0.69 | 0.69 | $0.0000 |
| full-context | 0.81 | 0.58 | — (7/25 overflow) |

† re-measured under calibrated compaction calls (zero empties); the
other arms make no summarization calls, so their suite-era numbers are
unaffected. At 64K the same ordering: lcm 1.00, rep 0.98, flat 0.97. At
256K on real agent sessions (flat 0.92, lcm 1.00, rep 1.00 †): everyone
wins. On real pi sessions the three summarizers all sit at 1.00 except
flat's 0.92 @256K.

The old tie claim survives in weakened form: differences among
archive-preservers are ≤6 pts and borderline (flat 0.94 vs lcm 1.00 at
n=88/cell, p≈0.06). What the clean data adds is the *shape* of the
residual gap: every summarizer miss is a needle lost at a compaction
event (flat 8, representation 5, lcm 0 across 176 SWE runs) — folds and
merges drop specifics occasionally; the tree's small calls drop none.
Trace-derived QA (questions about the session's *own* real facts —
actual errors, fixes, commands — 88 grounded pairs, gold-token recall)
finds the same tie on natural information: flat 0.86, lcm 0.85,
rep 0.82 @128K — real facts are retained about as well as injected
needles, with no architectural separation at deployment budget.
Calibration also halved summarizer overhead (flat $0.0065→$0.0031, lcm
$0.0097→$0.0026/run): the discarded-thinking tax is gone, and the
hierarchy is now the *cheapest* summarizer per run. What separates the
tiers is unchanged: eviction loses information (window 0.69), pruning
loses reasoning (editing 0.75), and overflow loses everything
(full-context 0.81→0.58 penalized).

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

### R3 — Instruction and call physics are entangled: the same prompt is poison on a shared-cap call and rescue on a calibrated one

If the R2 gap were a prompting deficiency, an explicit instruction —
"end every summary with a TOPICS line enumerating named projects and
their status" — should close it. Under the shared-cap regime it did the
opposite: the instruction *loads* the call (more target, more reasoning),
truncation answers, and the map blanks:

```
LoCoMo continuity recall (recall / % empty folds) — SHARED-CAP regime
                      no instruction     + TOPICS instruction
flat @8K   34%              │  19%   ← instruction made it worse
lcm  @8K   12%              │   0%
flat @16K  62%              │  30%   ← 100% of folds truncated empty
lcm  @16K  52%              │  15%
```

Re-run under calibrated calls (reasoning-off, stated budget, zero
empties), the same instruction flips sign:

```
LoCoMo continuity recall (recall / abstention) — CALIBRATED regime
                      no instruction     + TOPICS instruction
flat @8K   53% / 20%        84% / 0%   ← +31 pts, abstention gone
flat @16K  96% / 0%         92% / 0%   ← two runs lost topics at a fold;
                                           within the fold-loss distribution
lcm  @8K   99% / 0%        100% / 0%
lcm  @16K  99% / 0%        100% / 0%
```

Needles stay 100% and corpus QA stays in the parity band (23–30%) —
the instruction costs nothing else. So the finding is not "state isn't
promptable" and not "architecture doesn't matter" — it's that the two
interact: **on a length-starved reasoning call, prompt content is a
liability; on a calibrated utility call, an inventory instruction is the
cheapest continuity win in the study** (+31 pts for flat at stress, one
prompt line). The architecture gap narrows but survives: hierarchy holds
99% at both budgets with *no* instruction at all, while instructed flat
still folds its way to 84% — structure carries state natively, wording
carries it only as long as every compaction call lands.

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

### R5 — Models recall when occlusion is visible; the 0/511 was mostly
rational no-need

Across 511 suite runs, models spontaneously issue archive search
(flat: ~5-8 `history_search` calls/run at stress; retrieval's whole
design) but **never once** called ARC's deterministic `history_recall`.
We initially read that as a discoverability boundary. It isn't. With
needles deliberately planted beyond citation previews (mid-observation,
outside head/tail excerpts), unprompted models call `history_recall`
at healthy rates — and an explicit system-prompt instruction changes
nothing:

```
hidden needles (mid-placement), unprompted     acc     calls/needle
8K   (view ≈ wall of citations)               50%       0.75
16K  (partial citation coverage)             80%       0.52
64K  (mostly verbatim)                       89%       0.18
8K + explicit "call history_recall" line      46%       (no change)
```

Recovery works when it fires: needle accuracy is 83% among answers
with recall calls vs 44% without. Usage tracks actual occlusion almost
rationally — the suite's 0/511 happened because at deployment budgets
randomly-placed needles are rarely *truly* hidden (the verbatim window
and previews cover them), so there was nothing to recall. The residual
boundary is the trigger problem: some hidden needles still draw
"I don't have that information" instead of a fetch (a truncated preview
emits no error signal the model can notice). Strategies that inject
their map don't depend on the model noticing occlusion; strategies that
gate recovery behind tool calls work — but only as well as the model's
sense that something is missing.

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

### R9 — On real multi-million-token sessions, memory design separates on
### what survives, not on retrieval

Three real pi sessions (3,535–5,836 turns, 1–2M tokens each; the
longest contains 7 real compaction events), capped at a 64K view:
flat-compact / lcm / conclusions × 3 seeds × 3 sessions, with two
long-horizon probes built for this regime — *status currency* (topics
seeded open, flipped resolved/paused at 55–75% depth, uncued "status
now?") and *next-step* (a `[handoff] Next:` line seeded at 60% depth,
asked for uncued at the end).

- **Artifact retrieval is solved for every archive-preserving arm:**
  needles 1.00 and next-step 1.00 across all 27 runs — once the archive
  is searchable (and the search ranks phrases, see below), *what kind of
  memory you keep doesn't matter for anything the model can name*.
- **The separation is inventory:** continuity (uncued "name the topics
  we've discussed") — conclusions 0.77, lcm 0.02, flat 0.00. The
  rewrite arms lose topic *names* in their summaries; the append-only
  store carries them mechanically. Notably lcm's R2 dominance (99%
  inventory on LoCoMo) does not transfer to corpora 100× longer: its
  tree still fits in-budget, but its summaries stop enumerating topics
  when the segment is one-100th of the session.
- **Currency is structured > rewrite:** conclusions 0.80, lcm 0.64,
  flat 0.52 — status flips survive a latest-wins dedupe and a small-
  call tree better than a whole-context rewrite.
- **Cost inverts at scale:** conclusions answered with 466K tokens vs
  lcm 665K / flat 623K (−30%) with peak views of 18K vs 45K/57K at
  equal overhead — the mechanical inventory never grows with session
  length, so the per-turn view stays flat while rewrite views track the
  budget.

Three probe-realism failures had to be fixed before any of this was
measurable, each worth remembering when testing on real transcripts:
(a) **vocabulary collision** — the seeded `[handoff]` marker collided
with the project's real HANDOFF.md files and top-5 search truncation
buried the seeded line below 9 real matches; every arm *correctly*
answered "no such line" until search ranked exact-phrase matches first.
(b) **ambiguous probe wording** — "what was the agent about to do
next?" is answerable from the session's real next-steps; on synthetic
corpora the seeded line is the only answer, on real ones it is one of
dozens. (c) **models detect synthetic probes** — one run answered
"these are all distractor topics injected into the session" (and then
answered correctly anyway); inventory probes on real corpora measure
recall-with-skepticism, which is the deployment condition too.

### R10 — One sawtooth, four backs: the schedule is a commodity; the back
### must be structured or it is dead weight

The sawtooth (fill at full fidelity, delta-only prefill, one deep cut)
is what every deployed compactor converges on — cache- and UX-optimal,
and not the variable that matters. This experiment pins the schedule
*identical* across four arms — same 64K→~16K teeth (12K verbatim head +
≤4K back, boundaries bit-identical because the trigger prices the back
at its cap), same archive, same search tools — and swaps only what the
cut leaves resident (3 real sessions × 2 seeds, 13–20 teeth each):

| back | continuity | currency | needle | next_step |
|---|---|---|---|---|
| none (head only) | 0.05 | 0.58 | 0.96 | 1.00 |
| rewrite (fold/tooth) | 0.10 | 0.62 | 1.00 | 1.00 |
| segments (append-only summaries) | 0.05 | 0.57 | 1.00 | 1.00 |
| ledger (typed observations) | 0.82 | 0.70 | 0.96 | 1.00 |

- **Append-onlyness is not the active ingredient — structure is.**
  `segments` never re-derives anything (no compounding, no rewrite)
  and still ties `none` at 0.05. Prose summaries — frozen or re-rolled —
  carry no topic inventory at this back size. Only typed, topic-keyed
  records that code can dedupe (latest-wins) survive the cut.
  Append-only is what *enables* the mechanism; the mechanism is the
  typing.
- **The rewrite is nearly indistinguishable from evicting outright**
  (0.10 vs 0.05 continuity; currency 0.62 vs 0.58 with the fold winning
  one session 1.00 and losing another 0.45). The deployed default's
  summary buys ~nothing measurable beyond the tail at this cadence.
- **Ledger tuning (two further rounds, n=6 each):** naming continuity —
  passing the running TOPICS inventory into each extraction call as
  read-only vocabulary — collapsed topic fragmentation 147 keys → 4
  (over-consolidating ~5 real workstreams; a wording problem, not a
  mechanism problem). Topic-balanced residency + stripping citations
  from the resident block cut answer tokens 19%. Per-run floors moved
  decisively (worst-run continuity 0.10 → 0.50+); means stayed inside
  seed noise (0.82 / 0.78 / 0.67) — prompt micro-tuning is second-order
  once the structure exists, but floor-raising is real: v1's mean hides
  a run that lost *every* topic to one extraction miss.
- **A probe-validity finding fell out:** the tuned ledger's needle dip
  was the model *refusing* — "the string appears only as an injected
  INFO line buried in tool output." A tidier, more authoritative
  inventory makes synthetic needles more conspicuous: the better the
  memory reads as ground truth, the more anything absent from it looks
  like an injection attempt. R9c's stronger form.

## §5 What we'd tell an agent builder

Priced recipe, current evidence:

1. **Verbatim tail + searchable archive is the floor, not the strategy.**
   It near-ties the summarizers on cued recall at $0 compaction cost (R1) —
   and beats them on cost; what it cannot do is R2's uncued inventory.
2. **Calibrate every utility call: reasoning-off, stated budget,
   require-content retry.** A shared completion cap is a silent killer —
   measured 42–100% empty folds, invisible without telemetry, and bigger
   caps don't fix it (8192 still blanks: the model's reasoning appetite
   scales to the cap). `reasoning: {enabled: false}` + a word budget
   stated in the prompt + an empty-at-length retry took 60/60 grid runs
   to zero blanks at ~¼ of the previous thinking-token bill (R4, R8).
3. **Don't rely on the model calling memory tools at deployment
   budgets.** Tools fire when occlusion is visible (citation walls:
   0.75 calls/hidden-needle @8K, 83% recovery when used) and stay silent
   when the view reads as complete (R5) — and instruction doesn't change
   either regime. Inject the map into the view; treat tools as the
   stress-budget backstop, not the mechanism.
4. **If continuity matters at stress budgets, structure the map as a
   hierarchy of small summaries — and either way, instruct the
   inventory.** Uninstructed, lcm holds 99% uncued inventory at both
   budgets vs flat's 53% @8K (folding drops topics), at ≈1.3× flat's
   compaction overhead. One TOPICS prompt line buys flat +31 pts at
   stress once its calls are calibrated (53%→84%, R3) — free insurance
   even under hierarchy. Maintained-memory variants pay their own cost:
   merge-churn regenerates the whole doc 9–10×/run and erodes inventory.
5. **Hold the view in the model's effective band; compact on crossing,
   not on a clock.** Raw context degrades well below nominal — the
   full-context arm scored 0.81 at 128K while compacted views of the
   same sessions scored 0.94–1.00 — so "wait as long as possible"
   optimizes the wrong variable twice: quality falls with raw size AND
   every turn pays the whole view (our $/run replay costs hide this;
   production pays per turn). Keep the working view at a steady size
   inside the band you've validated (for flash: clean through 128K,
   thinning at 256K) and compact when it crosses the high-water mark.
6. **Event shape beats event frequency.** The loss variable is the
   compaction event, not how often compaction happens: flat loses ~3 pts
   at 0–1 folds and ~43 pts at 3.6 folds/run, while lcm's small-call
   tree absorbs 4 folds/run at 99% inventory. A big-bang re-derivation
   of the whole map is the wrong event shape at any frequency; small,
   bounded, per-segment calls make compaction lossless enough to run
   often. (Every measured summarizer miss was a needle lost at a
   compaction event.)
7. **Budget for degradation.** Choose the strategy whose *failure* you can
   afford: lost-delta (representation) vs wiped-map (flat) vs
   lost-everything (window) — and remember calibration (R4) prevents the
   first two from ever firing.
8. **For long horizons, make memory append-only and assemble in code.**
   On real 1–2M-token sessions the memory design's *loss function*
   decides what survives: a whole-context rewrite loses topic names
   (continuity 0.00) and status flips (0.52); a summary tree keeps
   statuses but loses names (0.64 / 0.02); an append-only observation
   store with mechanical latest-wins assembly keeps both (0.80 / 0.77)
   at −30% answer tokens and a flat 18K view because the inventory
   grows with distinct topics, not session length (R9). Its one failure
   surface is extraction recall — audit what the extractor drops, and
   mandate one-line status mentions explicitly (one prompt rule moved
   continuity 0.49→0.77 mid-grid).
9. **Fix the sawtooth; invest in the back.** The fill-cut schedule is a
   commodity — cache- and UX-optimal, convergent across every deployed
   agent. Under a pinned schedule, a prose back (frozen segment
   summaries or a re-derived fold) buys ≈ nothing over evicting
   outright (continuity 0.05–0.10 vs 0.05 for no back at all, R10).
   The back must be typed records the code can dedupe: that alone
   carries the inventory (0.82). Then tune the *floor*, not the mean:
   passing the running topic vocabulary into extraction and balancing
   residency across topics raised worst-run continuity 0.10 → 0.50+ and
   cut answer tokens 19%, while mean differences stayed inside seed
   noise — structure is first-order, prompt phrasing second-order.

The through-line of all nine: **the compaction call is the unit of
failure — and of design.** Calibrate it (2), size it small (4, 6),
state its budget (2), and let the view it maintains ride in the
effective band (5); everything else is preference. R9 sharpens it one
step further: the call's *loss function* — what it can never put back —
sets the ceiling for everything the agent will ever recall unprompted.

## §6 Threats to validity

- **Cost model is replay-based.** $/run prices one end-of-session
  question battery; production pays the view every turn, so absolute
  costs understate long sessions badly and per-turn framing (5) can
  invert "compact late" intuitions. Cost *ratios* between arms transfer;
  absolute spend does not.
- **One model class.** flash builds every arm's memory and answers every
  question. Stronger summarizer models could reorder R1/R2 — but the
  truncation physics (R4) is a property of reasoning models as a class
  (and so is its fix), and our single-model design is what makes the
  comparisons causal.
- **R9 rides on three sessions.** Real multi-million-token transcripts
  are rare; n=3 sessions × 3 seeds gives n=9 per arm per probe, but the
  corpus is one project's coding sessions on one model — continuity's
  0.77-vs-0.00 gap is large enough to survive session-count noise, the
  currency ordering (0.80/0.64/0.52) is one model's summarization
  quality, and models that recognize synthetic probes (R9c) may behave
  differently when the topics are load-bearing rather than distractors.
- **Judge = solver.** The judge is the same flash; ≥0.5 grading is coarse.
  Systematic bias would have to favor one arm's phrasing to reorder
  results.
- **Estimator gaps.** Calibrated against provider counts (24 live
  calls): true tokens = 0.928 × (chars/4) + 1.6/message — the estimator
  overcharges ~7.7% on content and ~2.4/message, both conservative and
  symmetric across arms, so comparisons are unaffected; constants kept
  to preserve budget comparability with recorded runs.
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
  strategies/        the nine arms (conclusions.py is the newest)
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
