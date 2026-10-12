# Resident Memory for Long-Horizon Agents: Theory, Law, and Dials

Status: theory v2.1 — incorporates SME review passes 1 (activation-law
lineage, single-locus weight, use-semantics, anti-closure contract,
lifecycle, re-encoding audit) and 2 (recording schema, sealing predicate,
allocation policy, identification strategy, prediction repairs).

## 1. The problem

An agent with a bounded resident view operating over an unbounded stream
must serve four distinct recall modes:

| mode | question | example |
|---|---|---|
| uncued enumeration | "what are we working on?" | resume, handoff |
| cued status | "where does X stand?" | "status of the split read?" |
| cued fact | "what was the number?" | "what was the spill cost?" |
| forward-looking | "what's next?" | "what was I about to do?" |

Measured ground truth from this benchmark (all cells n>=6, see README R1-R11):

- **Archive + search solves cued fact at any depth** (needle and next-step
  1.0 in every arm at 5.5M tokens, with one refusal artifact — a model
  declining to parrot an obviously injected line, R10's probe-validity
  signature). Search is free at compaction time and depth-robust.
- **Uncued enumeration collapses for every flat-grain design** at ~115
  compaction events ("teeth") / ~150 resident topics (continuity 0.0-0.3
  where 20-tooth runs scored 0.82-0.92). A flat index is treated as an
  index, not a list to enumerate.
- **Statuses and details decay differently**: a status index line survived
  115 teeth intact while all detail saturated. Status is the last thing
  forgotten; detail is the first.
- **Maintenance LLM calls are where churn enters**: re-summarizing and
  merge-based memories erode inventory and regenerate documents 9-10x/run.
- **Models do not search when the view reads complete** (tools fire only
  when occlusion is visible; instruction does not change this), and a tidy
  authoritative memory actively *harms* absence probes ("never happened",
  injection-distrust — R9c).
- **Birth-time extraction is stochastic**: single-mention recall ~60-70%.

The theory must explain these, not just tolerate them.

## 2. Axioms

1. **Memory is a hierarchy of abstraction, and each level exists to survive
   the decay of the level below it** (Ribot's law: regression from the
   particular to the general). A flat list has one decay rate for
   everything, so it must drown or amputate.
2. **The unit of forgetting is the re-activation, not the fact.** Age runs
   from last use. Disuse, not time, is the enemy.
3. **Specificity is a salary the view pays, not a property of the item.**
   Forgetting is being outbid: demotion when another item's activation
   outranks yours under the budget.
4. **Importance is durability, not elevation.** A load-bearing decision
   outlasts a typo fix *at full detail*, not merely as gist.
5. **Forgetting is selection over an append-only ledger, never
   regeneration.** No summary is ever rewritten; nothing goes stale;
   nothing is destroyed. The archive is the asymptote.

## 3. The architecture (six components)

```
 every message ──────► ARCHIVE  (full fidelity, never cut, FTS-searchable)
       │                        ▲
       │ encode ONCE:           │ search (query = the faded NAME,
       ▼                        │  expanded by aliases, ranked by ledger
    LEDGER                      │  weight not raw BM25)
  append-only facts             │
  (workstream/topic,            │
   weight, gist,                │
   supersedes-edges)            │
       │                        │
       │ activation (pure math, │
       │  render-time only)     │
       ▼                        │
     VIEW ──────────────────────┘
  budgeted render: each item    │
  at the rung it has earned     │ hit re-activates the item
       │                        │ (exogenous uses only)
       ▼                        │
     AGENT                      │
```

### 3.1 Two stores

The **archive** holds everything, forever, searchable — nothing is ever
fully forgotten. The **ledger** holds append-only facts about the stream:
each observation's tree position (workstream → topic), weight, gist, and
supersession edges. The ledger is never edited, merged, or re-summarized.

### 3.2 Encoding at birth — the only LLM step

One extraction call per segment assigns tree position, weight, and gist.
After that **no LLM maintains memory**. Delayed re-encoding (§3.7) is
allowed and is *append-only* — it is not maintenance.

### 3.3 The activation law

```
activation(i) = ln Σ_uses (teeth_since_use)^(-d(i))
                    clamped: teeth_since_use >= 1
d(i) ∈ {D_NOTE, D_DECISION, D_STATE}     (discrete, D_NOTE > D_DECISION > D_STATE)
```

Honest lineage: the **form** is ACT-R base-level activation (power-law
recency, sum over uses, log-compressed, decaying only logarithmically — a
fresh single-use item sits at exactly ln 1 = 0). The
**parameterization** — importance shrinking the decay exponent — is not
canonical ACT-R; its grounding is state-dependent decay (Pavlik & Anderson:
items practiced at higher strength decay slower), emotional-consolidation
modulation (Kleinsmith & Kaplan crossover: high-arousal items recalled worse
at minutes, better at days — salience modulates the *slope*), and rational
analysis (retention should track probability of future need; importance is
a need-prior).

Design decisions from review 1:

- **Weight acts in exactly one locus: the exponent.** It does not multiply
  activation and does not slide ladder boundaries.
- **d is discrete** (3 values, aligned with weight classes). Continuous
  d(w) would let extractor weight-noise dominate ranking: at 10^4 teeth a
  Δd of 0.1 shifts activation ~0.9 nats, more than typical inter-item gaps.
- **"Practice consolidates" is real but weak**: with d ≳ 0.5 the most
  recent use dominates the sum. The sum implements *re-entry promotion*;
  durable importance is carried by the exponent and the pinning rule.
- The **pinning rule is a second mechanism, admitted**: a state-change's
  name+status rung does not demote while its parent workstream renders
  (limit d → 0). This is measured fact (§1: statuses survive detail
  saturation), not derived from the law.

### 3.4 The view is a rendering, not a document

A pure function of (ledger, now, budget). Rungs per item:

```
 L0  full line     "split-read 26%→62% peak, spills −90%, 629µs"
 L1  gist          "split-read spill fix landed"
 L2  name+status   "split read: in progress"
 L3  name          "split read"
 L4  archive only  (not rendered; searchable forever)
```

**View-assembly policy (invariant, not a dial).** The budget is spent in
a fixed order, grounded in this benchmark's own history:

1. **Index reservation**: every unsealed workstream's name+status line is
   reserved first. Completeness of the enumeration surface is a hard
   guarantee — names are cheap; only detail is scarce. (R10: the flat
   inventory's enumeration failure; R9: continuity is the probe that
   dies first.)
2. **Topic-balanced detail fill**: remaining budget fills detail lines
   round-robin across workstreams, newest-per-topic first, greedy by
   activation within that order. (R10 v2: topic-balanced residency raised
   worst-run continuity 0.10 → 0.50+; a pure greedy fill lets one prolific
   workstream evict every other.)
3. **Exact accounting**: the assembled view never exceeds the budget under
   the same counter the trigger uses. (R11: this invariant was broken three
   ways and killed runs; it is restated here as a hard invariant.)

Claims that follow mechanically:

- **Dormancy demotes**: an unmentioned workstream surrenders detail but
  keeps name+status.
- **Re-entry promotes**: one fresh mention resets the clock; detail returns
  instantly (the full-fidelity fact was never destroyed, only unrendered).
- **Capacity displacement emerges**: a burst of new work outbids old work
  that has not aged a day. (This is *displacement*, not cue-based
  interference — the fan/overload form of interference reappears in search
  and is managed by alias expansion and ledger-weight ranking, §3.6.)
- **Saturation scales with workstream count, not topic count** — hierarchy
  buys its fan-out factor of horizon. Falsifiable; see §6 P6.

Composition up the tree: **max-weight descendant governs detail
allocation** (the durable core of a topic is its most consequential fact).
Max cannot cause forgetting — it gates only detail — but the recency floor
(`RECENCY_FLOOR_FULL`, §5) prevents a high-weight outlier from keeping a
dead topic vivid forever. Mass (sum of member activations) governs
index-overflow triage (§4).

### 3.5 The forgetting surface is exposed — the search trigger

Faded topics render as actions, not states:

```
 kvarn — in progress
   gate: resolved · port: in progress
   split read — detail faded · search "split read" to restore
   +4 more faded
```

Every render carries the anti-closure footer — the index is NOT exhaustive;
search before concluding anything never happened. (Directionally measured:
currency 0.47 → 0.82+ when the footer shipped with its sibling fixes; its
*isolated* contribution was not measured, and P2's absence battery is the
instrument that will isolate it.) Markers are phrased as actions ("search X
to restore") because state-phrased markers invite meta-answers and
premature closure. This is tip-of-the-tongue as an interface — with the
caveat that the system's metacognition is *exact* (it knows precisely what
it faded), which is better than human TOT and is a deliberate divergence.

### 3.6 Search closes the loop (with written use-semantics)

A search hit re-activates the item: the next render shows it at higher
resolution, with **no write**. This is retrieval-as-*strengthening* (the
testing effect; in ACT-R terms each retrieval adds a use). It is
deliberately NOT reconsolidation — retrieved memories are not revisable
here; that is a stated divergence from human memory.

**Use-semantics (the ossification guard):** uses are a render-time filter
over a *source-tagged use log* (recorded at collection time), never ledger
writes. Only *exogenous* events append uses — user messages, tool results,
agent actions in the world. The renderer never feeds the ledger (invariant,
not a dial). Of search results, only the result actually quoted/used
reactivates — operationalized as **intentional name-match**: a search
re-activates a topic when the query names it (the Matthew-effect guard:
incidental fan hits on shared words do not count). The literal reading
("only the result the agent quotes back") is unrecordable in this
benchmark — models do not cite message ids in answers — and the
recorded search log (query, result ids, matched topics) reconstructs
the name-match arm exactly (v2.2 amendment; see lit-review).
re-activates; incidental FTS hits on common names do not. Without this, the
resident set becomes self-sustaining (rendered → mentioned → used →
rendered) — a Matthew effect freezing whatever got rendered early.

Alias machinery: the ledger carries canonical names + alias sets; search
expands aliases and ranks results by ledger weight/recency, not raw BM25.
The extraction vocabulary is a read-only reference (naming continuity), not
a merge instruction.

### 3.7 Append-only repair (delayed encoding ≠ maintenance)

Birth-time errors must not be frozen forever:

- **Re-encoding audit**: mechanically triggered when archive segments show
  zero ledger coverage for `REAUDIT_GAP` teeth; appends facts; never edits.
  Its *action* is extraction — so audit outputs are **precomputed per
  archive segment at collection time** (cached, keyed by segment) and
  sweeps replay the cached appends without provider calls. Audit extraction
  is a pure function of the segment; exact while segments are disjoint.
- **Supersession edges**: the extractor asserts "replaces §id" at birth;
  rendering keeps the newest non-superseded fact. v1 and v2 coexist in the
  ledger; the view shows v2. Weight/gist revisions are appended facts.
- **Tree repair**: "moved under" records, append-only.

## 4. Workstream lifecycle

**Sealing predicate (defined).** A workstream seals when, for
`SEAL_AFTER_TEETH` consecutive teeth: no exogenous use of any member
topic, AND no member topic holds an open/in-progress status (after stale
retirement, below). The sealed tier renders as a **name + last-status
list** ("sealed: corvid (resolved), amphora (paused)"), not a bare
count — the workstream-level equivalent of the L2 rung: identity is the
slowest-decaying band and statuses are the last thing forgotten, so a
sealed workstream stays enumerable and status-readable while
surrendering only detail. (v2.2 amendment, from the mega-a qual
finding: a names-only sealed line made cued-status probes answer
"sealed" — memory-tier vocabulary — instead of the status, which the
index can cheaply carry.) "What are we working on" legitimately
excludes sealed work; "what did we work on" still gets them by name.
When the sealed list itself exceeds the index budget, lowest-mass
sealed workstreams (mass = sum of member activations) collapse first
to "+N earlier" — those are recoverable only by search.

**Stale-status retirement.** An open/in-progress status with no exogenous
use for `STALE_STATUS_TEETH` teeth demotes to "stalled" (still name+status,
still pinned while the parent renders, no longer blocks sealing).
Without this, the *norm* for abandoned work — one "in progress" line, then
silence — holds a pinned status forever and the unsealed index grows
without bound, eventually breaking the completeness guarantee at fixed
budget. Identity decays slowest — but not never, and "stalled" is the
rung where it stops.

## 5. The dials

Tuning dials change what the theory predicts; **diagnostic controls** are
configurations the theory forbids or warns about — they exist to
*demonstrate* failure modes (the Matthew effect, probe reactivity), never
as search directions. Sweep tuning dials under the constraint
`D_NOTE > D_DECISION > D_STATE` (the ordering is part of the law; an
unconstrained grid produces meaningless cells).

| dial | default | role | theory root |
|---|---|---|---|
| `D_NOTE` | 0.50 | tune | importance-is-durability; prior *informed by* ACT-R's global d≈0.5 (fitted in seconds — a tooth is ~50–100K tokens of displacement, so the anchor transfers as a range, not a value) |
| `D_DECISION` | 0.35 | tune | " (slope, not level) |
| `D_STATE` | 0.20 | tune | " state changes decay slowest (measured: statuses survive detail saturation) |
| `ACTIVATION_FORM` | power | diagnostic | power: nothing fully forgets, archive is the asymptote; exp: clean saturation — discriminating them needs 1.5+ orders of magnitude of age, i.e. mega sessions only |
| `RECENCY_FLOOR_FULL` | 8 teeth | tune | anti-zombie: full-line rung requires a use within K teeth regardless of weight. Scale-dependent: 8 is reasonable at ~115 teeth, binds nearly everything at grid scale (13–20 teeth) — identification arms run it at 0 |
| `PIN_STATE_STATUS` | on | tune | status-is-last-forgotten (measured). Off is an identification arm, not a deployment setting |
| `PIN_NEXT_STEP` | on | tune | **product prior, not theory-derived**: one pinned next-step per open workstream. Next-step probes are 1.0 via search in every arm, so the benchmark cannot validate it behaviorally; measure its structural budget share |
| `RESIDENT_BUDGET` | 4096 tok | tune | fidelity is the scarce resource; completeness is cheap; 4096 is the R11 stress point (the falsification target) |
| `COMPOSITION_DETAIL` | max | tune | durable core = most consequential descendant |
| `SEAL_AFTER_TEETH` | 30 | tune | identity decays slowest but not never; only meaningful in mega-scale regimes |
| `STALE_STATUS_TEETH` | 15 | tune | zombie-status retirement (§4) |
| `CLOCK` | teeth | diagnostic (NOT offline-sweepable) | teeth+virtual-idle is unimplemented: all aging is tooth-integer arithmetic, the recording carries per-tooth timestamps only (no per-use stamps), and `rerender_from_recording` rejects clock overrides |
| `REAUDIT_GAP` | 3 teeth | tune (live-anchored; NOT offline-sweepable) | extraction-miss recovery; audits run at collection time and their outputs (with audit-time teeth and prompt state) are baked into the recording — a different gap cannot be counterfactually re-run; `rerender_from_recording` rejects gap overrides |
| `USE_SOURCES` | exogenous | diagnostic (reserved; NOT offline-sweepable) | exogenous-only is the theory and the only implemented value: use events are tagged segment/search (the segment IS the exogenous stream). +rendered contradicts the renderer-never-feeds invariant; the interesting arm (drop search uses) is `SEARCH_REACTIVATES=off` |
| `SEARCH_REACTIVATES` | quoted-only | diagnostic (sweeps offline) | quoted-only is the testing effect; any-hit is the fan-noise control; off is the clean-instrument control (§7). All three arms rebuild offline from the recorded search log (query, result ids, matched topics) |

Invariants (not dials): the renderer never feeds the ledger; the ledger is
never edited; the archive is never cut; the anti-closure footer always
renders; index reservation and topic-balanced fill are fixed policy (§3.4);
the assembled view never exceeds budget under exact accounting (§3.4).

### 5.1 Recording schema (collection-time; cannot be retrofitted)

The "collect once, sweep many" contract requires the live replay to
record, beyond the ledger itself:

1. **Token positions** per tooth (cumulative archived tokens; per-message
   positions would allow re-toothing, which is moot while
   `RESIDENT_BUDGET` is clamped to `BACK_TOKENS` — a letter deviation,
   accepted).
2. **Source-tagged use events** (`segment` / `search` — the theory's
   user/tool-result/agent-action trichotomy collapses to this two-valued
   tag; accepted deviation) — sufficient to sweep `SEARCH_REACTIVATES`.
3. **Per-tooth timestamps** (the teeth+virtual-idle clock arm would need
   per-use stamps; unimplemented, see the dial table).
4. **Full search logs** (query, result ids, matched topics) — sufficient
   to rebuild every `SEARCH_REACTIVATES` arm offline; literal quoted
   spans are not recorded (models do not cite ids in answers; see the
   §3.6 name-match amendment).
5. **Cached audit outputs** — replayed at their recorded teeth under any
   render dial, but the gap itself is live-anchored: audits ran at
   collection time and their outputs are baked in (see the dial table);
   `REAUDIT_GAP` is not counterfactually sweepable.
6. **Per-tooth rung assignment logs** — the primary identification signal
   (§5.2).

### 5.2 Sweep methodology and identification

**Metrics (offline, no LLM):** per seeded topic, per tooth, per class — the
**full rung trajectory** (rung vs teeth-since-last-use). That is ~115
ordered observations per topic on mega sessions, not one binary at
question time. Plus: index completeness (fraction of unsealed workstreams
enumerated, per tooth), allocation shares (detail tokens per active
workstream), saturation tooth (first tooth where any guarantee breaks),
and observable budget cutoffs (highest-excluded / lowest-included
activation at each rung boundary — cutoffs are *observed*, not fitted).

**Identification strategy.** Budget and decay are confounded at a single
budget (faster decay mimics smaller budget), and pins/floors delete the
comparisons that identify d (the pin removes L2 competition for state
items; the floor removes L0 competition beyond K teeth; the d-signal
survives mainly in L1 transitions). Protocol, staged:

1. **Identification arms**: PIN off, FLOOR 0, composition fixed, budget
   swept so cutoffs stay interior.
2. **Fit d̂ per class**: for single-use seeded topics of class c, the
   rung-drop age Δ* satisfies A(Δ*) ≈ C_r. Under the power law,
   −ln C_r = d_c · ln Δ*, so d̂_c is the slope of −ln C vs ln Δ across
   (rung, budget, tooth) — over-identified; cross-consistency is the
   specification test. A fitted violation of D_NOTE > D_DECISION > D_STATE
   is a genuine falsification of the class taxonomy.
3. **Reintroduce pins and floors** as separately identified step
   mechanisms.
4. **Fit composition** on allocation shares; **fit sealing** on saturation
   tooth.

Absolute d is not identified — only d relative to the ledger's activation
distribution. That is acceptable: the deliverable is dial settings, not
"true human d." Form discrimination (power vs exp) needs 1.5+ orders of
magnitude of age — mega sessions only; grid-scale runs cannot separate the
forms or pin d.

**Objective function (stated before collection).** A constraint set —
index completeness = 100% of unsealed workstreams at every tooth; seeded
residency ≥ L2 at all probe teeth; overflow never — with saturation tooth
maximized within it; if infeasible, an explicit Pareto report plus live
anchors at the extremes.

**Counterfactual caveat (explicit).** Offline sweeps freeze the recorded
exogenous stream, but the recorded agent's mentions and searches were
responses to the *default-dial* view. Structural metrics measure residency
under the recorded stream, not counterfactual behavior; the live
confirmation runs at dial extremes exist precisely to close that gap.

## 6. Falsifiable predictions

1. **Partition test (falsify first)**: a hierarchical ~30-line index at
   ~115 teeth / ~150 topics / ~30 workstreams holds continuity ≥ 0.7 where
   every flat arm collapsed to 0.0-0.3. **Scoring spec:** the continuity
   probe counts a seeded topic as recalled if the model names it — sealed
   topics count, because the sealed tier renders names (§4). If a *small
   hierarchical* index still fails enumeration, the collapse is about how
   models treat index-shaped text — the central scaling rationale dies,
   and the problem restates as presentation, not memory.
2. **Marker → search conversion**: faded markers convert to searches at a
   materially higher rate than a no-marker control, **without** raising
   false-denials. Instrument (new): an absence battery —
   present-but-extraction-missed facts (false-denial rate) and
   never-occurred facts (correct-deny-after-search rate). The harness has
   no denial-class probe class today; building it is part of this
   prediction.
3. **Durability ordering — behavioral, not structural.** The structural
   sweep *sets* d, so structural ordering is true by construction
   (machinery consistency check only). The falsifiable version needs
   independent behavioral targets per class: currency probes ≈ state
   class, needle probes ≈ note class, and a **new decision-class probe** —
   seeded "we chose A over B because C" lines with cued recall — whose
   retention curve must fall between. If fitted behavioral decay violates
   the ordering, the class taxonomy is wrong.
4. **Displacement (not interference)**: a dense burst of *new* work
   mid-run degrades old-topic residency more than the same volume
   continuing *existing* work. Constructed offline with matched total
   tokens using the interleave machinery; live confirmation for the
   behavioral half.
5. **Reactivation**: one exogenous re-mention restores a dormant topic's
   detail within one tooth (structural, exact — the render is
   deterministic). The quoted-search half is only testable under the dials
   at which the search log was recorded (search behavior is
   counterfactual under swept dials); "without any write" is a unit-test
   assertion. Say so when reporting.
6. **Saturation scaling**: saturation tooth scales with workstream count,
   not topic count — testable offline by replicating workstreams in a
   recorded ledger to vary count at matched topic mass. This is the
   design's central scaling claim made mechanical.

## 7. Benchmark hygiene (probe reactivity)

With `SEARCH_REACTIVATES` on, probes are practice: answering a checkpoint-N
probe improves checkpoint-N+1 memory. The instrument must not grade its
own homework: score probes offline (assemble the view per probe, no
feedback into the session) or use fresh non-repeated probe sets with
matched exposure, and report first-occurrence vs repeat-probe scores
separately. `SEARCH_REACTIVATES=off` is the clean-instrument control.

## 8. Positioning

- Closest cognitive architecture: Soar's episodic memory (ACT-R-style
  activation, per-memory parameters) — verify exact parameterization
  before citing.
- **Verify before claiming novelty**: MemoryBank (Zhong et al., AAAI 2024)
  implements Ebbinghaus-curve forgetting with significance-dependent decay
  and retrieval reinforcement — if so, it is prior art for
  importance-modulated decay (not for the weight-in-exponent form, the
  ladder, or the exposed surface).
- Closest engineering lineage: **weighted cache eviction** (GreedyDual-Size,
  ARC) — "rank by importance × recency, evict the cheapest" is eviction
  policy. The contribution is the eviction policy **plus** a resolution
  ladder (demote before evict) **plus** an exposed forgetting surface
  (eviction you can see and reverse by search).
- Adjacent: EM-LLM (surprise-segmented context — contrast case for the
  teeth clock: budget-driven vs prediction-error-driven segmentation),
  Titans (surprise → encoding strength), MemOS (promotion mechanics without
  importance decay), Letta sleep-time compute (consolidation; our churn
  findings argue against), Mem0 (record ops, static importance), Zep
  (temporal invalidation, no decay), LCM (tree, re-derives instead of
  demoting), Generative Agents (importance as an *additive static term in
  a weighted sum alongside* exponential recency decay — not a multiplier
  on it).
- Deliberate divergences from human memory, stated: no reconsolidation
  (retrieval strengthens but never revises); exact metacognition (better
  than tip-of-the-tongue); power-law decay implemented as render-time
  selection, not trace decay; the archive makes verbatim memory perfect
  (Fuzzy-Trace Theory is satisfied by the archive's parallel verbatim
  trace, not by the ladder).

## 9. What this predicts about the benchmark's own history

The theory reads our results as its own boundary conditions — and they
split by arm, which is evidence *for* it: **v1's** collapse was the
flat-list failure mode (one decay rate, newest-records amputation — Axiom
1's "drown or amputate"); **ledger4's** collapse was detail-budget
saturation with statuses *differentially surviving* — the decay-rate
split the ladder formalizes, already present, but with no hierarchy to
allocate around and no exposure of what faded. The R9c refusal signature
= an exposed surface without the anti-closure contract. The rewrite≈
eviction result = regeneration that buys nothing selection doesn't.
Search's depth-robustness = the archive asymptote. A theory that cannot
re-derive its own inputs is decoration.
