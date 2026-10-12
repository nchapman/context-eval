# Dataset Plan: Stress-Testing Resident Memory

Status: draft v1 — researched 2026-12; the system (sawtooth-ledger5 + dials)
and its hypothesis are ahead of the data. This plan defines what the ideal
dataset looks like, inventories what exists (HuggingFace research), and
proposes assembly.

## 1. What the data must exercise (from our own findings)

1. **Mega-scale**: ≥500K tokens per session, multi-day shape. Sealing,
   decay, and the activation law only bite at depth (153 teeth needed
   ~2M input tokens to emerge).
2. **Engagement-weighted topics** (the "rando" critique): topics with
   natural mass — recurring work, real decisions — not just one-line
   plants. Single-mention recall is exhausted (10/10); the open question
   is recall as a function of conversational mass.
3. **Natural status evolution**: things that open, get worked, flip
   status, get superseded or abandoned — the currency probe's subject,
   arising from real work rather than injected `[status]` lines.
4. **Agentic coding density**: tool calls, code, test output — the
   note-taker's hardest regime (64K-char chunks of dense technical
   content is where crowding lives).
5. **Layered questions** (LoCoMo's real contribution): single-hop,
   multi-hop, temporal, adversarial/abstention, and uncued inventory —
   scored separately, because our findings show they dissociate
   (cued currency 1.00 while uncued continuity is mass-gated).
6. **Cued AND uncued probes**, plus abstention ("did we ever...?" where
   the answer is no) — our injection-distrust finding says tidy memory
   actively harms absence probes; that deserves external data.

## 2. Inventory: existing benchmarks (adopt where possible)

| Benchmark | Shape | Fit | Gap |
|---|---|---|---|
| **LongMemEval** (ICLR'25, `xiaowu0162/longmemeval-cleaned`) | 500 Qs, 5 abilities (extraction, multi-session, temporal, **knowledge-update**, **abstention**), S=115K tok/~40 sessions, M=500 sessions/~1.5M tok, timestamped | Best external match: knowledge-update ≈ our currency, abstention ≈ our never-happened; scale is right in M form; question pipeline is attribute-controlled and extensible | Chat only — no tools, no code, synthetic personas |
| **LoCoMo** (ours) | 10 dialogs, ~600 turns, 16K tok, 32 sessions | Question taxonomy is the model; already integrated | Far too short for sealing; saturated |
| **MemoryAgentBench** (ICLR'26, `ai-hyz/MemoryAgentBench`) | Incremental multi-turn streams, 4 competencies (retrieval, test-time learning, long-range understanding, **conflict resolution**); inject-once-query-many | Philosophy matches our replay loop; conflict resolution is untested for us | Transformed long-context docs, not agentic |
| **MemoryArena** (ICML'26, memoryarena.github.io) | Multi-session Memory-Agent-Environment loops, interdependent subtasks (web, planning, search, formal reasoning), 701 task groups / 4,850 subtasks, ~57 steps/40K tok | Tests whether memory supports **action**, not just recall; shows LoCoMo-saturated agents fail here | Interactive gym — needs a live agent, not our replay harness; not coding |
| **PersonaMem** (`bowen-upenn/PersonaMem-v1`) | Preference evolution, 32K/128K/1M splits | "Reasons behind preference updates" = status-flip rationale | Personalization niche, synthetic |
| **TheAgentCompany** (CMU, MIT) | Simulated software company, ~175 tasks (code/web/comm), workday-scale, trajectories open-sourced | Real long-horizon agentic structure with interdependencies | Per-task episodes, not one continuous thread; QA must be generated |
| **DABstep** (Adyen, `adyen/DABstep`) | 450 multi-step data-analysis tasks, sequential dependencies, factoid auto-scored | Dependency-chain idea; auto-scoring discipline | Small; analysis not coding |

**Rejected with numbers**: WildChat / LMSYS-Chat-1M / ShareChat (avg 2–4.6
turns, only ~3.7% exceed 10 turns — no long-horizon signal); MSC (tiny
sessions, superseded by LongMemEval); LongBench/InfiniteBench (documents,
not interaction).

**The gap**: nothing combines agentic coding × mega-scale × layered QA ×
natural topic mass. LongMemEval has QA+scale without code; the SWE trace
corpora have code+mass without QA. The ideal dataset is **assembled**.

## 3. Raw material for assembly

- **Open-SWE-Traces** (NVIDIA, `nvidia/Open-SWE-Traces`): 250K agent
  trajectories from **20K real PRs** (SWE-rebench-V2, permissive licenses
  only), 9 languages, OpenHands/SWE-agent/mini-swe-agent scaffolds.
  Individual trajectories are short (7–101 messages) — but grouped by
  repo and stitched in PR order they form **real workstream histories**:
  a hot module recurs across PRs (natural mass), drive-by PRs are
  randos (natural dose variation), PR lifecycle (open→merged/abandoned)
  is **real status evolution**, and approaches chosen/abandoned in
  commits are real decisions.
- Alternates: SWE-Gym trajectories (6K), nebius OpenHands (67K),
  Kwai SWE-smith 66K — same shape, kept as backup.
- Question generation: LongMemEval's attribute-controlled pipeline and
  LoCoMo's taxonomy are the templates; ground truth from PR/commit
  metadata (mechanically checkable: PR numbers, paths, error strings)
  with LLM-generated phrasings verified against the transcript.

## 4. Plan

### Phase 1 — adopt LongMemEval-S (external comparability, ~1 day, ~$3)
- Loader `datasets/longmemeval.py` → ReplaySample (timestamped sessions
  concatenated; questions mapped to kinds: knowledge-update→currency-style
  scoring, abstention→separate reporting).
- Run ledger5 vs flat-compact vs retrieval at 16K/64K. Read per-ability
  curves, not just acc: our prediction is knowledge-update and abstention
  dissociate from single-hop.
- Gate: preflight + sanity report as usual.

### Phase 2 — repo-mega corpora (the core build, ~1 week)
1. Download Open-SWE-Traces, group by repo; select ~6-10 repos whose
   stitched PR-chronology clears 500K tokens.
2. Stitch per-repo sessions: issue text → user turn; agent steps →
   assistant/tool messages (reuse pi_mega stitching machinery); PR
   outcome → closing turn. Real timestamps preserved.
3. Question layers from metadata (LoCoMo taxonomy):
   - single-hop: "what did PR #N fix?" / "what error did X address?"
   - temporal: "which came first, X or Y?"
   - multi-hop: "the module with recurring failures across PRs — what
     finally fixed it?"
   - knowledge-update: "what happened to approach A?" (later PR supersedes)
   - abstention: "did anyone ever fix Z?" (Z absent from history)
   - uncued inventory: "which modules did we work on?" (natural, not seeded)
4. **Dose-response axis**: measure recall vs natural mention-mass
   (mentions, PRs, token share per workstream) — the curve the needle
   paradigm couldn't give us. No injection; the mass is real.
5. Mechanical answer verification wherever possible; LLM-judged only
   for inventory/open-domain.
- Gates: SME review of the question generator before any live spend;
  2 smoke runs read by hand; preflight ceilings.

### Phase 3 — cross-benchmark positioning
- Report ledger5 vs baselines on: LongMemEval (external), repo-mega
  (ours), pi-mega (ours). One figure: ability × depth × arm.
- MemoryArena deferred: requires a live agent loop our replay harness
  doesn't have; revisit if we add an interactive mode.

## 5. Cost envelope
- Phase 1: ~$3 live (500 Qs × few arms × 2 budgets, cheap answers).
- Phase 2: loaders/generators free (metadata-grounded); smoke ~$2;
  full repo-mega suite ~$1.20/leg-class like today's mega.
- Total to a fully-stressed system: ~$10.
