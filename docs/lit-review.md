# Literature Review: "Lossless" Context Management for LLM Agents

*Compiled 2026-10. All arXiv IDs and claims below were verified against sources during research; community teardowns (of Claude Code, Codex, etc.) are labeled as such.*

## 1. The problem

LLM agents accumulate transcripts — messages, tool calls, observations — that grow monotonically while the context window stays fixed. Every strategy must answer two questions:

1. **What stays in context?** (the active view)
2. **How does the agent get the rest back?** (the recovery path)

"Lossless" in practice means one of three things, and the distinction matters for evaluation:

| Term | Definition | Examples |
|---|---|---|
| **Strictly lossless** | Verbatim archive + deterministic pointers; nothing ever transformed | ARC, Contexto, grep-your-own-transcript |
| **Effectively lossless** | Lossy in-context, but originals retained and recoverable via search/recall | LCM, hiro, MemGPT |
| **Lossy** | History replaced by summary; originals gone or unreferenced | Claude Code /compact, Codex compaction |

A key finding of the theory work (Context Compaction Theory, arXiv:2608.01326): lossy compaction introduces its own failure mode — later compactions build on earlier summaries, so errors and omissions **compound**. The paper surveyed deployed agents (Claude Code, Codex, Gemini CLI, OpenCode, Goose, Cursor) and found they all share this design. Community teardowns echo it: Codex's default compact prompt is ~6 lines, and "after 2 compactions, all reasoning is gone" (sigalovskinick gist, 2026).

## 2. Strategy families

### Family A — Truncation baselines (anchors)

- **Full context**: send everything. The oracle for effectiveness — but degrades with length ("lost in the middle"), costs linearly more per turn, and eventually overflows. LongMemEval (ICLR 2025) reports ~30% accuracy drop for long-context LLMs on sustained-interaction memory.
- **Sliding window / drop-oldest**: keep last N messages. The floor.
- **Observation masking** ("micro-compaction"): deterministically replace old tool outputs with placeholders/truncations. Cheap, lossy, no recovery. Used inline by Claude Code, OpenCode, Pi.

### Family B — Single-shot flat summarization (the industry default)

One LLM call summarizes the whole history; the summary replaces it.

- **Claude Code**: auto-compacts near the limit. Official docs: summarizes history, re-injects system prompt, CLAUDE.md, plan, auto-memory, fresh git status, and re-reads up to 5 recently-modified files. Community teardowns (claude-code-explain; badlogic gist) add: triggers with ~13K tokens remaining, ~50K post-compact budget (5K/file), 3-failure circuit breaker, and a 9-section structured summary prompt. Now server-side ("compact-2026-01-12").
- **Codex CLI**: manual `/compact` + auto at `model_auto_compact_token_limit` (`codex-rs/core/src/compact.rs`). Two paths: local LLM summarization for open models vs server-side `/responses/compact` with encrypted state (community teardown).
- **Gemini CLI**: LLM summary at configurable % of window, producing a structured XML snapshot, **plus a second LLM "probe verification" pass** checking completeness (documented in arXiv:2608.01326 — the only deployed agent with a verification step).
- **OpenCode**: two-phase — prune old tool outputs to a 40K budget, then LLM summarization (community teardown).

Strength: simple, effective short-term. Weakness: compounding loss, no recovery path, blocking.

### Family C — Lossy in-context, lossless in archive (recoverable)

The "lossless" family proper. Originals are always retained verbatim in a store; the active view is a bounded projection; recovery is a first-class operation.

#### C1. LCM — hierarchical summary DAG + recall tools
**LCM: Lossless Context Management** — Ehrlich & Blackman (Voltropy PBC), arXiv:2605.04050 (Feb 2026).

- Two token thresholds govern a deterministic engine loop: below τ_soft, zero overhead; between τ_soft and τ_hard, async compaction between turns; at τ_hard, blocking compaction.
- Compaction builds a **hierarchical summary DAG** over older messages while retaining pointers to every original.
- Their agent Volt (Opus 4.6) beat Claude Code on the OOLONG benchmark at every context length 32K–1M.
- The paper also proposes recursive task partitioning (LLM-Map style) — out of scope for us.
- **Reference implementations**: hermes-lcm (github.com/stephenschoettler/hermes-lcm, SQLite-backed DAG engine for Hermes Agent); hiro (github.com/nchapman/hiro) — depth-graded prompts (D0 leaf → D1 session → D2 phase → D3+ durable memory), escalation ladder (normal → aggressive → deterministic truncation), FTS5 HistorySearch + HistoryRecall tools, per-model parameter scaling.

#### C2. ARC — addressable citations + deterministic recall
**Addressable Recall Compaction for Long Context-Window Control in AI Agents** — Dang, Ichikawa, Fatima, Shirahata (Fujitsu), arXiv:2607.25066 (Jul 2026).

- Append-only, **ID-addressable store** of every tool observation; bounded active view replaces older/longer observations with compact `§id` citations (head/tail preview + metadata). Observations below a length threshold ρ stay verbatim.
- `_recall §id` tool, intercepted by the engine, restores full content on demand. Citation handles persist across later compactions.
- **No LLM is involved in compaction itself** — deterministic. Only recall costs extra tokens.
- **Regime boundary (applies to our clean-room and to context-editing alike):** both approaches only shrink TOOL payloads; on assistant-heavy transcripts (our pi corpora ≈ 54% untouchable assistant content) the view cannot fit and the driver records overflow — a property of the designs, not the harness. Reproduced on-synththetic: 23–24K edited view against a 16K budget with every tool observation cleared/cited.
- **Model-behavior boundary (our clean-room finding, deepseek-v4.1-flash):** id-addressed recall is never invoked. Across every smoke round the model made zero `_recall` calls — with the system prompt instructing recall use and the tool description teaching fan-out — while it willingly uses query-based `history_search` (0.5–0.9 calls/q). Guess-which-§id is not a step this model class performs; search-by-question-terms is. Measured ARC therefore runs in a preview-visibility-only regime: needles inside cited observations are recoverable only if they land in the head/tail preview. Tool discoverability predicts tool usage predicts recovery — a deployment truth the paper's single-model NIAH (99.4%) does not surface.
- Results: Needle-in-a-Haystack 99.40% vs 88.12% best baseline (Qwen3-8B @16K, Qwen3-32B @32K); LongBench-v2 Hard 29.97% vs 28.25%; reduced serving time and HBM traffic.
- Note: name collision with a different "ARC" (Active and Reflection-driven Context management, arXiv:2601.12030, ACL 2026 Findings — reflection-driven context reorganization for info-seeking agents).

#### C3. Search over verbatim history (no summarization at all)
Keep the raw transcript searchable; give the agent search tools (FTS5/BM25 or embeddings). This is hiro's HistorySearch generalized into a standalone strategy. Zero summarization cost; recall depends on the agent choosing to search and querying well. The EMem paper's baselines call this RAG-4096 (dense retrieval over chunks).

#### C4. Semantic organization — Contexto (AGNES clustering)
**ekailabs/contexto** — discussion #114, "Why We Chose Hierarchical Clustering (AGNES) for Agent Context".

- Core critique of LCM-style compaction: "**compression is temporal, not semantic**" — messages are grouped by arrival order, so related episodes scatter across summary nodes and unrelated topics blend together.
- Instead: **organize, don't compress**. Episodes are clustered bottom-up (AGNES) with UPGMA average linkage. Nothing is ever summarized away — claims zero information loss.
- Multi-resolution retrieval: sometimes a topic-level summary node, sometimes a specific episode; token-budget-aware packing (fill the window by topic coverage, not raw top-k).
- Hybrid maintenance: full O(n²) rebuild when few items or every 50 inserts; between rebuilds, greedy O(log N) tree insertion with O(d) centroid updates.
- The discussion usefully distinguishes **memory systems** (Mem0/Zep/Letta: "what does the agent know about the user?") from **context management** ("what past context is most relevant to inject right now?").

#### C5. RAPTOR — cluster + summarize trees
**RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval** — Sarthi, Abdullah, Tuli, Khanna, Goldie, Manning (Stanford), arXiv:2401.18059, ICLR 2024.

- Recursively: embed chunks → cluster (GMM, soft) → summarize each cluster → repeat on summaries, building a tree of increasing abstraction.
- Collapsed-tree retrieval lets a query hit the right abstraction level. +20% absolute on QuALITY (GPT-4).
- The bridge between C1 (temporal trees) and C4 (semantic clustering): semantically grouped, but nodes are summaries (lossy with verbatim leaves recoverable).
- Widely cited; retrieval-focused (offline index + query) rather than online transcript management — adaptation required.

### Family D — Extracted memory (change of abstraction level)

The agent "knows facts" rather than "recalls conversation."

- **Mem0** — Chhikara et al., arXiv:2504.19413 (Apr 2025). Two-phase: LLM extraction of salient facts from recent conversation, then consolidation with explicit ADD/UPDATE/DELETE/NOOP ops. Graph variant (Mem0-g) adds relational structure. LoCoMo: 26% relative improvement over OpenAI's memory, big token savings vs full-context. Production impl: github.com/mem0ai/mem0 (huge OSS following). Notable data point: their docs report full-context routinely consuming 25K+ tokens/query vs <7K for memory retrieval at comparable accuracy.
- **EMem (event-centric)** — "A Simple Yet Strong Baseline for Long-Term Conversational Memory," arXiv:2511.17208. Extract events, not facts/preferences. On LoCoMo and LongMemEval-S, matches or beats LangMem/Zep/Mem0 while cutting context from 101K to 0.6–3.6K tokens (gpt-4o-mini). Also reports memory retrieval (0.794–0.806) beating naive full-context prompting. The title says it: strong and simple.
- **MemGPT / Letta** — Packer et al., arXiv:2310.08560 (Oct 2023). The OS metaphor: main context vs external context (recall + archival storage), FIFO eviction with memory-pressure warnings at 70% (letting the model proactively page things out), recursive summary of evictions at 50%. Successor: Letta (github.com/letta-ai/letta). DMR benchmark origin.
- **Zep / Graphiti** — Rasmussen et al., arXiv:2501.13956 (Jan 2025). Temporal knowledge graph (bi-temporal: event time vs ingestion time) built from conversation. Beats MemGPT on DMR (94.8% vs 93.4%), strong on LongMemEval. Impl: github.com/getzep/graphiti. Complex — not clean-room friendly; candidate for library-baseline only.
- **Generative Agents** — Park et al., arXiv:2304.03442 (2023). Memory stream with retrieval scored by recency × importance × relevance, plus reflection (synthesizing higher-level memories). The root of most of this literature.
- **Filesystem-as-memory** — Letta blog, "Is a Filesystem All You Need?": storing conversation history in a file (agent reads/greps it) scores **74.0% on LoCoMo**, beating specialized memory libraries. Strong evidence for candidates 4/11 (search over verbatim history, transcript-as-file).
- Also noted, likely out of scope: A-Mem (Zettelkasten-style agentic notes, arXiv:2502.12110), MemoryBank (Ebbinghaus decay, Zhong et al.), H-MEM (hierarchical semantic abstraction with positional indexes, EACL 2026), Memory-R1 (RL-trained memory ops).

### Family E — Model-managed recursion (RLM)

**Recursive Language Models** — Zhang, Kraska, Khattab (MIT), arXiv:2512.24601 (Dec 2025, v3 May 2026).

- Treat the prompt as **part of the environment**: load it as a variable in a Python REPL; the model writes code to peek, decompose, and recursively call itself on snippets.
- Processes inputs up to 100× beyond context windows; on long-context tasks, beats GPT-5-with-compaction by 26% median, CodeAct sub-calls by 130%, Claude Code by 13% — at comparable cost.
- Post-trained RLM-Qwen3-8B: +28.3% over base Qwen3-8B. Code: github.com/alexzhang13/rlm.
- LCM positions itself as decomposing RLM's "symbolic recursion" into deterministic engine mechanisms. RLM is the maximal-flexibility end of the spectrum; hard to clean-room faithfully (the policy is the artifact), but an "RLM-lite" (transcript-as-file + code/search tools) is tractable.

## 3. Theory & learned policies (related work)

- **Context Compaction Theory** — arXiv:2608.01326. Formalizes how lossy compaction fails; the comparative anatomy of deployed agents' compaction (thresholds, what's re-injected, verification).
- **AutoCompact** — arXiv:2610.02163. Trains a coding agent to decide when to compact, what to preserve, how to continue (judge-labeled trajectories). Out of scope for clean-room, but relevant for "when" policy ablations.
- **ARC (reflection variant)** — arXiv:2601.12030. Context as active internal state, reorganized on detected degradation.
- Surveys: "Memory for Autonomous LLM Agents: Mechanisms…" arXiv:2603.07670 (2022–early 2026); security of long-term memory arXiv:2604.16548; Structured Agent Memory (ACL 2026 demo).

## 4. Benchmarks

| Benchmark | Venue | Shape | Measures | Notes |
|---|---|---|---|---|
| **LoCoMo** | ACL 2024 (Maharana et al.) | 10 conversations, ~600 turns, ~16K tokens, ≤32 sessions; ~2000 QA | single-hop, multi-hop, temporal, open-domain, adversarial | De facto standard; hiro already has an eval harness. Long-context+RAG gains 12–20% but lags humans by 36% (41% on temporal) |
| **LongMemEval** | ICLR 2025 (Wu et al.), arXiv:2410.10813 | 500 questions in scalable histories | info extraction, multi-session reasoning, temporal reasoning, **knowledge updates**, **abstention** | Harder than LoCoMo; the two bolded abilities specifically punish stale/compounding summaries. Code: github.com/xiaowu0162/LongMemEval |
| **OOLONG** | arXiv:2511.02817 | Long-context reasoning tasks | atomic chunk analysis + distributional aggregation | Used by LCM paper; closest to coding-agent workloads |
| **MemoryAgentBench** | ICLR 2026 (arXiv:2507.05257) | Incremental multi-turn | accurate retrieval, test-time learning, long-range understanding, **conflict resolution** | Critiques LoCoMo (~9K, short) & LongMemEval (synthetic). **Agent-shaped harness w/ pluggable methods** (see §4.1) |
| DMR, PerLTQA, RealTalk, StoryBench, LoCoMo-Plus | various | — | — | Secondary; LoCoMo-Plus targets implicit constraints (2026) |
| **BEAM** | via Mem0 docs | 1M & 10M token scales | ten task categories | Only public benchmark at production context volumes — "cannot be solved by simply expanding the context window" |

### 4.1 Eval harnesses (existing, reusable)

- **MemoryAgentBench** — github.com/HUST-AI-HYZ/MemoryAgentBench (MIT, 459★, ICLR 2026). The agent-focused harness: data is chunked and fed as **incremental multi-turn interactions** ("inject once, query multiple times"), with a pluggable `methods/` layer (ships adapters for mem0, letta, cognee, RAG, full-context, and more). Datasets: EventQA, FactConsolidation (new), plus reformulated RULER, DetectiveQA, ICL tasks, recsys, LongMemEval, InfBench. Metrics: substring/exact match, Recall@5, LLM-judge. Data on HuggingFace (`ai-hyz/MemoryAgentBench`). This is the substrate our strategies plug into — compaction happens *during* the conversation, which offline QA harnesses can't exercise.
- **mem0/evaluation** — github.com/mem0ai/mem0/evaluation. The LoCoMo harness the field copy-pastes (memobase forked it; EMem replicated its baselines). Offline: ingest → retrieve → answer → judge. Good for family D, blind to online cost.
- **memory_and_context_eval_harness** — github.com/maximem-ai/memory_and_context_eval_harness. Clean Provider-ABC design, 5-phase pipeline, LoCoMo + LongMemEval, but also offline.
- **letta-evals** — github.com/letta-ai/letta-evals. Evaluation kit for stateful agents (Letta-platform-coupled).
- Also relevant: "Beyond Memory Leaderboards: Evaluating Scientific Memory as Budgeted Context Restoration" (arXiv:2607.16848) — argues for budgeted-restoration framing, aligned with our efficiency axis.

**Gap nobody fills:** head-to-head comparison of the *families* (B vs C1 vs C2 vs C3 vs C4 vs D) under identical budgets, identical models, with efficiency (tokens, extra LLM calls, latency) reported alongside accuracy. Each paper evaluates against its own chosen baselines. That's the opportunity for this project.

## 5. Evaluation dimensions

**Effectiveness**
- Per-category accuracy (LoCoMo/LongMemEval categories), LLM-judge where standard
- Recovery success: needle questions after compaction events (ARC's NIAH protocol)
- Knowledge-update & abstention categories (punish stale memory)
- Compounding: accuracy as a function of number of compaction cycles

**Efficiency**
- Tokens in/out per turn (task LLM + any summarizer/embedder)
- Extra LLM calls per conversation (summarization, verification)
- Wall-clock latency added, split by LCM's regimes (none/async/blocking)
- Peak and mean context occupancy; $ cost per conversation at fixed budget

**Protocol notes**
- Fixed context budget sweep (e.g., 16K / 32K / 128K) — ARC's approach; the only fair comparison
- Same base model + same judge across strategies
- Both online (compaction during a live conversation) and offline (QA over assembled context) eval modes — efficiency is an online property; most benchmarks only test offline

## Reasoning-token persistence (added Oct 2026)

Do agents keep thinking/CoT in history, and should they? Measured on a real
pi session (warp3090, Oct 6): assistant **thinking = 48% of all content
tokens** (1.36M of 2.81M) vs tool results 45%, assistant prose 6%, user 1%.
Reasoning persistence is therefore the single largest context lever for
reasoning-model sessions -- larger than tool-output compaction, which is
what the classical papers optimize.

**Coverage in our references:** mostly absent. ARC compacts tool
observations only (our pi-corpus run shows the failure: peaks 232-415K on a
128K budget because assistant content is untouchable). LCM and CompactionRL
treat all history uniformly; Context Compaction Theory only observes that
deployed compaction destroys it ("after 2 compactions, all reasoning is
gone" -- Codex teardown).

**The 2026 wave asking exactly this:**
- *When Can Agents Forget Their Reasoning?* (arXiv:2609.29875) -- online
  compression of historical reasoning; removing it changes future actions;
  scores reasoning relevance given trajectory state; surveys ARC,
  SelfCompact, Sculptor, SWE Pruner, COMPASS.
- *SelfCompact* (arXiv:2606.23525) -- rubric-gated compaction over "closed
  reasoning units"; +18.1 pts math / +5-9 agentic search at 30-70% lower
  cost vs fixed-interval compaction.
- *Structured Context Eviction* (arXiv:2606.11213) -- explicit eviction
  ladder; **reasoning-trace stripping comes first** (exploratory episodes).
- *COMPASS* (ACL 2026) -- meta-thinker reasons over previous reasoning;
  keeps it as structured notes, not in-context.
- *Slipstream* (arXiv:2605.08580) -- validates compactions against the
  agent's continued reasoning.

**Vendor policy (production):**
- Anthropic: thinking blocks REQUIRED verbatim within a tool-use turn
  (signatures); across turns, pass all back and the API auto-filters
  (per-model defaults; Opus 4.5+/Sonnet 4.6+ keep all turns); explicit
  `clear_thinking_20251015` strategy with keep=N-turns|all; edited history
  invalidates preserved-thinking signatures.
- OpenAI: o1/o3-mini ignored reasoning on follow-ups; o3+ auto-includes
  "relevant" reasoning items (adjacent to function calls), recommends
  passing all and letting the API prune; GPT-5.6 exposes an explicit
  `reasoning.context: all_turns | current_turn | auto` dial; ChatGPT
  persists *reasoning recaps* (summaries), not raw CoT, across sessions.

**Gap nobody fills cleanly:** no published comparison of drop vs persist vs
recap (summarize-at-record-time) reasoning policies on real multi-day
sessions. Our pi corpus + open-loop replay can run exactly that ablation.

## Honcho & incremental representation memory (added Oct 2026)

Honcho v3 (Plastic Labs; plasticlabs.ai, honcho.dev) is out-of-band memory
infrastructure: messages go to a Storage service; async deriver workers
(Neuromancer, tiered Gemini/Anthropic) extract conclusions and merge them
into per-entity *representations* (peers = users/agents/groups/projects);
agents query `session.context(tokens=N)` (summary + messages + conclusions),
hybrid BM25+vector search, or the Dialectic chat endpoint at turn time.
Their thesis ("Memory as Reasoning"): representation maintenance is a
reasoning task, done incrementally off the critical path — never re-derived
from the growing transcript.

**Our port (`representation` arm) deliberately excludes** their async
scheduling, dialectic tiers, dreamer consolidation, and per-entity doc
partition — deployment machinery, not the mechanism under test. What it
keeps: one bounded memory doc (target 2048 tok) updated by merging each new
delta into maintained, sectioned state (facts, entities, open threads with
status), same trigger/tail/tools/model/accounting as flat-compact.

**Known deviations & confounds (from the pre-flight SME review):**
- single doc vs per-entity representations — results license claims about
  "single-doc incremental merge," not about Honcho-the-system
- no [§id] provenance in memory ("state conclusions, never quote
  transcripts") — prompt asymmetry beyond the merge rule
- MEMORY_TARGET_TOKENS=2048 vs flat's emergent ~1.1K summaries: resident
  size is conflated with the merge rule; attribute via view_tokens
  telemetry before claiming either
- corrected framing: flat's per-update input is also bounded (increments
  only since 52d1156); the isolated variable is update-rule semantics
  (merge into maintained state vs re-derive summary-of-summary), not input
  size. Empty-update truncation risk is identical in both arms; what
  differs structurally is that a truncated update after the first can no
  longer strip the view of its map
- the battery's comparison cells: flat P0' (no inventory instruction) and
  P1' (instructed inventory, failed via truncation) bracket the
  representation arm, whose inventory is maintained state, not per-fold
  output

**Why this arm matters:** the P0'/P1' experiments established that (a) the
true flat-vs-lcm continuity gap at fixed model is 10 pts @16K, and (b)
instructing fold-based summarizers to carry inventories degrades them
mechanically (truncation load). Incremental merge is the architecture-level
alternative — and if it wins, Honcho's docker stack (per-module OpenRouter
routing is first-class: `DERIVER_MODEL_CONFIG__OVERRIDES__BASE_URL`) is the
calibration point for what their full machinery adds on top.

**Battery result (LoCoMo n=10, 8K/16K, P0' baselines, 0 errors):**
continuity recall/abstention — representation 54%/20% @8K and 68%/10% @16K,
vs flat 34%/40% and 62%/30%, lcm 12%/80% and 52%/40%, instructed flat
(P1') 19%/70% and 30%/60%. Best continuity arm at both budgets with the
lowest abstention; QA parity with flat/retrieval (68-70% everywhere);
overhead ~55K tok/run (flat: 34K@8K, 18K@16K; lcm: 70K@8K, 30K@16K) —
the amortization premium, priced.

**Findings.** (1) Inventory persistence is architectural, not promptable:
maintained state beats both uninstructed folding and instructed folding
(which degrades mechanically via truncation load). (2) The arm degrades
gracefully under the shared truncation physics: 29-43% of merges came
back empty (completion-cap burn, same root cause measured at every level
this project) yet recall held — memory preserved + tail + tools absorb
the misses — whereas a fold arm's empty compaction wipes its entire map.
(3) Headroom: reasoning-separated merges (deferred 8192/provider-param
experiment) should recover the empty merges; continuity could rise
further at unchanged architecture.
