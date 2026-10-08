# Strategy Candidates for Evaluation

Each candidate gets a clean-room implementation sharing one harness, one
transcript format, and one model. Estimated LOC assumes Go (hiro-compatible)
or Python — the core logic is language-agnostic.

Selection criteria:
1. **Representative** — covers a distinct point on the design axes
2. **Clean-roomable** — mechanism fully described in a paper or public doc; ≤ ~500 LOC
3. **No heavyweight deps** — no external services; SQLite + optional local embedder

## Design axes

- **Temporal vs semantic grouping** (Contexto's critique)
- **Strictly lossless vs effectively lossless vs lossy**
- **LLM-driven vs deterministic compaction**
- **Engine-managed vs model-managed** (recall as engine tool vs agent skill)
- **Verbatim recall vs abstracted memory** (transcript vs facts/events)

## The matrix

| # | Strategy | Family | Losslessness | Compaction cost | Est. LOC | Verdict |
|---|---|---|---|---|---|---|
| 0a | Full context | baseline | strict | — | ~20 | **anchor (upper bound)** |
| 0b | Sliding window | baseline | lossy | — | ~20 | **anchor (lower bound)** |
| 1 | Flat compaction (Claude Code-style single-shot summary) | B | lossy | 1 LLM call per compaction | ~150 | **yes** — industry default, must be in |
| 2 | LCM summary DAG + recall tools | C1 | effective | LLM calls per sweep | ~500 | **yes** — hiro's strategy; paper + hiro as design refs only |
| 3 | ARC addressable citations + `_recall` | C2 | strict | **zero** (deterministic) | ~300 | **yes** — cheapest lossless; strongest per-token claims |
| 4 | BM25/FTS5 search over verbatim history | C3 | strict | zero | ~150 | **yes** — no LLM, isolates "search alone" |
| 5 | Embedding top-k RAG over verbatim history | C3 | strict | embedder only | ~250 | **yes** — the standard RAG baseline (RAG-4096) |
| 6 | Contexto AGNES clustering + budget packing | C4 | strict | rebuilds (no LLM) | ~450 | **yes** — the semantic-organization thesis |
| 7 | Mem0-lite fact store (extract + ADD/UPDATE/DELETE) | D | lossy (abstracted) | 1 LLM call per turn/batch | ~300 | **yes** — the "agent knows facts" point |
| 8 | EMem event memory | D | lossy (abstracted) | 1 LLM call per batch | ~300 | **yes** — "simple yet strong" claim worth testing |
| 9 | RAPTOR cluster+summarize tree | C5 | effective | LLM + clustering | ~400 | **maybe** — v2; overlaps 2+6 |
| 10 | MemGPT/Letta tiered memory | C/D hybrid | effective | LLM (recursive summary + paging) | ~500 | **maybe** — v2; historically important |
| 11 | RLM-lite (transcript-as-file + code tools) | E | strict | model-driven | ~200 scaffold | **maybe** — policy quality dominates; hard to make fair |
| 12 | Zep/Graphiti temporal KG | D | abstracted | heavy (LLM graph ops) | — | **no** — use library as reference point at most |

## Recommended first wave (0a, 0b, 1–8)

Ten implementations covering every axis with pairwise contrasts built in:

- **1 vs 2**: does hierarchy + recall beat single-shot? (compounding-loss question)
- **2 vs 3**: LLM summarization vs deterministic citation at the same recall-tool budget
- **3 vs 4 vs 5**: pointers vs lexical search vs dense search for recovery
- **4/5 vs 6**: flat retrieval vs semantic organization (temporal-vs-semantic axis)
- **2 vs 7/8**: transcript-anchored vs abstracted memory
- **0a vs everything**: is any of this better than just paying for context?
- **0b vs everything**: how much do the fancy strategies actually buy?

Second wave (9–11) once the harness proves out.

## Shared harness sketch (to scaffold next)

**Existing harnesses to be aware of before building** (details in lit-review §4.1):

- **MemoryAgentBench** (MIT, ICLR 2026) — closest fit: agent-shaped (incremental multi-turn), pluggable methods layer, built-in datasets/metrics. Borrow its interaction pattern and datasets at minimum; possibly extend rather than build from scratch.
- **mem0/evaluation** — the field's LoCoMo reference harness (offline ingest→retrieve→answer→judge). Borrow its judging prompts for comparability.
- **maximem memory_and_context_eval_harness** — clean Provider-ABC design, but offline only; compaction strategies need online interaction.

Decision deferred until scaffolding: extend MemoryAgentBench vs. clean harness that reuses its datasets and metric conventions. Either way our additions are the same — efficiency telemetry, budget enforcement, strategy adapters.

```
Strategy interface:
  OnTurn(transcript append)            // optional mutation/compaction trigger
  Assemble(query, budget) -> messages  // the active view for this LLM call
  Tools() -> []tool                    // optional recall/search tools
  Stats() -> Usage                     // tokens, LLM calls, latency, peak occupancy

Harness responsibilities:
  - drives a conversation against a benchmark dataset
  - enforces the fixed budget (hard fail on overflow)
  - interleaves tool calls (recall/search) per strategy
  - records per-turn telemetry + compaction events
  - scores with benchmark-specific judges
```

Transcript format: append-only JSONL of messages + tool observations with
stable IDs (needed by 2, 3, 4, 5, 6, 11 alike — the "addressable store" is
shared infrastructure, not a strategy).

## Efficiency/effectiveness reporting

Primary plot: **accuracy vs mean tokens-per-turn** Pareto frontier, one point
per (strategy × budget). Secondary: accuracy vs compaction count (compounding
curve); recovery-tool call rate; added latency by regime (none/async/blocking).

## Sources per candidate

1. Claude Code docs (code.claude.com/docs/en/context-window); Context Compaction Theory arXiv:2608.01326
2. LCM arXiv:2605.04050; hermes-lcm; hiro design (clean-room from paper)
3. ARC arXiv:2607.25066
4. standard BM25/FTS5; hiro HistorySearch concept
5. RAG-4096 baseline as described in arXiv:2511.17208
6. contexto discussion #114 (ekailabs/contexto); AGNES/UPGMA via standard clustering
7. Mem0 arXiv:2504.19413 (ops simplified to ADD/UPDATE/DELETE)
8. EMem arXiv:2511.17208
9. RAPTOR arXiv:2401.18059
10. MemGPT arXiv:2310.08560
11. RLM arXiv:2512.24601
