# Context budget tiers: what do real coding agents use?

Motivating question: is a 16K history budget realistic for coding agents,
given how large their fixed prompt overhead is? (Answer: no — it is a stress
tier. The deployment regime is ~96–165K. Details below.)

## Fixed overhead (system prompt + tool schemas), measured

| Agent | Fixed overhead | Notes / source |
|---|---|---|
| Claude Code (baseline) | ~27K tokens | measured from first-call `usage`: 27,169 from /tmp, 30,919 in a configured project (claudecodecamp.com). Tool definitions alone ~14–17K (GH anthropics/claude-code#26158: ~16.5K). System prompt text only 2.3–3.6K. |
| Claude Code (project + MCP) | 75–85K tokens | production setups with MCP tool schemas riding every turn (~40% of a 200K window) |
| OpenCode | ~7–10K | vs Claude Code 4.7x gap (developersdigest.tech comparison) |
| Cline | ~7K | system prompt + tool definitions |
| Pi | <1K | deliberate minimal-prompt design (tensorlake.ai writeup) |
| Generic per-tool cost | 550–1,400 tokens/tool | name + description + schema + field docs + few-shots |

Our harness: 239 tokens (minimal system prompt + recall tool schemas), and
`_build_engine` reserves fixed costs *outside* the history budget — so our
"budget" maps directly onto production's `window - overhead - reserve`.

## Auto-compact triggers / effective history budgets

| System | Trigger | Effective history budget |
|---|---|---|
| Claude Code, 200K models | ~95% of window, ~33K reserve | **~135–165K** |
| Claude Code, 1M models | ~967K default; configurable 100K–1M (`/autocompact`) | ~900K+ |
| Cline | ~80% of model window (128–200K models) | **~95–155K** |
| Anthropic API server-side compaction | near context window limit | window-bound |
| OpenHands LLMSummarizingCondenser | event-count threshold (~120 events); keep first N + last M | event-based, not token-based |
| MemGPT/Letta core memory | by design (2K chars/block, 5–15 blocks) | **2.5–6K tokens** |

Real-world compaction frequency (telemetry in GH anthropics/claude-code#55504,
607 compaction events / 153 sessions): 87% of multi-compaction sessions showed
sub-120s gaps between successive compactions — compaction thrash is a real
failure mode, and compaction runs when the model is at its most degraded
("context rot" — claude.com session-management writeup).

## Corpus distribution (data/swe_smoke.jsonl, exact DeepSeek tokenizer)

min 6.8K · p25 13.7K · median 32K · p75 51K · max 153K tokens.

## Tier framing used in reports

| Budget | Tier | Interpretation | Corpus impact |
|---|---|---|---|
| 16K | stress | MemGPT-class core memory with headroom; edge/local/cheap-model products. Below Claude Code's fixed overhead alone. | 17/22 traces overflow verbatim; compaction fires ~13x/session |
| 32K | stress+ | median trace just fits verbatim | 11/22 overflow |
| 64K | mid | compaction fires on the larger half of sessions only | 5/22 overflow |
| 128K | **deployment** | Claude-Code-on-200K-class (~135–165K history budget); Cline-class | 3/22 overflow; 0–3 compactions/session |
| 256K | uncapped | 1M-window / never-compact regime | 0 overflow; whole corpus verbatim |

The deployment tier answers the question production tools actually face:
compaction fires rarely — what breaks in the sessions where it does?

## Effective context: where does model ability actually top out?

Practitioner intuition (e.g., compaction sweet spot ~256K or ~70% of max)
is supported at two levels:

**Benchmark level — the coherence ceiling on hard tasks is much LOWER than
claimed windows:**

- NoLiMa (Adobe, ICML 2025; associative needles, minimal lexical overlap):
  13 models claiming >=128K; at 32K, 11 fall below 50% of their short-context
  baseline. Effective length (>=85% of base): **GPT-4.1 16K** (1M claimed),
  GPT-4o 8K (128K claimed), Llama 3.3 70B **2K** (128K claimed).
- RULER (NVIDIA, COLM 2024; 13 tasks incl. aggregation/multi-hop): most
  models' effective length 4–64K vs 32K–1M claims. Exceptions: Gemini 1.5
  Pro (1M) holds 96.7→94.4 from 4K to 128K — top models hold >128K on
  retrieval-flavored tasks; multi-hop/aggregation degrade much earlier.
- BABILong: popular LLMs effectively utilize **10–20%** of claimed context.
- "Same Task, More Tokens" (ACL 2024): padding alone drops reasoning
  0.92 → 0.68 by ~3K tokens — length hurts independent of retrieval.
- OpenAI's GPT-5.4 launch eval: 97.2% at 32K → 36.6% at 1M.

**Harness level — production tools cap usable context at ~256K or 70–90%
of the window, well below 1M:**

| Harness | Advertised | Usable / compact-at | Fraction |
|---|---|---|---|
| Codex / GPT-5.5 | 1.05M | session cap 400K; input allocation 272K − 5% headroom → **~258K usable** | ~25% |
| Claude Code / Opus (1M) | 1M | auto-compact **~420K** | ~42% |
| Claude Code / Sonnet 5 (1M) | 1M | auto-compact ~967K | ~97% |
| Google Antigravity | 1M (UI) | compacts **~256K** in practice | ~26% |
| Context Gateway (OSS proxy) | — | background compaction at 85% | 85% |
| Codex config example | 1M | `model_auto_compact_token_limit = 900000` | 90% |

The convergence is notable: harness authors do not trust the top half of
1M windows — the 256K figure that practitioners report as a sweet spot is
within a few percent of Codex's computed usable input (~258K).

**Implication for this eval's tiers:** our corpus caps at 153K, so the 256K
"uncapped" arm == the practical ceiling regime (Codex-class usable input).
The stress tiers (16–32K) sit inside NoLiMa/RULER effective-length territory
(8–64K) — i.e., where long-range attention is already degraded on hard
tasks, which is exactly the regime where moving information into a compact
structured view (LCM) or verbatim previews (ARC) should help most. The
deployment tier (~128K) is where top models still hold retrieval accuracy
but aggregation/multi-hop begin to slip.

## Related work: learned compaction policies (2026)

Two papers directly on our problem space — compaction decisions for
long-horizon coding agents, evaluated on SWE-bench-class tasks:

- **AutoCompact** (arXiv:2610.02163): SFT + RL for when-to-compact in
  coding agents; +9.2% pass@1 on SWE-bench Verified, evaluated at a 256K
  window that never overflows AND a 16K window with fallback compaction —
  the same two-regime frame as our stress/uncapped tiers.
- **CompactionRL** (arXiv:2607.05378): RL with context compaction for
  long-horizon agents; notes context scaling "does not fully solve degraded
  utilization over long sequences", echoing the effective-context literature.


- https://www.claudecodecamp.com/p/inside-claude-code-s-system-prompt (measured 27K/31K baseline)
- https://github.com/anthropics/claude-code/issues/26158 (tool defs ~16.5K)
- https://www.developersdigest.tech/blog/claude-code-token-overhead-opencode-comparison (33K vs 7K; 75–85K production)
- https://code.claude.com/docs/en/model-config (auto-compact windows, 967K/200K defaults, /autocompact 100K–1M)
- https://github.com/anthropics/claude-code/issues/55504 (trigger math, 33K buffer, thrash telemetry)
- https://github.com/anthropics/claude-code/issues/23711 (demand for configurable thresholds)
- https://cline-cline.mintlify.app/models/context-windows + cline/docs auto-compact.md (80% meter, auto-compact behavior; summary leverages prompt cache)
- https://claude.com/blog/using-claude-code-session-management-and-1m-context (compaction at weakest point)
- https://docs.letta.com/advanced/memory-management (2K-char blocks; archival/recall)
- https://docs.openhands.dev/sdk/arch/condenser (LLMSummarizingCondenser defaults)
## Sources

- NoLiMa: https://github.com/adobe-research/NoLiMa (effective lengths; GPT-4.1 16K, Llama 3.3 70B 2K)
- RULER: https://arxiv.org/abs/2404.06654 + https://github.com/NVIDIA/RULER (Gemini 1.5 Pro >128K; most models 4–64K effective)
- BABILong: https://arxiv.org/pdf/2406.10149 (10–20% effective utilization)
- Same Task, More Tokens: https://aclanthology.org/2024.acl-long.818.pdf (padding alone: 0.92→0.68 @ ~3K)
- Harness usable-context table: https://noqta.tn/en/blog/effective-context-window-ai-coding-agents-1m-token-myth-2026 (Codex ~258K usable; Claude Code ~420K/~967K; Antigravity ~256K) + https://community.openai.com/t/1-million-context-to-enable-professional-workloads/1390013 (Codex 1M config) + https://news.ycombinator.com/item?id=47367526 (Context Gateway 85%; GPT-5.4 eval 97.2%→36.6%)
- AutoCompact: https://arxiv.org/abs/2610.02163 (RL when-to-compact, SWE-bench +9.2%)
- CompactionRL: https://arxiv.org/pdf/2607.05378v1
