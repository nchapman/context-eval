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

## Sources

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
- https://www.tensorlake.ai/blog/pi-coding-agent-efficient-system-prompting (<1K system+tools)
