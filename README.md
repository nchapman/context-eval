# context-eval

A benchmark project evaluating strategies for "lossless" context management in LLM agents.

**Goal:** measure both **effectiveness** (can the agent answer questions about its past?) and **efficiency** (tokens, LLM calls, latency, cost) of the popular context-management strategies, head-to-head, under identical budgets and models.

**Method:** clean-room implementations of each strategy, derived from papers and public documentation rather than ported code. All implementations are judged practical: roughly ≤500 LOC each, minimal dependencies.

## Status

- [x] Literature review ([docs/lit-review.md](docs/lit-review.md))
- [x] Strategy candidate matrix ([docs/candidates.md](docs/candidates.md))
- [ ] Eval harness scaffold
- [ ] Implementations
- [ ] Results

## Context

Hiro (github.com/nchapman/hiro) implements an LCM-style summary-DAG strategy
(soft/hard thresholds, depth-graded hierarchical summarization, FTS5 history
recall). This project exists to answer: is that the right strategy, and where
are the tradeoffs? See the [LCM paper](https://arxiv.org/abs/2605.04050) for
the architecture hiro follows.
