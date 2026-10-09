# context-eval

Clean-room benchmark of context-management strategies for coding agents.
Replays recorded agent transcripts (SWE traces, LoCoMo dialogs, multi-day pi
sessions) through pluggable strategies at fixed token budgets and measures
needle/QA recovery, overflow behavior, and cache-aware cost. Headline
question: does LCM-style hierarchical compaction beat the deployed default
(flat `/compact`), and where?

## Layout

- `src/context_eval/strategies/` — the seven arms (see REGISTRY in
  `__init__.py`): `full-context`, `sliding-window` (anchors),
  `context-editing` (Anthropic clear_tool_uses), `flat-compact` (Claude
  Code /compact, OpenHands condenser), `retrieval` (BM25, recall.ts
  policy), `arc` (arXiv:2607.25066), `lcm` (arXiv:2605.04050)
- `driver.py` — replay loop: append → `on_turn` → assemble → budget check;
  question-time tool interleaving; per-turn telemetry
- `questions.py` — seeded needle injection (randomized line placement)
- `judge.py`, `report.py` — scoring (needle substring, LoCoMo F1) and
  aggregation (acc / acc* overflow-penalized, depth curves, Pareto)
- `tokens.py` — `TokenCounter.count_message()` is the ONLY per-message
  cost authority (content + overhead + tool_call args)
- `provider.py` — OpenRouter client, any-fp8 routing, 7-attempt backoff
- `datasets/` — loaders (`swe`, `locomo`, `pi_session`); `scripts/` —
  `preflight.py`, `run_suite.sh`, `thinking_arms.py`
- `docs/` — `budgets.md` (tier framing), `lit-review.md` (per-paper
  summaries + regime/model-behavior findings), `candidates.md`
- `data/`, `results/` — corpora and run outputs (gitignored)

## Commands

```bash
uv run pytest -q                 # tests (must be green to commit)
uv run ruff check src tests scripts
uv run python scripts/preflight.py --leg swe     # BEFORE any live leg
uv run python -m context_eval.run_eval --path data/swe_smoke.jsonl \
  --strategies lcm --budgets 16000 --samples 3 --out results/x.jsonl
uv run python -m context_eval.report results/x.jsonl --out results/report.md
./scripts/run_suite.sh [--skip-<leg>]            # 3-worker full suite
```

Model: `deepseek/deepseek-v4.1-flash` via OpenRouter (`OPENROUTER_API_KEY`
in `.env`; `env.load_env()` reads it). Live runs are paid and sequential
per worker; never exceed 3 concurrent workers.

## Best practices

**Sanity checks at every level — they must not depend on the author
noticing.** Tests written by the author of a bug pass; every real catch in
this project came from an independent gate:
- `scripts/preflight.py` before ANY live run leg: prints question counts,
  expected calls/wall-clock/cost, hard-fails on ceilings (3000 questions,
  90 min/leg) unless `--allow` with a stated reason.
- `report.py` sanity warnings: zero-answer cells unexplained by overflow,
  0.00-acc cells, errored runs. Verify flagged cells before dismissing
  (blank predictions = provider failure; real wrong answers = true zero).
- SME subagent review for new components and component-boundary changes
  (injection × previews, counter × strategies). Run it before live spend,
  not after.

**Regression-first tests.** When fixing a bug, first write the test that
would have caught it, against the regime where it lived (e.g. flat's
pre-trigger view), and confirm it fails for the right reason.

**Contract changes propagate everywhere in the same commit.** When
changing shared accounting (`count_message`, budget semantics), grep every
consumer and update all internal arithmetic. Half-propagated fixes become
live-only bugs (flat overflowed at 16K because its trigger skipped
tool-call args after `count_messages` was fixed).

**Diff invocations when promoting smoke → suite.** The LoCoMo runaway
happened because the question cap lived in the smoke command and not the
suite script. Parameter-for-parameter comparison, then commit.

**Budget semantics.** Budget bounds the strategy view only; system prompt,
  tool schemas, and the question are reserved outside. All per-message
  cost math goes through `TokenCounter.count_message()` — never
  `count_text(content) + MESSAGE_OVERHEAD_TOKENS` by hand. Synthesized
  strings (summaries, stubs, citations) use `count_text`.

**Strategy fidelity.** Each arm implements its reference design as
competently as possible; deviations go in the module docstring and
`docs/lit-review.md`. Do not bolt cross-arm capabilities onto an arm (no
search tools for editing, no auto-recall for ARC). Documented regime
boundaries (ARC/editing overflow on assistant-heavy input; editing ≡
full-context on tool-free corpora) are findings, not bugs.

**Needle discipline.** Seeded, position-randomized within observations
(random line boundary — never end-append: ARC's tail preview read every
needle for free once). Same seed ⇒ same needles across all arms.

**Failure signatures.** 429 exhaustion looks like blank predictions and
zero-acc cells, not error rows — check `error` fields AND predictions
before diagnosing strategy behavior. Overflowed runs answer nothing and
score 0 by design (that is `acc*`).

**Never edit `run_suite.sh` (or any bash script) while it executes** —
bash reads incrementally. Wait for completion or work on a copy.

**Background jobs & process matching** (three incidents, one rule each):
- `pgrep/pkill -f` patterns must not match your own wrapper: use a
  character class (`context_eval[.]run_eval`) — a literal pattern in your
  command line matches the command itself (a watcher that never exits, a
  pkill that dies mid-sweep at exit 143).
- A function that backgrounds its work internally (`run() { uv ... & }`)
  must be called WITHOUT a trailing `&`: the extra `&` turns the call into
  a grandchild the invoking shell's `wait` never sees — the script
  'completes' while workers are orphaned mid-run.
- A task's completion only covers its own process tree. Anything you
  backgrounded separately keeps running: verify with `ps` (by PID list,
  not pattern assumptions) before treating work as finished.

**Commits.** Imperative subject, evidence in the body (measurements,
file:line, what would have caught it). Tests + lint green per commit.
Runs stay out of git (`results/`); corpora in `data/` are reproducible via
loaders and `scripts/thinking_arms.py`.
