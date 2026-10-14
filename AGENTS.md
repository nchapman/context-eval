# context-eval

Clean-room benchmark of context-management strategies for coding and
conversational agents. Replays recorded transcripts (SWE traces, LoCoMo
dialogs, LongMemEval S/M sessions, real multi-day pi sessions, and
~5.5M-token interleaved "mega" sessions) through pluggable strategies at
fixed token budgets and measures needle/QA recovery, uncued inventory
(continuity), status currency, overflow behavior, and cost. The findings
live in `docs/findings.md` (data-first; recomputed by `analysis/`); the
README is the main report.

## Layout

- `src/context_eval/strategies/` — the arms (REGISTRY in `__init__.py`):
  base arms `full-context`, `sliding-window`, `retrieval`, `context-editing`,
  `flat-compact` (+`-v2` cue-card), `arc`, `lcm`, `representation`,
  `conclusions`; and the `sawtooth-*` family (one pinned schedule,
  interchangeable backs: `none`/`rewrite`/`segments`/`ledger`..`ledger5` —
  ledger5 implements `docs/resident-memory-theory.md`)
- `driver.py` — replay loop: append → `on_turn` → assemble → budget check;
  question-time tool interleaving; per-turn telemetry
- `questions.py` — seeded needle/continuity/currency injection
- `judge.py`, `report.py` — scoring (needle substring, LoCoMo F1, official
  LongMemEval check) and aggregation (acc / acc* overflow-penalized, depth
  curves, sanity gates — refuses cells with blank compactions)
- `tokens.py` — `TokenCounter.count_message()` is the ONLY per-message
  cost authority (content + overhead + tool_call args)
- `provider.py` — OpenRouter client, vetted-fp8 allowlist rotation, retry
  ladder; `sweeps.py`/`long_horizon.py` — offline dial sweeps + long-horizon
  probes over recordings
- `datasets/` — loaders (`swe`, `locomo`, `pi_session`, `longmemeval`);
  `scripts/` — corpus builders (`build_devset`, `mega_sessions`,
  `thinking_arms`, `trace_qa`), run tooling (`preflight`, `run_suite`,
  `run_sharded`, `smoke_provider`, `snapshot_views`), theory tooling
  (`dial_sweeps`, `seal_render_probe`, `extraction_bench`)
- `analysis/` — independent recomputation of every findings number from
  raw run rows; imports NOTHING from src (see its README)
- `docs/` — `findings.md` (the evidence report), `lit-review.md` (per-arm
  sources + full run history), `resident-memory-theory.md`,
  `dataset-plan.md`, `budgets.md`
- `data/`, `results/` — corpora and run outputs (gitignored; see the
  tracked `data/README.md` and `results/README.md` manifests)

## Commands

```bash
uv run pytest -q                 # tests (must be green to commit)
uv run ruff check src tests scripts analysis
uv run python scripts/preflight.py --leg swe     # BEFORE any live leg
                                  # legs: swe, locomo, pi, thinking, mega
uv run python -m context_eval.run_eval --path data/longmemeval_dev100.jsonl \
  --strategies flat-compact,lcm,sawtooth-ledger5 --budgets 64000 \
  --samples 100 --out results/x.jsonl   # --shard i/n for parallel workers
uv run python -m context_eval.report results/x.jsonl --out results/report.md
./scripts/run_suite.sh [--skip-<leg>]            # 3-worker full suite
```

Model: `deepseek/deepseek-v4.1-flash` via OpenRouter (`OPENROUTER_API_KEY`
in `.env`; `env.load_env()` reads it). Live runs are paid and sequential
per worker; never exceed 3 concurrent workers. LongMemEval legs answer
with the official LLM check; budget via `scripts/preflight.py` first —
corpus rows are 0.5–5 MB, so samples not questions dominate cost.

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

**Provider economics (measured, 2025-10-14).** Cheap iteration pins
baidu/fp8 ($0.10/M in) but its shared-pool TPM dies even under a
SEQUENTIAL 16K fold loop — allowlists need >=2 vetted tags so the
client's 429 rotation has somewhere to go (`baidu/fp8,inference-net/fp8`
absorbed 5 throttles with zero run deaths). Cost estimates come from
measured $/run of the SAME run shape (a modeled estimate missed 2.6x;
measured projections landed within 4%). Measured @16K fold run:
$0.26 baidu / $0.31-0.35 inference-net / $0.42 with a bigger artifact.
Zero cached tokens on fold streams — do not budget for KV reuse.

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
