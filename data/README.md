# data/ — corpus manifest (this file is tracked; everything else here is not)

~6.2 GB on disk. All corpora are reproducible from loaders in
`src/context_eval/datasets/` and scripts in `scripts/`; nothing here is
committed. If a file is missing, rebuild it with the command listed.

## LongMemEval (ICLR 2025, Wu et al. — the conversational-QA axis, findings §5)

| file | what it is | how to rebuild |
|---|---|---|
| `longmemeval_s_cleaned.json` / `_m_cleaned.json` | official cleaned downloads of `xiaowu0162/longmemeval-cleaned` (S = 500 sessions ~120K tok; M = 500 ~1.2M tok) | HF download |
| `longmemeval_s.jsonl` / `_m.jsonl` | replay-format conversions | `uv run python -m context_eval.datasets.longmemeval data/longmemeval_s_cleaned.json data/longmemeval_s.jsonl` |
| `longmemeval_dev25.jsonl` | 25-sample stratified TUNING split (seed 25) — never use for eval numbers | `uv run python scripts/build_devset.py 25` |
| `longmemeval_dev100.jsonl` | dev25's 25 seen + 75 fresh (seed 100) — the seen/fresh overfit check | `uv run python scripts/build_devset.py 100` |
| `longmemeval_m25*.jsonl`, `_s25*.jsonl` | 25-sample stratified M/S subsamples; `fresh`/`fresh2` suffixes = disjoint question sets A/B paired with the base-vs-fixed sweeps in `results/longmemeval/m25/` | ad-hoc stratified subsampling of the loaders (commands in git history around 10-12/13) |
| `longmemeval_m30_v2dev.jsonl` | 30 unseen M samples for the 16K/128K extremes bracket | same |
| `longmemeval_m6_promptdev.jsonl` | 6 hardest samples (3 multi-session + 3 knowledge-update) for the cue-card pilot | same |

## LoCoMo (ACL 2024, Maharana et al. — dialog QA, stress tier)

| file | what it is | how to rebuild |
|---|---|---|
| `locomo10.json` | raw corpus download (10 dialogs, ~300 turns each) | fetched by the loader |
| `locomo.jsonl` | replay-format conversion | `uv run python -m context_eval.datasets.locomo` |

## SWE traces (NVIDIA Open-SWE-Traces — coding-agent haystack)

| file | what it is | how to rebuild |
|---|---|---|
| `swe_smoke.jsonl` | stratified trajectory sample | `uv run python -m context_eval.datasets.open_swe` |
| `swe_trace_qa.json` | 88 grounded QA pairs about real session facts | `uv run python scripts/trace_qa.py` |

## Real pi agent sessions (the 1–2M-token corpus, findings §2/§6)

| file | what it is | how to rebuild |
|---|---|---|
| `pi_sessions.jsonl` | recorded multi-day sessions from `~/.pi/agent/sessions/` | `uv run python -m context_eval.datasets.pi_session` |
| `pi_mega.jsonl`, `pi_mega_a/b.jsonl` | ~5.5M-token interleaves ("many projects over months", rotating order, ~115–153 compaction teeth) | `uv run python scripts/mega_sessions.py --cycles 4 --out data/pi_mega.jsonl` |
| `thinking_drop/persist/recap.jsonl` + `thinking_recaps.json` | reasoning-representation variants of the pi sessions (R7 ablation) | `uv run python scripts/thinking_arms.py` |

## Tokenizer

| file | what it is |
|---|---|
| `deepseek_tokenizer.json` | DeepSeek tokenizer backing `TokenCounter` (the one cost authority) |
