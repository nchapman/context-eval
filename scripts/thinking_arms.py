"""Build the thinking-persistence ablation arms from one pi session.

Three corpora over the SAME message span (identical needle positions,
seed-matched), differing only in how assistant thinking is represented:

- drop:    thinking removed (what most harnesses do at turn end)
- persist: thinking inline, verbatim (keep-all history)
- recap:   thinking replaced by a short LLM summary at record time
           (ChatGPT-style reasoning recaps), from a hash-keyed cache

Usage:
    uv run python scripts/thinking_arms.py build <session.jsonl> \
        --span-tokens 1000000 --seed 7
    uv run python scripts/thinking_arms.py recap <session.jsonl> \
        --span-tokens 1000000 --batch 8

`build` writes data/thinking_drop.jsonl and data/thinking_persist.jsonl
(recap arm after `recap` fills data/thinking_recaps.json). Span is chosen
so the *persist* variant totals <= --span-tokens.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from context_eval.datasets.base import ReplaySample
from context_eval.datasets.pi_session import convert
from context_eval.tokens import default_counter
from context_eval.types import Role

DATA = Path("data")


def _span_index(persist: ReplaySample, span_tokens: int, counter) -> int:
    total = 0
    for i, m in enumerate(persist.messages):
        total += counter.count_text(m.content) + 4
        if total > span_tokens:
            return i
    return len(persist.messages)


def _slice(sample: ReplaySample, n: int, dataset: str) -> ReplaySample:
    return ReplaySample(
        dataset=dataset,
        sample_id=sample.sample_id,
        messages=sample.messages[:n],
        meta=dict(sample.meta),
    )


def build(path: Path, span_tokens: int, seed: int, needles: int) -> None:
    counter = default_counter()
    persist_full = convert(path, thinking="persist")
    drop_full = convert(path, thinking="drop")
    n = _span_index(persist_full, span_tokens, counter)
    print(f"span: first {n} messages of {len(persist_full.messages)}")

    # NOTE: needles are NOT injected here. run_eval injects them with its
    # own seed; identical spans + identical roles give identical needle
    # positions across arms. Injecting here would double-needle at run time.
    for name, sample in (
        ("drop", drop_full),
        ("persist", persist_full),
    ):
        sliced = _slice(sample, n, dataset=f"pi-thinking-{name}")
        total = counter.count_messages(sliced.messages)
        print(f"{name:8} {len(sliced.messages)} msgs, {total:,} tokens")
        out = DATA / f"thinking_{name}.jsonl"
        out.write_text(sliced.to_json() + "\n")
        print(f"wrote {out}")


def recap(path: Path, span_tokens: int, batch: int) -> None:
    """Summarize thinking blocks into data/thinking_recaps.json (hash-keyed).

    Works on the raw session so each thinking part is recapped exactly once;
    the cache is shared across arms and runs.
    """
    from context_eval.env import load_env
    from context_eval.provider import OpenRouterClient

    load_env()
    counter = default_counter()
    persist_full = convert(path, thinking="persist")
    drop_full = convert(path, thinking="drop")
    n = _span_index(persist_full, span_tokens, counter)

    # Recover per-message thinking text: persist content starts with the
    # thinking block(s); drop content is the prose remainder.
    cache_path = DATA / "thinking_recaps.json"
    cache: dict[str, str] = {}
    if cache_path.is_file():
        cache = json.loads(cache_path.read_text())

    pending: list[tuple[str, str]] = []  # (hash, thinking text)
    for pm, dm in zip(persist_full.messages[:n], drop_full.messages[:n], strict=False):
        if pm.role is not Role.ASSISTANT or pm.content == dm.content:
            continue
        if dm.content and pm.content.endswith(dm.content):
            thinking = pm.content[: len(pm.content) - len(dm.content)]
        else:
            thinking = pm.content
        thinking = thinking.rstrip("\n")
        if not thinking.strip():
            continue
        key = hashlib.sha1(thinking.encode()).hexdigest()
        if key not in cache:
            pending.append((key, thinking))

    print(f"thinking blocks in span: {sum(1 for _ in _iter_thinking(persist_full, drop_full, n))}, "
          f"uncached: {len(pending)}")
    if not pending:
        return

    llm = OpenRouterClient()
    # flush every `batch` blocks: one numbered-list call per batch
    for i in range(0, len(pending), batch):
        chunk = pending[i : i + batch]
        numbered = "\n".join(f"[{j+1}] {text[:4000]}" for j, (_, text) in enumerate(chunk))
        prompt = (
            "Summarize each numbered reasoning excerpt in <=2 sentences each. "
            "Preserve conclusions, decisions, unresolved questions, and any "
            "specific values, identifiers, or file names. Drop exploration "
            "dead-ends. Reply with the same numbered list, one recap per item.\n\n"
            + numbered
        )
        result = llm.complete([{"role": "user", "content": prompt}])
        lines = [ln.strip() for ln in result.text.splitlines() if ln.strip()]
        recaps: dict[int, str] = {}
        for line in lines:
            head, _, rest = line.partition("]")
            num = head.lstrip("[")
            if num.isdigit():
                recaps[int(num)] = rest.strip()
        for j, (key, _) in enumerate(chunk, start=1):
            cache[key] = recaps.get(j, "")
        cache_path.write_text(json.dumps(cache))
        print(f"  recapped {min(i + batch, len(pending))}/{len(pending)}", file=sys.stderr)


def _iter_thinking(persist: ReplaySample, drop: ReplaySample, n: int):
    for pm, dm in zip(persist.messages[:n], drop.messages[:n], strict=False):
        if pm.role is Role.ASSISTANT and pm.content != dm.content:
            yield pm


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("build", "recap"):
        p = sub.add_parser(name)
        p.add_argument("session", type=Path)
        p.add_argument("--span-tokens", type=int, default=1_000_000)
        if name == "build":
            p.add_argument("--seed", type=int, default=7)
            p.add_argument("--needles", type=int, default=6)
        else:
            p.add_argument("--batch", type=int, default=8)
    args = parser.parse_args(argv)

    if args.cmd == "build":
        build(args.session, args.span_tokens, args.seed, args.needles)
    else:
        recap(args.session, args.span_tokens, args.batch)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
