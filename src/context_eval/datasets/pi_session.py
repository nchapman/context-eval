"""Convert pi session transcripts into replay samples.

pi (the coding agent harness) persists verbatim multi-day session
transcripts under ~/.pi/agent/sessions/<project-slug>/<timestamp>.jsonl,
including compaction events. Those transcripts are the most authentic
extreme-length corpus available: real multi-day agent sessions with
user/assistant/tool-result messages at 0.3M-1.5M content tokens.

CLI:
    uv run python -m context_eval.datasets.pi_session <file.jsonl...> \
        --out data/pi_sessions.jsonl --max-tokens 1000000 \
        [--thinking drop|persist]

Role mapping: user -> user, assistant -> assistant, toolResult -> tool.
System messages (the harness's own prompts) and non-message records are
skipped; their counts are kept in sample meta.

Thinking policy (assistant "thinking" content parts):
- drop (default): excluded, matching what most deployed harnesses do at
  turn end; counts land in meta["thinking_parts"]
- persist: rendered inline before the assistant prose, matching what a
  keep-all history (Anthropic Opus 4.5+/Sonnet 4.6+ default) would show
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from context_eval.datasets.base import ReplayMessage, ReplaySample
from context_eval.tokens import default_counter
from context_eval.types import Role

_ROLE_MAP: dict[str, Role] = {
    "user": Role.USER,
    "assistant": Role.ASSISTANT,
    "toolResult": Role.TOOL,
}


def convert(
    path: str | Path,
    max_tokens: int | None = None,
    thinking: str = "drop",
    recap_cache: dict[str, str] | None = None,
) -> ReplaySample:
    """Parse one pi session transcript into a ReplaySample.

    ``max_tokens`` keeps the longest message-prefix whose cumulative exact
    token count stays within the cap (None keeps the whole session).
    ``thinking`` selects the reasoning-persistence policy (see module doc).
    """
    counter = default_counter()
    messages: list[ReplayMessage] = []
    skipped: dict[str, int] = {}
    thinking_parts = 0
    compactions = 0
    first_ts = last_ts = ""

    with Path(path).open() as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                skipped["unparseable"] = skipped.get("unparseable", 0) + 1
                continue
            rtype = rec.get("type")
            if rtype == "compaction":
                compactions += 1
                continue
            if rtype != "message":
                continue
            ts = rec.get("timestamp", "")
            if ts:
                last_ts = ts
                first_ts = first_ts or ts
            msg = rec.get("message", {})
            role = _ROLE_MAP.get(msg.get("role"))
            if role is None:  # system prompts and anything exotic
                skipped[msg.get("role", "?")] = skipped.get(msg.get("role", "?"), 0) + 1
                continue
            content = msg.get("content")
            thinking_texts: list[str] = []
            prose_texts: list[str] = []
            if isinstance(content, list):
                for part in content:
                    if not isinstance(part, dict):
                        continue
                    if part.get("type") == "thinking":
                        thinking_texts.append(part.get("thinking", ""))
                    else:
                        prose_texts.append(part.get("text", ""))
            elif isinstance(content, str):
                prose_texts.append(content)
            thinking_parts += len(thinking_texts)
            if thinking == "persist" and thinking_texts:
                body = "\n".join(thinking_texts + prose_texts)
            elif thinking == "recap" and thinking_texts:
                # ChatGPT-style reasoning recaps: each thinking block
                # replaced by its cached <=2-sentence summary at record time.
                # Key must match scripts/thinking_arms.py recap: sha1 of the
                # joined thinking text, rstrip'd. Uncached blocks are dropped.
                joined = "\n".join(thinking_texts).rstrip("\n")
                recap = (recap_cache or {}).get(hashlib.sha1(joined.encode()).hexdigest(), "")
                body = "\n".join(([recap] if recap.strip() else []) + prose_texts)
            else:
                body = "\n".join(prose_texts)
            messages.append(ReplayMessage(role=role, content=body))

    if max_tokens is not None:
        total = 0
        kept = 0
        for i, m in enumerate(messages):
            total += counter.count_text(m.content) + 4
            if total > max_tokens:
                break
            kept = i + 1
        messages = messages[:kept]

    session_id = Path(path).stem
    return ReplaySample(
        dataset="pi-session",
        sample_id=session_id,
        messages=tuple(messages),
        meta={
            "source": str(path),
            "skipped_roles": skipped,
            "compaction_events": compactions,
            "first_timestamp": first_ts,
            "last_timestamp": last_ts,
            "prefix_capped": max_tokens is not None,
            "thinking": thinking,
            "thinking_parts": thinking_parts,
        },
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, default=Path("data/pi_sessions.jsonl"))
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--thinking", choices=["drop", "persist"], default="drop")
    args = parser.parse_args(argv)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    counter = default_counter()
    with args.out.open("w") as out:
        for path in args.files:
            sample = convert(path, args.max_tokens, thinking=args.thinking)
            total = counter.count_messages(sample.messages)
            print(
                f"{sample.sample_id}: {len(sample.messages)} messages, "
                f"{total:,} tokens, {sample.meta['compaction_events']} compactions, "
                f"{sample.meta['thinking_parts']} thinking parts "
                f"(skipped: {sample.meta['skipped_roles']})",
                file=sys.stderr,
            )
            out.write(sample.to_json() + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
