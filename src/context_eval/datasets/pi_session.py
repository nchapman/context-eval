"""Convert pi session transcripts into replay samples.

pi (the coding agent harness) persists verbatim multi-day session
transcripts under ~/.pi/agent/sessions/<project-slug>/<timestamp>.jsonl,
including compaction events. Those transcripts are the most authentic
extreme-length corpus available: real multi-day agent sessions with
user/assistant/tool-result messages at 0.3M-1.5M content tokens.

CLI:
    uv run python -m context_eval.datasets.pi_session <file.jsonl...> \
        --out data/pi_sessions.jsonl --max-tokens 1000000

Role mapping: user -> user, assistant -> assistant, toolResult -> tool.
System messages (the harness's own prompts) and non-message records are
skipped; their counts are kept in sample meta.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from context_eval.datasets.base import ReplayMessage, ReplaySample
from context_eval.tokens import default_counter
from context_eval.types import Role

_ROLE_MAP: dict[str, Role] = {
    "user": Role.USER,
    "assistant": Role.ASSISTANT,
    "toolResult": Role.TOOL,
}


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            part.get("text", "") for part in content if isinstance(part, dict)
        )
    return ""


def convert(
    path: str | Path,
    max_tokens: int | None = None,
) -> ReplaySample:
    """Parse one pi session transcript into a ReplaySample.

    ``max_tokens`` keeps the longest message-prefix whose cumulative exact
    token count stays within the cap (None keeps the whole session).
    """
    counter = default_counter()
    messages: list[ReplayMessage] = []
    skipped: dict[str, int] = {}
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
            messages.append(ReplayMessage(role=role, content=_message_text(msg.get("content"))))

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
        },
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, default=Path("data/pi_sessions.jsonl"))
    parser.add_argument("--max-tokens", type=int, default=None)
    args = parser.parse_args(argv)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    counter = default_counter()
    with args.out.open("w") as out:
        for path in args.files:
            sample = convert(path, args.max_tokens)
            total = counter.count_messages(sample.messages)
            print(
                f"{sample.sample_id}: {len(sample.messages)} messages, "
                f"{total:,} tokens, {sample.meta['compaction_events']} compactions "
                f"(skipped: {sample.meta['skipped_roles']})",
                file=sys.stderr,
            )
            out.write(sample.to_json() + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
