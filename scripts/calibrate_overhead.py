"""Calibrate the per-message overhead term (MESSAGE_OVERHEAD_TOKENS).

Sends single-message and two-message chat calls with real transcript
slices at varied lengths, records provider prompt_tokens, and fits

    actual = a * (chars / CHARS_PER_TOKEN) + b * n_messages

b is the per-message framing overhead our budget math charges via
MESSAGE_OVERHEAD_TOKENS; a is the char-estimator scale. Usage: run with
OPENROUTER_API_KEY set; prints the fit and the recommended constant.
"""

import statistics
import sys

from context_eval.env import load_env
from context_eval.provider import ChatMessage, OpenRouterClient

load_env()


def slices_from_corpus(path="data/locomo.jsonl", n=16):
    import json

    texts = []
    for line in open(path):
        rec = json.loads(line)
        for m in rec.get("messages", []):
            c = m.get("content") or ""
            if 200 < len(c) < 4000:
                texts.append(c)
        if len(texts) >= n:
            break
    return texts[:n]


def main() -> int:
    import httpx

    client = OpenRouterClient(
        model="deepseek/deepseek-v4.1-flash",
        http_client=httpx.Client(timeout=120),
    )
    texts = slices_from_corpus()
    single, double = [], []
    for t in texts:
        c = client.complete(
            [ChatMessage(role="user", content=t)],
            max_tokens=1,
            reasoning={"enabled": False},
        )
        single.append((len(t), c.usage.prompt_tokens))
        print(f"1msg len={len(t):5d} est={len(t)//4:5d} actual={c.usage.prompt_tokens:5d}",
              file=sys.stderr)
    for t in texts[: len(texts) // 2]:
        c = client.complete(
            [ChatMessage(role="user", content=t), ChatMessage(role="user", content=t)],
            max_tokens=1,
            reasoning={"enabled": False},
        )
        double.append((len(t), c.usage.prompt_tokens))
        print(f"2msg len={len(t):5d} actual={c.usage.prompt_tokens:5d}", file=sys.stderr)

    # per-message increment at matched lengths (same text twice)
    increments = [p2 - p1 for (n, p1) in single[: len(double)] for (n2, p2) in double if n2 == n]
    est = [n / 4 for n, _ in single]
    act = [p for _, p in single]
    a = sum(e * x for e, x in zip(est, act, strict=True)) / sum(e * e for e in est)
    resid = [x - a * e for x, e in zip(act, est, strict=True)]
    print(f"\nscale a={a:.3f} (1.000 = 4 chars/token exact)")
    print(f"intercept b={statistics.median(resid):.1f} tokens/message")
    if increments:
        print(f"paired 1msg->2msg increment: median {statistics.median(increments):.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
