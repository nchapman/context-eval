"""Generate trace-derived Q&A pairs for SWE corpora (cached).

Needles measure retention of injected synthetic facts; trace QA measures
retention of the session's own real information (actual errors, files,
decisions). For each sample, seeded-deterministic evidence messages are
selected, then one reasoning-off LLM call writes a question whose gold
answer appears verbatim in the trace (grounding is validated locally;
ungrounded pairs are retried once, then dropped).

Output: data/swe_trace_qa.json — {sample_id: [{question, answer}]}
Loaded by run_eval --trace-qa. Regeneration reuses the cache unless
--force.
"""

import hashlib
import json
import sys
from pathlib import Path

import httpx

from context_eval.env import load_env
from context_eval.provider import ChatMessage, OpenRouterClient

CACHE = Path("data/swe_trace_qa.json")
QUESTIONS_PER_SAMPLE = 4

_PROMPT = """You write eval questions about an agent's coding session.
Below is one message from the transcript. Write ONE question about the
specific information in it, answerable only from this session, with a
SHORT gold answer (a few words, quoted exactly from the message).
Do not ask about trivia like line numbers or exact whitespace; prefer
errors and their fixes, commands and outcomes, files and their roles,
ids/versions/values, decisions and reasons.
Reply as exactly two lines:
Q: <question>
A: <gold answer, copied verbatim from the message>

MESSAGE:
{msg}"""


def _interesting(msgs):
    """Rank messages by information density; drop pure scaffolding."""
    scored = []
    for i, m in enumerate(msgs):
        t = m.get("content") or ""
        if len(t) < 120 or len(t) > 6000:
            continue
        score = sum(
            kw in t.lower()
            for kw in ("error", "failed", "traceback", "test", "fix",
                       "passed", "assert", "def ", "import", "git")
        )
        role = m.get("role")
        if role == "assistant":
            score += 1
        scored.append((score, i, t))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [(i, t) for _, i, t in scored]


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--path", default="data/swe_smoke.jsonl")
    ap.add_argument("--force", action="store_true", help="regenerate cached samples")
    ap.add_argument("--out", type=Path, default=CACHE)
    args = ap.parse_args()
    load_env()

    cache: dict[str, list] = {}
    if args.out.exists() and not args.force:
        cache = json.loads(args.out.read_text())

    llm = OpenRouterClient(
        model="deepseek/deepseek-v4.1-flash",
        http_client=httpx.Client(timeout=120),
    )
    samples = [json.loads(line) for line in open(args.path)]
    trace_text = {s["sample_id"]: "\n".join(m.get("content") or "" for m in s["messages"])
                  for s in samples}

    for s in samples:
        sid = s["sample_id"]
        if sid in cache and not args.force:
            continue
        rng = int(hashlib.sha256(sid.encode()).hexdigest()[:8], 16)
        pool = _interesting(s["messages"])
        picked = pool[rng % len(pool):] + pool[: rng % len(pool)]  # seeded rotation
        pairs, seen = [], set()
        for _idx, text in picked:
            if len(pairs) >= QUESTIONS_PER_SAMPLE:
                break
            try:
                c = llm.complete(
                    [ChatMessage(role="user", content=_PROMPT.format(msg=text[:4000]))],
                    max_tokens=512, reasoning={"enabled": False}, require_content=True,
                )
            except Exception as exc:  # provider exhausted: keep what we have
                print(f"{sid}: {exc}", file=sys.stderr)
                break
            lines = [ln.strip() for ln in c.text.strip().splitlines() if ln.strip()]
            if len(lines) < 2 or not lines[0].startswith("Q:") or not lines[-1].startswith("A:"):
                continue
            q = lines[0][2:].strip()
            a = lines[-1][2:].strip().strip('"')
            # grounding: gold must be copied from the trace, and answers
            # must not repeat across questions
            if a and a.lower() in trace_text[sid].lower() and a.lower() not in seen:
                pairs.append({"question": q, "answer": a})
                seen.add(a.lower())
        cache[sid] = pairs
        print(f"{sid}: {len(pairs)}/{QUESTIONS_PER_SAMPLE} grounded pairs", file=sys.stderr)
        args.out.write_text(json.dumps(cache, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
