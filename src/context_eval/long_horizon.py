"""Long-horizon probes: status currency and next-step coherence.

Compaction benchmarks usually measure *presence* (was the fact kept?).
Long sessions fail differently: the map carries a stale average — a
resolved topic still listed open, a superseded plan resurrected, the
current next-step lost in a fold. These probes measure *currency*:

- currency: a topic thread is seeded `open` at depth d, flipped to
  `resolved` at a later depth; the uncued question asks its status now.
  Gold = the final status; the stale status in the answer is a
  confabulation signal scored separately.
- next_step: a forward-looking intent ("Next: X") is seeded before the
  compaction boundary; the uncued question asks what was about to
  happen. Tests plan continuity, not fact retention.

Deterministic and seeded like the needle layer; injected into tool
observations (SWE-style) or as speaker turns (LoCoMo-style), matching
the corpus.
"""

from __future__ import annotations

import random
import re

from context_eval.datasets.base import ReplayMessage, ReplaySample
from context_eval.questions import Question
from context_eval.types import Role

CURRENCY_QUESTION = (
    "Earlier in this session some topics were opened and some were later "
    "resolved or paused. For each topic below, what is its status right "
    "now — open, resolved, or paused? Answer one line per topic: "
    "`topic: status`. Include every topic listed."
)
NEXT_STEP_QUESTION = (
    "At one point this session contains an explicit handoff note — a line "
    "beginning `[handoff] Next:`. What immediate next action did it state, "
    "and on which codename? Answer in one sentence."
)

_STATUSES = ("open", "resolved", "paused")


def inject_currency(
    sample: ReplaySample, topics: list[str], seed: int = 0
) -> tuple[ReplaySample, list[Question]]:
    """Seed each topic `open` early and flip it later; ask status now.

    Flips land in the second half of the session (deep enough that any
    budget-bounded strategy must have compacted the opening). One
    question covers all topics; per-topic golds ride in meta.
    """
    if not topics:
        return sample, []
    rng = random.Random(seed ^ 0xC0FFEE)
    messages = list(sample.messages)
    n = len(messages)
    tool_idx = [i for i, m in enumerate(messages) if m.role is Role.TOOL]
    pool = tool_idx or list(range(n))
    # opening: first quarter of the pool; flip: 55-75% depth
    opens = pool[: max(1, len(pool) // 4)]
    flip_zone = pool[int(len(pool) * 0.55) : int(len(pool) * 0.75)] or pool
    gold: dict[str, str] = {}
    lines_by_pos: dict[int, list[str]] = {}

    for t in topics:
        open_at = opens[rng.randrange(len(opens))]
        flip_at = flip_zone[rng.randrange(len(flip_zone))]
        if flip_at <= open_at:
            flip_at = pool[min(len(pool) - 1, open_at + max(1, len(pool) // 2))]
        final = rng.choice(("resolved", "paused"))
        gold[t] = final
        lines_by_pos.setdefault(open_at, []).append(
            f"[status] {t}: open — investigation started"
        )
        lines_by_pos.setdefault(flip_at, []).append(
            f"[status] {t}: {final} — closed out, superseded or finished"
        )

    for pos in sorted(lines_by_pos, reverse=True):
        target = messages[pos]
        block = "\n".join(lines_by_pos[pos])
        if target.role is Role.TOOL:
            messages[pos] = ReplayMessage(
                role=target.role,
                content=target.content + "\n" + block,
                meta=dict(target.meta),
            )
        else:
            messages.insert(
                pos + 1,
                ReplayMessage(role=Role.USER, content=block, meta={"probe": True}),
            )

    question = Question(
        kind="currency",
        question=CURRENCY_QUESTION + "\n" + "\n".join(f"- {t}" for t in topics),
        answer=" | ".join(f"{t}:{g}" for t, g in gold.items()),
        category="currency",
        meta={"statuses": gold, "topics": list(topics)},
    )
    return (
        ReplaySample(
            dataset=sample.dataset,
            sample_id=sample.sample_id,
            messages=messages,
            meta=dict(sample.meta),
        ),
        [question],
    )


def score_currency(question: Question, prediction: str) -> tuple[float, float]:
    """Returns (accuracy, staleness): per-topic fraction correct, and
    fraction answered with the *stale* (pre-flip) status."""
    low = prediction.lower()
    hits, stale = 0, 0
    topics = question.meta.get("topics", [])
    for t in topics:
        # last mention of the topic wins (statuses update in the answer)
        mentions = [m.start() for m in re.finditer(re.escape(t.lower()), low)]
        if not mentions:
            continue
        tail = low[mentions[-1] : mentions[-1] + len(t) + 40]
        final = question.meta["statuses"][t]
        if final in tail:
            hits += 1
        elif "open" in tail:
            stale += 1
    n = max(1, len(topics))
    return hits / n, stale / n


def inject_next_step(
    sample: ReplaySample, seed: int = 0
) -> tuple[ReplaySample, Question]:
    """Seed a forward-looking intent mid-session; ask what was next."""
    from context_eval.questions import _upper, _word  # shared seeded vocab

    rng = random.Random(seed ^ 0x5EED_2)
    codename = f"{_word(rng)}-{_upper(rng, 3)}"
    action = rng.choice(
        (
            "migrate the config loader",
            "write the regression test",
            "rerun the benchmark suite",
            "refactor the retry ladder",
            "ship the patched release",
        )
    )
    line = f"[handoff] Next: {action} on {codename} — picked up from here."
    messages = list(sample.messages)
    pool = [i for i, m in enumerate(messages) if m.role is Role.TOOL] or list(
        range(len(messages))
    )
    at = pool[int(len(pool) * 0.6)]
    target = messages[at]
    if target.role is Role.TOOL:
        messages[at] = ReplayMessage(
            role=target.role,
            content=target.content + "\n" + line,
            meta=dict(target.meta),
        )
    else:
        messages.insert(
            at + 1, ReplayMessage(role=Role.USER, content=line, meta={"probe": True})
        )
    question = Question(
        kind="next_step",
        question=NEXT_STEP_QUESTION,
        answer=action,
        category="next_step",
        meta={"codename": codename, "action": action},
    )
    return (
        ReplaySample(
            dataset=sample.dataset,
            sample_id=sample.sample_id,
            messages=messages,
            meta=dict(sample.meta),
        ),
        question,
    )
