"""Needle-injection eval layer.

Novel facts are injected into a replayed trace at controlled depths, then
queried after replay — directly measuring what a context strategy retained
or can recover. Fully deterministic (seeded): no LLM involved in injection,
ground truth is exact-match. Contamination-immune (fresh random values).

SWE traces: facts ride as audit/log lines appended to tool observations.
LoCoMo: facts ride as new speaker turns inserted after a chosen turn.
"""

from __future__ import annotations

import random
import string
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from context_eval.datasets.base import ReplayMessage, ReplaySample
from context_eval.types import Role

Rng = random.Random


@dataclass(frozen=True)
class Question:
    """One eval question with deterministic ground truth."""

    kind: str  # "needle" | "locomo" | "trace_qa"
    question: str
    answer: str
    category: str = ""
    meta: Mapping[str, Any] = field(default_factory=dict)


def _hex(rng: Rng, n: int) -> str:
    return "".join(rng.choice("0123456789abcdef") for _ in range(n))


def _digits(rng: Rng, n: int) -> str:
    return "".join(rng.choice(string.digits) for _ in range(n))


def _word(rng: Rng) -> str:
    words = (
        "amber basalt cobalt dune ember fable granite harbor iris juniper "
        "kernel lumen mantle nectar opal prism quiver ridge slate tundra umber vortex willow zephyr"
    ).split()
    return rng.choice(words)


def _upper(rng: Rng, n: int) -> str:
    return "".join(rng.choice(string.ascii_uppercase) for _ in range(n))


@dataclass(frozen=True)
class _FactSpec:
    # Log-line container for tool observations (SWE).
    container: str
    # Spoken sentence for conversational turns (LoCoMo).
    sentence: str
    question: str
    value_gen: Callable[[Rng], str]


FACTS: tuple[_FactSpec, ...] = (
    _FactSpec(
        container="NOTE (workspace audit): the canary deployment token is {v}.",
        sentence="oh, heads up — the canary deployment token for our staging push is {v}",
        question="What is the canary deployment token?",
        value_gen=lambda r: f"{_hex(r, 4)}-{_upper(r, 2)}{_digits(r, 2)}Q",
    ),
    _FactSpec(
        container="rollback tag recorded by CI: {v}",
        sentence="we tagged the rollback point as {v}, in case tonight's deploy goes sideways",
        question="What is the rollback tag?",
        value_gen=lambda r: f"rb-{_word(r)}-{_digits(r, 3)}",
    ),
    _FactSpec(
        container="INFO (on-call handoff): the pager override code is {v}.",
        sentence="the on-call pager override code this week is {v}, don't lose it",
        question="What is the on-call pager override code?",
        value_gen=lambda r: f"{_word(r).upper()}-{_digits(r, 4)}",
    ),
    _FactSpec(
        container="metrics export finished; dashboard passphrase: {v}",
        sentence="the metrics dashboard passphrase is {v} now, they rotated it",
        question="What is the metrics dashboard passphrase?",
        value_gen=lambda r: f"{_word(r)}{_digits(r, 2)}-{_word(r)}",
    ),
    _FactSpec(
        container="audit: ephemeral signing key id (last 8): {v}",
        sentence="the new signing key ends in {v}, the full id is in the vault",
        question="What are the last 8 characters of the signing key id?",
        value_gen=lambda r: _hex(r, 8),
    ),
    _FactSpec(
        container="scheduler (recurring): internal review host is {v}",
        sentence="the internal review board lives at {v} these days",
        question="What is the internal review host?",
        value_gen=lambda r: f"{_word(r)}-{_word(r)}-{_digits(r, 2)}.internal",
    ),
    _FactSpec(
        container="quota broker: per-minute request ceiling set to {v}",
        sentence="ops bumped our request ceiling to {v} per minute",
        question="What is the per-minute request ceiling?",
        value_gen=lambda r: f"{_word(r)}-{r.randint(100, 999)}",
    ),
    _FactSpec(
        container="release train manifest: candidate suffix for tomorrow is {v}",
        sentence="tomorrow's release candidate suffix is {v}, if the tests stay green",
        question="What is the release candidate suffix?",
        value_gen=lambda r: f"rc.{r.randint(1, 9)}-{_word(r)}",
    ),
)


_TOPIC_SPECS: tuple[tuple[str, str], ...] = (
    # (distinctive token, conversational mention spoken by the user)
    ("amphora", "Let's call the retry-queue rework 'amphora' — still open, we haven't started it."),
    ("FIRE-1147", "I filed the flaky-test investigation as FIRE-1147; that one's resolved now."),
    ("kestrel", "The API migration is codename 'kestrel', paused until the security audit clears."),
    ("ORION-3", "ORION-3 is the new auth service, still mid-rollout."),
    ("tessellate", "The design-system refresh is nicknamed 'tessellate', about halfway done."),
    ("waypoint", "The CI dashboard project is 'waypoint' — that shipped last month."),
    ("halcyon", "The data-lake consolidation goes by 'halcyon', still just planning."),
    ("VELA-9", "Feature-flag cleanup is tracked as VELA-9, resolved."),
    ("bramble", "The parser rewrite we keep deferring is 'bramble', still open."),
    ("osprey", "The new CLI is 'osprey', currently in beta."),
    ("LATTICE-2", "Schema normalization is LATTICE-2, blocked on review."),
    ("corvid", "The admin-panel revamp is 'corvid', paused for now."),
    ("sundial", "The scheduler migration is 'sundial' — that one shipped."),
    ("THETA-7", "The on-call rotation redesign is THETA-7, still open."),
    ("palimpsest", "The docs-site redo is 'palimpsest', in progress."),
    ("grebe", "The profiler work is 'grebe', just started."),
    ("NOVA-11", "Billing reconciliation is NOVA-11, resolved last sprint."),
    ("inkwell", "The audit-log service is 'inkwell', deployed to prod."),
    ("curlew", "Search-relevance tuning goes by 'curlew', ongoing."),
    ("PERIDOT-4", "The email-digest redesign is PERIDOT-4, on hold."),
)

CONTINUITY_QUESTION = (
    "We've been working together for a while. In a short list, name the "
    "main projects, topics, or investigations we've discussed — one line "
    "each, with a word on status if you recall it. Only list things you're "
    "confident actually came up between us; if you can't recall the topic "
    "inventory, say so."
)


def inject_topics(
    sample: ReplaySample, count: int, rng: Rng
) -> tuple[ReplaySample, list[str], list[str]]:
    """Seed conversational topic markers; return (sample, seeded, distractors).

    Markers ride natural USER (fallback ASSISTANT) turns, never a message
    already needle-bearing. Distractors are pool tokens NOT seeded here —
    listing one is a false memory. The question that probes them (see
    CONTINUITY_QUESTION) contains no searchable terms: recovery is only
    possible if the resident view carries a map of the topics.
    """
    if count <= 0:
        return sample, [], []
    # Naming mentions stay in the FIRST HALF of the session: the currency
    # flip zone starts at 55%, and a status-bearing naming line landing
    # after its flip makes the flip's gold contradict the latest evidence.
    half = len(sample.messages) // 2
    eligible = [
        i
        for i, m in enumerate(sample.messages[:half])
        if m.role in (Role.USER, Role.ASSISTANT) and not m.meta.get("needle")
    ]
    if not eligible:  # tiny corpus: fall back to the full range
        eligible = [
            i
            for i, m in enumerate(sample.messages)
            if m.role in (Role.USER, Role.ASSISTANT) and not m.meta.get("needle")
        ]
    if not eligible:
        return sample, [], []
    order = list(range(len(_TOPIC_SPECS)))
    rng.shuffle(order)
    chosen = order[: min(count, len(_TOPIC_SPECS), len(eligible))]
    positions = _spread(len(chosen), eligible)
    messages = list(sample.messages)
    seeded: list[str] = []
    for token_idx, pos in zip(chosen, positions, strict=True):
        token, mention = _TOPIC_SPECS[token_idx]
        target = messages[pos]
        # End-of-turn append is natural chat phrasing here and safe: only
        # TOOL observations get previewed/cited (arc), so the needle-style
        # random-line placement rule does not apply to user/assistant turns.
        content = target.content + "\n" + mention
        messages[pos] = ReplayMessage(role=target.role, content=content, meta=dict(target.meta))
        seeded.append(token)
    distractors = [t for i, (t, _) in enumerate(_TOPIC_SPECS) if i not in chosen]
    rng.shuffle(distractors)
    return (
        ReplaySample(
            dataset=sample.dataset,
            sample_id=sample.sample_id,
            messages=tuple(messages),
            meta=sample.meta,
        ),
        seeded,
        distractors[:4],
    )


def _eligible_indices(sample: ReplaySample) -> list[int]:
    tool_idx = [i for i, m in enumerate(sample.messages) if m.role is Role.TOOL]
    if tool_idx:
        return tool_idx
    return [i for i, m in enumerate(sample.messages) if m.role is Role.USER]


def _spread(count: int, pool: list[int]) -> list[int]:
    """Deterministic even spread over the eligible pool."""
    return [pool[min(int((k + 0.5) * len(pool) / count), len(pool) - 1)] for k in range(count)]


def inject_needles(
    sample: ReplaySample, n: int = 5, seed: int = 0, position: str = "random"
) -> tuple[ReplaySample, list[Question]]:
    """position="mid" plants needles in the body of long tool
    observations, guaranteed outside any head/tail citation preview
    (ARC keeps ~100 head + ~60 tail chars) — recovery then requires the
    recall tool, not the view."""
    """Return a needle-injected copy of the sample plus its questions."""
    if n <= 0:
        return sample, []
    pool = _eligible_indices(sample)
    if position == "mid":
        pool = [
            i for i in pool
            if sample.messages[i].role is Role.TOOL
            and len(sample.messages[i].content) > 800
        ]
    if not pool:
        return sample, []

    count = min(n, len(pool), len(FACTS))
    positions = sorted(set(_spread(count, pool)))
    rng = random.Random(seed)

    messages = list(sample.messages)
    questions: list[Question] = []
    # Descending so insertions never shift later positions.
    for pos, fact_index in zip(
        reversed(positions), range(len(positions) - 1, -1, -1), strict=True
    ):
        fact = FACTS[fact_index % len(FACTS)]
        value = fact.value_gen(rng)
        target = messages[pos]

        if target.role is Role.TOOL:
            line = fact.container.format(v=value)
            lines = target.content.split("\n")
            if position == "mid":
                # insert at a line boundary whose char offset is inside
                # [250, len-250]: outside head:100/tail:60 previews with
                # margin, deterministic per seed
                offsets, acc = [], 0
                for ln in lines:
                    offsets.append(acc)
                    acc += len(ln) + 1
                lo, hi = 250, len(target.content) - 250
                eligible = [k for k, o in enumerate(offsets) if lo <= o <= hi]
                at = eligible[rng.randrange(len(eligible))] if eligible else len(lines) // 2
            else:
                # Insert at a seeded random line boundary, not always the end:
                # deterministic per seed, but the needle's position inside the
                # observation must not systematically land in any strategy's
                # preview window (e.g. ARC keeps head+tail of cited messages).
                at = rng.randrange(len(lines) + 1)
            content = "\n".join([*lines[:at], line, *lines[at:]])
            new_meta = dict(target.meta)
            new_meta["needle"] = True
            messages[pos] = ReplayMessage(role=target.role, content=content, meta=new_meta)
        else:
            speaker = target.meta.get("speaker") or target.content.split(":", 1)[0]
            content = f"{speaker}: {fact.sentence.format(v=value)}."
            messages.insert(
                pos + 1, ReplayMessage(role=Role.USER, content=content, meta={"needle": True})
            )

        questions.append(
            Question(
                kind="needle",
                question=fact.question,
                answer=value,
                category=f"needle@{pos / len(messages):.0%}",
                meta={"depth": pos / len(messages), "fact": fact.question},
            )
        )

    return (
        ReplaySample(
            dataset=sample.dataset,
            sample_id=sample.sample_id,
            messages=tuple(messages),
            meta=sample.meta,
        ),
        list(reversed(questions)),
    )
