"""Flat compaction v2: the summary as a cue-card index.

Baseline flat-compact writes an answer-shaped digest (EVENTS/FACTS/USER/
STATE prose). At long horizons the digest either fossilizes or swells,
and the verbatim archive behind history_search is only as good as the
queries the agent thinks to issue. v2 changes the summary's JOB: not to
answer, but to make retrieval precise -- a topic index where each line
carries the exact entities, keywords, amounts, and ss-ids that (a)
resolve the common case directly and (b) generate a high-precision
history_search query when they don't. Human-memory framing: the summary
is recognition (``I know this topic exists and where''), search is
recall (verbatim recovery). Mechanics (tau trigger, kept tail, FTS
tools, two-step fold against the previous summary) are inherited from
baseline flat-compact -- ONLY the compaction artifact differs, so
benchmark deltas are attributable to the cue-card design alone.
"""

from __future__ import annotations

from context_eval.strategies.flat_compact import (
    FLAT_TAU,
    KEEP_TAIL_FRAC,
    FlatCompact,
)

_V2_INDEX_WORDS = 450

# The index line is the unit of memory: one topic, one line, keywords
# that match how questions get asked (entities + amounts + dates).
_V2_PROMPT = f"""You maintain a TOPIC INDEX for an agent with search access to the
verbatim session archive. The index is recognition memory: each line
must let the agent either answer directly or issue one precise
history_search query. It is NOT prose.

Build in two steps:
STEP 1 - from the NEW MESSAGES above only (not the prior index), list
every topic touched (a person, project, purchase, plan, event,
preference, or recurring activity), with its entities, specific
numbers/amounts/dates, and the [<id>] stamps of its messages.
STEP 2 - merge into the prior index above: one line per topic, newest
info first within the line. Topics never merge. When a value changes,
update it in place and keep one trailing "(was X)" only if the old
value was recent. Never drop a topic that was not superseded -- a
dropped topic is a memory the agent no longer knows it has.

Each topic line: `TOPIC (first date~last date) <id>,<id>: entity $N /
YYYY-MM-DD / keyword, keyword, keyword`
- keywords are SEARCH TERMS: nouns, names, amounts -- the words a
  question about this topic would contain; no verbs, no adjectives.
- the first 2-3 ids are the best entry points into the verbatim
  archive for this topic.

Sections, in order:
USER: identity card, 3 lines max, CURRENT values only.
TOPIC INDEX: one line per topic, most recently active first.
THREADS: open loops and next steps, one line each.

Keep the whole index under ~{_V2_INDEX_WORDS} words. Resolve relative
dates to absolute YYYY-MM-DD using the message stamps."""


class FlatCompactV2(FlatCompact):
    """Cue-card variant; inherits all mechanics, overrides the artifact."""

    name = "flat-compact-v2"

    def __init__(
        self,
        tau: float = FLAT_TAU,
        keep_tail_frac: float = KEEP_TAIL_FRAC,
    ) -> None:
        super().__init__(tau=tau, keep_tail_frac=keep_tail_frac)

    def _fold_instruction(self) -> str:
        return _V2_PROMPT
