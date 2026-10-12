"""Offline probe: does the sealed tier's FORMAT gate uncued enumeration?

The validation run sealed all 10 seeded topics deep (naming now lands in
the first half) and rendered them as one inline `sealed: name (status),
...` line; the model's continuity answer enumerated the rich active
sections and scored 0.0 -- while currency (cued) scored 10/10 from the
same line. This probe re-asks continuity against render variants of the
SAME final block, via the driver's own _ask machinery (search tools
live, full archive):

  ctrl      block as recorded (reproduce the 0.0)
  sect      sealed line -> its own bulleted section at the same position
  sect_top  same section, placed directly under the block header

Usage:
  uv run python scripts/seal_render_probe.py --rec results/validation/...
"""

from __future__ import annotations

import argparse
import json
import re

from context_eval.datasets.base import read_jsonl
from context_eval.driver import SYSTEM_PROMPT, _ask
from context_eval.env import load_env
from context_eval.judge import score_question
from context_eval.provider import OpenRouterClient
from context_eval.questions import Question
from context_eval.store import TranscriptStore
from context_eval.strategies.base import Engine, PromptMessage
from context_eval.strategies.sawtooth import SawtoothLedger5
from context_eval.tokens import default_counter
from context_eval.types import Role

SEALED_RE = re.compile(r"sealed: (.+)")


def sealed_section(block: str) -> str:
    """Rewrite the inline sealed line into a bulleted section."""
    matches = list(SEALED_RE.finditer(block))
    if not matches:
        raise SystemExit("no sealed line found in block")
    m = matches[-1]  # last sealed line wins
    items = [it.strip() for it in m.group(1).split(",") if it.strip()]
    body = "\n".join(f"- {it}" for it in items)
    sect = (
        "## Sealed workstreams — older or closed-out topics from this session\n"
        f"{body}"
    )
    return block[: m.start()] + sect + block[m.end() :]


class Stub:
    """Strategy stub: fixed block, live sawtooth search tools."""

    name = "stub"

    def __init__(self, block: str, strat: SawtoothLedger5) -> None:
        self.block = block
        self.strat = strat

    def assemble(self, engine) -> list:
        return [PromptMessage(role=Role.USER, content=self.block)]

    def tools(self, engine):
        return self.strat.tools(engine)

    def stats(self):
        return type("S", (), {})()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rec", default=(
        "results/validation/mega-a-s1.sawtooth-ledger5."
        "64000.mega-a.recording.json"))
    ap.add_argument("--path", default="data/pi_mega_a.jsonl")
    args = ap.parse_args()
    load_env()

    rec = json.loads(open(args.rec).read())
    block = rec["renders"][-1]["block"]

    # engine + index over the injected corpus (naming-first-half, seed 1)
    from context_eval.run_eval import build_questions

    sample = read_jsonl(args.path)[0]
    injected, qs = build_questions(
        sample, needles=4, seed=1, max_q=8, skip_locomo=True,
        topics=10, probes="currency,next_step",
    )
    store = TranscriptStore()
    for m in injected.messages:
        store.append(m.role, m.content, m.meta)
    strat = SawtoothLedger5()
    from context_eval.budget import Budget

    engine = Engine(store=store, counter=default_counter(),
                    budget=Budget(64000), llm=None)

    cont = next(q for q in qs if q.kind == "continuity")
    q = Question(kind="continuity", question=cont.question, answer="",
                 category="continuity", meta=cont.meta)

    variants = {
        "ctrl": block,
        "sect": sealed_section(block),
        "sect_top": _top(sealed_section(block)),
        "wording_ctrl": block,
        "wording_sect": sealed_section(block),
    }
    strat._index.index(store.messages())  # live archive behind history_search
    llm = OpenRouterClient()
    for name, blk in variants.items():
        stub = Stub(blk, strat)
        rec2 = _ask(strategy=stub, engine=engine, question=q, llm=llm,
                    max_answer_tokens=512, temperature=0.0,
                    system_prompt=SYSTEM_PROMPT)
        sc = score_question(q, rec2.prediction)
        named = [t for t in cont.meta["terms"] if t.lower() in rec2.prediction.lower()]
        print(f"[{name:8s}] score={sc.value:.2f} named={len(named)}/10 {named}")
        print("  pred:", rec2.prediction[:280].replace(chr(10), " / "))


def _top(block_with_sect: str) -> str:
    """Move the sealed section to just under the block header."""
    m = re.search(r"## Sealed workstreams.*?(?=\n#|\Z)", block_with_sect, re.S)
    if m is None:
        raise SystemExit("sealed section not found")
    sect = m.group(0)
    rest = block_with_sect[: m.start()] + block_with_sect[m.end():]
    lines = rest.split("\n")
    # insert after the header paragraph (first blank line)
    for i, ln in enumerate(lines[1:], 1):
        if not ln.strip():
            return "\n".join(lines[: i + 1]) + "\n" + sect + "\n" + "\n".join(lines[i + 1 :])
    return rest


if __name__ == "__main__":
    main()
