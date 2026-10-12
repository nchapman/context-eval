"""Focused iteration on extraction recall (the note-taking step).

Replays ONLY the extraction calls for recorded chunks that contain seeded
topic mentions (naming lines, [status] opens/flips), plus a control
sample -- no full runs. Known-topics context comes from the RECORDING
(lines born before the chunk's tooth), so every variant faces exactly
the live context the recorded extraction faced. A variant is only
trusted if the baseline reproduces the known live failures.

Cost: ~60 provider calls per (run, variant) -- cents.

Note: v0 is the prompt AS SHIPPED in sawtooth.py (the v1-codename clause
was promoted after bench validation, so v0 == v1-codename for fresh
checkouts; the archived results/ files predate the promotion).

Usage:
  uv run python scripts/extraction_bench.py --run mega-a-s1 --variant v0
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from pathlib import Path

from context_eval.datasets.base import read_jsonl
from context_eval.env import load_env
from context_eval.provider import OpenRouterClient
from context_eval.run_eval import build_questions
from context_eval.strategies.lcm import summary_call_kwargs
from context_eval.strategies.sawtooth import (
    _EXTRACT_PROMPT_V5,
    _bullet_or_row,
    _Ledger5Line,
)

RESULTS = Path("results/bench")


# -- prompt variants ----------------------------------------------------------

def variant_prompt(name: str) -> str:
    base = _EXTRACT_PROMPT_V5
    if name == "v0":
        return base
    codename = (
        "- EXCEPTION, never skipped and exempt from the budget: a one-line "
        "topic/status mention (any speaker, any surface -- chat, tool "
        "output, even a trailing '[status] x: paused' annotation) is ALWAYS "
        "its own line under a workstream named for that codename -- quoted "
        "nicknames ('corvid') and tracker ids (NOVA-11, FIRE-1147) alike. "
        "It is NEVER a note under another workstream, even when the rest of "
        "the segment is about something else. Several distinct codenames in "
        "one segment: one line each."
    )
    flip = (
        "- Status updates to a KNOWN topic are never skipped or absorbed: "
        "if the segment states a new status for a topic already in KNOWN "
        "TOPICS (shipped, paused, closed out, superseded -- even one line "
        "at the end of a long tool output), emit that topic's state-change "
        "line with the new status."
    )
    if name == "v1-codename":
        return _swap_exception(base, codename)
    if name == "v2-flip":
        return base.replace("- A topic reappearing", flip + "\n- A topic reappearing")
    if name == "v3-both":
        return _swap_exception(base, codename).replace(
            "- A topic reappearing", flip + "\n- A topic reappearing"
        )
    if name == "v4-secondlook":
        return _swap_exception(base, codename)
    raise ValueError(f"unknown variant {name!r}")


_SECOND_LOOK_PROMPT = """You maintain the ledger of an agent session. The messages below were
SKIPPED by the first extraction pass: no ledger line cites them. They are
short, event-shaped items (status pings, annotations, CI notes, one-line
mentions) that are easy to lose next to dense work output.
Give them a focused second look: extract ONLY observations that change the
ledger — a status/state change for a known topic, a new topic or codename
mention, a commitment or event future work depends on. Use exactly the
same line forms:
- WORKSTREAM/topic | note | ... [§<ids>]
- WORKSTREAM/topic | decision | ... [§<ids>]
- WORKSTREAM/topic | <status> | ... [§<ids>]
If nothing here merits a line, output nothing.
{known_topics}Skipped messages:
{segment}"""


def _swap_exception(base: str, new: str) -> str:
    old = [ln for ln in base.splitlines() if ln.startswith("- EXCEPTION")][0]
    return base.replace(old, new)


# -- bench --------------------------------------------------------------------


def rebuild_run(stem: str):
    """(recording, corpus, seed, messages-by-id, seeded-topics, gold)."""
    corpus, seed = stem.split("-")[1], int(stem.split("-s")[1])
    rec = json.loads(
        Path(f"results/collection/{stem}.sawtooth-ledger5.64000."
             f"mega-{corpus}.recording.json").read_text()
    )
    samples = read_jsonl(f"data/pi_mega_{corpus}.jsonl")
    injected, questions = build_questions(
        samples[0], needles=4, seed=seed, max_q=8, skip_locomo=True,
        topics=10, probes="currency,next_step",
    )
    cur = next(q for q in questions if q.kind == "currency")
    gold = dict(p.split(":") for p in cur.answer.split(" | "))
    by_id = {f"m{i + 1:06d}": m for i, m in enumerate(injected.messages)}
    return rec, corpus, seed, by_id, gold


def known_topics_from_lines(lines: list, upto_tooth: int) -> str:
    """Replicates _known_topics_v5 over recorded lines born before the
    chunk's tooth (frozen live context)."""
    born = [ln for ln in lines if ln["tooth"] < upto_tooth]
    if not born:
        return ""
    last, disp, ws_last, ws_disp = {}, {}, {}, {}
    for b in born:
        tkey = f"{b['ws']}/{b['topic']}".casefold()
        last[tkey] = max(last.get(tkey, 0), b["seq"])
        disp.setdefault(tkey, f"{b['ws']}/{b['topic']}")
        wkey = b["ws"].casefold()
        ws_last[wkey] = max(ws_last.get(wkey, 0), b["seq"])
        ws_disp.setdefault(wkey, b["ws"])
    ws = ", ".join(ws_disp[k] for k in sorted(ws_last, key=lambda k: -ws_last[k])[:30])
    topics = ", ".join(disp[k] for k in sorted(last, key=lambda k: -last[k])[:100])
    return f"WORKSTREAMS: {ws}\nKNOWN TOPICS (workstream/topic): {topics}\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="recording stem, e.g. mega-a-s1")
    ap.add_argument("--variant", default="v0")
    ap.add_argument("--controls", type=int, default=25)
    ap.add_argument("--sleep", type=float, default=0.25)
    args = ap.parse_args()

    load_env()
    rec, corpus, seed, by_id, gold = rebuild_run(args.run)
    prompt_tpl = variant_prompt(args.variant)
    topics = list(gold)

    # target chunks: any message containing a seeded token
    def msg_has_seed(m) -> str | None:
        for t in topics:
            if re.search(rf"\b{re.escape(t)}\b", m.content, re.IGNORECASE):
                return t
        return None

    chunks = sorted(rec["chunks"], key=lambda c: c["tooth"])
    targets, controls_pool = [], []
    for c in chunks:
        seeds = {t for mid in c["ids"]
                 if (m := by_id.get(mid)) and (t := msg_has_seed(m))}
        (targets if seeds else controls_pool).append((c, seeds))
    rng = random.Random(7)
    controls = rng.sample(controls_pool, min(args.controls, len(controls_pool)))

    llm = OpenRouterClient()
    out = {"run": args.run, "variant": args.variant, "chunks": [], "topics": {}}
    lines_by_topic: dict[str, list] = {t: [] for t in topics}
    flipped: dict[str, list] = {t: [] for t in topics}

    todo = [(c, s, "target") for c, s in targets] + [
        (c, set(), "control") for c, _ in controls
    ]
    todo.sort(key=lambda x: x[0]["tooth"])
    for c, seeds, kind in todo:
        text = "".join(
            f"§{mid} [{by_id[mid].role.value}]\n{by_id[mid].content}\n\n"
            for mid in sorted(c["ids"]) if mid in by_id
        )
        known = known_topics_from_lines(rec["lines"], c["tooth"])
        prompt = prompt_tpl.format(segment=text[:64_000], known_topics=known)
        completion = llm.complete(
            [{"role": "user", "content": prompt}], **summary_call_kwargs()
        )
        # fp8 routing intermittently returns empty 200s under rapid cadence
        # (live never sees this: empty_summaries=0 across 2000+ extractions);
        # one retry keeps the bench measuring extraction, not routing noise
        if not (completion.text or "").strip():
            time.sleep(max(args.sleep, 1.0))
            completion = llm.complete(
                [{"role": "user", "content": prompt}], **summary_call_kwargs()
            )
        got = []

        def _parse(text: str) -> list[dict]:
            out = []
            for line in (text or "").splitlines():
                row = _bullet_or_row(line)
                if row is None:
                    continue
                ln = _Ledger5Line(0, row)
                if ln.umbrella and ln.topic:
                    out.append({"ws": ln.umbrella, "topic": ln.topic,
                                "kind": ln.kind, "text": ln.text, "ids": ln.ids})
            return out

        got = _parse(completion.text)
        second = None
        if args.variant == "v4-secondlook":
            cited = {i for g in got for i in g["ids"]}
            mini = [mid for mid in sorted(c["ids"]) if mid in by_id
                    and by_id[mid].role.value == "tool"
                    and 20 <= len(by_id[mid].content) <= 1000
                    and mid not in cited]
            chars = sum(len(by_id[mid].content) for mid in mini)
            if chars >= 200:
                text2 = "".join(
                    f"§{mid} [{by_id[mid].role.value}]\n{by_id[mid].content}\n\n"
                    for mid in mini
                )[:32_000]
                p2 = _SECOND_LOOK_PROMPT.format(
                    segment=text2, known_topics=known
                )
                comp2 = llm.complete(
                    [{"role": "user", "content": p2}], **summary_call_kwargs()
                )
                if not (comp2.text or "").strip():
                    time.sleep(max(args.sleep, 1.0))
                    comp2 = llm.complete(
                        [{"role": "user", "content": p2}], **summary_call_kwargs()
                    )
                second = _parse(comp2.text)
                got.extend(second)
                time.sleep(args.sleep)
        for t in topics:
            for g in got:
                if g["ws"].casefold() == t.casefold():
                    lines_by_topic[t].append(
                        {"chunk": c["tooth"], "kind": g["kind"], "text": g["text"]}
                    )
                    if g["kind"] in ("open", "resolved", "paused", "in progress"):
                        flipped[t].append(g["kind"])
                elif re.search(rf"\b{re.escape(t)}\b", g["text"], re.IGNORECASE):
                    lines_by_topic[t].append(
                        {"chunk": c["tooth"], "kind": g["kind"],
                         "text": g["text"], "mis_scoped": g["ws"]}
                    )
        out["chunks"].append({
            "tooth": c["tooth"], "kind": kind, "seeds": sorted(seeds),
            "n_lines": len(got),
            "n_second": len(second) if second is not None else None,
            "lines": got if kind == "target" else len(got),
        })
        time.sleep(args.sleep)

    # score
    for t in topics:
        own = [x for x in lines_by_topic[t] if "mis_scoped" not in x]
        mis = [x for x in lines_by_topic[t] if "mis_scoped" in x]
        states = [x["kind"] for x in own
                  if x["kind"] in ("open", "resolved", "paused", "in progress")]
        final = states[-1] if states else None
        out["topics"][t] = {
            "gold": gold[t],
            "own_ws": bool(own),
            "final_status": final,
            "correct": bool(own) and final == gold[t],
            "mis_scoped": [x["mis_scoped"] for x in mis][:2],
        }
    ok = sum(1 for v in out["topics"].values() if v["correct"])
    ctl = [c["n_lines"] for c in out["chunks"] if c["kind"] == "control"]
    out["summary"] = {
        "topics_correct": f"{ok}/{len(topics)}",
        "mean_control_lines": (sum(ctl) / len(ctl)) if ctl else 0,
        "target_chunks": len(targets),
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / f"{args.run}.{args.variant}.json"
    path.write_text(json.dumps(out, indent=1))
    print(f"[bench] {args.run} {args.variant}: correct={ok}/{len(topics)} "
          f"control-lines/chunk={out['summary']['mean_control_lines']:.1f} "
          f"-> {path}")


if __name__ == "__main__":
    main()
