"""Offline dial-sweep harness (theory 5.2): metrics over re-rendered
recordings. Pure analysis -- no LLM, no strategy state. Every number
here is derivable from (recording, counter, dials) alone.

Three measurement families:
- block metrics (parse_block + render_metrics): allocation, completeness
- rung trajectories (rung_rows): the identification signal -- rung vs
  teeth-since-last-use, per item class
- dial sweeps (dial_grid + sweep_recording): re-render under varied
  dials at a tooth grid, deltas vs baseline
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .strategies.sawtooth import (
    Ledger5Dials,
    _l5_d_for,
    _Ledger5Line,
    replay_use_events,
    rerender_from_recording,
)

_WS_RE = re.compile(r"^## (?P<ws>.+?) — (?P<status>.+)$")
_TOPIC_RE = re.compile(r"^(?P<topic>.+?): (?P<status>.+)$")
# zero-detail topics under pin_state_status=False render name-only with
# the faded marker (no ': status') -- they are visible topics too
_FADED_RE = re.compile(r'^(?P<topic>.+?) · faded — search ')


@dataclass
class BlockView:
    """A parsed resident block: sections, sealed line, header."""

    header_tokens: int
    sections: list  # [(ws, status, topic_names, index_tokens, detail_tokens)]
    sealed_names: list
    sealed_tokens: int
    total_tokens: int

    @property
    def index_tokens(self) -> int:
        """Everything that is not detail: header, section headers, topic
        lines, sealed line."""
        return self.header_tokens + sum(s[3] for s in self.sections) + self.sealed_tokens


def parse_block(block: str, counter) -> BlockView:
    """Split a rendered block into its structural pieces. Detail lines are
    '- ' bullets; topic lines are 'name: status'; the sealed line carries
    'name (status)' entries (or a '+N earlier' collapse)."""
    header, sections, sealed = 0, [], ""
    cur = None
    for line in block.splitlines():
        if line.startswith("[Session memory"):
            header = counter.count_text(line)
        elif m := _WS_RE.match(line):
            cur = {
                "ws": m["ws"], "status": m["status"],
                "topics": [], "index": counter.count_text(line), "detail": 0,
            }
            sections.append(cur)
        elif line.startswith("sealed:"):
            sealed = line
        elif line.startswith("- ") and cur is not None:
            cur["detail"] += counter.count_text(line) + 1  # render fact_cost
        elif cur is not None and (
            (m := _TOPIC_RE.match(line)) or (m := _FADED_RE.match(line))
        ):
            cur["topics"].append(m["topic"])
            cur["index"] += counter.count_text(line)
    sealed_names = []
    if sealed:
        sealed_names = [n.strip() for n in re.findall(r"([\w-]+) \(", sealed)]
    return BlockView(
        header_tokens=header,
        sections=[(s["ws"], s["status"], s["topics"], s["index"], s["detail"])
                  for s in sections],
        sealed_names=sealed_names,
        sealed_tokens=counter.count_text(sealed) if sealed else 0,
        total_tokens=counter.count_text(block),
    )


def render_metrics(block: str, rungs: list, counter) -> dict:
    """Per-render measurements. completeness is the reservation guarantee
    made operational: every UNSEALED rung topic must be visible in the
    block (as a topic line under an active section, or by name in a
    section header context)."""
    view = parse_block(block, counter)
    unsealed = {r["topic"] for r in rungs if not r["sealed"]}
    visible = {t for _, _, ts, _, _ in view.sections for t in ts}
    # a workstream section with a single same-named topic may render the
    # topic line identically; also count ws names whose section exists
    visible |= {ws for ws, _, _, _, _ in view.sections}
    missing = unsealed - visible
    kept = [r["kept"] for r in rungs if not r["sealed"]]
    detail = sum(d for *_, d in view.sections)
    return {
        "n_topics": len(rungs),
        "n_unsealed": len(unsealed),
        "n_active_ws": len(view.sections),
        "n_sealed_ws": len(view.sealed_names),
        "completeness": (len(unsealed & visible) / len(unsealed)) if unsealed else 1.0,
        "missing_unsealed": sorted(missing),
        "mean_kept_unsealed": sum(kept) / len(kept) if kept else 0.0,
        "frac_faded": sum(1 for k in kept if k == 0) / len(kept) if kept else 0.0,
        "detail_tokens": detail,
        "index_tokens": view.total_tokens - detail,
        "total_tokens": view.total_tokens,
    }


# -- rung trajectories -------------------------------------------------------


def _class_of(newest) -> str:
    """Class of a topic = class of its NEWEST line (seq order), matching
    the render's d-selection."""
    if newest is None:
        return "note"
    if newest.kind == "decision":
        return "decision"
    if newest.is_state_change:
        return "state"
    return "note"


def rung_rows(rec: dict, counter, dials: Ledger5Dials | None = None, tooth: int | None = None):
    """Re-render and return rung rows annotated with teeth-since-last-use
    and item class -- the raw material for rung-vs-age curves. Ages are
    computed from the SAME use stream the render consumed (snapshot-
    clipped), so the x-axis matches the emitted rung values."""
    block, rungs = rerender_from_recording(rec, counter, tooth=tooth, dials=dials)
    if not rungs:
        return [], block
    lines = [_Ledger5Line.from_record(d) for d in rec["lines"]]
    by_key: dict[str, list] = {}
    for ln in lines:
        by_key.setdefault(f"{ln.umbrella}/{ln.topic}".casefold(), []).append(ln)
    _, use_events = replay_use_events(rec, rungs[0]["tooth"], dials)
    last_use: dict[str, int] = {}
    for u in use_events:
        last_use[u["key"]] = max(last_use.get(u["key"], 0), u["tooth"])
    active_dials = dials or Ledger5Dials(**rec["dials"])
    for r in rungs:
        key = f"{r['ws']}/{r['topic']}".casefold()
        members = by_key.get(key) or []
        newest = max(members, key=lambda m: m.seq, default=None)
        # no recorded use: age from the newest fact's birth (the render's
        # ulist fallback sums fact teeth; the newest is the stand-in)
        lu = last_use.get(key) or (max(m.tooth for m in members) if members else 1)
        r["age"] = max(1, r["tooth"] - lu)
        r["item_class"] = _class_of(newest)
        r["d"] = _l5_d_for(newest.kind if newest else "note", active_dials)
    return rungs, block


def rung_curve(rungs: list) -> dict:
    """Aggregate rung rows into mean-kept / sealed-fraction per (class, age
    bucket). The theory's prediction: rung decays with age, slower for
    smaller d (state < decision < note)."""
    buckets: dict[tuple, list] = {}
    for r in rungs:
        age = r["age"]
        b = 1 if age <= 2 else 2 if age <= 5 else 3 if age <= 10 else 4 if age <= 20 \
            else 5 if age <= 40 else 6
        buckets.setdefault((r["item_class"], b), []).append(r)
    out = {}
    for (cls, b), rows in sorted(buckets.items()):
        out[f"{cls}/age{b}"] = {
            "n": len(rows),
            "mean_kept": sum(r["kept"] for r in rows) / len(rows),
            "sealed_frac": sum(1 for r in rows if r["sealed"]) / len(rows),
            "mean_act": sum(r["act"] for r in rows) / len(rows),
        }
    return out


# -- dial sweeps -------------------------------------------------------------


def dial_grid(recorded: Ledger5Dials) -> list:
    """The sweep arms: baseline plus one arm per dial value. Names carry
    the dial and value for the report."""
    from dataclasses import replace

    out = [("baseline", recorded)]
    for v in (0.30, 0.70):
        out.append((f"d_note={v}", replace(recorded, d_note=v)))
        out.append((f"d_state={v}", replace(recorded, d_state=v)))
    for v in (15, 45, 60):
        out.append((f"seal_after_teeth={v}", replace(recorded, seal_after_teeth=v)))
    for v in (8, 25):
        out.append((f"stale_status_teeth={v}", replace(recorded, stale_status_teeth=v)))
    for v in (2048, 3072):
        out.append((f"resident_budget={v}", replace(recorded, resident_budget=v)))
    for v in (0, 16):
        out.append((f"recency_floor_full={v}", replace(recorded, recency_floor_full=v)))
    out.append(("pin_state_status=off", replace(recorded, pin_state_status=False)))
    out.append(("pin_next_step=off", replace(recorded, pin_next_step=False)))
    out.append(("composition=mass", replace(recorded, composition_detail="mass")))
    out.append(("activation_form=exp", replace(recorded, activation_form="exp")))
    for arm in ("off", "any-hit"):
        out.append((f"search={arm}", replace(recorded, search_reactivates=arm)))
    return out


def sweep_recording(rec: dict, counter, grid: list | None = None,
                    tooth_step: int = 10) -> list:
    """Re-render the recording at a tooth grid under each dial arm and
    measure every render. Returns metric rows keyed (arm, tooth)."""
    recorded = Ledger5Dials(**rec["dials"])
    grid = grid or dial_grid(recorded)
    teeth = sorted({r["render_tooth"] for r in rec["renders"]})
    if not teeth:
        return []
    # the final tooth joins the grid but is never duplicated
    coarse = sorted({t for t in teeth if t % tooth_step == 0} | {teeth[-1]})
    rows = []
    for arm, dials in grid:
        for t in coarse:
            block, rungs = rerender_from_recording(rec, counter, tooth=t, dials=dials)
            if not rungs:
                continue
            m = render_metrics(block, rungs, counter)
            m.update(arm=arm, tooth=t)
            rows.append(m)
    return rows
