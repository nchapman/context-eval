"""Offline dial sweeps over recorded ledger5 collections (theory 5.2).

Reads recording sidecars, re-renders across the dial grid and a tooth
grid, computes block metrics + rung trajectories, and writes:
  <out>/<stem>.sweep.json   raw metric rows + rung curves
  <out>/report.md           pooled tables and headline findings

No LLM calls: every number derives from (recording, counter, dials).
Usage:
  uv run python scripts/dial_sweeps.py \
      --recordings 'results/collection/*.recording.json' --out results/sweeps
"""

from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path

from context_eval.strategies.sawtooth import Ledger5Dials
from context_eval.sweeps import (
    dial_grid,
    rung_curve,
    rung_rows,
    sweep_recording,
)
from context_eval.tokens import default_counter


def aggregate_sweep(rows: list) -> dict:
    """Per-arm means over the tooth grid (pooled across recordings)."""
    by_arm: dict[str, list] = defaultdict(list)
    for r in rows:
        by_arm[r["arm"]].append(r)
    out = {}
    for arm, rs in by_arm.items():
        n = len(rs)
        out[arm] = {
            "renders": n,
            "mean_completeness": sum(r["completeness"] for r in rs) / n,
            "mean_detail_tokens": sum(r["detail_tokens"] for r in rs) / n,
            "mean_index_tokens": sum(r["index_tokens"] for r in rs) / n,
            "mean_sealed_ws": sum(r["n_sealed_ws"] for r in rs) / n,
            "mean_kept_unsealed": sum(r["mean_kept_unsealed"] for r in rs) / n,
            "mean_frac_faded": sum(r["frac_faded"] for r in rs) / n,
            "min_completeness": min(r["completeness"] for r in rs),
        }
    return out


def fmt_table(headers: list, rows: list) -> str:
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        out.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recordings", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tooth-step", type=int, default=10)
    args = ap.parse_args()

    counter = default_counter()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(glob.glob(args.recordings))
    if not files:
        raise SystemExit(f"no recordings match {args.recordings!r} -- nothing to sweep")

    pooled_rows: list = []
    pooled_rungs: list = []
    stems = []
    for f in files:
        rec = json.loads(Path(f).read_text())
        stem = Path(f).name.split(".")[0]
        stems.append(stem)

        # 1) dial sweep rows for this recording
        rows = sweep_recording(rec, counter, tooth_step=args.tooth_step)
        # 2) rung trajectories at the recorded dials, every recorded tooth
        for t in sorted({r["render_tooth"] for r in rec["renders"]}):
            rr, _ = rung_rows(rec, counter, tooth=t)
            for r in rr:
                r["run"] = stem
            pooled_rungs.extend(rr)
        json.dump(
            {"rows": rows, "rung_curve": rung_curve(
                [r for r in pooled_rungs if r["run"] == stem])},
            (out_dir / f"{stem}.sweep.json").open("w"), indent=1,
        )
        pooled_rows.extend({**r, "run": stem} for r in rows)
        print(f"[sweep] {stem}: {len(rows)} render rows")

    # -- pooled report ------------------------------------------------------
    agg = aggregate_sweep(pooled_rows)
    if "baseline" not in agg:
        raise SystemExit("no sweep rows produced (recordings have no renders?)")
    first = json.loads(Path(files[0]).read_text())
    order = [a for a, _ in dial_grid(Ledger5Dials(**first["dials"]))]
    arms = [a for a in order if a in agg]
    base = agg["baseline"]

    lines = [
        "# Dial sweeps -- ledger5 collection",
        "",
        f"Recordings: {', '.join(stems)} | counter: deepseek tokenizer | "
        f"tooth grid: every {args.tooth_step}",
        "",
        "## Dial sensitivity (pooled means over the tooth grid)",
        "",
        fmt_table(
            ["arm", "compl", "d compl", "detail tok", "d detail", "sealed ws",
             "kept", "d kept", "faded", "min compl"],
            [[a,
              f"{v['mean_completeness']:.3f}",
              f"{v['mean_completeness'] - base['mean_completeness']:+.3f}",
              f"{v['mean_detail_tokens']:.0f}",
              f"{v['mean_detail_tokens'] - base['mean_detail_tokens']:+.0f}",
              f"{v['mean_sealed_ws']:.1f}",
              f"{v['mean_kept_unsealed']:.2f}",
              f"{v['mean_kept_unsealed'] - base['mean_kept_unsealed']:+.2f}",
              f"{v['mean_frac_faded']:.2f}",
              f"{v['min_completeness']:.3f}"]
             for a, v in ((a, agg[a]) for a in arms)],
        ),
        "",
        "## Rung trajectories (recorded dials, all teeth, pooled)",
        "",
        "Mean kept-detail lines and sealed fraction by item class and age",
        "since last use. Theory: rung decays with age; decay is slower for",
        "state (d=0.20) than decision (0.35) than note (0.50).",
        "",
        fmt_table(
            ["class/age", "n", "mean kept", "sealed frac", "mean act"],
            [[k, v["n"], f"{v['mean_kept']:.2f}", f"{v['sealed_frac']:.2f}",
              f"{v['mean_act']:.2f}"]
             for k, v in rung_curve(pooled_rungs).items()],
        ),
        "",
        "## Notes",
        "",
        "- completeness = fraction of unsealed rung topics visible in the",
        "  block (the reservation guarantee; 1.0 unless triage force-sealed).",
        "- search=off / any-hit arms rebuild the use stream from the",
        "  recorded search log; collection-time dials are excluded (they",
        "  raise, by design).",
    ]
    (out_dir / "report.md").write_text("\n".join(lines) + "\n")
    json.dump({"sweep": agg, "rung_curve": rung_curve(pooled_rungs)},
              (out_dir / "pooled.json").open("w"), indent=1)
    print(f"[sweep] wrote {out_dir}/report.md")


if __name__ == "__main__":
    main()
