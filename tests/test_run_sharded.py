"""run_sharded.sh regression tests.

The script burned a full live run once (edited mid-execution) and its
merge logic once re-merged stale shards silently (review finding). These
execute the real script with a stub `uv` so the launch/wait/clean/merge
path runs end-to-end without spend.
"""

import json
import os
import stat
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_sharded.sh"

FAKE_UV = """#!/usr/bin/env python3
import json
import sys
from pathlib import Path

args = sys.argv[1:]
out = Path(args[args.index("--out") + 1])
for i, a in enumerate(args):  # the script appends its own --out last
    if a == "--out":
        out = Path(args[i + 1])
out.parent.mkdir(parents=True, exist_ok=True)
shard = out.stem.rsplit(".", 1)[-1]
with out.open("w") as f:
    f.write(json.dumps({"shard": shard}) + "\\n")
"""


def _fake_uv_bin(tmp_path: Path) -> Path:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    uv = bin_dir / "uv"
    if not uv.exists():  # tests may override with their own stub
        uv.write_text(FAKE_UV)
        uv.chmod(uv.stat().st_mode | stat.S_IEXEC)
    return bin_dir


def _run(tmp_path: Path, *extra: str) -> subprocess.CompletedProcess:
    env = dict(os.environ, PATH=f"{_fake_uv_bin(tmp_path)}:{os.environ['PATH']}")
    return subprocess.run(
        ["bash", str(SCRIPT), "3", *extra],
        env=env, capture_output=True, text=True, timeout=60,
        cwd=tmp_path,
    )


def test_merges_all_shards_and_exits_zero(tmp_path: Path):
    out = tmp_path / "results" / "run.jsonl"
    r = _run(tmp_path, "--path", "x.jsonl", "--out", str(out))
    assert r.returncode == 0, r.stderr
    rows = [json.loads(ln) for ln in out.read_text().splitlines()]
    assert sorted(row["shard"] for row in rows) == ["0", "1", "2"]
    assert "merged 3 runs from 3 shards" in r.stdout


def test_stale_shard_outputs_are_not_remerged(tmp_path: Path):
    out = tmp_path / "results" / "run.jsonl"
    stale_dir = out.parent / "shards" / "run"
    stale_dir.mkdir(parents=True)
    (stale_dir / "run.7.jsonl").write_text('{"stale": true}\n')  # larger prior n
    r = _run(tmp_path, "--path", "x.jsonl", "--out", str(out))
    assert r.returncode == 0, r.stderr
    rows = [json.loads(ln) for ln in out.read_text().splitlines()]
    assert len(rows) == 3 and not any("stale" in row for row in rows)


def test_out_equals_form_accepted(tmp_path: Path):
    out = tmp_path / "results" / "eq.jsonl"
    r = _run(tmp_path, "--path", "x.jsonl", f"--out={out}")
    assert r.returncode == 0, r.stderr
    assert out.exists()


def test_trailing_or_missing_out_fails_cleanly(tmp_path: Path):
    assert _run(tmp_path, "--path", "x.jsonl", "--out").returncode == 2
    assert _run(tmp_path, "--path", "x.jsonl").returncode == 2


def test_failing_worker_reports_warning(tmp_path: Path):
    bin_dir = _fake_uv_bin(tmp_path)
    (bin_dir / "uv").write_text("#!/bin/sh\nexit 3\n")
    out = tmp_path / "results" / "bad.jsonl"
    r = _run(tmp_path, "--path", "x.jsonl", "--out", str(out))
    assert r.returncode != 0
    assert "shard failed" in r.stderr
