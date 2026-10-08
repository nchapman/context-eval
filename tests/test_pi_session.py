"""Tests for the pi-session transcript converter."""

from __future__ import annotations

import json

from context_eval.datasets.pi_session import convert


def _session_file(tmp_path, records):
    p = tmp_path / "2026-10-06T00-00-00-000Z_deadbeef.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in records))
    return p


def test_convert_maps_roles_and_skips_non_messages(tmp_path):
    path = _session_file(
        tmp_path,
        [
            {"type": "session", "id": "s"},
            {"type": "message", "message": {"role": "user", "content": "hi"},
             "timestamp": "2026-10-06T00:00:01Z"},
            {"type": "message", "message": {"role": "assistant",
             "content": [{"type": "text", "text": "part a"}, {"type": "text", "text": "part b"}]}},
            {"type": "message", "message": {"role": "toolResult", "content": "output"}},
            {"type": "message", "message": {"role": "system", "content": "harness prompt"}},
            {"type": "compaction", "id": "c1"},
            {"type": "model_change", "model": "m"},
        ],
    )
    sample = convert(path)

    assert sample.dataset == "pi-session"
    assert sample.sample_id == "2026-10-06T00-00-00-000Z_deadbeef"
    assert [m.role.value for m in sample.messages] == ["user", "assistant", "tool"]
    assert sample.messages[1].content == "part a\npart b"
    assert sample.meta["skipped_roles"] == {"system": 1}
    assert sample.meta["compaction_events"] == 1
    assert sample.meta["first_timestamp"] == "2026-10-06T00:00:01Z"
    assert not sample.meta["prefix_capped"]


def test_convert_prefix_cap_keeps_messages_within_budget(tmp_path, monkeypatch):
    from context_eval.datasets import pi_session
    from context_eval.tokens import CharEstimator

    monkeypatch.setattr(pi_session, "default_counter", lambda: CharEstimator())
    path = _session_file(
        tmp_path,
        [
            {"type": "message", "message": {"role": "user", "content": "a" * 400}},
            {"type": "message", "message": {"role": "assistant", "content": "b" * 400}},
            {"type": "message", "message": {"role": "user", "content": "c" * 400}},
        ],
    )
    # CharEstimator: 400 chars = 100 tokens + 4 overhead = 104/msg; cap 210 keeps two.
    sample = convert(path, max_tokens=210)
    assert len(sample.messages) == 2
    assert sample.meta["prefix_capped"]


def test_convert_tolerates_unparseable_lines(tmp_path):
    path = tmp_path / "x.jsonl"
    path.write_text("not json\n" + json.dumps(
        {"type": "message", "message": {"role": "user", "content": "q"}}))
    sample = convert(path)
    assert len(sample.messages) == 1
    assert sample.meta["skipped_roles"].get("unparseable") == 1
