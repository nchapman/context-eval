from context_eval.store import TranscriptStore
from context_eval.types import Role


def test_append_assigns_stable_monotonic_ids():
    store = TranscriptStore()
    a = store.append(Role.USER, "hello")
    b = store.append(Role.ASSISTANT, "hi")
    c = store.append(Role.USER, "again")

    assert (a.id, b.id, c.id) == ("m000001", "m000002", "m000003")
    assert store.get("m000002") is b
    assert store.get("m999999") is None


def test_require_raises_for_unknown_id():
    store = TranscriptStore()
    store.append(Role.USER, "x")
    try:
        store.require("m000009")
        raise AssertionError("expected KeyError")
    except KeyError:
        pass


def test_tail_and_length():
    store = TranscriptStore()
    for i in range(5):
        store.append(Role.USER, f"m{i}")

    assert len(store) == 5
    assert [m.content for m in store.tail(2)] == ["m3", "m4"]
    assert store.tail(0) == ()
    assert store.tail(100) == store.messages()


def test_messages_returns_ordered_snapshot():
    store = TranscriptStore()
    store.append(Role.USER, "one")
    store.append(Role.TOOL, "observation", meta={"tool": "grep"})
    msgs = store.messages()

    assert [m.role for m in msgs] == [Role.USER, Role.TOOL]
    assert msgs[1].meta == {"tool": "grep"}
    # Snapshot: later appends do not mutate the returned tuple
    store.append(Role.USER, "two")
    assert len(msgs) == 2
