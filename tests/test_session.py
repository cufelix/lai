"""Session: transcript persistence, image pruning, and safe compaction."""

from __future__ import annotations

import time

from lai.agent.providers.base import (
    ImageBlock,
    Message,
    TextBlock,
    ThinkingBlock,
    ToolCall,
    ToolResultBlock,
    Usage,
)
from lai.agent.session import Session

PNG = b"\x89PNG\r\n\x1a\n" + b"0" * 64


def test_append_and_text_access():
    session = Session()
    session.append(Message.user("hello"))
    session.append(Message.assistant("hi there"))
    assert len(session.messages) == 2
    assert session.last_assistant_text() == "hi there"


def test_last_assistant_text_skips_empty_turns():
    session = Session()
    session.append(Message.assistant("real answer"))
    session.append(Message("assistant", [ToolCall("c1", "echo", {})]))
    assert session.last_assistant_text() == "real answer"


def test_usage_accumulates():
    session = Session()
    session.add_usage(Usage(10, 5))
    session.add_usage(Usage(3, 2))
    assert session.usage.input_tokens == 13
    assert session.usage.output_tokens == 7
    assert session.usage.total == 20


def test_persist_and_reload_roundtrip(tmp_path):
    session = Session(task="do a thing")
    session.bind(tmp_path)
    session.append(Message.user("open the editor"))
    session.append(Message("assistant", [TextBlock("on it"), ToolCall("c1", "app_open", {"name": "Xed"})]))
    session.append(Message("user", [ToolResultBlock("c1", "opened", is_error=False)]))

    reloaded = Session.load(session.path)
    assert reloaded.task == "do a thing"
    assert len(reloaded.messages) == 3
    assert reloaded.messages[1].tool_calls[0].name == "app_open"
    assert reloaded.messages[1].tool_calls[0].input == {"name": "Xed"}
    results = [b for b in reloaded.messages[2].content if isinstance(b, ToolResultBlock)]
    assert results[0].content == "opened"


def test_reloading_replaces_images_with_a_placeholder(tmp_path):
    session = Session()
    session.bind(tmp_path)
    session.append(Message.user("look", images=[PNG]))
    reloaded = Session.load(session.path)
    assert not any(isinstance(b, ImageBlock) for b in reloaded.messages[0].content)
    assert "earlier session" in reloaded.messages[0].text


def test_load_of_a_missing_file_is_empty(tmp_path):
    assert Session.load(tmp_path / "ghost.jsonl").messages == []


def test_load_skips_corrupt_lines(tmp_path):
    path = tmp_path / "s.jsonl"
    path.write_text('{"kind":"session_start","id":"x","task":"t","at":1}\nnot json\n', encoding="utf-8")
    session = Session.load(path)
    assert session.task == "t" and session.messages == []


def test_unbound_session_does_not_write():
    session = Session()
    session.append(Message.user("x"))  # must not raise
    assert session.path is None


def test_estimate_tokens_grows_with_content():
    session = Session()
    before = session.estimate_tokens()
    session.append(Message.user("a" * 4000))
    after_text = session.estimate_tokens()
    session.append(Message.user("look", images=[PNG]))
    after_image = session.estimate_tokens()
    assert before < after_text < after_image
    assert after_image - after_text >= 1000  # an image costs about a thousand tokens


def test_estimate_counts_every_block_type():
    session = Session()
    session.append(
        Message("assistant", [
            TextBlock("t" * 400),
            ThinkingBlock("h" * 400),
            ToolCall("c1", "x", {"a": "b" * 100}),
        ])
    )
    session.append(Message("user", [ToolResultBlock("c1", "r" * 400, images=(PNG,))]))
    assert session.estimate_tokens() > 1000


def test_prune_images_keeps_only_the_newest():
    session = Session()
    for index in range(5):
        session.append(Message.user(f"shot {index}", images=[PNG]))
    removed = session.prune_images(keep=2)
    assert removed == 3
    remaining = [b for m in session.messages for b in m.content if isinstance(b, ImageBlock)]
    assert len(remaining) == 2
    placeholders = [
        b for m in session.messages for b in m.content
        if isinstance(b, TextBlock) and "older screenshot removed" in b.text
    ]
    assert len(placeholders) == 3


def test_prune_images_strips_tool_result_images_too():
    session = Session()
    for index in range(4):
        session.append(Message("user", [ToolResultBlock(f"c{index}", f"shot {index}", images=(PNG,))]))
    session.prune_images(keep=1)
    with_images = [
        b for m in session.messages for b in m.content
        if isinstance(b, ToolResultBlock) and b.images
    ]
    assert len(with_images) == 1
    stripped = [
        b for m in session.messages for b in m.content
        if isinstance(b, ToolResultBlock) and not b.images
    ]
    assert all("removed to save context" in b.content for b in stripped)


def test_prune_is_a_noop_when_under_the_limit():
    session = Session()
    session.append(Message.user("one", images=[PNG]))
    assert session.prune_images(keep=3) == 0


def test_compact_is_a_noop_for_a_short_transcript():
    session = Session()
    for index in range(4):
        session.append(Message.user(f"m{index}"))
    assert session.compact("summary", keep_recent=8) == 0
    assert len(session.messages) == 4


def test_compact_replaces_history_with_a_summary():
    session = Session()
    for index in range(20):
        session.append(Message.user(f"m{index}"))
    dropped = session.compact("here is what happened", keep_recent=5)
    assert dropped > 0
    assert "here is what happened" in session.messages[0].text
    assert len(session.messages) == 6


def test_compact_never_orphans_a_tool_result():
    """The naive cut point here lands on a tool_result whose tool_use would be
    dropped — the boundary must move back so the pairing survives."""
    session = Session()
    for index in range(12):
        session.append(Message.user(f"filler {index}"))
    # Tail: assistant tool_use, then the matching user tool_result.
    session.append(Message("assistant", [ToolCall("call-A", "echo", {})]))
    session.append(Message("user", [ToolResultBlock("call-A", "done")]))
    session.append(Message.assistant("finished"))

    session.compact("summary", keep_recent=2)

    ids_used = {b.id for m in session.messages for b in m.content if isinstance(b, ToolCall)}
    ids_returned = {
        b.tool_use_id for m in session.messages for b in m.content if isinstance(b, ToolResultBlock)
    }
    assert ids_returned <= ids_used, "a tool_result survived without its tool_use"


def test_summary_shape():
    session = Session(task="t")
    session.append(Message.user("x"))
    summary = session.summary()
    assert summary["task"] == "t"
    assert summary["messages"] == 1
    assert "usage" in summary and "estimated_tokens" in summary


def test_list_sessions_is_newest_first(tmp_path):
    first = Session(task="older")
    first.bind(tmp_path)
    time.sleep(0.01)
    second = Session(task="newer")
    second.bind(tmp_path)

    listed = Session.list_sessions(tmp_path)
    assert [entry["task"] for entry in listed][:2] == ["newer", "older"]
    assert listed[0]["id"] == second.id


def test_list_sessions_on_a_missing_directory(tmp_path):
    assert Session.list_sessions(tmp_path / "nope") == []


def test_list_sessions_respects_the_limit(tmp_path):
    for index in range(5):
        Session(task=f"t{index}").bind(tmp_path)
    assert len(Session.list_sessions(tmp_path, limit=2)) == 2
