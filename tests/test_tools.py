"""Tool contract: schema validation, registry, and the dispatch safety gate."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from lai.config import Config, SafetyConfig, load_config
from lai.errors import InvalidToolInput, LaiError, ToolNotFound
from lai.safety.policy import PolicyEngine, Risk
from lai.tools import build_registry
from lai.tools.base import ToolContext, ToolRegistry, ToolResult, ToolSpec, validate_input
from lai.tools.control import BLOCKED_MARKER, DONE_MARKER
from lai.tools.control import register as register_control


def spec(name="probe", *, risk=Risk.READ, handler=None, parameters=None) -> ToolSpec:
    return ToolSpec(
        name=name,
        description="a probe tool",
        parameters=parameters or {"properties": {}},
        handler=handler or (lambda ctx, args: ToolResult.text("ok")),
        risk=risk,
    )


def context(**kwargs) -> ToolContext:
    kwargs.setdefault("config", Config())
    return ToolContext(**kwargs)


# -- validate_input ------------------------------------------------------

SCHEMA = {
    "properties": {
        "name": {"type": "string"},
        "count": {"type": "integer", "minimum": 1, "maximum": 10},
        "ratio": {"type": "number"},
        "flag": {"type": "boolean"},
        "items": {"type": "array"},
        "mode": {"type": "string", "enum": ["a", "b"]},
    },
    "required": ["name"],
}


def test_valid_input_passes():
    validate_input({"name": "x", "count": 3, "mode": "a"}, SCHEMA)


def test_missing_required_field():
    with pytest.raises(InvalidToolInput, match="missing required"):
        validate_input({"count": 1}, SCHEMA)


def test_none_counts_as_missing():
    with pytest.raises(InvalidToolInput):
        validate_input({"name": None}, SCHEMA)


@pytest.mark.parametrize(
    ("payload", "fragment"),
    [
        ({"name": 5}, "must be string"),
        ({"name": "x", "count": "3"}, "must be integer"),
        ({"name": "x", "flag": "yes"}, "must be boolean"),
        ({"name": "x", "items": "a,b"}, "must be array"),
    ],
)
def test_type_errors(payload, fragment):
    with pytest.raises(InvalidToolInput, match=fragment):
        validate_input(payload, SCHEMA)


def test_bool_is_not_an_integer():
    with pytest.raises(InvalidToolInput, match="must be integer"):
        validate_input({"name": "x", "count": True}, SCHEMA)


def test_enum_violation():
    with pytest.raises(InvalidToolInput, match="must be one of"):
        validate_input({"name": "x", "mode": "z"}, SCHEMA)


@pytest.mark.parametrize("count", [0, 11])
def test_bounds(count):
    with pytest.raises(InvalidToolInput):
        validate_input({"name": "x", "count": count}, SCHEMA)


def test_integer_is_acceptable_where_number_expected():
    validate_input({"name": "x", "ratio": 3}, SCHEMA)


def test_unknown_keys_are_tolerated():
    validate_input({"name": "x", "surprise": 1}, SCHEMA)


def test_non_dict_payload_rejected():
    with pytest.raises(InvalidToolInput):
        validate_input(["nope"], SCHEMA)  # type: ignore[arg-type]


# -- registry ------------------------------------------------------------


def test_duplicate_registration_raises_unless_replacing():
    registry = ToolRegistry()
    registry.register(spec())
    with pytest.raises(LaiError, match="already registered"):
        registry.register(spec())
    registry.register(spec(), replace=True)
    assert len(registry) == 1


def test_unknown_tool_message_suggests_alternatives():
    registry = ToolRegistry()
    registry.register(spec("ui_click"))
    with pytest.raises(ToolNotFound) as info:
        registry.get("ui_clik")
    assert "unknown tool" in str(info.value)


def test_membership_names_and_unregister():
    registry = ToolRegistry()
    registry.register(spec("a"))
    registry.register(spec("b"))
    assert "a" in registry and registry.names == ["a", "b"]
    registry.unregister("a")
    assert "a" not in registry and len(registry) == 1
    registry.unregister("missing")  # must not raise


def test_group_filtering_and_exclusion():
    registry = ToolRegistry()
    registry.register(replace(spec("x"), group="one"))
    registry.register(replace(spec("y"), group="two"))
    assert [s.name for s in registry.specs(groups={"one"})] == ["x"]
    assert [s.name for s in registry.specs(exclude={"x"})] == ["y"]


def test_decorator_registration():
    registry = ToolRegistry()

    @registry.tool("decorated", "desc", {"properties": {}}, risk=Risk.WRITE)
    def handler(ctx, args):
        return ToolResult.text("hi")

    assert registry.get("decorated").risk is Risk.WRITE


def test_schema_renderings():
    registry = build_registry()
    for entry in registry.to_anthropic():
        assert set(entry) == {"name", "description", "input_schema"}
        assert entry["input_schema"]["type"] == "object"
    for entry in registry.to_openai():
        assert entry["type"] == "function"
        assert entry["function"]["parameters"]["type"] == "object"
    for entry in registry.to_mcp():
        assert set(entry) == {"name", "description", "inputSchema"}
        assert entry["inputSchema"]["type"] == "object"


def test_build_registry_group_restriction():
    registry = build_registry(groups={"ui"})
    assert registry.names
    assert all(name.startswith("ui_") for name in registry.names)


# -- dispatch ------------------------------------------------------------


def test_handler_exception_becomes_a_failed_result():
    registry = ToolRegistry()

    def boom(ctx, args):
        raise ValueError("kaboom")

    registry.register(spec(handler=boom))
    result = registry.call("probe", {}, context())
    assert result.ok is False
    assert "ValueError" in result.content and "kaboom" in result.content


def test_lai_error_is_reported_with_its_code():
    registry = ToolRegistry()
    registry.register(spec(handler=lambda ctx, args: (_ for _ in ()).throw(LaiError("nope"))))
    result = registry.call("probe", {}, context())
    assert result.ok is False
    assert result.data["error"] == "lai_error"


def test_unknown_tool_returns_failure_not_exception():
    result = ToolRegistry().call("ghost", {}, context())
    assert result.ok is False
    assert "unknown tool" in result.content


def test_plain_dict_return_is_coerced():
    registry = ToolRegistry()
    registry.register(spec(handler=lambda ctx, args: {"a": 1}))
    result = registry.call("probe", {}, context())
    assert result.ok and result.data == {"a": 1}


def test_invalid_input_is_reported_and_handler_never_runs():
    ran = []
    registry = ToolRegistry()
    registry.register(
        spec(handler=lambda ctx, args: ran.append(1), parameters={"properties": {"x": {"type": "string"}}, "required": ["x"]})
    )
    result = registry.call("probe", {}, context())
    assert result.ok is False and "invalid input" in result.content
    assert ran == []


def test_deny_short_circuits_before_the_handler():
    ran = []
    registry = ToolRegistry()
    registry.register(spec(risk=Risk.INPUT, handler=lambda ctx, args: ran.append(1)))
    policy = PolicyEngine(SafetyConfig(mode="readonly"))
    result = registry.call("probe", {}, context(policy=policy))
    assert result.ok is False and result.content.startswith("BLOCKED")
    assert ran == []


def test_ask_with_refusing_approver_does_not_run_the_handler():
    ran = []
    registry = ToolRegistry()
    registry.register(spec(risk=Risk.INPUT, handler=lambda ctx, args: ran.append(1)))
    policy = PolicyEngine(SafetyConfig(mode="ask"))
    result = registry.call("probe", {}, context(policy=policy, approver=lambda *_: False))
    assert result.ok is False and "NOT APPROVED" in result.content
    assert ran == []


def test_ask_with_approving_approver_runs_the_handler():
    seen = []
    registry = ToolRegistry()
    registry.register(spec(risk=Risk.INPUT, handler=lambda ctx, args: seen.append(1) or ToolResult.text("done")))
    policy = PolicyEngine(SafetyConfig(mode="ask"))
    result = registry.call("probe", {}, context(policy=policy, approver=lambda *_: True))
    assert result.ok and seen == [1]


def test_missing_approver_in_ask_mode_is_a_refusal():
    registry = ToolRegistry()
    registry.register(spec(risk=Risk.WRITE))
    policy = PolicyEngine(SafetyConfig(mode="ask"))
    result = registry.call("probe", {}, context(policy=policy))
    assert result.ok is False and "NOT APPROVED" in result.content


def test_approver_that_raises_is_contained():
    registry = ToolRegistry()
    registry.register(spec(risk=Risk.WRITE))
    policy = PolicyEngine(SafetyConfig(mode="ask"))

    def bad_approver(*_):
        raise RuntimeError("ui gone")

    result = registry.call("probe", {}, context(policy=policy, approver=bad_approver))
    assert result.ok is False and "approval failed" in result.content


def test_rate_limiter_only_counts_successful_non_read_calls():
    registry = ToolRegistry()
    registry.register(spec("reader", risk=Risk.READ))
    registry.register(spec("writer", risk=Risk.WRITE))
    policy = PolicyEngine(SafetyConfig(mode="yolo", max_actions_per_minute=2))
    ctx = context(policy=policy)
    for _ in range(5):
        assert registry.call("reader", {}, ctx).ok
    assert registry.call("writer", {}, ctx).ok
    assert registry.call("writer", {}, ctx).ok
    blocked = registry.call("writer", {}, ctx)
    assert blocked.ok is False and "rate limit" in blocked.content


def test_output_is_truncated_to_the_configured_limit():
    registry = ToolRegistry()
    registry.register(spec(handler=lambda ctx, args: ToolResult.text("x" * 5000)))
    config = load_config().with_overrides(
        limits=replace(load_config().limits, max_tool_output_chars=1000)
    )
    result = registry.call("probe", {}, context(config=config))
    assert len(result.content) <= 1000
    assert "truncated" in result.content


def test_audit_records_call_and_result(tmp_path):
    from lai.safety.audit import AuditLog

    audit = AuditLog(tmp_path / "audit.jsonl")
    registry = ToolRegistry()
    registry.register(spec())
    registry.call("probe", {}, context(audit=audit))
    kinds = [entry["kind"] for entry in audit.read()]
    assert "tool_call" in kinds and "tool_result" in kinds


def test_duration_is_measured():
    registry = ToolRegistry()
    registry.register(spec())
    assert registry.call("probe", {}, context()).duration >= 0


# -- ToolResult ----------------------------------------------------------


def test_failure_accepts_a_splatted_error_dict():
    # Regression: LaiError.to_dict() carries a "message" key, which used to
    # collide with the parameter name and raise TypeError inside dispatch.
    result = ToolResult.failure("boom", **LaiError("inner", detail="d").to_dict())
    assert result.ok is False and result.data["message"] == "inner"


def test_truncated_is_a_noop_below_the_limit():
    result = ToolResult.text("short")
    assert result.truncated(1000) is result


def test_json_result_renders_content():
    result = ToolResult.json({"a": 1})
    assert '"a": 1' in result.content and result.data == {"a": 1}


# -- control tools -------------------------------------------------------


def test_task_complete_and_blocked_markers():
    registry = ToolRegistry()
    register_control(registry)
    done = registry.call("task_complete", {"summary": "did it", "verification": "checked"}, context())
    assert done.content == DONE_MARKER
    assert done.data["summary"] == "did it" and done.data["done"] is True

    blocked = registry.call("task_blocked", {"reason": "no app"}, context())
    assert blocked.content == BLOCKED_MARKER
    assert blocked.data["blocked"] is True


def test_task_complete_requires_a_summary():
    registry = ToolRegistry()
    register_control(registry)
    assert registry.call("task_complete", {}, context()).ok is False


def test_plan_update_stores_the_plan_on_the_session():
    from lai.agent.session import Session

    registry = ToolRegistry()
    register_control(registry)
    session = Session()
    result = registry.call(
        "plan_update", {"steps": ["one", "two"], "current": 1}, context(session=session)
    )
    assert session.metadata["plan"] == {"steps": ["one", "two"], "current": 1}
    assert "→ 2. two" in result.content


# -- system tools --------------------------------------------------------


def test_file_write_read_roundtrip_and_append(tmp_path):
    registry = build_registry()
    ctx = context(cwd=tmp_path)
    assert registry.call("file_write", {"path": "a.txt", "content": "hello"}, ctx).ok
    assert registry.call("file_read", {"path": "a.txt"}, ctx).content == "hello"
    registry.call("file_write", {"path": "a.txt", "content": " world", "append": True}, ctx)
    assert registry.call("file_read", {"path": "a.txt"}, ctx).content == "hello world"


def test_file_read_missing_fails_gracefully(tmp_path):
    result = build_registry().call("file_read", {"path": "nope.txt"}, context(cwd=tmp_path))
    assert result.ok is False and "not a file" in result.content


def test_file_list_with_and_without_glob(tmp_path):
    (tmp_path / "one.py").write_text("x")
    (tmp_path / "two.txt").write_text("y")
    (tmp_path / "sub").mkdir()
    registry = build_registry()
    ctx = context(cwd=tmp_path)
    plain = registry.call("file_list", {}, ctx)
    assert "one.py" in plain.content and "d sub" in plain.content
    globbed = registry.call("file_list", {"pattern": "*.py"}, ctx)
    assert "one.py" in globbed.content and "two.txt" not in globbed.content


def test_file_list_rejects_a_non_directory(tmp_path):
    (tmp_path / "f.txt").write_text("x")
    result = build_registry().call("file_list", {"path": "f.txt"}, context(cwd=tmp_path))
    assert result.ok is False


def test_shell_exec_success_failure_and_timeout(tmp_path):
    registry = build_registry()
    ctx = context(cwd=tmp_path)
    ok = registry.call("shell_exec", {"command": "echo hello"}, ctx)
    assert ok.ok and "hello" in ok.content and ok.data["exit_code"] == 0

    bad = registry.call("shell_exec", {"command": "exit 3"}, ctx)
    assert bad.ok is False and bad.data["exit_code"] == 3

    slow = registry.call("shell_exec", {"command": "sleep 5", "timeout": 1}, ctx)
    assert slow.ok is False and "timed out" in slow.content


def test_shell_exec_rejects_a_missing_cwd(tmp_path):
    result = build_registry().call(
        "shell_exec", {"command": "echo x", "cwd": str(tmp_path / "ghost")}, context(cwd=tmp_path)
    )
    assert result.ok is False and "does not exist" in result.content


def test_shell_exec_captures_stderr(tmp_path):
    result = build_registry().call(
        "shell_exec", {"command": "echo oops >&2; exit 1"}, context(cwd=tmp_path)
    )
    assert "oops" in result.content and result.ok is False


# -- failure messages ----------------------------------------------------


def test_a_failure_hint_reaches_the_model_not_just_the_json():
    """Only `content` is sent to the model; guidance left in `data` helps nobody."""
    result = ToolResult.failure("no coding agent is installed", hint="install one of: claude, codex")
    assert "install one of: claude, codex" in result.content
    assert result.data["hint"], "and it stays in data for JSON consumers"


def test_a_failure_detail_is_folded_in_too():
    result = ToolResult.failure("claude exited 1", detail="Invalid API key")
    assert "Invalid API key" in result.content


def test_a_hint_already_in_the_message_is_not_repeated():
    result = ToolResult.failure("run `lai setup` to fix this", hint="run `lai setup` to fix this")
    assert result.content.count("lai setup") == 1


# -- a screenshot for a model that cannot see ----------------------------


class FakeScreen:
    """Just enough desktop to take a screenshot of."""

    def grab(self, region=None, **kwargs):
        from lai.osl.screen import Rect

        return type("Shot", (), {
            "region": Rect(0, 0, 800, 600), "size": (800, 600), "scale": 1.0,
            "png": b"\x89PNG", "to_dict": lambda self: {"width": 800},
        })()

    def virtual_bounds(self):
        from lai.osl.screen import Rect

        return Rect(0, 0, 800, 600)


class FakeDesktop:
    screen = FakeScreen()
    windows = type("W", (), {"active_window": lambda self: None})()
    last_snapshot = None


def screenshot_registry() -> ToolRegistry:
    from lai.tools.computer import register as register_computer

    reg = ToolRegistry()
    register_computer(reg)
    return reg


def test_a_screenshot_becomes_text_when_the_model_has_no_vision():
    """An image a model cannot open is a dead end. Reading it out loud is not."""

    class Words:
        text = "137 x 24 = 3288"
        words = ["137", "x", "24", "=", "3288"]

    class Engine:
        def read(self, region, **kwargs):
            return Words()

    ctx = ToolContext(desktop=FakeDesktop(), extra={"vision": False, "ocr_engine": Engine()})
    result = screenshot_registry().call("computer_screenshot", {"scope": "desktop"}, ctx)
    assert result.ok
    assert result.images == [], "sending an image it cannot read wastes the turn"
    assert "3288" in result.content
    assert "cannot see images" in result.content


def test_a_seeing_model_still_gets_the_picture():
    ctx = ToolContext(desktop=FakeDesktop(), extra={"vision": True})
    result = screenshot_registry().call("computer_screenshot", {"scope": "desktop"}, ctx)
    assert result.ok and result.images == [b"\x89PNG"]


def test_vision_is_assumed_when_nobody_said_otherwise():
    ctx = ToolContext(desktop=FakeDesktop(), extra={})
    assert screenshot_registry().call("computer_screenshot", {}, ctx).images


def test_no_ocr_available_says_what_to_do_instead():
    class Broken:
        def read(self, region, **kwargs):
            raise RuntimeError("tesseract is not installed")

    ctx = ToolContext(desktop=FakeDesktop(), extra={"vision": False, "ocr_engine": Broken()})
    result = screenshot_registry().call("computer_screenshot", {"scope": "desktop"}, ctx)
    assert result.ok
    assert "ui_snapshot" in result.content
    assert "tesseract is not installed" in result.content


def test_a_screenshot_can_be_kept():
    """A screenshot nobody kept is a claim. Saving one is the proof."""
    import tempfile

    target = Path(tempfile.mkdtemp()) / "proof.png"
    ctx = ToolContext(desktop=FakeDesktop(), extra={"vision": True})
    result = screenshot_registry().call("computer_screenshot", {"path": str(target)}, ctx)
    assert result.ok
    assert target.read_bytes() == b"\x89PNG"
    assert str(target) in result.content
    assert result.data["path"] == str(target)


def test_a_saved_screenshot_is_always_a_png():
    import tempfile

    target = Path(tempfile.mkdtemp()) / "proof.jpg"
    ctx = ToolContext(desktop=FakeDesktop(), extra={"vision": True})
    result = screenshot_registry().call("computer_screenshot", {"path": str(target)}, ctx)
    assert result.data["path"].endswith(".png")


def test_a_missing_directory_is_created():
    import tempfile

    target = Path(tempfile.mkdtemp()) / "a" / "b" / "proof.png"
    ctx = ToolContext(desktop=FakeDesktop(), extra={"vision": True})
    screenshot_registry().call("computer_screenshot", {"path": str(target)}, ctx)
    assert target.exists()
