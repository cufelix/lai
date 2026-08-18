"""Agent CLIs used as models.

The load-bearing part is the JSON protocol: these CLIs return prose, and the
loop needs tool calls, so the parser has to pull a reply object out of whatever
the model actually printed — and must refuse to invent tool calls when it
cannot. The rest is about failing usefully: an unauthenticated CLI is the most
likely outcome on a fresh machine, and it must say so rather than pass its own
error message off as the model's answer.
"""

from __future__ import annotations

import json

import pytest

from lai.agent.providers.base import Message, TextBlock, ToolCall, ToolResultBlock
from lai.agent.providers.cli_agent import (
    CLI_SPECS,
    CLIAgentProvider,
    CLISpec,
    _extract_json,
    _last_meaningful_lines,
    _parse_calls,
    _transcript,
    _with_hint,
    available_clis,
)
from lai.errors import ProviderError

TOOLS = [
    {"name": "window_list", "description": "List windows.", "input_schema": {"type": "object"}},
    {"name": "task_complete", "description": "Finish.", "input_schema": {"type": "object"}},
]


def provider(monkeypatch, output: str, *, returncode: int = 0, spec: CLISpec | None = None,
             stderr: str = "", record: list | None = None) -> CLIAgentProvider:
    """A provider whose CLI is a stub returning exactly `output`."""
    import subprocess

    spec = spec or CLISpec(name="stub", command="stub-cli", args=("{prompt}",))
    monkeypatch.setattr("lai.agent.providers.cli_agent.shutil.which", lambda name: "/usr/bin/" + name)

    class Result:
        def __init__(self):
            self.returncode = returncode
            self.stdout = output
            self.stderr = stderr

    def fake_run(argv, **kwargs):
        if record is not None:
            record.append((argv, kwargs.get("input")))
        return Result()

    monkeypatch.setattr(subprocess, "run", fake_run)
    return CLIAgentProvider(spec)


def sequence_provider(monkeypatch, outcomes: list[tuple[int, str, str]], record: list | None = None):
    """A CLI stub that fails/succeeds in the given order: (returncode, stdout, stderr)."""
    import subprocess

    monkeypatch.setattr("lai.agent.providers.cli_agent.shutil.which", lambda name: "/usr/bin/" + name)
    monkeypatch.setattr("lai.agent.providers.cli_agent.RETRY_DELAYS", (0.0, 0.0))

    class Result:
        def __init__(self, rc, out, err):
            self.returncode, self.stdout, self.stderr = rc, out, err

    def fake_run(argv, **kwargs):
        index = len(record) if record is not None else 0
        if record is not None:
            record.append(argv)
        rc, out, err = outcomes[min(index, len(outcomes) - 1)]
        return Result(rc, out, err)

    monkeypatch.setattr(subprocess, "run", fake_run)
    return CLIAgentProvider(CLISpec(name="stub", command="stub-cli", args=("{prompt}",)))


# -- reply parsing -------------------------------------------------------


def test_a_clean_json_reply_becomes_a_tool_call(monkeypatch):
    reply = json.dumps({"text": "Looking.", "tool_calls": [{"name": "window_list", "input": {}}]})
    turn = provider(monkeypatch, reply).complete([Message.user("hi")], tools=TOOLS)
    assert turn.text == "Looking."
    assert [c.name for c in turn.tool_calls] == ["window_list"]
    assert turn.stop_reason == "tool_use"


def test_a_reply_in_a_markdown_fence_still_parses(monkeypatch):
    reply = 'Sure!\n```json\n{"text": "ok", "tool_calls": [{"name": "window_list", "input": {}}]}\n```\n'
    turn = provider(monkeypatch, reply).complete([Message.user("hi")], tools=TOOLS)
    assert [c.name for c in turn.tool_calls] == ["window_list"]


def test_a_reply_wrapped_in_prose_still_parses(monkeypatch):
    reply = 'Here is my answer:\n{"text": "ok", "tool_calls": []}\nHope that helps.'
    turn = provider(monkeypatch, reply).complete([Message.user("hi")], tools=TOOLS)
    assert turn.text == "ok" and turn.tool_calls == []


def test_prose_with_no_json_is_treated_as_the_answer(monkeypatch):
    """Never invent tool calls: unparseable prose is a final answer, not a tool."""
    calls: list = []
    p = provider(monkeypatch, "I think you have three windows open.", record=calls)
    turn = p.complete([Message.user("hi")], tools=TOOLS)
    assert turn.tool_calls == []
    assert "three windows" in turn.text
    assert turn.stop_reason == "end_turn"
    assert len(calls) == 2, "it should retry once before giving up on the format"


def test_the_retry_carries_a_correction(monkeypatch):
    calls: list = []
    provider(monkeypatch, "no json here", record=calls).complete([Message.user("hi")], tools=TOOLS)
    first_prompt, second_prompt = calls[0][0][-1], calls[1][0][-1]
    assert "ONLY the JSON object" in second_prompt
    assert "ONLY the JSON object" not in first_prompt


def test_no_retry_when_no_tools_were_offered(monkeypatch):
    calls: list = []
    provider(monkeypatch, "just chatting", record=calls).complete([Message.user("hi")], tools=[])
    assert len(calls) == 1


def test_thinking_and_text_are_streamed(monkeypatch):
    events: list[tuple[str, str]] = []
    reply = json.dumps({"thinking": "considering", "text": "here goes",
                        "tool_calls": [{"name": "window_list", "input": {}}]})
    provider(monkeypatch, reply).complete(
        [Message.user("hi")], tools=TOOLS, stream=lambda kind, payload: events.append((kind, payload))
    )
    kinds = [k for k, _ in events]
    assert "thinking" in kinds and "text" in kinds and "tool" in kinds


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ('{"text": "a"}', {"text": "a"}),
        ('{"tool_calls": []}', {"tool_calls": []}),
        ('{"unrelated": 1}', None),
        ("[1, 2, 3]", None),
        ("", None),
        ("not json at all", None),
        ('{"text": "with }brace inside"}', {"text": "with }brace inside"}),
    ],
)
def test_extract_json_accepts_only_protocol_shaped_objects(raw, expected):
    assert _extract_json(raw) == expected


def test_extract_json_ignores_an_unrelated_object_in_the_prose():
    """A stray JSON blob in the model's chatter must not be mistaken for the reply."""
    text = 'Config looks like {"host": "x", "port": 1} to me. {"text": "the real reply"}'
    assert _extract_json(text) == {"text": "the real reply"}


# -- tool call parsing ---------------------------------------------------


def test_parse_calls_handles_both_key_names():
    calls = _parse_calls([
        {"name": "a", "input": {"x": 1}},
        {"tool": "b", "arguments": {"y": 2}},
    ])
    assert [(c.name, c.input) for c in calls] == [("a", {"x": 1}), ("b", {"y": 2})]


def test_parse_calls_decodes_stringified_arguments():
    calls = _parse_calls([{"name": "a", "input": '{"x": 1}'}])
    assert calls[0].input == {"x": 1}


def test_parse_calls_skips_unusable_entries():
    calls = _parse_calls([{"input": {}}, "nonsense", 42, {"name": "  "}, {"name": "ok"}])
    assert [c.name for c in calls] == ["ok"]


def test_parse_calls_gives_each_call_a_unique_id():
    calls = _parse_calls([{"name": "a"}, {"name": "a"}])
    assert calls[0].id != calls[1].id


def test_parse_calls_tolerates_a_non_list():
    assert _parse_calls(None) == [] and _parse_calls({"name": "a"}) == []


# -- prompt rendering ----------------------------------------------------


def test_the_transcript_renders_every_block_type():
    text = _transcript([
        Message("user", [TextBlock("do a thing")]),
        Message("assistant", [ToolCall("1", "window_list", {"scope": "all"})]),
        Message("user", [ToolResultBlock("1", "7 windows", images=(b"png",))]),
    ])
    assert "do a thing" in text
    assert "window_list" in text
    assert "7 windows" in text
    assert "not visible here" in text, "the model must know it cannot see the screenshot"


def test_the_transcript_marks_an_error_result():
    text = _transcript([Message("user", [ToolResultBlock("1", "boom", is_error=True)])])
    assert "ERROR" in text


def test_an_empty_transcript_says_so():
    assert "nothing yet" in _transcript([])


def test_the_prompt_carries_the_tools_and_the_protocol(monkeypatch):
    calls: list = []
    provider(monkeypatch, '{"text": "x"}', record=calls).complete(
        [Message.user("hello")], system="You drive a desktop.", tools=TOOLS
    )
    prompt = calls[0][0][-1]
    assert "window_list" in prompt and "task_complete" in prompt
    assert "ONE JSON object" in prompt
    assert "You drive a desktop." in prompt
    assert "hello" in prompt


# -- failure handling ----------------------------------------------------


def test_a_failing_cli_raises_rather_than_answering(monkeypatch):
    """An auth failure must not be passed off as the model's reply."""
    p = provider(monkeypatch, "", returncode=1, stderr="ERROR: 401 Unauthorized")
    with pytest.raises(ProviderError, match="exited 1"):
        p.complete([Message.user("hi")], tools=TOOLS)


def test_empty_output_is_an_error(monkeypatch):
    with pytest.raises(ProviderError, match="no output"):
        provider(monkeypatch, "   ").complete([Message.user("hi")], tools=TOOLS)


def test_a_transient_api_error_is_waited_out(monkeypatch):
    """A hosted-model hiccup must not kill a two-hour run at step twelve."""
    calls: list = []
    p = sequence_provider(monkeypatch, [
        (1, "", '{"is_error":true,"terminal_reason":"api_error"}'),
        (0, json.dumps({"text": "recovered"}), ""),
    ], record=calls)
    turn = p.complete([Message.user("hi")], tools=TOOLS)
    assert len(calls) == 2, "the first failure should have been retried"
    assert turn.message.content[0].text == "recovered"


def test_a_persistent_api_error_raises_after_every_retry(monkeypatch):
    calls: list = []
    p = sequence_provider(monkeypatch, [
        (1, "", '{"is_error":true,"terminal_reason":"api_error"}'),
    ], record=calls)
    with pytest.raises(ProviderError, match="exited 1"):
        p.complete([Message.user("hi")], tools=TOOLS)
    assert len(calls) == 3, "initial attempt plus both retries"


def test_an_auth_failure_is_not_retried(monkeypatch):
    calls: list = []
    p = sequence_provider(monkeypatch, [(1, "", "ERROR: 401 Unauthorized")], record=calls)
    with pytest.raises(ProviderError, match="401"):
        p.complete([Message.user("hi")], tools=TOOLS)
    assert len(calls) == 1, "retrying a login prompt just wastes a minute"


def test_a_missing_cli_is_reported_at_construction(monkeypatch):
    monkeypatch.setattr("lai.agent.providers.cli_agent.shutil.which", lambda name: None)
    with pytest.raises(ProviderError, match="not installed"):
        CLIAgentProvider("claude")


def test_an_unknown_cli_name_lists_the_known_ones():
    with pytest.raises(ProviderError, match="unknown agent CLI"):
        CLIAgentProvider("definitely-not-a-cli")


def test_a_timeout_is_reported_as_one(monkeypatch):
    import subprocess

    monkeypatch.setattr("lai.agent.providers.cli_agent.shutil.which", lambda name: "/usr/bin/x")

    def explode(*a, **kw):
        raise subprocess.TimeoutExpired(cmd="stub", timeout=1)

    monkeypatch.setattr(subprocess, "run", explode)
    spec = CLISpec(name="stub", command="stub-cli", args=("{prompt}",))
    with pytest.raises(ProviderError, match="timed out"):
        CLIAgentProvider(spec, timeout=1).complete([Message.user("hi")], tools=TOOLS)


def test_the_claude_json_envelope_is_unwrapped(monkeypatch):
    envelope = json.dumps({"is_error": False, "result": '{"text": "unwrapped", "tool_calls": []}'})
    turn = provider(monkeypatch, envelope, spec=CLI_SPECS["claude"]).complete(
        [Message.user("hi")], tools=TOOLS
    )
    assert turn.text == "unwrapped"


def test_an_error_flag_in_the_envelope_raises(monkeypatch):
    envelope = json.dumps({"is_error": True, "result": "credit balance too low"})
    with pytest.raises(ProviderError, match="reported an error"):
        provider(monkeypatch, envelope, spec=CLI_SPECS["claude"]).complete(
            [Message.user("hi")], tools=TOOLS
        )


# -- diagnostics ---------------------------------------------------------


def test_the_reason_is_pulled_out_of_a_retry_storm():
    noisy = "\n".join([
        "session id: abc",
        "reasoning effort: none",
        *["2026-08-18T09:00:00Z ERROR failed to connect: HTTP error: 401 Unauthorized"] * 6,
        "ERROR: Reconnecting... 3/5",
    ])
    summary = _last_meaningful_lines(noisy)
    assert "401 Unauthorized" in summary
    assert summary.count("401") == 1, "the same reason repeated is still one reason"
    assert "2026-08-18T" not in summary, "timestamps are noise here"


def test_without_a_recognisable_reason_it_shows_the_tail():
    summary = _last_meaningful_lines("session id: x\nsomething odd\nlast line")
    assert "last line" in summary and "session id" not in summary


def test_an_auth_failure_gains_a_sign_in_hint():
    assert "codex login" in _with_hint("codex", "401 Unauthorized")
    assert "GEMINI_API_KEY" in _with_hint("gemini", "you must specify the GEMINI_API_KEY")


def test_an_unrelated_failure_gains_no_hint():
    assert _with_hint("codex", "disk full") == "disk full"


# -- specs ---------------------------------------------------------------


def test_every_spec_names_a_command_and_explains_itself():
    for name, spec in CLI_SPECS.items():
        assert spec.command, name
        assert spec.describe, f"{name} must explain what it needs"
        assert spec.stdin or any("{prompt}" in arg for arg in spec.args), (
            f"{name} has no way to receive the prompt"
        )


def test_build_places_the_prompt_and_the_model():
    spec = CLI_SPECS["claude"]
    argv, stdin = spec.build("PROMPT", "opus")
    assert "PROMPT" in argv and stdin is None
    assert argv[argv.index("--model") + 1] == "opus"


def test_build_uses_stdin_when_the_cli_wants_it():
    spec = CLI_SPECS["codex"]
    argv, stdin = spec.build("PROMPT", "", "/tmp/answer.txt")
    assert stdin == "PROMPT"
    assert "PROMPT" not in argv
    assert argv[argv.index("--output-last-message") + 1] == "/tmp/answer.txt"


def test_available_clis_reflects_the_path(monkeypatch):
    monkeypatch.setattr("lai.agent.providers.cli_agent.shutil.which",
                        lambda name: "/usr/bin/claude" if name == "claude" else None)
    assert [spec.name for spec in available_clis()] == ["claude"]


def test_a_file_answer_wins_over_stdout(monkeypatch, tmp_path):
    """codex prints a banner to stdout and the real answer to a file."""
    import subprocess

    monkeypatch.setattr("lai.agent.providers.cli_agent.shutil.which", lambda name: "/usr/bin/x")

    class Result:
        returncode = 0
        stdout = "banner noise, config dump, progress lines"
        stderr = ""

    def fake_run(argv, **kwargs):
        target = argv[argv.index("--output-last-message") + 1]
        with open(target, "w", encoding="utf-8") as fh:
            fh.write('{"text": "from the file", "tool_calls": []}')
        return Result()

    monkeypatch.setattr(subprocess, "run", fake_run)
    turn = CLIAgentProvider(CLI_SPECS["codex"]).complete([Message.user("hi")], tools=TOOLS)
    assert turn.text == "from the file"


def test_the_answer_file_is_cleaned_up(monkeypatch):
    import subprocess
    from pathlib import Path

    monkeypatch.setattr("lai.agent.providers.cli_agent.shutil.which", lambda name: "/usr/bin/x")
    seen: list[str] = []

    class Result:
        returncode = 0
        stdout = '{"text": "x"}'
        stderr = ""

    def fake_run(argv, **kwargs):
        seen.append(argv[argv.index("--output-last-message") + 1])
        return Result()

    monkeypatch.setattr(subprocess, "run", fake_run)
    CLIAgentProvider(CLI_SPECS["codex"]).complete([Message.user("hi")], tools=TOOLS)
    assert seen and not Path(seen[0]).exists()


def test_usage_is_reported_as_unknown_rather_than_guessed(monkeypatch):
    turn = provider(monkeypatch, '{"text": "x"}').complete([Message.user("hi")], tools=TOOLS)
    assert turn.usage.total == 0, "a CLI reports cost, not tokens; inventing a count would be a lie"
