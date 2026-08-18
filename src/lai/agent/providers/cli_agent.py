"""Use an installed coding-agent CLI as the model.

Claude Code, Codex and Gemini CLI all authenticate against a *subscription*
rather than a metered API key, and a lot of people already have one working on
their machine. This provider borrows it: LAI shells out to the CLI for each
turn instead of calling an API, so a desktop agent runs with no API key at all.

The catch is that those CLIs return prose, and LAI's loop needs tool calls. So
this provider carries its own function-calling protocol: the tool schemas and
the transcript are rendered into a prompt, and the CLI is asked to reply with a
single JSON object naming the tools to run. That is genuinely second-class —
no streaming, no images, a slower turn, and a model that can decline to follow
the format — and it is worth stating plainly rather than papering over:

* **Vision goes through the filesystem.** These CLIs cannot take an image in
  the prompt, but they can *read* one: screenshots are written to a temporary
  directory and their paths handed over, which is how the CLI ends up looking
  at the screen. It costs a round trip inside the CLI's own loop, so it is
  slower than an API that takes images inline — but the agent can see.
* **Whole transcript per turn.** These CLIs are stateless per invocation, so
  every turn re-sends the conversation. Fine for a desktop task, wasteful for a
  long one.
* **Parse failures happen.** A model that answers in prose gets one corrective
  retry, then its prose is treated as the final answer.

Everything else in LAI — the safety gate, the a11y grounding, the budgets — is
unchanged, because this is just another `Provider`.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

from ...errors import ProviderError
from .base import Message, TextBlock, ToolCall, TurnResult, Usage

DEFAULT_TIMEOUT = 600.0
MAX_PROMPT_CHARS = 400_000
MAX_STAGED_IMAGES = 2

# An agent CLI fronts a hosted model, and hosted models have bad minutes. A
# failure that smells like the network or the far end is worth waiting out —
# three instant attempts all landing inside the same outage is how a
# two-hour-long run dies at step twelve.
TRANSIENT_MARKERS = (
    "api_error", "api error", "overloaded", "rate limit", "rate_limit",
    "service unavailable", "internal server error", "bad gateway",
    "502", "503", "529", "econnreset", "econnrefused", "connection reset",
    "connection refused", "network error", "temporarily unavailable",
)
RETRY_DELAYS = (3.0, 8.0)


def _is_transient(detail: str) -> bool:
    lowered = detail.lower()
    return any(marker in lowered for marker in TRANSIENT_MARKERS)


@dataclass(frozen=True, slots=True)
class CLISpec:
    """How to drive one agent CLI non-interactively."""

    name: str
    command: str
    args: tuple[str, ...]
    """Arguments around the prompt; ``{prompt}`` marks where it goes, if at all."""
    stdin: bool = False
    """True when the prompt is delivered on stdin rather than as an argument."""
    result_key: str = ""
    """Key holding the reply in JSON output; empty means the output is plain text."""
    model_flag: str = ""
    """Flag used to select a model, if the CLI supports one."""
    output_file_flag: str = ""
    """Flag naming a file the CLI writes its final message to.

    Preferable to parsing stdout whenever a CLI offers it: `codex exec` prints
    a configuration banner, an echo of the prompt and progress lines around the
    answer, and picking the answer back out of that is guesswork.
    """
    image_args: tuple[str, ...] = ()
    """Extra arguments needed before the CLI may read image files.

    ``{dir}`` is replaced with the directory holding this turn's screenshots.
    Empty means the CLI cannot see images at all.
    """
    describe: str = ""
    extra_env: dict = field(default_factory=dict)

    @property
    def sees_images(self) -> bool:
        return bool(self.image_args)

    def build(
        self, prompt: str, model: str = "", output_file: str = "", image_dir: str = ""
    ) -> tuple[list[str], str | None]:
        """Return (argv, stdin_payload)."""
        argv = [self.command]
        argv.extend(prompt if arg == "{prompt}" else arg for arg in self.args)
        if model and self.model_flag:
            argv.extend([self.model_flag, model])
        if output_file and self.output_file_flag:
            argv.extend([self.output_file_flag, output_file])
        if image_dir and self.image_args:
            argv.extend(arg.replace("{dir}", image_dir) for arg in self.image_args)
        return argv, (prompt if self.stdin else None)


# Each entry is one way somebody already has a model on their machine.
CLI_SPECS: dict[str, CLISpec] = {
    "claude": CLISpec(
        name="claude",
        command="claude",
        args=("-p", "{prompt}", "--output-format", "json"),
        result_key="result",
        model_flag="--model",
        image_args=("--allowedTools", "Read", "--add-dir", "{dir}"),
        describe="Claude Code CLI (subscription or ANTHROPIC_API_KEY)",
    ),
    "codex": CLISpec(
        name="codex",
        command="codex",
        args=("exec", "--skip-git-repo-check", "--sandbox", "read-only", "-"),
        stdin=True,
        output_file_flag="--output-last-message",
        model_flag="-m",
        image_args=("-c", "sandbox_permissions=[\"disk-full-read-access\"]"),
        describe="OpenAI Codex CLI (ChatGPT plan or OPENAI_API_KEY)",
    ),
    "gemini": CLISpec(
        name="gemini",
        command="gemini",
        args=("-o", "json", "{prompt}"),
        result_key="response",
        model_flag="-m",
        image_args=("--include-directories", "{dir}"),
        describe="Gemini CLI (Google account or GEMINI_API_KEY)",
    ),
    "opencode": CLISpec(
        name="opencode",
        command="opencode",
        args=("run", "{prompt}"),
        model_flag="-m",
        describe="opencode CLI (whatever backend it is configured with)",
    ),
}

PROTOCOL = """\
You are the reasoning engine for an autonomous desktop agent. You do not talk to \
a human: your entire reply is parsed by a program.

Reply with ONE JSON object and nothing else. No prose before it, no prose after \
it, no markdown fence. The object has these keys:

  "thinking":   optional string, your private reasoning
  "text":       optional string, what you would say to the user
  "tool_calls": a list (possibly empty) of {"name": "<tool>", "input": {...}}

Call tools by putting them in "tool_calls". Do not describe a tool call in \
prose — a description does nothing. When the task is finished, call the \
"task_complete" tool. When you are stuck, call "task_blocked".

Example of a valid reply:
{"text": "Opening the calculator.", "tool_calls": [{"name": "app_open", "input": {"name": "Calculator"}}]}
"""

REPAIR = (
    "That was not a single JSON object. Reply again with ONLY the JSON object "
    "described earlier — no prose, no markdown fence."
)


class CLIAgentProvider:
    """A coding-agent CLI, dressed up as a tool-calling model."""

    def __init__(
        self,
        spec: CLISpec | str,
        *,
        model: str = "",
        timeout: float = DEFAULT_TIMEOUT,
        cwd: str = "",
        max_tokens: int = 0,
    ) -> None:
        if isinstance(spec, str):
            resolved = CLI_SPECS.get(spec)
            if resolved is None:
                raise ProviderError(
                    f"unknown agent CLI {spec!r}",
                    detail=f"known: {', '.join(sorted(CLI_SPECS))}",
                )
            spec = resolved
        self.spec = spec
        self.name = f"cli:{spec.name}"
        self.model = model or spec.name
        self.timeout = timeout
        # A coding CLI inspects the directory it runs in. Somewhere neutral
        # keeps it from wandering into a repository and reading the code.
        self.cwd = cwd or os.path.expanduser("~")
        self.max_tokens = max_tokens
        self.calls = 0
        self._image_dir: Path | None = None

        if not shutil.which(spec.command):
            raise ProviderError(
                f"{spec.command!r} is not installed",
                detail=spec.describe or f"install {spec.command} and try again",
            )

    # -- Provider protocol ------------------------------------------------

    def complete(
        self,
        messages: list[Message],
        *,
        system: str = "",
        tools: list[dict] | None = None,
        stream=None,
    ) -> TurnResult:
        staged = self._stage_images(messages)
        try:
            prompt = self._render(messages, system=system, tools=tools or [], images=staged)
            if stream:
                stream("status", f"asking {self.spec.command}…")

            raw = self._invoke(prompt, image_dir=staged[0].parent if staged else None)
            parsed = _extract_json(raw)

            if parsed is None and (tools or []):
                # One corrective attempt: models drift out of the format, and a
                # retry is far cheaper than losing the turn.
                if stream:
                    stream("status", "reply was not JSON — asking again")
                raw = self._invoke(
                    prompt + "\n\n" + REPAIR + "\n",
                    image_dir=staged[0].parent if staged else None,
                )
                parsed = _extract_json(raw)
        finally:
            self._clear_images()

        if parsed is None:
            # Treat prose as a plain answer rather than inventing tool calls.
            return self._turn(text=raw.strip(), calls=[], raw=raw)

        text = str(parsed.get("text") or "").strip()
        thinking = str(parsed.get("thinking") or "").strip()
        if stream and thinking:
            stream("thinking", thinking)
        if stream and text:
            stream("text", text)

        calls = _parse_calls(parsed.get("tool_calls"))
        for call in calls:
            if stream:
                stream("tool", call.name)
        return self._turn(text=text, calls=calls, raw=raw)

    def close(self) -> None:
        return None

    # -- internals --------------------------------------------------------

    def _turn(self, *, text: str, calls: list[ToolCall], raw: str) -> TurnResult:
        blocks: list = []
        if text:
            blocks.append(TextBlock(text))
        blocks.extend(calls)
        if not blocks:
            blocks.append(TextBlock(""))
        return TurnResult(
            message=Message("assistant", blocks),
            stop_reason="tool_use" if calls else "end_turn",
            # A CLI reports cost, not tokens, and not in a shape worth guessing
            # at. Reporting zero is honest; inventing an estimate is not.
            usage=Usage(),
            model=self.model,
            raw=raw,
        )

    def _invoke(self, prompt: str, *, image_dir=None) -> str:
        if len(prompt) > MAX_PROMPT_CHARS:
            prompt = prompt[:MAX_PROMPT_CHARS] + "\n[transcript truncated]\n"

        model = self.model if self.model != self.spec.name else ""
        # (0.0, 3.0, 8.0): the first try, then two waits spaced far enough apart
        # that a far-end hiccup can clear between them.
        for attempt, delay in enumerate((0.0, *RETRY_DELAYS)):
            if delay:
                time.sleep(delay)
            answer_file = ""
            if self.spec.output_file_flag:
                handle, answer_file = tempfile.mkstemp(prefix="lai-cli-", suffix=".txt")
                os.close(handle)
            argv, stdin_payload = self.spec.build(prompt, model, answer_file, str(image_dir or ""))
            env = {**os.environ, **self.spec.extra_env}
            self.calls += 1
            try:
                try:
                    result = subprocess.run(
                        argv,
                        input=stdin_payload,
                        capture_output=True,
                        text=True,
                        timeout=self.timeout,
                        cwd=self.cwd,
                        env=env,
                        check=False,
                    )
                except subprocess.TimeoutExpired as exc:
                    raise ProviderError(
                        f"{self.spec.command} timed out after {self.timeout:.0f}s",
                        detail="agent CLIs are slow; raise provider.timeout if this is normal for you",
                    ) from exc
                except OSError as exc:
                    raise ProviderError(f"could not run {self.spec.command}: {exc}") from exc

                if result.returncode != 0:
                    combined = ((result.stderr or "") + "\n" + (result.stdout or "")).strip()
                    raise ProviderError(
                        f"{self.spec.command} exited {result.returncode}",
                        detail=_with_hint(self.spec.name, _last_meaningful_lines(combined)) or "no output",
                    )

                if answer_file:
                    try:
                        written = Path(answer_file).read_text(encoding="utf-8", errors="replace").strip()
                    except OSError:
                        written = ""
                    if written:
                        return written

                output = (result.stdout or "").strip()
                if not output:
                    raise ProviderError(
                        f"{self.spec.command} produced no output",
                        detail=_last_meaningful_lines(result.stderr or ""),
                    )
                return self._unwrap(output)
            except ProviderError as exc:
                final = attempt >= len(RETRY_DELAYS)
                if final or not _is_transient(str(exc)):
                    raise
                # transient — the loop's next delay is the backoff
            finally:
                if answer_file:
                    Path(answer_file).unlink(missing_ok=True)
        raise ProviderError(f"{self.spec.command} failed after {len(RETRY_DELAYS) + 1} attempts")

    def _stage_images(self, messages: list[Message]) -> list:
        """Write the newest screenshots to a temp dir the CLI is allowed to read.

        These CLIs take no inline images, but they can open a file — so the
        filesystem is the channel. Only the most recent few are staged: the
        loop already prunes old screenshots, and handing a model six versions
        of the same desktop invites it to reason about a stale one.
        """
        if not self.spec.sees_images:
            return []

        blobs: list[bytes] = []
        for message in reversed(messages):
            for block in message.content:
                if getattr(block, "type", "") == "image":
                    blobs.append(block.data)
                elif getattr(block, "type", "") == "tool_result":
                    blobs.extend(block.images)
            if len(blobs) >= MAX_STAGED_IMAGES:
                break
        if not blobs:
            return []

        self._image_dir = self._image_dir or Path(tempfile.mkdtemp(prefix="lai-vision-"))
        staged: list[Path] = []
        for index, blob in enumerate(blobs[:MAX_STAGED_IMAGES]):
            path = self._image_dir / f"screen-{index}.png"
            try:
                path.write_bytes(blob)
            except OSError:
                continue
            staged.append(path)
        return staged

    def _clear_images(self) -> None:
        if self._image_dir is None:
            return
        shutil.rmtree(self._image_dir, ignore_errors=True)
        self._image_dir = None

    def _unwrap(self, output: str) -> str:
        """Pull the model's reply out of the CLI's own JSON envelope."""
        if not self.spec.result_key:
            return output
        try:
            payload = json.loads(output)
        except json.JSONDecodeError:
            return output
        if isinstance(payload, dict):
            if payload.get("is_error"):
                raise ProviderError(
                    f"{self.spec.command} reported an error",
                    detail=str(payload.get("result") or payload.get("error") or "")[:400],
                )
            value = payload.get(self.spec.result_key)
            if isinstance(value, str):
                return value
        return output

    def _render(self, messages: list[Message], *, system: str, tools: list[dict],
                images: list | None = None) -> str:
        parts = [PROTOCOL]
        if images:
            listing = "\n".join(f"  {path}" for path in images)
            parts.append(
                "## What the screen looks like right now\n"
                "Screenshots of the desktop have been written to these files. "
                "READ THEM before deciding what to do — they are how you see:\n"
                f"{listing}\n"
                "Use your own file-reading tool to open them, then continue. "
                "Your final reply must still be the single JSON object."
            )
        if tools:
            parts.append("## Tools you may call\n")
            for tool in tools:
                schema = json.dumps(tool.get("input_schema", {}), ensure_ascii=False)
                parts.append(f"### {tool.get('name')}\n{tool.get('description', '')}\ninput schema: {schema}\n")
        if system:
            parts.append("## Your operating context\n" + system)
        parts.append("## Conversation so far\n" + _transcript(messages))
        parts.append(
            "Now reply with the single JSON object for your next step. "
            "Remember: tool calls belong in \"tool_calls\", not in prose."
        )
        return "\n\n".join(parts)


# -- parsing --------------------------------------------------------------


def _transcript(messages: list[Message]) -> str:
    lines: list[str] = []
    for message in messages:
        for block in message.content:
            kind = getattr(block, "type", "")
            if kind == "text":
                lines.append(f"{message.role}: {block.text}")
            elif kind == "tool_use":
                lines.append(f"assistant called {block.name}({json.dumps(block.input, ensure_ascii=False)[:600]})")
            elif kind == "tool_result":
                marker = "ERROR" if block.is_error else "result"
                extra = f" (+{len(block.images)} image(s), not visible here)" if block.images else ""
                lines.append(f"{marker} of that tool: {block.content[:2000]}{extra}")
            elif kind == "image":
                lines.append("[a screenshot was taken — see the image files listed above]")
    return "\n".join(lines) if lines else "(nothing yet)"


_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def _extract_json(text: str) -> dict | None:
    """Find the reply object in whatever the CLI actually printed."""
    if not text:
        return None
    candidate = text.strip()

    for attempt in (candidate, *(m.group(1).strip() for m in _FENCE.finditer(candidate))):
        parsed = _try_object(attempt)
        if parsed is not None:
            return parsed

    # Last resort: the first balanced {...} run in the output.
    start = candidate.find("{")
    while start != -1:
        depth, in_string, escaped = 0, False, False
        for index in range(start, len(candidate)):
            char = candidate[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    parsed = _try_object(candidate[start : index + 1])
                    if parsed is not None:
                        return parsed
                    break
        start = candidate.find("{", start + 1)
    return None


def _try_object(text: str) -> dict | None:
    try:
        value = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(value, dict):
        return None
    # Only accept something shaped like our protocol, so a stray JSON blob in
    # the model's prose is not mistaken for the reply.
    if not any(key in value for key in ("tool_calls", "text", "thinking")):
        return None
    return value


def _parse_calls(raw) -> list[ToolCall]:
    if not isinstance(raw, list):
        return []
    calls: list[ToolCall] = []
    for index, entry in enumerate(raw):
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("name") or entry.get("tool") or "").strip()
        if not name:
            continue
        arguments = entry.get("input", entry.get("arguments", {}))
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except (json.JSONDecodeError, ValueError):
                arguments = {}
        if not isinstance(arguments, dict):
            arguments = {}
        calls.append(ToolCall(id=f"cli_{index}_{name}", name=name, input=arguments))
    return calls


# Words that mark the line actually worth showing a user, as opposed to the
# retry spam and configuration banner an agent CLI wraps it in.
_SIGNAL = re.compile(
    r"unauthoriz|forbidden|api[_ -]?key|not logged in|log ?in|authenticat|credential|"
    r"quota|rate limit|billing|payment|not found|no such model|invalid model|permission denied",
    re.I,
)
_NOISE = re.compile(r"^(reconnecting|session id:|workdir:|model:|provider:|approval:|sandbox:|reasoning )", re.I)
_TIMESTAMP = re.compile(r"^\d{4}-\d{2}-\d{2}T[\d:.]+Z?\s*", re.M)

# What to tell someone whose CLI is installed but not usable yet.
LOGIN_HINTS = {
    "claude": "run `claude` once and sign in, or set ANTHROPIC_API_KEY",
    "codex": "run `codex login` (ChatGPT plan), or set OPENAI_API_KEY",
    "gemini": "run `gemini` once and sign in, or set GEMINI_API_KEY",
    "opencode": "run `opencode auth login`",
}


def _last_meaningful_lines(text: str, limit: int = 4) -> str:
    """Pull the actual reason out of a CLI failure.

    Agent CLIs surround the one useful line with a configuration banner, an
    echo of the prompt and a wall of retry messages, so neither the head nor
    the tail of their output is reliably the answer. Lines that name a cause
    win; otherwise the tail, minus the noise.
    """
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    if not lines:
        return ""
    signal = [line for line in lines if _SIGNAL.search(line)]
    if signal:
        # Deduplicate: a retry loop repeats one reason many times, and the
        # lines differ only by their timestamp, so that has to come off first.
        seen, unique = set(), []
        for line in signal:
            key = _TIMESTAMP.sub("", line)[:120]
            if key not in seen:
                seen.add(key)
                unique.append(_TIMESTAMP.sub("", line).strip())
        return "\n".join(unique[:limit])[:600]
    quiet = [line for line in lines if not _NOISE.match(line)]
    return "\n".join((quiet or lines)[-limit:])[:600]


def _with_hint(command: str, detail: str) -> str:
    """Append the sign-in instruction when the failure looks like an auth one."""
    hint = LOGIN_HINTS.get(command)
    if hint and _SIGNAL.search(detail or ""):
        return f"{detail}\n\nhint: {hint}"
    return detail


def available_clis() -> list[CLISpec]:
    """Every agent CLI that is actually installed, in preference order."""
    return [spec for spec in CLI_SPECS.values() if shutil.which(spec.command)]


__all__ = ["CLI_SPECS", "PROTOCOL", "CLIAgentProvider", "CLISpec", "available_clis"]
