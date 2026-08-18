"""Hand a coding job to a coding agent, and keep the desktop for yourself.

LAI can already write files and run commands, so it *can* build software —
but its tools are aimed at a desktop, and it pays for every line of code in
agent steps. An installed coding CLI is a specialist at exactly the part LAI
is worst at, and most people who run LAI already have one signed in.

So this is the division of labour that plays to both sides: the coding agent
writes and edits inside one directory, and LAI does the thing no coding agent
can do — open the result on a real screen, look at it, press its buttons, and
say whether it actually works. Architect and orchestrator here, coder there.

What keeps it safe rather than reckless:

* **One directory.** The workspace is explicit and the CLI is confined to it.
  There is no "just fix my whole home directory" mode.
* **The gate still applies.** This is a destructive tool, so it goes through
  the same permission modes as `shell_exec` — in `ask` mode a human approves
  the job before a single file is touched.
* **The result is evidence, not a claim.** What comes back is the agent's
  message *plus* the files that actually changed on disk, so the model
  verifies against the filesystem rather than trusting a summary.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path

from ..safety.policy import Risk
from .base import ToolContext, ToolRegistry, ToolResult

DEFAULT_TIMEOUT = 900.0
MAX_TIMEOUT = 3600.0
MAX_OUTPUT_CHARS = 6_000
MAX_LISTED_FILES = 40

# How to run each CLI as a *worker* that may edit files in one directory —
# which is a different invocation from using it as a model (see
# agent/providers/cli_agent.py, where the same binaries answer questions).
CODERS: dict[str, tuple[str, ...]] = {
    "claude": ("claude", "-p", "{task}", "--permission-mode", "acceptEdits", "--add-dir", "{workspace}"),
    "codex": ("codex", "exec", "--skip-git-repo-check", "--full-auto", "-C", "{workspace}", "{task}"),
    "gemini": ("gemini", "-y", "-i", "{task}"),
    "opencode": ("opencode", "run", "{task}"),
}
PREFERENCE = ("claude", "codex", "gemini", "opencode")


def available_coders() -> list[str]:
    """Which coding CLIs this machine actually has."""
    return [name for name in PREFERENCE if shutil.which(CODERS[name][0])]


def register(registry: ToolRegistry) -> None:
    @registry.tool(
        "code_agent",
        "Hand a self-contained programming job to an installed coding agent (Claude Code, "
        "Codex, Gemini CLI, opencode) working inside one directory, and get back what it "
        "did plus the files that actually changed. Use this for anything that is mostly "
        "writing or editing code — building a page or a script, fixing a failing test, "
        "refactoring — because it is far faster and better at that than doing it yourself "
        "one file_write at a time. Then do the part it cannot: open the result on this "
        "desktop, look at it, click it, and confirm it really works. Describe the job "
        "completely, including how to tell it succeeded — the coding agent cannot see your "
        "screen and cannot ask you anything.",
        {
            "properties": {
                "task": {
                    "type": "string",
                    "description": "The complete job, as you would brief a competent "
                                   "colleague who cannot ask questions.",
                },
                "workspace": {
                    "type": "string",
                    "description": "Directory it may read and write. Created if absent.",
                },
                "coder": {
                    "type": "string",
                    "description": f"Which CLI to use: {', '.join(PREFERENCE)}. Default: whichever is installed.",
                },
                "timeout": {
                    "type": "number",
                    "description": f"Seconds to allow (default {DEFAULT_TIMEOUT:.0f}, max {MAX_TIMEOUT:.0f}).",
                },
            },
            "required": ["task", "workspace"],
        },
        risk=Risk.DESTRUCTIVE,
        group="agentic",
    )
    def code_agent(ctx: ToolContext, args: dict) -> ToolResult:
        task = str(args.get("task") or "").strip()
        if not task:
            return ToolResult.failure("task is required — say what should be built or changed")

        try:
            workspace = _prepare_workspace(args.get("workspace"))
        except (OSError, ValueError) as exc:
            return ToolResult.failure(str(exc))

        installed = available_coders()
        if not installed:
            return ToolResult.failure(
                "no coding agent is installed",
                hint="install one of: " + ", ".join(PREFERENCE)
                + " — or write the files yourself with file_write",
            )
        coder = str(args.get("coder") or "").strip().lower() or installed[0]
        if coder not in CODERS:
            return ToolResult.failure(f"unknown coder {coder!r}", hint=f"known: {', '.join(PREFERENCE)}")
        if coder not in installed:
            return ToolResult.failure(
                f"{coder} is not installed on this machine",
                hint="available here: " + ", ".join(installed),
            )

        timeout = min(float(args.get("timeout") or DEFAULT_TIMEOUT), MAX_TIMEOUT)
        before = _snapshot_files(workspace)
        started = time.monotonic()

        argv = [
            part.replace("{task}", task).replace("{workspace}", str(workspace))
            for part in CODERS[coder]
        ]
        try:
            completed = subprocess.run(
                argv, cwd=str(workspace), capture_output=True, text=True,
                timeout=timeout, check=False,
            )
        except subprocess.TimeoutExpired:
            return ToolResult.failure(
                f"{coder} did not finish within {timeout:.0f}s",
                hint="break the job into smaller pieces, or raise `timeout`",
            )
        except OSError as exc:
            return ToolResult.failure(f"could not run {coder}: {exc}")

        elapsed = time.monotonic() - started
        changed = _changed_files(workspace, before)
        output = _tail((completed.stdout or "") + ("\n" + completed.stderr if completed.stderr else ""))

        if completed.returncode != 0 and not changed:
            return ToolResult.failure(
                f"{coder} exited {completed.returncode} and changed nothing",
                detail=output or "no output",
                hint="it may not be signed in — try `lai models test cli:" + coder + "`",
            )

        summary = [
            f"{coder} worked for {elapsed:.0f}s in {workspace}.",
            "",
            _describe_changes(changed),
            "",
            "What it reported:",
            output or "(no output)",
            "",
            "This is its own account. Verify it: open the result on screen and check.",
        ]
        return ToolResult.text(
            "\n".join(summary),
            coder=coder,
            workspace=str(workspace),
            changed=[str(path) for path in changed[:MAX_LISTED_FILES]],
            changed_count=len(changed),
            elapsed=round(elapsed, 1),
            exit_code=completed.returncode,
        )


# -- internals -----------------------------------------------------------


def _prepare_workspace(raw) -> Path:
    if not raw or not str(raw).strip():
        raise ValueError("workspace is required — name the directory it may write in")
    workspace = Path(str(raw)).expanduser()
    if not workspace.is_absolute():
        workspace = (Path.cwd() / workspace).resolve()
    if workspace.exists() and not workspace.is_dir():
        raise ValueError(f"{workspace} is a file, not a directory")
    workspace.mkdir(parents=True, exist_ok=True)
    return workspace


def _snapshot_files(workspace: Path) -> dict:
    """Path → (mtime, size), so 'what changed' is measured rather than claimed."""
    found: dict = {}
    for path in _walk(workspace):
        try:
            stat = path.stat()
        except OSError:
            continue
        found[path] = (stat.st_mtime, stat.st_size)
    return found


def _changed_files(workspace: Path, before: dict) -> list[Path]:
    after = _snapshot_files(workspace)
    changed = [path for path, stamp in after.items() if before.get(path) != stamp]
    changed += [path for path in before if path not in after]
    return sorted(set(changed))


def _walk(workspace: Path):
    skip = {".git", "node_modules", "__pycache__", ".venv", "venv", "dist", "build", ".next"}
    for root, dirnames, filenames in os.walk(workspace):
        dirnames[:] = [d for d in dirnames if d not in skip]
        for name in filenames:
            yield Path(root) / name


def _describe_changes(changed: list[Path]) -> str:
    if not changed:
        return "No files changed on disk — whatever it says, nothing was written."
    listing = "\n".join(f"  {path}" for path in changed[:MAX_LISTED_FILES])
    more = f"\n  … and {len(changed) - MAX_LISTED_FILES} more" if len(changed) > MAX_LISTED_FILES else ""
    return f"{len(changed)} file(s) changed on disk:\n{listing}{more}"


def _tail(text: str) -> str:
    text = (text or "").strip()
    if len(text) <= MAX_OUTPUT_CHARS:
        return text
    return "[…earlier output trimmed…]\n" + text[-MAX_OUTPUT_CHARS:]
