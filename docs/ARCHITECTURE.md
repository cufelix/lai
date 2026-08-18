# LAI — Architecture

## The problem

An agent driving a desktop has one hard problem: **grounding**. Given a
screenshot, a model must decide *what is on screen* and *where to click*. Pure
pixel agents guess, and they guess wrong often enough that multi-step tasks
collapse — each blind click compounds the error from the last.

LAI's answer is to stop guessing. Linux already publishes a machine-readable
description of every widget on screen through **AT-SPI2**, the accessibility
bus that screen readers use. That is a DOM for the desktop. LAI reads it.

## Layers

```
┌─────────────────────────────────────────────────────────────┐
│  Interfaces   cli · repl · daemon(HTTP+SSE) · MCP server     │
├─────────────────────────────────────────────────────────────┤
│  Agent        loop · providers · session · prompt            │
├─────────────────────────────────────────────────────────────┤
│  Tools        36 specs → one dispatch path → safety gate     │
├─────────────────────────────────────────────────────────────┤
│  OS layer     a11y · windows · screen · input · apps · clip  │
├─────────────────────────────────────────────────────────────┤
│  X11 / AT-SPI / freedesktop                                  │
└─────────────────────────────────────────────────────────────┘
```

Each layer depends only on the one below it. The OS layer knows nothing about
models; the agent knows nothing about Xlib.

## Perception: four sources, fused

| Source | Module | What it answers |
|---|---|---|
| AT-SPI tree | `osl/a11y.py` | *What widgets exist, named, with exact bounds?* |
| Window tree | `osl/windows.py` | *What is open, where, which is focused?* |
| Screen | `osl/screen.py` | *What does it actually look like?* |
| Processes | `osl/apps.py` | *What is running; did my launch work?* |

`Desktop.observe()` fuses them into one `Observation`, rendered for the model as
a text summary plus an optional screenshot.

Three details that matter more than they look:

- **Frame extents.** GTK client-side-decorated windows include an invisible
  shadow in their X geometry. Untrimmed, a calculator reports
  `(-26, -23, 412, 538)` — partly off-screen. LAI subtracts
  `_GTK_FRAME_EXTENTS` and reports the visible `(0, 0, 360, 486)`.
- **Coordinate mapping.** A `Screenshot` carries its region and scale, so a
  point in image space converts back to screen space exactly. The model is told
  the formula in the tool result.
- **Perception never raises.** Every observation call is wrapped: an app that
  dies mid-walk yields a partial snapshot, not a crashed run.

## Actuation: semantic first

```
ui_click(name="Save")
   → find the element in the a11y tree
   → invoke its accessible action        ← deterministic
   → if it has none, click its centre    ← fallback
computer_click(x, y)
   → raw XTEST event                     ← last resort
```

The same ordering applies to text: `ui_type` sets the value through the
`EditableText` interface; only if that fails does it focus, select-all and
simulate keystrokes.

## The dispatch choke point

Every tool call — from the loop, from the daemon, from an MCP client — goes
through `ToolRegistry.call()`. That single function does:

1. resolve the tool (unknown → structured error, not an exception)
2. **policy check** → allow / ask / deny
3. approval callback if the verdict is *ask*
4. schema validation
5. audit `tool_call`
6. run the handler, containing **every** exception into a failed `ToolResult`
7. rate-limit accounting, output truncation, audit `tool_result`

Because there is exactly one path, no tool can accidentally skip the safety
gate — and adding a tool cannot introduce a new way to crash the loop.

## The loop

```
observe → act → verify, repeated under explicit budgets

for step in 1..max_steps:
    check interrupt, wall-clock and token budgets
    compact the transcript if it is getting heavy
    ask the model (streaming, with the full tool schema)
    if it called tools:  run them, append results, continue
    if it called task_complete / task_blocked:  stop, report
    if it produced bare text:  nudge once, then accept it
```

Termination is explicit. `task_complete` requires a summary and asks for a
verification statement; `task_blocked` requires a reason. Budget exhaustion,
repeated provider failures and interrupts are reported as themselves — a run
that ran out of steps never reports success.

**Context economy.** Screenshots dominate a desktop transcript. Only the newest
few survive; older ones become placeholders. When the transcript still grows
past threshold, the model writes a handoff summary and the history is replaced
— with the cut point moved back if it would orphan a `tool_result` from its
`tool_use`.

## Providers

One neutral message format (`agent/providers/base.py`), several wire protocols:

- `anthropic_api.py` — the Messages API over raw httpx. Because it is written
  to the protocol rather than an SDK, the *same class* serves api.anthropic.com,
  z.ai/GLM and any compatible gateway; only the base URL and auth change.
- `openai_api.py` — chat-completions with function tools: OpenAI, OpenRouter,
  Ollama.

Credential discovery (`providers/registry.py`) checks env vars, then a
compatible-gateway pair, then a local `glm`-style wrapper script, then a running
Ollama — so `lai` works on a machine that already has *any* of them configured.

## Extensibility

**Skills** are Claude Code's format, unchanged: a directory with `SKILL.md` and
frontmatter. Progressive disclosure keeps them cheap — names and descriptions go
in the system prompt, bodies load on demand via `skill_load`. They can be
installed at runtime from git, GitHub shorthand, an archive URL or a path;
archives are path-checked and size-capped before extraction.

**MCP** works both ways:

- *Client* — reads `mcp.json` / `.mcp.json` / Claude Code's config, connects to
  each server, and registers its tools as `mcp__<server>__<tool>` with a
  conservatively-guessed risk level.
- *Server* — `lai mcp` publishes the whole desktop toolset over stdio, which is
  how Claude Code gains native OS control.

The MCP SDK is async and LAI is synchronous. One background event loop bridges
them, and each server is owned by a **single long-lived task** — because anyio
cancel scopes must be exited by the task that entered them, so enter/exit from
different tasks would deadlock.

## Safety

Four modes (`readonly` / `ask` / `auto` / `yolo`) scale what runs unattended.
Independent of mode, some things are always refused:

- destructive shell patterns (`rm -rf`, `mkfs`, `dd if=`, `curl … | sh`, …)
- input directed at a password manager or an authentication prompt

Plus: per-tool allow/deny lists, an actions-per-minute rate limit, a dry-run
mode, secret redaction on everything logged or transmitted, and an append-only
JSONL audit trail of every call and verdict.

The daemon binds loopback only, requires a bearer token stored `0600`, and
refuses a non-loopback bind without an explicit opt-in.

## Testing

685 tests, 82% line coverage.

Pure logic is tested directly. Anything touching the display is marked `x11` and
only ever drives a window the test launched itself, terminating it in fixture
teardown. The MCP integration test spawns a real `lai mcp` subprocess and speaks
the protocol to it — the same path Claude Code uses.

```bash
pytest tests/ -q                              # everything
pytest tests/ -q -m "not slow and not x11"    # no display needed
```

## Deliberate limitations

- **X11 only.** The backend boundary exists; the Wayland implementation
  (portal capture + libei input) does not.
- **Chromium/Electron publish no a11y tree** without
  `--force-renderer-accessibility`. LAI degrades to pixels there, which works
  but is less precise.
- **No OCR**, so text in canvases is only as readable as the model's vision.
