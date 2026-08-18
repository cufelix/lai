# LAI — Feature Catalog

**LAI (Local Agent Interface)** — a native OS-level autonomous agent for Linux.
Think: *Claude Chrome extension, but the "page" is your entire desktop*, driven by an
OpenClaw-style persistent autonomous loop, able to consume any skill or MCP tool from the internet.

Legend: **[C]** = MVP core (build now) · **[1]** = phase 1 follow-up · **[2]** = later

---

## 1. Perception — "reading the desktop"

The Chrome extension reads the DOM. The OS equivalent is a *fused* view: pixels + accessibility tree + window tree + process tree.

| # | Feature | Notes | Tier |
|---|---------|-------|------|
| 1.1 | Full-screen screenshot | X11 capture, multi-monitor aware | **C** |
| 1.2 | Per-window / per-region screenshot | crop to focused window | **C** |
| 1.3 | Image downscaling + token budgeting | resize to model-optimal dims, report scale factor | **C** |
| 1.4 | Window tree enumeration | id, title, class, pid, geometry, state, desktop | **C** |
| 1.5 | Focused window introspection | what the user is actually looking at | **C** |
| 1.6 | AT-SPI accessibility tree | the desktop "DOM": roles, names, states, bounds | **C** |
| 1.7 | A11y element search | by name/role/regex → clickable coordinates | **C** |
| 1.8 | Interactive-element indexing | numbered overlay refs (`ref=12`) like browser a11y snapshots | **C** |
| 1.9 | Process/app inventory | running apps, resource use | **C** |
| 1.10 | Clipboard read | text + image | **C** |
| 1.11 | OCR fallback | for apps with no a11y (Electron, games, VMs) | 1 |
| 1.12 | Visual diffing | detect "did the screen change after my click?" | 1 |
| 1.13 | Screen recording | video/GIF of an agent run | 1 |
| 1.14 | Multi-monitor geometry model | per-output coordinate mapping | **C** |
| 1.15 | Notification capture | listen to DBus notifications | 2 |
| 1.16 | Audio in/out awareness | what's playing, mic state | 2 |
| 1.17 | Wayland backend | portal-based capture + libei input | 2 |
| 1.18 | Idle/user-presence detection | pause when human takes over | 1 |

## 2. Actuation — "driving the desktop"

| # | Feature | Notes | Tier |
|---|---------|-------|------|
| 2.1 | Mouse move / click / double / right / middle | absolute coords | **C** |
| 2.2 | Drag & drop | press-move-release | **C** |
| 2.3 | Scroll (v/h) | wheel buttons | **C** |
| 2.4 | Type text | unicode-safe | **C** |
| 2.5 | Key / chord / sequence | `ctrl+alt+t`, `Return` | **C** |
| 2.6 | Semantic click | "click the *Save* button" via a11y, no pixels | **C** |
| 2.7 | Semantic set-value | fill a text field via a11y | **C** |
| 2.8 | Window focus / raise / close | **C** |
| 2.9 | Window move / resize / maximize / tile | **C** |
| 2.10 | Workspace switching | virtual desktops | 1 |
| 2.11 | App launch by name | `.desktop` index, fuzzy match | **C** |
| 2.12 | App launch + wait-for-window | the critical primitive: launch and know it's ready | **C** |
| 2.13 | App terminate | graceful then force | **C** |
| 2.14 | Clipboard write | **C** |
| 2.15 | Shell exec (sandboxed, gated) | with timeout, cwd, streaming | **C** |
| 2.16 | File read/write/list/glob | **C** |
| 2.17 | Browser bridge | reuse Chrome/Firefox via CDP for real web work | 1 |
| 2.18 | Terminal-app driving | spawn a PTY, drive TUI apps | 1 |
| 2.19 | dbus method calls | control apps properly (mpris, etc.) | 1 |
| 2.20 | Global hotkey to summon agent | 1 |
| 2.21 | Human-handoff / takeover pause | 1 |

## 3. Agent runtime — "the OpenClaw loop"

| # | Feature | Notes | Tier |
|---|---------|-------|------|
| 3.1 | Tool-calling agentic loop | multi-turn, parallel tool calls | **C** |
| 3.2 | Streaming responses | **C** |
| 3.3 | Provider abstraction | swap models freely | **C** |
| 3.4 | Provider: Anthropic API | **C** |
| 3.5 | Provider: GLM / z.ai (Anthropic-compatible) | **C** |
| 3.6 | Provider: OpenAI-compatible | **C** |
| 3.7 | Provider: local `claude` CLI (subscription auth, no key) | **C** |
| 3.8 | Provider: Ollama (local, offline) | **C** |
| 3.9 | Model routing (cheap ↔ smart per step) | 1 |
| 3.10 | Session persistence + resume | JSONL transcripts | **C** |
| 3.11 | Context compaction | summarize when near window | **C** |
| 3.12 | Vision message support | screenshots into the loop | **C** |
| 3.13 | Autonomous goal loop | plan → act → verify → repeat until done | **C** |
| 3.14 | Self-verification step | "did that actually work?" before claiming done | **C** |
| 3.15 | Retry / backoff / error recovery | **C** |
| 3.16 | Subagent spawning | delegate isolated subtasks | 1 |
| 3.17 | Cost + token accounting | **C** |
| 3.18 | Interruptibility (ESC / signal) | **C** |
| 3.19 | Scheduled/cron tasks | 1 |
| 3.20 | Event triggers (file, window, notification) | 2 |
| 3.21 | Long-term memory store | 1 |
| 3.22 | Prompt caching | 1 |

## 4. Extensibility — "skills and tools from the internet"

| # | Feature | Notes | Tier |
|---|---------|-------|------|
| 4.1 | Skill discovery (local dirs) | `~/.lai/skills`, `~/.claude/skills`, project `.lai/skills` | **C** |
| 4.2 | SKILL.md frontmatter parsing | name/description, Claude-Code compatible | **C** |
| 4.3 | Progressive disclosure | inject names+descriptions; load body on demand | **C** |
| 4.4 | `skill` tool (load a skill mid-run) | **C** |
| 4.5 | Skill install from git URL | `lai skill install <git>` | **C** |
| 4.6 | Skill install from zip/http | **C** |
| 4.7 | Skill marketplace/registry index | 1 |
| 4.8 | Skill-bundled executable scripts | run helper scripts shipped in a skill | **C** |
| 4.9 | MCP client (stdio) | consume any MCP server → tools | **C** |
| 4.10 | MCP client (HTTP/SSE) | 1 |
| 4.11 | MCP server mode (expose LAI's desktop tools) | **so Claude Code itself gains desktop control** | **C** |
| 4.12 | Reuse Claude Code's `.mcp.json` config | **C** |
| 4.13 | Custom tool plugins (python entrypoints) | 1 |
| 4.14 | Slash-command / prompt library | 1 |

## 5. Safety & control

| # | Feature | Notes | Tier |
|---|---------|-------|------|
| 5.1 | Permission modes: `readonly`/`ask`/`auto`/`yolo` | **C** |
| 5.2 | Per-tool allow/deny policy | **C** |
| 5.3 | Dangerous-command detection | `rm -rf`, `dd`, `mkfs`, sudo, curl\|sh | **C** |
| 5.4 | Protected screen regions / apps | never touch password managers, banking | **C** |
| 5.5 | Secret redaction in transcripts + screenshots | **C** |
| 5.6 | Kill switch / panic stop | mouse-to-corner or hotkey | **C** |
| 5.7 | Dry-run mode | **C** |
| 5.8 | Full audit log of every action | **C** |
| 5.9 | Rate limiting of actions | **C** |
| 5.10 | Step budget + wall-clock budget | **C** |
| 5.11 | Confirmation for irreversible ops | **C** |
| 5.12 | Sandboxed/VM execution profile | 2 |

## 6. Interfaces

| # | Feature | Notes | Tier |
|---|---------|-------|------|
| 6.1 | CLI one-shot: `lai do "<task>"` | **C** |
| 6.2 | Interactive REPL | shipped; `lai chat` is the richer default | **C** |
| 6.3 | `lai doctor` env diagnostics | **C** |
| 6.4 | Daemon mode + HTTP API | **C** |
| 6.5 | WebSocket live event stream | **C** |
| 6.6 | MCP stdio server (`lai mcp`) | **C** |
| 6.7 | Web dashboard | shipped as `lai web` — chat, live screen, settings | 1 |
| 6.8 | Desktop overlay HUD (what agent sees/does) | 1 |
| 6.9 | Tray icon + global hotkey | 1 |
| 6.10 | Telegram / chat channel | 1 |
| 6.11 | Voice in/out | 2 |
| 6.12 | systemd user service | **C** |

## 7. Observability

| # | Feature | Notes | Tier |
|---|---------|-------|------|
| 7.1 | Structured JSONL event log | **C** |
| 7.2 | Screenshot trail per step | **C** |
| 7.3 | Run replay | 1 |
| 7.4 | Metrics (steps, tokens, cost, success) | **C** |
| 7.5 | Trace export (HTML report) | 1 |

---

## MVP core (what gets built and tested now)

Everything marked **[C]**. Concretely, the MVP is:

1. **`lai.osl`** — OS layer: X11 screenshot, input, windows, apps, AT-SPI a11y tree, clipboard, multi-monitor.
2. **`lai.tools`** — 25+ tools with JSON schemas: computer, ui (semantic), window, app, shell, files, skill, plus MCP-proxied tools.
3. **`lai.agent`** — streaming tool-calling loop, 5 providers, sessions, compaction, budgets, self-verification.
4. **`lai.skills`** — discovery + progressive disclosure + install-from-internet.
5. **`lai.mcp`** — MCP client (consume) **and** MCP server (expose desktop to Claude Code).
6. **`lai.safety`** — permission modes, danger detection, protected apps, redaction, audit log, kill switch.
7. **`lai.cli` / `lai.daemon`** — `do`, `repl`, `doctor`, `mcp`, `serve`, `skill`.
8. **Tests** — unit + integration against a real X11 display, ≥80% on pure logic.

### Why this is the right core
The single hardest thing about a desktop agent is *reliable perception and grounding*. Pixel-only agents (pure screenshot + coordinate guessing) are brittle. LAI's differentiator is **fusing the AT-SPI accessibility tree with screenshots** — the agent gets a real element list with roles, names and exact bounds (like a DOM), and falls back to pixels only when a11y is unavailable. That's what makes "open a program and use it" actually work.

---

## Status — what shipped in the MVP

Every **[C]** item above is implemented, wired into the CLI/daemon/MCP surfaces, and covered by tests.

| Subsystem | Module | Shipped |
|---|---|---|
| Perception | `osl/a11y.py` `osl/windows.py` `osl/screen.py` | AT-SPI tree with refs/roles/values/bounds; EWMH window tree with GTK frame-extent correction; multi-monitor capture with exact coordinate mapping; screen-settle detection |
| Actuation | `osl/inputs.py` `osl/desktop.py` | click/drag/scroll/type/key with validation; semantic click & set-value via accessible actions; pixel fallback; window focus/move/resize/state |
| Apps | `osl/apps.py` | 157-entry `.desktop` index, fuzzy match, launch-and-wait-for-window, terminate |
| Tools | `tools/` | 36 tools at MVP, 53 after phase 2; JSON schemas, one dispatch path |
| Agent | `agent/` | streaming tool-calling loop, 4 provider families, sessions, image pruning, safe compaction, explicit termination, budgets, interrupts |
| Skills | `skills/` | Claude-Code-compatible discovery, progressive disclosure, install from git/GitHub/archive/path |
| MCP | `mcp/` | client (consume any server) **and** server (`lai mcp` exposes the desktop) |
| Safety | `safety/` | 4 modes, always-on hard denials, protected windows, rate limit, dry-run, redaction, audit log |
| Interfaces | `cli.py` `daemon/` | `do` `repl` `doctor` `observe` `tools` `skills` `sessions` `serve` `mcp`; HTTP + SSE daemon with bearer auth |

### Defects found and fixed while building

Each of these was caught by a test or an end-to-end run, not by inspection:

1. **`WindowManager.close` defined twice** — the window-closing overload shadowed the connection-closing one, so `Desktop.close()` raised `TypeError`. Renamed to `close_window`.
2. **`ToolResult.failure(message=…)` collision** — callers splat `LaiError.to_dict()`, which carries a `message` key, so every error path raised `TypeError` instead of reporting the error. Parameter renamed to `content`.
3. **Interrupt lost before a run** — `Agent.run()` cleared the stop flag on entry, so a `/stop` racing `/task` was silently dropped. Now cleared on exit.
4. **`DISPLAY` stripped by MCP clients** — the MCP SDK passes a restricted environment, so every OS call failed under Claude Code. Added X-session discovery (`osl/session.py`) plus session passthrough for spawned servers.
5. **GTK CSD shadow inflating window bounds** — a calculator reported `(-26,-23,412,538)`, partly off-screen. Now trimmed via `_GTK_FRAME_EXTENTS` to the visible `(0,0,360,486)`.
6. **SSE responses hung clients** — the daemon advertised `keep-alive` with no `Content-Length`, so a client never saw the stream end. Now `Connection: close`.
7. **`xdotool mousemove --sync` hangs** — blocks until the 10 s subprocess timeout when the pointer is already at the target. Replaced with a bounded position poll; this was also causing intermittent `ui_click` failures via the pointer fallback.
8. **`lai.skills.install` ambiguity** — the package re-exported a function shadowing the submodule of the same name. Exported as `install_skill`.

### Next up (phase 1)

OCR fallback for a11y-less apps · browser bridge via CDP · subagent spawning ·
model routing · scheduled/triggered runs · overlay HUD · Wayland backend.

---

## Phase 2 — shipped

Everything previously marked **[1]** that is now implemented, plus the connector
layer, which was originally 6.10.

| Area | Delivered |
|---|---|
| 1.11 OCR fallback | `osl/ocr.py` — tesseract with per-word screen coordinates, so the agent can click a word it read |
| 1.13 Screen recording | `osl/recorder.py` — ffmpeg x11grab, graceful stop, two-pass GIF export |
| 1.15 Notification capture | `osl/notifications.py` — DBus eavesdrop monitor **and** outbound notifications |
| 1.18 Idle / presence | `osl/idle.py` — MIT-SCREEN-SAVER; the agent can check whether the human is present before taking the pointer |
| 2.10 Workspace switching | `osl/windows.py` — count, names, switch, move-window-to-workspace |
| 3.9 Model routing | `agent/router.py` — data-driven escalation rules, per-model usage accounting |
| 3.16 Subagent spawning | `agent/subagent.py` — isolated context, own budget, depth-capped, shares the safety gate |
| 3.19 Scheduled tasks | `scheduler.py` — hand-written cron parser + `every:<n>`, JSON task store, daemon runner |
| 3.21 Long-term memory | `agent/memory.py` — SQLite + FTS5 (LIKE fallback), upsert-by-key, prompt context blocks |
| 6.2 Interface | `tui/` — full-screen Textual UI: live feed, plan, desktop panel, inline approval modal |
| 6.10 Connectors | `channels/` — Telegram, Discord, webhook, local; pairing, allowlist, remote approvals |
| 6.1 CLI | `lai schedule list/add/remove/enable/disable/run` — the scheduler was reachable only from the agent and the daemon before |

### Connector security model

A connector is a remote control for a desktop, so the token is explicitly *not*
the security boundary:

- unknown senders get a flat refusal that does not reveal pairing exists
- pairing needs a 6-digit code minted on the operator's own terminal:
  single-use, 15-minute expiry, 5 attempts, constant-time compare
- the first principal to pair becomes admin; the allowlist is stored `0600`
- `ask`-mode approvals are delivered to the chat and block the run until
  answered, so remote use does not require `yolo`

### Defects found and fixed in phase 2

9. **Circular import** — `tools/agentic` → `agent.subagent` → `agent.loop` → `tools.base` → `tools/__init__`. Fixed by loading the optional tool families lazily inside `build_registry`, which also made them genuinely optional rather than import-time-mandatory.
10. **`ToolResult.text(content=…)` collision** — the same trap as defect 2, hit again by splatting a dict carrying a `content` key. Caught during implementation.
11. **Three entry points raced for one desktop** — `/task` checked "am I busy?" and then set the flag in a separate step, and channel runs did not touch that flag at all, so a chat message and an HTTP task could drive the mouse simultaneously. Replaced with an atomic `claim → engage → release` gate on `DaemonState` that all three paths (HTTP, channels, scheduler) share; `/stop` arriving during agent construction is now remembered and abandons the run instead of being lost. Found by a live end-to-end run, not by a test.

### Still open

Browser bridge via CDP · terminal/PTY driving · dbus app control · global hotkey ·
tray icon · web dashboard · run replay · voice · Wayland backend · sandbox profile.

---

## Phase 3 — onboarding

The shortest honest description of the goal: *one command from a bare machine
to an agent doing something on your desktop.*

| Piece | Delivered |
|---|---|
| `checks.py` | Every environment check carries the repair for it — one source of truth shared by `doctor` and `setup`, so a diagnosis cannot drift from its fix |
| `setup_wizard.py` | `lai setup`: probe → repair → backend → permission mode → a real first task. Nothing is written that has not been verified |
| `config_file.py` | Annotated `config.toml` writer: atomic, `0600`, only the settings actually chosen |
| `lai doctor --fix` | Applies every automatic repair; lists exact commands for the rest |
| Bare `lai` | Wizard if there is no model backend, the full-screen interface otherwise |
| `packaging/install.sh` | Multi-distro: packages, venv, accessibility, `~/.local/bin/lai`, then hands over to the wizard |
| TUI first screen | Concrete example tasks instead of an empty prompt; a missing backend names `lai setup` |

Design decisions worth stating:

- **The key is spent before it is saved.** A pasted API key is verified with a
  real 16-token request. A typo fails in the wizard, not on the first task.
- **`--yes` means "take the default", not "say yes to everything".** A question
  whose safe answer is no (switch backends, really use `yolo`) does not flip
  just because the run is unattended.
- **sudo is never run without a terminal.** It would block on a password prompt
  nobody can answer, so those fixes are reported instead.
- **The demo runs `readonly`.** The first thing a new user sees must not change
  their machine.

## Phase 4 — every route to a model

"Connect whatever you already have" turned out to mean three different things,
so there are three mechanisms rather than one long list of vendors.

| Piece | Delivered |
|---|---|
| `providers/cli_agent.py` | Claude Code, Codex, Gemini CLI and opencode used *as the model*, via a JSON tool-calling protocol. No API key: it borrows their login |
| `providers/catalog.py` | 20 OpenAI-compatible endpoints as data — 14 hosted (Groq, DeepSeek, Mistral, xAI, Together, Fireworks, Cerebras, Perplexity, Nebius, Moonshot, Qwen, Gemini, OpenAI, OpenRouter) and 6 local (Ollama, LM Studio, llama.cpp, vLLM, LiteLLM, Jan) |
| `models.py` + `lai models` | One listing of what is ready, what needs a sign-in, and what is merely known; `test` proves one works, `use` makes it the default |
| Wizard | The backend menu is built from that discovery, so an already-signed-in CLI is a first-class choice next to the API vendors |

### Using a coding CLI as the model

This is the piece worth explaining, because it is a compromise rather than a
clean win. Those CLIs return prose; the loop needs tool calls. So the provider
renders the tool schemas and the transcript into a prompt and asks for a single
JSON object naming the tools to run — function calling, emulated.

What that costs, stated where someone will find it before they are surprised:

- **No vision.** LAI works from the accessibility tree alone on this route.
- **Whole transcript per turn**, because the CLIs are stateless per invocation.
- **No token accounting** — a CLI reports cost, not tokens. Reporting zero is
  honest; inventing an estimate is not.
- **Format drift is possible.** A model answering in prose gets one corrective
  retry, then its prose is treated as the final answer — never as tool calls,
  because inventing a desktop action from ambiguous text is the one failure
  mode that could actually break something.

It is the answer to "I have no API key", not the recommended path. Verified
end-to-end: a full observe→act→verify run through `claude -p`, two steps,
sixteen seconds, no key anywhere.

## Phase 5 — comfort

Three complaints, one root: LAI was a set of commands rather than something you
*sit in*. Phase 5 answers them.

**A backend running out no longer ends the run.** Quotas are hit mid-task; that
is when it hurts most. `provider.fallback` is an ordered chain — hosted keys,
then signed-in coding CLIs, then local models — that is lazy (a standby is only
built when it is reached), sticky (no flapping mid-task) and narrow about what
counts: quota, auth and outages move on, a malformed request is raised, because
it would fail identically everywhere. The switch is announced in every
interface and written to the audit log. On by default; `LAI_FALLBACK=off`
disables it.

**`lai` is now a conversation.** The chat interface is the default command:
prompt_toolkit history and completion when it is installed, plain input when it
is not, and slash commands for everything you would otherwise quit for —
`/model` (with a menu of what actually works), `/fallback`, `/mode`, `/status`,
`/doctor`, `/observe`, `/new`. Changes persist to `config.toml`, so the choice
survives the session.

**`lai web` puts it in a browser.** One self-contained page served by the
daemon — no CDN, no bundler, `default-src 'none'` — streaming the same SSE
events the terminal renders, next to a live view of the actual screen. Settings
switch backend and permission mode through the same code path the chat uses, so
a choice made in a tab and a choice made in a terminal end up in the same file.

### Security of the browser interface

The daemon can drive the whole desktop, so the page is deliberately powerless
on its own:

- The token reaches the page in the URL **fragment**, which browsers never send
  to a server — it cannot land in an access log or a proxy. From there it moves
  to `sessionStorage`, scoped to the tab and dropped when the tab closes.
- `GET /` needs no token and contains none; every endpoint that reads the
  desktop or changes state demands one.
- Only `/screen` accepts its token in the query string, because an `<img>` tag
  cannot send a header. That exemption is tested to apply to nothing else.
- No CORS headers are sent, so another origin cannot read a response; the token
  is not a cookie, so nothing is attached automatically to a forged request.
- Tool output is written with `textContent` everywhere. A window title is
  untrusted input, and this page will never be the thing that executes it.

## Phase 6 — it gets better at this machine

The gap phase 6 closes: LAI had memory tools and never used them on its own, so
every run rediscovered the same desktop. Which launcher entry opens the editor,
where a canvas starts below a toolbar, what a save dialog calls its field — all
worked out, then thrown away.

**The journal.** `~/.lai/notes/*.md`, one file per topic, with the same
frontmatter shape skills use. Markdown rather than a database on purpose: an
agent's beliefs about someone's machine must be readable, correctable and
deletable by that someone. A note that cannot be audited will quietly mislead
every future run.

**Reading.** Notes relevant to the task are injected into the system prompt,
under a heading that tells the model to treat them as starting points and to
say so when one turns out to be wrong. Bounded to a few thousand characters so
the journal can never crowd out the task.

**Writing.** After a run that did more than one thing, the model is shown a
compressed trace — the calls it made, what came back, what failed — and asked
what is worth remembering about *this machine*. The answer is merged, not
appended: the same lesson learned three times stays one line, because a journal
that grows a copy per run is worse than none. `{"notes": []}` is an explicitly
good answer, and prose that ignores the format is never filed as fact.

Three rules keep it from becoming a liability: never fabricate (only what the
trace shows), never block (reflection runs after the result exists and swallows
its own failures), never grow without bound (one call, capped trace, four
lessons maximum).

**Pages, everywhere.** The same journal is a CLI command (`lai notes
list|show|edit|add|rm`), a set of chat commands (`/notes`, `/edit` in `$EDITOR`,
`/learn`, `/forget`, `/learning on|off`), and a page in the browser with a real
markdown editor. `/settings` and the browser's Settings page show the same
state and write the same `config.toml`.

---

### Defects found and fixed in phase 5

18. **A large transcript killed the CLI backends** — the prompt was passed as a single argv entry, and Linux caps one at 128 KiB (`MAX_ARG_STRLEN`). With MCP servers connected the prompt sailed past it and `execve` failed with `E2BIG`, so a perfectly working `claude` reported "could not run claude: Argument list too long" three times and ended the run. Found live, in the browser, on the first real task. Prompts over 96 KB now go down stdin; a CLI that cannot read stdin gets a prompt truncated to fit instead.
19. **`lai serve` and `lai web` ignored `--no-mcp`** — `serve()` built its own runtime with MCP always on, which is both slow and how defect 18 was reached in the first place.
20. **The browser rendered every finished run twice** — the daemon ends a stream with `done` while the blocking API calls the same thing `result`; the page handled both.
21. **A reload logged the browser out** — the token was read from the fragment and then erased from the URL, so refreshing the page left it with no credentials at all. It is kept in `sessionStorage` now.
22. **`favicon.ico` answered 401** — browsers ask unprompted, and there is nothing secret about not having one; it is a 204 now.

---

### Defects found and fixed in phase 4

16. **A failed CLI invocation was returned as the model's answer** — `codex` exits non-zero on an auth failure but still prints its configuration banner, and the first implementation handed that banner to the loop as if the model had said it. Now a non-zero exit raises, and the reason is pulled out of the retry storm rather than shown as the first 500 characters of a config dump.
17. **`3.7 Provider: local claude CLI` was marked shipped and was not** — the MVP table claimed subscription-auth support that no code provided. It exists now; the claim was the defect.

---

### Defects found and fixed in phase 3

12. **`lai doctor` took 22 seconds** — it built a full runtime including every configured MCP server, so the command you run *because* something is already wrong made you wait through 207 external tool connections. MCP is now opt-in via `--mcp`; the environment check itself takes 0.4 s.
13. **A saved API key made `lai` re-run the wizard forever** — `needs_setup()` asked `discover_credentials()`, which reads only the environment, so a key written to `config.toml` by a successful setup was invisible to it. Now both sources are consulted.
14. **Nested config tables were rendered as scalars** — `[channels]` emitted `telegram = "{'token': …}"` before `[channels.telegram]`, and TOML rejects the file as overwriting a value. Caught by a round-trip test that parses what it writes.
15. **Archive path check was a string prefix** — `_safe_extract_path` compared `str(resolved).startswith(str(root))`, so a skill archive containing `../skills-evil/payload.sh` extracted to `/tmp/skills-evil/` while passing the guard, since that path does start with `/tmp/skills`. Now uses component-wise `Path.is_relative_to`. Found by `bandit` follow-up, verified by reproducing the escape.
