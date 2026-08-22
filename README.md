# LAI — a native OS agent for Linux

**The desktop is the page.**

The Claude Chrome extension gave an agent the browser. LAI gives it the whole
operating system: it opens real applications, reads their interfaces, clicks
their buttons, types into their fields, manages their windows, runs shell
commands and works with files — autonomously, in a loop, until the task is done.

It is not a screenshot-and-guess agent. LAI reads the **AT-SPI accessibility
tree** — the desktop's equivalent of a DOM — so it knows there is a button
literally named *Save* at exactly those coordinates, and can invoke it through
the toolkit's own action API. Pixels are the fallback, not the plan.

```
$ lai do "Open the text editor, write a haiku about X11, save it to ~/haiku.txt"

zai/glm-5.2 → Open the text editor, write a haiku about X11...
────────────────────────────────────────────────────────────────
▸ app_open {"name": "Text Editor"}
  ✓ Opened 'Text Editor'. Window 'Unsaved Document 1' [Xed] after 0.6s.
▸ ui_snapshot {}
  ✓ 57 accessible element(s)
▸ ui_type {"ref": 12, "text": "Pixels in a grid..."}
  ✓ Set text via a11y_set_text
▸ computer_key {"key": "ctrl+s"}
  ✓ Pressed ctrl+s
▸ ui_type {"name": "Name:", "text": "/home/felix/haiku.txt"}
  ✓ Set text of the filename field
▸ ui_click {"name": "Save"}
  ✓ Clicked push button 'Save' via a11y_action
▸ file_read {"path": "/home/felix/haiku.txt"}
  ✓ Pixels in a grid...
▸ task_complete
────────────────────────────────────────────────────────────────
✓ completed — 8 steps in 34s
```

---

## Why it works

Desktop agents fail for one reason: **bad grounding**. A model given only a
screenshot has to guess where things are, and it guesses wrong constantly. LAI
fuses four sources of truth:

| Source | What it gives | Analogy |
|---|---|---|
| **AT-SPI tree** | Every widget with role, name, value, state, exact bounds | the DOM |
| **Window tree** | Every window with title, class, pid, geometry, state | the tab list |
| **Screen capture** | What it actually looks like | the rendered page |
| **Process tree** | What is running, what a launch produced | — |
| **OCR** | Text in apps that publish no a11y tree | the fallback |

Actions follow the same priority. `ui_click(name="Save")` invokes the widget's
own accessible action — deterministic, and it still works when the window moves.
`computer_click(x, y)` is there for apps that publish nothing (Electron, games,
VMs), not as the default.

Every step runs **observe → act → verify**. The agent does not report success
because it sent a click; it reports success after it re-read the state and saw
the effect.

---

## Install

Linux with **X11**, Python 3.10+. One line:

```bash
curl -fsSL https://raw.githubusercontent.com/cufelix/lai/main/packaging/bootstrap.sh | sh
```

It checks the machine, clones to `~/.local/share/lai`, installs the system
packages, builds the virtualenv, turns on accessibility, puts `lai` on your
`PATH`, and hands over to the setup wizard. Everything it changes is printed
before it runs, and re-running it updates an existing install.

```bash
LAI_DIR=~/opt/lai  …| sh        # install somewhere else
LAI_REF=v0.2       …| sh        # a tag or branch
…| sh -s -- --no-setup          # stop before the wizard
```

Prefer to read it first? That is the better instinct:

```bash
curl -fsSL https://raw.githubusercontent.com/cufelix/lai/main/packaging/bootstrap.sh -o lai-install.sh
less lai-install.sh && sh lai-install.sh
```

Or clone and run the installer directly:

```bash
git clone https://github.com/cufelix/lai.git && cd lai && ./packaging/install.sh
```

### Setup

```bash
lai setup
```

Four steps, and it refuses to leave you with something that does not work:

```
LAI setup — a native agent for your Linux desktop
Four steps. Nothing is changed without asking.

1/4  Checking this machine
  ✓ platform                 Linux 6.14.0 / Linux Mint 22.2
  ✓ display server           x11 (DISPLAY=:0)
  ✓ input (xdotool)          available
  ✓ accessibility (AT-SPI)   28 app(s) registered, 57 element(s) in the focused window
  ! OCR (tesseract)          missing — apps without an accessibility tree are harder to read
    fix: sudo apt-get install -y tesseract-ocr
    run it? [Y/n]

2/4  Model backend
  Which model backend?
    1. Claude (Anthropic)  [console.anthropic.com/settings/keys]  (default)
    2. GLM (z.ai)  [z.ai — API keys]
    3. OpenAI  [platform.openai.com/api-keys]
    4. OpenRouter (many models, one key)  [openrouter.ai/keys]
    5. Ollama — runs locally, no key needed
    6. Skip for now
  Paste your Claude (Anthropic) key (it is not echoed):
  verifying the key…
  ✓ works  anthropic/claude-sonnet-4-5 replied 'OK'

3/4  How much should it ask?
  ✓ mode: ask

4/4  First run
  asking: what is on this desktop right now?
  ✓ Seven windows are open (Firefox, two terminals, Chrome, Nemo, VS Code, Xviewer),
    with the focused window being the maximized Google Chrome.
  2 steps, 11s

LAI is ready.
```

The key is **spent on a real request before it is saved**, so a typo is caught
in the wizard rather than on your first real task. The config lands in
`~/.lai/config.toml`, mode `0600`. Running `lai setup` again is safe: it keeps
what is already there and changes only what you pick.

`--yes` takes every default without asking (for scripted installs); `--no-demo`
skips the first run.

### If something is wrong

```bash
lai doctor        # what is broken, and the exact command that fixes each thing
lai doctor --fix  # apply every automatic fix
```

Every failure carries its own repair — the diagnosis and the fix are the same
code, so they cannot drift apart.

### Model backends — bring whatever you already have

LAI does not care where the thinking comes from. `lai models` shows every route
this machine has:

```
$ lai models

Ready now
  zai            api   glm-5.2
                 glm wrapper script (~/.local/bin/glm)
  cli:claude     cli   claude
                 claude CLI on PATH (sign-in not verified)
  ollama         local qwen3-vl:2b
                 serving 4 model(s): qwen3.5:2b, gemma4:latest, qwen:latest…

…and 19 more — `lai models --all` to see them

6 usable now, 25 known in total
lai models test <name>   prove one works
lai models use <name>    make it the default
```

**Choosing a model, not just a vendor.** OpenRouter carries hundreds and adds
more weekly, so the list is asked for rather than hard-coded:

```
$ lai models models openrouter glm
  z-ai/glm-5.2:free      256k ctx · free
  z-ai/glm-4.7-flash     202k ctx · $0.06/M in
  z-ai/glm-4.7           204k ctx · $0.4/M in
…422 model(s) · `lai models use openrouter <model>` to pick one
```

Free first, then cheapest, with context windows. The same list is a numbered
menu in the chat (`/models openrouter glm`) and a searchable picker in the
browser's Settings page. Pinning one verifies it first — "OpenRouter works" and
"this model works" are different claims.

Adding the key takes one line and never leaves the conversation:

```
› /key openrouter sk-or-v1-…
saved and verified openrouter/z-ai/glm-5.2:free replied 'OK'
```

**Three kinds of backend:**

| Kind | What it is | Key needed |
|---|---|---|
| `api` | Anthropic, OpenAI, GLM/z.ai, OpenRouter, Gemini, Groq, DeepSeek, Mistral, xAI, Together, Fireworks, Cerebras, Perplexity, Nebius, Moonshot, Qwen | yes |
| `local` | Ollama, LM Studio, llama.cpp, vLLM, LiteLLM, Jan | no |
| `cli` | **Claude Code, Codex, Gemini CLI, opencode** | no — it uses their login |

That last row is the interesting one. If you already have Claude Code signed in
on a subscription, or Codex on a ChatGPT plan, LAI can borrow it:

```bash
lai models use cli:claude
lai do "open the calculator and work out 12 * 34"
```

```
$ lai do "How many windows are open?" --provider cli:claude
▸ window_list {}
  ✓ 7 window(s) (* = focused)
▸ task_complete
✓ completed — 2 steps in 16s
```

No API key anywhere. And because someone's Claude Code might itself be pointed
at Ollama or a proxy, this works for whatever *they* configured — LAI just asks
the CLI.

**Vision works too — through the filesystem.** These CLIs take no inline
images, but they can *read a file*: each turn, LAI stages the newest
screenshots to a temp dir, lists the paths in the prompt, and grants the CLI
read access (`--allowedTools Read --add-dir` for claude, equivalents for the
others). The model reads the screen the same way you would paste a screenshot
into a chat — verified working, including identifying the focused window from
a live capture. `codex` and `gemini` get the same treatment where their flags
allow.

**What you give up with a CLI backend**, stated plainly:

- **Slower turns**, and the whole transcript is re-sent each time, because
  these CLIs are stateless per invocation.
- **No token accounting** — a CLI reports cost, not tokens, so LAI reports zero
  rather than inventing a number.
- Tool calling is emulated through a JSON protocol rather than native, so a
  model that ignores the format costs a retry.

An API key is still the better experience. The CLI route exists so that "I have
no API key" is never the end of the conversation.

### Without the wizard

LAI auto-detects a backend. Any of these is enough:

```bash
export ANTHROPIC_API_KEY=sk-ant-...        # Anthropic
export ZAI_API_KEY=...                     # GLM / z.ai
export OPENAI_API_KEY=...                  # OpenAI
export OPENROUTER_API_KEY=...              # OpenRouter — one key, most models
export GROQ_API_KEY=...                    # …and a dozen more, see `lai models --all`
ollama serve                               # fully local, no key
claude / codex / gemini                    # already signed in? that works too
```

It also reads an existing `ANTHROPIC_BASE_URL` + `ANTHROPIC_AUTH_TOKEN` pair
(any Anthropic-compatible gateway), and picks up a local `glm`-style wrapper
script if you already have one. If a key is already in your environment,
`lai setup` finds it and does not ask.

---

## Use

```bash
lai                             # setup if new, otherwise the chat interface
lai --continue                  # pick up the last conversation
lai do "<task>"                 # one autonomous run
lai do "<task>" --here          # …on *your* desktop, for tasks about your open windows
lai web                         # the same agent, in your browser
lai tui                         # full-screen dashboard
lai repl                        # plain interactive session
lai doctor                      # environment diagnostics
lai notes                       # what it has learned about this machine
lai observe                     # print exactly what the agent sees right now
lai tools                       # list the 53 tools
lai skills list|install|show    # manage skills
lai sessions                    # past runs
lai sessions <id>               # replay one, readably
lai models                      # every backend this machine can use
lai schedule                    # recurring tasks (run by the daemon)
lai channels                    # remote connectors and who may use them
lai serve                       # HTTP daemon with live event streaming
lai mcp                         # expose the desktop over MCP (see below)
```

### The interface

Type `lai` and you are talking to it:

```
LAI v0.4.0 — your desktop, driven
zai/glm-5.2 · mode ask · 53 tools · 38 skills
failover ready: cli:claude → cli:codex → ollama

Say what you want done. /help for commands.

› open the text editor and write me a haiku about X11
```

Slash commands cover everything you would otherwise have to quit for:

| | |
|---|---|
| `/model` | switch backend — with no argument it offers a menu of what works |
| `/fallback` | the standby order when a backend refuses (`auto`, a list, or `off`) |
| `/mode` | `readonly` · `ask` · `auto` · `yolo` |
| `/status` | who is answering, what stands behind them, what already stepped aside |
| `/doctor` `/observe` `/tools` `/skills` | inspect without leaving |
| `/notes` `/edit` `/learn` `/forget` | what it has learned here — read, correct, teach, delete |
| `/settings` | everything you can change, and what it is set to |
| `/new` | forget this conversation and start clean |
| `/sessions` `/resume` | past conversations, and picking one up again |

Everything you change is written to `~/.lai/config.toml`, so the next start
remembers it. History and tab-completion come from prompt_toolkit when it is
installed; without it, the same interface runs on plain input.

### In the browser

It comes up on its own. Typing `lai` starts the conversation *and* the browser
view beside it, sharing one runtime — so the two are the same session rather
than two agents fighting over one desktop:

```
LAI v1.0.0 — your desktop, driven
openrouter/z-ai/glm-4.6 · mode auto · 54 tools · 38 skills
browser view: http://127.0.0.1:8788/#<token>
```

`[web] autostart = false` turns it off; `open_browser = true` makes it launch a
window rather than print the link. `lai web` still runs it on its own.

The same agent, the same desktop gate, reached from a tab — in three views:

- **Chat** — streaming tool calls, a live view of the actual screen beside the
  conversation, stop button, failover announced inline.
- **Learned** — everything it believes about this machine, as an editable
  markdown list. Fix a wrong note, write one yourself, delete what is stale.
- **Settings** — backend, failover, permission mode and learning, written to
  the same `config.toml` the terminal reads.

The page is one self-contained file served by the daemon — no CDN, no bundler,
a strict `default-src 'none'` policy — and the token travels in the URL
*fragment*, which browsers never send to a server.

### The dashboard

`lai tui` gives you the agent with its work visible next to it:

```
 LAI  zai/glm-5.2  ask  step 6  42s  4,102 tok
┌────────────────────────────────────────┬──────────────────────────┐
│ ▸ ui_snapshot                          │ PLAN                     │
│   ✓ 57 accessible element(s)           │   ✓ open the editor      │
│ ▸ ui_type ref=12 text='Dear Anna,'     │   → write the letter     │
│   ✓ Set text via a11y_set_text         │     save as letter.txt   │
│ ▸ computer_key key='ctrl+s'            │                          │
│   ✓ Pressed ctrl+s                     ├──────────────────────────┤
│                                        │ DESKTOP                  │
│                                        │  focused                 │
│                                        │   *Unsaved Document 1    │
│                                        │   Xed  1920x1008         │
│                                        │  windows (5)             │
│                                        │  → Xed        Unsaved…   │
│                                        │    firefox    Claude     │
├────────────────────────────────────────┴──────────────────────────┤
│ > Tell me what to do on this desktop…                             │
└───────────────────────────────────────────────────────────────────┘
```

Permission prompts appear as a modal where you are already looking — `y` allows,
`n` refuses. `ctrl+c` interrupts, `f2` cycles permission mode, `f5` re-observes,
`ctrl+n` starts a fresh session.

Useful flags on `do` / `chat` / `tui`: `--mode`, `--model`, `--provider`,
`--steps`, `--timeout`, `--dry-run`, `--json`, `--verbose`.

### It arrives knowing things

An agent that starts every task from nothing repeats solved problems,
re-derives facts it was told, and cannot answer *"carry on with what we were
doing"*. Before a run begins, LAI gathers three different kinds of knowing:

| | |
|---|---|
| **Notes** | how *this machine* behaves, learned by working on it — markdown you can correct |
| **Memory** | what it was *told* to remember: preferences, decisions, keyed facts |
| **Recently** | what was happening lately, so "carry on" has a referent |

```
## What you have learned on this machine
### Top panel reserves 64px of vertical space
- Maximized windows start at y=64 …

## Memory
- [preference] the user prefers windows tiled to the left half

## Recently on this machine
- 1 min ago: How many windows are open on this desktop right now?
```

All of it ranked against the task and capped at 4,000 characters — the point
is to spend *fewer* tokens, so a recall block that crowded out the task would
be a loss dressed as a feature. Every source fails soft: a broken database
costs its section, never the run.

**Compaction is when facts move out.** A summary put back into the transcript
survives until the next compaction and then goes too, so anything that will
still be true tomorrow is written to the journal before the conversation is
dropped.

### It learns this machine

An agent that rediscovers your desktop every run wastes most of its steps on
things it already worked out yesterday. So after a run that taught it
something, it writes a note:

```
$ lai notes
  drawing   the canvas starts below the toolbar at about y=140 (app, drawing)
  editor    "Text Editor" in the launcher is actually Xed        (app)
  firefox   ctrl+l focuses the URL bar; the a11y tree is slow    (browser)

3 note(s) in ~/.lai/notes
```

Those notes go into the next run's prompt, marked as *starting points to
verify* rather than facts — a stale note followed blindly is worse than none.

They are plain markdown in `~/.lai/notes`, deliberately: an agent's beliefs
about your machine should be something you can read, correct and throw away.

```bash
lai notes                 # everything it believes
lai notes show drawing    # read one
lai notes edit drawing    # open it in $EDITOR
lai notes add "drawing: the canvas starts at y=140"
lai notes rm drawing
```

The same list is editable in the browser under **Learned**, and in the chat
with `/notes`, `/edit`, `/learn` and `/forget`. Reflection costs one extra
model call at the end of a run that actually did something; `/learning off`
(or `[learning] enabled = false`) stops it, and existing notes are still read.

### It can hire a coder

LAI can write files itself, but its tools are aimed at a desktop and it pays
for every line in agent steps. A coding CLI is a specialist at exactly that —
and most people running LAI already have one signed in. So the division of
labour plays to both sides:

```
▸ code_agent {"task": "build a playable Snake in snake.html…", "workspace": "~/games"}
  ✓ claude worked for 74s. 1 file(s) changed on disk: ~/games/snake.html
▸ app_open {"name": "Firefox", "args": ["~/games/snake.html"]}
▸ computer_key {"key": "ArrowUp"}
▸ computer_screenshot {}
  ✓ the snake turned — it works
```

**The coding agent writes; LAI checks.** Opening the result on a real screen,
pressing its keys and looking at what happened is the thing no coding agent
can do for itself. LAI is the architect and the tester; `claude`, `codex`,
`gemini` or `opencode` is the coder.

What comes back is *evidence, not a claim*: the worker's own summary plus the
files that actually changed on disk, so "I have created the file for you" with
an empty directory is reported as exactly that. The worker is confined to one
named directory, and the tool is classified `destructive`, so in `ask` mode a
human approves the job before a file is touched.

### When a backend runs out

Subscriptions hit quotas and hosted endpoints have bad minutes, usually
half-way through something. LAI treats that as a reason to carry on, not to
stop:

```
▸ computer_drag {"from": [350, 320], "to": [550, 320]}
  ✓ dragged
↻ zai stepped aside (HTTP 429 rate_limit_error: Usage limit reached for 5 hour)
  continuing on cli:claude/claude
▸ screenshot {}
```

The chain is ordered (hosted keys, then signed-in coding CLIs, then local
models — a 2B model on the CPU is a last resort, not a first choice), lazy
(a standby is only built when it is reached) and sticky (no flapping between
two models mid-task). Only failures another backend could plausibly survive
move the run on — quota, auth, an outage; a malformed request is raised, since
it would fail identically everywhere.

**And it remembers.** A quota lasts hours, so rediscovering it every run — send,
wait, 429, fail over — is pure waste. Failures are written down with a recovery
time, and vendors usually state one:

```
$ lai models
Ready now
  cli:codex      cli   codex
  ollama         local qwen3-vl:2b
  zai            api   glm-5.2
                 ⏳ out of quota, retry in 59 min
  cli:claude     cli   claude
                 ⏳ out of quota, retry in 59 min
```

That "59 min" is z.ai's own stated reset time, parsed rather than guessed. A
backend still cooling is skipped as a standby, and `auto` will not start on one.
An *explicitly configured* backend is always tried anyway — being wrong about a
recovery time must cost a retry, never an outage.

`/fallback off` or `LAI_FALLBACK=off` turns it off.

**Never using a particular backend** is a setting, not a matter of avoiding it:

```toml
[provider]
deny = ["cli:claude", "anthropic"]
```

A denied backend is not auto-detected, not fallen back to, not listed and not
offered in a menu — and asking for one explicitly is an error rather than a
silent substitution. `LAI_DENY=anthropic` does the same for one run.

### A screen of its own

**LAI works on its own desktop by default.** It starts a second X server with
its own root window, its own pointer and its own focus. Applications it opens
live there, screenshots come from there, and nothing it does reaches your
keyboard, your clipboard or your window stack. You keep working; it keeps
working; neither waits for the other.

This is the default because the alternative is not a fair trade. Sharing one
desktop means the agent's click lands in the window you just switched to, its
typing goes wherever your focus went, and the thing it opens covers what you
were reading. Taking turns makes that less frequent, not less annoying.

```bash
lai do "open the calculator and work out 12 * 34"   # its screen, in a window
lai do "..." --unwatched                            # its screen, off-screen entirely
lai do "..." --here                                 # your desktop, taking turns
```

Its screen starts empty, and applications on it have no session — no logged-in
browser profile, no open documents. So a task *about the windows you have open*
needs `--here`; the agent is told this and will say so rather than guessing.
`lai observe` always reports your desktop, because that is the whole point of
the command.

**When the run ends, the work comes to you.** An X window belongs to the
display its client connected to — the agent's Firefox cannot cross to yours,
and dies with the server. What was *in* it can:

```
✓ completed — 4 steps in 18s
→ opened on your desktop: https://jspaint.app/#local:b1f7a5b7 (from firefox)
→ opened on your desktop: /home/you/report.md
```

Pages the agent had open are read out of the address bar (matched on the value,
so it works in any language) and files the run wrote are reopened, both through
`xdg-open` on your display. Something already handed over is not sent again,
because a browser left open stays open and every later task would find the same
page. Unsaved state in an editor is not reconstructed: that is gone when the
server stops, and pretending otherwise would be worse than saying so.
`[desktop] handover = false` turns it off.

Browsers the agent launches on its own screen get their own profile
(`--no-remote --profile`, or `--user-data-dir`) — without it they hand the URL
to the copy running on *your* desktop and exit, and every browser task is
unwinnable. Chromium is also started with `--force-renderer-accessibility`,
without which its window is a single opaque rectangle to AT-SPI: no page, not
even the address bar.

By default that screen is shown in a window on yours (Xephyr), so you can watch
it work. The window is deliberately unobtrusive: it does not take focus when it
appears — focus goes straight back to whatever you were typing into — and it
never grabs your keyboard or mouse when the pointer crosses it. It draws the
agent's own cursor inside itself, so you can see where it is actually clicking.
Minimise it and the agent carries on; it is a window onto a server, not the
server.

| | |
|---|---|
| **Xephyr** | nested in a window on your desktop. The default, and what `--watch` asks for. |
| **Xvfb** | entirely off-screen. `--unwatched`, or `watch = false`. |

```toml
[desktop]
own_display = "auto"   # auto = its own screen when possible · always · never
watch = true           # false to keep it off-screen entirely
```

`auto` falls back to your desktop and says so if no virtual X server is
installed; `always` refuses to run rather than touch your session. `lai doctor`
offers to install Xvfb. A window manager is started alongside, because without
one applications have no decorations, cannot be maximised, and frequently never
receive focus — which looks exactly like a broken agent.

### It gets out of your way

When it *is* on your desktop — `--here`, or no virtual X server installed — one
mouse is shared between two hands, and that does not split the work. So before
it touches the mouse or keyboard, LAI checks how long you have been still:

```
▸ computer_click {"x": 812, "y": 344}
⏸ you are using the machine — waiting for you to finish
▶ carrying on after 7s
```

Only for actions that drive the input devices — reading the screen while you
type is harmless, so observation never waits. Bounded, too: if you simply keep
working it gives up its turn and says so rather than hanging. And on a machine
whose X server cannot report idle time, the agent works normally rather than
refusing to move.

`[safety] yield_to_user = false` turns it off; `user_idle_seconds` sets how
still you have to be. On its own screen none of this applies — there is nothing
to take turns over.

### Permission modes

| Mode | Observe | Click / type | Write files, open apps | Shell, kill |
|---|---|---|---|---|
| `readonly` | ✓ | ✗ | ✗ | ✗ |
| `ask` *(default)* | ✓ | ask | ask | ask |
| `auto` | ✓ | ✓ | ✓ | ask |
| `yolo` | ✓ | ✓ | ✓ | ✓ |

Some things are refused in **every** mode, `yolo` included:

- destructive shell patterns — `rm -rf`, `mkfs`, `dd if=`, `curl … | sh`, `shutdown`
- sending input to password managers or authentication prompts
  (KeePassXC, Bitwarden, 1Password, polkit dialogs, anything titled *password*)

Every action is rate-limited, written to an append-only audit log at
`~/.lai/logs/`, and passed through secret redaction before it is logged or sent
to a model.

---

## Reach it from anywhere

Connectors let the agent be driven from outside the machine it runs on.

```bash
export LAI_TELEGRAM_TOKEN=123456:AA...      # from @BotFather
lai channels test telegram                   # verify the token
lai serve --channels telegram                # prints a one-time pairing code
```

Then message the bot `/pair 041823`. From your phone:

```
you  open the text editor and write down today's meeting notes
LAI  Working… step 3/25 · ui_type
LAI  ✅ completed — 7 steps, 41s
     Opened Xed, typed the notes, saved to ~/notes/2026-08-17.txt
     Verified: re-read the file; contents match.
```

| Connector | Direction | Notes |
|---|---|---|
| `telegram` | two-way | long polling, no inbound port — works behind NAT |
| `discord` | two-way | bot gateway with heartbeat + RESUME |
| `webhook` | two-way | generic JSON; Slack- and Discord-webhook payload shapes |
| `local` | in-process | scripting, embedding and tests |

Chat commands: `/status` `/stop` `/new` `/screenshot` `/windows` `/skills`
`/mode` `/whoami` `/help`, plus `/allow` `/revoke` `/who` for admins.

**Access control.** The bot token is not the security boundary — the allowlist
is. An unknown sender gets a flat `Not authorised.` with no hint that pairing
exists. Pairing needs a six-digit code minted on your own terminal, single-use,
15-minute expiry, five attempts. The first person to pair becomes admin; the
allowlist is stored `0600` at `~/.lai/channels.json`.

**Remote approvals.** In `ask` mode the permission prompt is delivered to the
chat and the run blocks on your reply, so operating remotely does not force you
into `yolo`:

```
LAI  ⚠️ Approval needed
     shell_exec
     command='git push origin main'
     command needs confirmation (matched 'git push')
     Reply /yes to allow or /no to refuse (expires in 180s).
```

---

## Give Claude Code the desktop

This is the shortest path from "Claude in the browser" to "Claude on the OS":

```bash
claude mcp add lai -- /path/to/lai/.venv/bin/lai mcp
```

Claude Code now has all 53 desktop tools — `app_open`, `ui_snapshot`,
`ui_click`, `window_focus`, `computer_screenshot`, and the rest — driving your
real machine, with LAI's safety gate still in force.

Set `LAI_MODE=auto` for the MCP server, since an MCP client cannot answer
interactive approval prompts.

---

## What it can do

53 native tools, in families:

| Family | Tools | For |
|---|---|---|
| `ui` | snapshot, find, click, type, focus, read, wait_for | semantic control through the a11y tree — the primary path |
| `computer` | screenshot, click, move, drag, scroll, type, key, cursor | pixel fallback for apps with no a11y tree |
| `window` | list, focus, close, arrange | window management |
| `app` | list, open, close | launch-and-wait-for-window |
| `perception` | ocr_read, ocr_find, notify_user, notifications_recent, user_idle, workspace_* , record_start/stop | reading pixels, watching notifications, knowing whether the human is present, virtual desktops, screen recording |
| `agentic` | memory_save/search/forget, delegate, schedule_* | learning across sessions, isolated subtasks, recurring runs |
| `system` | shell_exec, file_read/write/list, clipboard_read/write | non-GUI work |
| `skill` | list, load, install | procedures, including installing new ones mid-task |
| `control` | plan_update, task_complete, task_blocked | planning and explicit termination |

Three of these are worth calling out:

- **Memory** — `~/.lai/memory.db` (SQLite, FTS5 where available). The agent
  records what it learns about your applications — *"Xed's save dialog names its
  field `Name:`"* — so the next session does not rediscover it.
- **`delegate`** — runs a self-contained subtask in a fresh context with its own
  step budget and returns only the conclusion. Depth-capped at 2, and it cannot
  escape the parent's safety gate.
- **`user_idle`** — an autonomous agent should not fight you for the mouse. It
  can check whether you are actually at the keyboard before taking over.

**Scheduling.** `schedule_task` accepts standard 5-field cron (`*`, lists,
ranges, `*/n`, `@hourly`/`@daily`/`@weekly`/`@monthly`) or `every:<seconds>`.
The daemon runs due tasks, skips them while a user-initiated run is in flight,
and pushes the result to your connectors. Manage them from the shell too:

```bash
lai schedule add nightly "@daily" "summarise today's notes into ~/journal"
lai schedule list                # what is scheduled, when each is next due
lai schedule run <id>            # fire one now, ignoring its schedule
lai schedule disable <id>        # keep it, stop it firing
```

**One desktop, one agent.** The HTTP API, the connectors and the scheduler all
claim the same gate before they start, so a chat message arriving mid-task is
refused with an explanation rather than fighting the running agent for the
mouse.

**OCR** needs `sudo apt install tesseract-ocr`. Without it, `ocr_*` report
cleanly that they are unavailable and everything else keeps working — the same
pattern as every other optional dependency here.

---

## Skills

LAI reads **Claude Code's skill format**, unchanged. Any `SKILL.md` with
frontmatter is a skill:

```markdown
---
name: invoice-filing
description: Use when filing a PDF invoice into the accounting folder
---
1. Open the PDF in the document viewer …
```

Discovered from `~/.lai/skills`, `./.lai/skills`, `./.claude/skills`,
`~/.claude/skills` and `~/.openclaw/skills`. Names and descriptions go into the
system prompt; the body loads on demand via `skill_load` — so a hundred skills
cost almost nothing until one is actually used.

Install from anywhere:

```bash
lai skills install owner/repo               # GitHub
lai skills install https://…/skills.git     # any git URL
lai skills install https://…/skill.zip      # archive
lai skills install ./my-skill               # local
```

The agent can also install a skill mid-task when it finds it lacks a capability.

### Models without function calling

Native tool calling is an API feature, and plenty of capable models do not have
it — Hermes, Qwen, most things served straight off Ollama or vLLM. They were
trained to write the call instead:

```
<tool_call>
{"name": "window_list", "arguments": {}}
</tool_call>
```

LAI reads both forms, so those models work without configuration. It also
writes the schemas into the prompt in that shape when a backend says it has no
function calling — discovered from the refusal, once, the same way a model with
no vision is. Mistral's `[TOOL_CALLS]` and Llama's `<|python_tag|>` are
understood too. A bare JSON object in the prose is deliberately *not*: a model
explaining what it would call has not called it.

```toml
[provider]
tool_dialect = "auto"   # auto · native (never fall back) · text (always ask in the prompt)
```

Reasoning that arrives in its own field — DeepSeek's `reasoning_content`, and
anything OpenRouter proxies from it — is kept rather than dropped.

## MCP tools

LAI is an MCP **client** too. It reads `~/.lai/mcp.json`, `./.mcp.json` and
Claude Code's own MCP config, connects to each server, and registers their tools
as `mcp__<server>__<tool>` — so anything reachable over MCP becomes something
the desktop agent can do.

Connect a handful of servers and that is several hundred tools. Sending every
schema on every turn is the most expensive thing a well-equipped machine does —
measured here at 242 KB of JSON, about sixty thousand tokens, to answer *how
many windows do I have open*. So LAI's own tools always go, and the connected
ones are matched against the task. What is left out is named rather than
hidden:

```
# Connected services
195 further tools are connected but not listed above, from: confluence (25),
firecrawl (27), github (26)… If you need one, call `tool_find(query)`.
```

`tool_find` searches every connected tool and makes the matches callable for
the rest of the run. Naming the service works — *read a github pull request*
finds the github tools — because servers are matched before word frequencies
are consulted.

---

## HTTP daemon

```bash
lai serve                       # 127.0.0.1:8787, bearer token at ~/.lai/daemon.token
```

| Endpoint | Purpose |
|---|---|
| `GET /health` | liveness (no auth) |
| `GET /status` | provider, tools, skills, current task |
| `GET /observe` | what the agent sees now |
| `GET /tools` `GET /skills` `GET /sessions` | inventories |
| `POST /task` | run a task — Server-Sent Events stream by default |
| `POST /stop` | interrupt the running task |

```bash
TOKEN=$(cat ~/.lai/daemon.token)
curl -sN -H "Authorization: Bearer $TOKEN" \
     -d '{"task":"open the calculator","mode":"auto"}' \
     http://127.0.0.1:8787/task
```

It refuses to bind a non-loopback address without `--allow-remote`. This API
controls your desktop; treat the token like a password.

---

## Architecture

```
lai/
├── osl/          OS layer — the hard part
│   ├── a11y.py       AT-SPI tree: the desktop's DOM
│   ├── windows.py    EWMH window management via python-xlib
│   ├── screen.py     multi-monitor capture + coordinate mapping
│   ├── inputs.py     mouse/keyboard via xdotool (XTEST)
│   ├── apps.py       .desktop index, launch-and-wait-for-window
│   ├── clipboard.py  GTK clipboard
│   └── desktop.py    the facade: semantic-first, pixels-as-fallback
├── tools/        53 tools with JSON schemas, one safety choke point
├── agent/        the loop, 5 providers, sessions, compaction, prompt
├── skills/       discovery, progressive disclosure, install-from-internet
├── channels/     telegram · discord · webhook · local, with pairing + allowlist
├── tui/          full-screen textual interface
├── mcp/          client (consume) and server (expose)
├── scheduler.py  hand-rolled cron, task store, background runner
├── safety/       permission policy, redaction, audit log
├── daemon/       HTTP + SSE service
├── config.py     file + env + defaults, immutable
└── cli.py
```

Design decisions worth knowing:

- **Immutable config.** Policy cannot be mutated mid-run.
- **One dispatch path.** Every tool call goes through `ToolRegistry.call`, which
  is where permission checks, validation, auditing and error containment live.
  No tool can bypass the gate by accident.
- **Perception never crashes the loop.** Every observation degrades to `None`
  rather than raising; a dead app mid-walk is normal, not exceptional.
- **Explicit termination.** The agent finishes by calling `task_complete` with a
  verification statement, not by trailing off. Budget exhaustion, blocking and
  errors are reported as themselves, never dressed up as success.
- **Screenshots are pruned.** Only the newest few survive in context; older ones
  become placeholders. Context compaction summarises via the model and refuses
  to cut at a point that would orphan a `tool_result`.

---

## Limitations

- **X11 only.** Wayland needs portal-based capture and libei input; the backend
  interface is there, the implementation is not.
- **Chromium and Electron publish no accessibility tree** unless started with
  `--force-renderer-accessibility`. Without it, LAI falls back to screenshots
  and coordinates in those apps — it works, just less precisely.
- **OCR needs tesseract installed**; without it, a11y-less apps are only as
  readable as the model's vision.
- Vision quality depends on the model. GLM and Claude both work; small local
  models struggle with dense UIs.

---

## Development

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/ -q          # 1063 tests
PYTHONPATH=src .venv/bin/python -m pytest tests/ -q -m "not slow and not x11"   # no display needed
PYTHONPATH=src .venv/bin/python -m pytest tests/ -q --cov=src/lai --cov-report=term   # 81%
.venv/bin/ruff check src/ tests/                             # clean
.venv/bin/bandit -q -r src/lai                               # 5 documented false positives
```

The ruff config in `pyproject.toml` encodes this codebase's deliberate choices
rather than fighting them — blind `except Exception` at subsystem boundaries is
the degradation strategy, local imports are how optional dependencies stay
optional, and the cron engine is local-time on purpose. Everything else is
enforced.

`x11`-marked tests drive the real display. They only ever touch windows they
launched themselves and always clean up.

See `docs/FEATURES.md` for the full feature catalogue and what is planned next.
