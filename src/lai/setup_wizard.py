"""``lai setup`` — from a fresh machine to a working agent, in one command.

The design rule here is that the wizard never leaves someone with a
configuration that does not work. Every choice is verified before it is
written: a package is re-probed after installing, an API key is spent on a real
one-token request before it is saved, and the final step runs an actual task so
the first thing a new user sees is the agent doing something rather than a
promise that it would.

It is also safe to run repeatedly. Nothing here is destructive: it re-reads the
existing config, changes only what was chosen, and every system change (a
package install, a gsettings flag) is shown as the exact command and applied
only after an explicit yes.
"""

from __future__ import annotations

import shutil
import sys
from dataclasses import dataclass, replace
from pathlib import Path

from . import config_file
from .checks import FAIL, OK, WARN, Check, Report, run_checks
from .config import Config, load_config

# Offered in this order: the first two are what most people have, ollama is the
# no-key escape hatch, and "I'll do it later" must always be reachable.
BACKENDS = [
    ("anthropic", "Claude (Anthropic)", "https://console.anthropic.com/settings/keys", "sk-ant-…"),
    ("zai", "GLM (z.ai)", "https://z.ai/manage-apikey/apikey-list", "…"),
    ("openai", "OpenAI", "https://platform.openai.com/api-keys", "sk-…"),
    ("openrouter", "OpenRouter (many models, one key)", "https://openrouter.ai/keys", "sk-or-…"),
]

DEMO_TASK = (
    "Tell me what is on this desktop right now: how many windows are open and "
    "which one is focused. Use window_list, then call task_complete with a "
    "one-sentence answer. Do not open, close or click anything."
)


@dataclass(slots=True)
class Answers:
    """What the wizard decided, so a caller can assert on it."""

    fixed: list[str]
    skipped: list[str]
    provider: str = ""
    model: str = ""
    mode: str = "ask"
    config_written: Path | None = None
    demo_ran: bool = False
    demo_ok: bool = False


class Prompt:
    """Terminal questions, with a non-interactive fallback.

    Scripted installs and CI have no tty. Rather than hanging on input(), every
    question returns its default, so `lai setup --yes` is a valid unattended
    path and an accidental `lai setup < /dev/null` cannot wedge.
    """

    def __init__(self, *, assume_yes: bool = False, interactive: bool | None = None) -> None:
        self.assume_yes = assume_yes
        if interactive is None:
            interactive = sys.stdin.isatty() and sys.stdout.isatty()
        self.interactive = bool(interactive)

    def confirm(self, question: str, *, default: bool = True) -> bool:
        # `--yes` means "take the default", not "say yes to everything": a
        # question whose safe answer is no (switch backends, really use yolo)
        # must not flip just because the run is unattended.
        if self.assume_yes or not self.interactive:
            return default
        suffix = "[Y/n]" if default else "[y/N]"
        try:
            answer = input(f"{question} {suffix} ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return False
        if not answer:
            return default
        return answer in ("y", "yes")

    def choose(self, question: str, options: list[str], *, default: int = 0) -> int:
        if not self.interactive or self.assume_yes:
            return default
        print(question)
        for index, option in enumerate(options, 1):
            marker = " (default)" if index - 1 == default else ""
            print(f"  {index}. {option}{marker}")
        try:
            answer = input(f"choose [1-{len(options)}]: ").strip()
        except (EOFError, KeyboardInterrupt):
            return default
        if not answer:
            return default
        try:
            picked = int(answer) - 1
        except ValueError:
            return default
        return picked if 0 <= picked < len(options) else default

    def secret(self, question: str) -> str:
        """Read a key without echoing it to the screen or the shell history."""
        if not self.interactive:
            return ""
        import getpass  # noqa: PLC0415

        try:
            return getpass.getpass(f"{question} ").strip()
        except (EOFError, KeyboardInterrupt):
            return ""


def run_setup(
    out,
    *,
    assume_yes: bool = False,
    interactive: bool | None = None,
    skip_demo: bool = False,
    config: Config | None = None,
) -> tuple[int, Answers]:
    """Walk the whole path. Returns (exit code, what happened)."""
    prompt = Prompt(assume_yes=assume_yes, interactive=interactive)
    config = config or load_config()
    answers = Answers(fixed=[], skipped=[])

    _banner(out)

    # 1. What is missing, and can we fix it?
    out.write("[bold]1/4  Checking this machine[/bold]")
    report = _probe(config)
    _render(out, report)

    repairs = [c for c in report if c.status != OK and c.fix is not None]
    if repairs:
        out.write("")
        _repair(out, prompt, repairs, answers)
        report = _probe(config)
        out.write("")
        out.write("[bold]After fixes:[/bold]")
        _render(out, report)

    # 2. A model to think with.
    out.write("")
    out.write("[bold]2/4  Model backend[/bold]")
    settings = config_file.read(config.home)
    provider_settings = _setup_provider(out, prompt, report, answers)
    if provider_settings:
        settings = config_file.merge(settings, {"provider": provider_settings})

    # 3. How much freedom it gets.
    out.write("")
    out.write("[bold]3/4  How much should it ask?[/bold]")
    mode = _setup_mode(out, prompt)
    answers.mode = mode
    settings = config_file.merge(settings, {"safety": {"mode": mode}})

    written = config_file.write(config.home, settings)
    answers.config_written = written
    out.write(f"  [green]✓[/green] saved [dim]{written}[/dim]")

    # 4. Prove it works.
    out.write("")
    out.write("[bold]4/4  First run[/bold]")
    if skip_demo:
        out.write("  [dim]skipped[/dim]")
    else:
        _run_demo(out, prompt, answers)

    out.write("")
    return _finish(out, config, answers)


# -- steps ---------------------------------------------------------------


def _probe(config: Config) -> Report:
    """Run the checks against a real runtime, degrading if it cannot be built."""
    runtime = None
    try:
        from .runtime import build_runtime  # noqa: PLC0415

        runtime = build_runtime(config, with_mcp=False)
        return run_checks(runtime, config)
    except Exception:
        return run_checks(None, config)
    finally:
        if runtime is not None:
            try:
                runtime.close()
            except Exception:
                pass


def _repair(out, prompt: Prompt, repairs: list[Check], answers: Answers) -> None:
    automatic = [c for c in repairs if c.fix and c.fix.automatic]
    manual = [c for c in repairs if c.fix and not c.fix.automatic]

    for check in automatic:
        fix = check.fix
        assert fix is not None
        out.write(f"  [yellow]{check.label}[/yellow]: {check.detail}")
        out.write(f"    fix: [cyan]{fix.shell() or fix.description}[/cyan]")
        if fix.needs_sudo and not prompt.interactive:
            # sudo would block on a password prompt nobody can answer.
            answers.skipped.append(check.key)
            out.write("    [dim]skipped — needs sudo, and this is not an interactive terminal[/dim]")
            continue
        if fix.needs_sudo:
            out.write("    [dim](needs sudo — you may be asked for your password)[/dim]")
        if not prompt.confirm("    run it?", default=True):
            answers.skipped.append(check.key)
            out.write("    [dim]skipped[/dim]")
            continue
        ok, output = fix.run()
        if ok:
            answers.fixed.append(check.key)
            out.write("    [green]✓ done[/green]")
        else:
            answers.skipped.append(check.key)
            out.write(f"    [red]✗ failed[/red] [dim]{output.strip()[:300]}[/dim]")
        if fix.manual:
            out.write(f"    [dim]{fix.manual}[/dim]")

    for check in manual:
        fix = check.fix
        assert fix is not None
        answers.skipped.append(check.key)
        icon = "[red]✗[/red]" if check.status == FAIL else "[yellow]![/yellow]"
        out.write(f"  {icon} [bold]{check.label}[/bold]: {check.detail}")
        for line in (fix.manual or fix.description).splitlines():
            out.write(f"    [dim]{line}[/dim]")


def _setup_provider(out, prompt: Prompt, report: Report, answers: Answers) -> dict:
    """Find or obtain a working model backend. Verified before it is saved.

    The menu is built from :func:`lai.models.discover`, not a hand-written
    list, so an agent CLI somebody already has signed in — Claude Code, Codex,
    Gemini — shows up here as a first-class choice alongside the API vendors.
    """
    from . import models as backends  # noqa: PLC0415

    try:
        found = backends.discover()
    except Exception:
        found = []
    ready = [b for b in found if b.usable]

    if ready:
        best = ready[0]
        out.write(f"  [green]✓[/green] found [bold]{best.name}[/bold] ({best.model}) — {best.detail}")
        answers.provider, answers.model = best.name, best.model
        if len(ready) > 1:
            others = ", ".join(b.name for b in ready[1:6])
            out.write(f"  [dim]also usable here: {others}[/dim]")
        if not prompt.confirm("  use a different backend instead?", default=False):
            return {}

    # Anything already working is a one-keystroke choice; keys come after.
    choices: list[tuple[str, object]] = [("ready", b) for b in ready]
    labels = [
        f"{b.name} ({b.model}) — {b.detail}"[:96] + ("  [no vision]" if not b.vision else "")
        for b in ready
    ]
    for name, label, where, _example in BACKENDS:
        choices.append(("key", name))
        labels.append(f"paste an API key for {label}  [{where}]")
    choices.append(("ollama", None))
    labels.append("Ollama — runs locally, no key needed")
    choices.append(("other", None))
    labels.append("something else (Groq, DeepSeek, LM Studio, vLLM, …)")
    choices.append(("skip", None))
    labels.append("Skip for now")

    picked = prompt.choose("  Which model backend?", labels, default=0)
    kind, payload = choices[picked] if 0 <= picked < len(choices) else ("skip", None)

    if kind == "ready":
        backend = payload
        out.write(f"  [green]✓[/green] using [bold]{backend.name}[/bold]")
        if backend.kind == backends.KIND_CLI:
            out.write("  [dim]note: an agent CLI is slower per turn, and sees screenshots by[/dim]")
            out.write("  [dim]reading them from disk rather than inline[/dim]")
        answers.provider, answers.model = backend.name, backend.model
        return {"name": backend.name}

    if kind == "ollama":
        return _setup_ollama(out, prompt, answers)

    if kind == "other":
        return _setup_catalog(out, prompt, answers)

    if kind == "skip":
        out.write("  [yellow]no backend configured — `lai setup` again when you have a key[/yellow]")
        return {}

    return _setup_key(out, prompt, answers, str(payload))


def _setup_key(out, prompt: Prompt, answers: Answers, name: str) -> dict:
    """Paste a key for one of the well-known vendors, and prove it works."""
    entry = next((b for b in BACKENDS if b[0] == name), None)
    if entry is None:
        return {}
    _name, label, where, example = entry

    out.write(f"  Get a key from [cyan]{where}[/cyan]")
    if prompt.interactive and prompt.confirm("  open that page in your browser?", default=True):
        _open_url(out, where)
    key = prompt.secret(f"  Paste your {label} key (it is not echoed):")
    if not key:
        out.write("  [yellow]nothing pasted — skipping[/yellow]")
        return {}
    if not _plausible(name, key):
        out.write(f"  [yellow]that does not look like a {label} key (expected {example})[/yellow]")
        if not prompt.confirm("  use it anyway?", default=False):
            return {}

    out.write("  [dim]verifying the key…[/dim]")
    ok, detail, model = _verify_key(name, key)
    if not ok:
        out.write(f"  [red]✗ the key was rejected:[/red] {detail}")
        if not prompt.confirm("  save it anyway?", default=False):
            return {}
    else:
        out.write(f"  [green]✓ works[/green] [dim]{detail}[/dim]")

    answers.provider, answers.model = name, model
    return {"name": name, "api_key": key, "model": model}


def _setup_catalog(out, prompt: Prompt, answers: Answers) -> dict:
    """The long tail: every other vendor LAI knows how to reach."""
    from .agent.providers.catalog import LOCAL_VENDORS, VENDORS  # noqa: PLC0415

    offered = [v for v in VENDORS if v.name not in {b[0] for b in BACKENDS}] + list(LOCAL_VENDORS)
    labels = [
        f"{v.label}"
        + ("  [runs locally, no key]" if v.local else f"  [{', '.join(v.env_keys)}]")
        + ("" if v.vision else "  [no vision]")
        for v in offered
    ]
    labels.append("back")
    picked = prompt.choose("  Which one?", labels, default=0)
    if picked >= len(offered):
        return {}

    vendor = offered[picked]
    if vendor.local:
        out.write(f"  [dim]{vendor.label} at {vendor.base_url}[/dim]")
        if vendor.notes:
            out.write(f"  [dim]{vendor.notes}[/dim]")
        answers.provider, answers.model = vendor.name, vendor.default_model
        return {"name": vendor.name, "model": vendor.default_model, "base_url": vendor.base_url}

    if vendor.signup:
        out.write(f"  Get a key from [cyan]{vendor.signup}[/cyan]")
        if prompt.interactive and prompt.confirm("  open that page in your browser?", default=True):
            _open_url(out, vendor.signup)
    key = prompt.secret(f"  Paste your {vendor.label} key (it is not echoed):")
    if not key:
        out.write("  [yellow]nothing pasted — skipping[/yellow]")
        return {}

    out.write("  [dim]verifying the key…[/dim]")
    ok, detail, _model = _verify_key(vendor.name, key)
    out.write(f"  [green]✓ works[/green] [dim]{detail}[/dim]" if ok
              else f"  [red]✗ the key was rejected:[/red] {detail}")
    if not ok and not prompt.confirm("  save it anyway?", default=False):
        return {}

    answers.provider, answers.model = vendor.name, vendor.default_model
    return {"name": vendor.name, "api_key": key, "model": vendor.default_model}


def _setup_ollama(out, prompt: Prompt, answers: Answers) -> dict:
    from .agent.providers.registry import OLLAMA_BASE_URL, _ollama_model  # noqa: PLC0415

    if not shutil.which("ollama"):
        out.write("  [yellow]ollama is not installed[/yellow]")
        out.write("  [dim]install it from https://ollama.com, then: ollama pull qwen3-vl:2b[/dim]")
        return {}
    model = ""
    try:
        model = _ollama_model() or ""
    except Exception:
        model = ""
    if not model:
        out.write("  [yellow]ollama is installed but not serving a model[/yellow]")
        out.write("  [dim]start it with: ollama serve   (then: ollama pull qwen3-vl:2b)[/dim]")
        if not prompt.confirm("  configure it anyway?", default=True):
            return {}
        model = "qwen3-vl:2b"
    out.write(f"  [green]✓[/green] local ollama, model [bold]{model}[/bold]")
    out.write("  [dim]note: small local models struggle with dense interfaces[/dim]")
    answers.provider, answers.model = "ollama", model
    return {"name": "ollama", "model": model, "base_url": OLLAMA_BASE_URL}


def _setup_mode(out, prompt: Prompt) -> str:
    picked = prompt.choose(
        "  Permission mode:",
        [
            "ask   — confirm before anything that changes the machine (recommended)",
            "auto  — click and type freely, still confirm shell commands",
            "readonly — look but never touch",
            "yolo  — never ask (only for a machine you can afford to break)",
        ],
        default=0,
    )
    mode = ["ask", "auto", "readonly", "yolo"][picked]
    if mode == "yolo":
        out.write("  [red]yolo: LAI will not ask before anything. Destructive shell patterns[/red]")
        out.write("  [red]and password managers stay blocked, but nothing else does.[/red]")
        if not prompt.confirm("  really?", default=False):
            mode = "ask"
    out.write(f"  [green]✓[/green] mode: [bold]{mode}[/bold]")
    return mode


def _run_demo(out, prompt: Prompt, answers: Answers) -> None:
    """Do something real, so the first impression is the agent working."""
    if not answers.provider and not _has_any_backend():
        out.write("  [dim]no model backend — skipping the demo[/dim]")
        return
    if not prompt.confirm("  Run a harmless first task to prove it works?", default=True):
        out.write("  [dim]skipped[/dim]")
        return

    answers.demo_ran = True
    out.write("  [dim]asking: what is on this desktop right now?[/dim]")
    try:
        from .runtime import build_runtime  # noqa: PLC0415

        config = load_config()
        config = config.with_overrides(
            safety=replace(config.safety, mode="readonly"),
            limits=replace(config.limits, max_steps=6, max_seconds=120.0),
        )
        runtime = build_runtime(config, with_mcp=False)
    except Exception as exc:
        out.write(f"  [red]could not start:[/red] {exc}")
        return

    try:
        if runtime.provider is None:
            out.write(f"  [red]no model provider:[/red] {runtime.provider_error}")
            return
        agent = runtime.agent(approver=lambda *_: False)
        result = agent.run(DEMO_TASK)
        answers.demo_ok = result.ok
        if result.ok:
            out.write(f"  [green]✓[/green] {result.summary}")
            out.write(f"  [dim]{result.steps} steps, {result.elapsed:.0f}s[/dim]")
        else:
            out.write(f"  [yellow]{result.status}[/yellow] {result.error or result.summary}")
    except Exception as exc:
        out.write(f"  [red]the demo failed:[/red] {type(exc).__name__}: {exc}")
    finally:
        try:
            runtime.close()
        except Exception:
            pass


def _finish(out, config: Config, answers: Answers) -> tuple[int, Answers]:
    report = _probe(config)
    ready = report.ready

    out.rule()
    if ready and answers.demo_ok:
        out.write("[green bold]LAI is ready.[/green bold]")
    elif ready:
        out.write("[green bold]Setup complete.[/green bold]")
    else:
        out.write("[yellow bold]Setup finished, but some things still need attention.[/yellow bold]")
        for check in report.blockers:
            out.write(f"  [red]✗[/red] {check.label}: {check.detail}")
            if check.fix and check.fix.manual:
                for line in check.fix.manual.splitlines():
                    out.write(f"      [dim]{line}[/dim]")

    out.write("")
    out.write("[bold]Try this:[/bold]")
    for command, purpose in (
        ("lai", "the full-screen interface"),
        ('lai do "open the calculator"', "one task, right now"),
        ("lai doctor", "re-check this machine"),
    ):
        out.write(f"  [cyan]{command:<30}[/cyan] [dim]{purpose}[/dim]")
    out.write("")
    return (0 if ready else 1), answers


# -- helpers -------------------------------------------------------------


def _banner(out) -> None:
    out.write("")
    out.write("[bold]LAI setup[/bold] — a native agent for your Linux desktop")
    out.write("[dim]Four steps. Nothing is changed without asking.[/dim]")
    out.write("")


def _render(out, report: Report) -> None:
    for check in report:
        if check.status == OK:
            icon, style = "[green]✓[/green]", "dim"
        elif check.status == WARN:
            icon, style = "[yellow]![/yellow]", "yellow"
        else:
            icon, style = "[red]✗[/red]", "red"
        out.write(f"  {icon} [bold]{check.label:24}[/bold] [{style}]{check.detail}[/{style}]")


def _open_url(out, url: str) -> None:
    """Open a page in the user's browser. Never fatal — it is a convenience."""
    import subprocess  # noqa: PLC0415

    opener = shutil.which("xdg-open") or shutil.which("gio")
    if not opener:
        out.write("  [dim](no xdg-open here — open it yourself)[/dim]")
        return
    command = [opener, url] if opener.endswith("xdg-open") else [opener, "open", url]
    try:
        subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        out.write("  [dim]opened in your browser[/dim]")
    except OSError as exc:
        out.write(f"  [dim](could not open it: {exc})[/dim]")


def _plausible(provider: str, key: str) -> bool:
    """Catch an obviously-wrong paste before spending a request on it."""
    key = key.strip()
    if len(key) < 12 or " " in key:
        return False
    prefixes = {"anthropic": "sk-ant-", "openai": "sk-", "openrouter": "sk-or-"}
    expected = prefixes.get(provider, "")
    return key.startswith(expected) if expected else True


def _verify_key(provider: str, key: str) -> tuple[bool, str, str]:
    """Spend one tiny request to prove the key works. (ok, detail, model)."""
    from .agent.providers.base import Message  # noqa: PLC0415
    from .agent.providers.catalog import get as get_vendor  # noqa: PLC0415
    from .agent.providers.registry import DEFAULT_MODELS, ZAI_BASE_URL, _instantiate  # noqa: PLC0415
    from .config import ProviderConfig  # noqa: PLC0415

    vendor = get_vendor(provider)
    model = DEFAULT_MODELS.get(provider) or (vendor.default_model if vendor else "")
    base_url = ZAI_BASE_URL if provider == "zai" else (vendor.base_url if vendor else "")
    config = ProviderConfig(name=provider, api_key=key, model=model, base_url=base_url, max_tokens=16)

    instance = None
    try:
        instance = _instantiate(provider, config, None)
        turn = instance.complete([Message.user("Say OK.")], system="Reply with one word.")
        text = (turn.text or "").strip()[:40]
        return True, f"{provider}/{model} replied {text!r}" if text else f"{provider}/{model}", model
    except Exception as exc:
        return False, f"{type(exc).__name__}: {str(exc)[:200]}", model
    finally:
        if instance is not None:
            try:
                instance.close()
            except Exception:
                pass


def _has_any_backend() -> bool:
    from .agent.providers.registry import discover_credentials  # noqa: PLC0415

    try:
        return bool(discover_credentials())
    except Exception:
        return False


def needs_setup(config: Config | None = None) -> bool:
    """Would a bare ``lai`` be better off running the wizard than the interface?

    The question is not "is there a config file" but "is there a model to think
    with", because that is the only thing that makes LAI unusable. Note that a
    key saved by a previous setup lives in config.toml and is invisible to
    ``discover_credentials()``, which reads the environment — so both sources
    have to be consulted, or a configured machine would be sent back through
    the wizard forever.
    """
    config = config or load_config()
    if (config.provider.api_key or "").strip():
        return False
    return not _has_any_backend()


__all__ = ["BACKENDS", "DEMO_TASK", "Answers", "Prompt", "needs_setup", "run_setup"]
