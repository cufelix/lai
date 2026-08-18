"""HTTP daemon — LAI as a always-on service.

Deliberately stdlib-only (``ThreadingHTTPServer`` + Server-Sent Events): this
process can control the whole desktop, so its attack surface should be as small
as possible. It binds to loopback, requires a bearer token, and refuses to start
on a non-loopback interface without an explicit opt-in.

Endpoints
    GET  /health                  liveness + capability summary
    GET  /status                  runtime, provider, tool and skill inventory
    GET  /observe                 what the agent sees right now
    GET  /tools                   model-facing tool schemas
    GET  /skills                  available skills
    GET  /sessions                recent sessions
    GET  /channels                connector + allowlist status
    GET  /schedule                scheduled tasks
    POST /task                    run a task; SSE stream unless {"stream": false}
    POST /stop                    interrupt the running task
    POST /channels/pair           mint a one-time pairing code
    POST /channels/webhook        inbound message for the webhook connector
"""

from __future__ import annotations

import json
import secrets
import sys
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from ..agent.session import Session
from ..config import Config, load_config
from ..errors import LaiError
from ..runtime import Runtime, build_runtime

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8787
LOOPBACK = ("127.0.0.1", "::1", "localhost")


class _Claim:
    """Placeholder owner while an agent is being constructed.

    A ``/stop`` in that window is remembered, and the run that was about to
    start is abandoned instead of launched.
    """

    def __init__(self) -> None:
        self.interrupted = False

    def interrupt(self) -> None:
        self.interrupted = True


@dataclass(slots=True)
class DaemonState:
    """Shared mutable state, guarded by ``lock``.

    The desktop is a single resource with three entry points — the HTTP API,
    the channels and the scheduler — so owning it is an atomic claim, not a
    check-then-set: :meth:`claim` marks it taken under the lock, :meth:`engage`
    swaps in the real agent once built, :meth:`release` frees it.
    """

    runtime: Runtime
    token: str
    lock: threading.Lock = field(default_factory=threading.Lock)
    current_agent: Any = None
    current_task: str = ""
    channels: Any = None
    scheduler: Any = None
    scheduler_sink: Any = None
    started_at: float = field(default_factory=time.time)
    completed: int = 0
    failed: int = 0

    @property
    def busy(self) -> bool:
        return self.current_agent is not None

    def claim(self, task: str) -> _Claim | None:
        """Take the desktop atomically; ``None`` means someone else has it."""
        with self.lock:
            if self.busy:
                return None
            gate = _Claim()
            self.current_agent, self.current_task = gate, task
            return gate

    def engage(self, agent: Any) -> None:
        """Replace the placeholder with the agent that will actually run."""
        with self.lock:
            if isinstance(self.current_agent, _Claim):
                self.current_agent = agent

    def release(self) -> None:
        with self.lock:
            self.current_agent, self.current_task = None, ""


class Handler(BaseHTTPRequestHandler):
    server_version = "LAI/0.1"
    state: DaemonState  # injected on the server instance

    # -- plumbing --------------------------------------------------------

    def log_message(self, fmt: str, *args) -> None:
        # Default logging writes to stderr on every request; keep it quiet but
        # keep errors visible.
        if not str(args[1] if len(args) > 1 else "").startswith("2"):
            super().log_message(fmt, *args)

    def _authorized(self) -> bool:
        state = self.server.state  # type: ignore[attr-defined]
        if not state.token:
            return True
        header = self.headers.get("Authorization", "")
        supplied = header[7:].strip() if header.lower().startswith("bearer ") else ""
        if not supplied:
            supplied = self.headers.get("X-LAI-Token", "").strip()
        if not supplied:
            # An <img> tag cannot send a header, so the live desktop view — and
            # only that — may carry its token in the query string. Everything
            # that changes state still requires the header.
            supplied = self._query_token()
        return secrets.compare_digest(supplied, state.token)

    def _query_token(self) -> str:
        path, _, query = self.path.partition("?")
        if path.rstrip("/") != "/screen" or not query:
            return ""
        from urllib.parse import parse_qs  # noqa: PLC0415

        return (parse_qs(query).get("token") or [""])[0].strip()

    def _send(self, status: int, payload: dict | list, *, content_type: str = "application/json") -> None:
        body = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _send_page(self) -> None:
        from ..web import page  # noqa: PLC0415

        try:
            body = page()
        except OSError as exc:
            self._send(500, {"error": "no_ui", "message": str(exc)})
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        # Served straight from disk, so an upgrade must not leave a stale page
        # talking to a newer daemon.
        self.send_header("Cache-Control", "no-store")
        # The page talks only to its own origin; forbidding everything else
        # means a compromised dependency has nowhere to send the desktop.
        self.send_header(
            "Content-Security-Policy",
            "default-src 'none'; img-src 'self' data:; style-src 'unsafe-inline'; "
            "script-src 'unsafe-inline'; connect-src 'self'; form-action 'none'; base-uri 'none'",
        )
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _send_screenshot(self, runtime) -> None:
        """A still of the desktop, for the browser's live panel."""
        try:
            body = runtime.desktop.screen.grab().png
        except Exception as exc:
            self._send(503, {"error": "no_screen", "message": str(exc)})
            return
        self.send_response(200)
        self.send_header("Content-Type", "image/png")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _read_json(self) -> dict:
        try:
            length = int(self.headers.get("Content-Length", 0))
        except ValueError:
            return {}
        if length <= 0 or length > 2_000_000:
            return {}
        try:
            return json.loads(self.rfile.read(length) or b"{}")
        except (json.JSONDecodeError, OSError):
            return {}

    # -- routes ----------------------------------------------------------

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0].rstrip("/") or "/"
        if path == "/health":
            self._send(200, self._health())
            return
        if path == "/favicon.ico":
            # Browsers ask unprompted; a 401 here is noise in the log and in
            # the console, and there is nothing secret about not having one.
            self.send_response(204)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        if path in ("/", "/index.html"):
            # The page itself carries no secrets — the token reaches it through
            # the URL fragment, which the browser never sends anywhere.
            self._send_page()
            return
        if not self._authorized():
            self._send(401, {"error": "unauthorized", "hint": "send Authorization: Bearer <token>"})
            return

        state: DaemonState = self.server.state  # type: ignore[attr-defined]
        runtime = state.runtime
        try:
            if path == "/status":
                self._send(200, self._status())
            elif path == "/observe":
                observation = runtime.desktop.observe(screenshot=False, scope="focused")
                self._send(200, {"summary": observation.summary(), **observation.to_dict()})
            elif path == "/tools":
                self._send(200, {"tools": runtime.registry.to_anthropic()})
            elif path == "/skills":
                self._send(200, {"skills": [s.to_dict() for s in runtime.skills.list()]})
            elif path == "/sessions":
                self._send(200, {"sessions": Session.list_sessions(runtime.config.sessions_dir)})
            elif path == "/schedule":
                store = getattr(runtime, "task_store", None)
                tasks = [t.to_dict() for t in store.list()] if store else []
                self._send(200, {"tasks": tasks, "running": state.scheduler is not None})
            elif path == "/channels":
                manager = state.channels
                self._send(200, manager.status() if manager else {"channels": [], "enabled": False})
            elif path == "/models":
                from ..chat import backends as backend_tools  # noqa: PLC0415

                found = backend_tools.catalogue()
                self._send(200, {
                    "active": runtime.provider.name if runtime.provider else "",
                    "backends": [b.to_dict() for b in found],
                })
            elif path == "/screen":
                self._send_screenshot(runtime)
            else:
                self._send(404, {"error": "not_found", "path": path})
        except LaiError as exc:
            self._send(500, exc.to_dict())
        except Exception as exc:  # a handler crash must not kill the daemon
            self._send(500, {"error": type(exc).__name__, "message": str(exc)})

    def do_POST(self) -> None:
        path = self.path.split("?", 1)[0].rstrip("/") or "/"
        if not self._authorized():
            self._send(401, {"error": "unauthorized"})
            return
        state: DaemonState = self.server.state  # type: ignore[attr-defined]

        if path == "/channels/pair":
            manager = state.channels
            if manager is None:
                self._send(400, {"error": "no_channels", "message": "no connectors are running"})
                return
            code = manager.access.new_pairing_code()
            self._send(200, {"code": code, "expires_in": 900, "send": f"/pair {code}"})
            return

        if path == "/channels/webhook":
            manager = state.channels
            channel = (manager.channels.get("webhook") if manager else None)
            if channel is None:
                self._send(400, {"error": "no_webhook", "message": "the webhook connector is not running"})
                return
            payload = self._read_json()
            signature = self.headers.get("X-LAI-Signature", "")
            accepted = channel.deliver(payload, signature=signature)
            if accepted is None:
                self._send(400, {"error": "rejected", "message": "bad signature or empty text"})
            else:
                self._send(202, {"accepted": True, "route": accepted.route})
            return

        if path in ("/provider", "/mode"):
            from ..chat import backends as backend_tools  # noqa: PLC0415

            body = self._read_json()
            try:
                if path == "/mode":
                    self._send(200, {"mode": backend_tools.set_mode(state.runtime, str(body.get("mode", "")))})
                elif body.get("fallback") is not None:
                    wanted = str(body["fallback"]).strip().lower()
                    chain = backend_tools.set_fallback(
                        state.runtime, [] if wanted in ("off", "none", "") else [wanted]
                    )
                    self._send(200, {"fallback": chain})
                else:
                    label = backend_tools.use(state.runtime, str(body.get("name", "")),
                                              model=str(body.get("model", "")))
                    self._send(200, {"provider": label})
            except (LaiError, ValueError) as exc:
                self._send(400, {"error": "rejected", "message": str(exc)})
            return

        if path == "/stop":
            with state.lock:
                agent = state.current_agent
            if agent is None:
                self._send(200, {"stopped": False, "reason": "nothing running"})
            else:
                agent.interrupt()
                self._send(200, {"stopped": True, "task": state.current_task})
            return

        if path != "/task":
            self._send(404, {"error": "not_found", "path": path})
            return

        body = self._read_json()
        task = str(body.get("task", "")).strip()
        if not task:
            self._send(400, {"error": "bad_request", "message": "'task' is required"})
            return

        gate = state.claim(task)
        if gate is None:
            self._send(409, {"error": "busy", "current_task": state.current_task})
            return

        if body.get("stream", True):
            self._run_streaming(state, task, body, gate)
        else:
            self._run_blocking(state, task, body, gate)

    # -- task execution --------------------------------------------------

    def _build_agent(self, state: DaemonState, body: dict, on_event=None):
        runtime = state.runtime
        overrides = {}
        if body.get("mode"):
            from dataclasses import replace  # noqa: PLC0415

            overrides["safety"] = replace(runtime.config.safety, mode=str(body["mode"]))
        if overrides:
            runtime.config = runtime.config.with_overrides(**overrides)
            runtime.policy.config = runtime.config.safety

        session = Session()
        session.bind(runtime.config.sessions_dir)
        agent = runtime.agent(session=session, on_event=on_event, approver=lambda *_: False)
        if body.get("steps"):
            agent.config = agent.config.with_overrides(
                limits=_replace_limits(agent.config.limits, int(body["steps"]))
            )
        return agent

    def _run_blocking(self, state: DaemonState, task: str, body: dict, gate: _Claim) -> None:
        try:
            agent = self._build_agent(state, body)
        except Exception as exc:
            state.release()
            self._send(500, {"error": type(exc).__name__, "message": str(exc)})
            return
        state.engage(agent)
        if gate.interrupted:  # /stop arrived before the agent existed
            state.release()
            self._send(200, {"status": "interrupted", "ok": False, "error": "stopped before it started"})
            return
        try:
            result = agent.run(task)
            with state.lock:
                state.completed += 1 if result.ok else 0
                state.failed += 0 if result.ok else 1
            self._send(200, result.to_dict())
        except Exception as exc:
            self._send(500, {"error": type(exc).__name__, "message": str(exc)})
        finally:
            state.release()

    def _run_streaming(self, state: DaemonState, task: str, body: dict, gate: _Claim) -> None:
        # An SSE body has no Content-Length, so the client can only know the
        # response ended when the connection closes. Announcing keep-alive here
        # leaves every client hanging after the final event.
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Accel-Buffering", "no")
        self.send_header("Connection", "close")
        self.close_connection = True
        self.end_headers()

        closed = threading.Event()

        def emit(kind: str, payload: dict) -> None:
            if closed.is_set():
                return
            try:
                data = json.dumps({"kind": kind, **payload}, ensure_ascii=False, default=str)
                self.wfile.write(f"event: {kind}\ndata: {data}\n\n".encode())
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, ValueError):
                closed.set()

        try:
            agent = self._build_agent(state, body, on_event=emit)
        except Exception as exc:
            state.release()
            emit("error", {"error": type(exc).__name__, "message": str(exc)})
            return
        state.engage(agent)
        if gate.interrupted:
            state.release()
            emit("result", {"status": "interrupted", "ok": False, "error": "stopped before it started"})
            self._end_stream(closed)
            return
        try:
            result = agent.run(task)
            with state.lock:
                state.completed += 1 if result.ok else 0
                state.failed += 0 if result.ok else 1
            emit("result", result.to_dict())
        except Exception as exc:
            emit("error", {"error": type(exc).__name__, "message": str(exc)})
        finally:
            state.release()
            self._end_stream(closed)

    def _end_stream(self, closed: threading.Event) -> None:
        if closed.is_set():
            return
        try:
            self.wfile.write(b"event: end\ndata: {}\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, ValueError):
            pass

    # -- payloads --------------------------------------------------------

    def _health(self) -> dict:
        state: DaemonState = self.server.state  # type: ignore[attr-defined]
        return {
            "ok": True,
            "service": "lai",
            "uptime": round(time.time() - state.started_at, 1),
            "busy": state.busy,
            "auth_required": bool(state.token),
        }

    def _status(self) -> dict:
        state: DaemonState = self.server.state  # type: ignore[attr-defined]
        runtime = state.runtime
        provider = runtime.provider
        return {
            **self._health(),
            "current_task": state.current_task,
            "completed": state.completed,
            "failed": state.failed,
            "provider": {
                "name": provider.name,
                "model": provider.model,
                "chain": list(getattr(provider, "chain", []) or [provider.name]),
                "failures": dict(getattr(provider, "failures", {}) or {}),
                "fallback": list(runtime.config.provider.fallback),
            } if provider else None,
            "provider_error": runtime.provider_error,
            "mode": runtime.config.safety.mode,
            "tools": len(runtime.registry),
            "skills": len(runtime.skills),
            "mcp_tools": len(runtime.mcp_tools),
            "config": runtime.config.redacted(),
        }


def _start_scheduler(runtime, state: DaemonState):
    """Run scheduled tasks in the background.

    A scheduled run must not fight a user-initiated one for the desktop, so it
    is skipped while the daemon is busy rather than queued — a missed hourly
    check is cheaper than two agents racing for the mouse.
    """
    store = getattr(runtime, "task_store", None)
    if store is None:
        return None
    try:
        from ..scheduler import Scheduler  # noqa: PLC0415
    except Exception:
        return None

    def fire(task) -> None:
        if runtime.provider is None:
            return
        label = f"[scheduled] {task.name}"
        if not state.claim(label):
            print(f"[scheduler] skipping {task.name!r}: a task is already running")
            return
        agent = runtime.agent(approver=lambda *_: False)
        state.engage(agent)
        try:
            result = agent.run(task.task)
            print(f"[scheduler] {task.name}: {result.status} in {result.steps} steps")
            sink = state.scheduler_sink
            if sink is not None and result.summary:
                sink.broadcast(f"⏰ {task.name}\n\n{result.summary}")
        finally:
            state.release()

    try:
        scheduler = Scheduler(store, fire)
        scheduler.start()
    except Exception as exc:  # scheduling is optional
        print(f"[scheduler] not started: {exc}")
        return None
    state.scheduler = scheduler
    pending = len(store.list()) if hasattr(store, "list") else 0
    if pending:
        print(f"  schedule : {pending} task(s)")
    return scheduler


def _replace_limits(limits, max_steps: int):
    from dataclasses import replace  # noqa: PLC0415

    return replace(limits, max_steps=max_steps)


def serve(
    config: Config | None = None,
    *,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    token: str | None = None,
    allow_remote: bool = False,
    with_mcp: bool = True,
) -> None:
    """Run the daemon until interrupted."""
    config = config or load_config()
    # Under systemd, a pipe or a container, stdout is block-buffered, so the
    # startup banner (including the pairing code) stays invisible until the
    # process exits. Line buffering makes it appear when it is useful.
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except (AttributeError, OSError):
        pass

    if host not in LOOPBACK and not allow_remote:
        raise LaiError(
            f"refusing to bind {host}: this API controls the desktop",
            detail="pass allow_remote=True (or --allow-remote) if you really mean it, "
            "and set a strong token",
        )

    runtime = build_runtime(config, with_mcp=with_mcp)
    resolved_token = token if token is not None else _load_or_create_token(config)
    state = DaemonState(runtime=runtime, token=resolved_token)

    scheduler = _start_scheduler(runtime, state)

    manager = None
    if config.channels.enabled:
        from ..channels import build_manager  # noqa: PLC0415

        def _on_channel_event(kind: str, payload: dict) -> None:
            # The manager drives the desktop gate (claim/engage/release) itself;
            # the daemon only folds channel runs into its /status counters.
            if kind == "run_finished":
                with state.lock:
                    state.completed += 1 if payload.get("ok") else 0
                    state.failed += 0 if payload.get("ok") else 1

        manager = build_manager(runtime, on_event=_on_channel_event, desktop=state)
        state.channels = manager
        state.scheduler_sink = manager

    server = ThreadingHTTPServer((host, port), Handler)
    server.state = state  # type: ignore[attr-defined]
    server.daemon_threads = True

    provider = runtime.provider
    print(f"LAI daemon on http://{host}:{port}")
    print(
        f"  provider : {provider.name}/{provider.model}"
        if provider
        else f"  provider : none ({runtime.provider_error})"
    )
    print(f"  mode     : {config.safety.mode}")
    print(f"  tools    : {len(runtime.registry)}   skills: {len(runtime.skills)}")
    print(f"  token    : {config.home / 'daemon.token'}")
    if manager is not None:
        started = manager.start()
        print(f"  channels : {', '.join(started) or 'none started'}")
        if not manager.access.principals():
            code = manager.access.new_pairing_code()
            print(f"\n  Nobody is authorised yet. Send this to your bot:  /pair {code}")
    print(f"\n  curl -H 'Authorization: Bearer $(cat {config.home / 'daemon.token'})' \\")
    print(f"       -d '{{\"task\":\"open the calculator\"}}' http://{host}:{port}/task\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down")
    finally:
        if scheduler is not None:
            scheduler.stop()
        if manager is not None:
            manager.stop()
        server.shutdown()
        server.server_close()
        runtime.close()


def _load_or_create_token(config: Config) -> str:
    """Persist a token with restrictive permissions so only this user can call in."""
    path = Path(config.home) / "daemon.token"
    if path.is_file():
        existing = path.read_text(encoding="utf-8").strip()
        if existing:
            return existing
    token = secrets.token_urlsafe(32)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(token + "\n", encoding="utf-8")
    try:
        path.chmod(0o600)
    except OSError:
        pass
    return token
