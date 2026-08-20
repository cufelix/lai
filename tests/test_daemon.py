"""HTTP daemon: auth, endpoints, task execution and the busy guard."""

from __future__ import annotations

import json
import socket
import threading
import time
from dataclasses import replace
from http.server import ThreadingHTTPServer

import httpx
import pytest

from lai.agent.loop import RunResult
from lai.config import load_config
from lai.daemon.server import DaemonState, Handler, _load_or_create_token, serve
from lai.errors import LaiError

TOKEN = "test-token-123"


class FakeAgent:
    def __init__(self, result: RunResult, *, on_run=None) -> None:
        self.result = result
        self.interrupted = False
        self._on_run = on_run
        self.on_event = None

    def run(self, task: str) -> RunResult:
        if self._on_run:
            self._on_run(self)
        return self.result

    def interrupt(self) -> None:
        self.interrupted = True


class FakeRuntime:
    """Stands in for a real Runtime — no desktop, no model."""

    def __init__(self, config, agent_factory=None) -> None:
        self.config = config
        self.provider = type("P", (), {"name": "fake", "model": "fake-1"})()
        self.provider_error = ""
        self.registry = _tiny_registry()
        self.skills = _FakeSkills()
        self.mcp_tools = []
        self.policy = type("Pol", (), {"config": config.safety})()
        self.desktop = _FakeDesktop()
        self._agent_factory = agent_factory or (lambda **kw: FakeAgent(RunResult(status="completed", summary="ok")))

    def agent(self, **kwargs):
        agent = self._agent_factory(**kwargs)
        agent.on_event = kwargs.get("on_event")
        return agent

    def close(self) -> None:
        pass


class _FakeSkills:
    def list(self):
        return []

    def __len__(self):
        return 0


class _FakeDesktop:
    def observe(self, **kwargs):
        from lai.osl.desktop import Observation

        return Observation(active_window=None, windows=[], monitors=[])


def _tiny_registry():
    from lai.tools.base import ToolRegistry, ToolResult, ToolSpec

    registry = ToolRegistry()
    registry.register(
        ToolSpec("probe", "a probe", {"properties": {}}, lambda ctx, args: ToolResult.text("ok"))
    )
    return registry


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture
def daemon(tmp_path, monkeypatch):
    """A live daemon on a free port, backed by a fake runtime."""
    monkeypatch.setenv("LAI_HOME", str(tmp_path / "home"))
    config = load_config()
    config.ensure_dirs()
    holder: dict = {}

    def make_agent(**kwargs):
        factory = holder.get("factory")
        agent = factory(**kwargs) if factory else FakeAgent(RunResult(status="completed", summary="ok"))
        holder["last_agent"] = agent
        return agent

    runtime = FakeRuntime(config, agent_factory=make_agent)
    state = DaemonState(runtime=runtime, token=TOKEN)
    port = free_port()
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.state = state
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    holder.update({"url": f"http://127.0.0.1:{port}", "state": state, "config": config})
    # Wait for the socket to accept connections.
    for _ in range(50):
        try:
            httpx.get(f"{holder['url']}/health", timeout=1.0)
            break
        except httpx.HTTPError:
            time.sleep(0.05)
    try:
        yield holder
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def auth() -> dict:
    return {"Authorization": f"Bearer {TOKEN}"}


# -- auth ----------------------------------------------------------------


def test_health_needs_no_auth(daemon):
    response = httpx.get(f"{daemon['url']}/health", timeout=5)
    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True and body["service"] == "lai" and body["auth_required"] is True


@pytest.mark.parametrize("path", ["/status", "/observe", "/tools", "/skills", "/sessions"])
def test_endpoints_require_auth(daemon, path):
    assert httpx.get(f"{daemon['url']}{path}", timeout=5).status_code == 401


def test_wrong_token_is_rejected(daemon):
    response = httpx.get(
        f"{daemon['url']}/status", headers={"Authorization": "Bearer nope"}, timeout=5
    )
    assert response.status_code == 401


def test_token_accepted_via_the_header_alternative(daemon):
    response = httpx.get(f"{daemon['url']}/status", headers={"X-LAI-Token": TOKEN}, timeout=5)
    assert response.status_code == 200


def test_post_requires_auth(daemon):
    response = httpx.post(f"{daemon['url']}/task", json={"task": "x"}, timeout=5)
    assert response.status_code == 401


# -- read endpoints ------------------------------------------------------


def test_status_reports_the_runtime(daemon):
    body = httpx.get(f"{daemon['url']}/status", headers=auth(), timeout=5).json()
    assert body["provider"]["name"] == "fake" and body["provider"]["model"] == "fake-1"
    assert body["provider"]["chain"] == ["fake"], "a single backend is a chain of one"
    assert body["tools"] == 1
    assert body["busy"] is False
    assert "config" in body and body["config"]["provider"]["api_key"] in ("set", "unset")


def test_tools_endpoint(daemon):
    body = httpx.get(f"{daemon['url']}/tools", headers=auth(), timeout=5).json()
    assert body["tools"][0]["name"] == "probe"


def test_skills_and_sessions_endpoints(daemon):
    assert "skills" in httpx.get(f"{daemon['url']}/skills", headers=auth(), timeout=5).json()
    assert "sessions" in httpx.get(f"{daemon['url']}/sessions", headers=auth(), timeout=5).json()


def test_observe_endpoint(daemon):
    body = httpx.get(f"{daemon['url']}/observe", headers=auth(), timeout=5).json()
    assert "summary" in body and "windows" in body


def test_unknown_path_is_404(daemon):
    assert httpx.get(f"{daemon['url']}/nope", headers=auth(), timeout=5).status_code == 404
    assert httpx.post(f"{daemon['url']}/nope", headers=auth(), json={}, timeout=5).status_code == 404


# -- task execution ------------------------------------------------------


def test_blocking_task_returns_the_result(daemon):
    response = httpx.post(
        f"{daemon['url']}/task",
        headers=auth(),
        json={"task": "do it", "stream": False},
        timeout=20,
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "completed" and body["summary"] == "ok"


def test_task_requires_a_non_empty_task(daemon):
    for payload in ({}, {"task": "   "}):
        response = httpx.post(f"{daemon['url']}/task", headers=auth(), json=payload, timeout=5)
        assert response.status_code == 400


def test_streaming_task_emits_sse_events(daemon):
    def factory(**kwargs):
        emit = kwargs.get("on_event")

        def run_hook(agent):
            if emit:
                emit("step", {"step": 1, "of": 3})
                emit("tool_call", {"name": "probe", "input": {}})

        return FakeAgent(RunResult(status="completed", summary="streamed"), on_run=run_hook)

    daemon["factory"] = factory
    chunks: list[str] = []
    with httpx.stream(
        "POST", f"{daemon['url']}/task", headers=auth(), json={"task": "stream me"}, timeout=20
    ) as response:
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]
        for line in response.iter_lines():
            chunks.append(line)

    body = "\n".join(chunks)
    assert "event: step" in body
    assert "event: tool_call" in body
    assert "event: result" in body
    assert "event: end" in body
    payloads = [json.loads(line[5:]) for line in chunks if line.startswith("data:") and line[5:].strip()]
    assert any(p.get("status") == "completed" for p in payloads)


def test_busy_guard_returns_409(daemon):
    release = threading.Event()

    def factory(**kwargs):
        return FakeAgent(
            RunResult(status="completed", summary="slow"),
            on_run=lambda agent: release.wait(timeout=10),
        )

    daemon["factory"] = factory
    first = threading.Thread(
        target=lambda: httpx.post(
            f"{daemon['url']}/task", headers=auth(), json={"task": "slow", "stream": False}, timeout=30
        )
    )
    first.start()
    try:
        for _ in range(100):
            if daemon["state"].busy:
                break
            time.sleep(0.05)
        assert daemon["state"].busy, "the first task should be running"
        second = httpx.post(
            f"{daemon['url']}/task", headers=auth(), json={"task": "another"}, timeout=10
        )
        assert second.status_code == 409
        assert second.json()["error"] == "busy"
    finally:
        release.set()
        first.join(timeout=15)


def test_stop_with_nothing_running(daemon):
    body = httpx.post(f"{daemon['url']}/stop", headers=auth(), json={}, timeout=5).json()
    assert body["stopped"] is False


def test_stop_interrupts_the_running_agent(daemon):
    release = threading.Event()
    agents: list[FakeAgent] = []

    def factory(**kwargs):
        agent = FakeAgent(
            RunResult(status="interrupted"), on_run=lambda a: release.wait(timeout=10)
        )
        agents.append(agent)
        return agent

    daemon["factory"] = factory
    worker = threading.Thread(
        target=lambda: httpx.post(
            f"{daemon['url']}/task", headers=auth(), json={"task": "long", "stream": False}, timeout=30
        )
    )
    worker.start()
    try:
        for _ in range(100):
            if daemon["state"].busy:
                break
            time.sleep(0.05)
        body = httpx.post(f"{daemon['url']}/stop", headers=auth(), json={}, timeout=5).json()
        assert body["stopped"] is True
        assert agents[0].interrupted is True
    finally:
        release.set()
        worker.join(timeout=15)


def test_completed_and_failed_counters(daemon):
    httpx.post(f"{daemon['url']}/task", headers=auth(), json={"task": "a", "stream": False}, timeout=20)
    status = httpx.get(f"{daemon['url']}/status", headers=auth(), timeout=5).json()
    assert status["completed"] == 1 and status["failed"] == 0

    daemon["factory"] = lambda **kw: FakeAgent(RunResult(status="error", error="boom"))
    httpx.post(f"{daemon['url']}/task", headers=auth(), json={"task": "b", "stream": False}, timeout=20)
    status = httpx.get(f"{daemon['url']}/status", headers=auth(), timeout=5).json()
    assert status["failed"] == 1


def test_mode_override_is_applied(daemon):
    httpx.post(
        f"{daemon['url']}/task",
        headers=auth(),
        json={"task": "x", "mode": "yolo", "stream": False},
        timeout=20,
    )
    assert daemon["state"].runtime.config.safety.mode == "yolo"


def test_malformed_json_body_is_a_bad_request(daemon):
    response = httpx.post(
        f"{daemon['url']}/task",
        headers={**auth(), "Content-Type": "application/json"},
        content=b"{not json",
        timeout=5,
    )
    assert response.status_code == 400


# -- token management ----------------------------------------------------


def test_token_file_is_created_once_with_tight_permissions(tmp_path):
    config = load_config().with_overrides(home=tmp_path / "h")
    config.ensure_dirs()
    first = _load_or_create_token(config)
    assert len(first) > 20
    path = config.home / "daemon.token"
    assert path.is_file()
    assert path.stat().st_mode & 0o077 == 0, "the token must not be group/world readable"
    assert _load_or_create_token(config) == first


def test_serve_refuses_a_non_loopback_bind_without_opt_in(tmp_path):
    config = load_config().with_overrides(home=tmp_path / "h")
    with pytest.raises(LaiError, match="refusing to bind"):
        serve(config, host="0.0.0.0", port=free_port())


def test_daemon_state_busy_property():
    state = DaemonState(runtime=None, token="t")
    assert state.busy is False
    state.current_agent = object()
    assert state.busy is True


def test_claim_engage_release_cycle():
    state = DaemonState(runtime=None, token="t")
    gate = state.claim("first")
    assert gate is not None and state.busy and state.current_task == "first"
    assert state.claim("second") is None, "a second claim while busy must fail"
    agent = FakeAgent(RunResult(status="completed"))
    state.engage(agent)
    assert state.current_agent is agent, "the placeholder must be swapped for the real agent"
    other = FakeAgent(RunResult(status="completed"))
    state.engage(other)
    assert state.current_agent is agent, "engage must never overwrite a live run's agent"
    state.release()
    assert not state.busy and state.current_task == ""
    assert state.claim("again") is not None, "the desktop must be claimable again"


def test_stop_during_construction_abandons_the_run(daemon):
    """A /stop that lands before the agent exists must still stop the task."""
    build = threading.Event()

    def factory(**kwargs):
        build.wait(timeout=10)
        return FakeAgent(RunResult(status="completed"))

    daemon["factory"] = factory
    outcome: dict = {}

    def post() -> None:
        outcome["response"] = httpx.post(
            f"{daemon['url']}/task", headers=auth(), json={"task": "slow build", "stream": False}, timeout=30
        ).json()

    worker = threading.Thread(target=post)
    worker.start()
    try:
        for _ in range(200):
            if daemon["state"].busy:
                break
            time.sleep(0.05)
        assert daemon["state"].busy, "the claim should hold the desktop during construction"
        stop = httpx.post(f"{daemon['url']}/stop", headers=auth(), json={}, timeout=5).json()
        assert stop["stopped"] is True
        build.set()
        worker.join(timeout=15)
        assert outcome["response"]["status"] == "interrupted"
        assert outcome["response"]["ok"] is False
        assert not daemon["state"].busy, "the gate must be released"
    finally:
        build.set()
        worker.join(timeout=15)


def test_state_replace_is_immutable_on_config(tmp_path, monkeypatch):
    """Isolated deliberately: reading the developer's own config made this test
    depend on whatever mode they last used."""
    monkeypatch.setenv("LAI_HOME", str(tmp_path))
    monkeypatch.delenv("LAI_MODE", raising=False)
    config = load_config().with_overrides(safety=replace(load_config().safety, mode="ask"))
    changed = config.with_overrides(safety=replace(config.safety, mode="yolo"))
    assert config.safety.mode == "ask", "the original must not be mutated"
    assert changed.safety.mode == "yolo"
