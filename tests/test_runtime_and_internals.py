"""Runtime assembly, the async bridge, archive installs, and provider retries."""

from __future__ import annotations

import io
import tarfile
import zipfile

import httpx
import pytest

from lai.errors import ProviderError, SkillError
from lai.runtime import build_runtime

SKILL_MD = "---\nname: packaged\ndescription: shipped in an archive\n---\nDo the thing.\n"


@pytest.fixture(autouse=True)
def isolated_home(monkeypatch, tmp_path):
    monkeypatch.setenv("LAI_HOME", str(tmp_path / "home"))


# -- runtime -------------------------------------------------------------


def test_runtime_assembles_without_a_provider():
    runtime = build_runtime(with_provider=False, with_mcp=False)
    try:
        assert runtime.provider is None
        assert len(runtime.registry) > 20
        assert runtime.policy is not None and runtime.audit is not None
        assert runtime.desktop is not None
        assert runtime.config.home.is_dir()
    finally:
        runtime.close()


def test_runtime_registers_control_and_skill_tools():
    runtime = build_runtime(with_provider=False, with_mcp=False)
    try:
        for name in ("task_complete", "task_blocked", "plan_update", "skill_list", "skill_load"):
            assert name in runtime.registry
    finally:
        runtime.close()


def test_runtime_records_a_provider_failure_instead_of_raising(monkeypatch):
    monkeypatch.setattr(
        "lai.runtime.build_provider",
        lambda config, **kwargs: (_ for _ in ()).throw(ProviderError("no key anywhere")),
    )
    runtime = build_runtime(with_mcp=False)
    try:
        assert runtime.provider is None
        assert "no key anywhere" in runtime.provider_error
    finally:
        runtime.close()


def test_agent_without_a_provider_raises_a_helpful_error(monkeypatch):
    monkeypatch.setattr(
        "lai.runtime.build_provider",
        lambda config, **kwargs: (_ for _ in ()).throw(ProviderError("nothing configured")),
    )
    runtime = build_runtime(with_mcp=False)
    try:
        with pytest.raises(ProviderError, match="no model provider"):
            runtime.agent()
    finally:
        runtime.close()


def test_group_restriction_limits_the_registry():
    runtime = build_runtime(with_provider=False, with_mcp=False, groups={"ui"})
    try:
        # Control and skill tools are added after the group filter by design.
        assert any(name.startswith("ui_") for name in runtime.registry.names)
        assert not any(name.startswith("computer_") for name in runtime.registry.names)
    finally:
        runtime.close()


def test_broken_mcp_layer_does_not_break_the_runtime(monkeypatch):
    monkeypatch.setattr(
        "lai.mcp.client.load_mcp_configs",
        lambda config, cwd: (_ for _ in ()).throw(RuntimeError("config exploded")),
    )
    runtime = build_runtime(with_provider=False, with_mcp=True)
    try:
        assert runtime.mcp_tools == []
        assert len(runtime.registry) > 20  # desktop tools still present
    finally:
        runtime.close()


def test_runtime_close_is_idempotent():
    runtime = build_runtime(with_provider=False, with_mcp=False)
    runtime.close()
    runtime.close()


def test_agent_binds_a_session_to_disk():
    runtime = build_runtime(with_provider=False, with_mcp=False)
    try:
        runtime.provider = type("P", (), {"name": "fake", "model": "m", "close": lambda self: None})()
        agent = runtime.agent()
        assert agent.session.path is not None
        assert agent.session.path.parent == runtime.config.sessions_dir
    finally:
        runtime.close()


# -- async bridge --------------------------------------------------------


def test_bridge_runs_a_coroutine_from_sync_code():
    from lai.mcp.bridge import run_sync

    async def add():
        return 40 + 2

    assert run_sync(add()) == 42


def test_bridge_propagates_exceptions():
    from lai.mcp.bridge import run_sync

    async def boom():
        raise ValueError("inner failure")

    with pytest.raises(ValueError, match="inner failure"):
        run_sync(boom())


def test_bridge_times_out():
    import asyncio

    from lai.mcp.bridge import run_sync

    async def slow():
        await asyncio.sleep(10)

    with pytest.raises(TimeoutError):
        run_sync(slow(), timeout=0.2)


def test_bridge_loop_is_reused():
    from lai.mcp.bridge import get_loop

    assert get_loop() is get_loop()


def test_bridge_spawn_returns_a_future():
    from lai.mcp.bridge import spawn

    async def work():
        return "done"

    assert spawn(work()).result(timeout=5) == "done"


# -- skill archives ------------------------------------------------------


def make_zip() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("packaged/SKILL.md", SKILL_MD)
        archive.writestr("packaged/scripts/run.sh", "#!/bin/sh\necho hi\n")
    return buffer.getvalue()


def make_tar() -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        data = SKILL_MD.encode()
        info = tarfile.TarInfo("packaged/SKILL.md")
        info.size = len(data)
        archive.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


def install_from_bytes(monkeypatch, payload: bytes, url: str, target):
    import lai.skills.install as install_module

    class FakeResponse:
        content = payload

        def raise_for_status(self):
            return None

    monkeypatch.setattr(install_module.httpx, "get", lambda *a, **kw: FakeResponse())
    return install_module.install(url, target)


def test_install_from_a_zip_url(monkeypatch, tmp_path):
    result = install_from_bytes(monkeypatch, make_zip(), "https://example.test/s.zip", tmp_path)
    assert result.installed == ("packaged",)
    assert (tmp_path / "packaged" / "SKILL.md").is_file()
    assert (tmp_path / "packaged" / "scripts" / "run.sh").is_file()


def test_install_from_a_tarball_url(monkeypatch, tmp_path):
    result = install_from_bytes(monkeypatch, make_tar(), "https://example.test/s.tar.gz", tmp_path)
    assert result.installed == ("packaged",)


def test_install_rejects_a_corrupt_zip(monkeypatch, tmp_path):
    with pytest.raises(SkillError, match="not a valid zip"):
        install_from_bytes(monkeypatch, b"definitely not a zip", "https://example.test/s.zip", tmp_path)


def test_install_rejects_a_corrupt_tarball(monkeypatch, tmp_path):
    with pytest.raises(SkillError, match="not a valid tar"):
        install_from_bytes(monkeypatch, b"nope", "https://example.test/s.tar.gz", tmp_path)


def test_install_rejects_an_oversized_archive(monkeypatch, tmp_path):
    import lai.skills.install as install_module

    payload = b"x" * (install_module.MAX_ARCHIVE_BYTES + 1)
    with pytest.raises(SkillError, match="too large"):
        install_from_bytes(monkeypatch, payload, "https://example.test/s.zip", tmp_path)


def test_install_reports_a_download_failure(monkeypatch, tmp_path):
    import lai.skills.install as install_module

    def fail(*a, **kw):
        raise httpx.ConnectError("dns failure")

    monkeypatch.setattr(install_module.httpx, "get", fail)
    with pytest.raises(SkillError, match="download failed"):
        install_module.install("https://example.test/s.zip", tmp_path)


def test_install_reports_a_git_failure(monkeypatch, tmp_path):
    import subprocess

    import lai.skills.install as install_module

    monkeypatch.setattr(install_module.shutil, "which", lambda name: "/usr/bin/git")

    def fake_run(*a, **kw):
        return subprocess.CompletedProcess(a[0], 128, "", "repository not found")

    monkeypatch.setattr(install_module.subprocess, "run", fake_run)
    with pytest.raises(SkillError, match="git clone failed"):
        install_module.install("https://github.test/nobody/nothing.git", tmp_path)


def test_github_shorthand_is_expanded(monkeypatch, tmp_path):
    import lai.skills.install as install_module

    seen: dict = {}
    monkeypatch.setattr(install_module.shutil, "which", lambda name: "/usr/bin/git")

    def fake_run(cmd, *a, **kw):
        import subprocess

        seen["url"] = next(a for a in cmd if a.startswith("https://"))
        return subprocess.CompletedProcess(cmd, 1, "", "stopped here")

    monkeypatch.setattr(install_module.subprocess, "run", fake_run)
    with pytest.raises(SkillError):
        install_module.install("owner/repo", tmp_path)
    assert seen["url"] == "https://github.com/owner/repo.git"


def test_install_without_git_available(monkeypatch, tmp_path):
    import lai.skills.install as install_module

    monkeypatch.setattr(install_module.shutil, "which", lambda name: None)
    with pytest.raises(SkillError, match="git is not installed"):
        install_module.install("https://github.test/a/b.git", tmp_path)


# -- anthropic provider error handling -----------------------------------


def build_provider():
    from lai.agent.providers.anthropic_api import AnthropicProvider

    return AnthropicProvider(api_key="k", model="m", base_url="https://api.test")


def test_provider_requires_a_key():
    from lai.agent.providers.anthropic_api import AnthropicProvider

    with pytest.raises(ProviderError, match="no API key"):
        AnthropicProvider(api_key="", model="m")


def test_provider_sends_both_auth_header_styles():
    provider = build_provider()
    try:
        headers = provider._client.headers
        assert headers["x-api-key"] == "k"
        assert headers["authorization"] == "Bearer k"
        assert headers["anthropic-version"]
    finally:
        provider.close()


def test_non_retryable_status_raises_with_detail(monkeypatch):
    provider = build_provider()

    class Response:
        status_code = 401
        headers: dict = {}

        def json(self):
            return {"error": {"type": "authentication_error", "message": "bad key"}}

    monkeypatch.setattr(provider._client, "post", lambda *a, **kw: Response())
    try:
        with pytest.raises(ProviderError) as info:
            provider.complete([])
        assert "401" in str(info.value)
        assert "bad key" in (info.value.detail or "")
    finally:
        provider.close()


def test_retryable_status_is_retried_then_succeeds(monkeypatch):
    provider = build_provider()
    attempts = {"n": 0}

    class Response:
        def __init__(self, status):
            self.status_code = status
            self.headers = {"retry-after": "0"}

        def json(self):
            if self.status_code == 200:
                return {"content": [{"type": "text", "text": "recovered"}], "stop_reason": "end_turn"}
            return {"error": {"type": "overloaded_error", "message": "busy"}}

    def post(*a, **kw):
        attempts["n"] += 1
        return Response(200 if attempts["n"] > 1 else 529)

    monkeypatch.setattr(provider._client, "post", post)
    monkeypatch.setattr("lai.agent.providers.anthropic_api._backoff", lambda *a, **kw: None)
    try:
        assert provider.complete([]).text == "recovered"
        assert attempts["n"] == 2
    finally:
        provider.close()


def test_network_errors_are_retried_then_reported(monkeypatch):
    provider = build_provider()

    def post(*a, **kw):
        raise httpx.ConnectError("no route to host")

    monkeypatch.setattr(provider._client, "post", post)
    monkeypatch.setattr("lai.agent.providers.anthropic_api._backoff", lambda *a, **kw: None)
    try:
        with pytest.raises(ProviderError, match="failed after"):
            provider.complete([])
    finally:
        provider.close()


def test_thinking_budget_replaces_temperature():
    from lai.agent.providers.anthropic_api import AnthropicProvider

    provider = AnthropicProvider(api_key="k", model="m", thinking_budget=2048)
    try:
        payload = provider._payload([], "sys", None)
        assert payload["thinking"] == {"type": "enabled", "budget_tokens": 2048}
        assert "temperature" not in payload, "extended thinking requires the default temperature"
        # The block may also carry a cache marker; what matters here is the text.
        assert payload["system"][0]["text"] == "sys"
    finally:
        provider.close()


def test_payload_omits_empty_optional_fields():
    provider = build_provider()
    try:
        payload = provider._payload([], "", None)
        assert "system" not in payload and "tools" not in payload
        assert payload["temperature"] == 1.0
    finally:
        provider.close()
