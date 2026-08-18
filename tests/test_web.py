"""The browser interface.

The page is served to anyone who asks, so the load-bearing property is that it
carries no authority: every endpoint that reads the desktop or changes state
still demands the token, and the token itself only ever travels in the URL
fragment, which browsers do not send to servers.
"""

from __future__ import annotations

import socket
import threading
import time
from http.server import ThreadingHTTPServer

import httpx
import pytest

from lai.config import load_config
from lai.daemon.server import DaemonState, Handler
from lai.web import page, url

TOKEN = "web-test-token"


class FakeScreen:
    def grab(self, *args, **kwargs):
        return type("Shot", (), {"png": b"\x89PNG\r\n\x1a\nfake"})()


class FakeDesktop:
    screen = FakeScreen()

    def observe(self, **kwargs):
        from lai.osl.desktop import Observation

        return Observation(active_window=None, windows=[], monitors=[])


class FakeProvider:
    def __init__(self, name="zai", model="glm-5"):
        self.name, self.model = name, model

    def close(self):
        pass


class FakeRuntime:
    def __init__(self, config):
        self.config = config
        self.provider = FakeProvider()
        self.provider_error = ""
        self.registry = type("R", (), {"__len__": lambda self: 4, "to_anthropic": lambda self: []})()
        self.skills = type("S", (), {"__len__": lambda self: 0, "list": lambda self: []})()
        self.mcp_tools = []
        self.policy = type("P", (), {"config": config.safety})()
        self.desktop = FakeDesktop()

    def agent(self, **kwargs):
        raise AssertionError("no task should be run by these tests")

    def close(self):
        pass


@pytest.fixture
def web(tmp_path, monkeypatch):
    monkeypatch.setenv("LAI_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("LAI_FALLBACK", raising=False)
    config = load_config()
    config.ensure_dirs()
    runtime = FakeRuntime(config)
    state = DaemonState(runtime=runtime, token=TOKEN)

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.state = state
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{port}"
    for _ in range(50):
        try:
            httpx.get(f"{base}/health", timeout=1.0)
            break
        except httpx.HTTPError:
            time.sleep(0.05)
    try:
        yield {"url": base, "state": state, "runtime": runtime}
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def auth() -> dict:
    return {"Authorization": f"Bearer {TOKEN}"}


# -- the page ------------------------------------------------------------


def test_the_page_is_served_without_a_token(web):
    """It has to be: the browser fetches it before it has read the fragment."""
    response = httpx.get(f"{web['url']}/", timeout=5)
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "<title>LAI</title>" in response.text


def test_the_page_carries_no_credentials(web):
    body = httpx.get(f"{web['url']}/", timeout=5).text
    assert TOKEN not in body, "serving the token to an unauthenticated caller would hand over the desktop"


def test_the_page_is_locked_down_by_a_content_security_policy(web):
    policy = httpx.get(f"{web['url']}/", timeout=5).headers.get("content-security-policy", "")
    assert "default-src 'none'" in policy
    assert "connect-src 'self'" in policy, "the page must not be able to phone anywhere else"


def test_the_url_puts_the_token_in_the_fragment():
    """Fragments never reach a server, so the token cannot land in a log."""
    assert url("127.0.0.1", 8787, "abc") == "http://127.0.0.1:8787/#abc"
    assert "#" not in url("127.0.0.1", 8787, "")


def test_the_page_ships_with_the_package():
    assert b"<title>LAI</title>" in page()


# -- endpoints the page uses ---------------------------------------------


def test_status_exposes_the_failover_chain(web):
    web["runtime"].provider = type("P", (), {
        "name": "ollama", "model": "q", "chain": ["zai", "ollama"],
        "failures": {"zai": "429 quota"},
    })()
    body = httpx.get(f"{web['url']}/status", headers=auth(), timeout=5).json()
    assert body["provider"]["chain"] == ["zai", "ollama"]
    assert body["provider"]["failures"]["zai"] == "429 quota"


def test_models_lists_backends_and_marks_the_active_one(web, monkeypatch):
    from lai.models import READY, Backend

    monkeypatch.setattr(
        "lai.models.discover",
        lambda **kwargs: [Backend(name="zai", label="zai", kind="api", status=READY,
                                  detail="via ZAI_API_KEY", model="glm-5")],
    )
    body = httpx.get(f"{web['url']}/models", headers=auth(), timeout=5).json()
    assert body["active"] == "zai"
    assert body["backends"][0]["name"] == "zai"


def test_the_browser_can_change_the_permission_mode(web, tmp_path):
    from lai import config_file

    response = httpx.post(f"{web['url']}/mode", headers=auth(), json={"mode": "readonly"}, timeout=5)
    assert response.status_code == 200 and response.json()["mode"] == "readonly"
    assert web["runtime"].policy.config.mode == "readonly", "the live policy must change, not just the file"
    assert config_file.read(tmp_path / "home")["safety"]["mode"] == "readonly"


def test_an_impossible_mode_is_refused(web):
    response = httpx.post(f"{web['url']}/mode", headers=auth(), json={"mode": "banana"}, timeout=5)
    assert response.status_code == 400


def test_the_browser_can_switch_backend(web, monkeypatch):
    monkeypatch.setattr(
        "lai.agent.providers.registry.build_provider",
        lambda config, **kwargs: FakeProvider(name=config.name, model="m"),
    )
    response = httpx.post(f"{web['url']}/provider", headers=auth(), json={"name": "ollama"}, timeout=5)
    assert response.status_code == 200 and response.json()["provider"] == "ollama/m"
    assert web["runtime"].provider.name == "ollama"


def test_a_backend_that_cannot_be_built_is_a_clean_400(web, monkeypatch):
    from lai.errors import ProviderError

    def refuse(config, **kwargs):
        raise ProviderError("openai: no API key configured")

    monkeypatch.setattr("lai.agent.providers.registry.build_provider", refuse)
    response = httpx.post(f"{web['url']}/provider", headers=auth(), json={"name": "openai"}, timeout=5)
    assert response.status_code == 400
    assert "no API key" in response.json()["message"]
    assert web["runtime"].provider.name == "zai", "a failed switch must leave the working one alone"


def test_failover_can_be_turned_off_from_the_browser(web):
    response = httpx.post(f"{web['url']}/provider", headers=auth(), json={"fallback": "off"}, timeout=5)
    assert response.status_code == 200 and response.json()["fallback"] == []
    assert web["runtime"].config.provider.fallback == ()


# -- the live desktop view -----------------------------------------------


def test_the_screen_endpoint_returns_a_png(web):
    response = httpx.get(f"{web['url']}/screen", headers=auth(), timeout=5)
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.content.startswith(b"\x89PNG")


def test_the_screen_accepts_a_query_token_because_an_image_tag_cannot_send_headers(web):
    response = httpx.get(f"{web['url']}/screen?token={TOKEN}", timeout=5)
    assert response.status_code == 200


def test_a_query_token_does_not_unlock_anything_else(web):
    """Only the image endpoint may take its token that way."""
    for path in ("/status", "/models", "/observe"):
        assert httpx.get(f"{web['url']}{path}?token={TOKEN}", timeout=5).status_code == 401


def test_the_screen_still_refuses_a_wrong_token(web):
    assert httpx.get(f"{web['url']}/screen?token=nope", timeout=5).status_code == 401
    assert httpx.get(f"{web['url']}/screen", timeout=5).status_code == 401
