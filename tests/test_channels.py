"""Channels: access control, message shapes, routing, and the Telegram wire format."""

from __future__ import annotations

import json
import threading
import time

import pytest

from lai.channels.access import Access, AccessPolicy, Principal
from lai.channels.base import Attachment, BaseChannel, IncomingMessage, OutgoingMessage
from lai.channels.manager import ChannelManager
from lai.channels.telegram import TelegramChannel, _format, _split
from lai.channels.webhook import LocalChannel, WebhookChannel


def msg(text: str = "hello", *, sender: str = "u1", channel: str = "local", chat: str = "c1") -> IncomingMessage:
    return IncomingMessage(channel=channel, sender=sender, chat=chat, text=text, message_id="1")


# -- message primitives --------------------------------------------------


def test_command_parsing():
    assert msg("/status").command == ("status", "")
    assert msg("/mode auto").command == ("mode", "auto")
    assert msg("/pair  123456 ").command == ("pair", "123456")
    assert msg("not a command").command == ("", "")
    assert msg("/status@my_bot").command == ("status", ""), "group-mention suffix must be stripped"


def test_is_command_and_route():
    assert msg("/help").is_command
    assert not msg("hello").is_command
    assert msg(chat="42", channel="telegram").route == "telegram:42"


def test_outgoing_image_helper():
    out = OutgoingMessage.image("c1", b"PNGDATA", caption="look")
    assert out.attachments[0].kind == "image"
    assert out.attachments[0].data == b"PNGDATA"
    assert out.attachments[0].caption == "look"


def test_attachment_size():
    assert Attachment("file", b"12345").size == 5


def test_base_channel_contains_a_broken_handler():
    channel = BaseChannel()
    channel.start(lambda m: (_ for _ in ()).throw(RuntimeError("handler exploded")))
    channel._deliver(msg())
    assert channel.errors and "handler exploded" in channel.errors[0]


def test_base_channel_status_shape():
    channel = BaseChannel()
    assert set(channel.status()) == {"name", "available", "running", "recent_errors"}


# -- access control ------------------------------------------------------


def test_unknown_sender_is_denied():
    policy = AccessPolicy()
    decision = policy.check(msg())
    assert decision.access is Access.DENIED
    assert not decision.allowed


def test_allowlisted_sender_is_permitted():
    policy = AccessPolicy()
    policy.allow("local", "u1")
    assert policy.check(msg()).allowed


def test_open_access_lets_anyone_in():
    policy = AccessPolicy(open_access=True)
    assert policy.check(msg(sender="stranger")).allowed


def test_pairing_flow_first_user_becomes_admin():
    policy = AccessPolicy()
    code = policy.new_pairing_code()
    assert len(code) == 6 and code.isdigit()
    assert policy.check(msg(f"/pair {code}")).access is Access.PAIRING

    outcome = policy.redeem(code, "local", "u1", name="Felix")
    assert outcome.allowed and outcome.admin
    assert policy.check(msg()).allowed

    # Second person pairs as a plain user.
    code2 = policy.new_pairing_code()
    second = policy.redeem(code2, "local", "u2")
    assert second.allowed and not second.admin


def test_pairing_code_is_single_use():
    policy = AccessPolicy()
    code = policy.new_pairing_code()
    assert policy.redeem(code, "local", "u1").allowed
    assert not policy.redeem(code, "local", "u2").allowed


def test_wrong_code_is_refused_and_rate_limited():
    policy = AccessPolicy()
    code = policy.new_pairing_code()
    wrong = "000000" if code != "000000" else "111111"
    for _ in range(5):
        assert not policy.redeem(wrong, "local", "attacker").allowed
    # The code is cancelled after too many attempts, so even the right one fails.
    assert not policy.redeem(code, "local", "attacker").allowed
    assert not policy.pairing_active


def test_expired_pairing_code_is_refused():
    policy = AccessPolicy()
    code = policy.new_pairing_code(ttl=-1)
    assert not policy.pairing_active
    assert not policy.redeem(code, "local", "u1").allowed


def test_revoke():
    policy = AccessPolicy()
    policy.allow("local", "u1")
    assert policy.revoke("local", "u1")
    assert not policy.revoke("local", "u1")
    assert not policy.check(msg()).allowed


def test_policy_persists_and_reloads(tmp_path):
    path = tmp_path / "channels.json"
    first = AccessPolicy(path)
    first.allow("telegram", "12345", name="Felix", admin=True)

    second = AccessPolicy(path)
    principal = second.get("telegram", "12345")
    assert principal is not None and principal.admin and principal.name == "Felix"
    assert path.stat().st_mode & 0o077 == 0, "the allowlist must not be world-readable"


def test_corrupt_policy_file_fails_closed(tmp_path):
    path = tmp_path / "channels.json"
    path.write_text("{not json", encoding="utf-8")
    policy = AccessPolicy(path)
    assert policy.principals() == []
    assert not policy.check(msg()).allowed


def test_principal_roundtrip():
    principal = Principal("telegram", "1", "Felix", admin=True)
    assert Principal.from_dict(principal.to_dict()).admin


def test_summary_shape():
    policy = AccessPolicy()
    policy.allow("local", "u1")
    summary = policy.summary()
    assert set(summary) == {"open_access", "principals", "pairing_active", "path"}


# -- local channel -------------------------------------------------------


def test_local_channel_roundtrip():
    channel = LocalChannel()
    seen = []
    channel.start(seen.append)
    channel.inject("do a thing")
    assert seen[0].text == "do a thing"
    channel.send(OutgoingMessage(chat="local", text="done"))
    assert channel.texts == ["done"]
    assert channel.last().text == "done"


def test_local_channel_edit_replaces_in_place():
    channel = LocalChannel()
    first = channel.send(OutgoingMessage(chat="local", text="Working…"))
    channel.send(OutgoingMessage(chat="local", text="Finished", edit_id=first))
    assert channel.texts == ["Finished"]


# -- webhook channel -----------------------------------------------------


def test_webhook_signature_roundtrip():
    channel = WebhookChannel("https://example.test/hook", secret="s3cret")
    payload = {"text": "hello", "chat": "c1"}
    assert channel.verify(payload, channel.sign(payload))
    assert not channel.verify(payload, "deadbeef")


def test_webhook_without_a_secret_accepts_anything():
    assert WebhookChannel("https://example.test/hook").verify({"a": 1}, "")


def test_webhook_deliver_rejects_a_bad_signature():
    channel = WebhookChannel(secret="s3cret")
    seen = []
    channel.start(seen.append)
    assert channel.deliver({"text": "hi"}, signature="wrong") is None
    assert seen == []


def test_webhook_deliver_accepts_a_signed_payload():
    channel = WebhookChannel(secret="s3cret")
    seen = []
    channel.start(seen.append)
    payload = {"text": "run this", "sender": "ci", "chat": "builds"}
    accepted = channel.deliver(payload, signature=channel.sign(payload))
    assert accepted is not None and accepted.text == "run this"
    assert seen[0].chat == "builds"


def test_webhook_ignores_an_empty_payload():
    channel = WebhookChannel()
    channel.start(lambda m: None)
    assert channel.deliver({}) is None


@pytest.mark.parametrize(
    ("style", "key"), [("json", "text"), ("slack", "text"), ("discord", "content")]
)
def test_webhook_payload_styles(style, key):
    channel = WebhookChannel("https://example.test", style=style)
    payload = channel._payload(OutgoingMessage(chat="c", text="hello"))
    assert key in payload


def test_webhook_send_without_a_url_is_a_noop():
    assert WebhookChannel().send(OutgoingMessage(chat="c", text="x")) == ""


# -- telegram wire format ------------------------------------------------


def test_telegram_without_a_token_is_unavailable():
    assert not TelegramChannel("").available


def test_telegram_html_escaping():
    assert _format("<b>bold</b> & more", True) == "&lt;b&gt;bold&lt;/b&gt; &amp; more"
    assert _format("<raw>", False) == "<raw>"


def test_telegram_split_on_line_boundaries():
    text = "\n".join(f"line {i}" for i in range(500))
    chunks = _split(text, 200)
    assert all(len(c) <= 200 for c in chunks)
    assert "".join(chunks) == text


def test_telegram_split_hard_splits_a_long_line():
    chunks = _split("x" * 500, 100)
    assert all(len(c) <= 100 for c in chunks)
    assert "".join(chunks) == "x" * 500


def test_telegram_short_text_is_not_split():
    assert _split("short", 100) == ["short"]


def test_telegram_decode_of_a_text_update():
    channel = TelegramChannel("123:abc")
    update = {
        "update_id": 7,
        "message": {
            "message_id": 42,
            "date": 1700000000,
            "chat": {"id": -100123},
            "from": {"id": 555, "first_name": "Felix", "last_name": "K"},
            "text": "open the editor",
        },
    }
    decoded = channel._decode(update)
    assert decoded is not None
    assert decoded.sender == "555" and decoded.sender_name == "Felix K"
    assert decoded.chat == "-100123" and decoded.text == "open the editor"
    assert decoded.message_id == "42"
    channel.stop()


def test_telegram_decode_ignores_a_non_message_update():
    channel = TelegramChannel("123:abc")
    assert channel._decode({"update_id": 1, "poll": {}}) is None
    channel.stop()


def test_telegram_uses_the_caption_when_there_is_no_text():
    channel = TelegramChannel("123:abc")
    decoded = channel._decode({
        "message": {"message_id": 1, "chat": {"id": 1}, "from": {"id": 2}, "caption": "look at this"}
    })
    assert decoded.text == "look at this"
    channel.stop()


# -- manager -------------------------------------------------------------


class FakeRuntime:
    def __init__(self, tmp_path, agent_factory=None):
        from lai.config import load_config

        self.config = load_config().with_overrides(home=tmp_path)
        self.config.ensure_dirs()
        self.provider = type("P", (), {"name": "fake", "model": "m1"})()
        self.registry = type("R", (), {"__len__": lambda self: 3})()
        self.skills = type("S", (), {"__len__": lambda self: 2, "list": lambda self: []})()
        self.policy = type("Pol", (), {"config": self.config.safety})()
        self.desktop = FakeDesktop()
        self._factory = agent_factory

    def agent(self, **kwargs):
        return self._factory(**kwargs) if self._factory else FakeAgent()


class FakeDesktop:
    class _Screen:
        def grab(self):
            return type("Shot", (), {"png": b"\x89PNGfake"})()

    class _Windows:
        def active_window(self):
            return type("W", (), {"title": "Editor", "wm_class": "Xed", "active": True})()

        def list_windows(self):
            return [
                type("W", (), {"title": "Editor", "wm_class": "Xed", "active": True})(),
                type("W", (), {"title": "Browser", "wm_class": "firefox", "active": False})(),
            ]

    screen = _Screen()
    windows = _Windows()


class FakeResult:
    status = "completed"
    summary = "did the thing"
    verification = "checked it"
    artifacts: list = []
    steps = 3
    elapsed = 1.5
    error = ""

    def to_dict(self):
        return {"status": self.status, "summary": self.summary}


class FakeAgent:
    def __init__(self, *, on_event=None, approver=None, result=None, hook=None, **kwargs):
        self.on_event = on_event
        self.approver = approver
        self.result = result or FakeResult()
        self.hook = hook
        self.interrupted = False

    def run(self, task):
        if self.hook:
            self.hook(self)
        return self.result

    def interrupt(self):
        self.interrupted = True


@pytest.fixture
def wired(tmp_path):
    """A manager with a local channel and an allowlisted sender."""
    holder: dict = {}

    def factory(**kwargs):
        agent = (holder.get("factory") or FakeAgent)(**kwargs)
        holder["agent"] = agent
        return agent

    runtime = FakeRuntime(tmp_path, agent_factory=factory)
    channel = LocalChannel()
    policy = AccessPolicy(tmp_path / "channels.json")
    policy.allow("local", "local", admin=True)
    manager = ChannelManager(runtime, access=policy, approval_timeout=2.0)
    manager.add(channel)
    manager.start()
    holder.update({"manager": manager, "channel": channel, "runtime": runtime, "policy": policy})
    yield holder
    manager.stop()


def wait_until(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


def test_denied_sender_gets_a_flat_refusal(wired):
    wired["channel"].inject("do something", sender="intruder")
    assert wired["channel"].texts[-1] == "Not authorised."


def test_denial_does_not_advertise_pairing(wired):
    wired["channel"].inject("hello", sender="intruder")
    assert "pair" not in wired["channel"].texts[-1].lower()


def test_help_command(wired):
    wired["channel"].inject("/help")
    assert "desktop agent" in wired["channel"].texts[-1]


def test_status_command(wired):
    wired["channel"].inject("/status")
    text = wired["channel"].texts[-1]
    assert "fake/m1" in text and "mode" in text


def test_windows_command(wired):
    wired["channel"].inject("/windows")
    assert "Xed" in wired["channel"].texts[-1]


def test_screenshot_command_sends_an_image(wired):
    wired["channel"].inject("/screenshot")
    last = wired["channel"].last()
    assert last.attachments and last.attachments[0].kind == "image"


def test_mode_command_requires_admin_and_validates(wired):
    wired["channel"].inject("/mode auto")
    assert "now auto" in wired["channel"].texts[-1]
    assert wired["runtime"].config.safety.mode == "auto"

    wired["channel"].inject("/mode banana")
    assert "must be one of" in wired["channel"].texts[-1]


def test_unknown_command(wired):
    wired["channel"].inject("/frobnicate")
    assert "Unknown command" in wired["channel"].texts[-1]


def test_empty_message_is_nudged(wired):
    wired["channel"].inject("   ")
    assert "instruction" in wired["channel"].texts[-1]


def test_a_task_runs_and_reports_back(wired):
    wired["channel"].inject("open the editor")
    assert wait_until(lambda: any("completed" in t for t in wired["channel"].texts))
    final = [t for t in wired["channel"].texts if "completed" in t][-1]
    assert "did the thing" in final
    assert "checked it" in final


def test_a_second_task_while_busy_is_refused(wired):
    release = threading.Event()
    wired["factory"] = lambda **kw: FakeAgent(hook=lambda a: release.wait(timeout=5), **kw)
    wired["channel"].inject("slow task")
    assert wait_until(lambda: wired["manager"].conversations and
                      list(wired["manager"].conversations.values())[0].busy)
    wired["channel"].inject("another task")
    assert any("Still working" in t for t in wired["channel"].texts)
    release.set()


def test_stop_interrupts_the_run(wired):
    release = threading.Event()
    wired["factory"] = lambda **kw: FakeAgent(hook=lambda a: release.wait(timeout=5), **kw)
    wired["channel"].inject("slow task")
    assert wait_until(lambda: wired["manager"].conversations and
                      list(wired["manager"].conversations.values())[0].busy)
    wired["channel"].inject("/stop")
    assert wired["agent"].interrupted
    release.set()


def test_stop_unblocks_a_pending_approval(wired):
    """/stop must not leave a run parked in the approval wait for its full timeout."""
    release = threading.Event()

    def hook(agent):
        verdict = type("V", (), {"reason": "needs confirmation"})()
        agent.approver("shell_exec", {"command": "x"}, verdict)
        release.wait(timeout=5)

    wired["factory"] = lambda **kw: FakeAgent(hook=hook, **kw)
    wired["channel"].inject("do it")
    assert wait_until(lambda: any("Approval needed" in t for t in wired["channel"].texts))
    wired["channel"].inject("/stop")
    assert wait_until(lambda: wired["agent"].interrupted)
    release.set()


def test_a_task_from_another_chat_is_refused_while_one_runs(wired):
    """Two chats share one desktop: the second run must not start."""
    release = threading.Event()
    wired["factory"] = lambda **kw: FakeAgent(hook=lambda a: release.wait(timeout=5), **kw)
    wired["channel"].inject("slow task", chat="c1")
    assert wait_until(lambda: wired["manager"].conversations.get("local:c1", None) is not None
                      and wired["manager"].conversations["local:c1"].busy)
    wired["channel"].inject("task from another chat", chat="c2")
    assert any("Another task is already running" in t for t in wired["channel"].texts)
    second = wired["manager"].conversations.get("local:c2")
    assert second is not None and second.runs == 0 and not second.busy
    release.set()


class FakeDesktopGate:
    """claim/engage/release gate with the daemon's semantics."""

    def __init__(self, *, refuses=False):
        self.refuses = refuses
        self.claims: list[str] = []
        self.engaged: list[object] = []
        self.releases = 0

    def claim(self, task: str) -> bool:
        if self.refuses:
            return False
        self.claims.append(task)
        return True

    def engage(self, agent) -> None:
        self.engaged.append(agent)

    def release(self) -> None:
        self.releases += 1


def test_the_desktop_gate_is_claimed_and_released(wired):
    desktop = FakeDesktopGate()
    wired["manager"].desktop = desktop
    wired["channel"].inject("open the editor")
    assert wait_until(lambda: any("completed" in t for t in wired["channel"].texts))
    assert desktop.claims and "open the editor" in desktop.claims[0]
    assert desktop.engaged, "the real agent must be engaged after construction"
    assert wait_until(lambda: desktop.releases == 1)


def test_a_refused_claim_means_no_run(wired):
    wired["manager"].desktop = FakeDesktopGate(refuses=True)
    wired["channel"].inject("open the editor")
    assert any("Another task is already running" in t for t in wired["channel"].texts)
    conversation = wired["manager"].conversations["local:local"]
    assert conversation.runs == 0 and not conversation.busy, "no run may start"


def test_run_events_carry_task_and_agent(wired):
    events: list[tuple[str, dict]] = []
    wired["manager"].on_event = lambda kind, payload: events.append((kind, payload))
    wired["channel"].inject("open the editor")
    assert wait_until(lambda: any(k == "run_finished" for k, _ in events))
    started = next(p for k, p in events if k == "run_started")
    assert started["task"] == "open the editor"
    assert started["agent"] is wired["agent"]


def test_stop_with_nothing_running(wired):
    wired["channel"].inject("/stop")
    assert "Nothing is running" in wired["channel"].texts[-1]


def test_new_command_resets_the_session(wired):
    wired["channel"].inject("first task")
    assert wait_until(lambda: any("completed" in t for t in wired["channel"].texts))
    conversation = list(wired["manager"].conversations.values())[0]
    before = conversation.session.id
    wired["channel"].inject("/new")
    assert conversation.session.id != before


def test_approval_travels_over_the_channel(wired):
    """In ask mode the permission prompt must reach the human remotely."""
    captured: dict = {}

    def hook(agent):
        verdict = type("V", (), {"reason": "input action needs confirmation"})()
        captured["granted"] = agent.approver("computer_click", {"x": 5, "y": 5}, verdict)

    wired["factory"] = lambda **kw: FakeAgent(hook=hook, **kw)
    wired["channel"].inject("click something")

    assert wait_until(lambda: any("Approval needed" in t for t in wired["channel"].texts))
    wired["channel"].inject("/yes")
    assert wait_until(lambda: "granted" in captured)
    assert captured["granted"] is True


def test_refused_approval(wired):
    captured: dict = {}

    def hook(agent):
        verdict = type("V", (), {"reason": "needs confirmation"})()
        captured["granted"] = agent.approver("shell_exec", {"command": "rm x"}, verdict)

    wired["factory"] = lambda **kw: FakeAgent(hook=hook, **kw)
    wired["channel"].inject("delete something")
    assert wait_until(lambda: any("Approval needed" in t for t in wired["channel"].texts))
    wired["channel"].inject("/no")
    assert wait_until(lambda: "granted" in captured)
    assert captured["granted"] is False


def test_approval_times_out_as_a_refusal(wired):
    captured: dict = {}

    def hook(agent):
        verdict = type("V", (), {"reason": "needs confirmation"})()
        captured["granted"] = agent.approver("shell_exec", {"command": "x"}, verdict)

    wired["factory"] = lambda **kw: FakeAgent(hook=hook, **kw)
    wired["channel"].inject("do it")
    assert wait_until(lambda: "granted" in captured, timeout=8)
    assert captured["granted"] is False


def test_answering_when_nothing_is_pending(wired):
    wired["channel"].inject("/yes")
    assert "Nothing is waiting" in wired["channel"].texts[-1]


def test_pairing_through_the_manager(tmp_path):
    runtime = FakeRuntime(tmp_path)
    channel = LocalChannel()
    policy = AccessPolicy(tmp_path / "c.json")
    manager = ChannelManager(runtime, access=policy)
    manager.add(channel)
    manager.start()
    try:
        channel.inject("hello")
        assert channel.texts[-1] == "Not authorised."

        code = policy.new_pairing_code()
        channel.inject(f"/pair {code}")
        assert "Paired" in channel.texts[-1]

        channel.inject("/whoami")
        assert "admin" in channel.texts[-1]
    finally:
        manager.stop()


def test_manager_status_shape(wired):
    status = wired["manager"].status()
    assert set(status) == {"channels", "conversations", "access"}


def test_handler_errors_are_reported_not_raised(wired):
    wired["factory"] = lambda **kw: (_ for _ in ()).throw(RuntimeError("agent build failed"))
    wired["channel"].inject("do a thing")
    assert wait_until(lambda: any("failed" in t for t in wired["channel"].texts))


def test_broadcast_reaches_known_conversations(wired):
    wired["channel"].inject("/status")
    assert wired["manager"].broadcast("scheduled report") == 1
    assert wired["channel"].texts[-1] == "scheduled report"


# -- factory -------------------------------------------------------------


def test_build_channels_reports_what_is_missing(tmp_path):
    from dataclasses import replace

    from lai.channels import build_channels
    from lai.config import load_config

    config = load_config().with_overrides(home=tmp_path)
    config = config.with_overrides(
        channels=replace(config.channels, enabled=("telegram", "webhook", "nonsense"))
    )
    built, problems = build_channels(config)
    assert [c.name for c in built] == ["webhook"]
    assert "telegram" in problems and "token" in problems["telegram"]
    assert "nonsense" in problems


def test_build_channels_with_a_telegram_token(tmp_path):
    from dataclasses import replace

    from lai.channels import build_channels
    from lai.config import load_config

    config = load_config().with_overrides(home=tmp_path)
    config = config.with_overrides(
        channels=replace(config.channels, enabled=("telegram",), telegram_token="123:abc")
    )
    built, problems = build_channels(config)
    assert [c.name for c in built] == ["telegram"] and not problems
    for channel in built:
        channel.stop()


def test_config_never_leaks_the_bot_token(tmp_path):
    from dataclasses import replace

    from lai.config import load_config

    config = load_config().with_overrides(home=tmp_path)
    config = config.with_overrides(
        channels=replace(config.channels, telegram_token="123:supersecret")
    )
    assert "supersecret" not in json.dumps(config.redacted())


# -- discord -------------------------------------------------------------


def test_discord_without_a_token_is_unavailable():
    from lai.channels.discord import DiscordChannel

    assert not DiscordChannel("").available


def test_discord_decodes_a_user_message():
    from lai.channels.discord import DiscordChannel

    channel = DiscordChannel("fake.token")
    decoded = channel._decode({
        "id": "999",
        "channel_id": "555",
        "content": "open the editor",
        "author": {"id": "77", "username": "felix", "global_name": "Felix"},
        "attachments": [],
    })
    assert decoded is not None
    assert decoded.sender == "77" and decoded.sender_name == "Felix"
    assert decoded.chat == "555" and decoded.text == "open the editor"
    channel.stop()


def test_discord_ignores_bots_including_itself():
    from lai.channels.discord import DiscordChannel

    channel = DiscordChannel("fake.token")
    assert channel._decode({
        "id": "1", "channel_id": "2", "content": "hi",
        "author": {"id": "3", "username": "otherbot", "bot": True},
    }) is None
    channel.stop()


def test_discord_ignores_an_empty_message():
    from lai.channels.discord import DiscordChannel

    channel = DiscordChannel("fake.token")
    assert channel._decode({
        "id": "1", "channel_id": "2", "content": "  ", "author": {"id": "3"}, "attachments": []
    }) is None
    channel.stop()


def test_discord_tracks_sequence_and_session_from_ready():
    from lai.channels.discord import DiscordChannel

    channel = DiscordChannel("fake.token")
    channel._on_frame({
        "op": 0, "s": 5, "t": "READY",
        "d": {
            "user": {"id": "1", "username": "laibot"},
            "session_id": "sess-abc",
            "resume_gateway_url": "wss://resume.example",
        },
    })
    assert channel._sequence == 5
    assert channel._session_id == "sess-abc"
    assert "resume.example" in channel._resume_url
    channel.stop()


def test_discord_invalid_session_clears_resume_state():
    from lai.channels.discord import DiscordChannel

    channel = DiscordChannel("fake.token")
    channel._session_id, channel._sequence = "old", 9
    channel._on_frame({"op": 9, "d": False})
    assert channel._session_id == "" and channel._sequence is None
    channel.stop()


def test_discord_message_split_respects_the_2000_char_cap():
    from lai.channels.discord import MAX_MESSAGE_CHARS, _split

    chunks = _split("y" * 5000, MAX_MESSAGE_CHARS)
    assert all(len(c) <= MAX_MESSAGE_CHARS for c in chunks)
    assert "".join(chunks) == "y" * 5000


def test_discord_dispatch_of_a_message_reaches_the_handler():
    from lai.channels.discord import DiscordChannel

    channel = DiscordChannel("fake.token")
    seen = []
    channel.start = lambda h: None  # avoid opening a real gateway
    channel._handler = seen.append
    channel._on_frame({
        "op": 0, "s": 1, "t": "MESSAGE_CREATE",
        "d": {"id": "1", "channel_id": "2", "content": "hello", "author": {"id": "3", "username": "u"}},
    })
    assert seen and seen[0].text == "hello"
