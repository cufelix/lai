"""Channels — how the outside world reaches the desktop agent."""

from __future__ import annotations

from ..config import Config
from .access import Access, AccessDecision, AccessPolicy, Principal
from .base import (
    Attachment,
    BaseChannel,
    Channel,
    ChannelError,
    IncomingMessage,
    OutgoingMessage,
)
from .discord import DiscordChannel
from .manager import ChannelManager, Conversation
from .telegram import TelegramChannel
from .webhook import LocalChannel, WebhookChannel

# Everything a user can name in `channels.enabled`.
REGISTRY: dict[str, str] = {
    "telegram": "a Telegram bot — drive the desktop from your phone",
    "discord": "a Discord bot over the gateway (needs the websockets package)",
    "webhook": "generic HTTP in/out, Slack- and Discord-webhook compatible",
    "local": "in-process channel for scripting and tests",
}


def build_channels(config: Config) -> tuple[list[Channel], dict[str, str]]:
    """Construct the channels named in the config.

    Returns the usable channels plus a ``{name: reason}`` map for the ones that
    could not be built, so `lai doctor` can explain what is missing rather than
    failing silently.
    """
    wanted = [name.strip().lower() for name in config.channels.enabled if name.strip()]
    built: list[Channel] = []
    problems: dict[str, str] = {}

    for name in wanted:
        if name == "telegram":
            token = config.channels.telegram_token
            if not token:
                problems[name] = "no bot token (set LAI_TELEGRAM_TOKEN or channels.telegram.token)"
                continue
            built.append(TelegramChannel(token))
        elif name == "discord":
            token = config.channels.discord_token
            if not token:
                problems[name] = "no bot token (set LAI_DISCORD_TOKEN or channels.discord.token)"
                continue
            built.append(DiscordChannel(token))
        elif name == "webhook":
            built.append(
                WebhookChannel(
                    config.channels.webhook_url,
                    secret=config.channels.webhook_secret,
                    style=config.channels.webhook_style,
                )
            )
        elif name == "local":
            built.append(LocalChannel())
        else:
            problems[name] = f"unknown channel (known: {', '.join(sorted(REGISTRY))})"
    return built, problems


def build_manager(runtime, *, channels: list[Channel] | None = None, on_event=None, desktop=None) -> ChannelManager:
    """Assemble a manager with the configured access policy and channels.

    ``desktop`` is an optional claim/engage/release gate; the daemon passes its
    state so channel runs serialise with HTTP tasks and the scheduler.
    """
    config = runtime.config
    policy = AccessPolicy(config.channels_file, open_access=config.channels.open_access)
    manager = ChannelManager(
        runtime,
        access=policy,
        on_event=on_event,
        approval_timeout=config.channels.approval_timeout,
        desktop=desktop,
    )
    if channels is None:
        channels, _ = build_channels(config)
    for channel in channels:
        manager.add(channel)
    return manager


__all__ = [
    "REGISTRY",
    "Access",
    "AccessDecision",
    "AccessPolicy",
    "Attachment",
    "BaseChannel",
    "Channel",
    "ChannelError",
    "ChannelManager",
    "Conversation",
    "DiscordChannel",
    "IncomingMessage",
    "LocalChannel",
    "OutgoingMessage",
    "Principal",
    "TelegramChannel",
    "WebhookChannel",
    "build_channels",
    "build_manager",
]
