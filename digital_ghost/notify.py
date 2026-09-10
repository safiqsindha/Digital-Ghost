"""Push notifications for an unattended sweep.

A 45-cell sweep runs for over a day with nobody watching the terminal. This
module pushes two things to a phone: a cell failed, and the sweep finished.

Every notification path is best-effort. A notification is worth nothing next
to the sweep it reports on, so nothing here is allowed to raise into the
caller — see `_post`.

Backends are selected by `notifications.backend` in runtime.yaml. The topic
or webhook URL is read ONLY from the environment variable named in the
config (the same rule the GPU credentials follow): it is a secret and
runtime.yaml is committed.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.request
from abc import ABC, abstractmethod
from collections.abc import Iterable

from digital_ghost.config import NotificationConfig

logger = logging.getLogger(__name__)

NTFY_BASE_URL = "https://ntfy.sh"

# Short: a hung connection must not stall the cell that is reporting through it.
REQUEST_TIMEOUT_S = 10.0

EVENT_FAILURE = "failure"
EVENT_COMPLETION = "completion"

# Lock screens truncate anyway, and a full CUDA traceback in a push
# notification tells the reader less than its first line does.
MAX_ERROR_CHARS = 240


def _post(url: str, data: bytes, headers: dict[str, str]) -> bool:
    request = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_S) as response:
            if 200 <= response.status < 300:
                return True
            logger.warning("notification rejected with HTTP %s by %s", response.status, url)
            return False
    # Bare Exception on purpose: a dead notification service, a typo'd webhook
    # and a DNS blip must all be non-events for a 27-hour sweep. Anything that
    # escaped here would kill the run it was only supposed to describe.
    except Exception as e:  # noqa: BLE001 - failure to notify is never fatal
        logger.warning("notification to %s failed: %s", url, e)
        return False


class Notifier(ABC):
    def __init__(self, notify_on: Iterable[str] | None = None) -> None:
        # None means every event; a list gates which ones reach the phone.
        self.notify_on = None if notify_on is None else frozenset(notify_on)

    def wants(self, event: str) -> bool:
        return self.notify_on is None or event in self.notify_on

    @abstractmethod
    def send(self, title: str, message: str, priority: str = "default") -> bool:
        """Returns whether the notification was delivered. Never raises."""
        ...


class NtfyNotifier(Notifier):
    def __init__(
        self,
        topic: str,
        notify_on: Iterable[str] | None = None,
        base_url: str = NTFY_BASE_URL,
    ) -> None:
        super().__init__(notify_on)
        self.topic = topic
        self.base_url = base_url.rstrip("/")

    def send(self, title: str, message: str, priority: str = "default") -> bool:
        return _post(
            f"{self.base_url}/{self.topic}",
            message.encode("utf-8"),
            {
                "Content-Type": "text/plain; charset=utf-8",
                "Title": title,
                "Priority": priority,
            },
        )


class DiscordNotifier(Notifier):
    def __init__(self, webhook_url: str, notify_on: Iterable[str] | None = None) -> None:
        super().__init__(notify_on)
        self.webhook_url = webhook_url

    def send(self, title: str, message: str, priority: str = "default") -> bool:
        # Discord webhooks carry no title or priority field; fold the title
        # into the body so the first line still reads as a headline.
        payload = {"content": f"{title}\n{message}"}
        return _post(
            self.webhook_url,
            json.dumps(payload).encode("utf-8"),
            {"Content-Type": "application/json"},
        )


class NullNotifier(Notifier):
    """Drops everything. The default, and the fallback for misconfiguration."""

    def send(self, title: str, message: str, priority: str = "default") -> bool:
        logger.debug("notification suppressed (no backend configured): %s", title)
        return True


def get_notifier(config: NotificationConfig) -> Notifier:
    if config.backend == "ntfy":
        topic = os.environ.get(config.ntfy_topic_env, "").strip()
        if not topic:
            logger.warning(
                "notifications.backend is 'ntfy' but %s is not set in the environment; "
                "the sweep will run without notifications. Export it and restart to "
                "get them.",
                config.ntfy_topic_env,
            )
            return NullNotifier(config.notify_on)
        return NtfyNotifier(topic, notify_on=config.notify_on)

    if config.backend == "discord":
        webhook = os.environ.get(config.discord_webhook_env, "").strip()
        if not webhook:
            logger.warning(
                "notifications.backend is 'discord' but %s is not set in the environment; "
                "the sweep will run without notifications. Export it and restart to "
                "get them.",
                config.discord_webhook_env,
            )
            return NullNotifier(config.notify_on)
        return DiscordNotifier(webhook, notify_on=config.notify_on)

    if config.backend != "none":
        logger.warning(
            "unknown notifications.backend %r; the sweep will run without notifications",
            config.backend,
        )
    return NullNotifier(config.notify_on)


def _describe_error(error: BaseException | str) -> str:
    text = f"{type(error).__name__}: {error}" if isinstance(error, BaseException) else str(error)
    text = " ".join(text.split())
    if len(text) > MAX_ERROR_CHARS:
        text = text[: MAX_ERROR_CHARS - 3] + "..."
    return text


def notify_cell_failure(
    notifier: Notifier,
    cell_id: str,
    error: BaseException | str,
    cells_done: int,
    cells_total: int,
) -> bool:
    if not notifier.wants(EVENT_FAILURE):
        return False
    remaining = max(0, cells_total - cells_done)
    message = (
        f"{_describe_error(error)}\n"
        f"Progress: {cells_done}/{cells_total} cells done, {remaining} to go.\n"
        "The sweep is still running; the failed cell can be re-run on resume."
    )
    return notifier.send(f"Cell failed: {cell_id}", message, priority="high")


def notify_sweep_complete(
    notifier: Notifier,
    n_done: int,
    n_failed: int,
    total_cost_usd: float,
    elapsed_hours: float,
) -> bool:
    if not notifier.wants(EVENT_COMPLETION):
        return False
    verdict = (
        f"{n_failed} cell(s) failed - re-run the sweep with resume to retry them."
        if n_failed
        else "No failures. Nothing to intervene on."
    )
    message = (
        f"{n_done} cells done, {n_failed} failed.\n"
        f"Elapsed {elapsed_hours:.1f}h, spend ${total_cost_usd:.2f}.\n"
        f"{verdict}"
    )
    return notifier.send(
        "Sweep complete" if not n_failed else "Sweep complete with failures",
        message,
        priority="high" if n_failed else "default",
    )
