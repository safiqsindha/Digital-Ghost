"""Notification tests. `urlopen` is monkeypatched in every test that could
reach the network — nothing here is allowed to make a real request.
"""

from __future__ import annotations

import json
import urllib.request

import pytest

from digital_ghost.config import NotificationConfig
from digital_ghost.notify import (
    DiscordNotifier,
    NtfyNotifier,
    NullNotifier,
    get_notifier,
    notify_cell_failure,
    notify_sweep_complete,
)


class FakeResponse:
    def __init__(self, status: int = 200):
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture
def captured_requests(monkeypatch) -> list:
    """Replaces urlopen and records every Request it was handed."""
    calls = []

    def fake_urlopen(request, timeout=None):
        calls.append((request, timeout))
        return FakeResponse()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    return calls


class RecordingNotifier(NullNotifier):
    def __init__(self):
        super().__init__()
        self.sent = []

    def send(self, title: str, message: str, priority: str = "default") -> bool:
        self.sent.append((title, message, priority))
        return True


def test_ntfy_posts_to_topic_url_with_title_and_priority_headers(captured_requests):
    assert NtfyNotifier("my-topic").send("Cell failed", "it broke", priority="high")

    request, timeout = captured_requests[0]
    assert request.full_url == "https://ntfy.sh/my-topic"
    assert request.get_method() == "POST"
    assert request.data == b"it broke"
    assert request.get_header("Title") == "Cell failed"
    assert request.get_header("Priority") == "high"
    assert timeout == pytest.approx(10.0)


def test_discord_posts_content_json(captured_requests):
    assert DiscordNotifier("https://discord.example/webhook/abc").send("Sweep complete", "45 cells")

    request, _ = captured_requests[0]
    assert request.full_url == "https://discord.example/webhook/abc"
    assert request.get_header("Content-type") == "application/json"
    payload = json.loads(request.data.decode("utf-8"))
    assert list(payload) == ["content"]
    assert "Sweep complete" in payload["content"]
    assert "45 cells" in payload["content"]


def test_null_notifier_is_the_default(captured_requests):
    notifier = get_notifier(NotificationConfig())
    assert isinstance(notifier, NullNotifier)
    assert notifier.send("t", "m")
    assert captured_requests == []


@pytest.mark.parametrize(
    "backend,env_var",
    [("ntfy", "DIGITAL_GHOST_NTFY_TOPIC"), ("discord", "DIGITAL_GHOST_DISCORD_WEBHOOK")],
)
def test_missing_env_var_falls_back_to_null_notifier(backend, env_var, monkeypatch, caplog):
    monkeypatch.delenv(env_var, raising=False)
    notifier = get_notifier(NotificationConfig(backend=backend))
    assert isinstance(notifier, NullNotifier)
    assert env_var in caplog.text


def test_backend_is_selected_from_env(monkeypatch):
    monkeypatch.setenv("DIGITAL_GHOST_NTFY_TOPIC", "topic-from-env")
    notifier = get_notifier(NotificationConfig(backend="ntfy"))
    assert isinstance(notifier, NtfyNotifier)
    assert notifier.topic == "topic-from-env"

    monkeypatch.setenv("DIGITAL_GHOST_DISCORD_WEBHOOK", "https://discord.example/hook")
    notifier = get_notifier(NotificationConfig(backend="discord"))
    assert isinstance(notifier, DiscordNotifier)
    assert notifier.webhook_url == "https://discord.example/hook"


@pytest.mark.parametrize(
    "notifier",
    [NtfyNotifier("topic"), DiscordNotifier("https://discord.example/hook")],
)
def test_network_exception_returns_false_without_raising(notifier, monkeypatch):
    def boom(request, timeout=None):
        raise OSError("connection reset by peer")

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    assert notifier.send("title", "message") is False


def test_non_2xx_response_returns_false(monkeypatch):
    monkeypatch.setattr(urllib.request, "urlopen", lambda request, timeout=None: FakeResponse(500))
    assert NtfyNotifier("topic").send("title", "message") is False


def test_cell_failure_message_has_the_key_facts():
    notifier = RecordingNotifier()
    notify_cell_failure(
        notifier, "meme_d40_s2", RuntimeError("CUDA out of memory"), cells_done=12, cells_total=45
    )

    title, message, priority = notifier.sent[0]
    assert "meme_d40_s2" in title
    assert "RuntimeError" in message
    assert "CUDA out of memory" in message
    assert "12/45" in message
    assert priority == "high"


def test_cell_failure_truncates_a_huge_traceback():
    notifier = RecordingNotifier()
    notify_cell_failure(notifier, "c1", "x" * 5000, cells_done=1, cells_total=45)
    assert len(notifier.sent[0][1]) < 500


def test_sweep_complete_message_has_the_key_facts():
    notifier = RecordingNotifier()
    notify_sweep_complete(notifier, n_done=44, n_failed=1, total_cost_usd=31.5, elapsed_hours=26.75)

    title, message, priority = notifier.sent[0]
    assert "Sweep complete" in title
    assert "44" in message
    assert "1 cell" in message
    assert "$31.50" in message
    assert "26.8h" in message
    assert priority == "high"


def test_sweep_complete_says_no_action_needed_when_clean():
    notifier = RecordingNotifier()
    notify_sweep_complete(notifier, n_done=45, n_failed=0, total_cost_usd=30.0, elapsed_hours=27.0)

    title, message, priority = notifier.sent[0]
    assert "failure" not in title.lower()
    assert "No failures" in message
    assert priority == "default"


def test_notify_on_gates_events():
    notifier = RecordingNotifier()
    notifier.notify_on = frozenset({"failure"})

    assert notify_cell_failure(notifier, "c1", "boom", cells_done=1, cells_total=45)
    assert notify_sweep_complete(notifier, 45, 0, 30.0, 27.0) is False
    assert len(notifier.sent) == 1
