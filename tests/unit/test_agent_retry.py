"""Unit tests for Agent.do_action_request()'s transient-network-error retry.

Confirmed live 2026-09-09: a real RemoteDisconnected from ARC-AGI-3's own
game server (wrapped by requests as ConnectionError) killed a 51-action-deep
run outright, since do_action_request() had no retry at all. These tests
cover the fix in isolation via a minimal concrete Agent subclass, independent
of any specific template.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import requests
from arcengine import FrameData, GameAction, GameState

from agents.agent import Agent


class _MinimalAgent(Agent):
    """The smallest possible concrete Agent, for testing base-class behavior
    that has nothing to do with any specific template's choose_action logic."""

    def is_done(self, frames, latest_frame) -> bool:  # type: ignore[no-untyped-def]
        return False

    def choose_action(self, frames, latest_frame) -> GameAction:  # type: ignore[no-untyped-def]
        return GameAction.ACTION1


def _make_agent() -> _MinimalAgent:
    return _MinimalAgent(
        card_id="card-abc",
        game_id="ls20-test",
        agent_name="minimal-test",
        ROOT_URL="http://localhost",
        record=False,
        arc_env=MagicMock(),
    )


def _fake_raw_frame() -> SimpleNamespace:
    """A stand-in for FrameDataRaw with exactly the attributes
    _convert_raw_frame_data reads."""
    return SimpleNamespace(
        game_id="ls20-test",
        frame=[],
        state=GameState.NOT_FINISHED,
        levels_completed=0,
        win_levels=0,
        guid="guid-1",
        full_reset=False,
        available_actions=[],
    )


@pytest.mark.unit
class TestDoActionRequestRetry:
    def test_succeeds_immediately_with_no_retry_needed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        agent = _make_agent()
        agent.arc_env.step.return_value = _fake_raw_frame()
        sleep_calls = []
        monkeypatch.setattr("agents.agent.time.sleep", lambda s: sleep_calls.append(s))

        result = agent.do_action_request(GameAction.ACTION1)

        assert isinstance(result, FrameData)
        assert agent.arc_env.step.call_count == 1
        assert sleep_calls == []

    def test_retries_on_connection_error_then_succeeds(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        agent = _make_agent()
        agent.arc_env.step.side_effect = [
            requests.exceptions.ConnectionError("Remote end closed connection"),
            _fake_raw_frame(),
        ]
        sleep_calls = []
        monkeypatch.setattr("agents.agent.time.sleep", lambda s: sleep_calls.append(s))

        result = agent.do_action_request(GameAction.ACTION1)

        assert isinstance(result, FrameData)
        assert agent.arc_env.step.call_count == 2
        assert len(sleep_calls) == 1  # backed off exactly once before the retry

    def test_retries_on_timeout_too(self, monkeypatch: pytest.MonkeyPatch) -> None:
        agent = _make_agent()
        agent.arc_env.step.side_effect = [
            requests.exceptions.Timeout("timed out"),
            _fake_raw_frame(),
        ]
        monkeypatch.setattr("agents.agent.time.sleep", lambda s: None)

        result = agent.do_action_request(GameAction.ACTION1)
        assert isinstance(result, FrameData)
        assert agent.arc_env.step.call_count == 2

    def test_raises_after_exhausting_all_retry_attempts(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        agent = _make_agent()
        agent.arc_env.step.side_effect = requests.exceptions.ConnectionError(
            "still down"
        )
        monkeypatch.setattr("agents.agent.time.sleep", lambda s: None)

        with pytest.raises(requests.exceptions.ConnectionError, match="still down"):
            agent.do_action_request(GameAction.ACTION1)

        assert agent.arc_env.step.call_count == 3  # the configured attempt count

    def test_does_not_retry_a_real_http_error_response(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An HTTPError means the server actually responded with a
        rejection -- retrying could resubmit an action it already
        processed, so this must propagate on the first failure."""
        agent = _make_agent()
        agent.arc_env.step.side_effect = requests.exceptions.HTTPError("400 Bad Request")
        sleep_calls = []
        monkeypatch.setattr("agents.agent.time.sleep", lambda s: sleep_calls.append(s))

        with pytest.raises(requests.exceptions.HTTPError):
            agent.do_action_request(GameAction.ACTION1)

        assert agent.arc_env.step.call_count == 1
        assert sleep_calls == []

    def test_does_not_retry_unrelated_exceptions(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        agent = _make_agent()
        agent.arc_env.step.side_effect = ValueError("something else entirely")
        monkeypatch.setattr("agents.agent.time.sleep", lambda s: None)

        with pytest.raises(ValueError, match="something else entirely"):
            agent.do_action_request(GameAction.ACTION1)

        assert agent.arc_env.step.call_count == 1

    def test_backoff_increases_with_attempt_number(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        agent = _make_agent()
        agent.arc_env.step.side_effect = [
            requests.exceptions.ConnectionError("1"),
            requests.exceptions.ConnectionError("2"),
            _fake_raw_frame(),
        ]
        sleep_calls: list[float] = []
        monkeypatch.setattr("agents.agent.time.sleep", lambda s: sleep_calls.append(s))

        agent.do_action_request(GameAction.ACTION1)

        assert len(sleep_calls) == 2
        assert sleep_calls[1] > sleep_calls[0]
