"""Unit tests for HumanAgent (agents/templates/human_agent.py): the action
validation/re-prompt loop and is_done, driven via a mocked input()."""

from unittest.mock import MagicMock

import pytest
from arcengine import FrameData, GameAction, GameState

from agents.templates.human_agent import HumanAgent


def _make_agent() -> HumanAgent:
    return HumanAgent(
        card_id="card-abc",
        game_id="ls20-test",
        agent_name="human-test",
        ROOT_URL="http://localhost",
        record=False,
        arc_env=MagicMock(),
    )


def _frame(available=None, state=GameState.NOT_FINISHED) -> FrameData:
    return FrameData(
        frame=[[[0, 5], [0, 0]]],
        state=state,
        available_actions=available or [GameAction.ACTION1, GameAction.ACTION2],
    )


@pytest.mark.unit
class TestIsDone:
    def test_done_on_win_and_game_over(self) -> None:
        agent = _make_agent()
        assert agent.is_done([], _frame(state=GameState.WIN)) is True
        assert agent.is_done([], _frame(state=GameState.GAME_OVER)) is True

    def test_not_done_while_playing(self) -> None:
        agent = _make_agent()
        assert agent.is_done([], _frame(state=GameState.NOT_FINISHED)) is False


@pytest.mark.unit
class TestChooseAction:
    def test_valid_first_input_is_accepted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        agent = _make_agent()
        monkeypatch.setattr("builtins.input", lambda _: "ACTION1")
        frame = _frame()
        action = agent.choose_action([frame], frame)
        assert action is GameAction.ACTION1

    def test_lowercase_input_is_accepted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        agent = _make_agent()
        monkeypatch.setattr("builtins.input", lambda _: "action2")
        frame = _frame()
        action = agent.choose_action([frame], frame)
        assert action is GameAction.ACTION2

    def test_unavailable_action_reprompts(self, monkeypatch: pytest.MonkeyPatch) -> None:
        agent = _make_agent()
        responses = iter(["ACTION3", "ACTION1"])  # ACTION3 not in available_actions
        monkeypatch.setattr("builtins.input", lambda _: next(responses))
        frame = _frame(available=[GameAction.ACTION1, GameAction.ACTION2])
        action = agent.choose_action([frame], frame)
        assert action is GameAction.ACTION1

    def test_garbage_input_reprompts(self, monkeypatch: pytest.MonkeyPatch) -> None:
        agent = _make_agent()
        responses = iter(["not a real action", "ACTION1"])
        monkeypatch.setattr("builtins.input", lambda _: next(responses))
        frame = _frame()
        action = agent.choose_action([frame], frame)
        assert action is GameAction.ACTION1

    def test_quit_raises_system_exit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        agent = _make_agent()
        monkeypatch.setattr("builtins.input", lambda _: "quit")
        frame = _frame()
        with pytest.raises(SystemExit):
            agent.choose_action([frame], frame)

    def test_complex_action_prompts_for_coordinates(self, monkeypatch: pytest.MonkeyPatch) -> None:
        agent = _make_agent()
        responses = iter(["ACTION6", "3", "4"])
        monkeypatch.setattr("builtins.input", lambda _: next(responses))
        frame = _frame(available=[GameAction.ACTION6])
        action = agent.choose_action([frame], frame)
        assert action is GameAction.ACTION6
        assert action.action_data.x == 3
        assert action.action_data.y == 4
