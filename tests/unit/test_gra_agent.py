"""Unit tests for GRAAgent and FrameTranslator (agents/templates/gra_agent.py).

No real API calls or LLM calls: GRAAgent's Reasoner is the plain
HeuristicReasoner, so these tests exercise the actual frame-to-Observation
translation, entity tracking, and action-mapping logic end to end.
"""

from unittest.mock import MagicMock

import pytest
from arcengine import FrameData, GameAction, GameState

from agents.templates.gra_agent import FrameTranslator, GRAAgent


def _frame(
    grid: list[list[int]] | None = None,
    available: list[GameAction] | None = None,
    state: GameState = GameState.NOT_FINISHED,
    levels_completed: int = 0,
) -> FrameData:
    return FrameData(
        frame=[grid] if grid is not None else [],
        state=state,
        levels_completed=levels_completed,
        available_actions=(
            [GameAction.ACTION1, GameAction.ACTION2, GameAction.RESET]
            if available is None
            else available
        ),
    )


def _make_agent() -> GRAAgent:
    return GRAAgent(
        card_id="card-abc",
        game_id="ls20-test",
        agent_name="gra-test",
        ROOT_URL="http://localhost",
        record=False,
        arc_env=MagicMock(),
    )


@pytest.mark.unit
class TestFrameTranslatorSegmentation:
    def test_background_is_the_most_frequent_color(self) -> None:
        translator = FrameTranslator()
        grid = [
            [0, 0, 0],
            [0, 5, 0],
            [0, 0, 0],
        ]
        obs = translator.translate(_frame(grid), tick=1, actions_remaining=10)
        assert len(obs.entities) == 1
        entity = next(iter(obs.entities.values()))
        assert entity["color"] == 5

    def test_two_disconnected_same_color_regions_are_separate_entities(self) -> None:
        translator = FrameTranslator()
        grid = [
            [5, 0, 5],
            [0, 0, 0],
            [5, 0, 5],
        ]
        obs = translator.translate(_frame(grid), tick=1, actions_remaining=10)
        assert len(obs.entities) == 4  # four isolated corner cells

    def test_connected_cells_form_one_entity(self) -> None:
        translator = FrameTranslator()
        grid = [
            [0, 0, 0],
            [0, 5, 5],
            [0, 5, 5],
        ]
        obs = translator.translate(_frame(grid), tick=1, actions_remaining=10)
        assert len(obs.entities) == 1
        assert next(iter(obs.entities.values()))["size"] == 4

    def test_empty_frame_produces_no_entities_not_a_crash(self) -> None:
        translator = FrameTranslator()
        obs = translator.translate(_frame(grid=None), tick=1, actions_remaining=10)
        assert obs.entities == {}


@pytest.mark.unit
class TestFrameTranslatorObjectPermanence:
    def test_moved_entity_keeps_its_id(self) -> None:
        translator = FrameTranslator()
        grid1 = [[0, 0, 0], [0, 5, 0], [0, 0, 0]]
        grid2 = [[0, 0, 0], [0, 0, 5], [0, 0, 0]]  # same shape, shifted right

        obs1 = translator.translate(_frame(grid1), tick=1, actions_remaining=10)
        id1 = next(iter(obs1.entities))
        obs2 = translator.translate(_frame(grid2), tick=2, actions_remaining=9)
        id2 = next(iter(obs2.entities))

        assert id1 == id2
        assert any("moved" in e for e in obs2.events)

    def test_new_entity_gets_a_new_id_and_an_appeared_event(self) -> None:
        translator = FrameTranslator()
        grid1 = [[0, 0], [0, 0]]
        grid2 = [[0, 5], [0, 0]]

        translator.translate(_frame(grid1), tick=1, actions_remaining=10)
        obs2 = translator.translate(_frame(grid2), tick=2, actions_remaining=9)

        assert len(obs2.entities) == 1
        assert any("appeared" in e for e in obs2.events)

    def test_disappeared_entity_produces_an_event_and_is_dropped(self) -> None:
        translator = FrameTranslator()
        grid1 = [[0, 5], [0, 0]]
        grid2 = [[0, 0], [0, 0]]

        translator.translate(_frame(grid1), tick=1, actions_remaining=10)
        obs2 = translator.translate(_frame(grid2), tick=2, actions_remaining=9)

        assert obs2.entities == {}
        assert any("disappeared" in e for e in obs2.events)

    def test_different_colored_entities_never_match(self) -> None:
        translator = FrameTranslator()
        grid1 = [[0, 5], [0, 0]]
        grid2 = [[0, 6], [0, 0]]  # same position, different color

        obs1 = translator.translate(_frame(grid1), tick=1, actions_remaining=10)
        id1 = next(iter(obs1.entities))
        obs2 = translator.translate(_frame(grid2), tick=2, actions_remaining=9)
        id2 = next(iter(obs2.entities))

        assert id1 != id2
        assert any("disappeared" in e for e in obs2.events)
        assert any("appeared" in e for e in obs2.events)


@pytest.mark.unit
class TestFrameTranslatorObservationFields:
    def test_objective_text_never_leaks_a_goal(self) -> None:
        translator = FrameTranslator()
        obs = translator.translate(_frame([[0]]), tick=1, actions_remaining=10)
        # The only real requirement: no environment/game-specific hint --
        # generic vocabulary like "win condition" is fine (it names the
        # *category* of thing to look for, not what counts as winning
        # here). Checked by content, not just presence, so a future edit
        # can't quietly start leaking actual game-specific wording in.
        for leaky_phrase in ("goal is", "reach the", "target is", "you must"):
            assert leaky_phrase not in obs.objective_text.lower()

    def test_goal_progress_is_none_not_fabricated(self) -> None:
        translator = FrameTranslator()
        obs = translator.translate(_frame([[0]]), tick=1, actions_remaining=10)
        assert obs.goal_progress is None

    def test_done_true_on_win_and_game_over(self) -> None:
        translator = FrameTranslator()
        for state in (GameState.WIN, GameState.GAME_OVER):
            obs = translator.translate(
                _frame([[0]], state=state), tick=1, actions_remaining=10
            )
            assert obs.done is True

    def test_not_done_while_not_finished(self) -> None:
        translator = FrameTranslator()
        obs = translator.translate(
            _frame([[0]], state=GameState.NOT_FINISHED), tick=1, actions_remaining=10
        )
        assert obs.done is False

    def test_goal_satisfied_only_on_win(self) -> None:
        translator = FrameTranslator()
        obs = translator.translate(
            _frame([[0]], state=GameState.GAME_OVER), tick=1, actions_remaining=10
        )
        assert obs.goal_satisfied is False

    def test_available_actions_pass_through_as_names(self) -> None:
        translator = FrameTranslator()
        obs = translator.translate(
            _frame([[0]], available=[GameAction.ACTION3, GameAction.RESET]),
            tick=1,
            actions_remaining=10,
        )
        assert obs.available_actions == ["ACTION3", "RESET"]

    def test_levels_completed_change_is_a_real_event(self) -> None:
        translator = FrameTranslator()
        translator.translate(_frame([[0]], levels_completed=0), tick=1, actions_remaining=10)
        obs2 = translator.translate(
            _frame([[0]], levels_completed=1), tick=2, actions_remaining=9
        )
        assert any("levels_completed" in e for e in obs2.events)


@pytest.mark.unit
class TestGRAAgentConstruction:
    def test_constructs_with_fresh_cortex_and_control_loop(self) -> None:
        agent = _make_agent()
        assert agent.cortex is not None
        assert agent.control_loop is not None
        assert agent.cortex.tick_count == 0

    def test_two_instances_share_no_state(self) -> None:
        """The zero-context-per-game guarantee: nothing about one game's
        agent instance can leak into another's."""
        agent1 = _make_agent()
        agent2 = _make_agent()
        assert agent1.cortex is not agent2.cortex
        assert agent1.translator is not agent2.translator

        # Advance agent1's translator's entity counter; agent2 must be
        # unaffected.
        agent1.translator.translate(_frame([[0, 5]]), tick=1, actions_remaining=10)
        assert agent1.translator._next_entity_num == 1
        assert agent2.translator._next_entity_num == 0

    def test_is_done_reflects_terminal_states(self) -> None:
        agent = _make_agent()
        assert agent.is_done([], _frame([[0]], state=GameState.WIN)) is True
        assert agent.is_done([], _frame([[0]], state=GameState.GAME_OVER)) is True
        assert agent.is_done([], _frame([[0]], state=GameState.NOT_FINISHED)) is False


@pytest.mark.unit
class TestGRAAgentChooseAction:
    def test_returns_one_of_the_available_actions(self) -> None:
        agent = _make_agent()
        frame = _frame(
            [[0, 5], [0, 0]],
            available=[GameAction.ACTION1, GameAction.ACTION2, GameAction.RESET],
        )
        action = agent.choose_action([frame], frame)
        assert action in (GameAction.ACTION1, GameAction.ACTION2, GameAction.RESET)

    def test_complex_action_gets_coordinates_filled_in(self) -> None:
        agent = _make_agent()
        frame = _frame([[0, 5], [0, 0]], available=[GameAction.ACTION6])
        action = agent.choose_action([frame], frame)
        assert action is GameAction.ACTION6
        assert action.action_data.x is not None
        assert action.action_data.y is not None

    def test_no_available_actions_defaults_to_reset_without_crashing(self) -> None:
        agent = _make_agent()
        frame = _frame([[0]], available=[])
        action = agent.choose_action([frame], frame)
        assert action is GameAction.RESET

    def test_repeated_ticks_advance_cortex_state(self) -> None:
        agent = _make_agent()
        frame1 = _frame([[0, 5], [0, 0]])
        agent.choose_action([frame1], frame1)
        assert agent.cortex.tick_count == 1
        frame2 = _frame([[0, 0], [0, 5]])
        agent.choose_action([frame1, frame2], frame2)
        assert agent.cortex.tick_count == 2
