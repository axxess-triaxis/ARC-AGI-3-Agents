"""Unit tests for OpenClawVision: the stall-detection and multimodal-content
additions layered on top of the base OpenClaw gateway agent.

These exercise the two behaviors this agent adds over its parent without a
live gateway, following the same client-mocking pattern already used in
test_openclaw_model_override.py.
"""

from unittest.mock import MagicMock

import httpx
import openai
import pytest
from arcengine import FrameData, GameState

from agents.templates.openclaw_agent import openclaw_agent as oc_module
from agents.templates.openclaw_agent.openclaw_vision_agent import OpenClawVision


def _make_response(content: str) -> MagicMock:
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    resp.usage = MagicMock(total_tokens=0)
    return resp


def _bad_request_error(message: str = "multimodal content not supported") -> openai.BadRequestError:
    request = httpx.Request("POST", "http://127.0.0.1:18789/v1/chat/completions")
    response = httpx.Response(400, request=request)
    return openai.BadRequestError(message, response=response, body=None)


def _make_agent(monkeypatch: pytest.MonkeyPatch) -> tuple[OpenClawVision, MagicMock]:
    """Instantiate OpenClawVision with the OpenAI client patched out."""
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _make_response(
        '{"action":"ACTION1"}'
    )
    monkeypatch.setattr(oc_module, "OpenAIClient", lambda **kw: mock_client)

    agent = OpenClawVision(
        card_id="card-abc",
        game_id="ls20-test",
        agent_name="openclawvision-test",
        ROOT_URL="http://localhost",
        record=False,
        arc_env=MagicMock(),
    )
    return agent, mock_client


def _frame(grid_value: int = 0) -> FrameData:
    return FrameData(
        game_id="ls20-test",
        state=GameState.NOT_FINISHED,
        available_actions=[1, 2, 3, 4, 5],
        frame=[[[grid_value] * 4 for _ in range(4)]],
    )


@pytest.mark.unit
class TestVisionContent:
    def test_first_turn_sends_multimodal_content(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        agent, client = _make_agent(monkeypatch)
        agent.choose_action([], _frame())

        content = client.chat.completions.create.call_args.kwargs["messages"][0][
            "content"
        ]
        assert isinstance(content, list)
        assert content[0]["type"] == "text"
        assert content[1]["type"] == "image_url"
        assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")

    def test_falls_back_to_text_after_bad_request(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        agent, client = _make_agent(monkeypatch)
        # First call (multimodal) rejected; retry (text-only) succeeds.
        client.chat.completions.create.side_effect = [
            _bad_request_error(),
            _make_response('{"action":"ACTION1"}'),
        ]

        action = agent.choose_action([], _frame())

        assert action.name == "ACTION1"
        assert agent._send_images is False
        assert client.chat.completions.create.call_count == 2
        first_content = client.chat.completions.create.call_args_list[0].kwargs[
            "messages"
        ][0]["content"]
        second_content = client.chat.completions.create.call_args_list[1].kwargs[
            "messages"
        ][0]["content"]
        assert isinstance(first_content, list)
        assert isinstance(second_content, str)

    def test_stays_text_only_after_one_fallback(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        agent, client = _make_agent(monkeypatch)
        client.chat.completions.create.side_effect = [
            _bad_request_error(),
            _make_response('{"action":"ACTION1"}'),
            _make_response('{"action":"ACTION2"}'),
        ]

        agent.choose_action([], _frame())
        client.chat.completions.create.reset_mock()
        agent.choose_action([], _frame(grid_value=1))

        # Second turn onward: no more multimodal attempt, straight to text.
        assert client.chat.completions.create.call_count == 1
        content = client.chat.completions.create.call_args.kwargs["messages"][0][
            "content"
        ]
        assert isinstance(content, str)

    def test_real_bad_request_not_caused_by_images_still_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        agent, client = _make_agent(monkeypatch)
        agent._send_images = False  # already text-only; a 400 here is a real error
        client.chat.completions.create.side_effect = _bad_request_error()

        with pytest.raises(openai.BadRequestError):
            agent.choose_action([], _frame())


@pytest.mark.unit
class TestStallDetection:
    def test_no_warning_on_first_turn(self, monkeypatch: pytest.MonkeyPatch) -> None:
        agent, client = _make_agent(monkeypatch)
        agent.choose_action([], _frame())

        prompt = client.chat.completions.create.call_args.kwargs["messages"][0][
            "content"
        ][0]["text"]
        assert "STALL WARNING" not in prompt

    def test_no_warning_when_frame_changes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        agent, client = _make_agent(monkeypatch)
        agent.choose_action([], _frame(grid_value=0))
        agent.choose_action([], _frame(grid_value=1))
        agent.choose_action([], _frame(grid_value=2))

        prompt = client.chat.completions.create.call_args.kwargs["messages"][0][
            "content"
        ][0]["text"]
        assert "STALL WARNING" not in prompt

    def test_warning_after_repeated_identical_frames(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        agent, client = _make_agent(monkeypatch)
        # Same frame three turns running (the action never changes anything).
        agent.choose_action([], _frame(grid_value=5))
        agent.choose_action([], _frame(grid_value=5))
        agent.choose_action([], _frame(grid_value=5))

        prompt = client.chat.completions.create.call_args.kwargs["messages"][0][
            "content"
        ][0]["text"]
        assert "STALL WARNING" in prompt
        assert "ACTION1" in prompt  # names the repeated action

    def test_reset_on_game_over_clears_stall_state(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        agent, client = _make_agent(monkeypatch)
        agent.choose_action([], _frame(grid_value=5))
        agent.choose_action([], _frame(grid_value=5))

        game_over_frame = FrameData(
            game_id="ls20-test",
            state=GameState.GAME_OVER,
            available_actions=[],
            frame=[[[5] * 4 for _ in range(4)]],
        )
        action = agent.choose_action([], game_over_frame)
        assert action.name == "RESET"
        assert agent._repeat_streak == 0
        assert agent._last_frame_hash is None
        assert agent._last_action_name is None


@pytest.mark.unit
class TestRenderGridImage:
    def test_empty_frame_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        agent, _ = _make_agent(monkeypatch)
        assert agent._render_grid_image(None) is None
        assert agent._render_grid_image([]) is None
        assert agent._render_grid_image([[]]) is None

    def test_real_grid_returns_png_bytes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        agent, _ = _make_agent(monkeypatch)
        png = agent._render_grid_image([[[0, 1], [2, 3]]])
        assert png is not None
        assert png.startswith(b"\x89PNG")
