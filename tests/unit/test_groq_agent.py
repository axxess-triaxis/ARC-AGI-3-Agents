"""Unit tests for GroqReasoningAgent: the client-construction override and
the fail-fast guard on a missing GROQ_API_KEY. The inherited
ReasoningLLM/LLM behavior (message building, function calling, token
tracking) is already covered by the base class's own tests; this file
only covers what GroqReasoningAgent itself changes.
"""

from unittest.mock import MagicMock

import pytest

from agents.templates.groq_agent import GroqReasoningAgent


def _make_agent(monkeypatch: pytest.MonkeyPatch) -> GroqReasoningAgent:
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test_key")
    return GroqReasoningAgent(
        card_id="card-abc",
        game_id="ls20-test",
        agent_name="groq-test",
        ROOT_URL="http://localhost",
        record=False,
        arc_env=MagicMock(),
    )


@pytest.mark.unit
class TestGroqReasoningAgent:
    def test_raises_without_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
            GroqReasoningAgent(
                card_id="card-abc",
                game_id="ls20-test",
                agent_name="groq-test",
                ROOT_URL="http://localhost",
                record=False,
                arc_env=MagicMock(),
            )

    def test_constructs_with_api_key_present(self, monkeypatch: pytest.MonkeyPatch) -> None:
        agent = _make_agent(monkeypatch)
        assert agent.MODEL == "openai/gpt-oss-120b"

    def test_build_client_points_at_groq_with_groq_key(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        agent = _make_agent(monkeypatch)
        client = agent._build_client()
        assert str(client.base_url).rstrip("/") == GroqReasoningAgent.GROQ_BASE_URL
        assert client.api_key == "gsk_test_key"

    def test_does_not_use_openai_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Even if OPENAI_API_KEY happens to be set in the environment,
        # GroqReasoningAgent must never pick it up -- it's a distinct
        # provider with its own key.
        monkeypatch.setenv("OPENAI_API_KEY", "sk-should-not-be-used")
        agent = _make_agent(monkeypatch)
        client = agent._build_client()
        assert client.api_key == "gsk_test_key"


@pytest.mark.unit
class TestPrettyPrint3dHexEncoding:
    """The whole point of this override: the base LLM.pretty_print_3d
    renders a Python list literal per row (verbose); this one renders one
    hex character per cell. Confirmed live to cut a real ARC-AGI-3 grid
    dump from ~13,000 tokens to comfortably under Groq's 8,000/min cap."""

    def test_single_row_renders_as_hex_string(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        agent = _make_agent(monkeypatch)
        out = agent.pretty_print_3d([[[0, 1, 10, 15]]])
        assert "01af" in out.replace(" ", "")

    def test_output_is_much_shorter_than_list_repr(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        agent = _make_agent(monkeypatch)
        row = list(range(16)) * 4  # 64 cells, values 0-15
        grid = [[row for _ in range(64)]]
        hex_out = agent.pretty_print_3d(grid)
        list_repr_len = len(str(row)) * 64
        assert len(hex_out) < list_repr_len / 3

    def test_multiple_grid_planes_each_labeled(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        agent = _make_agent(monkeypatch)
        out = agent.pretty_print_3d([[[0]], [[1]]])
        assert "Grid 0" in out
        assert "Grid 1" in out


@pytest.mark.unit
class TestBuildToolsDropsStrictMode:
    """Confirmed live against a real ARC-AGI-3 game: Groq's schema
    validator rejects the base template's strict=True tool definitions
    for no-argument actions (RESET, ACTION1-5) with 'required present but
    properties is missing' -- a real vendor-specific validator difference
    from OpenAI's own, not a malformed schema."""

    def test_no_tool_sets_strict_flag(self, monkeypatch: pytest.MonkeyPatch) -> None:
        agent = _make_agent(monkeypatch)
        tools = agent.build_tools()
        assert tools  # sanity: there are tools to check at all
        for tool in tools:
            assert "strict" not in tool["function"]

    def test_still_covers_all_seven_actions(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        agent = _make_agent(monkeypatch)
        names = {t["function"]["name"] for t in agent.build_tools()}
        assert names == {
            "RESET",
            "ACTION1",
            "ACTION2",
            "ACTION3",
            "ACTION4",
            "ACTION5",
            "ACTION6",
        }

    def test_action6_keeps_its_xy_parameters(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        agent = _make_agent(monkeypatch)
        action6 = next(
            t for t in agent.build_tools() if t["function"]["name"] == "ACTION6"
        )
        assert set(action6["function"]["parameters"]["properties"]) == {"x", "y"}
