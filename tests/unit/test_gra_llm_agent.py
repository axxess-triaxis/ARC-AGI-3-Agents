"""Unit tests for GRALLMAgent: the API-key guard and the chat_fn wiring
(MultiChainReasoner itself is already covered by general-reasoning-agent's
own test suite -- these tests cover only what this class adds: client
construction and the real-vs-fake API boundary). No real network calls.
"""

import json
from unittest.mock import MagicMock

import pytest
from arcengine import FrameData, GameAction, GameState

from agents.templates.gra_llm_agent import GRALLMAgent
from gra.llm_reasoner import MultiChainReasoner

_VALID_RESPONSE = {
    "action": "ACTION1",
    "known": [], "inferred": [], "assumed": [], "unknown": [],
    "competing_hypotheses": [
        {"statement": "a", "confidence": 0.6},
        {"statement": "b", "confidence": 0.4},
    ],
    "falsification_test": "if nothing changes, wrong",
    "is_novel_situation": True,
    "predicted_outcome": "something changes",
    "predicted_confidence": 0.6,
    "reversible": True,
    "expected_benefit": 0.5,
    "expected_cost": 0.1,
    "is_exploration": True,
    "resource_check": "fine",
    "rationale": "test",
}


def _make_agent(monkeypatch: pytest.MonkeyPatch) -> GRALLMAgent:
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test_key")
    return GRALLMAgent(
        card_id="card-abc",
        game_id="ls20-test",
        agent_name="gra-llm-test",
        ROOT_URL="http://localhost",
        record=False,
        arc_env=MagicMock(),
    )


def _mock_openai_client(response_text: str) -> MagicMock:
    client = MagicMock()
    completion = MagicMock()
    completion.choices = [MagicMock(message=MagicMock(content=response_text))]
    client.chat.completions.create.return_value = completion
    return client


@pytest.mark.unit
class TestGRALLMAgentConstruction:
    def test_raises_without_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
            GRALLMAgent(
                card_id="card-abc",
                game_id="ls20-test",
                agent_name="gra-llm-test",
                ROOT_URL="http://localhost",
                record=False,
                arc_env=MagicMock(),
            )

    def test_constructs_with_multi_chain_reasoner(self, monkeypatch: pytest.MonkeyPatch) -> None:
        agent = _make_agent(monkeypatch)
        assert isinstance(agent.cortex.reasoner, MultiChainReasoner)

    def test_reasoner_uses_configured_chain_count(self, monkeypatch: pytest.MonkeyPatch) -> None:
        agent = _make_agent(monkeypatch)
        assert agent.cortex.reasoner.num_chains == GRALLMAgent.NUM_CHAINS
        assert agent.cortex.reasoner.agreement_threshold == GRALLMAgent.AGREEMENT_THRESHOLD


@pytest.mark.unit
class TestGRALLMAgentChooseAction:
    def test_chat_fn_calls_the_real_model_name_with_no_history(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        agent = _make_agent(monkeypatch)
        fake_client = _mock_openai_client(json.dumps(_VALID_RESPONSE))
        monkeypatch.setattr(agent, "_build_client", lambda: fake_client)
        # Rebuild the reasoner now that _build_client is patched.
        agent.cortex.reasoner = agent._build_reasoner()

        frame = FrameData(
            frame=[[[0, 5], [0, 0]]],
            state=GameState.NOT_FINISHED,
            available_actions=[GameAction.ACTION1, GameAction.ACTION2],
        )
        action = agent.choose_action([frame], frame)

        assert action in (GameAction.ACTION1, GameAction.ACTION2)
        call_kwargs = fake_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["model"] == GRALLMAgent.MODEL
        assert len(call_kwargs["messages"]) == 1  # single-shot, no history
        assert call_kwargs["messages"][0]["role"] == "user"

    def test_two_agreeing_calls_stop_before_a_third(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        agent = _make_agent(monkeypatch)
        fake_client = _mock_openai_client(json.dumps(_VALID_RESPONSE))
        monkeypatch.setattr(agent, "_build_client", lambda: fake_client)
        agent.cortex.reasoner = agent._build_reasoner()

        frame = FrameData(
            frame=[[[0, 5], [0, 0]]],
            state=GameState.NOT_FINISHED,
            available_actions=[GameAction.ACTION1, GameAction.ACTION2],
        )
        agent.choose_action([frame], frame)

        # Every call returns the same action -> consensus reached on the
        # 2nd call, 3rd never made.
        assert fake_client.chat.completions.create.call_count == 2
