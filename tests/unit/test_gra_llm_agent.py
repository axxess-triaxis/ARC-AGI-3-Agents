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


# --- Groq rate limits (first live run, 2026-10-05, died on a per-minute 429) ---

import httpx  # noqa: E402
from openai import RateLimitError  # noqa: E402

from agents.templates import gra_llm_agent as gla  # noqa: E402


def _rate_limit_error(message: str, retry_after: str | None = None) -> RateLimitError:
    headers = {"retry-after": retry_after} if retry_after else {}
    response = httpx.Response(429, headers=headers, request=httpx.Request("POST", "https://api.groq.com"))
    return RateLimitError(message, response=response, body=None)


_TPM = ("Rate limit reached for model `openai/gpt-oss-20b` on tokens per minute (TPM): "
        "Limit 8000, Used 6965, Requested 1690. Please try again in 4.9125s.")
_TPD = ("Rate limit reached for model `openai/gpt-oss-20b` on tokens per day (TPD): "
        "Limit 200000, Used 199990, Requested 1700. Please try again in 12m3.5s.")


def _ok(text: str):
    return MagicMock(choices=[MagicMock(message=MagicMock(content=text))])


def test_per_minute_429_waits_as_told_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = _make_agent(monkeypatch)
    sleeps: list[float] = []
    monkeypatch.setattr(gla.time, "sleep", sleeps.append)
    client = MagicMock()
    client.chat.completions.create.side_effect = [_rate_limit_error(_TPM), _ok("answer")]

    assert agent._make_chat_fn(client)("prompt") == "answer"
    assert sleeps == [pytest.approx(5.4125)]
    assert client.chat.completions.create.call_count == 2


def test_retry_after_header_wins_over_the_message(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = _make_agent(monkeypatch)
    sleeps: list[float] = []
    monkeypatch.setattr(gla.time, "sleep", sleeps.append)
    client = MagicMock()
    client.chat.completions.create.side_effect = [_rate_limit_error(_TPM, retry_after="7"), _ok("x")]

    agent._make_chat_fn(client)("prompt")
    assert sleeps == [7.5]


def test_daily_quota_stops_immediately_without_waiting(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = _make_agent(monkeypatch)
    sleeps: list[float] = []
    monkeypatch.setattr(gla.time, "sleep", sleeps.append)
    client = MagicMock()
    client.chat.completions.create.side_effect = _rate_limit_error(_TPD)

    with pytest.raises(gla.GroqDailyQuotaExhausted):
        agent._make_chat_fn(client)("prompt")
    assert sleeps == [] and client.chat.completions.create.call_count == 1


def test_gives_up_after_max_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = _make_agent(monkeypatch)
    monkeypatch.setattr(gla.time, "sleep", lambda s: None)
    client = MagicMock()
    client.chat.completions.create.side_effect = _rate_limit_error(_TPM)

    with pytest.raises(RateLimitError):
        agent._make_chat_fn(client)("prompt")
    assert client.chat.completions.create.call_count == GRALLMAgent.MAX_RATE_LIMIT_RETRIES + 1
