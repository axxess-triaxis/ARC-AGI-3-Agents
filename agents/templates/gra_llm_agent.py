"""GRALLMAgent: GRAAgent with general-reasoning-agent's MultiChainReasoner
(language reasoning + graph-based exploration, see gra.llm_reasoner) in
place of the plain HeuristicReasoner.

Model choice: openai/gpt-oss-120b's daily Groq quota was already exhausted
this session from AGI-2 testing; gpt-oss-20b has its own independent
200,000 TPD pool and is already confirmed live to work mechanically on
real ARC-AGI-3 turns (48-129 actions taken in earlier runs this session).
Same base_url/client pattern as GroqReasoningAgent (arc-agi-benchmarking
repo) and groq_agent.py (this repo) -- nothing new about the transport,
only the reasoning architecture behind it.

max_completion_tokens is set high enough for gpt-oss-20b's own internal
chain-of-thought (confirmed live this session: it reasons before emitting
content, even on trivial prompts) to actually complete before also writing
the required structured JSON. Kept only as large as needed, not the
model's max, since MultiChainReasoner already makes up to 3x the calls of
a single-reasoner agent per tick -- output-token cost matters more here.
"""

from __future__ import annotations

import logging
import os
import re
import time

from openai import OpenAI, RateLimitError

from gra.llm_reasoner import MultiChainReasoner
from gra.reasoner import Reasoner

from .gra_agent import GRAAgent

logger = logging.getLogger(__name__)

# Groq's 429 body says which limit was hit and how long to wait, e.g.
# "... on tokens per minute (TPM): Limit 8000, Used 6965, Requested 1690.
# Please try again in 4.9125s." (first live GRALLMAgent run, 2026-10-05).
_RETRY_IN = re.compile(r"try again in (?:(\d+)m)?([\d.]+)s", re.IGNORECASE)
_DAILY_LIMIT = re.compile(r"per day \((?:TPD|RPD)\)", re.IGNORECASE)


class GroqDailyQuotaExhausted(RuntimeError):
    """The model's daily Groq quota is spent; waiting minutes won't help."""


def _retry_after_seconds(error: RateLimitError) -> float | None:
    """Seconds Groq asked us to wait, from the retry-after header or the message."""
    header = getattr(getattr(error, "response", None), "headers", {}) or {}
    try:
        if header.get("retry-after"):
            return float(header["retry-after"])
    except (TypeError, ValueError):
        pass
    match = _RETRY_IN.search(str(error))
    if match:
        return int(match.group(1) or 0) * 60 + float(match.group(2))
    return None


class GRALLMAgent(GRAAgent):
    MODEL = "openai/gpt-oss-20b"
    GROQ_BASE_URL = "https://api.groq.com/openai/v1"
    MAX_COMPLETION_TOKENS = 1500
    NUM_CHAINS = 3
    AGREEMENT_THRESHOLD = 2
    # Free tier: 8,000 tokens/minute per model and each call is ~1.7k tokens, so a
    # 3-chain tick can trip the per-minute limit. Wait as told and retry.
    MAX_RATE_LIMIT_RETRIES = 5
    MAX_RATE_LIMIT_WAIT_S = 90.0

    def _build_reasoner(self) -> Reasoner:
        chat_fn = self._make_chat_fn(self._build_client())
        return MultiChainReasoner(
            chat_fn=chat_fn,
            num_chains=self.NUM_CHAINS,
            agreement_threshold=self.AGREEMENT_THRESHOLD,
        )

    def _build_client(self) -> OpenAI:
        api_key = os.environ.get("GROQ_API_KEY", "")
        if not api_key:
            raise RuntimeError(
                "GRALLMAgent requires GROQ_API_KEY to be set (see .env.example)."
            )
        return OpenAI(api_key=api_key, base_url=self.GROQ_BASE_URL, max_retries=0)

    def _make_chat_fn(self, client: OpenAI):
        def chat_fn(prompt: str) -> str:
            # Deliberately a single-message, single-shot call -- no
            # conversation history -- every time this is invoked. That
            # statelessness is the whole point (see llm_reasoner.py's
            # module docstring on "0 context" and the Groq tool-call-bleed
            # bug it avoids).
            for attempt in range(self.MAX_RATE_LIMIT_RETRIES + 1):
                try:
                    response = client.chat.completions.create(
                        model=self.MODEL,
                        messages=[{"role": "user", "content": prompt}],
                        max_completion_tokens=self.MAX_COMPLETION_TOKENS,
                        temperature=0.7,  # nonzero: 3 sequential calls with an identical
                        # prompt need some sampling variance to ever disagree at all.
                    )
                    return response.choices[0].message.content or ""
                except RateLimitError as error:
                    if _DAILY_LIMIT.search(str(error)):
                        raise GroqDailyQuotaExhausted(
                            f"Groq daily quota for {self.MODEL} is exhausted; stopping this run."
                        ) from error
                    if attempt == self.MAX_RATE_LIMIT_RETRIES:
                        raise
                    wait = _retry_after_seconds(error)
                    # Small margin past Groq's own estimate; a sane default when it gives none.
                    wait = min((wait + 0.5) if wait is not None else 10.0, self.MAX_RATE_LIMIT_WAIT_S)
                    logger.info(
                        "Groq per-minute limit on %s; waiting %.1fs (retry %d/%d)",
                        self.MODEL, wait, attempt + 1, self.MAX_RATE_LIMIT_RETRIES,
                    )
                    time.sleep(wait)
            raise AssertionError("unreachable")

        return chat_fn
