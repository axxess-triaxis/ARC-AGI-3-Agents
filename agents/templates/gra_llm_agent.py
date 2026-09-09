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

import os

from openai import OpenAI

from gra.llm_reasoner import MultiChainReasoner
from gra.reasoner import Reasoner

from .gra_agent import GRAAgent


class GRALLMAgent(GRAAgent):
    MODEL = "openai/gpt-oss-20b"
    GROQ_BASE_URL = "https://api.groq.com/openai/v1"
    MAX_COMPLETION_TOKENS = 1500
    NUM_CHAINS = 3
    AGREEMENT_THRESHOLD = 2

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
            response = client.chat.completions.create(
                model=self.MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=self.MAX_COMPLETION_TOKENS,
                temperature=0.7,  # nonzero: 3 sequential calls with an identical
                # prompt need some sampling variance to ever disagree at all.
            )
            return response.choices[0].message.content or ""

        return chat_fn
