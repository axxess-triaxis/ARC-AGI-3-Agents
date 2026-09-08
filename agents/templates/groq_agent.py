"""Groq-backed reasoning agent.

Calls Groq's OpenAI-compatible API directly with the repo's existing
lean, hand-built prompt (agents.LLM/ReasoningLLM) -- no OpenClaw gateway
in the path. This exists specifically because the OpenClaw gateway route
(see templates/openclaw_agent/) turned out to be structurally
incompatible with Groq's free tier: OpenClaw's own agent-wrapper
overhead (system prompt, tool definitions, its memory-core plugin)
measured ~20K tokens on a single trivial turn, live, against a gateway
onboarded with a real Groq key -- comfortably over every Groq free-tier
reasoning model's 8,000 tokens/minute cap before any real game content
is even sent. Calling Groq directly with ReasoningLLM's own minimal
prompt keeps real per-turn usage in the low thousands, which fits.

Text-only by design: an image encoded as base64 in a multimodal message
costs meaningful tokens on top of an already-tight 8K/min budget, so
this deliberately does not send the grid as an image (unlike
OpenClawVision) even though the chosen model may support vision --
ARC-AGI-3's grid is already a small, lossless integer matrix in text
form, so an image adds tokens without adding information here.

Verified live end-to-end against a real ARC-AGI-3 game (re86-8af5384d):
27 real actions submitted (RESET plus a genuine mix of ACTION2-6, not a
single action repeated), ~23 minutes, with the model producing real,
coherent turn-by-turn reasoning about the hex grid. The run ended not on
a bug but on Groq's separate **daily** quota for this model (tokens per
day, 200,000 -- distinct from the 8,000/minute cap this file's other
fixes address): "Rate limit reached ... on tokens per day (TPD): Limit
200000". That is a real, load-bearing operational constraint for anyone
running this agent through a full competition session on the free tier,
not a defect -- budget for it (fewer, longer-lived processes rather than
many short restarts; each restart re-pays a full RESET's context) or
expect to need Groq's paid Dev Tier for sustained play.
"""

import os
from typing import Any

from openai import OpenAI as OpenAIClient

from .llm_agents import ReasoningLLM


class GroqReasoningAgent(ReasoningLLM):
    """ReasoningLLM pointed at Groq's OpenAI-compatible endpoint instead
    of OpenAI's. Reuses ReasoningLLM's function-calling protocol as-is --
    Groq's `openai/gpt-oss-120b` advertises `tools` support in its own
    model catalog (confirmed live via `GET /openai/v1/models`), so no
    JSON-in-text fallback (like OpenClaw's) is needed here.

    Overrides `pretty_print_3d` for the same reason the OpenClaw gateway
    route turned out to be free-tier-incompatible, but confirmed live to
    be a *bigger* factor than that gateway's own overhead: the base
    `LLM.pretty_print_3d()` renders each grid row as a Python list literal
    (e.g. "[0, 1, 2, 3, ...]"), which alone measured ~13,000 tokens for a
    single grid dump on a real ARC-AGI-3 game -- already over every Groq
    free-tier model's 8,000 tokens/minute cap before any other context is
    added. A dense hex-string encoding (one hex char per cell, matching
    OpenClaw's own `_render_grid` approach) cuts that by roughly 10x while
    staying exact and lossless -- cell values are 0-15, i.e. one hex
    digit each."""

    MODEL = "openai/gpt-oss-120b"
    GROQ_BASE_URL = "https://api.groq.com/openai/v1"
    # Confirmed live: with DO_OBSERVATION on, the model kept emitting
    # tool-call-shaped output on the plain observation call too (which
    # declares no `tools` at all), and Groq's validator then rejects it
    # with "Tool choice is none, but model called a tool" -- a real
    # gpt-oss-120b/Groq behavior once tool_calls exist earlier in the
    # message history, not something a request-side flag can fix. Skipping
    # the observation step (same as the repo's own FastLLM) sidesteps it
    # and also shrinks per-turn token usage, which helps the 8K/min budget.
    DO_OBSERVATION = False

    def _build_client(self) -> OpenAIClient:
        return OpenAIClient(
            api_key=os.environ.get("GROQ_API_KEY", ""),
            base_url=self.GROQ_BASE_URL,
        )

    def pretty_print_3d(self, array_3d: list[list[list[Any]]]) -> str:
        lines = []
        for i, block in enumerate(array_3d):
            lines.append(f"Grid {i} (hex, one char per cell, values 0-f):")
            for row in block:
                lines.append("  " + "".join(f"{c:x}" for c in row))
            lines.append("")
        return "\n".join(lines)

    def build_tools(self) -> list[dict[str, Any]]:
        """Drops the base template's `"strict": True` tool flag. Confirmed
        live: Groq's schema validator rejects the base RESET/ACTIONn tool
        definitions (whose params are `{"type":"object","properties":{},
        "required":[],"additionalProperties":False}`) under strict mode
        with 'required present but properties is missing' -- a real
        vendor-specific JSON-Schema-strictness difference from OpenAI's
        own validator, not a malformed schema (the same schema works
        fine, non-strict, against Groq; ACTION6's x/y schema is unaffected
        either way since it always had properties)."""
        functions = self.build_functions()
        tools: list[dict[str, Any]] = []
        for f in functions:
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": f["name"],
                        "description": f["description"],
                        "parameters": f.get("parameters", {}),
                    },
                }
            )
        return tools

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if not os.environ.get("GROQ_API_KEY"):
            # Fail loudly and immediately rather than let every turn 401
            # against Groq with an opaque error deep in the openai SDK.
            raise RuntimeError(
                "GroqReasoningAgent requires GROQ_API_KEY to be set "
                "(see .env.example)."
            )
