"""OpenClaw Vision agent.

A competition-facing extension of `OpenClaw` (see `openclaw_agent.py`) that
closes the two gaps its own README names as unaddressed:

1. **No vision.** The base agent serializes the grid as hex text only,
   because OpenClaw's OpenAI-compat endpoint "does not document image
   input." Claude (the provider this agent targets, via OPENCLAW_MODEL=
   anthropic/claude-*) is strong at grid/spatial reasoning from images, and
   the underlying `/v1/chat/completions` proxy is OpenAI-format, which does
   support `image_url` content blocks for vision-capable models. This agent
   sends both the hex text (exact, lossless) and a rendered PNG (fast
   pattern/shape recognition) every turn, and falls back to text-only
   automatically if the gateway/model rejects multimodal content — a real
   possibility per the README's own uncertainty, not a guaranteed win.
2. **No stagnation detection.** The base agent has no memory of whether its
   last action actually changed anything. Across unmapped games it is easy
   to burn the 80-action budget repeating a no-op (e.g. walking into a
   wall). This agent hashes each frame, tracks a run of identical hashes,
   and — once an action produces zero visible change twice in a row —
   injects an explicit warning naming the repeated action so the model is
   pushed to try something else, without ever telling it *which* game rule
   caused the stall (that stays the model's job, preserving generalization).

Both additions are game-agnostic: neither reads or reacts to any specific
game_id, color code, or object shape. They only use signals every ARC-AGI-3
game exposes identically (frame arrays, available_actions, state).
"""

import base64
import hashlib
import io
import logging
from typing import Any, List, Optional

import openai
from arcengine import FrameData, GameAction, GameState
from PIL import Image, ImageDraw

from .openclaw_agent import OpenClaw

logger = logging.getLogger()


class OpenClawVision(OpenClaw):
    """OpenClaw gateway agent with an added image frame and stall detection."""

    # ARC-AGI-3's published 16-color key (see reasoning_agent.py's template
    # for the same table) -- fixed across every game, not game-specific.
    _PALETTE = {
        0: "#FFFFFF",
        1: "#CCCCCC",
        2: "#999999",
        3: "#666666",
        4: "#333333",
        5: "#000000",
        6: "#E53AA3",
        7: "#FF7BCC",
        8: "#F93C31",
        9: "#1E93FF",
        10: "#88D8F1",
        11: "#FFDC00",
        12: "#FF851B",
        13: "#921231",
        14: "#4FCC30",
        15: "#A356D6",
    }
    _CELL_PX = 10
    _STALL_WARN_AFTER = 2  # repeats of an identical frame before we speak up

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._send_images = True  # flips to False permanently after a 400
        self._last_frame_hash: Optional[str] = None
        self._repeat_streak = 0
        self._last_action_name: Optional[str] = None

    @property
    def name(self) -> str:
        return f"{super().name}.vision"

    def choose_action(
        self, frames: List[FrameData], latest_frame: FrameData
    ) -> GameAction:
        if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            self._repeat_streak = 0
            self._last_frame_hash = None
            self._last_action_name = None
            return GameAction.RESET

        stall_note = self._update_stall_tracking(latest_frame)
        prompt_text = self._build_prompt(latest_frame)
        if stall_note:
            prompt_text = f"{stall_note}\n\n{prompt_text}"

        content: Any = prompt_text
        if self._send_images:
            image_bytes = self._render_grid_image(latest_frame.frame)
            if image_bytes is not None:
                b64 = base64.b64encode(image_bytes).decode()
                content = [
                    {"type": "text", "text": prompt_text},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    },
                ]

        extra_headers = (
            {"x-openclaw-model": self._model_override}
            if self._model_override
            else None
        )
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": content}],
                extra_headers=extra_headers,
            )
        except openai.BadRequestError as e:
            if self._send_images and content is not prompt_text:
                logger.warning(
                    f"OpenClaw rejected multimodal content ({e}); "
                    "falling back to text-only for the rest of this run."
                )
                self._send_images = False
                return self.choose_action(frames, latest_frame)
            logger.error(f"OpenClaw 400: {e}")
            logger.error(f"prompt: {prompt_text[:500]}")
            raise

        msg = response.choices[0].message
        if response.usage:
            self._track_tokens(response.usage.total_tokens, msg.content or "")

        blob = self._parse_blob(msg)
        action = self._action_from_blob(blob)
        action.reasoning = self._extract_reasoning(blob)
        self._last_action_name = action.name
        return action

    def _update_stall_tracking(self, latest_frame: FrameData) -> Optional[str]:
        """Hash the current frame, compare to the last one, and return a
        warning string once the same action has produced an identical frame
        `_STALL_WARN_AFTER` times in a row. Returns None otherwise."""
        current_hash = hashlib.sha256(
            repr(latest_frame.frame).encode("utf-8")
        ).hexdigest()

        if current_hash == self._last_frame_hash:
            self._repeat_streak += 1
        else:
            self._repeat_streak = 0
        self._last_frame_hash = current_hash

        if self._repeat_streak >= self._STALL_WARN_AFTER and self._last_action_name:
            return (
                "# STALL WARNING\n"
                f"Your last action ({self._last_action_name}) has produced "
                f"the exact same frame {self._repeat_streak} times in a row "
                "-- it is very likely having no effect right now (a wall, an "
                "already-open door, an out-of-range click, etc). Choose a "
                "genuinely different action or coordinates this turn instead "
                "of repeating it."
            )
        return None

    def _render_grid_image(
        self, frame_3d: Optional[List[List[List[int]]]]
    ) -> Optional[bytes]:
        """Render the last grid plane as a flat-color PNG. Returns None for
        an empty/missing frame rather than raising -- a rendering failure
        should never take down the whole turn."""
        if not frame_3d:
            return None
        try:
            grid = frame_3d[-1]
            if not grid or not grid[0]:
                return None
            height = len(grid)
            width = len(grid[0])
            img = Image.new(
                "RGB", (width * self._CELL_PX, height * self._CELL_PX), "white"
            )
            draw = ImageDraw.Draw(img)
            for y in range(height):
                for x in range(width):
                    color = self._PALETTE.get(grid[y][x], "#888888")
                    x0, y0 = x * self._CELL_PX, y * self._CELL_PX
                    draw.rectangle(
                        [x0, y0, x0 + self._CELL_PX, y0 + self._CELL_PX],
                        fill=color,
                    )
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            return buffer.getvalue()
        except Exception as e:  # defensive: never let rendering crash a turn
            logger.warning(f"Grid image render failed, continuing text-only: {e}")
            return None
