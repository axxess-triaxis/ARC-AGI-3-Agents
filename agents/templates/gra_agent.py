"""GRAAgent: wires general-reasoning-agent's Cortex/ControlLoop into a real
ARC-AGI-3 game, in place of the one-shot "prompt an LLM with the current
frame" pattern every other template in this repo uses.

Why: ARC-AGI-3 is deliberately an unmapped, instruction-free environment --
no stated objective, novel per-game rules, sparse feedback. A single-turn LLM
call has no persistent world model, no hypothesis memory, no goal inference
carried between turns; each tick is independent of every other. Cortex was
built for exactly this class of problem (see general-reasoning-agent's own
README) but has never been pointed at a real interactive environment before
-- only its own synthetic benchmark levels.

Hard requirement: every game must start from EXACT ZERO prior context. This
holds by construction, not by convention -- GRAAgent.__init__ builds a brand
new Cortex/ControlLoop/FrameTranslator every time (ARC-AGI-3-Agents itself
already instantiates one Agent object per game), nothing is loaded from disk,
and general-reasoning-agent's own persistence.py is deliberately never
imported here. Nothing at module scope holds state across instances either.

Scope of this first version (see module README / MILESTONES for follow-ups):
- Reasoner is the plain HeuristicReasoner (no LLM). This lets the wiring
  itself -- frame-to-Observation translation, action mapping, the control
  loop's state machine -- be verified in isolation before adding an
  LLM-backed Reasoner as a separate, later change.
- objective_text is deliberately non-leaking: nothing here hints at what
  "winning" looks like, matching the benchmark's own no-instructions design.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from arcengine import FrameData, GameAction, GameState

from envs.base import Observation
from gra.control.loop import ControlLoop
from gra.cortex import Cortex

from ..agent import Agent

logger = logging.getLogger()

_NO_OBJECTIVE_TEXT = (
    "unknown -- infer the win condition and useful actions from how the "
    "environment responds; nothing here states the goal"
)

# Two entities of the same color are considered "the same object, moved" if
# their centroids are within this many cells of each other tick-to-tick.
# ls20-class games are small grids (well under 64x64 in practice); this is
# generous enough to track a moving object without conflating two distinct
# same-colored objects that happen to be near each other.
_MATCH_DISTANCE = 6.0

def _action_names(actions: Optional[list[Any]]) -> list[str]:
    """FrameData.available_actions comes back as raw ints, not GameAction
    instances -- GameAction is a plain Enum (not IntEnum), and pydantic
    coerces to the annotated int type during validation. Same quirk already
    handled in openclaw_agent.py's _action_names(); duplicated rather than
    imported since that one lives on an unrelated agent class."""
    out: list[str] = []
    for a in actions or []:
        if isinstance(a, GameAction):
            out.append(a.name)
            continue
        matched = next((m for m in GameAction if m.value == a), None)
        out.append(matched.name if matched else str(a))
    return out


# Hard cap on tracked entities per tick. A pathological grid (e.g. a fine
# checkerboard) could otherwise produce thousands of one-cell "entities";
# WorldModel has no cap of its own, so this repo enforces one here and logs
# when it's hit rather than silently degrading or crashing.
_MAX_ENTITIES = 60


class FrameTranslator:
    """Converts ARC-AGI-3's raw grid frames into gra's Observation shape.

    Stateful across ticks (holds the previous tick's entities for matching),
    but scoped to one GRAAgent instance -- i.e. one game -- satisfying the
    zero-context-per-game requirement automatically.
    """

    def __init__(self) -> None:
        self._next_entity_num = 0
        # entity_id -> {"color": int, "centroid": (r, c), "bbox": tuple, "size": int}
        self._prev_entities: dict[str, dict[str, Any]] = {}
        self._prev_levels_completed: int = 0

    def translate(
        self,
        frame: FrameData,
        tick: int,
        actions_remaining: float,
    ) -> Observation:
        grid = self._latest_grid(frame)
        entities = self._segment(grid) if grid else {}
        matched, events = self._match_and_diff(entities)

        levels_completed = frame.levels_completed or 0
        if levels_completed != self._prev_levels_completed:
            events.append(
                f"levels_completed changed: {self._prev_levels_completed} -> {levels_completed}"
            )
        self._prev_levels_completed = levels_completed

        available = _action_names(frame.available_actions)
        state = frame.state
        done = state in (GameState.WIN, GameState.GAME_OVER)
        goal_satisfied = state == GameState.WIN

        self._prev_entities = matched
        return Observation(
            tick=tick,
            objective_text=_NO_OBJECTIVE_TEXT,
            entities={
                eid: {"color": e["color"], "centroid": e["centroid"], "size": e["size"]}
                for eid, e in matched.items()
            },
            reliability=1.0,
            available_actions=available,
            resource_readouts={"actions_remaining": float(actions_remaining)},
            # ARC-AGI-3 exposes no measurable within-level progress signal --
            # leaving this None (rather than fabricating one) is the honest
            # reading, and is exactly the case Cortex's goal-ambiguity path
            # already handles.
            goal_progress=None,
            goal_satisfied=goal_satisfied,
            done=done,
            events=events,
            contradicts_prior=False,
        )

    def suggest_click_target(self) -> tuple[int, int]:
        """Where to click for a complex (coordinate) action when the
        reasoner picked one but supplied no coordinates. Targets the
        most-recently-appeared entity (an honest "poke at what's new" default,
        not a guess at game-specific semantics); falls back to a fixed
        interior point if nothing is tracked yet."""
        for eid, e in self._prev_entities.items():
            if e.get("is_new"):
                r, c = e["centroid"]
                return int(round(c)), int(round(r))
        if self._prev_entities:
            any_entity = next(iter(self._prev_entities.values()))
            r, c = any_entity["centroid"]
            return int(round(c)), int(round(r))
        return 32, 32

    # -- internals ---------------------------------------------------------
    def _latest_grid(self, frame: FrameData) -> Optional[list[list[int]]]:
        if not frame.frame:
            return None
        return frame.frame[-1]

    def _segment(self, grid: list[list[int]]) -> list[dict[str, Any]]:
        if not grid or not grid[0]:
            return []
        n_rows, n_cols = len(grid), len(grid[0])

        counts: dict[int, int] = {}
        for row in grid:
            for v in row:
                counts[v] = counts.get(v, 0) + 1
        background = max(counts, key=lambda v: counts[v])

        visited = [[False] * n_cols for _ in range(n_rows)]
        components: list[dict[str, Any]] = []

        for r0 in range(n_rows):
            for c0 in range(n_cols):
                if visited[r0][c0] or grid[r0][c0] == background:
                    continue
                color = grid[r0][c0]
                stack = [(r0, c0)]
                visited[r0][c0] = True
                cells: list[tuple[int, int]] = []
                while stack:
                    r, c = stack.pop()
                    cells.append((r, c))
                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nr, nc = r + dr, c + dc
                        if (
                            0 <= nr < n_rows
                            and 0 <= nc < n_cols
                            and not visited[nr][nc]
                            and grid[nr][nc] == color
                        ):
                            visited[nr][nc] = True
                            stack.append((nr, nc))
                rs = [p[0] for p in cells]
                cs = [p[1] for p in cells]
                components.append(
                    {
                        "color": color,
                        "centroid": (sum(rs) / len(rs), sum(cs) / len(cs)),
                        "bbox": (min(rs), max(rs), min(cs), max(cs)),
                        "size": len(cells),
                    }
                )

        if len(components) > _MAX_ENTITIES:
            logger.warning(
                "FrameTranslator: %d connected components found, capping to "
                "the %d largest (grid likely too fine-grained for object-level "
                "tracking as-is)",
                len(components),
                _MAX_ENTITIES,
            )
            components.sort(key=lambda c: c["size"], reverse=True)
            components = components[:_MAX_ENTITIES]

        return components

    def _match_and_diff(
        self, components: list[dict[str, Any]]
    ) -> tuple[dict[str, dict[str, Any]], list[str]]:
        events: list[str] = []
        matched: dict[str, dict[str, Any]] = {}
        remaining_prev = dict(self._prev_entities)

        for comp in components:
            best_id, best_dist = None, _MATCH_DISTANCE
            for eid, prev in remaining_prev.items():
                if prev["color"] != comp["color"]:
                    continue
                dr = prev["centroid"][0] - comp["centroid"][0]
                dc = prev["centroid"][1] - comp["centroid"][1]
                dist = (dr * dr + dc * dc) ** 0.5
                if dist < best_dist:
                    best_id, best_dist = eid, dist

            if best_id is not None:
                prev = remaining_prev.pop(best_id)
                if best_dist > 0.5:
                    events.append(
                        f"{best_id} (color={comp['color']}) moved "
                        f"{prev['centroid']} -> {comp['centroid']}"
                    )
                matched[best_id] = {**comp, "is_new": False}
            else:
                new_id = f"entity-{self._next_entity_num}"
                self._next_entity_num += 1
                events.append(
                    f"{new_id} (color={comp['color']}) appeared at {comp['centroid']}"
                )
                matched[new_id] = {**comp, "is_new": True}

        for eid, prev in remaining_prev.items():
            events.append(f"{eid} (color={prev['color']}) disappeared")

        return matched, events


class GRAAgent(Agent):
    """Plays one ARC-AGI-3 game via a fresh general-reasoning-agent Cortex."""

    MAX_ACTIONS = 200

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # Fresh every time -- see module docstring on the zero-context
        # requirement. No persistence.* import anywhere in this class.
        self.cortex = Cortex(objective=_NO_OBJECTIVE_TEXT)
        self.control_loop = ControlLoop(self.cortex)
        self.translator = FrameTranslator()
        self._finalized = False

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        return latest_frame.state in (GameState.WIN, GameState.GAME_OVER)

    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        observation = self.translator.translate(
            latest_frame,
            tick=self.action_counter,
            actions_remaining=max(0, self.MAX_ACTIONS - self.action_counter),
        )

        if observation.done:
            if not self._finalized:
                self.control_loop.finalize(observation)
                self._finalized = True
            # Base Agent.main() checks is_done() before calling choose_action
            # again, but guard here too in case this is ever driven directly.
            return GameAction.RESET

        if not observation.available_actions:
            logger.warning(
                "%s: no available_actions on this frame; defaulting to RESET",
                self.game_id,
            )
            return GameAction.RESET

        action_name, args = self.control_loop.tick(observation)
        action = self._resolve_action(action_name, observation.available_actions)
        self._apply_args(action, args)
        rationale = self._current_rationale()
        if rationale:
            action.reasoning = {"text": rationale}
        return action

    # -- internals ---------------------------------------------------------
    def _resolve_action(self, action_name: str, available: list[str]) -> GameAction:
        """The reasoner picks from `available` (real per-tick legality), so
        this should always resolve -- the fallback exists for the case where
        an intervening state change made the chosen action illegal between
        proposal and here, not expected in practice but never silently
        crashing on an unfamiliar string."""
        try:
            return GameAction.from_name(action_name)
        except (KeyError, ValueError, AttributeError):
            logger.warning(
                "%s: could not resolve action '%s' to a GameAction; using "
                "first available action instead",
                self.game_id,
                action_name,
            )
            return GameAction.from_name(available[0]) if available else GameAction.ACTION1

    def _apply_args(self, action: GameAction, args: dict[str, Any]) -> None:
        if not action.is_complex:
            return
        x = args.get("x")
        y = args.get("y")
        if x is None or y is None:
            x, y = self.translator.suggest_click_target()
        action.set_data({"game_id": self.game_id, "x": x, "y": y})

    def _current_rationale(self) -> str:
        proposal = self.cortex._last_action_proposal
        return proposal.rationale if proposal is not None else ""
