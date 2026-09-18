"""HumanAgent: lets a person play a real ARC-AGI-3 game from the terminal,
same real API a real agent uses -- for a human baseline to compare agents
against (the Google AI Overview screenshot earlier this session cited a
66% average human score on ARC-AGI-2's public evaluation; this is the
AGI-3 equivalent: play it yourself, see the real scorecard).
"""

from __future__ import annotations

from typing import Any, Optional

from arcengine import FrameData, GameAction, GameState


def _action_names(actions: Optional[list[Any]]) -> list[str]:
    """Same quirk as gra_agent.py/openclaw_agent.py: available_actions
    comes back as raw ints (GameAction is a plain Enum, pydantic coerces to
    the annotated int type), not GameAction instances."""
    out: list[str] = []
    for a in actions or []:
        if isinstance(a, GameAction):
            out.append(a.name)
            continue
        matched = next((m for m in GameAction if m.value == a), None)
        out.append(matched.name if matched else str(a))
    return out


from ..agent import Agent


class HumanAgent(Agent):
    """Prints the current grid and prompts for the next action every turn.

    Deliberately does not track any world model or hint at the objective --
    a human baseline should face exactly what an agent faces: no
    instructions, infer everything from what the grid does in response.
    """

    MAX_ACTIONS = 500

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        return latest_frame.state in (GameState.WIN, GameState.GAME_OVER)

    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        self._print_frame(latest_frame)
        available = _action_names(latest_frame.available_actions)

        while True:
            raw = input(f"Action {available} (or 'quit'): ").strip().upper()
            if raw in ("QUIT", "Q"):
                print("Quitting -- your scorecard so far will still be reported.")
                raise SystemExit(0)

            try:
                action = GameAction.from_name(raw)
            except (KeyError, ValueError, AttributeError):
                print(f"'{raw}' isn't a real action. Try one of {available}.")
                continue

            if action.name not in available:
                print(f"'{raw}' isn't available on this turn. Try one of {available}.")
                continue

            if action.is_complex():  # a METHOD, not a property -- see gra_agent.py's note
                try:
                    x = int(input("  x (column): ").strip())
                    y = int(input("  y (row): ").strip())
                except ValueError:
                    print("x and y must be whole numbers. Try again.")
                    continue
                action.set_data({"game_id": self.game_id, "x": x, "y": y})

            return action

    def _print_frame(self, frame: FrameData) -> None:
        grid = frame.frame[-1] if frame.frame else []
        state_name = frame.state.name if hasattr(frame.state, "name") else str(frame.state)
        print(
            f"\n=== {self.game_id} | turn {self.action_counter} | state={state_name} "
            f"| levels_completed={frame.levels_completed} ==="
        )
        for row in grid:
            print(" ".join(f"{c:2d}" for c in row))
