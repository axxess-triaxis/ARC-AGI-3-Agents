"""Play a real ARC-AGI-3 game yourself, from the terminal.

    python play.py                 # plays ls20 (this session's usual test game)
    python play.py --game sk48     # plays a different game
    python play.py --game ls20 --tags human-baseline

Thin wrapper around main.py's own setup (API auth, scorecard, game
resolution) with --agent fixed to "humanagent" -- see
agents/templates/human_agent.py for what actually runs. Reads real actions
from your keyboard and submits them to the real API, same as any agent
here; your run gets a real scorecard.
"""

from __future__ import annotations

import argparse
import os
import sys

import main as main_module

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play an ARC-AGI-3 game yourself.")
    parser.add_argument("-g", "--game", default="ls20", help="Game id to play (default: ls20).")
    parser.add_argument("-t", "--tags", default=None, help="Comma-separated scorecard tags.")
    args = parser.parse_args()

    forwarded = ["play.py", "--agent", "humanagent", "--game", args.game]
    if args.tags:
        forwarded += ["--tags", args.tags]
    sys.argv = forwarded

    os.environ["TESTING"] = "False"
    main_module.main()
