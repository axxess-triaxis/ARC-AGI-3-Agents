# GRALLMAgent: first live runs on `ls20`, 2026-10-05

Agent: `GRALLMAgent` (general-reasoning-agent's `MultiChainReasoner` with up to 3 chains per tick,
Groq `openai/gpt-oss-20b`, free tier, this repo's own Groq key). Game: `ls20` (7 levels). Each run
starts from zero prior context. Every number below is read from that run's log in `runs/` or its
scorecard, not estimated.

| Run | Code | Scorecard | Levels | Actions | Ended by | Groq calls (200 / 429) |
|---|---|---|---|---|---|---|
| 1 | `gra-agent-integration` as of 2026-09-18 | [9ef4640d…](https://arcprize.org/scorecards/9ef4640d-591a-4691-9448-87d69e2cc0f8) | 0/7 | 4 | **Crash**: an uncaught Groq per-minute 429 (TPM limit 8000, used 6965, requested 1690) killed the agent thread | 6 / 1 |
| 2 | + retry on per-minute 429 (`6fc1083`) | [736cce6b…](https://arcprize.org/scorecards/736cce6b-a5a5-4a1f-95df-b9d242c094e4) | 0/7 | 201 | The 200-action cap. **Moves 8–200 were all `ACTION1` with no LLM call** (dead-end lock-in, below) | 7 / 1 |
| 3 | + dead-end escape rotation (general-reasoning-agent `9465022`) | [78633fa3…](https://arcprize.org/scorecards/78633fa3-2b1b-4615-9e34-6186b6bd4c97) | 0/7 | 129 | **The game's own `GAME_OVER`** at action 129 (128 `NOT_FINISHED` frames, then `GAME_OVER`, per the recording) | 19 / 10 |

Run 3's action mix: `ACTION1` 39, `ACTION2` 30, `ACTION3` 30, `ACTION4` 30. All 10 per-minute
429s were waited out; none crashed the run. It ran for 186 s at an average of 0.69 actions/s.

## Bugs found by these runs and fixed

1. **Per-minute 429 crashed the run.** The client used `max_retries=0` and `chat_fn` didn't catch
   `RateLimitError`.
   - Now a per-minute 429 waits for the time Groq states (the `retry-after` header, else the
     "try again in Xs" text) plus 0.5 s, capped at 90 s, for up to 5 retries.
   - A per-day 429 raises `GroqDailyQuotaExhausted` at once.
   - Commit `6fc1083`, branch `fix/grallm-groq-rate-limit-retry`.
2. **Escaping a dead end always chose the first action.**
   - Cause: `MultiChainReasoner._escape_without_llm` ranked actions by `frontier_score`, which is
     0.0 for every action at a blocked node. `max()` therefore always returned `ACTION1`, which
     changed nothing, so the agent never left the dead end.
   - Fix: a new `GraphExplorer.escape_score` (untried first, then `1/(2 + attempts)`) rotates
     through every action.
   - Commit `9465022` in general-reasoning-agent, branch `fix/dead-end-escape-rotation`.
   - GRAAgent (no LLM) uses the same `GraphExplorer`. Whether this bug also affected its earlier
     0/7 has not been checked.

## What these numbers do and do not show

- **No level was completed in any run.** Run 3 is the only one where the agent reasoned and
  explored freely until the game itself ended.
- **The agent stops at `GAME_OVER`.** `GRAAgent.is_done` treats `GAME_OVER` like `WIN`, so it never
  sends `RESET` to retry the level. A longer, fairer attempt at level 1 needs `RESET` handling first.
- **Small sample.** Three runs, one game, one model on a free tier. This is not a capability
  measurement of gpt-oss-20b or of the architecture.
- **Exit code.** Every run, including the clean ones, ended with process exit code 2 after the
  scorecard was closed. The cause is not investigated.
