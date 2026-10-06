# GRALLMAgent: first live runs on `ls20`, 2026-10-05 and 06

Agent: `GRALLMAgent` (general-reasoning-agent's `MultiChainReasoner` with up to 3 chains per tick,
Groq `openai/gpt-oss-20b`, free tier, this repo's own Groq key). Game: `ls20` (7 levels). Each run
starts from zero prior context. Every number below is read from that run's log in `runs/` or its
scorecard, not estimated.

| Run | Code | Scorecard | Levels | Actions | Ended by | Groq calls (200 / 429) |
|---|---|---|---|---|---|---|
| 1 | `gra-agent-integration` as of 2026-09-18 | [9ef4640d…](https://arcprize.org/scorecards/9ef4640d-591a-4691-9448-87d69e2cc0f8) | 0/7 | 4 | **Crash**: an uncaught Groq per-minute 429 (TPM limit 8000, used 6965, requested 1690) killed the agent thread | 6 / 1 |
| 2 | + retry on per-minute 429 (`6fc1083`) | [736cce6b…](https://arcprize.org/scorecards/736cce6b-a5a5-4a1f-95df-b9d242c094e4) | 0/7 | 201 | The 200-action cap. **Moves 8–200 were all `ACTION1` with no LLM call** (dead-end lock-in, below) | 7 / 1 |
| 3 | + dead-end escape rotation (general-reasoning-agent `9465022`) | [78633fa3…](https://arcprize.org/scorecards/78633fa3-2b1b-4615-9e34-6186b6bd4c97) | 0/7 | 129 | **The game's own `GAME_OVER`** at action 129 (128 `NOT_FINISHED` frames, then `GAME_OVER`, per the recording) | 19 / 10 |
| 4 (2026-10-06) | + `RESET` and retry on `GAME_OVER` (`b8e7211`; general-reasoning-agent `c4455f4`) | [067e8eeb…](https://arcprize.org/scorecards/067e8eeb-fe8e-451f-bdc5-22e1aee89a2e) | 0/7 | 194 (attempt 1: 129, then `RESET`, attempt 2: 64) | **Groq daily token quota** for gpt-oss-20b (TPD limit 200,000; used 198,606, requested 1,537). The run stopped cleanly with `GroqDailyQuotaExhausted`, as designed | 148 / 147 |

Run 4: `ACTION1` 96, `ACTION2` 35, `ACTION3` 32, `ACTION4` 30, `RESET` 1. Attempt 1 again hit
`GAME_OVER` at exactly action 129, the same count as run 3, which suggests a fixed move budget per
attempt in `ls20` rather than one specific fatal move (not verified against the game's rules). It
took 1,461 s at 0.13 actions/s: 146 per-minute waits. The model was consulted far more often than
in run 3 (148 successful calls against 19), which used up the model's whole free daily token budget
in one game.

Run 3's action mix: `ACTION1` 39, `ACTION2` 30, `ACTION3` 30, `ACTION4` 30. All 10 per-minute
429s were waited out; none crashed the run. It ran for 186 s at an average of 0.69 actions/s.

## No-LLM GRAAgent re-test with the `RESET` fix (2026-10-06)

| Agent | Code | Scorecard | Levels | Actions | Ended by | Groq calls |
|---|---|---|---|---|---|---|
| `graagent` (Cortex `HeuristicReasoner`, no LLM) | `b8e7211` (`RESET` on `GAME_OVER`) | [114410b1…](https://arcprize.org/scorecards/114410b1-d7c4-4068-8e2a-5783e1b9b5f8) | 0/7 | 201 (attempt 1: 129, then `RESET`, attempt 2: 71) | The 200-action cap | 0 |

Action mix: `ACTION1` 139, `ACTION2` 12, `ACTION3` 41, `ACTION4` 8, `RESET` 1. It took 155 s. Only the
`RESET` fix applies to this agent; the dead-end escape fix does not (see below).

**Three agents now hit `GAME_OVER` at exactly action 129 on the first attempt**: GRALLMAgent runs 3
and 4, and this GRAAgent run, with very different action mixes. That is strong evidence that `ls20`
ends an attempt after a fixed number of moves (128 actions, then `GAME_OVER`), whatever moves are
made. It is not yet confirmed against the game's own rules. If it holds, level 1 has to be solved
within 128 moves, and no agent so far has changed `levels_completed` at all within that budget.

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
   - This bug cannot have affected the no-LLM GRAAgent: `GraphExplorer` is used only by
     `MultiChainReasoner`, and GRAAgent runs Cortex's `HeuristicReasoner`, which never consults it.

## What these numbers do and do not show

- **No level was completed in any run.** Run 3 is the only one where the agent reasoned and
  explored freely until the game itself ended.
- **`GAME_OVER` used to end the run.** `GRAAgent.is_done` treated `GAME_OVER` like `WIN`, so no run
  ever retried the level. Fixed for run 4 (`b8e7211`): the agent now `RESET`s and keeps what it
  learned within the game.
- **The free tier allows about one game a day.** At this model's call rate, one 200-action game
  needs about all of gpt-oss-20b's 200,000 daily tokens on this key.
- **Small sample.** Four runs, one game, one model on a free tier. This is not a capability
  measurement of gpt-oss-20b or of the architecture.
- **Exit code.** Every run, including the clean ones, ended with process exit code 2 after the
  scorecard was closed. The cause is not investigated.
