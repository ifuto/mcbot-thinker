#!/usr/bin/env bash
# One-shot: behavior-clone warm-start from a scripted swordsman, then run PPO
# self-play on the C simulator. Designed to be left running (auto-resumes).
# On this 2-core box it sustains tens of thousands of agent-steps/sec, so a
# day of running = billions of simulated fight ticks across parallel arenas.
set -euo pipefail
cd "$(dirname "$0")/.."

export PYTHONPATH="$PWD"
PY=".venv/bin/python"
[ -x "$PY" ] || PY=python3

RUN="${RUN:-sword1v1}"
CKPT="${CKPT:-checkpoints}"
NMATCHES="${NMATCHES:-512}"
ROLLOUT="${ROLLOUT:-128}"

"$PY" mcbot/sim/build.py

# --warmstart gives a fighting baseline so PPO refines tactics instead of
# rediscovering "walk toward the enemy". Skip by passing --no-warmstart.
if [ "${WARMSTART:-1}" = "1" ]; then
  "$PY" -m mcbot.rl.train --warmstart 6000 --nmatches "$NMATCHES" \
        --run "$RUN" --ckpt-dir "$CKPT" --iterations "${ITERATIONS:-0}" \
        --rollout "$ROLLOUT" --eval-every "${EVAL_EVERY:-50}" --save-every "${SAVE_EVERY:-200}"
else
  "$PY" -m mcbot.rl.train --nmatches "$NMATCHES" \
        --run "$RUN" --ckpt-dir "$CKPT" --iterations "${ITERATIONS:-0}" \
        --rollout "$ROLLOUT" --eval-every "${EVAL_EVERY:-50}" --save-every "${SAVE_EVERY:-200}"
fi
