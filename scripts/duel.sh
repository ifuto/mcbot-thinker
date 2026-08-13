#!/usr/bin/env bash
# Bot-vs-bot duel. Sides are a checkpoint path or 'scripted:<style>'.
#   --a scripted:aggro --b models/sword1v1_warm.pt
#   --a models/sword1v1_best.pt --b models/sword1v1_best.pt  (mirror)
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD"
PY=".venv/bin/python"; [ -x "$PY" ] || PY=python3
"$PY" -m mcbot.duel "$@"
