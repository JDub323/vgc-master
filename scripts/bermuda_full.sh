#!/usr/bin/env bash
# The real experiment (big box): 4 generations of paths -> fit -> gate with
# the fictitious-play reservoir, then export the bundle for the pile.
#
#   bash scripts/bermuda_full.sh
#   python round_robin.py play bermuda baseline     # from any coordinator
#   python round_robin.py star                     #   checkout with a pile
#   python round_robin.py standings
#
# Tune via env: GENS GAMES ARENA_GAMES WORKERS TEAMS SCENARIOS CANDIDATES
set -euo pipefail
cd "$(dirname "$0")/.."
MAIN="${MAIN:-../vgc-bot}"
export VGC_NODE_DIR="${VGC_NODE_DIR:-$(cd "$MAIN" && pwd)/artifacts/node}"
PY="${PY:-$MAIN/.venv/bin/python}"

"$PY" bermuda/loop.py \
    --gens "${GENS:-4}" \
    --games "${GAMES:-3000}" \
    --arena-games "${ARENA_GAMES:-200}" \
    --workers "${WORKERS:-8}" \
    --teams "${TEAMS:-pool}" \
    --scenarios "${SCENARIOS:-8}" \
    --candidates "${CANDIDATES:-12}" \
    --epochs "${EPOCHS:-6}"

echo "== exporting the bundle =="
"$PY" export_agent.py bermuda --agent bermuda \
    --ckpt artifacts/checkpoints/bermuda.pt \
    --architecture "BERMUDA LSMC-CE" \
    --notes "Bermudan-exercise regression Monte Carlo, entropic CE risk"

echo "bundle in the pile — run round_robin.py play/star/all to rate it."
