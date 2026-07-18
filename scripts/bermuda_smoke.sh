#!/usr/bin/env bash
# Tiny end-to-end proof on a laptop CPU: a handful of gen-0 paths, a 2-epoch
# LSMC fit, and two arena games with a minimal scenario budget. Minutes, not
# hours — budget knobs only, no code differences from the full run.
set -euo pipefail
cd "$(dirname "$0")/.."
MAIN="${MAIN:-../vgc-bot}"
export VGC_NODE_DIR="${VGC_NODE_DIR:-$(cd "$MAIN" && pwd)/artifacts/node}"
PY="${PY:-$MAIN/.venv/bin/python}"

echo "== 1/3 gen-0 paths (8 games, heuristic measure) =="
"$PY" bermuda/paths.py --games 8 --out artifacts/bermuda/paths/smoke \
    --behavior heuristic --teams replicas --workers 2 --seed 0 \
    --label smoke

echo "== 2/3 LSMC fit (2 epochs, tiny) =="
"$PY" bermuda/train.py --shards artifacts/bermuda/paths/smoke \
    --out artifacts/checkpoints/bermuda_smoke.pt --epochs 2 --batch 256

echo "== 3/3 exercise-policy arena (2 games, K=2 A=5) =="
"$PY" bermuda/eval.py arena --ckpt artifacts/checkpoints/bermuda_smoke.pt \
    --vs heuristic --games 2 --workers 1 --scenarios 2 --candidates 5

echo
echo "smoke OK. Full run: bash scripts/bermuda_full.sh"
