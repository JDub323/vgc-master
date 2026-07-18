#!/usr/bin/env bash
# One-time worktree setup: share the main checkout's behavior assets and
# Node install. Usage: bash scripts/bermuda_setup.sh [path-to-main-checkout]
set -euo pipefail
cd "$(dirname "$0")/.."
MAIN="${1:-../vgc-bot}"

test -d "$MAIN/artifacts" || {
    echo "main checkout not found at '$MAIN' (pass its path)"; exit 1; }

mkdir -p artifacts/checkpoints artifacts/bermuda
for f in dex.json usage_stats.json spreads.json vocab.json \
         selfplay_teams.json; do
    if [ -f "$MAIN/artifacts/$f" ]; then
        cp "$MAIN/artifacts/$f" artifacts/
        echo "copied artifacts/$f"
    fi
done

echo
echo "Node install is shared via \$VGC_NODE_DIR — the run scripts set it to:"
echo "  $MAIN/artifacts/node"
echo "setup done."
