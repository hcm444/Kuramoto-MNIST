#!/usr/bin/env bash
# Pull checkpoints and generated digits from a remote GPU server (Vast, RunPod, …).
set -euo pipefail

REMOTE="${1:?usage: ./cloud/fetch_results.sh user@host [remote-repo-dir]}"
REMOTE_DIR="${2:-/workspace/Kuramoto-MNIST}"
SSH_PORT="${SSH_PORT:-22}"
RSYNC_SSH="ssh -p $SSH_PORT"

echo "==> Fetching checkpoints/"
rsync -avz -e "$RSYNC_SSH" --progress "$REMOTE:$REMOTE_DIR/checkpoints/" ./checkpoints/

echo "==> Fetching digits/"
rsync -avz -e "$RSYNC_SSH" --progress "$REMOTE:$REMOTE_DIR/digits/" ./digits/

echo ""
echo "Done."
echo "  View grid: open digits/progress_10x10.png"
echo "  Regenerate: python make_progress_grid.py --skip-train --device cuda"
