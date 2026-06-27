#!/usr/bin/env bash
# Pull checkpoints and generated digits from a remote Ubuntu GPU server.
set -euo pipefail

REMOTE="${1:?usage: ./cloud/fetch_results.sh user@server-ip [remote-repo-dir]}"
REMOTE_DIR="${2:-un0-mnist-bench}"

echo "==> Fetching checkpoints/"
rsync -avz --progress "$REMOTE:$REMOTE_DIR/checkpoints/" ./checkpoints/

echo "==> Fetching digits/"
rsync -avz --progress "$REMOTE:$REMOTE_DIR/digits/" ./digits/

echo ""
echo "Done."
echo "  View grid: open digits/progress_10x10.png"
echo "  Regenerate: python make_progress_grid.py --skip-train --device cuda"
