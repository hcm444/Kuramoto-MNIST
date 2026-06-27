#!/usr/bin/env bash
# Train with snapshots and build digits/progress_10x10.png on CUDA
set -euo pipefail

cd "$(dirname "$0")/.."
source .venv/bin/activate

EPOCHS="${EPOCHS:-100}"
SNAPSHOTS="${SNAPSHOTS:-10}"
BATCH_SIZE="${BATCH_SIZE:-512}"

python make_progress_grid.py \
  --epochs "$EPOCHS" \
  --snapshots "$SNAPSHOTS" \
  --batch-size "$BATCH_SIZE" \
  --device cuda \
  --candidates 8 \
  --cell-scale 4

echo ""
echo "Progress grid ready. Download to your Mac:"
echo "  ./cloud/fetch_results.sh root@YOUR_DROPLET_IP"
