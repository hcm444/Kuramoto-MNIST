#!/usr/bin/env bash
# Train on full MNIST (60k) and build digits/progress_10x10.png on CUDA.
#
# Quality defaults: 100 epochs, batch 512, DINO 0.5, 32 candidates per digit.
# Override: EPOCHS=150 CANDIDATES=32 BATCH_SIZE=256 ./cloud/train_progress.sh
set -euo pipefail

cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
source .venv/bin/activate

EPOCHS="${EPOCHS:-100}"
SNAPSHOTS="${SNAPSHOTS:-10}"
BATCH_SIZE="${BATCH_SIZE:-512}"
CANDIDATES="${CANDIDATES:-32}"
CELL_SCALE="${CELL_SCALE:-4}"

echo "==> Quality 10×10 progress grid"
echo "    Full MNIST train set (60,000 images)"
echo "    Epochs: $EPOCHS  Snapshots: $SNAPSHOTS  Batch: $BATCH_SIZE"
echo "    Candidates per digit: $CANDIDATES"
echo ""

python make_progress_grid.py \
  --epochs "$EPOCHS" \
  --snapshots "$SNAPSHOTS" \
  --batch-size "$BATCH_SIZE" \
  --device cuda \
  --candidates "$CANDIDATES" \
  --cell-scale "$CELL_SCALE"

echo ""
echo "Done."
echo "  Grid:       digits/progress_10x10.png"
echo "  Row images: digits/progress_rows/epoch_XXXX.png"
echo "  Checkpoint: checkpoints/kuramoto/final.pt"
echo ""
echo "Download to your laptop:"
echo "  ./cloud/fetch_results.sh user@your-server"
