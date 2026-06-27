#!/usr/bin/env bash
# Train on full MNIST (60k) and export digits/0.png … digits/9.png on CUDA.
set -euo pipefail

cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
source .venv/bin/activate

EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-512}"
CANDIDATES="${CANDIDATES:-32}"
CHECKPOINT="${CHECKPOINT:-checkpoints/kuramoto/final.pt}"

echo "==> Train ten MNIST digits (full dataset, CUDA quality preset)"
echo "    Epochs: $EPOCHS  Batch: $BATCH_SIZE  Candidates: $CANDIDATES"
echo ""

python make_digits.py \
  --device cuda \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --candidates "$CANDIDATES"

echo ""
echo "Done."
echo "  Digits:     digits/0.png … digits/9.png"
echo "  Grid:       digits/grid.png"
echo "  Checkpoint: $CHECKPOINT"
echo ""
echo "Download to your laptop:"
echo "  ./cloud/fetch_results.sh user@your-server"
