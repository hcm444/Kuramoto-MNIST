#!/usr/bin/env bash
# Train on full MNIST (60k) and build digits/progress_10x10.png on CUDA.
#
# MNIST-tuned losses; auto-selects 6 GB vs cloud preset from GPU VRAM.
# Override: EPOCHS=150 CANDIDATES=32 BATCH_SIZE=256 ./cloud/train_progress.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/env.sh"

cd "$REPO_DIR"
# shellcheck disable=SC1091
source .venv/bin/activate

export KURAMOTO_PRESET="${KURAMOTO_PRESET:-}"

EPOCHS="${EPOCHS:-}"
SNAPSHOTS="${SNAPSHOTS:-10}"
CANDIDATES="${CANDIDATES:-32}"
CELL_SCALE="${CELL_SCALE:-4}"

ARGS=(--device cuda --snapshots "$SNAPSHOTS" --candidates "$CANDIDATES" --cell-scale "$CELL_SCALE")
if [[ -n "$EPOCHS" ]]; then
  ARGS+=(--epochs "$EPOCHS")
fi
if [[ -n "${BATCH_SIZE:-}" ]]; then
  ARGS+=(--batch-size "$BATCH_SIZE")
fi
if [[ -n "${RESUME:-}" ]]; then
  ARGS+=(--resume "$RESUME")
fi

echo "==> Progress grid training"
echo "    Preset: ${KURAMOTO_PRESET:-auto}  Snapshots: $SNAPSHOTS  Candidates: $CANDIDATES"
echo ""

python make_progress_grid.py "${ARGS[@]}"

echo ""
echo "Done."
echo "  Grid:       digits/progress_10x10.png"
echo "  Row images: digits/progress_rows/epoch_XXXX.png"
echo "  Checkpoint: checkpoints/kuramoto/final.pt"
echo ""
echo "Download to your laptop:"
echo "  ./cloud/fetch_results.sh user@your-server"
