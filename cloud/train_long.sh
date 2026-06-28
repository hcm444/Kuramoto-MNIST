#!/usr/bin/env bash
# Full Un-0-scale MNIST training on a cloud GPU (Vast.ai, RunPod, …).
#
# Default: 1200 epochs, MNIST-tuned loss, 10×10 progress grid at the end.
# Resume:  RESUME=checkpoints/kuramoto/final.pt ./cloud/train_long.sh
#
# Overrides:
#   EPOCHS=400 SNAPSHOTS=10 ./cloud/train_long.sh
#   BATCH_SIZE=256 ./cloud/train_long.sh   # if OOM
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/env.sh"

cd "$REPO_DIR"
# shellcheck disable=SC1091
source .venv/bin/activate

export KURAMOTO_PRESET=cloud

EPOCHS="${EPOCHS:-1200}"
SNAPSHOTS="${SNAPSHOTS:-10}"
CANDIDATES="${CANDIDATES:-32}"
CELL_SCALE="${CELL_SCALE:-4}"

ARGS=(
  --device cuda
  --epochs "$EPOCHS"
  --snapshots "$SNAPSHOTS"
  --candidates "$CANDIDATES"
  --cell-scale "$CELL_SCALE"
)
if [[ -n "${BATCH_SIZE:-}" ]]; then
  ARGS+=(--batch-size "$BATCH_SIZE")
fi
if [[ -n "${RESUME:-}" ]]; then
  ARGS+=(--resume "$RESUME")
fi

echo "==> Cloud long training (KURAMOTO_PRESET=cloud)"
echo "    Epochs: $EPOCHS  Snapshots: $SNAPSHOTS  Candidates: $CANDIDATES"
if [[ -n "${RESUME:-}" ]]; then
  echo "    Resume: $RESUME"
fi
echo ""

python make_progress_grid.py "${ARGS[@]}"

echo ""
echo "Done."
echo "  Grid:       digits/progress_10x10.png"
echo "  Checkpoint: checkpoints/kuramoto/final.pt"
echo ""
echo "Download to laptop:"
echo "  ./cloud/fetch_results.sh user@HOST -p SSH_PORT"
