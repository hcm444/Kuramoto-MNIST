# Shared cloud training defaults. Source from cloud/*.sh scripts.
#
#   source "$(dirname "$0")/env.sh"
#
# Presets (set KURAMOTO_PRESET before sourcing to override):
#   cloud — 1200 epochs, 1024 oscillators, batch 512 (Vast / ≥8 GB)
#   6gb   — 60 epochs, 512 oscillators, batch 128 (laptop GPUs)
#   (unset) — auto-detect from GPU VRAM

export KURAMOTO_PRESET="${KURAMOTO_PRESET:-cloud}"
export REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export REPO_URL="${REPO_URL:-https://github.com/hcm444/Kuramoto-MNIST.git}"

export EPOCHS="${EPOCHS:-}"
export SNAPSHOTS="${SNAPSHOTS:-10}"
export CANDIDATES="${CANDIDATES:-32}"
export CELL_SCALE="${CELL_SCALE:-4}"
export BATCH_SIZE="${BATCH_SIZE:-}"
export RESUME="${RESUME:-}"
