#!/usr/bin/env bash
# Vast.ai onstart script — paste into the instance "On-start Script" field,
# or pass as:  --onstart-cmd "bash -lc 'curl -fsSL .../onstart.sh | bash'"
#
# Trains Kuramoto-MNIST to 1200 epochs and writes digits/progress_10x10.png.
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive
export KURAMOTO_PRESET=cloud
export REPO_DIR="${REPO_DIR:-/workspace/Kuramoto-MNIST}"
export REPO_URL="${REPO_URL:-https://github.com/hcm444/Kuramoto-MNIST.git}"
export RECREATE_VENV="${RECREATE_VENV:-0}"

LOG="${REPO_DIR}/train.log"
mkdir -p "$(dirname "$REPO_DIR")"

exec > >(tee -a "$LOG") 2>&1
echo "==> Kuramoto-MNIST vast onstart $(date -Is)"

if [[ ! -d "$REPO_DIR/.git" ]]; then
  echo "==> Clone $REPO_URL"
  git clone "$REPO_URL" "$REPO_DIR"
fi

cd "$REPO_DIR"
git pull --ff-only || true
chmod +x cloud/*.sh cloud/vast/*.sh 2>/dev/null || chmod +x cloud/*.sh

echo "==> Bootstrap"
./cloud/bootstrap.sh

echo "==> Train (tmux session: train)"
if command -v tmux >/dev/null 2>&1; then
  tmux kill-session -t train 2>/dev/null || true
  tmux new-session -d -s train "./cloud/train_long.sh"
  echo "Attached log: tail -f $LOG"
  echo "Tmux: tmux attach -t train"
else
  nohup ./cloud/train_long.sh >> "$LOG" 2>&1 &
  echo "PID: $!  log: tail -f $LOG"
fi

echo "==> Onstart finished $(date -Is)"
