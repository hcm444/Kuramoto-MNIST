#!/usr/bin/env bash
# One-time setup on a fresh Ubuntu GPU droplet (DigitalOcean, etc.).
set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/un0-mnist-bench}"

echo "==> System packages"
sudo apt-get update
sudo apt-get install -y git python3.12 python3.12-venv python3-pip

if [[ ! -d "$REPO_DIR/.git" ]]; then
  echo "==> Clone repo (set REPO_URL if you use a fork)"
  REPO_URL="${REPO_URL:-https://github.com/hcm444/Kuramoto-MNIST.git}"
  git clone "$REPO_URL" "$REPO_DIR"
fi

cd "$REPO_DIR"

echo "==> Python venv"
python3.12 -m venv .venv
source .venv/bin/activate

echo "==> PyTorch (CUDA)"
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -e .

echo "==> Verify GPU"
python - <<'PY'
import torch
from un0.common import resolve_device
d = resolve_device("cuda")
print("device:", d)
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY

echo ""
echo "Done. Next:"
echo "  cd $REPO_DIR && source .venv/bin/activate"
echo "  ./cloud/train_digits.sh"
