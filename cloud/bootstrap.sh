#!/usr/bin/env bash
# One-time setup on Ubuntu with an NVIDIA GPU (RunPod, Vast, AWS, bare metal, …).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"

# un0 requires Python >=3.11,<3.14 (3.14 is not supported yet).
pick_python() {
  for candidate in python3.13 python3.12 python3.11; do
    if command -v "$candidate" >/dev/null 2>&1; then
      echo "$candidate"
      return 0
    fi
  done
  echo "Need Python 3.11, 3.12, or 3.13 (un0 does not support 3.14)." >&2
  echo "Install on Ubuntu: sudo apt install python3.12 python3.12-venv" >&2
  exit 1
}

check_python_version() {
  local py="$1"
  "$py" - <<'PY'
import sys
major, minor = sys.version_info[:2]
if not (major == 3 and 11 <= minor <= 13):
    raise SystemExit(
        f"Python {major}.{minor} is not supported. "
        "un0 requires >=3.11,<3.14 — use 3.12 or 3.13."
    )
PY
}

echo "==> System packages"
if command -v apt-get >/dev/null 2>&1; then
  sudo apt-get update
  sudo apt-get install -y git curl
  if ! pick_python >/dev/null 2>&1; then
    echo "Installing python3.12 …"
    sudo apt-get install -y python3.12 python3.12-venv python3-pip || {
      echo "Could not install python3.12 via apt." >&2
      echo "Try: sudo add-apt-repository ppa:deadsnakes/ppa && sudo apt update" >&2
      echo "     sudo apt install python3.12 python3.12-venv" >&2
      exit 1
    }
  fi
fi

PYTHON="${PYTHON:-$(pick_python)}"
check_python_version "$PYTHON"
echo "Using $PYTHON ($("$PYTHON" --version))"

if [[ ! -f "$REPO_DIR/pyproject.toml" ]]; then
  echo "==> Clone repo (set REPO_URL if you use a fork)"
  REPO_URL="${REPO_URL:-https://github.com/hcm444/Kuramoto-MNIST.git}"
  mkdir -p "$(dirname "$REPO_DIR")"
  git clone "$REPO_URL" "$REPO_DIR"
fi

cd "$REPO_DIR"

echo "==> Python venv"
rm -rf .venv
"$PYTHON" -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate

echo "==> PyTorch (CUDA)"
pip install --upgrade pip
if python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
  echo "PyTorch with CUDA already available — skipping wheel install"
else
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
fi
pip install -e .

echo "==> Verify GPU"
python - <<'PY'
import sys
import torch
from un0.common import resolve_device

d = resolve_device("cuda")
print("device:", d)
print("torch:", torch.__version__)
print("python:", sys.version.split()[0])
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
    print("vram_gb:", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1))
else:
    raise SystemExit(
        "CUDA not available. Use an Ubuntu machine with an NVIDIA GPU and driver installed."
    )
PY

echo ""
echo "Setup complete."
echo ""
echo "Train a quality 10×10 progress grid (full 60k MNIST, ~1–2 hr on T4):"
echo "  cd $REPO_DIR"
echo "  tmux new -s train"
echo "  ./cloud/train_progress.sh"
echo ""
echo "Or ten final digits only:"
echo "  ./cloud/train_digits.sh"
