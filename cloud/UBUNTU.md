# Ubuntu GPU training

Train on **Ubuntu 22.04+** with an **NVIDIA GPU**. This is the recommended path for a **quality 10×10 progress grid** (full 60,000-image MNIST dataset, 100 epochs).

**Requirements:** Ubuntu 22.04+, **Python 3.12 or 3.13** (not 3.14 — `un0` requires `<3.14`), NVIDIA GPU with ≥8 GB VRAM, CUDA driver, outbound HTTPS.

### Install Python 3.12 on Ubuntu

If you only have 3.14 installed:

```bash
sudo apt update
sudo apt install -y python3.12 python3.12-venv python3-pip
```

On older Ubuntu releases, use deadsnakes:

```bash
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install -y python3.12 python3.12-venv
```

Then bootstrap (auto-picks 3.13 → 3.12 → 3.11):

```bash
PYTHON=python3.12 ./cloud/bootstrap.sh
```

---

## Quick start (copy-paste)

SSH into your Ubuntu GPU machine, then:

```bash
# Option A: clone from GitHub
git clone https://github.com/hcm444/Kuramoto-MNIST.git un0-mnist-bench
cd un0-mnist-bench

# Option B: if you already uploaded the repo, just cd into it
# cd ~/un0-mnist-bench

chmod +x cloud/*.sh
./cloud/bootstrap.sh          # one-time: venv + PyTorch CUDA + pip install

tmux new -s train             # keeps training alive if SSH disconnects
./cloud/train_progress.sh     # full dataset → digits/progress_10x10.png
# Detach tmux: Ctrl-B, then D
```

**Runtime:** ~1–2 hours on a T4, ~30–60 min on RTX 4090.

**Output:**

```
digits/progress_10x10.png       # 10 rows × 10 digit columns
digits/progress_rows/           # one PNG per training snapshot
checkpoints/kuramoto/final.pt   # trained model
```

---

## Upload from your laptop instead of cloning

From your Mac or laptop:

```bash
rsync -avz --exclude .venv --exclude checkpoints --exclude data \
  ./ user@YOUR_SERVER_IP:~/un0-mnist-bench/
```

Then on the server:

```bash
ssh user@YOUR_SERVER_IP
cd ~/un0-mnist-bench
chmod +x cloud/*.sh
./cloud/bootstrap.sh
./cloud/train_progress.sh
```

---

## What the quality preset uses

| Setting | Value |
|---------|-------|
| Dataset | Full MNIST train set (60,000) |
| Epochs | 100 |
| Batch size | 512 |
| Oscillators | 1024 |
| DINO weight | 0.5 |
| Precision | bf16 |
| Grid candidates | 32 samples per digit (best kept) |

These are the `CUDA_TRAIN_KWARGS` defaults in `mnist_bench/digits.py`.

---

## Other commands

**Ten final digits only** (no progress grid):

```bash
./cloud/train_digits.sh
```

**Regenerate grid from existing checkpoints** (no retraining):

```bash
source .venv/bin/activate
python make_progress_grid.py --skip-train --device cuda --candidates 32
```

**Train longer for sharper digits:**

```bash
EPOCHS=150 CANDIDATES=32 ./cloud/train_progress.sh
```

**If you run out of GPU memory:**

```bash
BATCH_SIZE=256 ./cloud/train_progress.sh
# or
BATCH_SIZE=128 ./cloud/train_progress.sh
```

---

## Download results to your laptop

On your laptop (from the repo directory):

```bash
./cloud/fetch_results.sh user@YOUR_SERVER_IP
open digits/progress_10x10.png   # macOS
```

---

## Manual Python commands (without shell scripts)

```bash
source .venv/bin/activate

# Quality 10×10 grid
python make_progress_grid.py --device cuda --epochs 100 --candidates 32

# Ten digits only
python make_digits.py --device cuda --epochs 100 --candidates 32
```

---

## Verify GPU before training

```bash
source .venv/bin/activate
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

If `False`, you need an NVIDIA GPU instance — CPU-only Ubuntu will not work for reasonable training times.

---

## Providers

Any Ubuntu + NVIDIA GPU works: [RunPod](https://www.runpod.io/), [Vast.ai](https://vast.ai/), AWS `g4dn`, GCP T4, Lambda Cloud, etc. Pick **Ubuntu 22.04** with CUDA preinstalled when possible.

**Google Colab** (free T4): see the Colab section in this folder's scripts or run:

```python
!git clone https://github.com/hcm444/Kuramoto-MNIST.git un0-mnist-bench
%cd un0-mnist-bench
!pip install -q -e .
!./cloud/train_progress.sh
```

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `un0 requires Python <3.14` | Install 3.12: `sudo apt install python3.12 python3.12-venv`, then `PYTHON=python3.12 ./cloud/bootstrap.sh` |
| `No module named venv` | `sudo apt install python3.12-venv` |
| `cuda available: False` | Wrong instance — need NVIDIA GPU + driver |
| OOM at batch 512 | `BATCH_SIZE=256 ./cloud/train_progress.sh` |
| `python: command not found` | Run `source .venv/bin/activate` first |
| SSH disconnect killed training | Use `tmux` or `screen` |
| Digits still blurry | `EPOCHS=150 CANDIDATES=32 ./cloud/train_progress.sh` |
| Blobs / mode collapse (all columns identical) | Documented on full CUDA preset — see **[FINDINGS.md](../FINDINGS.md)**. Try retuned Kuramoto (`--epochs 400 --dino-weight 0.2 --pixel-weight 0.06 --collapse-weight 0.01`) or train DCGAN: `python train_dcgan.py --device cuda` |
| Stale `digits/0.png` … `9.png` | Re-export: `python make_digits.py --skip-train --device cuda --candidates 32` |

---

## After training

Stop or delete the cloud instance to avoid ongoing charges.
