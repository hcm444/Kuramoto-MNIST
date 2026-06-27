# Cloud GPU training

**→ For Ubuntu step-by-step instructions, see [UBUNTU.md](UBUNTU.md).**

DigitalOcean GPU droplets are often **sold out**. Use any machine with **Ubuntu + NVIDIA GPU**.

## Scripts

| Script | Purpose |
|--------|---------|
| `bootstrap.sh` | One-time venv + PyTorch CUDA + `pip install -e .` |
| `train_progress.sh` | **Quality 10×10 grid** (full 60k MNIST, 100 epochs) |
| `train_digits.sh` | Ten final digits only |
| `fetch_results.sh` | Download `checkpoints/` and `digits/` to your laptop |

## Google Colab (free T4)

```python
!git clone https://github.com/hcm444/Kuramoto-MNIST.git un0-mnist-bench
%cd un0-mnist-bench
!pip install -q -e .
!./cloud/train_progress.sh
```

Download `digits/progress_10x10.png` from the Files panel.

## Providers

RunPod, Vast.ai, Lambda Cloud, AWS g4dn, GCP T4 — pick **Ubuntu 22.04**, CUDA 12.x, ≥8 GB VRAM.

## Overrides

```bash
EPOCHS=150 CANDIDATES=32 ./cloud/train_progress.sh
BATCH_SIZE=256 ./cloud/train_progress.sh   # if OOM
```

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `cuda available: False` | Need NVIDIA GPU instance |
| OOM at batch 512 | `BATCH_SIZE=256 ./cloud/train_progress.sh` |
| Colab disconnects | Use `EPOCHS=50` or Colab Pro |

See **[UBUNTU.md](UBUNTU.md)** for the full guide.
