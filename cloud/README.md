# Cloud GPU training

**→ [Vast.ai deploy guide](vast/README.md)** (recommended)  
**→ [Generic Ubuntu GPU](UBUNTU.md)**

## Scripts

| Script | Purpose |
|--------|---------|
| `bootstrap.sh` | One-time venv + PyTorch CUDA + `pip install -e .` |
| `train_long.sh` | **1200 epochs** + 10×10 progress grid (cloud preset) |
| `train_progress.sh` | Shorter grid run (default 100 epochs) |
| `train_digits.sh` | Ten final digits only |
| `fetch_results.sh` | Download `checkpoints/` and `digits/` to laptop |
| `vast/onstart.sh` | Paste into Vast.ai on-start field |

## Vast.ai quick start

1. Rent **≥8 GB** GPU, Ubuntu + CUDA 12 PyTorch template  
2. On-start script:

```bash
curl -fsSL https://raw.githubusercontent.com/hcm444/Kuramoto-MNIST/main/cloud/vast/onstart.sh | bash
```

3. SSH → `tmux attach -t train`  
4. Laptop → `./cloud/fetch_results.sh root@IP /workspace/Kuramoto-MNIST`

## Resume

```bash
RESUME=checkpoints/kuramoto/final.pt EPOCHS=1200 ./cloud/train_long.sh
```

## Presets (`KURAMOTO_PRESET`)

| Preset | Epochs | Oscillators | Batch | Use case |
|--------|--------|-------------|-------|----------|
| `cloud` | 1200 | 1024 | 512 | Vast / ≥8 GB |
| `6gb` | 60 | 512 | 128 | Laptop 6 GB |
| (auto) | 100/60 | by VRAM | by VRAM | Local CUDA |

## Overrides

```bash
EPOCHS=400 ./cloud/train_long.sh
BATCH_SIZE=256 ./cloud/train_long.sh   # OOM
CANDIDATES=32 ./cloud/train_progress.sh
```

## Other providers

RunPod, Lambda, AWS `g4dn`, GCP T4 — same scripts; use `cloud/UBUNTU.md` for SSH setup.
