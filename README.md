# Kuramoto-MNIST

Train [Un-0](https://github.com/unconv-ai/Un-0) Kuramoto dynamics on MNIST and generate **ten grayscale digits** (0–9) plus a **10×10 training progress grid**.

## Vast.ai (recommended for 1200 epochs)

Rent a GPU, paste one on-start script, download results when done.

**Guide:** **[cloud/vast/README.md](cloud/vast/README.md)**

**On-start script** (Vast instance field):

```bash
curl -fsSL https://raw.githubusercontent.com/hcm444/Kuramoto-MNIST/main/cloud/vast/onstart.sh | bash
```

**Manual SSH workflow:**

```bash
git clone https://github.com/hcm444/Kuramoto-MNIST.git
cd Kuramoto-MNIST
chmod +x cloud/*.sh cloud/vast/*.sh
./cloud/bootstrap.sh
tmux new -s train
./cloud/train_long.sh    # 1200 epochs → digits/progress_10x10.png
```

**Resume** from a laptop checkpoint on the instance:

```bash
RESUME=checkpoints/kuramoto/final.pt EPOCHS=1200 ./cloud/train_long.sh
```

**Download** (set `SSH_PORT` for Vast):

```bash
SSH_PORT=12345 ./cloud/fetch_results.sh root@IP /workspace/Kuramoto-MNIST
```

| Preset | GPU | Epochs | ~time |
|--------|-----|--------|-------|
| `cloud` | ≥8 GB (4090, A5000, T4) | 1200 | 2–5 days |
| `6gb` (auto) | 6 GB laptop | 60 | ~9 hr |
| `train_progress.sh` | ≥8 GB | 100 | ~1–2 hr |

---

## Mac (Apple Silicon)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
python make_progress_grid.py --device mps --epochs 40 --candidates 16
```

Faster subset: `python make_progress_grid.py --fast --device mps`

---

## Outputs

```
digits/
  progress_10x10.png     # 10 training snapshots × 10 digits
  progress_rows/         # one row per snapshot
  0.png … 9.png          # final digits (make_digits.py)
checkpoints/kuramoto/
  final.pt               # trained model
  snapshots/             # epoch checkpoints
  samples/               # live sample grids during training
```

---

## How it works

1. **Train** — Un-0 drift loss + DINO on padded 32×32 RGB MNIST  
2. **Generate** — `model.sample(class_id)` per digit; progress grid keeps best candidate per class  

**MNIST-tuned loss** (all CUDA presets): `dino=0.2`, `pixel=0.06`, `collapse=0.01` — higher pixel weight than Un-0 CIFAR defaults.

Training details and DCGAN comparison: **[FINDINGS.md](FINDINGS.md)**

---

## Local options

```bash
python make_progress_grid.py --skip-train --device cuda --candidates 16
python make_digits.py --skip-train --device cuda
python train_dcgan.py --device cuda --epochs 200   # GAN baseline
```

---

## References

- [unconv-ai/Un-0](https://github.com/unconv-ai/Un-0) — oscillator image generator  
- [Un-0 blog](https://unconv.ai/blog/introducing-un-0-generating-images-with-coupled-oscillators/)

MIT
