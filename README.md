# Kuramoto-MNIST

Benchmark **Kuramoto dynamics** (via [Un-0](https://github.com/hcm444/Un-0)) against a **DCGAN baseline** on MNIST at 32×32 RGB (padded from 28×28 grayscale).

## Inspiration

This project builds on [Un-0](https://github.com/unconv-ai/Un-0) by [Unconventional AI](https://unconv.ai) — a Kuramoto-based image generator with no diffusion schedule and no adversarial training. The MNIST benchmark uses the [hcm444/Un-0](https://github.com/hcm444/Un-0) fork for training utilities, the drift-loss recipe, and Apple Silicon inference helpers.

**References:**

- [unconv-ai/Un-0](https://github.com/unconv-ai/Un-0) — original research implementation
- [hcm444/Un-0](https://github.com/hcm444/Un-0) — MPS inference optimizations and photomosaic tooling used here
- [Introducing Un-0 (blog)](https://unconv.ai/blog/introducing-un-0-generating-images-with-coupled-oscillators/)

## What it compares

| Axis | Kuramoto (Un-0) | DCGAN |
|------|-----------------|-------|
| Training | Drift loss + DINO features | Adversarial (BCE) |
| Generation | ODE integration of coupled oscillators | Single forward pass |
| Classes | 10 digits | 10 digits |
| Resolution | 32×32 RGB | 32×32 RGB |

## Setup

Requires Python ≥ 3.11. Install [Un-0](https://github.com/hcm444/Un-0) as a dependency (editable path or from GitHub).

```bash
git clone https://github.com/hcm444/Kuramoto-MNIST.git
cd Kuramoto-MNIST
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[eval]"
pip install -e "git+https://github.com/hcm444/Un-0.git#egg=un0"
```

For local development with Un-0 checked out alongside this repo, edit `pyproject.toml` or run `pip install -e ../Un-0`.

## Quick smoke test (~2 min on MPS)

```bash
python scripts/dump_mnist_reals.py --max-images 500
python train_kuramoto.py --max-steps 10 --batch-size 64 --save-every 1 --sample-every 1
python train_dcgan.py --max-steps 20 --batch-size 64 --save-every 1 --sample-every 1
```

## Full workflow

### 1. Dump real MNIST images (FID reference)

```bash
python scripts/dump_mnist_reals.py
```

Writes 60k PNGs to `data/mnist_reals/`.

### 2. Train models

```bash
# Kuramoto (default n=1024 oscillators, 400 epochs)
python train_kuramoto.py --checkpoint-dir checkpoints/kuramoto

# DCGAN baseline (200 epochs)
python train_dcgan.py --checkpoint-dir checkpoints/dcgan
```

Training is fastest on **CUDA**. MPS works for inference benchmarking after checkpoints exist.

### 3. Evaluate

```bash
python eval/fid.py --model kuramoto --checkpoint checkpoints/kuramoto/final.pt
python eval/fid.py --model dcgan --checkpoint checkpoints/dcgan/final.pt
python eval/benchmark.py --model kuramoto --checkpoint checkpoints/kuramoto/final.pt --device mps
python eval/benchmark.py --model dcgan --checkpoint checkpoints/dcgan/final.pt --device mps
```

### 4. Side-by-side comparison

```bash
python eval/compare.py \
  --kuramoto checkpoints/kuramoto/final.pt \
  --dcgan checkpoints/dcgan/final.pt \
  --device mps
```

Writes `results/comparison.json` and prints a summary table.

## Project layout

```
mnist_bench/          # data loader, DCGAN, FID helpers
train_kuramoto.py     # Un-0 drift-loss training on MNIST
train_dcgan.py        # GAN baseline
scripts/              # dump real images for FID
eval/                 # fid, benchmark, compare
checkpoints/          # trained weights (gitignored)
results/              # metrics JSON (gitignored)
```

## Notes

- MNIST is padded 28→32 and repeated to 3 channels so the CIFAR-10 Un-0 architecture applies without modification.
- FID uses [clean-FID](https://github.com/GaParmar/clean-fid) with custom statistics built from `data/mnist_reals/`.
- For fair comparison, evaluate both models at the same `--num-samples` and resolution.

## License

MIT
