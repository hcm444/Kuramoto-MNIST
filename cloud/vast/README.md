# Vast.ai deployment

Rent a GPU, paste an onstart script, and come back to `digits/progress_10x10.png` plus `checkpoints/kuramoto/final.pt`.

**Target hardware:** Ubuntu + NVIDIA **≥8 GB VRAM** (RTX 3090/4090, A5000, A10, T4 16GB).  
**Default training:** 1200 epochs, 1024 oscillators, MNIST-tuned Un-0 loss mix (~2–5 days on a 4090).

---

## 1. Pick an instance

On [vast.ai](https://vast.ai), search for:

| Filter | Value |
|--------|--------|
| GPU RAM | ≥ 8 GB |
| CUDA | 12.x |
| Image | PyTorch (Ubuntu 22.04) |
| Disk | ≥ 30 GB |

RTX **4090** or **A5000** spot instances are usually the best price/performance.

---

## 2. On-start script (easiest)

In the Vast **On-start Script** field, paste:

```bash
curl -fsSL https://raw.githubusercontent.com/hcm444/Kuramoto-MNIST/main/cloud/vast/onstart.sh | bash
```

Or upload `cloud/vast/onstart.sh` from this repo after cloning.

The script will:

1. Clone/pull this repo into `/workspace/Kuramoto-MNIST`
2. Run `cloud/bootstrap.sh` (venv + `pip install -e .`)
3. Start `cloud/train_long.sh` inside **tmux** session `train`

---

## 3. SSH in and monitor

```bash
# Vast shows: ssh -p PORT root@IP
ssh -p PORT root@IP

tmux attach -t train          # live training
tail -f /workspace/Kuramoto-MNIST/train.log
ls checkpoints/kuramoto/samples/
```

Detach tmux: `Ctrl+B`, then `D`.

---

## 4. Resume from a checkpoint

Upload your laptop checkpoint before starting, or resume on the instance:

```bash
cd /workspace/Kuramoto-MNIST
source .venv/bin/activate
RESUME=checkpoints/kuramoto/final.pt EPOCHS=1200 ./cloud/train_long.sh
```

From your laptop:

```bash
rsync -avz -e "ssh -p PORT" checkpoints/kuramoto/final.pt root@IP:/workspace/Kuramoto-MNIST/checkpoints/kuramoto/
```

---

## 5. Download results

```bash
# From your laptop (set Vast SSH port)
./cloud/fetch_results.sh root@IP /workspace/Kuramoto-MNIST
# If non-default SSH port:
rsync -avz -e "ssh -p PORT" root@IP:/workspace/Kuramoto-MNIST/checkpoints/ ./checkpoints/
rsync -avz -e "ssh -p PORT" root@IP:/workspace/Kuramoto-MNIST/digits/ ./digits/
```

---

## Overrides

```bash
EPOCHS=400 ./cloud/train_long.sh              # shorter run
BATCH_SIZE=256 ./cloud/train_long.sh          # OOM fallback
CANDIDATES=32 ./cloud/train_long.sh           # more picks per digit in grid
RESUME=checkpoints/kuramoto/final.pt ./cloud/train_long.sh
```

Quick 100-epoch grid (no 1200-epoch budget):

```bash
KURAMOTO_PRESET=cloud EPOCHS=100 ./cloud/train_progress.sh
```

---

## Cost estimate

| GPU | ~1200 epochs | ~$/hr | ~total |
|-----|--------------|-------|--------|
| RTX 4090 | 2–3 days | $0.30–0.60 | $15–45 |
| T4 16GB | 4–5 days | $0.15–0.30 | $15–35 |
| A100 | 1.5–2 days | $1.00+ | $40–80 |

Stop the instance when `train_long.sh` finishes to avoid idle charges.

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| CUDA OOM | `BATCH_SIZE=256 ./cloud/train_long.sh` |
| `un0 requires Python <3.14` | Use a PyTorch template with Python 3.11–3.13 |
| Training stopped after disconnect | `tmux attach -t train` or re-run with `RESUME=` |
| Stale snapshots in grid | `rm checkpoints/kuramoto/snapshots/epoch_*.pt` then `--skip-train` |

See also **[../UBUNTU.md](../UBUNTU.md)** for generic Ubuntu GPU setup.
