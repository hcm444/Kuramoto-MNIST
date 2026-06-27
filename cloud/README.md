# Cloud GPU training

DigitalOcean GPU droplets are often **sold out**. Use any machine with an **NVIDIA GPU + Ubuntu** — the same scripts work everywhere.

## Fastest option: Google Colab (free GPU)

1. Open [Google Colab](https://colab.research.google.com/)
2. **Runtime → Change runtime type → T4 GPU**
3. Paste and run:

```python
!git clone https://github.com/hcm444/Kuramoto-MNIST.git un0-mnist-bench
# Or upload your local repo: Files → Upload, then skip clone
%cd un0-mnist-bench

!pip install -q -e .

# Ten digits (0–9) — ~30–60 min on T4
!./cloud/train_digits.sh

# Or 10×10 progress grid
# !./cloud/train_progress.sh
```

4. Download results: **Files** panel → `digits/` and `checkpoints/kuramoto/` → right-click Download

Colab free tier may disconnect — use **100 epochs** max or Colab Pro for longer runs.

---

## Other GPU providers (usually in stock)

| Provider | Notes | Typical cost |
|----------|-------|--------------|
| [RunPod](https://www.runpod.io/) | On-demand RTX 4090 / A5000, hourly | ~$0.30–0.70/hr |
| [Vast.ai](https://vast.ai/) | Cheapest spot market | ~$0.10–0.40/hr |
| [Lambda Cloud](https://lambdalabs.com/service/gpu-cloud) | Simple UI, A10/A100 | ~$0.60–1.10/hr |
| [Paperspace Gradient](https://www.paperspace.com/gpu-cloud) | Notebooks + SSH | ~$0.50/hr |
| [AWS EC2](https://aws.amazon.com/ec2/instance-types/g4/) | `g4dn.xlarge` (T4) | ~$0.50/hr |
| [Google Cloud](https://cloud.google.com/compute/docs/gpus) | `n1-standard-4` + T4 | varies |

Pick **Ubuntu 22.04** with **CUDA 12.x** and at least **16 GB GPU RAM**.

---

## Generic setup (RunPod, Vast, Lambda, AWS, …)

### 1. Create a GPU instance and SSH in

```bash
ssh root@YOUR_GPU_HOST
```

### 2. Upload the repo from your Mac

```bash
rsync -avz --exclude .venv --exclude checkpoints --exclude data \
  /Users/henrymeier/Documents/un0-mnist-bench/ \
  root@YOUR_GPU_HOST:~/un0-mnist-bench/
```

### 3. Bootstrap and train

```bash
ssh root@YOUR_GPU_HOST
cd ~/un0-mnist-bench
chmod +x cloud/*.sh
./cloud/bootstrap.sh

# Use tmux so disconnect doesn't kill training
tmux new -s train
./cloud/train_digits.sh        # ten final digits
# or
./cloud/train_progress.sh      # 10×10 progress grid
# Detach: Ctrl-B, then D
```

### 4. Download to your Mac

```bash
cd /Users/henrymeier/Documents/un0-mnist-bench
./cloud/fetch_results.sh root@YOUR_GPU_HOST
```

---

## What runs on the GPU

Same as before, but not tied to DigitalOcean:

| Setting | Mac (MPS) | Cloud (CUDA) |
|---------|-----------|--------------|
| Batch size | 64 | 512 |
| Precision | fp32 | bf16 |
| Speed | hours | ~10–30× faster |

Scripts: `cloud/bootstrap.sh`, `cloud/train_digits.sh`, `cloud/train_progress.sh`, `cloud/fetch_results.sh`

Override training length:

```bash
EPOCHS=150 ./cloud/train_digits.sh
EPOCHS=100 SNAPSHOTS=10 ./cloud/train_progress.sh
```

---

## RunPod quick start (example)

1. [runpod.io](https://www.runpod.io/) → **Deploy** → **GPU Pod**
2. Template: **RunPod PyTorch 2.x** (CUDA preinstalled)
3. GPU: RTX 4090 or A5000
4. Open **Web Terminal** or SSH
5. Run the **Generic setup** steps above

---

## After training

- Pull `checkpoints/` and `digits/` to your Mac with `fetch_results.sh`
- **Stop / delete** the cloud instance to stop billing
- Regenerate locally anytime: `python make_digits.py --skip-train --device mps`

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `cuda available: False` | Wrong instance type — need NVIDIA GPU |
| OOM at batch 512 | `BATCH_SIZE=256 ./cloud/train_digits.sh` |
| Colab disconnects | Shorter run: `EPOCHS=50 ./cloud/train_digits.sh` |
| `un0` install fails | Instance needs outbound HTTPS |
