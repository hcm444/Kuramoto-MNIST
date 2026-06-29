# Research — Kuramoto vs DCGAN on MNIST

Experiments comparing **Un-0 Kuramoto** (drift loss + digit encoder) against a **class-conditional DCGAN** baseline on full MNIST (60k train).

## Documents

| Study | Epochs | Summary |
|-------|--------|---------|
| [**Fair comparison (main)**](fair_comparison.md) | 400 + 800 | Matched protocol, full write-up — **start here** |
| [Fair comparison — 400 ep run](fair_comparison_400epoch.md) | 400 | First leg: fresh train, progress every 10 |
| [Fair comparison — 800 ep resume](fair_comparison_800epoch.md) | 400→800 | Second leg: resume from `final.pt` |
| [Early 100 ep pilot](comparison_100epoch.md) | 100 | Pilot grids; DCGAN v1/v2 history |

## Key result (800 epochs, fair protocol)

| | Kuramoto (`6gb`) | DCGAN (stable + `dcgan=True`) |
|---|---|---|
| Readable 0–9 | Yes — all classes | Yes — all classes |
| Visual quality | Thick, jagged, high-contrast | Thin strokes, MNIST-like |
| Plateau | ~epoch 100–150 (loss flat) | Good by ~epoch 100 |
| **Verdict** | Learns structure; needs loss/arch work for polish | **Stronger generator** on this benchmark |

Progress grids (80 rows × 10 digits):

- Kuramoto: [`digits/kuramoto/progress_80x10.png`](../digits/kuramoto/progress_80x10.png)
- DCGAN: [`digits/dcgan/progress_80x10.png`](../digits/dcgan/progress_80x10.png)

Wait I made a typo - DCGAN is progress_80x10.png

## Reproduce

```bash
# Fresh 400-epoch fair run (wipes prior comparison artifacts)
./scripts/run_fair_comparison_400.sh

# Resume both models 400 → 800 (keeps existing progress rows)
./scripts/resume_fair_comparison.sh

# Custom target
EPOCHS=1200 ./scripts/resume_fair_comparison.sh
```

Checkpoints are gitignored; train locally or fetch from your GPU host via `cloud/fetch_results.sh`.
