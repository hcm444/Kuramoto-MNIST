"""Train Un-0 Kuramoto generator on MNIST (32x32 RGB, 10 classes)."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from tqdm.auto import tqdm

from torch.nn import functional as F

from mnist_bench.constants import IMAGE_SIZE, NUM_CLASSES
from mnist_bench.data import build_mnist_dataloader
from mnist_bench.digits import anti_collapse_loss, grayscale_consistency_loss
from un0.common import (
    autocast_context,
    disable_torchscript_gpu_fuser_on_blackwell,
    make_scheduler,
    resolve_device,
    save_sample_grid,
    seed_everything,
)
from un0.decoupled_adamw import DecoupledAdamW
from un0.losses import DINOFeatureExtractor, PerClassQueue, conditional_drift_loss
from un0.model import build_cifar10_model

WEIGHT_DECAY = 1e-3
BETA1 = 0.9
BETA2 = 0.95
WARMUP_FRACTION = 0.1
GRAD_CLIP_NORM = 1.0
GAMMA = 0.2
LOG_EVERY = 10


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--precision", choices=("fp32", "tf32", "bf16", "fp16"), default="bf16")
    parser.add_argument("--dino-weight", type=float, default=1.0)
    parser.add_argument("--pixel-weight", type=float, default=0.004)
    parser.add_argument(
        "--collapse-weight",
        type=float,
        default=0.0,
        help="Penalize flat black/white generator outputs.",
    )
    parser.add_argument(
        "--channel-weight",
        type=float,
        default=0.0,
        help="Penalize RGB mismatch so generated digits stay grayscale.",
    )
    parser.add_argument("--n-oscillators", type=int, default=1024)
    parser.add_argument("--n-conditional-oscillators", type=int, default=8)
    parser.add_argument("--class-dropout-prob", type=float, default=0.1)
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--solver", choices=("euler", "rk4"), default="euler")
    parser.add_argument("--queue-size", type=int, default=1024)
    parser.add_argument("--num-pos", type=int, default=64)
    parser.add_argument("--feature-batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints/kuramoto"))
    parser.add_argument("--sample-every", type=int, default=50)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument(
        "--snapshot-every",
        type=int,
        default=0,
        help="Save checkpoints/kuramoto/snapshots/epoch_XXXX.pt every N epochs (0=off).",
    )
    parser.add_argument("--max-steps", type=int, default=0, help="Stop after N optimizer steps (smoke test).")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Train on first N MNIST images only (0 = full 60k). Speeds up Mac runs.",
    )
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--device", default="auto", help="Device: auto, mps, cuda, or cpu.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    disable_torchscript_gpu_fuser_on_blackwell()
    seed_everything(int(args.seed))
    device = resolve_device(str(args.device))
    if args.precision == "tf32":
        torch.set_float32_matmul_precision("high")

    loader = build_mnist_dataloader(
        root=str(args.data_root),
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        max_samples=int(args.max_samples),
        pin_memory=device.type == "cuda",
    )

    if int(args.max_samples) > 0:
        print(f"Using {len(loader.dataset)} MNIST samples ({len(loader)} steps/epoch)")

    model = build_cifar10_model(
        n_oscillators=int(args.n_oscillators),
        n_conditional_oscillators=int(args.n_conditional_oscillators),
        class_dropout_prob=float(args.class_dropout_prob),
        num_steps=int(args.num_steps),
        parameterization="standard",
        relativization="mean_relative",
        encoding="sin_cos",
        solver=str(args.solver),
    ).to(device)

    use_dino = float(args.dino_weight) > 0.0
    dino = None
    if use_dino:
        dino = DINOFeatureExtractor().to(device)
        if device.type != "mps":
            dino = torch.compile(dino, dynamic=False)

    optimizer = DecoupledAdamW(
        model.parameters(),
        lr=float(args.lr),
        betas=(BETA1, BETA2),
        weight_decay=WEIGHT_DECAY,
    )
    steps_per_epoch = len(loader)
    total_steps = int(args.epochs) * steps_per_epoch
    if args.max_steps > 0:
        total_steps = min(total_steps, int(args.max_steps))
    scheduler = make_scheduler(optimizer, total_steps=total_steps, warmup_fraction=WARMUP_FRACTION)
    scaler = torch.amp.GradScaler("cuda") if args.precision == "fp16" else None

    queue = PerClassQueue(
        num_classes=NUM_CLASSES,
        queue_size=int(args.queue_size),
        data_dim=3 * IMAGE_SIZE * IMAGE_SIZE,
        device=device,
        dtype=torch.float32,
        track_sample_ids=False,
    )

    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    snapshot_dir = checkpoint_dir / "snapshots"
    eval_class_ids = torch.arange(NUM_CLASSES, device=device).repeat_interleave(10)

    global_step = 0
    stop = False
    for epoch in range(int(args.epochs)):
        model.train()
        if dino is not None:
            dino.train()
        progress = tqdm(enumerate(loader), total=steps_per_epoch, desc=f"epoch {epoch + 1}")
        for _step, batch in progress:
            x_real = batch["data"].to(device)
            class_id = batch["class_id"].to(device)
            class_id_gen = class_id

            queue.push(x_real.detach(), class_id, None)
            gen_classes = torch.unique(class_id_gen)
            queue_ready = bool(queue.ready_mask(int(args.num_pos))[gen_classes].all())
            x_real_pos = class_id_pos = None
            if queue_ready:
                x_real_pos, class_id_pos, _ = queue.draw(gen_classes, num_pos=int(args.num_pos))

            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, args.precision):
                x_gen = model(class_id_gen)
                if queue_ready:
                    loss, metrics = conditional_drift_loss(
                        x_real,
                        x_gen,
                        class_id,
                        class_id_gen,
                        dino=dino,
                        dino_weight=float(args.dino_weight),
                        pixel_weight=float(args.pixel_weight),
                        gamma=GAMMA,
                        feature_batch_size=int(args.feature_batch_size),
                        image_size=IMAGE_SIZE,
                        x_real_pos=x_real_pos,
                        class_id_pos=class_id_pos,
                    )
                    channel_w = float(args.channel_weight)
                    if channel_w > 0.0:
                        channel_loss = grayscale_consistency_loss(x_gen)
                        loss = loss + channel_w * channel_loss
                        metrics["loss/channel"] = channel_loss.detach()
                    collapse_w = float(args.collapse_weight)
                    if collapse_w > 0.0:
                        collapse_loss = anti_collapse_loss(x_gen)
                        loss = loss + collapse_w * collapse_loss
                        metrics["loss/collapse"] = collapse_loss.detach()
                else:
                    loss = F.mse_loss(x_gen, x_real)
                    metrics = {"loss/warmup": loss.detach(), "loss/total": loss.detach()}
                    channel_w = float(args.channel_weight)
                    if channel_w > 0.0:
                        channel_loss = grayscale_consistency_loss(x_gen)
                        loss = loss + channel_w * channel_loss
                        metrics["loss/channel"] = channel_loss.detach()
                    collapse_w = float(args.collapse_weight)
                    if collapse_w > 0.0:
                        collapse_loss = anti_collapse_loss(x_gen)
                        loss = loss + collapse_w * collapse_loss
                        metrics["loss/collapse"] = collapse_loss.detach()

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                grad_norm = nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
                optimizer.step()
            scheduler.step()
            global_step += 1

            if global_step % LOG_EVERY == 0:
                postfix = {k: float(v.detach().cpu()) for k, v in metrics.items()}
                postfix["grad_norm"] = float(grad_norm.detach().cpu())
                progress.set_postfix(postfix)

            if args.max_steps > 0 and global_step >= int(args.max_steps):
                stop = True
                break

        epoch_num = epoch + 1
        if epoch_num % int(args.sample_every) == 0 or stop:
            sample_path = checkpoint_dir / "samples" / f"epoch_{epoch_num:04d}.png"
            samples = model.sample(eval_class_ids)
            save_sample_grid(samples, sample_path, image_size=IMAGE_SIZE)

        if epoch_num % int(args.save_every) == 0 or stop:
            state = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "global_step": global_step,
                "config": config,
            }
            torch.save(state, checkpoint_dir / "latest.pt")
            if int(args.snapshot_every) > 0 and epoch_num % int(args.snapshot_every) == 0:
                snapshot_dir.mkdir(parents=True, exist_ok=True)
                snapshot_path = snapshot_dir / f"epoch_{epoch_num:04d}.pt"
                torch.save(state, snapshot_path)
            if stop:
                torch.save(state, checkpoint_dir / "smoke.pt")

        if stop:
            print(f"Stopped at global_step={global_step}")
            break

    final = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": int(args.epochs) - 1,
        "global_step": global_step,
        "config": config,
    }
    torch.save(final, checkpoint_dir / "final.pt")
    print(f"Wrote {checkpoint_dir / 'final.pt'}")


if __name__ == "__main__":
    main()
