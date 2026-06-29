"""Train a class-conditional DCGAN baseline on MNIST."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from tqdm.auto import tqdm

from mnist_bench.constants import NUM_CLASSES
from mnist_bench.data import build_mnist_dataloader, flat_to_images
from mnist_bench.dcgan import (
    Discriminator,
    Generator,
    count_parameters,
    dcgan_discriminator_loss,
    dcgan_generator_loss,
)
from mnist_bench.digits import (
    init_progress_manifest,
    save_dcgan_progress_row,
    stitch_progress_grid,
)
from un0.common import resolve_device, seed_everything
from torchvision.utils import save_image


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr-g", type=float, default=2e-4, help="Generator learning rate.")
    parser.add_argument("--lr-d", type=float, default=1e-4, help="Discriminator learning rate.")
    parser.add_argument("--g-steps", type=int, default=2, help="Generator steps per discriminator step.")
    parser.add_argument("--label-smooth", type=float, default=0.9, help="Real label for D (label smoothing).")
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints/dcgan"))
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume training from a checkpoint .pt (generator, discriminator, optimizers).",
    )
    parser.add_argument("--sample-every", type=int, default=0, help="Save sample grid (0=off).")
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument(
        "--progress-every",
        type=int,
        default=0,
        help="Save best-of-N progress row every N epochs (0=off).",
    )
    parser.add_argument("--progress-candidates", type=int, default=32)
    parser.add_argument("--progress-rows-dir", type=Path, default=Path("digits/dcgan/progress_rows"))
    parser.add_argument("--progress-manifest", type=Path, default=Path("digits/dcgan/progress_manifest.json"))
    parser.add_argument("--progress-output", type=Path, default=Path("digits/dcgan/progress_10x10.png"))
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--device", default="auto")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    seed_everything(int(args.seed))
    device = resolve_device(str(args.device))

    loader = build_mnist_dataloader(
        root=str(args.data_root),
        batch_size=int(args.batch_size),
        pin_memory=device.type == "cuda",
        dcgan=True,
    )

    generator = Generator(latent_dim=int(args.latent_dim)).to(device)
    discriminator = Discriminator().to(device)
    print(f"Generator params: {count_parameters(generator):,}")
    print(f"Discriminator params: {count_parameters(discriminator):,}")
    print(
        f"Stable DCGAN: lr_g={args.lr_g} lr_d={args.lr_d} g_steps={args.g_steps} "
        f"label_smooth={args.label_smooth}",
    )

    opt_g = torch.optim.Adam(generator.parameters(), lr=float(args.lr_g), betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=float(args.lr_d), betas=(0.5, 0.999))

    start_epoch = 0
    global_step = 0
    resume_state = None
    if args.resume is not None:
        resume_path = Path(args.resume)
        if not resume_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        resume_state = torch.load(resume_path, map_location=device, weights_only=False)
        generator.load_state_dict(resume_state["generator"])
        discriminator.load_state_dict(resume_state["discriminator"])
        start_epoch = int(resume_state.get("epoch", 0)) + 1
        global_step = int(resume_state.get("global_step", 0))
        print(f"Resuming from {resume_path} at epoch {start_epoch} (global_step={global_step})")

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}

    if resume_state is not None:
        if "opt_g" in resume_state and "opt_d" in resume_state:
            opt_g.load_state_dict(resume_state["opt_g"])
            opt_d.load_state_dict(resume_state["opt_d"])
        else:
            print("  (checkpoint has no optimizer state — using fresh Adam states)")

    progress_every = int(args.progress_every)
    if progress_every > 0 and start_epoch == 0:
        init_progress_manifest(
            manifest_path=Path(args.progress_manifest),
            candidates_per_digit=int(args.progress_candidates),
            seed=int(args.seed),
        )

    stop = False
    g_steps = max(1, int(args.g_steps))
    for epoch in range(start_epoch, int(args.epochs)):
        generator.train()
        discriminator.train()
        progress = tqdm(enumerate(loader), total=len(loader), desc=f"epoch {epoch + 1}")
        for _step, batch in progress:
            real = batch["data"].to(device)
            class_id = batch["class_id"].to(device)

            opt_d.zero_grad(set_to_none=True)
            d_loss, _fake = dcgan_discriminator_loss(
                discriminator,
                generator,
                real,
                class_id,
                latent_dim=int(args.latent_dim),
                device=device,
                label_smooth=float(args.label_smooth),
            )
            d_loss.backward()
            opt_d.step()

            g_loss = torch.tensor(0.0, device=device)
            for _ in range(g_steps):
                opt_g.zero_grad(set_to_none=True)
                g_loss = dcgan_generator_loss(
                    discriminator,
                    generator,
                    class_id,
                    latent_dim=int(args.latent_dim),
                    device=device,
                )
                g_loss.backward()
                opt_g.step()

            global_step += 1
            if global_step % 20 == 0:
                progress.set_postfix(
                    d_loss=float(d_loss.detach()),
                    g_loss=float(g_loss.detach()),
                )

            if args.max_steps > 0 and global_step >= int(args.max_steps):
                stop = True
                break

        epoch_num = epoch + 1
        if int(args.sample_every) > 0 and (epoch_num % int(args.sample_every) == 0 or stop):
            generator.eval()
            with torch.no_grad():
                class_ids = torch.arange(NUM_CLASSES, device=device).repeat_interleave(8)
                noise = torch.randn(class_ids.shape[0], int(args.latent_dim), device=device)
                flat = generator(noise, class_ids)
                images = flat_to_images(flat)
            sample_path = checkpoint_dir / "samples" / f"epoch_{epoch_num:04d}.png"
            sample_path.parent.mkdir(parents=True, exist_ok=True)
            save_image(images, sample_path, nrow=8)
            generator.train()

        if progress_every > 0 and epoch_num % progress_every == 0:
            row_path = save_dcgan_progress_row(
                generator,
                epoch_num,
                latent_dim=int(args.latent_dim),
                rows_dir=Path(args.progress_rows_dir),
                manifest_path=Path(args.progress_manifest),
                device=device,
                candidates=int(args.progress_candidates),
                seed=int(args.seed),
            )
            print(f"Progress row epoch {epoch_num}: {row_path}")

        if epoch_num % int(args.save_every) == 0 or stop:
            state = _checkpoint_state(
                generator,
                discriminator,
                opt_g,
                opt_d,
                epoch=epoch,
                global_step=global_step,
                config=config,
            )
            torch.save(state, checkpoint_dir / "latest.pt")
            if stop:
                torch.save(state, checkpoint_dir / "smoke.pt")

        if stop:
            print(f"Stopped at global_step={global_step}")
            break

    final_epoch_idx = epoch if stop else int(args.epochs) - 1
    final = _checkpoint_state(
        generator,
        discriminator,
        opt_g,
        opt_d,
        epoch=final_epoch_idx,
        global_step=global_step,
        config=config,
    )
    torch.save(final, checkpoint_dir / "final.pt")
    print(f"Wrote {checkpoint_dir / 'final.pt'}")

    if progress_every > 0:
        final_epoch = final_epoch_idx + 1
        if final_epoch % progress_every != 0:
            save_dcgan_progress_row(
                generator,
                final_epoch,
                latent_dim=int(args.latent_dim),
                rows_dir=Path(args.progress_rows_dir),
                manifest_path=Path(args.progress_manifest),
                device=device,
                candidates=int(args.progress_candidates),
                seed=int(args.seed),
            )
        manifest = stitch_progress_grid(
            manifest_path=Path(args.progress_manifest),
            output_image=Path(args.progress_output),
        )
        print(f"Wrote {manifest.output_image}")


def _checkpoint_state(
    generator: Generator,
    discriminator: Discriminator,
    opt_g: torch.optim.Optimizer,
    opt_d: torch.optim.Optimizer,
    *,
    epoch: int,
    global_step: int,
    config: dict,
) -> dict:
    return {
        "generator": generator.state_dict(),
        "discriminator": discriminator.state_dict(),
        "opt_g": opt_g.state_dict(),
        "opt_d": opt_d.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "config": config,
    }


if __name__ == "__main__":
    main()
