"""Train a class-conditional DCGAN baseline on MNIST."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from tqdm.auto import tqdm

from mnist_bench.constants import IMAGE_SIZE, NUM_CLASSES
from mnist_bench.data import build_mnist_dataloader, flat_to_images
from mnist_bench.dcgan import Discriminator, Generator, count_parameters, dcgan_losses
from un0.common import resolve_device, seed_everything
from torchvision.utils import save_image


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints/dcgan"))
    parser.add_argument("--sample-every", type=int, default=25)
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--data-root", default="data")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    seed_everything(int(args.seed))
    device = resolve_device("auto")

    loader = build_mnist_dataloader(
        root=str(args.data_root),
        batch_size=int(args.batch_size),
        pin_memory=device.type == "cuda",
    )

    generator = Generator(latent_dim=int(args.latent_dim)).to(device)
    discriminator = Discriminator().to(device)
    print(f"Generator params: {count_parameters(generator):,}")
    print(f"Discriminator params: {count_parameters(discriminator):,}")

    opt_g = torch.optim.Adam(generator.parameters(), lr=float(args.lr), betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=float(args.lr), betas=(0.5, 0.999))

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}

    global_step = 0
    stop = False
    for epoch in range(int(args.epochs)):
        generator.train()
        discriminator.train()
        progress = tqdm(enumerate(loader), total=len(loader), desc=f"epoch {epoch + 1}")
        for _step, batch in progress:
            real = batch["data"].to(device)
            class_id = batch["class_id"].to(device)

            opt_d.zero_grad(set_to_none=True)
            d_loss, g_loss, fake = dcgan_losses(
                discriminator,
                generator,
                real,
                class_id,
                latent_dim=int(args.latent_dim),
                device=device,
            )
            d_loss.backward()
            opt_d.step()

            opt_g.zero_grad(set_to_none=True)
            _, g_loss, _ = dcgan_losses(
                discriminator,
                generator,
                real,
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
        if epoch_num % int(args.sample_every) == 0 or stop:
            generator.eval()
            with torch.no_grad():
                class_ids = torch.arange(NUM_CLASSES, device=device).repeat_interleave(8)
                noise = torch.randn(class_ids.shape[0], int(args.latent_dim), device=device)
                flat = generator(noise, class_ids)
                images = flat_to_images(flat)
            sample_path = checkpoint_dir / "samples" / f"epoch_{epoch_num:04d}.png"
            sample_path.parent.mkdir(parents=True, exist_ok=True)
            save_image(images, sample_path, nrow=8)

        if epoch_num % int(args.save_every) == 0 or stop:
            state = {
                "generator": generator.state_dict(),
                "discriminator": discriminator.state_dict(),
                "epoch": epoch,
                "global_step": global_step,
                "config": config,
            }
            torch.save(state, checkpoint_dir / "latest.pt")
            if stop:
                torch.save(state, checkpoint_dir / "smoke.pt")

        if stop:
            print(f"Stopped at global_step={global_step}")
            break

    final = {
        "generator": generator.state_dict(),
        "discriminator": discriminator.state_dict(),
        "epoch": int(args.epochs) - 1,
        "global_step": global_step,
        "config": config,
    }
    torch.save(final, checkpoint_dir / "final.pt")
    print(f"Wrote {checkpoint_dir / 'final.pt'}")


if __name__ == "__main__":
    main()
