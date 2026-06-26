"""Small DCGAN baseline for 32x32 class-conditional MNIST."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from mnist_bench.constants import FLAT_DIM, IMAGE_SIZE, NUM_CLASSES


def _weights_init(module: nn.Module) -> None:
    classname = module.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(module.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(module.weight.data, 1.0, 0.02)
        nn.init.constant_(module.bias.data, 0)


class Generator(nn.Module):
    """Map (noise, class) -> flat image in [-1, 1]."""

    def __init__(self, *, latent_dim: int = 128, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.num_classes = int(num_classes)
        self.label_emb = nn.Embedding(self.num_classes, self.latent_dim)
        self.fc = nn.Linear(self.latent_dim * 2, 128 * 4 * 4)
        self.net = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
            nn.Tanh(),
        )
        self.apply(_weights_init)

    def forward(self, noise: Tensor, class_id: Tensor) -> Tensor:
        label = self.label_emb(class_id)
        x = torch.cat([noise, label], dim=1)
        x = self.fc(x).view(noise.shape[0], 128, 4, 4)
        return self.net(x).reshape(noise.shape[0], FLAT_DIM)


class Discriminator(nn.Module):
    """Class-conditional discriminator on flat images."""

    def __init__(self, *, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.label_emb = nn.Embedding(self.num_classes, IMAGE_SIZE * IMAGE_SIZE)
        self.net = nn.Sequential(
            nn.Conv2d(4, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.fc = nn.Linear(512 * 2 * 2, 1)
        self.apply(_weights_init)

    def forward(self, flat: Tensor, class_id: Tensor) -> Tensor:
        x = flat.reshape(flat.shape[0], 3, IMAGE_SIZE, IMAGE_SIZE)
        label = self.label_emb(class_id).view(flat.shape[0], 1, IMAGE_SIZE, IMAGE_SIZE)
        x = torch.cat([x, label], dim=1)
        x = self.net(x).reshape(flat.shape[0], -1)
        return self.fc(x)


def dcgan_losses(
    discriminator: Discriminator,
    generator: Generator,
    real: Tensor,
    class_id: Tensor,
    *,
    latent_dim: int,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return (discriminator_loss, generator_loss, fake_flat)."""
    batch_size = real.shape[0]
    real_logits = discriminator(real, class_id)
    noise = torch.randn(batch_size, latent_dim, device=device)
    fake = generator(noise, class_id)
    fake_logits = discriminator(fake.detach(), class_id)
    real_loss = F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits))
    fake_loss = F.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits))
    d_loss = real_loss + fake_loss

    gen_logits = discriminator(fake, class_id)
    g_loss = F.binary_cross_entropy_with_logits(gen_logits, torch.ones_like(gen_logits))
    return d_loss, g_loss, fake


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)
