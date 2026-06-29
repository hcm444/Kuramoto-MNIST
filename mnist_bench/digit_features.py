"""MNIST-trained CNN features for Kuramoto drift loss (DINO replacement)."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from mnist_bench.constants import IMAGE_SIZE
from mnist_bench.data import build_mnist_dataloader

DEFAULT_ENCODER_PATH = Path("checkpoints/mnist_digit_encoder.pt")


def _flat_to_gray(x_flat: Tensor, *, image_size: int = IMAGE_SIZE) -> Tensor:
    """(B, 3*H*W) in [-1, 1] → (B, 1, H, W) grayscale in [0, 1]."""
    images = x_flat.reshape(x_flat.shape[0], 3, image_size, image_size)
    images = (images + 1.0) * 0.5
    return images.mean(dim=1, keepdim=True)


FEATURE_DIM = 128


class DigitEncoder(nn.Module):
    """Small conv tower; multi-scale pooled views for drift loss."""

    def __init__(self, *, feature_dim: int = FEATURE_DIM) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.proj1 = nn.Conv2d(32, self.feature_dim, 1)
        self.proj2 = nn.Conv2d(64, self.feature_dim, 1)
        self.proj3 = nn.Conv2d(128, self.feature_dim, 1)
        self.proj4 = nn.Conv2d(256, self.feature_dim, 1)
        self.head = nn.Linear(256, 10)

    def _views_from_map(self, x: Tensor) -> list[Tensor]:
        views: list[Tensor] = []
        pooled = F.adaptive_avg_pool2d(x, output_size=(2, 2))
        for row in range(2):
            for col in range(2):
                views.append(F.normalize(pooled[:, :, row, col], p=2, dim=1))
        global_vec = F.adaptive_avg_pool2d(x, output_size=1).flatten(1)
        views.append(F.normalize(global_vec, p=2, dim=1))
        return views

    def encode_views(self, gray: Tensor) -> list[Tensor]:
        """Return L2-normalized feature vectors (one per spatial/global view)."""
        views: list[Tensor] = []
        x = self.block1(gray)
        views.extend(self._views_from_map(self.proj1(x)))
        x = self.block2(x)
        views.extend(self._views_from_map(self.proj2(x)))
        x = self.block3(x)
        views.extend(self._views_from_map(self.proj3(x)))
        x = self.block4(x)
        views.extend(self._views_from_map(self.proj4(x)))
        return views

    def forward(self, gray: Tensor) -> Tensor:
        x = self.block4(self.block3(self.block2(self.block1(gray))))
        return self.head(F.adaptive_avg_pool2d(x, output_size=1).flatten(1))


class DigitFeatureExtractor(nn.Module):
    """Frozen MNIST encoder used by conditional_drift_loss (DINO drop-in)."""

    def __init__(self, *, checkpoint: Path | str | None = None) -> None:
        super().__init__()
        self.encoder = DigitEncoder()
        path = Path(checkpoint) if checkpoint is not None else DEFAULT_ENCODER_PATH
        if not path.is_file():
            pretrain_digit_encoder(output_path=path)
        state = torch.load(path, map_location="cpu", weights_only=False)
        self.encoder.load_state_dict(state["encoder"])
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval()

    def train(self, mode: bool = True) -> DigitFeatureExtractor:
        super().train(mode)
        self.encoder.eval()
        return self

    def forward(self, x_flat: Tensor, *, image_size: int = IMAGE_SIZE) -> list[Tensor]:
        gray = _flat_to_gray(x_flat, image_size=image_size)
        gray = gray.to(dtype=next(self.encoder.parameters()).dtype)
        return self.encoder.encode_views(gray)


def pretrain_digit_encoder(
    *,
    output_path: Path | str = DEFAULT_ENCODER_PATH,
    data_root: str = "data",
    epochs: int = 5,
    batch_size: int = 256,
    lr: float = 1e-3,
    device: str = "auto",
) -> Path:
    """Train the digit encoder on MNIST classification; save for drift loss."""
    from un0.common import resolve_device, seed_everything

    seed_everything(42)
    torch_device = resolve_device(device)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = DigitEncoder().to(torch_device)
    loader = build_mnist_dataloader(
        root=data_root,
        batch_size=batch_size,
        num_workers=2,
        pin_memory=torch_device.type == "cuda",
    )
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        correct = 0
        total = 0
        progress = tqdm(loader, desc=f"digit-encoder epoch {epoch + 1}/{epochs}")
        for batch in progress:
            x = batch["data"].to(torch_device)
            y = batch["class_id"].to(torch_device)
            gray = _flat_to_gray(x)
            logits = model(gray)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            preds = logits.argmax(dim=1)
            correct += int((preds == y).sum())
            total += int(y.shape[0])
            progress.set_postfix(loss=float(loss.detach()), acc=correct / max(total, 1))

    acc = correct / max(total, 1)
    print(f"Digit encoder accuracy: {acc * 100:.2f}%")
    torch.save(
        {
            "encoder": model.state_dict(),
            "accuracy": acc,
            "epochs": epochs,
        },
        output_path,
    )
    print(f"Wrote {output_path}")
    return output_path


__all__ = [
    "DEFAULT_ENCODER_PATH",
    "DigitEncoder",
    "DigitFeatureExtractor",
    "pretrain_digit_encoder",
]
