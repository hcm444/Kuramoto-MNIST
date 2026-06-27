"""MNIST dataloader: pad 28x28 grayscale to 32x32 RGB in [-1, 1]."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from mnist_bench.constants import CHANNELS, IMAGE_SIZE, NUM_CLASSES

_MNIST_MEAN = (0.1307,)
_MNIST_STD = (0.3081,)


def _pad_and_rgb(tensor: Tensor) -> Tensor:
    """Pad 28x28 to 32x32 and repeat the channel to RGB."""
    if tensor.shape[0] != 1:
        msg = f"Expected single-channel MNIST tensor, got shape {tensor.shape}."
        raise ValueError(msg)
    padded = torch.nn.functional.pad(tensor, (2, 2, 2, 2), mode="constant", value=0.0)
    return padded.repeat(CHANNELS, 1, 1)


def build_mnist_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(_MNIST_MEAN, _MNIST_STD),
            transforms.Lambda(_pad_and_rgb),
        ]
    )


def collate_mnist_batch(batch: list[tuple[Tensor, int]]) -> dict[str, Tensor]:
    images = torch.stack([item[0] for item in batch])
    labels = torch.as_tensor([int(item[1]) for item in batch], dtype=torch.long)
    flat = images.reshape(images.shape[0], -1) * 2.0 - 1.0
    return {"data": flat, "class_id": labels}


class _MNISTWithId(datasets.MNIST):
    def __getitem__(self, index: int) -> tuple[Tensor, int, int]:
        image, label = super().__getitem__(index)
        return image, int(label), int(index)


def build_mnist_dataloader(
    *,
    root: str = "data",
    train: bool = True,
    batch_size: int = 512,
    num_workers: int = 4,
    pin_memory: bool = False,
    include_sample_id: bool = False,
    shuffle: bool = True,
    drop_last: bool = True,
    max_samples: int = 0,
) -> DataLoader[dict[str, Tensor]]:
    transform = build_mnist_transform()
    dataset: Any
    if include_sample_id:
        dataset = _MNISTWithId(root=root, train=train, download=True, transform=transform)

        def _collate(batch: list[tuple[Tensor, int, int]]) -> dict[str, Tensor]:
            images = torch.stack([item[0] for item in batch])
            labels = torch.as_tensor([item[1] for item in batch], dtype=torch.long)
            sample_ids = torch.as_tensor([item[2] for item in batch], dtype=torch.long)
            flat = images.reshape(images.shape[0], -1) * 2.0 - 1.0
            return {"data": flat, "class_id": labels, "sample_id": sample_ids}

        collate_fn = _collate
    else:
        dataset = datasets.MNIST(root=root, train=train, download=True, transform=transform)
        collate_fn = collate_mnist_batch

    if max_samples > 0 and max_samples < len(dataset):
        dataset = Subset(dataset, range(int(max_samples)))

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=collate_fn,
    )


def flat_to_images(flat: Tensor) -> Tensor:
    """Convert flat [-1, 1] tensors to (N, 3, H, W) in [0, 1]."""
    images = flat.reshape(-1, CHANNELS, IMAGE_SIZE, IMAGE_SIZE)
    return ((images + 1.0) * 0.5).clamp(0.0, 1.0)


def images_to_grayscale(images: Tensor) -> Tensor:
    """Convert (N, 3, H, W) in [0, 1] to (N, 1, H, W) for MNIST-style export."""
    if images.shape[1] == 1:
        return images
    return images.mean(dim=1, keepdim=True)


__all__ = [
    "NUM_CLASSES",
    "build_mnist_dataloader",
    "build_mnist_transform",
    "collate_mnist_batch",
    "flat_to_images",
    "images_to_grayscale",
]
