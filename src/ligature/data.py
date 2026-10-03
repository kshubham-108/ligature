"""Random contiguous training windows from a uint16 token file."""

from pathlib import Path

import numpy as np
import torch


def load_tokens(path: str | Path) -> np.memmap:
    """Map a token file into memory without reading it; pages load on first access."""
    return np.memmap(path, dtype=np.uint16, mode="r")


def get_batch(
    tokens: np.ndarray,
    batch_size: int,
    block_size: int,
    device: torch.device,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample (x, y) of shape (batch_size, block_size), where y is x shifted left by one token."""
    starts = torch.randint(len(tokens) - block_size, (batch_size,), generator=generator).tolist()
    # int64 because uint16 is not a valid index type for nn.Embedding or cross_entropy.
    x = torch.stack([torch.from_numpy(tokens[i : i + block_size].astype(np.int64)) for i in starts])
    y = torch.stack(
        [torch.from_numpy(tokens[i + 1 : i + 1 + block_size].astype(np.int64)) for i in starts]
    )
    if device.type == "cuda":
        # Page-locked host memory lets the non_blocking copy run asynchronously with GPU work.
        x, y = x.pin_memory(), y.pin_memory()
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)
