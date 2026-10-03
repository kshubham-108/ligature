"""Device selection, seeding and a minimal CSV logger."""

import csv
import random
from pathlib import Path

import numpy as np
import torch


def get_device() -> torch.device:
    """Return the best available device: CUDA, then MPS, then CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch (all devices) for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    # torch.manual_seed also seeds every CUDA device.
    torch.manual_seed(seed)


class CSVLogger:
    """Append rows of metrics to a CSV file with a fixed set of columns."""

    def __init__(self, path: str | Path, fieldnames: list[str]) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=fieldnames)
        self._writer.writeheader()

    def log(self, row: dict[str, float | int | str]) -> None:
        """Write one row; unknown keys raise, missing keys are left blank."""
        self._writer.writerow(row)
        # Flush every row so a crashed or interrupted run still leaves a usable log.
        self._file.flush()

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> "CSVLogger":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
