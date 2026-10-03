"""Training configuration: a flat dataclass loaded from YAML, with command-line overrides."""

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import yaml

from ligature.model import GPTConfig


@dataclass
class TrainConfig:
    data_dir: str = "data"
    seed: int = 1337
    device: str = "auto"  # "auto" picks CUDA, then MPS, then CPU

    n_layer: int = 6
    n_head: int = 6
    d_model: int = 384
    block_size: int = 256
    dropout: float = 0.0
    pos_encoding: str = "rope"
    bias: bool = False
    use_sdpa: bool = False

    batch_size: int = 64  # sequences per micro-batch
    grad_accum_steps: int = 1
    max_steps: int = 4000  # optimiser updates
    learning_rate: float = 1e-3
    min_lr: float = 1e-4
    warmup_steps: int = 200
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    compile: bool = False

    eval_interval: int = 250
    eval_iters: int = 50

    @property
    def tokens_per_step(self) -> int:
        return self.batch_size * self.grad_accum_steps * self.block_size

    def model_config(self, vocab_size: int) -> GPTConfig:
        return GPTConfig(
            vocab_size=vocab_size,
            block_size=self.block_size,
            n_layer=self.n_layer,
            n_head=self.n_head,
            d_model=self.d_model,
            dropout=self.dropout,
            pos_encoding=self.pos_encoding,
            bias=self.bias,
            use_sdpa=self.use_sdpa,
        )

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "TrainConfig":
        types = {f.name: f.type for f in fields(cls)}
        unknown = sorted(set(values) - set(types))
        if unknown:
            raise ValueError(f"unknown config keys: {unknown}")
        return cls(**{key: _coerce(value, types[key]) for key, value in values.items()})

    def to_yaml(self, path: str | Path) -> None:
        Path(path).write_text(yaml.safe_dump(asdict(self), sort_keys=False), encoding="utf-8")


def load_config(path: str | Path | None, overrides: list[str]) -> TrainConfig:
    """Load a YAML config, then apply --key value overrides on top."""
    values = yaml.safe_load(Path(path).read_text(encoding="utf-8")) if path else None
    values = values or {}
    values.update(parse_overrides(overrides))
    return TrainConfig.from_dict(values)


def parse_overrides(args: list[str]) -> dict[str, str]:
    """Turn ["--max_steps", "10"] into {"max_steps": "10"}; dashes in keys become underscores."""
    if len(args) % 2 != 0:
        raise ValueError(f"overrides must be --key value pairs, got {args}")
    overrides = {}
    for key, value in zip(args[::2], args[1::2], strict=True):
        if not key.startswith("--"):
            raise ValueError(f"expected --key, got {key!r}")
        overrides[key[2:].replace("-", "_")] = value
    return overrides


def _coerce(value: Any, type_: type) -> Any:
    # Command-line values are always strings, and PyYAML reads "1e-3" (no decimal point) as one.
    if type_ is bool and isinstance(value, str):
        if value.lower() in ("true", "1", "yes"):
            return True
        if value.lower() in ("false", "0", "no"):
            return False
        raise ValueError(f"cannot read {value!r} as a bool")
    return type_(value)
