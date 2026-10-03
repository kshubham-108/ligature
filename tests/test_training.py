import csv
from pathlib import Path

import numpy as np
import pytest
import torch

from ligature.config import TrainConfig, load_config
from ligature.model import GPT, GPTConfig
from ligature.tokenizer import Tokenizer
from ligature.train import (
    configure_optimizer,
    format_duration,
    get_lr,
    load_checkpoint,
    save_checkpoint,
    train,
)

TINY_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "tiny.yaml"


def test_lr_schedule_values() -> None:
    cfg = TrainConfig(learning_rate=1e-3, min_lr=1e-4, warmup_steps=10, max_steps=101)
    assert get_lr(0, cfg) == pytest.approx(1e-4)  # 1e-3 * 1 / 10: the first step is not wasted
    assert get_lr(9, cfg) == pytest.approx(1e-3)  # last warmup step reaches the peak
    assert get_lr(10, cfg) == pytest.approx(1e-3)  # cosine starts at the peak
    assert get_lr(55, cfg) == pytest.approx(5.5e-4)  # halfway through the decay: the midpoint
    assert get_lr(100, cfg) == pytest.approx(1e-4)  # last step lands on min_lr


def test_checkpoint_round_trip_gives_identical_logits(tmp_path: Path) -> None:
    torch.manual_seed(0)
    cfg = TrainConfig(n_layer=2, n_head=2, d_model=32, block_size=16)
    model = GPT(cfg.model_config(vocab_size=64)).eval()
    optimizer = configure_optimizer(model, cfg)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, model, optimizer, scaler, cfg, {"step": 0})

    loaded, checkpoint = load_checkpoint(path, torch.device("cpu"))
    idx = torch.randint(0, 64, (2, 16))
    assert torch.equal(model(idx)[0], loaded.eval()(idx)[0])
    assert GPTConfig(**checkpoint["model_config"]) == model.config


@pytest.fixture
def pattern_data(tmp_path: Path) -> Path:
    """A data folder whose tokens repeat 0..99, so the next token is easy to learn."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    Tokenizer(merges=[]).save(data_dir / "tokenizer.json")
    tokens = np.tile(np.arange(100, dtype=np.uint16), 200)
    tokens.tofile(data_dir / "train.bin")
    tokens.tofile(data_dir / "val.bin")
    return data_dir


def test_tiny_config_trains_ten_steps_and_loss_falls(pattern_data: Path, tmp_path: Path) -> None:
    # tiny.yaml evaluates every 20 steps, so step 10 is logged only because it is the last one.
    cfg = load_config(TINY_CONFIG, ["--data_dir", str(pattern_data), "--max_steps", "10"])
    out_dir = tmp_path / "run"
    train(cfg, out_dir)

    with (out_dir / "log.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert [int(r["step"]) for r in rows] == [0, 10]
    assert float(rows[-1]["val_loss"]) < float(rows[0]["val_loss"])
    for name in ("ckpt.pt", "ckpt_final.pt", "config.yaml", "run_info.yaml", "tokenizer.json"):
        assert (out_dir / name).exists()
    _, final = load_checkpoint(out_dir / "ckpt_final.pt", torch.device("cpu"))
    assert final["step"] == 10


def test_format_duration() -> None:
    assert format_duration(None) == "-:--:--"
    assert format_duration(59.6) == "0:01:00"
    assert format_duration(3 * 3600 + 25 * 60 + 7) == "3:25:07"
