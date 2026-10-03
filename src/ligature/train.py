"""Train a GPT on the token files written by scripts/prepare_data.py."""

import argparse
import contextlib
import csv
import math
import shutil
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

from ligature.config import TrainConfig, load_config
from ligature.data import get_batch, load_tokens
from ligature.model import GPT, GPTConfig
from ligature.tokenizer import Tokenizer
from ligature.utils import CSVLogger, get_device, set_seed

LOG_FIELDS = ["step", "train_loss", "val_loss", "val_ppl", "lr", "tokens_per_sec", "elapsed_s"]


def get_lr(step: int, cfg: TrainConfig) -> float:
    """Linear warmup to learning_rate, then cosine decay that reaches min_lr at the last step."""
    if step < cfg.warmup_steps:
        return cfg.learning_rate * (step + 1) / cfg.warmup_steps
    decay_steps = max(1, cfg.max_steps - 1 - cfg.warmup_steps)
    progress = min(1.0, (step - cfg.warmup_steps) / decay_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))  # 1 -> 0
    return cfg.min_lr + cosine * (cfg.learning_rate - cfg.min_lr)


def configure_optimizer(model: GPT, cfg: TrainConfig) -> torch.optim.AdamW:
    """AdamW that decays 2-D weight matrices only, not norms, biases or embeddings."""
    embedding_ids = {id(m.weight) for m in model.modules() if isinstance(m, nn.Embedding)}
    decay, no_decay = [], []
    # parameters() yields the tied embedding / LM head weight once, as an embedding.
    for p in model.parameters():
        if p.dim() == 2 and id(p) not in embedding_ids:
            decay.append(p)
        else:
            no_decay.append(p)
    groups = [
        {"params": decay, "weight_decay": cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=cfg.learning_rate, betas=(cfg.beta1, cfg.beta2))


def autocast_dtype(device: torch.device) -> torch.dtype | None:
    """bf16 on GPUs with native bf16 (compute capability 8.0+), fp16 on older GPUs, else None."""
    if device.type != "cuda":
        return None
    # torch.cuda.is_bf16_supported() can return True on GPUs that only emulate bf16 (slowly),
    # such as the T4 (7.5), so check the compute capability instead.
    major, _ = torch.cuda.get_device_capability(device)
    return torch.bfloat16 if major >= 8 else torch.float16


@torch.no_grad()
def estimate_loss(
    model: nn.Module,
    splits: dict[str, np.ndarray],
    cfg: TrainConfig,
    device: torch.device,
    autocast: contextlib.AbstractContextManager,
) -> dict[str, float]:
    """Mean loss over eval_iters batches per split, with the same batches at every evaluation."""
    model.eval()
    losses = {}
    for name, tokens in splits.items():
        generator = torch.Generator().manual_seed(cfg.seed + 1)
        total = 0.0
        for _ in range(cfg.eval_iters):
            x, y = get_batch(tokens, cfg.batch_size, cfg.block_size, device, generator)
            with autocast:
                _, loss = model(x, y)
            total += loss.item()
        losses[name] = total / cfg.eval_iters
    model.train()
    return losses


def save_checkpoint(
    path: Path,
    model: GPT,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    cfg: TrainConfig,
    progress: dict,
) -> None:
    """Save everything needed to resume; progress holds step, best_val_loss, elapsed_s, data_rng."""
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "model_config": asdict(model.config),
        "train_config": asdict(cfg),
        **progress,
    }
    torch.save(checkpoint, path)


def load_checkpoint(path: str | Path, device: torch.device) -> tuple[GPT, dict]:
    """Rebuild the model from a checkpoint; also return the raw checkpoint dict."""
    checkpoint = torch.load(path, map_location=device)
    model = GPT(GPTConfig(**checkpoint["model_config"])).to(device)
    model.load_state_dict(checkpoint["model"])
    return model, checkpoint


def open_log(path: Path, resume_step: int | None) -> CSVLogger:
    """Start log.csv; on resume keep only rows before resume_step, which is about to be re-run."""
    kept = []
    if resume_step is not None and path.exists():
        with path.open(newline="", encoding="utf-8") as f:
            kept = [row for row in csv.DictReader(f) if int(row["step"]) < resume_step]
    logger = CSVLogger(path, LOG_FIELDS)
    for row in kept:
        logger.log(row)
    return logger


def write_run_info(
    out_dir: Path, cfg: TrainConfig, model: GPT, device: torch.device, dtype
) -> None:
    cfg.to_yaml(out_dir / "config.yaml")
    info = {
        "torch_version": str(torch.__version__),  # a str subclass yaml cannot dump
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "precision": {torch.bfloat16: "bf16", torch.float16: "fp16", None: "fp32"}[dtype],
        "non_embedding_params": model.num_params(),
    }
    (out_dir / "run_info.yaml").write_text(yaml.safe_dump(info, sort_keys=False), encoding="utf-8")


def format_duration(seconds: float | None) -> str:
    """Format seconds as h:mm:ss, or a placeholder when unknown (e.g. before any training)."""
    if seconds is None:
        return "-:--:--"
    s = round(seconds)
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}"


def synchronize(device: torch.device) -> None:
    """Wait for queued GPU work so wall-clock timings are honest."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def train(cfg: TrainConfig, out_dir: Path, resume: bool = False) -> float:
    """Train for cfg.max_steps optimiser updates and return the best validation loss."""
    set_seed(cfg.seed)
    device = get_device() if cfg.device == "auto" else torch.device(cfg.device)
    dtype = autocast_dtype(device)
    autocast = torch.autocast("cuda", dtype=dtype) if dtype else contextlib.nullcontext()

    data_dir = Path(cfg.data_dir)
    tokenizer = Tokenizer.load(data_dir / "tokenizer.json")
    splits = {
        "train": load_tokens(data_dir / "train.bin"),
        "val": load_tokens(data_dir / "val.bin"),
    }

    model = GPT(cfg.model_config(tokenizer.vocab_size)).to(device)
    optimizer = configure_optimizer(model, cfg)
    # Loss scaling is needed for fp16 only; when disabled, every scaler call is a passthrough.
    scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16)
    data_rng = torch.Generator().manual_seed(cfg.seed)
    start_step, best_val_loss, elapsed_before = 0, math.inf, 0.0

    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "ckpt.pt"
    if resume:
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        data_rng.set_state(checkpoint["data_rng"])
        start_step = checkpoint["step"]
        best_val_loss = checkpoint["best_val_loss"]
        elapsed_before = checkpoint["elapsed_s"]
    shutil.copy(data_dir / "tokenizer.json", out_dir / "tokenizer.json")
    write_run_info(out_dir, cfg, model, device, dtype)
    logger = open_log(out_dir / "log.csv", start_step if resume else None)

    train_model = torch.compile(model) if cfg.compile else model
    print(f"{model.num_params():,} non-embedding parameters on {device}")
    t_start = t_last = time.perf_counter()
    tokens_since_last = 0

    for step in range(start_step, cfg.max_steps + 1):
        if step % cfg.eval_interval == 0 or step == cfg.max_steps:
            synchronize(device)
            now = time.perf_counter()
            losses = estimate_loss(train_model, splits, cfg, device, autocast)
            row = {
                "step": step,
                "train_loss": round(losses["train"], 4),
                "val_loss": round(losses["val"], 4),
                "val_ppl": round(math.exp(losses["val"]), 3),
                "lr": get_lr(step, cfg),
                "tokens_per_sec": round(tokens_since_last / (now - t_last)),
                "elapsed_s": round(elapsed_before + now - t_start, 1),
            }
            logger.log(row)
            # The ETA covers the remaining training steps only, not the evaluations still to come.
            remaining_tokens = (cfg.max_steps - step) * cfg.tokens_per_step
            tokens_per_sec = row["tokens_per_sec"]
            eta = remaining_tokens / tokens_per_sec if tokens_per_sec > 0 else None
            print(
                f"step {step:>5} | train {row['train_loss']:.4f} | val {row['val_loss']:.4f} "
                f"| ppl {row['val_ppl']:8.2f} | lr {row['lr']:.2e} "
                f"| {tokens_per_sec:>7,} tok/s | {row['elapsed_s']:>6.1f} s "
                f"| eta {format_duration(eta)}"
            )

            is_best = losses["val"] < best_val_loss
            best_val_loss = min(best_val_loss, losses["val"])
            progress = {
                "step": step,
                "best_val_loss": best_val_loss,
                "elapsed_s": row["elapsed_s"],
                "data_rng": data_rng.get_state(),
            }
            if is_best:
                save_checkpoint(ckpt_path, model, optimizer, scaler, cfg, progress)
            if step == cfg.max_steps:
                # Keep the final weights even when they are not the best, e.g. to train further.
                final_path = out_dir / "ckpt_final.pt"
                save_checkpoint(final_path, model, optimizer, scaler, cfg, progress)
            # Restart the throughput clock after evaluating so eval time is not counted.
            t_last, tokens_since_last = time.perf_counter(), 0
        if step == cfg.max_steps:
            break

        lr = get_lr(step, cfg)
        for group in optimizer.param_groups:
            group["lr"] = lr
        for _ in range(cfg.grad_accum_steps):
            x, y = get_batch(splits["train"], cfg.batch_size, cfg.block_size, device, data_rng)
            with autocast:
                _, loss = train_model(x, y)
            scaler.scale(loss / cfg.grad_accum_steps).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        tokens_since_last += cfg.tokens_per_step

    logger.close()
    return best_val_loss


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--config", type=Path, help="YAML config (ignored with --resume)")
    parser.add_argument("--out", type=Path, required=True, help="run folder, e.g. runs/base")
    parser.add_argument("--resume", action="store_true", help="continue from OUT/ckpt.pt")
    args, overrides = parser.parse_known_args()
    # A resumed run reuses its own saved config, so it continues exactly what it started.
    config_path = args.out / "config.yaml" if args.resume else args.config
    if config_path is None:
        parser.error("--config is required unless --resume is given")
    train(load_config(config_path, overrides), args.out, resume=args.resume)


if __name__ == "__main__":
    main()
