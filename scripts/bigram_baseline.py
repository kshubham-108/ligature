"""Add-one-smoothed bigram baseline from train.bin: the loss a model with context must beat."""

import argparse
import json
import math
from pathlib import Path

import numpy as np

from ligature.tokenizer import Tokenizer

CHUNK_TOKENS = 10_000_000  # bounds the int64 temporaries to a few hundred MB


def count_bigrams(tokens: np.ndarray, vocab_size: int) -> np.ndarray:
    """Return counts (V, V) where counts[a, b] is how often token b follows token a."""
    V = vocab_size
    counts = np.zeros(V * V, dtype=np.int64)
    for start in range(0, len(tokens) - 1, CHUNK_TOKENS):
        # One token of overlap so the pair spanning two chunks is counted exactly once.
        chunk = tokens[start : start + CHUNK_TOKENS + 1].astype(np.int64)
        # Each pair (a, b) becomes the single index a * V + b, so one bincount counts them all.
        counts += np.bincount(chunk[:-1] * V + chunk[1:], minlength=V * V)
    return counts.reshape(V, V)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=Path("results/bigram.json"))
    args = parser.parse_args()

    V = Tokenizer.load(args.data_dir / "tokenizer.json").vocab_size
    train = np.memmap(args.data_dir / "train.bin", dtype=np.uint16, mode="r")
    val = np.memmap(args.data_dir / "val.bin", dtype=np.uint16, mode="r")

    counts = count_bigrams(train, V)
    # Add-one smoothing: P(b | a) = (count(a, b) + 1) / (count(a) + V), so unseen pairs keep a
    # small non-zero probability and the validation loss stays finite.
    log_probs = np.log(counts + 1) - np.log(counts.sum(axis=1, keepdims=True) + V)  # (V, V)
    a, b = val[:-1].astype(np.int64), val[1:].astype(np.int64)
    val_loss = float(-log_probs[a, b].mean())

    result = {
        "vocab_size": V,
        "train_tokens": len(train),
        "val_tokens": len(val),
        "val_loss": round(val_loss, 4),
        "val_ppl": round(math.exp(val_loss), 3),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"bigram val loss {result['val_loss']:.4f}, perplexity {result['val_ppl']:.2f}")


if __name__ == "__main__":
    main()
