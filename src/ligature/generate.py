"""Generate stories from a trained checkpoint."""

import argparse
import sys
from pathlib import Path

import torch

from ligature.tokenizer import Tokenizer
from ligature.train import load_checkpoint
from ligature.utils import get_device


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, required=True, help="e.g. runs/base/ckpt.pt")
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument("--n", type=int, default=3, help="number of samples")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8, help="0 means greedy")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-cache", action="store_true", help="recompute every step")
    args = parser.parse_args()

    device = get_device()
    model, _ = load_checkpoint(args.ckpt, device)
    tokenizer = Tokenizer.load(args.ckpt.parent / "tokenizer.json")

    # Every training story follows an <|endoftext|>, so starting with one puts the prompt where
    # a story begins.
    prompt_ids = [tokenizer.eot_id] + tokenizer.encode(args.prompt)
    idx = torch.tensor([prompt_ids] * args.n, device=device)  # (n, T): the samples form a batch
    out = model.generate(
        idx,
        args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        use_cache=not args.no_cache,
        seed=args.seed,
    )

    # Some consoles (cp1252 on Windows) cannot show every character a story may contain.
    sys.stdout.reconfigure(errors="replace")
    for i, row in enumerate(out.tolist(), start=1):
        story = row[1:]  # drop the leading <|endoftext|>
        if tokenizer.eot_id in story:
            story = story[: story.index(tokenizer.eot_id)]  # the story ends at the next one
        print(f"Sample {i}:\n{tokenizer.decode(story)}\n")


if __name__ == "__main__":
    main()
