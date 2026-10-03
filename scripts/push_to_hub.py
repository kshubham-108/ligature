"""Export a run's weights in fp16 and upload them with tokenizer.json to a Hugging Face model repo.

The token is read from the HF_TOKEN environment variable and is never printed or stored.
"""

import argparse
import os
import sys
from pathlib import Path

import torch
from huggingface_hub import HfApi

from ligature.train import export_weights, load_checkpoint

MB = 1024 * 1024


def require_token() -> str:
    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("Set the HF_TOKEN environment variable to a token with write access.")
    return token


def max_logit_difference(ckpt_path: Path, weights_path: Path) -> float:
    """Largest absolute logit difference between the checkpoint and the fp16 export."""
    cpu = torch.device("cpu")
    original, _ = load_checkpoint(ckpt_path, cpu)
    exported, _ = load_checkpoint(weights_path, cpu)
    idx = torch.randint(
        original.config.vocab_size,
        (2, original.config.block_size),
        generator=torch.Generator().manual_seed(0),
    )
    with torch.no_grad():
        return (original.eval()(idx)[0] - exported.eval()(idx)[0]).abs().max().item()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("runs/base"))
    parser.add_argument("--repo-id", default="JPSProject/ligature")
    parser.add_argument("--dry-run", action="store_true", help="export and check, upload nothing")
    args = parser.parse_args()

    ckpt_path = args.run_dir / "ckpt.pt"
    weights_path = args.run_dir / "model.pt"
    tokenizer_path = args.run_dir / "tokenizer.json"
    export_weights(ckpt_path, weights_path)
    print(
        f"{ckpt_path}: {ckpt_path.stat().st_size / MB:.1f} MB -> "
        f"{weights_path}: {weights_path.stat().st_size / MB:.1f} MB"
    )
    difference = max_logit_difference(ckpt_path, weights_path)
    print(f"largest logit difference after fp16 export: {difference:.2e}")

    if args.dry_run:
        print("dry run: nothing uploaded")
        return
    api = HfApi(token=require_token())
    api.create_repo(args.repo_id, repo_type="model", exist_ok=True)
    for path in (weights_path, tokenizer_path):
        api.upload_file(
            path_or_fileobj=path,
            path_in_repo=path.name,
            repo_id=args.repo_id,
            repo_type="model",
            commit_message=f"Upload {path.name}",
        )
        print(f"uploaded {path.name}")
    print(f"https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
