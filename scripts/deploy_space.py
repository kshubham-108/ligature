"""Assemble the Docker Space in a temporary folder and upload it to Hugging Face Spaces.

The token is read from the HF_TOKEN environment variable and is never printed or stored.
"""

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

REPO_ROOT = Path(__file__).resolve().parents[1]

# The Space front matter lives only here, so the GitHub README stays free of it.
SPACE_README = """---
title: Ligature
sdk: docker
app_port: 7860
---

# Ligature

A small GPT-style language model written from scratch in PyTorch and trained on TinyStories.
Give it the start of a story and it continues it.

- Code: https://github.com/kshubham-108/ligature
- Weights: https://huggingface.co/{model_repo}
"""


def require_token() -> str:
    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("Set the HF_TOKEN environment variable to a token with write access.")
    return token


def assemble(folder: Path, model_repo: str) -> list[Path]:
    """Copy what the Docker image needs into folder; return the files, relative to it."""
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc", "*.egg-info")
    shutil.copy(REPO_ROOT / "Dockerfile", folder)
    shutil.copy(REPO_ROOT / "pyproject.toml", folder)
    shutil.copytree(REPO_ROOT / "app", folder / "app", ignore=ignore)
    shutil.copytree(REPO_ROOT / "src", folder / "src", ignore=ignore)
    (folder / "README.md").write_text(SPACE_README.format(model_repo=model_repo), encoding="utf-8")
    return sorted(p.relative_to(folder) for p in folder.rglob("*") if p.is_file())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--space-id", default="JPSProject/ligature")
    parser.add_argument("--model-repo", default="JPSProject/ligature", help="where the weights are")
    parser.add_argument("--dry-run", action="store_true", help="assemble and list, upload nothing")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        files = assemble(folder, args.model_repo)
        print("Space contents:", *(f"  {path.as_posix()}" for path in files), sep="\n")
        if args.dry_run:
            print("--- README.md ---", (folder / "README.md").read_text(encoding="utf-8"), sep="\n")
            print("dry run: nothing uploaded")
            return

        api = HfApi(token=require_token())
        api.create_repo(args.space_id, repo_type="space", space_sdk="docker", exist_ok=True)
        # Set before uploading, so the first build already knows where to fetch the weights.
        api.add_space_variable(args.space_id, "HF_REPO_ID", args.model_repo)
        api.upload_folder(
            repo_id=args.space_id,
            repo_type="space",
            folder_path=folder,
            commit_message="Deploy Ligature",
        )
    print(f"https://huggingface.co/spaces/{args.space_id}")


if __name__ == "__main__":
    main()
