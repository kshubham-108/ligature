"""Download TinyStories, train the tokeniser and encode both splits to uint16 token files."""

import argparse
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download

from ligature.tokenizer import EOT, Tokenizer

REPO_ID = "roneneldan/TinyStories"
TRAIN_FILE = "TinyStoriesV2-GPT4-train.txt"
VAL_FILE = "TinyStoriesV2-GPT4-valid.txt"
MB = 1024 * 1024
STORIES_PER_TASK = 1000

# Each worker process loads its own tokeniser once, so its chunk cache lives across tasks.
_worker_tokenizer: Tokenizer | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--train-mb", type=float, default=300, help="MB of the train file to use")
    parser.add_argument("--tok-train-mb", type=float, default=20, help="MB to train the tokeniser")
    parser.add_argument("--vocab-size", type=int, default=4096)
    parser.add_argument(
        "--local",
        action="store_true",
        help="download only the validation file and split it 90/10 into train and val",
    )
    return parser.parse_args()


def download(filename: str, data_dir: Path) -> Path:
    return Path(hf_hub_download(REPO_ID, filename, repo_type="dataset", local_dir=data_dir / "raw"))


def read_stories(path: Path, max_mb: float | None = None) -> list[str]:
    """Read whole stories from a file, optionally only from its first max_mb megabytes."""
    max_bytes = -1 if max_mb is None else int(max_mb * MB)
    with path.open("rb") as f:
        raw = f.read(max_bytes)
    # A read cut short can end mid-character; that tail belongs to the story dropped below.
    stories = raw.decode("utf-8", errors="ignore").split(EOT)
    if len(raw) == max_bytes:
        stories = stories[:-1]
    return [s.strip() for s in stories if s.strip()]


def take_mb(stories: list[str], mb: float) -> list[str]:
    """Return leading stories until their UTF-8 size reaches mb megabytes."""
    taken, size = [], 0
    for story in stories:
        if size >= mb * MB:
            break
        taken.append(story)
        size += len(story.encode("utf-8"))
    return taken


def _init_worker(tokenizer_path: str) -> None:
    global _worker_tokenizer
    _worker_tokenizer = Tokenizer.load(tokenizer_path)


def _encode_stories(stories: list[str]) -> np.ndarray:
    ids: list[int] = []
    for story in stories:
        ids.extend(_worker_tokenizer.encode(story))
        ids.append(_worker_tokenizer.eot_id)
    return np.array(ids, dtype=np.uint16)


def encode_split(stories: list[str], tokenizer_path: Path, out_path: Path) -> tuple[int, float]:
    """Encode stories in parallel, each followed by <|endoftext|>; return (tokens, seconds)."""
    start = time.perf_counter()
    tasks = [stories[i : i + STORIES_PER_TASK] for i in range(0, len(stories), STORIES_PER_TASK)]
    with Pool(initializer=_init_worker, initargs=(str(tokenizer_path),)) as pool:
        tokens = np.concatenate(pool.map(_encode_stories, tasks))
    tokens.tofile(out_path)
    return len(tokens), time.perf_counter() - start


def main() -> None:
    args = parse_args()
    if args.vocab_size > 2**16:
        raise ValueError("vocab_size must fit in uint16")
    args.data_dir.mkdir(parents=True, exist_ok=True)

    if args.local:
        stories = read_stories(download(VAL_FILE, args.data_dir))
        n_train = int(0.9 * len(stories))
        train, val = stories[:n_train], stories[n_train:]
    else:
        train = read_stories(download(TRAIN_FILE, args.data_dir), args.train_mb)
        val = read_stories(download(VAL_FILE, args.data_dir))

    tok_stories = take_mb(train, args.tok_train_mb)
    tok_mb = sum(len(s.encode("utf-8")) for s in tok_stories) / MB
    start = time.perf_counter()
    tokenizer = Tokenizer.train(EOT.join(tok_stories), args.vocab_size)
    tok_seconds = time.perf_counter() - start
    tokenizer_path = args.data_dir / "tokenizer.json"
    tokenizer.save(tokenizer_path)
    print(
        f"tokeniser: {tokenizer.vocab_size} tokens trained on {tok_mb:.1f} MB "
        f"in {tok_seconds:.1f} s"
    )

    for name, split in (("train", train), ("val", val)):
        n_bytes = sum(len(s.encode("utf-8")) for s in split)
        n_tokens, seconds = encode_split(split, tokenizer_path, args.data_dir / f"{name}.bin")
        print(
            f"{name}: {len(split):,} stories, {n_bytes / MB:.1f} MB -> {n_tokens:,} tokens "
            f"in {seconds:.1f} s ({n_bytes / MB / seconds:.2f} MB/s)"
        )
        if name == "val":
            # Exclude the <|endoftext|> appended after each story: it stands for no input bytes.
            print(f"val bytes per token: {n_bytes / (n_tokens - len(split)):.3f}")


if __name__ == "__main__":
    main()
