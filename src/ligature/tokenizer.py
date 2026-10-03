"""Byte-level BPE tokeniser in the style of GPT-2, written from scratch."""

import heapq
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import regex

# GPT-2 pre-tokenisation: contractions, runs of letters, digits or punctuation (each optionally
# led by one space), and whitespace. Merges are learned and applied within these chunks only.
GPT2_SPLIT_PATTERN = (
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)
EOT = "<|endoftext|>"

Pair = tuple[int, int]


class Tokenizer:
    """Byte-level BPE with ids ordered as 256 bytes, then merges, then special tokens."""

    def __init__(self, merges: list[Pair], special_tokens: tuple[str, ...] = (EOT,)) -> None:
        # A merge's id doubles as its rank: lower ids were learned earlier and are applied first.
        self.merges: dict[Pair, int] = {pair: 256 + i for i, pair in enumerate(merges)}
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        for (a, b), new_id in self.merges.items():
            self.vocab[new_id] = self.vocab[a] + self.vocab[b]

        first_special = 256 + len(merges)
        self.special_tokens = {tok: first_special + i for i, tok in enumerate(special_tokens)}
        for tok, i in self.special_tokens.items():
            self.vocab[i] = tok.encode("utf-8")

        self._pattern = regex.compile(GPT2_SPLIT_PATTERN)
        self._special_pattern = _special_split_pattern(special_tokens)
        self._cache: dict[str, list[int]] = {}

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def eot_id(self) -> int:
        return self.special_tokens[EOT]

    @classmethod
    def train(
        cls, text: str, vocab_size: int, special_tokens: tuple[str, ...] = (EOT,)
    ) -> "Tokenizer":
        """Learn merges until the vocabulary, special tokens included, has vocab_size entries."""
        n_merges = vocab_size - 256 - len(special_tokens)
        if n_merges < 0:
            raise ValueError(f"vocab_size must be at least {256 + len(special_tokens)}")

        pattern = regex.compile(GPT2_SPLIT_PATTERN)
        special_pattern = _special_split_pattern(special_tokens)
        pieces = special_pattern.split(text) if special_pattern else [text]
        chunk_counts = Counter(
            chunk
            for piece in pieces
            if piece not in special_tokens
            for chunk in pattern.findall(piece)
        )
        return cls(_learn_merges(chunk_counts, n_merges), special_tokens)

    def encode(self, text: str) -> list[int]:
        """Encode text, mapping each special token straight to its id."""
        pieces = self._special_pattern.split(text) if self._special_pattern else [text]
        ids: list[int] = []
        for piece in pieces:
            if piece in self.special_tokens:
                ids.append(self.special_tokens[piece])
                continue
            for chunk in self._pattern.findall(piece):
                if chunk not in self._cache:
                    self._cache[chunk] = self._encode_chunk(chunk)
                ids.extend(self._cache[chunk])
        return ids

    def decode(self, ids: list[int]) -> str:
        """Decode ids; invalid UTF-8 (e.g. half a character at the end) becomes U+FFFD."""
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")

    def save(self, path: str | Path) -> None:
        data = {
            "merges": [list(pair) for pair in self.merges],
            "special_tokens": list(self.special_tokens),
        }
        Path(path).write_text(json.dumps(data), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Tokenizer":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        merges = [(a, b) for a, b in data["merges"]]
        return cls(merges, tuple(data["special_tokens"]))

    def _encode_chunk(self, chunk: str) -> list[int]:
        ids = list(chunk.encode("utf-8"))
        while len(ids) >= 2:
            # Apply the earliest-learned merge present, replaying the order used in training.
            pair = min(zip(ids, ids[1:], strict=False), key=lambda p: self.merges.get(p, math.inf))
            if pair not in self.merges:
                break
            ids = _merge(ids, pair, self.merges[pair])
        return ids


def _special_split_pattern(special_tokens: tuple[str, ...]) -> regex.Pattern | None:
    """Pattern whose split() keeps special tokens as separate pieces, longest match first."""
    if not special_tokens:
        return None
    alternatives = sorted(special_tokens, key=len, reverse=True)
    return regex.compile("(" + "|".join(regex.escape(t) for t in alternatives) + ")")


def _merge(ids: list[int], pair: Pair, new_id: int) -> list[int]:
    """Replace every non-overlapping occurrence of pair, left to right, with new_id."""
    out = []
    i = 0
    while i < len(ids):
        if i + 1 < len(ids) and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


def _learn_merges(chunk_counts: Counter[str], n_merges: int) -> list[Pair]:
    """Run BPE on a chunk-frequency table, updating only the chunks each merge touches."""
    words = [list(chunk.encode("utf-8")) for chunk in chunk_counts]
    freqs = list(chunk_counts.values())

    # Pair counts are weighted by chunk frequency; pair_to_words lets a merge find its chunks.
    pair_counts: Counter[Pair] = Counter()
    pair_to_words: defaultdict[Pair, set[int]] = defaultdict(set)
    for w, (ids, freq) in enumerate(zip(words, freqs, strict=True)):
        for pair in zip(ids, ids[1:], strict=False):
            pair_counts[pair] += freq
            pair_to_words[pair].add(w)

    # Max-heap of (-count, pair): the most frequent pair wins and ties go to the smallest pair,
    # which makes training deterministic. Entries go stale when a count changes, so a popped
    # entry is used only if it still matches the live count.
    heap = [(-count, pair) for pair, count in pair_counts.items()]
    heapq.heapify(heap)

    merges: list[Pair] = []
    while len(merges) < n_merges and heap:
        neg_count, pair = heapq.heappop(heap)
        if pair_counts.get(pair, 0) != -neg_count:
            continue
        new_id = 256 + len(merges)
        merges.append(pair)

        changed: set[Pair] = set()
        for w in pair_to_words.pop(pair):
            ids, freq = words[w], freqs[w]
            for p in zip(ids, ids[1:], strict=False):
                pair_counts[p] -= freq
                pair_to_words[p].discard(w)
                changed.add(p)
            ids = words[w] = _merge(ids, pair, new_id)
            for p in zip(ids, ids[1:], strict=False):
                pair_counts[p] += freq
                pair_to_words[p].add(w)
                changed.add(p)

        for p in changed:
            if pair_counts[p] > 0:
                heapq.heappush(heap, (-pair_counts[p], p))
            else:
                del pair_counts[p]
                pair_to_words.pop(p, None)
    return merges
