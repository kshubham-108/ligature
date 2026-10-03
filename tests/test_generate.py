import pytest
import torch

from ligature.model import GPT, GPTConfig, sample_next_token


def small_model(**overrides) -> GPT:
    torch.manual_seed(0)
    defaults = dict(vocab_size=64, block_size=32, n_layer=2, n_head=4, d_model=32)
    return GPT(GPTConfig(**(defaults | overrides))).eval()


def incremental_logits(model: GPT, idx: torch.Tensor, chunks: list[int]) -> torch.Tensor:
    """Feed idx through the KV-cache in chunks of the given sizes; return all logits."""
    caches = model.make_caches(idx.size(0), idx.device)
    logits, start = [], 0
    for size in chunks:
        chunk_logits, _ = model(idx[:, start : start + size], caches=caches)
        logits.append(chunk_logits)
        start += size
    return torch.cat(logits, dim=1)


@pytest.mark.parametrize("pos_encoding", ["rope", "learned"])
@pytest.mark.parametrize("use_sdpa", [False, True])
def test_greedy_with_cache_matches_without(pos_encoding: str, use_sdpa: bool) -> None:
    model = small_model(pos_encoding=pos_encoding, use_sdpa=use_sdpa)
    prompt = torch.randint(0, 64, (2, 3))
    cached = model.generate(prompt, max_new_tokens=20, temperature=0, use_cache=True)
    uncached = model.generate(prompt, max_new_tokens=20, temperature=0, use_cache=False)
    assert cached.shape == (2, 23)
    assert torch.equal(cached, uncached)


@pytest.mark.parametrize("pos_encoding", ["rope", "learned"])
def test_cached_logits_match_full_forward_on_both_attention_paths(pos_encoding: str) -> None:
    idx = torch.randint(0, 64, (2, 16))
    # A prompt of 5, single tokens, then a chunk of 4 on top of a non-empty cache.
    chunks = [5, 1, 1, 4, 1, 1, 1, 1, 1]
    results = {}
    for use_sdpa in (False, True):
        model = small_model(pos_encoding=pos_encoding, use_sdpa=use_sdpa)
        full, _ = model(idx)
        results[use_sdpa] = incremental_logits(model, idx, chunks)
        torch.testing.assert_close(results[use_sdpa], full, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(results[True], results[False], rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("use_cache", [True, False])
def test_generation_stops_at_block_size(use_cache: bool) -> None:
    model = small_model()
    prompt = torch.randint(0, 64, (1, 3))
    out = model.generate(prompt, max_new_tokens=1000, temperature=0, use_cache=use_cache)
    assert out.shape == (1, model.config.block_size)


def test_cache_never_grows_past_block_size() -> None:
    model = small_model()
    block_size = model.config.block_size
    caches = model.make_caches(1, torch.device("cpu"))
    model(torch.randint(0, 64, (1, block_size - 1)), caches=caches)
    model(torch.randint(0, 64, (1, 1)), caches=caches)
    assert all(c.length == block_size and c.k.size(2) == block_size for c in caches)

    with pytest.raises(ValueError):
        model(torch.randint(0, 64, (1, 1)), caches=caches)
    assert all(c.length == block_size for c in caches)


def test_top_k_never_samples_outside_the_top_k() -> None:
    torch.manual_seed(0)
    logits = torch.randn(4, 50)
    allowed = torch.topk(logits, k=5).indices  # (4, 5)
    generator = torch.Generator().manual_seed(0)
    for _ in range(500):
        token = sample_next_token(logits, temperature=1.0, top_k=5, generator=generator)  # (4, 1)
        assert (allowed == token).any(dim=1).all()


def test_tiny_top_p_keeps_only_the_most_likely_token() -> None:
    torch.manual_seed(0)
    logits = torch.randn(4, 50)
    token = sample_next_token(logits, temperature=1.0, top_p=1e-6)
    assert torch.equal(token, logits.argmax(dim=-1, keepdim=True))


def test_seed_makes_sampling_reproducible() -> None:
    model = small_model()
    prompt = torch.randint(0, 64, (2, 3))
    first = model.generate(prompt, max_new_tokens=10, temperature=1.0, top_k=10, seed=7)
    second = model.generate(prompt, max_new_tokens=10, temperature=1.0, top_k=10, seed=7)
    assert torch.equal(first, second)
