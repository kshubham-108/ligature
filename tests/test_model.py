import math

import pytest
import torch

from ligature.model import GPT, GPTConfig, RotaryEmbedding


def small_config(**overrides) -> GPTConfig:
    defaults = dict(vocab_size=64, block_size=16, n_layer=2, n_head=4, d_model=32)
    return GPTConfig(**(defaults | overrides))


@pytest.mark.parametrize("pos_encoding", ["rope", "learned"])
def test_output_shapes(pos_encoding: str) -> None:
    torch.manual_seed(0)
    config = small_config(pos_encoding=pos_encoding)
    model = GPT(config)
    idx = torch.randint(0, config.vocab_size, (2, 8))

    logits, loss = model(idx)
    assert logits.shape == (2, 8, config.vocab_size)
    assert loss is None

    _, loss = model(idx, targets=idx)
    assert loss.shape == ()


def test_initial_loss_is_close_to_uniform() -> None:
    # With weights drawn from N(0, 0.02) the logits are all close to zero, so the predicted
    # distribution is nearly uniform over the vocabulary and the cross-entropy is about
    # -ln(1 / vocab_size) = ln(vocab_size). A much higher value would point to a bad init.
    torch.manual_seed(0)
    config = small_config(vocab_size=1024)
    model = GPT(config)
    idx = torch.randint(0, config.vocab_size, (8, 16))
    targets = torch.randint(0, config.vocab_size, (8, 16))
    _, loss = model(idx, targets)
    assert abs(loss.item() - math.log(config.vocab_size)) < 0.1


@pytest.mark.parametrize("pos_encoding", ["rope", "learned"])
@pytest.mark.parametrize("use_sdpa", [False, True])
def test_changing_a_token_leaves_earlier_logits_unchanged(
    pos_encoding: str, use_sdpa: bool
) -> None:
    torch.manual_seed(0)
    config = small_config(pos_encoding=pos_encoding, use_sdpa=use_sdpa)
    model = GPT(config).eval()
    T = 8
    idx = torch.randint(0, config.vocab_size, (2, T))
    logits, _ = model(idx)

    for t in range(T):
        changed = idx.clone()
        changed[:, t] = (changed[:, t] + 1) % config.vocab_size
        new_logits, _ = model(changed)
        torch.testing.assert_close(new_logits[:, :t], logits[:, :t], rtol=0, atol=1e-6)
        # The changed position itself must move, or the check above proves nothing.
        assert not torch.allclose(new_logits[:, t], logits[:, t])


@pytest.mark.parametrize("pos_encoding", ["rope", "learned"])
def test_explicit_attention_matches_sdpa(pos_encoding: str) -> None:
    torch.manual_seed(0)
    config = small_config(pos_encoding=pos_encoding)
    model = GPT(config).eval()
    idx = torch.randint(0, config.vocab_size, (2, config.block_size))

    explicit, _ = model(idx)
    for block in model.blocks:
        block.attn.use_sdpa = True
    sdpa, _ = model(idx)
    torch.testing.assert_close(sdpa, explicit, rtol=1e-5, atol=1e-5)


def test_rope_score_depends_only_on_relative_offset() -> None:
    torch.manual_seed(0)
    head_dim = 16
    rope = RotaryEmbedding(head_dim, block_size=64)
    q = torch.randn(1, 1, 1, head_dim)  # (B, n_head, T, head_dim) with a single position
    k = torch.randn(1, 1, 1, head_dim)

    def score(q_pos: int, k_pos: int) -> float:
        return (rope(q, start_pos=q_pos) * rope(k, start_pos=k_pos)).sum().item()

    offset_3 = [score(3, 0), score(10, 7), score(50, 47)]
    assert offset_3 == pytest.approx([offset_3[0]] * 3, abs=1e-5)
    assert score(10, 2) != pytest.approx(offset_3[0], abs=1e-3)


def test_lm_head_shares_storage_with_token_embedding() -> None:
    model = GPT(small_config())
    assert model.lm_head.weight is model.tok_emb.weight
    assert model.lm_head.weight.data_ptr() == model.tok_emb.weight.data_ptr()


@pytest.mark.parametrize("bias", [False, True])
def test_num_params_matches_closed_form(bias: bool) -> None:
    config = small_config(bias=bias)
    d, n_layer = config.d_model, config.n_layer
    # Weights per block: qkv 3d^2 + attention proj d^2 + MLP fc 4d^2 + MLP proj 4d^2 = 12d^2,
    # plus two LayerNorm scales (2d). Biases add qkv 3d + proj d + fc 4d + proj d + two LayerNorm
    # shifts 2d = 11d. The final LayerNorm adds d (2d with bias). Embeddings are excluded.
    per_block = 12 * d**2 + 2 * d + (11 * d if bias else 0)
    expected = n_layer * per_block + (2 * d if bias else d)
    assert GPT(config).num_params() == expected


def test_tiny_model_overfits_one_batch() -> None:
    torch.manual_seed(0)
    config = small_config()
    model = GPT(config)
    idx = torch.randint(0, config.vocab_size, (4, config.block_size))
    targets = torch.randint(0, config.vocab_size, (4, config.block_size))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

    for _ in range(200):
        _, loss = model(idx, targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    assert loss.item() < 0.05
