"""GPT-style decoder-only transformer, with every component written by hand."""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 4096
    block_size: int = 256
    n_layer: int = 6
    n_head: int = 6
    d_model: int = 384
    dropout: float = 0.0
    pos_encoding: str = "rope"  # "rope" or "learned"
    bias: bool = False
    use_sdpa: bool = False

    def __post_init__(self) -> None:
        if self.d_model % self.n_head != 0:
            raise ValueError("d_model must be divisible by n_head")
        if self.pos_encoding not in ("rope", "learned"):
            raise ValueError(f"unknown pos_encoding: {self.pos_encoding!r}")
        if self.pos_encoding == "rope" and self.head_dim % 2 != 0:
            raise ValueError("RoPE needs an even head_dim")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_head


class RotaryEmbedding(nn.Module):
    """Rotary position embeddings (Su et al., 2021) with cos/sin precomputed up to block_size."""

    def __init__(self, head_dim: int, block_size: int, base: float = 10000.0) -> None:
        super().__init__()
        # Channel pair i rotates at frequency base^(-2i / head_dim): fast for early pairs, slow
        # for late ones. Position m rotates pair i by the angle m * freq_i.
        inv_freq = base ** (-torch.arange(0, head_dim, 2).float() / head_dim)  # (head_dim/2,)
        angles = torch.outer(torch.arange(block_size).float(), inv_freq)  # (block_size, head_dim/2)
        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)

    def forward(self, x: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        """Rotate x (B, n_head, T, head_dim) as if its first position were start_pos."""
        T = x.size(-2)
        cos = self.cos[start_pos : start_pos + T]  # (T, head_dim/2)
        sin = self.sin[start_pos : start_pos + T]
        x1, x2 = x[..., 0::2], x[..., 1::2]  # (B, n_head, T, head_dim/2) each
        # 2D rotation of each pair (x1, x2); re-interleave so channel order is unchanged.
        rotated = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)
        return rotated.flatten(-2).type_as(x)  # (B, n_head, T, head_dim)


class KVCache:
    """Keys and values of one attention layer, pre-allocated up to block_size positions."""

    def __init__(
        self,
        batch_size: int,
        n_head: int,
        block_size: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        shape = (batch_size, n_head, block_size, head_dim)
        self.k = torch.zeros(shape, device=device, dtype=dtype)
        self.v = torch.zeros(shape, device=device, dtype=dtype)
        self.length = 0  # positions filled so far, which is also the next token's position

    def append(self, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Store k, v (B, n_head, T, head_dim); return all keys and values cached so far."""
        end = self.length + k.size(2)
        if end > self.k.size(2):
            raise ValueError(f"KV-cache full: {end} positions exceed block_size {self.k.size(2)}")
        self.k[:, :, self.length : end] = k
        self.v[:, :, self.length : end] = v
        self.length = end
        return self.k[:, :, :end], self.v[:, :, :end]  # (B, n_head, end, head_dim) views


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with an explicit path and an SDPA path."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.n_head = config.n_head
        self.head_dim = config.head_dim
        self.dropout = config.dropout
        self.use_sdpa = config.use_sdpa
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=config.bias)
        self.proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.rope = (
            RotaryEmbedding(config.head_dim, config.block_size)
            if config.pos_encoding == "rope"
            else None
        )
        mask = torch.tril(torch.ones(config.block_size, config.block_size, dtype=torch.bool))
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x: torch.Tensor, cache: KVCache | None = None) -> torch.Tensor:
        B, T, C = x.shape
        # Absolute position of the first token in x: the number of positions already cached.
        start = cache.length if cache is not None else 0
        q, k, v = self.qkv(x).split(C, dim=2)  # (B, T, C) each
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, n_head, T, head_dim)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        if self.rope is not None:
            q, k = self.rope(q, start), self.rope(k, start)
        if cache is not None:
            k, v = cache.append(k, v)  # (B, n_head, start + T, head_dim)

        if self.use_sdpa:
            y = self._sdpa_attention(q, k, v, start)
        else:
            y = self._explicit_attention(q, k, v, start)

        y = y.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, n_head, head_dim) -> (B, T, C)
        return self.resid_dropout(self.proj(y))

    def _sdpa_attention(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, start: int
    ) -> torch.Tensor:
        T = q.size(-2)
        dropout_p = self.dropout if self.training else 0.0
        if start == 0:
            return F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p, is_causal=True)
        # With a cache, is_causal=True would align its mask to the top-left of the (T, start + T)
        # score matrix and hide most of the cache. One new token may see every cached position,
        # so it needs no mask; a longer chunk needs its rows of the causal mask.
        mask = None if T == 1 else self.mask[start : start + T, : start + T]
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=dropout_p)

    def _explicit_attention(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, start: int
    ) -> torch.Tensor:
        T, T_k = q.size(-2), k.size(-2)  # T_k = start + T keys, cached ones included
        # Dividing by sqrt(head_dim) keeps score variance near 1 when q and k have unit variance,
        # so the softmax does not saturate and its gradients do not vanish.
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, n_head, T, T_k)
        # Query i sits at absolute position start + i, so it uses that row of the causal mask.
        att = att.masked_fill(~self.mask[start : start + T, :T_k], float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        return att @ v  # (B, n_head, T, head_dim)


class MLP(nn.Module):
    """Position-wise feed-forward network with a 4x hidden layer."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.fc = nn.Linear(config.d_model, 4 * config.d_model, bias=config.bias)
        self.gelu = nn.GELU()
        self.proj = nn.Linear(4 * config.d_model, config.d_model, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.proj(self.gelu(self.fc(x))))


class Block(nn.Module):
    """Pre-LayerNorm transformer block: attention then MLP, each added to the residual stream."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.d_model, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.d_model, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor, cache: KVCache | None = None) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x), cache)
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    """Decoder-only language model with a weight-tied LM head."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = (
            nn.Embedding(config.block_size, config.d_model)
            if config.pos_encoding == "learned"
            else None
        )
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.d_model, bias=config.bias)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        # Weight tying (Press and Wolf, 2017): reading a token in and predicting it out share one
        # matrix, which saves vocab_size * d_model parameters and tends to help small models.
        self.lm_head.weight = self.tok_emb.weight

        self.apply(self._init_weights)
        # Each block adds two outputs (attention and MLP) to the residual stream, so its variance
        # grows with every one of the 2 * n_layer additions. Shrinking the std of the layers that
        # write into the stream by 1/sqrt(2 * n_layer) keeps the stream's variance at
        # initialisation roughly independent of depth (GPT-2, Radford et al., 2019).
        residual_std = 0.02 / math.sqrt(2 * config.n_layer)
        for block in self.blocks:
            nn.init.normal_(block.attn.proj.weight, mean=0.0, std=residual_std)
            nn.init.normal_(block.mlp.proj.weight, mean=0.0, std=residual_std)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self) -> int:
        """Count parameters, excluding the token embedding (tied to the head) and pos_emb."""
        # parameters() yields the tied weight once, so it is subtracted once.
        n = sum(p.numel() for p in self.parameters())
        n -= self.tok_emb.weight.numel()
        if self.pos_emb is not None:
            n -= self.pos_emb.weight.numel()
        return n

    def make_caches(self, batch_size: int, device: torch.device) -> list[KVCache]:
        """One empty KV-cache per layer, each holding up to block_size positions."""
        c = self.config
        dtype = self.tok_emb.weight.dtype
        return [
            KVCache(batch_size, c.n_head, c.block_size, c.head_dim, device, dtype)
            for _ in range(c.n_layer)
        ]

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        caches: list[KVCache] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Return logits (B, T, vocab_size) and, given targets (B, T), the mean cross-entropy.

        With caches, idx holds only the new tokens, which continue after the cached positions.
        """
        B, T = idx.shape
        start = caches[0].length if caches is not None else 0
        if start + T > self.config.block_size:
            raise ValueError(
                f"sequence length {start + T} exceeds block_size {self.config.block_size}"
            )

        x = self.tok_emb(idx)  # (B, T, d_model)
        if self.pos_emb is not None:
            positions = torch.arange(start, start + T, device=idx.device)
            x = x + self.pos_emb(positions)  # (T, d_model) broadcasts over B
        x = self.drop(x)
        for i, block in enumerate(self.blocks):
            x = block(x, caches[i] if caches is not None else None)
        logits = self.lm_head(self.ln_f(x))  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            # Flatten to (B * T, vocab_size) and (B * T,): one classification per position.
            loss = F.cross_entropy(logits.view(B * T, -1), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        use_cache: bool = True,
        seed: int | None = None,
    ) -> torch.Tensor:
        """Extend idx (B, T) by up to max_new_tokens, stopping when it reaches block_size."""
        B, T = idx.shape
        if T > self.config.block_size:
            raise ValueError(f"prompt length {T} exceeds block_size {self.config.block_size}")
        generator = None
        if seed is not None:
            generator = torch.Generator(device=idx.device).manual_seed(seed)
        was_training = self.training
        self.eval()

        caches = self.make_caches(B, idx.device) if use_cache else None
        new_tokens = idx  # with a cache, the first step feeds the whole prompt to fill it
        for _ in range(min(max_new_tokens, self.config.block_size - T)):
            logits, _ = self(new_tokens if use_cache else idx, caches=caches)
            next_token = sample_next_token(logits[:, -1, :], temperature, top_k, top_p, generator)
            idx = torch.cat((idx, next_token), dim=1)  # (B, T + 1)
            new_tokens = next_token  # (B, 1)

        self.train(was_training)
        return idx


def sample_next_token(
    logits: torch.Tensor,
    temperature: float,
    top_k: int | None = None,
    top_p: float | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Pick one token per row of logits (B, vocab_size); temperature 0 means greedy."""
    if temperature == 0:
        return logits.argmax(dim=-1, keepdim=True)  # (B, 1)
    logits = logits / temperature
    if top_k is not None:
        kth_largest = torch.topk(logits, min(top_k, logits.size(-1))).values[:, -1:]  # (B, 1)
        logits = logits.masked_fill(logits < kth_largest, float("-inf"))
    if top_p is not None:
        # Nucleus sampling: keep the smallest set of most likely tokens whose probabilities
        # reach top_p. A token is dropped once the tokens ranked above it already reach top_p,
        # so the most likely token is always kept.
        sorted_logits, order = logits.sort(dim=-1, descending=True)
        probs = sorted_logits.softmax(dim=-1)
        mass_before = probs.cumsum(dim=-1) - probs
        sorted_logits = sorted_logits.masked_fill(mass_before >= top_p, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(-1, order, sorted_logits)
    probs = logits.softmax(dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=generator)  # (B, 1)
