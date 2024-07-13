
from dataclasses import dataclass
import time
from typing import Tuple

from einops import rearrange, repeat
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

device = torch.device("mps")

@dataclass
class ModelConfig:
    n_layers: int = 12
    hidden_dim: int = 3072
    hidden_size: int = 768

    n_heads: int = 12
    n_kv_heads: int = 3
    head_dim: int = 256

    vocab_size: int = 50257
    norm_eps: float = 1e-05
    rope_theta: float = 10000.0

class RoPE(nn.Module):
    """
    rotary position embeddings

    adapted from lucidrains/rotary-embedding-torch
    https://github.com/lucidrains/rotary-embedding-torch/tree/main
    """

    def __init__(self, dim: int, theta: float = 10000.0):
        super().__init__()

        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))

        self.register_buffer("cache", None, persistent=False)
        self.freqs = nn.Parameter(freqs, requires_grad=False)

    @staticmethod
    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        x = rearrange(x, "... (d r) -> ... d r", r=2)
        x1, x2 = x.unbind(dim=-1)
        x = torch.stack((-x2, x1), dim=-1)
        return rearrange(x, "... d r -> ... (d r)")

    @torch.no_grad()
    def forward(self, x: torch.Tensor, seq_dim: int = -2, offset: int = 0) -> torch.Tensor:
        seq_len = x.shape[seq_dim]
        seq = torch.arange(seq_len, device=x.device, dtype=x.dtype) + offset

        if seq_len is not None and self.cache is not None \
            and offset + seq_len <= self.cache.shape[0]:
            freqs = self.cache[offset:(offset + seq_len)].detach()
        else:
            freqs = self.freqs
            freqs = torch.einsum("..., f -> ... f", seq.type(freqs.dtype), freqs)
            freqs = repeat(freqs, "... n -> ... (n r)", r=2)
            if seq_len is not None:
                self.cache = freqs.detach()

        if x.ndim == 3:
            freqs = freqs[-seq_len:]
        start_index = 0
        end_index = freqs.shape[-1]
        xl, x, xr = x[..., :start_index], x[..., start_index:end_index], x[..., end_index:]
        x = (x * freqs.cos()) + (RoPE.rotate_half(x) * freqs.sin())
        out = torch.cat((xl, x, xr), dim=-1)
        return out.type(x.dtype)

class CausalAttention(nn.Module):
    """
    causal self-attention

    some comment convensions:
        B: (b)atch size
        L: sequence (l)ength
        D: embedding (d)imensionality
    """

    def __init__(self, config: ModelConfig):
        super().__init__()

        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.scale = config.head_dim ** -0.5

        self.wq = nn.Linear(config.hidden_size, config.head_dim * config.n_heads, bias=False)
        self.wk = nn.Linear(config.hidden_size, config.head_dim * config.n_kv_heads, bias=False)
        self.wv = nn.Linear(config.hidden_size, config.head_dim * config.n_kv_heads, bias=False)
        self.wo = nn.Linear(config.n_heads * config.head_dim, config.hidden_size, bias=False)
        self.rope = RoPE(config.head_dim, theta=config.rope_theta)

    def forward(self, x: torch.Tensor, cache: Tuple[torch.Tensor, torch.Tensor] | None = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # B, L, D = x.shape

        q, k, v = self.wq(x), self.wk(x), self.wv(x)
        q = rearrange(q, "b l (h d) -> b l h d", h=self.n_heads)
        k = rearrange(k, "b l (h d) -> b l h d", h=self.n_heads)
        v = rearrange(v, "b l (h d) -> b l h d", h=self.n_heads)

        if cache is not None:
            k_cache, v_cache = cache
            q = self.rope(q, offset=k_cache.shape[2])
            k = torch.cat([self.rope(k, offset=k_cache.shape[2]), k], dim=2)
            v = torch.cat([v_cache, v], dim=2)
        else:
            q = self.rope(q)
            k = self.rope(k)

        attn = torch.einsum("blhd,bldh->blhh", [v * self.scale, k])
        attn = torch.softmax(attn, dim=-1)
        out = torch.einsum("blhh,blhd->blhd", [attn, v])
        out = rearrange("b h l d -> b h (l d)")
        return self.wo(out), (k, v)

class FeedForward(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        self.w1 = nn.Linear(config.hidden_size, config.hidden_dim, bias=False)
        self.w2 = nn.Linear(config.hidden_dim, config.hidden_size, bias=False)
        self.w3 = nn.Linear(config.hidden_size, config.hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class Block(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        self.attn = CausalAttention(config)
        self.ff = FeedForward(config)
        self.attn_norm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps)
        self.ff_norm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps)

    def forward(self, x: torch.Tensor, cache: Tuple[torch.Tensor, torch.Tensor] | None = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        r, cache = self.attn(self.attn_norm(x), cache=cache)
        h = x + r
        r = self.ff(self.ff_norm(h))
        out = h + r
        return out, cache

class Arthur(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        self.embedder = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [Block(config) for _ in range(config.n_layers)]
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps)
        self.final = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedder(x)
        for l in self.layers:
            x, _ = l(x)
        x = self.norm(x)
        return self.final(x)

if __name__ == "__main__":
    model = Arthur(ModelConfig())
    print(sum(p.numel() for p in model.parameters()))
    #rope = RoPE(dim=4)
    #x = torch.tensor([[1, 2, 3, 4], [4, 5, 6, 7], [7, 8, 9, 10]], dtype=torch.float)
    #x = rope(x)
    #print(x)
