from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        std = math.sqrt(2.0 / (in_features + out_features))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: Tensor) -> Tensor:
        return x @ self.weight.T


class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(self, token_ids: Tensor) -> Tensor:
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.d_model = d_model
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: Tensor) -> Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        out = (x / rms) * self.weight
        return out.to(in_dtype)


def silu(x: Tensor) -> Tensor:
    return x * torch.sigmoid(x)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(silu(self.w1(x)) * self.w3(x))


def softmax(x: Tensor, dim: int) -> Tensor:
    x_max = x.amax(dim=dim, keepdim=True)
    e = torch.exp(x - x_max)
    return e / e.sum(dim=dim, keepdim=True)


def scaled_dot_product_attention(
    Q: Tensor, K: Tensor, V: Tensor, mask: Optional[Tensor] = None
) -> Tensor:
    d_k = Q.shape[-1]
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    attn = softmax(scores, dim=-1)
    return attn @ V


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        inv_freq = 1.0 / (theta ** (torch.arange(0, d_k, 2, device=device, dtype=torch.float32) / d_k))
        positions = torch.arange(max_seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)  # (max_seq_len, d_k/2)
        self.register_buffer("cos", torch.cos(freqs), persistent=False)
        self.register_buffer("sin", torch.sin(freqs), persistent=False)

    def forward(self, x: Tensor, token_positions: Tensor) -> Tensor:
        # x: (..., seq, d_k); token_positions: (..., seq)
        cos = self.cos[token_positions]  # (..., seq, d_k/2)
        sin = self.sin[token_positions]
        # broadcast shape: insert a head dim if needed
        while cos.dim() < x.dim():
            cos = cos.unsqueeze(-3)
            sin = sin.unsqueeze(-3)
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        rotated_1 = x1 * cos - x2 * sin
        rotated_2 = x1 * sin + x2 * cos
        out = torch.empty_like(x)
        out[..., 0::2] = rotated_1
        out[..., 1::2] = rotated_2
        return out


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        rope: Optional[RotaryPositionalEmbedding] = None,
        device=None,
        dtype=None,
    ):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.rope = rope

    def forward(self, x: Tensor, token_positions: Optional[Tensor] = None) -> Tensor:
        *batch, seq, _ = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        # reshape to (..., num_heads, seq, d_head)
        q = q.reshape(*batch, seq, self.num_heads, self.d_head).transpose(-2, -3)
        k = k.reshape(*batch, seq, self.num_heads, self.d_head).transpose(-2, -3)
        v = v.reshape(*batch, seq, self.num_heads, self.d_head).transpose(-2, -3)

        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(seq, device=x.device)
                # broadcast to match batch dims
                for _ in range(len(batch)):
                    token_positions = token_positions.unsqueeze(0)
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        # causal mask (query_i can attend to key_j where j <= i)
        causal = torch.tril(torch.ones(seq, seq, device=x.device, dtype=torch.bool))
        out = scaled_dot_product_attention(q, k, v, mask=causal)
        # merge heads: (..., num_heads, seq, d_head) -> (..., seq, d_model)
        out = out.transpose(-2, -3).reshape(*batch, seq, self.d_model)
        return self.output_proj(out)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        device=None,
        dtype=None,
    ):
        super().__init__()
        d_head = d_model // num_heads
        self.rope = RotaryPositionalEmbedding(theta, d_head, max_seq_len, device=device)
        self.attn = MultiHeadSelfAttention(d_model, num_heads, rope=self.rope, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)

    def forward(self, x: Tensor, token_positions: Optional[Tensor] = None) -> Tensor:
        x = x + self.attn(self.ln1(x), token_positions=token_positions)
        x = x + self.ffn(self.ln2(x))
        return x


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(d_model, num_heads, d_ff, context_length, rope_theta, device=device, dtype=dtype)
                for _ in range(num_layers)
            ]
        )
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, token_ids: Tensor) -> Tensor:
        x = self.token_embeddings(token_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_final(x)
        return self.lm_head(x)
