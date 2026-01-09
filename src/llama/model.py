"""
LLaMA-like Transformer model implementation from scratch.

This module implements a minimal, self-contained LLaMA-style transformer
with the following components:
- RMSNorm: Root Mean Square Layer Normalization
- RoPE: Rotary Position Embeddings
- Multi-head attention with causal masking
- SwiGLU feed-forward network
- Full transformer with pre-normalization

Based on the LLaMA architecture but simplified for clarity and minimal dependencies.
No fairscale or model parallelism - just pure PyTorch.

Reference: /home/NagaiYoru/LLM_Tutorial/llama3/llama/model.py

Usage:
    from src.llama.model import ModelArgs, Transformer

    args = ModelArgs(
        vocab_size=100,
        dim=128,
        n_layers=2,
        n_heads=4,
        n_kv_heads=4,
        max_seq_len=256,
    )
    model = Transformer(args)
    logits = model(tokens)
"""

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class ModelArgs:
    """
    Model configuration arguments.

    For a "Hello World" demo, use small values:
        dim=128, n_layers=2, n_heads=4, max_seq_len=256
    """

    # Model architecture
    dim: int = 128  # Embedding dimension
    n_layers: int = 2  # Number of transformer layers
    n_heads: int = 4  # Number of attention heads
    n_kv_heads: Optional[int] = None  # Number of key-value heads (for GQA)
    vocab_size: int = -1  # Vocabulary size (set at runtime)
    hidden_dim: int = 256  # Feed-forward hidden dimension

    # Normalization and RoPE
    norm_eps: float = 1e-5  # RMSNorm epsilon
    rope_theta: float = 10000.0  # RoPE theta parameter

    # Sequence length
    max_seq_len: int = 256  # Maximum sequence length

    # Other
    dropout: float = 0.0  # Dropout rate


class RMSNorm(torch.nn.Module):
    """
    Root Mean Square Layer Normalization.

    RMSNorm is a simplified version of Layer Norm that removes the mean centering.
    It normalizes by the root mean square of activations.

    Formula: output = (x / sqrt(mean(x^2) + eps)) * weight
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        """
        Initialize RMSNorm.

        Args:
            dim: Dimension to normalize
            eps: Small constant for numerical stability
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute RMS normalization.

        Args:
            x: Input tensor of shape (..., dim)

        Returns:
            Normalized tensor
        """
        # Compute RMS: sqrt(mean(x^2) + eps)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor

        Returns:
            Normalized and scaled output
        """
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    """
    Precompute frequency tensor for rotary position embeddings.

    This computes the complex exponential frequencies that will be used
    to rotate queries and keys in the attention mechanism.

    Args:
        dim: Dimension of the attention head (must be even)
        end: Maximum sequence length
        theta: Base frequency for RoPE

    Returns:
        Complex tensor of shape (end, dim // 2)
    """
    # Compute frequencies: 1 / (theta^(2i/d)) for i = 0, ..., dim/2
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))

    # Create position indices: 0, 1, 2, ..., end-1
    t = torch.arange(end, dtype=torch.float32)

    # Outer product: freqs[t, i] = t * freqs[i]
    freqs = torch.outer(t, freqs)

    # Convert to complex numbers: exp(i * freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)

    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Reshape frequency tensor for broadcasting with input tensor.

    Args:
        freqs_cis: Frequency tensor of shape (seq_len, dim)
        x: Input tensor of shape (batch, seq_len, n_heads, head_dim)

    Returns:
        Reshaped frequency tensor for broadcasting
    """
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])

    # Shape: (1, seq_len, 1, dim) for broadcasting with (batch, seq_len, n_heads, head_dim)
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary position embeddings to queries and keys.

    RoPE rotates the queries and keys by their position, encoding
    relative position information directly into the attention mechanism.

    Args:
        xq: Query tensor of shape (batch, seq_len, n_heads, head_dim)
        xk: Key tensor of shape (batch, seq_len, n_kv_heads, head_dim)
        freqs_cis: Precomputed frequency tensor

    Returns:
        Tuple of (rotated_queries, rotated_keys)
    """
    # Reshape to complex numbers: (batch, seq_len, n_heads, head_dim/2)
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

    # Reshape freqs for broadcasting
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)

    # Apply rotation: multiply by complex exponential
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(-2)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(-2)

    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Repeat key-value heads to match query heads (for grouped-query attention).

    When n_kv_heads < n_heads, we need to repeat the KV heads to match
    the number of query heads for the attention computation.

    Args:
        x: KV tensor of shape (batch, seq_len, n_kv_heads, head_dim)
        n_rep: Number of times to repeat each head

    Returns:
        Repeated tensor of shape (batch, seq_len, n_kv_heads * n_rep, head_dim)
    """
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x

    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    """
    Multi-head attention with RoPE and causal masking.

    This implements the core attention mechanism of LLaMA:
    - Linear projections for Q, K, V, and output
    - Rotary position embeddings
    - Causal masking (tokens can only attend to previous tokens)
    - Scaled dot-product attention
    """

    def __init__(self, args: ModelArgs):
        """
        Initialize attention layer.

        Args:
            args: Model configuration
        """
        super().__init__()

        # Number of KV heads (default: same as number of Q heads)
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        self.n_heads = args.n_heads
        self.n_rep = self.n_heads // self.n_kv_heads

        # Head dimension
        self.head_dim = args.dim // args.n_heads

        # Linear projections for Q, K, V, and output
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

        self.dropout = args.dropout

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Forward pass of attention layer.

        Args:
            x: Input tensor of shape (batch, seq_len, dim)
            freqs_cis: Precomputed RoPE frequencies
            mask: Optional causal mask

        Returns:
            Output tensor of shape (batch, seq_len, dim)
        """
        bsz, seqlen, _ = x.shape

        # Project to Q, K, V
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        # Reshape for multi-head attention
        # (batch, seq_len, n_heads, head_dim)
        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_kv_heads, self.head_dim)

        # Apply rotary position embeddings
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        # Repeat KV heads if needed (for grouped-query attention)
        keys = repeat_kv(xk, self.n_rep)
        values = repeat_kv(xv, self.n_rep)

        # Transpose for attention computation
        # (batch, n_heads, seq_len, head_dim)
        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        # Compute attention scores: Q * K^T / sqrt(d_k)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)

        # Apply causal mask if provided
        if mask is not None:
            scores = scores + mask

        # Softmax to get attention weights
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)

        # Apply dropout to attention weights
        if self.dropout > 0:
            scores = F.dropout(scores, p=self.dropout, training=self.training)

        # Compute weighted sum: scores * V
        output = torch.matmul(scores, values)

        # Transpose back and reshape
        # (batch, seq_len, n_heads * head_dim)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)

        # Output projection
        return self.wo(output)


class FeedForward(nn.Module):
    """
    Feed-forward network with SwiGLU activation.

    SwiGLU is a variant of GLU that uses SiLU (swish) activation:
        FFN(x) = down(SiLU(gate(x)) * up(x))

    Where gate, up, down are linear projections.
    """

    def __init__(self, args: ModelArgs):
        """
        Initialize feed-forward network.

        Args:
            args: Model configuration
        """
        super().__init__()

        # SwiGLU has two parallel projections (gate and up)
        self.w1 = nn.Linear(args.dim, args.hidden_dim, bias=False)  # gate
        self.w3 = nn.Linear(args.dim, args.hidden_dim, bias=False)  # up
        self.w2 = nn.Linear(args.hidden_dim, args.dim, bias=False)  # down

        self.dropout = args.dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, dim)

        Returns:
            Output tensor of shape (batch, seq_len, dim)
        """
        # SwiGLU: SiLU(w1(x)) * w3(x)
        hidden = F.silu(self.w1(x)) * self.w3(x)

        # Dropout
        if self.dropout > 0:
            hidden = F.dropout(hidden, p=self.dropout, training=self.training)

        # Output projection
        return self.w2(hidden)


class TransformerBlock(nn.Module):
    """
    Transformer decoder block with pre-normalization.

    Architecture:
        x = x + Attention(RMSNorm(x))
        x = x + FFN(RMSNorm(x))

    Pre-normalization means we apply normalization before the sub-layer,
    which helps with training stability.
    """

    def __init__(self, layer_id: int, args: ModelArgs):
        """
        Initialize transformer block.

        Args:
            layer_id: Layer index (for logging/debugging)
            args: Model configuration
        """
        super().__init__()
        self.layer_id = layer_id
        self.attention = Attention(args)
        self.feed_forward = FeedForward(args)

        # Pre-normalization
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor
            start_pos: Starting position (for KV caching, unused in this simple implementation)
            freqs_cis: RoPE frequencies
            mask: Causal mask

        Returns:
            Output tensor
        """
        # Self-attention with residual connection
        h = x + self.attention(self.attention_norm(x), freqs_cis, mask)

        # Feed-forward with residual connection
        out = h + self.feed_forward(self.ffn_norm(h))

        return out


class Transformer(nn.Module):
    """
    Full LLaMA-style transformer model.

    Components:
        - Token embeddings
        - N transformer blocks
        - Final normalization
        - Output projection to vocabulary

    Usage:
        model = Transformer(args)
        logits = model(input_ids)  # (batch, seq_len, vocab_size)
    """

    def __init__(self, params: ModelArgs):
        """
        Initialize transformer.

        Args:
            params: Model configuration
        """
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size

        # Token embeddings
        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)

        # Transformer blocks
        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        # Final normalization
        self.norm = RMSNorm(params.dim, eps=params.norm_eps)

        # Output projection to vocabulary
        self.output = nn.Linear(params.dim, params.vocab_size, bias=False)

        # Precompute RoPE frequencies
        self.freqs_cis = precompute_freqs_cis(
            params.dim // params.n_heads,
            params.max_seq_len * 2,  # Extra room for caching
            params.rope_theta,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        start_pos: int = 0,
    ) -> torch.Tensor:
        """
        Forward pass of the transformer.

        Args:
            input_ids: Token IDs of shape (batch, seq_len)
            start_pos: Starting position (for generation with caching)

        Returns:
            Logits of shape (batch, seq_len, vocab_size)
        """
        _bsz, seqlen = input_ids.shape

        # Token embeddings
        h = self.tok_embeddings(input_ids)

        # Get RoPE frequencies for this sequence
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

        # Create causal mask
        mask = None
        if seqlen > 1:
            # Mask where future positions are -inf
            mask = torch.full((seqlen, seqlen), float("-inf"), device=input_ids.device)
            mask = torch.triu(mask, diagonal=1)

            # Handle KV cache: only mask new tokens relative to cache
            if start_pos > 0:
                mask = torch.hstack(
                    [torch.zeros((seqlen, start_pos), device=input_ids.device), mask]
                )

        # Pass through transformer blocks
        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)

        # Final normalization
        h = self.norm(h)

        # Output projection
        output = self.output(h).float()

        return output
