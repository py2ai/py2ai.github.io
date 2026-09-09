---
layout: post
title: "Understanding Multihead Attention in LLaMA: From First Principles to RoPE, GQA, and KV Cache"
description: "Multihead attention is the mechanism that lets a transformer look at every other token in a sequence when producing a representation for the current token. LLaMA - the open-weight model family from Meta - keeps the classic multihead attention skeleton but makes three specific changes that matter for inference speed and quality: Rotary Position Embeddings (RoPE) instead of additive position vectors, Grouped-Query Attention (GQA) to cut KV cache memory, and a KV cache that lets autoregressive decoding reuse past keys and values. This post walks through the whole stack: how a single attention head computes scaled dot-product attention, how multihead attention splits Q, K, V into parallel heads and recombines them, what LLaMA changes under the hood, and how the attention block fits inside a full LLaMA decoder layer alongside the SwiGLU feed-forward network and residual connections."
date: 2026-09-08
header-img: "img/post-bg.jpg"
permalink: /Understanding-Multihead-Attention-in-LLaMA/
featured-img: ai-coding-frameworks/ai-coding-frameworks
tags:
  - LLaMA
  - Multihead Attention
  - Transformers
  - RoPE
  - GQA
  - KV Cache
  - Deep Learning
  - NLP
author: PyShine
---

## Why Attention

A transformer encoder or decoder builds a contextual representation for every token in a sequence. A token's representation should depend on the other tokens around it - not just on a fixed word embedding. Attention is the operation that performs this contextual mixing: for each query token, it computes a weighted sum of value representations of all tokens, with weights derived from how strongly the query matches each key. Multihead attention runs this in parallel across multiple heads, each with its own notion of similarity, then recombines the results.

## Scaled Dot-Product Attention (One Head)

A single attention head takes three matrices derived from the input: a query matrix `Q`, a key matrix `K`, and a value matrix `V`, each of shape `[seq_len, d_k]` (the head dimension). The output is a weighted sum of the values, where the weights come from how strongly each query matches each key.

![Scaled dot-product attention per head](/assets/img/diagrams/llama-multihead-attention/mha-scaled-dot-product.svg)

The computation in one head:

1. **Scores** - `Q . K^T` gives a `[seq_len, seq_len]` matrix. Cell `[i, j]` measures how much query `i` attends to key `j`.
2. **Scale** - divide by `sqrt(d_k)`. This keeps the dot products from growing large as `d_k` grows, which would push softmax into regions with vanishing gradients.
3. **Causal mask** - set the upper triangle to `-inf` so each query can only attend to past and current positions. This is what makes a decoder autoregressive: at position `i`, the model cannot peek at future tokens.
4. **Softmax** - normalise each row so the attention weights sum to 1.
5. **Weighted sum** - multiply the attention weights by `V` to get the head's output `[seq_len, d_v]`.

In code:

```python
def attention(q, k, v, d_k, mask=None):
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    return weights @ v
```

## Multihead Attention

A single head can only learn one kind of relationship. Multihead attention projects the input into `n_heads` separate `(Q, K, V)` triples, runs attention in parallel on each, concatenates the results, and projects back to `d_model` with a final output projection.

![Multihead attention overview](/assets/img/diagrams/llama-multihead-attention/mha-overview.svg)

The dimensions: if `d_model` is the model dimension (e.g. 4096 in LLaMA-7B) and `n_heads` is the number of heads, then the head dimension is `d_k = d_model / n_heads`. Each head gets its own slice of the projected Q, K, and V. After all heads produce their outputs, concatenation gives a `[seq_len, n_heads * d_k] = [seq_len, d_model]` tensor, which the output projection `W_O` mixes back together.

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        b, s, _ = x.shape
        q = self.w_q(x).view(b, s, self.n_heads, self.d_k).transpose(1, 2)
        k = self.w_k(x).view(b, s, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(x).view(b, s, self.n_heads, self.d_k).transpose(1, 2)
        out = attention(q, k, v, self.d_k, mask=causal_mask(s))
        out = out.transpose(1, 2).contiguous().view(b, s, -1)
        return self.w_o(out)
```

## What LLaMA Changes

LLaMA keeps the multihead skeleton but makes three specific changes.

![LLaMA-specific modifications to multihead attention](/assets/img/diagrams/llama-multihead-attention/mha-llama-modifications.svg)

### Rotary Position Embeddings (RoPE)

Classic transformers add a learned position vector to each token embedding. The trouble is that this gives an absolute notion of position rather than a relative one, and generalisation to longer sequences suffers. LLaMA replaces additive position vectors with Rotary Position Embeddings, which rotate pairs of dimensions in Q and K based on their position. The rotation angles are chosen so that the dot product `Q . K^T` depends only on the *relative* offset between query and key positions, not their absolute positions. This means the model can attend based on relative distance, which generalises better to sequence lengths unseen during training.

RoPE is applied to Q and K (not V). The rotation is applied per head, in the head dimension, by pairing adjacent dimensions and rotating each pair by an angle proportional to the position index.

### Grouped-Query Attention (GQA)

In standard multihead attention, each query head has its own key and value head. During autoregressive decoding, this means the KV cache (the cached past K and V tensors) stores one entry per head - and that memory grows linearly with both sequence length and `n_heads`. LLaMA-2 (70B) and LLaMA-3 use Grouped-Query Attention: instead of `n_heads` key and value heads, there are `n_kv_heads` where `n_kv_heads < n_heads`. Each key and value head is shared by several query heads. This cuts the KV cache size and the bandwidth to read it on every token step, which is the main bottleneck in long-context inference. Multi-Query Attention is the extreme case where `n_kv_heads = 1`; LLaMA uses intermediate values (e.g. 8 KV heads for 64 query heads) which preserves most of the quality while still shrinking the cache.

### KV Cache

During autoregressive decoding, each new token only appends one new row to Q, K, and V. The keys and values for all past positions do not change, so they are cached and reused on every step. The KV cache holds `[past_seq_len, n_kv_heads, d_k]` for K and V. On each forward pass, the model computes K and V only for the new token, appends them to the cache, and runs attention against the full cached K and V. This turns the per-step complexity from `O(seq_len^2)` (recomputing everything) into `O(seq_len)` (only the new query against the cached keys). The downside is the cache grows with sequence length, which is exactly the problem GQA helps with.

## Full LLaMA Decoder Layer

The attention block is one of two sublayers in each LLaMA decoder layer. The other is a SwiGLU feed-forward network. Both are wrapped with residual connections and pre-norm RMSNorm.

![LLaMA decoder layer with attention and SwiGLU FFN](/assets/img/diagrams/llama-multihead-attention/mha-decoder-layer.svg)

A LLaMA decoder layer:

1. **Pre-norm** - `RMSNorm` normalises the input. RMSNorm is a simpler variant of LayerNorm: it normalises by the root-mean-square of the activations and applies a learnable scale, with no mean subtraction and no bias.
2. **Multihead attention** - RoPE, GQA, KV cache, causal mask, as described above.
3. **Residual** - add the attention output back to the layer input.
4. **Pre-norm** - another RMSNorm before the feed-forward sublayer.
5. **SwiGLU feed-forward** - `W_gate` and `W_up` both project from `d_model` to `d_ff` (typically `d_ff = 2/3 * 4 * d_model` in LLaMA), the gate branch goes through SiLU (swish), the two are multiplied elementwise, and `W_down` projects back to `d_model`. SwiGLU replaces the ReLU activation used in the original transformer FFN.
6. **Residual** - add the FFN output back to the post-attention residual.

The output of this layer feeds into the next, and so on for `n_layers` (32 in LLaMA-7B, 80 in LLaMA-2 70B, and more in later models).

## Why These Choices Matter

The three LLaMA-specific changes all bend the same dial: inference efficiency without giving up quality.

- **RoPE** drops the position embedding table (no learned vector per position) and gives relative attention that extrapolates to longer sequences.
- **GQA** cuts the KV cache size and the memory bandwidth spent re-reading it on every token step, which is the dominant cost in long-context decoding.
- **KV cache** reuses past K and V instead of recomputing them, turning per-step cost from quadratic to linear in context length.

Together with RMSNorm (cheaper than LayerNorm), SwiGLU (better than ReLU FFN at equal parameter count), and the absence of bias terms, these are the changes that distinguish a LLaMA decoder from the original Transformer. The multihead attention skeleton - Q, K, V projections, parallel heads, scaled dot-product, causal mask, softmax, weighted sum, concat, output projection - is unchanged. What changed is how positions are encoded, how KV heads are shared, and how past computations are cached.

## Putting It Together

Multihead attention in LLaMA is the same multihead attention from "Attention Is All You Need", with three engineering choices layered on top. The single-head computation stays `softmax((Q . K^T) / sqrt(d_k)) . V` with a causal mask. The multihead split into `n_heads` parallel heads and the concat-plus-output-projection recombination are unchanged. What is new is RoPE on Q and K (rotary position embeddings instead of additive ones), GQA (fewer KV heads than query heads, shared across groups), and the KV cache (past K and V kept across decoding steps). The attention block sits inside a LLaMA decoder layer between two RMSNorms and a SwiGLU feed-forward, with residual connections around each sublayer. The result is a model that runs faster on long contexts, generalises better to unseen sequence lengths, and reuses computation across decoding steps without sacrificing the core attention mechanism.
