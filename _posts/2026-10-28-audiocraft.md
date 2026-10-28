---
layout: post
title: "AudioCraft Source Tour: How Meta's MusicGen, AudioGen and EnCodec Fit Together"
description: "A code-level walk through Meta's AudioCraft: the EnCodec quantizer core, the LMModel over interleaved codebooks, the T5 and chroma conditioners, and the generation loop that stitches them together."
date: 2026-10-28
header-img: "img/post-bg.jpg"
permalink: /audiocraft/
featured-img: "ai-coding-frameworks/ai-coding-frameworks"
image: https://pyshine.com/assets/img/diagrams/audiocraft/facebookresearch-audiocraft-overview-architecture.svg
tags: [LLM, Python, Open Source]
categories: [AI Inference]
keywords: audiocraft, musicgen, audiogen, encodec, meta ai, music generation, audio generation, neural codec, residual vector quantization, classifier-free guidance, pytorch, python
author: "PyShine"
---

Meta's AudioCraft is the library behind MusicGen and AudioGen, and it is one of the most instructive generative-audio codebases you can read. It commits to one architectural idea: treat audio generation as language modeling over discrete tokens produced by a neural codec. Everything else — the conditioners, the pattern providers, the streaming machinery — exists to serve that single idea. If you have read our tours of minimal text-only stacks or high-throughput serving layers, this is what the same transformer story looks like when the vocabulary is learned codebook entries rather than BPE tokens.

The repository splits cleanly along that seam. The codec half lives in `audiocraft/models/encodec.py`, `audiocraft/modules/seanet.py` and `audiocraft/quantization/`, trainable on its own through the solver in `audiocraft/solvers/compression.py`. The generative half lives in `audiocraft/models/lm.py` behind the facade classes `audiocraft/models/musicgen.py` and `audiocraft/models/audiogen.py`, which share their inference skeleton in `audiocraft/models/genmodel.py`. Between the two, `audiocraft/modules/conditioners.py` turns text descriptions, melody excerpts and CLAP embeddings into tensors the transformer can consume.

This post walks that path with line-verified references from the `main` branch, anchored by two diagrams: an overview of the module graph and a detailed view including data IO and codec training.

<div style="overflow-x:auto;"><img src="/assets/img/diagrams/audiocraft/facebookresearch-audiocraft-overview-architecture.svg" alt="AudioCraft overview architecture: MusicGen facade, BaseGenModel loop, LMModel with streaming transformer and codebook patterns, conditioning stack, EnCodec with SEANet and residual vector quantization" style="min-width:900px;"></div>

<p><em>AudioCraft at a glance: a thin facade (left) drives a language model over codebook streams (center), which is steered by conditioners and grounded by the EnCodec codec (right).</em></p>

Reading the overview from left to right:

- Everything user-facing starts at the [MusicGen, L40](https://github.com/facebookresearch/audiocraft/blob/main/audiocraft/models/musicgen.py#L40) facade, whose [get_pretrained, L57](https://github.com/facebookresearch/audiocraft/blob/main/audiocraft/models/musicgen.py#L57) maps names like `facebook/musicgen-small` (300M) or `facebook/musicgen-large` (3.3B) to downloaded checkpoints.
- The facade inherits the actual inference skeleton from [BaseGenModel, L28](https://github.com/facebookresearch/audiocraft/blob/main/audiocraft/models/genmodel.py#L28), which owns both the codec and the LM and exposes `generate` for text and `generate_continuation` for audio prompts.
- Inside the LM, [LMModel, L120](https://github.com/facebookresearch/audiocraft/blob/main/audiocraft/models/lm.py#L120) wraps one embedding table and one output projection per codebook, plus a shared transformer trunk.
- Codebooks are not modeled in parallel naively: a [DelayedPatternProvider, L305](https://github.com/facebookresearch/audiocraft/blob/main/audiocraft/modules/codebooks_patterns.py#L305) interleaves the K streams into a single flat sequence so one causal pass can predict them all.
- On the conditioning side, [T5Conditioner, L422](https://github.com/facebookresearch/audiocraft/blob/main/audiocraft/modules/conditioners.py#L422) embeds the text prompt through a frozen T5 encoder, and melody variants reuse chroma features instead.
- The codec core, [EncodecModel, L125](https://github.com/facebookresearch/audiocraft/blob/main/audiocraft/models/encodec.py#L125), runs a convolutional encoder into a residual quantizer and back out through a mirrored decoder.
- That quantizer is [ResidualVectorQuantizer, L16](https://github.com/facebookresearch/audiocraft/blob/main/audiocraft/quantization/vq.py#L16) — with the default `bins=1024` vocabulary per stage — and it is the reason the LM problem is finite at all.

## The Big Picture: Tokens All the Way Down

The cleanest way into the code is `audiocraft/models/genmodel.py`. `BaseGenModel.__init__` takes a `CompressionModel` and an `LMModel`, puts both in eval mode, and derives `max_duration` from the LM's training config. The three public entry points — `generate` (L151), `generate_continuation` (L166) and `generate_unconditional` (L135) — all funnel into `_generate_tokens` at L193.

In the simple case the whole job is one call: `self.lm.generate(prompt_tokens, attributes, ...)` inside an autocast context at L227–230. Tokens come back shaped `[B, K, T]`, and `generate_audio` at L262 hands them straight to `self.compression_model.decode(gen_tokens, None)`. No sampler network, no vocoder stage — the codec decoder is the vocoder.

The interesting branch is long-form generation. AudioCraft models are trained on 30-second windows, but you can ask for minutes. `_generate_tokens` then enters a while loop at L243 that generates a window, appends the tokens, and slides the context forward by `extend_stride` seconds (18 by default, per `set_generation_params` at L96 of `musicgen.py`). Each window reconditions on the tail of the previous one, so continuity is real but soft — a design we revisit in the limits section. For audio-prompt continuation, `generate_continuation` resamples your waveform with `convert_audio` (`audiocraft/data/audio_utils.py`, L54) and encodes it with `self.compression_model.encode(prompt)` at L129 — the codec path we look at next.

## The Codec: EnCodec, SEANet and Residual Quantization

`EncodecModel` at L125 of `audiocraft/models/encodec.py` composes three pieces: `self.encoder`, `self.quantizer`, `self.decoder`. Its `forward` at L206 makes the flow explicit — encode, quantize, decode — and trims decoder padding. `preprocess` at L186 optionally renormalizes loudness, dividing the waveform by an RMS-derived scale that `postprocess` undoes after decoding. The public contract is `encode` at L223, returning `(codes, scale)` with codes shaped `[B, K, T]`, and `decode` at L240. `decode_latent` at L257 stops at the continuous quantized embedding, which is what diffusion-based refinement models like MultiBandDiffusion consume.

The convolutional muscle comes from `audiocraft/modules/seanet.py`: `SEANetEncoder` at L63 progressively downsamples the waveform with strided convolutions and residual blocks, and `SEANetDecoder` at L156 mirrors it — the same "strided conv autoencoder" family found in most neural codecs.

The quantization stack is the part most worth stealing. `ResidualVectorQuantizer` at L16 of `audiocraft/quantization/vq.py` defaults to `n_q=8` stages of `bins=1024` entries with EMA decay 0.99. Residual means stage two quantizes the error left by stage one, stage three the remainder, so effective bandwidth grows with depth. The math is explicit at L80: `bw_per_q = math.log2(self.bins) * frame_rate / 1000` — 10 bits per codebook per frame, expressed in kilobits per second. Underneath, `audiocraft/quantization/core_vq.py` holds the machinery: `EuclideanCodebook` at L87 maintains codebook vectors with exponential moving averages, optionally initialized by a small `kmeans` run at L53, and revives dead codes whose usage falls below a threshold. `VectorQuantization` at L222 and `ResidualVectorQuantization` at L351 stack that book into single- and multi-stage forms, while `audiocraft/quantization/base.py` defines the shared `QuantizedResult` container at L19 and the `BaseQuantizer` interface at L27.

One more codec detail matters for generation: `InterleaveStereoCompressionModel` at L397 of `encodec.py` builds stereo from mono by interleaving left/right codes into a taller codebook axis, so the same LM code handles stereo.

## The Language Model: One Sequence, Many Codebooks

`LMModel` at L120 of `audiocraft/models/lm.py` is a `StreamingModule` holding `n_q` parallel `ScaledEmbedding` tables (L165), one `StreamingTransformer` trunk (L168), and `n_q` output projections (L174). Its `forward` at L221 sums the per-codebook embeddings, gets condition tensors, calls `self.fuser` to merge them and produce a cross-attention source, runs the transformer, then stacks per-codebook logits into `[B, K, S, card]` at L262.

The clever part is how K codebooks fit into one causal sequence, and the answer lives in `audiocraft/modules/codebooks_patterns.py`. A `Pattern` (L22) is a layout of `(timestep, codebook)` coordinates; `CodebooksPatternProvider` (L272) is the abstract factory. MusicGen's default, `DelayedPatternProvider` at L305, shifts each codebook one step later than the previous — shown concretely in its docstring — so a single sequence step contains at most one new token per codebook and the causal LM predicts all K streams in one flat pass. `ParallelPatternProvider` (L359) removes the delay, `UnrolledPatternProvider` (L372) flattens fully, and `MusicLMPattern` (L530) is the flat fallback.

`LMModel.generate` at L421 orchestrates decoding. It asks the pattern provider for a layout over `max_gen_len` timesteps (L521), folds the prompt into the interleaved sequence via `build_pattern_sequence` (L530), then loops over sequence steps at L540. Each iteration calls `_sample_next_token` at L323, where classifier-free guidance happens: AudioCraft batches the conditional and unconditional passes into one forward by concatenating real conditions with nullified ones (L497–509) and splitting the logits, computing `uncond + (cond - uncond) * cfg_coef` at L399 — one transformer call instead of two. When `two_step_cfg` is set, it alternates two passes while swapping the streaming cache in and out (L381–387). Sampling then applies top-k (250 by default) or top-p at L407–416, with plain argmax at temperature zero. Note what is absent: no beam search anywhere in the decode path. After the loop, `revert_pattern_sequence` at L576 un-interleaves the flat sequence back into `[B, K, T]` codes for the codec.

The streaming machinery behind per-step caching lives in `audiocraft/modules/streaming.py`: `StreamingModule` at L20 tracks a state dict via `set_streaming_state` (L88) and `reset_streaming` (L68), and `StreamingSequential` at L122 propagates it through composed modules. In `audiocraft/modules/transformer.py`, `StreamingMultiheadAttention` (L138), `StreamingTransformerLayer` (L454) and `StreamingTransformer` (L577) use it for key/value caches, with optional `RotaryEmbedding` from `audiocraft/modules/rope.py` (L49) for positions.

<div style="overflow-x:auto;"><img src="/assets/img/diagrams/audiocraft/facebookresearch-audiocraft-architecture.svg" alt="AudioCraft detailed architecture: generation API, LM internals with rope and patterns, conditioners with chroma, codec with SEANet and RVQ layers, audio IO, compression solver training, builders and checkpoint loaders" style="min-width:900px;"></div>

<p><em>The detailed view adds the data path, the codec training loop with its adversarial losses, and the builders/loaders pair that turns config into running models.</em></p>

## Conditioning: Text, Melody, and the Fuser

`audiocraft/modules/conditioners.py` is a 1700-line file with a simple shape. A `ConditioningAttributes` (L78) is a bag of named attributes — a description string, maybe a waveform, maybe a joint embedding. A `ConditioningProvider` (L1469) holds one `BaseConditioner` (L345) per attribute name and dispatches each, returning `ConditionType` pairs — a `(tensor, mask)` tuple defined at L43.

The flagship conditioner is `T5Conditioner` at L422, which loads any T5-family encoder — the `MODELS` list at L434 ranges from `t5-small` to `google/flan-t5-xxl` — through Hugging Face `T5Tokenizer` and `T5EncoderModel.from_pretrained` (L475–476), projects the output to the model dimension, and masks padding. By default the T5 weights are frozen and excluded from checkpoints (L484). Melody models add `ChromaStemConditioner` at L571, extracting chromagram features from source-separated stems via `ChromaExtractor` (`audiocraft/modules/chroma.py`, L16); MusicGen exposes this through `generate_with_chroma` at L155 of `musicgen.py`.

Merging conditions into the LM input is `ConditionFuser`'s job (L1672), with five strategies at L1687: `sum`, `prepend`, `cross`, `ignore`, `input_interpolate`. Text descriptions are typically prepended as extra sequence positions and fed as cross-attention sources. Training-time robustness comes from `AttributeDropout` (L1380) and `ClassifierFreeGuidanceDropout` (L1427), the latter producing the null conditions inference reuses at sampling time. The design is deliberately model-agnostic: conditioning-prefix construction in the /transformers/ sense, with audio-flavored attribute types.

## From Install to Your First Model

Installation is a plain `pip install audiocraft` (a CUDA-enabled PyTorch build helps a lot; see our notes on /torchaudio/ for the audio-side tensors). The smallest complete program is:

```python
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write

model = MusicGen.get_pretrained('facebook/musicgen-small')
model.set_generation_params(duration=8, top_k=250, cfg_coef=3.0)
wav = model.generate(['lo-fi hip hop with a rainy city ambience'])
audio_write('out', wav[0].cpu(), model.sample_rate)
```

Under the hood, `get_pretrained` calls `load_lm_model` and `load_compression_model` from `audiocraft/models/loaders.py` (L111 and L78), which `hf_hub_download` the config and weights and rebuild the modules through the factories in `audiocraft/models/builders.py` — `get_compression_model` (L70), `get_lm_model` (L136), `get_conditioner_provider` (L178), `get_codebooks_pattern_provider` (L240). `audio_write` (L159 of `audiocraft/data/audio.py`) shells out to ffmpeg for mp3/wav output, while `audio_read` (L116) picks soundfile or ffmpeg by format. The same pair powers continuation prompts: read any clip, pass it to `generate_continuation`, and the LM completes it. If your workflow involves transcription-before-generation, /whisperx/ pairs naturally here; for lightweight on-device speech, /piper/ occupies the opposite corner of audio generation.

## Honest Limits

- **Long-form is stitchy.** Beyond the 30-second training window, `_generate_tokens` re-prompts with the last `extend_stride` seconds of audio. Transitions are usually acceptable and occasionally audible; there is no architectural memory beyond that context.
- **Decoding is sampling, not search.** There is no beam search in `LMModel.generate` — only top-k/top-p sampling with CFG, so you steer outputs through coefficients rather than search.
- **Sequential codebooks cost latency.** The delayed pattern still emits up to K tokens per step through one transformer pass; long durations add up, and batching CFG doubles effective batch size.
- **Melody conditioning has prerequisites.** `ChromaStemConditioner` expects stems, so quality depends on the source separation model upstream.
- **The framework footprint is real.** Hydra/OmegaConf configs, xformers attention backends, and multi-GPU solver code make the training side heavier than inference; the `CompressionSolver` at L27 balances reconstruction and adversarial objectives — `AdversarialLoss` at L26 and `FeatureMatchingLoss` at L201 of `audiocraft/adversarial/losses.py`, rescaled by the `Balancer` at L14 of `audiocraft/losses/balancer.py` — and expects that infrastructure.
- **Weights are separate from code.** The permissive license covers the source; check `LICENSE_weights` before shipping anything commercial.

For anyone who has toured text-only repos — NanoGPT, llm.c — AudioCraft is the refreshing variant where the tokenizer is a differentiable codec you can train, inspect, and reuse. Start at `genmodel.py`, jump to `lm.py` for the decode loop, and keep `core_vq.py` open in a second window.

<script type="application/ld+json">
{ "@context": "https://schema.org", "@type": "TechArticle", "headline": "AudioCraft Source Tour: How Meta's MusicGen, AudioGen and EnCodec Fit Together", "description": "A code-level walk through Meta's AudioCraft: the EnCodec quantizer core, the LMModel over interleaved codebooks, the T5 and chroma conditioners, and the generation loop.", "datePublished": "2026-10-28", "author": { "@type": "Person", "name": "PyShine" }, "keywords": "audiocraft, musicgen, encodec, audio generation, neural codec", "publisher": { "@type": "Organization", "name": "PyShine" } }
</script>
