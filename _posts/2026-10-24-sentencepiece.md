---
layout: post
title: "SentencePiece: A Source Tour of Google's Language-Agnostic Tokenizer"
description: "Walk through the C++ source: the processor API, Viterbi unigram segmentation, BPE merges, normalization rules, and the trainer that Llama and T5 depend on."
date: 2026-10-24
header-img: "img/post-bg.jpg"
permalink: /sentencepiece/
featured-img: "ai-coding-frameworks/ai-coding-frameworks"
image: https://pyshine.com/assets/img/diagrams/sentencepiece/google-sentencepiece-overview-architecture.svg
tags: [LLM, Python, Open Source]
categories: [AI Inference]
keywords: SentencePiece, tokenizer, unigram language model, BPE, Viterbi, subword segmentation
author: "PyShine"
---

Ask a Llama checkpoint what "tokenization" means and it will answer in the dialect of SentencePiece. Google's tokenizer library — a compact C++ core with Python bindings — has quietly become the substrate under an enormous slice of modern NLP: the Llama family, T5, ALBERT, XLNet, and countless fine-tuned descendants all ship a `.model` file produced by `spm_train`. Unlike tooling that assumes spaces delimit words, SentencePiece treats the raw byte stream as the source of truth, so Japanese, Chinese, code, and English can pass through the exact same training script without a pre-tokenizer stage. That single design decision — "no language-dependent preprocessing" — is visible everywhere in the source, and it is the thread we will follow in this tour.

The repository is refreshingly small for something this load-bearing. The inference path lives in a handful of files under `src/`: a processor facade, a normalizer, and four interchangeable segmentation models. The training path reuses those models under the hood but adds an expectation-maximization loop, a pruning pass, and a protobuf describing every choice made along the way. The Python side is no longer the old SWIG wrapper; since v0.2.2 the binding is pybind11 (`python/src/sentencepiece/sentencepiece_pybind.cc`), which also explains recent niceties like NumPy output and free-threading support.

What follows is a read of that source on the `master` branch, guided by two diagrams. The first gives the bird's-eye view — who calls whom, from your Python script down to the piece-to-id map. The second zooms into the lattice machinery and the unigram trainer, where the interesting algorithms live.

<div style="overflow-x:auto;"><img src="https://pyshine.com/assets/img/diagrams/sentencepiece/google-sentencepiece-overview-architecture.svg" alt="SentencePiece architecture overview: SentencePieceProcessor, normalizer, model subclasses, trainer, and ModelProto" style="min-width:900px;"></div>

<p><em>The overview graph: pybind11 wraps the C++ facade, the processor fans out to a normalizer plus one of four models, and the trainer side collapses everything into a serialized ModelProto.</em></p>

Reading the overview from left to right:

- [SentencePieceProcessor::Encode, L396](https://github.com/google/sentencepiece/blob/master/src/sentencepiece_processor.cc#L396) — the front door: normalize the input, ask the model for pieces, then reattach character offsets to the original text.
- [Normalizer::Normalize, L88](https://github.com/google/sentencepiece/blob/master/src/normalizer.cc#L88) — trie-driven character rewriting plus whitespace escaping to the U+2581 meta symbol.
- [ModelInterface::InitializePieces, L83](https://github.com/google/sentencepiece/blob/master/src/model_interface.cc#L83) — walks the model's pieces and builds the lookup tables every subclass depends on.
- [ModelInterface::PieceToId, L69](https://github.com/google/sentencepiece/blob/master/src/model_interface.cc#L69) — the abacus: a flat hash map with reserved ids (`<unk>`, `<s>`, `</s>`, `<pad>`) checked first.
- [Lattice::Viterbi, L185](https://github.com/google/sentencepiece/blob/master/src/unigram_model.cc#L185) — the dynamic-programming core that picks the highest-likelihood split of a normalized string.
- [bpe::Model::SampleEncode, L111](https://github.com/google/sentencepiece/blob/master/src/bpe_model.cc#L111) — merge-based segmentation driven by a priority queue over adjacent symbol pairs.
- [UnigramTrainer::Train, L824](https://github.com/google/sentencepiece/blob/master/src/unigram_model_trainer.cc#L824) — seed pieces, EM sub-iterations, and pruning until the vocabulary hits its target size.

## The Processor Facade: One Entry Point, Two Halves

`SentencePieceProcessor` is deliberately thin. [Load, L100](https://github.com/google/sentencepiece/blob/master/src/sentencepiece_processor.cc#L100) accepts a `ModelProto` and wires two members: a `Normalizer` and a `ModelInterface`. From then on, `Encode` is just three moves — normalize, segment, repopulate. The repopulation step, [PopulateSentencePieceText, L298](https://github.com/google/sentencepiece/blob/master/src/sentencepiece_processor.cc#L298), is the part users rarely think about but constantly benefit from: the normalizer records a `norm_to_orig` mapping, and this function converts piece-level positions in the *normalized* string back into byte offsets in the *original* string, filling the `SentencePieceText` proto with `surface`, `begin`, and `end` per token. Decode runs the inverse: [Decode, L492](https://github.com/google/sentencepiece/blob/master/src/sentencepiece_processor.cc#L492) concatenates pieces, drops control tokens, expands byte-fallback tokens, and reverses the whitespace escape so you get human-readable text out.

The two halves talk through a narrow interface. The normalizer produces a string where every space is now `▁` (U+2581, "LOWER ONE EIGHTH BLOCK"), so a piece like `▁This` carries its own leading boundary. The model consumes that string and returns `(piece, id)` pairs. Note that the processor also exposes the map directly — [PieceToId, L1101](https://github.com/google/sentencepiece/blob/master/src/sentencepiece_processor.cc#L1101) and `IdToPiece` — which is all the Python `piece_to_id` / `id_to_piece` helpers funnel into.

## Unigram Segmentation: A Viterbi Walk Over the Lattice

The unigram model is the default `model_type`, and it is the most interesting file in the repo. Its premise: each piece carries a log-probability, and the best tokenization of a string is the split with maximum total likelihood. Finding that split exactly is a shortest-path problem over a lattice, and the code builds it honestly. [Model::PopulateNodes, L599](https://github.com/google/sentencepiece/blob/master/src/unigram_model.cc#L599) runs a common-prefix search over the vocabulary and inserts a node for every piece that matches some suffix of the input.

Classic inference then walks the DAG: [Lattice::Viterbi, L185](https://github.com/google/sentencepiece/blob/master/src/unigram_model.cc#L185) computes, for each node, the best `backtrace_score` over incoming edges and keeps one back-pointer per node; reconstructing from the EOS node yields the segmentation. But the hot path does not allocate a lattice at all. [Model::EncodeOptimized, L796](https://github.com/google/sentencepiece/blob/master/src/unigram_model.cc#L796) performs the same left-to-right dynamic program with per-position hypotheses kept in flat arrays, plus score recentering to survive very long inputs, and it falls back to the classic path only when timeouts or degenerate scores appear.

The same lattice supports three more modes that matter for training-time regularization: `NBestEncode` enumerates the k best segmentations, and `SampleEncode` draws segmentations with probability proportional to their likelihood (the "subword regularization" trick that measurably improves downstream robustness). During training, [Lattice::PopulateMarginal, L268](https://github.com/google/sentencepiece/blob/master/src/unigram_model.cc#L268) computes how often each node appears across the whole posterior — the expected counts that the EM loop consumes.

## BPE Merges, Word and Character Fallbacks

BPE in this codebase is a compact lesson in data-structure choice. [bpe::Model::SampleEncode, L111](https://github.com/google/sentencepiece/blob/master/src/bpe_model.cc#L111) first splits the input into single-character symbols linked in a doubly-linked list, then loads an "agenda" priority queue with every adjacent pair that exists in the merge table. The comparator, `SymbolPairComparator`, packs scores into integers for branch-free comparison — a reminder that tokenizer code is latency-sensitive. Popping the best pair merges the two symbols in place, then pushes the new neighbor pairs back onto the agenda. The same function doubles as BPE-dropout: when `alpha > 0`, each merge is skipped with probability alpha, and the base class shows that plain `Encode` is just `SampleEncode(normalized, 0.0)` — deterministic merges.

One subtlety worth knowing: BPE uses `PieceToIdNoReserved` rather than the public lookup, so control symbols like `<s>` can never be merged into larger pieces by accident. The word and character models are intentionally boring. [word::Model::Encode, L29](https://github.com/google/sentencepiece/blob/master/src/word_model.cc#L29) looks up whitespace-delimited tokens whole, and [char::Model::Encode, L29](https://github.com/google/sentencepiece/blob/master/src/char_model.cc#L29) emits one piece per character — useful baselines, and handy when a language resists subword segmentation.

## From Install to Your First Model

Everything above ships as a wheel; there is no compiler between you and a working model. Install, train on a plain text file, and round-trip some text:

```python
import sentencepiece as spm

# Train a 2000-piece unigram model from a UTF-8 corpus.
spm.SentencePieceTrainer.train(input='input.txt', model_prefix='m', vocab_size=2000)

# Load and use it.
sp = spm.SentencePieceProcessor(model_file='m.model')

print(sp.encode('hello world'))                 # [0, 1, ...] ids
print(sp.encode('hello world', return_type=str))  # ['▁hello', '▁world']
print(sp.decode(sp.encode('hello world')))      # 'hello world'
print(sp.get_piece_size())                      # 2000
```

The trainer call emits the two artifacts: `m.model` (the serialized protobuf) and `m.vocab` (a human-readable piece/score listing). The same `train()` accepts `sentence_iterator` and `model_writer` arguments, so you can train from a URL stream into a `BytesIO` on machines without a writable local disk. Under the Python surface, `spm.SentencePieceTrainer.train` maps keyword arguments onto the C++ flag parser of [SentencePieceTrainer::Train, L163](https://github.com/google/sentencepiece/blob/master/src/sentencepiece_trainer.cc#L163), and every call logs the effective `trainer_spec` to stderr — worth skimming once, because it is the ground truth of what your model was configured to do.

## Inside the Trainer: Seeds, EM, and Pruning

The trainer's shared plumbing lives in [TrainerInterface::LoadSentences, L368](https://github.com/google/sentencepiece/blob/master/src/trainer_interface.cc#L368): it streams input files (or an iterator), drops empty sentences, and computes the `required_chars_` set from `character_coverage` — characters frequent enough that they must survive into the final vocabulary. A companion filter, `IsValidSentencePiece`, rejects candidate pieces that break whitespace rules or exceed `max_sentencepiece_length`.

The unigram trainer then runs its three-phase loop. First, [MakeSeedSentencePieces, L223](https://github.com/google/sentencepiece/blob/master/src/unigram_model_trainer.cc#L223) harvests a deliberately oversized seed vocabulary from frequent substrings (or single characters for tiny corpora). Second, the EM phase alternates [RunEStep, L519](https://github.com/google/sentencepiece/blob/master/src/unigram_model_trainer.cc#L519) — lattice-based expected counts over sampled sentences — with `RunMStep`, which converts those counts into fresh piece log-probs. Third, [PruneSentencePieces, L601](https://github.com/google/sentencepiece/blob/master/src/unigram_model_trainer.cc#L601) removes the pieces whose removal degrades the corpus likelihood the least, looping until the vocabulary shrinks to `vocab_size`. The result is serialized by [TrainerInterface::Serialize, L651](https://github.com/google/sentencepiece/blob/master/src/trainer_interface.cc#L651) into the proto defined by `sentencepiece_model.proto` — a schema where [ModelProto.pieces, L298](https://github.com/google/sentencepiece/blob/master/src/sentencepiece_model.proto#L298) is literally the ordered id table, and `trainer_spec` records the training configuration forever. The default `vocab_size` of 8000 is declared at [L57](https://github.com/google/sentencepiece/blob/master/src/sentencepiece_model.proto#L57) of the proto, next to the `model_type` enum defaulting to `UNIGRAM`.

<div style="overflow-x:auto;"><img src="https://pyshine.com/assets/img/diagrams/sentencepiece/google-sentencepiece-architecture.svg" alt="SentencePiece detail diagram: inference path, lattice and Viterbi, model internals, unigram trainer EM loop, and ModelProto" style="min-width:900px;"></div>

<p><em>The detail graph: the inference path on the left, the lattice/Viterbi cluster and the EM-plus-pruning trainer loop on the right, both meeting at the ModelProto schema.</em></p>

## Honest Limits

A few constraints are structural, and knowing them up front saves hours. Building the trainers from source requires a real C++ toolchain — CMake plus a modern compiler on every platform; the pip wheels cover common Linux/macOS/Windows targets, but embedded or exotic environments must compile the core themselves. The vocabulary is fixed before training starts: you must choose `vocab_size` ahead of time, and the trainer will either prune down to it or refuse to run if it cannot fit `required_chars` and meta pieces into that budget. There is no byte-fallback by default — unless you enable the `byte_fallback` flag (and include the 256 byte pieces), any character outside the vocabulary collapses into `<unk>`, which is exactly how many older models "lose" emoji and rare scripts. Finally, whitespace handling changes round-tripping: spaces are replaced by the `▁` underscore meta symbol and a dummy prefix is typically added, so the decoded text equals the input for ordinary sentences, but pieces you inspect in logs or embeddings tables look like `▁This` rather than `This`, and exotic whitespace configurations can surprise you if you assume byte-identical reconstruction.

Where to go next depends on what you are building. If you are comparing tokenizers as such, the Rust-based [tokenizers](/tokenizers/) library is the natural counterpart, and [transformers](/transformers/) glues either one onto model checkpoints. If your interest is serving models end to end, see how [vllm-2](/vllm-2/) and [llama.cpp-2](/llama.cpp-2/) handle preprocessing around the weights, or how the original [llama2](/llama2/) architecture pairs with its SentencePiece vocabulary.

```json
{
  "@context": "https://schema.org",
  "@type": "TechArticle",
  "headline": "SentencePiece: A Source Tour of Google's Language-Agnostic Tokenizer",
  "description": "Walk through the C++ source: the processor API, Viterbi unigram segmentation, BPE merges, normalization rules, and the trainer that Llama and T5 depend on.",
  "datePublished": "2026-10-24",
  "author": {"@type": "Person", "name": "PyShine"},
  "publisher": {"@type": "Organization", "name": "PyShine"},
  "keywords": "SentencePiece, tokenizer, unigram language model, BPE, Viterbi, subword segmentation"
}
```
