---
layout: post
title: "One Framework, Four Kinds of Retrieval - Inside UKPLab/sentence-transformers"
description: "A source-level tour of UKPLab/sentence-transformers: how SentenceTransformer, CrossEncoder, SparseEncoder, and MultiVectorEncoder share one BaseModel of composable modules, one HF Trainer bridge, and one set of similarity utilities to serve dense, reranked, sparse, and late-interaction retrieval."
date: 2026-10-16
header-img: "img/post-bg.jpg"
permalink: /sentence-transformers/
featured-img: ai-coding-frameworks/ai-coding-frameworks
image: https://pyshine.com/assets/img/diagrams/sentence-transformers/ukplab-sentence-transformers-overview-architecture.svg
tags: [sentence-transformers, embeddings, semantic search, CrossEncoder, SPLADE, ColBERT, retrieval]
categories: [AI, Open Source]
keywords: "sentence-transformers source code, SentenceTransformer encode, CrossEncoder rank, SparseEncoder SPLADE, MultiVectorEncoder MaxSim, BaseModel nn.Sequential, SentenceTransformerTrainer, CachedMultipleNegativesRankingLoss, mine_hard_negatives, ONNX OpenVINO export"
author: "PyShine"
---

In 2019, two researchers at the UKP Lab in TU Darmstadt published Sentence-BERT and open-sourced the code that made transformer embeddings practical for sentences. Six years later that codebase, UKPLab/sentence-transformers under the Apache-2.0 license and now maintained within Hugging Face, has grown into the framework behind more than fifteen thousand pretrained models on the Hub and a permanent fixture of the MTEB retrieval leaderboard. The current development line, version 6.2.0, is no longer just sentence embeddings: the same package now ships four model families, SentenceTransformer for dense embeddings, CrossEncoder for reranking, SparseEncoder for SPLADE-style sparse vectors, and MultiVectorEncoder for ColBERT-style late interaction, all behind one import and one training story. As always in this series, what follows is an educational tour of published source code.

The architecture earns its simplicity through a shared core. Every family inherits from a single BaseModel that is literally an nn.Sequential of modules: a Transformer input module loads any Hugging Face checkpoint, then pooling, projection, and normalization modules shape the output. Each family is a thin subclass that adds its own encode-style methods, its own trainer glue, and its own loss library, while saving, hub uploads, multimodal inputs, batching, and backend exports live once in the base. Reading the overview from left to right follows that story: four facades, one core, one training bridge, and a utility belt from similarity math to ONNX export.

<div style="text-align:center;"><img src="https://pyshine.com/assets/img/diagrams/sentence-transformers/ukplab-sentence-transformers-overview-architecture.svg" alt="sentence-transformers repository architecture overview" style="max-width:100%;"/></div>

<p><em>Overview of sentence-transformers: four family facades extend one BaseModel core built from composable modules, the BaseTrainer bridges Hugging Face's Trainer, and utilities plus export backends round out the package.</em></p>

Reading the overview from left to right:

- [SentenceTransformer](https://github.com/UKPLab/sentence-transformers/blob/main/sentence_transformers/sentence_transformer/model.py) is the flagship at line 38 of 1341 lines, with encode implemented at 753 and the similarity method at 1056.
- [CrossEncoder](https://github.com/UKPLab/sentence-transformers/blob/main/sentence_transformers/cross_encoder/model.py) is the reranker at line 32 of 826 lines: predict at 450 scores text pairs, rank at 642 returns ordered passages.
- [SparseEncoder](https://github.com/UKPLab/sentence-transformers/blob/main/sentence_transformers/sparse_encoder/model.py) at line 30 of 1473 lines produces vocabulary-sized sparse vectors, with a static sparsity reporter at 1157.
- [MultiVectorEncoder](https://github.com/UKPLab/sentence-transformers/blob/main/sentence_transformers/multi_vector_encoder/model.py) at line 63 of 1623 lines emits one vector per token, scored by MaxSim at 977.
- [BaseModel](https://github.com/UKPLab/sentence-transformers/blob/main/sentence_transformers/base/model.py) is the shared core at line 66 of 2045 lines, with save_pretrained at 806 and push_to_hub at 950.
- [The Transformer module](https://github.com/UKPLab/sentence-transformers/blob/main/sentence_transformers/base/modules/transformer.py) is the input module at line 644 that wraps any Hugging Face checkpoint, dispatching by [Router](https://github.com/UKPLab/sentence-transformers/blob/main/sentence_transformers/base/modules/router.py) at line 30 for non-text modalities.
- [BaseTrainer](https://github.com/UKPLab/sentence-transformers/blob/main/sentence_transformers/base/trainer.py) is the training bridge at line 89 of 1334 lines, extending the Hugging Face Trainer for all four families.
- [util/similarity.py](https://github.com/UKPLab/sentence-transformers/blob/main/sentence_transformers/util/similarity.py) is the math belt of 940 lines: cos_sim at 59, dot_score at 101, manhattan at 135, each with pairwise variants.

## Why You Need This

Retrieval is not one algorithm, it is a menu of trade-offs, and this repository is the only mainstream library that puts the whole menu behind one API. Dense embeddings compress a sentence into one vector: fast to search, occasionally lossy. A CrossEncoder reads query and passage together and scores them jointly: accurate, too slow to run over a whole corpus, which is exactly why it exists as a reranker. Sparse vectors keep the vocabulary dimension, so matching is transparent and lexical. Late interaction keeps one vector per token and scores with MaxSim, trading storage for fine-grained matching. Learning all four from scratch means four codebases; here they are four classes with the same load, save, encode, and similarity conventions.

The second reason is the module system, which is genuinely compositional rather than nominally so. A model is a chain: put a Transformer at the front, append Pooling to turn tokens into one vector, optionally a Dense projection or Normalize at the end, save it, and the chain definition travels with the weights. Swap Pooling for a sparse pooling module and the same skeleton produces SPLADE vectors; swap the front for StaticEmbedding lookup tables and you get CPU-friendly models that the project blogs about running dramatically faster than attention-based ones. Because every module implements the same save and load contract at the Module base class, custom research modules slot into pretrained models without forks.

The third reason is the training story, which is where most embedding projects stall. The library ships roughly twenty-eight embedding loss functions, from MultipleNegativesRankingLoss to cached variants that grow batch sizes to thousands of negatives, plus a dozen reranker losses and a dozen sparse losses, all feeding one Hugging Face Trainer subclass so you get mixed precision, multi-GPU, and logging for free. Evaluation runs during training with eleven ready evaluators, hard negatives can be mined from any corpus with one function call, and trained models auto-generate full model cards for the Hub. That is a complete path from raw pairs to a searchable, shareable artifact.

<div style="text-align:center;"><img src="https://pyshine.com/assets/img/diagrams/sentence-transformers/ukplab-sentence-transformers-architecture.svg" alt="sentence-transformers detailed architecture" style="max-width:100%;"/></div>

<p><em>The detailed map: family facades and their key methods, the BaseModel core with save, hub, cards, adapters, and modality handling, the module stack, the training bridge with losses and samplers, evaluators, and the utility and backend layer.</em></p>

The cleanest place to understand this codebase is the inheritance diamond that is not. [BaseModel](https://github.com/UKPLab/sentence-transformers/blob/main/sentence_transformers/base/model.py) at line 66 declares itself as nn.Sequential, PeftAdapterMixin, and ABC, which means a model instance is simultaneously the module chain, a loadable adapter host, and a family contract. Its 2045 lines carry the universal verbs: save_pretrained at 806 walks the module chain and serializes each module's config and state, push_to_hub at 950 packages the folder with the model card machinery from a 2052-line [model_card.py](https://github.com/UKPLab/sentence-transformers/blob/main/sentence_transformers/base/model_card.py), and the 941-line [modality.py](https://github.com/UKPLab/sentence-transformers/blob/main/sentence_transformers/base/modality.py) defines how text, image, audio, and video inputs are named, processed, and routed. The [Router](https://github.com/UKPLab/sentence-transformers/blob/main/sentence_transformers/base/modules/router.py) at line 30 picks the right input module per modality, which is how the CLIP module and the text Transformer coexist in one framework.

The family facades then differentiate on output shape. SentenceTransformer's encode at 753 is the single-vector path, with encode_multi_process at 1101 for corpus-scale jobs and similarity at 1056 dispatching to cosine, dot, euclidean, or manhattan. CrossEncoder's predict at 450 feeds pairs through the transformer and a scoring head, and rank at 642 wraps predict with sorting so the README's three-passage Berlin query returns corpus ids, scores, and texts in order, with the ms-marco MiniLM checkpoint printing 8.61, 6.35, and 5.51 for its three candidates. SparseEncoder's encode at 578 produces a vector the width of the vocabulary, and its static sparsity method at 1157 reports what the README demonstrates on a SPLADE model: ninety-nine point eight percent of activations are zero. MultiVectorEncoder splits into encode_query at 355 and encode_document at 531, then scores with MaxSim at 977, where each query token picks its best-matching document token, which is why the README example encodes a ten-token query into a 10 by 128 matrix and scores a Paris document at 9.60 against a Berlin one at 9.41.

Training is deliberately boring in the best way. [SentenceTransformerTrainer](https://github.com/UKPLab/sentence-transformers/blob/main/sentence_transformers/sentence_transformer/trainer.py) is only 168 lines at line 36 because the 1334-line [BaseTrainer](https://github.com/UKPLab/sentence-transformers/blob/main/sentence_transformers/base/trainer.py) at line 89 already extends the Hugging Face Trainer with column remapping, multi-dataset sampling through an 806-line sampler, and a 187-line collator that pairs texts with labels. The loss libraries are where the research lives: the [CachedMultipleNegativesRankingLoss](https://github.com/UKPLab/sentence-transformers/blob/main/sentence_transformers/sentence_transformer/losses/cached_multiple_negatives_ranking.py) at line 26 is one of twenty-eight embedding losses, caching gradient-safe negatives so contrastive training scales without OOM. The legacy fit method survives at line 160 of [fit_mixin.py](https://github.com/UKPLab/sentence-transformers/blob/main/sentence_transformers/sentence_transformer/fit_mixin.py) as a compatibility path that compiles old-style arguments into the trainer stack.

The last mile is deployment and data, and both get real code rather than afterthoughts. [mine_hard_negatives](https://github.com/UKPLab/sentence-transformers/blob/main/sentence_transformers/util/hard_negatives.py) at line 25 of a 948-line module scans a corpus with an existing model to manufacture difficult negatives, the single highest-leverage trick for embedding quality. [quantization.py](https://github.com/UKPLab/sentence-transformers/blob/main/sentence_transformers/util/quantization.py) compresses embeddings to binary and scalar forms across 564 lines for cheaper indexes. And the [backend layer](https://github.com/UKPLab/sentence-transformers/blob/main/sentence_transformers/backend/load.py) loads ONNX models at line 14 and OpenVINO at 92, with export helpers for optimized ONNX at optimize.py line 19, dynamic ONNX quantization at quantize.py line 24, and static OpenVINO quantization at line 109, so a trained model leaves Python behind without leaving the framework.

## From Install to Embeddings

The stack asks for Python 3.10 or newer, PyTorch 2.2 or newer, and transformers v5 or newer:

```sh
pip install -U sentence-transformers
```

Then the canonical example from the repository's README:

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

sentences = [
    "The weather is lovely today.",
    "It's so sunny outside!",
    "He drove to the stadium.",
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# => (3, 384)

similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[1.0000, 0.6660, 0.1046],
#         [0.6660, 1.0000, 0.1411],
#         [0.1046, 0.1411, 1.0000]])
```

Three sentences become three 384-dimensional vectors, and one matrix multiplication answers which sentences mean the same thing. Swap the class name for CrossEncoder, SparseEncoder, or MultiVectorEncoder and the same four-line shape serves reranking, sparse search, or late interaction, with the repo even shipping an agent skill, installable via hf skills add train-sentence-transformers, that lets coding agents drive the training APIs directly.

Honest limits: the four-family breadth is new, and the seams show in version churn, with the jump to Trainer-based training in v3 and the multimodal expansion still leaving deprecation trails that the codebase keeps politely shimming. The legacy fit path exists precisely because a decade of tutorials depends on it. Rerankers and late-interaction models are compute-hungry at query time, and the library will not make a CrossEncoder cheap, only easy. Sparse and multi-vector support, while first-class, leans on a smaller model zoo than the dense family. And the module-chain design, elegant as it is, means a model directory is a contract: load a chain you did not build and you inherit its preprocessing decisions. Still, the core achievement stands: one import gives you four decades-worth of retrieval research in a shape you can actually ship, which is why sentence-transformers remains the default answer to how do I embed text.
