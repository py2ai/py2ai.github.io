---
layout: post
title: "Glyd: Bit-for-Bit Compression for Model Weights and Data Lakes - Inside surya-koritala/Glyd"
description: "A source tour of Glyd, a Rust lossless compression engine that holds bf16 model weights and KV caches packed in GPU memory bit for bit, reads 3-7x faster than zstd on servers, turns logs into typed columns, and diffs new object versions against old ones."
date: 2026-10-03
header-img: "img/post-bg.jpg"
permalink: /Glyd-Lossless-Compression-For-Models-And-Data/
featured-img: "ai-coding-frameworks/ai-coding-frameworks"
image: https://pyshine.com/assets/img/diagrams/glyd/surya-koritala-glyd-architecture.svg
tags:
  - Rust
  - Compression
  - GPU
  - LLM
categories: [AI, Open Source]
keywords: "lossless compression, model weights, KV cache, GPU memory, vLLM, Rust, zstd alternative, LZ4, object storage, safetensors, delta compression, typed columns, bf16"
author: "PyShine"
---

Every layer of the AI stack has a storage problem. Training runs checkpoint terabytes; inference packs GPUs to the brim with weights and KV caches; data lakes drown in logs and dumps that are written once and read constantly. Compression tools exist for all of this, but they treat each problem separately - and they all make the same trade: smaller means slower, or lossy, or both. [Glyd](https://github.com/surya-koritala/Glyd), a Rust workspace now at version 0.26.0, refuses that trade and goes after the interesting part: lossless compression whose decode is the fastest thing in the room.

The headline result is the GPU side. Glyd holds a bf16 model's weights and KV cache in fewer bits in GPU memory and rebuilds the exact values on the GPU as the model runs - the model bit for bit. Every Linear layer's matrix packs and unpacks to itself, and the repository publishes the evidence: Qwen3-32B drops from 65.5 GB of weights to 44.5 GB, fitting on one 48 GB GPU instead of two; Qwen2.5-72B fits on three instead of four; a Qwen2.5-7B KV cache at 16K tokens shrinks from 947 MB to 651 MB. MMLU answers match bf16's on 98-100% of questions because nothing is approximated - only the arithmetic order of the products differs, as it does between any two kernels.

Underneath that sits a general-purpose engine: a drop-in alternative to LZ4, Snappy and zstd for object storage, data lakes, logs, backups and RPC payloads, with a C ABI and Python and Go bindings. The full program is in `docs/benchmarks/suite-2026-09-21.md` - an 8.7 GB real-data corpus measured on AWS Graviton3, where Glyd's default tier decompresses at 3,309 MB/s per core against LZ4's 1,394 and zstd -3's 1,425, at a better ratio than LZ4. The claim is not "best at everything"; it is precise: reads are the right side of the trade for data written once and read many times.

The source is worth a tour because it is three products in one repository - a codec, an object store, and a GPU residency system - sharing one format discipline, with every number in the README backed by committed logs under `benchmarks/`.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/glyd/surya-koritala-glyd-overview-architecture.svg" alt="Architecture overview of the surya-koritala/Glyd repository" style="max-width:100%;height:auto;" />
</div>

*Overview of the Glyd repository: the Rust codec core and its CLI, the smart modes for records, deltas and model files, the CUDA kernels that keep weights compressed on the GPU, and the store that deduplicates across objects.*

Reading the overview from left to right: `src/bin/glyd.rs` drives `src/lib.rs`, the codec core that reads and writes its own `v7` container format and layers three special modes on top. `src/record.rs` turns record-shaped data into typed columns before compressing; `src/finder.rs` compresses a new version of an object against the old one wherever content moved; `src/safetensors.rs` opens model files tensor by tensor. The store crate wraps the codec with cross-object deltas, and the GPU path drives the CUDA kernels in `gpu/` from Python, ending in the vLLM integration that serves packed models.

## Why You Need This

If your bill is dominated by reads - analytics over cold logs, serving model weights, loading checkpoints - the decompress speed of your codec is the throughput of your system. Glyd's decode is parallel by construction: its output decodes across all cores in one stream, while a classic zstd or LZ4 frame decodes on one thread. On the eight-core benchmark that is 22,707 MB/s of read throughput against zstd -3 -T8's 1,422, with sizes in the same class. For write-heavy, rarely-read data, the README says plainly that zstd still wins on write cost - a rare piece of honesty in a benchmark section.

For anything versioned, `--base` mode changes the economics of keeping history. A new version is compressed against the old one with its content found wherever it moved, so dumps, images and source trees store at 1-5% of their plain size - fifteen kernel releases in 228 MB instead of 3 GB. The same idea generalized across objects is the store: `put` an object and it is kept as a delta against the stored object it most resembles, when that pays, and a read costs at most five decodes. A 39 GB bucket of images, kernel releases, Wikipedia tables and GitHub events stores in 1,334 MB where zstd -3 needs 6,132 MB.

For ML artifacts specifically, the codec understands the containers. safetensors files are opened tensor by tensor, PyTorch training checkpoints (optimizer state included) compress against the previous checkpoint, and the GPU path finishes the job by keeping the weights compressed *in memory* - the only stage where decompression would otherwise be paid on every token.

## How It Works

One container format, one entropy stack, and three layers of specialization over it.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/glyd/surya-koritala-glyd-architecture.svg" alt="Detailed architecture of the surya-koritala/Glyd repository" style="max-width:100%;height:auto;" />
</div>

*The detailed view: codec core and v7 container, the entropy and SIMD layers, format interop with deflate, zstd, snappy and JPEG, the smart modes for records, shapes, deltas and model files, the store crate, and the CUDA path.*

### Understanding the Architecture

**The v7 container is the spine.** `src/v7_format.rs` defines it, `src/v7_encode.rs` writes it, `src/v7_decode.rs` reads it, and `src/v7_ultra.rs` implements the slowest, smallest tier on top of the encoder. Members carry their own coding choices, so a file can mix record columns, delta patches and tensor blocks. SIMD decode paths in `src/x86_decompress.rs` (AVX2) and `src/neon_decompress.rs` (NEON) make that decode fast per core, which is what the benchmark tables are really measuring.

**Entropy coding is hand-rolled and layered.** `src/huffman.rs` and `src/huff8.rs` build static Huffman tables, `src/tans.rs` implements an asymmetric numeral systems coder, `src/bits.rs` does the bit-level I/O underneath, and `src/ldm.rs` finds long-distance repeats before coding. Everything composes through `src/lib.rs`, with `src/error.rs` as the shared error model and `src/mmap.rs` feeding inputs without extra copies.

**Interoperability is a reader, not a reimplementation.** The `src/reflate/`, `src/rezstd/` and `src/resnappy/` modules reconstruct deflate, zstd and snappy streams, and `src/jpg/` carries a JPEG model - covering files that earlier releases and other tools wrote, so the codec can ingest a zstd lake without a transcode step. `src/parquet.rs` reads Parquet columns into the record flow, where `src/record.rs` re-shapes logs, SQL dumps, CSV and JSON lines into typed columns and `src/shape.rs` trains shape dictionaries for small single objects.

**The modes that matter are small modules.** `src/finder.rs` locates the stored object a new version resembles and `src/split.rs` partitions the stream so unchanged regions become references - that pair is `--base` mode and the engine under the store. `src/streaming.rs` keeps the same format usable on pipes. `src/safetensors.rs` and `src/torchzip.rs` walk model files tensor by tensor so each matrix gets its own coding decision.

**The GPU path keeps compression past load time.** The Rust crate `glyd-gpu` packs checkpoints on the CPU with no GPU required (`glyd pack Qwen/Qwen3-8B out`); the CUDA kernels in `gpu/glyd_gpu.cu` decode matrices and the KV cache into tensor-core registers as the model runs, with `mma` and `mma12` layouts tuned for different GPUs. `gpu/glyd_gpu.py` exposes `from_pretrained` - packed as it loads, optionally `exact=True` for bit-for-bit bf16 logits - and the `gpu/vllm/` integration lets vLLM hold Linears and MoE experts packed while its KV cache takes the memory saved.

**End to end:** `pip install "glyd[gpu]"`, then `glyd run Qwen/Qwen3-8B` downloads the model, starts it packed, and opens a chat - the weights land in 9.4 GB instead of 13.9 on a 16 GB card, generate 1.25-1.32x faster than bf16 from 1 to 32 sequences, and decode to themselves bit for bit. The same CLI, pointed at a directory of logs with `-r`, a bucket with the store, or a checkpoint with `--base`, applies the same format discipline to plain data.

## Advantages

- **Lossless with receipts.** Every decode is compared byte for byte with its input, every benchmark against the reference codec on the same machine in the same run, logs committed under `benchmarks/`.
- **Decode-first design.** Parallel per-stream decode and SIMD paths make reads 3-7x zstd's on multicore servers - the right optimization for read-heavy storage.
- **Structure-aware modes.** Record mode, shape dictionaries, packs and base/delta mode exploit the structure that generic codecs leave on the table.
- **GPU residency for inference.** Weights and KV cache stay packed in memory at 10.80 bits per value, bit for bit, with model families Qwen3, Qwen2, Llama, Mistral and Granite supported by the packer.
- **One format everywhere.** The same v7 container covers CPU files, store objects and GPU-packed checkpoints, readable through C, Python and Go bindings.
- **Honest scoping.** The README states where it loses - write cost, H100 `mma` latency, GH200 saturation - with numbers attached.

## Benefits

- **Cut GPU count, not quality.** A 32B model that needed two 48 GB GPUs runs on one; the MMLU answers are bf16's on 98-100% of questions.
- **Shrink versioned storage by an order of magnitude.** `--base` mode keeps release history at 1-5% of plain size without a restore ceremony.
- **Store a lake in a fraction.** The cross-object store kept a 39 GB mixed bucket in 1.3 GB, every object read back byte-exact.
- **Speed up the read path you already have.** Faster-than-memory decompression turns cold logs and dumps into warm analytics inputs.
- **Pack models for distribution.** `glyd pack` writes checkpoints 33% smaller on CPU alone; fine-tunes against their base store 44% smaller.
- **Adopt incrementally.** Use it as a CLI, a library over the C ABI, Python or Go bindings, the store crate, or the vLLM integration - each works alone.

## Usage

Install and serve a packed model (Linux, NVIDIA GPU):

```bash
curl -LsSf https://getglyd.com/install.sh | sh
glyd run Qwen/Qwen3-8B
glyd serve MODEL        # OpenAI-compatible API
glyd doctor             # what this machine has, which models fit
```

Python, packed on load:

```python
import glyd
model = glyd.from_pretrained("Qwen/Qwen3-8B")               # 11.2 GB of weights, not 16.4
model = glyd.from_pretrained("Qwen/Qwen3-8B", exact=True)   # logits bit for bit bf16's
```

Pack a checkpoint without Python or a GPU:

```bash
glyd pack Qwen/Qwen3-8B qwen3-8b-glyd
glyd pack Qwen/Qwen3-8B qwen3-8b-glyd12 --layout mma12
```

General data work with the codec CLI:

```bash
glyd -r logs.jsonl logs.glyd        # record mode: typed columns
glyd --base old.tar new.tar new.glyd   # delta against the old version
glyd --cold archive.tar archive.glyd   # archive tier
```

## Conclusion

Glyd is that rare compression project with a thesis: the decode is the product, structure is free wins, and lossless is table stakes. Whether the artifact is a tensor, a log column, a versioned tarball or a bucket of related objects, the same Rust core and the same container format give it a smaller, faster-restorable form - and on the GPU, the compression finally follows the weights all the way into memory.

Links:

- Repository: [https://github.com/surya-koritala/Glyd](https://github.com/surya-koritala/Glyd)
- Benchmark suite: [https://github.com/surya-koritala/Glyd/blob/main/docs/benchmarks/suite-2026-09-21.md](https://github.com/surya-koritala/Glyd/blob/main/docs/benchmarks/suite-2026-09-21.md)
- GPU documentation: [https://github.com/surya-koritala/Glyd/blob/main/gpu/README.md](https://github.com/surya-koritala/Glyd/blob/main/gpu/README.md)
- vLLM integration: [https://github.com/surya-koritala/Glyd/blob/main/gpu/vllm/README.md](https://github.com/surya-koritala/Glyd/blob/main/gpu/vllm/README.md)
