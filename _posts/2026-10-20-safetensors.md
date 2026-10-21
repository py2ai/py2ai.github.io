---
layout: post
title: "Safetensors Source Tour: The Zero-Copy Weight Container"
description: "How Hugging Face's safetensors stores model weights: an 8-byte header, validated JSON metadata, a no_std Rust core, and zero-copy mmap, pread and DLPack paths through PyO3 into every major framework."
date: 2026-10-20
header-img: "img/post-bg.jpg"
permalink: /safetensors/
featured-img: "ai-coding-frameworks/ai-coding-frameworks"
image: https://pyshine.com/assets/img/diagrams/safetensors/huggingface-safetensors-overview-architecture.svg
tags: [LLM, Python, Open Source]
categories: [AI Inference]
keywords: [safetensors, serialization, mmap, zero-copy, Rust, Hugging Face, model weights]
author: "PyShine"
---

Every large model you download today is, on disk, mostly just a bag of numbers. The interesting engineering question is what wraps around that bag. For years the answer in the PyTorch world was `torch.save`, which wraps Python pickle — arbitrary code execution on load, no partial loading, no framework neutrality. Safetensors is Hugging Face's answer: a tiny, deliberately boring container format that stores tensors as raw bytes preceded by a small JSON header, with serialization and deserialization implemented once in a dependency-light Rust core and exposed to Python, plus ports for numpy, JAX, TensorFlow, Paddle and MLX. If you have read our [tokenizers](/tokenizers/) source tour, the shape of the project will feel familiar: a compact Rust crate, a PyO3 binding layer, and thin per-framework Python modules on top.

The design bet is that a checkpoint should be mmap-able: the loader should not have to read, copy or even deserialize tensor bytes to put them where a framework wants them. That single bet explains almost everything in the codebase — the 8-byte header preamble, the strict offset validation, the `torch.UntypedStorage` trick, the `pread(2)` backend, and the DLPack hand-off for CUDA and Metal. It is also why serving stacks like [vLLM](/vllm-2/), [TGI](/tgi/) and the [Transformers](/transformers/) library made safetensors the default weight format; fast weight streaming is a load-time problem, and safetensors attacks it at the file-format level rather than with application-side tricks.

This post tours the actual source: the format contract, the Rust validation core, the zero-copy paths, the async prefetch engine, and the per-framework ports. As usual, every linked symbol below points at the exact line on `main` at the time of writing, so you can follow along in the repository.

<div style="overflow-x:auto;"><img src="https://pyshine.com/assets/img/diagrams/safetensors/huggingface-safetensors-overview-architecture.svg" alt="Safetensors overview architecture" style="max-width:100%;"></div>

<p><em>Overview of safetensors: a no_std Rust core parses and validates the header, the PyO3 binding layer exposes safe_open with mmap/pread backends and a background prefetch engine, and pure-Python ports adapt the result to torch, numpy, JAX and friends.</em></p>

Reading the overview from left to right:

- The format contract is documented in the crate README: [8 bytes: N, L76](https://github.com/huggingface/safetensors/blob/main/safetensors/README.md#L76) — an unsigned little-endian 64-bit integer giving the size of the JSON header, followed by N bytes of header and then the raw tensor bytes.
- [MAX_HEADER_SIZE, L10](https://github.com/huggingface/safetensors/blob/main/safetensors/src/tensor.rs#L10) caps that header at 100,000,000 bytes, which bounds the cost of parsing metadata no matter how many tensors a file declares.
- [read_metadata, L390](https://github.com/huggingface/safetensors/blob/main/safetensors/src/tensor.rs#L390) reads the u64 length, slices the header bytes, checks UTF-8 and JSON validity, then runs validation before anything is returned.
- [validate, L627](https://github.com/huggingface/safetensors/blob/main/safetensors/src/tensor.rs#L627) is the security heart: offsets must be contiguous and increasing, and each tensor's span must exactly equal shape product times dtype bitsize.
- [SafeTensors::deserialize, L446](https://github.com/huggingface/safetensors/blob/main/safetensors/src/tensor.rs#L446) turns a validated buffer into a `SafeTensors` whose tensors are borrowed views — zero bytes are copied.
- [safe_open, L1704](https://github.com/huggingface/safetensors/blob/main/bindings/python/src/lib.rs#L1704) is the PyO3 front door: a Python context manager that parses the header once and serves tensors lazily.
- [Storage, L594](https://github.com/huggingface/safetensors/blob/main/bindings/python/src/lib.rs#L594) enumerates the serving strategies: a plain Rust `Mmap`, a torch-owned `UntypedStorage`, a Paddle `MmapStorage`, or an open file handle for pread.
- [Loader, L755](https://github.com/huggingface/safetensors/blob/main/bindings/python/src/engine/unix.rs#L755) is the background engine behind `safe_open.prefetch`: worker threads read planned byte spans into pinned slabs and stage them into CUDA buffers.
- [save_file, L288](https://github.com/huggingface/safetensors/blob/main/bindings/python/py_src/safetensors/torch.py#L288) is the most-used port API; it flattens tensors to raw pointers and delegates to the Rust serializer.

## The 8-Byte Header That Kills Pickle

A safetensors file is `N` (8 bytes, little-endian) + N bytes of JSON + raw tensor data. The JSON is a dict keyed by tensor name, where each value is a [TensorInfo, L799](https://github.com/huggingface/safetensors/blob/main/safetensors/src/tensor.rs#L799) holding `dtype`, `shape`, and `data_offsets` — a `(begin, end)` pair into the data section. An optional `__metadata__` key carries free-form string annotations. The README states the dict layout precisely at [the header grammar, L80](https://github.com/huggingface/safetensors/blob/main/safetensors/README.md#L80).

Two implementation details are worth noticing. First, when writing, the serializer pads the JSON header with spaces to the next multiple of 8 bytes, so the data section starts at an aligned offset; you can see this in [serialize, L268](https://github.com/huggingface/safetensors/blob/main/safetensors/src/tensor.rs#L268) and its file-based sibling [serialize_to_file, L348](https://github.com/huggingface/safetensors/blob/main/safetensors/src/tensor.rs#L348), which streams through a buffered writer and a temp-file rename so existing mmaps never see a truncated file. Second, the [Dtype enum, L811](https://github.com/huggingface/safetensors/blob/main/safetensors/src/tensor.rs#L811) has 22 variants ranging from `BOOL` and 16-bit floats to FP8 variants and 4-bit `F4`, each reporting its width through [bitsize, L867](https://github.com/huggingface/safetensors/blob/main/safetensors/src/tensor.rs#L867). Sub-byte dtypes are the reason bitsize exists: a 4-bit tensor's byte length is no longer `n_elements * size_in_bytes`, and several APIs had to grow `bitsize`-aware arithmetic.

Compared with pickle-based checkpoints, this header buys three concrete properties: loading is language-neutral (the format is trivially parseable in C, Rust, Go or JavaScript), loading is partial (you can mmap the file, read one tensor's span, and touch nothing else), and loading is safe (the header cannot contain code). The [Transformers docs and our PyTorch tour](/pytorch/) cover the migration story from `.bin` files; here we care about how the container enforces its own safety.

## Validation Before a Single Byte Moves

The Rust core treats every input as hostile. The [SafeTensorError enum, L16](https://github.com/huggingface/safetensors/blob/main/safetensors/src/tensor.rs#L16) lists the failure modes: header too small, header too large, invalid UTF-8, invalid JSON, invalid header length, tensor not found, invalid offsets, and incomplete buffer coverage. The parse path in `read_metadata` uses checked arithmetic throughout — `checked_add` on the header end, `try_into` on the u64 length — so a malformed length cannot wrap around into a bogus slice.

The deepest check is [Metadata::validate, L627](https://github.com/huggingface/safetensors/blob/main/safetensors/src/tensor.rs#L627). It walks tensors in offset order and requires that each tensor starts exactly where the previous one ended, that `end >= start`, that the element count times bitsize (computed with `checked_mul`) matches the declared span, and it returns the final offset so the caller can confirm the header plus data exactly covers the file. If the last tensor's end does not land on the buffer end, deserialization fails with `MetadataIncompleteBuffer`. This is what makes the zero-copy guarantee sound: a view handed to a framework can never point outside the file, overlap a neighbor, or disagree with its declared shape.

The JSON itself is parsed into [HashMetadata, L540](https://github.com/huggingface/safetensors/blob/main/safetensors/src/tensor.rs#L540), a serde helper with `__metadata__` skipped when absent and tensor entries flattened into a map, then converted into the ordered `Metadata` struct at [Metadata, L532](https://github.com/huggingface/safetensors/blob/main/safetensors/src/tensor.rs#L532). The repo also carries an `attacks/` directory full of adversarial scripts (DoS-style giant tensors, aliasing tricks against torch and TF loaders), and a fuzz target at [fuzz_target!, L6](https://github.com/huggingface/safetensors/blob/main/safetensors/fuzz/fuzz_targets/fuzz_target_1.rs#L6) that feeds random bytes to `deserialize` — validation is not an afterthought here, it is the product.

## The Zero-Copy Path: mmap, UntypedStorage, pread

Opening a file through `safe_open` constructs an [Open, L752](https://github.com/huggingface/safetensors/blob/main/bindings/python/src/lib.rs#L752) state object that picks one of the strategies in the [Backend enum, L402](https://github.com/huggingface/safetensors/blob/main/bindings/python/src/lib.rs#L402): `mmap` (map the file copy-on-read and serve views into it) or `pread` (keep the handle open and read exact byte ranges on demand). The [Framework enum, L436](https://github.com/huggingface/safetensors/blob/main/bindings/python/src/lib.rs#L436) and [Device enum, L487](https://github.com/huggingface/safetensors/blob/main/bindings/python/src/lib.rs#L487) are parsed from Python strings like `"pt"`, `"cuda:0"`, `"mps"` via [parse_device, L520](https://github.com/huggingface/safetensors/blob/main/bindings/python/src/lib.rs#L520), and the cross product decides what "zero-copy" means:

- numpy-style frameworks get a view of the mmap handed to `np.frombuffer`-style construction, zero copies.
- PyTorch (1.11+) gets the best path: the file is re-opened through `torch.UntypedStorage.from_file`, so the mmap is owned by torch itself, and each tensor is a `torch.asarray(...).view(dtype)` over a byte slice of that storage — the returned tensor literally aliases the file's page cache.
- Paddle 3.1.1+ mirrors that with `paddle.MmapStorage`.
- For CUDA/MPS destinations, bytes must move, and the code moves them with intent: [parallel_pread, L2377](https://github.com/huggingface/safetensors/blob/main/bindings/python/src/lib.rs#L2377) reads ranges concurrently via [read_exact_at, L2343](https://github.com/huggingface/safetensors/blob/main/bindings/python/src/lib.rs#L2343), with macOS page-cache bypass via [disable_page_cache_macos, L2321](https://github.com/huggingface/safetensors/blob/main/bindings/python/src/lib.rs#L2321).

The prefetch engine generalizes the direct-to-GPU idea. `safe_open.prefetch(plan)` builds a [LoadPlan, L200](https://github.com/huggingface/safetensors/blob/main/bindings/python/src/engine/unix.rs#L200) of disjoint byte [Spans, L193](https://github.com/huggingface/safetensors/blob/main/bindings/python/src/engine/unix.rs#L193), runs [worker, L885](https://github.com/huggingface/safetensors/blob/main/bindings/python/src/engine/unix.rs#L885) threads that chunk reads into pinned staging slabs, and stages results into [CudaBuffer, L296](https://github.com/huggingface/safetensors/blob/main/bindings/python/src/engine/unix.rs#L296) allocations through the hand-rolled driver API in [CudaApi, L82](https://github.com/huggingface/safetensors/blob/main/bindings/python/src/engine/cuda.rs#L82). Device buffers then reach Python through DLPack: [to_capsule, L233](https://github.com/huggingface/safetensors/blob/main/bindings/python/src/dlpack.rs#L233) wraps a `DLManagedTensor` whose data pointer trait is implemented for CUDA buffers and for shared Metal buffers in [MTLBuffer, L61](https://github.com/huggingface/safetensors/blob/main/bindings/python/src/metal.rs#L61) — the same mechanism Apple's [MLX](/mlx/) stack consumes. The result is `for name, tensor in loader:` yielding ready CUDA tensors while the disk reads continue in the background.

## Slicing Without Loading

A lesser-known API makes even partial reads lazy. `f.get_slice("name")` returns a [PySafeSlice, L1908](https://github.com/huggingface/safetensors/blob/main/bindings/python/src/lib.rs#L1908), a handle carrying the tensor's [TensorMeta](https://github.com/huggingface/safetensors/blob/main/bindings/python/src/lib.rs#L1893) — dtype, shape, offsets — without touching data. Indexing it invokes [\_\_getitem\_\_, L2102](https://github.com/huggingface/safetensors/blob/main/bindings/python/src/lib.rs#L2102), which parses arbitrary Python indexing (ints, slices, ellipsis) via [parse_indexers, L307](https://github.com/huggingface/safetensors/blob/main/bindings/python/src/lib.rs#L307) into Rust [TensorIndexer, L58](https://github.com/huggingface/safetensors/blob/main/safetensors/src/slice.rs#L58) values, then computes the touched byte ranges with the [SliceIterator, L275](https://github.com/huggingface/safetensors/blob/main/safetensors/src/slice.rs#L275). For sub-byte dtypes, [slice_byte_ranges, L335](https://github.com/huggingface/safetensors/blob/main/safetensors/src/slice.rs#L335) handles the alignment puzzles. Only the sliced regions are materialized — useful for sharded inference where you want one row of a giant embedding table. The type stub published to users documents all of this at [class safe_open, L213](https://github.com/huggingface/safetensors/blob/main/bindings/python/py_src/safetensors/__init__.pyi#L213).

## From Install to Your First Model

Install the wheel (the Rust extension is prebuilt; the build itself is maturin plus PyO3 configured at [the maturin section, L94](https://github.com/huggingface/safetensors/blob/main/bindings/python/pyproject.toml#L94) and pinned at [pyo3 with abi3-py310, L13](https://github.com/huggingface/safetensors/blob/main/bindings/python/Cargo.toml#L13), one ABI for all Python 3.10+):

```bash
pip install safetensors
```

Round-trip a tensor dict with the torch port, then read it back lazily:

```python
import torch
from safetensors.torch import save_file, load_file
from safetensors import safe_open

tensors = {"weight": torch.randn(512, 1024), "bias": torch.randn(512)}
save_file(tensors, "model.safetensors", metadata={"format": "pt"})

loaded = load_file("model.safetensors")            # zero-copy mmap path

with safe_open("model.safetensors", framework="pt") as f:
    print(f.keys())                                # tensor names from the header
    meta = f.get_tensor_meta("weight")             # dtype/shape/offsets, no data read
    row = f.get_slice("weight")[0, :64]            # lazy slice, only 256 bytes touched
```

`save_file` refuses to silently duplicate shared storages; the torch port detects aliased tensors with [_find_shared_tensors, L76](https://github.com/huggingface/safetensors/blob/main/bindings/python/py_src/safetensors/torch.py#L76) and resolves keep-names via [_remove_duplicate_names, L97](https://github.com/huggingface/safetensors/blob/main/bindings/python/py_src/safetensors/torch.py#L97) — the same machinery behind `save_model` and `load_model`. The numpy port at [save_file, L61](https://github.com/huggingface/safetensors/blob/main/bindings/python/py_src/safetensors/numpy.py#L61) is the reference implementation every other port imitates; JAX, TensorFlow, MLX and Paddle ports are thin numpy bridges ([flax, L40](https://github.com/huggingface/safetensors/blob/main/bindings/python/py_src/safetensors/flax.py#L40), [tensorflow, L41](https://github.com/huggingface/safetensors/blob/main/bindings/python/py_src/safetensors/tensorflow.py#L41), [mlx, L41](https://github.com/huggingface/safetensors/blob/main/bindings/python/py_src/safetensors/mlx.py#L41), [paddle, L51](https://github.com/huggingface/safetensors/blob/main/bindings/python/py_src/safetensors/paddle.py#L51)). For bulk conversion of existing checkpoints, the repo ships [convert_file, L216](https://github.com/huggingface/safetensors/blob/main/bindings/python/convert.py#L216), which rewrites pickle checkpoints in place.

## Honest Limits

- The header is not zero-copy, by the author's own admission at [design notes, L158](https://github.com/huggingface/safetensors/blob/main/safetensors/README.md#L158): JSON is parsed eagerly on open. For a 7B model the header is kilobytes, so this is fine — but a file with millions of tiny tensors pays a real parse cost.
- The 100MB header cap at [MAX_HEADER_SIZE, L10](https://github.com/huggingface/safetensors/blob/main/safetensors/src/tensor.rs#L10) is a hard ceiling; files with pathological tensor counts cannot be opened at all.
- No compression, no encryption, no chunked/tiled layout, no checksums. Safetensors is deliberately a dumb container; if you need quantized block layouts or shard manifests you build them above it — [GGUF's approach](/ggml/) in the llama.cpp world packs more into the file, at the cost of a much bigger spec.
- Sub-byte dtypes error out on unaligned operations ([MisalignedSlice](https://github.com/huggingface/safetensors/blob/main/safetensors/src/tensor.rs#L16)) rather than silently rounding.
- Prefetch-to-GPU is Unix-only today; the [Windows Loader stub, L54](https://github.com/huggingface/safetensors/blob/main/bindings/python/src/engine/windows.rs#L54) returns an error, and CUDA staging requires the hand-rolled driver bindings.
- Saving requires dense, contiguous tensors; sparse or strided views must be materialized first, and concurrent mutation of tensors during `save_file` can corrupt output (the docstring warns about exactly this).

The takeaway: safetensors wins by being small. Eight bytes of preamble, a validated JSON header, and the discipline to never copy a byte you can alias. That is a good design lesson for anyone building infrastructure around [inference runtimes](/inference/) — the fastest code is the load you never do.

<script type="application/ld+json">
{
  "@context": "https://schema.org",
  "@type": "TechArticle",
  "headline": "Safetensors Source Tour: The Zero-Copy Weight Container",
  "description": "How Hugging Face's safetensors stores model weights: an 8-byte header, validated JSON metadata, a no_std Rust core, and zero-copy mmap, pread and DLPack paths through PyO3 into every major framework.",
  "author": {"@type": "Person", "name": "PyShine"},
  "datePublished": "2026-10-20",
  "keywords": "safetensors, serialization, mmap, zero-copy, Rust, Hugging Face, model weights",
  "publisher": {"@type": "Organization", "name": "PyShine"}
}
</script>
