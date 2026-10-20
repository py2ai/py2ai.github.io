---
layout: post
title: "ONNX Runtime GenAI Source Tour: One C++ Core Wearing Two Hats"
description: "A guided walk through the microsoft/onnxruntime-genai repository: the C and Python bindings, the model layer that wraps ONNX Runtime sessions, the Generators loop with greedy and top-k/top-p sampling, and the newer serving engine with dynamic batching, paged KV caches, and prefix reuse."
date: 2026-10-19
header-img: "img/post-bg.jpg"
permalink: /onnxruntime-genai/
featured-img: "ai-coding-frameworks/ai-coding-frameworks"
image: https://pyshine.com/assets/img/diagrams/onnxruntime-genai/microsoft-onnxruntime-genai-overview-architecture.svg
tags: [LLM, Python, Open Source]
categories: [AI Inference]
keywords: [ONNX Runtime GenAI, onnxruntime-genai, OgaCreateModel, GeneratorParams, KV cache, continuous batching, paged cache, prefix cache, Microsoft, LLM inference]
author: "PyShine"
---

Most repositories we have toured on this site do one job well. [ONNX](/onnx/) defines a model format. [llama.cpp](/llama.cpp-2/) runs quantized matrices on a laptop. [vLLM](/vllm-2/) serves fleets of requests. **onnxruntime-genai**, Microsoft's generative-AI layer over [ONNX Runtime](/onnxruntime-2/), is unusual because it deliberately does two jobs at once, inside one C++ core. The first job is compact and stable: the *Generators API*, a library you embed into a desktop app, a phone, or a game engine to run a single chat session end to end. The second job is newer and fast-moving: a *serving engine* with request objects, continuous batching, paged KV caches, and prefix reuse — the same class of problems [Triton Inference Server](/triton-server/) or a dedicated inference stack solves, but rebuilt inside a library that still has to fit next to a DirectML GPU on a consumer laptop.

The dual identity is visible the moment you open `src/`. There is a `generator/` folder containing the classic single-session loop, and an `engine/` folder containing a scheduler, a cache manager, and request bookkeeping that could just as plausibly front a small service. Both compile into the same `onnxruntime_genai` binary and share the model layer underneath, so a model converted once can be driven either way. Where [KTransformers](/ktransformers/) grafts experts onto existing runtimes and [MIGraphX](/migraphx/) is a graph compiler with a serving shell, this project is a runtime consumer with two separate control planes — that seam is what makes the source worth a tour.

The map below follows the bindings at the top, down through the shared model layer, and then forks into the generator loop on one side and the serving engine on the other.

<div style="overflow-x:auto;"><img src="https://pyshine.com/assets/img/diagrams/onnxruntime-genai/microsoft-onnxruntime-genai-overview-architecture.svg" alt="onnxruntime-genai overview architecture" style="max-width:100%;"></div>

<p><em>Overview of onnxruntime-genai: C and Python bindings wrap a shared model layer (config, sessions, staged execution, logits and KV inputs); the Generators API runs one sampling loop while the engine side adds requests, a scheduler, and static/paged cache managers; the Python model builder converts Hugging Face checkpoints for both.</em></p>

Reading the overview from left to right:

- The C entry point is [OgaCreateModel, L489](https://github.com/microsoft/onnxruntime-genai/blob/main/src/ort_genai_c.cpp#L489) and its partner [OgaCreateGenerator, L585](https://github.com/microsoft/onnxruntime-genai/blob/main/src/ort_genai_c.cpp#L585) — two calls that stand in for the entire embedding surface.
- The Python binding mirrors them in [python.cpp, L656](https://github.com/microsoft/onnxruntime-genai/blob/main/src/python/python.cpp#L656), where `generate_next_token` is exported to the `onnxruntime_genai` module.
- Every model is described by [Config, L16](https://github.com/microsoft/onnxruntime-genai/blob/main/src/config.h#L16), parsed from the JSON the builder emits.
- The shared model layer centers on [Model, L148](https://github.com/microsoft/onnxruntime-genai/blob/main/src/models/model.h#L148), whose [CreateState, L156](https://github.com/microsoft/onnxruntime-genai/blob/main/src/models/model.h#L156) manufactures per-request state.
- The Generators control plane is [Generator, L126](https://github.com/microsoft/onnxruntime-genai/blob/main/src/generator/generators.h#L126) driven by [GeneratorParams, L86](https://github.com/microsoft/onnxruntime-genai/blob/main/src/generator/generators.h#L86).
- Sampling policy lives behind [SamplingMethod, L194](https://github.com/microsoft/onnxruntime-genai/blob/main/src/generator/generators.h#L194) with greedy, top-k, and top-p flavors.
- The serving control plane starts at [Engine, L129](https://github.com/microsoft/onnxruntime-genai/blob/main/src/engine/engine.h#L129), which owns a [Scheduler, L21](https://github.com/microsoft/onnxruntime-genai/blob/main/src/engine/scheduler.h#L21) and a [CacheManager, L53](https://github.com/microsoft/onnxruntime-genai/blob/main/src/engine/cache_manager.h#L53).
- Dynamic request admission is decided in [DynamicBatchScheduler::Schedule, L164](https://github.com/microsoft/onnxruntime-genai/blob/main/src/engine/scheduler.cpp#L164).
- Checkpoint conversion is a Python-side tool whose CLI lives at [builder.py `__main__`, L1248](https://github.com/microsoft/onnxruntime-genai/blob/main/src/python/py/models/builder.py#L1248).

## Two Doors into One Core

The library's public faces are deliberately thin. The C ABI in `ort_genai_c.cpp` is the contract other languages bind to: [OgaCreateModelWithRuntimeSettings, L351](https://github.com/microsoft/onnxruntime-genai/blob/main/src/ort_genai_c.cpp#L351) accepts raw JSON for hosts that want full control, [OgaCreateModelFromConfig, L478](https://github.com/microsoft/onnxruntime-genai/blob/main/src/ort_genai_c.cpp#L478) takes a prebuilt `Config`, and [OgaCreateGeneratorParams, L507](https://github.com/microsoft/onnxruntime-genai/blob/main/src/ort_genai_c.cpp#L507) produces the parameter bag a generator consumes. Every call returns an opaque handle plus an error string; there is no C++ object crossing the boundary, which is what makes the same binary embeddable from C#, Rust, Java, or Python without an ABI fight.

The pybind11 surface in [python.cpp](https://github.com/microsoft/onnxruntime-genai/blob/main/src/python/python.cpp#L656) is a direct reflection of that ABI — `generate_next_token`, `is_done`, token getters, and knobs like [set_max_generated_tokens, L855](https://github.com/microsoft/onnxruntime-genai/blob/main/src/python/python.cpp#L855). What is more interesting than the bindings is the lifecycle discipline behind them. The single global ORT environment is lazily created by `GetOrtGlobals()` and torn down by [Shutdown, L171](https://github.com/microsoft/onnxruntime-genai/blob/main/src/generator/generators.cpp#L171), guarded by a mutex so a host that re-initializes the library in-process cannot double-construct it. Leak checking is built in: typed instance counters dump any `Model`, `Generator`, or sequence that escaped cleanup, and the code even documents why the ORT environment itself must be deliberately leaked at process exit rather than destroyed during static teardown — a rare case of shutdown-ordering wisdom written directly into comments rather than tribal knowledge.

## The Model Layer: Sessions, States, and Stages

Beneath both control planes sits the model layer. [Config, L16](https://github.com/microsoft/onnxruntime-genai/blob/main/src/config.h#L16) is the parsed JSON describing providers, device settings, and per-model options — including a runtime profile that records device facts like [RuntimeProfileDeviceFacts, L882](https://github.com/microsoft/onnxruntime-genai/blob/main/src/config.h#L882) so later decisions do not have to re-probe hardware. Turning that config into compute happens in [model.cpp](https://github.com/microsoft/onnxruntime-genai/blob/main/src/models/model.cpp#L854): `CreateSessionOptions` assembles ORT session options from the config, and `CreateSession` opens the ONNX file — the same protobuf message whose [GraphProto](https://github.com/pytorch/executorch/blob/main/schema/onnx/onnx.in.proto#L564) we discussed in the [ONNX tour](/onnx/), executed here by ONNX Runtime rather than inspected by it.

The base class [Model, L148](https://github.com/microsoft/onnxruntime-genai/blob/main/src/models/model.h#L148) is architecture-agnostic on purpose. Its per-request counterpart, [State, L33](https://github.com/microsoft/onnxruntime-genai/blob/main/src/models/model.h#L33), holds the input and output bindings for exactly one sequence of generation. The decoder-only family — [DecoderOnly_Model, L14](https://github.com/microsoft/onnxruntime-genai/blob/main/src/models/decoder_only.h#L14) and [DecoderOnly_State, L23](https://github.com/microsoft/onnxruntime-genai/blob/main/src/models/decoder_only.h#L23) — is the workhorse for GPT-style checkpoints, wiring tokenizer preprocessing, the graph session, [Logits, L7](https://github.com/microsoft/onnxruntime-genai/blob/main/src/models/io/logits.h#L7), and the [KeyValueCache, L27](https://github.com/microsoft/onnxruntime-genai/blob/main/src/models/io/kv_cache.h#L27) input wrapper into one runnable unit. One structural detail deserves attention: execution is staged. Rather than firing the whole graph in a single `Run`, requests flow through [ExecutionParams, L30](https://github.com/microsoft/onnxruntime-genai/blob/main/src/models/graph_executor.h#L30) in ordered sub-runs — preprocessing, the model itself, logits processing — coordinated by a graph executor. (The file that stitches those stages together is named after the assembly-line word we would rather not repeat here, but you will find it one directory away.) Finally, [Sequences, L5](https://github.com/microsoft/onnxruntime-genai/blob/main/src/sequences.h#L5) is the humble container of generated token ids that everything else reads and appends; its simplicity is a feature, because every batch of complexity in this repo lands somewhere above it.

## The Generator Loop and Sampling

The Generators API is the oldest hat. Its header sets the vocabulary early — forward-declared structs at [L46-51](https://github.com/microsoft/onnxruntime-genai/blob/main/src/generator/generators.h#L46), then [GeneratorParams, L86](https://github.com/microsoft/onnxruntime-genai/blob/main/src/generator/generators.h#L86), [Generator, L126](https://github.com/microsoft/onnxruntime-genai/blob/main/src/generator/generators.h#L126), and an [Action, L127](https://github.com/microsoft/onnxruntime-genai/blob/main/src/generator/generators.h#L127) that composes steps like logits processors and samplers into a chain. The parameter object exposes two hats of its own: raw input tensors for C++ callers and convenience setters for Python. The loop body in [generators.cpp](https://github.com/microsoft/onnxruntime-genai/blob/main/src/generator/generators.cpp#L958) is refreshingly legible: run the model, pick or sample a token, append it via [AppendTokens, L744](https://github.com/microsoft/onnxruntime-genai/blob/main/src/generator/generators.cpp#L744), check stopping criteria, repeat.

Sampling policy is resolved once, not per step: [InitializeSamplingMethod, L693](https://github.com/microsoft/onnxruntime-genai/blob/main/src/generator/generators.cpp#L693) reads the search options and locks in one of the [SamplingMethod](https://github.com/microsoft/onnxruntime-genai/blob/main/src/generator/generators.h#L194) variants, with the greedy case spelled out at [IsGreedySampling, L690](https://github.com/microsoft/onnxruntime-genai/blob/main/src/generator/generators.cpp#L690) and its helpers nearby at [L659](https://github.com/microsoft/onnxruntime-genai/blob/main/src/generator/generators.cpp#L659) and [L677](https://github.com/microsoft/onnxruntime-genai/blob/main/src/generator/generators.cpp#L677). When context must be truncated, [RewindToLength, L980](https://github.com/microsoft/onnxruntime-genai/blob/main/src/generator/generators.cpp#L980) rewinds sequences and cache state together — an operation that sounds trivial until you realize every consumer of the KV cache has to agree on what "gone" means. Beyond plain decoding, the same folder hosts speculative execution: [SpeculativeDecodingModel, L14](https://github.com/microsoft/onnxruntime-genai/blob/main/src/models/speculative_decoding.h#L14) pairs a draft and a target model, and the decoding strategy layer ([StandardDecodingStrategy, L17](https://github.com/microsoft/onnxruntime-genai/blob/main/src/decoding/standard_decoding_strategy.h#L17)) decides how many tokens to attempt per step, which is precisely the bet that makes speculative decoding pay off when the draft is right and costs you when it is wrong.

## The Serving Engine: Scheduler, Cache, Prefix Reuse

The engine folder is where this repository stops being a library and starts being a server-in-waiting. Requests are first-class objects: [Request, L154](https://github.com/microsoft/onnxruntime-genai/blob/main/src/engine/request.h#L154) carries tokenized prompts plus per-request [RequestOptions, L37](https://github.com/microsoft/onnxruntime-genai/blob/main/src/engine/request.h#L37) and chat-style [TurnOptions, L46](https://github.com/microsoft/onnxruntime-genai/blob/main/src/engine/request.h#L46). The [Engine, L129](https://github.com/microsoft/onnxruntime-genai/blob/main/src/engine/engine.h#L129) ties a scheduler, an executor, and cache management together, and even exposes multi-token-prediction stepping through [PrepareMtpStep, L261](https://github.com/microsoft/onnxruntime-genai/blob/main/src/engine/engine.h#L261) — the same MTP idea we saw in the KTransformers and speculative corners of the ecosystem, here as a first-class scheduling primitive.

Admission control lives in [scheduler.cpp](https://github.com/microsoft/onnxruntime-genai/blob/main/src/engine/scheduler.cpp#L460): a factory ([Scheduler::Create, L460](https://github.com/microsoft/onnxruntime-genai/blob/main/src/engine/scheduler.cpp#L460)) picks a policy, [CreateScheduledRequests, L32](https://github.com/microsoft/onnxruntime-genai/blob/main/src/engine/scheduler.cpp#L32) turns waiting requests into a runnable batch, and two implementations differ exactly where you would expect. [StaticBatchScheduler::Schedule, L83](https://github.com/microsoft/onnxruntime-genai/blob/main/src/engine/scheduler.cpp#L83) admits a fixed group and waits for all of it, while [DynamicBatchScheduler::Schedule, L164](https://github.com/microsoft/onnxruntime-genai/blob/main/src/engine/scheduler.cpp#L164) admits continuously as capacity frees — the distinction that decides whether a chatty request waits for a slow neighbor. The batch itself is just bookkeeping over [ScheduledRequests, L64](https://github.com/microsoft/onnxruntime-genai/blob/main/src/engine/scheduled_requests.h#L64), and execution is delegated to [ModelExecutor, L43](https://github.com/microsoft/onnxruntime-genai/blob/main/src/engine/model_executor.h#L43) with a decoder-only specialization at [DecoderModelExecutor, L65](https://github.com/microsoft/onnxruntime-genai/blob/main/src/engine/model_executor.h#L65), speaking to models through the generic [Decoder, L69](https://github.com/microsoft/onnxruntime-genai/blob/main/src/engine/decoders/decoder.h#L69) interface and its [ModelIO, L18](https://github.com/microsoft/onnxruntime-genai/blob/main/src/engine/decoders/decoder.h#L18) / [DecoderIO, L41](https://github.com/microsoft/onnxruntime-genai/blob/main/src/engine/decoders/decoder.h#L41) payload structs.

Cache management is the engine's real substance. The [CacheManager, L53](https://github.com/microsoft/onnxruntime-genai/blob/main/src/engine/cache_manager.h#L53) interface answers two questions — [CanAllocate, L62](https://github.com/microsoft/onnxruntime-genai/blob/main/src/engine/cache_manager.h#L62) and [Allocate, L64](https://github.com/microsoft/onnxruntime-genai/blob/main/src/engine/cache_manager.h#L64) — and tracks per-step growth through [ReserveStep, L155](https://github.com/microsoft/onnxruntime-genai/blob/main/src/engine/cache_manager.h#L155) with explicit [CacheStepReservation, L17](https://github.com/microsoft/onnxruntime-genai/blob/main/src/engine/cache_manager.h#L17) receipts. Two implementations ship: a [StaticCacheManager, L166](https://github.com/microsoft/onnxruntime-genai/blob/main/src/engine/cache_manager.h#L166) that pre-reserves contiguous blocks per sequence, and a [PagedCacheManager, L199](https://github.com/microsoft/onnxruntime-genai/blob/main/src/engine/cache_manager.h#L199) backing [PagedKeyValueCache, L84](https://github.com/microsoft/onnxruntime-genai/blob/main/src/engine/paged_key_value_cache.h#L84) — the block-table trick that lets fragments of sequences share physical memory instead of each owning a rectangle of it. On top sits prefix reuse: [PrefixCache::Match, L99](https://github.com/microsoft/onnxruntime-genai/blob/main/src/engine/prefix_cache.cpp#L99) hashes each prompt into a chain of blocks (a [RootHash, L55](https://github.com/microsoft/onnxruntime-genai/blob/main/src/engine/prefix_cache.cpp#L55) plus per-block [ChainHash, L59](https://github.com/microsoft/onnxruntime-genai/blob/main/src/engine/prefix_cache.cpp#L59)), so a second conversation with the same system prompt reuses computed attention state, and [Register, L175](https://github.com/microsoft/onnxruntime-genai/blob/main/src/engine/prefix_cache.cpp#L175) publishes finished sequences for future matches. If you have read the [vLLM design story](/vllm-2/), none of these ideas are new to you — what is new is seeing them implemented in a codebase whose other half must also run inside a 300 MB mobile app budget.

## From Install to Your First Model

The quickest path is Python. Install the wheel, convert a checkpoint with the builder, and generate:

```bash
pip install onnxruntime-genai
python -m onnxruntime_genai.models.builder -m microsoft/Phi-3-mini-4k-instruct -o ./phi3-int4 -p int4 -e cpu
```

The builder CLI at [builder.py `__main__`, L1248](https://github.com/microsoft/onnxruntime-genai/blob/main/src/python/py/models/builder.py#L1248) wraps an argparse surface ([L863](https://github.com/microsoft/onnxruntime-genai/blob/main/src/python/py/models/builder.py#L863)) that downloads a Hugging Face model, optionally quantizes it, exports or re-links the ONNX graph, and writes the `genai_config.json` that `Config` parses at load time. Then generation is a five-liner:

```python
import onnxruntime_genai as og

model = og.Model("./phi3-int4")
tokenizer = og.Tokenizer(model)
stream = tokenizer.create_stream()

params = og.GeneratorParams(model)
params.set_search_options(max_length=256)
params.input_ids = tokenizer.encode("Explain a KV cache in one sentence.")[None, :]

generator = og.Generator(model, params)
while not generator.is_done():
    generator.generate_next_token()
    print(stream.decode(generator.get_next_tokens()[0]), end="", flush=True)
```

Every line maps to a section above: `og.Model` goes through the C ABI's create-model call, `GeneratorParams` and `Generator` mirror [OgaCreateGeneratorParams](https://github.com/microsoft/onnxruntime-genai/blob/main/src/ort_genai_c.cpp#L507) and [OgaCreateGenerator, L585](https://github.com/microsoft/onnxruntime-genai/blob/main/src/ort_genai_c.cpp#L585), and the while-loop is the `generate_next_token` binding. For C and C++ hosts the same shape holds with `Oga*` calls instead. If you want to watch a single machine juggle many conversations, the engine-side API (`CreateEngine`, `AddRequest`, step-and-decode) is where to point your experiments — it is the same core doing [continuous batching](/inference/) locally.

## Honest Limits

The two hats are not equally mature. The Generators API is battle-tested and documented; the engine side is visibly younger — interfaces still shift between releases, and some advanced scheduling paths carry TODOs, so treat it as a foundation to study rather than a stable serving contract. The builder covers the popular model families but not everything on the Hub; unusual architectures may export but need hand-edited config. Device support varies by provider: CPU, CUDA, DirectML, WebGPU, QNN, OpenVINO, RyzenAI, and AMD GPU paths all exist in the execution-provider interface layer, yet feature parity (flash attention variants, quantized matmul support, prefix-cache compatibility) differs per vendor, and the code is honest about falling back when a device cannot do continuous decoding. Documentation trails the source — reading headers is genuinely the best available reference.

That honesty is also the reason to read this repository. It shows, in one place, what it costs to serve the same [GGUF-free, ONNX-standard](/onnx/) models everywhere from a phone to a workstation, and lets you compare those trade-offs against [Triton's](/triton-server/) datacenter focus, [llama.cpp's](/llama.cpp-2/) minimalist single-binary philosophy, and [KTransformers'](/ktransformers/) heterogeneous expert placement. The format layer ([ONNX](/onnx/)) and the runtime layer ([ONNX Runtime](/onnxruntime-2/)) got their own tours; this is the layer where a standard runtime finally learned to talk.

<script type="application/ld+json">
{
  "@context": "https://schema.org",
  "@type": "TechArticle",
  "headline": "ONNX Runtime GenAI Source Tour: One C++ Core Wearing Two Hats",
  "description": "Guided source tour of microsoft/onnxruntime-genai: C and Python bindings, the shared model layer, the Generators sampling loop, and the serving engine with dynamic batching, paged KV caches, and prefix reuse.",
  "author": {"@type": "Person", "name": "PyShine"},
  "datePublished": "2026-10-19",
  "keywords": "onnxruntime-genai, ONNX Runtime, GeneratorParams, KV cache, continuous batching, paged cache, prefix cache, LLM inference",
  "publisher": {"@type": "Organization", "name": "PyShine"}
}
</script>
