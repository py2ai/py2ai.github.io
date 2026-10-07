---
layout: post
title: "Backburner: Your iPhone Becomes a 27B Model Co-Processor - Inside StayLameBro/backburner"
description: "A source-level tour of Backburner, the project that plugs an iPhone into a MacBook over USB-C and splits a 27B language model across both: staged split prefill, phone-held context past 64k, Neural Engine attention pages, and an OpenAI-compatible server."
date: 2026-10-07
header-img: "img/post-bg.jpg"
permalink: /backburner-iphone-mac-llm/
featured-img: "ai-coding-frameworks/ai-coding-frameworks"
image: https://pyshine.com/assets/img/diagrams/backburner/staylamebro-backburner-architecture.svg
tags: [LLM, Apple Silicon, Inference, On-Device AI]
categories: [AI, Open Source]
keywords: Backburner, iPhone LLM inference, split prefill, llama.cpp fork, Neural Engine, Metal, SME2, speculative decoding, USB-C, Qwen3.8-27B
author: "PyShine"
---

Every Apple Silicon laptop owner eventually hits the same wall: a 24 GB MacBook can run serious open-weight models, but long agent sessions and long contexts do not fit comfortably, and there is a very fast computer sitting in your pocket doing nothing. Backburner, published by StayLameBro, turns that pocket computer into a co-processor. Plug your iPhone into your MacBook with a 10 Gb/s USB-C cable and the two of them run Qwen3.8-27B together: the Mac handles most of the model, and the iPhone runs the rest on its GPU while also holding the oldest part of a very long context in its memory.

The headline capabilities are concrete. For prefill, every batch of prompt tokens is split so the Mac runs layers 1-40 and the iPhone runs layers 41-64 on its GPU, staged over the cable, which the project measures as 29-44% faster prefill at 16k-48k context when an agent reads a file or tool result of more than about 512 tokens. For context, the Mac alone fits 64k tokens of 8-bit context next to the model; the iPhone holds the oldest KV pages past that and computes attention over them, which the server sizes from the phone's free memory at startup, reaching 196k-229k tokens at 8-bit on an iPhone 17 Pro Max. And for correctness, greedy output is token-identical with and without the phone in the project's own tests.

This repository is the orchestration half of the system: the iPhone app under [ios/Backburner/](https://github.com/StayLameBro/backburner/tree/main/ios/Backburner), the on-device attention kernels in [phone-attn/](https://github.com/StayLameBro/backburner/tree/main/phone-attn), and the Mac-side scripts in [scripts/](https://github.com/StayLameBro/backburner/tree/main/scripts). The inference engine itself is a llama.cpp fork maintained separately, and it contributes its own Mac-only speedups: SME2 CPU co-attention, Metal fusions, and DFlash2 speculative decoding. This tour walks the repository, explains the division of labor, and shows how to set it up.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/backburner/staylamebro-backburner-overview-architecture.svg" alt="Architecture overview of the Backburner repository, showing the iPhone app with its RPC bridge and tunnels, the on-device attention kernels, the Mac server launcher with the prompt cache and SME2 kernels, and the installer and security suite" style="max-width:100%;">
</div>
<p><em>Architecture overview of the Backburner repository, from the iPhone app to the Mac-side server scripts.</em></p>

Reading the overview from left to right:

- The **iPhone app** centers on the interface in [ios/Backburner/Sidecar/ContentView.swift](https://github.com/StayLameBro/backburner/blob/main/ios/Backburner/Sidecar/ContentView.swift) and the RPC bridge in [ios/Backburner/Sidecar/RPCBridge.mm](https://github.com/StayLameBro/backburner/blob/main/ios/Backburner/Sidecar/RPCBridge.mm), which speaks to the Mac through the USB-C tunnel in [ios/Backburner/Sidecar/Tunnel.swift](https://github.com/StayLameBro/backburner/blob/main/ios/Backburner/Sidecar/Tunnel.swift).
- The **on-device kernels** live in [phone-attn/](https://github.com/StayLameBro/backburner/tree/main/phone-attn): the GPU matrix-unit kernel in [phone-attn/pa-metal.mm](https://github.com/StayLameBro/backburner/blob/main/phone-attn/pa-metal.mm) and the Neural Engine pages in [phone-attn/pa-ane.mm](https://github.com/StayLameBro/backburner/blob/main/phone-attn/pa-ane.mm), both speaking the wire protocol declared in [phone-attn/phone-attn.h](https://github.com/StayLameBro/backburner/blob/main/phone-attn/phone-attn.h).
- The **Mac host** is driven by [scripts/serve.sh](https://github.com/StayLameBro/backburner/blob/main/scripts/serve.sh), which launches the OpenAI-compatible server, arms the SSD prompt cache in [scripts/proxy.py](https://github.com/StayLameBro/backburner/blob/main/scripts/proxy.py), and coordinates the SME2 co-attention helper in [scripts/sme/sme_attn.c](https://github.com/StayLameBro/backburner/blob/main/scripts/sme/sme_attn.c).
- The **setup and security** tooling covers the one-command installer in [install.sh](https://github.com/StayLameBro/backburner/blob/main/install.sh) and the security test suite in [tests/security/run.sh](https://github.com/StayLameBro/backburner/blob/main/tests/security/run.sh).

## Why You Need This

The first reason is latency you can feel. When a coding agent reads a source file or a tool result into its session, it must re-read the whole context window through prefill, and at tens of thousands of tokens that wait dominates every interaction. Backburner attacks exactly that: the phone's half of the model runs on the iPhone's GPU while the Mac starts the next micro-batch, so the two devices overlap. In the project's turn benchmark, reading a 2,000-token file into a 16k session went from 109 tokens per second on the Mac alone to 157 with the phone, and a real 27k-token agent session that took 228 seconds cold on the Mac alone finished in 168 seconds with the phone.

The second reason is context depth. Past the Mac's 64k cells at 8-bit, a Mac-only setup must drop to 4-bit precision to reach 128k, and precision loss in the context is a quiet quality tax. With the phone holding the oldest KV pages, the total context stays at 8-bit and stretches to 196k-229k tokens depending on the phone's free memory, tested end to end to 128k. The project's long-context test recalled planted facts at positions 1.5k, 40k, and 100k with the phone's thermal state nominal.

The third reason is honest engineering. The README keeps its claims testable: greedy outputs are compared token by token with and without the phone, benchmark rows are committed under [bench/results/](https://github.com/StayLameBro/backburner/tree/main/bench/results), the compare tool in [tools/neo-air/](https://github.com/StayLameBro/backburner/tree/main/tools/neo-air) checks split execution against a reference run, and the limits section says plainly what does not work, including that small reads stay on the Mac and that the phone must stay in the foreground during long-context work. That kind of measurement culture is rarer than it should be.

## How It Works

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/backburner/staylamebro-backburner-architecture.svg" alt="Detailed architecture of the Backburner repository, showing the iPhone app internals, the on-device kernels and their protocol, Mac host scripts, model preparation tools, build and benchmark tooling, and the security suite" style="max-width:100%;">
</div>
<p><em>Detailed architecture of the Backburner repository, including model preparation, benchmarks, and the security suite.</em></p>

### Understanding the Architecture

**Split prefill overlaps the two devices.** The Mac engine runs layers 1-40 of each 256-token micro-batch and streams the residual to the phone, which runs layers 41-64 on its GPU while the Mac starts the next micro-batch. The phone keeps a mirror of its layers' KV rows and recurrent state, so only new rows cross the cable, and the last micro-batch of each batch runs on the Mac so outputs stay local. On the A19 Pro the phone's layers use the GPU's matrix units through Metal 4 tensor operations, which the project measures as 2.4x faster than without them. The phone side of this conversation is implemented in the RPC bridge in [ios/Backburner/Sidecar/RPCBridge.mm](https://github.com/StayLameBro/backburner/blob/main/ios/Backburner/Sidecar/RPCBridge.mm), with the app logic in [ios/Backburner/Sidecar/ContentView.swift](https://github.com/StayLameBro/backburner/blob/main/ios/Backburner/Sidecar/ContentView.swift).

**Phone-held context extends past 64k.** Beyond the Mac's capacity, the oldest KV pages, 4,096 keys each, move to the phone. Each attention step sends the queries to the phone and merges its partial result with the Mac's, using the protocol in [phone-attn/phone-attn.h](https://github.com/StayLameBro/backburner/blob/main/phone-attn/phone-attn.h). The GPU kernel in [phone-attn/pa-metal.mm](https://github.com/StayLameBro/backburner/blob/main/phone-attn/pa-metal.mm) computes attention over the held keys, and while the phone holds keys, 512-token micro-batches run as two staggered halves so Mac and phone overlap, lifting prefill at 140k from 58 to 68 tokens per second. Two phones can even share the old pages, as documented in [docs/TWO-PHONES.md](https://github.com/StayLameBro/backburner/blob/main/docs/TWO-PHONES.md).

**The Neural Engine takes over static work.** Old keys never change, so each 16,384-key page of a layer is compiled into a Neural Engine model with the keys and values baked in as weights, built by [phone-attn/ane-kv/build.py](https://github.com/StayLameBro/backburner/blob/main/phone-attn/ane-kv/build.py) and verified by [phone-attn/ane-kv/check_real.py](https://github.com/StayLameBro/backburner/blob/main/phone-attn/ane-kv/check_real.py). While writing tokens past 64k, the Neural Engine takes part of each old-key attention call and the GPU takes the rest, cutting per-token attention time at 140k from 279 to 176 milliseconds. The launcher in [scripts/serve.sh](https://github.com/StayLameBro/backburner/blob/main/scripts/serve.sh) pushes the page template to the phone the first time it sees it, and the measured trade-offs, including the two uses the project tried and set aside, are written up in [docs/ANE.md](https://github.com/StayLameBro/backburner/blob/main/docs/ANE.md).

**The Mac contributes its own kernels.** The SME2 helper in [scripts/sme/sme_attn.c](https://github.com/StayLameBro/backburner/blob/main/scripts/sme/sme_attn.c) lets the M4's SME units take about 30% of the rows of each big prefill matmul while the GPU handles the rest, and past 40k keys they also take the oldest keys of each attention layer during decoding. Model preparation is scripted: [scripts/split-gguf.py](https://github.com/StayLameBro/backburner/blob/main/scripts/split-gguf.py) cuts the tail layers into a separate GGUF with no head, and [scripts/make-drafter.sh](https://github.com/StayLameBro/backburner/blob/main/scripts/make-drafter.sh) with [scripts/convert-dflash2.py](https://github.com/StayLameBro/backburner/blob/main/scripts/convert-dflash2.py) produce the DFlash2 draft model used for speculative decoding.

**Transport and security are one design.** The cable tunnel in [ios/Backburner/Sidecar/Tunnel.swift](https://github.com/StayLameBro/backburner/blob/main/ios/Backburner/Sidecar/Tunnel.swift) answers only the Mac, enforced by the policy header in [ios/Backburner/Sidecar/CableOnly.h](https://github.com/StayLameBro/backburner/blob/main/ios/Backburner/Sidecar/CableOnly.h), and Wi-Fi through [ios/Backburner/Sidecar/WifiTunnel.swift](https://github.com/StayLameBro/backburner/blob/main/ios/Backburner/Sidecar/WifiTunnel.swift) is only enabled for a Mac paired over the cable, through an encrypted, authenticated tunnel described in [docs/WIFI.md](https://github.com/StayLameBro/backburner/blob/main/docs/WIFI.md). The suite in [tests/security/run.sh](https://github.com/StayLameBro/backburner/blob/main/tests/security/run.sh) covers the cable policy in [tests/security/cable-policy-test.cpp](https://github.com/StayLameBro/backburner/blob/main/tests/security/cable-policy-test.cpp), Noise protocol test vectors, tunnel behavior, and the exposure checker in [scripts/check-phone-exposure.py](https://github.com/StayLameBro/backburner/blob/main/scripts/check-phone-exposure.py).

End to end, a request flows like this: an agent POSTs to the OpenAI-compatible endpoint that [scripts/serve.sh](https://github.com/StayLameBro/backburner/blob/main/scripts/serve.sh) exposes on port 8080. If the prompt matches a cached prefix, [scripts/proxy.py](https://github.com/StayLameBro/backburner/blob/main/scripts/proxy.py) restores it from the SSD in a fraction of a second. Otherwise the engine prefills, splitting each micro-batch across Mac and phone; past 64k, old KV pages live on the phone and every attention step consults it through the protocol. Decoding runs on the Mac with the draft model, and past 64k the phone's GPU and Neural Engine join each step for old-key attention. If the phone stops answering for 15 seconds, the server stops with a clear message rather than silently degrading.

## Advantages

- **Hardware you already own.** No new GPU purchase: an Apple Silicon Mac plus an iPhone 15 Pro or newer becomes a two-device inference system over the cable the phone ships with, provided it is a 10 Gb/s cable.
- **Faster agent loops.** 29-44% faster prefill at 16k-48k means every file read and tool result lands sooner, which compounds across an entire coding session.
- **Long context at full precision.** Up to 196k-229k tokens of 8-bit context by sizing from the phone's free memory, instead of dropping to 4-bit past 64k.
- **Bit-exact outputs.** Greedy decoding is token-identical with and without the phone in the project's tests, so the accelerator changes speed, not answers.
- **Layered security posture.** Cable-only by default, paired-MAC Wi-Fi tunnels, Noise test vectors, an exposure checker, and a secret-scanning pre-commit hook in [scripts/hooks/check-sensitive.py](https://github.com/StayLameBro/backburner/blob/main/scripts/hooks/check-sensitive.py).
- **Reproducible measurements.** Benchmarks, results, and the compare tooling are committed, and each setting in [scripts/serve.sh](https://github.com/StayLameBro/backburner/blob/main/scripts/serve.sh) documents the measurement that chose it.

## Benefits

- **Lower wait per interaction.** The SSD prompt cache restores a 27k-token session start in 0.3-5 seconds after the first time, so restarting an agent does not re-pay the cold start.
- **Privacy by locality.** The model, the context, and the KV pages never leave the two devices; nothing is advertised on the network.
- **Graceful degradation.** A phone failure during a read turns the phone off for 60 seconds and the batch reruns on the Mac, and PHONE=0 forces a Mac-only run for debugging.
- **Fits real agent workloads.** One request at a time matches how coding agents actually issue requests, and the memory loader keeps the model wired so macOS cannot page it out.
- **Documented limits.** The README states what stays on the Mac, what the phone cannot do past 64k, and what happens when the app leaves the foreground, so expectations match reality.
- **A fork that pays off on the Mac alone.** The SME2 kernels, Metal fusions, and DFlash2 speculative decoding speed up Mac-only operation too, so the investment is useful even unplugged.

## Usage

The one-command path downloads the engine and models, about 24 GB, and is safe to re-run:

```bash
curl -fsSL https://raw.githubusercontent.com/StayLameBro/backburner/main/install.sh | bash
backburner phone   # once per phone: plug in, open Backburner, copies the phone's half over the cable
backburner         # OpenAI-compatible server at http://127.0.0.1:8080/v1
```

To build step by step from source:

```bash
git clone --recursive https://github.com/StayLameBro/backburner && cd backburner

# the Mac engine
cmake -S llama.cpp -B llama.cpp/build-metal -DCMAKE_BUILD_TYPE=Release
cmake --build llama.cpp/build-metal --target llama-server llama-quantize -j

# the draft model for speculative decoding
huggingface-cli download z-lab/Qwen3.8-27B-DFlash2 --local-dir ~/Models/qwen38-27b-dflash2
scripts/make-drafter.sh ~/Models/qwen38-27b-dflash2 ~/Models/dflash2-v2-q4km-self16.gguf

# the phone's half of the model, copied over the cable
python3 scripts/split-gguf.py ~/Models/Qwen3.8-27B-IQ4_XS.gguf ~/Models/tail-iq4xs-L40-nohead.gguf -L 40 --no-head
scripts/phone-tail.sh L40

# run with or without the phone
scripts/serve.sh
PHONE=0 scripts/serve.sh      # the Mac alone
```

The first run with the phone pushes the Neural Engine page model and relaunches the app, about a minute. After that, the startup line should read split prefill on, remote KV on, and ANE pages on. You need an Apple Silicon Mac, an iPhone 15 Pro or newer, a 10 Gb/s USB-C cable, and an Apple ID to install the app through AltStore with no developer account.

## Conclusion

Backburner is a reminder that the most interesting systems-level AI projects are no longer coming only from large labs. With a forked inference engine, an iPhone app, GPU and Neural Engine kernels written from scratch, a wire protocol over USB-C, and a security suite to keep the phone from listening to anyone but your Mac, this repository treats a consumer phone as a first-class accelerator and proves each claim with committed benchmarks. If you run local models on a Mac and have an iPhone in your pocket, the marginal hardware cost is a cable, and the payoff is a faster, longer-context coding agent.

Links:

- Repository: [https://github.com/StayLameBro/backburner](https://github.com/StayLameBro/backburner)
- Engine fork: [https://github.com/StayLameBro/backburner-llama.cpp](https://github.com/StayLameBro/backburner-llama.cpp)
- iPhone install guide: [docs/INSTALL-IPHONE.md](https://github.com/StayLameBro/backburner/blob/main/docs/INSTALL-IPHONE.md)
- License: MIT
