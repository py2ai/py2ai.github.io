---
title: "The Best AMD Ryzen for Local AI in 2026: LLM Speed, Model Sizes, and Clusters, With Sources"
description: "Deep research into AMD Ryzen AI Max (Strix Halo) for local LLMs: real decode tok/s for gpt-oss-120b, Qwen3, Llama 4 Scout and dense 70B models, memory tiers from 32GB to 128GB, llama.cpp RPC clusters, fine-tuning, and 2026 prices, with a verified source for every claim."
permalink: /Best-AMD-Ryzen-For-Local-AI-2026-Deep-Research/
featured-img: ai-coding-frameworks/ai-coding-frameworks
image: https://pyshine.com/assets/img/diagrams/amd-ryzen-local-ai/pyshine-amd-ryzen-model-map.svg
author: PyShine
categories: [AI, Open Source]
tags: [AMD, Ryzen AI Max, Strix Halo, Local AI, LLM, llama.cpp, ROCm, Benchmarks]
---

If you have been following our local-AI hardware series - the [Mac mini deep dive](https://pyshine.com/Best-Mac-Mini-For-Local-AI-2026-Deep-Research/), the [Mac Studio deep dive](https://pyshine.com/Best-Mac-Studio-For-Local-AI-2026-Deep-Research/), and the [NVIDIA DGX Spark deep dive](https://pyshine.com/Best-DGX-Spark-For-Local-AI-2026-Deep-Research/) - this is the fourth and most price-aggressive chapter. AMD's Ryzen AI Max family, codenamed **Strix Halo**, puts up to 128GB of unified LPDDR5X memory and a 40-compute-unit Radeon 8060S GPU in mini PCs that start well under the price of a DGX Spark or a fully loaded Mac Studio. The question, as always: how fast does it actually run large language models, and how big can those models be?

Every number below carries a link to its source. We read AMD's official spec sheets, AMD's own 2026 benchmark blog, the community Strix Halo benchmark repos, Framework's published machine-learning figures, and storefront pricing pages, all crawled in September 2026.

![AMD Ryzen AI Max for local AI - 2026 picks by budget](https://pyshine.com/assets/img/diagrams/amd-ryzen-local-ai/pyshine-amd-ryzen-local-ai-picks.svg)

## What "Ryzen AI Max" Actually Is

Strix Halo comes in three main bins. The flagship is the [AMD Ryzen AI Max+ 395](https://www.amd.com/en/products/processors/laptop/ryzen/ai-300-series/amd-ryzen-ai-max-plus-395.html): 16 Zen 5 cores / 32 threads up to 5.1 GHz, Radeon 8060S graphics with 40 RDNA 3.5 compute units at up to 2900 MHz, a 50-TOPS XDNA 2 NPU (up to 126 total TOPS), and 256-bit LPDDR5X-8000 memory, 128GB maximum, with a configurable TDP of 45-120W. All of these figures come straight from [AMD's official specification page](https://www.amd.com/en/products/processors/laptop/ryzen/ai-300-series/amd-ryzen-ai-max-plus-395.html).

The step-down [Ryzen AI Max 385](https://www.amd.com/en/products/processors/laptop/ryzen/ai-300-series/amd-ryzen-ai-max-plus-395.html) pairs the same memory subsystem with 8 cores and 32 graphics cores, which is the configuration Framework sells as its 32GB entry tier at [$1,269](https://frame.work/products/desktop-diy-amd-aimax300/). AMD also sells its own reference box, the [Ryzen AI Halo developer platform](https://www.amd.com/en/products/processors/desktops/ryzen/ryzen-ai-halo/ryzen-ai-max-plus-395.html), a 150 x 150 x 45.4 mm machine with 128GB at 8000 MT/s, 256 GB/s of memory bandwidth, 10GbE, and a Linux-first ROCm software stack.

| Spec | Value | Source |
|---|---|---|
| CPU | 16x Zen 5, 32 threads, up to 5.1 GHz | [AMD specs](https://www.amd.com/en/products/processors/laptop/ryzen/ai-300-series/amd-ryzen-ai-max-plus-395.html) |
| GPU | Radeon 8060S, 40 CU RDNA 3.5 @ 2900 MHz | [AMD specs](https://www.amd.com/en/products/processors/laptop/ryzen/ai-300-series/amd-ryzen-ai-max-plus-395.html) |
| NPU | 50 TOPS XDNA 2 (126 TOPS total) | [AMD specs](https://www.amd.com/en/products/processors/laptop/ryzen/ai-300-series/amd-ryzen-ai-max-plus-395.html) |
| Memory | 256-bit LPDDR5X-8000, max 128GB | [AMD specs](https://www.amd.com/en/products/processors/laptop/ryzen/ai-300-series/amd-ryzen-ai-max-plus-395.html) |
| Memory bandwidth | 256 GB/s | [AMD Halo page](https://www.amd.com/en/products/processors/desktops/ryzen/ryzen-ai-halo/ryzen-ai-max-plus-395.html) |
| TDP | 45-120W configurable | [AMD specs](https://www.amd.com/en/products/processors/laptop/ryzen/ai-300-series/amd-ryzen-ai-max-plus-395.html) |
| VRAM allocation | Up to 96GB on Windows, ~115GB GTT pool on Linux | [Framework](https://frame.work/desktop?tab=machine-learning), [seehiong blog](https://seehiong.github.io/posts/2026/08/running-llama.cpp-on-amd-strix-halo/) |

## The 256 GB/s Rule

Just like the DGX Spark's 273 GB/s ([NVIDIA](https://www.nvidia.com/en-us/products/workstations/dgx-spark/)), Strix Halo's decode speed is governed by memory bandwidth. The theoretical 256 GB/s works out to roughly 215 GB/s measured in real inference runs - about a 16 percent gap - according to the community benchmark repository [visorcraft/strix-halo-llm-perf](https://github.com/visorcraft/strix-halo-llm-perf).

The consequence is identical to every other bandwidth-limited box in this series: **mixture-of-experts models with few active parameters fly, dense models crawl.** A 30B MoE with 3B active parameters decodes at 86-100 tok/s, while a dense 70B at the same quantization manages 5.1 tok/s - the whole dense weight matrix streams from memory for every single token.

| Model type | Decode speed | Source |
|---|---|---|
| MoE 30B-A3B (3B active) | 86-100 tok/s | [visorcraft](https://github.com/visorcraft/strix-halo-llm-perf), [RunAIHome](https://runaihome.com/blog/ryzen-ai-max-395-strix-halo-local-llm-2026/) |
| MoE 120B (5.1B active) | 53-56 tok/s | [visorcraft](https://github.com/visorcraft/strix-halo-llm-perf) |
| Dense 32B Q4 | ~10 tok/s | [llmrun](https://llmrun.dev/device/framework-desktop-128gb) |
| Dense 70B Q4_K_M | 5.1 tok/s | [RunAIHome](https://runaihome.com/blog/ryzen-ai-max-395-strix-halo-local-llm-2026/) |

## Decode Throughput: The Community Numbers

The single best public resource for this chip is the [strix-halo-llm-perf](https://github.com/visorcraft/strix-halo-llm-perf) GitHub repository, which benchmarks llama.cpp across GMKtec EVO-X2 and Beelink GTR9 Pro machines. Combined with [Framework's official LM Studio figures](https://frame.work/desktop?tab=machine-learning), [llmrun's device page](https://llmrun.dev/device/framework-desktop-128gb), [RunAIHome's 2026 analysis](https://runaihome.com/blog/ryzen-ai-max-395-strix-halo-local-llm-2026/), and [llamaperf's Strix Halo 128GB page](https://llamaperf.com/gpu/strix-halo-128gb), a consistent picture emerges on a 128GB box:

- **gpt-oss-20b**: 58 tok/s in Framework's official [LM Studio testing](https://frame.work/desktop?tab=machine-learning); the 8B-class LFM2.5 reaches [152.5 tok/s](https://llmrun.dev/device/framework-desktop-128gb).
- **Qwen3-30B-A3B**: 86.1 tok/s Vulkan ([visorcraft](https://github.com/visorcraft/strix-halo-llm-perf)) and up to 100.04 tok/s at IQ4_XS on RADV drivers ([RunAIHome](https://runaihome.com/blog/ryzen-ai-max-395-strix-halo-local-llm-2026/)). Qwen3-Coder-30B hits [96.8-98.5 tok/s](https://llmrun.dev/device/framework-desktop-128gb).
- **gpt-oss-120b**: the headline act. Its MXFP4 weights occupy [63.39 GB](https://community.frame.work/t/tracking-will-the-ai-max-395-128gb-be-able-to-run-gpt-oss-120b/73280), so it only fits on 128GB models. llama.cpp Vulkan runs deliver [53.4-55.57 tok/s](https://github.com/visorcraft/strix-halo-llm-perf); Framework's official figure is [38 tok/s](https://frame.work/desktop?tab=machine-learning); GMKtec's Ollama build manages [19.25 tok/s](https://www.gmktec.com/products/amd-ryzen%E2%84%A2-ai-max-395-evo-x2-ai-mini-pc). Your backend choice matters more than your brand of mini PC.
- **Qwen3-Coder-Next 80B-A3B**: [42.7 tok/s](https://llmrun.dev/device/framework-desktop-128gb). **Qwen3.6 35B Q6**: [50 tok/s](https://llmrun.dev/device/framework-desktop-128gb).
- **Qwen3.8 Flash-Next (~125B MoE)**: [38-45 tok/s with multi-token prediction](https://runaihome.com/blog/ryzen-ai-max-395-strix-halo-local-llm-2026/).
- **GLM-5.3-Flash**: [14.63 tok/s on ROCm vs 8.57 on Vulkan](https://runaihome.com/blog/ryzen-ai-max-395-strix-halo-local-llm-2026/) - the ROCm/Vulkan split flips depending on the model.
- **Frontier-adjacent MoE on one box**: MiniMax M2.5 (228.7B, Q3) at [32.8 tok/s](https://github.com/visorcraft/strix-halo-llm-perf); Qwen3-235B-A22B Q3 at [17.2 tok/s](https://github.com/visorcraft/strix-halo-llm-perf); Llama 4 Scout 109B at [13.83 tok/s](https://runaihome.com/blog/ryzen-ai-max-395-strix-halo-local-llm-2026/); Heretic2 at IQ4_XS in a 92GB footprint runs [24 tok/s](https://llamaperf.com/gpu/strix-halo-128gb).
- **MLPerf Client v6.1** on Strix Halo 128GB decodes Atlas NVFP4 at [20.6 tok/s](https://llamaperf.com/gpu/strix-halo-128gb), and DeepSeek V4.1 Flash Q2 (streamed from SSD) decodes at [7.12 tok/s](https://llamaperf.com/gpu/strix-halo-128gb).
- **Dense big models**: Llama 3.1 70B Q4_K_M at [5.1 tok/s](https://runaihome.com/blog/ryzen-ai-max-395-strix-halo-local-llm-2026/); dense 27-32B quantized models sit around [9.6-11.3 tok/s](https://llmrun.dev/device/framework-desktop-128gb).

AMD's own [January 2026 blog](https://www.amd.com/en/blogs/2026/amd-ryzen-ai-max-ai-pcs-deliver-exceptional-intelligence.html) frames the same story marketing-first: gpt-oss-120b runs roughly **10x faster than Llama 3 70B** on the AI Max+ 395, and the platform delivers about **1.7x the tokens-per-dollar of DGX Spark** across four models tested in LM Studio.

## Prompt Processing: ROCm vs Vulkan

Backends matter on this chip. The [strix-halo-llm-perf](https://github.com/visorcraft/strix-halo-llm-perf) maintainers found that **ROCm 7.x builds give the best prompt processing**, while **AMDVLK Vulkan builds give the best token generation (+16 percent in some tests)**, and the kyuz0 [amd-strix-halo-toolboxes](https://github.com/kyuz0/amd-strix-halo-toolboxes) project packages prebuilt toolchains so you can A/B test both against your own workloads. On Linux, Ubuntu 24.04 with recent ROCm builds unlocks a ~115GB GTT memory pool for the GPU, versus the 96GB VRAM slider on Windows ([seehiong blog](https://seehiong.github.io/posts/2026/08/running-llama.cpp-on-amd-strix-halo/), [Framework](https://frame.work/desktop?tab=machine-learning)).

## Model Map by Cluster Size

![Ryzen AI Max memory tiers and model map](https://pyshine.com/assets/img/diagrams/amd-ryzen-local-ai/pyshine-amd-ryzen-model-map.svg)

| Setup | Biggest practical models | Speed | Source |
|---|---|---|---|
| 32GB (Max 385) | 8-14B dense, gpt-oss-20b Q4 | fast small models | [Framework](https://frame.work/products/desktop-diy-amd-aimax300/) |
| 64GB (Max+ 395) | Qwen3-30B-A3B, Qwen3-Coder-30B, dense 32B Q4 | 86-100 / ~97 / ~10 tok/s | [llmrun](https://llmrun.dev/device/framework-desktop-128gb) |
| 128GB (Max+ 395) | gpt-oss-120b, Qwen3.8 Flash-Next, MiniMax M2.5 Q3, Llama 4 Scout 109B, dense 70B Q4 | 53-56 down to 5.1 tok/s | [visorcraft](https://github.com/visorcraft/strix-halo-llm-perf) |
| 2 boxes (USB4/10GbE RPC) | MiniMax M2.5-REAP 228.7B, Qwen3.5-397B | 15.35 / ~12 tok/s | [visorcraft](https://github.com/visorcraft/strix-halo-llm-perf) |
| Strix Halo + Mac M5 Max | GLM-5.3 321B | 24 tok/s | [llamaperf](https://llamaperf.com/gpu/strix-halo-128gb) |

Note the 64GB catch: gpt-oss-120b's [63.39 GB of MXFP4 weights](https://community.frame.work/t/tracking-will-the-ai-max-395-128gb-be-able-to-run-gpt-oss-120b/73280) cannot squeeze into a 64GB machine's VRAM allocation. If 120B-class models are the goal, 128GB is the floor.

## Clustering: USB4, 10GbE, and Mixed Mac+AMD

llama.cpp's RPC mode works fine over Strix Halo's USB4 ports, measuring about [9.4 Gbps of effective throughput](https://github.com/visorcraft/strix-halo-llm-perf). Two hosts split [MiniMax M2.5-REAP 228.7B at 15.35 tok/s and Qwen3.5-397B at roughly 12 tok/s](https://github.com/visorcraft/strix-halo-llm-perf). Because RPC is vendor-agnostic, the community also runs **hybrid clusters**: a Strix Halo box plus a Mac Studio M5 Max ran GLM-5.3 321B at [24 tok/s](https://llamaperf.com/gpu/strix-halo-128gb) - the same mixed-vendor trick we covered in the [Mac Studio deep dive](https://pyshine.com/Best-Mac-Studio-For-Local-AI-2026-Deep-Research/).

For rack ambitions, the [Minisforum MS-S1 MAX](https://store.minisforum.com/products/minisforum-ms-s1-max-mini-pc) ships with dual 10GbE, a PCIe 4.0 x16 slot, and explicit dual-unit-to-2U-rack cluster support. Framework documents [llama.cpp RPC clustering](https://frame.work/desktop?tab=machine-learning) as a supported workflow for its Desktop, and has announced a 192GB memory tier is coming.

## Fine-Tuning

Fine-tuning on Strix Halo is a Linux-first affair: AMD's ROCm 7.x stack supports gfx1151, and the [Ryzen AI Halo developer platform](https://www.amd.com/en/products/processors/desktops/ryzen/ryzen-ai-halo/ryzen-ai-max-plus-395.html) ships ROCm preinstalled with a 120W power envelope, while the [strix-halo-llm-perf](https://github.com/visorcraft/strix-halo-llm-perf) and [toolboxes](https://github.com/kyuz0/amd-strix-halo-toolboxes) projects document the ROCm builds that work. With 96-115GB of GPU-visible memory, QLoRA on 30-70B models and LoRA on small MoE models fit where a 16GB consumer GPU cannot. There is no CUDA ecosystem here - expect to debug - but for inference-first buyers that is a non-issue.

## Ryzen AI Max vs DGX Spark vs Mac Studio

| Platform | Memory | Bandwidth | gpt-oss-120b decode | Price (128GB class) |
|---|---|---|---|---|
| Ryzen AI Max+ 395 | 128GB LPDDR5X | 256 GB/s | 38-56 tok/s | from $3,449 |
| DGX Spark (GB10) | 128GB LPDDR5X | 273 GB/s | ~55-60 tok/s | $3,999+ |
| Mac Studio M5 Max | up to 128GB | far higher | see [our deep dive](https://pyshine.com/Best-Mac-Studio-For-Local-AI-2026-Deep-Research/) | from ~$4,999+ |

Sources: [AMD Halo page](https://www.amd.com/en/products/processors/desktops/ryzen/ryzen-ai-halo/ryzen-ai-max-plus-395.html), [NVIDIA](https://www.nvidia.com/en-us/products/workstations/dgx-spark/), [visorcraft](https://github.com/visorcraft/strix-halo-llm-perf), [Framework](https://frame.work/products/desktop-diy-amd-aimax300/). The bandwidth gap to Apple silicon (M5 Max/Ultra reach 460-1,228 GB/s, documented in [our Mac Studio post](https://pyshine.com/Best-Mac-Studio-For-Local-AI-2026-Deep-Research/)) is real - but so is the price gap. AMD's [own blog](https://www.amd.com/en/blogs/2026/amd-ryzen-ai-max-ai-pcs-deliver-exceptional-intelligence.html) claims 1.7x tokens-per-dollar over DGX Spark, and the community numbers above back the direction of that claim: you trade roughly 6 percent bandwidth and a CUDA license for hundreds of dollars and full x86 tooling.

## Where to Buy (2026 Prices)

- **[Framework Desktop](https://frame.work/products/desktop-diy-amd-aimax300/)**: DIY Max 385 32GB $1,269 (currently out of stock), Max+ 395 64GB $1,959, Max+ 395 128GB $3,449. Mini-ITX mainboard, repairable, [5GbE plus dual USB4](https://frame.work/desktop?tab=machine-learning).
- **[GMKtec EVO-X2](https://www.gmktec.com/products/amd-ryzen%E2%84%A2-ai-max-395-evo-x2-ai-mini-pc)**: $2,199 (64GB) / $3,649 (128GB), re-verified September 2026 amid the DRAM shortage.
- **[BOSGAME M5 AI](https://www.newegg.com/p/2W1-003S-00009)**: 128GB/2TB at $3,499.99 on Newegg.
- **[Minisforum MS-S1 MAX](https://store.minisforum.com/products/minisforum-ms-s1-max-mini-pc)**: 128GB "Max AI Compute" edition at $3,799 (regular $4,749), with dual 10GbE and PCIe x16.
- **[AMD Ryzen AI Halo](https://www.newegg.com/amd-rah-001-linux-os-mini-pc/p/N82E16859992001)**: official developer platform, $4,699.99 at Newegg, Linux + ROCm preinstalled.

## Conclusion

The Ryzen AI Max family is the value play of the 2026 local-AI scene. For $3,449-$3,649 you get a 128GB box that runs **gpt-oss-120b at 38-56 tok/s**, **Qwen3-30B MoE at ~100 tok/s**, and clusters into two-box MiniMax/Qwen3.5-397B territory or hybrid Mac+AMD rigs running GLM-5.3 321B. It loses the bandwidth race to Apple silicon and the software race to NVIDIA, but per dollar of decode throughput on big-memory MoE models, it is the cheapest door into frontier-adjacent local AI today. Pair the purchase decision with the DGX Spark analysis in [our Spark deep dive](https://pyshine.com/Best-DGX-Spark-For-Local-AI-2026-Deep-Research/) and pick by ecosystem, not spec sheets.

## All Sources

1. [AMD Ryzen AI Max+ 395 official specifications](https://www.amd.com/en/products/processors/laptop/ryzen/ai-300-series/amd-ryzen-ai-max-plus-395.html)
2. [AMD Ryzen AI Halo developer platform](https://www.amd.com/en/products/processors/desktops/ryzen/ryzen-ai-halo/ryzen-ai-max-plus-395.html)
3. [AMD blog: Ryzen AI Max AI PCs deliver exceptional intelligence (Jan 2026)](https://www.amd.com/en/blogs/2026/amd-ryzen-ai-max-ai-pcs-deliver-exceptional-intelligence.html)
4. [visorcraft/strix-halo-llm-perf community benchmark repo](https://github.com/visorcraft/strix-halo-llm-perf)
5. [kyuz0/amd-strix-halo-toolboxes](https://github.com/kyuz0/amd-strix-halo-toolboxes)
6. [llamaperf: Strix Halo 128GB](https://llamaperf.com/gpu/strix-halo-128gb)
7. [RunAIHome: Ryzen AI Max 395 Strix Halo local LLM analysis](https://runaihome.com/blog/ryzen-ai-max-395-strix-halo-local-llm-2026/)
8. [llmrun: Framework Desktop 128GB device page](https://llmrun.dev/device/framework-desktop-128gb)
9. [Framework Desktop store and pricing](https://frame.work/products/desktop-diy-amd-aimax300/)
10. [Framework Desktop machine learning benchmarks](https://frame.work/desktop?tab=machine-learning)
11. [Framework community: running gpt-oss-120b on AI Max 395 128GB](https://community.frame.work/t/tracking-will-the-ai-max-395-128gb-be-able-to-run-gpt-oss-120b/73280)
12. [GMKtec EVO-X2 product page](https://www.gmktec.com/products/amd-ryzen%E2%84%A2-ai-max-395-evo-x2-ai-mini-pc)
13. [Minisforum MS-S1 MAX product page](https://store.minisforum.com/products/minisforum-ms-s1-max-mini-pc)
14. [seehiong: running llama.cpp on AMD Strix Halo (Linux GTT)](https://seehiong.github.io/posts/2026/08/running-llama.cpp-on-amd-strix-halo/)
15. [Newegg: AMD Ryzen AI Halo developer platform](https://www.newegg.com/amd-rah-001-linux-os-mini-pc/p/N82E16859992001)
16. [Newegg: BOSGAME M5 AI 128GB](https://www.newegg.com/p/2W1-003S-00009)
17. [NVIDIA DGX Spark official page](https://www.nvidia.com/en-us/products/workstations/dgx-spark/)
18. [PyShine: Best Mac Studio for local AI deep research](https://pyshine.com/Best-Mac-Studio-For-Local-AI-2026-Deep-Research/)
19. [PyShine: Best DGX Spark for local AI deep research](https://pyshine.com/Best-DGX-Spark-For-Local-AI-2026-Deep-Research/)
20. [PyShine: Best Mac mini for local AI deep research](https://pyshine.com/Best-Mac-Mini-For-Local-AI-2026-Deep-Research/)
