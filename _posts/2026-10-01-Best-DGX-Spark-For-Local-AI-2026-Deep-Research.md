---
layout: post
title: "The Best DGX Spark for Local AI in 2026: LLM Speed, Model Sizes, and Clusters, With Sources"
description: "A deep research guide to NVIDIA DGX Spark local LLM performance: what the GB10 box really delivers in tokens per second, the biggest models one, two, four and eight Sparks can run, the 273 GB/s decode wall, two-box clustering over 200 GbE, fine-tuning numbers, and 2026 prices - every figure linked to its source."
date: 2026-10-01
permalink: /Best-DGX-Spark-For-Local-AI-2026-Deep-Research/
featured-img: ai-coding-frameworks/ai-coding-frameworks
image: https://pyshine.com/assets/img/diagrams/dgx-spark-local-ai/pyshine-dgx-spark-model-map.svg
tags: [DGX Spark, NVIDIA, Local AI, LLM, GB10, CUDA, Benchmarks]
categories: [AI, Open Source]
author: PyShine
---

In our [Mac mini deep research](https://pyshine.com/Best-Mac-Mini-For-Local-AI-2026-Deep-Research/) and the [Mac Studio follow-up](https://pyshine.com/Best-Mac-Studio-For-Local-AI-2026-Deep-Research/) we followed Apple Silicon through the local-LLM stack. NVIDIA answers with a completely different machine: the [DGX Spark](https://www.nvidia.com/en-us/products/workstations/dgx-spark/), a desk-sized box built around the GB10 Grace Blackwell Superchip with **128 GB of coherent unified memory**, **up to 1 petaFLOP of FP4 AI compute** and a datacenter-class **ConnectX-7 200 GbE port** for clustering. It shipped in October 2025, and a year later it is the reference CUDA machine for running 120B-700B-class open models at home.

The honest headline from a year of published benchmarks: the Spark is **a capacity machine and a prompt-processing monster, not a decode-speed demon**. Its 273 GB/s memory bandwidth is roughly a quarter of the Mac Studio M5 Ultra's 1.2 TB/s, and that single number decides decode speed. On Mixture-of-Experts models - where only a few billion parameters fire per token - it is genuinely fast. On dense 70B models it can drop below reading pace. Everything in this post is a measured figure with a link so you can double-check it.

This guide covers the hardware and the 2026 lineup of GB10 clones, the bandwidth math that predicts decode, what NVIDIA and the community actually measured in tokens per second, how big a model one, two, four or eight Sparks can run, the clustering playbook, fine-tuning numbers, and where to buy without overpaying.

![DGX Spark cluster picks](https://pyshine.com/assets/img/diagrams/dgx-spark-local-ai/pyshine-dgx-spark-local-ai-picks.svg)

*The four buying tiers: one Spark for the 35B-120B MoE class, two Sparks as the validated sweet spot for 235B-405B FP4, four for 700B-class, and community racks of six to eight for 750B-plus.*

Reading the ladder top to bottom: every tier multiplies pooled memory and the biggest runnable model - from gpt-oss-120b at 55-60 tok/s on a single box, to 235B FP4 and DeepSeek V4 Flash at up to 1M context on two boxes, to NVIDIA-validated 700B-class on four, to a 753B model at 7.4 tok/s on a six-node community rack.

## What the DGX Spark Actually Is

The [DGX Spark](https://www.nvidia.com/en-us/products/workstations/dgx-spark/) is a 150 x 150 x 51 mm, 1.2-1.5 kg desktop appliance running DGX OS (Ubuntu-based). It was announced as Project DIGITS at CES on January 6, 2025, renamed and priced at GTC in March 2025, and started shipping the week of October 13, 2025 ([NVIDIA press release](https://nvidianews.nvidia.com/news/nvidia-dgx-spark-arrives-for-worlds-ai-developers)). The official specification table ([NVIDIA product page](https://www.nvidia.com/en-us/products/workstations/dgx-spark/)):

| Spec | DGX Spark (GB10) | Source |
|---|---|---|
| AI compute | Up to 1 PFLOP FP4 (with sparsity) | [NVIDIA](https://www.nvidia.com/en-us/products/workstations/dgx-spark/) |
| Memory | 128 GB LPDDR5x, coherent unified, 256-bit | [NVIDIA](https://www.nvidia.com/en-us/products/workstations/dgx-spark/) |
| Memory bandwidth | 273 GB/s | [NVIDIA](https://www.nvidia.com/en-us/products/workstations/dgx-spark/) |
| CPU | 20 Arm cores (10 Cortex-X925 + 10 Cortex-A725) | [ASUS GX10 spec sheet](https://eshop.asus.com/us/ascent-gx10.html) |
| Fine-tuning | Up to 70B parameters | [NVIDIA](https://www.nvidia.com/en-us/products/workstations/dgx-spark/) |
| Inference | Up to 200B parameters | [NVIDIA](https://www.nvidia.com/en-us/products/workstations/dgx-spark/) |
| Clustering | ConnectX-7, up to four systems, models up to 700B | [NVIDIA](https://www.nvidia.com/en-us/products/workstations/dgx-spark/) |
| Networking | ConnectX-7 SmartNIC (200 Gb), 10 GbE RJ-45, Wi-Fi 7 | [ASUS GX10 spec sheet](https://eshop.asus.com/us/ascent-gx10.html) |

Two practical numbers hide behind those marketing figures. First, the GPU does not see all 128 GB: llama.cpp reports about **124.5 GB of "VRAM"** ([NVIDIA Developer Forums](https://forums.developer.nvidia.com/t/fastest-llama-cpp-docker-image/376946)), and an operator running a six-node rack budgets **roughly 104 GB of GPU-visible memory per node after DGX OS takes its share** ([6-node build report](https://forums.developer.nvidia.com/t/6x-dgx-spark-200g-roce-fabric-my-learning-journey/377871)). Second, the ConnectX-7's two QSFP56 cages top out at **200 Gb of usable bandwidth** because the NIC sits behind a pair of PCIe Gen5 x4 links - the second port is for topology flexibility, not more throughput ([StorageReview cluster review](https://www.storagereview.com/review/nvidia-dgx-spark-cluster-review-distributed-inference-on-dell-gigabyte-and-hp)).

The same GB10 chip ships in partner systems: ASUS Ascent GX10, Acer Veriton GN100, Gigabyte AI TOP ATOM, Dell Pro Max with GB10, HP ZGX Nano, Lenovo ThinkStation PGX and MSI EdgeXpert ([NVIDIA GTC announcement](https://nvidianews.nvidia.com/news/nvidia-dgx-spark-arrives-for-worlds-ai-developers) names the OEM lineup). Software-wise it is the only desktop in this class with the full CUDA stack: TensorRT-LLM, SGLang, vLLM, PyTorch and NGC containers all run unmodified, which is exactly what Apple Silicon cannot offer.

## The 273 GB/s Rule: Why MoE Wins and Dense Loses

Token generation is memory-bandwidth-bound: every generated token must read the model's active weights once, so decode speed is approximately bandwidth divided by gigabytes-per-token. A DGX Spark operator worked the formula explicitly: **tok/s = 273 / (active weights + KV cache)**, which puts the comfortable interactive sweet spot at roughly 27B active parameters at Q4 - fine for MoE models with 3B-17B active, painful for anything dense ([NVIDIA Developer Forums](https://forums.developer.nvidia.com/t/why-273-gb-s-less-is-more-until-it-isn-t/359555)).

The clearest measured proof of the MoE-versus-dense split on identical hardware:

| Model | Architecture | Quant | Decode | Source |
|---|---|---|---|---|
| Qwen3 30B (A3B active) | MoE | Q4_K_M | ~89 tok/s | [RunAIHome](https://runaihome.com/blog/nvidia-rtx-spark-local-ai-2026/) |
| Qwen3 32B dense | Dense | Q4_K_M | ~10.7 tok/s | [RunAIHome](https://runaihome.com/blog/nvidia-rtx-spark-local-ai-2026/) |
| Llama 3.1 70B dense | Dense | FP8 | 2.7 tok/s | [LocalAIMaster (LMSYS data)](https://www.localaimaster.com/blog/dgx-spark-local-ai-review) |

Same box, same class of model size, an 8x generation gap driven purely by active parameters. This is the single most important thing to understand before buying a Spark: **it is a MoE machine**.

For bandwidth context against the Macs we covered earlier: the M5 Max MacBook Pro moves 614 GB/s and the DGX Spark 273 GB/s on the same 128 GB memory class ([Elton Stoneman's comparison table](https://blog.sixeyed.com/mac-studio-llm-workstation/)), and the [Mac Studio M5 Ultra](https://pyshine.com/Best-Mac-Studio-For-Local-AI-2026-Deep-Research/) reaches 1.2 TB/s. Apple wins raw decode; NVIDIA wins software (CUDA, FP4 checkpoints, fine-tuning) and clustering (RDMA fabric instead of Thunderbolt).

## Decode Throughput: What NVIDIA and the Community Measured

NVIDIA's own published inference table ([Technical Blog, Oct 24, 2025](https://developer.nvidia.com/blog/how-nvidia-dgx-sparks-performance-enables-intensive-ai-tasks/)) - batch size 1, input/output context 2048/128:

| Model | Quant | Engine | Prefill (tok/s) | Decode (tok/s) |
|---|---|---|---|---|
| gpt-oss-20b | MXFP4 | llama.cpp | 3,670 | 82.7 |
| gpt-oss-120b | MXFP4 | llama.cpp | 1,725 | 55.4 |
| Llama 3.1 8B | NVFP4 | TensorRT-LLM | 10,257 | 38.7 |

Community numbers agree and extend the picture:

- **gpt-oss-120b (117B MoE, MXFP4)**: the llama.cpp maintainer measured 1,956 tok/s prefill and 60.6 tok/s generation (40.6 at 32K context depth), and Qwen3 Coder 30B-A3B at Q8_0 hit 44.3 tok/s ([LocalAIMaster's attributed table](https://www.localaimaster.com/blog/dgx-spark-local-ai-review)). Level1Techs reproduced 59.5 tok/s at depth zero falling to 41.9 at 32K on the Gigabyte AI TOP ATOM ([Level1Techs](https://forum.level1techs.com/t/gigabyte-ai-tops-dgx-spark/243812)), and The AI Bench's July 2026 calibration archive logs 42 tok/s with cross-verified runner builds ([TheAIBench](https://theaibench.ai/methodology/calibration/)).
- **gpt-oss-20b**: NVIDIA's 82.7 tok/s was reproduced at 86 tok/s with llama.cpp build 7067 - and The AI Bench documents the same box jumping from 2,009 to 3,798 tok/s of prompt processing purely from a llama.cpp software bump ([TheAIBench](https://theaibench.ai/methodology/calibration/)). Their verdict: an M4 Max MacBook out-decodes the Spark 118 to 86 on this identical model, but the Spark runs 120B models no consumer GPU can load.
- **Qwen3.5-35B-A3B (MoE)**: 72.3 tok/s tg128 in llama-bench with an NVFP4 checkpoint ([NVIDIA Developer Forums](https://forums.developer.nvidia.com/t/fastest-llama-cpp-docker-image/376946)); a long-term Claude Code deployment averaged 51.0 tok/s with 85-246 ms TTFT at 128K context ([NVIDIA Developer Forums](https://forums.developer.nvidia.com/t/implementation-guide-dgx-spark-with-qwen3-5-35b-a3b-via-llama-cpp-for-claude-code/365382)).
- **Qwen3.8-27B dense**: 34 tok/s under SGLang with NVFP4 weights ([llamaperf](https://llamaperf.com/gpu/dgx-spark)).
- **DeepSeek V4 Flash (IQ2 imatrix)**: 28-31 tok/s sustained through a 131K-context agentic workflow using the ds4 server ([NVIDIA Developer Forums](https://forums.developer.nvidia.com/t/best-model-for-single-spark/374834)).
- **Llama 3.1 8B**: 20.5 tok/s single-stream but 368 tok/s aggregate at batch 32 - the Spark serves small teams well on small models ([LocalAIMaster (LMSYS data)](https://www.localaimaster.com/blog/dgx-spark-local-ai-review)).
- **Dense 70B caveat**: LMSYS measured Llama 3.1 70B FP8 at 2.7 tok/s decode - slower than reading pace - and concluded dense 70B-120B models are for prototyping, not production, on this box ([LocalAIMaster](https://www.localaimaster.com/blog/dgx-spark-local-ai-review)).

Early scattered reviews are worth a calibration note: ServeTheHome's launch-day 14.5 tok/s gpt-oss-120b figure was later attributed to configuration issues, and model-load times dropped from 104 s to 22 s with a kernel fix ([LocalAIMaster](https://www.localaimaster.com/blog/dgx-spark-local-ai-review)). On this platform, always check the engine build before believing a number.

## Prompt Processing Is the Hidden Superpower

Decode is bandwidth-bound, but prefill is compute-bound - and the GB10's Blackwell tensor cores crush it:

- Up to **10,257 tok/s** of prompt processing on an 8B NVFP4 model in TensorRT-LLM, and 1,725-1,956 tok/s on a 117B MoE ([NVIDIA Technical Blog](https://developer.nvidia.com/blog/how-nvidia-dgx-sparks-performance-enables-intensive-ai-tasks/); [LocalAIMaster](https://www.localaimaster.com/blog/dgx-spark-local-ai-review)).
- **2,637-3,798 tok/s pp512** with current llama.cpp builds ([NVIDIA Developer Forums](https://forums.developer.nvidia.com/t/fastest-llama-cpp-docker-image/376946); [TheAIBench](https://theaibench.ai/methodology/calibration/)).
- On the two-box 235B FP4 setup, TensorRT-LLM pushed prefill to **23,477 tok/s** ([On-Prem Hardware Sizing Guide](https://supermicrovexpo.com)).
- TTFT in real agentic use: 85-246 ms across task types on the 35B-A3B deployment ([NVIDIA Developer Forums](https://forums.developer.nvidia.com/t/implementation-guide-dgx-spark-with-qwen3-5-35b-a3b-via-llama-cpp-for-claude-code/365382)).

This is why NVIDIA positions the Spark as an **agent computer**: workloads that read 30K-250K token contexts repeatedly benefit far more from fast prefill than from fast decode ([NVIDIA Technical Blog, Mar 16, 2026](https://developer.nvidia.com/blog/scaling-autonomous-ai-agents-and-workloads-with-nvidia-dgx-spark/)).

## How Big an LLM Each Cluster Size Can Run

![DGX Spark model map](https://pyshine.com/assets/img/diagrams/dgx-spark-local-ai/pyshine-dgx-spark-model-map.svg)

*Memory tier, not chip tier, decides the model class - and FP4 checkpoints halve the footprint compared to 8-bit.*

| Cluster | Pooled memory | Biggest model class | Measured speed | Source |
|---|---|---|---|---|
| 1 Spark (~120 GB GPU-visible) | 128 GB | gpt-oss-120b MXFP4 (59 GiB); NVIDIA rates up to 200B FP4 | 55-60 tok/s on 120b; 2.7 tok/s dense 70B | [Level1Techs](https://forum.level1techs.com/t/gigabyte-ai-tops-dgx-spark/243812); [NVIDIA](https://www.nvidia.com/en-us/products/workstations/dgx-spark/); [LocalAIMaster](https://www.localaimaster.com/blog/dgx-spark-local-ai-review) |
| 2 Sparks (200 GbE RDMA) | ~240 GB | Qwen3-235B-A22B FP4; 405B FP4 class | 11.7 tok/s on 235B; ~60-67 tok/s DeepSeek V4 Flash at up to 1M ctx | [NVIDIA Technical Blog](https://developer.nvidia.com/blog/how-nvidia-dgx-sparks-performance-enables-intensive-ai-tasks/); [NVIDIA Developer Forums](https://forums.developer.nvidia.com/t/deepseek-v4-flash-dspark-on-2x-dgx-spark-gb10-big-single-stream-speed-boost-60-67-tok-s-1m-context-now-with-concurrency/374846) |
| 4 Sparks | ~480 GB | 700B-class (NVIDIA-validated) | ~4x TP speedup on 70B; ~74,600 tok/s batched | [NVIDIA](https://www.nvidia.com/en-us/products/workstations/dgx-spark/); [NVIDIA Technical Blog](https://developer.nvidia.com/blog/scaling-autonomous-ai-agents-and-workloads-with-nvidia-dgx-spark/) |
| 6-8 Sparks | ~630-840 GB | GLM-5.2 753B; Mimo 2.6 1020B-A42B | 7.4 tok/s (753B); 68.3 tok/s (1020B MoE, speculative) | [NVIDIA Developer Forums](https://forums.developer.nvidia.com/t/6x-dgx-spark-200g-roce-fabric-my-learning-journey/377871); [llamaperf](https://llamaperf.com/gpu/dgx-spark) |

The dense-versus-MoE split matters as much here as it does on Macs: a 405B dense model at FP4 still streams ~200 GB of weights per token, so it lands at the slow end even when it fits, while a 1020B MoE with 42B active decoded at 68.3 tok/s on eight Sparks with DFlash speculative decoding ([llamaperf](https://llamaperf.com/gpu/dgx-spark)). Choose MoE checkpoints whenever the workload allows.

## Clustering: Two Boxes Over a Datacenter Fabric

The Spark's party trick is the [ConnectX-7 SmartNIC](https://www.nvidia.com/en-us/products/workstations/dgx-spark/) - a real 200 Gb RDMA fabric on a desktop appliance. What is actually validated:

- **Two Sparks, directly connected**, is the configuration NVIDIA actively markets and StorageReview benchmarked end-to-end across Dell, Gigabyte and HP units. Key finding: for batched inference, **pipeline parallelism beat tensor parallelism 554.7 to 252.0 tok/s** on gpt-oss-120b at batch 128, because TP's all-reduce traffic saturates the 200 Gb link; TP keeps a narrow lead for single-stream chat latency ([StorageReview](https://www.storagereview.com/review/nvidia-dgx-spark-cluster-review-distributed-inference-on-dell-gigabyte-and-hp)).
- **Tensor-parallel speedups scale**: ~2x per-token speed at two nodes and ~4x at four nodes for Llama 3.3 70B, with batched throughput growing ~18,400 to ~35,900 to ~74,600 tok/s from one to four nodes ([NVIDIA Technical Blog](https://developer.nvidia.com/blog/scaling-autonomous-ai-agents-and-workloads-with-nvidia-dgx-spark/)).
- **But do not cluster what already fits**: an operator measured Qwen3.6-35B-A3B at 99.0 tok/s with TP=2 dropping to 89.2 with TP=4 - sharing a model that fits one box costs throughput ([NVIDIA Developer Forums](https://forums.developer.nvidia.com/t/6x-dgx-spark-200g-roce-fabric-my-learning-journey/377871)).
- **Concurrency is the cluster's real payoff**: the same six-node rack sustained 128 simultaneous streams at 1,403 tok/s aggregate on 35B-A3B, and a single Spark served 765 tok/s across 32 lanes of a 14B model ([NVIDIA Developer Forums](https://forums.developer.nvidia.com/t/6x-dgx-spark-200g-roce-fabric-my-learning-journey/377871)).
- **Beyond eight**: operators run 6-node rings on MikroTik 400G switches at ~196 Gbps per link; GLM-5.2 753B at TP=2 x PP=3 across 6 nodes yields 7.4 tok/s ([NVIDIA Developer Forums](https://forums.developer.nvidia.com/t/6x-dgx-spark-200g-roce-fabric-my-learning-journey/377871)).
- **Heterogeneous disaggregation exists too**: EXO Labs demonstrated 2x Spark for prefill plus an M3 Ultra Mac Studio for decode at ~2.8x end-to-end improvement on Llama 3.1 8B with 8K prompts ([NVIDIA Developer Forums](https://forums.developer.nvidia.com/t/dgx-spark-rtx-6000-pro-blackwell-disaggregated-inference/368860)).

## Fine-Tuning: the Quiet Reason to Pick CUDA

Apple Silicon cannot fine-tune 70B models on-device; the Spark can. NVIDIA's published training numbers ([Technical Blog](https://developer.nvidia.com/blog/how-nvidia-dgx-sparks-performance-enables-intensive-ai-tasks/)): Llama 3.2 3B full fine-tuning at 82,739 tok/s, Llama 3.1 8B LoRA at 53,658 tok/s, and Llama 3.3 70B QLoRA at 5,079 tok/s - none of which fit a 32 GB consumer GPU. NVIDIA explicitly rates the box for fine-tuning up to 70B parameters ([product page](https://www.nvidia.com/en-us/products/workstations/dgx-spark/)).

A six-day independent stress test confirmed the training numbers are real but documented the production gotchas: FP16 precision traps, memory fragmentation requiring hard reboots, and CUDA version mismatches that cost 3.6x performance until fixed ([RunDataRun](https://ai.rundatarun.io/practical-applications/dgx-lab-benchmarks-vs-reality-day-4)). Budget for the software stack, not just the box.

## DGX Spark vs Mac Studio: Which One for Which Job

Having covered both families, here is the honest split:

| Need | Buy | Why |
|---|---|---|
| Fastest single-stream decode on 70B-120B class | [Mac Studio](https://pyshine.com/Best-Mac-Studio-For-Local-AI-2026-Deep-Research/) | 460-1,228 GB/s bandwidth vs 273; gpt-oss-20b decode 118 vs 86 ([TheAIBench](https://theaibench.ai/methodology/calibration/); [sixeyed](https://blog.sixeyed.com/mac-studio-llm-workstation/)) |
| CUDA software, FP4 checkpoints, fine-tuning up to 70B | DGX Spark | Full TensorRT-LLM/SGLang/PyTorch stack; 5,079 tok/s QLoRA on 70B ([NVIDIA](https://developer.nvidia.com/blog/how-nvidia-dgx-sparks-performance-enables-intensive-ai-tasks/)) |
| 235B-700B models on a desk | DGX Spark (2-4 boxes) | RDMA fabric + NVIDIA-validated clustering ([NVIDIA](https://developer.nvidia.com/blog/scaling-autonomous-ai-agents-and-workloads-with-nvidia-dgx-spark/)) |
| Always-on agents with huge prompts | Either, slight edge to Spark | Fast prefill; NVIDIA's agent positioning ([NVIDIA Technical Blog](https://developer.nvidia.com/blog/scaling-autonomous-ai-agents-and-workloads-with-nvidia-dgx-spark/)) |

A final wrinkle: NVIDIA is also bringing **RTX Spark N1X** - the same GB10-class silicon in Windows laptops and compact desktops - in late 2026 starting above $2,899, with ~300 GB/s bandwidth ([RunAIHome](https://runaihome.com/blog/nvidia-rtx-spark-local-ai-2026/)). The desktop DGX Spark remains the capacity-and-cluster play.

## Where to Buy and What to Pay

Prices moved a lot in year one. At launch in October 2025 the Founders Edition was **$3,999** ([NVIDIA press release](https://nvidianews.nvidia.com/news/nvidia-dgx-spark-arrives-for-worlds-ai-developers)) and ASUS listed the Ascent GX10 at **$2,999.99** ([NVIDIA Marketplace](https://marketplace.nvidia.com/en-us/enterprise/personal-ai-supercomputers/asus-ascent-gx10/)). By 2026, LPDDR5x supply constraints pushed the Founders Edition to **$4,699** ([LocalAIMaster price history](https://www.localaimaster.com/blog/dgx-spark-local-ai-review); [PromptQuorum](https://www.promptquorum.com/local-llms/70b-models-consumer-hardware)), and street prices for partner boxes now run higher - Newegg's AI-system best-sellers show the ASUS GX10 sold around $5,999, Gigabyte AI TOP ATOM at $6,499, Acer Veriton GN100 at $6,499, MSI EdgeXpert at $6,442 and a 4 TB DGX Spark at $6,500 ([Newegg](https://www.newegg.com/d/Best-Sellers/Thin-Client-Systems/s/ID-3087)).

Practical buying advice: the cheapest entry is watching for restocks at launch-tier pricing; the same GB10 chip and 128 GB ship in every partner clone, so pay for storage and warranty, not brand ([LocalAIMaster](https://www.localaimaster.com/blog/dgx-spark-local-ai-review)). If you plan to cluster, confirm the box exposes the ConnectX-7 QSFP56 ports - every GB10 system does, but check the exact SKU's I/O panel before ordering ([ASUS GX10](https://eshop.asus.com/us/ascent-gx10.html)).

## Conclusion

The DGX Spark in 2026 is the desktop to buy when your bottleneck is **model capacity and the CUDA software ecosystem**, not raw decode speed. One box runs the 35B-120B MoE class at 34-86 tok/s with elite prompt processing and full fine-tuning up to 70B. Two boxes over the 200 GbE fabric are the validated sweet spot for 235B-405B FP4 models. Four reach NVIDIA-validated 700B-class, and community racks of six to eight have served a 753B model and a 1T-parameter MoE. The 273 GB/s decode wall is real - dense 70B at 2.7 tok/s is a prototype, not a product - but paired with the right MoE checkpoints and the machine's unmatched prefill throughput, it is the most capable little CUDA box you can put on a desk. Check every number against the sources above, because on this platform the engine build and the checkpoint quant matter as much as the hardware.

## Links

- [NVIDIA DGX Spark product page](https://www.nvidia.com/en-us/products/workstations/dgx-spark/) - official specs
- [NVIDIA press release: DGX Spark arrives](https://nvidianews.nvidia.com/news/nvidia-dgx-spark-arrives-for-worlds-ai-developers) - launch and OEM lineup
- [NVIDIA Technical Blog: how DGX Spark's performance enables intensive AI tasks](https://developer.nvidia.com/blog/how-nvidia-dgx-sparks-performance-enables-intensive-ai-tasks/) - official inference and fine-tuning benchmarks
- [NVIDIA Technical Blog: scaling autonomous AI agents](https://developer.nvidia.com/blog/scaling-autonomous-ai-agents-and-workloads-with-nvidia-dgx-spark/) - clustering and agent workloads
- [LocalAIMaster DGX Spark review](https://www.localaimaster.com/blog/dgx-spark-local-ai-review) - llama.cpp maintainer and LMSYS tables, price history
- [The AI Bench July 2026 calibration archive](https://theaibench.ai/methodology/calibration/) - cross-verified runs, Spark vs MacBook decode
- [Level1Techs: Gigabyte AI TOP ATOM benchmarks](https://forum.level1techs.com/t/gigabyte-ai-tops-dgx-spark/243812) - llama-bench at multiple context depths
- [NVIDIA Developer Forums: fastest llama.cpp docker image](https://forums.developer.nvidia.com/t/fastest-llama-cpp-docker-image/376946) - 35B-A3B NVFP4 runs
- [NVIDIA Developer Forums: Qwen3.5-35B-A3B for Claude Code](https://forums.developer.nvidia.com/t/implementation-guide-dgx-spark-with-qwen3-5-35b-a3b-via-llama-cpp-for-claude-code/365382) - real agent deployment numbers
- [NVIDIA Developer Forums: best model for single Spark](https://forums.developer.nvidia.com/t/best-model-for-single-spark/374834) - DeepSeek V4 Flash on one box
- [NVIDIA Developer Forums: DeepSeek-V4-Flash-DSpark on 2x Spark](https://forums.developer.nvidia.com/t/deepseek-v4-flash-dspark-on-2x-dgx-spark-gb10-big-single-stream-speed-boost-60-67-tok-s-1m-context-now-with-concurrency/374846) - two-box 1M-context runs
- [NVIDIA Developer Forums: 6x DGX Spark RoCE fabric](https://forums.developer.nvidia.com/t/6x-dgx-spark-200g-roce-fabric-my-learning-journey/377871) - six-node rack measurements
- [NVIDIA Developer Forums: why 273 GB/s](https://forums.developer.nvidia.com/t/why-273-gb-s-less-is-more-until-it-isn-t/359555) - the bandwidth sweet-spot math
- [NVIDIA Developer Forums: Spark + RTX 6000 Pro disaggregation](https://forums.developer.nvidia.com/t/dgx-spark-rtx-6000-pro-blackwell-disaggregated-inference/368860) - EXO prefill/decode split
- [StorageReview: DGX Spark cluster review](https://www.storagereview.com/review/nvidia-dgx-spark-cluster-review-distributed-inference-on-dell-gigabyte-and-hp) - two-node PP vs TP benchmarks
- [llamaperf: DGX Spark reports](https://llamaperf.com/gpu/dgx-spark) - community-reported speeds incl. 8-box 1020B run
- [RunAIHome: RTX Spark for local AI](https://runaihome.com/blog/nvidia-rtx-spark-local-ai-2026/) - MoE vs dense, N1X outlook
- [RunDataRun: DGX Spark benchmarks vs reality](https://ai.rundatarun.io/practical-applications/dgx-lab-benchmarks-vs-reality-day-4) - fine-tuning stress test
- [Elton Stoneman: Mac Studio LLM workstation](https://blog.sixeyed.com/mac-studio-llm-workstation/) - bandwidth comparison table
- [PromptQuorum: 70B on consumer hardware](https://www.promptquorum.com/local-llms/70b-models-consumer-hardware) - Spark 70B specs and price
- [On-Prem AI Hardware Sizing Guide](https://supermicrovexpo.com) - 235B FP4 prefill numbers
- [NVIDIA Marketplace: ASUS Ascent GX10](https://marketplace.nvidia.com/en-us/enterprise/personal-ai-supercomputers/asus-ascent-gx10/) - launch-tier pricing
- [ASUS eShop: Ascent GX10](https://eshop.asus.com/us/ascent-gx10.html) - current spec sheet and pricing
- [Newegg AI systems best-sellers](https://www.newegg.com/d/Best-Sellers/Thin-Client-Systems/s/ID-3087) - 2026 street prices
- [PyShine: Best Mac mini for local AI](https://pyshine.com/Best-Mac-Mini-For-Local-AI-2026-Deep-Research/) - the entry-point comparison
- [PyShine: Best Mac Studio for local AI](https://pyshine.com/Best-Mac-Studio-For-Local-AI-2026-Deep-Research/) - the bandwidth-king comparison
