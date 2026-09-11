---
layout: post
title: "nano-kpu: The Inference Chip Kimi-K3 Designed for Itself"
description: "nano-kpu is the RTL design of a nano-scale hybrid-architecture inference chip, designed and implemented fully by Kimi-K3, released by Moonshot AI under Apache 2.0. The chip runs a nano instance of the Moonshot hybrid text stack: one KDA (Kimi Delta Attention) linear-attention layer, one NoPE multi-head latent attention (MLA) layer, a sigmoid-routed MoE MLP with one shared expert per layer, and attention-residual mixing with block size 2. Weights are int4 group-128 (AWQ/GPTQ-style), d_model 64, vocab 512, max context 64 positions, ~238k parameters. The repo ships the RTL (top module msh_chip_top), LUT ROMs, a bit-exact Python fixed-point self-model, a float32 golden reference, and a full Verilator + yosys/Nangate45 evaluation harness with correctness gates (cosine >= 0.98, argmax >= 0.99). The interface is a RUN/DONE command stream over a 128-bit DRAM port with latency-elastic in-order reads, and all on-chip storage must use the msh_sram/msh_rom macros (inferred memories are rejected). It is a demonstration of Kimi K3, not an official Moonshot AI project."
date: 2026-09-11
header-img: "img/post-bg.jpg"
permalink: /nano-kpu-Kimi-K3-Inference-Chip/
featured-img: ai-coding-frameworks/ai-coding-frameworks
tags:
  - nano-kpu
  - Kimi K3
  - RTL Design
  - Hardware
  - LLM Chip Design
  - KDA
  - Open Source
  - Verilator
author: PyShine
---

## What is nano-kpu

nano-kpu is the RTL design of a nano-scale hybrid-architecture inference chip, and the headline is who made it: **the chip was designed and implemented fully by Kimi-K3**. The repository, released by Moonshot AI at [github.com/MoonshotAI/nano-kpu](https://github.com/MoonshotAI/nano-kpu) under Apache 2.0, contains the complete chip RTL, its functional-simulation flow, and its timing/area synthesis flow. It is a demonstration of Kimi K3's capability, not an official Moonshot AI project.

The chip runs a **nano instance of the Moonshot hybrid text stack** - the same architectural family that powers the 2.8T-parameter Kimi K3, but scaled down to 2 layers, a 64-wide model dimension, a 512-token vocabulary, and a 64-position context. Every architectural primitive that defines the full Kimi K3 is present here: KDA linear attention, NoPE multi-head latent attention, sigmoid-routed mixture-of-experts with a shared expert, attention-residual mixing, and int4 group-128 quantization. The model the chip computes is a faithful, miniature version of the one that designed it.

What makes this more than a curiosity is that the deliverable is real hardware engineering: synthesizable Verilog-2005/SystemVerilog RTL, a Verilator C++ testbench, a yosys+Nangate45 synthesis flow, correctness gates against a float32 golden reference, and a measurement methodology that is deliberately fixed and frozen so results are comparable and reproducible. An LLM did not just write a Python script; it produced a chip datapath, a memory protocol, a macro discipline, and an evaluation harness.

## The Model Architecture: A Nano Kimi K3

The chip computes a 2-layer hybrid-attention MoE transformer. The configuration is pinned in `docs/architecture.md` and normative in `reference/model.py`.

![nano-kpu model architecture](/assets/img/diagrams/nano-kpu/nano-kpu-model-architecture.svg)

### Understanding the Architecture

**Two layers, one of each attention type.** Layer 0 is a KDA (Kimi Delta Attention) linear-attention layer; Layer 1 is a NoPE multi-head latent attention (MLA) layer. This is the same hybrid pattern as the full Kimi K3, where 69 of 93 layers are KDA and 24 are Gated MLA - here the ratio is 1:1 because there are only two layers, but the architectural ingredients are identical. KDA provides the linear-recurrent path with a fixed-size state; MLA provides the full-softmax global-attention path.

**KDA at nano scale.** The KDA layer has 2 heads of 32 dimensions each. It uses the delta rule with per-channel decay - the same `Diag(alpha_t)` channel-wise gating that distinguishes Kimi Delta Attention from Gated DeltaNet, just at head dimension 32 instead of the production scale. A kernel-4 depthwise causal convolution runs on the query, key, and value streams before the SiLU and the delta update. The decay is `alpha = exp(-5 * sigmoid(A_h * (f + dt_bias)))` with `A_h = exp(A_log)` shipped as unsigned Q8.8 per head, giving `alpha` a range of `(e^-5, 1)`. This is the exact recurrence described in the [Kimi Linear paper](https://arxiv.org/abs/2510.26692), in its token-by-token recurrent form.

**NoPE MLA.** The MLA layer has 2 heads with `dk=32` and `dr=16` (shared across heads) and `dv=32`, over a compressed KV latent of `dc=128`. Nothing is rotated - there is no RoPE anywhere in the design. Positional information is carried entirely by KDA's decaying recurrent state and the causal convolutions, which is the same NoPE-plus-KDA strategy the full Kimi K3 uses.

**Sigmoid-routed MoE with a shared expert.** Every layer has 8 SwiGLU experts (`d_ff=32`) with top-2 routing plus one always-on shared expert. The router is bias-free sigmoid with a score-correction bias that affects selection only, and the selected weights are renormalized by a factor of `2.828`. The shared expert is unweighted and always fires - the same Stable LatentMoE shape as Kimi K3's 16-of-896 routing, shrunk to top-2-of-8.

**Attention-residual mixing.** Block size 2: at block boundaries the running residual is snapshotted, the prefix sum restarts from the attention output, and the past re-enters through softmax mixtures over the snapshots. This is the AttnRes operation from Kimi K3, where each layer can attend to a weighted combination of representations from preceding blocks rather than only the immediately previous layer.

**Int4 group-128 weights.** All matmul tensors ship as AWQ/GPTQ-style int4 packed along the fan-in dimension, with int16 Q3.12 scales and uint8 zero points per group of 128. The embedding is token-major so a row lookup is one contiguous burst. Small parameters (norms, conv kernels, biases) ship as int16 fixed point in Q2.13, Q4.11, or Q8.8. The whole model is about 238k parameters, roughly 135.5 KB packed.

**Untied embedding and LM head.** The embedding and the LM head are separate tensors - the head is a standard-layout int4 tensor with `fan_out = vocab = 512`. The chip consumes token ids (the tokenizer and any MTP heads are explicitly out of scope).

## The Chip Interface and Run Protocol

The chip exposes a single clock, an active-low reset released after 16 cycles, a command/response stream, and a 128-bit DRAM port. The top module is `msh_chip_top`.

![nano-kpu interface and protocol](/assets/img/diagrams/nano-kpu/nano-kpu-interface-protocol.svg)

### Understanding the Interface

**Command and response streams.** The testbench sends one `RUN = 0x0000_0001` command after reset and nothing else, over a standard valid/ready handshake. The chip signals completion by sending `DONE = 0x0000_D0DE` exactly when the run is complete, after the final status word has been written. Cycle counting stops at DONE.

**The 128-bit DRAM port.** One request channel and one read-response channel. Peak throughput is one 16-byte beat per cycle. Reads arrive in order, each at least 24 cycles after acceptance, with a deterministic pseudo-random jitter - but the design must be **latency-elastic**: it cannot bake the constant 24 into its control logic, because the evaluation re-runs with the base latency doubled (`--lat-base=48`) and reports an elasticity ratio. Any number of requests may be in flight. Writes are posted with no acknowledgement. On performance-measured long runs the ready signal stays high to preserve the full 16 B/cycle roofline; short runs apply about 5% pseudo-random back-pressure as a protocol stress.

**A weight-stream floor.** The dense tensors (everything except the data-dependent routed experts and embedding rows) participate in every position's forward pass, so a clean run must read at least `floor_weight_bytes` from memory. Fewer read beats than this floor is a protocol violation and the run cannot PASS. The floor is deliberately conservative for MoE - an honest design also reads its selected experts, which the roofline (not the floor) prices.

**On-chip storage is macro-enforced.** All storage beyond individual flip-flops - the KV cache, the KDA recurrent state, the conv histories, buffers - must be instantiated as the harness-provided `msh_sram` macro. Inferred memories (`reg [W-1:0] mem [0:D-1]` arrays) are rejected: the `macros_only` gate fails synthesis if any `$mem_v2` survives. The `msh_sram` macro is a 1-read, 1-write, single-clock, synchronous-read SRAM with per-byte write enables, priced at about 1 Mbit/mm2 on Nangate45 6T high-density cells. Constant lookup tables (activation tables for sigmoid, the KDA decay alpha, expneg, rsqrt, recip) must use the `msh_rom` macro, which has two asynchronous read ports so one lookup can fetch `tab[idx]` and `tab[idx+1]` in the same cycle for linear interpolation. Both macros count their bits toward the SRAM capacity budget and the area budget.

**The run protocol.** The testbench preloads the self-describing memory image (a header plus a descriptor table) and asserts RUN. The chip reads the header, self-configures, streams weights as it pleases, and processes the token sequence causally under teacher forcing - the MLA KV cache, KDA state, and conv histories must be maintained across positions, and expert selection happens per token from the router scores. The chip writes one logits row plus one argmax id per position in any order at any time, then writes the status word last, then raises DONE. The testbench dumps the output region and the harness compares per-row cosine similarity and argmax agreement against the float32 reference.

## The Evaluation Flow

The harness is `harness/evaluate.py`, and it runs both functional simulation and synthesis in a single fixed, frozen parameterization so results are comparable across designs.

![nano-kpu evaluation flow](/assets/img/diagrams/nano-kpu/nano-kpu-evaluation-flow.svg)

### Understanding the Evaluation

**Setup is conda-centric and idempotent.** `scripts/setup_env.sh` uses an existing conda or bootstraps Miniconda, creates a dedicated `nano-kpu` environment from conda-forge with Verilator 5.x (baseline 5.050), yosys >= 0.64, and python 3.12 + numpy. It never touches the base env. It also downloads the Nangate45 standard-cell library (`NangateOpenCellLibrary_typical.lib`) from [The-OpenROAD-Project/OpenROAD-flow-scripts](https://github.com/The-OpenROAD-Project/OpenROAD-flow-scripts), pinned by commit and SHA-256, into `harness/lib/`. The cell library is deliberately not bundled - synthesis needs it, simulation does not.

**Functional simulation.** The Verilator C++ testbench (`harness/tb/tb_main.cpp` is normative) models the 16 B/cycle DRAM port and runs the chip against the self-describing memory image. Correctness is judged against `reference/model.py`, a float32 implementation that is the arbiter of model semantics. The gates are strict: per-row logits cosine must be at least 0.98, pooled argmax agreement must be at least 0.99, and the design must produce byte-identical outputs when all flip-flops are initialized to pseudo-random values (`+verilator+rand+reset+2`) - real silicon does not power up to zeros, so the design cannot rely on zero initialization.

**Performance metrics.** `cycles_per_token = long-run simulated cycles / seq_len`, and `tokens/s = clock / cpt`. The testbench also reports `read_beats` and `write_beats`, which feed a bandwidth-energy score (data movement is the batch-1 energy proxy) and a realism review - a design claiming fewer read beats than the dense int4 stream it must consume will be noticed. The throughput roofline assumes the active int4 weight stream flows once per token with dequant fused into the datapath; a design that expands weights to a wider format in external memory pays double bandwidth and cannot reach the roofline.

**Synthesis.** Area and timing are measured at the synthesis stage using yosys with abc mapping and Nangate45 static timing. The flow deliberately does not search for design-specific optima - it uses a fixed, frozen synthesis parameterization so reported numbers are a conservative, reproducible baseline rather than the best achievable quality of results. These are pre-layout estimates, not sign-off values validated through backend place and route. The storage rule is enforced here: all arrays beyond flip-flops go through the `msh_sram`/`msh_rom` macros, and macro bits are priced at about 1 Mbit/mm2 and count toward the area budget.

**Scoring and audit.** `harness/scoring.py` provides correctness judgment plus synthesis metric helpers (NE/latch/lint thresholds; there is no numeric scoring). `harness/audit.py` checks integrity, including the rule that any ROM a design uses must have a disclosed generator - no frozen numeric artifacts ship with the task. `harness/memmap.py` and `harness/check_integrity.py` validate the memory image and the repo tree.

## The Repository Layout: What Kimi-K3 Produced

The repository is the complete deliverable of an LLM designing a chip. Every directory is a piece of real hardware engineering work.

![nano-kpu repository layout](/assets/img/diagrams/nano-kpu/nano-kpu-repository-layout.svg)

### What is in the Repo

**`rtl/`** is the chip RTL. The top module is `msh_chip_top` and `filelist.f` is the compile manifest. The language is Verilog-2005 or the SystemVerilog subset that both Verilator 5 and yosys 0.66 accept. The RTL must be plain readable source - no vendor IP blobs, no pre-synthesized netlists, no encrypted modules. Inside `rtl/`, the `roms/` subdirectory holds the LUT init hex files (for sigmoid, alpha, expneg, rsqrt, and recip) plus generator notes, and the `selfmodel/` subdirectory holds a bit-exact Python fixed-point model that serves as a simulation aid.

**`reference/`** is the float32 golden reference model (`reference/model.py` is normative - prose never overrides it). Goldens are recomputed at evaluation time, not checked in frozen, which closes the loophole of a design overfitting to a specific numeric artifact.

**`harness/`** is the functional simulation and performance evaluation flow. It contains `evaluate.py` (the main entry point for correctness, cycles/token, and throughput), the Verilator C++ testbench in `tb/`, the `msh_sram` and `msh_rom` macro models in `macros/` (sim injection plus synth blackbox), the yosys synthesis scripts (`synth_area.ys`, `synth_tech.ys`, `synth_netlist.ys`), the cell libraries fetched at setup time, and the scoring/audit/memmap/integrity helpers.

**`docs/`** holds the specifications: `architecture.md` (what to compute), `interface.md` (the bus protocol), `memory_map.md` (the image layout), `quantization.md` (the weight format and numeric freedom), and `TASK_SPEC.md` (deliverables, environment, and evaluation mechanics). **`weights/`** has weight format notes. **`scripts/`** has the conda-centric setup script and the generated PATH file. A **`Makefile`** provides unified entry points: `setup`, `synth`, `lint`, `audit`, and `selftest`.

### Quick Start

```bash
# First time: toolchain env + cell library
bash scripts/setup_env.sh
# Put the toolchain env on PATH
source scripts/kpu-env.sh

# Functional sim, short sequence (minutes)
python3 harness/evaluate.py --quick

# Full evaluation: sim + timing/area synthesis (hours)
python3 harness/evaluate.py

# Functional + randreset + latency, no synthesis
python3 harness/evaluate.py --skip-synth

# Area + timing flow only (no sim)
make synth

# Individual checks
make lint
make audit
make selftest
```

## Key Features

| Feature | Description |
|---------|-------------|
| Designed by Kimi-K3 | The RTL was designed and implemented fully by Kimi-K3, not a human |
| Nano hybrid stack | KDA layer + NoPE MLA layer + sigmoid MoE + attention-residual mixing |
| KDA linear attention | Delta rule with per-channel decay, kernel-4 causal conv, fixed recurrent state |
| NoPE MLA | Full softmax attention, no positional encoding anywhere, compressed KV latent |
| Sigmoid MoE | 8 SwiGLU experts, top-2 routed + 1 shared, score-correction bias |
| Attention residuals | Block size 2, softmax mixture over preceding-block snapshots |
| Int4 group-128 | AWQ/GPTQ-style packed weights, ~238k params, ~135.5 KB |
| msh_chip_top | Synthesizable Verilog-2005/SV, top module with exact port stub |
| 128-bit DRAM port | 16 B/cycle, in-order reads, latency-elastic, posted writes |
| msh_sram / msh_rom | Macro-enforced storage; inferred $mem_v2 is rejected |
| Correctness gates | cosine >= 0.98, argmax >= 0.99, rand-reset byte-identical |
| Synthesis-stage metrics | yosys + Nangate45, frozen parameterization, not sign-off P&R |
| Verilator 5 + yosys | Conda env, Nangate45 fetched from OpenROAD (pinned by SHA-256) |
| Apache 2.0 | Open source; Nangate45 cell lib not bundled |

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| `macros_only` gate fails | Inferred `reg [W-1:0] mem [0:D-1]` arrays survive synthesis | Replace all arrays with `msh_sram` / `msh_rom` macro instances |
| Run cannot PASS | Read beats below `floor_weight_bytes` | Ensure the dense int4 weight stream is actually read each token |
| Outputs differ under rand-reset | Design relies on zero-initialized flops | Make all state initialization explicit via `rst_n` only |
| Elasticity ratio far from 1.0 | Constant 24 baked into read control | Make the design latency-elastic; keep multiple reads in flight |
| Timing/area not improving | Flow uses frozen parameterization | Expected - the flow does not search for design-specific optima |
| `verilator`/`yosys` not found | PATH not sourced | Run `source scripts/kpu-env.sh` after setup |
| Nangate45 missing | Cell lib not fetched | Re-run `scripts/setup_env.sh` (network access to raw.githubusercontent.com required) |
| ROM audit fails | Frozen numeric artifact without a disclosed generator | Provide a generator script for every LUT init hex |

## Conclusion

nano-kpu is the most concrete evidence yet that frontier LLMs can do real hardware engineering, not just write code. Kimi-K3 produced a synthesizable inference chip that runs a miniature version of its own architecture - the same KDA, NoPE MLA, sigmoid MoE, and attention-residual mixing that define the 2.8T-parameter model, scaled to 2 layers and 64 dimensions. The deliverable is not a toy: it includes a Verilator testbench, a yosys/Nangate45 synthesis flow, strict correctness gates against a float32 reference, a macro-enforced storage discipline, a latency-elastic memory protocol, and a fixed, frozen measurement methodology.

For anyone tracking the frontier of what LLMs can do in hardware design, this repo is the reference artifact. The connection to the [Kimi Delta Attention](/Understanding-Kimi-Delta-Attention/) work is direct: the chip is, in a real sense, K3 building a tiny version of itself in silicon.

## Links

- [nano-kpu on GitHub (MoonshotAI/nano-kpu)](https://github.com/MoonshotAI/nano-kpu)
- [nano-kpu architecture.md](https://github.com/MoonshotAI/nano-kpu/blob/main/docs/architecture.md)
- [nano-kpu interface.md](https://github.com/MoonshotAI/nano-kpu/blob/main/docs/interface.md)
- [nano-kpu quantization.md](https://github.com/MoonshotAI/nano-kpu/blob/main/docs/quantization.md)
- [Kimi Linear paper (arXiv:2510.26692)](https://arxiv.org/abs/2510.26692)
- [Nangate45 from OpenROAD-flow-scripts](https://github.com/The-OpenROAD-Project/OpenROAD-flow-scripts)
- [Verilator](https://www.veripool.org/verilator/)
- [yosys synthesis framework](https://github.com/YosysHQ/yosys)
