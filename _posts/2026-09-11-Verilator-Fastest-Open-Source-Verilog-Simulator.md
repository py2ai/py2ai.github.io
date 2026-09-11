---
layout: post
title: "Verilator: The Fastest Open Source Verilog and SystemVerilog Simulator"
description: "Verilator is the fastest Verilog and SystemVerilog simulator, open source under LGPL-3.0 and Artistic-2.0, guided by the CHIPS Alliance under the Linux Foundation. Rather than interpreting RTL at runtime, Verilator compiles Verilog or SystemVerilog into optimized, optionally multithreaded C++ or SystemC that runs as native code. On a single thread it is about 100x faster than interpreted simulators like Icarus Verilog and about 10x faster than standalone SystemC; multithreading adds another 2-10x, for 200-1000x total over interpreted simulators. It performs similar to or better than closed-source commercial simulators (VCS, Questa, Riviera-Pro, NC-Verilog) at zero license cost. It accepts Verilog-2005 and the SystemVerilog subset that Yosys accepts, performs lint checks, optionally inserts assertions and coverage, supports VCD/FST traces, and has out-of-the-box support for Arm and RISC-V vendor IP. Over 700 contributors. This post breaks down the verilation pipeline, the performance math, the usage modes, and the ecosystem."
date: 2026-09-11
header-img: "img/post-bg.jpg"
permalink: /Verilator-Fastest-Open-Source-Verilog-Simulator/
featured-img: ai-coding-frameworks/ai-coding-frameworks
tags:
  - Verilator
  - Verilog
  - SystemVerilog
  - Hardware Simulation
  - EDA
  - Open Source
  - RISC-V
  - CHIPS Alliance
author: PyShine
---

## What is Verilator

Verilator is the fastest Verilog and SystemVerilog simulator. It is open source, guided by the [CHIPS Alliance](https://chipsalliance.org/) under the [Linux Foundation](https://www.linuxfoundation.org/), with over 700 contributors, and licensed under the GNU Lesser General Public License Version 3 or the Perl Artistic License Version 2.0. The code is at [github.com/verilator/verilator](https://github.com/verilator/verilator), the documentation is at [verilator.org](https://verilator.org), and the main site is [veripool.org/verilator](https://www.veripool.org/verilator/).

The core idea is simple and it is the reason for the speed. Verilator does **not** interpret RTL at runtime. It **compiles** Verilog or SystemVerilog into optimized, optionally thread-partitioned C++ or SystemC that runs as native code. The simulation cost is paid once at compile time; every cycle of the run afterward executes as compiled native instructions, not as an interpreter walking a runtime data structure. The result is a compiled Verilog model that, on a single thread, runs over 10x faster than standalone SystemC and about 100x faster than interpreted simulators such as [Icarus Verilog](https://steveicarus.github.io/iverilog).

Verilator accepts Verilog or SystemVerilog, performs lint code-quality checks, and optionally inserts assertion checks and coverage-analysis points. It can automatically generate a simulator executable using `--binary`, or you can write your own C++/SystemC wrapper to instantiate the model. It supports all design constructs, most verification constructs, intra-assignment delays, and events. Tristate-bus (`z`) and unknowns (`x`) are handled in limited contexts for performance, which is the main trade-off: if you need SDF annotation, mixed-signal simulation, or a full drop-in replacement for every closed-source simulator feature, Verilator may not be the best fit. If you want high-speed simulation or a path to migrate SystemVerilog into C++/SystemC, it is the tool for the job.

## The Verilation Pipeline

Verilator is invoked with parameters similar to GCC or Synopsys VCS. The process of turning RTL into a runnable simulator is called "verilation."

![Verilator verilation pipeline](/assets/img/diagrams/verilator/verilator-verilation-pipeline.svg)

### Understanding the Pipeline

**Parse and lint.** Verilator reads the Verilog or SystemVerilog source files and performs lint checks. The lint step is not cosmetic: because Verilator compiles rather than interprets, it catches a class of issues at compile time that an interpreter would only surface at runtime, and it refuses to emit code for constructs it cannot prove are sound.

**Assertions and coverage (optional).** Verilator can insert assertion checks and coverage-analysis points into the generated code. These are opt-in via flags like `--assert` and `--coverage`, and they add runtime overhead only when enabled.

**Optimize and thread-partition.** This is where the speed comes from. Verilator folds constants, eliminates dead logic, and optionally splits the design across threads for parallel simulation. The optimization is static, done at compile time, so the runtime cost is only the optimized logic that survived. When multithreading is enabled, Verilator partitions the design into thread-local regions that can be evaluated concurrently, which is where the 2-10x multithread speedup over its own single-thread mode comes from.

**Emit C++ or SystemC.** The optimized model is emitted as `.cpp` and `.h` files, the "Verilated" code. This is the key architectural decision: the output is C++ that a standard compiler can optimize further, not a proprietary runtime blob.

**Build the executable.** With `--binary`, Verilator generates the wrapper and compiles it in one step. Without `--binary`, you write a C++ or SystemC testbench that instantiates the Verilated model, compile it with g++ or clang++, and produce the simulator executable. Either way, the final artifact is a native executable running compiled RTL.

**Why this is fast.** An interpreted simulator walks a runtime data structure every cycle, dispatching on signal types and scheduling events. Verilator instead runs compiled C++ that has already been through the host compiler's optimizer. The per-cycle cost is a function call into straight-line native code, not an interpreter loop. This is the structural reason for the 100x single-thread and 200-1000x multithread speedups over interpreted simulators.

## Performance: Where Verilator Lands

The performance numbers are the headline. Verilator does not directly translate Verilog HDL to C++ or SystemC; it compiles into a much faster, optimized, and optionally thread-partitioned model that is then wrapped inside a C++/SystemC module.

![Verilator performance comparison](/assets/img/diagrams/verilator/verilator-performance-comparison.svg)

### Understanding the Numbers

**Versus interpreted simulators.** On a single thread, Verilator is about 100x faster than interpreted simulators like Icarus Verilog. Multithreading adds another 2-10x, yielding 200-1000x total over interpreted simulators. This is the gap that makes Verilator the default for large, long-running regressions where interpreted simulators are too slow to be usable.

**Versus standalone SystemC.** Verilator on a single thread is over 10x faster than standalone SystemC. The reason is the same: standalone SystemC still carries the SystemC kernel's scheduling overhead, while Verilator folds the scheduling into compiled code when possible.

**Versus commercial simulators.** Verilator has typically similar or better performance versus closed-source simulators including Aldec Riviera-Pro, Cadence Incisive/NC-Verilog, Mentor ModelSim/Questa, Synopsys VCS, VTOC, and Pragmatic CVer/CVC. The performance is comparable, but the cost is not: Verilator is open source, so you spend on compute rather than licenses. This is what "best simulation cycles per dollar" means in practice: a CI farm can run Verilator across thousands of cores without per-seat license servers.

**Where it is not the best fit.** The speed comes from compile-time optimization and limited X/Z handling, which means there are tasks where a commercial simulator remains the better choice: SDF annotation (back-annotated timing for sign-off), mixed-signal simulation, and any flow that depends on a closed-source simulator feature Verilator does not replicate. For everything else, Verilator is the default.

## Usage Modes and Build Flow

Verilator supports several output modes, chosen by CLI flags. All modes produce the same Verilated C++ model; they differ in how that model is wrapped and compiled.

![Verilator usage modes](/assets/img/diagrams/verilator/verilator-usage-modes.svg)

### The Four Modes

**Mode 1: `--binary` (quickest path).** Verilator generates the wrapper and compiles it in one step, producing a simulator executable directly. This is the fastest way to get from RTL to a runnable simulation, and it is the mode used for quick checks and smoke tests.

**Mode 2: C++ custom wrapper (most common).** You write a C++ testbench that instantiates the Verilated model, drives the inputs, and checks the outputs. This is the standard mode for real verification, because it gives you full control over the testbench, the stimulus, and the checking. The nano-kpu chip we covered [earlier](/nano-kpu-Kimi-K3-Inference-Chip/) uses exactly this pattern: a Verilator C++ testbench (`harness/tb/tb_main.cpp`) drives the `msh_chip_top` module and models the 128-bit DRAM port.

**Mode 3: SystemC wrapper.** The Verilated model is instantiated as a SystemC module, for Transaction-Level Modeling (TLM) and Electronic System Level (ESL) flows where the design needs to interoperate with other SystemC IP.

**Mode 4: Library link (co-simulation).** Verilator generates a Verilated library that can be linked into other simulators, optionally encrypted. This is the path for integrating Verilator's speed into a larger simulation environment without abandoning the existing toolchain.

### Common CLI Flags

| Flag | Purpose |
|------|---------|
| `--cc` | Emit C++ (the default output mode) |
| `--binary` | Auto-generate wrapper and compile in one step |
| `--sc` | Emit SystemC |
| `--trace` | Enable VCD waveform output |
| `--trace-fst` | Enable FST (compressed) traces |
| `--coverage` | Insert coverage-analysis points |
| `--assert` | Insert assertion checks |
| `--public-flat` | Expose internal signals for debugging |
| `--pins <N>` | Set the port pin width |
| `--Mdir <dir>` | Set the output directory |
| `-j <N>` | Enable multithreaded partitioning across N threads |

A minimal `--binary` invocation:

```bash
verilator --cc --binary --trace top.v --exe tb_main.cpp
```

This reads `top.v`, emits the Verilated C++, wraps it, compiles against `tb_main.cpp`, and produces a simulator executable that also writes VCD traces.

## The Ecosystem

Verilator sits at the center of the open hardware verification ecosystem. Its speed and openness make it the default simulator for a wide range of projects.

![Verilator ecosystem](/assets/img/diagrams/verilator/verilator-ecosystem.svg)

### Governance and IP Support

Verilator is guided by the [CHIPS Alliance](https://chipsalliance.org/) under the [Linux Foundation](https://www.linuxfoundation.org/), with over 700 contributors. It has out-of-the-box support for Arm IP and RISC-V vendor IP, which means the standard processor cores used across open hardware and much of industry simulate without custom setup. [Cocotb](https://www.cocotb.org/) provides a Python coroutine-based cosimulation library that officially supports Verilator, so teams that write testbenches in Python can drive Verilator directly.

### Waveform Viewers

Verilator emits VCD and FST trace files, which can be viewed with [GTKWave](https://gtkwave.sourceforge.net/) (the classic open-source waveform viewer) or [Surfer](https://surfer-project.org/) (a modern web and offline viewer). The trace output is opt-in via `--trace` or `--trace-fst`.

### Fallback: Icarus Verilog

[Icarus Verilog (iverilog)](https://steveicarus.github.io/iverilog) is a highly-featured interpreted Verilog simulator. It is slower than Verilator by roughly 100x, but it has fuller feature coverage for edge cases Verilator does not handle. The common pattern is to use Verilator for the fast path (regressions, CI, long runs) and fall back to Icarus when a specific construct or feature requires it.

### Real-World Use: nano-kpu

The [nano-kpu](/nano-kpu-Kimi-K3-Inference-Chip/) repo we covered earlier is a concrete example of Verilator in production use. Kimi-K3's chip design uses Verilator 5.x (baseline 5.050) as the functional simulation testbench: the C++ testbench `harness/tb/tb_main.cpp` is normative, it models the 16 B/cycle DRAM port, and it drives the `msh_chip_top` module through the RUN/DONE command protocol. The correctness gates (per-row cosine, argmax agreement, random-reset byte-identical outputs) are all evaluated against the Verilator simulation output. This is the pattern: Verilator is the simulation engine, the testbench is C++, and the evaluation harness checks the simulation output against a reference.

## Key Features

| Feature | Description |
|---------|-------------|
| Compiled, not interpreted | RTL is compiled to optimized C++/SystemC that runs as native code |
| 100x faster than interpreted (single-thread) | About 100x faster than Icarus Verilog on one thread |
| 200-1000x with multithreading | 2-10x additional speedup from thread partitioning |
| Comparable to commercial | Similar or better than VCS, Questa, Riviera-Pro, NC-Verilog |
| Open source | LGPL-3.0 or Artistic-2.0, no license fees |
| Verilog + SystemVerilog | Accepts Verilog-2005 and the SystemVerilog subset Yosys accepts |
| Lint checks | Code-quality checks at compile time |
| Assertions + coverage | Optional `--assert` and `--coverage` insertion |
| VCD / FST traces | `--trace` and `--trace-fst` for waveform output |
| `--binary` one-step | Auto-generate wrapper and compile in one command |
| C++ / SystemC wrappers | Full control over testbench via custom wrappers |
| Library link | Verilated libraries can link into other simulators |
| Multithreaded | `-j N` for thread-partitioned parallel simulation |
| Arm + RISC-V IP | Out-of-the-box support for major vendor IP |
| CHIPS Alliance / Linux Foundation | Community governance, 700+ contributors |

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| Verilator rejects a construct | SystemVerilog feature outside the supported subset | Check the manual; fall back to Icarus Verilog for that module |
| Simulation mismatches vs commercial sim | X/Z propagation handled differently | Verilator handles X/Z in limited contexts; constrain inputs to 0/1 where possible |
| No speedup over interpreted | Running without `--binary` or not compiling | Ensure the Verilated C++ is compiled to an executable, not run through an interpreter |
| Poor multithread scaling | Design not partitionable across threads | Check the thread partition report; reduce combinational paths between partitions |
| No VCD traces | Trace output not enabled | Add `--trace` (VCD) or `--trace-fst` (FST) to the verilator command |
| Lint errors block build | Verilator enforces lint at compile time | Fix the lint warnings; use `--Wno-<warning>` to suppress specific ones if intentional |
| Commercial simulator feature missing | SDF annotation, mixed-signal, etc. | Use a commercial simulator for that flow; Verilator does not cover every feature |
| Slow compile time | Large design + full optimization | Use `--no-` flags to disable unused optimizations; incremental builds help |

## Conclusion

Verilator's position in the hardware verification stack is earned, not claimed. It is the fastest open source Verilog and SystemVerilog simulator because it makes a different architectural bet than interpreted simulators: it compiles RTL to optimized C++ that runs as native code, paying the cost once at compile time instead of every cycle at runtime. The 100x single-thread and 200-1000x multithread speedups over interpreted simulators, combined with comparable performance to commercial tools at zero license cost, make it the default for regressions, CI, and any long-running simulation where interpreted tools are too slow and commercial tools are too expensive.

For open hardware, RISC-V development, AI chip design, and any team that wants to spend on compute rather than licenses, Verilator is the simulation engine. The [nano-kpu](/nano-kpu-Kimi-K3-Inference-Chip/) repo is a recent, concrete example: the chip Kimi-K3 designed is verified by a Verilator C++ testbench, and the correctness gates are evaluated against its simulation output. When an LLM designs a chip and the verification tool of choice is Verilator, that tells you where the tool sits in the ecosystem.

## Links

- [Verilator website (veripool.org)](https://www.veripool.org/verilator/)
- [Verilator on GitHub (verilator/verilator)](https://github.com/verilator/verilator)
- [Verilator documentation (verilator.org)](https://verilator.org/verilator_doc.html)
- [Verilator installation guide](https://verilator.org/install)
- [CHIPS Alliance](https://chipsalliance.org/)
- [Linux Foundation](https://www.linuxfoundation.org/)
- [Cocotb (Python cosimulation)](https://www.cocotb.org/)
- [Icarus Verilog (iverilog)](https://steveicarus.github.io/iverilog)
- [GTKWave (waveform viewer)](https://gtkwave.sourceforge.net/)
- [Surfer (waveform viewer)](https://surfer-project.org/)
