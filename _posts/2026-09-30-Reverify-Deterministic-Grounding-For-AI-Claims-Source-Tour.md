---
layout: post
title: "Reverify: Deterministic Grounding for AI Claims - Inside 2akouwu/reverify"
description: "Reverify makes deterministic tools the judge of everything a language model says: the model proposes claims, pure-Python checkers test each one against ground truth, and only VERIFIED evidence with receipts survives. A source-level tour of the propose-verify pipeline in 2akouwu/reverify — claim extraction, the checker registry, verdict gating, and the durable ledger that carries grounded facts across context resets."
date: 2026-09-30
header-img: "img/post-bg.jpg"
permalink: /Reverify-Deterministic-Grounding-For-AI-Claims-Source-Tour/
featured-img: "ai-coding-frameworks/ai-coding-frameworks"
image: https://pyshine.com/assets/img/diagrams/reverify/2akouwu-reverify-architecture.svg
tags:
  - AI Agents
  - Hallucination
  - Code Verification
  - Open Source
categories: [AI, Open Source]
keywords: "reverify, AI hallucination, claim verification, deterministic verification, MCP server, binary analysis, reverse engineering, LLM grounding, context rollover, Python, agent tools, code equivalence"
author: "PyShine"
---

Language models are fluent explainers and unreliable witnesses. Ask one to read a binary, audit a refactor, or describe a function it cannot see, and it will happily produce offsets, struct fields, and API names that simply do not exist — delivered in the same confident tone as everything else. The team behind 2akouwu/reverify states the problem in one line on the project page: "Stop your AI from making things up." What makes the repository interesting is that the answer is not another prompt or guardrail; it is an architectural rule — *the model proposes, deterministic tools decide*.

Reverify is a Python package (MIT licensed, installable from PyPI as `reverify`, currently at version 0.11.0) that turns that rule into a working pipeline. The model emits structured JSON *claims* about an artifact; a verifier checks each claim against the actual bytes with parsers, a disassembler, an emulator, a pattern scanner, and an SMT-backed equivalence prover; and the claim comes back `VERIFIED`, `REFUTED`, or `INCONCLUSIVE` with the observed evidence attached. Nothing the model says on its own is ever promoted to a fact.

The source is worth a tour because it is a rare, complete implementation of an idea many teams talk about: grounding. Rather than asking a model to be honest, Reverify removes the option of being believed. In this post we walk the two architecture diagrams generated from the actual repository tree — an overview and a detailed module graph — and trace how a claim travels from a proposer to a checker to a verdict, and how the grounded facts that survive are stored so they outlive the context window that produced them.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/reverify/2akouwu-reverify-overview-architecture.svg" alt="Architecture overview of the 2akouwu/reverify repository" style="max-width:100%;height:auto;" />
</div>

*Overview of the Reverify architecture: agent-facing interfaces feed a propose-verify loop whose verdicts are gated by deterministic checkers, with a durable ledger and session orchestrator carrying state across context resets.*

Reading the overview from left to right: agents reach Reverify through two interfaces — the command-line entrypoint in `reverify/cli.py` and the Model Context Protocol server in `reverify/mcp_server.py`, which exposes tools like `re_verify_claim` to Claude Code, Cursor, and other MCP hosts. Both funnel into the loop at the center: the reconstruction agent in `reverify/agent.py` elicits claim proposals, and the verifier in `reverify/verifier.py` judges them. The verifier leans on the deterministic checker bank at the right — `binary.py` for parsing, `disasm.py` for instructions, `emulator.py` for execution, `behavior.py` for equivalence and proofs, `semantic.py` for angr-derived call graphs, and `backends.py` as the registry that decides whether each subsystem runs on an installed engine or the pure-Python fallback. Below the loop, `ledger.py` persists what the tools established and refuted, and `rollover.py` runs multi-session goals so that a fresh context starts from the ledger rather than a lossy summary.

## Why You Need This

The core failure mode Reverify targets is confident fabrication. In binary reverse engineering it is at its worst: a model asked to reconstruct a function prologue will propose the textbook `push rbp; mov rbp, rsp` sequence whether or not those bytes are actually there. The project's own benchmark documentation ([BENCHMARK.md](https://github.com/2akouwu/reverify/blob/main/BENCHMARK.md)) reports that on 71 real Windows system files, the model's textbook answer was wrong 97% of the time — and that the verification gate accepted none of those wrong claims. Whether or not you reproduce those exact numbers, the mechanism that makes them possible is in the code, and it is worth understanding.

Conventional mitigations do not fix this. Self-consistency checks ask the model to grade itself, which it is famously bad at. Prompt-based "cite your sources" does not work when there is no source, only bytes. Retrieval grounding helps for text corpora but says nothing about whether a claimed machine instruction exists at a claimed offset. The only judge that cannot be sweet-talked is a deterministic program that looks at the artifact — and that is precisely what Reverify's checker bank is.

There is a second, subtler problem: even an honest verifier can be gamed. "All claims verified" is trivially reachable by asserting trivia — the file starts with `MZ`, a `.text` section exists. Reverify counters this with an information-weighted scoring pass in `reverify/verifier.py`: every result carries a `weight`, and claims that merely restate the fact sheet the model was shown, duplicate earlier claims, echo the tools' own previous output, or reference inline content that does not occur in the binary score zero. A reconstruction counts as *grounded* only when nothing is refuted and the verified weight reaches the `--min-information` threshold (default 1.0). Being not-wrong is not enough; the verified set has to actually say something.

Finally, agent sessions die. Long tasks hit the context window, the harness compacts or the operator types `/clear`, and whatever the model had established evaporates into a summary — including, if you are unlucky, its unverified guesses, which re-enter the fresh context dressed as facts. Reverify treats that state problem as part of the verification problem, which is why a ledger and a rollover subsystem sit at the bottom of the architecture rather than being an afterthought.

## How It Works

The pipeline is best understood as three stations — claim extraction, a checker registry, and verdict gating — connected by the ledger that everything writes to.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/reverify/2akouwu-reverify-architecture.svg" alt="Detailed module architecture of the 2akouwu/reverify repository" style="max-width:100%;height:auto;" />
</div>

*Detailed module graph of the repository: CLI and MCP surfaces dispatch into the propose-verify loop, the verifier dispatches claims across the checker bank, optional engines upgrade checkers in place, and the ledger plus rollover harness persist grounded state.*

### Understanding the Architecture

**Claim extraction.** A claim is a small JSON object — a kind, kind-specific parameters, an optional note, an optional `id`, and optional `depends_on` references — parsed by `Claim.from_dict` in `reverify/verifier.py`. The supported kinds read like a checklist of things a model is tempted to assert: `bytes_at` and the typed reads `u16_at`/`u32_at`/`u64_at`, `pattern_present`, `string_present`, `instructions`, `emulate_result`, `behavior_equiv`, `prove_equiv`, `protobuf_field`, `import_present`, `export_present`, `section_present`, plus the semantic kinds `function_at`, `calls`, `references`, and `reachable_from_entry`. Offsets are file offsets unless the claim says `"space": "rva"` or `"va"`, and the verifier translates through the section table and echoes all three addresses in the evidence. Setting `"observe": true` flips the claim from assertion to question — the tools read the value instead of judging one.

**The proposal side of the loop.** `reverify/agent.py` implements `ReconstructionAgent`, which drives the verifier automatically: it shows the model a keyed fact sheet, asks for structured claims, checks every one, and feeds refutations — with the address where the expected bytes actually are — back for the next round. The model never sees raw bytes beyond a small addressed header; anything else it learns through `observe`. Multiple `--samples` per round let the verifier, not the model's confidence, select among competing proposals. The language model itself is injected as a plain `propose` callable (`openai_proposer` builds one; `demo_proposer` runs offline), so the whole loop is testable without network access.

**The checker registry.** `Verifier.verify` in `reverify/verifier.py` dispatches each claim to a `_check_<kind>` handler, and those handlers delegate to the deterministic bank: `reverify/binary.py` (`parse_binary`) for PE/ELF/Mach-O structure, `reverify/disasm.py` for x86/x64/ARM/ARM64 disassembly and AOB pattern scans, `reverify/emulator.py` for register-level execution, `reverify/protocol_parser.py` for schema-less Protobuf/TLV dissection, `reverify/behavior.py` for running a candidate implementation against a reference and for Z3 proof-grade equivalence, `reverify/exebench.py` for sandboxed re-executability checks via `reverify/sandbox.py`, and `reverify/semantic.py` for angr-backed functions and call graphs. `reverify/backends.py` is the registry's registry: it detects which optional engines (capstone, unicorn, lief, z3, angr) are importable and reports, per subsystem, whether an engine or the pure-Python core is judging — `reverify backends` prints it.

**Verdict gating.** Every check returns one of five verdicts defined at the top of `reverify/verifier.py`: `VERIFIED`, `REFUTED`, `INCONCLUSIVE`, `OBSERVED`, or `INVALIDATED` — the last for claims whose `depends_on` root was refuted, so a wrong foundation poisons everything built on it. `verify_all` then applies dependencies, runs the information-weighting `summarize`, and attaches a receipt from `Verifier.receipt()`: the binary's SHA-256, the reverify version, the Python and platform strings, and which engines judged. A report plus its receipt is replayable evidence rather than a claim to be trusted.

**Durable state.** `reverify/ledger.py` writes what the tools verified, observed, proved, and refuted to `.reverify/ledger/<sha256>.json` — content-keyed per binary, atomic, checkpointed every round. Facts carry strength tiers (`PROVEN`, `TESTED`, `VERIFIED`, `DERIVED`, `OBSERVED`), proof-grade facts are pinned in the bounded context view, and refutations come back as `KNOWN FALSE` so a fresh context does not re-propose the same wrong prior. Nothing the model merely said is ever stored.

**Sessions that roll over instead of rotting.** `reverify/rollover.py` runs a goal across fresh-context sessions with drivers for Claude (Agent SDK), OpenAI-compatible endpoints, and an offline mock; the model requests a rollover, and token budgets and drift detection force one when restatements dominate. `reverify/rollover_harness.py` wires the same discipline into interactive CLIs — Claude Code, Codex, Gemini CLI, OpenCode — via hooks and a launcher, with the hand-off written to files and receipts logged before a session is replaced.

The end-to-end flow, then: a model proposes claims through the CLI or MCP surface; the agent normalizes and batches them; the verifier resolves addresses, dispatches each claim to the right checker, and returns verdicts with evidence; the scoring pass weighs what verified; the ledger records what survived; and `reconstruct` iterates until the report says *grounded* — or the round cap hits and everything unverified stays unverified.

## Advantages

- **The judge cannot be persuaded.** Verdicts come from deterministic code paths in `reverify/verifier.py` executing against the artifact's actual bytes, not from a model's self-assessment — the same input always yields the same verdict.
- **Grounded is not just "nothing refuted."** Information weighting measures each verified claim against the binary itself — occurrence count and entropy — so trivia, padding, and echoed tool output score zero and cannot carry a reconstruction past the `--min-information` gate.
- **Graceful strength ladder.** With no dependencies installed the pure-Python core judges everything; installing the extras upgrades checkers in place (capstone, unicorn, lief, z3, angr per `reverify/backends.py`), and semantic verdicts are honestly recorded at a `DERIVED` tier below `VERIFIED`.
- **Refutations are teachable.** A wrong `bytes_at` reports where the expected bytes actually are; a wrong `calls` lists the function's real callees — the model can fix the claim instead of guessing again.
- **State survives context resets losslessly.** Because only tool-verified content is ever stored, the ledger can restore a fresh context exactly, including negative memory of known-false claims.
- **Agent-native surfaces.** The same verification is reachable as a CLI subcommand or as MCP tools (`re_verify_claim`, `re_ledger`), so an existing agent stack adopts it without rewrites.

## Benefits

- **Catches hallucinations before users see them.** Claims are checked against ground truth before they are reported, which is the exact insertion point where fabrication should be stopped.
- **Replayable evidence.** Every report carries a receipt with the binary's SHA-256, tool version, and engines used, so a verdict can be re-run and audited rather than believed.
- **Beyond binaries.** The `equiv` subcommand applies the same discipline to ordinary source code — running a candidate refactor and its reference over shared inputs, with a concrete counterexample on mismatch — so AI-written code is tested, not trusted.
- **Long tasks without compaction decay.** Ledger plus rollover lets multi-session work resume from established facts, and refuted priors do not resurface in later contexts.
- **Zero-dependency install.** `pip install reverify` works on Python 3.8+ with nothing else required; engines are optional accelerations, not requirements.
- **Honest about uncertainty.** When the tools cannot decide, the verdict is `INCONCLUSIVE` — never a guess dressed as a fact.

## Usage

Install from PyPI and triage a binary with the pure-Python core (no extra engines required):

```bash
pip install reverify        # pure-Python core; or "reverify[full]" for capstone+unicorn+lief
reverify auto sample.bin --json
```

Check a claim against the actual bytes — verdicts come back with the evidence the tools observed:

```bash
reverify verify sample.bin --claim '{
  "kind": "instructions", "offset": 4096,
  "mnemonics": ["push", "mov", "sub"], "note": "function prologue"
}'
```

Run the closed propose-verify loop, where a model proposes and the tools gate until the reconstruction is grounded, with state checkpointed to the ledger every round:

```bash
reverify reconstruct target.exe --goal "reconstruct the export table stubs" --json
reverify ledger target.exe    # what is established, what is known false
```

Compare two implementations of the same routine over shared inputs (execution requires opting in):

```bash
reverify equiv reference.py candidate.py --lang python
```

Expose the whole toolkit to an MCP host such as Claude Code or Cursor:

```bash
python reverify/mcp_server.py
```

## Conclusion

Reverify's contribution is an inversion of trust: the model is demoted from authority to proposer, and promotion to fact requires passing through deterministic checkers whose verdicts, weights, and receipts are all inspectable in the source. Reading `reverify/verifier.py` next to `reverify/agent.py` and `reverify/ledger.py` shows how far that principle is pushed — even the loop's memory refuses to store anything the tools did not establish. For teams wiring language models into binary analysis, security review, or code-equivalence checking, it is one of the clearest working reference implementations of propose-then-verify grounding available today.

Links:

- GitHub repository: [2akouwu/reverify](https://github.com/2akouwu/reverify)
- PyPI package: [reverify](https://pypi.org/project/reverify/)
- Walkthrough example: [EXAMPLE.md](https://github.com/2akouwu/reverify/blob/main/EXAMPLE.md)
- Benchmark methodology: [BENCHMARK.md](https://github.com/2akouwu/reverify/blob/main/BENCHMARK.md)
