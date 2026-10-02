---
layout: post
title: "EvoOntology: A Self-Evolving Ontology Layer for Data Agents - Inside ruc-datalab/EvoOntology"
description: "EvoOntology is an MIT-licensed Python research platform from Renmin University of China that gives data agents a versioned, workload-grounded ontology layer over MCP, then evolves it from recorded interaction trajectories under a gated Accept/Reject evaluation. We tour the source behind its typed semantic store, semantic MCP runtime, evolution state machine, and three benchmark adapters."
date: 2026-10-02
header-img: "img/post-bg.jpg"
permalink: /EvoOntology-A-Self-Evolving-Ontology-Layer-For-Data-Agents/
featured-img: "ai-coding-frameworks/ai-coding-frameworks"
image: https://pyshine.com/assets/img/diagrams/evoontology/ruc-datalab-evoontology-architecture.svg
tags:
  - Python
  - MCP
  - Data Agents
  - AI Agents
categories: [AI, Open Source]
keywords: "EvoOntology, self-evolving ontology, data agents, ontology layer, MCP, Claude Code plugin, Codex plugin, text-to-SQL, BIRD, DDR-10K, InsightBench, semantic layer, trajectory recording, evaluation gate, RUC DataLab"
author: "PyShine"
---

Raw data leaves its meaning implicit. Table names, column headers, file paths, and isolated observations rarely explain what a metric actually counts, which entities relate to which, or what constraints the business assumes. Agents that work over heterogeneous tables, files, and databases have to infer that semantics repeatedly, one task at a time, and they get it wrong in ways that are hard to notice. [ruc-datalab/EvoOntology](https://github.com/ruc-datalab/EvoOntology), a research platform from Renmin University of China's DataLab described in an arXiv paper, attacks this agent-data gap with a versioned Ontology Layer that agents query through MCP tools, grounded in real workload evidence, and — the interesting part — continuously adapted from the agent's own execution trajectories.

The design borrows a discipline from training loops: treat the ontology layer as trainable agent state rather than model weights. A builder initializes grounded semantic objects from the workload and the underlying data. After the agent works, recorded trajectories trigger an evolution session that diagnoses recurring behavior, attributes it to a specific layer, and proposes a bounded patch. Every candidate is validated against its parent with the same data, agent, and decoding settings, and is published only when paired evaluation shows a reproducible improvement. Changes stay inspectable, comparable, and reversible.

The source is worth a tour because the Python core is deliberately deterministic — all the judgment-heavy work lives in agent skills, while the runtime implements storage, validation, trajectory recording, and a strict evolution state machine with zero runtime dependencies. Let us walk through it.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/evoontology/ruc-datalab-evoontology-overview-architecture.svg" alt="Architecture overview of the ruc-datalab/EvoOntology repository" style="max-width:100%;height:auto;" />
</div>

*Overview of the EvoOntology codebase: the Claude Code and Codex plugins on the left, the deterministic core runtime in the middle, the versioned ontology data underneath, and the evolution session with its evaluation gate and benchmark adapters on the right.*

The walkthrough starts at the clients. Both a Claude Code plugin and a Codex plugin ship the same three skills — build, evolve, and explore the ontology — and launch the semantic MCP server over stdio. The server exposes two retrieval tools plus a set of workspace operations over a versioned semantic store. While the agent works, its tool calls are recorded as trajectories; when enough accumulate, an evolution run compares a candidate ontology against its parent and either publishes the next version or keeps the old one.

## Why You Need This

The first reason is semantic uncertainty. The Ontology Layer makes domain concepts, data mappings, relationships, and constraints explicit in a typed semantic graph with four node families — Terms, Mappings, Constraints, and Evidence — connected by semantic relations and structural references. Instead of guessing what a column means for the hundredth time, an agent looks it up and gets the definition plus the evidence that grounds it.

The second reason is that static semantic layers do not scale with use. Hand-authored layers need sustained expert maintenance, go stale as data and workloads shift, and consume growing context when injected in full. EvoOntology inverts that: the agent retrieves only the semantics needed for the current step through `browse_semantics` and `resolve_semantics`, initialized by a compact session manifest rather than a full ontology dump. And because interaction trajectories are recorded, the layer adapts to how the work actually behaves instead of how it was specified.

The third reason is controlled evolution. Ontology updates are not free-form edits. They happen inside an evolution session with a round budget, a frozen evaluation batch, and a gate that publishes a candidate only when paired evaluation shows it beats its parent. The repository's benchmark table reports that this loop pays: across the four-backbone analysis subset, the initial ontology layer improves over ReAct without one, and self-evolution adds a further gain on all three benchmarks — including a trajectory-wise score of 89.5 versus 69.5 on DDR-Bench, execution accuracy of 72.4 versus 63.6 on BIRD, and an insight score of 54.2 versus 53.2 on InsightBench.

## How It Works

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/evoontology/ruc-datalab-evoontology-architecture.svg" alt="Detailed architecture of the ruc-datalab/EvoOntology repository" style="max-width:100%;height:auto;" />
</div>

*Detailed architecture of EvoOntology: the plugin layer with skills and hooks, the JSON-RPC MCP runtime with its retrieval tools and workspace operations, the versioned semantic store with typed models and validation, the evolution lifecycle from trajectory recording through the evaluation gate, the three benchmark environments, and the docs, tests, and release tooling.*

**The semantic store is versioned like a git branch.** The ontology package defines the five record types — terms, mappings, relations, constraints, and evidence — each carrying lifecycle state and confidence. A store class loads, saves, lists, and promotes versions, publishes a candidate as the next numbered version, and switches the active pointer, all under a `.evoontology` workspace directory whose project file also records the mode. A standalone validator checks that every reference between records points at something that exists, exposed both as a library function and as a console entry point.

**The runtime speaks MCP over stdio with two retrieval tools.** `browse_semantics` finds up to six catalog entries by query and kind; `resolve_semantics` resolves up to five mentioned concepts with their evidence and relations. A compact manifest initializes each session with tool descriptions and usage guidance, so the agent learns how to ask without anyone injecting the full layer. Alongside the retrieval tools, a workspace-operations module exposes the whole lifecycle to the agent: validate, visualize, list and set versions, start, resume and finalize evolution runs, begin and record rounds, record evaluations, confirm trajectory sources, accept, mark incomplete, extend the budget, and publish a completed build.

**Trajectories are recorded at tool-call granularity, deliberately shallow.** A trajectory store appends one record per task with the tool calls and outcomes, filtering for the semantic tools and truncating results — no chain-of-thought is captured. An evolution trigger watches the accumulated count since the last checkpoint and fires when thresholds are met, then advances its checkpoint after a successful run.

**The evolution session is an explicit state machine.** A run freezes its budget and evaluation data, then loops through rounds: begin with a hypothesis and candidate version, record the diagnosis and patch, record the evaluation, and either reject (which supplies input to the next round) or accept. Only accept or a valid incomplete state is terminal — a budget that runs out or an external block leaves the run incomplete, publishing nothing and advancing nothing. Acceptance passes a publication gate that switches the active version and advances the trigger checkpoint.

**Evaluation is paired, gated, and blind by default.** An evaluation gate decides between two modes: absolute ground-truth scores for benchmarks that have them, or an LLM judge doing an A/B comparison where the parent and candidate answers are anonymized before judging and decoded after. The benchmark integration contract keeps this honest: each environment implements an adapter with a single evaluate call returning normalized metrics and case results, so the gate compares parent and candidate under identical conditions.

**Two modes separate research from production.** In `fixed_split` mode, for benchmarks with fixed questions and ground truth, a construction pool supports building and diagnosis while a validation reserve is used only for the final gate and never flows back into construction. In `rolling_trajectory` mode, for production use or cold starts without a test set, a seed workload initializes the first version and later tasks accumulate until the trigger freezes a batch for evaluation. Both modes share the same workspace, versions, and checkpoints.

**Three benchmark environments plug in through one contract.** BIRD covers text-to-SQL over real-world databases, DDR-10K covers open-ended research over heterogeneous financial data, and InsightBench covers iterative business analysis and insight generation. A registry discovers them all, listable with a single command, and each one preserves its native rollout and evaluation protocol behind the adapter. The same repository also ships the tooling that keeps the plugins honest: a synchronizer script copies the deterministic core into both plugin packages, and tests guard that parity along with the state machine, runtime, store, and visualization.

The end-to-end flow reads cleanly. Build probes the workload, persists evidence, and publishes the first version. The agent resolves semantics through MCP while its calls are recorded. The trigger fires, a session freezes data and budget, the agent diagnoses and patches a candidate, the gate evaluates parent against candidate, and accept publishes the next version — or the rejection feeds the next round. The layer ends up smarter about the exact workload it serves, and every step of that is inspectable.

## Advantages

- **Explicit semantics**: terms, mappings, constraints, and evidence are typed records with lifecycle and confidence, not comments in a schema file.
- **On-demand retrieval**: the compact manifest plus two MCP tools keep context small; the full ontology is never injected wholesale.
- **Gated self-evolution**: candidates publish only when paired evaluation shows a reproducible improvement over the parent, with reject looping forward as signal.
- **Deterministic core, intelligent skills**: Python handles storage, validation, and the state machine; judgment stays in the agent skills that drive it.
- **Pluggable evaluation**: one adapter contract connects text-to-SQL, financial research, and business-analysis benchmarks, plus a documented path for adding new ones.
- **Zero runtime dependencies**: the core package requires nothing beyond Python itself, with the ontology explorer rendered from a bundled template.

## Benefits

- **Agents stop re-deriving the same meaning**: grounded knowledge is reused across tasks instead of rediscovered per request.
- **Errors become evidence**: failed or awkward trajectories are recorded and later attributed to a specific layer, so the same mistake is less likely twice.
- **Maintenance is reversible**: versioned stores and an active-version pointer mean a bad candidate never overwrites a good parent.
- **Blind judging by default**: anonymized A/B evaluation keeps the gate honest when ground truth is unavailable.
- **Research reproducibility**: fixed data splits, frozen batches, and identical rollout conditions make reported gains checkable rather than anecdotal.
- **Fits existing agent workflows**: installation through the Claude Code or Codex plugin marketplaces takes three commands, and the ontology explorer opens automatically after builds and evolutions.

## Usage

Install the Claude Code plugin from the GitHub marketplace — no clone, no virtual environment:

```bash
claude plugin marketplace add ruc-datalab/EvoOntology
claude plugin install evoontology@evoontology
claude plugin list
```

Start a session and run the three skills as slash commands: `/evoontology:build-ontology` to probe the workload and publish the first version, `/evoontology:explore-ontology` to inspect what was built, and `/evoontology:evolve-ontology` when the reminder hook nudges you that enough trajectories have accumulated. For Codex, the marketplace flow is the same shape, and the skills are invoked with dollar-style commands in a thread.

Once built, the data agent calls `browse_semantics` and `resolve_semantics` with no further configuration. To work with the benchmark environments, the registry lists what is available:

```sh
python -m benchmarks list
```

Each environment ships its own configuration for baseline and ontology conditions, a run script for rollouts, and an evaluation runner; the integration guide in the docs explains the adapter, data loader, and seed-skill contract for adding your own. A usage guide covers workspace layout, the lifecycle, and configuration end to end, and the architecture guide explains module boundaries and the evolution state machine in detail.

## Conclusion

EvoOntology makes a claim that is easy to state and hard to do: an agent's semantics can be trained the way skills are trained — proposed from evidence, evaluated against a parent, and published only on demonstrated improvement. The repository backs the claim with an unusually clean separation. Skills hold the judgment; a dependency-free Python core holds the store, the tools, the trajectories, and a state machine that refuses to publish what it cannot justify; benchmark adapters hold the evaluation honest. If your agents work over tables, files, or databases and you are tired of semantics that live in one person's head, this project is a serious and well-engineered place to start.

Links:

- [ruc-datalab/EvoOntology on GitHub](https://github.com/ruc-datalab/EvoOntology)
- [arXiv paper](https://arxiv.org/abs/2609.15779)
- [Usage guide](https://github.com/ruc-datalab/EvoOntology/blob/master/USAGE.md)
- [Architecture guide](https://github.com/ruc-datalab/EvoOntology/blob/master/docs/architecture.md)
- [Claude Code plugin](https://github.com/ruc-datalab/EvoOntology/tree/master/plugins/claude-code)
