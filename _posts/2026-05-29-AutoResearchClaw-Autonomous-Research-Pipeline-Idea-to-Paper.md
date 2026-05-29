---
layout: post
title: "AutoResearchClaw: Autonomous Research Pipeline From Idea to Paper"
description: "Learn how AutoResearchClaw transforms research ideas into conference-ready papers using a 23-stage autonomous pipeline with multi-agent debate, self-healing experiments, and human-in-the-loop co-pilot mode."
date: 2026-05-29
header-img: "img/post-bg.jpg"
permalink: /AutoResearchClaw-Autonomous-Research-Pipeline-Idea-to-Paper/
featured-img: ai-coding-frameworks/ai-coding-frameworks
categories: [AI Agents, Python, Research Tools]
tags: [AutoResearchClaw, autonomous research, AI research pipeline, paper generation, LLM agents, multi-agent systems, academic writing, self-evolving AI, research automation, Python]
keywords: "how to use AutoResearchClaw, AutoResearchClaw tutorial, autonomous research pipeline, AI paper generation tool, AutoResearchClaw vs alternatives, research automation with AI, how to generate academic papers with AI, self-evolving research agent, multi-agent debate research, AutoResearchClaw installation guide, conference paper automation"
author: "PyShine"
---

# AutoResearchClaw: Autonomous Research Pipeline From Idea to Paper

AutoResearchClaw is a fully autonomous and self-evolving research pipeline that transforms a single research idea into a conference-ready academic paper. Developed by AIMING Lab at UNC, this 23-stage Python pipeline handles everything from literature discovery and hypothesis generation to sandbox experiments, multi-agent peer review, and LaTeX export -- all without requiring human intervention, or with deep collaboration through its Co-Pilot mode.

> **Key Insight:** AutoResearchClaw's 23-stage pipeline processes research through 8 distinct phases -- from topic decomposition to citation verification -- producing NeurIPS/ICML/ICLR-ready papers with real literature from OpenAlex, Semantic Scholar, and arXiv, not hallucinated references.

## What Is AutoResearchClaw?

AutoResearchClaw is an open-source Python framework (MIT license) that automates the entire academic research workflow. Given a research topic, it:

1. Decomposes the topic into structured research questions
2. Searches real academic databases (OpenAlex, Semantic Scholar, arXiv) for relevant papers
3. Screens and extracts knowledge from collected literature
4. Synthesizes findings and generates testable hypotheses via multi-agent debate
5. Designs and runs experiments in a sandboxed environment
6. Analyzes results and makes autonomous PROCEED/REFINE/PIVOT decisions
7. Writes a conference-grade paper with anti-fabrication guards
8. Exports to LaTeX with 4-layer citation verification

The project has earned 12,891+ GitHub stars and includes an arXiv paper documenting its methodology.

> **Takeaway:** With just `researchclaw run --topic "Your idea" --auto-approve`, you get a complete research pipeline that handles literature review, experiment execution, paper writing, and citation verification -- all from a single command.

![AutoResearchClaw Architecture](/assets/img/diagrams/autoresearchclaw/autoresearchclaw-architecture.svg)

### Understanding the Architecture

The architecture diagram above illustrates the complete 23-stage pipeline organized across 8 phases. Let's break down each component:

**Phase A: Research Scoping (Stages 1-2)**
The pipeline begins by decomposing a research topic into a structured problem tree with specific research questions. The LLM analyzes the topic, identifies key dimensions, and creates a hierarchical decomposition that guides the entire pipeline.

**Phase B: Literature Discovery (Stages 3-6)**
This is where AutoResearchClaw distinguishes itself from tools that hallucinate references. It queries real academic databases -- OpenAlex, Semantic Scholar, and arXiv -- with query expansion and deduplication. Stage 5 acts as a gate, screening papers by relevance before knowledge extraction in Stage 6.

**Phase C: Knowledge Synthesis (Stages 7-8)**
Collected findings are clustered, research gaps identified, and hypotheses generated through structured multi-agent debate. This debate mechanism ensures hypotheses are challenged and refined before proceeding to experiment design.

**Phase D: Experiment Design (Stages 9-11)**
The pipeline designs experiment plans, generates hardware-aware Python code (adapting to GPU/MPS/CPU availability), and estimates resource needs. Stage 9 is a gate stage requiring human approval in Co-Pilot mode.

**Phase E: Experiment Execution (Stages 12-13)**
Experiments run in a sandboxed environment with AST-validated code, NaN/Inf fast-fail detection, and self-healing repair. If experiments fail, the system automatically diagnoses and repairs the code, retrying up to 10 rounds.

**Phase F: Analysis and Decision (Stages 14-15)**
Multi-agent analysis of results leads to an autonomous decision: PROCEED (continue to writing), REFINE (tweak parameters and re-run experiments), or PIVOT (explore a new research direction). This decision loop is one of AutoResearchClaw's most powerful features.

**Phase G: Paper Writing (Stages 16-19)**
Section-by-section drafting produces 5,000-6,500 word papers with anti-fabrication guards, methodology-evidence consistency checks, and revision length guards. Peer review uses multi-agent debate to evaluate paper quality.

**Phase H: Finalization (Stages 20-23)**
A quality gate, knowledge archival, LaTeX export with NeurIPS/ICML/ICLR templates, and 4-layer citation verification (arXiv ID check, CrossRef/DataCite DOI, Semantic Scholar title match, LLM relevance scoring) ensure the final paper meets conference standards.

**Cross-Cutting Components:**
- **Sentinel Watchdog** monitors quality across all stages, detecting NaN/Inf values, verifying paper-evidence consistency, and scoring citation relevance
- **MetaClaw Bridge** enables cross-run learning, converting failures and warnings into reusable skills injected into future runs
- **HITL Co-Pilot** provides 6 intervention modes for human-AI collaboration at critical decision points
- **LLM Providers** support OpenAI, Anthropic, Google, local models, and ACP-compatible agents

## Installation and Quick Start

```bash
# 1. Clone and install
git clone https://github.com/aiming-lab/AutoResearchClaw.git
cd AutoResearchClaw
python3 -m venv .venv && source .venv/bin/activate
pip install -e .

# 2. Setup (interactive -- installs OpenCode beast mode, checks Docker/LaTeX)
researchclaw setup

# 3. Configure
researchclaw init          # Interactive: choose LLM provider, creates config.arc.yaml

# 4. Run fully autonomous
export OPENAI_API_KEY="sk-..."
researchclaw run --config config.arc.yaml --topic "Your research idea" --auto-approve

# Or run in Co-Pilot mode for human-AI collaboration
researchclaw run --topic "Your research idea" --mode co-pilot
```

Output is saved to `artifacts/rc-YYYYMMDD-HHMMSS-<hash>/deliverables/` containing compile-ready LaTeX, BibTeX, experiment code, and charts.

The minimum configuration requires just a project name, research topic, and LLM provider settings:

```yaml
project:
  name: "my-research"

research:
  topic: "Your research topic here"

llm:
  base_url: "https://api.openai.com/v1"
  api_key_env: "OPENAI_API_KEY"
  primary_model: "gpt-4o"
  fallback_models: ["gpt-4o-mini"]

experiment:
  mode: "sandbox"
  sandbox:
    python_path: ".venv/bin/python"
```

![AutoResearchClaw Features](/assets/img/diagrams/autoresearchclaw/autoresearchclaw-features.svg)

### Understanding the Key Features

The features diagram above shows the six major capability areas of AutoResearchClaw and their specific components:

**Multi-Source Literature**
The literature system queries three real academic databases -- OpenAlex, Semantic Scholar, and arXiv -- with query expansion and deduplication. A circuit breaker pattern provides graceful degradation if any source is unavailable. The 4-layer citation verification system (arXiv ID check, CrossRef/DataCite DOI, Semantic Scholar title match, LLM relevance scoring) ensures no hallucinated references make it into the final paper.

**Human-in-the-Loop Co-Pilot**
Six intervention modes range from fully autonomous (`--auto-approve`) to step-by-step (`--mode step-by-step`). The Co-Pilot mode (`--mode co-pilot`) enables deep collaboration at three critical stages: the Idea Workshop (Stages 7-8) for hypothesis co-creation, the Baseline Navigator (Stage 9) for experiment design review, and the Paper Co-Writer (Stages 16-17) for collaborative drafting. SmartPause automatically detects when human input would be valuable and pauses the pipeline.

**Sandbox Experiments**
The experiment execution system auto-detects hardware (NVIDIA CUDA, Apple MPS, or CPU-only) and adapts code generation accordingly. AST validation ensures generated code is syntactically correct before execution. When experiments fail, the self-healing system diagnoses the error and generates targeted LLM repairs, retrying up to 10 rounds. Domain-specific agents handle high-energy physics (ColliderAgent), biology (COBRApy), and statistics experiments.

**Conference-Grade Writing**
The paper writing system produces 5,000-6,500 word drafts using NeurIPS, ICLR, and ICML LaTeX templates. Anti-fabrication guards (VerifiedRegistry) enforce ground-truth experiment data in papers. Revision length guards prevent the LLM from over-expanding or under-delivering sections. The anti-disclaimer enforcement removes AI-generated caveats.

**Self-Learning MetaClaw**
MetaClaw adds cross-run knowledge transfer. When enabled, the pipeline captures lessons from failures and warnings, converts them into reusable skills, and injects those skills into all 23 pipeline stages on subsequent runs. Controlled experiments show an 18.3% improvement in robustness score and 24.8% reduction in stage retry rate.

**Multi-Platform Integration**
AutoResearchClaw works standalone via CLI, through OpenClaw for messaging platform integration (Discord, Telegram, Lark, WeChat), or with any ACP-compatible agent including Claude Code, Codex CLI, Copilot CLI, Gemini CLI, and Kimi CLI. No API keys are needed when using ACP mode -- the agent handles its own authentication.

## The 23-Stage Pipeline in Detail

![AutoResearchClaw Workflow](/assets/img/diagrams/autoresearchclaw/autoresearchclaw-workflow.svg)

### Understanding the Pipeline Workflow

The workflow diagram above shows how a research topic flows through the 8 phases of the pipeline, with three key feedback loops:

**The PIVOT Loop (Stage 15 to Stage 8)**
When the analysis phase determines that the current hypothesis is fundamentally flawed, the pipeline pivots to an entirely new research direction. This is not a simple parameter tweak -- it generates new hypotheses through multi-agent debate and redesigns the experiment from scratch. The PIVOT loop is what makes AutoResearchClaw genuinely self-correcting.

**The REFINE Loop (Stage 15 to Stage 13)**
When results are promising but need improvement, the pipeline refines experiment parameters and re-runs. This iterative refinement loop can adjust hyperparameters, modify experimental conditions, or improve code quality. Up to 10 refinement rounds are supported.

**The Sentinel Watchdog**
Running alongside the entire pipeline, the Sentinel Watchdog monitors quality at key stages (experiment execution, research decision, peer review, and citation verification). It detects NaN/Inf values in experiment results, verifies paper-evidence consistency, and scores citation relevance. When issues are found, it can trigger automatic repair or pause for human review.

**Gate Stages**
Three gate stages (5, 9, and 20) serve as quality checkpoints. In autonomous mode, they auto-approve. In Co-Pilot mode, they pause for human review. On rejection, the pipeline rolls back to the previous stage and tries an alternative approach.

> **Amazing:** AutoResearchClaw's PIVOT/REFINE decision loop at Stage 15 is what separates it from simple paper generators. When experiments don't support the hypothesis, it doesn't just write around the problem -- it pivots to a new research direction or refines the experiment, just like a real researcher would.

## Human-in-the-Loop Co-Pilot System

The HITL Co-Pilot system introduced in v0.4.0 transforms the pipeline from purely autonomous to a human-AI collaborative research engine. Six intervention modes provide different levels of human involvement:

| Mode | Command | What It Does |
|------|---------|-------------|
| **Full Auto** | `--auto-approve` | Original behavior -- no human intervention |
| **Gate Only** | `--mode gate-only` | Pause at 3 gate stages (5, 9, 20) for approval |
| **Checkpoint** | `--mode checkpoint` | Pause at each phase boundary (8 checkpoints) |
| **Co-Pilot** | `--mode co-pilot` | Deep collaboration at critical stages, auto elsewhere |
| **Step-by-Step** | `--mode step-by-step` | Pause after every stage -- learn the pipeline |
| **Express** | `--mode express` | Quick review -- only 3 most critical gates |
| **Custom** | `--mode custom` | Define per-stage policies via `stage_policies` config |

The Co-Pilot mode provides three deep collaboration interfaces:

**Idea Workshop (Stages 7-8)**: Brainstorm, evaluate, and refine hypotheses collaboratively. The AI presents multiple hypotheses with novelty scores, and you can inject guidance, edit hypotheses, or start collaborative chats.

**Baseline Navigator (Stage 9)**: Review and modify the experiment design. The AI suggests baselines, and you can add, remove, or modify them. A reproducibility checklist ensures all experiments are properly configured.

**Paper Co-Writer (Stages 16-17)**: Section-by-section collaborative drafting. You can edit AI-generated sections, inject specific writing guidance, and the AI polishes your contributions while maintaining academic tone.

> **Important:** SmartPause is a confidence-driven dynamic intervention system that automatically detects when human input would be valuable -- even in autonomous mode. It monitors confidence scores at each stage and pauses when scores drop below thresholds, preventing the pipeline from continuing with low-quality outputs.

## MetaClaw: Self-Learning Across Runs

MetaClaw adds cross-run knowledge transfer to AutoResearchClaw. When enabled, the pipeline automatically captures lessons from failures and warnings, converts them into reusable skills, and injects those skills into all 23 pipeline stages on subsequent runs.

The learning cycle works as follows:

1. **Run N executes** -- failures and warnings are captured as structured lessons
2. **Lesson-to-Skill conversion** -- each lesson is converted into a SKILL.md file with YAML frontmatter
3. **Skills are stored** -- in `~/.metaclaw/skills/arc-*/SKILL.md`
4. **Run N+1 injects skills** -- `build_overlay()` injects learned skills into every LLM prompt
5. **LLM avoids known pitfalls** -- resulting in higher quality output and fewer retries

Controlled A/B experiments demonstrate measurable improvements:

| Metric | Baseline | With MetaClaw | Improvement |
|--------|----------|---------------|-------------|
| Stage retry rate | 10.5% | 7.9% | -24.8% |
| Refine cycle count | 2.0 | 1.2 | -40.0% |
| Pipeline stage completion | 18/19 | 19/19 | +5.3% |
| Overall robustness score | 0.714 | 0.845 | +18.3% |

## Domain-Specific Experiment Agents

v0.5.0 introduced domain-specialist execution agents that route beyond the default ML sandbox to specialist agents per field:

- **High-Energy Physics**: ColliderAgent uses Lagrangian definitions, FeynRules, MadGraph5, and Delphes via the Magnus cloud for particle physics simulations
- **Biology**: COBRApy agent handles genome-scale metabolic modelling for systems biology
- **Statistics**: Simulation-study agent manages statistical simulation experiments
- **Chemistry/Materials**: Generic Docker executor covers computational chemistry and materials science
- **ML**: Default sandbox handles standard machine learning experiments with GPU/MPS/CPU auto-detection

The pipeline auto-selects the right executor from the research domain, requiring no manual configuration.

## Skills Library

AutoResearchClaw ships with 20 pre-loaded built-in skills covering scientific writing, experiment design, chemistry, biology, and more. You can also load custom skills:

```bash
# Option 1: Install a skill (persists across projects)
researchclaw skills install /path/to/my-skill/

# Option 2: Drop a SKILL.md into the project
mkdir -p .claude/skills/my-custom-skill

# Option 3: Configure shared skill directories in config.arc.yaml
```

Skills are loaded and injected into LLM prompts automatically -- no manual activation needed. Browse community skills at [K-Dense-AI/claude-scientific-skills](https://github.com/K-Dense-AI/claude-scientific-skills) (150+ scientific skills across multiple disciplines).

## Multi-Platform Integration

AutoResearchClaw is not locked to a single platform. It supports multiple ways to run:

| Method | How |
|--------|-----|
| **Standalone CLI** | `researchclaw run --topic "..." --auto-approve` (autonomous) or `--mode co-pilot` (collaborative) |
| **Python API** | `from researchclaw.pipeline import Runner; Runner(config).run()` |
| **Claude Code** | Reads `RESEARCHCLAW_CLAUDE.md` -- just say "Run research on [topic]" |
| **Copilot CLI** | `researchclaw run --topic "..."` with `llm.acp.agent: "gh"` |
| **OpenCode** | Reads `.claude/skills/` -- same natural language interface |
| **Any AI CLI** | Provide `RESEARCHCLAW_AGENTS.md` as context |
| **OpenClaw** | Share the GitHub repo URL, say "Research [topic]" |

The ACP (Agent Client Protocol) integration means you can use any ACP-compatible coding agent as the LLM backend without API keys -- the agent handles its own authentication.

## Key Technical Details

| Detail | Value |
|--------|-------|
| **Language** | Python 3.11+ |
| **License** | MIT |
| **Version** | v0.5.0 |
| **Tests** | 2,699 passed |
| **Pipeline Stages** | 23 across 8 phases |
| **Gate Stages** | 3 (Stages 5, 9, 20) |
| **HITL Modes** | 6 (full-auto, gate-only, checkpoint, co-pilot, step-by-step, custom) |
| **Literature Sources** | OpenAlex, Semantic Scholar, arXiv |
| **Citation Verification** | 4-layer (arXiv, CrossRef/DataCite, Semantic Scholar, LLM) |
| **LaTeX Templates** | NeurIPS 2025, ICLR 2026, ICML 2026 |
| **Domain Agents** | HEP (ColliderAgent), Biology (COBRApy), Statistics, Chemistry, ML |
| **LLM Providers** | OpenAI, Anthropic, Google, Local, ACP-compatible agents |
| **Package Name** | `researchclaw` (pip install -e .) |
| **Entry Point** | `researchclaw` CLI command |

## Troubleshooting

| Issue | Solution |
|-------|----------|
| **LLM API rate limits** | Configure `fallback_models` in config; the pipeline automatically retries with exponential backoff |
| **Experiment failures** | The self-healing system automatically repairs code up to 10 rounds; check `experiment_diagnosis/` for details |
| **Docker not available** | Set `experiment.mode: "local"` in config; sandbox mode requires Docker |
| **LaTeX compilation errors** | Ensure `texlive-latex-recommended` is installed; the pipeline includes NeurIPS/ICML/ICLR templates |
| **Cost exceeding budget** | Set `hitl.cost_budget_usd` in config; the pipeline pauses at 50%/80%/100% thresholds |
| **Citation verification failures** | Check `verification_report.json` for which layer failed; the pipeline auto-removes unverified citations |

## Conclusion

AutoResearchClaw represents a significant leap forward in AI-assisted academic research. Its 23-stage pipeline with multi-agent debate, self-healing experiments, PIVOT/REFINE decision loops, and 4-layer citation verification produces conference-ready papers from a single research idea. The Co-Pilot mode enables genuine human-AI collaboration, while MetaClaw ensures the system learns from every run.

Whether you are a researcher looking to accelerate your workflow, a student learning the research process, or a team managing multiple research projects, AutoResearchClaw provides the tools to go from idea to paper with unprecedented automation and quality assurance.

**Links:**
- GitHub: [https://github.com/aiming-lab/AutoResearchClaw](https://github.com/aiming-lab/AutoResearchClaw)
- arXiv Paper: [https://arxiv.org/abs/2605.20025](https://arxiv.org/abs/2605.20025)
- ARC-Bench Dataset: [https://huggingface.co/datasets/AIMING-Lab-UNC/ARC-Bench](https://huggingface.co/datasets/AIMING-Lab-UNC/ARC-Bench)
- Discord Community: [https://discord.gg/u4ksqW5P](https://discord.gg/u4ksqW5P)