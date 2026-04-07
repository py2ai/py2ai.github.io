---
layout: post
title: "AgentSkillOS: An Operating System for Agent Skills"
description: "Learn how AgentSkillOS helps discover, compose, and orchestrate 200,000+ AI agent skills into working pipelines with DAG-based workflows."
date: 2026-04-08
header-img: "img/post-bg.jpg"
permalink: /AgentSkillOS-Skill-Orchestration-System/
featured-img: ai-coding-frameworks/ai-coding-frameworks
tags:
  - Open Source
  - AI Agents
  - Python
  - Tutorial
author: "PyShine"
---

# AgentSkillOS: An Operating System for Agent Skills

The agent skill ecosystem is exploding - over 200,000+ skills are now publicly available. But with so many options, how do you find the right skills for your task? And when one skill isn't enough, how do you compose and orchestrate multiple skills into a working pipeline?

**AgentSkillOS** is the operating system for agent skills - helping you discover, compose, and run skill pipelines end-to-end.

![AgentSkillOS Architecture](/assets/img/diagrams/agentskillos-architecture.svg)

## Introduction

AgentSkillOS addresses a fundamental challenge in the AI agent ecosystem: skill discovery and orchestration at scale. With hundreds of thousands of skills available across platforms like GitHub, npm, and PyPI, finding the right combination of tools for complex tasks has become increasingly difficult.

Traditional approaches rely on semantic search, which often misses skills that look unrelated in embedding space but are crucial for solving tasks. AgentSkillOS introduces a novel capability tree structure that organizes skills hierarchically, enabling more creative and effective skill discovery.

### Why AgentSkillOS Matters

- **Scale**: Manages 200,000+ skills efficiently
- **Discovery**: Finds non-obvious but functionally relevant skills
- **Orchestration**: Composes multiple skills into coordinated workflows
- **Control**: Provides human-in-the-loop GUI for intervention

## Key Features

| Feature | Description |
|---------|-------------|
| Skill Search & Discovery | Creatively discover task-relevant skills with a skill tree that organizes skills into a hierarchy based on their capabilities |
| Skill Orchestration | Compose and orchestrate multiple skills into a single workflow with a directed acyclic graph, automatically managing execution order, dependencies, and data flow |
| GUI (Human-in-the-Loop) | A built-in GUI enables human intervention at every step, making workflows controllable, auditable, and easy to steer |
| High-Quality Skill Pool | A curated collection of high-quality skills, selected based on Claude's implementation, GitHub stars, and download volume |
| Observability & Debugging | Trace each step with logs and metadata to debug faster and iterate on workflows with confidence |
| Extensible Skill Registry | Easily plug in new skills, bring your own skills via a flexible registry |
| Benchmark | 30 multi-format creative tasks across 5 categories, evaluated with pairwise comparison and Bradley-Terry aggregation |

## Architecture Overview

AgentSkillOS follows a modular architecture with pluggable retrieval and orchestration components.

![Skill Retrieval](/assets/img/diagrams/agentskillos-skill-retrieval.svg)

### Core Components

**Entry Points**: The system provides multiple interfaces including Web UI, Batch CLI, and Python API for different use cases.

**Manager Layer**: Handles skill discovery through two approaches:
- **Tree-based**: Uses capability trees for hierarchical skill organization
- **Vector-based**: Uses semantic embeddings for similarity search

**Orchestrator Layer**: Manages skill execution:
- **DAG Engine**: Plans and executes directed acyclic graphs
- **Direct Engine**: Simple sequential execution
- **Freestyle Engine**: Claude Code-style execution

**Runtime**: Executes skills using Claude Code or other LLM providers.

## How Skill Retrieval Works

### Why Skill Tree?

Pure semantic retrieval prioritizes textual similarity, often missing skills that look unrelated in embedding space but are crucial for actually solving the task. This leads to narrow, myopic skill usage.

AgentSkillOS uses LLM + Skill Tree to navigate the capability hierarchy, surfacing non-obvious but functionally relevant skills. This enables broader, more creative, and more effective skill composition.

### Tree-based vs Vector-based Search

| Approach | Strengths | Best For |
|----------|-----------|----------|
| Tree-based | Hierarchical organization, creative discovery | Complex tasks requiring diverse skills |
| Vector-based | Semantic similarity, fast lookup | Known skill patterns, straightforward tasks |

### Pre-built Skill Trees

| Tree | Skills | Description |
|------|--------|-------------|
| `skill_seeds` | ~50 | Curated skill set (default) |
| `skill_200` | 200 | 200 skills |
| `skill_1000` | ~1,000 | 1,000 skills |
| `skill_10000` | ~10,000 | 10,000 active + layered dormant skills |

## DAG Orchestration

![DAG Orchestration](/assets/img/diagrams/agentskillos-dag-orchestration.svg)

### Plan Generation

The DAG orchestrator analyzes task requirements and generates execution plans:

1. **Task Analysis**: Understands user requirements
2. **Skill Selection**: Chooses relevant skills from the pool
3. **Dependency Resolution**: Determines execution order
4. **Plan Generation**: Creates the DAG structure

### Parallel Execution

The orchestrator supports three strategies:

| Strategy | Description | Use Case |
|----------|-------------|----------|
| Quality-First | Deep, multi-stage pipelines | High-quality outputs |
| Efficiency-First | Wide, parallel execution | Speed optimization |
| Simplicity-First | Essential steps only | Simple tasks |

### Execution Flow

```
User Request -> Skill Discovery -> Plan Generation -> DAG Execution -> Output
                    |                    |                  |
                    v                    v                  v
              Skill Tree          Dependency Graph    Parallel Tasks
```

## Installation

### Prerequisites

- Python 3.10+
- [Claude Code](https://github.com/anthropics/claude-code) (must be installed and available in PATH)
- Use [cc-switch](https://github.com/farion1231/cc-switch) to switch to other LLM providers

### Install and Run

```bash
# Clone the repository
git clone https://github.com/ynulihao/AgentSkillOS.git
cd AgentSkillOS

# Install in development mode
pip install -e .

# Configure environment
cp .env.example .env
# Edit .env with your API keys

# Start the web interface
python run.py --port 8765
```

### Configuration

```bash
# .env configuration
LLM_MODEL=openai/anthropic/claude-opus-4.5
LLM_BASE_URL=https://openrouter.ai/api/v1
LLM_API_KEY=your-key

EMBEDDING_MODEL=openai/text-embedding-3-large
EMBEDDING_BASE_URL=https://api.openai.com/v1
EMBEDDING_API_KEY=your-key
```

## Usage Examples

### Web UI

The Web UI provides a visual workflow overview in the browser:

1. Navigate to `http://localhost:8765`
2. Enter your task description
3. Select skill discovery mode (tree or vector)
4. Choose orchestration strategy
5. Review and approve the generated plan
6. Monitor execution in real-time

### Batch CLI

Run multiple tasks in parallel without the Web UI:

```bash
# Run a batch configuration
python run.py cli --task config/batch.yaml

# Override parallel task count
python run.py cli -T config/batch.yaml --parallel 4

# Resume interrupted runs
python run.py cli -T config/batch.yaml --resume ./runs/my_batch_20260306_120000

# Dry run to preview tasks
python run.py cli -T config/batch.yaml --dry-run
```

### Batch Configuration (YAML)

```yaml
batch_id: my_batch

defaults:
  skill_mode: auto          # "auto" (discover) or "specified"
  skill_group: skill_200    # Which skill pool to use
  output_dir: ./runs
  continue_on_error: true

execution:
  parallel: 2               # Max concurrent tasks
  retry_failed: 0

tasks:
  - file: path/to/task1.json
  - file: path/to/task2.json
  - dir: path/to/tasks/     # Scan directory
    pattern: "*.json"
```

### CLI Flags

| Flag | Description |
|------|-------------|
| `--task PATH`, `-T` | Path to batch YAML config (required) |
| `--parallel N`, `-p` | Override parallel task count |
| `--resume PATH`, `-R` | Resume an interrupted batch run |
| `--output-dir PATH`, `-o` | Override output directory |
| `--dry-run` | Preview tasks without execution |
| `--verbose`, `-v` | Show detailed logs |
| `--manager PLUGIN`, `-m` | Override skill manager (e.g., `tree`, `vector`) |
| `--orchestrator PLUGIN` | Override orchestrator (e.g., `dag`, `free-style`) |

### Custom Skill Groups

Create your own skill collections:

1. Create `data/my_skills/skill-name/SKILL.md`
2. Register in `src/config.py` -> `SKILL_GROUPS`
3. Build: `python run.py build -g my_skills -v`

## Benchmark Results

AgentSkillOS includes a benchmark of **30 multi-format creative tasks** spanning **5 categories**, evaluated via pairwise comparison with Bradley-Terry aggregation.

### Key Findings

- **Substantial Gains over Baselines**: All three AgentSkillOS variants achieve the highest Bradley-Terry scores across 200 / 1K / 200K ecosystems
- **Both Retrieval and Orchestration Are Essential**: Removing components reveals clear degradation
- **Strategy Choice Shapes Execution Structure**: Each orchestration strategy faithfully translates its design intent into a distinct DAG topology

### Example Use Cases

| Example | Description |
|---------|-------------|
| Bug Diagnosis Report | Mobile bug localization, fix validation, and visual bug report generation with before/after evidence |
| UI Design Research | Design-language research, report generation, and multi-direction concept mockups for knowledge software |
| Paper Promotion | Transforms academic papers into social slides, scientific pages, and platform-specific promotion content |
| Meme Video | Green-screen compositing, subtitle timing, and viral short-video production with multi-version outputs |

## Academic Reference

AgentSkillOS is backed by academic research. If you find it useful, consider citing the paper:

```bibtex
@article{li2026organizing,
  title={Organizing, Orchestrating, and Benchmarking Agent Skills at Ecosystem Scale},
  author={Li, Hao and Mu, Chunjiang and Chen, Jianhao and Ren, Siyue and Cui, Zhiyao and Zhang, Yiqun and Bai, Lei and Hu, Shuyue},
  journal={arXiv preprint arXiv:2603.02176},
  year={2026}
}
```

**Paper Link**: [arXiv:2603.02176](https://arxiv.org/abs/2603.02176)

**Dataset**: [Hugging Face - agentskillos-benchmark](https://huggingface.co/datasets/NPULH/agentskillos-benchmark)

## Future Roadmap

AgentSkillOS is actively developed with planned features:

- [x] Recipe Generation & Storage
- [ ] Interactive Agent Execution
- [ ] Plan Refinement
- [ ] Auto Skill Import
- [ ] Dependency Detection
- [ ] History Management
- [ ] Multi-CLI Support (Codex, Gemini CLI, Cursor)

## Conclusion

AgentSkillOS represents a significant advancement in managing AI agent skills at ecosystem scale. By combining hierarchical skill trees with DAG-based orchestration, it enables developers to:

1. **Discover** relevant skills from 200,000+ options
2. **Compose** multiple skills into working pipelines
3. **Execute** complex tasks with human oversight
4. **Debug** and iterate with full observability

The modular architecture allows plugging in different retrieval methods and orchestration strategies, making it adaptable to various use cases from simple automation to complex multi-stage workflows.

### Links

- **GitHub**: [https://github.com/ynulihao/AgentSkillOS](https://github.com/ynulihao/AgentSkillOS)
- **Documentation**: [https://ynulihao.github.io/AgentSkillOS/](https://ynulihao.github.io/AgentSkillOS/)
- **Paper**: [arXiv:2603.02176](https://arxiv.org/abs/2603.02176)
- **Dataset**: [Hugging Face](https://huggingface.co/datasets/NPULH/agentskillos-benchmark)

## Related Posts

- [DESIGN.md: AI-Powered Design Systems](/DESIGN-md-AI-Powered-Design-Systems/)
- [AI Coding Frameworks Comparison](/ai-coding-frameworks/)