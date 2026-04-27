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

### Understanding the Architecture Diagram

The architecture diagram above illustrates the complete AgentSkillOS system, designed as a layered operating system for managing AI agent skills at scale. Let's break down each component and understand how they work together.

#### Entry Points Layer

At the top of the architecture, we find three distinct entry points that make AgentSkillOS accessible to different user types and use cases:

- **Web UI**: A browser-based graphical interface that provides visual workflow management, real-time execution monitoring, and human-in-the-loop intervention capabilities. This is ideal for developers who want visual feedback and control over skill orchestration.

- **Batch CLI**: A command-line interface designed for automated, headless execution of multiple tasks. Perfect for CI/CD pipelines, scheduled jobs, or bulk processing scenarios where manual intervention isn't needed.

- **Python API**: A programmatic interface for developers who want to integrate AgentSkillOS directly into their applications. This enables custom workflows, embedded skill orchestration, and programmatic control over all system features.

#### Manager Layer: The Brain of Skill Discovery

The Manager Layer is responsible for discovering and selecting relevant skills from the vast skill pool. It implements two complementary approaches:

- **Tree-based Manager**: This innovative approach organizes skills into a hierarchical capability tree. Instead of relying solely on semantic similarity, it navigates through skill categories and subcategories, enabling discovery of non-obvious but functionally relevant skills. For example, when searching for "image processing," it might discover skills in "data visualization" or "document generation" that could enhance the workflow.

- **Vector-based Manager**: A traditional semantic search approach using embedding models. Skills are converted to vector representations, and similarity search finds the closest matches. This is fast and effective for known skill patterns but may miss creative combinations.

#### Orchestrator Layer: Coordinating Complex Workflows

The Orchestrator Layer takes selected skills and coordinates their execution. It offers three distinct execution strategies:

- **DAG Engine**: The most sophisticated orchestrator that builds directed acyclic graphs to manage complex dependencies. It automatically determines execution order, handles parallel execution where possible, and manages data flow between skills.

- **Direct Engine**: A simpler approach for straightforward tasks where skills execute sequentially without complex dependency management.

- **Freestyle Engine**: Inspired by Claude Code's execution model, this engine provides more flexible, conversational-style skill invocation for dynamic scenarios.

#### Runtime Layer: Where Skills Execute

At the bottom of the architecture sits the Runtime Layer, which handles the actual execution of skills. It supports multiple LLM backends including Claude Code and other providers through the cc-switch utility. This abstraction allows developers to use their preferred AI models while maintaining consistent skill execution interfaces.

#### Data Flow Through the System

When a user submits a task through any entry point, the request flows downward through the layers. The Manager Layer discovers relevant skills, the Orchestrator Layer plans and coordinates execution, and the Runtime Layer executes each skill. Results flow back upward, with logging and state management at each level ensuring observability and debugging capabilities.

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

### Understanding the Skill Retrieval Flow

The skill retrieval diagram above demonstrates how AgentSkillOS discovers relevant skills from its vast pool of 200,000+ available skills. This process is critical because finding the right skills determines the quality and efficiency of the entire workflow execution.

#### The Challenge of Skill Discovery at Scale

With over 200,000 skills available across platforms like GitHub, npm, and PyPI, traditional keyword search falls short. A simple query like "process images" might return hundreds of results, many of which are tangentially related but not truly useful for the specific task at hand. AgentSkillOS addresses this through two complementary retrieval mechanisms.

#### Tree-Based Retrieval: Navigating the Capability Hierarchy

The tree-based approach organizes skills into a hierarchical capability structure. Imagine a tree where:

- **Root nodes** represent broad capability categories like "Data Processing," "Content Generation," or "System Operations"
- **Branch nodes** represent sub-categories like "Image Manipulation" under "Data Processing"
- **Leaf nodes** contain the actual skills with their descriptions and metadata

When a user submits a task, the LLM doesn't just search for keywords—it navigates this tree intelligently. For example, a task to "create a marketing video from product images" might traverse:

1. **Content Generation** → **Video Production** → skills for video editing
2. **Data Processing** → **Image Manipulation** → skills for image preparation
3. **Content Generation** → **Marketing** → skills for promotional content

This traversal surfaces skills that semantic search might miss—skills that are functionally relevant even if their descriptions don't match the query textually.

#### Vector-Based Retrieval: Semantic Similarity Search

The vector-based approach converts skill descriptions into dense vector embeddings using models like OpenAI's text-embedding-3-large. When a query comes in:

1. The query text is converted to a vector representation
2. Similarity search (typically cosine similarity) finds the closest skill vectors
3. Top-k results are returned as candidate skills

This approach excels at finding skills with similar meanings even when using different terminology. However, it can miss skills that are functionally complementary but semantically distant.

#### Hybrid Retrieval: Best of Both Worlds

AgentSkillOS can combine both approaches for optimal results. The tree-based method provides creative, non-obvious skill suggestions, while vector-based search ensures no relevant skills are missed due to vocabulary gaps. This hybrid approach significantly outperforms either method alone, as demonstrated in the benchmark results.

#### Practical Implications for Developers

For developers building AI agent workflows, this retrieval system means:

- **Less manual skill hunting**: The system automatically surfaces relevant skills
- **More creative solutions**: Non-obvious skill combinations lead to innovative workflows
- **Better task coverage**: Complex tasks get decomposed into appropriate skill sequences
- **Scalable architecture**: The system handles skill pools from 50 to 200,000+ without degradation

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

### Understanding DAG Orchestration

The DAG (Directed Acyclic Graph) orchestration diagram above illustrates how AgentSkillOS coordinates multiple skills into coherent, executable workflows. This is where the magic happens—transforming a list of relevant skills into a coordinated execution plan.

#### What is DAG Orchestration?

A Directed Acyclic Graph is a mathematical structure where:

- **Nodes** represent individual skills or operations
- **Edges** represent dependencies between skills
- **Directed** means edges have a direction (A → B means B depends on A)
- **Acyclic** means there are no circular dependencies (no infinite loops)

This structure is perfect for skill orchestration because it naturally captures the dependencies between tasks while enabling parallel execution where possible.

#### The Orchestration Pipeline

The orchestration process follows a sophisticated pipeline:

**1. Task Analysis Phase**

When a user submits a task like "Create a bug diagnosis report for a mobile app," the orchestrator first analyzes the requirements. It breaks down the high-level goal into sub-tasks:

- Bug reproduction and localization
- Error log analysis
- Fix suggestion generation
- Visual documentation creation
- Report compilation

**2. Skill Selection Phase**

From the retrieved skills, the orchestrator selects the most appropriate ones for each sub-task. This selection considers:

- Skill capabilities and specializations
- Input/output compatibility between skills
- Historical performance metrics
- Resource requirements

**3. Dependency Resolution Phase**

The orchestrator determines which skills must run before others. For example:

- Bug localization must complete before fix suggestion
- Visual documentation requires both bug evidence and fix results
- Report compilation depends on all previous outputs

**4. Plan Generation Phase**

The final DAG structure is generated, optimizing for:

- **Parallelism**: Independent skills run simultaneously
- **Resource efficiency**: Minimize redundant computations
- **Fault tolerance**: Isolate failures to prevent cascade effects

#### Execution Strategies: Quality vs. Speed vs. Simplicity

AgentSkillOS offers three distinct orchestration strategies:

**Quality-First Strategy**

This strategy builds deep, multi-stage pipelines with extensive validation and refinement steps. Each skill's output is verified before passing to the next stage. Ideal for:

- Production deployments requiring high accuracy
- Complex tasks with significant consequences for errors
- Scenarios where iteration and refinement add value

**Efficiency-First Strategy**

This strategy maximizes parallel execution, running as many skills simultaneously as possible. Dependencies are minimized to reduce wait times. Ideal for:

- Time-sensitive tasks
- Batch processing scenarios
- When approximate results are acceptable

**Simplicity-First Strategy**

This strategy uses only essential skills, avoiding complex pipelines. It's the "minimum viable workflow" approach. Ideal for:

- Simple, well-defined tasks
- Quick prototyping and testing
- When complexity overhead isn't justified

#### Real-World Example: Bug Diagnosis Workflow

Consider a bug diagnosis task. The DAG might look like:

```
[Mobile Bug Report]
       ↓
[Parse Stack Trace] ←→ [Extract Error Logs]
       ↓                      ↓
[Localize Bug] ←─────── [Analyze Patterns]
       ↓
[Generate Fix Suggestion]
       ↓
[Create Visual Evidence] ←→ [Validate Fix]
       ↓
[Compile Report]
```

This DAG shows parallel execution opportunities (Parse Stack Trace and Extract Error Logs can run simultaneously) while maintaining necessary dependencies (Localize Bug needs both inputs).

#### Human-in-the-Loop Control

A key feature of AgentSkillOS orchestration is human oversight. At each stage:

- Users can review the generated plan before execution
- Intermediate results can be inspected and approved
- Manual intervention can redirect or modify the workflow
- Execution can be paused, resumed, or terminated

This control is essential for production systems where AI autonomy must be balanced with human judgment.

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
