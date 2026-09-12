---
layout: post
title: "MathModelAgent: The AI That Turns a 3-Day Math Competition Into 1 Hour"
description: "MathModelAgent is an open source agent designed for mathematical modeling that automatically completes the full pipeline: problem analysis, model selection, code writing, error correction, and paper generation. What used to take 3 days in a math modeling competition now takes 1 hour. Four specialized agents (Coordinator, Modeler, Coder, Writer) collaborate in a pipeline driven by SKILLS that work with Claude Code and Codex. Supports all LLM models via litellm, with each agent configurable to use a different model. 17 Typst paper templates for major competitions (national, MCM/ICM, Huawei Cup, etc.). 9-step automated validation ensures zero low-level errors. Four-layer fault tolerance (retry, fallback hand-off, evaluator shadow mode, feedback rerun). RAG knowledge base (ChromaDB + Rerank) for modeling methods, code templates, and paper writing. Human-in-loop with 6 decision actions (confirm, edit, regenerate, ask, skip, abort). Code interpreter supports local Jupyter, E2B, and Daytona. Web search via Tavily API for real data. Sister project sci-box provides scientific figure templates (SHAP, ROC, Taylor, chord diagrams) and draw.io flow charts. Desktop apps for macOS and Windows. Online version at mathmodel.top."
date: 2026-09-12
header-img: "img/post-bg.jpg"
permalink: /MathModelAgent-AI-Math-Modeling-3-Days-to-1-Hour/
featured-img: ai-coding-frameworks/ai-coding-frameworks
tags:
  - MathModelAgent
  - Mathematical Modeling
  - Open Source
  - AI Agents
  - Claude Code
  - Multi-Agent
  - Typst
  - Skills
author: PyShine
---

If you have ever participated in a mathematical modeling competition, you know the drill: three days of intense work, barely any sleep, one person writing code, another building the model, a third formatting the paper, and the clock never stops. The Mathematical Contest in Modeling (MCM) and its cousins give teams 96 hours to go from a vague problem statement to a polished, submission-ready paper.

MathModelAgent compresses that into one hour.

## What is MathModelAgent

[MathModelAgent](https://github.com/jihe520/MathModelAgent) is an open source agent designed specifically for mathematical modeling. Give it a problem statement, and it automatically analyzes the problem, selects an appropriate mathematical model, writes and debugs the code, generates publication-quality figures, and produces a properly formatted paper ready for submission.

The vision is bold: **turn 3 days of competition into 1 hour, and produce an award-level paper.**

The project started as a multi-agent system with a custom backend, but it has since been distilled into a pure SKILLS-driven approach. Instead of building its own agent framework, it leverages existing harnesses like Claude Code and Codex, loading skills that teach the agent the full mathematical modeling pipeline. This is a philosophical shift: why build an agent framework when Claude Code already is one?

## The Four-Agent Pipeline

MathModelAgent breaks the modeling task into four specialized roles, each handled by a dedicated agent.

![MathModelAgent workflow](/assets/img/diagrams/mathmodelagent/mathmodelagent-workflow.svg)

### Coordinator Agent

The Coordinator is the entry point. It validates and parses the problem statement, determines the competition type (national, MCM/ICM, Huawei Cup, etc.), selects the appropriate Typst paper template from 17 built-in options, and routes the task to the modeling stage.

### Modeler Agent

The Modeler analyzes the problem background and selects the right mathematical model. It has access to a decision tree covering AHP, TOPSIS, ARIMA, genetic algorithms, and many more. It references the RAG knowledge base (ChromaDB + Rerank) for modeling methods, code templates, and paper writing references, which reduces model hallucination.

### Coder Agent

The Coder translates the model plan into runnable code. It writes Python (and will support R and MATLAB), executes it through a code interpreter, and debugs errors iteratively. Code interpreters include:
- **Local**: Jupyter, saving notebooks (.ipynb) for easy re-editing
- **Cloud**: E2B and Daytona for sandboxed execution

The Coder also uses Tavily API for web search, fetching real-world data instead of fabricating numbers.

### Writer Agent

The Writer organizes the modeling process and results into a formatted paper using Typst templates. It generates `res.md` (markdown) and `res.docx` (Word), with figures, tables, and references. The 17 Typst templates cover major Chinese and international competitions, and the output is submission-ready.

## System Architecture

MathModelAgent has evolved from a traditional three-service architecture (FastAPI + Vue 3 + Redis) into a SKILLS-first design that rides on top of existing AI coding harnesses.

![MathModelAgent architecture](/assets/img/diagrams/mathmodelagent/mathmodelagent-architecture.svg)

### The SKILLS Revolution

The key insight: MathModelAgent no longer builds its own harness layer. Instead, it distills everything into SKILLS that are loaded by existing agent harnesses:

```bash
# Install the skills
npx skills add jihe520/MathModelAgent --all

# Run with Claude Code
claude --dangerously-skip-permissions
# Then: /1start-mathmodel complete this math modeling task

# Run with Codex
codex --yolo
# Then: $start-mathmodel complete this math modeling task
```

Each stage is an independent skill. You can call just the analysis, just the paper writing, or the full pipeline. Templates and the knowledge base are extensible. This is the composable agent economy in action.

### Traditional Backend (Alternative Path)

For those who prefer a traditional deployment, the FastAPI backend remains available:
- **Backend**: FastAPI on port 8000, workflow orchestration engine, agent system, LLM integration via litellm
- **Frontend**: Vue 3 + Vite on port 5173, chat interface, task management, agent editor, API configuration
- **Redis**: Pub/sub message queue, task state management, WebSocket message relay

Docker Compose starts all three services with one command: `docker-compose up -d`.

### Multi-LLM Support

Every agent can use a different LLM. Through litellm, MathModelAgent supports all providers: DeepSeek, OpenAI, Claude, Gemini, Qwen, and local models. This means you can use a strong reasoning model for the Modeler, a fast coding model for the Coder, and a quality writing model for the Writer.

### RAG Knowledge Base

The RAG system (ChromaDB + Rerank) retrieves modeling methods, code templates, and paper writing references. It includes a complete model selection decision tree, common error patterns, and MCM/ICM scoring criteria. Every stage automatically references this knowledge, which reduces hallucination and improves quality.

## 9-Step Paper Validation

Generating a paper is one thing. Generating one with zero errors is another. MathModelAgent runs a 9-step validation pipeline on every paper.

![MathModelAgent validation pipeline](/assets/img/diagrams/mathmodelagent/mathmodelagent-validation.svg)

| Step | What it checks |
|------|----------------|
| 1. Text leak detection | Scans for leaked prompts or internal instructions in output |
| 2. Numerical consistency | Cross-checks all numbers between code output and paper text |
| 3. Model validity | Verifies the selected model matches the problem type |
| 4. Code execution check | Re-runs code in interpreter, verifies no runtime errors |
| 5. Figure validation | Checks all figures are present, properly labeled, and scaled |
| 6. Reference check | Verifies citations exist and formatting is correct |
| 7. Typst compilation | Compiles Typst source to PDF, checks for errors |
| 8. PDF visual inspection | Visual check of layout, spacing, page breaks |
| 9. Final acceptance | All checks passed, paper is zero-error and ready |

This is why the paper quality approaches award-level. It is not just generation. It is generation plus rigorous validation.

## Four-Layer Fault Tolerance + Human-in-Loop

Things go wrong. Models hallucinate. Code fails. Numbers do not add up. MathModelAgent has a four-layer defense, plus a human-in-loop system for when you want control.

![MathModelAgent fault tolerance](/assets/img/diagrams/mathmodelagent/mathmodelagent-fault-tolerance.svg)

### Layer 1: Limited Retry

When an agent encounters an error, it automatically retries with a configurable maximum number of attempts. Same model, same prompt. Fast recovery for transient errors.

### Layer 2: Fallback Hand-Off

If retries are exhausted, the task hands off to a different (usually smarter) model. This is A2A (Agent-to-Agent) hand-off: automatic model selection and seamless degradation. The Coder might fail on a complex problem with a small model, then succeed when handed to a larger one.

### Layer 3: Evaluator Shadow Mode

A shadow evaluator runs in parallel, scoring the output for quality. It checks for hallucination, logic errors, and completeness. Issues are flagged for revision.

### Layer 4: Feedback Rerun

Evaluator feedback is injected back into the pipeline. The Writer agent rewrites, the Coder re-implements. This is a targeted fix, not a full re-run. The paper improves with each iteration.

### Human-in-Loop (HIL)

At key decision points, the pipeline pauses and waits for your approval. You have six actions:

| Action | What it does |
|--------|-------------|
| `confirm` | Approve and continue |
| `edit` | Modify the output and continue |
| `regenerate` | Redo with feedback |
| `ask` | Ask the agent a question |
| `skip` | Skip this step |
| `abort` | Stop the task |

This gives you full control when you want it, and full autonomy when you do not.

## sci-box: Scientific Figures and Flow Charts

MathModelAgent has a sister project: [sci-box](https://github.com/jihe520/sci-box), a skills package for scientific figures and flow charts.

```bash
npx skills add jihe520/sci-box
```

| Skill | What it provides |
|-------|-----------------|
| `scibox-figure` | SHAP, ROC, Taylor, chord, ring heatmap, rain cloud templates (Python + Matplotlib, exports PNG/PDF/SVG) |
| `scibox-diagram` | Editable draw.io templates: 5-layer technical roadmap, 3-column research framework, 3-column flow chart, horizontal task pipeline |

These are the kind of publication-quality figures that make a paper stand out. They are independent of MathModelAgent and can be used in any project.

## Quick Start

### Option 1: Desktop App (Simplest)

Download the desktop app from the [Releases page](https://github.com/jihe520/MathModelAgent/releases/latest):

| Platform | File |
|----------|------|
| macOS (Apple Silicon) | `mathmodel-<version>-arm64.dmg` |
| macOS (Intel) | `mathmodel-<version>-x64.dmg` |
| Windows 64-bit | `mathmodel-<version>-x64.exe` |

macOS builds are Developer ID signed and notarized. Windows builds are not yet signed (SmartScreen warning is expected on first run).

### Option 2: SKILLS with Claude Code or Codex

```bash
# Install skills
npx skills add jihe520/MathModelAgent --all

# Run with Claude Code
claude --dangerously-skip-permissions
# In Claude Code: /1start-mathmodel complete this math modeling task

# Run with Codex
codex --yolo
# In Codex: $start-mathmodel complete this math modeling task
```

Other commands:
- `/doctor`: Check environment configuration
- `/typst-author`: Typst knowledge

### Option 3: Docker Deployment

```bash
git clone https://github.com/jihe520/MathModelAgent.git
cd MathModelAgent
cp backend/.env.dev.example backend/.env.dev
cp frontend/.env.example frontend/.env.development
# Fill in API keys and model configuration
docker-compose up -d
```

Access:
- Frontend: `http://localhost:5173`
- Backend API: `http://localhost:8000`

### Option 4: Online Version

An online version is hosted at [mathmodel.top](https://mathmodel.top/home) for quick experimentation.

## Key Features

| Feature | Description |
|---------|-------------|
| 4-agent pipeline | Coordinator, Modeler, Coder, Writer |
| SKILLS-driven | Pure skills, no harness, works with Claude Code and Codex |
| 17 Typst templates | National, MCM/ICM, Huawei Cup, Huashu Cup, and more |
| 9-step validation | Text leak, numerical consistency, Typst compile, PDF visual check |
| 4-layer fault tolerance | Retry, fallback hand-off, evaluator shadow, feedback rerun |
| Human-in-loop | 6 actions: confirm, edit, regenerate, ask, skip, abort |
| Multi-LLM | Each agent uses a different model via litellm |
| RAG knowledge base | ChromaDB + Rerank for modeling methods and templates |
| Code interpreter | Local Jupyter, E2B, Daytona |
| Web search | Tavily API for real-world data |
| sci-box | Scientific figures (SHAP, ROC, Taylor) and draw.io flow charts |
| Desktop apps | macOS (Apple Silicon + Intel), Windows 64-bit |
| Online version | mathmodel.top |
| Low cost | Workflow agentless, no agent framework dependency |
| Custom templates | Prompt injection for each subtask |
| Docker support | One-command deployment |
| 10 languages | Frontend UI supports EN, ZH, ES, JA, KO, DE, FR, PT, RU, AR |

## Why MathModelAgent Matters

Mathematical modeling competitions are a microcosm of the broader knowledge-work challenge: take a vague problem, apply specialized expertise across multiple domains, produce a polished deliverable. The traditional approach is brute force: throw three smart people and 72 hours at it.

MathModelAgent shows a different path. Break the task into specialized roles, give each role to an AI agent with deep domain knowledge, validate the output rigorously, and let a human step in only at key decision points. The result is a paper that approaches award-level quality in one hour instead of three days.

The SKILLS-driven architecture is the most interesting part. By not building its own agent framework and instead riding on Claude Code or Codex, MathModelAgent benefits from every improvement to those harnesses. When Claude Code gets better at code execution, MathModelAgent gets better. When Codex adds new capabilities, MathModelAgent inherits them. This is the composable future of AI agents: specialized skills on top of general-purpose harnesses.

And the cost? The project claims single-task costs as low as about 1 RMB (roughly $0.14). That is a fraction of what most LLM API calls cost for a single conversation. The workflow agentless design, which avoids the overhead of agent frameworks, keeps costs down.

Whether you are a student preparing for a modeling competition, a researcher who needs to generate a quick analysis, or a developer interested in the architecture of multi-agent systems, MathModelAgent is worth studying.

## Related Posts

- [HyperFrames: Write HTML, Render Video, Built for AI Agents](/HyperFrames-Write-HTML-Render-Video-Built-for-Agents/)
- [TeamAI: Make Every Team AI Native with Git-Native Skill Sync](/TeamAI-Make-Every-Team-AI-Native/)
- [WeMM Embedding: Tencent Universal Multimodal Embedding](/WeMM-Embedding-Tencent-Universal-Multimodal-Embedding/)
