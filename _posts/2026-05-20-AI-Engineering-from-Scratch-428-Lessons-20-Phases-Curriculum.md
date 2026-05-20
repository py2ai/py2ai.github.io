---
layout: post
title: "AI Engineering from Scratch: 428 Lessons, 20 Phases, From Linear Algebra to Autonomous Agent Swarms"
description: "AI Engineering from Scratch is a comprehensive open-source curriculum that takes you from math foundations to multi-agent systems in 428 lessons across 20 phases. Every algorithm built from raw math first, every lesson ships a reusable artifact."
date: 2026-05-20
permalink: /AI-Engineering-from-Scratch-428-Lessons-20-Phases-Curriculum/
featured-img: /assets/img/diagrams/ai-engineering/ai-engineering-architecture.svg
categories: [AI, Education, Open Source]
tags: [ai-engineering, curriculum, machine-learning, deep-learning, llm, agents, open-source, education]
keywords: [AI engineering from scratch, AI curriculum, machine learning course, deep learning from scratch, LLM engineering, agent engineering, open source AI education]
author: "PyShine"
---

## What is AI Engineering from Scratch?

**AI Engineering from Scratch** is a massive open-source curriculum by **Rohit Ghumare** that covers the entire AI engineering stack — from linear algebra to autonomous agent swarms — in **428 lessons across 20 phases**, totaling approximately **320 hours** of content. It's free, MIT-licensed, and built to run on your own laptop.

The core philosophy: **you don't just learn AI, you build it. End-to-end. By hand.** Every algorithm gets built from raw math first. Backprop. Tokenizer. Attention. Agent loop. By the time PyTorch shows up, you already know what it's doing under the hood.

![AI Engineering Architecture](/assets/img/diagrams/ai-engineering/ai-engineering-architecture.svg)

## The Problem It Solves

> *84% of students already use AI tools. Only 18% feel prepared to use them professionally.*

Most AI material teaches in scattered pieces. A paper here, a fine-tuning post there, a flashy agent demo somewhere else. The pieces rarely line up. You ship a chatbot but can't explain its loss curve. You hook a function to an agent but can't say what attention does inside the model that's calling it.

This curriculum is the **spine** — 20 phases, 428 lessons, four languages (Python, TypeScript, Rust, Julia). Linear algebra at one end, autonomous swarms at the other.

## The 20-Phase Curriculum

Twenty phases stack on top of each other. Math is the floor. Agents and production are the roof.

### Foundation Layer (Phases 0-2)

| Phase | Topic | Lessons | What You Build |
|-------|-------|---------|----------------|
| **P0** | Setup and Tooling | 12 | Dev environment, GPU setup, Docker, Jupyter |
| **P1** | Math Foundations | 22 | Linear algebra, calculus, probability, optimization, SVD, Fourier transform |
| **P2** | ML Fundamentals | 18 | Linear regression, decision trees, SVMs, ensemble methods, pipelines |

### Deep Learning Core (Phase 3)

| Phase | Topic | Lessons | What You Build |
|-------|-------|---------|----------------|
| **P3** | Deep Learning Core | 13 | Perceptron, backprop from scratch, optimizers, mini framework, PyTorch intro |

### Modality Branches (Phases 4-6, 9)

| Phase | Topic | Lessons | What You Build |
|-------|-------|---------|----------------|
| **P4** | Computer Vision | 28 | CNNs, YOLO, U-Net, GANs, diffusion, ViT, 3D Gaussian splatting, world models |
| **P5** | NLP Foundations | 29 | Tokenization, Word2Vec, NER, attention, machine translation, RAG chunking |
| **P6** | Speech and Audio | 17 | ASR, Whisper, TTS, voice cloning, music generation, neural codecs |
| **P9** | Reinforcement Learning | 12 | Q-learning, DQN, PPO, RLHF, multi-agent RL |

### Transformer Revolution (Phases 7-8)

| Phase | Topic | Lessons | What You Build |
|-------|-------|---------|----------------|
| **P7** | Transformers Deep Dive | 14 | Self-attention, multi-head attention, BERT, GPT, MoE, Flash Attention |
| **P8** | Generative AI | 14 | VAEs, GANs, diffusion models, Stable Diffusion, ControlNet, video generation |

### LLM Layer (Phases 10-12)

| Phase | Topic | Lessons | What You Build |
|-------|-------|---------|----------------|
| **P10** | LLMs from Scratch | 22 | Tokenizers, pre-training mini GPT, RLHF, DPO, quantization, DeepSeek-V3 walkthrough |
| **P11** | LLM Engineering | 15 | Prompt engineering, RAG, LoRA fine-tuning, function calling, MCP, guardrails |
| **P12** | Multimodal AI | 25 | CLIP, LLaVA, BLIP-2, video-language, embodied VLAs, multimodal RAG |

### Agent Layer (Phases 13-16)

| Phase | Topic | Lessons | What You Build |
|-------|-------|---------|----------------|
| **P13** | Tools and Protocols | 23 | MCP servers/clients, A2A protocol, OAuth 2.1, OpenTelemetry, skill SDKs |
| **P14** | Agent Engineering | 42 | Agent loop, ReWOO, LangGraph, CrewAI, OpenAI/Claude SDKs, workbench |
| **P15** | Autonomous Systems | 22 | Self-improvement, kill switches, constitutional AI, METR, safety frameworks |
| **P16** | Multi-Agent and Swarms | 25 | Supervisor patterns, A2A, consensus, swarm optimization, MARL |

### Production and Ethics (Phases 17-18)

| Phase | Topic | Lessons | What You Build |
|-------|-------|---------|----------------|
| **P17** | Infrastructure and Production | 28 | vLLM, SGLang, TensorRT-LLM, chaos engineering, FinOps, compliance |
| **P18** | Ethics, Safety, Alignment | 30 | Red-teaming, watermarking, differential privacy, EU AI Act, dual-use risk |

### Capstone (Phase 19)

| Phase | Topic | Projects | What You Build |
|-------|-------|----------|----------------|
| **P19** | Capstone Projects | 17 | Terminal coding agent, RAG chatbot, voice assistant, multi-agent team, and more |

## The 6-Beat Lesson Structure

Every lesson follows the same six beats. The *Build It / Use It* split is the spine — you implement the algorithm from scratch first, then run the same thing through the production library.

![AI Engineering Features](/assets/img/diagrams/ai-engineering/ai-engineering-features.svg)

1. **MOTTO** — One-line core idea
2. **PROBLEM** — Concrete pain point
3. **CONCEPT** — Diagrams and intuition
4. **BUILD IT** — Raw math, no frameworks
5. **USE IT** — Same thing in PyTorch / sklearn
6. **SHIP IT** — Produce a reusable artifact

## Every Lesson Ships Something

Other curricula end with "congratulations, you learned X." Each lesson here ends with a **reusable tool** you can install or paste into your daily workflow:

| Artifact | What It Is |
|----------|-----------|
| **Prompts** | Paste into any AI assistant for expert-level help on a narrow task |
| **Skills** | Drop into Claude, Cursor, Codex, OpenClaw, Hermes, or any agent that reads `SKILL.md` |
| **Agents** | Deploy as autonomous workers — you wrote the loop yourself in Phase 14 |
| **MCP Servers** | Plug into any MCP-compatible client. Built end-to-end in Phase 13 |

By the end of the curriculum, you have a portfolio of **428 artifacts** you actually understand because you built them.

## A Worked Example: The Agent Loop

Phase 14, lesson 1: the agent loop. ~120 lines of pure Python, no dependencies.

```python
def run(query, tools):
    history = [user(query)]
    for step in range(MAX_STEPS):
        msg = llm(history)
        if msg.tool_calls:
            for call in msg.tool_calls:
                result = tools[call.name](**call.args)
                history.append(tool_result(call.id, result))
            continue
        return msg.content
    raise StepLimitExceeded
```

And the shipped artifact — a skill you can drop into any agent:

```markdown
---
name: agent-loop
description: ReAct-style loop for any tool list
phase: 14
lesson: 01
---

Implement a minimal agent loop that...
```

## Built-in Agent Skills

The curriculum includes two built-in skills for AI coding agents:

| Skill | What It Does |
|-------|-------------|
| `/find-your-level` | Ten-question placement quiz. Maps your knowledge to a starting phase and produces a personalized path with hour estimates |
| `/check-understanding <phase>` | Per-phase quiz, eight questions, with feedback and specific lessons to review |

## Four Languages

The curriculum uses **Python, TypeScript, Rust, and Julia** — each chosen where it makes the most sense:

- **Python** for ML/DL core and rapid prototyping
- **TypeScript** for production agents and web interfaces
- **Rust** for performance-critical inference and real-time processing
- **Julia** for mathematical foundations and numerical computing

## Where to Start

| Background | Start at | Estimated Time |
|------------|----------|----------------|
| New to programming and AI | Phase 0 — Setup | ~306 hours |
| Know Python, new to ML | Phase 1 — Math Foundations | ~270 hours |
| Know ML, new to deep learning | Phase 3 — Deep Learning Core | ~200 hours |
| Know deep learning, want LLMs and agents | Phase 10 — LLMs from Scratch | ~100 hours |
| Senior engineer, only want agent engineering | Phase 14 — Agent Engineering | ~60 hours |

## Quick Start

```bash
# Clone and run
git clone https://github.com/rohitg00/ai-engineering-from-scratch.git
cd ai-engineering-from-scratch
python phases/01-math-foundations/01-linear-algebra-intuition/code/vectors.py
```

Or read any completed lesson on [aiengineeringfromscratch.com](https://aiengineeringfromscratch.com) — no setup, no cloning.

## Why This Curriculum Matters

AI Engineering from Scratch fills a critical gap in AI education. While most resources teach you to *call APIs*, this curriculum teaches you to *understand what those APIs are doing* — and then build production systems on top of that understanding.

Key differentiators:

1. **Build-first pedagogy** — Every algorithm from raw math before touching a framework
2. **Shippable artifacts** — 428 reusable tools, not just homework exercises
3. **Complete stack coverage** — From linear algebra to autonomous swarms in one coherent path
4. **Multi-language** — Python, TypeScript, Rust, Julia where each excels
5. **Agent-native** — Built-in SkillKit integration for Claude, Cursor, Codex, and more
6. **Production-ready** — Phase 17 covers real infrastructure: vLLM, SGLang, FinOps, compliance

For anyone serious about understanding AI from the ground up — not just using it — this is the most comprehensive open-source curriculum available.

**Repository**: [github.com/rohitg00/ai-engineering-from-scratch](https://github.com/rohitg00/ai-engineering-from-scratch)  
**Stars**: 8.9K+ | **License**: MIT | **Website**: [aiengineeringfromscratch.com](https://aiengineeringfromscratch.com)