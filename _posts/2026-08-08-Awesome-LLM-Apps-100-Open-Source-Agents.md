---
layout: post
title: "Awesome LLM Apps: 100+ Open-Source AI Agents, Agent Skills, and RAG Apps"
date: 2026-08-08 00:00:00 +0800
categories: [ai, llm, agents, rag, open-source]
tags: [AI, LLM, Agents, RAG, Open Source, AI Apps, Claude, GPT, Gemini, DeepSeek, Llama, Qwen, Agent Skills, Voice AI]
description: "Discover Awesome LLM Apps — a curated collection of 100+ hand-built, end-to-end tested open-source AI agents, agent skills, and RAG apps. Apache-2.0 licensed, multi-model compatible."
author: "PyShine"
---

The LLM app landscape has exploded. Every week brings a new agent framework, a fresh RAG technique, or a voice AI demo that goes viral on Hacker News. But here's the problem: most of these projects are toy demos — half-baked proofs of concept that crash on the second run, lack proper error handling, or are locked behind a paywall and a specific model provider.

If you've ever cloned an "awesome" repo only to find 200 lines of README and zero runnable code, you know the frustration. The gap between "looks great in a tweet" and "actually works in production" is wider than ever.

That's where [**Awesome LLM Apps**](https://github.com/Shubhamsaboo/awesome-llm-apps) comes in. This isn't a list of links — it's a hand-crafted, end-to-end tested collection of 100+ open-source AI agents, agent skills, and RAG applications that you can clone, run, and actually ship.

## What Awesome LLM Apps Offers

At its core, Awesome LLM Apps is a curated repository of production-grade AI applications that spans the full spectrum of modern LLM engineering:

- **100+ agents, skills, and RAG apps** — each one hand-built and tested end-to-end
- **Apache-2.0 licensed** — free to clone, modify, and even sell commercially
- **Multi-model support** — works with Claude, Gemini, GPT, DeepSeek, Llama, Qwen, and other open-source models
- **Real code, not placeholders** — every template includes working code, proper `requirements.txt`, and setup scripts
- **Weekly updates** — new templates drop every week, covering the latest techniques

The repo is the brainchild of Shubham Saboo, an AI engineer who decided the ecosystem needed something better than vaporware. Each template is built with the same philosophy: **run it in 30 seconds, understand it in 30 minutes, ship it in 30 days.**

## Categories and Standout Examples

Awesome LLM Apps organizes its 100+ templates into distinct categories, each addressing a different aspect of AI agent development. Let's explore the standout examples.

### 🧩 Agent Skills

Agent skills are the fastest way to give your coding agent new abilities. Installable with a single command, skills are plain-English instructions that work with Claude Code, Codex, Cursor, and other coding assistants. Every skill ships real code and passes a security + evaluation CI gate.

**Standout examples:**

- **⚰️ Project Graveyard** — Finds every side project you abandoned, tells you why each one died, and helps you finish the one worth going back to. It scans your repositories, analyzes commit patterns, and delivers a forensic post-mortem.

- **♾️ Self-Improving Agent Skills** — Automatically optimizes agent skills using Gemini and the Agent Development Kit (ADK). It's a skill that improves itself — genuinely meta.

- **🧠 Advisor Orchestrator Worker** — A meta-loop with Claude Fable 5 as advisor, GPT-5.6 as orchestrator, and Gemini 3.5 Flash as worker. Each model plays to its strengths.

### 🌱 Starter AI Agents

Single-file agents that run with just an API key — the perfect starting point for learning how LLM agents work.

**Standout examples:**

- **🛫 AI Travel Agent** — Personalized day-by-day travel itineraries, available in both local and cloud modes. It's a complete agent in under 200 lines of Python.

- **📊 AI Data Analysis Agent** — Ask questions of any CSV or Excel file in plain English. Upload your spreadsheet and say "what was the top-selling product last quarter?" — no pandas required.

- **🎙️ AI Blog to Podcast Agent** — Turn any blog URL into a narrated podcast episode. It scrapes the content, generates a script, and synthesizes natural-sounding speech.

- **🔄 Mixture of Agents** — Multiple LLMs answer the same question, and an aggregator picks the best response. A practical pattern for reducing hallucinations.

### 🚀 Advanced AI Agents

Production-style agents with tools, memory, and multi-step reasoning. These are the templates you'd build on for real applications.

**Standout examples:**

- **🏚️ AI Home Renovation Agent** — Photos of your space go in, renovation plans and photorealistic renders come out. Uses Nano Banana Pro for multimodal understanding.

- **🔍 AI Fraud Investigation Agent** — Cross-references public records to flag facilities that don't add up. Built for investigative workflows with multi-step verification.

- **🧬 AI Self-Evolving Agent** — Agents that rewrite their own workflows using EvoAgentX. The meta-learning approach takes agent improvement to the next level.

- **📰 AI Journalist Agent** — Researches, writes, and edits articles on any topic. Includes multi-source verification and editorial review passes.

### 🛰️ Always-on Agents

Background agents that run on schedules or events, monitoring changing context and proactively delivering updates when something needs attention.

**Standout examples:**

- **📰 Always-on Hacker News Briefing Agent** — A scheduled scout that ships a ranked daily brief to Slack or email. It crawls HN, scores stories by your interests, and delivers a digest before your morning coffee.

- **📡 Release Radar Agent** — Watches your dependency releases and briefs you on breaking changes, deprecated APIs, security advisories, and major version bumps. Never get blindsided by a breaking change again.

### 🗣️ Voice AI Agents

Speech-in, speech-out agents using real-time voice APIs. These are the most interactive templates in the collection.

**Standout examples:**

- **🛡️ Insurance Claim Live Agent Team** — Real-time voice claim intake with Gemini Live. A multi-agent system that guides customers through the claims process conversationally.

- **📞 Customer Support Voice Agent** — Voice answers grounded in your own documentation. Upload your help center, and the agent speaks the answers — no training required.

### 📀 RAG (Retrieval Augmented Generation)

The largest category, with 20+ RAG templates ranging from basic chains to agentic and multi-source systems.

**Standout examples:**

- **🧐 Agentic RAG with Reasoning** — Watch the agent's step-by-step reasoning process as it retrieves information. Great for understanding *why* a RAG system produces a particular answer.

- **⛓️ Basic RAG Chain** — The minimal retrieval pipeline, applied to pharma research. Perfect if you want to understand RAG from the ground up before moving to agentic versions.

- **🧬 Multimodal Agentic RAG** — Text, PDFs, images, audio, and video — all answered with citations. A comprehensive template for building RAG over mixed media.

- **🦙 Local RAG Agent** — Llama 3.2 and Qdrant running completely offline, no API keys required. Privacy-preserving RAG for sensitive data.

## Quick Start

Enough talking — let's run something. Here are three ways to get started, from zero to agent in under a minute.

### Option 1: Install a Skill in 10 Seconds

Give your coding agent a new skill with a single command:

```bash
npx skills add https://github.com/Shubhamsaboo/awesome-llm-apps/tree/main/agent_skills/project-graveyard
```

Then ask it: *"why do I never finish my side projects?"*

### Option 2: Clone and Run Any Agent

```bash
git clone https://github.com/Shubhamsaboo/awesome-llm-apps.git
cd awesome-llm-apps/starter_ai_agents/ai_travel_agent
pip install -r requirements.txt
streamlit run travel_agent.py
```

That's it. No complex setup, no infrastructure configuration — just a working agent in your browser.

### Option 3: Try a RAG Tutorial

```bash
cd awesome-llm-apps/rag_tutorials/basic_rag_chain
pip install -r requirements.txt
python rag_chain.py
```

The basic RAG chain template walks you through the fundamentals: document loading, chunking, embedding, vector storage, and retrieval — all in a single script.

### What You Get with Each Template

Every agent in the repo includes:

- A complete Python script you can run immediately
- A `requirements.txt` with all dependencies pinned
- Clear setup instructions in the README
- Environment variable templates (`.env.example`) for API keys
- A Streamlit or CLI interface for interaction
- Testing notes and common troubleshooting solutions

## Who Should Use It

Awesome LLM Apps isn't for everyone — but if any of these describe you, it's definitely for you:

- **Developers learning LLM engineering** — The starter agents and RAG tutorials provide a structured learning path from basic prompts to multi-agent systems. Each template teaches a specific pattern you can reuse.

- **Founders building AI products** — Clone a template that matches your use case and ship a prototype in hours, not weeks. The Apache-2.0 license lets you use the code in commercial products.

- **Teams evaluating agent architectures** — With templates spanning single agents, multi-agent teams, always-on agents, and voice agents, you can compare approaches without building each one from scratch.

- **Researchers exploring RAG techniques** — The 20+ RAG templates cover every major variant: basic chains, agentic RAG, corrective RAG (CRAG), hybrid search, multimodal RAG, local RAG, and knowledge graph RAG.

- **Coding assistant power users** — The agent skills let you extend Claude Code, Codex, or Cursor with domain-specific abilities without writing prompts from scratch.

## The Bigger Picture

The real value of Awesome LLM Apps isn't any single template — it's the pattern library. Each agent demonstrates a specific engineering technique: how to structure multi-agent conversations, how to implement memory in production, how to handle tool use gracefully, how to build RAG with proper evaluation.

Instead of reimplementing these patterns for every new project, you can study the working examples, adapt them to your needs, and focus on the part that's unique to your product.

The repo also publishes [step-by-step tutorials on theunwindai.com](https://www.theunwindai.com/) that walk through the architecture and design decisions behind many of the templates — so you don't just *use* the code, you *understand* it.

## Conclusion

Awesome LLM Apps fills a critical gap in the AI developer ecosystem: production-ready, tested, multi-model LLM applications you can actually use. With 100+ templates across agent skills, starter agents, advanced agents, always-on agents, voice agents, and RAG systems, it's the most comprehensive collection of working LLM applications available under a permissive license.

Whether you're learning the ropes of LLM engineering, prototyping your next AI product, or extending your coding assistant's capabilities, this repo provides the building blocks — tested, working, and ready to ship.

**Key numbers to remember:**

- **100+** open-source templates
- **131k+** GitHub stars
- **Apache-2.0** license — free for commercial use
- **6+** model providers supported
- **Weekly** new template drops

Check out the [Awesome LLM Apps repository on GitHub](https://github.com/Shubhamsaboo/awesome-llm-apps) to explore the full collection. Your next AI agent is already written — you just need to run it.