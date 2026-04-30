---
layout: post
title: "WorldX: One-Sentence AI World Simulation with Autonomous Agents"
description: "Learn how WorldX turns a single text prompt into a living AI world with autonomous agents, procedural generation, emergent narratives, and god mode intervention -- built with Phaser 3, React 19, and LLM-driven simulation."
date: 2026-04-30
header-img: "img/post-bg.jpg"
permalink: /WorldX-AI-Powered-Virtual-World-Simulation/
featured-img: ai-coding-frameworks/ai-coding-frameworks
categories: [AI Agents, Open Source, Game Development]
tags: [WorldX, AI simulation, autonomous agents, procedural generation, emergent narrative, LLM agents, Phaser 3, pixel art, AI world building, open source]
keywords: "how to use WorldX, WorldX tutorial, AI world simulation, one-sentence world generation, autonomous AI agents, LLM procedural generation, WorldX vs AI dungeon, emergent narrative simulation, pixel art AI game, WorldX installation guide"
author: "PyShine"
---

# WorldX: One-Sentence AI World Simulation with Autonomous Agents

WorldX is an open-source project that turns a single text prompt into a fully autonomous AI world. Describe any scenario -- "A cyberpunk noodle shop where hackers and androids share rumors" -- and WorldX designs the world, generates original pixel-art maps and character sprites, then runs a living simulation where AI agents make decisions, form relationships, hold conversations, and create emergent narratives without any human scripting.

![Pipeline](/assets/img/diagrams/worldx/worldx-pipeline.svg)

### Understanding the One-Sentence-to-World Pipeline

The pipeline diagram above shows how WorldX transforms a single sentence into a living simulation through four distinct stages:

**One-Sentence Prompt**
Everything starts with a natural language description. You type something like "A cozy autumn mountain village with a blacksmith, a tavern owner, a wandering monk, and a curious child" and WorldX handles the rest. No configuration files, no manual character definitions, no map editors.

**Orchestrator -- LLM World Design**
The Orchestrator model (a strong reasoning LLM like Gemini 3.1 Pro) takes your prompt and designs the entire world: the map layout, character personalities, relationships, rules, and scene descriptions. It outputs a structured configuration that defines every element of the simulation.

**Art Generation -- Maps and Sprites**
The Image Gen model creates original pixel-art map tiles and character sprites based on the Orchestrator's configuration. A Vision model then reviews the generated art for quality, locates regions and elements, and provides revision feedback if the art does not match the world description. This review loop ensures the visual output matches your creative vision.

**Simulation Engine -- Autonomous Agents**
The Simulation model drives runtime character behavior. Each agent perceives its environment, makes decisions based on its personality and memories, acts (moves, speaks, interacts with objects), and updates its internal state. The result is a living world where emergent narratives unfold without any pre-written scripts.

**God Mode**
At any point, you can intervene as a "god" -- broadcasting events, editing character memories or personalities, and starting sandbox conversations with any character to see how they respond. This creates a unique blend of emergent storytelling and creative direction.

## Four Model Roles

![Model Roles](/assets/img/diagrams/worldx/worldx-model-roles.svg)

### Understanding the Four Model Roles

WorldX uses four distinct LLM roles, each configurable independently. All use the OpenAI-compatible `chat/completions` protocol, so any compatible platform works:

**Orchestrator -- World Design**
The Orchestrator is the creative brain. It takes your one-sentence prompt and designs the entire world structure: map layout, character personalities, relationships, rules, and scene descriptions. This role requires a strong reasoning model because it needs to create coherent, internally consistent worlds from minimal input. Recommended: Gemini 3.1 Pro Preview or equivalent.

**Image Gen -- Art Generation**
The Image Gen model creates original pixel-art map tiles and character sprites. Unlike template-based systems, WorldX generates unique art for each world based on your description. This role requires an image-capable model. Recommended: Gemini 3.1 Flash Image Preview.

**Vision -- Art Review**
The Vision model reviews generated art for quality, locates regions and elements on the map, and provides revision feedback. This creates a quality control loop: if the art does not match the world description, the Vision model flags it and the Image Gen model regenerates. Recommended: Gemini 3.1 Pro Preview (multimodal).

**Simulation -- Agent Behavior**
The Simulation model drives runtime character behavior -- what agents perceive, decide, say, and do. This role runs at high frequency (every simulation tick), so a cheaper, faster model works well. Recommended: Gemini 2.5 Flash or DeepSeek Chat.

The key insight is that you can mix and match providers. Use Google AI Studio's free tier for world creation and art generation, then switch to a cheaper provider like DeepSeek for the high-volume simulation calls.

## Project Architecture

![Architecture](/assets/img/diagrams/worldx/worldx-architecture.svg)

### Understanding the Project Architecture

The architecture diagram shows the four main components of WorldX and how they interact:

**Client Layer -- Phaser 3 + React 19**
The game client uses Phaser 3 for pixel-art rendering and React 19 for overlay UI panels (character dialogue, god mode controls, timeline management). The client connects to the server via HTTP and Server-Sent Events (SSE) for real-time state updates.

**Server Layer -- Express + SQLite + LLM**
The Express.js server hosts the simulation engine, which manages world state, character decisions, and dialogue generation. It communicates with LLM providers through the OpenAI-compatible API and persists all state to SQLite with per-timeline isolation.

**Orchestrator and Generators**
The Orchestrator handles world design and configuration generation. The Generators module handles map and character art creation with a multi-step review loop. These run during world creation, not during simulation, so they only incur LLM costs when you create a new world.

**SQLite Storage**
Each timeline gets its own SQLite database, enabling you to branch, replay, and compare different simulation runs. World state, character memories, relationships, and dialogue history are all persisted, so you can close the browser and resume later.

## Agent Simulation Loop

![Simulation Loop](/assets/img/diagrams/worldx/worldx-simulation-loop.svg)

### Understanding the Agent Simulation Loop

The simulation loop diagram shows the five-step cycle that each AI agent follows on every simulation tick:

**1. Perceive -- Observe Environment and Other Agents**
Each agent observes its surroundings: what objects are nearby, which other agents are present, what conversations are happening, and what time of day it is. This perception step provides the context for decision-making.

**2. Decide -- LLM-Driven Choice Based on Personality and Memory**
The Simulation model receives the agent's perception, personality traits, and accumulated memories, then generates a decision. This is not random -- the LLM considers the agent's character, past experiences, and current goals to produce contextually appropriate behavior.

**3. Act -- Move, Speak, Interact**
The agent executes its decision: moving to a new location, starting a conversation with another agent, interacting with an object, or performing any number of world actions. These actions change the world state that other agents perceive.

**4. Remember -- Update Memory and Relationship State**
After acting, the agent updates its internal memory with what happened and how relationships with other agents have evolved. A blacksmith who was helped by the tavern owner will remember that kindness and factor it into future decisions.

**5. Evolve -- Day/Night Cycle and Scene Transitions**
The world evolves across day/night cycles with scene transitions. Markets close at dusk, taverns get rowdy at night, and characters adjust their behavior based on the time of day. This creates a natural rhythm that makes the world feel alive.

**God Mode Intervention**
At any point in this cycle, you can intervene as a "god" -- broadcasting events (a storm approaches, a stranger arrives), editing character memories (the blacksmith now remembers a debt), or starting a sandbox conversation with any character. These interventions create ripple effects through the simulation, producing emergent narratives that no one scripted.

## Installation

### Quick Start (Preview Mode)

Two pre-built worlds are included. You only need a Simulation model key:

```bash
git clone https://github.com/YGYOOO/WorldX.git
cd WorldX
cp .env.example .env
# Edit .env -- fill in SIMULATION_* fields only
npm install
npm run dev
```

Open `http://localhost:3200`, pick a pre-built world, and hit Play.

### Full Creation Mode

Generate your own worlds from scratch. Requires all 4 model keys:

```bash
# Edit .env -- fill in all 4 model sections
npm run dev
```

Open `http://localhost:3200/create`, type a sentence, and watch your world come to life.

Or use the CLI:

```bash
npm run create -- "A cyberpunk noodle shop where hackers and androids share rumors"
```

### Model Configuration

Each of the 4 model roles needs 3 environment variables:

```env
{ROLE}_BASE_URL=https://openrouter.ai/api/v1    # API base URL
{ROLE}_API_KEY=sk-or-v1-xxxx                     # API key
{ROLE}_MODEL=google/gemini-3.1-pro-preview       # Model identifier
```

| Role | Env Prefix | What It Does | Recommended |
|------|-----------|-------------|-------------|
| Orchestrator | `ORCHESTRATOR_` | Designs world structure, characters, rules | Strong reasoning (gemini-3.1-pro) |
| Image Gen | `IMAGE_GEN_` | Generates map art and character sprites | Image-capable (gemini-flash-image) |
| Vision | `VISION_` | Reviews map quality, locates regions | Multimodal (gemini-3.1-pro) |
| Simulation | `SIMULATION_` | Drives runtime character behavior | Any model, cheaper is fine (gemini-2.5-flash) |

## Key Features

| Feature | Description |
|---------|-------------|
| One-Sentence World Creation | Describe any scenario and watch it materialize into a full simulation |
| AI-Generated Maps and Characters | Original pixel-art created to match your description, not templates |
| Autonomous Agent Simulation | Characters make decisions, form relationships, and hold conversations independently |
| Memory and Personality | Agents remember past events and act according to distinct personalities |
| Multi-Day Evolution | Worlds evolve across day/night cycles with scene transitions |
| God Mode | Broadcast events, edit character profiles/memories, run sandbox chats |
| Timeline System | Branch, replay, and compare different simulation runs |
| Bilingual UI | Chinese and English interface with one-click switching |
| Mix-and-Match Models | Use different LLM providers for each role to optimize cost and quality |
| OpenAI-Compatible API | Any provider supporting chat/completions protocol works |

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "npm install" fails | Ensure Node.js 18+ is installed: `node --version` |
| Blank screen on localhost:3200 | Check that the server is running on port 3100 and the client on 3200 |
| World generation stalls | Verify all 4 model API keys are valid and have sufficient quota |
| Art looks wrong or misaligned | The Vision review loop should catch this; try regenerating the world |
| Simulation feels slow | Switch the Simulation model to a faster/cheaper option like gemini-2.5-flash |
| Port already in use | Client runs on 3200, server on 3100; change with `PORT` env var |

## Conclusion

WorldX represents a fascinating intersection of LLM capabilities and game simulation. By decomposing the world-creation process into four distinct model roles -- Orchestrator for design, Image Gen for art, Vision for quality review, and Simulation for runtime behavior -- it achieves something that no single model could do alone: turning a single sentence into a living, breathing pixel-art world where AI agents create their own stories.

The mix-and-match model architecture is particularly clever. You can use Google AI Studio's free tier for world creation and art generation (which only runs once per world), then switch to a cheaper provider like DeepSeek for the high-frequency simulation calls that drive ongoing agent behavior. This keeps ongoing costs minimal while still producing rich, emergent narratives.

Whether you are interested in AI agent simulation, procedural generation, emergent storytelling, or just want to watch a blacksmith and a tavern owner argue about the price of ale in a pixel-art village, WorldX offers a compelling and accessible entry point.

**Links:**

- GitHub: [https://github.com/YGYOOO/WorldX](https://github.com/YGYOOO/WorldX)