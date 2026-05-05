---
layout: post
title: "WorldSeed: Emergent Multi-Agent World Engine for AI Simulation"
description: "Learn how WorldSeed creates stateful AI worlds where agents live autonomously with asymmetric information, tick-based execution, and LLM-powered Dungeon Masters. Build emergent multi-agent simulations with YAML configs."
date: 2026-05-05
header-img: "img/post-bg.jpg"
permalink: /WorldSeed-Emergent-Multi-Agent-World-Engine/
featured-img: ai-coding-frameworks/ai-coding-frameworks
categories: [AI Agents, Python, Open Source]
tags: [WorldSeed, multi-agent systems, AI simulation, emergent behavior, Python framework, LLM Dungeon Master, YAML world definition, asymmetric information, autonomous agents, game engine]
keywords: "how to use WorldSeed, WorldSeed multi-agent simulation tutorial, WorldSeed vs other agent frameworks, WorldSeed installation guide, emergent AI behavior simulation, Python world engine setup, LLM Dungeon Master for agents, YAML world definition tutorial, asymmetric information AI agents, open source multi-agent framework"
author: "PyShine"
---

# WorldSeed: Emergent Multi-Agent World Engine

WorldSeed is a stateful world engine that enables AI agents to live autonomously in simulated environments where emergent behavior arises from simple rules and asymmetric information. Unlike traditional agent frameworks that orchestrate tasks, WorldSeed creates living worlds where agents perceive, decide, and act within a shared reality -- each seeing only what their perception scope allows.

> **Key Insight:** WorldSeed processes 10 ticks per second with full state persistence, enabling real-time multi-agent simulations where each agent receives a filtered view of the world based on their perception scope.

## How It Works

WorldSeed operates on a tick-based execution model inspired by game engines. Each tick, the engine pulls actions from a queue, validates preconditions, resolves effects through either deterministic DSL operations or LLM-powered Dungeon Master judgment, and delivers filtered perceptions to each agent. This creates a world where agents can cooperate, compete, and surprise their creators.

![WorldSeed Architecture Overview](/assets/img/diagrams/worldseed/worldseed-architecture.svg)

### Understanding the Architecture

The architecture diagram above illustrates the core components and their interactions in WorldSeed. Let's break down each layer:

**Agent Layer (AI Agents)**
- OpenClaw and Codex agents connect to the world through REST API endpoints and WebSocket connections
- Agents register themselves, submit actions, and receive filtered perceptions through dedicated endpoints
- The WebSocket connection enables real-time notifications when an agent's inbox receives new information

**API Layer (REST + WebSocket)**
- `POST /register` -- Agents join the world and receive their initial state
- `GET /perceive` -- Agents request their filtered view of the world
- `POST /act` -- Agents submit actions to the action queue
- `WebSocket /ws` -- Real-time push notifications for inbox updates

**Processing Layer (Tick Engine)**
- The TickRunner fires at a configurable interval (default 10 ticks/second)
- The WorldEngine orchestrates each tick: pulling actions, checking preconditions, resolving effects
- The RulesEngine validates actions against preconditions defined in the scene YAML
- DSL Effect Ops handle deterministic state changes (math, property updates, list operations)
- The DM Provider uses LLM calls (Anthropic Claude) for uncertain outcomes that require judgment

**State Layer (Persistence)**
- StateStore manages entity CRUD operations with full state history
- EventLog records all events with configurable TTL (time-to-live)
- ConsequenceScanner detects reactive rules triggered by state changes

**Delivery Layer (Perception)**
- The Perceiver applies visibility rules to create agent-specific views
- InboxManager maintains per-agent message queues
- ConnectorProvider pushes notifications via WebSocket or webhook callbacks

![WorldSeed Tick Cycle](/assets/img/diagrams/worldseed/worldseed-tick-cycle.svg)

### Understanding the Tick Cycle

The tick cycle diagram shows the 10-step process that runs every tick in WorldSeed. This is the heartbeat of the simulation:

**1. TickRunner Fires Step** -- The clock loop triggers a new tick at the configured interval. Each tick is a complete processing cycle that advances the world state.

**2. Pull Actions From Queue** -- All actions submitted by agents since the last tick are pulled from the ActionQueue. Actions are processed in FIFO order within each tick.

**3. Check Preconditions** -- The RulesEngine validates each action against its preconditions. For example, a "trade" action might require both parties to have sufficient resources.

**4a. DSL Effects (Deterministic)** -- Actions with clear outcomes are resolved through the DSL effect engine. These include arithmetic operations, property assignments, list manipulations, and conditional logic. DSL effects are fast, reproducible, and never require LLM calls.

**4b. DM Judgment (LLM Call)** -- Actions with uncertain outcomes are routed to the Dungeon Master provider. The DM receives the action context, world state, and relevant history, then returns a structured judgment. This is where emergent behavior truly emerges -- the LLM can introduce creative consequences that the scene designer never explicitly programmed.

> **Amazing:** The Dungeon Master can introduce consequences that the scene designer never explicitly programmed. A simple "explore" action might result in discovering a hidden passage, triggering a chain of events that reshapes the entire simulation.

**5. Apply Effects To StateStore** -- All resolved effects (both DSL and DM) are applied to the StateStore atomically within each tick. This ensures consistency -- no agent sees a partially-updated world.

**6. Scan Consequences** -- The ConsequenceScanner checks if any reactive rules are triggered by the new state. Consequences are like database triggers -- they fire automatically when conditions are met, enabling cascading effects.

**7. Run AutoTick (Decay/Progress)** -- AutoTick rules handle time-based changes: resource decay, relationship progression, environmental shifts. These run every tick without requiring agent actions.

**8. Perceiver Delivers Inboxes** -- Each agent's perception is computed based on their visibility scope. The Perceiver filters the world state so agents only see what they should see.

**9. Notify Agents Via WebSocket** -- Agents receive real-time notifications about their updated inboxes. This enables reactive behavior without polling.

**10. Record Event To Stream Log** -- Every tick produces an event record in the EventLog with configurable TTL. This creates an audit trail for debugging and analysis.

![WorldSeed Perception Model](/assets/img/diagrams/worldseed/worldseed-perception-model.svg)

### Understanding the Perception Model

The perception model is what makes WorldSeed fundamentally different from other agent frameworks. Instead of giving all agents the same world view, WorldSeed implements asymmetric information -- each agent sees only what their perception scope allows.

**World State (Complete)**
The complete world state contains all entities, properties, relationships, and events. No single agent has access to this full picture -- it is the ground truth that the Perceiver uses to generate filtered views.

**Perceiver Engine (Visibility Rules)**
The Perceiver is the core of the information asymmetry system. It takes the complete world state and applies visibility rules defined in the scene YAML to produce agent-specific views. Rules can be based on:
- Location (agents in the same room see each other)
- Relationships (allies share information, enemies do not)
- Properties (hidden doors are only visible to agents with "perception" skill)
- Custom DSL predicates (arbitrary conditions defined by the scene designer)

**Agent Views (Asymmetric)**
- **Agent A** sees the room and their allies -- standard social perception
- **Agent B** sees the room and hidden secrets -- perhaps they have a "perception" skill
- **Agent C** sees only the room -- a newcomer with no special abilities

**Perception Scopes**
- `global` -- All agents see this information (public announcements, weather)
- `target_only` -- Directed to a specific agent (private messages, secrets)
- `admin` -- Omniscient view for debugging and game master oversight
- `custom` -- DSL-defined visibility rules for complex scenarios

**Hidden Properties**
Some properties are never sent to any agent. These are internal state variables used by the engine for tracking quest progress, relationship scores, or other mechanics that should remain invisible to all participants.

> **Important:** Asymmetric information is what drives emergent behavior in WorldSeed. When agents have different knowledge, they make different decisions, form different alliances, and create outcomes that no single agent or designer could predict.

![WorldSeed Scene Configuration](/assets/img/diagrams/worldseed/worldseed-scene-config.svg)

### Understanding Scene Configuration

WorldSeed worlds are defined through YAML configuration files called "scenes." The scene configuration diagram shows how a single YAML file defines every aspect of the simulated world.

**Required Sections**
- `scene` -- Metadata and unique identifier for the world
- `entities` -- What exists in the world (characters, objects, locations)
- `actions` -- What agents can do (move, trade, attack, explore)

**Optional Sections**
- `templates` -- Reusable stat templates for entities
- `agents` -- Who lives in the world and their initial configuration
- `consequences` -- Auto-detected reactive rules
- `auto_tick` -- Automatic changes that happen every tick (decay, progression)
- `perception` -- Visibility rules for asymmetric information
- `sanity_checks` -- Scripted tests to verify world behavior
- `narrator` -- Auto-narrator that generates story text from events

**Runtime Engine**
All configuration sections feed into the Runtime Engine, which processes the tick loop and produces an emergent multi-agent world. The engine reads the YAML once at startup and builds the in-memory state that drives the simulation.

> **Takeaway:** A single YAML file can define an entire world with characters, rules, and emergent dynamics. WorldSeed handles the execution -- you define the rules and let the agents surprise you.

## Installation

```bash
# Clone the repository
git clone https://github.com/AIScientists-Dev/WorldSeed.git
cd WorldSeed

# Install dependencies using uv (recommended)
pip install uv
uv sync

# Or install with pip
pip install -e ".[dm]"
```

The `[dm]` extra installs the Dungeon Master dependencies including `anthropic` for LLM-powered judgment. Without it, you can still use the DSL-only mode for deterministic simulations.

## Usage

### Creating a World

Define your world in a YAML file:

```yaml
scene:
  id: tavern-night
  name: "The Crossroads Tavern"
  description: "A bustling tavern at the crossroads of three kingdoms"

entities:
  - id: tavern
    type: location
    properties:
      name: "The Crossroads Tavern"
      warmth: 7
      noise_level: 5

  - id: stranger
    type: character
    properties:
      name: "Mysterious Stranger"
      gold: 50
      suspicion: 0
      location: tavern

actions:
  - id: eavesdrop
    name: "Eavesdrop on conversations"
    preconditions:
      - stranger.location == tavern
    effects:
      - property: stranger.suspicion
        operation: add
        value: 1

  - id: buy_drink
    name: "Buy a drink"
    preconditions:
      - stranger.gold > 0
    effects:
      - property: stranger.gold
        operation: subtract
        value: 2
      - property: tavern.warmth
        operation: add
        value: 1

perception:
  - scope: global
    includes: [tavern.name, tavern.warmth]
  - scope: target_only
    target: stranger
    includes: [stranger.gold, stranger.suspicion]

auto_tick:
  - property: tavern.noise_level
    operation: add
    value: 1
    condition: "tavern.warmth > 5"
```

### Running the World

```bash
# Start the WorldSeed server
worldseed serve --scene scene.yaml

# Or use the CLI for quick testing
worldseed run --scene scene.yaml --ticks 100
```

### Connecting Agents

Agents connect through the REST API:

```python
import httpx

# Register an agent
response = httpx.post("http://localhost:8000/register", json={
    "agent_id": "my-agent",
    "name": "Explorer",
    "persona": "A curious traveler seeking adventure"
})

# Perceive the world
perception = httpx.get("http://localhost:8000/perceive", params={
    "agent_id": "my-agent"
})

# Take an action
result = httpx.post("http://localhost:8000/act", json={
    "agent_id": "my-agent",
    "action_id": "explore",
    "parameters": {"direction": "north"}
})
```

## Features

| Feature | Description |
|---------|-------------|
| YAML World Definition | Define entire worlds with entities, actions, and rules in simple YAML files |
| Tick-Based Execution | Deterministic processing at 10 ticks/second with full state persistence |
| Asymmetric Information | Each agent sees only what their perception scope allows |
| DSL Effect Engine | Deterministic state changes through math, property, and list operations |
| LLM Dungeon Master | Route uncertain outcomes to LLM for creative, emergent judgment |
| Consequence Scanner | Reactive rules that fire automatically when conditions are met |
| AutoTick Rules | Time-based changes (decay, progression) that run every tick |
| OpenClaw/Codex Support | Connect AI agents through standard REST and WebSocket APIs |
| FastAPI Server | Production-ready HTTP server with WebSocket support |
| State Persistence | Full state history with configurable event TTL |
| Sanity Checks | Built-in testing framework for verifying world behavior |

## Architecture Deep Dive

### StateStore

The StateStore is the single source of truth for all entity state. It provides:
- CRUD operations for entities and their properties
- Atomic updates within each tick
- Deep diff tracking for state changes
- Query capabilities for the Perceiver and ConsequenceScanner

### RulesEngine

The RulesEngine evaluates preconditions before actions execute. Preconditions are expressed as Python-like expressions in the YAML:

```yaml
preconditions:
  - agent.gold >= 10
  - target.location == agent.location
  - "quest_status" in agent.flags
```

### DSL Effect Operations

The DSL effect engine supports these operations:
- `add`, `subtract`, `multiply`, `divide` -- Arithmetic on numeric properties
- `set` -- Direct property assignment
- `append`, `remove` -- List operations
- `conditional` -- If/then/else logic
- `for_each` -- Iterate over collections

### Dungeon Master Provider

When an action's outcome is uncertain (marked with `dm: true`), the DM provider:
1. Receives the action context and relevant world state
2. Sends a structured prompt to the LLM (Anthropic Claude by default)
3. Parses the LLM response into structured effects
4. Applies the effects to the StateStore

This enables truly emergent behavior where the LLM can introduce creative consequences that the scene designer never explicitly programmed.

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: worldseed` | Run `pip install -e .` from the project root |
| `anthropic.AuthenticationError` | Set `ANTHROPIC_API_KEY` environment variable for DM features |
| `YAML parse error` | Validate your scene YAML with `worldseed validate --scene scene.yaml` |
| `Agent not receiving updates` | Check WebSocket connection and ensure the agent is registered |
| `Tick processing too slow` | Reduce the number of DM-judged actions or increase tick interval |
| `State drift between agents` | Verify perception rules are correctly defined in the scene YAML |

## Conclusion

WorldSeed represents a paradigm shift in multi-agent AI simulation. Instead of orchestrating agents through explicit task delegation, it creates worlds where agents live autonomously with asymmetric information, and emergent behavior arises naturally from simple rules and LLM-powered judgment. The YAML-based configuration makes it accessible to both researchers and game designers, while the tick-based engine provides the deterministic foundation needed for reproducible experiments.

Whether you are building AI research simulations, interactive fiction, or testing multi-agent strategies, WorldSeed provides the infrastructure to create worlds that surprise even their creators.

## Links

- **GitHub Repository:** [https://github.com/AIScientists-Dev/WorldSeed](https://github.com/AIScientists-Dev/WorldSeed)