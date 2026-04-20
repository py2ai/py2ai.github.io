---
layout: post
title: "Hermes Agent Orange Book: A Practical Guide to Self-Improving AI Agents"
description: "Deep dive into the Hermes Agent Orange Book - the definitive practical guide to Nous Research's Hermes Agent framework, covering its learning loop, three-layer memory system, 40+ built-in tools, and self-improving skill architecture."
date: 2026-04-20
header-img: "assets/img/diagrams/hermes-agent-orange-book/hermes-agent-architecture.svg"
permalink: /Hermes-Agent-Orange-Book-Practical-AI-Agent-Guide/
featured-img: ai-coding-frameworks/ai-coding-frameworks
tags: [AI, Agent, Hermes, Nous-Research, Self-Improving, LLM, Open-Source]
author: PyShine
---

The AI agent landscape is evolving at breakneck speed. Among the most compelling entries is the Hermes Agent from Nous Research -- an open-source framework that does not merely execute tasks, but actively learns, remembers, and improves itself over time. The **Hermes Agent Orange Book**, authored by HuaShu, is the definitive practical guide to understanding and mastering this framework. With the Hermes repository surpassing 27,000 GitHub stars within just two months of launch, the Orange Book has become essential reading for anyone building production-grade AI agents.

This post walks through the core concepts, architecture, and practical applications covered in the Orange Book's 17 chapters across 5 parts -- from the foundational Harness-to-Hermes mapping to the philosophical boundaries of self-improving agents.

--

## From Harness to Hermes

The Orange Book begins by tracing the evolution from **Harness** -- Nous Research's earlier engineering-focused framework -- to **Hermes**, its productized successor. This is not a simple rename. The transition represents a fundamental shift from a developer tool to an autonomous agent platform.

The book identifies five critical layers that were transformed in this evolution:

- **Instruction Layer -> Skill System**: Where Harness relied on static prompt templates, Hermes introduces a dynamic Skill system that can be created, improved, and shared. Skills are not hardcoded -- they are living artifacts that evolve with use.
- **Constraint Layer -> Tool Permissions**: Harness had rigid execution boundaries. Hermes implements a granular permission model where each tool can be selectively enabled or restricted, giving users fine-grained control over what the agent can and cannot do.
- **Feedback Layer -> Learning Loop**: The most transformative change. Harness treated feedback as logs; Hermes closes the loop, turning every interaction into a learning opportunity through its 5-step Learning Loop.
- **Memory Layer -> Three-Layer Memory**: Harness had ephemeral context. Hermes introduces a structured three-layer memory system (Session, Persistent, Skill) that gives the agent genuine continuity across conversations and over time.
- **Orchestration Layer -> Sub-Agent Delegation**: Where Harness was single-threaded, Hermes can spawn up to 3 concurrent sub-agents, each with independent context, enabling parallel task execution and sophisticated delegation patterns.

This mapping is not just historical context -- it provides a mental model for understanding why Hermes works the way it does, and how to leverage each layer effectively.

--

## Architecture Overview

![Hermes Agent Architecture](/assets/img/diagrams/hermes-agent-orange-book/hermes-agent-architecture.svg)

The Hermes Agent architecture is designed around a central orchestration core that coordinates five major subsystems: the Skill system, the Memory system, the Tool ecosystem, the Gateway layer, and the Learning Loop. At the center sits the agent's reasoning engine, powered by configurable LLM providers (OpenRouter, OpenAI, Anthropic, or local models via Ollama). This engine receives user input through one of 14 platform gateways -- Telegram, Discord, Slack, WhatsApp, Signal, CLI, and more -- each acting as a bidirectional bridge between the user and the agent core.

When a message arrives, the orchestration engine follows a structured pipeline: first, it consults the Memory system to load relevant context (session history, persistent identity, and applicable skills). Then, it determines which Skills are relevant to the request, loading their instructions and associated tool permissions. The engine constructs a prompt that weaves together the user's message, retrieved memories, active skill instructions, and available tool definitions. This composite prompt is sent to the LLM, which returns both a textual response and any tool invocations.

Tool calls are executed within the permission boundaries defined by the active Skill, and results are fed back into the reasoning loop. If the agent determines that a task requires parallel execution, it can delegate to sub-agents -- up to 3 concurrent sub-agents, each operating with its own isolated context while reporting back to the parent. After the interaction completes, the Learning Loop activates: the agent evaluates whether new skills should be distilled, whether persistent memories should be updated, and whether the user model should be refined. This architecture ensures that every conversation leaves the agent smarter than it was before.

The entire state is persisted in a single SQLite database (`state.db`) that stores conversation history, FTS5 search indexes, and skill metadata, making Hermes portable and self-contained. Configuration lives in a single `config.yaml` file, and persistent memories are stored as Markdown files (`MEMORY.md` and `USER.md`) in the `~/.hermes/memories/` directory, ensuring transparency and human-editability.

--

## The Learning Loop Flywheel

![Hermes Learning Loop](/assets/img/diagrams/hermes-agent-orange-book/hermes-learning-loop.svg)

The Learning Loop is the most distinctive feature of the Hermes Agent, and the Orange Book dedicates significant attention to explaining how it creates a self-reinforcing flywheel of improvement. The loop consists of five steps that form a closed cycle:

**1. Curate Memory** -- After each interaction, the agent examines the conversation and identifies information worth retaining. This is not a simple dump of everything said; the agent applies relevance filtering, extracting only the facts, preferences, and patterns that have long-term value. These curated memories are written to the persistent memory layer (`MEMORY.md`), creating a growing knowledge base that informs future interactions.

**2. Create Skill** -- When the agent encounters a repeated pattern of behavior -- a sequence of tool calls, a reasoning pattern, or a response format that recurs -- it can distill that pattern into a new Skill. This is the key innovation: Skills are not just pre-defined; they emerge organically from usage. The agent writes a Skill definition (in Markdown with YAML frontmatter) that captures the pattern, making it instantly reusable.

**3. Skill Self-Improvement** -- Existing Skills are not static. The Learning Loop includes a mechanism for Skills to be refined over time. When a Skill produces suboptimal results, the agent can modify its instructions, add edge cases, or improve its tool selections. This creates a Darwinian pressure within the Skill ecosystem: well-performing Skills survive and propagate, while poorly-performing ones are iteratively improved or replaced.

**4. FTS5 Recall** -- The agent uses SQLite's FTS5 (Full-Text Search) engine to index all memories, skills, and conversation history. When a new conversation begins, the agent does not rely solely on recent context; it performs semantic retrieval across the entire indexed history, pulling in relevant skills, past decisions, and user preferences. This ensures that the agent's responses are informed by its complete experience, not just the current session.

**5. User Modeling** -- The final step closes the loop by updating the agent's model of the user. Through Honcho integration, the agent builds a progressively detailed understanding of who the user is -- their communication style, technical expertise, preferences, and goals. This user model is stored in `USER.md` and is consulted at the start of every interaction, ensuring that the agent's behavior adapts to each individual user over time.

Together, these five steps create a flywheel: better memories lead to better skills, better skills lead to better outcomes, better outcomes generate richer feedback, richer feedback refines the user model, and a refined user model produces more relevant memories. The agent literally gets better with every conversation.

--

## Three-Layer Memory System

![Hermes Memory System](/assets/img/diagrams/hermes-agent-orange-book/hermes-memory-system.svg)

Memory is the foundation upon which the Learning Loop operates, and the Orange Book provides a detailed analysis of Hermes's three-layer memory architecture. Each layer serves a distinct purpose and operates with different persistence guarantees:

**Session Memory (Episodic)** -- This is the shortest-lived layer, encompassing everything that happens within a single conversation. It includes the full message history, tool call results, and intermediate reasoning steps. Session memory is stored in the SQLite database and is loaded in its entirety when a conversation resumes. It answers the question: "What happened in this conversation?" When the session ends, the most valuable elements are promoted to Persistent Memory, while the rest remains searchable via FTS5 for future recall.

**Persistent Memory (Semantic)** -- This layer stores the agent's understanding of identity, relationships, and long-term knowledge. It is implemented as two Markdown files: `MEMORY.md` (general knowledge and facts the agent has learned) and `USER.md` (the user model built through Honcho). Persistent Memory answers the question: "Who am I and who is the user?" Because these files are plain Markdown, they are human-readable and editable, giving users direct control over what the agent remembers. The Orange Book emphasizes that this transparency is a deliberate design choice -- users should always be able to see and modify what the agent knows about them.

**Skill Memory (Procedural)** -- This is the most structured layer, storing the agent's knowledge of how to perform tasks. Each Skill is a self-contained unit that includes instructions, tool permissions, trigger conditions, and examples. Skill Memory answers the question: "How do I do things?" Skills are stored as Markdown files with YAML frontmatter in the `~/.hermes/skills/` directory, following the `agentskills.io` standard. The three sources of Skills -- Bundled (40+ built-in), Agent-created (auto-distilled), and Skills Hub (community) -- each contribute to this layer, creating a rich and growing procedural knowledge base.

The Orange Book highlights a critical insight: these three layers mirror the human cognitive architecture of episodic, semantic, and procedural memory. This is not accidental -- it reflects a deliberate design philosophy that treats the agent as a cognitive system rather than a simple stateless API. The addition of Honcho for user modeling adds a fourth dimension: a continuously refined model of the user that personalizes the agent's behavior across all three memory layers.

--

## Skill System Deep Dive

The Skill system is where Hermes truly differentiates itself from other agent frameworks. The Orange Book identifies three distinct sources of Skills:

**Bundled Skills (40+)** -- These ship with Hermes out of the box, covering common tasks like web search, code execution, file management, data analysis, and more. Each bundled Skill is a Markdown file with YAML frontmatter that defines its name, description, trigger phrases, tool permissions, and instruction body. They are immediately available without any configuration.

**Agent-Created Skills** -- This is the most innovative source. When the Learning Loop detects a repeated pattern, the agent can automatically distill a new Skill from its experience. For example, if a user repeatedly asks the agent to summarize articles in a specific format, the agent will create a Skill that captures this pattern, including the preferred format, the tools to use, and the reasoning steps to follow. These Skills are stored locally and can be further refined through the self-improvement cycle.

**Skills Hub (Community)** -- Hermes supports a community-driven Skills Hub where users can share and discover Skills. These follow the `agentskills.io` standard, ensuring compatibility and discoverability. The Orange Book walks through the process of publishing a Skill to the Hub and installing Skills from other users.

Each Skill follows the `agentskills.io` standard, which defines a consistent structure:

```yaml
---
name: article-summarizer
description: Summarizes articles in a structured format
triggers:
  - summarize
  - summarize this article
  - tl;dr
tools:
  - web_search
  - web_scraper
  - file_write
---

# Article Summarizer

You are an expert at distilling articles into clear, structured summaries.

## Instructions

1. Use web_search or web_scraper to retrieve the article content
2. Identify the key thesis, supporting arguments, and conclusions
3. Format the summary as:
   - **Thesis**: One-sentence core argument
   - **Key Points**: 3-5 bullet points
   - **Conclusion**: Implications and takeaways
4. Save the summary if requested
```

This structure makes Skills self-documenting, composable, and easy to debug -- a significant improvement over opaque prompt engineering.

--

## 40+ Built-In Tools

Hermes comes with over 40 built-in tools organized into five categories:

**Execution Tools** -- These enable the agent to take direct action in the world: running shell commands, executing Python code, managing files and directories, and interacting with the operating system. Examples include `shell_exec`, `python_exec`, `file_read`, `file_write`, and `file_search`.

**Information Tools** -- These allow the agent to gather information from external sources: web search, web scraping, RSS feed parsing, and API queries. Examples include `web_search`, `web_scrape`, `rss_read`, and `http_request`.

**Media Tools** -- These handle multimedia content: image generation, image analysis, audio transcription, and text-to-speech. Examples include `image_generate`, `image_analyze`, `audio_transcribe`, and `tts_speak`.

**Memory Tools** -- These manage the agent's own memory: reading and writing persistent memories, searching the FTS5 index, and managing the user model. Examples include `memory_read`, `memory_write`, `memory_search`, and `user_model_update`.

**Coordination Tools** -- These enable multi-agent interaction: spawning sub-agents, delegating tasks, and managing inter-agent communication. Examples include `subagent_spawn`, `subagent_delegate`, and `subagent_message`.

Each tool operates within the permission boundaries defined by the active Skill, ensuring that the agent can only access the tools it needs for the current task. This principle of least privilege is a core security feature of the Hermes architecture.

--

## MCP Integration

Beyond its 40+ built-in tools, Hermes supports the **Model Context Protocol (MCP)**, enabling connection to over 6,000 external applications and services. The Orange Book covers two connection modes:

**Stdio Mode** -- For local integrations, MCP servers communicate with Hermes through standard input/output streams. This is the simplest mode and is suitable for tools running on the same machine, such as local databases, file systems, and development tools.

**SSE Mode** -- For remote integrations, MCP servers communicate through Server-Sent Events over HTTP. This enables connection to cloud services, APIs, and remote tools without requiring local installation.

Configuration is straightforward:

```yaml
mcp:
  servers:
    - name: filesystem
      command: mcp-filesystem
      args: ["--root", "/home/user/documents"]
    - name: github
      url: https://mcp.github.com/sse
      headers:
        Authorization: Bearer ${GITHUB_TOKEN}
```

This extensibility means that Hermes is not limited to its built-in capabilities. Any service that exposes an MCP interface can become a tool in the agent's arsenal, making it a truly universal agent platform.

--

## Multi-Platform Gateway

One of Hermes's most practical features is its **multi-platform gateway**, which allows a single agent instance to communicate across 14 different platforms simultaneously:

- **Messaging**: Telegram, Discord, Slack, WhatsApp, Signal
- **Social**: Twitter/X, Instagram
- **Voice**: Phone (Twilio)
- **Development**: GitHub, GitLab
- **Interface**: CLI, Web UI, API

The key innovation is **cross-platform conversation continuity**. A user can start a conversation on Telegram, continue it on Discord, and review the results on the Web UI -- all within the same session and with full context preserved. This is possible because the gateway layer abstracts away platform-specific details, routing all messages through a unified interface to the agent core.

Configuration for multi-platform deployment:

```yaml
gateway:
  telegram:
    token: YOUR_TELEGRAM_BOT_TOKEN
  discord:
    token: YOUR_DISCORD_BOT_TOKEN
  slack:
    token: YOUR_SLACK_BOT_TOKEN
    signing_secret: YOUR_SLACK_SIGNING_SECRET
  whatsapp:
    phone_number: YOUR_WHATSAPP_NUMBER
    verify_token: YOUR_VERIFY_TOKEN
```

The Orange Book provides detailed setup instructions for each platform, including webhook configuration, authentication, and platform-specific quirks.

--

## Hermes vs OpenClaw vs Claude Code

![Hermes Comparison](/assets/img/diagrams/hermes-agent-orange-book/hermes-comparison.svg)

The Orange Book's final chapters provide a rigorous three-way comparison between Hermes, OpenClaw, and Claude Code -- three of the most prominent agent frameworks in the current landscape. The comparison is structured across six dimensions:

**Architecture Philosophy** -- Hermes is built around self-improvement and memory continuity. Its Learning Loop and three-layer memory system make it fundamentally oriented toward agents that get better over time. OpenClaw takes a more modular approach, emphasizing composability and interchangeability of components. Claude Code is deeply integrated with Anthropic's ecosystem, prioritizing seamless developer experience within the Claude toolchain.

**Memory Model** -- Hermes's three-layer memory (Session, Persistent, Skill) with Honcho user modeling provides the most comprehensive memory architecture. OpenClaw offers configurable memory backends but lacks the structured three-layer separation. Claude Code relies primarily on session-level context with CLAUDE.md files for persistent instructions, but does not have the automatic skill distillation or user modeling that Hermes provides.

**Skill System** -- Hermes's Skill system is the most mature, with three sources (Bundled, Agent-created, Skills Hub) and the `agentskills.io` standard for interoperability. OpenClaw has a plugin system but lacks the auto-distillation mechanism. Claude Code's skill system is tightly coupled to its ecosystem and does not support community sharing in the same way.

**Tool Ecosystem** -- Hermes offers 40+ built-in tools plus 6,000+ MCP integrations. OpenClaw has a comparable built-in tool set but a smaller MCP ecosystem. Claude Code's tool access is primarily through its terminal and file system integration, with MCP support being more recent.

**Multi-Platform** -- Hermes leads with 14 platform gateways and cross-platform conversation continuity. OpenClaw supports several platforms but with less seamless continuity. Claude Code is primarily a CLI tool, with web access through the Anthropic API.

**Self-Improvement** -- This is Hermes's clearest differentiator. The Learning Loop creates a genuine flywheel where the agent improves with every interaction. Neither OpenClaw nor Claude Code has an equivalent mechanism. Claude Code can learn from CLAUDE.md instructions, but this is manual rather than automatic. OpenClaw has experimental learning features but they are not yet production-ready.

The Orange Book concludes that Hermes is the strongest choice for users who want an agent that improves over time and operates across multiple platforms, while Claude Code excels for developers deeply embedded in the Anthropic ecosystem, and OpenClaw is best for those who prioritize modularity and component interchangeability.

--

## Getting Started

Setting up Hermes is straightforward. The Orange Book provides a step-by-step guide:

**Installation**:

```bash
# Clone the repository
git clone https://github.com/nous-research/hermes.git
cd hermes

# Install dependencies
pip install -r requirements.txt

# Or use the Docker image
docker pull nousresearch/hermes:latest
```

**Minimal Configuration**:

```yaml
model:
  provider: openrouter
  api_key: sk-or-xxxxx
  model: anthropic/claude-sonnet-4
terminal: local
gateway:
  telegram:
    token: YOUR_BOT_TOKEN
```

**First Conversation**:

```bash
# Start with CLI (simplest gateway)
python hermes.py --gateway cli

# Or start with Telegram
python hermes.py --gateway telegram
```

The directory structure after setup:

```
~/.hermes/
├── config.yaml          # Agent configuration
├── state.db             # SQLite database (conversation history + FTS5 index)
├── skills/
│   └── bundled/         # 40+ built-in Skills
└── memories/
    ├── MEMORY.md         # General persistent memory
    └── USER.md           # User model (Honcho)
```

The Orange Book walks through each configuration option, explains the security implications of tool permissions, and provides troubleshooting tips for common setup issues.

--

## Real-World Applications

The Orange Book dedicates four chapters to real-world application scenarios:

**Knowledge Assistant** -- Hermes can serve as a persistent knowledge companion that remembers your research interests, preferred sources, and summarization style. Over time, it learns to proactively suggest relevant information and filter noise, becoming increasingly valuable with each interaction.

**Development Automation** -- With its 40+ built-in tools and MCP integration, Hermes can automate development workflows: running tests, managing git operations, reviewing code, and generating documentation. The Skill system allows developers to create custom automation patterns that are refined through the Learning Loop.

**Content Creation** -- The multi-platform gateway makes Hermes ideal for content creation workflows that span multiple channels. A single agent can draft content, generate images, post to social media, and track engagement -- all while learning the creator's voice and style preferences over time.

**Multi-Agent Scenarios** -- The sub-agent delegation system enables sophisticated multi-agent workflows. A parent agent can spawn specialized sub-agents for research, writing, and review, each operating with its own context and tools. The parent coordinates their outputs, synthesizing a final result that leverages the strengths of each sub-agent.

--

## Conclusion

The Hermes Agent Orange Book is more than a technical manual -- it is a comprehensive exploration of what it means to build AI agents that genuinely improve over time. By mapping the evolution from Harness to Hermes, detailing the Learning Loop flywheel, explaining the three-layer memory architecture, and comparing Hermes against its competitors, the Orange Book provides both the conceptual framework and the practical guidance needed to build production-grade autonomous agents.

The framework's emphasis on self-improvement through the Learning Loop, transparency through human-readable memory files, and extensibility through the Skill system and MCP integration positions Hermes as a leading choice for developers who want agents that grow smarter with every interaction. As the AI agent ecosystem continues to mature, the principles documented in the Orange Book -- memory continuity, skill distillation, user modeling, and cross-platform operation -- will likely become the standard against which all agent frameworks are measured.

For developers, researchers, and anyone interested in the future of autonomous AI, the Hermes Agent Orange Book is essential reading. The repository is available at [alchaincyf/hermes-agent-orange-book](https://github.com/alchaincyf/hermes-agent-orange-book) on GitHub.