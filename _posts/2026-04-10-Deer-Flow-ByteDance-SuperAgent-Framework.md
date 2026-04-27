---
layout: post
title: "Deer Flow: ByteDance's Open-Source Long-Horizon SuperAgent Framework"
description: "Explore Deer Flow, ByteDance's cutting-edge open-source SuperAgent framework for long-horizon AI tasks with advanced planning and execution capabilities."
date: 2026-04-10
header-img: "img/post-bg.jpg"
permalink: /Deer-Flow-ByteDance-SuperAgent-Framework/
featured-img: ai-coding-frameworks/ai-coding-frameworks
tags:
  - AI Agents
  - ByteDance
  - Open Source
  - SuperAgent
  - Long-Horizon Tasks
author: "PyShine"
---

# Deer Flow: ByteDance's Open-Source Long-Horizon SuperAgent Framework

DeerFlow (**D**eep **E**xploration and **E**fficient **R**esearch **Flow**) is an open-source **super agent harness** developed by ByteDance that orchestrates sub-agents, memory, and sandboxes to accomplish complex, long-horizon tasks. With over 60,000 stars on GitHub and the highest monthly growth rate of +33,722 stars, DeerFlow has become one of the most popular AI agent frameworks in the open-source community.

## What is DeerFlow?

DeerFlow started as a Deep Research framework, but the community pushed it far beyond its original scope. Developers began using it to build data pipelines, generate slide decks, create dashboards, and automate content workflows. This evolution revealed that DeerFlow was not just a research tool but a **harness** - a runtime that gives agents the infrastructure to actually get work done.

DeerFlow 2.0 is a ground-up rewrite that shares no code with version 1. It is built on LangGraph and LangChain, shipping with everything an agent needs out of the box: a filesystem, memory, skills, sandbox-aware execution, and the ability to plan and spawn sub-agents for complex, multi-step tasks.

## Architecture Overview

![DeerFlow Architecture](/assets/img/diagrams/deer-flow-architecture.svg)

### Understanding the Architecture

The architecture diagram above illustrates the core components of DeerFlow and their interactions. This design represents a significant advancement in AI agent architecture, moving beyond simple request-response patterns to a fully orchestrated multi-agent system.

**User Request Layer**

The entry point for all DeerFlow interactions is the User Request. This represents any task or query that a user submits to the system, whether it is a simple question, a complex research task, or a multi-step workflow request. The system is designed to handle requests of varying complexity, automatically determining the appropriate execution strategy based on the task requirements.

**Gateway API (REST Endpoint)**

The Gateway API serves as the unified entry point for all external interactions with DeerFlow. Built as a REST endpoint, it provides:

- Standard HTTP interfaces for easy integration with existing systems
- Authentication and authorization handling
- Request validation and routing
- Rate limiting and quota management
- Streaming response support for long-running tasks

The Gateway API abstracts away the complexity of the underlying agent system, providing a clean interface for developers and users. It supports multiple transport protocols including HTTP, WebSocket for real-time updates, and SSE (Server-Sent Events) for streaming responses.

**Lead Agent (Orchestrator)**

At the heart of DeerFlow is the Lead Agent, which acts as the master orchestrator. This component is responsible for:

- Understanding and parsing user intent from natural language requests
- Generating execution plans that break down complex tasks into manageable steps
- Coordinating the activities of multiple sub-agents
- Managing context and state across the entire execution lifecycle
- Synthesizing results from various sub-agents into coherent outputs

The Lead Agent uses advanced planning algorithms to determine the optimal execution strategy. It can decide whether a task requires parallel sub-agent execution, sequential processing, or a hybrid approach. This intelligent orchestration is what enables DeerFlow to handle long-horizon tasks that span minutes to hours.

**Sub-Agents (Parallel Execution Units)**

DeerFlow can spawn multiple specialized sub-agents to handle different aspects of a complex task:

- **Research Sub-Agent**: Handles information gathering, web searches, and data collection from various sources
- **Code Sub-Agent**: Manages code generation, file operations, and development tasks
- **Analysis Sub-Agent**: Performs data analysis, synthesis, and insight extraction

Each sub-agent operates in its own isolated context, preventing interference between different task aspects. This isolation is crucial for maintaining clean execution boundaries and enabling true parallelism. Sub-agents can be dynamically spawned based on task requirements, and the Lead Agent manages their lifecycle from creation to termination.

**Skills Registry (Extensible Modules)**

The Skills Registry is DeerFlow's plugin architecture, allowing developers to extend the system's capabilities without modifying core code. Skills are structured capability modules defined in Markdown files that specify:

- Workflow definitions and best practices
- Tool references and integration points
- Input/output schemas and validation rules
- Execution parameters and constraints

DeerFlow ships with built-in skills for research, report generation, slide creation, web pages, image and video generation, and more. The progressive loading mechanism ensures that only necessary skills are loaded into context, keeping the context window lean and efficient.

**Long-Term Memory**

Unlike most agents that forget everything when a conversation ends, DeerFlow maintains persistent memory across sessions. This long-term memory system:

- Stores user preferences and accumulated knowledge
- Tracks recurring workflows and patterns
- Builds a profile of the user's writing style and technical stack
- Enables context continuity across multiple sessions

The memory system uses intelligent deduplication to prevent repeated information from accumulating endlessly. This ensures that the memory remains useful and relevant over time, rather than becoming a bloated repository of redundant data.

**Sandbox (File System)**

DeerFlow has its own execution environment with a full filesystem view. Each task gets isolated access to:

- `/mnt/user-data/uploads/` - User-provided files
- `/mnt/user-data/workspace/` - Agent working directory
- `/mnt/user-data/outputs/` - Final deliverables

With `AioSandboxProvider`, shell execution runs inside isolated containers, providing security and reproducibility. This is the difference between a chatbot with tool access and an agent with an actual execution environment.

**Deliverables Output**

The final stage of the pipeline produces structured outputs that can include:

- Research reports and documentation
- Generated code and applications
- Slide decks and presentations
- Web pages and dashboards
- Images, videos, and multimedia content

The output system supports multiple formats and can deliver results through various channels including the web interface, IM platforms (Telegram, Slack, Feishu, WeCom), and direct file downloads.

## Workflow Execution

![DeerFlow Workflow](/assets/img/diagrams/deer-flow-workflow.svg)

### Understanding the Workflow

The workflow diagram demonstrates how DeerFlow processes requests from initial input to final delivery. This execution model is designed for long-horizon tasks that require sophisticated planning and adaptive execution.

**User Request Received**

The workflow begins when a user submits a request through any supported channel. DeerFlow supports multiple input channels:

- Web interface at `http://localhost:2026`
- IM platforms (Telegram, Slack, Feishu, WeCom)
- Claude Code integration via the `/claude-to-deerflow` command
- Embedded Python client for programmatic access

Each channel provides the same core capabilities, ensuring consistent behavior regardless of how users interact with the system.

**Parse Intent and Context**

The first processing step involves understanding what the user actually wants. This goes beyond simple keyword matching:

- Natural language understanding extracts the core intent
- Context from previous interactions is retrieved from long-term memory
- User preferences and constraints are identified
- Available resources (files, skills, tools) are inventoried

This parsing stage is crucial for determining the appropriate execution strategy. A request like "analyze this paper" might trigger a research sub-agent, while "create a slide deck about this topic" would engage multiple skills including research, content generation, and slide creation.

**Generate Execution Plan**

Once the intent is understood, DeerFlow creates an execution plan. This plan:

- Identifies the skills and tools needed
- Determines the sequence of operations
- Estimates resource requirements
- Identifies potential parallelization opportunities

The planning algorithm considers dependencies between steps, available resources, and the complexity of the task. For simple requests, the plan might be a single step. For complex requests, it could involve multiple sub-agents working in parallel.

**Complex Task Decision Point**

A key innovation in DeerFlow is its ability to dynamically decide between direct execution and sub-agent spawning:

- **Simple tasks** (e.g., "What is the capital of France?") are executed directly without spawning sub-agents
- **Complex tasks** (e.g., "Research the history of AI and create a presentation") require sub-agent orchestration

This decision is based on factors including:
- Estimated number of steps required
- Need for specialized capabilities
- Potential for parallel execution
- Context window considerations

**Spawn Sub-Agents (for Complex Tasks)**

When the task complexity warrants it, DeerFlow spawns specialized sub-agents:

- Each sub-agent receives a scoped context focused on its specific task
- Sub-agents operate independently, enabling true parallelism
- The Lead Agent maintains coordination and monitors progress
- Results are collected and synthesized upon completion

This parallel execution model is what enables DeerFlow to handle tasks that would take minutes or hours to complete. A research task might fan out into a dozen sub-agents, each exploring a different angle, then converge into a single comprehensive report.

**Execute Skills and Tools**

Whether executing directly or through sub-agents, the actual work happens through skills and tools:

- **Skills** provide structured workflows and best practices
- **Tools** perform specific operations (web search, file operations, bash execution)
- **MCP servers** extend capabilities through the Model Context Protocol

The execution engine manages resource allocation, timeout handling, and result caching. It ensures that operations complete successfully and handles errors gracefully.

**Sandbox Operations**

All file and system operations occur within the sandbox environment:

- Isolated containers prevent security risks
- Per-thread directories ensure clean separation
- File operations are logged and auditable
- Generated artifacts are organized systematically

The sandbox is what transforms DeerFlow from a conversational AI into an agent that can actually build things. It can read, write, and edit files, view images, and execute shell commands (when configured safely).

**Synthesize Results**

After execution, DeerFlow synthesizes results from all sub-agents and operations:

- Individual results are aggregated
- Conflicts and inconsistencies are resolved
- A coherent narrative is constructed
- Output formatting is applied based on the deliverable type

This synthesis step is crucial for producing polished, professional outputs rather than raw data dumps.

**Update Memory**

The final step before delivery is updating the long-term memory:

- New facts and preferences are stored
- Successful patterns are reinforced
- Context is compressed to prevent bloat
- Duplicate information is skipped

This ensures that future interactions benefit from past experiences, making DeerFlow increasingly personalized and effective over time.

**Deliver Output**

Finally, the completed output is delivered through the appropriate channel:

- Web interface displays results inline
- IM platforms receive formatted messages and files
- Files are made available for download
- API responses return structured data

The delivery mechanism adapts to the channel's capabilities, ensuring the best possible user experience regardless of how they interact with DeerFlow.

## Core Features

![DeerFlow Features](/assets/img/diagrams/deer-flow-features.svg)

### Understanding the Features

The features diagram illustrates the five major capability areas of DeerFlow and how they connect to specific tools and functionalities. This modular architecture enables extensibility while maintaining a cohesive user experience.

**Skills System (Extensible Modules)**

The Skills System is DeerFlow's primary extensibility mechanism. Skills are structured capability modules that define workflows, best practices, and references to supporting resources. Key aspects include:

- **Progressive Loading**: Skills are loaded only when needed, keeping the context window lean and efficient for token-sensitive models
- **Built-in Skills**: DeerFlow ships with skills for deep research, code generation, slide creation, web page development, image generation, video creation, and more
- **Custom Skills**: Developers can create their own skills by defining Markdown files with workflow specifications
- **Skill Archives**: The Gateway accepts `.skill` archives with optional frontmatter metadata for version, author, and compatibility

The skills directory structure separates public built-in skills from custom user-defined skills:

```
/mnt/skills/public/
├── research/SKILL.md
├── report-generation/SKILL.md
├── slide-creation/SKILL.md
├── web-page/SKILL.md
└── image-generation/SKILL.md

/mnt/skills/custom/
└── your-custom-skill/SKILL.md
```

**Sub-Agents (Parallel Execution)**

Complex tasks rarely fit in a single pass. DeerFlow's sub-agent architecture enables:

- **Dynamic Spawning**: Sub-agents are created on-demand based on task requirements
- **Isolated Context**: Each sub-agent runs in its own context, preventing interference
- **Parallel Execution**: Multiple sub-agents can work simultaneously on different aspects
- **Structured Results**: Sub-agents report back with organized, parseable outputs
- **Lead Agent Synthesis**: The orchestrator combines results into coherent deliverables

This architecture is how DeerFlow handles tasks spanning minutes to hours: a research task might fan out into a dozen sub-agents, each exploring a different angle, then converge into a single report, website, or slide deck.

**Long-Term Memory (Persistent Context)**

Most agents forget everything when a conversation ends. DeerFlow remembers:

- **User Profile**: Preferences, writing style, technical stack
- **Accumulated Knowledge**: Facts and information learned across sessions
- **Recurring Workflows**: Patterns and templates for common tasks
- **Session Continuity**: Context that persists across multiple interactions

Memory is stored locally and stays under user control. The system uses intelligent deduplication to prevent repeated information from accumulating endlessly, ensuring memory remains useful over time.

**Sandbox Environment (Isolated Execution)**

DeerFlow doesn't just talk about doing things - it has its own computer:

- **Full Filesystem**: Each task gets isolated access to uploads, workspace, and outputs
- **Container Isolation**: With `AioSandboxProvider`, execution runs in isolated Docker containers
- **Security**: Host bash is disabled by default in local mode for safety
- **Flexibility**: Can read, write, edit files, view images, and execute commands when configured

This is the fundamental difference between a chatbot with tool access and an agent with an actual execution environment.

**IM Channels (Multi-Platform)**

DeerFlow supports receiving tasks from messaging apps without requiring a public IP:

| Channel | Transport | Difficulty |
|---------|-----------|------------|
| Telegram | Bot API (long-polling) | Easy |
| Slack | Socket Mode | Moderate |
| Feishu / Lark | WebSocket | Moderate |
| WeCom | WebSocket | Moderate |

Channels auto-start when configured, and all support commands like `/new` (new conversation), `/status` (thread info), `/models` (list models), `/memory` (view memory), and `/help`.

## Installation

### Prerequisites

- Python 3.12+
- Node.js 22+
- Docker (recommended for sandbox isolation)

### Quick Start

1. **Clone the repository:**

```bash
git clone https://github.com/bytedance/deer-flow.git
cd deer-flow
```

2. **Generate configuration files:**

```bash
make config
```

3. **Configure your model(s):**

Edit `config.yaml` to define at least one model:

```yaml
models:
  - name: gpt-4
    display_name: GPT-4
    use: langchain_openai:ChatOpenAI
    model: gpt-4
    api_key: $OPENAI_API_KEY
    max_tokens: 4096
    temperature: 0.7
```

4. **Set API keys:**

Create a `.env` file:

```bash
OPENAI_API_KEY=your-openai-api-key
TAVILY_API_KEY=your-tavily-api-key
```

5. **Start with Docker (Recommended):**

```bash
make docker-init    # Pull sandbox image (first time only)
make docker-start   # Start services
```

6. **Access the interface:**

Open `http://localhost:2026` in your browser.

### Local Development

For development without Docker:

```bash
make check   # Verify prerequisites
make install # Install dependencies
make dev     # Start development servers
```

## Usage Examples

### Basic Chat

```python
from deerflow.client import DeerFlowClient

client = DeerFlowClient()

# Simple chat
response = client.chat("Analyze this paper for me", thread_id="my-thread")
print(response)
```

### Streaming Responses

```python
# Streaming (LangGraph SSE protocol)
for event in client.stream("hello"):
    if event.type == "messages-tuple" and event.data.get("type") == "ai":
        print(event.data["content"])
```

### Model and Skill Management

```python
# List available models
models = client.list_models()

# List available skills
skills = client.list_skills()

# Enable/disable skills
client.update_skill("web-search", enabled=True)

# Upload files for analysis
client.upload_files("thread-1", ["./report.pdf"])
```

### Claude Code Integration

DeerFlow integrates with Claude Code for terminal-based interaction:

```bash
# Install the skill
npx skills add https://github.com/bytedance/deer-flow --skill claude-to-deerflow

# Use in Claude Code
/claude-to-deerflow
```

## Comparison with Other Agent Frameworks

| Feature | DeerFlow | AutoGPT | CrewAI | LangGraph |
|---------|----------|---------|--------|-----------|
| Sub-Agent Orchestration | Yes | Limited | Yes | Manual |
| Long-Term Memory | Yes | No | No | Optional |
| Sandbox Execution | Yes | Limited | No | No |
| Skills System | Yes | Plugins | Tools | Tools |
| IM Channel Support | Yes | No | No | No |
| Model Agnostic | Yes | Yes | Yes | Yes |
| Open Source | Yes | Yes | Yes | Yes |

DeerFlow stands out with its comprehensive approach to agent infrastructure, providing not just orchestration but also memory, sandbox execution, and multi-channel support out of the box.

## Security Considerations

DeerFlow has high-privilege capabilities including system command execution and file operations. The framework is designed for **local trusted environments** (accessible only via 127.0.0.1).

**Security Recommendations:**

- Deploy only in trusted networks
- Use IP allowlists for cross-device access
- Configure authentication gateways for public deployments
- Enable sandbox mode (Docker) for untrusted workloads
- Stay updated with security patches

## Conclusion

DeerFlow represents a significant advancement in AI agent frameworks, moving beyond simple request-response patterns to a fully orchestrated multi-agent system. With its combination of sub-agent orchestration, long-term memory, sandbox execution, and extensible skills system, DeerFlow provides the infrastructure for agents to actually accomplish complex, long-horizon tasks.

The framework's rapid adoption (60,000+ stars, fastest growing in its category) demonstrates the community's recognition of its value. Whether you're building research tools, content pipelines, or automated workflows, DeerFlow provides the foundation for sophisticated AI applications.

## Resources

- [GitHub Repository](https://github.com/bytedance/deer-flow)
- [Official Website](https://deerflow.tech)
- [Documentation](https://github.com/bytedance/deer-flow#documentation)
- [Contributing Guide](https://github.com/bytedance/deer-flow/blob/main/CONTRIBUTING.md)
