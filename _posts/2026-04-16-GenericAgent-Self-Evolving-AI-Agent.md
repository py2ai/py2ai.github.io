---
layout: post
title: "GenericAgent: Self-Evolving AI Agent with Skill Tree Growth"
description: "Explore GenericAgent, a self-evolving AI agent that grows its skill tree from a 3.3K-line seed, achieving full system control with 6x less code than traditional approaches."
date: 2026-04-16
header-img: "img/post-bg.jpg"
permalink: /GenericAgent-Self-Evolving-AI-Agent/
featured-img: ai-coding-frameworks/ai-coding-frameworks
tags:
  - AI Agents
  - Self-Evolving
  - Python
  - Open Source
author: "PyShine"
---

## Introduction

In the rapidly evolving landscape of AI agents, GenericAgent stands out as a remarkable achievement in minimalism and self-evolution. With just approximately 3,000 lines of core code, this framework enables any Large Language Model (LLM) to gain system-level control over a local computer, covering browser automation, terminal execution, filesystem operations, keyboard/mouse input, screen vision, and even mobile device control through ADB.

What truly sets GenericAgent apart is its design philosophy: **don't preload skills - evolve them**. Every time the agent solves a new task, it automatically crystallizes the execution path into a skill for direct reuse later. The longer you use it, the more skills accumulate, forming a personal skill tree that grows entirely from the 3K-line seed code.

## The Self-Evolution Mechanism

The fundamental innovation of GenericAgent lies in its self-evolution mechanism. Unlike traditional agents that come pre-loaded with hundreds of modules, GenericAgent starts minimal and grows its capabilities organically.

```
[New Task] --> [Autonomous Exploration] (install deps, write scripts, debug & verify) -->
[Crystallize Execution Path into skill] --> [Write to Memory Layer] --> [Direct Recall on Next Similar Task]
```

This approach means that after a few weeks of use, your agent instance will have a skill tree that no one else in the world has - all grown from the same seed code. Consider these practical examples:

| What you say | What the agent does the first time | Every time after |
|---|---|---|
| "Read my WeChat messages" | Install dependencies, reverse database, write read script, save skill | One-line invoke |
| "Monitor stocks and alert me" | Install mootdx, build selection flow, configure cron, save skill | One-line start |
| "Send this file via Gmail" | Configure OAuth, write send script, save skill | Ready to use |

## Architecture Overview

GenericAgent accomplishes complex tasks through a sophisticated combination of **Layered Memory**, **Minimal Toolset**, and **Autonomous Execution Loop**. Let's examine each component in detail.

![GenericAgent Architecture](/assets/img/diagrams/generic-agent/generic-agent-architecture.svg)

The architecture diagram above illustrates the core components of GenericAgent. At the center sits the Core Agent, a compact implementation of approximately 3,000 lines that orchestrates all operations. The Agent Loop, comprising only about 100 lines of code, handles the continuous cycle of perception, reasoning, execution, and memory updates.

The Layered Memory System provides a hierarchical approach to knowledge management. L0 contains Meta Rules that define the agent's core behavioral constraints. L1 serves as an Insight Index for rapid routing and recall. L2 stores Global Facts accumulated during long-term operation. L3 holds Task Skills and SOPs (Standard Operating Procedures) for reusable workflows. L4 archives Session Records for long-horizon recall.

The 9 Atomic Tools form the foundation for all external interactions. These include `code_run` for executing arbitrary code, `file_read` and `file_write` for filesystem operations, `file_patch` for precise modifications, `web_scan` for perceiving web content, `web_execute_js` for browser control, `ask_user` for human-in-the-loop confirmation, and two memory management tools for persisting context across sessions.

## Skill Tree Growth Mechanism

![Skill Growth Mechanism](/assets/img/diagrams/generic-agent/generic-agent-skill-growth.svg)

The skill tree growth mechanism represents the core innovation of GenericAgent. When encountering a new task, the agent enters an autonomous exploration phase where it installs necessary dependencies, writes scripts, and verifies the solution through debugging. Once successful, the execution path is crystallized into a skill and written to the appropriate memory layer.

This crystallization process is what enables GenericAgent to grow its capabilities over time. Each solved task contributes to a growing repository of skills that can be instantly recalled when similar tasks arise. The skill tree becomes increasingly specialized to your specific use cases and workflows.

The memory system ensures that relevant knowledge is always in scope. The layered approach prevents context pollution while maintaining access to critical information. When a new task arrives, the agent can quickly search through its skill repository to find relevant precedents, dramatically reducing the exploration required for similar tasks.

## System Control Workflow

![System Control Workflow](/assets/img/diagrams/generic-agent/generic-agent-control-workflow.svg)

The execution loop follows a clear pattern: perceive the environment state, perform task reasoning, execute tools, write experience to memory, and loop. This cycle continues until the task is complete or requires human intervention.

The perception phase involves scanning the current state of the browser, filesystem, or other controlled systems. The reasoning phase uses the LLM to determine the next best action based on the current state and accumulated memory. Tool execution happens through the 9 atomic tools, each designed for a specific type of interaction. After each action, relevant experiences are written to memory for future reference.

The workflow supports complex multi-step operations through its loop structure. Each iteration builds upon the previous one, maintaining context through the working memory and history tracking. The agent can switch between different tools and contexts seamlessly, enabling sophisticated task completion.

## Comparison with Traditional Agents

![Comparison with Traditional Agents](/assets/img/diagrams/generic-agent/generic-agent-comparison.svg)

When compared to traditional agent frameworks, GenericAgent demonstrates remarkable efficiency across multiple dimensions. The codebase is approximately 6x smaller than comparable solutions, with the core implementation fitting in roughly 3,000 lines versus hundreds of thousands in alternatives.

Deployment is significantly simpler, requiring only `pip install` plus an API key configuration. Traditional agents often require complex multi-service orchestration with numerous dependencies and configuration files. The browser control approach preserves login sessions by injecting into real browsers, whereas alternatives typically use sandboxed or headless browsers that lose session state.

The operating system control capabilities extend beyond file and terminal operations to include mouse/keyboard input, screen vision, and mobile device control through ADB. Most importantly, the self-evolution capability means the agent grows more capable over time, while traditional agents remain stateless between sessions.

Token efficiency is another significant advantage. GenericAgent operates with a context window under 30K tokens, a fraction of the 200K to 1M tokens consumed by other agents. This layered memory approach ensures the right knowledge is always in scope, reducing noise and hallucinations while improving success rates at a fraction of the cost.

## Core Features

### Self-Evolving Capabilities

The self-evolving nature of GenericAgent means that capabilities grow with every use. Each solved task automatically crystallizes into a skill, forming your personal skill tree. This organic growth ensures that the agent becomes increasingly specialized for your specific workflows and use cases.

### Minimal Architecture

The approximately 3K lines of core code and roughly 100-line Agent Loop represent a triumph of minimalist design. There are no complex dependencies and zero deployment overhead. This simplicity makes the codebase easy to understand, modify, and extend.

### Strong Execution

GenericAgent achieves strong execution capabilities through injection into real browsers, preserving login sessions and enabling interaction with authenticated web applications. The 9 atomic tools provide direct control over the system, enabling complex automation tasks.

### High Compatibility

The framework supports major LLM providers including Claude, Gemini, Kimi, and MiniMax. It runs cross-platform and can be extended to work with additional models through its modular LLM interface.

### Token Efficiency

Operating with a context window under 30K tokens, GenericAgent achieves remarkable efficiency. The layered memory system ensures that relevant information is always available while minimizing noise. This results in fewer hallucinations, higher success rates, and significantly lower costs compared to alternatives.

## Installation and Quick Start

Getting started with GenericAgent is straightforward:

```bash
# Clone the repository
git clone https://github.com/lsdefine/GenericAgent.git
cd GenericAgent

# Install minimal dependencies
pip install streamlit pywebview

# Configure API Key
cp mykey_template.py mykey.py
# Edit mykey.py and fill in your LLM API Key

# Launch
python launch.pyw
```

The minimal dependency list ensures quick setup. The framework supports multiple LLM backends, allowing you to choose the provider that best fits your needs.

## Bot Interface Options

GenericAgent supports multiple bot frontends for different use cases:

### Telegram Bot

Configure your Telegram bot token and allowed users in `mykey.py`:

```python
tg_bot_token = 'YOUR_BOT_TOKEN'
tg_allowed_users = [YOUR_USER_ID]
```

Then run with:
```bash
python frontends/tgapp.py
```

### WeChat Bot

For personal WeChat integration:
```bash
pip install pycryptodome qrcode requests
python frontends/wechatapp.py
```

The first launch will display a QR code for WeChat binding. After that, you can interact with the agent through WeChat messages.

### Alternative Frontends

Additional frontend options include:
- Qt-based desktop application: `python frontends/qtapp.py`
- Alternative Streamlit UI: `streamlit run frontends/stapp2.py`

## The 9 Atomic Tools

The power of GenericAgent comes from its carefully designed set of atomic tools:

| Tool | Function |
|------|----------|
| `code_run` | Execute arbitrary Python or shell code |
| `file_read` | Read file contents with keyword search support |
| `file_write` | Write or append content to files |
| `file_patch` | Make precise modifications to existing files |
| `web_scan` | Perceive web page content and tab information |
| `web_execute_js` | Execute JavaScript to control browser behavior |
| `ask_user` | Request human input for critical decisions |
| `update_working_checkpoint` | Update working memory with key information |
| `start_long_term_update` | Begin long-term memory crystallization |

These tools provide the foundation for all agent interactions. The `code_run` tool is particularly powerful, enabling dynamic installation of Python packages, writing new scripts, calling external APIs, or controlling hardware at runtime. Temporary abilities can be crystallized into permanent tools through this mechanism.

## Layered Memory System

The memory system is organized into five distinct layers:

**L0 - Meta Rules**: Core behavioral rules and system constraints that govern agent behavior. These rules define the fundamental operating principles and safety boundaries.

**L1 - Insight Index**: A minimal index layer for fast routing and recall. This layer enables quick searches through accumulated knowledge to find relevant precedents.

**L2 - Global Facts**: Stable knowledge accumulated over long-term operation. This includes environment facts, user preferences, and verified configurations.

**L3 - Task Skills/SOPs**: Reusable workflows for completing specific task types. These are the crystallized execution paths that form the skill tree.

**L4 - Session Archive**: Archived task records distilled from finished sessions. This layer enables long-horizon recall and learning from past experiences.

## Self-Bootstrap Proof

A remarkable aspect of GenericAgent is that everything in the repository, from installing Git and running `git init` to every commit message, was completed autonomously by the agent itself. The author never opened a terminal once. This self-bootstrap proof demonstrates the practical capabilities of the framework in real-world development scenarios.

## Demo Showcase

The repository includes several demonstration scenarios:

- **Food Delivery Order**: Navigate delivery apps, select items, and complete checkout automatically
- **Quantitative Stock Screening**: Screen stocks with quantitative conditions like EXPMA golden cross and turnover thresholds
- **Autonomous Web Exploration**: Browse and periodically summarize web content
- **Expense Tracking**: Query expenses through Alipay via ADB control
- **Batch Messaging**: Send bulk WeChat messages by driving the WeChat client

These demonstrations showcase the versatility of GenericAgent across different domains and interaction types.

## Latest Updates

The project continues to evolve with regular updates:

- **April 2026**: Introduced L4 session archive memory and scheduler cron integration
- **March 2026**: Added personal WeChat bot frontend support
- **March 2026**: Released million-scale Skill Library
- **March 2026**: Released "Dintal Claw" - a GenericAgent-powered government affairs bot
- **March 2026**: Featured by Jiqizhixin (Machine Heart)
- **January 2026**: GenericAgent V1.0 public release

## Conclusion

GenericAgent represents a paradigm shift in AI agent design. By starting with a minimal seed and growing capabilities through actual use, it achieves remarkable efficiency while maintaining flexibility. The layered memory system ensures relevant knowledge is always available, and the self-evolution mechanism means your agent becomes increasingly specialized over time.

For developers and researchers interested in AI agents, GenericAgent offers a clean, understandable codebase that demonstrates core principles without overwhelming complexity. The 3K-line core provides an excellent foundation for learning, experimentation, and extension.

The project is open source under the MIT License, making it accessible for both personal and commercial use. Whether you're building automation workflows, researching agent architectures, or simply exploring the possibilities of self-evolving AI systems, GenericAgent provides a compelling starting point.

## Resources

- **GitHub Repository**: [https://github.com/lsdefine/GenericAgent](https://github.com/lsdefine/GenericAgent)
- **Getting Started Guide**: Available in the repository as GETTING_STARTED.md
- **Community**: Join the WeChat or Feishu groups for discussion and support
- **License**: MIT License

---

*GenericAgent demonstrates that powerful AI agents don't require massive codebases. Through intelligent design and self-evolution, a 3K-line seed can grow into a sophisticated system capable of complex real-world tasks.*