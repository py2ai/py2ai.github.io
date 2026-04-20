---
layout: post
title: "Dive into Claude Code: Systematic AI Coding Analysis Reveals 98.4% Infrastructure"
description: "A source-level architectural analysis of Claude Code (v2.1.88, ~512K lines) that reveals only 1.6% of the codebase is AI decision logic while 98.4% is deterministic infrastructure across 5 layers, 7 safety systems, and 54 tools."
date: 2026-04-20
header-img: "assets/img/diagrams/dive-into-claude-code-vila/dive-into-claude-code-vila-layered-architecture.svg"
permalink: /Dive-into-Claude-Code-Systematic-AI-Coding-Analysis/
featured-img: ai-coding-frameworks/ai-coding-frameworks
tags: [AI, Claude-Code, Architecture, Agent-Design, LLM, Safety-Systems, Coding-Agent]
author: PyShine
---

## Introduction

How much of an AI coding agent is actually AI? The answer from [VILA-Lab/Dive-into-Claude-Code](https://github.com/VILA-Lab/Dive-into-Claude-Code) is startling: only 1.6%. The remaining 98.4% is deterministic infrastructure -- permission gates, context management pipelines, tool routing logic, and recovery mechanisms. This finding comes from a rigorous academic paper (arXiv:2604.14228) authored by Jiacheng Liu, Xiaohan Zhao, Xinyi Shang, and Zhiqiang Shen from VILA Lab, who performed a source-level architectural analysis of Claude Code version 2.1.88.

The scale of the analysis is impressive: approximately 1,900 TypeScript files comprising roughly 512,000 lines of code were systematically decomposed into 5 architectural layers containing 21 subsystems. The paper traces every design choice back through 13 design principles to 5 foundational human values, creating a values-to-implementation mapping that is rare in software architecture literature. Rather than simply cataloging features, the authors ask why each component exists and what design tension it resolves.

The core thesis challenges the intuition that AI agents are primarily AI. The agent loop itself -- the ReAct-pattern while-loop that drives every turn -- is deceptively simple. The real engineering complexity lives in the systems around it: 7 independent safety layers, a 5-stage context compaction pipeline, 54 tools assembled through a 5-step filtering process, and 27 hook events that intercept actions at every lifecycle stage. This post walks through the key architectural insights from the paper, illustrated with diagrams that map the system's structure, execution flow, safety architecture, and extensibility mechanisms.

## Layered Architecture: 5 Layers, 21 Subsystems

![Layered Architecture](/assets/img/diagrams/dive-into-claude-code-vila/dive-into-claude-code-vila-layered-architecture.svg)

The diagram above illustrates the 5-layer decomposition that the paper identifies as Claude Code's architectural backbone. Each layer has a distinct responsibility, and the separation is not merely logical -- it is enforced through module boundaries and dependency direction. Understanding these layers is essential because the 1.6%/98.4% ratio emerges from how these layers interact: the AI decision logic is concentrated in a narrow band within the Core layer, while the four surrounding layers are entirely deterministic.

The **Surface layer** handles all entry points and rendering. It encompasses the interactive CLI (built with React and Ink for terminal UI), the headless CLI mode (`claude -p`), the Agent SDK for programmatic access, and IDE integrations including Desktop and Browser variants. Critically, all four surface types converge on the same `queryLoop` -- there is one execution engine, not mode-specific ones. The `QueryEngine` class is a conversation wrapper, not the engine itself.

The **Core layer** contains the agent loop and context assembly. The `queryLoop` async generator in `query.ts` orchestrates the model call, tool dispatch, result collection, and repetition. This layer also houses the 5-stage compaction pipeline that runs before every model call, and the subagent spawning logic that delegates tasks to isolated context windows.

The **Safety/Action layer** is where the 98.4% infrastructure ratio becomes tangible. It contains 7 permission modes, the auto-mode ML classifier (`yoloClassifier.ts`), 27 hook events across 5 categories, the tool pool assembly pipeline, and shell sandboxing with filesystem and network isolation. Every tool invocation must pass through this layer before execution.

The **State layer** manages runtime state and persistence through append-only JSONL transcripts, the 4-level CLAUDE.md hierarchy, auto-memory files, and subagent sidechain files. The append-only design choice favors auditability over query power -- every event is human-readable and reconstructable without specialized tooling.

The **Backend layer** provides execution environments: shell execution with sandboxing, MCP connections supporting 7 transport types (stdio, SSE, HTTP, WebSocket, SDK, IDE), and 42 tool subdirectories that implement the 54 available tools. This layer is where actions actually execute, after passing through all upper layers.

## The Agentic Query Loop: 9 Steps Per Turn

![Agentic Query Loop](/assets/img/diagrams/dive-into-claude-code-vila/dive-into-claude-code-vila-query-loop.svg)

The diagram above shows the 9-step pipeline that executes on every turn of the agent loop. This pipeline is the runtime manifestation of the layered architecture -- each step corresponds to work done by one or more layers. Understanding this flow is critical because it reveals where the 1.6% AI logic actually lives: step 5 (the model call) is the only step where the LLM makes a decision. Steps 1-4 and 6-9 are entirely deterministic.

The loop begins with **Settings resolution** (step 1), which determines the active permission mode, tool availability, and configuration parameters. **State initialization** (step 2) loads the session transcript, prompt history, and any subagent sidechains. **Context assembly** (step 3) builds the context window from 9 ordered sources: system prompt, environment info, CLAUDE.md hierarchy, path-scoped rules, auto-memory, tool metadata, conversation history, tool results, and compact summaries.

Steps 4a through 4e are the **five pre-model context shapers** that run sequentially before every model call, cheapest first. Budget Reduction applies per-message size caps. Snip trims older history (feature-gated via `HISTORY_SNIP`). Microcompact performs cache-aware fine-grained compression. Context Collapse provides read-time virtual projection that is non-destructive. Auto-Compact generates a full model-produced summary as a last resort when all other strategies fail. This graduated approach ensures the least disruptive compression is always tried first.

After the **model call** (step 5), the system dispatches tools through one of **two execution paths**. The `StreamingToolExecutor` begins executing tools as they stream in from the model response, optimizing for latency. The fallback `runTools` path classifies tools as concurrent-safe or exclusive, executing safe tools in parallel and queuing exclusive ones sequentially. The **permission gate** (step 7) then evaluates the action against all 7 safety layers. Only after passing does **tool execution** (step 8) proceed in the Backend layer.

The loop terminates when one of **5 stop conditions** is met: no tool use in the model response, maximum turns exceeded, context overflow, hook intervention (a Stop hook returns a termination signal), or explicit abort by the user. Recovery mechanisms include max output token escalation with up to 3 retries per turn, reactive compaction that fires at most once per turn, and a prompt-too-long handler that tries context-collapse overflow, then reactive compaction, then terminates.

## 7 Safety Layers: Defense in Depth

![Safety Layers](/assets/img/diagrams/dive-into-claude-code-vila/dive-into-claude-code-vila-safety-layers.svg)

The diagram above depicts the 7 independent safety layers that every tool invocation must pass through. The paper's analysis of these layers reveals both the strength and the structural weakness of Claude Code's security architecture. The strength is that any single layer can block an action -- the system is designed so that a failure in one layer does not compromise the others. The weakness is that all layers share performance constraints, creating shared failure modes that undermine the defense-in-depth philosophy.

**Layer 1: Tool pre-filtering** removes blanket-denied tools from the model's view entirely. The model never sees tools it cannot use, which prevents it from even attempting to invoke them. This is the most efficient safety layer because it operates before any runtime evaluation.

**Layer 2: Deny-first rule evaluation** enforces the principle that a broad deny always overrides a narrow allow, even when the allow rule is more specific. This is the philosophical core of the safety system: unrecognized actions are blocked by default, not allowed by default. The strictest rule always wins.

**Layer 3: Permission mode constraints** apply the active mode's baseline handling. The 7 modes form a graduated trust spectrum: `plan` (user approves all plans) -> `default` (standard interactive approval) -> `acceptEdits` (file edits auto-approved) -> `auto` (ML classifier evaluates) -> `dontAsk` (no prompting, deny rules enforced) -> `bypassPermissions` (skips most prompts, safety-critical checks remain) plus the internal `bubble` mode for subagent escalation.

**Layer 4: Auto-mode ML classifier** is implemented in `yoloClassifier.ts`. This is a separate LLM call that evaluates safety independently from the main model. It loads the base system prompt plus permission templates (with separate internal and external variants), then performs a two-stage evaluation: a fast-filter pass followed by chain-of-thought reasoning. It races a pre-computed classification against a timeout to balance thoroughness with latency.

**Layer 5: Shell sandboxing** provides filesystem and network isolation for shell commands. This is the physical enforcement layer -- even if all logical layers fail, sandboxing limits what a malicious command can actually do to the system.

**Layer 6: Non-restoration on resume** ensures that permissions never persist across session boundaries. Trust is always re-established in the current session. This design choice accepts user friction as the cost of maintaining the safety invariant.

**Layer 7: Hook-based interception** allows PreToolUse hooks to modify or block actions before execution. With 27 hook events across 5 categories and 4 execution types (shell, LLM-evaluated, webhook, subagent verifier), this is the most flexible safety layer and the one most directly controlled by users.

The paper identifies a critical finding: **93% of permission prompts are approved without review**, revealing approval fatigue. The response was not more warnings but restructured boundaries -- sandboxing and classifiers that create safe zones for autonomous operation. Additionally, commands exceeding 50 subcommands bypass security analysis entirely to prevent event-loop starvation, meaning the defense-in-depth degrades precisely when it is most needed.

## Extensibility and Subagent Architecture

![Extensibility and Subagents](/assets/img/diagrams/dive-into-claude-code-vila/dive-into-claude-code-vila-extensibility-subagents.svg)

The diagram above illustrates the 4 extension mechanisms, 3 injection points, and subagent delegation architecture that together form Claude Code's extensibility model. The key insight is that not all extensions need to consume context tokens -- the system provides graduated mechanisms at different context costs, allowing users to choose the right level of integration for each customization.

**Hooks** operate at zero context cost. They handle 27 lifecycle events across 5 categories with 4 execution types: shell commands, LLM-evaluated expressions, webhook calls, and subagent verifiers. Hooks can intercept actions before they execute (PreToolUse), after they complete (PostToolUse), or when the session stops (Stop). Because they never enter the context window, hooks are the most scalable extension mechanism.

**Skills** operate at low context cost. Each skill is defined by a SKILL.md file with 15+ YAML frontmatter fields. Skills are injected via the SkillTool meta-tool, which means they only consume context when the model explicitly invokes them. This lazy-loading design prevents skill descriptions from cluttering the context window when they are not needed.

**Plugins** operate at medium context cost. The plugin manifest accepts 10 component types: commands, agents, skills, hooks, MCP servers, LSP servers, output styles, channels, settings, and user config. Plugins are more powerful than skills but consume more context because their component descriptions must be loaded during tool pool assembly.

**MCP Servers** operate at high context cost. They provide external tools via 7 transport types (stdio, SSE, HTTP, WebSocket, SDK, IDE). Each MCP server adds tool schemas to the context window, making them the most expensive extension mechanism but also the most capable -- they can expose entirely new tool surfaces that the core system was not designed for.

The **3 injection points** define where extensions intervene in the agent loop. The **assemble()** point controls what the model sees: CLAUDE.md content, skill descriptions, MCP resources, and hook-injected context. The **model()** point controls what the model can reach: built-in tools, MCP tools, SkillTool, and AgentTool. The **execute()** point controls whether and how an action runs: permission rules, PreToolUse/PostToolUse hooks, and Stop hooks.

The subagent architecture provides **6 built-in types**: Explore, Plan, General-purpose, Claude Code Guide, Verification, and Statusline-setup. Custom agents are defined via `.claude/agents/*.md` with YAML frontmatter supporting tools, model, permissions, hooks, skills, and more. The critical design trade-off is between **SkillTool** and **AgentTool**: SkillTool injects instructions into the current context window (cheap, same window), while AgentTool spawns a new isolated context window (expensive, approximately 7x token cost, but context-safe). Subagent sessions write their own `.jsonl` sidechain files, and only summaries return to the parent -- full history never enters the parent context. Three isolation modes are available: worktree (git worktree for filesystem isolation), remote (remote execution, internal-only), and in-process (shared filesystem, isolated conversation, the default).

## Context and Memory Management

Context is the binding constraint that shapes nearly every other architectural decision in Claude Code. The system assembles context from 9 ordered sources: system prompt, environment info, CLAUDE.md hierarchy, path-scoped rules, auto-memory, tool metadata, conversation history, tool results, and compact summaries. A critical design choice is that CLAUDE.md instructions are delivered as **user context** (probabilistic compliance) rather than system prompt (deterministic enforcement). Permission rules provide the deterministic enforcement layer separately.

The CLAUDE.md hierarchy operates at 4 levels with decreasing scope: Managed (`/etc/claude-code/CLAUDE.md` for system-wide enterprise configuration), User (`~/.claude/CLAUDE.md` for per-user settings), Project (`CLAUDE.md`, `.claude/CLAUDE.md`, `.claude/rules/*.md` for per-project rules), and Local (`CLAUDE.local.md` for personal settings, gitignored). This hierarchy allows different stakeholders to set different policies at different scopes.

The 5-layer compaction pipeline applies graduated lazy-degradation: Budget Reduction (always active, cheapest) -> Snip (feature-gated history trimming) -> Microcompact (cache-aware compression, always active) -> Context Collapse (read-time virtual projection, non-destructive, feature-gated) -> Auto-Compact (full model-generated summary, last resort). Memory is file-based with no vector database -- an LLM-based scan of memory-file headers selects up to 5 relevant files on demand, keeping the system fully inspectable, editable, and version-controllable by the user.

## Design Decisions for Agent Builders

The paper distills 6 design decisions that every coding agent must answer. **Decision 1: Where does reasoning live?** Claude Code places 1.6% in the model and 98.4% in the harness. As frontier models converge in capability (top 3 within 1% on SWE-bench), the operational harness becomes the differentiator, not the model. **Decision 2: What is the safety posture?** Deny-first with 7 independent layers, but shared failure modes degrade defense-in-depth. **Decision 3: How is context managed?** A graduated compaction pipeline outperforms single-pass truncation. Design for context scarcity from day one. **Decision 4: How does extensibility work?** Graduated context-cost mechanisms (hooks=0, skills=low, plugins=medium, MCP=high) scale better than a single unified API. **Decision 5: How do subagents work?** Isolated context with summary-only returns prevents context explosion, at the cost of ~7x tokens per subagent session. **Decision 6: How do sessions persist?** Append-only JSONL favors auditability over query power, and permissions are never restored on resume.

Three recurring meta-patterns emerge across all decisions. **Graduated layering over monolithic mechanisms**: safety, context, and extensibility all use stacked independent stages rather than single solutions. **Append-only designs favoring auditability**: everything can be reconstructed, nothing is destructively edited. **Model judgment within a deterministic harness**: the model decides freely within boundaries enforced by the harness. The 1.6%/98.4% ratio is not accidental -- it is the architectural expression of this principle.

## Key Findings and Security Insights

The paper's security analysis reveals several concerning findings. **4 CVEs** share a common root cause: a pre-trust execution window where hooks and MCP servers execute during initialization before the trust dialog appears, creating a structurally privileged attack window outside the deny-first pipeline. Commands with **50+ subcommands** bypass security analysis entirely to prevent event-loop starvation, meaning complex commands -- precisely those most likely to be dangerous -- receive the least scrutiny. The **93% prompt-approval rate** reveals that users approve nearly all permission prompts without review, indicating that the permission system creates friction without achieving its intended security benefit. The paper's analytical framework traces 5 human values (Human Decision Authority, Safety/Security/Privacy, Reliable Execution, Capability Amplification, Contextual Adaptability) through 13 design principles to specific implementation choices, demonstrating that the architecture is not ad hoc but systematically derived from value commitments.

## Getting Started

To explore the analysis, start with the [GitHub repository](https://github.com/VILA-Lab/Dive-into-Claude-Code) which contains the full paper, architecture documentation, and a design guide for agent builders. The [arXiv paper](https://arxiv.org/abs/2604.14228) provides the complete values-to-implementation framework with cross-system comparisons. For a guided reading path: agent builders should start with the Build Your Own Agent guide, then read the Architecture Deep Dive; security researchers should begin with the Safety and Permissions section; and researchers should read the full paper for the systematic analytical framework.

## Conclusion

The VILA-Lab analysis of Claude Code provides a rare systematic decomposition of a production AI coding agent. The central finding -- that 98.4% of the codebase is deterministic infrastructure -- reframes how we think about AI agent design. The agent loop is simple; the systems around it are complex. For the AI coding agent ecosystem, this means that competitive advantage lies not in the model or the prompt but in the harness: permission systems, context management, recovery logic, and extensibility mechanisms. The paper's values-to-principles-to-implementation framework gives agent builders a vocabulary and a decision map for navigating the same design space that Claude Code's architects navigated. As the authors demonstrate through OpenClaw comparison, cross-cutting integrative mechanisms -- not modular features -- are the true locus of engineering complexity, and the part that resists reimplementation.