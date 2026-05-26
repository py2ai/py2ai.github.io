---
layout: post
title: "ECC - Everything Claude Code: The Complete Agentic Operator System"
description: "Discover ECC, the harness-native operator system for Claude Code, Cursor, Codex, and more. 400+ skills, 61 agents, security auditing, and cross-harness parity."
date: 2026-05-25
header-img: "img/post-bg.jpg"
permalink: /ECC-Everything-Claude-Code-Agentic-Operator-System/
featured-img: ai-coding-frameworks/ai-coding-frameworks
categories: [AI Agents, Developer Tools, Open Source]
tags: [ECC, Claude Code, agentic AI, AI coding, Cursor, Codex, OpenCode, skills, agents, MCP, hooks, developer productivity, automation, cross-harness]
keywords: "ECC Everything Claude Code tutorial, how to set up ECC agentic operator system, Claude Code vs Cursor vs Codex comparison, ECC skills and agents guide, cross-harness AI coding assistant, ECC security scanning AgentShield, open source agentic coding framework, ECC installation and configuration, AI coding assistant operator system, ECC token optimization guide"
author: "PyShine"
---

> **182K+ stars** | **28K+ forks** | **170+ contributors** | **12+ language ecosystems** | **Anthropic Hackathon Winner**

[ECC](https://github.com/affaan-m/ECC) (Everything Claude Code) is not just a configuration pack. It is a complete **agentic operator system** for harness-native work, built by Affaan Mustafa (Anthropic hackathon winner) and evolved over 10+ months of intensive daily use building real products. From 61 specialized agents and 400+ workflow skills to security scanning, continuous learning, and token optimization, ECC provides a production-ready layer that works across every major AI coding harness.

---

## What Makes ECC Different

Most AI coding assistants ship with basic prompts or isolated commands. ECC takes a systems approach: it treats agentic coding as an operational discipline requiring **skills, instincts, memory optimization, continuous learning, security scanning, and research-first development**.

> **Key Insight:** ECC is the first plugin to maximize every major AI coding tool through a unified operator system, not just a collection of prompts.

The result is a plugin architecture that transforms Claude Code, Cursor, Codex, OpenCode, GitHub Copilot, Zed, and Gemini CLI from standalone tools into a unified, orchestrated development environment.

---

## Core Architecture

ECC is organized around a central plugin system with multiple component layers, each designed to be independently installable and composable:

| Component | Count | Purpose |
|-----------|-------|---------|
| **Agents** | 61 | Specialized subagents for delegation (planner, architect, reviewer, security auditor, build resolver) |
| **Skills** | 400+ | Workflow definitions and domain knowledge (TDD, security review, API design, content engine) |
| **Commands** | 76 | Maintained slash-entry compatibility for command-first workflows |
| **Rules** | 110+ | Always-follow guidelines organized by language and concern |
| **Hooks** | 8 event types | Trigger-based automations (session start, pre/post tool use, stop, compaction) |
| **MCP Configs** | 14 servers | Model Context Protocol integrations for external services |
| **Contexts** | Multiple | Prompt injection patterns for dynamic system context |
| **Examples** | Many | Config templates and reference implementations |

![ECC Core Architecture]({{ site.baseurl }}/assets/img/diagrams/ecc/ecc-architecture.svg)

### Understanding the Core Architecture

The architecture diagram above illustrates how ECC Core acts as the central plugin system, distributing capabilities across multiple harnesses and component categories. Let's break down each component and how they interact:

**Harness Layer (Top)**

At the top layer, seven AI coding harnesses feed into ECC Core: Claude Code, Cursor IDE, Codex CLI, OpenCode, GitHub Copilot, Zed, and Gemini CLI. This harness-agnostic design is what makes ECC unique. Instead of building separate configurations for each tool, ECC provides a single source of truth that adapts to each harness through DRY adapter patterns and shared configuration files. When you configure an agent or skill in ECC, it becomes available across all supported harnesses without duplication.

**Agents (61)**

The 61 specialized agents handle delegation tasks that would otherwise consume the main orchestrator's context window. Key agents include the planner (creates implementation blueprints), architect (designs system structure), reviewer (code quality and security), security auditor (OWASP Top 10 scanning), and build resolver (diagnoses compilation and runtime errors). Each agent is defined in AGENTS.md at the repository root, making it readable by Claude Code, Cursor, Codex, and OpenCode simultaneously.

**Skills (400+)**

Skills provide reusable workflow definitions that can be loaded on demand. Unlike static prompts, skills use the SKILL.md format with YAML frontmatter, enabling structured metadata, trigger conditions, and version tracking. Skills range from domain-specific patterns (django-patterns, golang-testing) to cross-cutting concerns (tdd-workflow, security-review, strategic-compact). The on-demand loading mechanism means only relevant skills consume context, keeping the working window efficient.

**Commands (76)**

Commands maintain backward compatibility with slash-entry workflows. These are the `/ecc:plan`, `/code-review`, `/security-scan` style interactions that developers type directly. Commands serve as entry points that often invoke agents and skills in sequence, providing a convenient human-readable interface to the underlying system.

**Rules (110+)**

Rules enforce always-follow guidelines organized by language and concern. TypeScript projects get TypeScript-specific rules, Python projects get Python-specific rules, and so on. Rules are loaded automatically based on the project context, ensuring that every interaction adheres to established best practices without manual configuration.

**Hooks (8 event types)**

Hooks provide event-driven automation that triggers on session lifecycle events: session start, pre-tool use, post-tool use, stop, and compaction. These hooks enable automatic behaviors like security scanning before file writes, cost tracking after API calls, and context compaction at strategic breakpoints.

**Data Flow**

The data flow is straightforward: a harness connects to ECC Core, which loads the appropriate agents, skills, rules, and hooks for the current task. Hooks fire on lifecycle events, agents delegate subtasks, skills provide structured workflows, and rules enforce constraints. The Dashboard GUI provides a visual interface for monitoring and controlling the system. This plugin architecture means teams can install only what they need, upgrade components independently, and maintain consistency across all their AI coding tools.

> **Takeaway:** The harness-agnostic core means you configure once and run everywhere. ECC does not duplicate effort across Claude Code, Cursor, and Codex; it shares the same agents, skills, and rules through adapter patterns.

---

## Cross-Harness Feature Parity

ECC is the first plugin to maximize every major AI coding tool. Here is how each harness compares:

| Feature | Claude Code | Cursor IDE | Codex CLI | OpenCode | GitHub Copilot |
|---------|------------|------------|-----------|----------|----------------|
| **Agents** | 61 | Shared (AGENTS.md) | Shared (AGENTS.md) | 12 | N/A |
| **Skills** | 400+ | Shared | 10 (native) | 37 | Via instructions |
| **Hook Events** | 8 types | 15 types | None yet | 11 types | None |
| **Rules** | 110+ | 110+ (YAML) | Instruction-based | 13 | 1 always-on file |
| **MCP Servers** | 14 | Shared | 7 (auto-merged) | Full | N/A |

![ECC Cross-Harness Parity]({{ site.baseurl }}/assets/img/diagrams/ecc/ecc-cross-harness.svg)

### Understanding Cross-Harness Feature Parity

The cross-harness parity diagram visualizes how ECC distributes features across five major AI coding tools. Reading from top to bottom, each row represents a feature category: Agents, Skills, Hooks, Rules, and MCP Servers. Reading from left to right, each column represents a harness: Claude Code, Cursor, Codex, OpenCode, and GitHub Copilot.

**Claude Code (Full Feature Set)**

Claude Code receives the complete ECC feature set: all 61 agents, 400+ skills, 8 hook event types, 110+ rules, and 14 MCP servers. This is the primary harness for which ECC was originally built, and it serves as the reference implementation. Every feature available in ECC is accessible through Claude Code's native plugin system, hook infrastructure, and settings.json configuration.

**Cursor IDE (Shared Configuration)**

Cursor achieves parity through shared configuration files, particularly AGENTS.md for agent delegation and YAML-based rules. The DRY adapter pattern lets Cursor reuse Claude Code hook scripts without duplication via `.cursor/hooks/adapter.js`. Cursor actually supports 15 hook event types, more than Claude Code's 8, because Cursor's event system exposes additional lifecycle points. Rules are delivered through Cursor's YAML rule format, maintaining the same content as Claude Code rules but in a different syntax.

**Codex CLI (Shared Agents, Limited Hooks)**

Codex receives shared agents and skills through AGENTS.md, with 10 native skills and 7 auto-merged MCP servers. Codex's lack of hooks is a current limitation, compensated by AGENTS.md, optional `model_instructions_file` overrides, and sandbox permissions. The Codex integration focuses on instruction-based rules rather than event-driven hooks, which means some automated behaviors available in Claude Code and Cursor are not yet possible in Codex.

**OpenCode (Growing Integration)**

OpenCode supports 12 agents, 37 skills, 11 hook types, 13 rules, and full MCP integration. This represents a rapidly maturing integration that covers most of ECC's capabilities. OpenCode's hook system is particularly robust, supporting 11 event types that enable similar automation patterns to Claude Code.

**GitHub Copilot (Minimal Integration)**

GitHub Copilot has the most limited integration, with rules delivered via a single always-on instruction file and skills accessed through instructions. This is not a limitation of ECC but rather of Copilot's current extension model, which does not support custom agents, hooks, or MCP servers. As Copilot's extension API evolves, ECC's integration depth will increase accordingly.

**Key Architectural Decisions**

AGENTS.md at the repository root serves as the universal cross-tool file, read by Claude Code, Cursor, Codex, and OpenCode. The SKILL.md format with YAML frontmatter works across Claude Code, Codex, and OpenCode. The color coding in the diagram tells an important story: green cells indicate full or shared feature availability, while gray cells indicate N/A or limited support.

> **Amazing:** A single AGENTS.md file at your repository root is read by Claude Code, Cursor, Codex, and OpenCode simultaneously. One file, four harnesses, zero duplication.

---

## Ecosystem Tools

Beyond the core plugin, ECC ships with three major ecosystem tools that extend its capabilities into security, learning, and skill generation.

### AgentShield - Security Auditor

Built at the Claude Code Hackathon (Cerebral Valley x Anthropic, February 2026), AgentShield runs **1282 tests with 98% coverage** across 102 static analysis rules. It scans CLAUDE.md, settings.json, MCP configs, hooks, agent definitions, and skills across five categories:

- Secrets detection (14 patterns including `sk-`, `ghp_`, `AKIA`)
- Permission auditing
- Hook injection analysis
- MCP server risk profiling
- Agent config review

The `--opus` flag runs three Claude Opus 4.6 agents in a red-team/blue-team/auditor pipeline for adversarial reasoning, not just pattern matching. Output formats include terminal (color-graded A-F), JSON (CI pipelines), Markdown, and HTML.

### Continuous Learning v2

The instinct-based learning system automatically extracts patterns from your coding sessions:

```bash
/instinct-status        # Show learned instincts with confidence scores
/instinct-import <file> # Import instincts from teammates
/instinct-export        # Export your instincts for sharing
/evolve                 # Cluster related instincts into reusable skills
```

Unlike static prompt libraries, Continuous Learning v2 observes how you actually work, builds confidence scores for each pattern, and evolves them into shareable skills over time.

### Skill Creator

Two ways to generate Claude Code skills from your repository:

- **Local Analysis**: `/skill-create` analyzes git history locally and generates SKILL.md files
- **GitHub App**: Advanced features for 10k+ commit repositories with auto-PRs and team sharing

![ECC Ecosystem]({{ site.baseurl }}/assets/img/diagrams/ecc/ecc-ecosystem.svg)

### Understanding the ECC Ecosystem

The ecosystem diagram maps the major tools and capabilities that orbit the ECC v2.0.0-rc.1 core. At the center sits ECC itself, with five primary ecosystem branches extending outward: AgentShield, Continuous Learning v2, Skill Creator, Language Ecosystems, and Token Optimization.

**AgentShield (Security Branch)**

AgentShield branches into three sub-components: Secrets Detection and Permission Audit, Auto-Fix with CVE Database and Risk Grading A-F, and Hook Injection analysis. These sub-components represent the depth of security coverage that AgentShield provides, going far beyond simple pattern matching to include adversarial reasoning and automated remediation.

The Secrets Detection module scans for 14 known secret patterns including AWS access keys (`AKIA`), GitHub personal access tokens (`ghp_`), and OpenAI API keys (`sk-`). The Permission Audit module checks for overly broad file permissions, dangerous command patterns, and privilege escalation vectors. The Auto-Fix module connects to a CVE database and provides risk-graded remediation suggestions from A (critical) to F (informational). The Hook Injection analysis examines hook scripts for supply chain attack vectors, ensuring that the very automation hooks that make ECC powerful cannot be weaponized against you.

**Continuous Learning v2 (Learning Branch)**

Continuous Learning v2 branches into Pattern Extraction with Confidence Scoring and Import/Export capabilities, Auto-Clustering with Skill Generation and Session Analysis. This represents the full lifecycle of institutional knowledge capture: from observing patterns, to scoring their reliability, to clustering them into reusable skills, to sharing them across teams.

The Pattern Extraction module watches your coding sessions and identifies recurring patterns: how you structure error handling, which debugging approaches you prefer, what naming conventions you follow. Each pattern receives a confidence score from 0.0 to 1.0, reflecting how consistently you apply it. Patterns with high confidence scores (above 0.8) are candidates for evolution into skills. The Import/Export system enables team-wide knowledge sharing: export your instincts to a file, share it with teammates, and they can import your learned patterns directly into their ECC instance.

The Auto-Clustering module groups related instincts into coherent skill definitions. When you run `/evolve`, ECC analyzes all instincts above the confidence threshold, identifies clusters of related patterns, and generates SKILL.md files that capture the collective knowledge. Session Analysis provides a retrospective view of what patterns were observed, which were adopted, and which were rejected, giving teams visibility into how their coding practices evolve over time.

**Skill Creator (Generation Branch)**

The Skill Creator branches into Local Analysis for SKILL.md Generation and the GitHub App for Auto-PRs and Team Sharing with 10k+ Commits support. This dual approach serves both individual developers working locally and enterprise teams managing large repositories.

Local Analysis runs `/skill-create` which examines your git history, identifies recurring patterns in your codebase, and generates SKILL.md files that codify your team's best practices. The GitHub App extends this capability for large repositories (10k+ commits) by running analysis in CI/CD, automatically creating pull requests with generated skills, and enabling team-wide review and adoption of new skills.

**Language Ecosystems (Language Branch)**

Language Ecosystems branch into TypeScript, Python, Go, Java/Kotlin, Rust/C++, Swift, and PHP/Perl. Each language receives dedicated rules, agents, and skills tailored to its ecosystem. TypeScript gets typescript-reviewer agent and frontend-patterns skill. Python gets python-reviewer and django-patterns. Go gets go-reviewer and golang-testing. This specialization ensures that ECC's guidance is contextually relevant rather than generically applicable.

**Token Optimization (Cost Branch)**

Token Optimization branches into Model Routing and Context Compaction, representing ECC's built-in cost management capabilities. Model Routing automatically selects the appropriate model (Sonnet for routine tasks, Opus for complex analysis) to minimize cost while maintaining quality. Context Compaction uses the strategic-compact skill to suggest `/compact` at logical breakpoints rather than waiting for auto-compaction at 95% context, preserving more relevant context while reducing token usage.

> **Important:** Each MCP tool description consumes tokens from your 200k window, potentially reducing it to ~70k. ECC recommends keeping under 10 MCPs enabled and under 80 tools active.

---

## Token Optimization

Claude Code usage can be expensive without token management. ECC provides battle-tested settings that significantly reduce costs:

| Setting | Default | Recommended | Impact |
|---------|---------|-------------|--------|
| `model` | opus | **sonnet** | ~60% cost reduction |
| `MAX_THINKING_TOKENS` | 31,999 | **10,000** | ~70% reduction in hidden thinking cost |
| `CLAUDE_AUTOCOMPACT_PCT_OVERRIDE` | 95 | **50** | Compacts earlier for better quality |

The `strategic-compact` skill suggests `/compact` at logical breakpoints (after research, before implementation, after milestones) instead of relying on auto-compaction at 95% context.

**Critical context window advice:** Your 200k context window before compacting might only be 70k with too many tools enabled. Performance degrades significantly. Have 20-30 MCPs in config, but keep under 10 enabled and under 80 tools active.

---

## What's New in v2.0.0-rc.1

The latest release adds the public Hermes operator story on top of the reusable layer:

- **Dashboard GUI** - Tkinter-based desktop application with dark/light theme toggle, font customization, and project logo
- **Operator workflows** - `brand-voice`, `social-graph-ranker`, `customer-billing-ops`, `google-workspace-ops`
- **Media tooling** - `manim-video`, `remotion-video-creation` for technical explainers
- **Ito prediction-market skill pack** - Public market/basket workflows with gated live API access
- **Optimization skill pack** - `parallel-execution-optimizer`, `benchmark-optimization-loop`, `latency-critical-systems`
- **ECC 2.0 alpha** - Rust control-plane prototype in `ecc2/` with `dashboard`, `start`, `sessions`, `status`, `stop`, `resume`, and `daemon` commands
- **Operator status snapshots** - `ecc status --markdown --write status.md` turns local state into portable handoffs
- **Package manager** - yarn@4.9.2 with Node.js >=18 required
- **Dependencies** - @iarna/toml, ajv, sql.js for configuration validation and local database support
- **Python dashboard** - `ecc_dashboard.py` provides an alternative to the Tkinter GUI
- **9 language translations** - Documentation available in de-DE, ja-JP, ko-KR, pt-BR, ru, tr, vi-VN, zh-CN, zh-TW

---

## Installation

### Plugin Install (Recommended)

```bash
# Add marketplace
/plugin marketplace add https://github.com/affaan-m/ECC

# Install plugin
/plugin install ecc@ecc
```

### Manual Install

```bash
git clone https://github.com/affaan-m/ECC.git
cd ECC

# Copy rules (plugin cannot distribute rules automatically)
mkdir -p ~/.claude/rules/ecc
cp -R rules/common ~/.claude/rules/ecc/
cp -R rules/typescript ~/.claude/rules/ecc/   # pick your stack
```

**Important:** Do not stack install methods. The most common broken setup is `/plugin install` followed by `./install.sh --profile full`. Choose one path.

---

## Usage

### Common Workflows

**Starting a new feature:**

```bash
/ecc:plan "Add user authentication with OAuth"  # planner creates blueprint
tdd-workflow skill                                # tdd-guide enforces tests-first
/code-review                                      # code-reviewer checks work
```

**Fixing a bug:**

```bash
tdd-workflow skill  # write failing test that reproduces it
                    # implement fix, verify test passes
/code-review        # catch regressions
```

**Preparing for production:**

```bash
/security-scan     # OWASP Top 10 audit
e2e-testing skill  # critical user flow tests
/test-coverage     # verify 80%+ coverage
```

### Key Commands and Skills

ECC provides 76 slash commands and 400+ skills. Key commands include:

```bash
/ecc:plan <feature>     # Create implementation blueprint
/code-review            # Run quality and security review
/security-scan          # AgentShield security audit
/test-coverage          # Verify test coverage thresholds
/compact                # Manual context compaction
/statusline             # Customize terminal status line
```

Key skills include:

```bash
tdd-workflow           # Test-driven development workflow
security-review        # Security audit checklist
frontend-patterns      # UI/UX implementation patterns
backend-patterns       # API and service design patterns
deployment-patterns    # Production deployment checklist
strategic-compact      # Context compaction optimization
```

### Dashboard Usage

ECC provides two dashboard options:

```bash
# Tkinter GUI dashboard
npm run dashboard

# Or directly
python3 ./ecc_dashboard.py
```

The dashboard provides operator readiness monitoring, session inspection, and worktree orchestration. For CI/CD integration, use the Node.js operator readiness dashboard:

```bash
npm run operator:dashboard
```

---

## Language Ecosystems

ECC supports **12+ language ecosystems** with dedicated rules, agents, and skills:

| Language | Rules | Agents | Skills |
|----------|-------|--------|--------|
| TypeScript/JavaScript | Yes | typescript-reviewer | frontend-patterns, api-design |
| Python | Yes | python-reviewer | django-patterns, fastapi-patterns |
| Go | Yes | go-reviewer | golang-patterns, golang-testing |
| Java / Kotlin | Yes | java-reviewer, kotlin-reviewer | springboot-patterns, quarkus-patterns |
| Rust | Yes | rust-reviewer | rust-patterns, rust-testing |
| C++ | Yes | cpp-reviewer | cpp-coding-standards, cpp-testing |
| Swift | Yes | - | swift-actor-persistence, swift-concurrency-6-2 |
| PHP / Perl | Yes | - | laravel-patterns, perl-patterns |
| C# / F# | Yes | - | csharp-testing, fsharp-testing |
| Dart / Flutter | Yes | - | dart-flutter-patterns |
| HarmonyOS / ArkTS | Yes | harmonyos-app-resolver | compose-multiplatform-patterns |
| Shell / DevOps | Yes | - | docker-patterns, deployment-patterns |

---

## Troubleshooting

### Context Window Overflow

**Symptom:** "Context too long" errors or incomplete responses

**Solutions:**

```bash
# Clear conversation history and start fresh
# Use Claude Code: "New Chat" or Cmd/Ctrl+Shift+N

# Reduce file size before analysis
head -n 100 large-file.log > sample.log

# Split tasks into smaller chunks
# Instead of: "Analyze all 50 files"
# Use: "Analyze files in src/components/ directory"
```

### Hooks Not Firing

**Symptom:** Pre/post hooks don't execute

**Causes:** Hooks not registered in settings.json, invalid hook syntax, or hook script not executable

**Solutions:**

```bash
# Check hooks are registered
grep -A 10 '"hooks"' ~/.claude/settings.json

# Verify hook files exist and are executable
ls -la ~/.claude/plugins/cache/*/hooks/

# Test hook manually
bash ~/.claude/plugins/cache/*/hooks/pre-bash.sh <<< '{"command":"echo test"}'
```

### Hook Runtime Controls

If a hook blocks legitimate commands or causes issues:

```bash
# Disable hook temporarily
# Edit ~/.claude/settings.json and remove the problematic hook

# Wrap dev servers in tmux to avoid false positives
tmux new-session -d -s dev "npm run dev"
tmux attach -t dev
```

### Token Budget Issues

**Symptom:** Excessive API costs or rapid context compaction

**Solutions:**

```bash
# Switch to Sonnet for 90% of coding tasks
# Reserve Opus for complex architecture and security analysis

# Reduce MAX_THINKING_TOKENS
export MAX_THINKING_TOKENS=10000

# Disable unused MCPs
# Navigate to /plugins and disable everything unused

# Use the strategic-compact skill
/skill strategic-compact
```

### Agent Not Found

**Symptom:** "Agent not loaded" or "Unknown agent" errors

**Solutions:**

```bash
# Check plugin installation
ls ~/.claude/plugins/cache/

# Verify agent exists
ls ~/.claude/plugins/cache/*/agents/

# Run ECC diagnostics
ecc doctor
ecc repair

# Reload plugin
# Claude Code -> Settings -> Extensions -> Reload
```

### Python/Node Version Mismatches

**Symptom:** "python3 not found" or "node: command not found"

**Solutions:**

```bash
# Verify installations
python3 --version  # Requires Python 3
node --version     # Requires Node.js >=18
npm --version

# ECC requires Node.js >=18
# Install from nodejs.org if needed
```

### Getting Help

If issues persist:

1. **Check GitHub Issues**: [github.com/affaan-m/ECC/issues](https://github.com/affaan-m/ECC/issues)
2. **Enable Debug Logging**:
   ```bash
   export CLAUDE_DEBUG=1
   export CLAUDE_LOG_LEVEL=debug
   ```
3. **Collect Diagnostic Info**:
   ```bash
   claude --version
   node --version
   python3 --version
   echo $CLAUDE_PACKAGE_MANAGER
   ls -la ~/.claude/plugins/cache/
   ```
4. **Open an Issue**: Include debug logs, error messages, and diagnostic info

---

## Conclusion

ECC represents a maturation of the AI coding assistant ecosystem. Rather than treating each harness as an isolated tool, it provides a **unified operator system** that brings consistency, security, and continuous improvement across Claude Code, Cursor, Codex, OpenCode, GitHub Copilot, Zed, and Gemini.

> **Key Insight:** With 61 agents, 400+ skills, 110+ rules, 14 MCP servers, security auditing with 1282 tests, instinct-based learning, and cross-harness parity, ECC is designed for teams that treat AI-assisted development as a core operational capability, not an experimental add-on.

Whether you are a solo developer looking to optimize your Claude Code workflow or an enterprise team standardizing across multiple AI coding tools, ECC provides the infrastructure to make agentic coding reliable, secure, and cost-effective.

---

## Links

- [GitHub Repository](https://github.com/affaan-m/ECC)
- [GitHub App](https://github.com/marketplace/ecc-tools)
- [npm Package](https://www.npmjs.com/package/ecc-universal)
- [Shorthand Guide](https://x.com/affaanmustafa/status/2012378465664745795)
- [Longform Guide](https://x.com/affaanmustafa/status/2014040193557471352)
- [Security Guide](https://x.com/affaanmustafa/status/2033263813387223421)
