---
layout: post
title: "ECC - The Everything Claude Code Agentic Operator System"
date: 2026-05-25 00:00:00 +0800
categories: [agentic-ai, claude-code, developer-tools, productivity]
tags: [ecc, claude-code, agentic-ai, ai-coding, cursor, codex, opencode, skills, agents]
permalink: /ECC-Everything-Claude-Code-Agentic-Operator-System/
---

> **182K+ stars** | **28K+ forks** | **170+ contributors** | **12+ language ecosystems** | **Anthropic Hackathon Winner**

[ECC](https://github.com/affaan-m/ECC) (Everything Claude Code) is not just a configuration pack. It is a complete **harness-native operator system** for agentic work, built by an Anthropic hackathon winner and evolved over 10+ months of intensive daily use building real products. From 61 specialized agents and 246 workflow skills to security scanning, continuous learning, and token optimization, ECC provides a production-ready layer that works across every major AI coding harness.

---

## What Makes ECC Different

Most AI coding assistants ship with basic prompts or isolated commands. ECC takes a systems approach: it treats agentic coding as an operational discipline requiring **skills, instincts, memory optimization, continuous learning, security scanning, and research-first development**.

The result is a plugin architecture that transforms Claude Code, Cursor, Codex, OpenCode, GitHub Copilot, Zed, and Gemini CLI from standalone tools into a unified, orchestrated development environment.

---

## Core Architecture

ECC is organized around five primary component layers, each designed to be independently installable and composable:

| Component | Count | Purpose |
|-----------|-------|---------|
| **Agents** | 61 | Specialized subagents for delegation (planner, architect, reviewer, security auditor, build resolver) |
| **Skills** | 246 | Workflow definitions and domain knowledge (TDD, security review, API design, content engine) |
| **Commands** | 76 | Maintained slash-entry compatibility for command-first workflows |
| **Rules** | 34 | Always-follow guidelines organized by language (common, TypeScript, Python, Go, Java, Swift, PHP) |
| **Hooks** | 8 event types | Trigger-based automations (session start, pre/post tool use, stop, compaction) |

![ECC Core Architecture]({{ site.baseurl }}/assets/img/diagrams/ecc/ecc-architecture.svg)

The architecture is **harness-agnostic at the core**. The same ECC installation can power Claude Code natively while simultaneously providing adapted configs for Cursor, Codex, OpenCode, GitHub Copilot, Zed, and Gemini CLI.

---

## Cross-Harness Feature Parity

ECC is the first plugin to maximize every major AI coding tool. Here is how each harness compares:

| Feature | Claude Code | Cursor IDE | Codex CLI | OpenCode | GitHub Copilot |
|---------|------------|------------|-----------|----------|----------------|
| **Agents** | 61 | Shared (AGENTS.md) | Shared (AGENTS.md) | 12 | N/A |
| **Skills** | 246 | Shared | 10 (native) | 37 | Via instructions |
| **Hook Events** | 8 types | 15 types | None yet | 11 types | None |
| **Rules** | 34 | 34 (YAML) | Instruction-based | 13 | 1 always-on file |
| **MCP Servers** | 14 | Shared | 7 (auto-merged) | Full | N/A |

![ECC Cross-Harness Parity]({{ site.baseurl }}/assets/img/diagrams/ecc/ecc-cross-harness.svg)

Key architectural decisions enable this parity:

- **AGENTS.md** at the repository root is the universal cross-tool file, read by Claude Code, Cursor, Codex, and OpenCode
- **DRY adapter pattern** lets Cursor reuse Claude Code hook scripts without duplication via `.cursor/hooks/adapter.js`
- **Skills format** (SKILL.md with YAML frontmatter) works across Claude Code, Codex, and OpenCode
- Codex's lack of hooks is compensated by `AGENTS.md`, optional `model_instructions_file` overrides, and sandbox permissions

---

## Ecosystem Tools

Beyond the core plugin, ECC ships with three major ecosystem tools:

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

---

## Token Optimization

Claude Code usage can be expensive without token management. ECC provides battle-tested settings that significantly reduce costs:

| Setting | Default | Recommended | Impact |
|---------|---------|-------------|--------|
| `model` | opus | **sonnet** | ~60% cost reduction |
| `MAX_THINKING_TOKENS` | 31,999 | **10,000** | ~70% reduction in hidden thinking cost |
| `CLAUDE_AUTOCOMPACT_PCT_OVERRIDE` | 95 | **50** | Compacts earlier for better quality |

The `strategic-compact` skill suggests `/compact` at logical breakpoints (after research, before implementation, after milestones) instead of relying on auto-compaction at 95% context.

**Critical context window advice**: Each MCP tool description consumes tokens from your 200k window, potentially reducing it to ~70k. ECC recommends keeping under 10 MCPs enabled and under 80 tools active.

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

**Important**: Do not stack install methods. The most common broken setup is `/plugin install` followed by `./install.sh --profile full`. Choose one path.

---

## Language Ecosystems

ECC supports **12+ language ecosystems** with dedicated rules, agents, and skills:

| Language | Rules | Agents | Skills |
|----------|-------|--------|--------|
| TypeScript/JavaScript | Yes | typescript-reviewer | frontend-patterns, nextjs-turbopack |
| Python | Yes | python-reviewer | django-patterns, pytorch-patterns |
| Go | Yes | go-reviewer | golang-patterns, golang-testing |
| Java / Kotlin | Yes | java-reviewer, kotlin-reviewer | springboot-patterns, quarkus-patterns |
| Rust | Yes | rust-reviewer | rust-build-resolver |
| C++ | Yes | cpp-reviewer | cpp-coding-standards |
| Swift | Yes | - | swift-actor-persistence, swift-concurrency-6-2 |
| PHP / Perl | Yes | - | laravel-patterns, perl-patterns |
| HarmonyOS / ArkTS | Yes | harmonyos-app-resolver | - |

---

## Common Workflows

**Starting a new feature:**
```
/ecc:plan "Add user authentication with OAuth"  → planner creates blueprint
tdd-workflow skill                                → tdd-guide enforces tests-first
/code-review                                      → code-reviewer checks work
```

**Fixing a bug:**
```
tdd-workflow skill  → write failing test that reproduces it
                    → implement fix, verify test passes
/code-review        → catch regressions
```

**Preparing for production:**
```
/security-scan     → OWASP Top 10 audit
e2e-testing skill  → critical user flow tests
/test-coverage     → verify 80%+ coverage
```

---

## Community and Ecosystem

Projects built on or inspired by ECC:

| Project | Description |
|---------|-------------|
| [EVC](https://github.com/SaigonXIII/evc) | Marketing agent workspace with 42 commands for content operators |
| [trading-skills](https://github.com/VictorVVedtion/trading-skills) | 68 trading-themed Claude Code skills with pre-trade review prompts |

---

## Conclusion

ECC represents a maturation of the AI coding assistant ecosystem. Rather than treating each harness as an isolated tool, it provides a **unified operator system** that brings consistency, security, and continuous improvement across Claude Code, Cursor, Codex, OpenCode, GitHub Copilot, Zed, and Gemini.

With 61 agents, 246 skills, security auditing, instinct-based learning, and cross-harness parity, ECC is designed for teams that treat AI-assisted development as a core operational capability, not an experimental add-on.

**Links:**
- [GitHub Repository](https://github.com/affaan-m/ECC)
- [GitHub App](https://github.com/marketplace/ecc-tools)
- [npm Package](https://www.npmjs.com/package/ecc-universal)
- [Shorthand Guide](https://x.com/affaanmustafa/status/2012378465664745795)
- [Longform Guide](https://x.com/affaanmustafa/status/2014040193557471352)
- [Security Guide](https://x.com/affaanmustafa/status/2033263813387223421)
