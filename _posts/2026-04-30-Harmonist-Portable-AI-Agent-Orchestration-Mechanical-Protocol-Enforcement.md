---
layout: post
title: "Harmonist: Portable AI Agent Orchestration with Mechanical Protocol Enforcement"
description: "Learn how Harmonist enforces engineering protocols as mechanical gates, not polite prompts. With 186 curated domain specialists, supply-chain verification, and schema-validated memory, it ensures AI agents cannot skip review steps."
date: 2026-04-30
header-img: "img/post-bg.jpg"
permalink: /Harmonist-Portable-AI-Agent-Orchestration-Mechanical-Protocol-Enforcement/
featured-img: ai-coding-frameworks/ai-coding-frameworks
categories: [AI Agents, Developer Tools, Open Source]
tags: [Harmonist, AI agent orchestration, mechanical enforcement, multi-agent framework, Cursor, Claude Code, Copilot, supply chain, protocol enforcement, open source]
keywords: "how to use Harmonist AI agent framework, Harmonist vs LangChain comparison, mechanical protocol enforcement AI agents, 186 agent catalogue orchestration, AI agent supply chain verification, Harmonist installation guide, Cursor AI agent orchestration, Claude Code multi-agent, AI coding assistant review gates, structured validated memory AI"
author: "PyShine"
---

# Harmonist: Portable AI Agent Orchestration with Mechanical Protocol Enforcement

Harmonist is a drop-in multi-agent framework for AI coding assistants that enforces engineering protocols as mechanical gates, not polite prompts. Built by GammaLab, it provides 186 curated domain specialists across 16 categories, supply-chain verification of agent definitions, and schema-validated memory with secret-pattern scanning. Unlike thin agent frameworks that leave enforcement to the prompt, Harmonist's `stop` hook refuses to let a turn complete if required reviewers didn't run, memory wasn't updated, or the protocol wasn't satisfied. The AI literally cannot ship a code change that skipped review.

## Architecture Overview

![Harmonist Architecture](/assets/img/diagrams/harmonist/harmonist-architecture.svg)

### Understanding the Harmonist Architecture

The architecture diagram above illustrates how Harmonist orchestrates 186 domain specialists through a data-driven routing system with mechanical enforcement. The system is built around five key components that work together to ensure protocol compliance.

**AGENTS.md Orchestrator**

The orchestrator is defined in a single `AGENTS.md` file that specifies the protocol, hook phases, invariants, and memory configuration. This file is the single source of truth for how agents interact in a project. It declares which domains and roles are active, which agents are mandatory gates, and how memory should be structured.

**agents/index.json Routing Table**

The routing table contains all 186 agent entries, organized by category and tagged with domains, roles, and disambiguation metadata. The orchestrator never hard-codes agent slugs. Instead, it extracts task tags from the current work, intersects them with the index, filters by the project's declared domains and roles, and picks the right specialist. This data-driven approach means adding a new agent is as simple as adding an entry to the index.

**Three Agent Categories**

Agents fall into two protocol tiers. Orchestration agents (scout, repo-map) and review agents (security, QA, SRE, performance, regression, accessibility) operate under `protocol: strict` -- they are mandatory gates that must run before code changes can ship. Persona agents (engineering, design, marketing, finance, and 11 more categories) operate under `protocol: persona` -- they are free-form domain specialists that provide depth in their area.

**Review Gates**

The review gates are the enforcement mechanism. When a session touches any file outside ignored patterns, the `stop` hook verifies that at least one review agent was invoked, that `qa-verifier` specifically ran, and that `session-handoff.md` was updated. If any check fails, the hook returns a `followup_message` telling the AI exactly what is missing.

**Persistent Memory**

Between sessions, state, decisions, and patterns live under `.cursor/memory/`, linked by correlation IDs that the LLM cannot forge. The next session reads the last three state snapshots and three decisions before planning, providing continuity across sessions.

## Mechanical Enforcement via Hooks

![Hook Enforcement](/assets/img/diagrams/harmonist/harmonist-hook-enforcement.svg)

### Understanding the Hook Enforcement Flow

The hook enforcement diagram above shows the five phases of Harmonist's enforcement lifecycle and how the `stop` gate works as a mechanical check that the AI cannot bypass.

**Five Hook Phases**

Every code-changing turn in Harmonist passes through five hook phases, each with a specific responsibility:

1. **sessionStart**: Bootstraps the correlation_id for the session, injects the last 3 state and decision memory entries, and warns about prior incidents. This gives the AI context from previous sessions before it starts working.

2. **afterFileEdit**: Records every file write to the session state. This is how the `stop` gate knows which files were touched during the session, which determines whether review requirements apply.

3. **subagentStart**: Parses the `AGENT: <slug>` marker from the dispatch, credits the reviewer, and enforces `readonly` capability scoping. Review agents are scoped to read-only access -- they cannot make changes, only verify.

4. **subagentStop**: Records the verdict from each subagent and updates telemetry. This is how the system tracks which reviewers have completed their work.

5. **stop (THE GATE)**: The critical enforcement point. Verifies that reviewers ran, memory was updated, and the protocol was satisfied. If any check fails, returns a `followup_message` telling the AI exactly what is missing. `loop_limit: 3` caps retries. On exhaustion, the incident is persisted and surfaced in the next session.

**The Stop Gate in Detail**

If the session touched any file outside ignored patterns, the stop gate checks three conditions: (1) at least one `category: review` agent was invoked via Task, (2) specifically `qa-verifier` was invoked, and (3) `.cursor/memory/session-handoff.md` was updated during the session. If any check fails, the turn does not complete. The model cannot argue with a state machine on disk.

## Supply Chain Verification and Memory

![Supply Chain and Memory](/assets/img/diagrams/harmonist/harmonist-supply-chain-memory.svg)

### Understanding Supply Chain Verification and Structured Memory

The diagram above shows two critical security features: supply-chain verification of agent definitions and schema-validated memory with secret-pattern scanning.

**Supply Chain Verification**

Every shipped file in Harmonist is hashed in `MANIFEST.sha256`. When `upgrade.py` runs, it SHA-verifies each source file before copying it into the project. If a file has been tampered with -- for example, a `security-reviewer.md` that returns `approve` for everything -- it is REFUSED and never enters the project. This is the first open-source agent catalogue with paranoid-level supply-chain posture. The `install_extras.py` script inherits the same guard for on-demand specialist installs.

**Schema-Validated Memory**

`memory.py append` is the only supported write path for memory entries. It validates every entry against a YAML schema (`memory/SCHEMA.md`), rejects duplicates, and scans the body for approximately 30 classes of secrets: AWS access keys, GitHub PATs, Stripe tokens, Slack webhooks, GCP service accounts, Azure connection strings, Telegram bot tokens, Discord tokens, Heroku/Postmark UUIDs, generic high-entropy tokens with `secret:` prefixes, and database connection strings with embedded credentials. Placeholder fences (`${VAR}`, `<NAME>`) suppress the scan so templates still write cleanly.

**Correlation IDs**

Every memory entry has a `correlation_id` of the form `<session_id>-<task_seq>` generated by the hooks at session start. The ID format (`<unix-seconds><pid4>`) is collision-safe across parallel sessions. The LLM reads the active ID via CLI but never writes the ID itself. This means the linkage between a state entry, a decision, and a pattern from the same task is cryptographically ordered from the hook's perspective, not trusted to the model.

## The 186-Agent Catalogue

![Agent Catalogue](/assets/img/diagrams/harmonist/harmonist-agent-catalogue.svg)

### Understanding the Agent Catalogue

The catalogue diagram above shows the 16 categories of domain specialists available in Harmonist. Each agent carries structured frontmatter with `description`, `tags`, `domains`, `distinguishes_from` (near-peers), `disambiguation` (one-line "when to pick this over X"), `version`, and `updated_at`. The orchestrator reads all of this for tie-breaking when multiple candidates match a task's tags.

**Strict Protocol Agents (8 total)**

The orchestration and review categories operate under `protocol: strict`. These are mandatory gates:

| Category | Count | Agents |
|----------|-------|--------|
| Orchestration | 2 | scout, repo-map |
| Review | 6 | security-reviewer, qa-verifier, sre-reviewer, perf-reviewer, regression-tester, a11y-reviewer |

**Persona Protocol Agents (178 total)**

The remaining 14 categories operate under `protocol: persona` and provide domain depth:

| Category | Count | Focus |
|----------|-------|-------|
| Engineering | 46 | Backend, frontend, DevOps, data, AI, embedded, Solidity, LLM eval |
| Design | 8 | UI/UX, brand, accessibility, visual storytelling |
| Testing | 8 | QA, performance, API testing, evidence collection |
| Product | 5 | Product management, sprints, feedback, trends |
| Project Management | 7 | Planning, studio production, coordination |
| Marketing | 30 | Growth, SEO, content, social, Douyin/WeChat/Xiaohongshu |
| Paid Media | 7 | PPC, tracking, campaign audits |
| Sales | 8 | Outbound, deals, discovery, proposals |
| Finance | 6 | FP&A, bookkeeping, tax, investments |
| Support | 5 | Customer support, compliance, analytics |
| Academic | 5 | Research, psychology, history, anthropology |
| Game Development | 20 | Unity, Unreal, Godot, Roblox, Blender |
| Spatial Computing | 6 | visionOS, WebXR, Metal, XR interaction |
| Specialized | 17 | Blockchain audit, MCP builder, Salesforce, ZK, niche |

## Installation

### Option 1: Integrate via Cursor (Recommended)

```bash
# 1. Clone into your project root
cd your-project/
git clone https://github.com/GammaLabTechnologies/harmonist.git

# 2. Open the project in Cursor, switch to Agent mode
# 3. Paste the contents of harmonist/integration-prompt.md
# 4. Follow the AI's walkthrough -- it will ask about your
#    project's domain and roles, then wire everything up.
# 5. Start a NEW chat when integration is done.
```

The AI reads `harmonist/agents/index.json`, picks the right specialists for your stack, writes a domain-specific `AGENTS.md`, bootstraps `.cursor/memory/`, installs the enforcement hooks, and records the integration state in `.cursor/pack-version.json`.

### Option 2: Integrate via CLI (No Cursor Needed)

```bash
cd your-project/
git clone https://github.com/GammaLabTechnologies/harmonist.git
python3 harmonist/agents/scripts/integrate.py --pack harmonist --project .
```

### Option 3: Manual Integration

See the [GUIDE_EN.md](https://github.com/GammaLabTechnologies/harmonist/blob/main/GUIDE_EN.md) for the step-by-step manual path.

## Requirements

| Requirement | Details |
|-------------|---------|
| Python | 3.9+ (stdlib only, no third-party dependencies) |
| Bash | 3.2+ for POSIX paths (macOS default works) |
| Windows | Pure-Python `hook_runner.py` handles native Windows; no WSL needed |
| Git | For version tracking |
| AI Assistant | Cursor (primary), Claude Code, Copilot, Windsurf, Aider, Kimi, Qwen, Gemini CLI, OpenCode, OpenClaw, Antigravity |

## Troubleshooting

| Issue | Solution |
|-------|----------|
| **Stop hook blocks every turn** | Check that `qa-verifier` was invoked and `session-handoff.md` was updated |
| **Agent not found in routing** | Verify the agent's tags match your project's declared domains and roles in `AGENTS.md` |
| **Memory validation errors** | Ensure entries match the schema in `memory/SCHEMA.md`; use `memory.py append` as the only write path |
| **Supply chain verification failure** | Re-clone the repository; `MANIFEST.sha256` hashes must match the shipped files |
| **Windows hook issues** | Use `hook_runner.py` instead of bash scripts; both paths are tested for parity |
| **Loop limit exhausted** | Check `.cursor/hooks/.state/incidents.json` for the specific failure; address the missing requirement |

## Conclusion

Harmonist addresses a fundamental gap in AI coding frameworks: the inability to enforce engineering protocols mechanically. While other frameworks leave enforcement to prompts that the model can silently skip, Harmonist implements protocol enforcement as IDE-level hooks that observe every subagent dispatch, file edit, and session stop. When the rules aren't met, the turn doesn't complete. With 186 curated domain specialists, supply-chain verification of agent definitions, schema-validated memory with secret scanning, and zero runtime dependencies (pure Python stdlib + bash), Harmonist provides the governance layer that serious engineering workflows require -- without the infrastructure overhead of enterprise platforms.

**Links:**
- GitHub Repository: [https://github.com/GammaLabTechnologies/harmonist](https://github.com/GammaLabTechnologies/harmonist)