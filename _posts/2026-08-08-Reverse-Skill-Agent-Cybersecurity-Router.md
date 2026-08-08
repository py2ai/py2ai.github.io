---
layout: post
title: "Reverse Skill: A Cybersecurity Skills Router for AI Agents"
description: "Reverse Skill is a routing engine that directs AI agents to the right tools and methodologies for security tasks — from APK analysis and binary reversing to penetration testing and CTF challenges. With 41 routing rules, 163 benchmark cases, MCP integration, and a three-layer authorization gate."
date: 2026-08-08
tags: [AI, Security, Cybersecurity, Agent, Penetration Testing, Reverse Engineering, MCP]
featured-img: ai-coding-frameworks/ai-coding-frameworks
image: /assets/img/diagrams/12-factor-agents/12-factor-agents-architecture.svg
---

# Reverse Skill: A Cybersecurity Skills Router for AI Agents

Imagine asking your AI coding assistant to analyze an APK file. What happens? It might try `jadx` first, then jump to `apktool`, then attempt `Frida` for dynamic analysis — all without knowing whether the target is actually an APK, whether those tools are installed, or whether you have authorization to interact with the target. Now imagine the same scenario for a binary executable, a JavaScript obfuscation challenge, a CTF puzzle, or a penetration test against a live web app. Each scenario demands a completely different playbook, and AI agents are notoriously bad at picking the right one.

This is the problem **reverse-skill** solves. It's a cybersecurity skills router — a package that sits between your AI agent and the security task at hand, classifies what needs to be done, checks which tools are available, and routes the agent to a repeatable, documented workflow instead of letting it guess commands.

The project is the work of security researcher **zhaoxuya520** and is available on GitHub at [zhaoxuya520/reverse-skill](https://github.com/zhaoxuya520/reverse-skill). It currently ships with **41 routing rules** (R0 through R40), **163 regression benchmark cases**, **42 tracked skill modules**, and MCP server integration for tool calling across multiple AI client platforms.

## What Reverse Skill Actually Is

At its core, reverse-skill is a routing engine. When you feed it a task description — "decompile this APK and check for anti-debugging," "recover the encryption key from this JavaScript obfuscation," or "perform reconnaissance on this target" — it follows a deterministic pipeline:

```
User task
  → RULES.md (global routing protocol)
  → MASTER-ROUTING.md / master-route.ps1 (PRIMARY fast ladder)
  → case-init / scope.md (authorization + network profile; NO target interaction until ready)
  → Scenario skill → tools / MCP / scripts
  → timeline + Evidence → Finding → Path → report + field-journal
```

The key insight is that **routing happens before execution**. The system never lets an agent "do first, route later." Instead, it enforces a strict order: classify the task, verify authorization, confirm tool availability, and only then take action.

### Why This Matters

AI agents (Claude Code, Codex CLI, Cursor, Cline, Windsurf, etc.) encounter security tasks across wildly different domains:

- **APK analysis** needs jadx, apktool, Frida, and smali knowledge
- **Binary reversing** needs IDA Pro, radare2, Ghidra, and understanding of PE/ELF/Mach-O formats
- **JS deobfuscation** needs CDP hooks, runtime capture, AST analysis, and domain-specific patterns
- **Penetration testing** needs nmap, Nuclei, SQLMap, Burp Suite, and an understanding of attack chains
- **CTF challenges** need 40+ sub-skills spanning reverse, pwn, crypto, web, and forensics

Each domain has its own playbook, its own toolchain, and its own safety considerations. Without a router, AI agents either guess poorly or fall back to generic approaches that waste tokens and produce unreliable results.

## How the Routing Works

The routing system is built around three layers: a master routing ladder, a full routing matrix, and skill-specific execution contracts.

### The Master Routing Ladder (41 Rules)

The `MASTER-ROUTING.md` file defines a priority-ordered ladder of 41 rules (R0–R40). Each rule maps a pattern of keywords to a skill module:

| Rule | Trigger Condition | Routes To |
|------|-------------------|-----------|
| **R1** | APK / smali / jadx / apktool | `skills/apk-reverse/` |
| **R2** | IPA / iOS / Objection / MobSF | `skills/mobile-reverse/` |
| **R3** | JS signing / frontend encryption / jshook / CDP | `skills/js-reverse/` |
| **R4** | DSL VM / fireeye / custom opcode VM | `skills/reverse-engineering/dsl-vm-reverse/` |
| **R5** | .NET / dnSpy / de4dot / ConfuserEx | `skills/dotnet-reverse/` |
| **R6** | IDA / decompile / deep disassembly | `skills/ida-reverse/` |
| **R7** | radare2 / r2 | `skills/radare2/` |
| **R8** | Firmware / binwalk / IoT / EMBA | `skills/firmware-pentest/` |
| **R9** | Malware sample / YARA / sandbox | `skills/malware-analysis/` |
| **R10** | Attack chain / red team / lateral movement | `skills/attack-chain/` |
| **R11** | Nmap / Nuclei / SQLMap / penetration tools | `skills/pentest-tools/` |
| **R12** | API / GraphQL / BOLA / JWT attacks | `skills/api-security/` |
| **R13** | SBOM / Trivy / supply chain | `skills/supply-chain-security/` |
| **R14** | LLM / prompt injection / agent security | `skills/llm-security/` |
| **R15** | bindiff / symbol migration / PDB | `skills/binary-diff/` |
| **R16** | N-day / patch diff / exploit development | `skills/patch-diff-exploit/` |
| **R17** | pwn / ROP / stack exploitation | `skills/pwn-chain/` |
| **R18** | EDR / evasion / syscall | `skills/edr-bypass-re/` |
| **R0** | Generic reverse / anti-debug / OLLVM / unknown binary | `skills/reverse-engineering/` |

The ladder is checked top-down. The first matching rule wins. If no strong keyword matches, the system defaults to **R0** (generic reverse engineering) and suggests consulting the full routing matrix.

### The Full Routing Matrix

When the master ladder produces ambiguous results, `skills/routing.md` provides a three-dimensional routing matrix covering **target type**, **user intent**, and **toolchain**. It maps over 100+ task phrasings to specific skill modules. For example:

- `"DSL VM / custom instruction set / risk engine reverse"` → `reverse-engineering/dsl-vm-reverse/SKILL.md`
- `"decompile / IDA analyze"` → `ida-reverse/SKILL.md`
- `"Frida hook / dynamic inject"` → `reverse-engineering/tools-dynamic.md`
- `"OLLVM deobfuscate / control flow flattening removal"` → `reverse-engineering/references/ollvm-deobfuscation.md`

The matrix also covers cross-module tasks — for instance, a CTF challenge might combine reverse engineering, cryptography, and web exploitation, each routed to separate skill modules that the orchestrator chains together.

### Running the Router from the Command Line

The router can be invoked directly via PowerShell or Bash scripts:

```powershell
# One-shot PRIMARY triage
powershell -File skills\scripts\master-route.ps1 -Hint "Analyze this APK for anti-debugging checks"

# Initialize a case directory with scope, timeline, and workitems
powershell -File skills\scripts\case-init.ps1 -Hint "Reverse this binary" -CaseName "malware-analysis"

# Smoke test the entire routing system
powershell -File skills\scripts\smoke.ps1
```

## Three-Layer Authorization Gate

One of the most important design decisions in reverse-skill is its **three-layer authorization gate**. The system refuses to interact with any target until all three layers are satisfied:

### Layer 1: Authorization Pre-Declaration

Before any tool touches a target, the system reads `skills/field-journal/precedent-auth.md` and requires an explicit authorization declaration. This ensures that the user has legal permission to test the target.

### Layer 2: Scope Contract

The `case-init` script generates a `scope.md` file that must be filled with:

- **auth.status**: Must be set to `granted` before any action occurs
- **target URL or IP**: Clearly identified and documented
- **network profile**: Authorized targets only, no unintended network access
- **role assignments**: Lead specialist and supporting roles defined

```powershell
# Initialize a case with all authorization fields set
powershell -File skills\scripts\case-init.ps1 `
  -Hint "Penetration test against staging environment" `
  -CaseName "staging-pentest" `
  -AuthGranted `
  -TargetUrl "https://staging.example.com" `
  -NetworkProfile authorized_target_only
```

### Layer 3: Case Guard

The `case-guard.ps1` script performs a runtime check before any tool execution. If authorization is not `granted` or the network profile is not set, the guard exits with code 2 (refusing to proceed). The `-Force` flag issues a warning but still blocks execution.

This three-layer approach is critical for security professionals who need demonstrable authorization trails. Every action is logged with its authorization context, creating an auditable chain from task receipt to final report.

## 42 Skill Modules

The skill modules are the execution engines behind each routing rule. Each module contains a `SKILL.md` file with:

- A clear **execution contract** — what the module does, prerequisites, and output format
- **Tool integration** — which tools to use, how to invoke them, and how to interpret results
- **MCP server connections** — for tools with MCP interfaces (Burp Suite, IDA Pro, etc.)
- **References and playbooks** — curated knowledge for specific sub-tasks

Here's a snapshot of the tracked skill modules:

| Category | Modules |
|----------|---------|
| **Mobile** | APK reverse, iOS/mobile reverse, browser extension reverse |
| **Binary** | IDA reverse, radare2, Ghidra reverse, Go/Rust reverse, .NET reverse, macOS reverse |
| **Web/JS** | JS reverse, DSL VM reverse, protocol reverse, browser automation |
| **Security Testing** | Pentest tools, attack chain, API security, firmware pentest |
| **Analysis** | Malware analysis, digital forensics, binary diff, code audit |
| **Offensive** | Pwn chain, EDR bypass, patch diff/exploit, supply chain security |
| **Defensive** | Threat hunting, identity federation, database security, email security |
| **Specialized** | LLM security, CTF sandbox orchestrator, hardware security, radio/SDR |
| **Utilities** | Diagram generator, docs generator, field journal |

### Example: APK Reverse Module

When R1 routes an APK task to `skills/apk-reverse/`, the agent follows a structured workflow:

```text
1. triage → classify the APK, identify architecture, check protection
2. static → jadx decompile, apktool unpack, smali analysis
3. dynamic → Frida hooking, runtime capture, SSL pinning bypass
4. analysis → identify anti-debugging, data exfiltration, vulnerable components
5. report → Evidence → Finding → Path chain, structured output
```

### Example: JS Reverse Module

When R3 routes a JavaScript obfuscation task to `skills/js-reverse/`, the agent follows a five-stage pipeline:

```text
Observe → Capture → Rebuild → Verify → Document
```

This includes support for CDP (Chrome DevTools Protocol) hooks, runtime capture via MCP servers, AST-level deobfuscation, and pattern matching against known obfuscation frameworks.

## MCP Server Integration

Reverse-skill integrates with MCP (Model Context Protocol) servers to provide tool-calling capabilities. The MCP integration enables:

- **Burp Suite MCP**: 78 tools for web application security testing, including proxy history analysis, Intruder brute forcing, Repeater replay, and Collaborator OOB testing
- **IDA Pro MCP**: Decompilation, cross-reference analysis, pattern search, and script execution
- ** anything-analyzer MCP**: HTTP capture, request replay, and browser interaction for security testing
- **Pentest Swarm AI**: Autonomous pentesting swarm that can coordinate multi-stage attacks
- **Reqable MCP**: Network traffic capture and API workflow testing

The MCP servers are registered through a bootstrap process that validates installation, checks authentication, and confirms connectivity before making them available to the agent.

## 163 Regression Benchmark Cases

The project maintains **163 regression benchmark cases** that validate routing accuracy across all skill modules. These cases ensure that updates to routing rules don't break existing classifications. The test suite covers:

- Correct rule matching for each of the 41 routing rules
- Edge cases (ambiguous inputs, overlapping triggers, multi-language targets)
- Tool availability checks across Windows and Linux platforms
- Authorization gate enforcement
- Evidence chain integrity

Running the benchmarks:

```powershell
# Full smoke test including routing matrix validation
powershell -File skills\scripts\smoke.ps1

# Case initialization and scope validation
powershell -File skills\scripts\case-init.ps1 -Hint "test task" -CaseName "benchmark-test"
```

## CI Platforms: Windows + Ubuntu

Reverse-skill is designed for cross-platform operation with CI pipelines on both Windows and Ubuntu:

- **Windows**: PowerShell-based scripts for master routing, case initialization, tool index refresh, and smoke testing
- **Ubuntu/Linux**: Bash equivalents for the same operations, with Kali Linux-specific profiles for security tool availability
- **Tool Index**: Auto-generated `tool-index.md` that reflects which tools are actually installed on the current machine. Agents never try to use tools that aren't available.

```bash
# Linux tool index refresh
bash skills/scripts/refresh-tool-index.sh

# Kali Linux specialized refresh
bash kali/scripts/refresh-tool-index.sh
```

## Client-Neutral Design

A key design principle is that reverse-skill is **client-neutral**. It works with any AI coding client that can read markdown files and execute scripts:

- Claude Code
- Codex CLI
- Cursor
- Cline
- Windsurf
- Kiro
- Aider
- Continue
- Reasonix

The routing core is pure markdown and scripts — no client-specific APIs or SDKs are required. Client-specific adapters are optional add-ons, not dependencies.

## Who Should Use Reverse Skill

**Security professionals** who want AI assistance without the guesswork:

- **Penetration testers** who need consistent, documented workflows across different target types
- **Reverse engineers** who want AI agents to follow established playbooks instead of improvising
- **CTF competitors** who need rapid classification of challenge types and access to 40+ sub-skills
- **Malware analysts** who want structured analysis pipelines with evidence tracking
- **Red team operators** who need multi-stage attack chain orchestration
- **Security educators** who want to teach structured analysis methodologies
- **Bug bounty hunters** who need repeatable reconnaissance and exploitation workflows

## Getting Started

### Prerequisites

- **Java / JDK** — for jadx and apktool
- **Node.js 22.12+** — for JS toolchain and MCP servers
- **Python 3.x** — for Frida and helper scripts
- **A code AI client** — Claude Code, Codex CLI, Cursor, etc.

### Installation

```bash
git clone https://github.com/zhaoxuya520/reverse-skill.git
```

Then refresh the tool index for your platform:

| Platform | Command |
|---|---|
| Windows | `powershell -File skills/scripts/refresh-tool-index.ps1` |
| Linux / macOS | `bash skills/scripts/refresh-tool-index.sh` |
| Kali Linux | `bash kali/scripts/refresh-tool-index.sh` |

Check `skills/tool-index.md` to see which tools were detected.

### Typical Workflow

```text
1. Clone the repo and refresh the tool index
2. Ask your AI client to read RULES.md
3. Run master-route.ps1 with your task description
4. Run case-init.ps1 to set up scope and authorization
5. The router directs you to the right SKILL.md
6. Follow the skill's execution contract
7. Log findings via Evidence → Finding → Path
8. Generate reports via docs-generator
```

## Conclusion

Reverse Skill fills a critical gap in the AI-assisted security toolkit: it gives agents a map before they start exploring. By enforcing routing-before-execution, implementing a three-layer authorization gate, and providing 42 skill modules with 163 regression-tested routing rules, it transforms AI agents from unpredictable tool guessers into structured security analysts.

The project's 121 commits, 15 contributors, and active community (QQ group 942400892, Discord server) demonstrate that this is a living, evolving toolkit. Whether you're analyzing an APK, reversing a binary, deobfuscating JavaScript, conducting a penetration test, or competing in a CTF, reverse-skill ensures your AI agent approaches the task with the right methodology, the right tools, and the right authorization — every single time.

**Get Started**: [GitHub Repository](https://github.com/zhaoxuya520/reverse-skill) | [Full Routing Matrix](https://github.com/zhaoxuya520/reverse-skill/blob/main/skills/routing.md) | [Master Routing Ladder](https://github.com/zhaoxuya520/reverse-skill/blob/main/skills/MASTER-ROUTING.md) | [AI Bootstrap Guide](https://github.com/zhaoxuya520/reverse-skill/blob/main/README_AI.md)