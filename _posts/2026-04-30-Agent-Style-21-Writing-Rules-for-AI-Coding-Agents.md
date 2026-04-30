---
layout: post
title: "Agent Style: 21 Writing Rules That Make AI Agents Write Like Tech Pros"
description: "Learn how agent-style applies 21 literature-backed writing rules to AI coding agents, reducing mechanical violations by 45-82% across Claude, GPT, and Gemini models with soft enforcement and deterministic review."
date: 2026-04-30
header-img: "img/post-bg.jpg"
permalink: /Agent-Style-21-Writing-Rules-for-AI-Coding-Agents/
featured-img: ai-coding-frameworks/ai-coding-frameworks
categories: [Developer Tools, AI Agents, Open Source]
tags: [agent-style, AI writing rules, coding agents, LLM writing quality, Claude Code, Codex CLI, technical writing, anti-patterns, developer tools, open source]
keywords: "how to use agent-style for AI agents, agent-style writing rules tutorial, AI agent writing quality improvement, agent-style vs prose linter, 21 writing rules for LLM, agent-style installation guide, Claude Code writing rules, Codex CLI style enforcement, AI agent anti-patterns, technical writing rules for AI"
author: "PyShine"
---

# Agent Style: 21 Writing Rules That Make AI Agents Write Like Tech Pros

Agent-style is a curated set of 21 English writing rules formatted for AI coding and writing agents to follow at generation time, not as a post-hoc linter. Created by Yue Zhao (USC CS faculty and author of PyOD), the project distills 12 canonical rules from Strunk & White, Orwell, Pinker, and Gopen & Swan, plus 9 field-observed rules logged from real LLM output across dozens of writing projects from 2022 to 2026. Available as both a Python package and npm module, agent-style integrates directly into Claude Code, Codex CLI, GitHub Copilot, Cursor, Aider, and 10+ other AI agent tools, reducing mechanical AI-tell violations by 45-82% in benchmark testing.

## Architecture Overview

![Agent Style Architecture](/assets/img/diagrams/agent-style/agent-style-architecture.svg)

### Understanding the Agent-Style Architecture

The architecture diagram above shows how agent-style's 21 rules flow from source through adapters into AI agent contexts. The system is designed around two complementary enforcement paths that work together to improve writing quality.

**Rule Sources**

The 21 rules come from two peer groups. The 12 canonical rules (RULE-01 through RULE-12) are distilled from four writing authorities: Strunk & White's *The Elements of Style* (1959), Orwell's "Politics and the English Language" (1946), Pinker's *The Sense of Style* (2014), and Gopen & Swan's "The Science of Scientific Writing" (1990). Every citation has been verified against the original works. The 9 field-observed rules (RULE-A through RULE-I) come from direct observation of LLM output patterns across research papers, grant proposals, technical documentation, and agent configurations from 2022 to 2026.

**Soft Enforcement Path**

When you run `agent-style enable <adapter>`, the CLI writes the appropriate adapter file to your project. The adapter contains the 21 rules in a format the target tool reads natively. For Claude Code, this means a marker block in `CLAUDE.md` that imports `.agent-style/claude-code.md`. For AGENTS.md-compliant tools, it appends a compact rule block. For Cursor, it writes `.cursor/rules/agent-style.mdc`. The rules are loaded at generation time, so the agent tries to follow them while writing the first draft.

**Review Skill Path**

The optional `style-review` skill provides a second pass. After a draft exists, you run `/style-review <file>` (in Claude Code) or `agent-style review <file>` (CLI). The deterministic audit checks 14 mechanical and structural rules (em-dashes, jargon, transition openers, cliches, contractions, sentence length, bullet overuse, same-starts, paragraph closers). The semantic audit uses the host model's reasoning for 7 context-aware rules (curse of knowledge, vague language, needless words, uncalibrated claims, stress position, term drift, citation discipline). The skill then offers to produce a polished draft alongside the original, never touching the source file.

## The 21 Rules

![Agent Style Rules](/assets/img/diagrams/agent-style/agent-style-rules.svg)

### Understanding the Rule System

The rules diagram above illustrates the two groups of rules and their enforcement mechanisms. Each rule is phrased as a negative constraint ("Do not X") because research by Zhang et al. (2026) found that negative-constraint phrasing is empirically more effective for coding-agent instructions than positive phrasing.

**Canonical Rules (RULE-01 through RULE-12)**

| # | Rule | Source |
|---|------|--------|
| 01 | Do not assume the reader shares your tacit knowledge | Pinker 2014, Ch. 3 |
| 02 | Do not use passive voice when the agent matters | Orwell 1946 Rule 3; S&W II.14 |
| 03 | Do not use abstract language when a concrete term exists | S&W II.16; Pinker 2014 Ch. 3 |
| 04 | Do not include needless words | S&W II.17; Orwell 1946 Rule 3 |
| 05 | Do not use dying metaphors or prefabricated phrases | Orwell 1946 Rule 1 |
| 06 | Do not use avoidable jargon where everyday English works | Orwell 1946 Rule 5; Pinker 2014 Ch. 2 |
| 07 | Use affirmative form for affirmative claims | S&W II.15 |
| 08 | Do not overstate or understate claims relative to evidence | Pinker 2014 Ch. 6; Gopen & Swan 1990 |
| 09 | Express coordinate ideas in similar form (parallel structure) | S&W II.19 |
| 10 | Keep related words together | S&W II.20; Gopen & Swan 1990 |
| 11 | Place new or important information at the end of the sentence | Gopen & Swan 1990 |
| 12 | Break long sentences; vary length (split over 30 words) | S&W II.18; Pinker 2014 Ch. 4 |

**Field-Observed Rules (RULE-A through RULE-I)**

| # | Rule |
|---|------|
| A | Do not convert prose into bullet points unless the content is a genuine list |
| B | Do not use em or en dashes as casual sentence punctuation |
| C | Do not start consecutive sentences with the same word or phrase |
| D | Do not overuse transition words ("Additionally", "Furthermore", "Moreover") |
| E | Do not close every paragraph with a summary sentence |
| F | Use consistent terms; do not redefine abbreviations mid-document |
| G | Use title case for section and subsection headings |
| H | Support factual claims with citation or concrete evidence (critical) |
| I | Prefer full forms over contractions in formal technical prose |

The escape hatch comes from Orwell's Rule 6: "Break any of these rules sooner than say anything outright barbarous." Rules are guides to clarity, not ends in themselves.

## Adapter Matrix

![Adapter Matrix](/assets/img/diagrams/agent-style/agent-style-adapter-matrix.svg)

### Understanding the Adapter System

The adapter matrix diagram above shows how agent-style integrates with each AI agent tool through different install modes. The system supports 10+ tools with four distinct installation strategies.

**Import-Marker Mode (Claude Code)**

For Claude Code, agent-style writes `.agent-style/RULES.md` and `.agent-style/claude-code.md`, then safe-appends an `@.agent-style/claude-code.md` import marker to your existing `CLAUDE.md`. This approach preserves any existing content in `CLAUDE.md` while adding the rules as a referenced import. Claude Code reads the import at session start and loads all 21 rules into its context.

**Append-Block Mode (AGENTS.md, Copilot)**

For AGENTS.md-compliant tools (Codex CLI, Jules, Zed, Warp, Gemini CLI, VS Code) and GitHub Copilot (repo-wide), agent-style safe-appends a marker-wrapped compact adapter block to your existing instruction file. Content above and below the marker is preserved, and the block can be cleanly removed with `agent-style disable`.

**Owned-File Mode (Cursor, Copilot path-scoped, Anthropic Skills, Kiro)**

For tools that use dedicated rule directories, agent-style writes a new file at the tool's expected path (`.cursor/rules/agent-style.mdc`, `.github/instructions/agent-style.instructions.md`, `.claude/skills/agent-style/SKILL.md`, `.kiro/steering/agent-style.md`). This mode fails closed if a non-agent-style file already exists at the target path, preventing accidental overwrites.

**Print-Only Mode (Codex API)**

For the Codex API, which has no CLI host, agent-style writes `.agent-style/codex-system-prompt.md` and prints the prompt body to stdout with manual-step instructions to stderr. You paste the prompt into your Codex API `system_prompt` field.

## Style Review Workflow

![Style Review Workflow](/assets/img/diagrams/agent-style/agent-style-workflow.svg)

### Understanding the Style Review Workflow

The workflow diagram above shows the two-pass review process that combines deterministic auditing with semantic judgment.

**Pass 1: Deterministic Audit**

The deterministic audit runs 14 mechanical and structural detectors against the draft. These detectors check for em-dash overuse, jargon, transition openers, cliches, contractions, sentence length violations, bullet overuse, same-start sentences, and paragraph closers. Each detector produces a per-rule violation count with the first 5 examples. This pass requires no model -- it is pure pattern matching and runs instantly.

**Pass 2: Semantic Judgment**

The semantic audit uses the host model's reasoning for 7 rules that need context-aware understanding: RULE-01 (curse of knowledge), RULE-03 (vague language), RULE-04 (needless words), RULE-08 (uncalibrated claims), RULE-11 (stress position), RULE-F (term drift), and RULE-H (citation discipline). These rules cannot be checked mechanically because they require understanding what the text means, not just what it looks like.

**Revision and Verification**

After the merged audit, the skill asks whether to produce a polished draft. On confirmation, it writes a revised copy alongside the original (e.g., `DESIGN.reviewed.md`) with hard invariants: no new facts, metrics, citations, or links; preserve Markdown structure; preserve meaning and length budget. The source file is never touched. A re-audit shows the before-and-after scorecard and a diff.

## Does It Work? Benchmark Results

The v0.3.0 sanity bench tested 10 fixed prose tasks (2 PR descriptions, 1 design-doc section, 1 commit message, 4 paper sections, 1 product description, 1 NSF-style specific aim) with 2 generations per condition across three flagship models:

| Model | Baseline Violations | With agent-style | Reduction |
|-------|-------------------|------------------|-----------|
| Claude Opus 4.7 | 105 | 58 | -45% |
| OpenAI GPT-5.4 (Codex CLI) | 51 | 28 | -45% |
| Gemini 3 Flash | 79 | 14 | -82% |

Numbers are directional, not statsig: 10 tasks x 2 generations x 2 conditions = 40 calls per runner. The takeaway is that the ruleset reduces mechanical AI-tell density across every frontier family where the instruction file actually reaches the model's context. The exact size of the drop varies with each model's baseline -- heavier in models that emit more long sentences and em-dashes by default, smaller or zero in already-clean models.

## Installation

### Python

```bash
pip install agent-style
```

### Node.js

```bash
npm install -g agent-style
```

### No-Install (Pinned curl Recipe)

```bash
AGENT_STYLE_REF=v0.3.5
mkdir -p .agent-style
curl -fsSLo .agent-style/RULES.md \
  "https://raw.githubusercontent.com/yzhao062/agent-style/${AGENT_STYLE_REF}/RULES.md"
curl -fsSLo .agent-style/claude-code.md \
  "https://raw.githubusercontent.com/yzhao062/agent-style/${AGENT_STYLE_REF}/agents/claude-code.md"
```

Then add one line to your `CLAUDE.md`:

```text
@.agent-style/claude-code.md
```

## Usage

### Soft Enforcement (Generation Time)

```bash
# See all supported tools
agent-style list-tools

# Enable for Claude Code
agent-style enable claude-code

# Enable for AGENTS.md-compliant tools (Codex, Jules, Zed, Warp, Gemini CLI, VS Code)
agent-style enable agents-md

# Enable for Cursor
agent-style enable cursor

# Enable for GitHub Copilot (repo-wide)
agent-style enable copilot

# Dry run first to see what will be created
agent-style enable claude-code --dry-run --json

# Disable when no longer needed
agent-style disable claude-code
```

### Style Review (Second Pass)

```bash
# Skill path (Claude Code / Anthropic Skills)
agent-style enable style-review
# Then inside Claude Code:
/style-review DESIGN.md

# CLI path (works anywhere)
agent-style review DESIGN.md              # human-readable audit
agent-style review DESIGN.md --audit-only  # machine-readable JSON
agent-style review --compare a.md b.md     # A/B delta per rule
```

### Self-Verification

After enabling, ask your agent:

```
Is agent-style active?
```

Expected reply:

```
agent-style v0.3.5 active: 21 rules (RULE-01..12 canonical + RULE-A..I field-observed);
full bodies at .agent-style/RULES.md.
```

### Bundled via anywhere-agents (Zero Config)

If your project uses [anywhere-agents](https://github.com/yzhao062/anywhere-agents), agent-style is the default rule pack with no `enable` step needed:

```bash
pipx run anywhere-agents   # Python path
npx anywhere-agents        # Node.js path
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| **Agent ignores the rules** | Restart the session; most agents only load instruction files at startup |
| **Version string missing in verification** | The file is on disk but not in the agent's active context; check tool reload behavior |
| **Training-prior vocabulary still appears** | Soft enforcement nudges but does not eliminate all patterns; use `style-review` for a second pass |
| **Manual step required (Codex API, Aider)** | The CLI prints the exact action needed; paste the snippet into your config |
| **Want to opt out of specific rules** | Planned for v0.3.0: `agent-style override <RULE-ID> disable` |
| **Conflicts with existing rules** | Agent-style uses marker blocks and import references; existing content is preserved |

## Conclusion

Agent-style addresses a fundamental quality gap in AI-generated technical prose: LLMs produce text that is grammatically correct but mechanically identifiable. The 21 rules -- 12 canonical from writing authorities and 9 field-observed from real LLM output -- target the specific failure modes that make AI writing recognizable: em-dash overuse, transition word stacking, bullet-point conversion, vague claims, and dying metaphors. With benchmark reductions of 45-82% across Claude, GPT, and Gemini, and adapters for 10+ AI agent tools, agent-style provides a practical, literature-backed approach to making AI-generated prose indistinguishable from skilled human writing.

**Links:**
- GitHub Repository: [https://github.com/yzhao062/agent-style](https://github.com/yzhao062/agent-style)
- PyPI Package: [https://pypi.org/project/agent-style/](https://pypi.org/project/agent-style/)
- npm Package: [https://www.npmjs.com/package/agent-style](https://www.npmjs.com/package/agent-style/)