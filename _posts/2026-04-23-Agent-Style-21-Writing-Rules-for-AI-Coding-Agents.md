---
layout: post
title: "Agent Style: 21 Writing Rules for AI Coding Agents"
description: "Agent Style provides 21 writing rules for AI coding agents like Claude Code and Copilot, eliminating passive voice, filler phrases, and AI tells from generated technical prose."
date: 2026-04-23
header-img: "img/post-bg.jpg"
permalink: /Agent-Style-21-Writing-Rules-for-AI-Coding-Agents/
featured-img: ai-coding-frameworks/ai-coding-frameworks
categories: [AI Agents, Developer Tools, Best Practices]
tags: [Open Source, AI Agents, Code Style, Best Practices, Claude Code, Writing Quality, AI Writing, LLM Output, Developer Tools, Technical Writing]
keywords: "AI coding agent writing rules, how to improve AI code quality, agent style guide for LLMs, Claude Code writing rules, AI agent output quality, technical writing for AI agents, reduce AI sycophancy in code, AI code style linter, LLM writing best practices, agent style npm package"
author: "PyShine"
---

## Introduction

AI coding agents have transformed software development, but they share a persistent weakness: the prose they generate often reads like it was written by a machine. Long sentences filled with passive voice, em dashes used as casual punctuation, filler phrases like "in order to" and "it is important to note that," and handwavy claims without citations -- these are the tells that betray AI authorship even when the code itself is correct.

**Agent Style** ([yzhao062/agent-style](https://github.com/yzhao062/agent-style)) tackles this problem head-on. It provides 21 curated English writing rules formatted specifically for AI coding and writing agents to follow at generation time -- not as a post-hoc linter, but as instructions loaded directly into the agent's context. The ruleset is the work of Yue Zhao, USC CS faculty and author of PyOD, who distilled 12 canonical rules from four classic writing authorities (Strunk & White, Orwell, Pinker, Gopen & Swan) and added 9 field-observed rules drawn from years of watching LLM output across research papers, grant proposals, technical documentation, and agent configurations.

The result is a drop-in style guide that works with Claude Code, OpenAI Codex, GitHub Copilot, Cursor, Aider, Kiro, and any AGENTS.md-compliant tool. With 254 GitHub stars and packages on both PyPI and npm, Agent Style has quickly become the de facto standard for enforcing writing quality in AI-generated technical prose.

## How It Works

Agent Style operates through two complementary enforcement paths:

**1. Soft enforcement (default):** The rules are loaded at generation time, so the agent tries to follow them while writing the first draft. A single `enable` command per tool wires the ruleset into the agent's context. The agent then reads Agent Style's 21 rules as part of its system prompt or project configuration.

**2. Skill-based review (opt-in second pass):** After the draft exists, you invoke `style-review` to audit the draft against the same 21 rules. The skill performs a deterministic audit, adds semantic judgment for rules that require context-aware reasoning, and on your confirmation writes a polished copy beside the original. Your source file is never touched.

The key insight is that soft enforcement nudges the model but does not guarantee compliance -- training-prior vocabulary like "leverages," "cutting-edge," and "Additionally"-openers can still appear, especially in prose over 200 words. That is why the second-pass review exists: it catches what the first pass misses.

## Architecture

![Agent Style Architecture](/assets/img/diagrams/agent-style/agent-style-architecture.svg)

The architecture diagram illustrates the full pipeline from rule definition to enforced output. At the top, the **Style Rules Source** layer contains RULES.md with all 21 full rule bodies including BAD/GOOD examples and rationale, fed by two peer sources: the canonical rules distilled from Strunk & White, Orwell, Pinker, and Gopen & Swan, and the field-observed rules logged from LLM output patterns between 2022 and 2026.

The **Adapter Layer** transforms the raw rules into platform-specific formats. Each adapter uses a different install mode: Claude Code uses `import-marker` (an `@path` reference in CLAUDE.md), AGENTS.md and Copilot use `append-block` (safe-appending a marker-wrapped block), Cursor and Anthropic Skills use `owned-file` (writing a new agent-style-owned file), Codex API uses `print-only` (manual system prompt paste), and Aider uses `multi-file-required` (convention file plus config snippet).

The **Agent Platforms** layer shows where the rules are consumed: Claude Code, Codex CLI, GitHub Copilot, and Cursor. Each platform applies the rules through its own enforcement mechanism. Finally, the **Enforcement Tiers** layer shows the four-tier system: Tier-1 (deny/ask for exact-phrase blocking and heuristic prompts), Tier-2 (linter-based checks via ProseLint and regex), Tier-3 (agent self-check requiring judgment), and Tier-4 (Codex review as the primary gate). Only after passing through these tiers does the output reach the **Enforced Output** node -- clean technical prose free of AI tells.

## The 21 Rules

![Agent Style Rules by Category](/assets/img/diagrams/agent-style/agent-style-rules.svg)

The rules diagram organizes all 21 rules into five categories based on the writing quality dimension they address. Each rule is labeled with its severity level (critical, high, medium, or low) and its rule ID. Dashed lines between categories show cross-category relationships -- for instance, RULE-04 (Needless Words) and RULE-05 (Dying Metaphors) both cut filler, while RULE-08 (Claim Calibration) and RULE-H (Citation Discipline) both fight handwavy prose.

### Clarity (5 Rules)

These rules ensure the reader can actually understand what is being said:

- **RULE-01: Curse of Knowledge [critical]** -- Do not assume the reader shares your tacit knowledge. Name the intended reader and write for someone one level below your expertise. This is the most common reviewer complaint on AI-generated technical prose.
- **RULE-03: Concrete Language [high]** -- Replace abstract category words ("factors," "aspects," "considerations") with specific items. "The system has performance issues" says nothing; "the checkout endpoint p95 latency rose from 120ms to 450ms at 14:00 UTC" names what, when, and how much.
- **RULE-09: Parallel Structure [medium]** -- Express coordinate ideas in the same grammatical form. If item 1 in a list is a verb-initial clause, items 2 and 3 should also be verb-initial clauses.
- **RULE-10: Related Words Together [medium]** -- Keep subject close to verb and modifier close to modified. If the gap between subject and verb exceeds 8 words, split the sentence.
- **RULE-11: Stress Position [medium]** -- Place new or important information at the end of the sentence. Readers unconsciously expect the stress position for new information, as shown by Gopen & Swan 1990.

### Consistency (5 Rules)

These rules ensure the prose maintains a uniform register and structure:

- **RULE-02: Passive Voice [high]** -- Prefer active voice when the agent is known and worth naming. "We ran the experiments on eight NVIDIA A100 GPUs" beats "The experiments were conducted on eight NVIDIA A100 GPUs."
- **RULE-07: Affirmative Form [medium]** -- Replace "not important" with "trivial," "did not remember" with "forgot," "does not succeed" with "fails." One affirmative word beats two negating words.
- **RULE-F: Term Consistency [medium]** -- Once you define a term or abbreviation, keep using it. Do not alternate "large language model," "LLM," "language model," and "foundation model" as synonyms for the same thing.
- **RULE-G: Title Case Headings [low]** -- Use title case for section and subsection headings. LLMs default to sentence case, which is a visible AI-tell in academic and engineering contexts.
- **RULE-I: No Contractions [low]** -- Prefer "it is" over "it's" in formal technical prose. Contractions are acceptable in informal registers but must be deliberate, not accidental register drift.

### Conciseness (4 Rules)

These rules eliminate wordy and structural filler:

- **RULE-04: Needless Words [high]** -- Cut filler phrases like "in order to" (use "to"), "due to the fact that" (use "because"), "it is important to note that" (delete and state the fact). These are mechanically enforceable with near-zero false-positive risk.
- **RULE-05: Dying Metaphors [high]** -- Delete cliches like "pushes the boundaries," "paradigm shift," "state of the art," and "industry-leading." If you cannot replace the cliche with a specific number or mechanism, the cliche was hiding the absence of substance.
- **RULE-12: Long Sentences [high]** -- Split any sentence over 30 words into two or more sentences. Vary sentence length across a paragraph. A paragraph of five 25-word sentences reads as monotone.
- **RULE-A: Bullet Overuse [medium]** -- Keep prose in paragraphs when ideas connect by cause-and-effect or argument. Use bullets only for genuine parallel enumerations. Resist the forced 3-item triad.

### Correctness (4 Rules)

These rules ensure claims are honest, verifiable, and precisely stated:

- **RULE-08: Claim Calibration [high]** -- Calibrate verbs to evidence. Experimental results "suggest" or "show"; theoretical derivations "imply" or "prove." Do not write "proves" when the evidence is "suggests."
- **RULE-H: Citation Discipline [critical]** -- Support factual claims with verifiable citation or concrete evidence. Never fabricate citations. If a cited paper cannot be verified, mark it `[UNVERIFIED]` and flag for review. This is the flagship rule of the three-rule cluster against handwavy prose (RULE-03, RULE-08, RULE-H).
- **RULE-06: Avoidable Jargon [medium]** -- Prefer "use" over "leverage," "method" over "methodology," "feature" over "functionality." Reserve the longer word for when it carries information the shorter word does not.
- **RULE-B: Dash Overuse [medium]** -- Do not use em or en dashes as casual sentence punctuation. Prefer commas, semicolons, colons, or parentheses. LLMs produce em dashes at several times the rate of skilled human technical writers.

### Context (3 Rules)

These rules address paragraph-level and document-level patterns:

- **RULE-C: Same-Starts [medium]** -- Do not open two or more consecutive sentences with the same word. The pattern signals a drafting template that the generation process locked into.
- **RULE-D: Transition Overuse [medium]** -- Do not open sentences with "Additionally," "Furthermore," "Moreover," "In addition" unless the logical move genuinely needs flagging. In most cases, a period ends the prior sentence and the next sentence makes the connection by content alone.
- **RULE-E: Summary Closers [medium]** -- Do not end every paragraph with a sentence that restates its point. If deleting the closer sentence leaves the paragraph's point intact, delete the closer.

### Escape Hatch

All 21 rules carry an escape hatch from Orwell 1946 Rule 6: *"Break any of these rules sooner than say anything outright barbarous."* Rules are guides to clarity, not ends in themselves.

## Adapter Matrix

![Agent Style Adapter Matrix](/assets/img/diagrams/agent-style/agent-style-adapter-matrix.svg)

The adapter matrix diagram shows how the 21 rules flow from RULES.md through five distinct install modes to reach six agent platforms, each with its own enforcement level profile. The **Install Modes** cluster defines how each adapter integrates with its target platform:

- **import-marker** (Claude Code): Writes `.agent-style/RULES.md` and `.agent-style/claude-code.md`, then safe-appends an `@.agent-style/claude-code.md` marker block to your existing CLAUDE.md. Claude Code resolves the `@path` directive at launch and loads the full rule bodies into active context.

- **append-block** (AGENTS.md, Copilot repo-wide): Safe-appends a marker-wrapped compact adapter block to your existing instruction file. Content above and below the marker is preserved. Future agent-style updates replace only the marker region.

- **owned-file** (Cursor, Copilot path-scoped, Anthropic Skills, Kiro): Writes a new agent-style-owned file at the tool's rule-directory path (e.g., `.cursor/rules/agent-style.mdc`, `.github/instructions/agent-style.instructions.md`). Fails closed if a non-agent-style file already occupies that path.

- **print-only** (Codex API): Writes `.agent-style/codex-system-prompt.md` and prints the prompt body to stdout with manual-step instructions on stderr. You paste the prompt into your Codex API `system_prompt` field.

- **multi-file-required** (Aider): Writes `.agent-style/aider-conventions.md` and prints a `.aider.conf.yml` snippet to stderr. You integrate both files into the Aider configuration manually.

The **Enforcement Levels** legend at the bottom clarifies the three tiers of enforcement strictness: **Strict** (Tier-1 deny with zero false positives, covering RULE-04 filler phrases and RULE-05 cliches), **Recommended** (Tier-2 linter plus Tier-3 agent self-check, covering most structural and semantic rules), and **Optional** (Tier-4 Codex review as a full audit pass for comprehensive verification).

## Installation

Agent Style is available on both PyPI and npm, making it accessible regardless of your technology stack:

```bash
pip install agent-style                              # Python users
```

Or for Node.js users:

```bash
npm install -g agent-style                           # Node users
# or: npx --yes agent-style@0.3.1 <subcommand>       # no install needed
```

For a no-install approach, you can pin to a specific release using curl:

```bash
AGENT_STYLE_REF=v0.3.1
mkdir -p .agent-style
curl -fsSLo .agent-style/RULES.md       "https://raw.githubusercontent.com/yzhao062/agent-style/${AGENT_STYLE_REF}/RULES.md"
curl -fsSLo .agent-style/claude-code.md "https://raw.githubusercontent.com/yzhao062/agent-style/${AGENT_STYLE_REF}/agents/claude-code.md"
```

Then add one line to your CLAUDE.md (create the file only if absent; never overwrite):

```text
@.agent-style/claude-code.md
```

## Usage

### Soft Enforcement -- Rules at Generation Time

After installing the CLI, enable the adapter for your specific tool:

```bash
agent-style enable claude-code                       # wire up Claude Code
agent-style enable agents-md                         # Codex, Jules, Zed, Warp, Gemini CLI, VS Code
agent-style enable cursor                            # Cursor
agent-style enable copilot                           # GitHub Copilot (repo-wide)
```

To see all supported tools:

```bash
agent-style list-tools                               # see all supported tools
```

To reverse an enable:

```bash
agent-style disable <tool>                           # reverse an enable
```

After enabling, verify the rules are active by asking your agent:

```text
Is agent-style active?
```

Expected reply:

```text
agent-style v0.3.1 active: 21 rules (RULE-01..12 canonical + RULE-A..I field-observed); full bodies at .agent-style/RULES.md.
```

### Skill-Based Review -- Second Pass

For skill-capable hosts (Claude Code / Anthropic Skills):

```bash
agent-style enable style-review                       # install the review skill
```

Then inside Claude Code:

```text
/style-review DESIGN.md
```

For the CLI path (works anywhere pip/npm runs; no skill host needed):

```bash
agent-style review DESIGN.md                          # human-readable audit
agent-style review DESIGN.md --audit-only             # machine-readable JSON
agent-style review --compare a.md b.md                # A/B delta per rule
```

The review workflow proceeds as follows: first, a deterministic audit runs against all 21 rules using mechanical and structural detectors (em-dashes, jargon, transition openers, cliches, contractions, sentence length, bullet overuse, same-starts, paragraph closers). Then, semantic judgment is added via the host model for the 7 rules that need context-aware reasoning: RULE-01 (curse of knowledge), RULE-03 (vague language), RULE-04 (needless words), RULE-08 (uncalibrated claims), RULE-11 (stress position), RULE-F (term drift), and RULE-H (citation discipline). The skill asks whether to produce a polished draft at `DESIGN.reviewed.md`. On confirmation, it writes a revised copy beside the original and shows the diff. The source file is never touched.

### Bundled via anywhere-agents (Zero-Config)

If your project uses [anywhere-agents](https://github.com/yzhao062/anywhere-agents), Agent Style is the default rule pack with no additional `enable` step:

```bash
pipx run anywhere-agents   # Python path
npx anywhere-agents        # Node.js path
```

## Features

![Agent Style Workflow](/assets/img/diagrams/agent-style/agent-style-workflow.svg)

The workflow diagram traces the complete enforcement pipeline from draft generation to verified output. The process begins when an agent **writes a draft** -- generating technical prose as part of a code commit message, design document, API specification, or research paper. At this point, **soft enforcement** is already active: the 21 rules are loaded in the agent's context, nudging the model toward cleaner prose from the start.

After the draft is written, the **style check** runs `agent-style review` to perform a deterministic audit. This audit applies all four enforcement tiers in sequence. **Tier-1 Deny** blocks exact-phrase matches outright -- filler phrases like "in order to" and "due to the fact that" are caught with zero false positives. **Tier-1 Ask** pauses for user confirmation on heuristic matches, suggesting substitutions like "leverage" to "use" or flagging sentences over 30 words for potential splitting. **Tier-2 Linter** runs ProseLint checks and structural regex patterns for passive voice, bullet overuse, and same-start detection. **Tier-3 Self-Check** applies agent judgment for semantic rules that require understanding context -- curse of knowledge, vague language, claim calibration, and citation discipline.

The decision node then branches: if **no violations** are found, the draft passes directly to the **Enforced Output** as clean technical prose. If **violations are detected**, the **Auto-Fix** path activates the `style-review` skill, which produces a polished draft at `DESIGN.reviewed.md` alongside the original. The source file remains untouched. A **re-audit** then runs on the revised draft, producing a before/after scorecard and diff for verification. Only after the re-audit confirms improvement does the output reach the **Enforced Output** node.

This two-pass design is critical because soft enforcement alone cannot guarantee compliance. Training-prior vocabulary and register patterns can still surface in the first draft, especially for prose exceeding 200 words. The deterministic audit catches mechanical violations reliably, while the semantic review addresses the deeper patterns that regex cannot detect. Together, they produce prose that reads as if a careful human editor reviewed every sentence.

### Key Features Summary

- **21 curated rules** split across two peer groups: 12 canonical (literature-backed) and 9 field-observed (LLM-specific patterns)
- **Multi-platform support** with adapters for Claude Code, Codex, Copilot, Cursor, Aider, Kiro, and AGENTS.md-compliant tools
- **Four-tier enforcement** from exact-phrase deny to full Codex review
- **Deterministic audit** via `agent-style review` with mechanical and structural detectors
- **Semantic review** for 7 context-aware rules via the style-review skill
- **Safe install** -- all adapters are idempotent and never overwrite user content
- **Dual package** -- available on both PyPI and npm
- **Zero-config option** via anywhere-agents bundling
- **Self-verification handshake** -- ask "Is agent-style active?" to confirm rules are loaded
- **Escape hatch** -- Orwell's Rule 6: break any rule sooner than say something barbarous

### Benchmark Results

The v0.3.0 sanity bench tested 10 fixed prose tasks across three flagship models, with and without the ruleset loaded at generation time:

| Model | Baseline Violations | With agent-style | Reduction |
|-------|-------------------|-------------------|-----------|
| Claude Opus 4.7 | 105 | 58 | -45% |
| OpenAI GPT-5.4 via Codex CLI | 51 | 28 | -45% |
| Gemini 3 Flash | 79 | 14 | -82% |

Numbers are directional (10 tasks x 2 generations x 2 conditions = 40 calls per runner), but the consistent reduction across all frontier families confirms that the ruleset reduces mechanical AI-tell density whenever the instruction file actually reaches the model's context.

## Conclusion

Agent Style fills a gap that most AI coding tool users have experienced but few have systematically addressed: the gap between correct code and readable prose. By distilling decades of writing advice into 21 actionable rules and packaging them as drop-in adapters for every major AI coding platform, it makes literature-backed writing quality accessible to any developer using AI agents.

The project's dual-path enforcement model -- soft enforcement at generation time plus optional skill-based review -- acknowledges the reality that AI models need both guidance and verification. The four-tier enforcement system provides a principled framework for deciding which rules can be mechanically enforced (Tier-1 deny for exact-phrase matches with zero false positives) and which require human or model judgment (Tier-3 and Tier-4 for semantic rules like curse of knowledge and citation discipline).

For teams that have noticed their AI-generated documentation, commit messages, and design docs reading like marketing copy -- full of "leverages," "cutting-edge," and "Additionally"-openers -- Agent Style provides a concrete, measurable, and immediately deployable fix. Install it, enable it for your platform, and ask your agent: "Is agent-style active?" The answer should be yes.

The project is actively maintained with a public roadmap (v0.3.0), dual licensing (CC BY 4.0 for rules content, MIT for code), and welcomes contributions that add canonical rules with cited sources. Visit the [GitHub repository](https://github.com/yzhao062/agent-style) to get started.