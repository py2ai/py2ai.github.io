---
layout: post
title: "PM Skills Marketplace: 68 AI-Powered Product Management Skills for Better Decisions"
description: "Learn how PM Skills Marketplace brings 68 structured PM skills and 42 chained workflows to AI coding assistants. From discovery to strategy, execution, launch, and growth -- Teresa Torres, Marty Cagan, and Alberto Savoia frameworks built into your daily workflow."
date: 2026-06-18
header-img: "img/post-bg.jpg"
permalink: /PM-Skills-Marketplace-AI-Powered-Product-Management/
featured-img: ai-coding-frameworks/ai-coding-frameworks
categories: [AI Agents, Product Management, Developer Tools]
tags: [PM Skills, product management, AI skills, Claude Code, product discovery, PRD, product strategy, AI workflows, go-to-market, product analytics]
keywords: "PM Skills Marketplace tutorial, how to use PM Skills with Claude Code, AI product management skills framework, product discovery AI assistant workflow, PRD writing with AI Claude Code, product strategy frameworks AI tools, go-to-market planning AI assistant, product management AI workflows open source, Claude Code plugins for product managers, AI shipping kit vibe coding audit"
author: "PyShine"
---

## Introduction

Product managers are drowning in AI-generated text but starving for structured decisions. Generic AI assistants can draft a PRD in seconds, but they cannot tell you whether you are solving the right problem or whether your assumptions are worth testing. PM Skills Marketplace changes that by bringing 68 domain-specific PM skills and 42 chained workflows to AI coding assistants, organized across 9 independent plugins that cover the entire product lifecycle from discovery through strategy, execution, launch, growth, and even shipping AI-built code.

The key differentiator is not speed -- it is rigor. Each skill encodes a proven PM framework, from Teresa Torres' Opportunity Solution Trees to Marty Cagan's INSPIRED methodology to Alberto Savoia's pretotyping, and walks you through it step by step. The result is better product decisions, not just faster documents. Quick start commands like `/discover`, `/strategy`, `/write-prd`, `/plan-launch`, and `/north-star` give you immediate access to structured workflows.

> **Amazing:** PM Skills Marketplace packs 68 domain-specific PM skills and 42 chained workflows across 9 plugins -- covering the entire product lifecycle from discovery through strategy, execution, launch, growth, and even shipping AI-built code.

## Why PM Skills Marketplace?

The problem with generic AI for product management is simple: it gives you text, not structure. Ask an AI to "write a PRD" and you get a document. Ask it to "help me decide what to build" and you get a list. What you do not get is a framework that forces you to confront your riskiest assumptions, rank them by impact, and design the cheapest test for each one.

PM Skills Marketplace solves this by encoding proven PM frameworks as skills that AI assistants can draw on. Instead of reading "Continuous Discovery Habits" and hoping you remember to apply it, you have Teresa Torres' Opportunity Solution Tree methodology built directly into your AI workflow. The skill prompts you to identify opportunities, generate solutions, test assumptions, and iterate -- exactly the structure the book prescribes, but active in your daily work rather than sitting on a bookshelf.

The system uses progressive disclosure, which means lean frontmatter (name and description) is always loaded for context matching, while the detailed skill body is loaded only when triggered. This means all 68 skills are available without bloating the AI's context window, and the right framework surfaces automatically based on conversation relevance.

PM Skills Marketplace works with Claude Code, Cowork, Codex CLI, Gemini CLI, Cursor, OpenCode, and Kiro. Skills follow the universal SKILL.md format and work with any AI assistant that reads it. Commands (slash commands) are Claude-specific, but on other platforms you can describe the workflow steps in plain language to achieve the same result.

> **Key Insight:** PM Skills Marketplace uses progressive disclosure -- lean frontmatter is always loaded for context matching, while detailed skill bodies are loaded only when triggered. This means 68 skills are available without bloating the AI's context window, and the right framework surfaces automatically based on conversation relevance.

## Architecture -- Skills, Commands, and Plugins

PM Skills Marketplace is built on a three-layer architecture: Skills (nouns), Commands (verbs), and Plugins (packages). Each layer serves a distinct purpose and they compose together to create end-to-end PM workflows.

![PM Skills Architecture](/assets/img/diagrams/pm-skills/pm-skills-architecture.svg)

**Skills** are the foundational building blocks. They encode proven PM frameworks as domain knowledge that AI assistants can draw on. Skills are nouns -- they represent concepts, analytical frameworks, and guided workflows. Some skills are auto-loaded when relevant to the conversation (shown with dashed borders in the diagram), while others are force-loaded through explicit invocation with `/plugin-name:skill-name` or `/skill-name`. A skill like `prioritization-frameworks` can serve multiple commands, making the system composable without duplication.

**Commands** are user-triggered workflows invoked with `/command-name`. Each command chains one or more skills into an end-to-end process. For example, `/discover` chains four skills: `brainstorm-ideas`, `identify-assumptions`, `prioritize-assumptions`, and `brainstorm-experiments`. After any command completes, it suggests relevant next commands, creating a natural workflow progression that mirrors how PMs actually work.

**Plugins** are installable packages that group related skills and commands. Each covers a distinct PM domain -- from product discovery through strategy, execution, market research, analytics, go-to-market, marketing, toolkit utilities, and AI shipping. Plugins never hard-reference commands from other plugins; instead, they suggest follow-ups in natural language, ensuring each plugin works standalone. Installing the marketplace gives all 9 plugins at once.

The architecture diagram above shows how these three layers connect. The 9 plugins at the top each contain their own skills and commands. The command layer in the middle shows five example commands and how they chain skills. The skill layer at the bottom distinguishes between auto-loaded skills (dashed borders, context-matched) and force-loaded skills (solid borders, explicit invocation). The side panel shows multi-platform compatibility: skills work everywhere, while commands are Claude-specific.

## The 9 Plugins in Detail

![PM Skills Features](/assets/img/diagrams/pm-skills/pm-skills-features.svg)

### Plugin 1: pm-product-discovery (13 skills, 5 commands)

The largest plugin covers the full discovery cycle: ideation, assumption mapping, prioritization, and experiment design. Key skills include `brainstorm-ideas` (multi-perspective ideation from PM, Designer, and Engineer viewpoints), `identify-assumptions` (risky assumption mapping across Value, Usability, Viability, and Feasibility dimensions), `prioritize-assumptions` (Impact x Risk matrix), and `opportunity-solution-tree` (Teresa Torres' OST methodology). Commands include `/discover` (chains four skills into a complete discovery cycle), `/brainstorm`, `/triage-requests`, `/interview`, and `/setup-metrics`. This plugin draws heavily on Teresa Torres' Continuous Discovery Habits and Alberto Savoia's pretotyping approach.

### Plugin 2: pm-product-strategy (12 skills, 5 commands)

The strategy toolkit provides frameworks for vision, business models, pricing, and competitive landscape. Key skills include `product-strategy` (comprehensive 9-section Product Strategy Canvas), `startup-canvas`, `lean-canvas`, `business-model`, and `monetization-strategy`. It uniquely combines vision crafting with competitive analysis tools like SWOT, PESTLE, Porter's Five Forces, and the Ansoff Matrix. Commands include `/strategy`, `/business-model`, `/value-proposition`, `/market-scan`, and `/pricing`. Based on Roger Martin's Playing to Win and Strategyzer's Business Model Generation.

### Plugin 3: pm-execution (16 skills, 11 commands)

The most command-heavy plugin covers day-to-day PM work: PRDs, OKRs, roadmaps, sprints, retros, and release notes. Key skills include `create-prd` (8-section PRD template), `brainstorm-okrs` (Christina Wodtke's Radical Focus methodology), `outcome-roadmap`, `sprint-plan`, `retro`, and `pre-mortem`. The `/red-team-prd` command is particularly notable -- it adversarially stress-tests a PRD by surfacing load-bearing assumptions and ranking them by cheapest test. Commands include `/write-prd`, `/plan-okrs`, `/transform-roadmap`, `/sprint`, and `/red-team-prd`.

### Plugin 4: pm-market-research (7 skills, 3 commands)

Focuses on understanding users and markets through personas, segmentation, journey maps, and market sizing. Key skills include `user-personas`, `market-segments`, `customer-journey-map`, `market-sizing`, and `competitor-analysis`. The `/research-users` command chains persona creation, user segmentation, and journey mapping into a single workflow. Based on Anthony Ulwick's Jobs to Be Done methodology.

### Plugin 5: pm-data-analytics (3 skills, 3 commands)

The most focused plugin addresses the most common PM data needs: generating SQL from natural language, analyzing cohort retention, and evaluating A/B test results with statistical significance calculations. Key skills include `sql-queries`, `cohort-analysis`, and `ab-test-analysis`. Commands include `/write-query`, `/analyze-cohorts`, and `/analyze-test`. Based on Alistair Croll and Benjamin Yoskovitz's Lean Analytics.

### Plugin 6: pm-go-to-market (6 skills, 3 commands)

Covers the critical transition from product to market: beachhead segment identification, ideal customer profiles, growth loop design, and competitive battlecards. Key skills include `gtm-strategy`, `beachhead-segment`, `ideal-customer-profile`, `growth-loops`, and `competitive-battlecard`. Commands include `/plan-launch`, `/growth-strategy`, and `/battlecard`. Based on Maja Voje's Go-To-Market Strategist methodology.

### Plugin 7: pm-marketing-growth (5 skills, 2 commands)

Handles product marketing with positioning, value propositions, naming, and North Star metrics. Key skills include `marketing-ideas`, `positioning-ideas`, `value-prop-statements`, `product-name`, and `north-star-metric`. Commands include `/market-product` and `/north-star`. Based on Sean Ellis' Hacking Growth methodology.

### Plugin 8: pm-toolkit (4 skills, 5 commands)

Provides essential PM utilities beyond core product work: resume review with the XYZ+S formula, NDA drafting, privacy policy generation, and grammar checking. Key skills include `review-resume`, `draft-nda`, `privacy-policy`, and `grammar-check`. Commands include `/review-resume`, `/tailor-resume`, `/draft-nda`, `/privacy-policy`, and `/proofread`.

### Plugin 9: pm-ai-shipping (2 skills, 5 commands)

Designed for the age of vibe coding, this plugin documents AI-built apps, audits the gap between documented intent and actual implementation, and compiles reviewer-ready shipping packets. Key skills include `shipping-artifacts` and `intended-vs-implemented`. Commands include `/ship-check`, `/document-app`, `/derive-tests`, `/security-audit-static`, and `/performance-audit-static`.

> **Important:** The pm-ai-shipping plugin is uniquely designed for the age of vibe coding. It does not just document what an AI-built app does -- it finds the gap between what the docs say the system should do and what the code actually does, surfacing the class of bug that generic scanners miss.

## How Commands Chain Skills -- Workflow Examples

Commands are where PM Skills Marketplace truly shines. Each command chains one or more skills into an end-to-end workflow that mirrors how experienced PMs actually work, not just how textbooks describe it.

![PM Skills Workflow](/assets/img/diagrams/pm-skills/pm-skills-workflow.svg)

**Example 1: The `/discover` workflow.** When you invoke `/discover`, it chains four skills in sequence. First, `brainstorm-ideas` generates options from multiple perspectives (PM, Designer, Engineer). Then `identify-assumptions` maps risky assumptions across Value, Usability, Viability, and Feasibility dimensions. Next, `prioritize-assumptions` ranks them using an Impact x Risk matrix. Finally, `brainstorm-experiments` designs lean startup pretotypes to test the riskiest ones. The result is a complete discovery cycle in one command.

**Example 2: The `/write-prd` workflow.** Invoking `/write-prd` uses the `create-prd` skill with an 8-section PRD template that covers problem statement, user stories, success metrics, scope, dependencies, open questions, milestones, and launch criteria. After completion, it suggests follow-up commands like `/plan-okrs` and `/transform-roadmap`.

**Example 3: The `/ship-check` workflow.** The AI Shipping Kit's `/ship-check` command documents the app, wires agent context, runs security and performance audits, maps test coverage, and compiles results into a reviewer-ready shipping packet. It uses both `shipping-artifacts` and `intended-vs-implemented` skills to surface the gap between documented intent and actual implementation.

The workflow diagram above shows how these commands map to the PM lifecycle. Each phase connects to the next with suggested commands, creating a natural progression from Discovery through Strategy, Research, Execution, Analytics, GTM, Marketing, and Shipping. The pm-toolkit plugin sits alongside the main pipeline as a utility belt, available at any phase for resume review, NDA drafting, privacy policy generation, and proofreading.

## Installation and Setup

PM Skills Marketplace supports three installation methods depending on your AI assistant.

**Claude Cowork (recommended for non-developers):**

1. Open Customize (bottom-left)
2. Go to Browse plugins, then Personal, then +
3. Select "Add marketplace from GitHub"
4. Enter: `phuryn/pm-skills`

**Claude Code (CLI):**

```bash
# Step 1: Add the marketplace
claude plugin marketplace add phuryn/pm-skills

# Step 2: Install individual plugins
claude plugin install pm-toolkit@pm-skills
claude plugin install pm-product-strategy@pm-skills
claude plugin install pm-product-discovery@pm-skills
claude plugin install pm-market-research@pm-skills
claude plugin install pm-data-analytics@pm-skills
claude plugin install pm-marketing-growth@pm-skills
claude plugin install pm-go-to-market@pm-skills
claude plugin install pm-execution@pm-skills
claude plugin install pm-ai-shipping@pm-skills
```

**Codex CLI (OpenAI):**

```bash
# Step 1: Add the marketplace
codex plugin marketplace add phuryn/pm-skills

# Step 2: Install the plugins you want
codex plugin add pm-toolkit@pm-skills
# ... (same pattern for all 9 plugins)
```

**Other AI assistants (skills only):**

- Gemini CLI: Copy skill folders to `.gemini/skills/`
- OpenCode: Copy skill folders to `.opencode/skills/`
- Cursor: Copy skill folders to `.cursor/skills/`
- Kiro: Copy skill folders to `.kiro/skills/`

Note: Commands (slash commands) are Claude-specific. For other assistants, describe the workflow steps in plain language to achieve the same result.

## Design Philosophy and PM Frameworks

The design philosophy behind PM Skills Marketplace is simple but powerful: Skills are nouns (domain knowledge, frameworks), Commands are verbs (workflows that chain skills), and Plugins are packages (groups of related skills and commands).

Progressive disclosure ensures that lean frontmatter is always loaded for context matching, while detailed skill bodies are loaded only when triggered. This means 68 skills are available without bloating the AI's context window, and the right framework surfaces automatically based on conversation relevance.

No cross-plugin references means plugins are independent. Each plugin works standalone without requiring any other plugin. When a command finishes, it suggests follow-up commands in natural language rather than hard-referencing commands from other plugins.

The frameworks encoded in PM Skills Marketplace draw from 12 proven PM thought leaders:

- **Teresa Torres** -- Continuous Discovery Habits (Opportunity Solution Trees)
- **Marty Cagan** -- INSPIRED and TRANSFORMED
- **Alberto Savoia** -- The Right It (pretotyping)
- **Dan Olsen** -- The Lean Product Playbook
- **Roger L. Martin** -- Playing to Win
- **Ash Maurya** -- Running Lean
- **Strategyzer** -- Business Model Generation, Value Proposition Design
- **Christina Wodtke** -- Radical Focus (OKRs)
- **Anthony W. Ulwick** -- Jobs to Be Done
- **Alistair Croll and Benjamin Yoskovitz** -- Lean Analytics
- **Sean Ellis** -- Hacking Growth
- **Maja Voje** -- Go-To-Market Strategist

> **Takeaway:** With a single `claude plugin marketplace add phuryn/pm-skills` command, you get the rigor of 12 product management thought leaders -- from Teresa Torres' Opportunity Solution Trees to Alberto Savoia's pretotyping methodology -- built directly into your AI workflow, not sitting on a bookshelf.

## Companion Project -- PM Brain

PM Brain is a companion project that serves as a second brain for product managers. It stores plain markdown files in a folder on your laptop -- no vector database, no cloud, no agent memory tricks. Claude reads your PM Brain files before answering, writes to them after completing tasks, and sweeps them every Friday for cleanup and consolidation.

PM Brain composes naturally with PM Skills: PM Brain provides persistent context (your product, users, decisions, and notes), while PM Skills provides structured frameworks (discovery, strategy, execution workflows). Together, they give your AI assistant both the knowledge of your specific product and the methodology to work through it systematically.

The project is available at [https://github.com/phuryn/pm-brain](https://github.com/phuryn/pm-brain).

## Validation and Contributing

PM Skills Marketplace includes a built-in validator that checks the integrity of all plugins. Run `python3 validate_plugins.py` from the repository root to verify:

- Plugin manifest (`plugin.json`) required fields: name, version, description, author, keywords
- Name matches between directory and manifest
- Semantic versioning compliance
- Skill frontmatter validation (name and description required)
- Command frontmatter validation (description and argument-hint required)
- Cross-reference validation (commands referencing skills that actually exist)
- README presence and section checks

Contributions are welcome via pull requests. Follow existing patterns: skills are nouns, commands are verbs, every skill needs frontmatter with `name` and `description`, and every command needs `description` and `argument-hint`. The project is released under the MIT License.

## Conclusion

PM Skills Marketplace brings structured PM frameworks to AI assistants in a way that generic AI text generation cannot. With 68 skills, 42 workflows, and 9 plugins covering the full product lifecycle -- from discovery through strategy, execution, analytics, go-to-market, marketing, and shipping -- it encodes the rigor of proven PM methodologies directly into your daily workflow.

The system works with Claude Code, Cowork, Codex CLI, Gemini CLI, Cursor, OpenCode, and Kiro. Skills auto-load based on conversation context, and commands suggest relevant next steps after completion. Whether you are writing a PRD, running a discovery sprint, planning a go-to-market strategy, or auditing a vibe-coded app, PM Skills Marketplace gives you the framework, not just the text.

Quick start: `/discover`, `/strategy`, `/write-prd`, `/plan-launch`, `/north-star`.

**Links:**

- GitHub: [https://github.com/phuryn/pm-skills](https://github.com/phuryn/pm-skills)
- PM Brain companion: [https://github.com/phuryn/pm-brain](https://github.com/phuryn/pm-brain)
- Product Compass Newsletter: [https://www.productcompass.pm](https://www.productcompass.pm)