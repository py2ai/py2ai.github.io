---
layout: post
title: "Khazix Skills - Open Source AI Writing and Research Methodology"
description: "Khazix Skills is an open-source AI toolkit by Digital Life Khazix featuring the Horizontal-Vertical Analysis deep research framework and a WeChat long-form writing skill with a 4-layer self-check system that eliminates AI-speak."
date: 2026-04-20
header-img: "assets/img/ai-coding-frameworks/ai-coding-frameworks"
permalink: /khazix-skills-open-source-ai-writing-research-methodology/
featured-img: ai-coding-frameworks/ai-coding-frameworks
tags: [AI, Writing, Research, Skills, Claude-Code, Methodology, Open-Source]
author: "PyShine"
---

Most AI writing tools produce content that reads like it was written by AI. The sentences are perfectly structured, the transitions are mechanically smooth, and every paragraph ends with a tidy summary. Khazix Skills takes the opposite approach: it is an open-source toolkit designed to make AI-assisted writing sound like a real person -- specifically, like Digital Life Khazix, one of China's most distinctive AI content creators.

The repository contains two types of tools: lightweight Prompts that you copy-paste into any AI chat, and heavyweight Skills that follow the Agent Skills open standard and auto-load in Claude Code, OpenClaw, or Codex. Both serve one purpose: turning hard-won methodology into reusable tools.

## Toolkit Overview

![Toolkit Overview](/assets/img/diagrams/khazix-skills/khazix-toolkit-overview.svg)

The toolkit is organized into two tiers. Prompts are lightweight and portable -- you copy the Horizontal-Vertical Analysis prompt, change the research subject variable, and paste it into any model that supports Deep Research. Skills are the heavyweight version: structured instruction sets with references, scripts, and quality gates that an AI agent loads and executes autonomously. The hv-analysis skill adds automatic web research, parallel sub-agent information gathering, and PDF report generation on top of the core methodology. The khazix-writer skill encodes an entire writing philosophy with style rules, a four-layer self-check system, content methodology, and a library of style examples.

Installation is straightforward. In Claude Code, OpenClaw, or Codex, you simply tell the agent: "Install this skill: https://github.com/KKKKhazix/khazix-skills". Manual installation involves downloading the .skill package from the Releases page and placing it in the appropriate skills directory (~/.claude/skills/, ~/.openclaw/skills/, or ~/.agents/skills/).

## The Horizontal-Vertical Analysis Method

![HV Analysis Method](/assets/img/diagrams/khazix-skills/khazix-hv-analysis-method.svg)

The Horizontal-Vertical Analysis method is the centerpiece of the toolkit. Created by Khazix, it fuses Saussure's diachronic-synchronic linguistic analysis, social science longitudinal-cross-sectional research design, business school case study methodology, and competitive strategy analysis into a unified research framework. The core principle is constant: vertical axis chases time depth, horizontal axis chases breadth at the current moment, and their intersection produces insight.

The Vertical Axis (Diachronic / Longitudinal) traces the complete life history of the research subject from birth to present. It covers origin stories (what background, technology, or need gave birth to it), the birth node (exact first release date and initial positioning), the full evolution timeline (major versions, funding events, strategic pivots, team changes, crises), and decision logic at each key node (why they chose A over B, what constraints they faced, which early decisions locked in future paths). The output is 6,000 to 15,000 words in narrative story format -- not a dry chronology, but a story with causal logic and temporal texture.

The Horizontal Axis (Synchronic / Cross-sectional) takes a current-time-slice view and compares the research subject against competitors. It first judges the competitive landscape: Scenario A (no direct competitors -- analyze why, and where future competition might emerge), Scenario B (1-2 competitors -- deep-dive each one), or Scenario C (3+ competitors -- select 3-5 most representative). Comparison covers core differences (technology, business model, target users, pricing), user perspectives (real community feedback, not marketing), ecological niche analysis, and trend judgment. Output is 3,000 to 10,000 words.

The H-V Intersection is where the method delivers its unique value. It answers: how did history shape the current competitive position? What are the historical roots of each advantage and disadvantage? If you put the main competitors on their own timelines, how do their different origin stories explain their current characteristics? And based on all this, three future scenarios: most likely, most dangerous, and most optimistic. This section is 1,500 to 3,000 words and must produce genuinely new insights, not a summary of what came before.

The full report totals 10,000 to 30,000 words and is output as a beautifully typeset PDF with cover page, color-coded headings, professional typography, and source attribution. The hv-analysis skill includes a md_to_pdf.py script based on WeasyPrint that handles the entire conversion with built-in CSS for A4 layout, Chinese-English mixed fonts, and academic-quality formatting.

## The Four-Layer Self-Check System

![Four Layer Check](/assets/img/diagrams/khazix-skills/khazix-four-layer-check.svg)

The khazix-writer skill's most distinctive feature is its four-layer quality assurance system, modeled on the software testing pyramid. Each layer escalates from mechanical rule-checking to subjective human-feel judgment, and all four must pass before an article is considered complete.

**L1: Hard Rules Check (Syntax Layer)** operates like a code linter. It scans for banned words ("shuolaihuaqu" / "to put it simply", "yizhiweizhe" / "this means", "benzhishang" / "essentially"), banned punctuation (colons, em-dashes, double quotes -- replaced with commas, periods, or corner brackets), structural cliches ("firstly...secondly...lastly", "it is worth noting that"), and vague tool names ("an AI tool" instead of the specific product name). Zero hits are required to pass. Every violation must be fixed before proceeding.

**L2: Style Consistency Check (Pattern Matching Layer)** functions like unit tests. It verifies the opening starts from a specific event (not grand narrative), rhythm has long-short sentence variation (3+ consecutive similar-length sentences equals flat rhythm), at least 3 one-sentence standalone paragraphs for emphasis, colloquial expressions appear 8-10+ times throughout, and the punctuation ban is re-verified (AI tends to re-introduce banned punctuation during edits). At least 3 out of 4 items must pass.

**L3: Content Quality Check (Integration Test Layer)** examines depth and persuasiveness. Every core claim must have specific evidence (a person, scene, detail, or data point). Knowledge must be delivered as if "casually pulled out while chatting" rather than "let me now explain to you." At least one cultural/philosophical/historical elevation must connect the specific topic to a larger reference. Opposing viewpoints must be acknowledged with empathy before presenting the author's different perspective. Article-type-specific checks apply: investigation articles need "personally went and did it" narrative, product reviews need real usage scenarios, methodology articles need actionable steps with honest learning curves.

**L4: Human Feel Final Review (Personality Layer)** is the most important and most subjective layer. It asks one question: "After reading this, do I feel like a knowledgeable ordinary person is earnestly discussing something that moved them, or is an AI outputting information?" It checks temperature (body-memory emotions like "I was stunned" vs. knowledge-descriptions like "I felt very shocked"), uniqueness (could only Khazix have written this angle?), posture (peer talking to peer, not teacher lecturing student), and flow (does attention ever break?).

## The AI-Human Collaboration Workflow

![AI Human Workflow](/assets/img/diagrams/khazix-skills/khazix-ai-human-workflow.svg)

A critical design decision in khazix-writer is the explicit boundary between what AI should and should not do. The skill is a style generator, not a thinking replacement. The ideal workflow is a five-step ping-pong between human and AI.

The human provides raw material: source documents, core viewpoints, personal experiences, and emotional nodes. The AI then supplements with background knowledge, finds supporting evidence and analogies, suggests structural improvements, and expands content along agreed-upon angles. The human does a second rewrite, adding their own voice, breaking the AI's too-smooth rhythm, and inserting real details the AI could never invent. The AI runs the four-layer self-check system and outputs a quality report with specific fix suggestions. The human does final review and publishes.

What AI excels at: finding citations and evidence across history, academia, and culture; generating multiple candidate analogies for abstract concepts; expanding content along determined angles; supplementing domain knowledge (Gestalt psychology, Jungian shadow theory, causal language model principles); and suggesting structural rearrangements. What AI fails at: first-hand observations (buying a 9.9-yuan DeepSeek, paying someone 499 yuan to install OpenClaw at home, sneaking to an internet cafe at 3 AM); core creative angles (connecting "selling DeepSeek on Taobao" to "Beijing Fold"); genuine emotional expression ("I was stunned" vs. "I felt very shocked"); and empathy-driven imagination from data to a specific person's life.

## Writing Style DNA

The khazix-writer skill encodes a comprehensive writing philosophy. The core identity is "a knowledgeable ordinary person earnestly discussing something that moved them." Key style elements include: rhythm variation (sentences vary in length, paragraphs jump naturally, one-sentence paragraphs create weight); deliberate breaks in formality (repeated emphasis, mid-thought interruptions, omitted subjects, intentional vagueness); knowledge delivered as if casually remembered rather than formally introduced; private perspective (using "I also face this problem" rather than "the lesson here is"); bold judgments backed by facts; empathy for opposing viewpoints before presenting your own; emotional punctuation ("..." for trailing shock, "???" for extreme surprise, "= =" for speechless吐槽); and cultural elevation that connects specific events to larger philosophical or historical references naturally.

The absolute forbidden zone is equally detailed: no cliches ("firstly...secondly...lastly"), no over-structuring (no bullet point lists, no excessive bold, no subheadings in most articles), no banned punctuation (colons, em-dashes, double quotes), no AI-telltale phrases ("to put it simply", "what does this mean", "essentially", "in other words"), no fabricated examples, no vague tool names, and no textbook openings ("in today's era of rapid AI development").

## The Horizontal-Vertical Analysis Prompt

For those who want the research methodology without installing a full skill, the standalone prompt is available. You copy it, change one variable ("Research Subject = Hermes Agent"), and paste it into any model with Deep Research capability. The prompt encodes the complete methodology: vertical analysis instructions, horizontal analysis with the three competitive scenarios, writing style requirements (narrative-driven, not list-driven), word count guidance, and output format specifications. It works with products, companies, concepts, and people.

## Installation

**Agent install (recommended):**

```
Install this skill: https://github.com/KKKKhazix/khazix-skills
```

**Manual install:**

1. Download the .skill package from the Releases page
2. Place in your skills directory:
   - Claude Code: `~/.claude/skills/`
   - OpenClaw: `~/.openclaw/skills/`
   - Codex: `~/.agents/skills/`

**Standalone prompt:**

Copy the content of `prompts/horizontal-vertical-analysis.md`, change the research subject, and paste into any AI chat.

## The Philosophy

Khazix's account mission is "to inspire curiosity about AI." The writing style is built on the belief that in the AI era, the most scarce quality is "human feel" -- the sense that a real person with real experiences and real imperfections is talking to you. The skills are not about making AI write more; they are about making AI write less like AI and more like someone who actually did the thing they are writing about.

The toolkit is the distillation of three years of content creation and entrepreneurship in the AI space. Every rule in the four-layer check system exists because it was violated by AI output and caught during editing. Every style element exists because it is what makes Khazix's writing recognizable. The methodology is open-sourced not because it is finished, but because sharing it makes it better.

**Repository:** [github.com/KKKKhazix/khazix-skills](https://github.com/KKKKhazix/khazix-skills)  
**License:** MIT  
**Stars:** 5,400+