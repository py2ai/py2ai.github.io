---
layout: post
title: "Nuwa Skill - Distill Anyone's Thinking Into Runnable AI Skills"
description: "Nuwa extracts cognitive operating systems from public figures - mental models, decision heuristics, expression DNA - into Claude Code skills you can query. Not role-play. Cognitive architecture extraction."
date: 2026-04-20
header-img: "assets/img/ai-coding-frameworks/ai-coding-frameworks"
permalink: /nuwa-skill-distill-thinking-into-ai-skills/
featured-img: ai-coding-frameworks/ai-coding-frameworks
tags: [AI, Claude-Code, Skills, Cognitive-Frameworks, Mental-Models, Open-Source]
author: "PyShine"
---

The best thinkers who ever lived spent decades explaining how they think. Munger left the Almanack plus 40 years of shareholder letters. Feynman left complete lecture series and three autobiographies. Naval left 300 tweets and a 20-hour podcast on wealth. Taleb left five books and ten thousand public arguments. High-purity cognitive ore, sitting there.

**Nuwa Skill** is an open-source Claude Code skill that takes a single name as input and automatically distills that person's cognitive operating system into a runnable perspective skill. Not role-play. Not quote recitation. Mental model extraction.

## What Nuwa Distills

Distilling the best minds in any field requires extracting something deeper than daily work habits. Nuwa extracts five layers:

| Layer | Description |
|---|---|
| **How they speak** | Expression DNA -- tone, rhythm, word preferences |
| **How they think** | Mental models, cognitive frameworks |
| **How they judge** | Decision heuristics |
| **What they won't do** | Anti-patterns, value floor |
| **Honest limits** | What the skill genuinely cannot do |

Work habits can be conveyed through process docs. But what makes Munger and Musk reach different conclusions about the same problem is their cognitive frameworks. Nuwa extracts the cognitive operating system.

Every skill explicitly states what it cannot do: it cannot distill intuition (frameworks can be extracted, inspiration cannot), it cannot capture change (only a snapshot up to the research cutoff), and public statements do not equal true beliefs (only based on public information). A skill that does not tell you its limits is not worth trusting.

## Architecture Overview

![Nuwa Architecture](/assets/img/diagrams/nuwa-skill/nuwa-architecture.svg)

The Nuwa architecture follows a five-phase pipeline that transforms a simple name input into a fully validated perspective skill. The process begins at Phase 0 with entry routing, which determines whether the user has provided a specific person's name (direct path) or a vague need such as "I want to make better decisions" (diagnostic path). The diagnostic path uses a structured needs assessment table covering ten dimensions -- from decision-making and expression to risk management and humor -- to reverse-engineer the most suitable distillation target. Once the target is confirmed, Phase 0.5 creates the skill directory structure with research folders, source archives, and utility scripts.

Phase 1 launches six parallel research agents that simultaneously gather information across different dimensions. Phase 2 applies triple-verification extraction to filter candidate claims into validated mental models. Phase 3 constructs the SKILL.md file by assembling all extracted components into the standard template. Phase 4 runs quality validation through three independent tests, and if the skill passes, Phase 5 applies dual-agent refinement inspired by the Darwin.skill evaluation framework. If validation fails, the system loops back to Phase 2 with a maximum of two iterations, ensuring the final output is honest about its limitations rather than artificially polished.

## The Six-Agent Research Swarm

![Six Agents](/assets/img/diagrams/nuwa-skill/nuwa-six-agents.svg)

The core of Nuwa's research capability is its six-agent parallel swarm, where each agent is responsible for a distinct information dimension. Agent 1 (Writings) searches for books, essays, papers, and newsletters, extracting core arguments that appear three or more times as genuine beliefs, along with self-coined terminology and recommended book lists that reveal intellectual lineage. Agent 2 (Conversations) targets podcasts, long interviews, and AMAs, capturing how the person responds when pushed, their impromptu analogies, moments of changing stance, and questions they refuse to answer. Agent 3 (Expression DNA) analyzes social media posts for high-frequency phrases, controversial positions, humor patterns, and public debate behavior.

Agent 4 (External Views) gathers criticism, biographies, and peer comparisons to provide an outside perspective on patterns the person themselves might not recognize. Agent 5 (Decisions) researches major life choices and turning points, examining the decision logic, retrospective reflections, and consistency between stated beliefs and actual behavior. Agent 6 (Timeline) constructs a complete chronological record from birth to the present, with special emphasis on the most recent twelve months to prevent the skill from becoming outdated. Each agent writes its findings to a dedicated markdown file in the research directory, annotates information sources with credibility ratings, and preserves contradictions rather than smoothing them over. A Phase 1.5 checkpoint reviews all agent outputs for quality before proceeding.

## Triple-Verification Extraction

![Triple Verification](/assets/img/diagrams/nuwa-skill/nuwa-triple-verification.svg)

The triple-verification pipeline is what separates genuine mental models from surface-level observations. Starting with 15 to 30 candidate claims gathered by the research agents, each claim must pass three rigorous tests before being elevated to a mental model. Test 1 (Cross-Domain Recurrence) checks whether the claim appears across two or more different domains or topics -- a one-off statement does not qualify. Test 2 (Generative Power) evaluates whether the claim can predict the person's position on new questions they have not publicly addressed, meaning it has actual predictive value rather than being merely descriptive. Test 3 (Exclusivity) verifies that the claim represents thinking unique to this person, not something any smart person would say.

Claims that pass all three tests become mental models, sorted by exclusivity strength with a target of three to seven models. Claims that pass only one or two tests are downgraded to decision heuristics (five to ten total), which are fast judgment rules expressible as "if X, then Y" with supporting case evidence. Claims that pass zero tests are discarded entirely. The sorting principle is critical: three deep models are far more valuable than ten shallow principles. This methodology is documented in the extraction-framework.md reference file and ensures that every mental model in the final skill has been rigorously validated rather than casually assembled.

## Anatomy of a Perspective Skill

![Skill Anatomy](/assets/img/diagrams/nuwa-skill/nuwa-skill-anatomy.svg)

Each perspective skill produced by Nuwa is a self-contained SKILL.md file with six core components. Mental Models (three to seven) form the cognitive lenses through which the skill views any problem, each with source evidence, application methods, and stated limitations. Decision Heuristics (five to ten) provide fast judgment rules with specific scenarios and case support. Expression DNA captures the person's tone, rhythm, word preferences, humor style, and certainty patterns so the skill speaks in their voice rather than generic AI language. Values and Anti-Patterns define core value rankings alongside behaviors the person explicitly opposes, including internal tensions between values that create depth.

The Agentic Protocol is a critical differentiator that elevates the skill from mere role-play to a reliable thinking partner. It implements a three-step workflow: first classifying whether a question requires factual research or can be answered purely from frameworks, then conducting research using dimensions derived from the person's mental models (for example, Munger would examine moats, incentive structures, and historical analogies), and finally synthesizing an answer that combines real information with the person's cognitive style. The Honest Limits section explicitly states what the skill cannot do, including the research cutoff date and which dimensions lack sufficient information. Supporting the SKILL.md are the raw research files and source archives, making the entire distillation process transparent and auditable.

## Live Examples

The repository includes 13 pre-distilled person skills and 1 topic skill, each with complete research data:

**Person Skills:** Paul Graham (startups/writing), Zhang Yiming (product/globalization), Andrej Karpathy (AI/engineering), Ilya Sutskever (AI safety/scaling), MrBeast (content creation), Trump (negotiation/power), Steve Jobs (product/design), Elon Musk (engineering/first principles), Charlie Munger (investing/inversion), Richard Feynman (learning/teaching), Naval Ravikant (wealth/leverage), Nassim Taleb (risk/antifragility), Zhang Xuefeng (education/career planning).

**Topic Skill:** X Mastery Mentor (Twitter/X full-stack operations).

Here is what distilled thinking looks like in practice. When asked about wanting to do content creation, write a book, and build an indie app simultaneously, Naval's skill responds: "You've listed three desires. Each desire is a contract you signed with unhappiness. This isn't about energy -- it's about too many contracts. Ask yourself: which one makes you lose track of time? That's where your specific knowledge lives. Not choose one forever. Just one first, then one, then one. Serial compounding, not parallel exhaustion."

When asked about high SaaS customer acquisition costs, Musk's skill responds: "Don't think about how to reduce it yet. Calculate the physical minimum first. What's the minimum necessary action to acquire a customer? What's the theoretically shortest path from knowing you to paying you? How many times longer is your actual path vs. the theoretical one? If it's more than 3x, there are steps you can eliminate. Don't optimize the funnel -- question whether the funnel should exist at all."

These are not recited quotes. Naval uses his "desire as contract" mental model. Musk uses "asymptotic limit" reasoning. They are analyzing your problem through the cognitive frameworks of these minds.

## Quality Validation

Every generated skill undergoes three validation tests before delivery. The Sanity Check tests the skill against three questions the person has publicly answered, verifying directional alignment. The Edge Case Test presents one question the person has never publicly discussed, where the expected result is measured uncertainty rather than false confidence. The Voice Check evaluates a 100-word sample for recognizable expression characteristics, ensuring it does not sound like generic AI output or a patchwork of direct quotes.

The pass criteria are strict: three to seven mental models each with source evidence, explicit failure conditions for every model, expression DNA that is identifiable in 100 words, at least three specific honest limits, at least two pairs of internal tensions, and more than 50 percent primary sources. If validation fails, the system loops back to Phase 2 with a maximum of two iterations, delivering the best achievable version with clearly标注d weak dimensions rather than endlessly polishing.

## Installation and Usage

Install Nuwa with a single command:

```bash
npx skills add alchaincyf/nuwa-skill
```

Then in Claude Code, distill anyone:

```
> Distill Paul Graham
> Build a Zhang Xiaolong perspective skill
> Create a Duan Yongping skill for me
```

After creation, invoke directly:

```
> Use Munger's perspective to analyze this investment decision
> How would Feynman explain quantum computing?
> Switch to Naval, I'm torn between three things
```

Nuwa also supports vague needs. Say "I want to make better decisions" or "I need a thinking advisor" and it will diagnose your need dimension, recommend two to three candidates from its existing skills or new distillation targets, and let you choose.

## The Darwin Connection

Nuwa creates skills, and **Darwin.skill** makes them evolve. Inspired by Karpathy's autoresearch, Darwin uses autonomous experiment loops to batch-optimize all skills with eight-dimension evaluation, a ratchet mechanism that only keeps improvements and automatically rolls back regressions, and independent sub-agent scoring. Nuwa's Phase 5 dual-agent refinement already incorporates Darwin's evaluation system, which is one reason Nuwa-generated skills achieve high quality.

```bash
npx skills add alchaincyf/darwin-skill
```

## The Philosophy

Nuwa is named after the Chinese goddess who created humans from clay. Here the clay is public information, and what is created is not a person -- it is a mirror. A good perspective skill lets you view your own problems through someone else's eyes. Not to imitate them, but to expand your own thinking boundaries.

colleague-skill distills what a person does. Nuwa distills how a person thinks. The next person you want to distill does not have to be a colleague.

**Repository:** [github.com/alchaincyf/nuwa-skill](https://github.com/alchaincyf/nuwa-skill)  
**License:** MIT  
**Stars:** 12,000+