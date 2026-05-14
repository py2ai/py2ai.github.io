---
layout: post
title: "Academic Research Skills: Supercharging Claude Code for Research"
description: "Academic Research Skills for Claude Code is a 35-agent, 25-mode suite covering research to publication with integrity gates, Socratic mentoring, and anti-sycophancy protocols."
date: 2026-05-14
header-img: "img/post-bg.jpg"
permalink: /Academic-Research-Skills-Claude-Code/
featured-img: ai-coding-frameworks/ai-coding-frameworks
categories: [AI, Python, Research]
tags: [Academic Research Skills, Claude Code, AI research, Python, open source, research tools, how to use, setup guide, tutorial, academic writing]
keywords: "how to use Academic Research Skills, Academic Research Skills tutorial, Academic Research Skills Claude Code, Academic Research Skills vs alternatives, Academic Research Skills installation guide, open source AI research tools, Claude Code research skills setup, best academic research AI tools, Academic Research Skills for beginners, AI-powered academic research"
author: "PyShine"
---

## From Research Question to Published Paper -- Inside the 35-Agent Suite That Guards Academic Integrity

Academic Research Skills (ARS) for Claude Code is an open-source suite of four interconnected skills that transform Anthropic's Claude Code CLI into a full-spectrum academic research copilot. With 35 specialized agents distributed across 25 modes, ARS covers every stage from initial research question formulation through peer review to final publication -- and it does so with mandatory human-in-the-loop checkpoints, anti-sycophancy protocols, and integrity verification gates that cannot be skipped. The project has amassed over 7,000 GitHub stars and continues to grow at nearly 2,000 stars per week, signaling a genuine need in the research community for AI-assisted workflows that respect human agency.

> **Key Insight**: ARS is explicitly designed as a human-in-the-loop tool, not an autonomous paper-writing system. Every pipeline stage ends with a user confirmation checkpoint, and the two integrity gates (Stage 2.5 and Stage 4.5) are mandatory -- they cannot be bypassed even if you want to skip them. This design choice is grounded in research by Lu et al. (2026, *Nature* 651:914-919), whose fully autonomous AI Scientist system published papers through blind peer review but exhibited failure modes including hallucinated results, shortcut reliance, and frame-lock.

## Architecture: Four Skills, One Pipeline

The ARS suite is organized as four Claude Code skills that can operate independently or be orchestrated through a 10-stage pipeline. The architecture diagram below shows how these skills relate to each other and to the shared infrastructure that underpins them.

![Academic Research Skills Architecture Overview](/assets/img/diagrams/academic-research-skills/academic-research-skills-architecture.svg)

The four skills are:

- **deep-research** (v2.9.2) -- 13 agents, 7 modes for research exploration, literature review, systematic review, fact-checking, and Socratic guided research
- **academic-paper** (v3.1.1) -- 12 agents, 10 modes for paper writing, revision, citation checking, format conversion, and AI disclosure generation
- **academic-paper-reviewer** (v1.9.0) -- 7 agents, 6 modes for multi-perspective peer review with 0-100 quality rubrics and Devil's Advocate challenges
- **academic-pipeline** (v3.7.0) -- the orchestrator that coordinates all three skills through a 10-stage workflow with integrity gates and collaboration evaluation

Each skill connects to shared resources including handoff schemas (Schema 9 Material Passport), ground truth isolation patterns, benchmark report schemas, sprint contracts, and cross-model verification protocols. The pipeline orchestrates the other three skills, dispatching the appropriate skill and mode at each stage while tracking state through the Material Passport.

## 25 Modes Across Three Spectrums

ARS offers 25 distinct modes categorized along a fidelity-originality spectrum. Fidelity modes produce template-heavy, predictable output. Balanced modes use default settings. Originality modes are exploratory and template-light, giving the user more room for creative thinking. The following diagram breaks down every mode across all four skills.

![Academic Research Skills Features Breakdown](/assets/img/diagrams/academic-research-skills/academic-research-skills-features.svg)

The mode spectrum is not just a labeling exercise -- it determines how much oversight the user has. Originality modes like `socratic` and `plan` have "Very High" oversight, meaning the user is actively guiding the process through dialogue. Fidelity modes like `citation-check` and `format-convert` have "Low" oversight because they perform mechanical, template-driven tasks that require minimal human input.

**Deep Research** offers seven modes. The standout is `socratic` mode, which uses intent-based activation to detect when a user wants guided thinking rather than a direct answer. It classifies user intent as exploratory versus goal-oriented, disables auto-convergence in exploratory mode, and includes a Dialogue Health Indicator that silently checks for persistent agreement, conflict avoidance, and premature convergence every five turns.

**Academic Paper** offers ten modes, from full paper writing to specialized tasks like citation checking and AI disclosure statement generation. The `plan` mode walks users through paper structure via Socratic dialogue, while `revision-coach` mode parses unstructured reviewer comments into a structured Revision Roadmap.

**Academic Paper Reviewer** offers six modes including a `calibration` mode that measures the reviewer's false negative rate (FNR), false positive rate (FPR), and balanced accuracy against a user-supplied gold set. This is not a tool that claims infallibility -- it offers a way to measure its own limitations.

## The 10-Stage Pipeline: Research to Publication

The academic-pipeline orchestrator manages a 10-stage workflow from initial research through final publication. The pipeline includes two mandatory integrity gates, Socratic coaching sub-stages, and a collaboration quality evaluation at the end. The workflow diagram below illustrates the complete pipeline flow.

![Academic Research Skills Pipeline Workflow](/assets/img/diagrams/academic-research-skills/academic-research-skills-workflow.svg)

The stages are:

1. **RESEARCH** -- deep-research generates an RQ Brief, Methodology Blueprint, Annotated Bibliography, and Synthesis Report
2. **WRITE** -- academic-paper produces a complete draft with bilingual abstracts and citation compliance
3. **2.5 INTEGRITY** -- mandatory gate with a 7-mode AI failure checklist (implementation bugs, hallucinated citations, hallucinated results, shortcut reliance, bug-as-insight reframing, methodology fabrication, frame-lock)
4. **REVIEW** -- academic-paper-reviewer produces 5 review reports (EIC + 3 field-adaptive reviewers + Devil's Advocate) with an editorial decision
5. **3 to 4 Revision Coaching** -- optional Socratic dialogue (max 8 rounds) to plan revision strategy
6. **REVISE** -- academic-paper revises the draft with point-by-point responses
7. **3' RE-REVIEW** -- verification review with a narrow team checking that revisions were actually made
8. **3' to 4' Residual Coaching** -- optional Socratic dialogue on residual issues (max 5 rounds)
9. **4' RE-REVISE** -- final revision with content frozen (no further review loops)
10. **4.5 FINAL INTEGRITY** -- zero-tolerance re-verification of all claims and citations
11. **FINALIZE** -- output in Markdown, DOCX (via Pandoc), LaTeX, or PDF (via tectonic)
12. **PROCESS SUMMARY** -- auto-generated paper creation process record with 6-dimension Collaboration Quality Evaluation

> **Important**: The pipeline enforces a hard cap of 2 revision loops total across Stages 4 and 4'. After that, remaining issues become "Acknowledged Limitations" rather than being silently resolved. This prevents infinite revision cycles and forces honest documentation of what could not be fixed.

## Data Access Levels and Integrity Architecture

ARS implements a three-tier data access model that mirrors how trust increases through the pipeline. Raw data enters at Stage 1, gets sanitized and verified at Stage 2, and only verified data reaches the review and final integrity stages. The ecosystem diagram below shows how data flows through the pipeline and connects to external tools.

![Academic Research Skills Ecosystem](/assets/img/diagrams/academic-research-skills/academic-research-skills-ecosystem.svg)

The three data access levels are:

- **RAW** (deep-research) -- arbitrary, possibly adversarial input. The research skill handles unverified web sources, user-provided PDFs, and raw search results
- **REDACTED** (academic-paper) -- sanitized material with no new raw ingestion. The writing skill operates on verified sources and structured outlines
- **VERIFIED_ONLY** (academic-paper-reviewer, academic-pipeline) -- only data that has passed through integrity gates. Reviewers never see unverified claims

The two integrity gates are the enforcement mechanism. Stage 2.5 runs a 7-mode failure checklist derived from Lu et al.'s analysis of autonomous AI research failure modes. Stage 4.5 performs a deeper re-verification with zero tolerance -- any mode that was SUSPECTED at Stage 2.5 must be CLEAR or explicitly user-overridden by Stage 4.5.

## Anti-Sycophancy: The Core Innovation

The most distinctive feature of ARS is its systematic approach to AI sycophancy -- the tendency of language models to agree with users rather than challenge them. Three mechanisms work together:

**Devil's Advocate Concession Threshold Protocol** -- When the Devil's Advocate reviewer challenges a thesis, it must score every rebuttal on a 1-5 scale before responding. Concession is only allowed at score 4 or above (the rebuttal directly addresses the core attack with evidence). At score 3 or below, the DA holds its position. No consecutive concessions are allowed, and concession rate is tracked throughout the pipeline.

**Intent Detection Layer** -- The Socratic Mentor classifies user intent as exploratory versus goal-oriented at dialogue start and re-assesses every 3 turns. In exploratory mode, auto-convergence is disabled, max rounds are raised to 60, and "want me to summarize?" prompts are prohibited. The user decides when to stop, not the AI.

**Dialogue Health Indicator** -- Every 5 turns, the Socratic Mentor silently self-assesses on three dimensions: persistent agreement, conflict avoidance, and premature convergence. When an agreement pattern is detected, challenging questions are auto-injected. This is invisible to the user to prevent gaming.

> **Takeaway**: These mechanisms do not eliminate AI sycophancy -- they make it visible and manageable. The DA will still eventually concede if pushed hard enough. The Socratic Mentor will still have some convergence bias. But now there are explicit checkpoints that slow down the sycophancy, force the DA to justify concessions, and prevent the Mentor from wrapping up before the user is ready.

## Installation and Quick Start

ARS installs in under 30 seconds using Claude Code's plugin system (v3.7.0+):

```text
/plugin marketplace add Imbad0202/academic-research-skills
/plugin install academic-research-skills
```

Alternatively, you can clone and symlink:

```bash
# Clone the repository
git clone https://github.com/Imbad0202/academic-research-skills.git ~/academic-research-skills

# Install each skill into your project
cd /path/to/your/project
mkdir -p .claude/skills
ln -s ~/academic-research-skills/deep-research .claude/skills/deep-research
ln -s ~/academic-research-skills/academic-paper .claude/skills/academic-paper
ln -s ~/academic-research-skills/academic-paper-reviewer .claude/skills/academic-paper-reviewer
ln -s ~/academic-research-skills/academic-pipeline .claude/skills/academic-pipeline
```

**Prerequisites:**

- Claude Code (latest version; plugin packaging requires recent builds)
- `ANTHROPIC_API_KEY` environment variable set
- Optional: Pandoc for DOCX export, tectonic + Source Han Serif TC for APA 7.0 PDF

Once installed, try these commands:

```text
# Start a Socratic research dialogue
/ars-plan "I want to explore AI's impact on higher education quality assurance"

# Quick literature review
/ars-lit-review "declining birth rates and private universities in Taiwan"

# Full pipeline from scratch
"I want to produce a complete research paper about how agentic AI reshapes student learning outcome measurement"

# Review an existing paper
/ars-full "Review this paper" (then paste or attach the paper)
```

## The Material Passport: Cross-Session Continuity

One of ARS's most sophisticated features is the Material Passport (Schema 9), a structured YAML document that tracks all pipeline state across sessions. With the v3.6.3 opt-in `ARS_PASSPORT_RESET=1` flag, every FULL checkpoint becomes a context-reset boundary, allowing you to resume a pipeline in a fresh Claude Code session using `resume_from_passport=<hash>`.

The passport includes:

- Research question brief and methodology blueprint
- Annotated bibliography with source verification status
- Paper configuration record (type, discipline, citation format, language)
- Style profile from Style Calibration
- Claim verification reports from integrity gates
- Score trajectory tracking across revision rounds
- Collaboration depth history from the observer agent
- Literature corpus entries with provenance tracking

This means you can start a research project, close your laptop, and pick up exactly where you left off in a new session -- with all context preserved in the passport rather than relying on the LLM's context window.

## Cross-Model Verification

ARS supports optional cross-model verification through the `ARS_CROSS_MODEL` environment variable. When enabled, integrity sample checks and Devil's Advocate critiques are run on a second model (GPT-5.4 Pro or Gemini 3.1 Pro) to provide independent verification. The Collaboration Depth Observer also supports cross-model comparison, flagging dimension disagreements greater than 2 points rather than silently averaging them.

This is not a claim that cross-model verification eliminates bias -- it is an acknowledgment that single-model verification has structural limitations. By running checks on a different model with different training data and different failure modes, ARS provides a second perspective that can catch errors the primary model misses.

## Supported Outputs and Citation Formats

ARS supports five citation formats (APA 7.0, Chicago, MLA, IEEE, Vancouver) and six paper structures (IMRaD, Thematic Literature Review, Theoretical Analysis, Case Study, Policy Brief, Conference Paper). Output formats include:

- **Markdown** -- always available, no dependencies
- **DOCX** -- via Pandoc when installed
- **LaTeX** -- APA 7.0 `apa7` document class with XeCJK for bilingual support
- **PDF** -- via tectonic compilation

Bilingual abstracts (Traditional Chinese + English) are generated automatically when Chinese content is detected, and the entire pipeline supports both English and Traditional Chinese input with intent-based mode activation that works in any language.

## The v3.0 Origin Story

The anti-sycophancy features in v3.0 were not designed in a vacuum. They were discovered through a real dialectic experiment where the author used ARS to write a reflection article about AI in higher education. Three structural problems emerged:

1. **Frame-lock** -- The Devil's Advocate debated four rounds against the thesis, but every round stayed inside the frame the user set. It attacked arguments, never premises. It never asked "are we even discussing the right question?"

2. **Sycophancy under pushback** -- Every time the user challenged the DA's attacks, it conceded too quickly. It retracted findings faster than it launched them. The model's training rewards conversational harmony, so "the user pushed back" was treated as evidence that the attack was wrong.

3. **Intent misdetection** -- The Socratic Mentor kept trying to converge and produce deliverables when the user was still exploring. It could not distinguish "the user wants a deep philosophical discussion" from "the user wants an RQ brief."

These discoveries led directly to the Concession Threshold Protocol, Intent Detection Layer, and Dialogue Health Indicator that now ship as core features.

> **Amazing**: The post-publication audit of ARS's own output found 21 issues out of 68 references (a 31% error rate) that passed three rounds of integrity checks. This is not hidden -- it is documented in the project's showcase as proof that external verification is necessary even after internal integrity gates. The tool is honest about its own limitations.

## Sprint Contracts and Generator-Evaluator Separation

v3.6.2 introduced Schema 13 Sprint Contracts, which force reviewers to pre-commit their scoring plan before reading the paper. This is a two-phase protocol:

- **Phase 1** (paper-content-blind) -- Each reviewer commits their scoring rubric, acceptance dimensions, and failure conditions without seeing the paper
- **Phase 2** (paper-visible) -- Reviewers then score the paper against their pre-committed plan, with the `<phase1_output>` data delimiter preventing self-injection

The `editorial_synthesizer_agent` runs a three-step mechanical protocol: build cross-reviewer matrix, evaluate each failure condition with panel-relative quantifiers, and resolve precedence by severity. A forbidden-operations list prevents post-hoc rubric edits.

This design prevents the common failure mode where reviewers adjust their criteria after seeing the paper to justify a predetermined conclusion -- a form of motivated reasoning that is particularly insidious because it feels like legitimate judgment.

## Literature Corpus Integration

v3.6.4 introduced the `literature_corpus[]` input port on the Material Passport, allowing users to bring their own curated literature from Zotero, Obsidian, or filesystem folders. Three reference Python adapters ship with ARS:

- `folder_scan.py` -- reads a filesystem folder of PDFs
- `zotero.py` -- reads Better BibTeX JSON exports
- `obsidian.py` -- reads Obsidian vault frontmatter

The consumer-side flow (v3.6.5) follows four Iron Rules:

1. **Same criteria** -- Apply identical inclusion/exclusion criteria to corpus entries and external database results
2. **No silent skip** -- Any skipped corpus entry is recorded with a reason
3. **No corpus mutation** -- Consumer agents never modify or backfill into the corpus
4. **Graceful fallback** -- On parse failure, emit `[CORPUS PARSE FAILURE]` and fall back to external-DB-only flow

## Plugin Packaging and Slash Commands

v3.7.0 introduced Claude Code plugin packaging, making installation a one-line command. The plugin ships 10 slash commands mapped to the mode registry:

| Command | Mode | Model |
|---------|------|-------|
| `/ars-plan` | deep-research socratic | opus |
| `/ars-full` | deep-research full | sonnet |
| `/ars-lit-review` | deep-research lit-review | sonnet |
| `/ars-outline` | academic-paper outline-only | sonnet |
| `/ars-abstract` | academic-paper abstract-only | sonnet |
| `/ars-revision` | academic-paper revision | sonnet |
| `/ars-revision-coach` | academic-paper revision-coach | opus |
| `/ars-citation-check` | academic-paper citation-check | sonnet |
| `/ars-disclosure` | academic-paper disclosure | sonnet |
| `/ars-format-convert` | academic-paper format-convert | sonnet |

Model routing is pinned in each command's frontmatter -- `opus` for `full` and `revision-coach` (which require architectural depth and review-interpretation depth), `sonnet` for the other 8. No Haiku per project policy.

## Performance and Cost

A full 15,000-word paper through the complete 10-stage pipeline costs approximately $4-6 in API costs and takes 2-4 hours of collaborative work. The pipeline is designed for human-AI collaboration, not autonomous generation -- the human decides at every gate, and the AI handles the grunt work of literature search, citation formatting, data verification, and logical consistency checking.

## Comparison with Alternatives

| Feature | ARS | Generic AI Chat | Autonomous Paper Generators |
|---------|-----|-----------------|---------------------------|
| Human-in-the-loop | Mandatory checkpoints | None | Minimal |
| Integrity gates | 2 mandatory (2.5 + 4.5) | None | Varies |
| Anti-sycophancy | Concession threshold + intent detection + dialogue health | None | None |
| Cross-model verification | Optional (ARS_CROSS_MODEL) | None | Rare |
| Citation verification | Semantic Scholar API + Levenshtein matching | None | Varies |
| Sprint contracts | Pre-commitment scoring (Schema 13) | None | None |
| Collaboration evaluation | 6-dimension rubric at pipeline completion | None | None |
| License | CC BY-NC 4.0 (non-commercial) | Varies | Varies |

> **Important**: ARS is licensed under CC BY-NC 4.0, which restricts commercial use. It is designed for academic researchers, students, and non-commercial scholarly work. Commercial use requires separate licensing from the maintainer.

## Getting Started

The fastest way to start is with the Socratic mode, which helps you clarify your research question before committing to a full pipeline:

```text
# In Claude Code, after installing the plugin:
"I have a vague idea about AI's impact on higher education quality assurance,
 but I'm not sure how to frame the research question. Can you guide me?"
```

ARS will enter Socratic mode, asking questions to help you clarify your thinking rather than giving you answers directly. After 5-15 rounds of dialogue, you will have a focused research question and methodology direction.

For a complete pipeline run:

```text
"I want to write a research paper on AI's impact on higher education QA"
```

This triggers the full 10-stage pipeline. Budget approximately $4-6 in API costs and 2-4 hours of collaborative work.

## Conclusion

Academic Research Skills for Claude Code represents a thoughtful approach to AI-assisted research that prioritizes integrity over convenience. Its 35 agents and 25 modes cover the full research-to-publication pipeline, but the real innovation is in the guardrails: mandatory integrity gates, anti-sycophancy protocols, sprint contracts that prevent motivated reasoning, and a collaboration depth observer that evaluates the quality of human-AI interaction. The project's willingness to document its own 31% reference error rate in the post-publication audit is perhaps the strongest endorsement of its design philosophy -- a tool that is honest about its limitations is more trustworthy than one that claims infallibility.

The project is actively maintained with frequent updates (v3.7.0 as of May 2026), a clear changelog, and comprehensive documentation including architecture docs, setup guides, and performance benchmarks. Whether you are a graduate student writing your first paper or an experienced researcher looking for an AI copilot that respects your agency, ARS provides a structured, integrity-first workflow that keeps you in the pilot seat.

---

**Links:**

- GitHub: [https://github.com/Imbad0202/academic-research-skills](https://github.com/Imbad0202/academic-research-skills)
- Architecture Documentation: [docs/ARCHITECTURE.md](https://github.com/Imbad0202/academic-research-skills/blob/main/docs/ARCHITECTURE.md)
- Setup Guide: [docs/SETUP.md](https://github.com/Imbad0202/academic-research-skills/blob/main/docs/SETUP.md)
- Performance Guide: [docs/PERFORMANCE.md](https://github.com/Imbad0202/academic-research-skills/blob/main/docs/PERFORMANCE.md)
- License: CC BY-NC 4.0