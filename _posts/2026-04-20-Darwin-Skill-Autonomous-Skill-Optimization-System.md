---
layout: post
title: "Darwin Skill - Autonomous Skill Optimization System"
description: "An autonomous evolution system that applies ML training loops to optimize AI agent skills, using an 8-dimension rubric, ratchet mechanism, and independent scoring to ensure skills only get better."
date: 2026-04-20
header-img: "assets/img/ai-coding-frameworks/ai-coding-frameworks"
permalink: /2026/04/20/darwin-skill-autonomous-skill-optimization/
featured-img: ai-coding-frameworks/ai-coding-frameworks
tags: [AI, Agent-Skills, Optimization, AutoResearch, Skill-Evolution, LLM]
author: "PyShine"
---

## Introduction

Darwin Skill is an autonomous skill optimization system that applies machine learning training loops to the problem of improving AI agent skills. Unlike persona distillation approaches such as Nuwa or zhangxuefeng, which focus on extracting and encoding human expertise into static skill files, darwin-skill treats the skill itself as a trainable asset that can be iteratively improved through evaluation, modification, and validation cycles.

The core insight is deceptively simple: **"Optimize your Agent Skills the way you train models."** In ML, you define a loss function, propose parameter updates, measure whether the update improves the objective, and keep only the changes that help. Darwin Skill applies this exact pattern to SKILL.md files -- define a rubric (the loss function), propose edits (the parameter update), re-score (the validation step), and keep or revert (the ratchet mechanism).

The project draws direct inspiration from Andrej Karpathy's autoresearch concept, where an agent writes goals and constraints in `program.md`, generates and tests code deltas indefinitely, and keeps only what measurably improves the objective. Darwin Skill adapts this autonomous experimentation loop to the domain of skill optimization, adding human-in-the-loop checkpoints and a dual evaluation mechanism that goes beyond pure structural analysis.

## The Core Problem

Most AI agent skills are written once and never improved. A developer crafts a SKILL.md file, deploys it, and moves on. There is no feedback loop. No quantifiable quality metric. No systematic process for identifying weaknesses and addressing them. Over time, skills accumulate technical debt -- vague instructions, missing edge cases, insufficient checkpoint design -- but nobody goes back to fix them because there is no objective way to measure whether a change actually helps.

This is precisely the problem that ML training loops solved for model weights. Before gradient descent with validation, model improvement was ad hoc: tweak some parameters, check a few examples, hope for the best. The introduction of quantifiable objectives, systematic evaluation, and ratchet mechanisms (only keep improvements) transformed model development from guesswork into engineering. Darwin Skill brings the same transformation to skill optimization. The 8-dimension rubric provides the quantifiable objective. The optimization loop provides the systematic evaluation. The git-based ratchet ensures that skills only get better, never worse.

## The Optimization Loop

![Darwin Skill Optimization Loop](/assets/img/diagrams/darwin-skill/darwin-skill-optimization-loop.svg)

The optimization loop is the engine that drives darwin-skill. It follows a strict iterative cycle: **Evaluate >>> Improve >>> Test >>> Human Confirm >>> Keep/Revert**. Each iteration targets the single weakest dimension identified in the previous evaluation, proposes a concrete improvement, and then re-scores the entire skill to determine whether the change should be preserved.

The ratchet mechanism is the critical safeguard. After each improvement attempt, the skill is re-evaluated using the full 8-dimension rubric. If the new total score is strictly greater than the previous score, the change is kept and committed to the git branch. If the score fails to improve, the change is reverted via `git revert` (not `git reset --hard`, which would destroy history). This ensures that the skill file only moves forward in quality, never backward. Failed experiments are not discarded silently -- they are logged in `results.tsv` with their scores and the dimension that was targeted, providing a record of what was tried and why it did not work.

The human-in-the-loop checkpoint occurs after each skill completes its optimization rounds. The system presents a diff of all changes, the score deltas for each dimension, and (when available) a comparison of test prompt outputs before and after the optimization. The user can approve the changes, reject them (triggering a full rollback), or request additional rounds. This checkpoint is essential because skill quality is more nuanced than a single loss value -- a change that improves the rubric score might introduce subtle issues that only human judgment can detect. The maximum number of optimization rounds defaults to 3 per skill, but this limit can be extended if the user chooses to continue.

## 8-Dimension Evaluation Rubric

![Darwin Skill Rubric Dimensions](/assets/img/diagrams/darwin-skill/darwin-skill-rubric-dimensions.svg)

The rubric is divided into two categories: **Structure** (60 points) and **Effectiveness** (40 points), for a maximum total of 100 points. Each dimension is scored on a 1-10 scale and then multiplied by its weight, with the final score computed as the sum of all weighted dimension scores divided by 10.

**Structure dimensions (60 pts)** evaluate the skill file through static analysis -- no execution required:

- **Frontmatter Quality (8 pts):** Checks whether the YAML frontmatter has a properly formatted `name` field, a `description` that covers what the skill does, when to use it, and trigger keywords, and whether the description stays within the 1024-character limit.
- **Workflow Clarity (15 pts):** The highest-weighted structure dimension. Evaluates whether the skill's steps are clearly numbered, explicitly actionable, and each step has defined inputs and outputs. Vague instructions like "process the data" score low; specific instructions like "read the CSV from `{input_path}`, extract columns X and Y, write to `{output_path}`" score high.
- **Boundary Condition Coverage (10 pts):** Assesses whether the skill handles exceptions, provides fallback paths, and includes error recovery instructions. A skill that only describes the happy path scores poorly here.
- **Checkpoint Design (7 pts):** Evaluates whether critical decision points include user confirmation steps. This prevents autonomous agents from making irreversible decisions without oversight.
- **Instruction Specificity (15 pts):** Tied for highest structure weight. Measures whether instructions are concrete enough to execute directly, with specific parameters, formats, and examples. Abstract guidance scores low; copy-paste-ready instructions score high.
- **Resource Integration (5 pts):** Checks whether referenced scripts, assets, and file paths actually exist and are reachable. The lowest-weighted dimension because broken references are easy to fix but rarely the primary quality issue.

**Effectiveness dimensions (40 pts)** require actual execution:

- **Overall Architecture (15 pts):** Evaluates whether the skill's structure is well-organized, not redundant, not missing critical sections, and consistent with the broader ecosystem conventions.
- **Live Test Performance (25 pts):** The single highest-weighted dimension in the entire rubric. This is what distinguishes darwin-skill from pure structural review. The skill is executed against 2-3 test prompts designed for its typical use cases, and the output quality is compared against a baseline (the same prompts run without the skill). The scoring considers whether the output fulfills the user's intent, whether the skill provides a meaningful improvement over the baseline, and whether the skill introduces any negative side effects such as excessive verbosity or format distortion.

The reason live test performance carries the most weight is simple: a skill that scores perfectly on structure but produces worse outputs than having no skill at all is actively harmful. The effectiveness dimension ensures that structural improvements translate into real-world performance gains.

## 5-Phase Lifecycle

![Darwin Skill Lifecycle Phases](/assets/img/diagrams/darwin-skill/darwin-skill-lifecycle-phases.svg)

The darwin-skill lifecycle consists of five phases, numbered with an unusual but deliberate scheme: Phase 0, Phase 0.5, Phase 1, Phase 2, and Phase 3.

**Phase 0 -- Initialize:** The system scans the project for SKILL.md files (either all skills under `.claude/skills/*/SKILL.md` or a user-specified subset), creates a git branch named `auto-optimize/YYYYMMDD-HHMM`, and initializes the `results.tsv` tracking file if it does not already exist. The branch-based approach ensures that all changes are isolated from the main working tree and can be cleanly merged or discarded.

**Phase 0.5 -- Design Test Prompts:** Before any evaluation can happen, test prompts must be designed for each skill. This phase reads each SKILL.md, understands what the skill does, and creates 2-3 test prompts covering the most typical usage scenario (the happy path) and a slightly more complex or ambiguous scenario. These prompts are saved to `test-prompts.json` in each skill's directory. The unusual "0.5" numbering reflects that this step precedes evaluation but comes after initialization -- it is a prerequisite, not a full phase. The quality of test prompts directly determines whether the optimization loop moves in the right direction, so the system presents all prompts to the user for confirmation before proceeding.

**Phase 1 -- Baseline Evaluation:** Each skill is scored across all 8 dimensions. Structure dimensions (1-7) are evaluated by the main agent through static analysis. The effectiveness dimension (8) requires spawning independent sub-agents: one executes the test prompts with the skill loaded, another executes the same prompts without it (baseline). The outputs are compared and scored. If sub-agents are unavailable due to timeout or environment constraints, a dry-run verification is performed instead -- the agent reads the skill and mentally simulates executing a typical prompt, then scores based on whether the flow seems reasonable. Dry-run evaluations are marked in `results.tsv` with `eval_mode=dry_run`. After all baselines are computed, a summary table is presented and the system pauses for user confirmation.

**Phase 2 -- Optimization Loop:** Skills are sorted from lowest to highest baseline score, so the weakest skills receive attention first. For each skill, the loop identifies the lowest-scoring dimension, proposes a specific improvement targeting that dimension, edits the SKILL.md, commits the change, re-evaluates using independent scoring, and decides whether to keep or revert. If the score improves, the change is kept and the next weakest dimension is targeted. If the score does not improve, the change is reverted and the system moves to the next skill. An optional Phase 2.5 (exploratory rewrite) can be triggered when hill-climbing gets stuck at a local optimum -- the skill is rewritten from scratch rather than incrementally tweaked.

**Phase 3 -- Summary Report:** A comprehensive report is generated showing the number of skills optimized, total experiments run, improvements kept, reverts performed, and a before/after score table with deltas for each skill. This report provides the evidence base for the user's final decision on whether to merge the optimization branch.

## AutoResearch Concept Mapping

![Darwin Skill AutoResearch Mapping](/assets/img/diagrams/darwin-skill/darwin-skill-autoresearch-mapping.svg)

The mapping between Karpathy's autoresearch and darwin-skill is not metaphorical -- it is structural. Each component in the autoresearch loop has a direct counterpart in the skill optimization system.

**program.md >>> SKILL.md (evaluation criteria):** In autoresearch, `program.md` defines the goals, constraints, and evaluation criteria that guide the agent's experimentation. In darwin-skill, the SKILL.md file itself (specifically, the rubric and constraint rules defined within the darwin-skill system) serves this role. The rubric specifies what "better" means across 8 dimensions. The constraint rules specify what changes are allowed -- no altering the skill's core purpose, no introducing new dependencies, only one dimension per round, and the optimized file must not exceed 150% of the original size.

**train.py >>> target SKILL.md (the asset being optimized):** In autoresearch, `train.py` is the code that gets modified and tested. In darwin-skill, each target SKILL.md is the asset being optimized. The skill file is the "model weights" -- the thing that changes from one iteration to the next, with each change evaluated against the objective.

**val_bpb >>> 8-dimension weighted score (quantifiable objective):** In autoresearch, validation bits-per-byte (`val_bpb`) is the single number that determines whether an experiment succeeded. In darwin-skill, the 8-dimension weighted score (maximum 100 points) serves the same function. The score must strictly improve for a change to be kept -- ties and regressions are both rejected.

**git ratchet >>> keep/revert mechanism:** Both systems use git as the ratchet mechanism. Only commits that improve the objective are retained. Failed experiments are reverted, creating a clean history where every commit represents a genuine improvement. Darwin-skill uses `git revert` rather than `git reset --hard` to preserve the full experiment history for debugging and analysis.

**test set >>> test-prompts.json:** In autoresearch, the test set provides the ground truth for evaluation. In darwin-skill, `test-prompts.json` provides the test cases for the live performance dimension. The quality and coverage of these test prompts directly determines the reliability of the effectiveness evaluation.

The key difference between the two systems is the human-in-the-loop requirement. Autoresearch is fully autonomous -- the agent runs indefinitely, keeping improvements and discarding failures without human intervention. Darwin-skill inserts human checkpoints after each skill's optimization completes, because skill quality involves subjective judgments that a numerical score cannot fully capture. A change might improve the rubric score while subtly shifting the skill's behavior in ways that only a human reviewer would notice.

## Independent Scoring Principle

One of the most important design decisions in darwin-skill is the separation of editing and scoring agents. **The agent that edits a skill is never the agent that scores it.** When the main agent proposes an improvement and modifies a SKILL.md, the re-evaluation of effectiveness is performed by an independent sub-agent that has no context about what was changed or why.

This principle is directly analogous to the separation of training and validation sets in machine learning. If you evaluate a model on the same data you trained it on, you get optimistic bias -- the model appears to perform better than it actually does because it has memorized the training examples. Similarly, if the same agent that made an edit then evaluates whether the edit helped, it is inherently biased toward confirming its own judgment. The sub-agent approach eliminates this self-grading bias.

When sub-agents are unavailable, the system falls back to dry-run verification rather than skipping the effectiveness dimension entirely. A dry run is less reliable than a full test but still provides more signal than ignoring effectiveness altogether. Every evaluation in `results.tsv` is tagged with its mode (`full_test` or `dry_run`), ensuring transparency about the confidence level of each score.

## Result Card System

After each skill optimization (and again after the full batch completes), darwin-skill generates visual result cards that display before/after scores, dimension-by-dimension improvements, and a summary of the most impactful changes. These cards are rendered as HTML and captured as PNG screenshots at 2x resolution for high-quality display.

The system offers three visual themes, randomly selected for each card:

| Theme | CSS Class | Visual Character |
|-------|-----------|-----------------|
| Warm Swiss | `.theme-swiss` | Warm white background, terracotta orange accents, Inter typeface, clean grid layout |
| Dark Terminal | `.theme-terminal` | Near-black background, fluorescent green text, monospace typeface, scanline effect |
| Newspaper | `.theme-newspaper` | Warm paper background, deep red accents, serif typeface, two-column editorial layout |

Each card includes the skill name, the before/after total score with delta, bar charts showing the 8 dimension scores before and after optimization, and the top 3 improvements made. The brand tagline -- "Train your Skills like you train your models" -- appears at the bottom of every card, reinforcing the core metaphor that drives the system's design.

## Getting Started

To use darwin-skill for optimizing your AI agent skills, follow these steps:

```yaml
# Step 1: Install darwin-skill
# Clone the repository and place the SKILL.md in your .claude/skills/ directory
git clone https://github.com/alchaincyf/darwin-skill

# Step 2: Run full optimization (recommended for first use)
# The system will scan all skills, create a git branch,
# design test prompts, evaluate baselines, and run the optimization loop
# Command: "Optimize all skills"

# Step 3: Review result cards
# After each skill is optimized, review the diff and score changes
# Approve or reject changes at each human checkpoint

# Step 4: Check the summary report
# Phase 3 generates a before/after score table
# Merge the optimization branch if satisfied with results
```

```bash
# Single skill optimization
# Command: "Optimize the huashu-slides skill"
# Runs Phase 0.5 through Phase 2 for the specified skill only

# Evaluation only (no changes)
# Command: "Evaluate all skills quality"
# Runs Phase 0.5 and Phase 1 only -- baseline scoring without modifications

# View optimization history
# Command: "Show skill optimization history"
# Reads and displays results.tsv with all past experiments
```

The `results.tsv` file tracks every optimization attempt with 9 columns: timestamp, commit hash, skill name, old score, new score, status (keep/revert/baseline/error), targeted dimension, improvement note, and evaluation mode (full_test or dry_run). This file accumulates over time, providing a longitudinal view of skill quality evolution.

## Conclusion

Darwin-skill addresses a fundamental gap in AI agent skill management: the absence of systematic, measurable improvement cycles. The ratchet mechanism ensures forward progress -- skills can only get better, never worse, because every change that fails to improve the score is automatically reverted. The 8-dimension rubric provides objectivity -- instead of subjective "this feels better" judgments, improvements are quantified across specific, weighted criteria. The independent scoring principle prevents bias -- the agent that edits is never the agent that scores, eliminating the self-grading problem that plagues ad hoc improvement attempts.

The ML training analogy is not just a framing device; it is the architectural foundation. Skills evolve the way models do: through iterative evaluation, targeted modification, and strict validation. The addition of human-in-the-loop checkpoints acknowledges that skill quality, unlike model loss, involves dimensions that a single number cannot fully capture. Darwin-skill brings engineering discipline to what was previously artisanal work -- and in doing so, it makes continuous skill improvement not just possible, but systematic and reliable.