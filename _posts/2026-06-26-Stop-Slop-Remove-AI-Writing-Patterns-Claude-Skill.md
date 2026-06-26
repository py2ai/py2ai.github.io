---
layout: post
title: "Stop Slop: The Open-Source Skill That Removes AI Writing Patterns From Your Prose"
description: "Learn how stop-slop detects and eliminates 40+ banned phrases, 9 structural anti-patterns, and scores your writing on 5 dimensions. A Claude Code skill that makes AI-generated text sound human."
date: 2026-06-26
header-img: "img/post-bg.jpg"
permalink: /Stop-Slop-Remove-AI-Writing-Patterns-Claude-Skill/
featured-img: ai-coding-frameworks/ai-coding-frameworks
tags:
  - Open Source
  - AI
  - Claude Code
  - Writing
  - LLM
  - Skill
  - Documentation
author: "PyShine"
---

# Stop Slop: The Open-Source Skill That Removes AI Writing Patterns From Your Prose

## Introduction

Language models produce text with recognizable fingerprints. Phrases like "Here’s the thing" and "It turns out" betray machine origin before the second sentence lands. With 12,500+ GitHub stars, stop-slop proves this problem resonates widely.

Stop-slop is a documentation-only Claude Code skill — no executable code, no dependencies. It provides a taxonomy of 40+ banned phrases across 7 categories, 9 structural anti-patterns, and a 5-dimension scoring rubric. The approach is recursive: use AI to detect and remove AI writing patterns. Named categories replace the vague sense that prose "just feels off" with concrete, fixable labels.

## The Problem: AI Writing Is Detectable

- LLMs produce text with recognizable patterns
- Throat-clearing openers: "Here's the thing:", "It turns out"
- Binary contrasts: "Not because X. Because Y."
- Emphasis crutches: "Full stop.", "Let that sink in."
- Business jargon: "Navigate challenges", "Deep dive", "Circle back"
- Meta-commentary: "In this section, we'll explore..."
- These patterns make AI text feel manufactured rather than written

## The Detection Pipeline

![Stop Slop Detection Pipeline](/assets/img/diagrams/stop-slop/stop-slop-detection-pipeline.svg)

### How Stop Slop Analyzes Text

Text flows through three analysis stages:

1. **Phrase Detection** — Scans for 40+ banned phrases across 7 categories
2. **Structural Analysis** — Identifies 9 structural anti-patterns at the sentence level
3. **Rule Enforcement** — Applies 8 core rules (active voice, specificity, rhythm variation)

If the text scores below 35/50 on the scoring rubric, it loops back for revision.

## The Taxonomy of AI Tells

![Stop Slop Tell Taxonomy](/assets/img/diagrams/stop-slop/stop-slop-tell-taxonomy.svg)

### Phrase-Level Patterns (7 Categories)

**Throat-Clearing Openers** (15 phrases)
Announcement phrases that delay the point. "Here's the thing:", "It turns out", "The uncomfortable truth is". The fix: state the content directly.

**Emphasis Crutches** (5 phrases)
Add no meaning. "Full stop.", "Let that sink in.", "Make no mistake". Delete them entirely.

**Business Jargon** (11 replacements)
Replace with plain language. "Navigate challenges" becomes "Handle challenges". "Deep dive" becomes "Analysis". "Circle back" becomes "Return to".

**Adverbs** (15+ words)
Kill all -ly words and filler phrases. "really", "just", "literally", "genuinely", "actually", "fundamentally". Also cut: "At its core", "In today's [X]", "It's worth noting".

**Meta-Commentary** (10 phrases)
Self-referential asides that announce structure instead of delivering content. "Hint:", "Plot twist:", "Let me walk you through...", "In this section, we'll...".

**Performative Emphasis** (3 phrases)
False intimacy or manufactured sincerity. "creeps in", "I promise", "They exist, I promise".

**Vague Declaratives** (5 phrases)
Announce importance without naming the specific thing. "The reasons are structural", "The implications are significant", "The stakes are high". Replace with the specific thing.

### Structural Patterns (9 Anti-Patterns)

**Binary Contrasts** — "Not because X. Because Y." creates false drama. State Y directly.

**Negative Listing** — "Not a X... Not a Y... A Z." is rhetorical striptease. State Z.

**Dramatic Fragmentation** — "[Noun]. That's it. That's the [thing]." reads as manufactured profundity. Use complete sentences.

**Rhetorical Setups** — "What if [reframe]?" is Socratic posturing. Make the point directly.

**Formulaic Constructions** — "By the time X, I was Y." is narrative template. Be specific.

**False Agency** — "The complaint becomes a fix." Complaints don't fix themselves. Name the person.

**Narrator-from-a-Distance** — "Nobody designed this." Floats above the scene. Put the reader in the room.

**Passive Voice** — "Mistakes were made." Name who made them.

**Wh- Sentence Starters** — "What makes this hard is..." becomes "The constraint is..." or name the specific constraint.

## The Scoring System

![Stop Slop Scoring Rubric](/assets/img/diagrams/stop-slop/stop-slop-scoring-rubric.svg)

### 5 Dimensions, 50 Points

Stop Slop rates text 1-10 on five dimensions:

| Dimension | Question | Low Score (1-3) | High Score (8-10) |
|-----------|----------|-----------------|-------------------|
| Directness | Statements or announcements? | Throat-clearing, announcements | Direct statements |
| Rhythm | Varied or metronomic? | Same-length sentences | Varied cadence |
| Trust | Respects reader intelligence? | Hand-holding, justification | Trusts reader |
| Authenticity | Sounds human? | AI patterns detectable | Human voice |
| Density | Anything cuttable? | Filler and redundancy | Every word earns its place |

**Below 35/50: revise.** This threshold catches text that still reads as AI-generated.

## Before and After: 5 Transformations

### Example 1: Throat-Clearing + Binary Contrast

**Before:**
> "Here's the thing: building products is hard. Not because the technology is complex. Because people are complex. Let that sink in."

**After:**
> "Building products is hard. Technology is manageable. People aren't."

**Why it works:** Removed the opener ("Here's the thing:"), eliminated the binary contrast structure ("Not because X. Because Y."), and cut the emphasis crutch ("Let that sink in."). Three direct statements replace three rhetorical devices.

### Example 2: Filler + Unnecessary Reassurance

**Before:**
> "It turns out that most teams struggle with alignment. The uncomfortable truth is that nobody wants to admit they're confused. And that's okay."

**After:**
> "Teams struggle with alignment. Nobody admits confusion."

**Why it works:** Cut hedging ("most"), removed two throat-clearing phrases ("It turns out", "The uncomfortable truth is"), and deleted the permission-granting ending ("And that's okay."). Two sentences replace three. The meaning is identical.

### Example 3: Business Jargon Stack

**Before:**
> "In today's fast-paced landscape, we need to lean into discomfort and navigate uncertainty with clarity. This matters because your competition isn't waiting."

**After:**
> "Move faster. Your competition is."

**Why it works:** Eliminated all jargon ("landscape", "lean into", "navigate", "clarity"). The core message — speed and competition — emerges in six words instead of twenty.

### Example 4: Dramatic Fragmentation

**Before:**
> "Speed. Quality. Cost. You can only pick two. That's it. That's the tradeoff."

**After:**
> "Speed, quality, cost — pick two."

**Why it works:** One sentence replaces five fragments. No performative emphasis. The content is identical; the delivery trusts the reader.

### Example 5: Rhetorical Setup

**Before:**
> "What if I told you that the best teams don't optimize for productivity? Here's what I mean: they optimize for learning. Think about it."

**After:**
> "The best teams optimize for learning, not productivity."

**Why it works:** Direct claim replaces rhetorical scaffolding. No "What if I told you", no "Here's what I mean", no "Think about it." The reader doesn't need to be guided through a reveal.

## The 8 Core Rules

1. **Cut filler phrases.** Remove throat-clearing openers, emphasis crutches, and all adverbs.
2. **Break formulaic structures.** Avoid binary contrasts, negative listings, dramatic fragmentation, rhetorical setups, false agency.
3. **Use active voice.** Every sentence needs a human subject doing something. No passive constructions.
4. **Be specific.** No vague declaratives. Name the specific thing. No lazy extremes.
5. **Put the reader in the room.** No narrator-from-a-distance voice. "You" beats "People."
6. **Vary rhythm.** Mix sentence lengths. Two items beat three. End paragraphs differently. No em dashes.
7. **Trust readers.** State facts directly. Skip softening, justification, hand-holding.
8. **Cut quotables.** If it sounds like a pull-quote, rewrite it.

## The 11-Point Quick Check

Before delivering any prose, run this checklist:

1. Any adverbs? Kill them.
2. Any passive voice? Find the actor, make them the subject.
3. Inanimate thing doing a human verb? Name the person.
4. Sentence starts with a Wh- word? Restructure it.
5. Any "here's what/this/that" throat-clearing? Cut to the point.
6. Any "not X, it's Y" contrasts? State Y directly.
7. Three consecutive sentences match length? Break one.
8. Paragraph ends with punchy one-liner? Vary it.
9. Em-dash anywhere? Remove it.
10. Vague declarative? Name the specific implication.
11. Narrator-from-a-distance? Put the reader in the scene.

## Skill Architecture and Integration

![Stop Slop Skill Architecture](/assets/img/diagrams/stop-slop/stop-slop-skill-architecture.svg)

### The Skill-as-Documentation Pattern

Stop Slop contains zero executable code. The entire skill is documentation:

```
stop-slop/
├── SKILL.md              # Core instructions (8 rules, quick checks, scoring)
├── references/
│   ├── phrases.md        # 7 categories of banned/flagged phrases
│   ├── structures.md     # 9 structural patterns with pattern/problem/fix
│   └── examples.md       # 5 before/after transformation examples
├── README.md
└── LICENSE
```

The `SKILL.md` file uses YAML frontmatter for discoverability:

```yaml
---
name: stop-slop
description: Remove AI writing patterns from prose. Use when drafting, editing, or reviewing text to eliminate predictable AI tells.
metadata:
  trigger: Writing prose, editing drafts, reviewing content for AI patterns
  author: Hardik Pandya (https://hvpandya.com)
---
```

This pattern — skill-as-documentation — means the skill works with any LLM that can read instructions. No API, no runtime, no dependencies.

### Integration Options

**Claude Code:** Add the `stop-slop` folder as a skill. Claude automatically loads `SKILL.md` and references the phrase/structure/example files on demand.

**Claude Projects:** Upload `SKILL.md` and reference files to project knowledge. The skill becomes part of the project context.

**Custom Instructions:** Copy the 8 core rules from `SKILL.md` into your custom instructions. Lightweight but loses the reference detail.

**API Calls:** Include `SKILL.md` in your system prompt. Reference files load on demand when the model needs specific phrase lists or examples.

## Getting Started

### Installation for Claude Code

```bash
# Clone the repository
git clone https://github.com/hardikpandya/stop-slop.git

# Add as a Claude Code skill
# Place the stop-slop folder in your .claude/skills/ directory
```

### Using with Custom Instructions

Copy the 8 core rules from `SKILL.md`:

```markdown
1. Cut filler phrases. Remove throat-clearing openers, emphasis crutches, and all adverbs.
2. Break formulaic structures. Avoid binary contrasts, negative listings, dramatic fragmentation.
3. Use active voice. Every sentence needs a human subject doing something.
4. Be specific. No vague declaratives. Name the specific thing.
5. Put the reader in the room. "You" beats "People." Specifics beat abstractions.
6. Vary rhythm. Mix sentence lengths. Two items beat three. No em dashes.
7. Trust readers. State facts directly. Skip softening, justification, hand-holding.
8. Cut quotables. If it sounds like a pull-quote, rewrite it.
```

### Using with API Calls

Include `SKILL.md` content in your system prompt:

```python
import openai

with open("stop-slop/SKILL.md") as f:
    skill_content = f.read()

response = openai.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": skill_content},
        {"role": "user", "content": "Edit this text to remove AI patterns: ..."}
    ]
)
```

## Why This Matters

- AI-generated content is flooding the internet
- Readers are developing "AI fatigue" — they can spot AI patterns
- Writing that sounds AI-generated loses credibility
- This skill gives developers and writers a systematic framework
- The scoring system provides a quantitative quality metric
- The skill-as-documentation pattern means zero dependencies

## Key Stats

- 12,500+ GitHub stars
- 867 forks
- MIT license
- Pure documentation — no runtime dependencies
- 40+ banned phrases across 7 categories
- 9 structural anti-patterns
- 5-dimension scoring rubric
- 5 before/after transformation examples
- 11-point quick check

## Conclusion

Stop Slop addresses a real and growing problem: AI writing has predictable patterns that make text feel manufactured. By providing a structured taxonomy of 40+ banned phrases, 9 structural anti-patterns, 8 core rules, and a 5-dimension scoring system, it gives writers and developers a systematic framework for producing prose that sounds human. The skill-as-documentation pattern means it works with any LLM — no API, no runtime, no dependencies. Just rules, examples, and a scoring rubric that catches what your ear might miss.

## Links

- GitHub: https://github.com/hardikpandya/stop-slop
- Author: https://hvpandya.com
