---
layout: post
title: "i-have-adhd: Stop Your Coding Agent From Burying the Answer"
description: "i-have-adhd is a single-file skill for coding agents that reshapes output: action first, numbered steps, no preamble, no tangents. A source-level tour of ayghri/i-have-adhd: the five cognitive premises behind its ten rules, the override cases, the pre-send check, and the manifest system that installs it into eight coding-agent platforms."
date: 2026-09-28
header-img: "img/post-bg.jpg"
permalink: /i-have-adhd-Stop-Your-Coding-Agent-From-Burying-the-Answer/
featured-img: "ai-coding-frameworks/ai-coding-frameworks"
image: https://pyshine.com/assets/img/diagrams/ihaveadhd/ayghri-ihaveadhd-architecture.svg
tags:
  - AI Agents
  - Prompt Engineering
  - Productivity
  - Open Source
categories: [AI, Open Source]
keywords: "i-have-adhd, ayghri, coding agent output style, answer first prompt, Claude Code skill, Cursor skill, Codex plugin, SKILL.md, prompt engineering, agent output formatting, numbered steps, no preamble, ADHD friendly output, AI productivity"
author: "PyShine"
---

You ask your coding assistant a question. It opens with "Great question!", narrates its reasoning, buries the actual command three paragraphs deep, tacks on two tangents you did not ask about, and closes with "Hope this helps!" You scroll. You lose your place. You forget what you were doing. Multiply that by every exchange in a working day and the tax is real - not because the model is wrong, but because its *shape* is wrong for a reader who needs to act.

[i-have-adhd](https://github.com/ayghri/i-have-adhd) by Ayoub G. attacks exactly that. It is a skill - a single markdown file of output rules - that you install into coding agents like Claude Code, Codex, Cursor, Kimi, Qwen, Gemini CLI and opencode. Once active, the agent leads with the next action, numbers multi-step work, gives estimates in real minutes, keeps errors flat and factual, and never opens with a pleasantry or closes with filler. The repository's own before/after pair says it in eight lines: before, a rambling paragraph about a JWT bug; after, "Run `npm install jsonwebtoken@latest`, then edit `src/auth.ts:42`," followed by three numbered steps and one next action.

What makes this repository worth studying is that it is not a demo of prompt tricks. It is a small, complete piece of agent infrastructure: a rule system grounded in five explicit premises about how some readers fail to act on text, override cases that keep the rules from destroying the answer, a self-check the agent runs before sending, and a distribution layer that packages the same file for eight different harnesses - which is how a repository with no application code at all earned its place at the top of the trending charts.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/ihaveadhd/ayghri-ihaveadhd-overview-architecture.svg" alt="Architecture overview of the ayghri/i-have-adhd repository" style="max-width:100%;height:auto;" />
</div>

*High-level overview: one SKILL.md defines the rules; platform manifests package it for each harness; install guides document the path; the coding agent loads it on invocation and its output changes shape.*

Reading the overview from left to right: everything starts from one file, `skills/i-have-adhd/SKILL.md`, which carries the entire rule system - there is no compiled code anywhere in the repository. Distribution is a set of small JSON manifests, one per platform: a Claude Code plugin plus its marketplace listing, a Codex plugin manifest, a Cursor skill copy, and manifests for Kimi, Qwen, Gemini CLI and opencode. Markdown install guides - translated into nine languages - document how to wire it in. At runtime, your coding agent loads the skill when you invoke `/i-have-adhd`, and from that point its responses follow the rule contract until you turn it off. The whole system is a prompt, packaged with the care usually reserved for software.

## Why You Need This

Consider where the time actually goes in an agent-assisted workflow. Rarely in waiting for the answer - more often in re-reading it. An assistant that leads with narration forces you to parse its entire response to find the one line that matters. An assistant that appends tangents makes every answer cost three decisions instead of one. And an assistant that says "this will take some work" leaves you to guess whether that means fifteen minutes or an afternoon - which is precisely the kind of vagueness that makes starting hard.

The repository's thesis is that this is a formatting problem with a formatting fix, and it argues the point from premises rather than taste. The SKILL.md opens with five observations that drive every rule: working memory is small, so nothing may be left off-screen; knowing the answer is not doing the answer; starting is the hardest step; vague time estimates all register as the same; and visible progress matters because it is motivating. Each of the ten rules traces back to one of these. "Lead with the next action" exists because the first step must be obvious and small. "Restate state every turn" exists because nothing survives between messages. "Specific time estimates" exist because "a bit of work" and "a few hours" are indistinguishable to a reader deciding whether to start now.

The second reason is scope honesty. This skill does not claim to make agents smarter or your code better - it makes output *actable*. That narrowness is why it works on every platform: the rules speak about responses, not about tools or model internals. And because the skill is a file rather than a fine-tune, you can read every behavior it will produce in ninety seconds, and fork it when you disagree with a rule.

There is also the simple fact of defaults. Agent harnesses are tuned for average conversational politeness - preamble, recap, hedging - because that is what reads as helpful in a demo. The pre-send check in this repository is effectively a de-linting pass against those defaults: delete the opening that announces what you are about to do, delete the closing that asks "anything else?", delete the "by the way" sidebar, replace idioms with the literal action. It is remarkable how much of the padding in typical agent output is exactly those four patterns.

## How It Works

The diagram below maps the real files of the repository and how they connect - from the rule core through its guardrails to the manifest layer that ships it everywhere.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/ihaveadhd/ayghri-ihaveadhd-architecture.svg" alt="Detailed architecture of i-have-adhd: skill core with five cognitive facts, the ten rules grouped into four families, override cases, the pre-send check, platform manifests and documentation" style="max-width:100%;height:auto;" />
</div>

*Detailed architecture: the SKILL.md core persists across a session; five cognitive facts motivate four rule families; overrides and a pre-send check guard the shape; eight platform manifests install the same rules; docs and slash commands wire the invocation.*

### Understanding the Architecture

**The skill core.** `skills/i-have-adhd/SKILL.md` is the entire product. Its YAML frontmatter (`name`, `description`, `disable-model-invocation`, metadata tags) tells the harness what the skill is and requires explicit invocation - the agent will not self-trigger it. The body is organized in four layers: the persistence clause, the five cognitive premises, the ten rules, and the override section. The persistence clause is the interesting one: it states that the rules apply to every response for the rest of the session, do not expire when the topic changes, and can only be turned off by the reader saying "stop adhd mode" - which the agent confirms in one line before returning to its default style. This is session-scoped configuration expressed entirely in prose, and it works because harnesses inject the skill text into the model's context, where the model treats it as standing instruction.

**The rule families.** The ten rules divide naturally into four groups. Rules one to three are the action spine: lead with the next action (the first line is something the reader can *do* - a command, a path, a snippet - prose comes after, if at all); number multi-step tasks (each step one bounded action, no step contains "and then" twice, and the guidance explicitly says to fold trivial steps because "a short path finished beats a complete path abandoned"); and end with one concrete next action the reader can complete in under two minutes. Rules four and five guard attention: suppress tangents (a second issue becomes a separate offer, not a mid-answer digression, though a question raised mid-work gets folded in) and restate state every turn ("Step 3 of 5 done: schema updated. Next: backfill the new column. Run the script?" - and if the harness has a task tool, the checklist does the restating instead of prose).

Rules six to eight fix clarity: estimates in concrete units ("About 15 minutes if tests already cover this. An afternoon if not."), completed work made visible in verifiable terms ("Login now works with magic links. Try: `npm run dev`, open `/login`."), and errors delivered flat - cause and fix, no "Uh oh". Rules nine and ten shape presentation: cap visible lists to five items by grouping and ranking (with the honest caveat that this shapes presentation only and must never limit analysis - retained items appear when asked), and ban the entire genre of preamble, recap and closer, with the forbidden phrases listed literally: "Great question," "Let me...", "I'll...", "Sure!", "Hope this helps," "Happy to clarify".

**The guardrails.** Two sections keep the rules from eating the answer. The override section names six cases where defaults lose: the reader asks to "explain" (run long, but keep the shape); a destructive action is ahead (confirmation beats brevity - safety wins); a debug spiral is happening (stop iterating, name the wrong assumption, ask one diagnostic question); the request is genuinely ambiguous (one clarifying question beats guessing); a rule fights the task ("what are my options" gets ranked options, because the options *are* the answer); and a rule fights the harness (the system prompt outranks the skill - announce tool calls when required, do the work instead of asking "want me to"). Then comes the pre-send check: delete the announcing opener, the "anything else?" closer, the "by the way" sidebar, hedging adverbs that carry no information (but keep hedges with real uncertainty - "deleting it manufactures confidence"), and idioms, replaced with the literal action. The final test is the reader-skims test: if only the first and last lines were read, does the reader know what to do next and what just happened?

**The distribution layer.** The same rules ship through eight manifests. `.claude-plugin/plugin.json` is the Claude Code plugin (version 0.3.0) and `.claude-plugin/marketplace.json` lists it for `claude plugin marketplace add`; `.codex-plugin/plugin.json`, `kimi.plugin.json` (which points at the shared `skills/` directory), `qwen-extension.json` and `gemini-extension.json` repeat the pattern for those CLIs. `.cursor/skills/i-have-adhd/SKILL.md` is a straight copy of the rules for Cursor's skill folder, and opencode gets both a JavaScript plugin wrapper (`.opencode/plugins/i-have-adhd.mjs`, referenced by `opencode.json`) and a slash command file (`.opencode/command/i-have-adhd.md`). A generic `.agents/plugins/marketplace.json` mirrors the listing for other harnesses. Documentation completes the picture: `INSTALL.md` plus nine translated install guides under `.github/install/`, and `AGENTS.md` at the root so a one-line "install the skill from this repository" prompt works on any agent that reads repository instructions.

**A response in flight.** Follow one message through the boxes. You invoke `/i-have-adhd`; the harness loads SKILL.md into context; the persistence clause takes effect. You ask about a failing test. The model drafts its default answer, then applies the contract: the "Great question" opener is deleted by rule ten, the actual command moves to line one by rule one, the fix becomes a numbered list by rule two, the estimate gets units by rule six, the error becomes cause-and-fix by rule eight, the tangent about your stale dependencies becomes a one-line separate offer by rule four, and the closer becomes one concrete next step by rule three. The pre-send check runs last. What you receive is shorter, and more importantly, it starts at the only line you were going to scroll to anyway.

## Advantages

- **One file, zero dependencies.** The entire system is a markdown document - nothing to build, run, or trust beyond reading it, which takes about ninety seconds.
- **Premises, not vibes.** Every rule traces to a stated premise about how readers fail to act on text, so the rules compose predictably and you can argue with them specifically.
- **Override-aware by design.** Six explicit break-glass cases - explanations, destructive actions, debug spirals, ambiguity, task conflicts, harness conflicts - keep the format from destroying the content.
- **Honest scoping.** Rule nine caps presentation but explicitly forbids limiting analysis or dropping relevant items - a formatting rule that knows the difference between shape and substance.
- **Portable across harnesses.** The same rules install into Claude Code, Codex, Cursor, Kimi, Qwen, Gemini CLI and opencode through per-platform manifests, so the behavior follows you between tools.
- **Forkable defaults.** Tuning is editing one file and swapping your copy in - the repository documents exactly how to uninstall the upstream and install your fork.

## Benefits

- **The scroll tax disappears.** Answers lead with the runnable line, so the common case - copy, paste, run - costs one glance instead of a full read.
- **Multi-step work stays on rails.** Numbered steps plus per-turn state restatement mean a context switch between messages no longer costs you the plan.
- **Estimates become decisions.** Concrete units turn "some work" into "fifteen minutes" - which is the difference between starting now and tabbing away.
- **Errors stop wasting attention.** Flat, cause-and-fix error reporting reads as work, not as mood.
- **It generalizes beyond coding.** The rules govern text shape, not code - the same skill disciplines planning answers, reviews, and ops runbooks, anywhere an agent writes for a reader who needs to act.

## Usage

The install path is a prompt - paste this into any CLI coding assistant:

```text
Install the i-have-adhd skill/plugin from https://github.com/ayghri/i-have-adhd,
refer to the repo's AGENTS.md for instructions.
```

Or use the Claude Code plugin commands directly:

```bash
claude plugin marketplace add ayghri/i-have-adhd
claude plugin install i-have-adhd@i-have-adhd
```

Restart your coding assistant, then invoke the skill:

```text
/i-have-adhd
```

The rules stay on for the whole session - across topics and tasks. Turn them off when you want the default style back:

```text
stop adhd mode
```

To tune the rules, fork the repository, edit the one file that matters, and swap your copy in:

```bash
claude plugin uninstall i-have-adhd            # drop the upstream copy first:
claude plugin marketplace remove i-have-adhd   # fork and upstream share both names
claude plugin marketplace add <your-username>/i-have-adhd
claude plugin install i-have-adhd@i-have-adhd
```

The full rule text lives in [SKILL.md](https://github.com/ayghri/i-have-adhd/blob/main/skills/i-have-adhd/SKILL.md), and [INSTALL.md](https://github.com/ayghri/i-have-adhd/blob/main/INSTALL.md) covers each supported platform.

## Conclusion

i-have-adhd succeeds because it refuses to be a model upgrade. It is a contract about response shape, written down in one file, enforced by the model that reads it, and packaged for every harness that matters. The source rewards the read: five premises that actually motivate ten rules, overrides that protect the answer from the format, a pre-send check that doubles as a style linter, and a distribution layer that treats a prompt with release discipline. If your agent's answers make you scroll past scenery to find the command, install the skill, ask something real, and compare the first line you get. The difference is one file wide.

**Links:**

- Repository: [https://github.com/ayghri/i-have-adhd](https://github.com/ayghri/i-have-adhd)
- Skill file: [https://github.com/ayghri/i-have-adhd/blob/main/skills/i-have-adhd/SKILL.md](https://github.com/ayghri/i-have-adhd/blob/main/skills/i-have-adhd/SKILL.md)
- Installation guide: [https://github.com/ayghri/i-have-adhd/blob/main/INSTALL.md](https://github.com/ayghri/i-have-adhd/blob/main/INSTALL.md)
