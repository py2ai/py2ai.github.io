---
layout: post
title: "Answer Me with HTML: Turn Agent Answers Into One Visual Page - Inside QingYunA/answer-me-with-html"
description: "A MIT-licensed agent skill where the model writes only a short Markdown draft and a bundled CLI renders a single-file HTML explainer page - diagrams auto-laid-out, writing checked, and even narrated videos generated."
date: 2026-10-05
header-img: "img/post-bg.jpg"
permalink: /answer-me-with-html-turn-agent-answers-into-one-visual-page/
featured-img: "ai-coding-frameworks/ai-coding-frameworks"
image: https://pyshine.com/assets/img/diagrams/answer-me-with-html/qingyuna-answer-me-with-html-architecture.svg
tags: [AI, Agent Skills, Data Visualization, Open Source]
categories: [AI, Open Source]
keywords: [answer me with html, agent skills, Claude Code, Cursor, Codex, explainer video]
author: "PyShine"
---

Ask a coding agent to explain the TCP three-way handshake and you get a wall of text. Ask it to draw the answer instead and you get a different problem: the agent types every line of CSS, every wrapper div, and every SVG coordinate by hand, and you sit there watching output tokens scroll for a minute or two. Andrej Karpathy framed this well - as LLMs do more of the work, keeping up with their output becomes the hard part, and a diagram or a page is far easier to absorb than a long block of text.

[Answer Me with HTML](https://github.com/QingYunA/answer-me-with-html) by QingYunA resolves this with a clean division of labor. The model writes only content - a short Markdown draft with headings as panels and fenced blocks for diagrams. A CLI that ships inside the skill does everything else: it picks the template, places the panels, applies the theme, lays out flow charts with the dagre library, and spaces sequence diagrams by actual label width. The agent never hand-writes HTML, CSS, or SVG.

The result is measurable. The repository's benchmark, run with the same model on both sides across 3 topics and 3 runs (medians, Claude Sonnet 5.5), shows 923 output tokens per answer versus 6,873 when asking for HTML directly - 7.4 times fewer - and 13 seconds versus 46 seconds, roughly 3.6 times faster, at about the same cost per answer. After install you just ask questions the way you always do, and the agent decides when a page is worth it: related concepts, multi-step flows, multi-way comparisons get pages, while one-line questions get one-line answers.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/answer-me-with-html/qingyuna-answer-me-with-html-overview-architecture.svg" alt="Answer me with HTML overview architecture" style="min-width:720px;width:100%;">
</div>

*The overview: the agent consults SKILL.md, pipes a short draft into the bundled CLI, and the CLI's parser, components, templates, and themes produce a single offline page or a narrated video.*

Reading the overview from left to right: the whole flow starts at `skills/answer-me-with-html/SKILL.md`, the playbook that tells the agent when to make a page and how to call the CLI in a single heredoc bash invocation. The CLI itself lives bundled in `skills/answer-me-with-html/scripts/am.mjs`, a single file with no install step, built from the modules under `src/`. Inside, `src/cli.js` dispatches the render, patch, video, lint, config, and clean commands; `src/parse.js` splits the draft into frontmatter and panels; `src/components/index.js` maps each panel to a component - flow, sequence, tree, timeline, limits, annot, kv, callout - and the diagram components compute real coordinates through `src/components/flow.js` and `src/components/sequence.js`. Templates in `src/templates/sheet.js` arrange panels on a grid, `src/themes/index.js` applies the blueprint or shadcn look, and `src/page.js` writes one self-contained HTML file. For video answers, `src/video/render.js` produces a beat-synced player. On the quality side, `src/lint/ste.js` runs an adapted ASD-STE100 writing check on every draft, and the optional always-on hook in `plugins/answer-me-with-html-always/hooks/remind.mjs` nudges the agent to attach a small page to every conclusion.

## Why You Need This

The token bill is only the visible part. When a model computes SVG coordinates by hand, arrows point at nothing, labels get cut off, and grids develop random gaps - the README is refreshingly blunt about this failure mode from the author's own experiments. Because layout here is computed by code rather than guessed by a language model, labels are measured before placement and panels snap to a real grid. When a draft does contain an error, the CLI does not just fail: it returns the line number, the offending component, and a correct example, and per the SKILL.md playbook the agent fixes that exact line and re-renders, typically in one try.

The second problem is prose quality, and this is where the project gets unusual. Drafts are checked against rules adapted from ASD-STE100, the controlled English first written for aircraft maintenance manuals - the same rules Karpathy noted make LLM writing far more readable. The machine-checkable parts became a rule set in `src/lint/ste.js` with English and Chinese wordlists: steps stay under 20 English words or 35 Chinese characters, descriptions under 25 words or 45 characters, paragraphs at most 6 sentences; wordy phrases like "utilize" yield to "use"; Chinese drafts drop empty verbs so 优化 replaces 进行优化; passive voice, three or more 的 in one sentence, and stock phrases such as 赋能 and 闭环 get flagged. A draft containing kana is treated as Japanese automatically. The check warns by default and only refuses to render in strict mode, which you control per page or globally.

The third reason is permanence. Every page embeds the Markdown that produced it inside a source textarea, so nothing is lost when the conversation scrolls away - click "Copy source" and the draft is back. Pages are plain single files with no CDN links or web fonts, they open offline, and they are trivial to share.

## How It Works

The agent's entire workload is the draft. A frontmatter block picks the template (sheet for a one-screen grid, doc for a linear walkthrough), the theme (blueprint or shadcn), the title, and optional metadata; every `## ` heading becomes a panel, with `span=2`, `rows=2`, and `bare` controlling layout and optional lettered IDs added automatically. Fenced blocks carry the diagrams. A flow block is plain text like `A -> B: label`; a sequence block lists messages between parties; tables use the status words ok, no, and warn in cells, which the renderer in `src/markdown.js` turns into check, cross, and warning badges.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/answer-me-with-html/qingyuna-answer-me-with-html-architecture.svg" alt="Answer me with HTML detailed architecture" style="min-width:900px;width:100%;">
</div>

*The detailed view: from the SKILL.md playbook through the parser, component set, themes, and the video stack, down to the benchmark and the 20-file test suite.*

### Understanding the Architecture

**One dispatcher, seven commands.** `src/cli.js` routes render, patch, video, lint, config, clean, list, and help. Render accepts a file or stdin - stdin is how agents call it, per the SKILL.md workflow - and prints either a success path or an error with the line number and a corrected example. Patch is the surgical command: it reads the source textarea back out of an existing page via `src/page.js`, replaces one named panel, and overwrites the file in place while keeping the original theme, mode, and style settings. Config writes to `~/.answer-me-with-html/config.json` through `src/config.js`, whose value parser accepts English and Chinese on/off words alike, and housekeeping in `src/housekeeping.js` watches the pages folder - if it grows past 200 MB, or passes 20 MB with no cleanup for 30 days, the CLI adds a throttled hint to the output, at most once every 7 days for cleanup and once every 3 days for updates, and it never deletes anything without your consent. A weekly background version check in `src/update.js` compares semver numbers and never updates by itself.

**The video stack is the surprise.** Ask for a 3b1b-style video and the agent writes the same draft plus one blockquote narration line per beat. `src/video/script.js` parses beats and camera focus markers - a node name in square brackets steers the virtual camera and highlights that node. When the Nth narration line plays, the Nth step of the diagram appears, and nodes with the same name in consecutive scenes glide to their new positions instead of cutting. `src/video/tts.js` handles voice: ElevenLabs if you set `ELEVENLABS_API_KEY`, a local OpenAI-compatible speech server returning 16-bit PCM WAV if you pass `--voice local` with `AM_TTS_URL` (the README suggests mlx-audio with Qwen3-TTS), otherwise the operating system voice, with captions only when neither exists. Audio is decoded and resampled at a 22,050 Hz sample rate, silence is trimmed, and each beat lasts exactly as long as its narration clip, so picture and voice stay in sync. The output is a single HTML file with the audio embedded - or pass `--mp4` to get a 1080p file via headless Chrome driven over the DevTools protocol and ffmpeg in `src/video/export.js`.

**Quality is enforced, not promised.** The linter in `src/lint/ste.js` applies length rules per sentence language, so a Chinese sentence in an English draft gets Chinese limits. The benchmark harness in `bench/run.mjs` reproduces the token and time comparison and stores results in `bench/results/results.json`, so the headline numbers are not marketing. The test directory holds 20 test files covering the CLI, components, sheet layout, patching, config, housekeeping, and a 27 KB video test module, and `npm run snapshot` compares rendered HTML against the main branch so refactors cannot silently change output.

Tracing one answer end to end: your question reaches the agent, SKILL.md's gate decides a page is warranted because the answer spans three or more interrelated concepts, the agent mentally sketches 3 to 8 panels and pipes a draft through the bundled CLI with a heredoc, the parser maps panels to components, dagre ranks the flow graph while measured text widths space the sequence lanes, the sheet template places everything, the theme paints it, the STE check counts words, and a single offline HTML file opens in your browser with the source draft embedded inside.

## Advantages

- **The model writes content, code does layout.** Diagram coordinates come from dagre and measured text, not model guesses - no more arrows into the void.
- **Self-correcting loop.** Errors come back with line numbers, component names, and correct examples, so the agent repairs drafts in one attempt.
- **Real writing discipline.** The ASD-STE100-derived check catches wordiness, passive voice, and stock phrases in English and Chinese, with a strict mode for CI-like enforcement.
- **Zero-install portability.** The CLI is one bundled file needing only Node.js 20 or newer; there is no npm install step on the user side.
- **Everything is inspectable.** Pages embed their source draft, the benchmark script and its results are committed, and 20 test files plus an HTML snapshot check guard regressions.
- **One skill, two media.** The same draft format produces both static explainer pages and narrated, beat-synced videos with local or cloud voices.

## Benefits

- **You stop waiting on the model.** Median answer time drops from about 46 seconds to about 13 in the published benchmark because the agent types roughly one seventh of the tokens.
- **Complex answers become scannable.** Sequence diagrams, state flows, folder trees, timelines, comparisons with status badges, and annotated sentences each get the visual form that fits the information shape.
- **Your terminal stays clean.** The agent replies with two or three lines - the core conclusion plus the page path - instead of pasting drafts or HTML back into the chat.
- **Pages accumulate into a personal knowledge base.** Everything lands in one folder, the copy-source button recovers any draft, and a consent-first cleanup command keeps the directory from growing forever.
- **Always-on mode suits note-taking workflows.** Install the companion plugin and every conclusion automatically gets a small 2-to-4 panel page rendered silently in the background.
- **It respects your attention.** No pop-ups when disabled, no pages for one-line answers, no pages in plan mode, and throttled housekeeping hints instead of nagging.

## Usage

Install with the skill installer, which asks which agents to set up (the installer supports more than 70 agents):

```bash
npx skills add QingYunA/answer-me-with-html
```

Or paste this into Claude Code, Codex, Cursor, or OpenCode and let the agent do it:

```text
Install the Answer me with HTML skill: run npx -y skills add QingYunA/answer-me-with-html -g -y,
and pass -a with your own agent name (for Claude Code, -a claude-code). Then read its SKILL.md
and use it to make a page that explains the TCP three-way handshake, so we know it works.
```

As a Claude Code plugin:

```bash
/plugin marketplace add QingYunA/answer-me-with-html
/plugin install answer-me-with-html@answer-me-with-html
```

Then just ask normally - "Explain the TCP three-way handshake", "Map out how the modules in this repo fit together", "Redis or Memcached?" - or explicitly say "explain it in HTML".

Drive the CLI directly when you want to:

```bash
AM=skills/answer-me-with-html/scripts/am.mjs

node $AM render examples/tcp.en.md                # render and open in the browser
node $AM render notes.md -o out.html --no-open    # choose output, stay quiet
node $AM render notes.md --theme shadcn           # theme for this run only
node $AM patch page.html --panel "Why three messages" < panel.md
node $AM lint notes.md                            # writing check only
node $AM list                                     # list components
node $AM config                                   # view settings
node $AM config set open off                      # stop auto-opening
node $AM clean --dry-run                          # preview cleanup
```

Make a narrated explainer video:

```bash
node $AM video examples/video-tcp.en.md           # player page with system voice
AM_TTS_URL=http://127.0.0.1:8000 AM_TTS_MODEL=mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-4bit \
  AM_TTS_VOICE=vivian node $AM video draft.md --voice local
node $AM video draft.md --mp4                     # 1080p file (Chrome + ffmpeg + Node 22+)
```

For development and verification:

```bash
git clone https://github.com/QingYunA/answer-me-with-html.git && cd answer-me-with-html
npm install
npm test
npm run build      # rebuild the bundled am.mjs after changing src/
npm run snapshot   # prove rendering did not change
```

## Conclusion

Answer Me with HTML is a rare kind of agent skill: it does not ask the model to be a better designer, it removes the design work from the model entirely and gives the job to deterministic code. The draft format is small enough to learn in a minute, the components cover the shapes that technical answers actually take, the writing check imports half a century of controlled-English practice, and the video stack turns the same drafts into narrated explainers - all in one offline HTML file. If you use coding agents daily, this skill changes what "explain it to me" means, and the committed benchmark means you can verify the speedup yourself.

- Repository: [github.com/QingYunA/answer-me-with-html](https://github.com/QingYunA/answer-me-with-html)
- Benchmark details: [bench/README.md](https://github.com/QingYunA/answer-me-with-html/tree/main/bench)
- Karpathy's observation that inspired it: [the post on X](https://x.com/karpathy/status/2105819303471976479)
- License: [MIT](https://github.com/QingYunA/answer-me-with-html/blob/main/LICENSE)
