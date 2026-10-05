---
layout: post
title: "yomiyasu: Deodorize AI-Scented Japanese Without Losing Meaning - Inside nanaism/yomiyasu"
description: "A MIT-licensed agent skill from ALGO ARTIS that rewrites AI-flavored Japanese into natural, information-dense prose - backed by a stdlib-only linter, a diff checker, and a 160-document verification corpus."
date: 2026-10-05
header-img: "img/post-bg.jpg"
permalink: /yomiyasu-deodorize-ai-scented-japanese-without-losing-meaning/
featured-img: "ai-coding-frameworks/ai-coding-frameworks"
image: https://pyshine.com/assets/img/diagrams/yomiyasu/nanaism-yomiyasu-architecture.svg
tags: [AI, Agent Skills, Technical Writing, Open Source]
categories: [AI, Open Source]
keywords: [yomiyasu, AI writing, Japanese, agent skills, Claude Code, Cursor, Codex]
author: "PyShine"
---

If you have read enough AI-generated Japanese, you recognize the smell before you can name it. Sentences open with "重要なのは" and close with "いかがでしたでしょうか". Systems "quietly break", time "melts away", and every other phrase is wrapped in bold. The prose looks polished, yet something is always off - the subject of a sentence is a concept that cannot act, and the core of a procedure has been flattened into bullet points.

The team behind [yomiyasu](https://github.com/nanaism/yomiyasu) - appropriately named "よみやす", or "easy to read" - traces this smell to specific linguistic pathologies rather than to a list of forbidden words. Their analysis, published in a detailed [Zenn article](https://zenn.dev/algoartis/articles/0b1c731881b25c), argues that earlier style-fixing prompts failed for four reasons: they only swapped surface vocabulary, they over-applied rhetorical rules until the model invented new awkwardness, they left subject-predicate relationships vague while attaching inanimate subjects to metaphorical verbs, and they piled up bold text and bullets while the actual information density dropped.

yomiyasu, released under the MIT license by optimization-focused startup ALGO ARTIS, is an agent skill that attacks the problem at the syntax level. It ships as a heavily documented skill definition for Codex, Claude Code, Cursor, and similar AI coding environments, plus two dependency-free Python checkers and a 160-document evaluation corpus. The promise is unusual for this genre: rewrites must preserve meaning first, and every claim the skill makes about "AI smell" is verifiable with scripts you can run yourself.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/yomiyasu/nanaism-yomiyasu-overview-architecture.svg" alt="yomiyasu overview architecture" style="min-width:720px;width:100%;">
</div>

*The yomiyasu overview: one skill entrypoint drives a set of rule documents, while two standard-library Python checkers and a generated corpus keep every rewrite accountable.*

Reading the overview from left to right, the story goes like this. Everything starts from `SKILL.md`, the 40 KB entrypoint that an agent loads when the skill is invoked. It defines the seven conversion principles, points to the vocabulary blacklist in `references/slop-catalog.md` and the structural rules in `references/gemini-syntax.md`, and selects one of the three tone guides under `references/domains/` - tech, business, or essay - depending on the task. When the rewrite is drafted, `SKILL.md` instructs the agent to run `scripts/yomiyasu_lint.py` as a static check and then `scripts/yomiyasu_diff.py` to compare the draft against the original. Those two tools draw their detection patterns from the same rule documents, so the skill and the checkers cannot drift apart. On the evidence side, `scripts/benchmark_corpus.py` imports the linter's `lint_text` function and measures it against the folders under `tests/corpus/`, while `tests/test_bold_multiline.py` guards the hardest parsing cases with regression tests. Packaging is handled twice over: `.claude-plugin/plugin.json` registers the skill as a Claude Code plugin, and `skills/yomiyasu/` holds the self-contained copy that gets distributed through skill installers.

## Why You Need This

The usual fix for AI-flavored Japanese is a blacklist prompt: "never write 手触り or 解像度 or 静かに壊れる". The yomiyasu authors tested that approach and found it circular - the model simply reaches for a different vague word, and the underlying sentence structure stays broken. That is why the corpus contains a dedicated `blacklist_ai` folder: documents generated with the ban-words prompt still score poorly on structure, which is evidence you can inspect in `tests/corpus/benchmark_results.json`.

There is also a purely mechanical problem that most writing prompts never notice. In Japanese text, bold markers (`**`) sit flush against full-width brackets and punctuation, and GitHub Flavored Markdown and CommonMark will refuse to render them - the raw asterisks appear in the published page. yomiyasu's maintainers hit this on their own README, filed it as Issue #2, and responded by building a proper delimiter-aware checker in `scripts/yomiyasu_lint.py` that understands multi-line bold, block boundaries, code spans, and the full set of Japanese brackets like 「」 and 『』. The regression cases live in `tests/fixtures/bold_regressions.json` - a 50-case fixture file - and the unittest suite in `tests/test_bold_multiline.py` replays them on both scripts.

Finally, there is an operational trap: if you activate two Japanese proofreading skills at once, their instructions fight each other and the output degrades. The README says this plainly and asks you to disable competing proofreading skills while yomiyasu is active. That kind of honesty about failure modes is rare, and it extends to the tooling - the linter explicitly labels every finding as a "candidate for review", stating that a clean report does not guarantee the rewrite preserved meaning.

## How It Works

The skill runs as a four-step procedure defined in `SKILL.md`. Step 1, "grasp context and paragraphs", has the agent identify the document's stance and how its paragraphs connect before touching a word. Step 2 performs the rewrite under the seven conversion principles. Step 3 invokes the static linter, resolving the script path relative to the skill installation directory so it works no matter where the agent stores skills. Step 4 saves the original and rewritten text to files and runs the diff checker to catch anything the rewrite added, dropped, or inverted.

The seven principles deserve a close read because they encode the linguistic theory. The first and highest rule is meaning preservation: conditions, exceptions, negations, parallel structures, quantities, and ordering must survive the rewrite traceable, and the agent may only fill in a missing subject when the original text or provided context confirms it. The second preserves each sentence's function - a request stays a request, an explanation stays an explanation - and keeps the register (polite or plain form) the document already uses. The third untangles personification: a tool or concept that "feels" or "wants" gets rewritten, while objectively correct statements about inanimate systems remain. The fourth opens metaphorical verbs - "quietly breaks", "melts time", "pushes to one side" - into literal descriptions of what actually happens, without shrinking the original nuance. The fifth reviews prefaces and the "AではなくB" contrast pattern, deleting them only when the argument's weight survives. The sixth forbids adding information the source never contained, including invented causes, numbers, or emotions. The seventh tunes length - an average sentence of 30 to 45 characters with 0 to 2 commas - and cleans up trailing colons, decorative markers, and stray half-width spaces.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/yomiyasu/nanaism-yomiyasu-architecture.svg" alt="yomiyasu detailed architecture" style="min-width:900px;width:100%;">
</div>

*The detailed view traces each step to the exact functions and data folders that implement it, from the seven principles down to the corpus folders that measure detection precision.*

### Understanding the Architecture

**The linter is a numeric report card.** `scripts/yomiyasu_lint.py` computes document metrics with `analyze_markdown_metrics` - characters, lines, bold density per 1,000 characters, and the ratio of list lines. When prose exceeds 300 characters, it warns above 3.0 bold marks per 1,000 characters and above a 25 percent list ratio. Pattern checks then run line by line: the `SLOP_WORDS` list flags pseudo-concrete vocabulary such as 手触り, 肌感, 解像度, 意思決定OS, 土台, 羅針盤, and 触媒; the `METAPHOR_VERB_PATTERNS` list catches English calques like "silently fails" (静かに壊れる) and "ignores silently" (黙って無視) with regex that avoids double-flagging the same verb span; `FILLER_PATTERNS` spots prefatory phrases and formulaic closings; a negative-parallelism regex marks "AではなくB" constructions as informational. Everything lands in a findings list where each warning costs 5 points and each informational note costs 2, producing a 100-point score, and any finding from the `bold_problems` analysis is severity "error" because unrendered asterisks are objectively broken output. The CLI supports `--strict`, which exits nonzero for CI and git hooks, and `--json` for machine-readable results.

**The diff checker audits the rewrite, not the style.** `scripts/yomiyasu_diff.py` takes the original and the rewritten document and answers three questions. First, did sentence-ending types shift? It maps every sentence ending to a stance - recommending, prescriptive, or explanatory - using the `STANCES` table, and reports any sentence whose stance flipped. Second, did content change? The `MARKERS` dictionary counts request forms like てください, obligation forms like なければならない, evaluations, conditionals, and connectives, so an added "must" or a lost "may" is surfaced as a marker count change; meanwhile `CONTENT`-regex word sets produce the list of words that appeared or vanished. Third, did structure change? Converting a list into prose, merging paragraphs, or changing the sentence count is reported with a pointer to re-check that section. As a final pass, `difflib.SequenceMatcher` computes character-level opcodes, and only inserted segments that introduce a marker or a brand-new word survive into the report - so you see exactly where the rewrite got creative.

**The corpus makes the claims falsifiable.** `scripts/build_corpus.py` generates raw AI text across 8 practical genres - architecture explainers, incident reports, PR descriptions, Slack announcements, comparison memos, spec drafts, code review comments, and retrospective essays - times 3 persona variations, calling the Claude CLI headlessly. That yields 24 raw documents, 24 blacklist-prompted documents, and 48 yomiyasu-rewritten documents. `scripts/setup_corpus_static.py` then seeds 16 human-written documents extracted from public pages with a stdlib `HTMLParser` subclass, plus 48 edge-case documents that contain the flagged verbs in perfectly legitimate usage - so false positives have a defined home. `scripts/benchmark_corpus.py` runs `lint_text` across all 160 documents, compares a naive keyword regex against the linter's lookaround-based patterns, and writes the outcome to `tests/corpus/benchmark_results.json`. The edge-case folder also doubles as a linter test input, confirming that "壊れる" describing a literal crash is not the same crime as "静かに壊れます" describing a vague process.

Tracing one document end to end: a draft lands in the agent, `SKILL.md` Step 1 maps its paragraphs and stance, Step 2 rewrites it under the seven principles with the selected domain guide from `references/domains/`, Step 3 scores it with `scripts/yomiyasu_lint.py`, Step 4 cross-examines it with `scripts/yomiyasu_diff.py`, and the agent returns the rewritten text, a list of what changed, any AI-flavored phrases deliberately kept because they carry meaning, and at most two questions for the author when the source text genuinely cannot settle a detail. The output format is fixed in `SKILL.md`, so results are comparable across runs and across agents.

## Advantages

- **Meaning preservation is rule one, not an afterthought.** The skill instructs the agent to leave natural sentences alone, keep evaluation and contrast when they carry weight, and route genuinely ambiguous cases to a writer-confirmation section instead of guessing.
- **Verification is standard-library Python.** Both checkers import nothing beyond the Python standard library, so you can run them on locked-down machines, in CI, or as pre-commit hooks without installing a single dependency.
- **The bold-rendering checker solves a real, documented bug class.** Delimiter-aware analysis of multi-line bold, block boundaries, code spans, and Japanese brackets turns "why does my README show asterisks" into a one-line finding with a suggested fix.
- **Findings are advisory with reasons attached.** Every message explains when the flagged pattern is legitimate - literal meanings, necessary contrasts, defined terminology - which keeps the tool from bullying good prose into blandness.
- **The evidence chain is public.** Eight genres, three personas, blacklist controls, human baselines, and legitimate-usage edge cases are all in the repository, and `benchmark_corpus.py` regenerates `tests/corpus/benchmark_results.json` on demand.
- **Distribution is already solved.** Claude Code plugin manifests, a packaged skill copy, skill-installer commands, and a registration-safe ZIP release cover every mainstream agent environment.

## Benefits

- **Your technical articles stop reading like translations.** Domain guide `references/domains/tech.md` pushes rewrites toward restored procedures and fewer bullets, which is exactly what a code walkthrough needs.
- **Business documents regain accountability.** `references/domains/business.md` targets specs, PR descriptions, and proposals: metaphor-free sentences, explicit boundary conditions, and named responsible parties.
- **Personal writing keeps its voice.** `references/domains/essay.md` protects genuine feeling and blocks the inflated moral-of-the-story ending that AI essays drift toward.
- **You get a reusable AI-prose detector.** Even without rewriting anything, running `yomiyasu_lint.py` over a suspicious document gives you a score, per-line findings, and snippet evidence you can act on.
- **Teams can standardize review.** `--strict` mode turns the linter into a merge gate, and `--json` output plugs into existing tooling without glue code.
- **It is honest about interference.** The README warns you to deactivate competing proofreading skills, preventing the quiet failure mode where two well-meaning prompts cancel each other.

## Usage

Install into Claude Code or another agent environment with the skill installer:

```bash
npx skills add nanaism/yomiyasu
```

For Cursor, Codex, and environments that read `AGENTS.md`:

```bash
npx openskills install nanaism/yomiyasu
npx openskills sync
```

As a Claude Code plugin:

```bash
/plugin marketplace add nanaism/yomiyasu
/plugin install yomiyasu@yomiyasu
```

For the Claude web app, download the registration-safe ZIP from the releases page and upload it in the skill settings - the repository warns that zipping the full checkout yourself will fail because of the duplicate packaged copy and the alternate plugin manifest.

Then simply paste a draft and ask:

```text
この文章を読みやすくして。
（ここに修正したい文章を貼り付け）
```

Target a specific style with a domain hint - "技術記事向けに" or explicitly "ドメイン tech" - choosing between tech, business, and essay guides.

Run the checkers directly on any Markdown file:

```bash
python3 scripts/yomiyasu_lint.py article.md
python3 scripts/yomiyasu_lint.py article.md --strict
python3 scripts/yomiyasu_lint.py article.md --json
python3 scripts/yomiyasu_diff.py original.md rewritten.md --stance=説明
python3 scripts/yomiyasu_diff.py --endings draft.md
```

Reproduce the evaluation corpus and benchmark locally:

```bash
python3 scripts/build_corpus.py
python3 scripts/setup_corpus_static.py
python3 scripts/benchmark_corpus.py
```

Run the regression suite for the bold-rendering analysis:

```bash
python3 -m unittest tests/test_bold_multiline.py
```

## Conclusion

yomiyasu is what happens when a team treats AI writing problems as engineering problems. The skill document encodes a linguistic diagnosis - broken subject-predicate chains, calqued metaphor verbs, format-heavy low-density formatting - and every rule is backed by a checker, a corpus folder, or a regression test. The tools admit their limits, the README documents its own bug history, and the whole thing runs on the Python standard library. If your team publishes Japanese technical content produced or polished by AI, this repository will improve the output and, just as valuably, show you how to hold an agent accountable for what it writes.

- Repository: [github.com/nanaism/yomiyasu](https://github.com/nanaism/yomiyasu)
- Background article: [Deodorizing AI-smelling Japanese with the yomiyasu Agent Skill (Zenn)](https://zenn.dev/algoartis/articles/0b1c731881b25c)
- License: [MIT](https://github.com/nanaism/yomiyasu/blob/main/LICENSE)
