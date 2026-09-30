---
layout: post
title: "fast-jev-compaction: Jev-Scored Verbatim Compaction for Claude Code - Inside tamaratran/fast-jev-compaction"
description: "fast-jev-compaction is a Claude Code plugin and npm library that replaces the lossy compaction summary with Jev-scored decisions. Every tool call and result is scored in one fast request, stale items are dropped or truncated, and everything kept stays verbatim. A source tour of tamaratran/fast-jev-compaction."
date: 2026-09-30
header-img: "img/post-bg.jpg"
permalink: /Fast-Jev-Compaction-Decision-Context-Plugin-Source-Tour/
featured-img: "ai-coding-frameworks/ai-coding-frameworks"
image: https://pyshine.com/assets/img/diagrams/fast-jev-compaction/tamaratran-fast-jev-compaction-architecture.svg
tags:
  - Claude Code
  - Context Compaction
  - AI Agents
  - TypeScript
categories: [AI, Open Source]
keywords: "fast-jev-compaction, Claude Code plugin, context compaction, Jev, TypeSafe, verbatim compaction, tool call scoring, npm library, AI agent context, session.compact hook, context window management, TypeScript"
author: "PyShine"
---

Every long agent session eventually hits the same wall: the context window fills up and something must be forgotten. The conventional answer — asking a model to summarize the older turns into prose — has a structural flaw. A summary is a rewrite, and every rewrite drops details: the exact file path, the verbatim error text, the constraint stated in the first message. Once those vanish, the assistant starts re-deriving facts it used to know, or worse, guessing them.

tamaratran/fast-jev-compaction — a TypeScript project with roughly 7,200 GitHub stars — takes the opposite route. It is a Claude Code plugin, and also a plain npm library, that replaces the built-in compaction summary with a set of decisions. Every tool call and tool result is scored in one fast request to Jev, TypeSafe's model; the items Jev judges stale are dropped or truncated, and everything that survives stays verbatim. User and assistant text is never touched.

The source is worth a tour because it is small, sharply factored, and solves a hard problem with unusual discipline: seven modules under `src/`, one hook file adapting them to Claude Code's function hooks, and a README that documents its own limitations honestly.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/fast-jev-compaction/tamaratran-fast-jev-compaction-overview-architecture.svg" alt="Architecture overview of the tamaratran/fast-jev-compaction repository" style="max-width:100%;height:auto;" />
</div>

*Overview of the fast-jev-compaction architecture: the Claude Code plugin surface on the left feeds the compaction library in the middle, which talks to Jev through a thin request layer, while the npm entry point and demo consume the same library from below.*

Reading the overview from left to right: the plugin manifests (`.claude-plugin/plugin.json` and `hooks/hooks.json`) configure and load `hooks/fast-jev.ts`, the adapter that intercepts Claude Code's `session.compact` event. The hook hands the transcript to `src/compact.ts`, the scoring engine, which leans on `src/state.ts` to pair tool calls with their results and fit the conversation state into a token budget. `src/request.ts` builds and validates the wire format, `src/client.ts` carries it over HTTP, and `src/types.ts` defines the shared shapes every module speaks. At the bottom, `src/messages.ts` and `src/index.ts` expose the same engine as a standalone npm package that `examples/demo.ts` exercises without Claude Code at all.

## Why You Need This

The core problem is information loss at the worst possible time. The built-in path asks a model to write a prose summary of everything that happened. Summaries are good at narrative and bad at payload: an error message with a stack trace, a `file_path` argument, a command that half-worked, the user's instruction to never edit a generated directory — these survive only by luck, and they are exactly what an agent needs later.

fast-jev-compaction never rewrites anything. Its only output operation is deletion: it removes the tool calls and results Jev scores as no longer needed, truncates a dropped result to its first `truncateHeadChars` characters plus a one-line note, and returns everything else as the original objects it received. As the state preamble in `src/state.ts` puts it, whatever is not kept is deleted permanently, but the assistant can always re-run a tool or re-read a file. A hallucinated detail in a summary cannot be re-run.

The plugin also controls when compaction happens and what counts as success. A second hook on `turn.complete` requests compaction once the session's context percentage crosses `compactAtPercent` (60 by default), guarded by an in-flight flag so it cannot recurse. And because a scoring pass can come back with almost nothing to delete, the hook compares the estimated reduction against `minReductionRatio` (25 percent) — if not enough was removed, it logs why and delegates to the built-in summary instead of pretending the problem was solved.

It is also observable. Every compaction logs the per-call decisions with both probabilities plus a stats block, and a toast in the Claude Code UI tells you whether the pruned history replaced the summary or the hook fell back. You are never left wondering what compaction did to your session.

## How It Works

The pipeline runs in five stages — pairing, state fitting, question batching, scoring over HTTP, and the rebuild — and the detailed graph below maps each stage to the file that owns it.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/fast-jev-compaction/tamaratran-fast-jev-compaction-architecture.svg" alt="Detailed architecture of the tamaratran/fast-jev-compaction repository" style="max-width:100%;height:auto;" />
</div>

*Detailed architecture of tamaratran/fast-jev-compaction: package and demo metadata, the Claude Code plugin surface with its hook and type reference, the six library modules of `src/` and their import graph, and the two Vitest suites that hold everything together.*

### Understanding the Architecture

**The pairing pass.** `collectToolCalls` in `src/state.ts` pairs every `tool_use` block with its `tool_result` by `tool_use_id`, assigning short ids (`t1`, `t2`, ...) used in the Jev state and questions. A call without a result is not a candidate — there is nothing to drop yet. Calls in the first message or in the newest `preserveRecentMessages` messages (six by default) are `pinned` and never become candidates, so the start of the session and the work in flight are always preserved.

**The state.** Jev sees the whole conversation in one `CompactionState`: a `context` preamble explaining the compaction, a `goal` (the last three user prompts, truncated to 500 characters each, by default), and a `history` array oldest-first. Every tool result appears only as a short note such as `ok, 4213 chars (omitted)`; tool inputs and message texts are included in full. `fitState` then shrinks the state in ordered stages until it fits `maxStateTokens` (25,000 estimated tokens): tool inputs truncated to 1000, then 200, then 60 characters; long texts abridged to a 400-character head plus a 150-character tail, oldest non-pinned messages first; old texts collapsed to a `[… N chars omitted …]` note; old tool calls reduced to one line each (`t12 Read file_path=src/a.ts → ok 480ch`); old messages without calls left out; runs of old call-only messages folded together. If even that does not fit, compaction throws rather than corrupting the picture Jev sees. Token sizes are estimated without a tokenizer — a word costs one token per six letters, a digit half a token, any other symbol nine tenths — calibrated in `estimateTokens` to land slightly above the usage Jev itself reports.

**The questions.** For every non-pinned call, `questionsFor` in `src/compact.ts` emits two `noul` questions — Jev's yes/no probability question type. `call_<id>` asks whether the call itself should stay, knowing it was made and with its input; `result_<id>` asks whether the full output should stay verbatim, because its contents are still needed and re-running the tool would not do. `batchCalls` splits the questions into as many requests as needed so that the state plus one batch stays under `maxRequestTokens` (30,000, under Jev's request limit), and the same complete state is resent with every batch. In `compact`, all batches fire concurrently through `Promise.all` and their answers are merged.

**The wire.** `src/request.ts` owns the protocol: `buildJevRequest` produces a POST to `https://api.typesafe.ai/v1/systemone` with a Bearer key and a body of `model` (default `jev-latest`), `state`, and `questions`; `parseJevResponse` validates that the reply contains an `answers` object; `noulAnswer` extracts one finite numeric probability or throws. `src/client.ts` wraps this in `JevClient`, whose key defaults to `process.env.TYPESAFE_API_KEY`. The plugin hook builds its own asker over Claude Code's `$.http.fetch`, reusing the same request builders — one protocol, two transports.

**The rebuild.** `decideCall` compares the two probabilities against `keepThreshold` (0.5): `keepResult` at or above it keeps call and result; otherwise `keepCall` at or above it keeps the call and truncates the result to its first `truncateHeadChars` (300) characters plus a note reading `[fast-jev-compaction truncated N chars of this tool result; re-run the tool if needed]`; otherwise call and result disappear together. `applyDecisions` then rebuilds the message list: messages that lose all their content are removed, no result is ever left without its call, and untouched messages are returned as the very same objects they arrived as. That identity matters — `toSessionMessages` in `hooks/fast-jev.ts` maps the output back onto session messages so the engine keeps its own handles for untouched content and takes only the edited messages from the plugin.

**End to end.** A turn finishes, `turn.complete` sees the context above 60 percent, and the hook calls `$.session.compact()`. The `session.compact` handler resolves the API key from plugin options, the environment, or settings, pairs the calls, fits the state, fires the concurrent Jev requests, applies the decisions, and either replaces the history with the pruned originals (logging `kept N/M messages, no summary` with the stats) or, on any Jev failure, malformed response, missing key, unfittable history, or an insufficient reduction, notifies the user and calls `next(event)` so the built-in summary takes over.

## Advantages

- **Verbatim where it matters.** Kept tool results are the original bytes — exact error text, paths, and command output survive compaction untouched.
- **Deletion-only, therefore recoverable.** The agent can always re-run a tool; the design leans on that instead of trusting a summary to preserve facts.
- **Typed decisions, not prose.** Every call yields a `CallDecision` with an action, a reason (`pinned`, `kept`, `result_dropped`, `call_dropped`), and both probabilities — data you can log, test, and audit.
- **Disciplined token budgeting.** Staged state fitting and request batching keep every Jev call under its limits, with a tokenizer-free estimator that overestimates slightly rather than overflowing.
- **Graceful fallback everywhere.** Missing key, Jev failure, malformed answers, unfittable history, or a reduction below 25 percent all degrade to the built-in summary.
- **Dual packaging.** The same engine works as a Claude Code plugin and as a plain npm library with injectable fetch.

## Benefits

- **Long sessions keep their anchors.** The first message and the most recent six are pinned, so the original task statement and the work in flight are never candidates.
- **Honest accounting.** `result.stats` reports message and character counts before and after, per-reason decision counts, state tokens, the fitting stage, request count, and duration.
- **Configurable strictness.** Keep threshold, preserved recency, state and request budgets, truncation length, trigger percentage, and minimum reduction are all exposed as options.
- **Debuggable decisions.** The hook logs a per-call `decisions:` line with both probabilities, so you can see exactly why a result was truncated.
- **Testable without the network.** The Vitest suites drive the engine through a fake Jev asker; the demo script is the only live-network code.
- **No tokenizer dependency.** The character-based estimator avoids shipping a tokenizer while staying calibrated against the usage Jev reports for real transcripts.

## Usage

As an npm library:

```sh
npm install fast-jev-compaction
export TYPESAFE_API_KEY=...
```

```ts
import { compactMessages, reductionRatio } from 'fast-jev-compaction';

const result = await compactMessages(transcript, { preserveRecentMessages: 4 });
console.log(result.messages, result.decisions, result.stats);
if (reductionRatio(result) < 0.25) {
  // not worth it: keep the original transcript, or summarize instead
}
```

As a Claude Code plugin, enable function hooks and provide the key wherever Claude Code runs — for example in `~/.claude/settings.json`:

```json
{
  "env": {
    "CLAUDE_CODE_ENABLE_FUNCTION_HOOKS": "1",
    "TYPESAFE_API_KEY": "<your key>"
  }
}
```

Then install from the repository's own marketplace:

```sh
claude plugin marketplace add tamaratran/fast-jev-compaction
claude plugin install fast-jev-compaction@fast-jev-compaction
```

Restart Claude Code or run `/reload-plugins`. From then on `/compact` and auto-compaction go through Jev, with the toast announcing either the pruned history or a fallback. To run from a checkout without installing:

```sh
CLAUDE_CODE_ENABLE_FUNCTION_HOOKS=1 claude --plugin-dir .
```

## Conclusion

fast-jev-compaction is a rare thing in agent tooling: a compaction strategy with a falsifiable policy. It replaces "summarize the past" with "score the past, then delete what no longer matters," keeps everything else byte-identical, measures its own reduction, and stands aside when it cannot beat the built-in summary. The code is compact, well-typed, and honest about its limits — token sizes are estimates, probabilities are not proofs. If you build on Claude Code, or on any framework with a compaction seam, this repository is one of the most instructive source tours available.

Links:

- GitHub repository: [tamaratran/fast-jev-compaction](https://github.com/tamaratran/fast-jev-compaction)
- Hook configuration and Claude Code type reference: [hooks/README.md](https://github.com/tamaratran/fast-jev-compaction/blob/main/hooks/README.md)
- Claude Code plugins documentation: [code.claude.com/docs/en/plugins](https://code.claude.com/docs/en/plugins)
