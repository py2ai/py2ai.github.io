---
layout: post
title: "Jev Review: Continuous Quality Review MCP Plugin - Inside NiazMorshed2007/jev-review"
description: "A source-level tour of NiazMorshed2007/jev-review, a local-first MCP plugin that gives AI coding agents a repeated, structured software-quality feedback loop. We walk through how review passes are triggered from the agent loop, the single jev_review tool surface, and how Jev turns focused diffs into per-dimension scores."
date: 2026-09-30
header-img: "img/post-bg.jpg"
permalink: /Jev-Review-Continuous-Quality-Review-MCP-Plugin-Source-Tour/
featured-img: "ai-coding-frameworks/ai-coding-frameworks"
image: https://pyshine.com/assets/img/diagrams/jev-review/niazmorshed2007-jev-review-architecture.svg
tags:
  - MCP
  - Code Review
  - AI Coding Agents
  - Software Quality
categories: [AI, Open Source]
keywords: "jev review, MCP plugin, AI coding agents, code review automation, software quality metrics, Claude Code, Codex, Cursor, OpenCode, local-first MCP server, jev_review tool, code quality scores, NiazMorshed2007, pyshine"
author: "PyShine"
---

AI coding agents are now good at producing code quickly, but the feedback they get back is usually binary: tests pass or they do not, the build compiles or it does not. What is missing is a cheap, structured signal about the *quality* of the change — is it readable, is it modular, is it secure, will the next person be able to change it without fear? **Jev Review**, from GitHub user NiazMorshed2007, is a small, sharply scoped TypeScript project that tries to fill exactly that gap. It ships as a Model Context Protocol (MCP) plugin that your coding agent can call while it works, receiving an independent 1–10 score per quality dimension instead of a wall of prose.

The project describes itself as a local-first MCP server for continuous software-quality review, powered by [Jev](https://typesafe.ai/). The design is deliberately narrow: one MCP tool, `jev_review`, no repository crawling, no daemon watching your files, no hosted backend beyond the direct call to the configured Jev API. The plugin works with Claude Code, Codex, Cursor, and OpenCode, and every client starts the exact same bundled Node.js process over stdio. The coding agent remains responsible for diagnosing weaknesses and editing code; Jev Review only supplies the scalar signal that tells the agent whether its last change actually moved the needle.

The source is worth a tour because it is a clean case study in doing one thing well. The whole evaluation engine lives in a handful of TypeScript modules under `src/`, the MCP boundary is about sixty lines, and the agent-facing behavior is encoded in a skill document and server instructions rather than hidden inside heuristics. You can read the entire pipeline — from an agent calling a tool, to a scored evaluation, to a score-delta comparison — in a single sitting, and every design decision in it is visible.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/jev-review/niazmorshed2007-jev-review-overview-architecture.svg" alt="Architecture overview of the NiazMorshed2007/jev-review repository" style="max-width:100%;height:auto;" />
</div>

*Overview of the Jev Review pipeline: the agent loop drives a single MCP tool, whose input is validated locally, orchestrated through the evaluation pipeline, and scored by the Jev HTTP client.*

Reading the overview from left to right: the agent loop, taught by the bundled skill at `skills/jev-review/SKILL.md`, drives calls into the MCP boundary in `src/mcp/server.ts`, which is started as a stdio server by the one-line entry point `src/server.ts`. The boundary validates arguments with the Zod schema in `src/evaluation/input.ts` and delegates to the review orchestrator in `src/evaluation/review.ts`, the heart of the Evaluation Pipeline group. The orchestrator builds the question rubric from `src/evaluation/questions.ts`, which itself derives every criterion from the metric definitions in `src/evaluation/metrics.ts`, sends the focused code state to the Jev HTTP client in `src/jev/client.ts`, and finally converts Jev's raw answers into a structured evaluation in `src/evaluation/transform.ts`, shaped by the schemas in `src/evaluation/types.ts`. On the far side, `src/config/environment.ts` supplies the local API key, and `src/jev/schema.ts` guarantees that whatever Jev returns actually matches its documented response shape.

## Why You Need This

The first problem Jev Review attacks is that "review" for AI agents is usually either absent or ad hoc. An agent that finishes a feature and stops has no reason to ask whether the implementation is cohesive or changeable — it just moves on. By registering an MCP tool with server instructions that explicitly say a single baseline call is not completion, Jev Review turns quality checking into part of the working loop: implement, score, inspect, improve, rescore.

The second problem is that prose reviews do not compose. A free-form paragraph of feedback cannot be compared against yesterday's feedback. Jev Review's response is a typed structure — per-metric score, confidence, prioritized weak dimensions, and, when the agent passes the previous evaluation back, per-metric deltas split into improvements and regressions. That makes progress measurable across an entire working session, not just narrated once.

The third problem is trust boundaries. Many review tools want filesystem access, hooks, or cloud accounts. Jev Review's MCP server never reads your repository automatically. The caller must explicitly send a `task`, a focused `diff`, optional `files`, and optional `repositoryContext` — and the Zod validation in `src/evaluation/input.ts` refuses calls that contain none of them. Your `JEV_API_KEY` is read from the environment once per request and used only in the TLS Authorization header sent directly to the Jev API.

Finally, it resists the classic failure mode of automated scoring: gaming. The skill document and the tool description both warn the agent against speculative architecture, meaningless tests, mechanical file splitting, and scope expansion to chase numbers. Correctness and the user's requirements always outrank score improvement, and the project deliberately refuses to synthesize a single blended "overall" percentage.

## How It Works

The entire runtime is one committed Node.js bundle driven by a small set of composable TypeScript modules, and the detailed diagram below maps every file that participates in a review pass.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/jev-review/niazmorshed2007-jev-review-architecture.svg" alt="Detailed architecture of the NiazMorshed2007/jev-review repository" style="max-width:100%;height:auto;" />
</div>

*Detailed architecture of NiazMorshed2007/jev-review: packaging manifests, the MCP boundary, the evaluation pipeline, Jev integration, and the local test suite that keeps all of it honest.*

### Understanding the Architecture

**The packaging layer is intentionally thin.** `plugin.json` and `mcp.json` form a portable Agent Plugins 1.0 package declaring a stdio MCP server that runs `node ./dist/server.js`. The `.claude-plugin/plugin.json`, `.codex-plugin/plugin.json`, and `.mcp.json` files are small adapters so Claude Code and Codex can install the same implementation, and `npx plugins add NiazMorshed2007/jev-review` wires the chosen client up from the repository directly — there is no npm publication.

**The MCP surface is one tool with strict typing on both ends.** `src/mcp/server.ts` registers `jev_review` with `reviewInputSchema` as its input schema and `evaluationSchema` as its output schema, so the agent sees a machine-readable contract rather than free text. The tool is annotated as read-only and non-destructive, errors are returned as `isError` text messages, and a joined `SERVER_INSTRUCTIONS` string tells the agent how to use the tool as a repeated feedback loop — baseline first, then rescore with the prior response in `previousEvaluation`.

**Review passes are triggered by the agent loop, not by hooks.** There is no file watcher and no post-commit trigger anywhere in the code. Instead, `skills/jev-review/SKILL.md` teaches the agent the required loop: after the first coherent implementation slice, call `jev_review` to establish a baseline; inspect the code yourself and form a hypothesis about why an important dimension is weak; make the smallest justified improvement; validate; then call again, passing the previous structured response unchanged as `previousEvaluation`. The skill also defines stopping conditions — stop when requirements pass, no justified improvement remains, or further score-seeking would add real risk.

**The evaluation pipeline builds a 19-metric rubric locally.** `src/evaluation/metrics.ts` defines nineteen quality dimensions — from correctness and cognitive complexity to coupling, changeability, security, and consistency — each with guidance, a suggestion, a map of named weaknesses, a priority weight, and a `conditional` flag. Four dimensions (performance, scalability, compatibility, observability) are conditional and only assessed when the supplied context contains concrete evidence. `src/evaluation/questions.ts` turns each definition into three typed Jev questions: an applicability decision, a 1–10 score against a fixed legend, and a single-consequence weakness choice.

**All the local computation happens in the transform.** `src/evaluation/transform.ts` validates Jev's answers, converts the raw 0–9 score into the reported 1–10 scale, caps confidence by the applicability certainty, and attaches an issue with a severity and a predefined suggestion when a score drops below 8 with a material weakness. It then ranks applicable metrics below 8 into at most five priorities using the formula `score − priorityWeight × 0.35`, so correctness (weight 3) and security outrank style-adjacent dimensions. When `previousEvaluation` is present, the comparison is computed entirely locally — the previous evaluation is stripped from the state sent to Jev by `toJevState` in `src/evaluation/input.ts`, and deltas of at least 0.75 are labeled improved or regressed.

**The Jev client is the only place that touches the network.** `src/jev/client.ts` posts the focused state, the question set, and the model name `jev-latest` directly to `https://api.typesafe.ai/v1/systemone` with the bearer key, using a 30-second timeout and up to two retries on 429, 529, and 5xx responses with capped exponential backoff. Every successful response is validated against the Zod schemas in `src/jev/schema.ts`, and specific error paths — an exceeded input ceiling, a rejected key, rate limiting — are translated into actionable messages the agent can act on. The README notes that Jev's live token ceiling is roughly 32,768 tokens for the submitted state, and on `max_tokens_exceeded` the server asks the agent to shrink the context or split the change into coherent review slices.

End to end, a review pass looks like this: the agent implements a slice, calls `jev_review` with its task and focused diff; `src/mcp/server.ts` validates the arguments; `src/evaluation/review.ts` parses the input, reads `JEV_API_KEY` via `src/config/environment.ts`, assembles the state and rubric, and dispatches through `JevClient`; Jev scores the state; `src/evaluation/transform.ts` converts the answers into a validated `Evaluation`; and the agent receives per-metric scores, priorities, and — on follow-ups — concrete improvements and regressions it can act on immediately.

## Advantages

- **One focused tool surface.** A single `jev_review` tool with typed input and output schemas keeps the agent's decision simple and makes the whole integration auditable in one file.
- **Local-first and private by construction.** The server runs as a local stdio process, never discovers or uploads repository files on its own, and only forwards the context the caller explicitly supplied.
- **Structured, comparable output.** Per-metric scores, confidence, prioritized weaknesses, and locally computed deltas turn quality review into data rather than prose.
- **Anti-gaming discipline.** Server instructions, the tool description, and the skill all explicitly forbid score-chasing through scope expansion, meaningless tests, or speculative abstraction.
- **No synthetic blended score.** Reporting `Readability 6.3 → 8.1` instead of an "82/100" keeps the signal honest and actionable per dimension.
- **Context-aware scoring.** Conditional metrics are only evaluated when evidence exists, and the rubric judges consequences in context rather than enforcing shallow rules like "more comments are better."

## Benefits

- **Faster quality loops.** The agent gets a scalar signal seconds after a coherent slice, catching weak dimensions while the change is still fresh rather than at review time.
- **Measurable improvement across a session.** Passing `previousEvaluation` back yields concrete improvements and regressions, so an agent can verify that its fix actually worked.
- **Clean separation of concerns.** Jev scores; the agent diagnoses and edits. The plugin never blurs responsibility for changing the code.
- **Portable across clients.** The same bundled server backs Claude Code, Codex, Cursor, and OpenCode, with packaging adapters instead of client-specific forks.
- **Robust remote integration.** Timeouts, retries, capped backoff, and schema validation mean flaky API responses fail loudly and legibly instead of corrupting results.
- **Well-tested core.** The `test/` directory covers the MCP protocol boundary, input validation, question construction, the score transform, and client retries using local fakes that consume no API quota.

## Usage

Set your Jev API key before starting the coding agent:

```bash
export JEV_API_KEY="your-key"
```

Install the plugin directly from GitHub — no npm publication is required:

```bash
npx plugins add NiazMorshed2007/jev-review
```

The installer prompts for your client (Claude Code, Codex, or Cursor). After restarting the client, ask the agent to use `jev-review` while implementing a nontrivial change. A typical tool call sends the task and the focused diff:

```json
{
  "task": "The requested behavior and acceptance constraints",
  "diff": "The current implementation diff after the latest changes",
  "repositoryContext": "Relevant conventions, invariants, and validation results"
}
```

For local development from a clone:

```bash
git clone https://github.com/NiazMorshed2007/jev-review.git
cd jev-review
npm install
npm run validate
```

`npm run validate` type-checks, runs the unit and MCP protocol tests, and rebuilds the committed `dist/server.js` bundle with esbuild.

## Conclusion

Jev Review is a refreshing example of restraint in the AI-tooling space. Rather than trying to be an autonomous reviewer or another agent framework, it reduces quality review to a single typed MCP tool, a well-documented agent loop, and a clean evaluation pipeline you can read in one sitting. The scoring itself is outsourced to the Jev API, while everything that can be deterministic — validation, rubric construction, answer checking, delta computation, and prioritization — stays local and testable. If you are building with Claude Code, Codex, Cursor, or OpenCode and want your agent to care about maintainability and not just green tests, this small TypeScript plugin is an easy add and an instructive read.

Links:

- GitHub repository: [NiazMorshed2007/jev-review](https://github.com/NiazMorshed2007/jev-review)
- Jev platform: [typesafe.ai](https://typesafe.ai/)
- TypeSafe console (API keys): [console.typesafe.ai](https://console.typesafe.ai/)
