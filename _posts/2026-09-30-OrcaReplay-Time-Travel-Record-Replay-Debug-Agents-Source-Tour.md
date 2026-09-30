---
layout: post
title: "OrcaReplay: Time Travel for AI Agents - Inside Continuum-AI-Corp/OrcaReplay"
description: "A source tour of OrcaReplay, the TypeScript toolkit that records any coding agent run, replays it byte-for-byte with no model called, forks it from any checkpoint onto a different model, and turns every run into a debuggable causal timeline. We walk the proxy, the trace format, the matching ladder and the fork engine."
date: 2026-09-30
header-img: "img/post-bg.jpg"
permalink: /OrcaReplay-Time-Travel-Record-Replay-Debug-Agents-Source-Tour/
featured-img: "ai-coding-frameworks/ai-coding-frameworks"
image: https://pyshine.com/assets/img/diagrams/orcareplay/continuum-ai-corp-orcareplay-architecture.svg
tags:
  - AI Agents
  - Debugging
  - TypeScript
  - Developer Tools
categories: [AI, Open Source]
keywords: "OrcaReplay, agent debugging, record and replay AI agents, deterministic replay, agent fork, checkpoint fork, TypeScript, LLM proxy, trace format, MCP server, Claude Code recording, offline replay, model comparison, coding agents, open source"
author: "PyShine"
---

Your coding agent just deleted a file, "fixed" a test that is still failing, and exited 0. What do you do at that point? Today the honest answer is archaeology: scroll a terminal transcript, re-run and get a different failure, sprinkle print statements into a harness you did not write. Observability dashboards will tell you what the run cost, which is not the question you have. The question you have is *why did it do that* — and that question needs the run back, not a chart of it.

OrcaReplay, from Continuum-AI-Corp, is a TypeScript toolchain that answers by giving you exactly that. It records any coding agent — Claude Code, Codex, goose, grok-cli, opencode, plain Python pipelines — through a local proxy plus several capture layers, stores the whole run as a file, then replays it offline byte-for-byte with no model called and no tokens spent. From any checkpoint it can fork the run onto a different model and let the two race with the model as the only variable.

The source is worth a tour because the interesting engineering is not in a demo, it is in the machinery: a normative trace specification, a proxy that parses several wire dialects, a matching ladder that refuses to guess, and a checkpoint system that is *derived* from the log rather than recorded into it. It is a monorepo of focused packages where every design decision is argued about in comments, and reading it teaches you what "time travel for agents" actually costs to build.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/orcareplay/continuum-ai-corp-orcareplay-overview-architecture.svg" alt="Architecture overview of the Continuum-AI-Corp/OrcaReplay repository" style="max-width:100%;height:auto;" />
</div>

*Overview of the OrcaReplay architecture: the CLI surfaces, the five capture layers, the trace store, and the replay engine that reads it back.*

Reading the overview from left to right: `orca record` (packages/cli/src/commands/record.ts) launches your agent through an adapter (packages/adapters/src/registry.ts) and points the agent's base-URL environment variables at the capture proxy (packages/proxy/src/server.ts). Side channels that the model protocol cannot see — shell commands, filesystem writes, MCP frames, hardcoded fetch calls — come from the PATH shim, the shadow git snapshotter, the MCP tee and the fetch hook. Every layer drains into one append-only trace store (packages/core/src/writer.ts) validated against the orca-trace v0 schema (packages/schema/src/validate.ts). On the right, the replay engine (packages/cli/src/commands/replay.ts) reads that trace back, serves recorded turns through the same proxy, materializes the recorded workspace from the shadow git store, and opens the single-file timeline viewer (packages/viewer/src/html.ts).

## Why You Need This

The first problem is that agent failures are not reproducible by re-running. Model APIs are non-deterministic: a second run makes different tool calls, hits different edge cases, and "fixes" a different bug than the one you saw. You cannot debug a failure you cannot summon on demand. OrcaReplay's answer is that the recording *is* the reproduction — `orca replay last` serves the conversation back from disk with egress blocked, executes the recorded tool calls for real, and reports zero divergences without a single token being billed.

The second problem is that the model transcript is a partial record. A run's exit code hides the truth: the agent's own log can show a clean finish while the shell check it ran exited 1 and the file it edited still has the wrong diff. OrcaReplay captures below the agent — at the process and socket boundary — so its timeline interleaves model turns with the shell commands' real exit codes and durations, the per-turn filesystem snapshots with diffs, and MCP frames, all ordered by when they actually happened rather than when they were drained.

The third problem is comparing models fairly. Anyone who has asked "would a cheaper model have got this right?" knows the trap: change the model and you also change the prompt the harness assembled, the context compaction point, everything. OrcaReplay forks a run from a checkpoint that provably existed — same files, same conversation prefix — and changes exactly one variable. `orca compare` grades each fork with a command you choose, such as a test suite, and prices the outcomes.

The fourth problem is reach. SDK wrappers only see agents you can edit and that hold an API key you can redirect. OrcaReplay's capture happens beneath the harness: base-URL variables for the willing, a fetch-hook preload for Node agents with hardcoded origins, and an opt-in TLS interception mode that mints a per-run certificate authority for agents — like a Codex CLI signed in with a subscription — that read no variable at all. There is even `orca attach` for agents running in a dev container or on another machine.

## How It Works

The whole system hangs off one property of stateless model APIs: every turn resends the entire conversation, so a proxy in front of the model sees the complete loop — requests, streamed responses, tool calls, tool results.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/orcareplay/continuum-ai-corp-orcareplay-architecture.svg" alt="Detailed architecture of the Continuum-AI-Corp/OrcaReplay repository" style="max-width:100%;height:auto;" />
</div>

*The detailed architecture: CLI commands, the capture layers, the trace store and its schema, the model dialect translators, and the views that read it all back.*

### Understanding the Architecture

**The trace format is the contract.** `spec/orca-trace-v0.md` is a normative, CC BY 4.0-licensed specification of one run: a manifest with an integrity root, an append-only `events.jsonl` with a dense `seq` order and a causal `causes` DAG, content-addressed blobs, and a shadow git object store under `fs/`. TypeScript types and JSON Schema in packages/schema/src are generated from or verified against that spec, never the reverse, so third-party readers can trust the file. Payloads over 4096 serialized bytes are spilled to blobs addressed by their SHA-256 digest — and because every turn resends the whole conversation, content addressing is what keeps a trace linear in new content rather than quadratic in turns.

**The writer is deliberately paranoid.** packages/core/src/writer.ts funnels every event through the same path — spill, redact, validate, append — because, as the comment puts it, a redactor that one caller can bypass is not a redactor. Scrubbing happens on serialized bytes rather than individual fields, so a secret hiding in a nested string is still caught; files are written 0600 into 0700 directories. The blob store in packages/core/src/blobs.ts and the redactor in packages/core/src/redaction.ts do the heavy lifting, and `orca scrub` (packages/cli/src/commands/scrub.ts) can rework a trace after the fact.

**Capture happens in layers below the agent.** The proxy (packages/proxy/src/server.ts) claims requests by wire dialect (packages/proxy/src/dialects.ts) — Anthropic, OpenAI chat, and the OpenAI Responses API — with translators in packages/providers/src/translate/. The PATH shim (packages/shell-shim/src/runner.ts) wraps spawned commands so the trace gets real exit codes, durations and the stdout/stderr split. The shadow git index (packages/fs-capture/src/shadow.ts) stages the workspace once per turn into a run-private git store, keeping raw bytes with no end-of-line normalization because the recording should stay honest even when Windows CRLF makes a one-character fix look like a full-file rewrite. The MCP tee (packages/mcp-shim/src/shim.ts) and the fetch-hook rewrite (packages/node-instrument/src/rewrite.ts) round out the layers for tool servers and hardcoded origins.

**Replay is a matching problem, and the matcher refuses to lie.** packages/proxy/src/matching.ts implements a four-rung ladder: agent harnesses are not deterministic, so a replayed request may differ in timestamps, generated ids or working directories, and matching degrades through exact, near-exact and weaker rungs. The governing rule is printed in the source: replay never silently approximates — anything below the exact rung produces a divergence the caller must record. Requests a harness makes for itself (quota probes, session naming) are only ever skipped onto a strong match and reported through a `reused` count, because a silently skipped exchange is a replay that lies.

**Checkpoints are derived, never recorded.** packages/core/src/graph.ts computes forkable points from the log itself: a checkpoint needs an `fs.snapshot` at or before it in the same turn and a complete conversation prefix — everything after an unanswered `model.request` is un-forkable, because the model's effect on the run is unknown there. The same module builds the causal graph behind `orca graph`, which separates *recorded* edges (a tool_use block physically inside the response that emitted it) from *inferred* ones (a file change attributed to a call by rule), and never writes inferred edges back into the trace.

**The same proxy serves the fork.** At replay time, packages/cli/src/commands/replay.ts positions a cursor in the recorded stream: everything before it is answered from disk with the network blocked, everything after it — if you passed `--from N --model X` — is answered live by the model you named, translated to the wire format the agent speaks. The replay restores the recorded workspace over your tree from the shadow store and puts it back, writes a run of its own recording what it *discovered* (divergences, unmatched requests), and is itself a forkable run. `orca compare` (packages/cli/src/commands/compare.ts) runs that fork several times from one checkpoint and grades each with your verify command, pricing results through packages/providers/src/pricing.ts.

End to end: you type `orca record claude`; the adapter launches Claude Code unmodified with its base URL pointed at the local proxy and installs the shim, snapshotter and tees; the agent works, and every model exchange, shell command, filesystem change and MCP frame lands in one validated, redacted, content-addressed trace under `.orca/runs/`; later, `orca show`, `orca graph` or the single-file timeline viewer lets you read what actually happened; `orca replay` reproduces it offline; `--from` and `--model` fork it; `orca compare` turns the whole thing into a verdict table.

## Advantages

- **Byte-for-byte offline replay.** The recorded conversation is served from the trace with egress blocked by default — no model called, no tokens, no variance — while tool calls still execute for real against a restored workspace.
- **Capture without modifying the agent.** Two environment variables and a PATH shim do the work for most harnesses; fetch-hook and TLS-interception modes cover agents that read nothing at all, and none of it patches your agent's code.
- **A file, not a dashboard.** A run is a directory with a JSONL log, content-addressed blobs and a manifest with an integrity root — greppable, exportable to a single self-contained HTML file, and push/pull-able to a gateway.
- **Honest by construction.** Redaction runs on the serialized bytes on the only path to disk, inferred graph edges and derived checkpoints are never written back into the trace, and the matcher reports divergences rather than quietly guessing.
- **Fair model comparison.** Forks start from a provable state — same files, same conversation prefix — so `orca compare`'s verdict table varies the model and nothing else.
- **Readable by machines.** Every command answers `--json` on stdout with diagnostics on stderr, and `orca mcp` exposes the store to an agent over six MCP tools, so an agent can replay and explain its own runs.

## Benefits

- **Debug at 9am what broke at 2am.** The failure is preserved exactly: which tool call wrote the file, what the shell really returned, what the model saw — regardless of what the run's exit code claimed.
- **Spend nothing while investigating.** Replays and timeline reads are local and free; only a deliberate fork onto a live model spends tokens, and the compare table shows what each candidate cost.
- **Trust the recording you attach to an issue.** Causal chain cards and single-file HTML exports carry their own legend and no external references, so the evidence renders anywhere, years later.
- **Onboard to any harness.** The adapter registry already covers Claude Code, Codex, goose, grok-cli, opencode, OpenClaw, a generic OpenAI-compatible shim and more, so support is a matter of which variables your harness reads — and unknown ones can be named per run.
- **Build on it as a library.** The programmatic API returns the same data the commands render, never writes to your stdout and never calls `process.exit`, so scripts and CI can sit on top of it.
- **Keep secrets out.** Credentials are attached to outbound requests only and never reach the trace, redaction is structural rather than a remembered rule, and push to a gateway is refused by default on secret-scan hits.

## Usage

Install the package — it puts one command, `orca`, on your PATH (Node 20 or newer, no native dependencies):

```console
npm i -g orcareplay
orca doctor                       # checks node, git, and which agents it can find
```

Record a real agent run, replay it offline, then fork it from a checkpoint onto a different model:

```console
orca record claude              # your agent, unmodified, doing whatever it does
orca replay last                # the same run again — no network, no tokens, no charge
orca replay last --from 4 --model claude-haiku-4-5 --ui
```

Compare several models from the same checkpoint and grade each fork with your own command:

```console
orca compare last --from 4 --models claude-sonnet-5,claude-haiku-4-5 --verify "npm test"
```

Inspect what actually happened, as a timeline, a causal graph, or a shareable card:

```console
orca show last
orca graph last
orca export last --card bug.svg
orca export last -o bug.html    # the full timeline as one self-contained HTML file
```

Serve the trace store to an agent over MCP, or use it from code:

```json
{ "mcpServers": { "orca": { "command": "orca", "args": ["mcp"] } } }
```

```ts
import { Orca } from 'orcareplay';

const orca = new Orca({ cwd: process.cwd() });
const { unmatched, divergences } = await orca.replay('last');
const timeline = await orca.show('last');
```

To work on the source itself, the README's from-source path is `npm ci && npm run build` followed by `npm install -g ./packages/cli`, with `npm test` running the vitest suite.

## Conclusion

OrcaReplay is a rare thing in the agent-tooling space: instead of adding another observability layer on top of the model API, it goes underneath the harness, records the whole loop into a specified, integrity-checked file, and then makes that file *executable* again — replayed offline, forked at any provable state, compared across models with one variable changed. The source tour rewards attention: the normative trace spec, the spill-redact-validate-append writer, the four-rung matcher that refuses to approximate silently, and the derived-checkpoint design all show a team thinking hard about what a recording must guarantee before you can trust it as a debugger. If you build with coding agents, reading this codebase will change how you think about reproducibility — and using it will change how you debug.

Links:

- GitHub repository: [Continuum-AI-Corp/OrcaReplay](https://github.com/Continuum-AI-Corp/OrcaReplay)
- Trace specification: [spec/orca-trace-v0.md](https://github.com/Continuum-AI-Corp/OrcaReplay/blob/main/spec/orca-trace-v0.md)
