---
layout: post
title: "Cortico: An Event-Stream Harness For Agents That Never Sleep"
permalink: /Cortico-An-Event-Stream-Harness-For-Agents-That-Never-Sleep/
image: https://pyshine.com/assets/img/diagrams/cortico/pal-ai-lab-cortico-architecture.svg
tags: [TypeScript, AI Agents, Node.js, Streaming, Minecraft]
---

Most agent frameworks are built around a conversation: a request comes in, the model answers, the session ends. [Cortico](https://github.com/Pal-AI-Lab/Cortico) by Pal AI Lab - v0.1.5 pre-release, MIT, TypeScript on Node 22+ - is built around a different primitive: the event stream. It is a harness for autonomous, continuously running agents that juggle mixed real-time input - persona bots, AI streamers, roleplay companions - where a gift donation, a chat message, a game event, and a timer can all arrive at once and the agent has to stay coherent through all of it.

## Four layers, strictly separated

The architecture holds five concerns apart (core, persona, memory, world, bot assembly), and the split is enforced by the directory layout. `src/core/` is deliberately semantics-free: it manages session lifecycles, the event stream, and model invocations, and knows nothing about what the agent is for. The persona - defined per bot under `bots/<name>/persona/` - carries the actual meaning: context synthesis, the cognitive loop, and memory protocols. Memory is the authoritative persistence store under `<deployment>/memory/`, with layout and lifecycle chosen by the persona rather than imposed by the framework - a refreshing inversion, since most frameworks hard-code their memory philosophy. A World (`src/worlds/<id>/`) is the isolated boundary to an external environment with event ingestion, tool declarations, and environment prompts; and a Bot (`bots/<name>/index.ts`) simply couples one persona with a designated set of worlds.

The event plumbing lives in `src/core/bus.ts` with a durable log in `src/core/event-store.ts`, and the agent loop in `src/core/loop.ts` - a substantial 80 KB module - consumes from it and drives model calls through the provider transport. `src/core/instance-lock.ts` keeps a single owner per deployment, `src/core/cost.ts` meters token spend, and `src/core/types.ts` (45 KB) declares the CoreApi contract that personas program against. The headline: one bot can simultaneously observe and act across chat platforms, a live game, and a physical environment, because worlds are just event sources and tool sinks.

## Worlds: Minecraft, Bilibili, QQ, terminal, web search

The built-in worlds show the range. The Minecraft world is enormous - `src/worlds/minecraft/world.ts` alone is over 300 KB, with a 232 KB `executor.ts`, dedicated modules for combat, melee and ranged tactics, terrain, blueprints with repair and resource planning, placement, chests, inventory, deaths, and a body-lease system so concurrent agents do not fight over the same avatar. It wraps mineflayer and the pathfinder, with performance patches and a receipt module that records what the agent actually did. This is not a demo toy; it is a long-running autonomous player.

The Bilibili world (`src/worlds/bilibili/world.ts`) ingests a live stream: audience admission policy, a coalescing buffer for bursty chat, gift frames, protobuf wire decoding, and a full OBS overlay stack - an editor UI with its own web app for the streamer to customize. The QQ world handles Chinese IM with vision: a VLM chain (`vision.ts`, `vlm.ts`) that describes images before the main model sees them, image downloads, and a console gate for human approval of roster changes. Terminal and web search (via a Brave client) round out the set. External extensions follow the same contract through `src/extensions.ts` with manifests and dry-mount testing (`src/extensions/dry-mount.ts`), so a third-party world is indistinguishable from a built-in one.

## Providers on the Responses protocol

Internally, everything speaks the Responses protocol: `src/protocol/open-responses/` carries the OpenAPI document (138 KB) with generated types and a streaming implementation. The provider registry resolves a configured endpoint to a transport; `src/providers/transport/` handles chat translation, native input, history, response assembly, HTTP details, and response meters. Local models are a first-class provider: the llama.cpp integration (`src/providers/llamacpp/runtime.ts`) downloads and manages runtimes, catalogs models from Hugging Face, and spins up a managed server - with its own console panels for runtime and model management. An OpenAI-responses-compat adapter covers relay endpoints.

## A console that treats the operator as a first-class user

`src/web/server.ts` (98 KB) serves the management console on `127.0.0.1:7788`, with auth (`src/web/auth.ts`) in front. The client (`src/web/client/main.ts`) is a feature-modular single-page app: a live timeline of what the agent is experiencing, a prompt editor, provider and pricing panels, usage charts, world dashboards, and an extension manager. Configuration changes apply to the next request; a managed runtime's launch parameters apply on its next start - small semantics, but the kind that make a long-running deployment operable. Deployments themselves are isolated configuration directories under `deployments/`, created with `pnpm start --new`, and the launcher rebuilds web assets when they are stale and restarts processes on request from the console.

The companion projects reveal the ambition. Cortina is an extension-creator workspace where a non-developer describes the extension they want to a coding agent and publishes the result to npm - built on the lab's TINA spec ("There Is No App: language is the code, the agent is the runtime"). And cortico-world-vtuber (AGPL-3.0 with a CLA, kept out of the main tree by license) drives a Live2D model through VTube Studio, speaks through streaming TTS, times subtitles with a forced aligner, and pushes overlays into OBS.

Pre-release software, honestly labeled - but the architecture is the most complete open answer I have seen to a question that is only getting louder: what does a harness look like when the agent is not answering a prompt but living in an environment?
