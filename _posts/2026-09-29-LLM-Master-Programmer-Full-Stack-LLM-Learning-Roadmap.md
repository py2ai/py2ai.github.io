---
layout: post
title: "llm-master: A Programmer's Full-Stack LLM Learning Roadmap - Inside youngyangyang04/llm-master"
description: "A guided tour of youngyangyang04/llm-master, a Chinese-language open source curriculum that turns more than 150 LLM tutorials into a six-stage learning path for programmers. We map how its roadmap layer, topic indexes, tutorial library, and interview bank fit together, and how to work through stages from first API call to RAG, agents, deployment, and interviews."
date: 2026-09-29
header-img: "img/post-bg.jpg"
permalink: /LLM-Master-Programmer-Full-Stack-LLM-Learning-Roadmap/
featured-img: "ai-coding-frameworks/ai-coding-frameworks"
image: https://pyshine.com/assets/img/diagrams/llm-master/youngyangyang04-llm-master-architecture.svg
tags:
  - LLM
  - Learning Roadmap
  - RAG
  - AI Agent
categories: [AI, Open Source]
keywords: "llm-master, LLM learning roadmap, large language model tutorial, RAG tutorial, AI agent engineering, prompt engineering, transformer explained, LLM fine-tuning, vLLM deployment, LLM interview questions, Claude Code, MCP protocol, AI coding workflow, kamacoder, open source AI curriculum"
author: "PyShine"
---

Most developers entering the LLM space do not lack material; they lack an order of operations. Every week produces new model releases, new agent frameworks, and new buzzwords, and the default response — reading whatever the algorithm surfaces — leaves you with fragments instead of a working mental model. What makes the problem stubborn is that LLM application engineering is a genuinely layered discipline: you cannot evaluate a RAG system without understanding embeddings and chunking, and you cannot reason about agents without understanding tool calling and context budgets.

[yangyangyang04/llm-master](https://github.com/youngyangyang04/llm-master) — branded "LLM Master" — is a direct answer to that problem. It is a Chinese-language, Markdown-only curriculum from 程序员Carl, the author behind programmercarl.com and the Kamacoder notes, and the material is drawn from the [Kamacoder LLM column](https://notes.kamacoder.com/llm/). The repository bills itself as "a programmer's full-stack LLM learning route" (程序员的大模型全栈学习路线): from the first model API call to evaluable RAG, reliable agents, and production-ready AI systems. It contains just over 150 tutorials and interview write-ups — 153 are indexed in `docs/README.md` — organized not as a dump but as a dependency-ordered course.

What earns it a source tour is that the value lives in the structure, not in any single document. The repo is small in bytes — a `docs/` tree of Markdown, a README, an MIT `LICENSE` — but it is carefully architected as a curriculum: a staged roadmap, a topic index layer, a tutorial library, and an interview bank that all cross-reference each other. Reading how those layers are wired tells you how a serious engineering curriculum is designed, and shows you exactly where to enter it based on your own level.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/llm-master/youngyangyang04-llm-master-overview-architecture.svg" alt="Architecture overview of the youngyangyang04/llm-master repository" style="max-width:100%;height:auto;" />
</div>

*Overview of the llm-master repository: two entry points on the left feed a six-stage roadmap and a topic index, both of which draw from a shared tutorial library, with the interview bank and resume corner closing the loop on the right.*

Reading the overview from left to right: you start at the root `README.md`, which offers three doors — the staged roadmap at `docs/roadmap/README.md`, the topic index at `docs/topics/README.md`, and the master catalog at `docs/README.md`. The roadmap layer fans out into per-stage files (`beginner.md`, `application.md`, `agent.md`, `interview.md`, with stages 2 and 4 routed through the topic layer), and every stage and topic ultimately "reads" or "curates" from the same tutorial library under `docs/llm/`. On the far right, the interview bank at `docs/interview/llm/README.md` and the resume corner at `docs/jianli/README.md` consume everything upstream, which is the repo's way of saying that learning here is always aimed at things you can build and defend in an interview.

## Why You Need This

The first problem llm-master solves is sequencing. The README opens with an observation that will feel familiar to anyone who has tried to self-study this field: the internet has no shortage of LLM material, but no real learning path for programmers. The repo's answer is a six-stage route — Stage 0 "Global Awareness" (全局认知), Stage 1 "Model Invocation" (模型调用), Stage 2 "RAG", Stage 3 "Agent", Stage 4 "Production Engineering" (生产工程), and Stage 5 "Principles and Interviews" (原理与面试) — where each stage has a core goal, a recommended entry file, and a completion marker. For example, Stage 1 is only "done" when you have built an AI application with streaming output, JSON Schema constraints, error handling, and cost accounting.

The second problem is depth versus buzzwords. A lot of free content stops at "what is an agent." This repo consistently pushes one level deeper, toward the engineering questions that actually decide whether something ships: when is a plain workflow better than an agent (`docs/llm/app/agent_vs_workflow.md`), why your RAG answers are wrong with a five-category retrieval taxonomy and a four-category generation taxonomy (`docs/llm/app/rag_problems.md`), or how KV Cache eats VRAM and what PagedAttention does about it (`docs/llm/app/kv_cache_paged_attention.md`).

The third problem is the gap between demos and production. Stage 4 and the deployment topic (`docs/topics/deployment.md`) are devoted to exactly the unglamorous middle: choosing between cloud APIs, hosted inference, and self-hosting with vLLM or SGLang; quantization schemes; load testing with TTFT, TPOT, P99, and Goodput; and capacity planning. The stated completion standard for that stage is a stress-test, capacity, reliability, and cost report for a real RAG or agent service — a deliverable, not a certificate of having watched something.

Finally, there is the career problem. If you are transitioning from Java, C++, Go, Python, or frontend work, you need to convert knowledge into interview performance and resume bullet points. The repo closes that loop explicitly: `docs/interview/llm/README.md` compiles 2026-era question sets across RAG, Agent, Transformer, fine-tuning, and "Vibe Coding," plus real interview experiences such as a four-round ByteDance agent development interview (`docs/interview/llm/20260506bytedance.md`), and the README prescribes a fixed answering structure: business problem → technology selection → system design → evaluation metrics → failure and optimization → final result.

## How It Works

Architecturally, llm-master is a hub-and-spoke curriculum: a small navigation layer (roadmaps and topic indexes) sits on top of a large article library, with an interview layer consuming both, and every arrow in the repository is a Markdown cross-link.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/llm-master/youngyangyang04-llm-master-architecture.svg" alt="Detailed architecture of the youngyangyang04/llm-master repository" style="max-width:100%;height:auto;" />
</div>

*Detailed view of the llm-master knowledge graph: the roadmap stages 0-5 and the six topic indexes both dereference into the tutorial library under docs/llm, whose flagship articles are shown alongside the interview bank and resume corner.*

### Understanding the Architecture

**The entry layer.** Everything starts from two files. The root `README.md` is the front door: it states the audience (developers with existing programming fundamentals), shows the six-stage mermaid flowchart, and links the four ways to use the repo — first-time learning starts at the roadmap, project work enters through the topic index, interview prep follows the interview route, and article lookup uses the full catalog. That full catalog is `docs/README.md` ("Master Index of All Materials," 全部资料索引), a flat, sectioned list of all 153 documents grouped into Foundations, Application Development, Transformer, AI Coding and Claude, Industry News, and Interview Topics.

**The roadmap layer.** `docs/roadmap/README.md` ("Complete Learning Route," 完整学习路线) is the spine: a table of six stages with goals, entry files, and completion markers, followed by a recommended sequence of four hands-on projects. Each stage then gets its own file. `beginner.md` ("Developer Beginner Path," 开发者入门路线) orders seven reads from `llm_learning_roadmap.md` through role comparisons, `llm_keywords.md`, `how_llm_trained.md`, and `token_cost_latency.md`, ending with the promise that you can draw the full chain of a model request. `application.md` (应用开发路线) covers the model-invocation fundamentals — prompt engineering, few-shot/CoT/reflection, structured output, streaming, function calling, context engineering — and `agent.md` (Agent 专项路线) lays out twelve ordered steps from `agent_intro.md` to `multi_agent_context_governance.md`, each file living under `docs/llm/app/`.

**The topic layer.** `docs/topics/README.md` (专题索引) catalogs six verticals: RAG, Agent, Fine-tuning, Deployment and Performance, Transformer, and AI Coding. Each topic file is itself a mini-roadmap. `rag.md` sequences ten articles from `why_rag.md` through chunking strategies, embeddings, vector databases, troubleshooting, optimization, long-document retrieval, evaluation, and Agentic RAG, then appends an "interview deepening" section linking back into the interview bank. `transformer.md` splits into a theory route (why Transformers beat RNN/CNN, data flow, Q/K/V, multi-head attention, positional encoding, block anatomy) and a code route of six hand-written implementations culminating in `tiny_transformer_code.md`. A note at the bottom of the topic index keeps time-sensitive model-release and industry-event coverage out of the core path by routing it to the news section of the master index.

**The tutorial library.** `docs/llm/` is where the actual content lives, organized by depth rather than by date. `docs/llm/intro/` holds the foundations — how models are trained, the thirteen core LLM concepts, API pricing arithmetic, distillation. `docs/llm/app/` is the largest shelf, with nearly forty application-engineering pieces covering prompts, RAG internals, agent design, multi-agent governance, MCP (`mcp_protocol.md`), and deployment. `docs/llm/transformer/` is the fifteen-part from-scratch series, `docs/llm/claude/` is a practical AI-coding track (CLAUDE.md, prompt caching, skills, hooks, loop engineering), and `docs/llm/news/` archives industry events separately so the curriculum itself stays stable.

**The interview and career layer.** `docs/interview/llm/README.md` aggregates the question banks — Transformer, RAG, Agent, fine-tuning, Vibe Coding, Claude Code deep dives, GraphRAG versus LightRAG, harness engineering, multi-agent communication — and, importantly, teaches the examiners' logic rather than canned answers. Two real ByteDance interview write-ups show how questions chain in practice. There are also legacy side-doors worth knowing about: `docs/interview/cpp/` and `docs/interview/java/` keep older language-specific material, and `docs/jianli/README.md` (the resume corner) deliberately delegates resume-building to the author's main site while pointing back to the interview route.

**The end-to-end flow.** Put together, a learner's path through the repository is a straight line with escape hatches: you land on `README.md`, enter `docs/roadmap/README.md`, and take whichever stage matches your level; each stage file hands you an ordered reading list into `docs/llm/`; each topic index lets you switch from sequential mode to problem-driven mode mid-project; and when you are ready to job-hunt, the stage-5 interview file pulls you into `docs/interview/llm/`, whose question banks explicitly reference the same articles you read during stages 2 and 3 — the same knowledge graph, viewed from an examiner's chair.

## Advantages

- **Dependency-ordered, not alphabetical.** The six-stage roadmap in `docs/roadmap/README.md` sequences topics by what depends on what, so you never read about agent evaluation before you understand function calling.
- **Completion markers instead of vague "mastery."** Every stage and topic defines a concrete exit criterion — an app with cost accounting, a RAG system with an offline eval set, a capacity report with P99 numbers — which turns passive reading into a checklist.
- **Engineering-first framing.** Articles are written from the application developer's seat: selection frameworks like `finetuning_vs_rag.md` (fine-tune or retrieve?) and failure-mode taxonomies matter more to practitioners than training math.
- **Theory with hands-on code.** The Transformer series in `docs/llm/transformer/` walks from attention math to writing each component yourself, ending with a tiny complete Transformer — the "unbox the black box" promise is backed by actual exercises.
- **Interview loop built in.** The question banks in `docs/interview/llm/` are cross-linked to the curriculum and include genuine multi-round interview experiences, so preparation and learning reinforce each other.
- **Living but stable.** Fast-moving model news is quarantined in `docs/llm/news/` and the master index, keeping the core learning path from rotting every time a model version ships.

## Benefits

- **A realistic on-ramp for career switchers.** If you already code in Java, C++, Go, Python, or frontend stacks, the beginner path assumes exactly that and nothing more, and the README is honest that pure algorithm research is out of scope.
- **Four resume-grade projects.** The README specifies four projects — AI business assistant, enterprise knowledge base, tool-using agent, production AI service — each with minimum delivery standards like eval sets, failure recovery, traces, and budget caps.
- **Faster problem-driven lookup.** Once you are building, the topic indexes let you jump straight to the article about, say, chunking strategies or PagedAttention without re-traversing the whole course.
- **Vocabulary fluency in weeks.** `docs/llm/intro/llm_keywords.md` and the beginner sequence compress the field's jargon — context, tokens, RAG, agents, fine-tuning — into coherent explanations rather than slogan definitions.
- **Cost and latency literacy.** Token billing, TTFT/TPOT, and pricing arithmetic get dedicated articles (`token_cost_latency.md`, `llm_pricing.md`), which is rare in beginner material but decisive in real projects.
- **Current-events coverage for interview small talk.** The news section tracks model releases and outages in depth, useful context when interviewers ask what you think of the latest model landscape.

## Usage

llm-master is a read-and-build curriculum, not an installable tool — there is nothing to `pip install`. To work through it locally, clone the repository and start from the roadmap:

```bash
git clone https://github.com/youngyangyang04/llm-master.git
cd llm-master
```

Then pick your entry point exactly as the README's "how to use this repo" section prescribes:

```text
docs/roadmap/README.md     # first-time learners: the six-stage route, start at your stage
docs/topics/README.md      # builders: jump to RAG / Agent / Fine-tuning / Deployment / Transformer / AI Coding
docs/roadmap/interview.md  # job seekers: organize what you know into interview answers
docs/README.md             # the full index of all 153 tutorials and interview write-ups
```

You can also read everything in the browser directly on GitHub, starting from the [roadmap](https://github.com/youngyangyang04/llm-master/blob/main/docs/roadmap/README.md). The material itself is in Chinese; section titles translate cleanly (the roadmap table, the topic index, the interview hub), and all file and article names in this post are the real paths in the repository.

## Conclusion

llm-master is that rarer thing on GitHub: a content repository that behaves like a well-designed course. Its source tree encodes an opinion about how programmers should learn LLM engineering — stage by stage, marker by marker, with evaluation and production concerns promoted from afterthoughts to graduation requirements — and its cross-linked indexes mean the same 153 documents serve first-time learners, working engineers, and interview candidates without duplicating content. If your 2026 goal is moving from "called the API once" to shipping evaluable RAG and agent systems you can defend in an interview, cloning this repo and following its roadmap is one of the highest-yield afternoons you can spend.

Links:

- GitHub repository: [youngyangyang04/llm-master](https://github.com/youngyangyang04/llm-master)
- Learning roadmap: [docs/roadmap/README.md](https://github.com/youngyangyang04/llm-master/blob/main/docs/roadmap/README.md)
- Full tutorial index: [docs/README.md](https://github.com/youngyangyang04/llm-master/blob/main/docs/README.md)
- Interview bank: [docs/interview/llm/README.md](https://github.com/youngyangyang04/llm-master/blob/main/docs/interview/llm/README.md)
- Originating course column: [Kamacoder LLM column](https://notes.kamacoder.com/llm/)
