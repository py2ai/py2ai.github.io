---
layout: post
title: "AI Engineering Course: An 18-Module Learning Path - Inside amitshekhariitbhu/ai-engineering-course"
description: "A source tour of amitshekhariitbhu/ai-engineering-course, a free Apache-2.0 AI Engineering curriculum that packs 18 modules and 146+ lessons into a single README. We map how the course is structured: the linear learning path from machine learning foundations to transformers, RAG, AI agents, LLM inference, evaluation, and system design, plus the lesson format that links every concept to a detailed blog."
date: 2026-09-29
header-img: "img/post-bg.jpg"
permalink: /AI-Engineering-Course-Learning-Path-Source-Tour/
featured-img: "ai-coding-frameworks/ai-coding-frameworks"
image: https://pyshine.com/assets/img/diagrams/ai-engineering-course/amitshekhariitbhu-ai-engineering-course-architecture.svg
tags:
  - AI Engineering
  - LLM
  - RAG
  - Open Source
categories: [AI, Open Source]
keywords: "AI engineering course, learn AI engineering, free AI curriculum, LLM engineering course, RAG course, AI agents course, fine-tuning LLM, prompt engineering, context engineering, LLM inference optimization, transformer architecture, AI system design, RLHF, vector search, machine learning foundations"
author: "PyShine"
---

Most "courses" on GitHub are folder graves: forty directories, a half-finished notebook, and a README that apologizes for the mess. [amitshekhariitbhu/ai-engineering-course](https://github.com/amitshekhariitbhu/ai-engineering-course) commits to the opposite extreme. The whole repository is four entries: one `README.md`, one `LICENSE`, one `.gitattributes`, and one banner image. The course does not live *next to* the README. The course **is** the README.

And what a README it is. Amit Shekhar, founder of Outcome School, has written a free, complete AI Engineering curriculum into that single file: a Module 0 primer plus 18 full modules and 146+ lessons, each lesson a detailed blog post (many with a video) linked in a precise reading order — from machine learning foundations all the way to LLM inference, evaluation, AI safety, and AI system design.

That single-file design is why the repo is worth a source tour. When there is no code to read, the *structure* is the engineering: how the modules are ordered, how each lesson unit is shaped, and how the navigation apparatus — table of contents, curriculum table, learning-path flowchart, glossary, FAQs — keeps a beginner from getting lost. This post walks the repository as we would walk a codebase: entry point to exit point.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/ai-engineering-course/amitshekhariitbhu-ai-engineering-course-overview-architecture.svg" alt="Architecture overview of the amitshekhariitbhu/ai-engineering-course repository" style="max-width:100%;height:auto;" />
</div>

*Overview of the ai-engineering-course repository: the README is both container and curriculum, opening with a navigation frame and then running one linear path through onramp, LLM core, applied AI, production, and capstone modules.*

Reading the overview from left to right — rendered top to bottom as Mermaid lays it out — the story is one unbroken chain. The `README.md` opens with a table of contents and a "Curriculum at a Glance" table; the path then enters at Module 0 ("Must Know"), climbs through foundations and the LLM core, crosses the applied band where RAG and agents live, descends into the production band of inference, evaluation, and safety, and exits through the wide-and-capstone modules. There are no branches or electives: every arrow is labeled "then."

## Why You Need This

The honest problem with learning AI engineering in 2026 is not scarcity of material — it is the absence of an order. There are great blogs on KV cache, videos on LoRA, threads on MCP, but nothing that tells you what to read first, what it assumes, and what it builds toward. Most self-taught engineers end up with islands of knowledge: they can recite what a reranker does but cannot explain why attention needs causal masking.

This course attacks that with two things fragmented content cannot give you: a strict sequence and a dependency discipline. The README states the rule plainly — follow the modules in order, and inside each module the lessons in order, because every lesson builds on the previous one. It even flags the two modules people are tempted to skip: Module 1 (Machine Learning Foundations) and Module 2 (Deep Learning and Neural Networks) are non-negotiable, because everything else rests on them.

The second problem is the paywall around so much AI education. The FAQ answers it directly: every lesson is a free blog, no sign-up, no paywall, and the repository is Apache-2.0 licensed, so the curriculum index can be forked or turned into a team onboarding plan without asking permission. (The author runs a separate paid live program at Outcome School and says so openly in the README — the course content itself is not gated behind it.)

The third problem is role confusion. AI job titles — AI Engineer, Gen AI Engineer, LLM Engineer, LLMOps Engineer — multiply faster than job descriptions. The README opens with a list of twelve such roles the course serves, and its FAQ draws the key distinction — a machine learning engineer trains and deploys models, while an AI engineer builds products on top of them — then maps each skill (prompting, RAG, agents, fine-tuning, inference optimization, evaluation, safety) to a specific module. That mapping alone is worth an afternoon if you are planning a career move.

## How It Works

Because the repository is a curriculum rather than a program, its architecture is the architecture of a learning path: a frame, a spine, an applied band, a production band, a lesson unit, and a reference ring — all inside `README.md`.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/ai-engineering-course/amitshekhariitbhu-ai-engineering-course-architecture.svg" alt="Detailed curriculum architecture of the amitshekhariitbhu/ai-engineering-course repository" style="max-width:100%;height:auto;" />
</div>

*Detailed architecture of the repository: the course frame (table of contents, curriculum table, learning-path flowchart, prerequisites) gates a strictly linear chain of Modules 0 through 18, with lesson sections, glossary, FAQs, the Apache-2.0 LICENSE, and a sister-repo link in the reference ring.*

### Understanding the Architecture

**The frame.** Everything starts at the top of `README.md`: a table of contents anchoring every section from "About This AI Engineering Course" to the License, then the audience, prerequisites, and "How to Use" sections. The navigational centerpiece is the "Curriculum at a Glance" table listing every module with its topic and lesson count (9 lessons in Module 1, 15 in Module 3, 13 in Module 9, 16 in Module 10, 17 in Module 12), followed by an "AI Engineering Learning Path" Mermaid flowchart rendering the course as one vertical chain from "Machine Learning Foundations" down to "AI Engineering Interviews."

**The spine.** The first half builds the model-centric foundation in strict order. Module 0 defines the six words of every AI conversation — LLM, RAG, MCP, Agent, Fine-tuning, Quantization — with one video covering all six. Modules 1 and 2 rebuild machine learning and deep learning from first principles: supervised versus unsupervised learning, precision versus recall, gradient descent, backpropagation step by step, normalization, RNNs, and how PyTorch and TensorFlow work. Module 3, the largest of this band at 15 lessons, walks the Transformer from BPE tokenization and embeddings through self-attention, the math behind Q, K, and V, causal masking, multi-head attention, RoPE, and the feed-forward network. Modules 4 through 6 cover how LLMs generate text (temperature, top-k and top-p sampling, token streaming, lost in the middle), modern architecture (Mixture of Experts, GQA, sliding window attention, Flash Attention), and the taxonomy of SLMs, Large Reasoning Models, and Diffusion Language Models. Module 7 closes the spine with 11 lessons on fine-tuning and alignment, from LoRA and knowledge distillation to the RLHF lineage through InstructGPT, PPO, DPO, and GRPO.

**The applied band.** Modules 8 through 11 turn "understanding the model" into "building on the model." Module 8 covers prompt and context engineering: chain-of-thought, prompt chaining, prompt caching, and context compaction. Module 9 delivers 13 lessons on vector search and RAG, from vector databases and ANN search up through hybrid search, rerankers, ColBERT, chunking strategies, HyDE, semantic caching, Agentic RAG, GraphRAG, and Vectorless RAG. Module 10, the longest in the course at 16 lessons, covers AI agents: function calling, the agent loop, ReAct, Plan-and-Execute, Reflection, agent memory, MCP, Agent Skills, multi-agent systems, SubAgents, orchestration, and computer-use agents. Module 11 zooms out to agentic engineering — harness, loop, and graph engineering — and dissects how LangChain, LangGraph, Claude Code, and Cursor work under the hood.

**The production band.** Modules 12 through 14 are the operational third of the course. Module 12 packs 17 lessons on LLM inference: prefill versus decode, KV cache and its compression, paged attention, continuous batching, speculative decoding with the Medusa and EAGLE variants, quantization, GGUF, and the serving engines llama.cpp, vLLM, SGLang, and TensorRT-LLM. Module 13 covers evaluation and observability — LLM evaluation, LLM as a judge, agent evaluation, and traces, spans, and metrics. Module 14 closes it with guardrails, prompt injection and its defenses, and LLM watermarking.

**The lesson unit.** Every module follows the same template, which is what makes the 148 lessons of the glance table feel like one document. A module opens with a two-sentence promise ("In this module, we will learn... By the end of this module, we will be able to..."), lists its lessons as ordered links, and gives each lesson a numbered section — for example, `### 9.1 How does a Vector Database work?` — with a short abstract, a "We will cover the following:" checklist, and a "Let's get started" link to the full blog on outcomeschool.com. Code appears inside the lessons, mostly in Python, and the prerequisites ask only for basic programming (preferably Python) and high-school math, with the needed linear algebra and calculus explained inline.

**The reference ring.** After Module 18 — a single lesson handing readers off to the separate [ai-engineering-interview-questions](https://github.com/amitshekhariitbhu/ai-engineering-interview-questions) repository — the README closes with a "Key Concepts Glossary" of one-paragraph definitions each linking back to its lesson, an FAQ covering pacing (three to four months at one or two lessons a day) and the AI-engineer-versus-ML-engineer question, and the Apache-2.0 `LICENSE`, with `assets/banner.png` rounding out the file tree.

Follow the flow end to end and the design clicks: a reader lands on the README, uses the glance table to see the whole territory, passes the prerequisites gate, learns six words in Module 0, and is carried up the spine and across the applied and production bands one lesson at a time, with the glossary as a rearview mirror and the interview repository as the exit door.

## Advantages

- **A single linear dependency chain.** The Mermaid learning-path flowchart and the module ordering make prerequisites explicit — you always know what you need before what you are reading.
- **One file, zero setup friction.** No environment to build, no repo to explore; the entire curriculum index is one `README.md` you can read on GitHub or keep locally.
- **Consistent lesson anatomy.** The "We will cover the following" checklist doubles as a self-test: the README tells you to skim it, and if you can explain every point, move on.
- **Honest prerequisites.** Basic Python and high-school math, with the harder math taught inside the lessons — the course genuinely starts from zero AI background.
- **Coverage that matches real job descriptions.** The module map lines up with what AI, Gen AI, LLM, and agentic AI engineer postings ask for, from MoE and GQA down to vLLM and semantic caching.
- **Permissive licensing.** Apache-2.0 means the curriculum structure can be reused in team training, study groups, or a forked cohort without legal friction.

## Benefits

- **A replacement for tab-hoarding.** Instead of 60 open browser tabs, one ordered index where every item's prerequisites were read the week before.
- **Interview readiness built in.** Module 18 plus the companion interview-questions repository turn the curriculum into preparation for AI Engineer, LLM Engineer, and MLOps Engineer roles.
- **A vocabulary fast track.** Module 0 and the glossary make AI engineering conversations legible within your first day.
- **Depth and breadth on the hot topics.** RAG and agents get 37 lessons between them (Modules 9 through 11), well past toy examples into Agentic RAG, GraphRAG, MCP, and multi-agent orchestration.
- **Production realism.** Inference engineering, evaluation, observability, and safety get their own modules rather than a closing paragraph, which is where most free curricula stop.
- **Maintained, not abandoned.** The README commits to growing the course as new blogs and videos are written, and the lesson format makes adding Module 19 as cheap as adding one section.

## Usage

There is nothing to install or run — the "binary" of this repository is prose. The whole file tree is:

```text
ai-engineering-course/
├── README.md      # the entire course: frame, Modules 0-18, glossary, FAQs
├── LICENSE        # Apache-2.0, copyright Outcome School
├── .gitattributes
└── assets/
    └── banner.png
```

The way you "run" it is the reading workflow the README itself prescribes:

```text
1. Follow the modules in order - each module builds on the previous one.
2. Inside each module, read the lessons in the given order.
3. Do not skip Module 1 and Module 2, even if you are in a hurry.
4. Already know a topic? Read its "We will cover the following" list;
   if you can explain every point, move to the next lesson.
5. After each module, explain the concepts to a friend in your own words.
```

Each lesson's numbered section in `README.md` links out to the full blog on outcomeschool.com — for instance, Module 9's first lesson opens `outcomeschool.com/blog/how-does-a-vector-database-work` — and many lessons include a video. The FAQ suggests one or two lessons per day, putting the full path at roughly three to four months.

## Conclusion

amitshekhariitbhu/ai-engineering-course is a reminder that a repository does not need source code to have architecture. With one README, one Apache-2.0 `LICENSE`, and one banner, it delivers a 19-stage curriculum and 146+ lessons whose real engineering is the ordering: foundations before transformers, transformers before generation, generation before fine-tuning, and only then RAG, agents, inference, evaluation, and system design. If you are a software engineer moving into AI work, a student building a first mental model, or a tech lead who needs the whole map in one afternoon, start at Module 0 and follow the arrows — the chain does the navigating for you.

Links:

- GitHub repository: [amitshekhariitbhu/ai-engineering-course](https://github.com/amitshekhariitbhu/ai-engineering-course)
- Lesson blogs: [Outcome School blog](https://outcomeschool.com/blog)
- Companion practice repo: [amitshekhariitbhu/ai-engineering-interview-questions](https://github.com/amitshekhariitbhu/ai-engineering-interview-questions)
