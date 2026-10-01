---
layout: post
title: "AI Engineering Lab: 24 Weeks From Python To Production AI - Inside zorost/AI-Engineering-Lab"
description: "AI Engineering Lab is a free, MIT-licensed 24-week training program that takes a beginner from Python to production AI systems - 43 runnable notebooks, one continuous freight case study, and CI-grade quality checks on the curriculum itself. We tour the repository."
date: 2026-10-01
header-img: "img/post-bg.jpg"
permalink: /AI-Engineering-Lab-Twenty-Four-Weeks-To-Production-AI/
featured-img: "ai-coding-frameworks/ai-coding-frameworks"
image: https://pyshine.com/assets/img/diagrams/ai-engineering-lab/zorost-ai-engineering-lab-architecture.svg
tags:
  - AI Engineering
  - Machine Learning
  - Education
  - Open Source
categories: [AI, Open Source]
keywords: "AI engineering course, 24 week AI program, RAG tutorial, LoRA fine-tuning, MCP agents, Databricks lakehouse, Jupyter notebooks, AI curriculum, vector search, open source education"
author: "PyShine"
---

The AI education market has a strange shape: an enormous number of two-hour prompt workshops on one end, research papers that assume you already live in the field on the other, and very little that actually walks a working developer from Python fundamentals to the systems companies are deploying. The gap is not content — it is sequence, discipline, and honest practice. Most free curricula die as lists of links you bookmark and never open.

AI Engineering Lab, from Zorost Intelligence AI Lab, is built against exactly that failure. It is a free, open, self-paced 24-week program that takes a motivated beginner from Python to production-grade AI systems: machine learning and deep learning, LLM internals, prompt and context engineering, retrieval augmented generation with vector search, quantization, fine-tuning with LoRA and DPO, evaluation harnesses, agent harnesses, MCP, the major cloud AI platforms, and a governed Databricks lakehouse. Forty-three runnable notebooks carry the hands-on load, no GPU and no paid API key are required for the start, and everything is MIT licensed.

What makes the source worth a tour is the engineering around the curriculum. This repository treats its course the way a good team treats software: a manifest describes every week, the program website is generated from that manifest so the two cannot drift, a Python script validates that every notebook parses, ends with a printed metric, and carries no committed outputs, and other checks hunt broken links and stale diagrams. It is a curriculum with CI, which is rarer than it should be.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/ai-engineering-lab/zorost-ai-engineering-lab-overview-architecture.svg" alt="Architecture overview of the zorost/AI-Engineering-Lab repository" style="max-width:100%;height:auto;" />
</div>

*Architecture overview of the zorost/AI-Engineering-Lab repository: an entry path feeds a 24-week curriculum of notebooks and scored exercises, seeded by a synthetic freight dataset, with build scripts generating the program site.*

Reading the overview from left to right: you enter through a start-here document that sends you to week one and starts a tracking habit. The 24 week folders pair runnable notebooks with exercises and quizzes, all seeded by the synthetic data package that week one generates. Scores flow into the tracker, the curriculum manifest feeds the generated program site, and a set of QA scripts validates notebooks and keeps the published page honest.

## Why You Need This

The first reason is sequence. The program is organized into seven phases — foundations, LLM core, model engineering, harnesses and loops, agents, cloud AI platforms, and a Databricks capstone — each phase building on artifacts from the last. You are never guessing what to read next: every week links only to what it needs, and the repository states plainly that you should never have to wonder. That is a solved problem in software onboarding that most AI courses have ignored.

The second reason is the case study. Rather than 24 disconnected demos, the whole program runs through ZoroLogistics, a fictional freight operator. Week one's seeded dataset becomes week two's SQL practice, week three's training data, week seven's retrieval corpus, week ten's fine-tuning set, week sixteen's agent tools, and week twenty-three's feature tables. By the end you hold a portfolio of interconnected artifacts — the difference between having done exercises and having built a system.

The third reason is the metric discipline. From week three on, nothing counts as finished until it carries a metric and an error-analysis note, and the repository's own notebook checker enforces the habit mechanically: a notebook is invalid if it does not end with a code cell that prints a number. For a field full of impressive demos with no error bars, learning to end every artifact with a score is arguably the most transferable skill in the program.

The fourth reason is accessibility. The start requires no prior Python, no GPU, and no paid API key; the synthetic data generator in the `zoro` package produces realistic freight data locally with deterministic seeds, deliberately including the flaws that data-cleaning weeks need something to find. Free cloud notebooks are an alternative for anyone who cannot install software, and the whole curriculum is browsable on the program website before you clone anything.

## How It Works

The repository is a curriculum engine: authored content at the leaves, a manifest as the spine, and generated surfaces verified against that spine.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/ai-engineering-lab/zorost-ai-engineering-lab-architecture.svg" alt="Detailed architecture of the zorost/AI-Engineering-Lab repository" style="max-width:100%;height:auto;" />
</div>

*Detailed architecture of the zorost/AI-Engineering-Lab repository, from entry points through the curriculum tree, the zoro data package, QA scripts, and the generated program site.*

### Understanding the Architecture

**The manifest is the spine.** `curriculum/manifest.json` describes the program: 24 weeks, seven phases, per-week objectives, topics, and the ZoroLogistics case study that ties them together. The file itself is generated by `scripts/build_tracker.py`, so the tracker and the manifest stay in step. `scripts/build_site.py` reads the manifest to emit the GitHub Pages site at `docs/index.html`, with a `--check` mode that fails when the published page has drifted from the curriculum — the page cannot say anything the weeks do not.

**A week is a repeatable unit.** Every `curriculum/week-NN/` folder carries the same four beats: a `README.md` for study days, `notebooks/` for build days, `exercises.md` for the ship artifact, and `quiz.md` for reflection — a ten question quiz where eight is passing, recorded in the tracker under `curriculum/tracking/`. The rhythm, about ten hours a week, is stated up front so learners can plan honestly.

**The data package makes it self-contained.** `zoro/data.py` holds seeded generators for carriers, shipments, and the rest of the freight world, built on numpy and pandas with no network and no API keys. Because the seeds are deterministic, every learner works the same data — which makes error analysis comparable and makes the program's later weeks stable: the retrieval corpus from week seven exists before you get there.

**QA scripts enforce the discipline.** `scripts/check_notebooks.py` validates every notebook in the repository: it must parse as nbformat 4, carry the required metadata block, open with a markdown title plus a requirements line, and end with a code cell that prints a number — the metric rule — with no non-empty outputs committed. `scripts/check_links.py` hunts broken links across the content, `scripts/check_mermaid.py` validates the diagrams in the visual deep-dive page, and `scripts/release_check.py` runs the gate so a release cannot ship a stale or broken curriculum.

**Reference material sits beside the path.** The `reference/` directory holds a glossary, a knowledge base, platform notes, and agent resources, so a week can point to depth without embedding it. `reference/GLOSSARY.md` keeps terminology consistent across 24 weeks of material, and `curriculum/learning-path.md` gives the visual deep dive of how phases connect, with its diagrams checked like everything else.

The end-to-end flow: clone and install requirements, start at week one, generate the dataset every later week reuses, then walk the weekly loop of study, build, ship, and reflect — each week ending with a score and an honest error note, the tracker marking the Monday rows as you go.

## Advantages

- **A real sequence, not a link list.** Seven phases and 24 weeks where every artifact builds on the last one.
- **One case study, six revisits.** ZoroLogistics turns scattered exercises into a portfolio of interconnected systems.
- **Metric discipline enforced by tooling.** The notebook checker rejects artifacts that do not end in a printed score.
- **Runs offline and for free.** Seeded synthetic data, no API keys, no GPU for the first eight weeks.
- **Deterministic for everyone.** Fixed seeds mean every learner trains and evaluates on identical data.
- **The site cannot drift.** The published program page is generated from the curriculum manifest and verified with a check mode.

## Benefits

- **Portfolio, not certificates.** You finish with a retrieval system, a fine-tuned model, agents, and a lakehouse — all built on the same dataset.
- **Production honesty from day one.** Error-analysis notes and metric thresholds are habits the program refuses to let you skip.
- **Modern coverage.** MCP, agent harnesses, LoRA and DPO, quantization, and the three major cloud AI platforms sit in one continuous path.
- **Low-risk entry.** No paid tools and no prior Python mean the cost of starting is genuinely just time.
- **Maintainable as open source.** QA scripts and generated surfaces mean contributors can extend 24 weeks of material without breaking it.
- **Instructor-ready.** The tracker, quizzes, and phase structure can be adopted directly by teams running cohort learning.

## Usage

Clone and set up:

```bash
git clone https://github.com/zorost/AI-Engineering-Lab.git
cd AI-Engineering-Lab
python -m pip install -r requirements.txt
```

Read `START-HERE.md` first if you are new to programming or to AI. Then open week one and generate the dataset the whole program reuses:

```python
from zoro import data
df = data.shipments(n=100_000, seed=42)
```

Follow the weekly loop — study the README, run and then deliberately break the notebooks, ship the exercise with a metric, and pass the quiz at eight of ten before ticking the tracker row. Browse all 24 weeks at the program site before you start if you want to see the map first.

## Conclusion

AI Engineering Lab is what happens when a team that ships AI systems for a living writes down how it would train a new hire — and then engineers the curriculum itself with the same discipline it teaches. The notebooks, the freight case study, and the metric habit make it a genuinely runnable roadmap rather than a reading list; the manifest-driven site and notebook QA make it a model for how open courseware should be maintained. If 2026 is the year you turn AI curiosity into an engineering practice, this repository is a serious place to spend twenty-four weeks.

Links:

- Repository: https://github.com/zorost/AI-Engineering-Lab
- Program site: https://zorost.github.io/AI-Engineering-Lab/
- Curriculum index: https://github.com/zorost/AI-Engineering-Lab/blob/main/curriculum/README.md
- Start here guide: https://github.com/zorost/AI-Engineering-Lab/blob/main/START-HERE.md
