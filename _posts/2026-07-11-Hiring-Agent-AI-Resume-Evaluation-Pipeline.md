---
layout: post
title: "Hiring Agent: An Open-Source AI Resume Evaluation Pipeline from HackerRank"
description: "Hiring Agent from interviewstreet (HackerRank) turns a resume PDF into a fair, explainable score by combining PyMuPDF extraction, GitHub enrichment, and an LLM evaluator with built-in fairness constraints. Runs locally on Ollama or on Google Gemini."
date: 2026-07-11
header-img: "img/post-bg.jpg"
permalink: /Hiring-Agent-AI-Resume-Evaluation-Pipeline/
featured-img: ai-coding-frameworks/ai-coding-frameworks
tags:
  - Hiring Agent
  - AI Agents
  - Resume Screening
  - HackerRank
  - Open Source
  - LLM
author: "PyShine"
---

# Hiring Agent: An Open-Source AI Resume Evaluation Pipeline from HackerRank

Resume screening at scale is one of the most repeated — and most inconsistently judged — tasks in software hiring. **Hiring Agent** is an open-source project from interviewstreet (HackerRank) that takes a resume PDF and produces a fair, explainable evaluation with category scores, evidence, bonuses, and deductions. With 5,500 stars on GitHub and an MIT license, it is a clean, practical example of an LLM pipeline that does real work end to end.

The project describes itself as a "Resume-to-Score pipeline." Let us walk through how it works.

## The Resume-to-Score Pipeline

Hiring Agent is structured as five sequential stages, each handled by a dedicated module.

![Hiring Agent Pipeline](/assets/img/diagrams/hiring-agent/hiring-agent-pipeline.svg)

The five stages:

1. **`pymupdf_rag.py`** — converts each PDF page to Markdown-like text using PyMuPDF
2. **`pdf.py`** — calls the LLM per section using Jinja templates, parsing six sections: Basics, Work, Education, Skills, Projects, and Awards
3. **`github.py`** — fetches the candidate's GitHub profile and repositories, classifies projects, and has the LLM select the top 7 with a minimum-commit threshold to prevent padding
4. **`evaluator.py`** — runs a strict-scored evaluation with fairness constraints encoded directly in evaluation templates
5. **`score.py`** — orchestrates the end-to-end flow and writes a CSV row when development mode is on

Intermediate results are cached under `cache/`, keyed by file basename, so you do not re-pay the LLM cost when iterating on a single resume. The structured-data schema is defined in Pydantic (`models.py`) as a `JSONResume` type, which keeps data flowing cleanly between stages.

## Key Features

The four pillars of the design are PDF extraction, GitHub enrichment, fair evaluation, and a local-or-cloud provider choice.

![Hiring Agent Features](/assets/img/diagrams/hiring-agent/hiring-agent-features.svg)

- **PDF Extraction** via PyMuPDF produces page-level Markdown that the LLM can parse section by section, using declarative Jinja templates under `prompts/templates`
- **GitHub Enrichment** fetches the profile and repositories, then asks the LLM to select exactly 7 unique projects with a meaningful-contribution and commit threshold — this is the mechanism that stops candidates from inflating their signal with trivial repos
- **Fair Evaluation** attaches evidence and explanation to every score, plus explicit bonus and deduction rules, with fairness constraints encoded directly in the evaluation templates
- **Local or Cloud** inference: run fully locally with Ollama (default model `gemma3:4b`) or in the cloud with Google Gemini (`gemini-2.5-pro`). Prompts are provider-agnostic and work across both without modification.

The provider abstraction lives in `models.py` as `OllamaProvider` and `GeminiProvider`, and `transform.py` normalizes loose LLM JSON into the JSON Resume style. This separation is what lets the same prompt templates work on a laptop or on a cloud key without changes.

## How Scoring Works

The evaluator scores across four categories, plus bonuses and deductions.

![Hiring Agent Scoring](/assets/img/diagrams/hiring-agent/hiring-agent-scoring.svg)

The four scoring categories:

- **Open Source** — contribution to public projects, weighted by GitHub signals
- **Self Projects** — independent project work, filtered through the top-7 selection
- **Production** — evidence of shipping real software at scale
- **Technical Skills** — depth and breadth of the stated skill set

On top of the category scores, the evaluator applies **bonuses** for standout evidence and **deductions** for missing or weak signals. Critically, every score carries evidence and an explanation — the output is not a black-box number but a readable report you can audit. That auditability is the "explainable" in "fair, explainable evaluation," and it is what separates this from a classifier that returns a single confidence score.

## Installation and Workflow

Hiring Agent runs as a standard Python project with one decision up front: which LLM provider to use.

![Hiring Agent Workflow](/assets/img/diagrams/hiring-agent/hiring-agent-workflow.svg)

The five-step setup:

1. **Clone**: `gh repo clone interviewstreet/hiring-agent` (or `git clone`)
2. **Venv + Install**: `python -m venv .venv && source .venv/bin/activate` then `pip install -r requirements.txt`
3. **Pull model** (if local): `ollama pull gemma3:4b`
4. **Choose provider** — set `LLM_PROVIDER=ollama` for local or `LLM_PROVIDER=gemini` with a `GEMINI_API_KEY` for cloud, configured via `cp .env.example .env`
5. **Score**: `python score.py /path/to/resume.pdf`

A single `DEVELOPMENT_MODE` flag in `config.py` toggles caching and CSV export, which is useful when you are processing a batch of resumes and want the per-candidate rows for comparison.

## Why It Matters

Hiring Agent is interesting for two reasons beyond resume screening itself.

First, it is a genuinely well-structured LLM pipeline. Each stage has a single responsibility, the schema is enforced with Pydantic, intermediate results are cached, and the prompts are declarative templates separated from the Python that calls them. That is a good blueprint for any "PDF in, structured judgment out" system — contracts, loan applications, grant reviews, insurance claims.

Second, it bakes fairness and explainability into the evaluation rather than treating them as an afterthought. Every score has evidence, deductions are explicit, and the GitHub top-7 selection with a commit threshold is a concrete anti-gaming mechanism. Whether that is enough fairness for production hiring is a separate question — but the architectural choices are worth studying.

If you hire engineers, or if you build LLM pipelines that have to defend their outputs, Hiring Agent is worth a careful read.

**Links:**

- GitHub: [https://github.com/interviewstreet/hiring-agent](https://github.com/interviewstreet/hiring-agent)
- License: MIT (© HackerRank)
- Default local model: `gemma3:4b` via Ollama
- Cloud option: Google Gemini (`gemini-2.5-pro`)