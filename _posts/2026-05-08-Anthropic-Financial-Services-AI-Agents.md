---
layout: post
title: "Anthropic Financial Services: AI Agents for Investment Banking, Equity Research, and Fund Administration"
description: "Learn how Anthropic's Financial Services agents bring Claude AI to investment banking, equity research, private equity, and fund administration with 10 named agents, 50+ skills, and 11 MCP data connectors."
date: 2026-05-08
header-img: "img/post-bg.jpg"
permalink: /Anthropic-Financial-Services-AI-Agents/
featured-img: ai-coding-frameworks/ai-coding-frameworks
categories: [AI Agents, Finance, Developer Tools]
tags: [Anthropic, Claude, financial services, AI agents, investment banking, equity research, fund administration, MCP connectors, private equity, wealth management]
keywords: "Anthropic financial services AI agents, Claude for investment banking, how to use Claude for finance, AI agents for equity research, financial services AI automation, Claude Cowork plugin finance, Anthropic managed agents API, MCP data connectors finance, AI-powered financial modeling, Claude for fund administration"
author: "PyShine"
---

# Anthropic Financial Services: AI Agents for Finance Workflows

Anthropic's Financial Services repository delivers a comprehensive collection of AI agents, skills, and data connectors purpose-built for financial services workflows. With 10 named end-to-end workflow agents, 7 vertical skill bundles, and 11 MCP data provider integrations, this open-source project under the Apache 2.0 license brings Claude's capabilities directly into investment banking, equity research, private equity, wealth management, and fund administration. The dual deployment model means the same agent logic runs as an interactive Claude Cowork plugin or as a headless Managed Agent API, giving financial institutions flexibility in how they integrate AI into existing workflows.

## The Dual Deployment Model: One Source, Two Surfaces

A defining architectural choice in this repository is the "one source, two surfaces" deployment model. Every agent, skill, and connector is authored once as Markdown, YAML, and JSON files, and then deployed through either of two surfaces:

**Claude Cowork Plugin** -- For interactive use, agents install directly into the Claude Cowork interface. Analysts trigger skills with slash commands like `/comps`, `/dcf`, or `/earnings`, and agents appear in the Cowork dispatch for full workflow automation. This surface is ideal for analysts who work alongside Claude in real time.

**Claude Managed Agent API** -- For headless, production-grade deployment, the same agent logic wraps into a Managed Agent template deployed via the `/v1/agents` API endpoint. The `deploy-managed-agent.sh` script resolves file references, uploads skills, creates leaf-worker subagents, and posts the orchestrator configuration. This surface suits firms that want to embed Claude agents into existing workflow engines, cron jobs, or event-driven pipelines.

> **Key Insight:** The dual deployment model means financial institutions can prototype interactively in Cowork, then promote the exact same agent logic to a production Managed Agent without rewriting a single prompt or skill definition.

This architecture eliminates the common problem of maintaining two separate codebases for interactive and automated workflows. The system prompt, skill definitions, and connector configurations are identical across both surfaces, ensuring behavioral consistency whether an analyst is working hands-on with Claude or a scheduled process is running autonomously overnight.

## Architecture Overview

![Financial Services Architecture](/assets/img/diagrams/financial-services/financial-services-architecture.svg)

### Understanding the Architecture

The architecture diagram above illustrates how Anthropic's Financial Services platform bridges Claude's AI capabilities with the specialized data sources and workflows that financial professionals rely on daily. Let us break down each layer and its role in the system.

**Claude AI Core Layer**

At the center sits Claude, providing the language understanding, reasoning, and generation capabilities that power every agent. Claude processes natural language instructions, interprets financial data from MCP connectors, and produces structured outputs like models, memos, and research notes. The agents do not replace Claude; they extend it with domain-specific context, conventions, and step-by-step methods that make Claude's output more precise and reliable for financial work.

**Agent Plugin Layer**

The agent plugin layer wraps Claude with self-contained workflow definitions. Each of the 10 named agents -- from Pitch Agent to KYC Screener -- bundles its own system prompt, skill set, and connector references. This self-contained design means installing a single agent plugin gives you everything needed for that workflow. The Pitch Agent, for example, carries its own comps analysis, LBO modeling, and deck QC skills, so it can produce a branded pitch deck from end to end without requiring additional vertical plugins.

**Vertical Skill Bundles**

Below the agents sit the 7 vertical skill bundles: financial-analysis, investment-banking, equity-research, private-equity, wealth-management, fund-admin, and operations. These bundles contain the domain expertise -- conventions, templates, and step-by-step methods -- that Claude draws on automatically when relevant. The financial-analysis bundle serves as the core, carrying shared modeling skills and all 11 data connectors. The remaining verticals layer on top, adding specialized skills for their domain.

**MCP Data Connectors**

The Model Context Protocol (MCP) connector layer wires Claude to external financial data providers. With 11 connectors -- including Daloopa, Morningstar, S&P Global, FactSet, Moody's, LSEG, and PitchBook -- agents can pull live market data, historical financials, and research directly into their workflows. Each connector is defined in a `.mcp.json` file, making it straightforward to swap providers or add internal data sources.

**Dual Deployment Surfaces**

The architecture supports two deployment surfaces that share the same underlying components. On the left, the Cowork Plugin surface provides interactive access through Claude's chat interface, where analysts use slash commands and natural language. On the right, the Managed Agent API surface enables headless deployment through the `/v1/agents` endpoint, where orchestrators route work to agents programmatically. Both surfaces reference the same agent definitions, skills, and connectors.

> **Important:** Everything in this repository is file-based -- Markdown, YAML, and JSON with no build step. This means financial teams can fork, edit, and customize agents using familiar tools without any compilation or bundling process.

## The 10 Named Agents and 7 Verticals

![Financial Services Agents](/assets/img/diagrams/financial-services/financial-services-agents.svg)

### Understanding the Agent Catalog

The agents diagram above maps the full landscape of Anthropic's Financial Services offering, showing how 10 named agents and 7 vertical skill bundles relate to each other and to the financial workflows they serve. Let us examine each component in detail.

**Coverage and Advisory Agents**

The Pitch Agent handles the complete pitch deck workflow: comparable company analysis, precedent transactions, LBO modeling, and branded deck generation. It bundles comps analysis, LBO modeling, and deck QC skills into a single self-contained plugin. The Meeting Prep Agent produces briefing packs before client meetings, synthesizing relevant data from connectors and prior interactions into a structured preparation document.

**Research and Modeling Agents**

The Market Researcher takes a sector or theme and produces an industry overview, competitive landscape analysis, peer comps, and an ideas shortlist. The Earnings Reviewer ingests earnings call transcripts and SEC filings, updates financial models, and drafts research notes. The Model Builder creates DCF, LBO, 3-statement, and comps models directly in Excel, leveraging the xlsx-author skill for headless spreadsheet generation.

**Fund Administration and Finance Operations Agents**

The Valuation Reviewer ingests GP packages, runs valuation templates, and stages LP reporting. The GL Reconciler finds breaks in general ledger data, traces root causes, and routes discrepancies for sign-off. The Month-End Closer handles accruals, roll-forwards, and variance commentary. The Statement Auditor audits LP statements before distribution, catching errors and inconsistencies that could affect investor reporting.

**Operations and Onboarding Agent**

The KYC Screener parses onboarding documents, runs a rules engine against KYC requirements, and flags gaps for human review. This agent demonstrates how the platform extends beyond analytical workflows into operational processes that still require domain expertise and regulatory compliance.

**Vertical Skill Bundles**

The 7 vertical skill bundles provide the domain expertise layer. The financial-analysis bundle is the core, containing shared modeling skills (comps, DCF, LBO, 3-statement) and all 11 MCP data connectors. Investment-banking adds CIMs, teasers, process letters, buyer lists, and merger models. Equity-research adds earnings notes, initiations, model updates, and thesis tracking. Private-equity adds sourcing, screening, diligence checklists, and IC memos. Wealth-management adds client reviews, financial plans, rebalancing, and tax-loss harvesting. Fund-admin adds GL recon, break tracing, accruals, and NAV tie-out. Operations adds KYC document parsing and rules-grid evaluation.

**Partner-Built Plugins**

Two partner-built plugins extend the platform with specialized data integrations. The LSEG plugin provides bond relative value, swap curve analysis, FX carry trade, options volatility, and macro-rates monitoring on London Stock Exchange Group data. The S&P Global plugin offers tear sheets, earnings previews, and funding digests powered by S&P Capital IQ data.

> **Amazing:** With over 50 domain-specific skills and slash commands like `/comps`, `/dcf`, `/earnings`, and `/ic-memo`, the platform covers the full spectrum of financial services workflows -- from pitch deck creation to KYC screening -- all authored as plain Markdown and YAML files.

## Multi-Agent Orchestration Workflow

![Financial Services Workflow](/assets/img/diagrams/financial-services/financial-services-workflow.svg)

### Understanding the Orchestration Workflow

The workflow diagram above illustrates how Anthropic's Financial Services agents coordinate using a depth-1 subagent delegation pattern, where an orchestrator agent dispatches work to specialized leaf workers and only the final worker in the chain has write permissions. This design ensures both efficiency and security in financial workflows.

**Orchestrator Agent**

The orchestrator sits at the top of the hierarchy and receives the initial request -- whether from an analyst in Cowork or from an automated pipeline via the Managed Agent API. The orchestrator's role is to understand the request, decompose it into subtasks, and delegate each subtask to the appropriate leaf worker. It does not produce any deliverable itself; instead, it coordinates the workflow and assembles the final output from worker results.

**Leaf Workers: Researcher, Modeler, Writer**

The three primary leaf worker types represent the core competencies needed in financial workflows. The Researcher worker gathers data from MCP connectors, searches filings, and synthesizes background information. The Modeler worker builds and updates financial models -- DCF, LBO, comps, 3-statement -- producing Excel workbooks or structured data. The Writer worker takes the research and model outputs and produces the final deliverable: a research note, a pitch deck, an IC memo, or a reconciliation report.

**Depth-1 Delegation Pattern**

The depth-1 constraint means the orchestrator can delegate to leaf workers, but leaf workers cannot delegate further. This keeps the delegation chain shallow and predictable, which is critical for financial workflows where auditability and traceability matter. Every action can be traced back to a single orchestrator decision, and no worker can spawn uncontrolled sub-processes.

**Write Permission Restriction**

A crucial security feature: only the final worker in the chain -- typically the Writer -- has write permissions. The Researcher and Modeler workers can read data and produce intermediate results, but they cannot write to the firm's systems of record. This ensures that no intermediate artifact accidentally becomes the authoritative version. The Writer stages its output for human review, and only a qualified professional can approve and publish.

**Human-in-the-Loop Checkpoint**

Every agent workflow ends at a human-in-the-loop checkpoint. The agent drafts analyst work product -- models, memos, research notes, reconciliations -- for review by a qualified professional. Agents do not make investment recommendations, execute transactions, bind risk, post to a ledger, or approve onboarding. This design principle is embedded in the repository's license and documentation, reflecting the regulatory reality of financial services.

**Steering Events and Handoff Requests**

The Managed Agent deployment uses steering events and handoff requests to coordinate between agents. The `orchestrate.py` script provides a reference event loop that routes `handoff_request` events between agents via your own orchestration layer. This allows firms to build custom routing logic -- for example, escalating complex valuations to senior analysts or routing KYC flags to compliance officers.

## MCP Data Connectors: 11 Financial Data Providers

The platform integrates with 11 financial data providers through the Model Context Protocol (MCP), giving agents direct access to live market data, historical financials, and research:

| Provider | Data Type | MCP Endpoint |
|----------|-----------|-------------|
| Daloopa | Financial data extraction | `mcp.daloopa.com` |
| Morningstar | Investment research and ratings | `mcp.morningstar.com` |
| S&P Global | Market intelligence (Capital IQ) | `kfinance.kensho.com` |
| FactSet | Financial data and analytics | `mcp.factset.com` |
| Moody's | Credit ratings and risk data | `api.moodys.com` |
| MT Newswires | Real-time news and alerts | `vast-mcp.blueskyapi.com` |
| Aiera | AI-powered earnings analysis | `mcp-pub.aiera.com` |
| LSEG | London Stock Exchange data | `api.analytics.lseg.com` |
| PitchBook | Private market data | `premium.mcp.pitchbook.com` |
| Chronograph | Portfolio analytics | `ai.chronograph.pe` |
| Egnyte | Document management | `mcp-server.egnyte.com` |

All connectors are centralized in the financial-analysis core plugin and shared across the remaining verticals. MCP access may require a subscription or API key from the provider. The `.mcp.json` configuration files make it straightforward to swap providers or add internal data sources by pointing to alternative endpoints.

## Microsoft 365 Integration

For firms that run Claude inside Excel, PowerPoint, Word, and Outlook via the Microsoft 365 add-in, the repository includes admin tooling in `claude-for-msft-365-install/`. This is a Claude Code plugin (not a Cowork plugin) that walks an IT admin through:

- Generating a customized add-in manifest for the tenant
- Granting Azure admin consent via Microsoft Graph
- Writing per-user routing configuration
- Deploying against the firm's own cloud infrastructure (Vertex AI, Bedrock, or an internal LLM gateway)

```bash
# Install the Microsoft 365 admin tooling
claude plugin install claude-for-msft-365-install@claude-for-financial-services

# Run the setup command
/claude-for-msft-365-install:setup
```

This is separate from the agents and vertical plugins -- it is the on-ramp that gets the add-in deployed in a tenant, after which the agents and skills run inside it.

## Installation and Getting Started

### Option 1: Claude Cowork

In Cowork, open **Settings > Plugins > Add plugin** and either:

- Paste the repository URL: `https://github.com/anthropics/claude-for-financial-services` and pick the agents and verticals you want from the marketplace list, or
- Upload a zip of any directory under `plugins/` (for example, `plugins/agent-plugins/pitch-agent/`).

### Option 2: Claude Code CLI

```bash
# Add the marketplace
claude plugin marketplace add anthropics/claude-for-financial-services

# Core skills and connectors (install first)
claude plugin install financial-analysis@claude-for-financial-services

# Named agents -- pick the ones you need
claude plugin install pitch-agent@claude-for-financial-services
claude plugin install gl-reconciler@claude-for-financial-services
claude plugin install market-researcher@claude-for-financial-services

# Vertical skill bundles
claude plugin install investment-banking@claude-for-financial-services
claude plugin install equity-research@claude-for-financial-services
```

Once installed, agents appear in Cowork dispatch, skills fire automatically when relevant, and slash commands are available in your session.

### Option 3: Managed Agent API

```bash
# Set your API key
export ANTHROPIC_API_KEY=sk-ant-...

# Deploy a named agent
scripts/deploy-managed-agent.sh gl-reconciler
```

The deploy script resolves file references, uploads skills, creates leaf-worker subagents, and posts the orchestrator to `/v1/agents`. See `scripts/orchestrate.py` for a reference event loop that routes `handoff_request` events between agents.

## Key Features Summary

| Feature | Description |
|---------|-------------|
| 10 Named Agents | End-to-end workflow agents for pitch decks, earnings, recon, KYC, and more |
| 7 Vertical Bundles | Domain skill packs for IB, ER, PE, WM, fund admin, and operations |
| 50+ Skills | Domain-specific skills with slash commands (`/comps`, `/dcf`, `/earnings`) |
| 11 MCP Connectors | Live data from Daloopa, Morningstar, S&P, FactSet, Moody's, LSEG, and more |
| Dual Deployment | Same agent logic runs as Cowork plugin or Managed Agent API |
| Depth-1 Orchestration | Orchestrator delegates to leaf workers; only final worker has write access |
| File-Based Architecture | Everything is Markdown, YAML, and JSON with no build step |
| Microsoft 365 | Admin tooling for Excel, PowerPoint, Word, and Outlook add-in deployment |
| Human-in-the-Loop | All outputs staged for professional review; no autonomous execution |
| Apache 2.0 License | Fully open source with commercial-friendly licensing |

## Human-in-the-Loop Design Philosophy

A core design principle throughout the repository is that agents draft analyst work product for review by qualified professionals. The documentation explicitly states that nothing in the repository constitutes investment, legal, tax, or accounting advice. Agents do not make investment recommendations, execute transactions, bind risk, post to a ledger, or approve onboarding.

This philosophy manifests in several architectural decisions:

- **Write permissions** are restricted to the final worker in the orchestration chain, preventing intermediate artifacts from becoming authoritative
- **Steering events** allow firms to insert custom routing logic, such as escalating complex valuations to senior analysts
- **Staged outputs** mean every deliverable is queued for human sign-off before publication
- **Customizable prompts** let firms tune agent behavior to match their specific processes and compliance requirements

> **Takeaway:** The human-in-the-loop design is not an afterthought -- it is the foundational principle. Every agent, from Pitch Agent to KYC Screener, produces drafts for professional review, making this platform suitable for regulated financial services environments where accountability and auditability are non-negotiable.

## Making It Yours

The repository is designed for customization. Financial teams can:

- **Swap connectors** by pointing `.mcp.json` at their own data providers and internal systems
- **Add firm context** by dropping terminology, processes, and formatting standards into skill files
- **Bring templates** using `/ppt-template` to teach Claude branded PowerPoint layouts
- **Adjust agent scope** by editing `agents/<slug>.md` to match how the team actually runs workflows
- **Add new workflows** by copying the structure for workflows the repository does not yet cover

New skills go under `plugins/vertical-plugins/<vertical>/skills/`, and running `python3 scripts/sync-agent-skills.py` propagates them to any agent that bundles them. The `scripts/check.py` linter verifies every manifest, resolves cross-file references, and fails if any bundled skill has drifted from its vertical source.

## Conclusion

Anthropic's Financial Services repository represents a significant step toward production-grade AI in financial services. By offering 10 named agents, 7 vertical skill bundles, 50+ domain-specific skills, and 11 MCP data connectors -- all authored as plain Markdown, YAML, and JSON with no build step -- it lowers the barrier for financial institutions to adopt AI while maintaining the human oversight that regulatory compliance demands.

The dual deployment model (Cowork plugin and Managed Agent API) gives teams the flexibility to prototype interactively and then promote to production without rewriting. The depth-1 orchestration pattern with write-restricted workers provides the auditability and control that financial workflows require. And the file-based architecture means any firm can fork, customize, and contribute back under the Apache 2.0 license.

Whether you are an investment banker building pitch decks, an equity research analyst covering earnings, a fund administrator reconciling ledgers, or a compliance officer screening KYC documents, this repository provides a starting point that you can tune to how your firm works.

**Links:**

- GitHub Repository: [https://github.com/anthropics/financial-services](https://github.com/anthropics/financial-services)