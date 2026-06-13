---
layout: post
title: "Superlog: Open-Source AI Self-Healing Observability with OpenTelemetry"
description: "Discover how Superlog combines OpenTelemetry observability with AI agents that automatically detect, diagnose, and self-heal software issues. Open-source, self-hosted, and built with TypeScript and React."
date: 2026-06-13
header-img: "img/post-bg.jpg"
permalink: /Superlog-AI-Self-Healing-Observability-OpenTelemetry/
featured-img: ai-coding-frameworks/ai-coding-frameworks
categories: [AI Agents, DevOps, Open Source]
tags: [superlog, AI self-healing, observability, OpenTelemetry, AI agents, self-hosted monitoring, LLM operations, TypeScript, React dashboard, AIOps]
keywords: "superlog AI self-healing observability, OpenTelemetry AI agent monitoring, self-hosted AIOps platform, AI automated incident remediation, LLM observability tool, superlog open source setup, AI agent memory system, OpenTelemetry self-healing software, TypeScript observability dashboard, AIOps open source alternative"
author: "PyShine"
---

## Introduction

What if your monitoring system did not just alert you about problems -- but actually investigated and opened a pull request with a fix? Traditional observability tools like Datadog, New Relic, and Grafana detect problems and alert humans, but the human must diagnose the root cause and write the fix. Mean time to resolution remains high because the loop from detection to remediation requires manual intervention at every step.

Every minute of downtime costs money. SREs spend hours triaging alerts, correlating traces with logs, and writing patches. The toil of incident response is one of the biggest pain points in modern software operations. Superlog is an open-source observability tool that uses AI agents to self-heal your software, and it does so by ingesting OpenTelemetry data, grouping noisy signals into incidents, and launching AI-powered investigations that can open pull requests with fixes -- all while you sleep.

![Superlog Architecture](/assets/img/diagrams/superlog/superlog-architecture.svg)

The diagram above shows Superlog's end-to-end architecture: your instrumented application emits OpenTelemetry telemetry through the OTLP protocol, which flows through the OTel Collector and Superlog's OTLP Proxy into ClickHouse for storage. The Worker process, powered by an AI Agent with persistent memory and LLM reasoning, investigates incidents and can open fix pull requests that feed back into your application -- closing the self-healing loop. Meanwhile, the Vite/React Dashboard visualizes everything in real time.

> **Key Insight**: Superlog represents a fundamental shift in observability: from passive monitoring that alerts humans to active self-healing that investigates and proposes fixes autonomously. By combining OpenTelemetry's vendor-neutral telemetry with LLM-powered agents that have persistent memory, Superlog closes the loop between detection and remediation -- the agent reads your code, identifies the root cause, writes a patch, and opens a pull request.

## What is Superlog?

Superlog is an open-source, self-hosted agentic telemetry system built by [Superlog Labs](https://superlog.sh), a Y Combinator P26 company. It ingests traces, logs, and metrics via the OpenTelemetry protocol, groups noisy signals into incidents, and gives teams a local-first product surface for debugging production systems. With 727 GitHub stars and growing, Superlog is written in TypeScript and uses a modern stack including Vite/React for the frontend, Hono for the API, and ClickHouse for high-performance telemetry queries.

The self-healing concept in Superlog is specific and practical: when an incident is detected, the AI agent investigates the issue by reading your source code on GitHub, identifying the root cause, and opening a pull request with a fix. This is not a generic "restart the service" approach -- the agent actually writes code patches tailored to the specific error. The agent can also classify incidents as noise, merge duplicate incidents, or pause and ask a human for clarification when it is uncertain.

Superlog is licensed under the Apache License 2.0, making it genuinely open-source. The repository contains the fully open-source community edition, which includes the web app, API, OTLP ingest proxy, worker processes, Postgres schema, ClickHouse-backed telemetry queries, and agent runner interfaces. A hosted Superlog Cloud edition with a free tier is also available for teams that prefer not to manage infrastructure.

## How Superlog's AI Agents Work

Superlog's agent system is the core of its self-healing capability. Here is how the investigation pipeline works:

**Observability Data Ingestion**: Your application, instrumented with the OpenTelemetry SDK, emits traces, logs, and metrics via the OTLP protocol. The data flows through the OTel Collector into Superlog's OTLP Proxy, which authenticates requests, stamps issue fingerprints, and forwards the data to ClickHouse for storage.

**Issue Grouping and Incident Creation**: The Worker process continuously groups related telemetry signals into issues based on fingerprinting -- errors with the same exception type, service, and stacktrace fingerprint are grouped together. When enough signals accumulate, an incident is created.

**Agent Investigation**: When an incident is triggered, the agent run lifecycle begins. The agent transitions through states: queued, repo discovery, running, and eventually complete or failed. During repo discovery, the agent identifies which GitHub repositories are most likely related to the incident by scoring repos against the service name and stacktrace frames. During the running state, the agent uses an LLM (Anthropic's Claude by default, or a community runner) to analyze the code, identify the root cause, and write a fix.

**Memory and Context**: Before each investigation, the agent loads all active memories for the project. These memories -- accumulated from previous runs and human feedback -- provide context about team-specific terminology, infrastructure layout, codebase structure, and lessons learned. This means the agent gets smarter over time.

**Resolution**: The agent can complete an investigation in several ways: opening a pull request with a fix, merging the incident into an existing one, classifying it as noise, or pausing to ask a human for clarification. Each outcome is recorded in the audit trail.

![Superlog Self-Healing Flow](/assets/img/diagrams/superlog/superlog-self-healing-flow.svg)

The diagram above illustrates the full agent investigation pipeline. Incoming telemetry is ingested and grouped into issues, which trigger incidents. The auto-investigate decision point determines whether the agent should investigate or classify the incident as noise. During investigation, the agent recalls relevant memories, performs root cause analysis, and decides on a course of action -- either opening a fix pull request or asking a human for help. After the fix is reviewed and merged, the incident is resolved and the agent records what it learned in memory for future investigations.

## Core Features Deep Dive

### OpenTelemetry Integration

Superlog is built natively on the OpenTelemetry standard. The OTel Collector configuration in the repository shows full pipeline support for traces, logs, and metrics -- all three signals are received via OTLP, processed through attribute enrichment and batch processing, and exported to ClickHouse. The collector strips any client-supplied `superlog.*` attributes to prevent tenant spoofing, then injects the authenticated project ID from request metadata. This is production-grade security on the ingest path.

The ClickHouse schema stores traces in `otel_traces`, logs in `otel_logs`, and metrics across five tables: `otel_metrics_gauge`, `otel_metrics_sum`, `otel_metrics_histogram`, `otel_metrics_summary`, and `otel_metrics_exp_histogram`. This comprehensive coverage means you get full-fidelity telemetry storage, not just sampled traces.

### AI Agent System

The agent runner backend defines a pluggable interface for investigation runtimes. The current providers are:

- **community** -- the default runner that records a local incident summary
- **anthropic** -- uses Anthropic's Claude to perform deep code investigation and open fix PRs
- **disabled** -- turns off automated investigations

The agent run lifecycle is governed by a state machine with well-defined transitions: queued, repo_discovery, running, awaiting_human, complete, failed, blocked_no_github, and pr_retry_queued. Each transition emits an audit event, creating a complete timeline of every investigation. The system supports human-in-the-loop workflows: the agent can pause and ask a clarifying question, and a human can resume the investigation with additional context.

### Memory System

Superlog's agent memory system is one of its most distinctive features. Memories are durable, project-scoped facts that persist across investigation runs. The memory kinds are:

- **feedback** -- lessons from human corrections and feedback
- **terminology** -- team-specific naming conventions and jargon
- **infra** -- deployment and infrastructure facts
- **project** -- codebase and project structure facts

The agent has three tools for working with memory: `save_memory` to persist a new fact, `update_memory` to correct or archive an existing fact, and `list_memories` to review what is already known. Memories are injected into every future investigation's prompt for the project, so the agent accumulates institutional knowledge over time. A memory that turns out to be wrong can be archived with the `update_memory` tool, ensuring the agent does not repeat mistakes.

### React Dashboard

The Vite/React frontend provides a real-time observability workspace. It connects to the Hono-based API to display traces, logs, metrics, issues, and incidents. The dashboard shows the full incident timeline including agent run events, pull request links, and resolution proposals. Teams can confirm or dismiss resolution proposals directly from the dashboard.

### Self-Hosted Deployment

Superlog runs entirely on your infrastructure. The Docker Compose configuration includes Postgres 16 for metadata, ClickHouse 26.1 for telemetry storage, and the OTel Collector for ingest. No data leaves your network. This makes Superlog suitable for privacy-sensitive and compliance-regulated environments where sending telemetry to a third-party cloud service is not acceptable.

### Integrations

Beyond the core observability features, Superlog integrates with GitHub for repository access and pull request management, Linear for issue tracking, Slack for notifications, and supports an MCP (Model Context Protocol) server for connecting AI coding tools. The billing system uses Autumn for plan management with a free tier available.

![Superlog Agent Memory](/assets/img/diagrams/superlog/superlog-agent-memory.svg)

The diagram above shows how the AI agent interacts with its four memory kinds and the LLM reasoning engine. The agent reads and writes feedback, terminology, infra, and project memories -- each scoped to the current project. The LLM provides the reasoning capability that drives the investigation decision, which can result in opening a fix PR, merging duplicate incidents, classifying noise, or asking a human for help.

> **Amazing**: In traditional observability, the mean time to resolution is measured in minutes or hours because a human must triage, diagnose, and write a fix. Superlog's AI agents can reduce this dramatically by automatically correlating traces, metrics, and logs, recalling similar past incidents from memory, and opening a pull request with a targeted code fix -- all without a human in the loop for straightforward issues.

## The Self-Healing Pipeline

Here is a step-by-step walkthrough of a self-healing scenario in Superlog:

1. **Application emits OpenTelemetry telemetry**: Your Node.js application, instrumented with the OTel SDK, sends traces, logs, and metrics to the OTel Collector via OTLP.

2. **Superlog ingests and indexes the data**: The OTLP Proxy authenticates the request using an API key, stamps issue fingerprints on the payload, and forwards it to the OTel Collector, which exports it to ClickHouse.

3. **Worker groups signals into issues**: The fingerprinting system groups related error signals together based on exception type, service name, and stacktrace. Each unique fingerprint becomes an issue.

4. **Incident is created from grouped issues**: When enough signals accumulate for a fingerprint, the Worker creates an incident and associates the relevant issues with it.

5. **Agent run is enqueued**: If auto-investigate is enabled for the project, the Worker enqueues an agent run for the incident. The run transitions to the `queued` state.

6. **Agent discovers relevant repositories**: The agent scores all accessible GitHub repositories against the incident's service name and stacktrace frames, selecting the most likely repository containing the bug.

7. **Agent investigates with LLM and memory**: The agent loads project memories, receives the incident context (issue summaries, stacktraces, service name), and begins an LLM-powered investigation session. It reads the source code, identifies the root cause, and formulates a fix.

8. **Agent opens a pull request**: If a fix is found, the agent pushes a branch and opens a pull request on GitHub. The PR policy can be configured per project: `never`, `on_ready_to_pr`, or `always`.

9. **Human reviews and merges**: The team reviews the PR in their normal GitHub workflow. Auto-merge policies can be configured: `never`, `when_checks_pass`, or `immediately`.

10. **Agent records learnings in memory**: After the investigation, the agent saves durable facts it learned -- what the root cause was, how the infra is structured, any terminology it picked up. These memories are injected into every future investigation for the project.

## Installation and Setup

### Prerequisites

- Node.js 20 or later
- pnpm 9 or later
- Docker and Docker Compose

### Quick Start

Clone the repository and start the local stack:

```bash
git clone https://github.com/superloglabs/superlog.git
cd superlog
pnpm install
docker compose up -d
pnpm --filter @superlog/db db:migrate
pnpm dev
```

The default local services are:

- **Web Dashboard**: `http://localhost:5173`
- **API Server**: `http://localhost:4100`
- **OTLP Intake**: `http://localhost:4101`

### Instrumenting Your Application

To send telemetry to Superlog, instrument your application with the OpenTelemetry SDK. For a Node.js application:

```bash
npm install @opentelemetry/api @opentelemetry/sdk-node @opentelemetry/auto-instrumentations-node
```

Then configure the OTLP exporter to point to your Superlog instance:

```bash
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
export OTEL_SERVICE_NAME=my-service
node --require @opentelemetry/auto-instrumentations-node/register app.js
```

### Installing via Skills

Superlog also provides a skills-based installation method for coding agents:

```
Run npx skills add superloglabs/skills --all and use the skills to install Superlog in this project
```

This approach uses the [superloglabs/skills](https://github.com/superloglabs/skills) repository, which contains agent-readable instructions for setting up Superlog instrumentation in your project.

### Configuring Agent Automation

Agent automation settings are configured per project through the API or dashboard. The key settings include:

- **agentRunProvider**: `community`, `anthropic`, or `disabled`
- **autoInvestigateIssuesEnabled**: whether to automatically trigger investigations
- **maxRuntimeMinutes**: maximum investigation duration (default: 90 minutes)
- **maxHumanResumeCount**: how many times a human can resume an investigation (default: 3)
- **prPolicy**: when to open PRs -- `never`, `on_ready_to_pr`, or `always`
- **autoMergeFixPrs**: auto-merge policy -- `never`, `when_checks_pass`, or `immediately`
- **customInstructions**: project-specific instructions for the agent (up to 8000 characters)

### Setting Up the LLM Provider

For the Anthropic agent runner, set the `ANTHROPIC_API_KEY` environment variable. The community runner does not require an external LLM provider -- it records a local incident summary without performing deep code analysis.

> **Takeaway**: Superlog's self-hosted architecture means your observability data never leaves your infrastructure. Unlike cloud-based AIOps platforms that require sending traces, metrics, and logs to their servers, Superlog runs entirely on your own infrastructure with Postgres and ClickHouse, making it suitable for privacy-sensitive and compliance-regulated environments.

## Usage Examples

### Example 1: Docker Compose Stack

The Docker Compose configuration starts Postgres, ClickHouse, and the OTel Collector:

```yaml
# docker-compose.yml (simplified)
services:
  postgres:
    image: postgres:16
    environment:
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
      POSTGRES_DB: superlog
    ports:
      - "5434:5432"

  clickhouse:
    image: clickhouse/clickhouse-server:26.1
    environment:
      CLICKHOUSE_DB: superlog
      CLICKHOUSE_USER: default
      CLICKHOUSE_PASSWORD: ""
    ports:
      - "8123:8123"
      - "9000:9000"

  collector:
    image: otel/opentelemetry-collector-contrib:0.150.1
    command: ["--config=/etc/collector/config.yaml"]
    volumes:
      - ./infra/collector/config.yaml:/etc/collector/config.yaml:ro
    ports:
      - "4317:4317"
      - "4318:4318"
```

### Example 2: OTel Collector Configuration

The collector configuration handles traces, logs, and metrics with project-scoped attribute injection:

```yaml
# infra/collector/config.yaml (key sections)
receivers:
  otlp:
    protocols:
      http:
        endpoint: 0.0.0.0:4318
      grpc:
        endpoint: 0.0.0.0:4317

processors:
  attributes/from_metadata:
    actions:
      - key: superlog.project_id
        action: insert
        from_context: metadata.x-superlog-project-id

exporters:
  clickhouse:
    endpoint: tcp://clickhouse:9000
    database: superlog
    logs_table_name: otel_logs
    traces_table_name: otel_traces

service:
  pipelines:
    traces:
      receivers: [otlp]
      processors: [attributes/from_metadata, batch]
      exporters: [clickhouse]
    logs:
      receivers: [otlp]
      processors: [attributes/from_metadata, batch]
      exporters: [clickhouse]
    metrics:
      receivers: [otlp]
      processors: [attributes/from_metadata, batch]
      exporters: [clickhouse]
```

### Example 3: Agent Memory Tools

The agent memory system provides three tools that the AI agent can use during investigations:

```json
{
  "save_memory": {
    "kind": "terminology",
    "title": "Payment service naming",
    "body": "The 'checkout' service is called 'payment-gateway' internally. All references to 'checkout' in traces mean the payment-gateway deployment."
  }
}
```

```json
{
  "update_memory": {
    "id": "mem_abc123",
    "status": "archived",
    "body": "This fact is no longer accurate after the Q2 migration."
  }
}
```

```json
{
  "list_memories": {}
}
```

### Example 4: Configuring PR Policy via API

```bash
curl -X PATCH http://localhost:4100/api/projects/{projectId}/automation \
  -H "Authorization: Bearer {token}" \
  -H "Content-Type: application/json" \
  -d '{
    "agentRunProvider": "anthropic",
    "autoInvestigateIssuesEnabled": true,
    "prPolicy": "on_ready_to_pr",
    "autoMergeFixPrs": "when_checks_pass",
    "customInstructions": "Always add unit tests for bug fixes. Our main branch is 'main' not 'master'."
  }'
```

### Example 5: Sample Application Instrumentation

Superlog includes a sample Next.js application in `apps/sample/` that demonstrates OpenTelemetry instrumentation:

```typescript
// apps/sample/instrumentation.ts
export async function register() {
  if (process.env.NEXT_RUNTIME === "nodejs") {
    const { NodeSDK } = await import("@opentelemetry/sdk-node");
    const { getNodeAutoInstrumentations } = await import(
      "@opentelemetry/auto-instrumentations-node"
    );
    const { OTLPTraceExporter } = await import(
      "@opentelemetry/exporter-trace-otlp-proto"
    );
    // SDK initialization with OTLP exporter pointing to Superlog
  }
}
```

### Example 6: Querying Telemetry via the API

```bash
# Get project stats (traces, logs, metrics, issues in the last hour)
curl http://localhost:4100/api/projects/{projectId}/stats \
  -H "Authorization: Bearer {token}"

# Query logs with filters
curl -X POST http://localhost:4100/api/projects/{projectId}/explore/logs \
  -H "Authorization: Bearer {token}" \
  -H "Content-Type: application/json" \
  -d '{"service": "api", "severity": "ERROR", "limit": 100}'

# Query traces with minimum duration filter
curl -X POST http://localhost:4100/api/projects/{projectId}/explore/traces \
  -H "Authorization: Bearer {token}" \
  -H "Content-Type: application/json" \
  -d '{"service": "api", "minDurationMs": 500, "limit": 50}'
```

![Superlog Observability Stack](/assets/img/diagrams/superlog/superlog-observability-stack.svg)

The diagram above shows how Superlog integrates with the broader OpenTelemetry ecosystem. Applications written in Node.js, Python, Go, Java, or any language with an OTel SDK export telemetry to the OTel Collector. The Collector forwards data through Superlog's OTLP Proxy, which authenticates and enriches the payload before storing it in ClickHouse's purpose-built telemetry tables. The Superlog Dashboard queries these tables to provide real-time visualization of traces, logs, and metrics.

## Comparison with Alternatives

### Superlog vs. Datadog

Datadog is a cloud-only observability platform with comprehensive monitoring capabilities, but it does not have AI agents that investigate incidents and open fix pull requests. Datadog's AI features are limited to anomaly detection and alerting. Pricing is usage-based and can become expensive at scale. Superlog is open-source and self-hosted, meaning no vendor lock-in and no per-ingest pricing.

### Superlog vs. Grafana/Prometheus/Loki

The Grafana ecosystem provides excellent monitoring and visualization, but it is purely passive -- it detects and displays problems but does not investigate or fix them. There are no AI agents, no memory system, and no automated remediation. Superlog adds the agentic layer on top of OpenTelemetry data that you might already be collecting with Grafana stack components.

### Superlog vs. New Relic

New Relic offers AI-powered alerting and some automated analysis, but it is a cloud-only SaaS platform with vendor lock-in. The AI features do not open pull requests with code fixes. Superlog's self-hosted architecture and code-level remediation through PRs provide a fundamentally different value proposition.

### Superlog vs. Enterprise AIOps (Moogsoft, BigPanda)

Enterprise AIOps platforms provide alert correlation and noise reduction, but they are expensive, closed-source, and typically do not write code fixes. They focus on reducing alert fatigue, not on automating remediation. Superlog is open-source, self-hosted, and goes beyond alert correlation to actual code-level fixes.

### Superlog vs. OpenTelemetry Collector

The OTel Collector is a telemetry pipeline, not an observability platform. It collects and routes data but provides no analysis, no incident management, and no AI agents. Superlog uses the OTel Collector as a component of its architecture and adds the intelligence layer on top.

> **Important**: While Superlog's AI agents can automatically investigate incidents and open fix pull requests, the automation level is fully configurable. Teams can start with the community runner that only records incident summaries, graduate to the Anthropic runner that investigates and suggests fixes, and control PR policy from `never` to `always`. Auto-merge policies provide an additional safety net. This graduated approach ensures trust before full automation.

## Conclusion

Superlog represents a paradigm shift from passive monitoring to active self-healing observability. By combining OpenTelemetry's vendor-neutral telemetry standard with AI agents that have persistent memory and can open pull requests with code fixes, Superlog closes the loop between detection and remediation in a way that no traditional observability tool does.

The best use cases for Superlog are self-hosted environments where data privacy is paramount, teams that want to reduce the toil of incident response, and organizations already invested in the OpenTelemetry standard. The project is young -- created in June 2026 -- but backed by Y Combinator and growing rapidly with an active community on [Discord](https://discord.gg/wJ56aRh8hx).

Getting started is straightforward: clone the repo, run Docker Compose, instrument your application with the OTel SDK, and configure the agent runner. The community edition is free and open-source under the Apache 2.0 license. As LLM capabilities improve and the memory system accumulates project-specific knowledge, Superlog's self-healing effectiveness will only increase over time.

- **GitHub**: [https://github.com/superloglabs/superlog](https://github.com/superloglabs/superlog)
- **Website**: [https://superlog.sh](https://superlog.sh)
- **Skills**: [https://github.com/superloglabs/skills](https://github.com/superloglabs/skills)
- **Discord**: [https://discord.gg/wJ56aRh8hx](https://discord.gg/wJ56aRh8hx)