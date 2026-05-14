---
layout: post
title: "AiToEarn: AI-Powered Earning Platform for Automated Income"
description: "Discover how AiToEarn uses AI agents to automate content creation, publishing, engagement, and monetization across 14+ social platforms for creators and one-person companies."
date: 2026-05-14
header-img: "img/post-bg.jpg"
permalink: /AiToEarn-AI-Powered-Earning-Platform/
featured-img: ai-coding-frameworks/ai-coding-frameworks
categories: [AI, TypeScript, Developer Tools]
tags: [AiToEarn, AI earning, automated income, TypeScript, open source, AI platform, how to use, setup guide, tutorial, passive income]
keywords: "how to use AiToEarn, AiToEarn tutorial, AiToEarn AI earning platform, AiToEarn vs alternatives, AiToEarn installation guide, open source AI earning, AiToEarn TypeScript setup, best AI earning platform, AiToEarn for beginners, AI-powered income automation"
author: "PyShine"
---

## What Is AiToEarn?

AiToEarn is an open-source AI-powered earning platform that helps one-person companies (OPCs), creators, brands, and businesses build, distribute, and monetize content through intelligent automation. With over 13,600 GitHub stars and a growth rate exceeding 3,300 stars per week, AiToEarn has rapidly become one of the most popular open-source projects in the AI content marketing space.

Built with TypeScript and powered by NestJS on the backend and Next.js on the frontend, AiToEarn provides four core AI agent capabilities -- Monetize, Publish, Engage, and Create -- that cover the entire content lifecycle from ideation to income. Whether you are a solo creator managing multiple social media accounts or a brand looking to scale your content operations, AiToEarn provides the infrastructure to automate the entire process.

> **Key Insight:** AiToEarn is not just another content scheduler. It is a full-stack AI agent platform that handles content creation, multi-platform distribution, audience engagement, and revenue generation -- all from a single codebase that you can self-host.

---

## Architecture Overview

AiToEarn follows a microservices architecture with clear separation between the AI agent service, the main API server, and the web frontend. The system is designed for Docker-first deployment, making it straightforward to run on any server.

![AiToEarn Architecture](/assets/img/diagrams/aitoearn/aitoearn-architecture.svg)

The architecture diagram above illustrates how the different components interact. Users access the platform through three client interfaces: the Next.js web application, the Electron desktop client, and the browser extension for engagement automation. All requests flow through Nginx as a reverse proxy, which routes traffic to either the `aitoearn-server` (NestJS API) or the `aitoearn-ai` (AI Agent Service) depending on the request type.

The `aitoearn-server` handles all business logic including user management, content scheduling, channel authentication, and monetization tracking. The `aitoearn-ai` service manages AI-powered content generation, invoking models from OpenAI, Anthropic, Google Gemini, Grok, and Volcengine to produce text, images, and video content. Both services share MongoDB for persistent storage and Redis for caching and task queuing. RustFS (an S3-compatible object store) handles media asset storage for images, videos, and generated content.

The MCP (Model Context Protocol) integration is particularly noteworthy. It allows external AI assistants like Claude Desktop and Cursor to connect directly to AiToEarn's agent capabilities, enabling a workflow where you can ask your AI assistant to create and publish content without ever opening the AiToEarn web interface.

---

## Four Core Agent Capabilities

AiToEarn organizes its functionality around four AI agent pillars, each addressing a distinct phase of the content monetization pipeline.

![AiToEarn Features](/assets/img/diagrams/aitoearn/aitoearn-features.svg)

### Monetize Agent

The Monetize agent is the core value proposition of AiToEarn. It connects creators with brand promotion tasks and handles all settlement logic. Three settlement models are supported:

- **CPS (Cost Per Sale)** -- Creators earn a percentage of each transaction generated through their content
- **CPE (Cost Per Engagement)** -- Payment based on measurable engagement metrics like likes, shares, and comments
- **CPM (Cost Per Mille)** -- Revenue calculated per thousand impressions or views

All settlements are results-driven, meaning creators only get paid when measurable outcomes occur. This creates a transparent and fair ecosystem where brands pay for actual performance rather than vague promises.

### Publish Agent

The Publish agent enables one-click distribution to 14 major social media platforms across both Chinese and international markets. Supported platforms include Douyin, Xiaohongshu (Rednote), Kuaishou, Bilibili, WeChat Channels, WeChat Official Accounts, TikTok, YouTube, Facebook, Instagram, Threads, X (Twitter), Pinterest, and LinkedIn.

A built-in calendar scheduler lets creators plan content across all platforms in a unified view, eliminating the need to manually log into each platform separately. The agent handles platform-specific formatting, optimal posting times, and content adaptation.

### Engage Agent

The Engage agent operates through a browser extension that automates audience interaction across all supported platforms. Key capabilities include:

- **Automated Actions** -- Auto-like, bookmark, and follow operations at scale
- **AI Smart Replies** -- Large language models generate contextually relevant responses to each comment
- **Comment Mining** -- Detects high-conversion signals such as "link please" or "how to buy" and responds instantly
- **Brand Monitoring** -- Real-time tracking of brand mentions across platforms with proactive engagement suggestions

### Create Agent

The Create agent handles the entire content production pipeline. For video content, it orchestrates video generation models (Grok, Veo, Seedance), video translation modules, and video editing tools to produce finished videos from a single prompt. For image and text content, it leverages models like Nano Banana to generate high-quality visual content. Batch generation is supported, allowing creators to submit multiple creation tasks that the agent processes in parallel -- ideal for matrix account operations and large-scale content distribution.

> **Takeaway:** The four-agent architecture is what separates AiToEarn from simple scheduling tools. Each agent operates semi-independently but shares data through the unified backend, creating a feedback loop where engagement data informs content creation, and monetization data guides publishing strategy.

---

## Content Creation and Publishing Workflow

Understanding how AiToEarn processes a content request from start to finish helps clarify the value of the agent-based approach.

![AiToEarn Workflow](/assets/img/diagrams/aitoearn/aitoearn-workflow.svg)

The workflow begins when a user submits a request -- this could be a simple text prompt, a content idea, or a specific monetization task from the marketplace. The AI Agent (Create) takes this input and generates appropriate content, whether that is a video, image, or text post. The generated content then goes through a quality review step.

If the content does not meet quality standards, the decision node routes it back to the AI Agent for revision. This loop continues until the content is approved. Once approved, the Publish Agent takes over and distributes the content to all configured social media platforms simultaneously -- TikTok, YouTube, Instagram, X (Twitter), LinkedIn, and Pinterest, among others.

After publication, the Engage Agent monitors audience interactions, automatically responding to comments and identifying high-conversion signals. Finally, the Monetize Agent tracks earnings across all platforms, providing creators with a unified view of their income from CPS, CPE, and CPM settlements.

This end-to-end automation means that a creator can go from a single content idea to published content across 14 platforms with active audience engagement and tracked revenue -- all without manual intervention at each step.

---

## Platform Ecosystem and Integrations

AiToEarn's strength lies in its extensive integration ecosystem, connecting to social platforms, AI model providers, and infrastructure services.

![AiToEarn Ecosystem](/assets/img/diagrams/aitoearn-ecosystem.svg)

### Social Platform Support

The platform supports 14 social media platforms divided into two groups. International platforms include TikTok, YouTube, Facebook, Instagram, Threads, X (Twitter), Pinterest, and LinkedIn. Chinese platforms include Douyin, Xiaohongshu (Rednote), Kuaishou, Bilibili, WeChat Channels, and WeChat Official Accounts. OAuth integration is handled through a Relay system that allows self-hosted instances to use the official AiToEarn credentials for platform authentication, eliminating the need to register as a developer on each platform individually.

### AI Model Integration

AiToEarn integrates with multiple AI model providers to power its content generation capabilities. OpenAI GPT models handle text generation and conversation. Anthropic Claude provides advanced reasoning for content strategy. Google Gemini offers multimodal content generation. Grok powers video generation capabilities. Volcengine (ByteDance's cloud platform) provides video-on-demand and additional AI services tailored to the Chinese market.

### Infrastructure Stack

The self-hosted deployment runs on a robust infrastructure stack: MongoDB with replica sets for data persistence, Redis for caching and task queuing, RustFS (an S3-compatible object store) for media asset storage, Docker Compose for container orchestration, and Nginx as a reverse proxy handling SSL termination and load balancing.

> **Amazing:** The entire AiToEarn stack -- including MongoDB, Redis, RustFS, the AI service, the API server, the web frontend, and Nginx -- can be deployed with just three commands: `git clone`, `cd`, and `docker compose up -d`. This makes it one of the easiest full-stack AI platforms to self-host.

---

## Getting Started with AiToEarn

AiToEarn offers five ways to get started, each suited to different use cases.

### Option 1: Use the Website (Simplest)

No setup required. Open your browser and navigate to [aitoearn.ai](https://aitoearn.ai/) (international) or [aitoearn.cn](https://aitoearn.cn/) (China). Create an account and start using all features immediately.

### Option 2: Use in OpenClaw

If you already use OpenClaw (a popular AI assistant platform), install the AiToEarn plugin:

```bash
npx -y @aitoearn/openclaw-plugin-cli
```

On first run, select your environment (China or international) and enter your API Key. After setup, you can receive and execute AiToEarn earning tasks directly inside OpenClaw.

### Option 3: Use in Claude, Cursor, or Other AI Assistants

AiToEarn supports the MCP (Model Context Protocol) standard, allowing integration with any MCP-compatible AI assistant. For Claude Desktop, add the following to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "aitoearn": {
      "type": "http",
      "url": "https://aitoearn.ai/api/unified/mcp",
      "headers": {
        "x-api-key": "your-api-key"
      }
    }
  }
}
```

For Cursor, add the MCP URL and authentication header in the MCP settings panel. The SSE transport is also available at `https://aitoearn.ai/api/unified/sse` for tools that prefer Server-Sent Events.

### Option 4: Docker One-Click Deploy

For teams wanting full control, deploy AiToEarn on your own server:

```bash
git clone https://github.com/yikart/AiToEarn.git
cd AiToEarn
docker compose up -d
```

Open `http://localhost:8080` and you are ready. Configure the Relay service in `docker-compose.yml` to enable OAuth login for social media platforms without registering developer accounts:

```yaml
# Add under aitoearn-server environment
RELAY_SERVER_URL: https://aitoearn.ai/api
RELAY_API_KEY: your-api-key
RELAY_CALLBACK_URL: http://127.0.0.1:8080/api/plat/relay-callback
```

Then restart: `docker compose restart aitoearn-server`

### Option 5: Build from Source

For developers who want to customize or contribute:

```bash
# Backend
cd project/aitoearn-backend
pnpm install
cp apps/aitoearn-ai/config/config.js apps/aitoearn-ai/config/local.config.js
cp apps/aitoearn-server/config/config.js apps/aitoearn-server/config/local.config.js
pnpm nx serve aitoearn-ai
# In another terminal
pnpm nx serve aitoearn-server

# Frontend
cd project/aitoearn-web
pnpm install
pnpm run dev
```

The backend uses Nx monorepo tooling with NestJS, while the frontend is built on Next.js 14 with React 18, Ant Design, Tailwind CSS, and Zustand for state management.

> **Important:** When using API Keys, make sure the environment matches. A China API Key (from aitoearn.cn) must be used with China endpoints, and an international API Key (from aitoearn.ai) must be used with international endpoints. A mismatch will result in 401 authentication errors.

---

## Technical Architecture Deep Dive

For developers interested in the internals, AiToEarn's codebase is well-organized and follows modern best practices.

### Backend (aitoearn-backend)

The backend is an Nx monorepo containing two main applications:

- **aitoearn-server** -- The main NestJS API server handling user accounts, API keys, content management, channel connections, publish records, credits, notifications, and the unified MCP endpoint. Key modules include `account/`, `api-key/`, `channel/`, `content/`, `credits/`, `publish-record/`, `relay/`, and `unified-mcp/`.

- **aitoearn-ai** -- The AI agent service responsible for content generation, agent orchestration, draft generation, and material adaptation. It integrates with OpenAI, Anthropic, Google Gemini, Grok, and Volcengine APIs.

Shared libraries include `aitoearn-ai-client/`, `aitoearn-auth/`, `aitoearn-queue/`, `aitoearn-server-client/`, `mongodb/`, `redis/`, `redlock/`, `ali-oss/`, `ali-sms/`, `aws-s3/`, `mail/`, and `nest-mcp/`.

### Frontend (aitoearn-web)

Built with Next.js 14, React 18, Ant Design 5, Tailwind CSS 4, and Zustand for state management. The frontend includes comprehensive i18n support via i18next, E2E testing with Playwright, and a rich component library using Radix UI primitives.

### Desktop Client (aitoearn-electron)

An Electron-based desktop application providing native OS integration, built with Vite and better-sqlite3 for local data storage.

---

## Monetization Models Explained

AiToEarn's monetization system is built around three settlement models, each designed for different campaign objectives:

| Model | Full Name | How It Works | Best For |
|-------|-----------|--------------|----------|
| **CPS** | Cost Per Sale | Creators earn a percentage of each sale generated through their content | E-commerce promotions, affiliate marketing |
| **CPE** | Cost Per Engagement | Payment based on measurable engagement actions (likes, shares, comments) | Brand awareness campaigns, community building |
| **CPM** | Cost Per Mille | Revenue per thousand impressions or views | Mass awareness campaigns, video content |

The content marketplace (launched in version 2.1) allows brands to post promotion tasks with specific settlement models, and creators can browse and accept tasks that match their audience and content style. All tracking and settlement is handled automatically by the platform.

---

## Version History and Roadmap

AiToEarn has evolved significantly since its first open-source release:

- **v0.1.1** (Feb 2025) -- First open-source release with Xiaohongshu, Douyin, Kuaishou, and WeChat Channels support
- **v1.0.18** (Sep 2025) -- First international version with Facebook, Instagram, Threads, Twitter, YouTube, TikTok, Pinterest
- **v1.3.2** (Nov 2025) -- First fully usable open-source version
- **v1.4.0** (Nov 2025) -- In-app auto-update, AI content features (abbreviation, expansion, image/video generation, tag generation)
- **v1.4.3** (Dec 2025) -- "All In Agent" -- super AI agent for automatic content generation and publishing
- **v1.8.0** (Feb 2026) -- Offline business promotion solutions for restaurants, retail, hotels, salons, gyms
- **v2.1** (Mar 2026) -- Content marketplace, OpenClaw integration, MCP protocol support
- **April 2026** -- OpenClaw earning workflow support

The project is licensed under MIT, making it fully open for commercial use, modification, and distribution.

---

## Conclusion

AiToEarn represents a significant step forward in AI-powered content marketing. By combining four specialized AI agents -- Monetize, Publish, Engage, and Create -- into a single open-source platform, it eliminates the need for creators to juggle multiple tools for content creation, scheduling, engagement, and revenue tracking. The support for 14 social media platforms across both Chinese and international markets, combined with MCP integration for AI assistants and Docker-based self-hosting, makes it accessible to everyone from individual creators to enterprise teams.

With 13,600+ GitHub stars and rapid growth, AiToEarn has clearly struck a chord with the creator economy. Whether you use the hosted version, integrate it through MCP, or deploy it on your own infrastructure, AiToEarn provides a comprehensive toolkit for turning content into income through intelligent automation.

> **Takeaway:** If you are a content creator, one-person company, or brand looking to scale your content operations without multiplying your workload, AiToEarn's agent-based approach offers the most complete open-source solution available today. The five deployment options -- from zero-setup web access to full self-hosting -- mean there is an entry point for every technical level.