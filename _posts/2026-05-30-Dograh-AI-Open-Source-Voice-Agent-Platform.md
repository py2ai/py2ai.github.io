---
layout: post
title: "Dograh AI: Open-Source Voice Agent Platform with Drag-and-Drop Workflow Builder"
description: "Learn how Dograh AI provides an open-source, self-hostable voice AI platform with drag-and-drop workflow builder, Pipecat voice pipeline, multi-provider LLM/STT/TTS integration, and MCP server for AI agent control."
date: 2026-05-30
header-img: "img/post-bg.jpg"
permalink: /Dograh-AI-Open-Source-Voice-Agent-Platform/
featured-img: ai-coding-frameworks/ai-coding-frameworks
categories: [AI, Open Source, Voice AI]
tags: [dograh, voice-ai, pipecat, webrtc, telephony, fastapi, nextjs, open-source, llm, tts, stt, mcp, docker, workflow-builder]
keywords: "open source voice AI platform, self-hosted voice agent builder, Dograh AI alternative to Vapi, drag and drop voice workflow builder, Pipecat voice pipeline framework, WebRTC voice agent deployment, multi-provider LLM STT TTS integration, telephony AI agent open source, MCP server voice AI integration, Python TypeScript voice AI SDK"
author: "PyShine"
---

# Dograh AI: Open-Source Voice Agent Platform with Drag-and-Drop Workflow Builder

Dograh AI is an open source voice AI platform that gives developers full control over building, deploying, and managing production voice agents. Unlike proprietary alternatives such as Vapi and Retell, Dograh is 100% open source under the BSD 2-Clause license, self-hostable, and designed with a bring-your-own-model philosophy that eliminates vendor lock-in. Whether you need a simple customer service bot or a complex multi-step voice workflow with telephony integration, Dograh provides the infrastructure to make it happen without depending on any single cloud provider.

Founded by Y Combinator alumni, Dograh has reached version 1.32.0 with active development and frequent releases. The platform ships with auto-generated API keys and its own built-in LLM, TTS, and STT stack, meaning you can talk to your first bot without configuring any external API keys. When you are ready to scale, you can plug in your own providers across 11 LLM services, 10 STT engines, 11 TTS systems, and 7 telephony carriers.

> **Key Insight:** Dograh ships with auto-generated API keys and its own LLM / TTS / STT stack -- no API keys needed to talk to your first bot. Connect your own keys for LLM, TTS, STT, or Telephony anytime.

## Architecture Overview

Dograh follows a full-stack architecture with clear separation between the frontend, backend, data layer, and voice pipeline. The entire system is containerized with Docker and deployable with a single `docker compose up` command.

![Architecture Overview](/assets/img/diagrams/dograh/dograh-architecture.svg)

### Understanding the Architecture

The architecture diagram above illustrates the layered design of the Dograh platform. Let us break down each component and its role in the system:

**UI Layer: Next.js 15 / React 19**

The frontend is built with Next.js 15 running React 19, TypeScript, and Tailwind CSS. The most distinctive UI component is the drag-and-drop workflow builder powered by `@xyflow/react`, which allows users to visually construct voice agent workflows by placing and connecting typed nodes on a canvas. The UI communicates with the backend through REST API calls and supports real-time updates for call monitoring and metrics dashboards.

**API Layer: FastAPI (Python 3.13)**

The backend runs on Python 3.13 with FastAPI, using SQLAlchemy in async mode for database operations and ARQ (async Redis queue) for background task management. ARQ handles campaign calls and pipeline runs asynchronously, ensuring the API remains responsive even when processing long-running voice sessions. The API layer also exposes the MCP server endpoint at `/api/v1/mcp`, enabling AI coding agents to interact with the platform programmatically.

**Data Layer: PostgreSQL 17 + Redis 7 + MinIO**

The persistence tier uses PostgreSQL 17 with the pgvector extension for vector storage and similarity search. Redis 7 serves dual purposes: as a cache layer for session data and as the message queue backend for ARQ background tasks. MinIO provides S3-compatible object storage in a bucket named "voice-audio" for all audio recordings generated during voice agent sessions.

**Voice Pipeline: Pipecat Framework**

The Pipecat framework, included as a git submodule, is the engine that drives all voice processing. It provides the transport layer, speech-to-text, language model inference, text-to-speech, and audio processing in a composable pipeline architecture. Pipecat supports both a standard STT-to-LLM-to-TTS pipeline and a realtime speech-to-speech pipeline for models that handle audio natively.

**External Integrations**

Dograh connects to external LLM providers (11 options including OpenAI, Groq, Google, Azure, and Bedrock), STT providers (10 options including Deepgram, OpenAI, and Google), TTS providers (11 options including ElevenLabs, Deepgram, and Cartesia), and telephony providers (7 options including Twilio, Vonage, and Plivo). All external integrations follow a plugin architecture, making it straightforward to add new providers.

**Docker Deployment**

The entire stack runs inside Docker containers, with the API container executing as a non-root "dograh" user for security. Docker Compose orchestrates all services, and a single command deploys the complete platform including the TURN server (Coturn) for WebRTC NAT traversal and Nginx as a reverse proxy for remote deployments.

## Workflow Graph Engine

The workflow graph engine is Dograh's visual programming system for constructing voice agent logic. Instead of writing code to define call flows, users drag and drop typed nodes onto a canvas and connect them with conditional edges. The engine validates the graph structure and extracts template variables from node prompts.

![Workflow Graph Engine](/assets/img/diagrams/dograh/dograh-workflow-graph.svg)

### Understanding the Workflow Graph Engine

The workflow graph engine diagram above shows the node types and their relationships within a directed graph. Let us examine each component:

**Core Node Types**

Dograh defines 7 core node types that form the building blocks of any voice workflow:

- **startCall**: The entry point of every workflow. This node defines the initial greeting (text or audio), the system prompt for the voice agent, and configuration options like interrupt handling and delayed start. Every valid workflow must have exactly one startCall node.
- **agentNode**: The primary processing node where the LLM interacts with the caller. Each agentNode has its own prompt, LLM provider configuration, and extraction settings. Multiple agentNodes can be chained together to create multi-step conversations.
- **endCall**: The terminal node that gracefully terminates the call. It can deliver a final message and trigger post-call actions like recording the conversation or updating a CRM system.
- **globalNode**: A special overlay node that applies a global prompt or configuration to all other nodes in the workflow. This is useful for setting system-wide behaviors like tone, language preferences, or compliance rules that should persist across all agent nodes.
- **trigger**: An entry point for scheduled or event-driven workflows. Triggers can initiate calls based on time schedules, allowing proactive outreach campaigns.
- **webhook**: An entry point for external systems to trigger workflows. Webhooks enable integration with CRM systems, helpdesk platforms, or any HTTP-capable service that needs to initiate a voice agent call.
- **qa**: A quality analysis node that evaluates conversation quality, providing automated scoring and feedback on agent performance.

**Edge Properties**

Edges connect nodes and define the transition logic between them. Each edge carries three properties:

- **label**: A human-readable name for the transition (e.g., "interested", "qualified")
- **condition**: A natural language description of when this edge should be followed (e.g., "caller expresses interest", "caller meets criteria"). The LLM evaluates these conditions during the conversation to determine which path to take.
- **transition_speech**: An optional spoken phrase the agent delivers when transitioning to the next node, creating smooth conversational handoffs between agent nodes.

**Template Variables**

Node prompts support template variables using the `{{ variable }}` syntax with optional filters. For example, `{{ caller_name | default:Friend }}` inserts the caller's name if available, or falls back to "Friend" if not. These variables can be populated from webhook payloads, pre-call data fetches, or extraction results from previous nodes. The template engine extracts all variables from the workflow definition at save time, making them available for configuration.

**Graph Validation**

The engine validates the workflow graph to ensure structural integrity: every workflow must have a startCall node, all agentNodes must be reachable from the start, and end conditions must be defined. Validation happens at save time through the UI and at call time through the SDK, catching configuration errors before they affect live calls.

## Voice Pipeline Deep Dive

The voice pipeline is the runtime engine that processes audio during a live call. Dograh uses the Pipecat framework to implement two distinct pipeline architectures: a standard pipeline for maximum provider flexibility and a realtime pipeline for speech-to-speech models.

![Voice Pipeline Flow](/assets/img/diagrams/dograh/dograh-voice-pipeline.svg)

### Understanding the Voice Pipeline

The voice pipeline diagram above illustrates the two parallel processing paths available in Dograh. Let us examine each stage:

**Standard Pipeline**

The standard pipeline follows the traditional STT-to-LLM-to-TTS pattern, giving developers maximum flexibility in mixing and matching providers:

1. **Transport Input**: Receives audio from the caller via WebRTC or telephony transport.
2. **STT (Speech-to-Text)**: Converts the incoming audio stream to text using any of the 10 supported STT providers. Deepgram is a popular choice for its low-latency streaming capabilities.
3. **Voicemail Detector**: Analyzes the transcribed text to detect whether the call has reached a voicemail system. The native Pipecat VoicemailDetector uses a non-blocking TTS gate to avoid wasting LLM tokens on automated answering systems.
4. **User Aggregator**: Collects and buffers user utterances to provide complete conversational context to the LLM, handling cases where the user pauses mid-sentence or speaks in fragments.
5. **LLM Gate**: Controls when and how the LLM is invoked, implementing rate limiting, circuit breaking (50% failure rate threshold over a 2-minute sliding window with minimum 5 calls), and context window management.
6. **LLM**: Processes the aggregated user input through the configured language model (any of 11 providers) to generate a response.
7. **Callback Processor**: Executes any configured callback functions, enabling real-time integration with external systems like CRMs or databases during the conversation.
8. **Recording Router**: Manages audio recording routing, determining which audio streams to capture and where to store them (MinIO S3 bucket).
9. **TTS (Text-to-Speech)**: Converts the LLM response back to audio using any of the 11 supported TTS providers.
10. **Transport Output**: Sends the synthesized audio back to the caller.
11. **Audio Buffer**: The AudioBufferProcessor records both input and output audio streams for complete conversation archival.
12. **Assistant Aggregator**: Collects assistant responses for metrics and logging.
13. **Metrics**: Collects performance data including latency, token usage, and call quality indicators.

**Realtime Pipeline**

The realtime pipeline is designed for speech-to-speech models that handle STT, LLM, and TTS internally, eliminating the need for separate processing stages:

1. **Transport Input**: Same as standard pipeline.
2. **User Aggregator**: Buffers user audio for the realtime model.
3. **Realtime LLM**: Models like OpenAI Realtime, Gemini Live, Gemini Live Vertex, xAI Grok Realtime, and Ultravox process audio input directly and produce audio output, handling speech recognition, language understanding, and speech synthesis in a single model.
4. **Callback Processor**: Same callback integration as the standard pipeline.
5. **Transport Output**: Sends the model's audio output to the caller.
6. **Audio Buffer / Assistant Aggregator / Metrics**: Same post-processing as the standard pipeline.

> **Takeaway:** The Pipecat pipeline architecture supports two modes: a standard STT to LLM to TTS pipeline for maximum provider flexibility, and a realtime speech-to-speech pipeline for models like OpenAI Realtime and Gemini Live that handle audio natively.

## Multi-Provider Integration

One of Dograh's strongest differentiators is its extensive multi-provider support. The platform uses a service factory pattern that abstracts provider-specific APIs behind common interfaces, allowing developers to swap providers without changing workflow logic.

![Multi-Provider Ecosystem](/assets/img/diagrams/dograh/dograh-provider-ecosystem.svg)

### Understanding the Provider Ecosystem

The provider ecosystem diagram above shows the breadth of integration options available in Dograh. Let us examine each category:

**LLM Providers (11)**

Dograh supports 11 LLM providers, giving developers the freedom to choose based on cost, latency, quality, and compliance requirements:

- **OpenAI**: GPT-4o and GPT-4o-mini for general-purpose conversations
- **Groq**: LPU-accelerated inference for ultra-low-latency responses
- **OpenRouter**: Unified API access to hundreds of models from multiple providers
- **Google**: Gemini models for multimodal reasoning
- **Google Vertex**: Enterprise-grade Gemini with compliance features
- **Azure**: OpenAI models hosted on Azure for enterprise compliance
- **AWS Bedrock**: Amazon's managed model service for cloud-native deployments
- **MiniMax**: Chinese LLM provider for multilingual support
- **Speaches**: Open-source speech-to-speech model hosting
- **Dograh**: Dograh's own built-in LLM for zero-config startup
- **xAI**: Grok models including realtime speech-to-speech capabilities

**STT Providers (10)**

Speech-to-text options cover a wide range of accuracy, latency, and language requirements:

- **Deepgram** (including Flux): Industry-leading streaming STT with sub-200ms latency
- **OpenAI**: Whisper-based transcription with broad language support
- **Google**: Cloud Speech-to-Text with automatic punctuation
- **Cartesia**: Real-time streaming transcription
- **Dograh**: Built-in STT for immediate deployment
- **Sarvam**: Indian language specialist with Hindi, Tamil, Telugu support
- **Speaches**: Open-source STT option
- **AssemblyAI**: Developer-friendly API with speaker diarization
- **Gladia**: Multi-language transcription with word-level timestamps
- **Speechmatics**: Enterprise-grade accuracy with custom vocabulary

**TTS Providers (11)**

Text-to-speech options range from natural-sounding neural voices to low-latency streaming:

- **Deepgram**: Aura models for fast, natural speech
- **OpenAI**: TTS-1 and TTS-1-HD for high-quality output
- **Google**: Cloud TTS with WaveNet voices
- **ElevenLabs**: Premium neural voices with voice cloning
- **Cartesia**: Sonic models for ultra-low-latency streaming
- **Dograh**: Built-in TTS for zero-config startup
- **Camb**: Specialized voice options
- **Speaches**: Open-source TTS option
- **Rime**: Custom voice creation platform
- **Sarvam**: Indian language TTS support
- **MiniMax**: Chinese language TTS

**Telephony Providers (7)**

The telephony provider registry uses a plugin architecture for PSTN connectivity:

- **Twilio**: The most widely used cloud communications API
- **Vonage**: Global communications platform with SMS and voice
- **Plivo**: Cloud communications API with competitive pricing
- **Telnyx**: Telecom infrastructure with SIP trunking
- **Vobiz**: Business VoIP provider
- **Cloudonix**: Real-time communications platform
- **ARI**: Asterisk REST Interface for on-premise PBX integration

> **Important:** The telephony provider registry uses a plugin architecture where adding a new provider requires only its own folder plus a single import line -- no edits needed outside the provider package.

## MCP Server for AI Agent Integration

Dograh exposes a Model Context Protocol (MCP) server at `/api/v1/mcp`, built on the FastMCP framework. This endpoint enables AI coding agents like Claude Code, Cursor, or GitHub Copilot to create, read, and modify voice workflows programmatically, bridging the gap between AI-assisted development and voice AI deployment.

The MCP server provides 13 tools organized into four categories:

**Workflow Management (5 tools)**
- `create_workflow`: Create a new voice agent workflow from a definition
- `get_workflow`: Retrieve a specific workflow by ID
- `save_workflow`: Update an existing workflow definition
- `get_workflow_code`: Get the code representation of a workflow
- `list_workflows`: List all workflows in the organization

**Node Type Introspection (2 tools)**
- `get_node_type`: Get the schema and configuration for a specific node type
- `list_node_types`: List all available node types with their specifications

**Knowledge and Search (3 tools)**
- `list_documents`: List documents in the knowledge base
- `list_docs`: List available documentation pages
- `read_doc`: Read a specific documentation page
- `search_docs`: Search documentation using semantic queries

**Operational (3 tools)**
- `list_credentials`: List configured API credentials
- `list_recordings`: List voice session recordings
- `list_tools`: List available integration tools

> **Amazing:** Dograh's MCP server at /api/v1/mcp enables AI coding agents to create, read, and modify voice workflows programmatically -- bridging the gap between AI-assisted development and voice AI deployment.

This MCP integration means a developer can describe a voice agent workflow in natural language to their AI coding assistant, and the assistant can use the MCP tools to create the workflow, configure the nodes, and deploy it -- all without touching the Dograh UI. This is particularly powerful for teams that want to version-control their voice agent configurations alongside their application code.

## SDK and Developer Experience

Dograh provides dual SDKs for programmatic workflow creation, supporting both Python and TypeScript developers with typed workflow builders and validation.

### Python SDK

Install the Python SDK from PyPI:

```bash
pip install dograh-sdk
```

The Python SDK offers two approaches to workflow creation. The first uses generated models for type-safe construction:

```python
from dograh_sdk import DograhClient
from dograh_sdk._generated_models import CreateWorkflowRequest

with DograhClient(base_url="http://localhost:8000", api_key="your-api-key") as client:
    workflow = client.create_workflow(
        body=CreateWorkflowRequest(
            name="My SDK-created agent",
            workflow_definition={
                "nodes": [{
                    "id": "1",
                    "type": "startCall",
                    "position": {"x": 271, "y": 4},
                    "data": {
                        "name": "start call",
                        "greeting_type": "text",
                        "prompt": "You are a helpful voice agent. Start by saying Hi.",
                        "allow_interrupt": False,
                        "add_global_prompt": False,
                        "delayed_start": False,
                        "extraction_enabled": False,
                        "pre_call_fetch_enabled": False,
                    },
                }],
                "edges": [],
                "viewport": {"x": 0, "y": 0, "zoom": 1},
            },
        )
    )
    print(f"Created workflow {workflow.id}: {workflow.name!r}")
```

The second approach uses the typed `Workflow` builder for a more fluent API:

```python
from dograh_sdk import DograhClient, Workflow

with DograhClient(base_url="http://localhost:8000", api_key="your-api-key") as client:
    wf = Workflow(client=client, name="loan_qual")
    start = wf.add(type="startCall", name="greeting", prompt="You are a loan qualification agent.")
    qualify = wf.add(type="agentNode", name="qualify", prompt="Ask about income and credit score.")
    end = wf.add(type="endCall", name="end", prompt="Thank the caller and end the conversation.")
    wf.edge(start, qualify, label="interested", condition="caller expresses interest")
    wf.edge(qualify, end, label="qualified", condition="caller meets criteria")
    payload = wf.to_json()
```

The `Workflow` builder validates node data against the spec catalog at call time, catching configuration errors early. This means you get immediate feedback if a node property is missing or has an invalid value, rather than discovering the issue when the workflow is deployed.

### TypeScript SDK

Install the TypeScript SDK from npm:

```bash
npm install @dograh/sdk
```

The TypeScript SDK provides the same typed workflow builder experience with full TypeScript type definitions, enabling IDE autocompletion and compile-time type checking for workflow construction.

## Self-Hosting and Deployment

Dograh is designed for Docker-first deployment, making self-hosting straightforward whether you are running locally or on a remote server.

### One-Command Self-Host

The fastest way to get Dograh running is with a single Docker Compose command:

```bash
curl -o docker-compose.yaml https://raw.githubusercontent.com/dograh-hq/dograh/main/docker-compose.yaml && REGISTRY=ghcr.io/dograh-hq ENABLE_TELEMETRY=true docker compose up --pull always
```

This command downloads the Docker Compose configuration and starts all services: the FastAPI backend, Next.js frontend, PostgreSQL database, Redis cache, MinIO storage, Coturn TURN server, and Nginx reverse proxy. The `REGISTRY=ghcr.io/dograh-hq` variable tells Docker to pull pre-built container images from GitHub Container Registry rather than building locally.

### Local vs Remote Deployment

For local development, Dograh runs directly on `localhost` without requiring a TURN server or Nginx. The WebRTC signaling works over localhost connections without NAT traversal.

For remote server deployment, the architecture includes:

- **Coturn TURN Server**: Handles WebRTC NAT traversal for callers behind firewalls. TURN authentication uses HMAC-SHA1 time-limited credentials via the TURN REST API, providing secure access without long-lived secrets.
- **Nginx Reverse Proxy**: Terminates TLS and routes traffic to the appropriate backend services. The Nginx configuration is templated and automatically generated during deployment.
- **Cloudflared Tunnel**: For development purposes, Dograh supports Cloudflared tunnels to expose local services to the internet without configuring port forwarding.

### Container Security

The API container runs as a non-root "dograh" user, following Docker security best practices. CORS is configured permissively (wildcard, no credentials) in open-source mode for ease of development, but switches to a strict allowlist in SaaS mode for production security.

### Observability

Dograh includes built-in observability through three integrated services:

- **Sentry**: Error tracking and crash reporting
- **PostHog**: Product analytics and usage metrics
- **Langfuse**: LLM tracing for debugging prompt engineering and response quality

Authentication uses JWT tokens with a 720-hour (30-day) default expiry in open-source mode, supporting both Local (email/password) and Stack (SaaS) auth providers.

## WebRTC and Telephony

Dograh supports two primary audio transport mechanisms: WebRTC for browser-based calls and telephony for PSTN connectivity.

### WebRTC Signaling

The WebRTC implementation uses WebSocket-based signaling with ICE trickling for efficient connection establishment. The signaling flow follows this sequence:

1. Client opens a WebSocket connection to the Dograh signaling server
2. Client creates an RTCPeerConnection and generates an offer SDP
3. Server responds with an answer SDP including ICE candidates
4. ICE trickling allows additional candidates to be exchanged after the initial offer/answer
5. Once ICE negotiation completes, audio flows directly between client and server (or through TURN if NAT traversal is required)

The TURN server uses HMAC-SHA1 time-limited credentials, generated via the TURN REST API. This approach provides secure, temporary access to the TURN relay without exposing long-lived shared secrets.

### Telephony Provider Registry

The telephony integration follows a plugin architecture where each provider is self-contained in its own directory. The registry pattern means that adding a new telephony provider requires only creating its own folder with the provider implementation and adding a single import line to the registry. No modifications are needed outside the provider package.

Each telephony provider implements a common interface that handles:

- Inbound call setup and signaling
- Outbound call initiation
- Call transfer to human agents
- DTMF tone processing
- Call status monitoring and hangup detection

The default concurrency limit is 2 concurrent calls per organization, configurable based on infrastructure capacity. The circuit breaker protects against cascading failures with a 50% failure rate threshold over a 2-minute sliding window, requiring a minimum of 5 calls before activating.

## Getting Started

Getting started with Dograh takes just a few minutes. The platform is designed for zero-config startup, meaning you can create and test your first voice bot without any external API keys.

### Step 1: Deploy Dograh

Run the one-command deployment:

```bash
curl -o docker-compose.yaml https://raw.githubusercontent.com/dograh-hq/dograh/main/docker-compose.yaml && REGISTRY=ghcr.io/dograh-hq ENABLE_TELEMETRY=true docker compose up --pull always
```

Wait for all containers to start. The UI will be available at `http://localhost:3000` and the API at `http://localhost:8000`.

### Step 2: Create Your First Voice Bot

Open the Dograh UI at `http://localhost:3000`. The platform auto-generates API keys and includes its own LLM, TTS, and STT stack, so you can create a voice bot immediately:

1. Click "Create Workflow" to open the drag-and-drop builder
2. Drag a **startCall** node onto the canvas
3. Configure the prompt: "You are a helpful assistant. Greet the caller warmly."
4. Drag an **endCall** node and connect it to the startCall node
5. Save the workflow

### Step 3: Test the Voice Bot

Use the built-in WebRTC test interface in the Dograh UI to make a test call. Click the "Test Call" button on your workflow, grant microphone access, and start talking to your voice agent.

### Step 4: Add Your Own Providers (Optional)

When you are ready to use your own LLM, TTS, or STT providers, navigate to Settings and add your API keys. Dograh supports 11 LLM providers, 10 STT providers, and 11 TTS providers. You can mix and match providers within a single workflow -- for example, using Deepgram for STT, OpenAI for the LLM, and ElevenLabs for TTS.

### Step 5: Connect Telephony (Optional)

To receive or make phone calls over the PSTN, configure one of the 7 supported telephony providers. Twilio is the most common choice. Add your Twilio credentials in Settings, purchase a phone number, and configure the webhook URL to point to your Dograh instance.

### Step 6: Use the SDK for Automation

For programmatic workflow creation, install the SDK and build workflows in code:

```bash
pip install dograh-sdk
```

```python
from dograh_sdk import DograhClient, Workflow

with DograhClient(base_url="http://localhost:8000", api_key="your-api-key") as client:
    wf = Workflow(client=client, name="my_first_agent")
    start = wf.add(type="startCall", name="greeting", prompt="You are a helpful voice agent.")
    end = wf.add(type="endCall", name="end", prompt="Thank the caller and goodbye.")
    wf.edge(start, end, label="done", condition="caller is finished")
    payload = wf.to_json()
```

Dograh combines the ease of visual workflow building with the power of programmatic SDK access, giving teams the flexibility to choose their preferred approach or combine both. The MCP server integration extends this further by enabling AI coding agents to build and modify voice workflows, making Dograh a compelling choice for teams that want to integrate voice AI into their existing development workflows without vendor lock-in.

## Links

- **GitHub Repository**: [https://github.com/dograh-hq/dograh](https://github.com/dograh-hq/dograh)
- **Python SDK (PyPI)**: [https://pypi.org/project/dograh-sdk/](https://pypi.org/project/dograh-sdk/)
- **TypeScript SDK (npm)**: [https://www.npmjs.com/package/@dograh/sdk](https://www.npmjs.com/package/@dograh/sdk)