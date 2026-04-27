---
layout: post
title: "OpenClaw: Personal AI Assistant You Run on Your Own Devices"
description: "OpenClaw is a 359K-star open-source personal AI assistant that runs locally on your devices, supporting 20+ messaging channels, voice wake words, live canvas, and multi-agent routing with enterprise-grade security."
date: 2026-04-19
header-img: "img/post-bg.jpg"
permalink: /OpenClaw-Personal-AI-Assistant/
featured-img: ai-coding-frameworks/ai-coding-frameworks
tags:
  - Open Source
  - AI Assistant
  - TypeScript
  - Self-Hosted
  - Privacy
author: "PyShine"
---

# OpenClaw: Personal AI Assistant You Run on Your Own Devices

OpenClaw is an open-source personal AI assistant that runs on your own devices, answers you on the channels you already use, and gives you full control over your data and privacy. With over 359,000 stars on GitHub, it has rapidly become one of the most popular self-hosted AI assistant projects in the world.

Unlike cloud-dependent AI assistants that lock you into vendor ecosystems, OpenClaw is local-first: the Gateway runs on your machine, your data stays on your machine, and you choose which model providers to trust. It supports 20+ messaging channels including WhatsApp, Telegram, Slack, Discord, Signal, iMessage, Microsoft Teams, Matrix, WeChat, and more.

![OpenClaw Architecture Overview](/assets/img/diagrams/openclaw/openclaw-architecture.svg)

## Understanding the Architecture

The architecture diagram above illustrates the core components of OpenClaw and how they interact. Let us break down each component:

**Gateway (Control Plane)**
The Gateway is the central control plane that manages all sessions, channels, tools, and events. It runs as a local daemon on your machine (via launchd on macOS or systemd on Linux) and stays running in the background. The Gateway handles routing messages from any connected channel to the appropriate agent session, manages tool execution, and coordinates all the moving parts.

**Channel Router**
The Channel Router is responsible for receiving messages from 20+ messaging platforms and normalizing them into a unified format. Whether a message comes from WhatsApp, Telegram, Slack, or Discord, the Channel Router ensures it reaches the Gateway in a consistent structure. This abstraction means you can add or remove channels without changing any core logic.

**Session Manager**
The Session Manager maintains conversation state across all your interactions. Each channel or conversation gets its own isolated session, ensuring context is preserved between messages. Sessions can be reset, compacted, or branched as needed, giving you fine-grained control over conversation history.

**Agent Router**
The Agent Router enables multi-agent routing, allowing you to direct different channels, accounts, or peers to isolated agents. Each agent has its own workspace and session configuration. This means you could have one agent handling your WhatsApp messages with a professional tone, while another handles Discord with a more casual personality.

**Security Layer**
Security is a first-class concern in OpenClaw. The Security Layer enforces DM pairing policies, manages allowlists, and provides Docker sandboxing for non-main sessions. By default, unknown senders receive a pairing code rather than having their messages processed, preventing unauthorized access to your assistant.

**Tool Engine**
The Tool Engine manages the execution of built-in and custom tools. OpenClaw ships with first-class tools for browser automation, canvas rendering, cron scheduling, session management, and Discord/Slack actions. The Skills system extends this further with community-maintained capabilities from ClawHub.

## Multi-Channel Messaging

![OpenClaw Multi-Channel Messaging](/assets/img/diagrams/openclaw/openclaw-channels.svg)

### Understanding Multi-Channel Support

The multi-channel diagram above shows how OpenClaw unifies communication across 20+ platforms into a single AI assistant experience. Here is how it works:

**Channel Integration Architecture**
Each messaging platform has its own channel plugin that handles the platform-specific API, authentication, and message format. The Channel Router normalizes all incoming messages into a unified internal format, so the AI agent sees a consistent interface regardless of which platform the user is on.

**Supported Channels**
OpenClaw supports an impressive range of messaging platforms:

| Category | Channels |
|----------|----------|
| Messaging | WhatsApp, Telegram, Signal, iMessage, BlueBubbles |
| Collaboration | Slack, Discord, Microsoft Teams, Google Chat |
| Social | IRC, Matrix, Nostr, Twitch |
| Regional | WeChat, QQ, Zalo, LINE, Feishu |
| Enterprise | Mattermost, Nextcloud Talk, Synology Chat |
| Web | WebChat (built-in), Tlon |

**Message Flow**
When a user sends a message on any channel, the flow is:
1. The channel plugin receives the message from the platform API
2. The Channel Router normalizes it and passes it to the Gateway
3. The Gateway routes it to the appropriate agent session
4. The agent processes the message using the configured AI model
5. The response is routed back through the same channel to the user

This means you can start a conversation on WhatsApp in the morning, continue it on Telegram in the afternoon, and pick it up on Slack in the evening -- all with the same AI assistant maintaining full context.

## Key Features

![OpenClaw Key Features](/assets/img/diagrams/openclaw/openclaw-features.svg)

### Understanding Key Features

The features diagram above highlights the eight core capabilities that set OpenClaw apart from other AI assistant projects:

**Local-First Gateway**
The Gateway runs entirely on your hardware. No cloud dependency, no vendor lock-in, no data leaving your machine unless you explicitly configure it. This is the fundamental design principle that makes OpenClaw different from hosted AI services.

**20+ Channel Support**
As covered in the multi-channel section, OpenClaw connects to more messaging platforms than any other self-hosted AI assistant. The plugin architecture makes it straightforward to add new channels.

**Voice Wake + Talk Mode**
OpenClaw supports voice interaction through wake words on macOS and iOS, and continuous voice on Android. It integrates with ElevenLabs for high-quality text-to-speech, with system TTS as a fallback. This transforms OpenClaw from a text-based assistant into a true voice companion.

**Live Canvas**
The Canvas feature provides an agent-driven visual workspace. Using the A2UI (Agent-to-UI) protocol, the agent can render interactive visual elements, dashboards, and data visualizations directly in the companion app. This goes beyond text responses to deliver rich, visual experiences.

**Skills and Plugins (ClawHub)**
OpenClaw has an extensive plugin and skill system. Skills are self-contained capability modules that can be installed from ClawHub, the community marketplace. Built-in skills include browser automation, canvas rendering, cron scheduling, and more. The plugin API allows third-party developers to extend OpenClaw with custom capabilities.

**Security Defaults**
Security is not an afterthought in OpenClaw -- it is a core design principle. DM pairing, allowlists, Docker sandboxing, and the `openclaw doctor` security audit tool all work together to keep your assistant safe by default while still allowing powerful workflows when explicitly configured.

**Multi-Agent Routing**
Multiple agents can run simultaneously, each with its own workspace, model configuration, and channel assignments. This enables use cases like having a professional agent for work channels and a casual agent for personal messaging.

**MCP Support**
Model Context Protocol (MCP) support is provided through mcporter, keeping MCP integration flexible and decoupled from the core runtime. You can add or change MCP servers without restarting the Gateway.

## Security Model

![OpenClaw Security Model](/assets/img/diagrams/openclaw/openclaw-security.svg)

### Understanding the Security Model

The security diagram above illustrates how OpenClaw handles inbound messages and protects your system:

**DM Policy System**
OpenClaw treats all inbound direct messages as untrusted input by default. The DM Policy system offers two modes:

- **Pairing Mode (default)**: Unknown senders receive a short pairing code. The bot does not process their message until the code is approved via `openclaw pairing approve <channel> <code>`. This prevents random users from accessing your AI assistant.
- **Open Mode**: Explicitly allows public inbound DMs. Requires setting `dmPolicy="open"` and including `"*"` in the channel allowlist. Use this only when you intentionally want public access.

**Docker Sandboxing**
For group and channel interactions, OpenClaw can run non-main sessions inside per-session Docker sandboxes. The typical sandbox default allows safe tools (bash, process, read, write, edit, sessions) while denying dangerous ones (browser, canvas, nodes, cron, discord, gateway). This means even if an agent is compromised in a group chat, it cannot escape its sandbox.

**Security Audit with openclaw doctor**
The `openclaw doctor` command performs a comprehensive security audit of your configuration, surfacing risky DM policies, misconfigured sandbox settings, and other potential vulnerabilities. It can also automatically fix common issues with `openclaw doctor --fix`.

**Main Session Access**
The main session (your personal session) has full host access by default, since it is just you interacting with your own machine. This is the intended design: full power for the owner, sandboxed isolation for everyone else.

## Installation

### Quick Install (Recommended)

OpenClaw requires Node 24 (recommended) or Node 22.16+.

```bash
# Install globally via npm
npm install -g openclaw@latest

# Or via pnpm
pnpm add -g openclaw@latest

# Run the onboarding wizard
openclaw onboard --install-daemon
```

The `openclaw onboard` command guides you step by step through:
- Gateway setup and configuration
- Workspace initialization
- Channel configuration (WhatsApp, Telegram, etc.)
- Model provider selection (OpenAI, Anthropic, etc.)
- Skill installation

### From Source (Development)

```bash
git clone https://github.com/openclaw/openclaw.git
cd openclaw

pnpm install
pnpm ui:build
pnpm build

pnpm openclaw onboard --install-daemon

# Dev loop with auto-reload
pnpm gateway:watch
```

### Docker

```bash
# Using Docker Compose
docker compose up -d

# Or build from Dockerfile
docker build -t openclaw .
```

## Configuration

Minimal configuration in `~/.openclaw/openclaw.json`:

```json5
{
  agent: {
    model: "<provider>/<model-id>",
  },
}
```

For example, to use OpenAI's GPT-4o:

```json5
{
  agent: {
    model: "openai/gpt-4o",
  },
}
```

Or Anthropic's Claude:

```json5
{
  agent: {
    model: "anthropic/claude-sonnet-4-20250514",
  },
}
```

Full configuration reference is available at [docs.openclaw.ai/gateway/configuration](https://docs.openclaw.ai/gateway/configuration).

## Usage

### Sending Messages

```bash
# Send a message to a contact
openclaw message send --to +1234567890 --message "Hello from OpenClaw"

# Talk to the assistant with high thinking
openclaw agent --message "Ship checklist" --thinking high
```

### Chat Commands

Within any connected channel, you can use these commands:

| Command | Description |
|---------|-------------|
| `/status` | Show current session status |
| `/new` | Start a new session |
| `/reset` | Reset the current session |
| `/compact` | Compact conversation history |
| `/think <level>` | Set thinking level (low/medium/high) |
| `/verbose on\|off` | Toggle verbose output |
| `/trace on\|off` | Toggle trace mode |
| `/usage off\|tokens\|full` | Control usage display |

### Voice Commands

On macOS with the companion app:
- Say the wake word to activate voice input
- Use push-to-talk overlay for manual activation
- Responses are spoken back using ElevenLabs TTS

On Android:
- Continuous voice mode with Connect/Chat/Voice tabs
- Camera and screen capture capabilities
- Canvas surface for visual responses

## Companion Apps

### macOS (OpenClaw.app)

The macOS companion app provides:
- Menu bar control for the Gateway and health monitoring
- Voice Wake + push-to-talk overlay
- WebChat + debug tools
- Remote gateway control over SSH

### iOS Node

- Pairs as a node over the Gateway WebSocket
- Voice trigger forwarding + Canvas surface
- Controlled via `openclaw nodes` commands

### Android Node

- Pairs as a WebSocket node via device pairing
- Exposes Connect/Chat/Voice tabs plus Canvas, Camera, Screen capture
- Android device command families for full device control

## Skills and Plugins

OpenClaw has a rich skills and plugins ecosystem:

- **Workspace root**: `~/.openclaw/workspace`
- **Injected prompt files**: `AGENTS.md`, `SOUL.md`, `TOOLS.md`
- **Skills directory**: `~/.openclaw/workspace/skills/<skill>/SKILL.md`
- **Community marketplace**: [ClawHub](https://clawhub.com)

Skills are self-contained modules that extend the assistant's capabilities. Each skill has a `SKILL.md` file that defines its interface, and they can be installed, updated, and removed independently.

## Development Channels

| Channel | Description |
|---------|-------------|
| **stable** | Tagged releases, npm `latest` |
| **beta** | Prerelease tags, npm `beta` |
| **dev** | Moving head of `main`, npm `dev` |

Switch channels: `openclaw update --channel stable|beta|dev`

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Gateway not starting | Run `openclaw doctor` to check configuration |
| DM pairing not working | Check `dmPolicy` setting in config |
| Model not responding | Verify API key and model ID in config |
| Channel not connecting | Check channel-specific docs at docs.openclaw.ai |
| Permission errors on macOS | See macOS Permissions docs |
| Docker sandbox issues | Run `openclaw doctor --fix` |

## Comparison with Alternatives

| Feature | OpenClaw | ChatGPT | Claude | Local LLMs |
|---------|----------|---------|--------|------------|
| Self-hosted | Yes | No | No | Yes |
| Multi-channel | 20+ | Web only | Web only | Varies |
| Voice input | Yes | Limited | No | Varies |
| Local data | Yes | No | No | Yes |
| Plugin system | ClawHub | Plugins | No | Varies |
| Docker sandbox | Yes | N/A | N/A | Varies |
| Open source | MIT | No | No | Varies |

## Conclusion

OpenClaw represents a paradigm shift in personal AI assistants. By running locally on your own devices, it gives you the power of modern AI models without surrendering your data to cloud services. With support for 20+ messaging channels, voice wake words, live canvas, multi-agent routing, and a robust security model, OpenClaw is the most comprehensive self-hosted AI assistant available today.

The project's 359K+ stars on GitHub reflect the community's enthusiasm for a privacy-first, locally-run AI assistant that actually works across all the platforms people already use. Whether you want a personal assistant on WhatsApp, a work bot on Slack, or a voice companion on your phone, OpenClaw delivers.

**Links:**
- [GitHub Repository](https://github.com/openclaw/openclaw)
- [Documentation](https://docs.openclaw.ai)
- [ClawHub Skills Marketplace](https://clawhub.com)
- [Discord Community](https://discord.gg/clawd)
