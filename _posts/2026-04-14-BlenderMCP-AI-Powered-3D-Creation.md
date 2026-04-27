---
layout: post
title: "BlenderMCP: AI-Powered 3D Content Creation"
description: "Connect Claude AI to Blender through the Model Context Protocol for prompt-assisted 3D modeling, scene creation, and manipulation with Poly Haven, Hyper3D, and Sketchfab integrations."
date: 2026-04-14
header-img: "img/post-bg.jpg"
permalink: /BlenderMCP-AI-Powered-3D-Creation/
featured-img: ai-coding-frameworks/ai-coding-frameworks
tags:
  - Open Source
  - Python
  - AI
  - 3D Graphics
  - Blender
  - MCP
author: "PyShine"
---

# BlenderMCP: AI-Powered 3D Content Creation

BlenderMCP is a groundbreaking integration that connects Blender to Claude AI through the Model Context Protocol (MCP), enabling natural language control of 3D modeling, scene creation, and manipulation. This powerful bridge transforms how artists and developers interact with Blender, making 3D content creation more accessible than ever.

## What is BlenderMCP?

BlenderMCP implements the Model Context Protocol to create a seamless communication channel between Claude AI and Blender 3D. With this integration, you can simply describe what you want to create in natural language, and Claude will execute the necessary commands in Blender to bring your vision to life.

The system supports multiple asset integrations including Poly Haven for HDRIs, textures, and models; Hyper3D Rodin for AI-generated 3D models; Sketchfab for searching and downloading models; and Hunyuan3D for text-to-3D generation. This comprehensive ecosystem makes it possible to create complex 3D scenes without manually modeling every element.

![BlenderMCP Architecture](/assets/img/diagrams/blender-mcp-architecture.svg)

### Understanding the Architecture

The architecture diagram above illustrates the complete data flow and component interactions within BlenderMCP. Let's examine each component in detail:

**Claude AI Layer**

The Claude AI layer serves as the intelligent orchestrator of the entire system. It provides natural language understanding capabilities that interpret user requests and translate them into actionable commands. The context awareness feature allows Claude to maintain awareness of the current Blender scene state, including existing objects, materials, and scene hierarchy. This contextual understanding enables intelligent decision-making when creating or modifying 3D content.

The tool orchestration component within Claude manages the execution sequence of various tools, ensuring that operations are performed in the correct order. For example, when creating a scene with multiple objects, Claude will first establish the environment lighting using an HDRI, then place the ground plane, and finally add the individual objects with appropriate materials.

**MCP Server Layer**

The MCP Server acts as the protocol handler and command router between Claude and Blender. Built using the FastMCP framework, it implements the Model Context Protocol specification, providing a standardized interface for tool discovery and execution. The server maintains a comprehensive registry of available tools, each with well-defined input schemas and output formats.

The command router within the MCP server handles the translation of Claude's tool calls into JSON-formatted commands that can be understood by the Blender addon. This layer also manages connection state, handles timeouts, and implements retry logic for robust communication. The JSON-based protocol ensures type safety and enables structured error handling throughout the system.

**Blender Addon Layer**

The Blender addon runs as a socket server within Blender, listening for incoming commands from the MCP server. It implements a command executor that translates JSON commands into actual Blender Python API calls. The addon maintains the scene state and provides real-time feedback about object positions, materials, and scene hierarchy.

The socket server implementation uses TCP for reliable communication, with configurable host and port settings. The addon also handles viewport screenshot capture, enabling Claude to "see" the current state of the Blender scene for better context-aware decision making.

**Integration Ecosystem**

The integration layer connects BlenderMCP to external asset sources:

- **Poly Haven**: Provides access to thousands of free, high-quality HDRIs, textures, and 3D models. The integration supports automatic downloading and importing with configurable resolution settings.

- **Hyper3D Rodin**: An AI-powered 3D generation service that creates models from text descriptions or reference images. The integration handles the asynchronous generation workflow, including job submission, status polling, and asset importing.

- **Sketchfab**: The world's largest platform for 3D content, offering millions of downloadable models. The integration supports search, preview thumbnails, and automatic scaling to target dimensions.

- **Hunyuan3D**: Tencent's text-to-3D generation model, supporting both text prompts and image inputs for creating custom 3D assets with built-in materials.

## Key Features

| Feature | Description |
|---------|-------------|
| Two-way Communication | Real-time bidirectional communication between Claude and Blender |
| Object Manipulation | Create, modify, and delete 3D objects using natural language |
| Material Control | Apply and modify materials, colors, and textures |
| Scene Inspection | Get detailed information about the current Blender scene |
| Code Execution | Run arbitrary Python code in Blender from Claude |
| Poly Haven Integration | Download HDRIs, textures, and models from Poly Haven API |
| Hyper3D Rodin | Generate 3D models using AI from text or images |
| Sketchfab Integration | Search and download models from Sketchfab |
| Hunyuan3D Support | Text-to-3D generation with built-in materials |
| Viewport Screenshots | Claude can see the current Blender viewport |

## Installation

### Prerequisites

- Blender 3.0 or newer
- Python 3.10 or newer
- uv package manager

**Install uv on Mac:**
```bash
brew install uv
```

**Install uv on Windows:**
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Add uv to the user path in Windows:
```powershell
$localBin = "$env:USERPROFILE\.local\bin"
$userPath = [Environment]::GetEnvironmentVariable("Path", "User")
[Environment]::SetEnvironmentVariable("Path", "$userPath;$localBin", "User")
```

### Claude Desktop Integration

Go to Claude > Settings > Developer > Edit Config > `claude_desktop_config.json`:

```json
{
    "mcpServers": {
        "blender": {
            "command": "uvx",
            "args": [
                "blender-mcp"
            ]
        }
    }
}
```

### Cursor Integration

For Mac users, go to Settings > MCP and paste:
```json
{
    "mcpServers": {
        "blender": {
            "command": "uvx",
            "args": [
                "blender-mcp"
            ]
        }
    }
}
```

For Windows users:
```json
{
    "mcpServers": {
        "blender": {
            "command": "cmd",
            "args": [
                "/c",
                "uvx",
                "blender-mcp"
            ]
        }
    }
}
```

### Installing the Blender Addon

1. Download the `addon.py` file from the repository
2. Open Blender
3. Go to Edit > Preferences > Add-ons
4. Click "Install..." and select the `addon.py` file
5. Enable the addon by checking the box next to "Interface: Blender MCP"

## Usage

### Starting the Connection

1. In Blender, go to the 3D View sidebar (press N if not visible)
2. Find the "BlenderMCP" tab
3. Turn on the Poly Haven checkbox if you want assets from their API (optional)
4. Click "Connect to Claude"
5. Make sure the MCP server is running in your terminal

### Example Commands

Here are some examples of what you can ask Claude to do:

- "Create a low poly scene in a dungeon, with a dragon guarding a pot of gold"
- "Create a beach vibe using HDRIs, textures, and models like rocks and vegetation from Poly Haven"
- "Generate a 3D model of a garden gnome through Hyper3D"
- "Get information about the current scene, and make a threejs sketch from it"
- "Make this car red and metallic"
- "Create a sphere and place it above the cube"
- "Make the lighting like a studio"
- "Point the camera at the scene, and make it isometric"

## Asset Creation Strategy

BlenderMCP implements an intelligent asset creation strategy that prioritizes the best source for each type of content:

1. **For specific existing objects**: First try Sketchfab, then Poly Haven
2. **For generic objects/furniture**: First try Poly Haven, then Sketchfab
3. **For custom or unique items**: Use Hyper3D Rodin or Hunyuan3D
4. **For environment lighting**: Use Poly Haven HDRIs
5. **For materials/textures**: Use Poly Haven textures

The system automatically falls back to scripting when:
- All integrations are disabled
- A simple primitive is explicitly requested
- No suitable asset exists in any library
- AI generation failed to produce the desired asset

## Environment Variables

Configure the Blender connection using environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `BLENDER_HOST` | Host address for Blender socket server | localhost |
| `BLENDER_PORT` | Port number for Blender socket server | 9876 |

Example:
```bash
export BLENDER_HOST='host.docker.internal'
export BLENDER_PORT=9876
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Connection issues | Ensure Blender addon server is running and MCP server is configured. Sometimes the first command won't go through but subsequent ones work. |
| Timeout errors | Simplify requests or break them into smaller steps |
| Poly Haven integration issues | Claude behavior can be erratic; try rephrasing requests |
| Persistent connection errors | Restart both Claude and the Blender server |

## Security Considerations

The `execute_blender_code` tool allows running arbitrary Python code in Blender, which is powerful but potentially dangerous. Always save your work before using it. Poly Haven requires downloading models, textures, and HDRI images - disable it in the checkbox if not needed.

## Telemetry Control

BlenderMCP collects anonymous usage data to improve the tool. Control telemetry in two ways:

1. **In Blender**: Go to Edit > Preferences > Add-ons > Blender MCP and uncheck the telemetry consent checkbox
2. **Environment Variable**: Completely disable all telemetry:
```bash
DISABLE_TELEMETRY=true uvx blender-mcp
```

## Conclusion

BlenderMCP represents a significant advancement in AI-assisted 3D content creation. By bridging Claude's natural language understanding with Blender's powerful 3D capabilities, it democratizes 3D modeling and makes scene creation accessible to everyone, regardless of their technical expertise.

The integration with Poly Haven, Hyper3D Rodin, Sketchfab, and Hunyuan3D provides a comprehensive ecosystem for sourcing and generating 3D assets. Whether you're a professional 3D artist looking to speed up your workflow or a beginner wanting to explore 3D creation, BlenderMCP offers an intuitive and powerful solution.
