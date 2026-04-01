---
layout: post
title: "MiroFish - AI Swarm Intelligence Engine for Predicting the Future"
date: 2026-04-01
categories: [AI, Machine Learning, Multi-Agent Systems, Python]
featured-img: 2026-apr/mirofish
description: "Explore MiroFish, an open-source multi-agent AI prediction engine that simulates thousands of intelligent agents to predict future outcomes. Learn how to build digital sandboxes for scenario simulation."
---

## MiroFish - AI Swarm Intelligence Engine for Predicting the Future

What if you could simulate thousands of AI agents with unique personalities, memories, and behaviors to predict future outcomes? **MiroFish** makes this possible - it's an open-source multi-agent prediction engine that creates high-fidelity digital worlds for scenario simulation.

## What is MiroFish?

MiroFish is a next-generation AI prediction engine powered by multi-agent technology. By extracting seed information from the real world (such as breaking news, policy drafts, or financial signals), it automatically constructs a parallel digital world where thousands of intelligent agents interact and evolve.

### Key Capabilities

**Upload seed materials** (data analysis reports, news articles, or even novels) and describe your prediction requirements in natural language. MiroFish returns:

- A detailed prediction report
- A deeply interactive high-fidelity digital world

## How It Works

### 5-Step Workflow

| Step | Description |
|------|-------------|
| **1. Graph Building** | Seed extraction, memory injection, GraphRAG construction |
| **2. Environment Setup** | Entity relationship extraction, persona generation, agent configuration |
| **3. Simulation** | Dual-platform parallel simulation (Twitter + Reddit), dynamic memory updates |
| **4. Report Generation** | ReportAgent with rich toolset for deep analysis |
| **5. Deep Interaction** | Chat with any agent in the simulated world |

### Functional Flow Diagram

![MiroFish Functional Flow]({{ site.baseurl }}/assets/img/posts/2026-apr/mirofish-flow-diagram.svg)

The diagram above illustrates the complete data flow through MiroFish's 5-step workflow:

1. **Input Layer** - Users upload documents (news, reports, novels) and describe prediction requirements
2. **Graph Building** - Text processing, knowledge graph construction via Zep Cloud, and entity memory injection
3. **Environment Setup** - Entity extraction, persona generation, and agent profile configuration
4. **Simulation** - OASIS engine runs Twitter/Reddit simulations with multi-agent interactions
5. **Output** - Prediction reports and interactive chat with simulated agents

## Technical Architecture

### Backend (Python + Flask)

```
backend/
├── app/
│   ├── api/           # REST API endpoints
│   ├── models/        # Data models
│   ├── services/      # Core services
│   │   ├── simulation_manager.py    # OASIS simulation management
│   │   ├── graph_builder.py         # Knowledge graph construction
│   │   ├── report_agent.py          # Report generation
│   │   └── oasis_profile_generator.py # Agent persona creation
│   └── utils/         # Utilities
```

### Frontend (Vue.js + Vite)

```
frontend/
├── src/
│   ├── components/    # Vue components
│   │   ├── Step1GraphBuild.vue
│   │   ├── Step2EnvSetup.vue
│   │   ├── Step3Simulation.vue
│   │   ├── Step4Report.vue
│   │   └── Step5Interaction.vue
│   └── views/         # Page views
```

### Key Dependencies

| Package | Purpose |
|---------|---------|
| `camel-oasis` | Social media simulation engine |
| `camel-ai` | Multi-agent framework |
| `zep-cloud` | Long-term memory management |
| `openai` | LLM API integration |
| `flask` | Backend web framework |

## Use Cases

### 1. Public Opinion Prediction

Upload news articles or social media data to simulate how public sentiment might evolve. The system can predict:

- Viral content spread patterns
- Crisis communication outcomes
- Brand reputation trajectories

### 2. Financial Market Simulation

Feed financial reports and market signals to create agent-based market simulations:

- Investor behavior modeling
- Market sentiment analysis
- Risk scenario testing

### 3. Creative Writing

Upload the first 80 chapters of a novel (like Dream of the Red Chamber) and let MiroFish predict the lost ending based on character personalities and plot dynamics.

### 4. Policy Impact Assessment

Test policy drafts in a zero-risk digital sandbox:

- Public reaction simulation
- Stakeholder behavior prediction
- Unintended consequence discovery

## Quick Start Guide

### Prerequisites

| Tool | Version | Purpose |
|------|---------|---------|
| Node.js | 18+ | Frontend runtime |
| Python | 3.11-3.12 | Backend runtime |
| uv | Latest | Python package manager |

### Installation

```bash
# Clone the repository
git clone https://github.com/666ghj/MiroFish.git
cd MiroFish

# Copy environment configuration
cp .env.example .env

# Configure API keys
# LLM_API_KEY - Your LLM API key (OpenAI SDK compatible)
# ZEP_API_KEY - Zep Cloud API key for memory management
```

### Environment Variables

```env
# LLM API Configuration (supports any OpenAI SDK compatible API)
LLM_API_KEY=your_api_key
LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
LLM_MODEL_NAME=qwen-plus

# Zep Cloud Configuration (free tier available)
ZEP_API_KEY=your_zep_api_key
```

### Run the Application

```bash
# Install all dependencies
npm run setup:all

# Start both frontend and backend
npm run dev
```

**Service URLs:**

- Frontend: `http://localhost:3000`
- Backend API: `http://localhost:5001`

### Docker Deployment

```bash
# Configure environment
cp .env.example .env

# Start with Docker Compose
docker compose up -d
```

## Core Concepts

### Agent Personas

Each agent in MiroFish has:

- **Independent personality** - Unique traits and behavioral patterns
- **Long-term memory** - Persistent context through Zep Cloud
- **Behavioral logic** - Consistent decision-making framework
- **Social relationships** - Connections with other agents

### GraphRAG Integration

MiroFish uses Graph-based Retrieval Augmented Generation:

- Extracts entities from seed materials
- Builds relationship graphs
- Enables context-aware agent interactions
- Supports temporal memory updates

### Dual-Platform Simulation

Simulates social dynamics across:

- **Twitter-like platform** - Short-form, viral content
- **Reddit-like platform** - Long-form, community discussions

## Example: Predicting Novel Endings

MiroFish can analyze literary works and predict plausible continuations:

```python
# Upload first 80 chapters of Dream of the Red Chamber
# System extracts:
# - Character relationships
# - Plot dynamics
# - Thematic elements
# - Character motivations

# Agents simulate character behaviors
# Generate prediction report with:
# - Multiple possible endings
# - Character arc conclusions
# - Thematic resolution
```

## Project Structure Deep Dive

### Simulation Manager

```python
class SimulationStatus(str, Enum):
    CREATED = "created"
    PREPARING = "preparing"
    READY = "ready"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    COMPLETED = "completed"
    FAILED = "failed"

class PlatformType(str, Enum):
    TWITTER = "twitter"
    REDDIT = "reddit"
```

### Agent Profile Generation

The `OasisProfileGenerator` creates detailed agent personas:

- Demographics and background
- Personality traits (Big Five model)
- Interests and expertise
- Social connections
- Communication style

## Why MiroFish Matters

### For Decision Makers

- **Zero-risk testing** - Try policies in simulation before real-world implementation
- **Scenario exploration** - Test multiple what-if scenarios
- **Stakeholder mapping** - Understand how different groups might react

### For Researchers

- **Multi-agent systems** - Study emergent behaviors
- **Social simulation** - Model complex social dynamics
- **LLM applications** - Explore large-scale agent coordination

### For Developers

- **Open source** - Full code access for customization
- **Modular architecture** - Easy to extend and modify
- **Modern stack** - Vue.js frontend, Flask backend, Python agents

## Acknowledgments

MiroFish is incubated by **Shanda Group** and powered by **OASIS (Open Agent Social Interaction Simulations)** from the CAMEL-AI team.

## Resources

- **GitHub Repository**: [https://github.com/666ghj/MiroFish](https://github.com/666ghj/MiroFish)
- **Live Demo**: [mirofish-live-demo](https://666ghj.github.io/mirofish-demo/)
- **Discord**: [Join Community](http://discord.gg/ePf5aPaHnA)
- **Twitter**: [@mirofish_ai](https://x.com/mirofish_ai)

## Conclusion

MiroFish represents a fascinating convergence of multi-agent systems, large language models, and social simulation. Whether you're a researcher studying emergent behaviors, a decision-maker testing scenarios, or a developer exploring AI applications, MiroFish offers a powerful platform for predicting the future through simulation.

The ability to create thousands of unique agents with persistent memories and let them interact in simulated social environments opens up possibilities that were previously confined to science fiction. As LLMs continue to improve, systems like MiroFish will become increasingly accurate at modeling complex social dynamics.

**Try it yourself** - clone the repository, set up your API keys, and start simulating your own scenarios!

---

*Have questions or want to share your MiroFish experiments? Join the [Discord community](http://discord.gg/ePf5aPaHnA) or check out the [GitHub repository](https://github.com/666ghj/MiroFish) for more details.*
