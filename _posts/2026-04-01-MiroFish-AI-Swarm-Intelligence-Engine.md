---
layout: post
title: "MiroFish - AI Swarm Intelligence Engine for Predicting the Future"
date: 2026-04-01
categories: [AI, Machine Learning, Multi-Agent Systems, Python]
featured-img: ai-coding-frameworks/ai-coding-frameworks
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

### Complete Workflow

| Step | Component | Description |
|------|-----------|-------------|
| **0** | Text Processing | Document parsing, preprocessing, and chunking |
| **1** | Ontology Generation | LLM-based entity and relationship type definition |
| **2** | Graph Building | Zep Cloud knowledge graph construction with GraphRAG |
| **3** | Entity Extraction | Filter and enrich entities from knowledge graph |
| **4** | Profile Generation | LLM + Zep search for detailed agent personas |
| **5** | Simulation Config | LLM-based simulation parameter generation |
| **6** | OASIS Simulation | Dual-platform parallel simulation (Twitter + Reddit) |
| **7** | Report Generation | ReACT-based ReportAgent with Zep tools |
| **8** | Deep Interaction | Chat with simulated agents and ReportAgent |

### Functional Flow Diagrams

The MiroFish workflow is divided into two main phases for better visualization:

#### Part 1: Data Preparation & Knowledge Graph Construction

![MiroFish Flow Part 1 - Data Preparation]({{ site.baseurl }}/assets/img/posts/2026-apr/mirofish-flow-part1.svg)

**Phase 1 Components:**

1. **Input Layer** - Users upload documents (news, reports, novels) and describe prediction requirements in natural language
2. **Text Processor** - FileParser extracts text, preprocessing cleans and normalizes, chunking splits into 500-character segments
3. **Ontology Generator** - LLM analyzes content to define 10 entity types (Person, Organization, etc.) and 6-10 edge types
4. **Zep Cloud** - Creates unique graph ID, sets ontology, uploads episodes in batches, performs GraphRAG extraction
5. **Entity Reader** - Filters entities by defined types, enriches with edges and relationships, outputs filtered entities

#### Part 2: Simulation & Output Generation

![MiroFish Flow Part 2 - Simulation & Output]({{ site.baseurl }}/assets/img/posts/2026-apr/mirofish-flow-part2.svg)

**Phase 2 Components:**

6. **Profile Generator** - Uses Zep hybrid search + LLM to generate detailed personas (age, MBTI, country, interests)
7. **Config Generator** - LLM generates simulation parameters (rounds, active hours, posting frequency)
8. **OASIS Engine** - CAMEL-AI powered social simulation on Twitter and Reddit platforms
9. **Report Agent** - ReACT pattern with InsightForge, Panorama, and Interview tools
10. **Chat Interface** - Interact with simulated agents or ask ReportAgent questions

## Technical Architecture

### Backend Services (Python + Flask)

```
backend/
├── app/
│   ├── api/                    # REST API endpoints
│   │   ├── graph.py           # Graph building API
│   │   ├── simulation.py      # Simulation management API
│   │   └── report.py          # Report generation API
│   ├── models/                 # Data models
│   │   ├── project.py         # Project model
│   │   └── task.py            # Task management
│   ├── services/               # Core services
│   │   ├── text_processor.py          # Document parsing & chunking
│   │   ├── ontology_generator.py      # LLM-based ontology definition
│   │   ├── graph_builder.py           # Zep Cloud graph construction
│   │   ├── zep_entity_reader.py       # Entity extraction & filtering
│   │   ├── oasis_profile_generator.py # Agent persona generation
│   │   ├── simulation_config_generator.py # LLM-based config
│   │   ├── simulation_manager.py      # Simulation orchestration
│   │   ├── simulation_runner.py       # Script execution
│   │   ├── report_agent.py            # ReACT report generation
│   │   └── zep_tools.py               # Zep search tools
│   └── utils/                  # Utilities
│       ├── file_parser.py      # Multi-format file parsing
│       ├── llm_client.py       # LLM API client
│       └── logger.py           # Logging utilities
```

### Frontend Components (Vue.js + Vite)

```
frontend/
├── src/
│   ├── components/             # Vue components
│   │   ├── Step1GraphBuild.vue    # Document upload & graph building
│   │   ├── Step2EnvSetup.vue      # Entity extraction & profile generation
│   │   ├── Step3Simulation.vue    # Simulation configuration & execution
│   │   ├── Step4Report.vue        # Report viewing & download
│   │   └── Step5Interaction.vue   # Chat interface
│   ├── views/                  # Page views
│   │   ├── MainView.vue        # Main workflow view
│   │   ├── SimulationView.vue  # Simulation monitoring
│   │   └── InteractionView.vue # Chat interface
│   └── api/                    # API clients
│       ├── graph.js            # Graph API
│       ├── simulation.js       # Simulation API
│       └── report.js           # Report API
```

### Key Dependencies

| Package | Purpose | Version |
|---------|---------|---------|
| `camel-oasis` | Social media simulation engine | Latest |
| `camel-ai` | Multi-agent framework | Latest |
| `zep-cloud` | Long-term memory & knowledge graph | Latest |
| `openai` | LLM API integration | Latest |
| `langchain` | ReACT agent framework | Latest |
| `flask` | Backend web framework | 3.x |
| `vue.js` | Frontend framework | 3.x |

## Core Components Deep Dive

### 1. Text Processor (`text_processor.py`)

Handles document ingestion and preparation:

- **FileParser**: Extracts text from PDF, DOCX, TXT, MD files
- **Preprocessing**: Removes excess whitespace, normalizes line endings
- **Chunking**: Splits text into 500-character chunks with 50-character overlap

```python
class TextProcessor:
    @staticmethod
    def split_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        return split_text_into_chunks(text, chunk_size, overlap)
```

### 2. Ontology Generator (`ontology_generator.py`)

LLM-powered entity and relationship type definition:

- **Entity Types**: Exactly 10 types including Person and Organization fallbacks
- **Edge Types**: 6-10 relationship types (WORKS_FOR, STUDIES_AT, etc.)
- **Validation**: Ensures Zep API compatibility (max 10 types each)

```python
class OntologyGenerator:
    def generate(self, document_texts: List[str], simulation_requirement: str) -> Dict[str, Any]:
        # Returns entity_types, edge_types, analysis_summary
```

### 3. Graph Builder (`graph_builder.py`)

Zep Cloud knowledge graph construction:

- **Create Graph**: Generates unique graph ID (`mirofish_<uuid>`)
- **Set Ontology**: Dynamically creates Pydantic models for entities/edges
- **Add Episodes**: Batch uploads text chunks (default 3 per batch)
- **Wait for Processing**: Monitors episode processing status
- **GraphRAG**: Automatic entity and relationship extraction

```python
class GraphBuilderService:
    def build_graph_async(self, text: str, ontology: Dict, ...) -> str:
        # Returns task_id for async processing
```

### 4. Entity Reader (`zep_entity_reader.py`)

Extracts and enriches entities from knowledge graph:

- **Filter by Type**: Only includes entities matching defined types
- **Enrich with Edges**: Adds related facts and relationships
- **Pagination**: Handles large graphs with paging

```python
class ZepEntityReader:
    def filter_defined_entities(self, graph_id: str, defined_entity_types: List[str]) -> FilteredEntities:
        # Returns filtered entities with enriched context
```

### 5. Profile Generator (`oasis_profile_generator.py`)

Creates detailed agent personas:

- **Zep Hybrid Search**: Searches nodes and edges for context
- **LLM Persona**: Generates age, MBTI, country, profession, interests
- **Individual vs Group**: Different prompts for persons vs organizations
- **Output Formats**: Twitter CSV and Reddit JSON

```python
class OasisProfileGenerator:
    def generate_profile_from_entity(self, entity: EntityNode, user_id: int, use_llm: bool = True) -> OasisAgentProfile:
        # Returns detailed agent profile
```

### 6. Simulation Manager (`simulation_manager.py`)

Orchestrates the entire simulation:

- **State Management**: Tracks simulation status (created, preparing, running, completed)
- **Platform Support**: Twitter and Reddit dual-platform simulation
- **Progress Callbacks**: Real-time progress updates

```python
class SimulationStatus(str, Enum):
    CREATED = "created"
    PREPARING = "preparing"
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
```

### 7. Report Agent (`report_agent.py`)

ReACT-based report generation with Zep tools:

- **InsightForge**: Deep insight retrieval with multi-dimensional analysis
- **Panorama Search**: Broad overview of simulation results
- **Quick Search**: Simple fact lookup
- **Interview Agents**: Real interviews with simulated agents

```python
class ReportAgent:
    def generate_report(self, simulation_id: str, graph_id: str, simulation_requirement: str) -> Report:
        # Returns structured prediction report
```

## Use Cases

### 1. Public Opinion Prediction

Upload news articles or social media data to simulate how public sentiment might evolve:

- Viral content spread patterns
- Crisis communication outcomes
- Brand reputation trajectories

### 2. Financial Market Simulation

Feed financial reports and market signals for agent-based market simulations:

- Investor behavior modeling
- Market sentiment analysis
- Risk scenario testing

### 3. Creative Writing

Upload the first 80 chapters of a novel and let MiroFish predict the lost ending based on character personalities and plot dynamics.

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

## API Reference

### Graph Building API

```http
POST /api/graph/build
Content-Type: application/json

{
  "documents": ["file1.pdf", "file2.docx"],
  "simulation_requirement": "Predict public reaction to policy X",
  "chunk_size": 500,
  "chunk_overlap": 50
}
```

### Simulation API

```http
POST /api/simulation/create
Content-Type: application/json

{
  "project_id": "proj_123",
  "graph_id": "mirofish_abc123",
  "enable_twitter": true,
  "enable_reddit": true
}
```

### Report API

```http
POST /api/report/generate
Content-Type: application/json

{
  "simulation_id": "sim_456",
  "graph_id": "mirofish_abc123",
  "simulation_requirement": "Analyze public sentiment trends"
}
```

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
