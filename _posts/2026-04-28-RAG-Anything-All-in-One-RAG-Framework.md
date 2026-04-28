---
layout: post
title: "RAG-Anything: All-in-One RAG Framework for Multi-Modal Retrieval Augmented Generation"
description: "Learn how to use RAG-Anything, the all-in-one RAG framework that supports multi-modal retrieval augmented generation. This guide covers installation, architecture, and building production-ready RAG pipelines."
date: 2026-04-28
header-img: "img/post-bg.jpg"
permalink: /RAG-Anything-All-in-One-RAG-Framework/
featured-img: ai-coding-frameworks/ai-coding-frameworks
categories: [AI, Python, Developer Tools]
tags: [RAG-Anything, RAG, retrieval augmented generation, multi-modal, AI framework, vector search, document processing, LLM, knowledge retrieval, open source]
keywords: "how to use RAG-Anything, RAG-Anything tutorial, RAG framework Python, multi-modal RAG pipeline, retrieval augmented generation framework, RAG-Anything vs LangChain, best RAG framework 2026, RAG-Anything installation guide, multi-modal document retrieval, open source RAG system"
author: "PyShine"
---

Modern documents are no longer just walls of text. Research papers embed figures alongside equations, financial reports weave tables between narrative paragraphs, and technical documentation mixes diagrams with code snippets. Traditional retrieval augmented generation systems treat everything as plain text, losing critical information in the process. **RAG-Anything** solves this problem as a comprehensive RAG framework that processes and queries documents containing interleaved text, images, tables, and mathematical equations through a single unified interface. Built on top of [LightRAG](https://github.com/HKUDS/LightRAG) by the HKUDS research group, RAG-Anything eliminates the need for multiple specialized tools and delivers end-to-end multimodal retrieval capabilities out of the box.

With over 19,000 stars on GitHub, RAG-Anything has quickly become one of the most popular open-source RAG frameworks for handling heterogeneous document content. This guide walks through its architecture, installation, and practical usage so you can build production-ready multimodal RAG pipelines.

## Table of Contents

- [What is RAG-Anything?](#what-is-rag-anything)
- [Architecture Overview](#architecture-overview)
- [Key Features](#key-features)
- [Installation](#installation)
- [Usage Examples](#usage-examples)
- [Configuration](#configuration)
- [Supported Content Types](#supported-content-types)
- [Troubleshooting](#troubleshooting)
- [Conclusion](#conclusion)

## What is RAG-Anything?

RAG-Anything is an all-in-one multimodal document processing RAG system developed by the Hong Kong University Data Science (HKUDS) lab. It extends the [LightRAG](https://github.com/HKUDS/LightRAG) framework to handle diverse content modalities that conventional text-only RAG pipelines cannot process effectively.

The core problem RAG-Anything addresses is straightforward: real-world documents contain mixed content types. A single research paper might include narrative text, experimental figures, result tables, and mathematical formulations. Traditional RAG systems strip away this multimodal richness, reducing everything to flat text. RAG-Anything preserves and leverages all content types through specialized processors, a multimodal knowledge graph, and modality-aware retrieval.

The framework supports three parser backends -- MinerU, Docling, and PaddleOCR -- giving you flexibility in how documents are initially parsed. It also provides VLM-enhanced query capabilities, allowing vision-language models to directly analyze images retrieved from the knowledge base.

## Architecture Overview

RAG-Anything implements a five-stage multimodal pipeline that extends traditional RAG architectures to handle diverse content modalities through intelligent orchestration and cross-modal understanding.

![RAG-Anything Architecture](/assets/img/diagrams/rag-anything/rag-anything-architecture.svg)

The architecture diagram above illustrates the complete five-stage pipeline that RAG-Anything employs. Starting from the top, documents in various formats (PDF, Office, Images) enter the **Document Parsing** stage where the MinerU parser performs adaptive content decomposition, breaking documents into their constituent elements while preserving contextual relationships. The parsed content then flows into the **Multi-Modal Understanding** stage, where content categorization and routing automatically identify and direct different content types through optimized channels. The concurrent multi-pipeline architecture processes textual and multimodal content in parallel, maximizing throughput while the document hierarchy extraction preserves the original organizational structure.

In the **Multimodal Analysis** stage, specialized processors handle each content type: the Visual Content Analyzer uses vision-language models to generate context-aware captions, the Structured Data Interpreter performs statistical pattern recognition on tabular data, the Math Expression Parser handles LaTeX formulas with conceptual mapping to domain knowledge, and the Extensible Modality Handler supports custom content types through a plugin architecture. The analysis results feed into the **Knowledge Graph** stage, where multi-modal entity extraction transforms significant elements into structured graph entities, cross-modal relationship mapping establishes semantic connections between textual and multimodal components, hierarchical structure preservation maintains document organization through "belongs_to" chains, and weighted relationship scoring assigns quantitative relevance based on semantic proximity. Finally, the **Retrieval** stage combines vector-graph fusion search with modality-aware ranking and relational coherence maintenance to deliver contextually integrated multimodal query results.

### Multimodal Processing Pipeline

![Processing Pipeline](/assets/img/diagrams/rag-anything/rag-anything-processing-pipeline.svg)

The processing pipeline diagram shows the left-to-right flow from document input to query output. An input document can be routed to one of three parser backends: MinerU (the default, selected with the "auto" mode), Docling (optimized for Office documents and HTML), or PaddleOCR (OCR-focused for images and PDFs). Each parser decomposes the document into content blocks categorized as Text, Images, Tables, or Equations. These blocks then enter dedicated processing pipelines: text flows through the LightRAG text pipeline, images go through the VLM pipeline using a vision model, tables are processed by the statistical pipeline, and equations enter the LaTeX parser pipeline. All pipeline outputs converge into the Multimodal Knowledge Graph. When a user issues a query, the system retrieves relevant content from the knowledge graph and returns answers with citations linking back to source documents.

### Knowledge Graph Construction

![Knowledge Graph](/assets/img/diagrams/rag-anything/rag-anything-knowledge-graph.svg)

The knowledge graph construction diagram details how multimodal entities are transformed into a structured semantic representation. Text entities, image entities, table entities, and equation entities all enter the entity extraction and annotation step, which transforms significant multimodal elements into structured knowledge graph entities with semantic annotations and metadata. The extracted entities then pass through cross-modal relationship mapping, where automated inference algorithms establish semantic connections and dependencies between textual entities and multimodal components. Hierarchy preservation maintains the original document organization through "belongs_to" relationship chains that preserve logical content hierarchy and sectional dependencies. Weighted relationship scoring assigns quantitative relevance scores based on semantic proximity and contextual significance. The resulting Multimodal Knowledge Graph contains three types of links: semantic links connecting conceptually related entities, dependency links capturing functional relationships, and hierarchical links preserving document structure.

### Retrieval System

![Retrieval System](/assets/img/diagrams/rag-anything/rag-anything-retrieval-system.svg)

The retrieval system diagram illustrates the hybrid retrieval architecture. A user query is processed through two parallel paths: vector similarity search, which embeds the query and finds semantically similar content, and graph traversal algorithms, which navigate the knowledge graph structure to find related entities. The results from both paths merge in the Vector-Graph Fusion step, combining semantic similarity scores with structural graph paths. The fused results then pass through modality-aware ranking, which adjusts scores based on content type relevance and query-specific modality preferences. Relational coherence maintenance ensures that the relationships between retrieved elements are preserved, delivering contextually integrated results rather than isolated fragments. This dual-path approach ensures both semantic relevance and structural coherence in the final answer.

## Key Features

| Feature | Description |
|---------|-------------|
| End-to-End Multimodal Pipeline | Complete workflow from document ingestion and parsing to intelligent multimodal query answering |
| Universal Document Support | Seamless processing of PDFs, Office documents (DOC/DOCX/PPT/PPTX/XLS/XLSX), images, and text files |
| Specialized Content Analysis | Dedicated processors for images (VLM), tables (statistical), equations (LaTeX), and custom types |
| Multimodal Knowledge Graph | Automatic entity extraction and cross-modal relationship discovery for enhanced understanding |
| Adaptive Processing Modes | Flexible MinerU, Docling, or PaddleOCR parsing workflows |
| Direct Content List Insertion | Bypass document parsing by directly inserting pre-parsed content lists from external sources |
| Hybrid Intelligent Retrieval | Vector similarity search combined with graph traversal for comprehensive content retrieval |
| VLM-Enhanced Queries | Vision-language models directly analyze images in retrieved context |
| Batch Processing | Process multiple documents concurrently with configurable workers |
| Context-Aware Processing | Intelligent integration of surrounding contextual information to enhance multimodal content processing |
| Extensible Architecture | Plugin-based modality handlers for custom and emerging content types |
| Multiple Query Modes | Naive, local, global, and hybrid search modes for different retrieval strategies |

## Installation

### Option 1: Install from PyPI (Recommended)

The quickest way to get started is installing from PyPI:

```bash
# Basic installation
pip install raganything

# With optional dependencies for extended format support:
pip install 'raganything[all]'              # All optional features
pip install 'raganything[image]'            # Image format conversion (BMP, TIFF, GIF, WebP)
pip install 'raganything[text]'             # Text file processing (TXT, MD)
pip install 'raganything[image,text]'       # Multiple features
```

### Option 2: Install from Source

For development or if you need the latest changes:

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and setup the project with uv
git clone https://github.com/HKUDS/RAG-Anything.git
cd RAG-Anything

# Install the package and dependencies in a virtual environment
uv sync

# If you encounter network timeouts (especially for opencv packages):
# UV_HTTP_TIMEOUT=120 uv sync

# Run commands directly with uv (recommended approach)
uv run python examples/raganything_example.py --help

# Install with optional dependencies
uv sync --extra image --extra text  # Specific extras
uv sync --all-extras                 # All optional features
```

### Office Document Processing Requirements

Office documents (.doc, .docx, .ppt, .pptx, .xls, .xlsx) require LibreOffice installed separately:

| Platform | Command |
|----------|---------|
| Windows | Download installer from [LibreOffice official website](https://www.libreoffice.org/download/download/) |
| macOS | `brew install --cask libreoffice` |
| Ubuntu/Debian | `sudo apt-get install libreoffice` |
| CentOS/RHEL | `sudo yum install libreoffice` |

### Verify Installation

After installation, verify that MinerU is properly configured:

```bash
# Verify MinerU installation
mineru --version

# Check if RAG-Anything can find the parser
python -c "from raganything import RAGAnything; rag = RAGAnything(); print('MinerU installed properly' if rag.check_parser_installation() else 'MinerU installation issue')"
```

Models are downloaded automatically on first use. For manual download, refer to the [MinerU Model Source Configuration](https://github.com/opendatalab/MinerU/blob/master/README.md).

## Usage Examples

### 1. End-to-End Document Processing

The most common use case is processing a document and then querying it. This example shows the complete workflow:

```python
import asyncio
from functools import partial
from raganything import RAGAnything, RAGAnythingConfig
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc

async def main():
    # Set up API configuration
    api_key = "your-api-key"
    base_url = "your-base-url"  # Optional

    # Create RAGAnything configuration
    config = RAGAnythingConfig(
        working_dir="./rag_storage",
        parser="mineru",  # Parser selection: mineru, docling, or paddleocr
        parse_method="auto",  # Parse method: auto, ocr, or txt
        enable_image_processing=True,
        enable_table_processing=True,
        enable_equation_processing=True,
    )

    # Define LLM model function
    def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
        return openai_complete_if_cache(
            "gpt-4o-mini",
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            api_key=api_key,
            base_url=base_url,
            **kwargs,
        )

    # Define vision model function for image processing
    def vision_model_func(
        prompt, system_prompt=None, history_messages=[], image_data=None, messages=None, **kwargs
    ):
        if messages:
            return openai_complete_if_cache(
                "gpt-4o", "", system_prompt=None, history_messages=[],
                messages=messages, api_key=api_key, base_url=base_url, **kwargs,
            )
        elif image_data:
            return openai_complete_if_cache(
                "gpt-4o", "", system_prompt=None, history_messages=[],
                messages=[
                    {"role": "system", "content": system_prompt} if system_prompt else None,
                    {"role": "user", "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {
                            "url": f"data:image/jpeg;base64,{image_data}"
                        }},
                    ]} if image_data else {"role": "user", "content": prompt},
                ],
                api_key=api_key, base_url=base_url, **kwargs,
            )
        else:
            return llm_model_func(prompt, system_prompt, history_messages, **kwargs)

    # Define embedding function
    embedding_func = EmbeddingFunc(
        embedding_dim=3072,
        max_token_size=8192,
        func=partial(
            openai_embed.func,
            model="text-embedding-3-large",
            api_key=api_key,
            base_url=base_url,
        ),
    )

    # Initialize RAGAnything
    rag = RAGAnything(
        config=config,
        llm_model_func=llm_model_func,
        vision_model_func=vision_model_func,
        embedding_func=embedding_func,
    )

    # Process a document
    await rag.process_document_complete(
        file_path="path/to/your/document.pdf",
        output_dir="./output",
        parse_method="auto"
    )

    # Query the processed content
    text_result = await rag.aquery(
        "What are the main findings shown in the figures and tables?",
        mode="hybrid"
    )
    print("Text query result:", text_result)

if __name__ == "__main__":
    asyncio.run(main())
```

### 2. Multimodal Queries

RAG-Anything supports three types of query methods, each suited to different use cases:

**Pure Text Queries** -- Direct knowledge base search using LightRAG:

```python
# Different query modes for text queries
text_result_hybrid = await rag.aquery("Your question", mode="hybrid")
text_result_local = await rag.aquery("Your question", mode="local")
text_result_global = await rag.aquery("Your question", mode="global")
text_result_naive = await rag.aquery("Your question", mode="naive")

# Synchronous version
sync_text_result = rag.query("Your question", mode="hybrid")
```

**VLM Enhanced Queries** -- Automatically analyze images in retrieved context using a vision-language model:

```python
# VLM enhanced query (automatically enabled when vision_model_func is provided)
vlm_result = await rag.aquery(
    "Analyze the charts and figures in the document",
    mode="hybrid"
)

# Manually control VLM enhancement
vlm_enabled = await rag.aquery(
    "What do the images show in this document?",
    mode="hybrid",
    vlm_enhanced=True   # Force enable VLM enhancement
)

vlm_disabled = await rag.aquery(
    "What do the images show in this document?",
    mode="hybrid",
    vlm_enhanced=False  # Force disable VLM enhancement
)
```

**Multimodal Content Queries** -- Enhanced queries with specific multimodal content analysis:

```python
# Query with equation content
equation_result = await rag.aquery_with_multimodal(
    "Explain this formula and its relevance to the document content",
    multimodal_content=[{
        "type": "equation",
        "latex": "P(d|q) = \\frac{P(q|d) \\cdot P(d)}{P(q)}",
        "equation_caption": "Document relevance probability"
    }],
    mode="hybrid"
)

# Query with table data
table_result = await rag.aquery_with_multimodal(
    "Compare these performance metrics with the document content",
    multimodal_content=[{
        "type": "table",
        "table_data": "Method,Accuracy,Speed\nRAGAnything,95.2%,120ms\nTraditional,87.3%,180ms",
        "table_caption": "Performance comparison"
    }],
    mode="hybrid"
)
```

### 3. Batch Processing

Process entire folders of documents concurrently:

```python
# Process multiple documents
await rag.process_folder_complete(
    folder_path="./documents",
    output_dir="./output",
    file_extensions=[".pdf", ".docx", ".pptx"],
    recursive=True,
    max_workers=4
)
```

### 4. Direct Content List Insertion

If you already have pre-parsed content from external sources, you can insert it directly without document parsing:

```python
# Pre-parsed content list from external source
content_list = [
    {
        "type": "text",
        "text": "This is the introduction section of our research paper.",
        "page_idx": 0
    },
    {
        "type": "image",
        "img_path": "/absolute/path/to/figure1.jpg",
        "image_caption": ["Figure 1: System Architecture"],
        "image_footnote": ["Source: Authors' original design"],
        "page_idx": 1
    },
    {
        "type": "table",
        "table_body": "| Method | Accuracy | F1-Score |\n|--------|----------|----------|\n| Ours | 95.2% | 0.94 |",
        "table_caption": ["Table 1: Performance Comparison"],
        "table_footnote": ["Results on test dataset"],
        "page_idx": 2
    },
    {
        "type": "equation",
        "latex": "P(d|q) = \\frac{P(q|d) \\cdot P(d)}{P(q)}",
        "text": "Document relevance probability formula",
        "page_idx": 3
    }
]

# Insert the content list directly
await rag.insert_content_list(
    content_list=content_list,
    file_path="research_paper.pdf",
    display_stats=True
)
```

### 5. Custom Modal Processors

Extend RAG-Anything with your own content type processors:

```python
from raganything.modalprocessors import GenericModalProcessor

class CustomModalProcessor(GenericModalProcessor):
    async def process_multimodal_content(self, modal_content, content_type, file_path, entity_name):
        # Your custom processing logic
        enhanced_description = await self.analyze_custom_content(modal_content)
        entity_info = self.create_custom_entity(enhanced_description, entity_name)
        return await self._create_entity_and_chunk(enhanced_description, entity_info, file_path)
```

### 6. Loading an Existing LightRAG Instance

If you already have a LightRAG instance with data, you can connect RAG-Anything to it:

```python
import asyncio
from functools import partial
from raganything import RAGAnything, RAGAnythingConfig
from lightrag import LightRAG
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.utils import EmbeddingFunc
import os

async def load_existing_lightrag():
    api_key = "your-api-key"
    base_url = "your-base-url"
    lightrag_working_dir = "./existing_lightrag_storage"

    # Create or load existing LightRAG instance
    lightrag_instance = LightRAG(
        working_dir=lightrag_working_dir,
        llm_model_func=lambda prompt, system_prompt=None, history_messages=[], **kwargs: openai_complete_if_cache(
            "gpt-4o-mini", prompt, system_prompt=system_prompt,
            history_messages=history_messages, api_key=api_key,
            base_url=base_url, **kwargs,
        ),
        embedding_func=EmbeddingFunc(
            embedding_dim=3072, max_token_size=8192,
            func=partial(openai_embed.func, model="text-embedding-3-large",
                         api_key=api_key, base_url=base_url),
        )
    )

    # Initialize storage (loads existing data if available)
    await lightrag_instance.initialize_storages()
    await initialize_pipeline_status()

    # Connect RAG-Anything to the existing instance
    rag = RAGAnything(
        lightrag=lightrag_instance,
        vision_model_func=vision_model_func,
    )

    # Query existing knowledge base
    result = await rag.aquery(
        "What data has been processed in this LightRAG instance?",
        mode="hybrid"
    )
    print("Query result:", result)

if __name__ == "__main__":
    asyncio.run(load_existing_lightrag())
```

## Configuration

### Environment Variables

Create a `.env` file in your project root (refer to the `.env.example` in the repository):

```bash
OPENAI_API_KEY=your_openai_api_key
OPENAI_BASE_URL=your_base_url  # Optional
OUTPUT_DIR=./output             # Default output directory for parsed documents
PARSER=mineru                   # Parser selection: mineru, docling, or paddleocr
PARSE_METHOD=auto              # Parse method: auto, ocr, or txt
```

### RAGAnythingConfig Parameters

The `RAGAnythingConfig` dataclass provides fine-grained control over the framework:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `working_dir` | `./rag_storage` | Directory for RAG storage and cache files |
| `parse_method` | `auto` | Parsing method: `auto`, `ocr`, or `txt` |
| `parser_output_dir` | `./output` | Output directory for parsed content |
| `parser` | `mineru` | Parser backend: `mineru`, `docling`, or `paddleocr` |
| `enable_image_processing` | `True` | Enable image content processing |
| `enable_table_processing` | `True` | Enable table content processing |
| `enable_equation_processing` | `True` | Enable equation content processing |
| `max_concurrent_files` | `1` | Maximum concurrent file processing |
| `recursive_folder_processing` | `True` | Recursively process subfolders in batch mode |
| `context_window` | `1` | Pages/chunks before and after for context |
| `context_mode` | `page` | Context mode: `page` or `chunk` |
| `max_context_tokens` | `2000` | Maximum tokens in extracted context |
| `use_full_path` | `False` | Use full file path or basename for references |

### Parser Comparison

| Parser | Strengths | Best For |
|--------|-----------|----------|
| MinerU | PDF, images, Office docs; powerful OCR and table extraction; GPU acceleration | General-purpose document processing |
| Docling | Optimized for Office documents and HTML; better structure preservation | Office-heavy workflows |
| PaddleOCR | OCR-focused for images and PDFs; produces text blocks | Image-heavy OCR tasks |

To install PaddleOCR parser extras:

```bash
pip install -e ".[paddleocr]"
# or
uv sync --extra paddleocr
```

Note that PaddleOCR also requires `paddlepaddle`. Install it following the [official PaddlePaddle guide](https://www.paddlepaddle.org.cn/install/quick).

### Advanced MinerU Configuration

MinerU 2.0 uses command-line parameters instead of config files:

```bash
# Common configurations
mineru -p input.pdf -o output_dir -m auto    # Automatic parsing mode
mineru -p input.pdf -o output_dir -m ocr     # OCR-focused parsing
mineru -p input.pdf -o output_dir -b pipeline --device cuda  # GPU acceleration
```

You can also pass MinerU parameters through the RAG-Anything API:

```python
await rag.process_document_complete(
    file_path="document.pdf",
    output_dir="./output/",
    parse_method="auto",
    parser="mineru",

    # MinerU special parameters
    lang="ch",                   # Document language for OCR optimization
    device="cuda:0",             # Inference device
    start_page=0,                # Starting page number (0-based)
    end_page=10,                 # Ending page number (0-based)
    formula=True,                # Enable formula parsing
    table=True,                  # Enable table parsing
    backend="pipeline",          # Parsing backend
    source="huggingface",        # Model source
)
```

## Supported Content Types

### Document Formats

| Format | Extensions | Notes |
|--------|-----------|-------|
| PDF | `.pdf` | Research papers, reports, presentations |
| Word | `.doc`, `.docx` | Requires LibreOffice |
| PowerPoint | `.ppt`, `.pptx` | Requires LibreOffice |
| Excel | `.xls`, `.xlsx` | Requires LibreOffice |
| Images | `.jpg`, `.png`, `.bmp`, `.tiff`, `.gif`, `.webp` | BMP/TIFF/GIF/WebP need `raganything[image]` |
| Text | `.txt`, `.md` | Requires `raganything[text]` |

### Multimodal Elements

| Element | Processor | Description |
|---------|-----------|-------------|
| Images | `ImageModalProcessor` | Photographs, diagrams, charts, screenshots |
| Tables | `TableModalProcessor` | Data tables, comparison charts, statistical summaries |
| Equations | `EquationModalProcessor` | Mathematical formulas in LaTeX format |
| Custom | `GenericModalProcessor` | Extensible for custom content types |

## Troubleshooting

### MinerU Installation Issues

If `mineru --version` fails or RAG-Anything reports a parser installation issue:

1. Ensure MinerU is installed: `pip install mineru[core]`
2. Verify the command is on your PATH: `mineru --version`
3. If models fail to download, check network connectivity or configure a mirror source:
   ```python
   # Use ModelScope mirror for Chinese users
   await rag.process_document_complete(
       file_path="document.pdf",
       output_dir="./output",
       source="modelscope"  # or "huggingface", "local"
   )
   ```

### LibreOffice Not Found for Office Documents

When processing .docx, .pptx, or .xlsx files, you may see errors about LibreOffice:

1. Install LibreOffice from the [official website](https://www.libreoffice.org/download/download/)
2. On Windows, ensure the LibreOffice installation directory is in your system PATH
3. Verify installation: `soffice --version`

### Network Timeouts During Installation

If `uv sync` or `pip install` times out, especially for opencv packages:

```bash
# For uv
UV_HTTP_TIMEOUT=120 uv sync

# For pip
pip install raganything --timeout 120
```

### Image Format Support Errors

If you encounter errors processing BMP, TIFF, GIF, or WebP images:

```bash
pip install 'raganything[image]'  # Installs Pillow
```

### Text File Processing Errors

For .txt and .md file processing issues:

```bash
pip install 'raganything[text]'  # Installs ReportLab
```

### VLM Query Returns Text-Only Results

If VLM-enhanced queries do not seem to analyze images:

1. Ensure `vision_model_func` is provided when initializing RAGAnything
2. Verify the vision model supports image inputs (e.g., GPT-4o, Claude 3.5 Sonnet)
3. Explicitly enable VLM enhancement:
   ```python
   result = await rag.aquery("Your question", mode="hybrid", vlm_enhanced=True)
   ```

### PaddleOCR Parser Issues

If using the PaddleOCR parser:

1. Install the extras: `pip install raganything[paddleocr]`
2. Install PaddlePaddle for your platform following the [official guide](https://www.paddlepaddle.org.cn/install/quick)
3. Note that PaddleOCR converts Office/TXT/MD files to PDF first, which may affect formatting

### Memory Issues with Large Documents

For large documents or batch processing:

1. Reduce `max_concurrent_files` to 1 in the configuration
2. Process documents individually rather than in batches
3. Use `start_page` and `end_page` parameters to process documents in chunks:
   ```python
   await rag.process_document_complete(
       file_path="large_document.pdf",
       output_dir="./output",
       start_page=0,
       end_page=50  # Process first 50 pages
   )
   ```

## Conclusion

RAG-Anything provides a production-ready, all-in-one RAG framework for multimodal document processing and retrieval. Its five-stage pipeline -- from document parsing through multimodal analysis, knowledge graph construction, and hybrid retrieval -- handles the full spectrum of content types found in real-world documents. The framework's extensibility through custom modal processors, its support for multiple parser backends, and its VLM-enhanced query capabilities make it a versatile choice for academic research, technical documentation, financial analysis, and enterprise knowledge management.

The project is actively maintained by the HKUDS lab and has a growing community. If you are building a RAG system that needs to handle more than just plain text, RAG-Anything is worth serious consideration.

**Links:**

- [GitHub Repository](https://github.com/HKUDS/RAG-Anything)
- [ArXiv Paper](https://arxiv.org/abs/2510.12323)
- [PyPI Package](https://pypi.org/project/raganything/)
- [LightRAG (Base Framework)](https://github.com/HKUDS/LightRAG)
- [MinerU Parser](https://github.com/opendatalab/MinerU)
- [Discord Community](https://discord.gg/yF2MmDJyGJ)

**Citation:**

```bibtex
@misc{guo2025raganythingallinoneragframework,
      title={RAG-Anything: All-in-One RAG Framework},
      author={Zirui Guo and Xubin Ren and Lingrui Xu and Jiahao Zhang and Chao Huang},
      year={2025},
      eprint={2510.12323},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2510.12323},
}