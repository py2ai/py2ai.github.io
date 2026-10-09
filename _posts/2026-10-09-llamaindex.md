---
layout: post
title: "LlamaIndex: The Data Framework That Taught LLMs to Read Your Files - Inside run-llama/llama_index"
description: "A source-code tour of LlamaIndex, the MIT-licensed data framework for LLM applications: readers and node parsers, the ingestion stage, StorageContext, vector store and index abstractions, composable retrievers, query engines, response synthesizers, workflows, and the Settings globals that wire it all together."
date: 2026-10-09
header-img: "img/post-bg.jpg"
permalink: /llamaindex/
featured-img: "ai-coding-frameworks/ai-coding-frameworks"
image: https://pyshine.com/assets/img/diagrams/llamaindex/run-llama-llamaindex-overview-architecture.svg
tags: [RAG, Python, Frameworks, Open Source]
categories: [AI, Open Source]
keywords: "llamaindex, rag, run-llama, vector store, retrieval, llm framework, open source, architecture"
author: "PyShine"
---

LlamaIndex is the framework that made "connect your data to an LLM" a teachable, layered discipline rather than a pile of glue scripts. Its monorepo at run-llama/llama_index is MIT-licensed Python organized with unusual clarity: llama-index-core holds the abstractions, llama-index-integrations holds the enormous ecosystem of adapters in roughly two dozen category folders, llama-index-instrumentation carries the observability primitives, and llama-dev holds the shared developer tooling. The core package currently sits at version 0.14.25 and targets Python 3.10 and newer. What makes this codebase worth a long read is that it solves the hardest problem in retrieval-augmented generation, which is not calling an API but deciding where each concern lives: loading, chunking, embedding, storing, retrieving, synthesizing, and routing each get their own interface.

The design idea that runs through everything is a grammar of five nouns: Documents come in through readers, get split into Nodes by parsers and transformations, land in an Index backed by stores, answer questions through Retrievers, and speak through query engines that call response synthesizers around your LLM. Every one of those nouns is a small abstract class in llama-index-core, and every provider integration is just an implementation of one of them. A global Settings object in llama_index/core/settings.py holds the default LLM, embedding model, and tokenizer so the rest of the library never needs you to pass the same three objects around by hand.

As always in this series, this is an educational tour of published source code. Ingestion frameworks handle your documents, your API keys, and sometimes your network endpoints, and LlamaIndex gives you the right seams to stay safe: connectors are plain classes you instantiate yourself, stores persist wherever you point them, and callbacks plus the instrumentation layer let you observe exactly what leaves your process. Read the interfaces before you wire anything to the internet; that habit is worth more than any tutorial.

<div style="overflow-x:auto;"><img src="https://pyshine.com/assets/img/diagrams/llamaindex/run-llama-llamaindex-overview-architecture.svg" alt="LlamaIndex overview architecture diagram" style="max-width:100%;"></div>
<p><em>LlamaIndex at a glance: readers and node parsers feed the ingestion transformations, StorageContext assembles document, index, and vector stores, indices build on the BaseIndex contract, retrievers fetch context for query engines, and response synthesizers drive the LLM to produce answers that agents can wrap as tools.</em></p>

Reading the overview from left to right:

- Everything enters through the reader contract at [llama-index-core/llama_index/core/readers/base.py](https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/readers/base.py), whose most famous implementation is SimpleDirectoryReader in [llama-index-core/llama_index/core/readers/file/base.py](https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/readers/file/base.py).
- Chunking lives in the node parser layer, from the interface in [llama-index-core/llama_index/core/node_parser/interface.py](https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/node_parser/interface.py) to the SentenceSplitter in [llama-index-core/llama_index/core/node_parser/text/sentence.py](https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/node_parser/text/sentence.py).
- The ingestion stage composes arbitrary sequences of transformations in [llama-index-core/llama_index/core/ingestion/transformations.py](https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/ingestion/transformations.py) and persists the results through StorageContext at [llama-index-core/llama_index/core/storage/storage_context.py](https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/storage/storage_context.py).
- Model access splits into the LLM interface at [llama-index-core/llama_index/core/base/llms/base.py](https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/base/llms/base.py) and BaseEmbedding at [llama-index-core/llama_index/core/base/embeddings/base.py](https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/base/embeddings/base.py).
- Vector backends implement the contracts in [llama-index-core/llama_index/core/vector_stores/types.py](https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/vector_stores/types.py), and indices build on BaseIndex at [llama-index-core/llama_index/core/indices/base.py](https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/indices/base.py).
- Questions flow through the retriever base at [llama-index-core/llama_index/core/base/base_retriever.py](https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/base/base_retriever.py) into the RetrieverQueryEngine at [llama-index-core/llama_index/core/query_engine/retriever_query_engine.py](https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/query_engine/retriever_query_engine.py), which synthesizes answers via [llama-index-core/llama_index/core/response_synthesizers/refine.py](https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/response_synthesizers/refine.py).
- Agents consume all of it as tools, with QueryEngineTool defined in [llama-index-core/llama_index/core/tools/query_engine.py](https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/tools/query_engine.py).

## Why You Need This

The first reason is that LlamaIndex is the clearest open implementation of the RAG abstraction ladder. A lot of systems hand you a black-box "chat with your data" function; this one hands you five interfaces and lets you stand at any rung. The reader contract in readers/base.py is a single method that returns documents, and the moment you implement it for your own weird data source, you get the whole rest of the stack for free. The same is true one level down: the node parser interface and the transformation sequences in the ingestion stage mean chunking stops being a hard-coded step and becomes a composable object graph you can inspect, log, and unit-test. If you have ever shipped a retrieval system where chunking was a 300-line function nobody dared to touch, this package will feel like ventilation.

The second reason is storage discipline. StorageContext in storage/storage_context.py is the assembly point that wires together a document store from [llama-index-core/llama_index/core/storage/docstore](https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/storage/docstore), an index store from [llama-index-core/llama_index/core/storage/index_store](https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/storage/index_store), and a vector store implementing [llama-index-core/llama_index/core/vector_stores/types.py](https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/vector_stores/types.py). Because indices only ever talk to that context, swapping in-memory dictionaries for a production database is a configuration change, not a rewrite. This is the pattern every data-intensive application should steal: small stores, one assembly object, zero hidden singletons.

The third reason is that retrieval is treated as a first-class, composable discipline rather than a single similarity search. The retriever base in base/base_retriever.py has siblings like the auto-merging retriever in [llama-index-core/llama_index/core/retrievers/auto_merging_retriever.py](https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/retrievers/auto_merging_retriever.py), the query fusion retriever in [llama-index-core/llama_index/core/retrievers/fusion_retriever.py](https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/retrievers/fusion_retriever.py), and the recursive retriever in [llama-index-core/llama_index/core/retrievers/recursive_retriever.py](https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/retrievers/recursive_retriever.py), each wrapping another retriever to add a strategy. Auto-merging reassembles parent chunks when enough of their children match; fusion generates multiple query variants and merges their rankings; recursive follows references between documents. Reading these three files teaches more about practical retrieval quality than most paid courses.

<div style="overflow-x:auto;"><img src="https://pyshine.com/assets/img/diagrams/llamaindex/run-llama-llamaindex-architecture.svg" alt="LlamaIndex detailed architecture diagram" style="max-width:100%;"></div>
<p><em>The detailed view: eight groups from ingestion to observability. Splitters feed the node parser interface, Settings binds default LLM and embedding models, StorageContext assembles three kinds of stores, three index families implement BaseIndex, three strategy retrievers wrap BaseRetriever, two query engines route and fetch, two synthesizers iterate the LLM, and workflows plus tools connect the stack to agents.</em></p>

The detailed diagram is worth a slow pass, group by group. In Ingestion, the SentenceSplitter implements the node parser interface and leans on the token-aware splitters in [llama-index-core/llama_index/core/text_splitter](https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/text_splitter); the transformations module chains readers, parsers, and any custom transform into a sequence that can also deduplicate and persist. In Models, the LLM base at base/llms/base.py defines completion, chat, and streaming contracts, while the bridge in [llama-index-core/llama_index/core/llms/custom.py](https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/llms/custom.py) lets you wrap any arbitrary provider function into the ecosystem without subclassing everything; Settings in [llama-index-core/llama_index/core/settings.py](https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/settings.py) binds whichever implementations you choose as process-wide defaults.

In Indices, three families implement BaseIndex: VectorStoreIndex from [llama-index-core/llama_index/core/indices/vector_store](https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/indices/vector_store), SummaryIndex from [llama-index-core/llama_index/core/indices/list](https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/indices/list), and PropertyGraphIndex from [llama-index-core/llama_index/core/indices/property_graph](https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/indices/property_graph), which builds knowledge graphs with extracted entities and relations. In Querying, the RetrieverQueryEngine fetches context and hands it to synthesizers, the RouterQueryEngine in [llama-index-core/llama_index/core/query_engine/router_query_engine.py](https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/query_engine/router_query_engine.py) uses the LLM selectors in [llama-index-core/llama_index/core/selectors](https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/selectors) to choose between engines per query, and node postprocessors from [llama-index-core/llama_index/core/postprocessor](https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/postprocessor) rerank or filter retrieved nodes before synthesis. The Refine synthesizer iterates over chunks and progressively refines an answer, while tree summarize in [llama-index-core/llama_index/core/response_synthesizers/tree_summarize.py](https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/response_synthesizers/tree_summarize.py) builds a hierarchy of partial summaries for when context exceeds one window.

## From Documents to Answers

The README quickstart is four lines, and each line lands on one of the interfaces above:

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

documents = SimpleDirectoryReader("YOUR_DATA_DIRECTORY").load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()
response = query_engine.query("YOUR_QUESTION")
```

Under the hood, load_data walks your directory through the file reader family in readers/file/base.py and returns Documents. from_documents runs the default node parser, embeds each chunk through the BaseEmbedding implementation in Settings, and writes vectors and metadata through the StorageContext into a default vector store. as_query_engine assembles a RetrieverQueryEngine with a Refine synthesizer, and query() executes retrieve, postprocess, synthesize. The beautiful part is that none of those steps are hidden: every default is a class you can import, subclass, or replace, which is exactly what makes the library a curriculum instead of a product.

The README's second example shows the knob you will use most. You import Settings, then assign your choices:

```python
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama

Settings.llm = Ollama(model="YOUR_MODEL", request_timeout=120.0)
Settings.embed_model = HuggingFaceEmbedding(model_name="YOUR_EMBED_MODEL")
```

Those two assignments change every index, retriever, and engine you build afterwards, including fully local stacks that never touch an external API. Persistence is equally explicit: the README imports StorageContext and load_index_from_storage to rebuild an index from disk, so re-embedding on every startup is a choice you make, not a behavior baked in.

## The Integrations Ring

The core stays small because everything provider-specific lives in llama-index-integrations, whose category folders cover LLMs, embeddings, vector stores, readers, node parsers, retrievers, query engines, tools, and more. The namespacing convention in the README makes the whole ring predictable: core abstractions import from llama_index.core.xxx and adapters import from llama_index.xxx.yyy, so `from llama_index.core.llms import LLM` names the contract while `from llama_index.llms.openai import OpenAI` names an implementation. When you evaluate a new provider, you are reading one adapter package against one core interface, which turns vendor comparison into a five-minute diff instead of a week of docs.

## Workflows, Tools, and Observability

The agent layer connects the stack to autonomous use. FunctionTool in [llama-index-core/llama_index/core/tools/function_tool.py](https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/tools/function_tool.py) turns plain Python functions into schema-declared tools, and QueryEngineTool wraps any query engine so an agent can consult your data as one action among many. Chat engines in [llama-index-core/llama_index/core/chat_engine](https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/chat_engine) add memory and turn-taking on top of retrieval. The workflow system, with its handler in [llama-index-core/llama_index/core/workflow/handler.py](https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/workflow/handler.py) and typed events in [llama-index-core/llama_index/core/workflow/events.py](https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/workflow/events.py), lets you compose multi-step, event-driven processes where steps wait for and emit events, which is how the library grew agentic capabilities without contaminating the RAG core.

Observability follows the same two-layer split. The callback manager in [llama-index-core/llama_index/core/callbacks/base.py](https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/callbacks/base.py) traces LLM calls, retrievals, and engine runs, while the instrumentation package under [llama-index-core/llama_index/core/instrumentation](https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/instrumentation) provides the dispatcher and span model that workflow events flow through. Because both are contract-first, the dozens of third-party tracer integrations plug in without touching core code.

## Try It Yourself

Install the core and any two adapters, point a reader at a folder of text, and run the quickstart above; the whole loop fits in an afternoon. Then do the exercise that actually teaches the architecture: write one custom reader implementing the contract from readers/base.py, one custom transform that logs node counts, and one custom vector store against the contracts in vector_stores/types.py. Nothing in the core changes when you do this, which is precisely the point, and it is the fastest way to internalize why the framework's seams are drawn where they are.

LlamaIndex earns its place in this series because it is a masterclass in layering a fast-moving domain: five interfaces, one assembly object, one settings registry, and an ecosystem that scales to hundreds of adapters without a single god class. Whether you end up using the library or not, its answer to the question "where does each concern live?" is a template worth copying in any retrieval or agent system you build.

Next up in this series: gpt-engineer, the project that turns a plain-English spec into a running codebase. Until then, read the seams, not the slogans.
