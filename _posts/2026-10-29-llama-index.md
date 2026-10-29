---
layout: post
title: "LlamaIndex Source Tour: How the Data Framework for LLM Apps Is Assembled"
description: "A guided tour of the llama_index repository: ingestion, indices, retrievers, response synthesis, and storage inside llama-index-core."
date: 2026-10-29
header-img: "img/post-bg.jpg"
permalink: /llama-index/
featured-img: "ai-coding-frameworks/ai-coding-frameworks"
image: https://pyshine.com/assets/img/diagrams/llama-index/run-llama-llama-index-overview-architecture.svg
tags: [LLM, Python, Open Source]
categories: [AI Inference]
keywords: LlamaIndex, llama-index-core, RAG, VectorStoreIndex, retrieval augmented generation, vector store, Python
author: "PyShine"
---

Ask a room of Python developers how they wire private documents into large language models, and sooner or later the answer will be LlamaIndex. The `run-llama/llama_index` repository is the home of that framework: a data layer that connects files, APIs, and databases to LLM prompts through indexing and retrieval. The README describes it as a "data framework" that offers connectors to ingest your existing data sources, ways to structure that data into indices, and an advanced retrieval and query interface that turns raw sources into knowledge-augmented output.

This post is a source tour, not a usage tutorial. We read the current `main` branch of the repository end to end, traced the classes that make up `llama-index-core`, and verified every symbol and line number below directly against the code before publishing. Two diagrams anchor the discussion: a compact overview of the moving parts, and a larger detail map that adds the retrieval stack, response synthesis, and storage internals.

The architecture is deliberately modular: reading, chunking, indexing, retrieval, and answer synthesis each live in their own module with little glue between them. Once you know the seams, you can swap any component — a different LLM, a different vector store, a different chunking strategy — without disturbing the rest. That swap-friendly design is why hundreds of integration packages coexist around a single core.

<div style="overflow-x:auto;"><img src="/assets/img/diagrams/llama-index/run-llama-llama-index-overview-architecture.svg" alt="Overview architecture of LlamaIndex: ingestion, model abstractions, indices, query flow, and storage" /></div>

<p><em>LlamaIndex overview: documents flow from SimpleDirectoryReader through SentenceSplitter into VectorStoreIndex, while Settings configures the LLM and embedding model used by the query flow.</em></p>

Reading the overview from left to right:

- [SimpleDirectoryReader, L208](https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/readers/file/base.py#L208) is the default entry point for local files; its `load_data()` method at [L718](https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/readers/file/base.py#L718) turns a folder of PDFs, markdown, and docs into `Document` objects.
- [TextNode, L765](https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/schema.py#L765) is the universal currency of the framework: everything that gets embedded, stored, retrieved, and cited is a node, and `Document` at [L1097](https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/schema.py#L1097) is just the top-level node that wraps a whole source file.
- [SentenceSplitter, L34](https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/node_parser/text/sentence.py#L34) chunks documents into nodes; its `split_text()` at [L176](https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/node_parser/text/sentence.py#L176) respects sentence boundaries while keeping chunks inside a token budget.
- [BaseIndex, L25](https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/indices/base.py#L25) defines the index contract — `from_documents()` at [L89](https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/indices/base.py#L89), `insert()` at [L229](https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/indices/base.py#L229), and factory methods that hand back engines and retrievers.
- [VectorStoreIndex, L36](https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/indices/vector_store/base.py#L36) is the flagship implementation; its constructor resolves the embedding model from [Settings, L291](https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/settings.py#L291), the module-level global every component falls back to.
- [RetrieverQueryEngine, L25](https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/query_engine/retriever_query_engine.py#L25) closes the loop on the query side: its `_query()` at [L203](https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/query_engine/retriever_query_engine.py#L203) retrieves nodes and hands them to a response synthesizer backed by [BaseLLM, L27](https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/base/llms/base.py#L27).
- [StorageContext, L53](https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/storage/storage_context.py#L53) sits under everything, bundling the docstore, index store, and vector store that give the index somewhere to live.

### The full detail map

The overview hides how much choice lives inside each box. The detail diagram below opens them up: the node data model, the retriever family, the synthesizer hierarchy, and the storage layer, all with the exact classes you will meet in the code.

<div style="overflow-x:auto;"><img src="/assets/img/diagrams/llama-index/run-llama-llama-index-architecture.svg" alt="Detailed architecture of llama-index-core: data model, ingestion, indices, retrieval, query execution, response synthesis, model abstractions, and storage" /></div>

<p><em>Detail map of llama-index-core — every node links back to the real class in the repository, grouped by responsibility.</em></p>

## Two Packages, One Namespace

LlamaIndex ships in two shapes. The `llama-index` starter package bundles core plus a selection of integrations for people who want to be productive immediately. The `llama-index-core` package contains only the engine, and you add exactly the integration packages your application needs — the README counts over three hundred of them on the integrations page. The import system makes the split visible: statements that include `core` come from the core package, and statements without `core` come from an integration. The README's own example is `from llama_index.core.llms import LLM` for the abstraction and `from llama_index.llms.openai import OpenAI` for a concrete provider.

This namespace rule is more than a style choice; it is the dependency direction of the whole ecosystem. Integrations subclass the abstract bases in core — [BaseLLM, L27](https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/base/llms/base.py#L27) with `chat()` at [L89](https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/base/llms/base.py#L89) and `complete()` at [L113](https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/base/llms/base.py#L113), or [BaseEmbedding, L72](https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/base/embeddings/base.py#L72) with `get_text_embedding()` at [L385](https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/base/embeddings/base.py#L385) — and core code only ever references the abstract interfaces. That is what keeps a vector store swap or an LLM swap a one-line change at the application level.

The coordination point is the `Settings` object at [L291](https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/settings.py#L291) of `settings.py`. It is a module-level singleton that remembers your default LLM, default embedding model, default tokenizer, and default transformations. Every component that needs a model but was not given one explicitly asks `Settings` first. This is why the README quickstart works with no configuration: unset components resolve to defaults that make sense for the installed starter package.

## From Install to Your First Model

The README's recommended path for a customized setup is plain pip:

```sh
pip install llama-index-core
pip install llama-index-llms-openai
pip install llama-index-llms-ollama
pip install llama-index-embeddings-huggingface
```

The minimal example from the README then builds a vector index in four lines of Python:

```python
import os
os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

documents = SimpleDirectoryReader("YOUR_DATA_DIRECTORY").load_data()
index = VectorStoreIndex.from_documents(documents)
```

Under the hood, `from_documents()` — the classmethod at [L89](https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/indices/base.py#L89) of `indices/base.py` — runs the reader output through the default transformation list, which chunks each document with the sentence splitter, and then embeds the resulting nodes into the vector store held by a freshly built `StorageContext`.

If you would rather keep everything local, the README shows the Ollama configuration. You assign onto `Settings` directly:

```python
from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from transformers import AutoTokenizer

Settings.llm = Ollama(model="llama-3.1:latest", request_timeout=360.0)
Settings.tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct"
)
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

documents = SimpleDirectoryReader("YOUR_DATA_DIRECTORY").load_data()
index = VectorStoreIndex.from_documents(documents)
```

Setting the tokenizer matters more than it looks: the sentence splitter measures chunk sizes in tokens, so if the tokenizer does not match the model, your chunks are measured with the wrong ruler. The tokenizer side of this story — vocabulary files and model-agnostic counting — is the same machinery behind libraries like [HuggingFace tokenizers](/tokenizers/), and the `transformers` package supplying `AutoTokenizer` is covered in our [transformers source tour](/transformers/).

Querying is symmetric in simplicity:

```python
query_engine = index.as_query_engine()
query_engine.query("YOUR_QUESTION")
```

And persistence is a one-liner each way, straight from the README:

```python
index.storage_context.persist()

from llama_index.core import StorageContext, load_index_from_storage
storage_context = StorageContext.from_defaults(persist_dir="./storage")
index = load_index_from_storage(storage_context)
```

## The Index Layer: BaseIndex and Its Heirs

[BaseIndex](https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/indices/base.py#L25) is an abstract generic class that fixes the lifecycle every index follows: documents go in through `from_documents()` or incremental `insert()`, nodes flow through a transformation list, an index-specific struct gets built, and the class exposes `as_query_engine()` at [L491](https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/indices/base.py#L491) and `as_retriever()` at [L489](https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/indices/base.py#L489) as the public exits.

[VectorStoreIndex](https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/indices/vector_store/base.py#L36) is the implementation most people meet first. Three details in its source are worth noticing. First, its constructor immediately resolves the embedding model with `resolve_embed_model(embed_model or Settings.embed_model)` — explicit arguments beat globals, and globals beat nothing. Second, `_build_index_from_nodes()` at [L260](https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/indices/vector_store/base.py#L260) embeds nodes in configurable batches (`insert_batch_size` defaults to 2048), so large corpora do not build one giant request. Third, `as_retriever()` at [L112](https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/indices/vector_store/base.py#L112) lazily imports and returns a [VectorIndexRetriever, L24](https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/indices/vector_store/retrievers/retriever.py#L24), which performs top-k similarity search against whatever vector store the index was built on.

The base class also offers `from_vector_store()` at [L87](https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/indices/vector_store/base.py#L87), which wraps an existing vector store that already contains your text — useful when an external service owns the data. For non-vector strategies, [SummaryIndex, L29](https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/indices/list/base.py#L29) keeps all nodes in a flat list and retrieves the whole corpus or filtered subsets, which suits summarization-style queries where everything might be relevant. Indices are also reloadable: `load_index_from_storage()` at [L12](https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/indices/loading.py#L12) rebuilds the right index class from a persisted storage context.

## Query Execution: Retrievers, Engines, and Synthesizers

A query travels through three small stages, each with its own abstract base. The unit of input is [QueryBundle, L1449](https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/schema.py#L1449), which wraps the raw query string plus an optional embedding vector, and the unit of output is [NodeWithScore, L1035](https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/schema.py#L1035), which pairs a node with a relevance score.

The retrieval stage is [BaseRetriever, L34](https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/base/base_retriever.py#L34). Its public `retrieve()` at [L192](https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/base/base_retriever.py#L192) handles callbacks and normalization, then dispatches to the private `_retrieve()` at [L263](https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/base/base_retriever.py#L263) that subclasses implement. The retriever family is where LlamaIndex's research surface lives: [AutoMergingRetriever, L26](https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/retrievers/auto_merging_retriever.py#L26) merges small chunk hits back into larger parent chunks, [QueryFusionRetriever, L33](https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/retrievers/fusion_retriever.py#L33) fans a query out over multiple retrievers and fuses the ranked results, [RecursiveRetriever, L22](https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/retrievers/recursive_retriever.py#L22) follows node relationships to pull in referenced material, and [RouterRetriever, L20](https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/retrievers/router_retriever.py#L20) delegates to one candidate chosen by a selector.

The synthesis stage is where the answer gets written. `as_query_engine()` wires the index's retriever together with a synthesizer produced by `get_response_synthesizer()` at [L38](https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/response_synthesizers/factory.py#L38), the factory behind the `ResponseMode` enum. The default mode builds a [CompactAndRefine, L13](https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/response_synthesizers/compact_and_refine.py#L13), which extends [Refine, L196](https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/response_synthesizers/refine.py#L196). Refinement is simple and effective: the LLM drafts an answer from the first compact batch of chunks, then each subsequent batch updates that draft. `Refine.get_response()` at [L599](https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/response_synthesizers/refine.py#L599) drives the loop, packing retrieved text as densely as the context window allows before spending model calls. The whole engine is thin: [RetrieverQueryEngine.\_query()](https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/query_engine/retriever_query_engine.py#L203) is essentially retrieve-then-synthesize, with [BaseQueryEngine.query()](https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/base/base_query_engine.py#L39) as the sync entry and an async twin beside it.

## Storage and Persistence

By default, everything lives in memory. The README is explicit about this, and the defaults confirm it: a fresh `StorageContext` at [L53](https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/storage/storage_context.py#L53) holds a [SimpleDocumentStore, L20](https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/storage/docstore/simple_docstore.py#L20) for nodes, a [SimpleVectorStore, L64](https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/vector_stores/simple.py#L64) for embeddings, and an index store — all plain dictionaries. `SimpleVectorStore.add()` at [L174](https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/vector_stores/simple.py#L174) appends vectors to a dict, and its similarity `query()` at [L244](https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/vector_stores/simple.py#L244) does a brute-force cosine comparison.

`StorageContext.from_defaults()` at [L74](https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/storage/storage_context.py#L74) is the assembly point: pass it a vector store, docstore, or persist directory and it wires the right combination together. Its `persist()` at [L151](https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/storage/storage_context.py#L151) writes each store under `./storage`, which is what `index.storage_context.persist()` reaches through the index. Swapping in a production vector store is therefore a matter of constructing a different `StorageContext` — the index code never knows the difference. One more ingestion-time convenience: `IngestionCache` at [L17](https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/ingestion/cache.py#L17) records transformed nodes so repeated ingestion runs can skip documents whose content hash has not changed.

## Honest Limits

The README itself is candid that the project's center of gravity has moved. The current focus, it says, is building an AI-powered engine for document parsing and extraction, with LlamaParse as the commercial platform and liteparse as the open-source parser; the OSS framework remains available but is no longer the primary focus. Judge recent commit activity accordingly.

A few other limits worth internalizing before you build on it. The "over 300 integration packages" figure is the README's own count — we did not audit it. Default storage is in-memory, so a process restart without `persist()` discards your vectors. The quickstart path assumes an OpenAI key by default, and the local Ollama path requires you to manually align the tokenizer with your model — get that wrong and chunk sizes are silently mismeasured. On packaging, the wheel ships an `_static` folder containing nltk and tiktoken caches; the README documents a GitHub attestation procedure for verifying those build assets in restricted environments. Finally, the project's own ParseBench and ExtractBench cover parsing and extraction quality, not RAG accuracy — we found no framework-level retrieval benchmarks in the README, and we decline to invent any.

Related reading on the model side of the stack: our [transformers source tour](/transformers/), the tokenizer internals behind [tokenizers](/tokenizers/), and weight serialization in [safetensors](/safetensors/).

<script type="application/ld+json">
{ "@context": "https://schema.org", "@type": "TechArticle", "headline": "LlamaIndex Source Tour: How the Data Framework for LLM Apps Is Assembled", "description": "A guided tour of the llama_index repository: ingestion, indices, retrievers, response synthesis, and storage inside llama-index-core.", "datePublished": "2026-10-29", "author": { "@type": "Person", "name": "PyShine" }, "publisher": { "@type": "Organization", "name": "PyShine" }, "keywords": "LlamaIndex, llama-index-core, RAG, VectorStoreIndex, Python" }
</script>
