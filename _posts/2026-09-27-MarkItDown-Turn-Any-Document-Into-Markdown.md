---
layout: post
title: "MarkItDown: Turn Any Document Into Markdown for Your LLMs"
description: "MarkItDown is Microsoft's Python tool that converts PDF, Word, PowerPoint, Excel, images, audio, and web pages into Markdown for LLM consumption. A source-level tour of microsoft/markitdown: the converter registry, StreamInfo content sniffing, per-format converters, the plugin system, and the MCP server."
date: 2026-09-27
header-img: "img/post-bg.jpg"
permalink: /MarkItDown-Turn-Any-Document-Into-Markdown/
featured-img: ai-coding-frameworks/ai-coding-frameworks
image: https://pyshine.com/assets/img/diagrams/markitdown/microsoft-markitdown-architecture.svg
tags:
  - Python
  - Markdown
  - LLM
  - Open Source
  - Microsoft
categories: [AI, Open Source]
keywords: "MarkItDown, microsoft markitdown, convert files to markdown, markitdown python, document conversion for LLMs, markitdown converters, magika content detection, markitdown-mcp, MCP server, RAG ingestion, pdf to markdown, docx to markdown, pptx to markdown, xlsx to markdown, markitdown plugins"
author: "PyShine"
---

Your knowledge lives in PDFs, Word contracts, PowerPoint decks, Excel exports, Outlook messages, and EPUB books. Your LLM speaks text. Between those two facts sits an uncomfortable amount of glue code: one library to parse PDFs, another for DOCX, a third for spreadsheets, each with its own API, its own failure modes, and its own idea of what to do with a table. [MarkItDown](https://github.com/microsoft/markitdown), a lightweight Python utility from Microsoft, collapses all of that into a single call that returns Markdown.

The choice of Markdown is deliberate. Mainstream LLMs were trained on enormous amounts of Markdown-formatted text, so they read it natively and often emit it unprompted. Markdown is nearly plain text - minimal markup, high token efficiency - yet it still preserves the structure that matters: headings, lists, tables, and links. MarkItDown converts PDF, PowerPoint, Word, Excel, images (EXIF metadata and OCR), audio (metadata and speech transcription), HTML, CSV, JSON, XML, ZIP files, YouTube URLs, and EPubs into exactly that representation.

This post walks through the tool as it is actually implemented in the repository: the three ways to call it, the converter registry that routes your file to the right handler, the content-sniffing step that decides what your file really is, the per-format converters, the third-party plugin system, and the companion MCP server that exposes the whole thing to coding agents.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/markitdown/microsoft-markitdown-overview-architecture.svg" alt="Architecture overview of the microsoft/markitdown repository" style="max-width:100%;height:auto;" />
</div>

*High-level overview: the CLI, the Python API, and the MCP server all funnel into one conversion engine, which dispatches by priority to built-in converters and optionally loads plugins or calls cloud services.*

Reading the overview from left to right: there are three front doors into the same engine. The `markitdown` command-line program and the public Python API both construct a `MarkItDown` object, while the separate `markitdown-mcp` package wraps the engine as a Model Context Protocol tool that agents can call. Everything converges on the conversion core - a converter registry plus a dispatch loop - which first infers what the input actually is (extension, MIME type, charset) and then walks a priority-sorted list of converters. Each converter implements the same small contract: an `accepts` check followed by a `convert` call that returns a result object carrying Markdown text and an optional title. The format layer is pluggable in both directions: built-in converters ship with the package, third-party plugins can register more at runtime, and cloud services such as Azure Document Intelligence or a vision LLM slot in as optional upgrades. Keep this layering in mind - the rest of the post zooms into each box.

## Why You Need This

If you have ever built a search index, a RAG system, or any kind of document-analysis workflow, you know the shape of the problem. The raw bytes of a PDF mean nothing to a language model. Strip the text out with a naive extractor and you lose the table structure, the heading hierarchy, and the reading order - and downstream chunking quality collapses with them. Worse, every format demands a different library, so a "simple" ingestion script becomes a museum of half-compatible dependencies.

MarkItDown attacks this from the direction that matters for LLM work. It is not trying to produce pixel-perfect documents for humans to read - the README says so explicitly. It is trying to produce the representation a language model consumes best, and that changes the trade-offs: preserving headings, lists, tables, and links matters more than fonts and margins. Because the output is Markdown rather than flat text, an LLM can tell a table from a paragraph, a heading from a shouted line, and a link target from its anchor text.

The second reason is uniformity. Whether the input is a local spreadsheet, an HTTPS URL, a raw byte stream from an upload endpoint, or a base64 data URI, you call the same object and get back the same result type. That uniformity is what makes the tool composable: the command line uses it, the Python API uses it, and the MCP server uses it - all without special-casing formats anywhere above the engine.

Finally, there is the ecosystem question. File formats never stop arriving, so the repository is designed to be extended without being forked: new formats are expected to arrive as third-party plugins that register themselves through Python entry points, and the maintainers keep the core package focused on fidelity improvements to the formats it already covers.

## How It Works

The diagram below maps the real subsystems of the repository and how they connect, from the user-facing entry points down to the individual converters and the services some of them call.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/markitdown/microsoft-markitdown-architecture.svg" alt="Detailed architecture of MarkItDown, from CLI, Python API and MCP server through the conversion engine to individual converters and external services" style="max-width:100%;height:auto;" />
</div>

*Detailed architecture: three interfaces feed the conversion engine, which builds StreamInfo guesses and dispatches through the DocumentConverter contract to per-format converters; utilities, plugins, and external services hang off the sides.*

### Understanding the Architecture

**Three front doors.** The CLI lives in `packages/markitdown/src/markitdown/__main__.py` and is registered as a console script named `markitdown` in the package's `pyproject.toml`. It is a thin `argparse` program: it accepts a filename (or reads stdin), optional hints like `-x` for extension, `-m` for MIME type, and `-c` for charset, plus switches for Document Intelligence (`-d`), Content Understanding (`--use-cu`), and plugins (`-p`). The public Python API is defined by `packages/markitdown/src/markitdown/__init__.py`, which exports `MarkItDown`, `DocumentConverter`, `DocumentConverterResult`, `StreamInfo`, and the exception types - everything a third party needs to build on the library. The third door is the MCP server in `packages/markitdown-mcp/src/markitdown_mcp/__main__.py`: it exposes exactly one tool, `convert_to_markdown(uri)`, accepting `http:`, `https:`, `file:`, and `data:` URIs, served over STDIO by default or Streamable HTTP and SSE via Starlette and uvicorn when started with `--http`.

**The conversion engine.** The class `MarkItDown` in `packages/markitdown/src/markitdown/_markitdown.py` is the heart of the system. It maintains a list of converter registrations, each pairing a converter with a numeric priority. Two constants set the stakes: `PRIORITY_SPECIFIC_FILE_FORMAT` (0.0) for format-specific converters and `PRIORITY_GENERIC_FILE_FORMAT` (10.0) for near catch-alls like plain text, HTML, and ZIP. Lower values are tried first, and because the sort is stable, converters registered later at the same priority win - which is exactly how a plugin can slot itself precisely between the built-ins.

**Knowing what the input is.** Every public entry method - `convert_local`, `convert_uri`, `convert_response`, `convert_stream` - reduces to the same shape: build one or more `StreamInfo` guesses and hand them to the internal `_convert` loop. `StreamInfo` (in `_stream_info.py`) is a small frozen dataclass describing mimetype, charset, extension, filename, local path, and URL. The guessing is layered: an initial guess comes from the file extension or HTTP headers, then the engine calls Google's `magika` library to identify the content type from the bytes themselves, and for text-like content it runs `charset-normalizer` over a 64 KiB sample to pin down the encoding. If the sniffed type disagrees with the claimed type, both guesses are kept and tried in order - a small design decision that quietly rescues mislabeled uploads.

**The dispatch loop.** `_convert` sorts the registry by priority and, for each guess, asks each converter two questions. First `accepts()` - a cheap check based on stream info, with the stream position restored afterward - then `convert()`, the real work. A converter that throws is recorded as a `FailedConversionAttempt` and the loop moves on, so a flaky optional dependency does not kill the conversion if another converter can cope. Only when every guess has been tried does the engine raise: `UnsupportedFormatException` if nothing even claimed the file, `FileConversionException` with the collected attempts if converters tried and failed. Successful output gets normalized - line endings flattened, triple newlines collapsed - before being returned as a `DocumentConverterResult` (the contract from `_base_converter.py`).

**The format converters.** Each file under `packages/markitdown/src/markitdown/converters/` handles one family. The PDF converter builds on `pdfminer.six` and `pdfplumber`, reconstructs Markdown tables, and even repairs MasterFormat-style partial numbering that PDF extraction splits across lines. Word documents go through `mammoth`, PowerPoint through `python-pptx` (with speaker notes and slide-image captions), Excel through `pandas` and `openpyxl` with sheet-level Markdown tables, and Outlook `.msg` files through `olefile`. HTML is cleaned up with BeautifulSoup and walked by a customized `_CustomMarkdownify` in `_markdownify.py` - the same walker the Wikipedia and RSS converters reuse. The image converter extracts EXIF metadata (optionally via `exiftool`) and, if you supply an `llm_client` and `llm_model`, asks a vision model to describe the picture. The audio converter transcribes WAV and MP3 through the `transcribe_audio` helper built on `pydub` and `SpeechRecognition`. Two converters deserve special mention: the ZIP converter is constructed with a reference back to the engine itself, so it iterates archive members and recursively converts each one through the full registry; and the Document Intelligence converter is only registered when you pass a `docintel_endpoint`, sitting at the top of the stack so it preempts the offline PDF path.

**Plugins.** The plugin system in `_markitdown.py` scans Python's `markitdown.plugin` entry-point group, lazily and only when enabled - plugins are off by default. A plugin is a package that exposes a `register_converters(markitdown, **kwargs)` function; the sample RTF plugin in `packages/markitdown-sample-plugin/src/markitdown_sample_plugin/_plugin.py` is about as small as one can be, and `markitdown-ocr` in `packages/markitdown-ocr` shows the pattern scaling up, layering LLM-vision OCR over the PDF, DOCX, PPTX, and XLSX converters using the same `llm_client`/`llm_model` options the engine already carries.

**A conversion in flight.** Follow a single command - `markitdown report.pdf` - through the boxes above. The CLI parses arguments, constructs `MarkItDown()`, and calls `convert()`, which sees a plain string with no URL scheme and routes to `convert_local`. The engine opens the file, builds a base guess from the `.pdf` extension, lets `magika` confirm `application/pdf` from the header bytes, and sorts the registry. The PDF converter's `accepts()` agrees, its `convert()` streams the document through pdfminer and pdfplumber, and the returned Markdown is normalized and printed to stdout. Nothing in that chain requires the caller to know anything about PDFs - and if the file had secretly been a DOCX renamed to `.pdf`, the sniffing layer and the fallback loop would have given the Word converter a chance instead of failing outright.

## Advantages

- **One contract, many formats.** Every converter implements the same `accepts`/`convert` interface over the same `StreamInfo`, so adding or replacing a converter never ripples upward. The engine does not know what a spreadsheet is.
- **Sniff, do not trust.** Content type is determined by `magika` over the actual bytes, cross-checked against extensions and headers, with charset detection for text. Mislabeled files get multiple guesses instead of a hard failure.
- **Graceful degradation.** Failed conversions are recorded, not raised; the loop moves to the next candidate converter and only reports once every path has been exhausted - with the attempts attached for debugging.
- **Local-first, cloud-optional.** The default path runs entirely offline on your own compute. Azure Document Intelligence, Azure Content Understanding, and LLM image captions are opt-in upgrades that slot into the same registry without changing call sites.
- **Extensible by entry point.** Third-party plugins register through the standard `markitdown.plugin` entry-point group, are disabled until asked for, and can position themselves anywhere in the priority order - including ahead of the built-ins.
- **Agent-ready by design.** The `markitdown-mcp` package exposes the engine as an MCP tool over STDIO, Streamable HTTP, or SSE, and the security documentation nudges callers toward the narrowest `convert_*` method for their use case - a rare, explicit acknowledgment that ingestion tools run on untrusted input.

## Benefits

- **Hours of glue code, deleted.** One `pip install` replaces a drawer full of single-format parsers and the try/except scaffolding around them.
- **Better retrieval quality.** Because headings, tables, and lists survive conversion, the chunks you feed a RAG system carry real structure instead of flattened noise.
- **Lower token bills.** Markdown is close to plain text, so you pay for content rather than markup ceremony - the README's own argument for choosing it.
- **Privacy by default.** Conversion happens in your process with no network calls unless you explicitly configure a cloud endpoint or an LLM client.
- **Predictable failure behavior.** Unsupported files raise a distinct exception from files that simply failed to convert, so batch jobs can route errors intelligently instead of dying on the first weird input.
- **Three integration surfaces, one engine.** Shell one-liners for quick jobs, a typed Python API for applications, and an MCP server for coding agents - all guaranteed to behave identically because they share the code you just read about.

## Usage

Install the package with every optional format dependency:

```bash
pip install 'markitdown[all]'
```

Or install only what you need - PDF, Word, and PowerPoint, for example:

```bash
pip install 'markitdown[pdf, docx, pptx]'
```

The other available extras are `xlsx`, `xls`, `outlook`, `az-doc-intel`, `az-content-understanding`, `audio-transcription`, and `youtube-transcription`. Then convert from the command line:

```bash
markitdown path-to-file.pdf > document.md
markitdown path-to-file.pdf -o document.md
cat path-to-file.pdf | markitdown
```

From Python, the whole API is two lines:

```python
from markitdown import MarkItDown

md = MarkItDown(enable_plugins=False)
result = md.convert("test.xlsx")
print(result.markdown)
```

Hand it an OpenAI-compatible client and it will caption images and slide graphics:

```python
from markitdown import MarkItDown
from openai import OpenAI

md = MarkItDown(llm_client=OpenAI(), llm_model="gpt-4o")
result = md.convert("example.jpg")
print(result.markdown)
```

Plugins ship separately and stay disabled until requested. List what is installed, then enable them for a run:

```bash
markitdown --list-plugins
markitdown --use-plugins path-to-file.pdf
```

For scanned or layout-heavy PDFs, you can route conversion through Azure Document Intelligence by passing an endpoint (or setting `MARKITDOWN_DOCINTEL_ENDPOINT` in the environment):

```bash
markitdown path-to-file.pdf -o document.md -d -e "<document_intelligence_endpoint>"
```

To let a coding agent convert documents for you, install the MCP server package and point your MCP client at it:

```bash
pip install markitdown-mcp
markitdown-mcp
```

By default it speaks STDIO, which is what local MCP clients expect. For an HTTP-based client, start it with:

```bash
markitdown-mcp --http --host 127.0.0.1 --port 3001
```

And for a dependency-free one-off, the repository's Dockerfile builds a container that converts from stdin:

```bash
docker build -t markitdown:latest .
docker run --rm -i markitdown:latest < ~/your-file.pdf > output.md
```

## Conclusion

MarkItDown succeeds because it picks its battle honestly: it does not try to be a beautiful document renderer, it tries to be the shortest correct path from "some file on disk" to "structured text an LLM can reason about." The source shows a small, disciplined architecture doing the work - a priority-ordered converter registry, byte-level content sniffing that distrusts file extensions, a fallback loop that records failure instead of amplifying it, and an extension story built on standard Python entry points. Whether you are indexing a document archive, feeding a retrieval system, or just tired of opening attachments to copy text by hand, clone the repository, run one file through it, and look at the Markdown you get back. The output is the pitch.

**Links:**

- Repository: [https://github.com/microsoft/markitdown](https://github.com/microsoft/markitdown)
- PyPI: [https://pypi.org/project/markitdown/](https://pypi.org/project/markitdown/)
- MCP server on PyPI: [https://pypi.org/project/markitdown-mcp/](https://pypi.org/project/markitdown-mcp/)
