---
layout: post
title: "Stirling PDF: Self-Hosted PDF Processing From Request to Pipeline - Inside Stirling-Tools/Stirling-PDF"
description: "A guided source tour of Stirling-Tools/Stirling-PDF, the open-source self-hosted PDF platform built on Java Spring Boot, Apache PDFBox, JPDFium, LibreOffice, OCRmyPDF and qpdf. We trace the full request-to-pipeline flow: the annotation-driven REST controller layer, the auto-job runtime, the PipelineProcessor executor, and the native tool integrations that do the heavy lifting."
date: 2026-09-29
header-img: "img/post-bg.jpg"
permalink: /Stirling-PDF-Request-to-Pipeline-Source-Tour/
featured-img: "ai-coding-frameworks/ai-coding-frameworks"
image: https://pyshine.com/assets/img/diagrams/stirling-pdf/stirling-tools-stirling-pdf-architecture.svg
tags:
  - PDF
  - Java
  - Spring Boot
  - Self-Hosted
categories: [AI, Open Source]
keywords: "stirling pdf, self-hosted pdf tools, pdf pipeline automation, spring boot pdf server, apache pdfbox, jpdfium, libreoffice automation, ocrmypdf, qpdf compression, pdf merge api, open source pdf editor, docker pdf server, java pdf toolkit, pdf workflow"
author: "PyShine"
---

If you have ever dropped a stack of PDFs into a web app, ticked a few boxes, and watched a clean merged-and-OCR'd document come out the other side, you have met the kind of machinery that most people never see. Stirling-Tools/Stirling-PDF is one of the most popular open-source implementations of that machinery: a self-hosted PDF platform that runs as a desktop app, in the browser, or on your own server, with a private REST API for everything it can do. The README describes it bluntly as a powerful, open-source PDF editing platform for editing, signing, redacting, converting, and automating PDFs without sending documents to external services.

What makes Stirling PDF interesting as a codebase is not just the breadth of its toolbox — the README advertises more than fifty tools covering merge, split, sign, redact, convert, OCR, and compress — but how those tools are wired together. The backend is Java on Spring Boot, but the repository has grown into a multi-module Gradle workspace: the core app lives in `app/core`, shared infrastructure in `app/common`, and separately licensed `app/proprietary` and `app/saas` modules (selected through the `STIRLING_FLAVOR` setting in `settings.gradle`) sit on top. A React editor under `frontend/editor` replaces the old server-rendered UI, and a separate Python `engine/` directory holds a FastAPI-based AI document service.

That combination — a big REST surface, a no-code pipeline system, and a layer cake of native libraries and external CLI engines — makes the request-to-pipeline flow genuinely worth a source tour. In this post we follow one HTTP request from the React editor into the controller layer, through an AspectJ-powered job runtime, into the pipeline executor that loops back over its own API, and finally down into the PDFBox, JPDFium, LibreOffice, OCRmyPDF, and qpdf integrations that do the real work.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/stirling-pdf/stirling-tools-stirling-pdf-overview-architecture.svg" alt="Architecture overview of the Stirling-Tools/Stirling-PDF repository" style="max-width:100%;height:auto;" />
</div>

*Overview of Stirling-PDF: the React editor talks to the Spring Boot controller layer, every endpoint flows through the auto-job aspect and executor, the pipeline engine re-dispatches steps over an internal loopback client, and PDF work lands either in in-JVM PDF libraries or in managed external processes.*

Reading the overview from left to right: the React editor (`frontend/editor`) issues multipart POSTs against the Spring Boot tool controllers under `app/core/src/main/java/stirling/software/SPDF/controller/api`, which are the public `/api/v1/*` surface of the server. Every endpoint declaration passes through the `AutoJobAspect` in `app/common`, which can turn a plain controller method into an async, queueable, retryable job handled by `JobExecutorService`. The pipeline controller and its `PipelineProcessor` form a special kind of client: instead of calling tool logic directly, they build multipart requests and post them back into the same controller surface via `InternalApiClient`. At the bottom, controllers either call in-JVM PDF utilities (PDFBox and a native PDFium binding) or hand files to `ProcessExecutor`, which runs LibreOffice, Tesseract/OCRmyPDF, qpdf, Calibre, and friends as carefully metered external processes.

## Why You Need This

Anyone who works with documents at any volume eventually hits the same wall: PDF tools that are online-only, metered per page, or locked to one operating system. Stirling PDF solves the ownership problem first. It runs where you run it — the README's quick start is a single `docker run` command — and the project's stated pitch is that documents never need to leave your infrastructure. For anyone handling contracts, medical records, financial statements, or anything else sensitive, that single property outweighs most feature checklists.

The second problem is fragmentation. A typical document workflow touches several tools: convert an Office file to PDF, OCR a scan, rotate and stamp pages, merge everything, compress the result, then redact or sign it. Stirling PDF puts all of those behind one coherent REST API and one web UI, with a consistency layer you can see directly in the source: shared marker annotations such as `GeneralApi` and `PipelineApi` in `app/common/src/main/java/stirling/software/common/annotations/api` stamp every controller with the right `/api/v1/...` base path, OpenAPI tags, and response conventions, so all fifty-plus tools behave the same way on the wire.

The third problem is automation. Clicking through a UI is fine for one document; it is useless for a folder that fills up every morning. Stirling PDF addresses this on two fronts. The pipeline system lets users compose JSON-described sequences of operations and execute them from the UI or the API, and `PipelineDirectoryProcessor` (`app/core/src/main/java/stirling/software/SPDF/controller/api/pipeline/PipelineDirectoryProcessor.java`) adds classic watched-folder automation, scanning configured directories on a schedule, waiting for files to be ready, running the pipeline, and moving results to a finished folder.

Finally, there is the extensibility problem. Because the tool surface is uniform and self-documented (the pipeline validates operations against `ApiDocService`, which reflects over the API definitions), adding a tool benefits everything at once: the UI, the REST API, and the pipeline engine all pick it up. The repository even documents the frontend recipe for this in `ADDING_TOOLS.md`, from the `useToolOperation` hook pattern to tool registration.

## How It Works

The cleanest way into this codebase is to follow a request from the browser to the PDF engines and back.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/stirling-pdf/stirling-tools-stirling-pdf-architecture.svg" alt="Detailed architecture of the Stirling-Tools/Stirling-PDF repository" style="max-width:100%;height:auto;" />
</div>

*Detailed architecture of Stirling-PDF: frontend hooks and the SPA router feed the annotation-decorated controller layer; the auto-job aspect, job executor, and task manager wrap execution; the pipeline controller, processor, and loopback client chain tool calls; external engines run under ProcessExecutor with shared temp-file and response utilities.*

### Understanding the Architecture

**The entry point wires four worlds together.** `SPDFApplication` (`app/core/src/main/java/stirling/software/SPDF/SPDFApplication.java`) is a standard Spring Boot main class, but its `scanBasePackages` list is the map of the whole product: `stirling.software.SPDF` (the core app), `stirling.software.common` (shared runtime), `stirling.software.proprietary`, and `stirling.software.saas`. The Gradle build in `settings.gradle` selects a flavor — `core`, `proprietary` (the default), or `saas` — by mapping the logical module names `:stirling-pdf`, `:common`, `:proprietary`, and `:saas` onto those directories. Boot itself handles external configuration: it locates settings and custom settings files through `InstallationPathConfig`, merges them as additional Spring config locations, and only then starts serving.

**The controller layer is annotation-driven and uniform.** Tool controllers are grouped by concern under `app/core/src/main/java/stirling/software/SPDF/controller/api`: `security` for encryption, redaction, and signatures; `converters` for Office, image, HTML, EPUB, and PDF/A conversions; `misc` for OCR, compression, metadata, and stamps; and the top level for general operations like split and rotate. Rather than repeating boilerplate, each class is meta-annotated with markers such as `@GeneralApi` or `@PipelineApi`, which bundle `@RestController`, the `/api/v1/...` request mapping, and an OpenAPI tag. The endpoint methods themselves are declared with `@AutoJobPostMapping` (`app/common/src/main/java/stirling/software/common/annotations/AutoJobPostMapping.java`) — a `@RequestMapping` shortcut that also carries operational metadata: timeout, retry count, progress tracking, queueability, and a mandatory `resourceWeight` used for load control.

**An aspect turns endpoints into jobs.** `AutoJobAspect` (`app/common/src/main/java/stirling/software/common/aop/AutoJobAspect.java`) wraps every annotated method with an AspectJ `@Around` advice. When a client appends `?async=true`, the call is routed into `JobExecutorService` (`app/common/src/main/java/stirling/software/common/service/JobExecutorService.java`), where it can be queued, timed out, and tracked; `TaskManager` records progress that can be polled over REST. The aspect also does quiet, practical work before the controller body ever runs: it resolves a `fileId` reference into an uploaded file via `FileStorage`, and for async jobs it persists a durable copy of the upload so the request's transient multipart data cannot vanish mid-job, propagating MDC logging context into the worker thread so background jobs remain traceable.

**The pipeline executor loops back through its own API.** The interesting design choice lives in `PipelineProcessor` (`app/core/src/main/java/stirling/software/SPDF/controller/api/pipeline/PipelineProcessor.java`). `PipelineController` receives files plus a JSON document that Jackson deserializes into `PipelineConfig` and `PipelineOperation` models, then hands off to the processor. The processor validates each operation against `ApiDocService`, asks `ToolMetadataService` whether the operation takes one file at a time or many, filters the working set by accepted extensions, and then — instead of invoking tool services directly — builds a multipart body and posts it to `InternalApiClient` (`app/common/src/main/java/stirling/software/common/service/InternalApiClient.java`), which performs an authenticated HTTP loopback call into the same `/api/v1/*` controllers a browser would hit. The client enforces a strict endpoint allowlist via a compiled regex, attaches an API key and an `X-Stirling-Automation` header, and groups every step of a run under an `AutomationRunContext` scope. Zip responses are exploded with `ZipExtractionUtils`, and `/api/v1/filter/filter-*` steps simply drop files that do not match, implementing conditional routing inside the pipeline.

**The merge executor shows the native-library strategy.** `MergeController` (`app/core/src/main/java/stirling/software/SPDF/controller/api/MergeController.java`) is the canonical multi-input tool, annotated with `@ToolIO(arity = ToolArity.MISO)` to advertise many-input/one-output behavior. It reorders inputs by an explicit `fileOrder` list or sorts by name, modification date, or PDF title; converts image uploads into PDF pages via `PdfUtils.imageToPdf`; and pre-validates every file by opening it with a native PDFium binding (`stirling.software.jpdfium`). The actual merge uses `PdfMerge.merge` with a rebuilt bookmark tree — including a generated table of contents when requested — and only falls back to Apache PDFBox for the one thing it still does better here: flattening signature fields with `PDAcroForm.flatten` when certificate signatures must be removed. Everything flows through `TempFileManager`, and the merged file returns via `WebResponseUtils` with a `_merged_unsigned.pdf` suffix.

**External engines are quarantined behind a process governor.** Tools that need real applications — LibreOffice for Office conversions, OCRmyPDF and Tesseract for recognition, qpdf and Ghostscript for compression, Calibre for e-books, WeasyPrint and pdftohtml for HTML — all go through `ProcessExecutor` (`app/common/src/main/java/stirling/software/common/util/ProcessExecutor.java`). It keeps a singleton per process type, each gated by a semaphore whose size comes from application properties, so a burst of LibreOffice conversions cannot exhaust the machine. `ConvertOfficeController` shows the pattern end to end: it sanitizes input to prevent SSRF through embedded URLs, tries `unoconvert` first, falls back to `soffice` with an isolated user profile, and always runs under the semaphore and timeout. In the Docker image, LibreOffice is additionally confined by a dedicated sandbox binary. `OCRController` prefers OCRmyPDF and degrades gracefully to a pure Tesseract path, while `CompressController` layers Ghostscript image recompression under a final qpdf structural optimization pass.

**End to end, a pipeline run looks like this:** the React editor posts files and a pipeline JSON to `/api/v1/pipeline/handleData`; `PipelineController` parses the config and materializes uploads into temp-backed resources; `PipelineProcessor` iterates the operations, dispatching each one through the authenticated loopback client into the normal controller layer; controllers do the heavy lifting in-JVM with PDFBox/JPDFium or spawn governed external engines via `ProcessExecutor`; results (unzipped as needed) become the input of the next stage; and `PipelineController` finally streams either a single file or a freshly zipped `output.zip` back to the caller, with every temp file tracked and cleaned by `TempFileManager`. The same choreography is available without HTTP at all — `PipelineDirectoryProcessor` feeds watched folders into the identical processor — and the proprietary `AiWorkflowService` (`app/proprietary/src/main/java/stirling/software/proprietary/service/AiWorkflowService.java`) reuses the same loopback client to execute AI-planned tool steps.

## Advantages

- **Everything under your roof.** The entire platform self-hosts with one Docker command; documents never pass through third-party services, and the REST API is private to your deployment.
- **One uniform tool surface.** Fifty-plus tools share the same annotation-driven controller conventions and `/api/v1/...` namespace, so learning one endpoint teaches you the shape of all of them, and the pipeline can compose any of them.
- **Real automation primitives.** The `@AutoJobPostMapping` framework gives every endpoint optional async execution, queueing, timeouts, retries, progress tracking, and resource-weight-aware admission control — infrastructure most projects bolt on per-feature.
- **Pipeline architecture that reuses the API itself.** `PipelineProcessor` executes steps by calling the public API over loopback, so there is exactly one behavior per tool whether a human, a pipeline, a watched folder, or an AI workflow triggered it.
- **Disciplined process management.** External engines run under per-type semaphores, timeouts, sandboxing, and isolated profiles, which is what makes it safe to embed heavyweight applications like LibreOffice in a web service.
- **Honest open-core boundaries.** The MIT-licensed core (`app/core`, `app/common`) is a complete product on its own, while `app/proprietary`, `app/saas`, and `engine/` are cleanly separated under their own licenses rather than tangled into the open code.

## Benefits

- **Privacy by default for sensitive documents.** Contracts, scans, and records can be OCR'd, redacted, signed, and converted entirely on infrastructure you control.
- **Reduced tool sprawl and licensing cost.** One self-hosted deployment replaces a patchwork of per-tool desktop apps and online converters, with a single API to integrate against.
- **Workflow leverage.** Composable pipelines and watched-folder automation turn multi-step document chores into unattended jobs that run the same way every time.
- **Operational confidence at scale.** Job queueing, retries, progress polling, and semaphore-limited external processes mean the server degrades predictably under load instead of thrashing.
- **A readable blueprint for your own services.** The codebase is a working reference for annotation-driven API design, AspectJ job wrapping, loopback orchestration, and pragmatic multi-module Gradle flavors in Spring Boot.
- **An active, documented extension path.** From `ADDING_TOOLS.md` to the developer guide, the project documents how to add tools and translations, so extending the platform is a supported activity rather than archaeology.

## Usage

The README's quick start runs the full server in Docker; the UI is then available at `http://localhost:8080`:

```bash
docker run -p 8080:8080 docker.stirlingpdf.com/stirlingtools/stirling-pdf
```

For development, the repository uses [Task](https://taskfile.dev/) as its unified command runner — `task dev` starts the editor and backend in watch mode, and `task` lists the most common commands:

```bash
task dev
```

Full installation options (desktop client and Kubernetes among them) are described in the project's documentation guide, and every tool endpoint is catalogued in the public API docs, which means anything you can do in the UI you can also script with `curl` against your own deployment's `/api/v1/...` routes.

## Conclusion

Stirling-PDF earns its reputation less through any single feature than through architecture. The request-to-pipeline flow we traced — React editor, annotation-marked controllers, an aspect that turns endpoints into governed jobs, a pipeline executor that composes tools by speaking the server's own API, and a process governor guarding LibreOffice, Tesseract, qpdf, and friends — is a genuinely reusable pattern for anyone building a tool platform on Spring Boot. The source is candid about its boundaries: an MIT-licensed core that stands alone, with proprietary and SaaS modules clearly fenced off. If you build backend services, it is one of the more instructive Java codebases you can spend an afternoon reading.

Links:

- GitHub repository: [Stirling-Tools/Stirling-PDF](https://github.com/Stirling-Tools/Stirling-PDF)
- Documentation: [docs.stirlingpdf.com](https://docs.stirlingpdf.com)
- API docs: [Stirling PDF Processing API](https://registry.scalar.com/@stirlingpdf/apis/stirling-pdf-processing-api/)
