---
layout: post
title: "PhoneInfoga: Scanning Phone Numbers With Free Resources - Inside sundowndev/phoneinfoga"
description: "An engineering tour of PhoneInfoga, the Go-based OSINT framework that parses any phone number into E164 form and fans it out through five concurrent scanners, from offline parsing to Google dorks, Numverify, and OVH."
date: 2026-10-07
header-img: "img/post-bg.jpg"
permalink: /phoneinfoga/
featured-img: "ai-coding-frameworks/ai-coding-frameworks"
image: https://pyshine.com/assets/img/diagrams/phoneinfoga/sundowndev-phoneinfoga-architecture.svg
tags: [OSINT, Golang, Security, Open Source]
categories: [AI, Open Source]
keywords: phoneinfoga, phone number osint, golang, cobra, gin, open source, scanners, google dorks
author: "PyShine"
---

A phone number is one of the few identifiers that crosses every boundary: it works on social networks, delivery apps, bank accounts, and paper business cards alike. PhoneInfoga, a Go project by sundowndev licensed under GPLv3, takes that observation and builds an information-gathering framework around it. Its own README describes it as "one of the most advanced tools to scan international phone numbers", first collecting basics such as country, area, carrier, and line type, then trying to find the VoIP provider or identify the owner with external APIs, phone books, and search engines. The codebase studied here is version 2 of the tool, a module named `github.com/sundowndev/phoneinfoga/v2` built with Go 1.20, cobra for the CLI, and gin for the web server.

What makes PhoneInfoga worth reading is its honesty, stated right on the project page. An "Anti-features" section declares what the tool will never do: it does not track a phone or its owner in real time, it does not get the precise location of a device, and it does not hack anything. It also warns that it "doesn't automate everything, it's just there to help investigating on phone numbers" and does not claim to provide verified data. The README further notes that the project is stable but unmaintained. That combination of capability and candor is rare, and it shapes the architecture: PhoneInfoga is a framework of small, replaceable scanners rather than a magic oracle.

This article reads the repository purely as a software engineering study. Phone number lookups against public directories are a recognized OSINT technique used by fraud analysts and researchers, but the same capability can feed harassment or doxxing, so the project's own disclaimer applies here too: verify what the tool reports, respect the terms of service of every API involved, respect privacy law, and use it only for lawful purposes on numbers you are entitled to investigate.

<div style="overflow-x:auto;"><img src="https://pyshine.com/assets/img/diagrams/phoneinfoga/sundowndev-phoneinfoga-overview-architecture.svg" alt="PhoneInfoga overview architecture diagram" style="max-width:100%;"></div>
<p><em>PhoneInfoga's end-to-end shape: a cobra CLI parses a number, a scanner library fans it out through five sources behind a DryRun gate, and results land in a tagged console writer or a gin-served REST API with an embedded web client.</em></p>

Reading the overview from left to right:

- **CLI entry (`main.go`, `cmd/`)** — cobra registers `scan`, `serve`, and `version` subcommands on a root command. See [root.go](https://github.com/sundowndev/phoneinfoga/blob/master/cmd/root.go).
- **Number parsing (`lib/number/number.go`)** — `NewNumber()` turns any input into a struct carrying E164, local, international, and carrier forms. See [number.go](https://github.com/sundowndev/phoneinfoga/blob/master/lib/number/number.go).
- **Scanner library (`lib/remote/remote.go`)** — `Library` holds registered scanners and runs one goroutine per scanner per scan. See [remote.go](https://github.com/sundowndev/phoneinfoga/blob/master/lib/remote/remote.go).
- **Data sources (`lib/remote/`)** — five built-in scanners: local, numverify, googlesearch, googlecse, and ovh.
- **Output (`lib/output/console.go`)** — struct-tag-driven console writer. See [console.go](https://github.com/sundowndev/phoneinfoga/blob/master/lib/output/console.go).
- **Server (`web/v2/api/server/server.go`)** — a gin REST API plus an embedded web client. See [server.go](https://github.com/sundowndev/phoneinfoga/blob/master/web/v2/api/server/server.go).

## Why You Need This

The first value is normalization. Human input like `+1 (555) 444-1212`, `33679368229`, or a locally formatted string all mean different things to different APIs. PhoneInfoga funnels everything through Google's libphonenumber via the `nyaruka/phonenumbers` Go port: `NewNumber()` prepends a plus sign, resolves the country with an ISO 3166 table (`ParseCountryCode`), and produces a struct with `Valid`, `RawLocal`, `Local`, `E164`, `International`, `CountryCode`, `Country`, and `Carrier` fields. Every scanner downstream consumes that single struct, which is why the tool can accept any format the docs show.

The second value is the scanner contract. Each data source implements a four-method interface — `Name`, `Description`, `DryRun`, and `Run` — and `DryRun` acts as a preflight gate. The numverify scanner refuses to run when `NUMVERIFY_API_KEY` is unset; the Google CSE scanner requires `GOOGLECSE_CX` plus `GOOGLE_API_KEY`; the OVH scanner only accepts country codes 33, 32, 44, 34, and 41, the zones the OVH Telecom API covers. Scanners that cannot contribute simply stay silent instead of polluting results with errors, and third-party scanners can be loaded at runtime from compiled Go plugins with `--plugin ./custom_scanner.so`.

The third value is the delivery surface. The same engine is reachable three ways: as a one-shot CLI command, as a REST API with a documented Swagger spec, and as a browser client whose built distribution is compiled into the server binary with `go:embed`. A Docker image `sundowndev/phoneinfoga` wraps all of it. For teams building fraud-detection or verification tooling, that means the scan logic can be embedded as Go modules, scripted over HTTP, or operated interactively without forking the project.

## How It Works

<div style="overflow-x:auto;"><img src="https://pyshine.com/assets/img/diagrams/phoneinfoga/sundowndev-phoneinfoga-architecture.svg" alt="PhoneInfoga detailed architecture diagram" style="max-width:100%;"></div>
<p><em>The full anatomy: cobra commands with dotenv loading, number normalization, the scanner framework with filter and plugin registry, five concrete scanners backed by two supplier clients, and the gin server with v2 and legacy routes.</em></p>

### Understanding the Architecture

**Cobra commands and environment.** `main.go` executes the root command, and `cmd/` registers three subcommands. `scan -n <number>` loads environment variables with godotenv — `.env` by default, more via `--env-file` — validates the input, and runs the library. `serve` starts the HTTP server on port 5000 with an optional `--no-client` flag for API-only operation, and `version` prints build info. Scanner selection is configurable per run with `-D`/`--disable`, which feeds a `filter.Engine` whose `Match()` skips scanners by name.

**Concurrent fan-out with panic isolation.** The `Library` in `lib/remote/remote.go` is the concurrency heart. `Scan()` starts one goroutine per registered scanner, synchronized with a `sync.WaitGroup`, and each goroutine is wrapped in a recover clause so a panicking scanner degrades to a named error instead of crashing the process. Results and errors land in two maps guarded by an `sync.RWMutex`. The DryRun gate runs first; only scanners that pass proceed to `Run`, whose return value is stored under the scanner's name.

**Five built-in scanners.** `local` simply reformats the parsed number offline. `numverify` calls the Numverify validation API through a supplier client and reports validity, carrier, location, and line type. `googlesearch` generates search-engine dorks with the author's `dorkgen` library, organized into five buckets: social media (Facebook, Twitter, LinkedIn, Instagram, VK), disposable SMS providers (twenty-one sites), reputation and complaint directories, individual-lookup phone books, and general queries including a compound dork matching the number inside documents of thirteen file extensions from doc and pdf to xls. `googlecse` executes the same style of dorks for real through Google's Custom Search JSON API, paginating results up to a configurable cap (10 by default, 100 maximum via `GOOGLECSE_MAX_RESULTS`) and translating HTTP 429 into a clear rate-limit error. `ovh` queries the OVH Telecom REST API to check whether a number belongs to a known number range, returning the range, city, and zip code for the five supported countries.

**Suppliers as seams.** The HTTP-calling parts of numverify and OVH are isolated behind supplier interfaces in `lib/remote/suppliers`, which is also where the project's test suite mocks external APIs with gock. This seam is what keeps the scanners testable and the API keys in one place, read through `ScannerOptions.GetStringEnv` so options can come from the call site or the environment — the REST API passes credentials per request, while the CLI reads them from `.env`.

**Server and API.** `web/v2/api` exposes POST `/v2/numbers` to register a number, POST `/v2/scanners/:scanner/dryrun` and `/run` to invoke one scanner, and GET `/v2/scanners` to list them. The legacy `/api` group keeps the older GET endpoints for validate and per-scanner scans. `web/client.go` embeds the compiled Vue client into the binary, so `phoneinfoga serve` ships a complete browser UI from a single executable, and the Swagger document in `web/docs/swagger.yaml` publishes the full contract.

**End to end.** A scan flows like this: cobra parses `scan -n`, godotenv loads credentials, the input is validated and normalized into a `Number`, `InitScanners` registers the five built-ins plus any plugins, the filter drops disabled scanners, one goroutine per scanner passes its DryRun gate and calls the relevant supplier or dork builder, and everything is written out through the struct-tag-driven console writer or served back over the REST API.

## Advantages

- **One normalization layer.** Every format converges to a single E164-carrying struct, so scanners stay format-agnostic.
- **Panic-isolated concurrency.** One goroutine per scanner with recover and mutex-guarded result maps means a failing source never breaks the scan.
- **Declarative availability.** DryRun gates make missing API keys or unsupported countries explicit and silent instead of error-noisy.
- **Plugin extensibility.** Compiled Go plugins plug into the same Scanner interface without touching core code.
- **Three delivery modes.** CLI, REST API, and embedded web client come from one codebase and one Docker image.
- **Test-friendly seams.** Supplier interfaces plus gock mocking cover the network edge with golden-file console output tests.

## Benefits

- **Fraud and spam triage.** Reputation and complaint-directory dorks surface whether a number has been reported before you call it back.
- **VoIP detection.** Carrier and line-type data from Numverify helps distinguish disposable VoIP numbers from subscriber lines.
- **Documentation-friendly.** The Swagger spec and anti-features section set exact expectations for what the tool can and cannot reveal.
- **Lightweight operations.** A static Go binary with an embedded UI deploys anywhere Docker runs, with `.env` for credentials.
- **Honest data posture.** Results are explicitly unverified leads, which keeps the tool useful for investigation rather than false certainty.
- **Educational clarity.** The scanner interface, filter engine, and supplier pattern form a compact case study in Go framework design.

## Usage

Install with the project script, Homebrew, or Go:

```bash
bash <( curl -sSL https://raw.githubusercontent.com/sundowndev/phoneinfoga/master/support/scripts/install )
brew install phoneinfoga
```

Scan a number in any accepted format:

```bash
phoneinfoga scan -n "+1 (555) 444-1212"
phoneinfoga scan -n "+33 06 79368229"
phoneinfoga scan -n "33679368229"
```

Enable key-gated scanners through environment or an env file:

```bash
NUMVERIFY_API_KEY=<key> phoneinfoga scan -n +4176418xxxx
phoneinfoga scan -n +4176418xxxx --env-file=.env.local
```

Load a custom scanner plugin:

```bash
phoneinfoga scan -n +4176418xxxx --plugin ./custom_scanner.so
```

Serve the web client and REST API, in a terminal or via Docker:

```bash
phoneinfoga serve
phoneinfoga serve -p 8080
docker run --rm -it -p 5000:5000 sundowndev/phoneinfoga serve
phoneinfoga serve --no-client
```

## Conclusion

PhoneInfoga is a reminder that a good OSINT tool is mostly plumbing done well: normalize the identifier once, define a small contract for data sources, isolate failures, and expose the result through whichever interface the user needs. Its anti-features list is as thoughtfully engineered as its scanner library, drawing a clear line between footprinting a number and pretending to track a person. The project's unmaintained status means its API-based scanners will decay over time, but the architecture remains a clean template for building extensible reconnaissance frameworks in Go.

Links:

- Repository: [https://github.com/sundowndev/phoneinfoga](https://github.com/sundowndev/phoneinfoga)
- Documentation: [https://sundowndev.github.io/phoneinfoga/](https://sundowndev.github.io/phoneinfoga/)
- Scanner library: [remote.go](https://github.com/sundowndev/phoneinfoga/blob/master/lib/remote/remote.go)
- Number parsing: [number.go](https://github.com/sundowndev/phoneinfoga/blob/master/lib/number/number.go)
