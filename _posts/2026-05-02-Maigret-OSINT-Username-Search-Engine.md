---
layout: post
title: "Maigret: OSINT Username Search Engine Across 3,000+ Sites"
description: "Learn how Maigret collects a complete dossier on any person by username alone, searching 3,000+ sites with recursive discovery, CAPTCHA bypass, Tor support, and multiple report formats. No API keys required."
date: 2026-05-02
header-img: "img/post-bg.jpg"
permalink: /Maigret-OSINT-Username-Search-Engine/
featured-img: ai-coding-frameworks/ai-coding-frameworks
categories: [OSINT, Security, Python]
tags: [Maigret, OSINT, username search, open source intelligence, Python, security tools, account discovery, digital forensics, privacy, cyber security]
keywords: "how to use Maigret OSINT tool, Maigret username search tutorial, OSINT username finder Python, Maigret vs Sherlock comparison, best OSINT tools 2026, Maigret installation guide, account discovery across social media, open source intelligence username search, Maigret recursive search feature, digital footprint analysis tool"
author: "PyShine"
---

# Maigret: OSINT Username Search Engine Across 3,000+ Sites

Maigret is a powerful open-source intelligence (OSINT) tool that collects a complete dossier on a person using nothing but a username. Created by Soxoj and battle-tested by security professionals worldwide, Maigret checks over 3,000 sites for account presence, extracts profile information, discovers linked accounts through recursive search, and generates comprehensive reports -- all without requiring any API keys.

## How Maigret Searches Across Sites

When you provide a username, Maigret launches an intelligent multi-phase search process that goes far beyond simple account existence checks. The tool queries its database of 3,000+ sites, performs async HTTP requests with built-in CAPTCHA bypass, extracts profile data and linked accounts, then recursively searches for newly discovered usernames and IDs.

![Maigret Search Workflow](/assets/img/diagrams/maigret/maigret-search-workflow.svg)

### Understanding the Search Workflow

The diagram above illustrates Maigret's search pipeline from username input to final reports. Here is how each phase works:

**Username Input** -- You provide one or more usernames to search. Maigret supports three search modes: a default search that checks the top 500 sites ranked by traffic, a full search of all 3,000+ sites with the `-a` flag, and a tagged search that filters sites by category or country using `--tags`.

**Site Database** -- Maigret maintains an auto-updated database of site definitions, fetched from GitHub once every 24 hours. Each entry contains the site URL pattern, detection logic, and metadata about what information can be extracted. If you are offline, Maigret falls back to its built-in database.

**Check Engine** -- The core of Maigret's power. It sends async HTTP requests to each site, checking whether the username exists. The engine includes built-in CAPTCHA bypass and block detection, handling rate limits and anti-bot measures automatically. It can also route requests through Tor, I2P, or any SOCKS/HTTP proxy.

**Profile Extraction** -- When an account is found, Maigret uses its socid_extractor module to pull all available information from the profile page: display names, bios, profile photos, links to other accounts, and site-specific IDs. This is where the real intelligence gathering happens.

**Recursive Search** -- This is Maigret's killer feature. When profile extraction discovers new usernames or IDs (like a Twitter handle found on an Instagram profile), Maigret automatically searches for those as well, building a comprehensive web of connected accounts. This recursive discovery can reveal accounts the target never intended to link.

**Results and Reports** -- All findings are compiled into your choice of report format: HTML with interactive graphs, PDF for documentation, CSV for data analysis, or D3.js interactive graphs for visual exploration.

## Architecture and Components

Maigret is designed as a modular system with multiple access points feeding into a shared core engine. Whether you use the CLI, web interface, Python library, or Telegram bot, the same powerful engine handles the work.

![Maigret Architecture](/assets/img/diagrams/maigret/maigret-architecture.svg)

### Understanding the Architecture

The architecture diagram shows four access points converging on the Maigret Core Engine, which orchestrates four subsystems:

**Access Points:**

- **CLI** -- The primary interface. Run `maigret username` to start searching. Supports all flags and output formats.
- **Web Interface** -- Launch with `maigret --web 5000` for a browser-based UI with interactive result graphs and downloadable reports.
- **Python Library** -- Import `maigret` directly in your Python projects for custom OSINT pipelines and automated workflows.
- **Telegram Bot** -- The `@maigret_search_bot` on Telegram provides a no-install option for quick username lookups.

**Core Subsystems:**

- **Site Database** -- 3,000+ site definitions, auto-updated daily from GitHub. Each entry specifies URL patterns, detection methods, and extractable data points.
- **Site Checker** -- Performs async HTTP requests with intelligent rate limiting, CAPTCHA bypass, and block detection. Routes through proxy/Tor/I2P when configured.
- **Profile Extractor** -- Uses the socid_extractor module to parse profile pages and extract IDs, links, and metadata. This is what enables recursive discovery.
- **Username Permutation** -- With `--permute`, generates likely username variants from multiple inputs (e.g., "john doe" becomes "johndoe", "j.doe", "doe.john") and searches for all of them.

**Report Generator** -- Converts results into HTML, PDF, CSV, Xmind, or interactive D3.js graph formats.

## Features and Capabilities

Maigret's feature set is organized into four categories, each addressing a different aspect of OSINT username investigation.

![Maigret Features Overview](/assets/img/diagrams/maigret/maigret-features-overview.svg)

### Understanding the Features

**Search Capabilities:**

- **3,000+ Sites Database** -- The largest site database of any username search tool, covering social media, forums, coding platforms, dating sites, and more. The default run checks the top 500 sites by traffic; use `-a` for all sites.
- **Recursive Search** -- Automatically discovers and searches for new usernames and IDs found in profiles, building a comprehensive account map.
- **Tag Filtering** -- Narrow your search by category (`--tags photo,dating`) or country (`--tags us`) to focus on relevant sites and reduce scan time.
- **Username Permutation** -- Generate likely username variants from multiple inputs with `--permute`, catching accounts that use different naming conventions.
- **Profile Parsing** -- Use `--parse URL` to extract IDs and usernames from a profile page, then use them to kick off recursive searches.

**Output and Reports:**

- **HTML Reports** -- Rich, formatted reports with embedded images and links, viewable in any browser.
- **PDF Reports** -- Professional documentation-ready reports for case files and reports.
- **CSV and JSON Export** -- Machine-readable formats for data analysis, `--csv` for spreadsheets, `--json ndjson` for programmatic processing.
- **Interactive D3 Graph** -- Visual network graph showing account connections and relationships, generated with `--graph`.
- **Xmind Mind Maps** -- Structured mind map exports for visual organization of findings.

**Integration and Access:**

- **Python Library** -- Embed Maigret in your own tools with `import maigret`. Build custom pipelines, automate searches, or integrate with other OSINT tools.
- **Web Interface** -- Browser-based UI at `http://127.0.0.1:5000` with result graphs and one-click report downloads.
- **Telegram Bot** -- Quick lookups without installation at `@maigret_search_bot`.
- **Docker Images** -- Two variants: `soxoj/maigret:latest` for CLI and `soxoj/maigret:web` for the web interface.
- **Cloud Shell Support** -- Run directly in Google Cloud Shell, Replit, Colab, or Binder without local installation.

**Privacy and Network:**

- **Tor Network Support** -- Route all checks through Tor with `--tor-proxy socks5://127.0.0.1:9050`.
- **I2P Network Support** -- Access `.i2p` sites with `--i2p-proxy http://127.0.0.1:4444`.
- **SOCKS/HTTP Proxy** -- Use any proxy with `--proxy socks5://host:port`.
- **CAPTCHA Bypass** -- Built-in detection and partial bypass of CAPTCHA challenges and WAF blocks.
- **Auto-Updated Database** -- Site definitions update automatically from GitHub, with offline fallback to the built-in database.

## Installation

### Quick Install (pip)

```bash
# Install from PyPI
pip3 install maigret

# Basic usage
maigret username
```

### From Source

```bash
# Clone and install
git clone https://github.com/soxoj/maigret && cd maigret
pip3 install .

# Run
maigret username
```

### Docker

```bash
# CLI mode (default)
docker pull soxoj/maigret
docker run -v /mydir:/app/reports soxoj/maigret:latest username --html

# Web UI mode (open http://localhost:5000)
docker run -p 5000:5000 soxoj/maigret:web
```

### Windows

Download the standalone EXE from the [Releases page](https://github.com/soxoj/maigret/releases).

## Usage Examples

```bash
# Search for a username on top 500 sites
maigret username

# Search on ALL 3,000+ sites
maigret username -a

# Generate HTML report
maigret username --html

# Generate PDF report
maigret username --pdf

# Search on sites tagged with specific categories
maigret username --tags photo,dating

# Search on sites from a specific country
maigret username --tags us

# Search multiple usernames
maigret user1 user2 user3 -a

# Parse a profile page and search for discovered IDs
maigret --parse https://example.com/profile/username

# Generate username permutations
maigret "john doe" --permute

# Use Tor for anonymous searching
maigret username --tor-proxy socks5://127.0.0.1:9050

# Launch web interface
maigret --web 5000
```

## Key Features Summary

| Feature | Description |
|---------|-------------|
| 3,000+ Sites | Largest site database of any username search tool |
| Recursive Search | Automatically discovers linked accounts |
| No API Keys | Works without any API keys or accounts |
| CAPTCHA Bypass | Built-in detection and partial bypass |
| Tor/I2P Support | Anonymous searching through privacy networks |
| Multiple Reports | HTML, PDF, CSV, JSON, D3 graph, Xmind |
| Web Interface | Browser-based UI with interactive graphs |
| Python Library | Embeddable in custom OSINT pipelines |
| Docker Support | CLI and Web UI container images |
| Auto-Update | Site database updates daily from GitHub |
| Username Permutation | Generate and search username variants |
| Tag Filtering | Filter by site category or country |

## Professional Use

Maigret is used by professional OSINT and social-media analysis tools:

- **Social Links API** -- Enterprise OSINT platform built on Maigret
- **Social Links Crimewall** -- Professional investigation tool
- **UserSearch** -- Online username search service

For commercial use requiring a daily-updated site database or a username-check API, the Maigret team offers a private database with 5,000+ sites updated daily, separate from the public open-source database.

## Disclaimer

Maigret is intended for educational and lawful purposes only. Users are responsible for complying with all applicable laws (GDPR, CCPA, etc.) in their jurisdiction. The authors bear no responsibility for misuse.

## Links

- GitHub Repository: [https://github.com/soxoj/maigret](https://github.com/soxoj/maigret)
- PyPI Package: [https://pypi.org/project/maigret/](https://pypi.org/project/maigret/)
- Documentation: [https://maigret.readthedocs.io](https://maigret.readthedocs.io)
- Telegram Bot: [https://t.me/maigret_search_bot](https://t.me/maigret_search_bot)
- Docker Hub: [https://hub.docker.com/r/soxoj/maigret](https://hub.docker.com/r/soxoj/maigret)