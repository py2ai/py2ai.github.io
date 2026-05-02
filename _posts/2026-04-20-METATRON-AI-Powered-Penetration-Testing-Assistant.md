---
layout: post
title: "METATRON: AI-Powered Penetration Testing Assistant That Runs 100% Locally"
description: "An in-depth look at METATRON, a CLI-based AI pentesting assistant that automates recon-to-report using a local LLM with an agentic loop, 6 recon tools, MariaDB persistence, and PDF/HTML report generation - all without API keys or cloud services."
date: 2026-04-20
header-img: "assets/img/ai-coding-frameworks/ai-coding-frameworks"
permalink: /2026/04/20/metatron-ai-penetration-testing/
featured-img: ai-coding-frameworks/ai-coding-frameworks
tags: [AI, penetration-testing, security, LLM, ollama, local-AI, cybersecurity]
author: "PyShine"
---

# METATRON: AI-Powered Penetration Testing Assistant That Runs 100% Locally

Penetration testing has traditionally required deep expertise, hours of manual reconnaissance, and careful documentation. METATRON changes this equation by combining a local large language model with an agentic execution loop that autonomously runs recon tools, analyzes results, searches for vulnerabilities, and produces structured reports -- all without sending a single packet to a cloud API. For security professionals who operate under strict data-handling policies or in air-gapped environments, this is a significant capability.

METATRON is a CLI-based penetration testing assistant built on top of Ollama, running a custom-tuned model called `metatron-qwen`. The system orchestrates six standard recon tools (nmap, whois, whatweb, curl, dig, nikto) through an agentic loop that can iteratively request additional tool runs and web searches based on its own analysis. Every finding is persisted to a MariaDB database and exported as professional PDF or HTML reports. The entire pipeline runs locally -- no API keys, no cloud services, no data leaving your machine.

The project is designed for security analysts, penetration testers, and red team operators who need repeatable, documentable assessments. Whether you are scanning a single host or running recurring audits, METATRON provides a structured workflow from initial reconnaissance through final report generation, with the LLM acting as an intelligent orchestrator at every stage.

## Architecture Overview

![METATRON Architecture](/assets/img/diagrams/metatron/metatron-architecture.svg)

The METATRON system is organized around six core modules, each with a clearly defined responsibility. At the top of the hierarchy sits `metatron.py`, the 431-line CLI entry point that orchestrates the entire assessment workflow. This module handles command-line argument parsing, initializes the database connection, triggers the reconnaissance phase, invokes the agentic LLM loop, and coordinates report export. It serves as the central nervous system, delegating work to specialized modules while maintaining the overall execution flow.

The `tools.py` module (252 lines) contains the recon tool runners. It defines a `ToolRunner` class that wraps each of the six reconnaissance tools -- nmap for port scanning and service detection, whois for domain registration lookups, whatweb for web technology fingerprinting, curl for HTTP response analysis, dig for DNS record enumeration, and nikto for web server vulnerability scanning. Each tool runner handles subprocess execution, output capture, and error handling. Critically, `tools.py` enforces a strict allowlist: only these six predefined tools can be executed, preventing the LLM from running arbitrary commands on the system.

The `llm.py` module (388 lines) is the intelligence engine. It interfaces with Ollama to communicate with the `metatron-qwen` model and implements the agentic loop. When the LLM emits a `[TOOL:]` tag in its response, `llm.py` parses the tool name and arguments, routes the request to `tools.py` for execution, and feeds the result back into the next LLM iteration. When the LLM emits a `[SEARCH:]` tag, the request is routed to `search.py` for web-based intelligence gathering. The module also handles context window management through self-summarization -- when the conversation grows too long, the LLM is prompted to compress its findings into 15 or fewer bullet points, keeping the context within the model's token limits.

The `search.py` module (176 lines) provides the smart search dispatch capability. It uses DuckDuckGo as its search backend and routes queries intelligently based on content. CVE identifiers trigger a direct lookup on vulnerability databases. Exploit-related queries are routed to exploit databases. Fix and remediation queries are directed to security advisory sources. This routing ensures that the LLM gets the most relevant information without wasting iterations on generic search results.

The `db.py` module (352 lines) manages all MariaDB interactions through full CRUD operations. It creates and manages five interconnected tables: `history` for scan metadata, `vulnerabilities` for discovered vulnerabilities, `fixes` for remediation suggestions, `exploits_attempted` for exploit records, and `summary` for assessment summaries. All tables are linked through the `sl_no` foreign key, enabling cascading queries across the entire assessment dataset.

Finally, `export.py` (422 lines) reads data from the database and generates professional PDF and HTML reports. It formats vulnerability tables, severity ratings, exploit details, and remediation recommendations into structured documents suitable for delivery to clients or compliance teams.

## The Agentic Loop

![METATRON Agentic Loop](/assets/img/diagrams/metatron/metatron-agentic-loop.svg)

The agentic loop is the core innovation that separates METATRON from simple tool-orchestration scripts. Rather than running a fixed sequence of commands and presenting raw output, METATRON gives the LLM the ability to autonomously decide when additional information is needed and which tools to use to obtain it. This creates a feedback-driven intelligence cycle that mirrors how an experienced penetration tester would approach a target.

The loop begins after the initial reconnaissance phase. All six recon tools are run against the target, and their raw output is collected and sent to the LLM along with a system prompt that instructs the model to analyze the findings. The LLM processes this data and produces one of two responses. If it determines that more information is needed -- for example, if it spotted an unusual service on a port and wants to search for known vulnerabilities -- it emits a structured tag in its response. The `[TOOL:]` tag requests execution of one of the six allowed recon tools with specific arguments. The `[SEARCH:]` tag requests a web search for CVEs, exploits, or remediation information.

When a `[TOOL:]` or `[SEARCH:]` tag is detected, the system parses the tag, executes the requested action, and captures the output. Before feeding this output back to the LLM, the system runs a summarization step. The LLM is prompted to compress the new information into 15 or fewer bullet points. This serves a dual purpose: it keeps the context window within manageable limits, and it forces the model to extract only the most operationally relevant information from each tool run. The summarized result is then fed back into the LLM for the next iteration.

The loop continues until one of two conditions is met. First, the LLM may stop emitting `[TOOL:]` and `[SEARCH:]` tags, indicating that it has gathered sufficient information to produce a final assessment. Second, the loop counter may reach `MAX_TOOL_LOOPS`, which is set to 9. This hard cap prevents runaway loops where the model keeps requesting additional information indefinitely. In practice, most assessments complete within 3 to 5 iterations.

Once the loop terminates, the LLM's final output is parsed for structured data. The model is instructed to format its findings using specific tags: `VULN:` for vulnerability names, `SEVERITY:` for severity ratings, `EXPLOIT:` for exploit details, and `RISK_LEVEL:` for overall risk assessment. This structured output is then stored in the MariaDB database and used to generate the final report.

```python
# Example of the agentic loop tag format
# The LLM emits these tags in its response:

[TOOL: nmap -sV -p 443 10.0.0.1]
# or
[SEARCH: CVE-2024-1234 nginx vulnerability]

# The system parses these and routes accordingly:
# [TOOL:] --> tools.py (allowlist-enforced execution)
# [SEARCH:] --> search.py (smart dispatch to CVE/exploit/fix sources)
```

The context window management strategy is particularly noteworthy. Rather than simply truncating the conversation history when it exceeds the model's token limit, METATRON uses self-summarization. The LLM is asked to compress its own findings into a concise summary, which replaces the full conversation history. This preserves the most important findings while freeing up context space for additional tool runs. The result is a system that can conduct deep, multi-iteration investigations without running into context length limitations.

## Database Schema

![METATRON Database Schema](/assets/img/diagrams/metatron/metatron-database-schema.svg)

METATRON uses MariaDB as its persistence layer, storing all assessment data in five interconnected tables. The schema is designed around a central `history` table that serves as the primary record for each scan, with four satellite tables linked through the `sl_no` foreign key. This design enables efficient queries across all findings for a given assessment while maintaining clean separation between different data categories.

The `history` table is the anchor of the schema. It contains four columns: `sl_no` (an auto-incrementing integer primary key), `target` (the IP address or domain that was scanned), `date` (a datetime stamp of when the assessment was performed), and `status` (a string indicating the current state of the assessment, such as "in_progress" or "completed"). Every scan creates a new row in this table, and the `sl_no` value becomes the linking identifier used across all other tables.

The `vulnerabilities` table stores discovered security weaknesses. Each row references a `sl_no` foreign key linking it to a specific scan in the history table, along with `vuln_name` (the name or title of the vulnerability), `severity` (a rating such as "Critical", "High", "Medium", or "Low"), and `description` (a detailed text explanation of the vulnerability and its potential impact). The relationship between `history` and `vulnerabilities` is one-to-many, since a single scan can uncover multiple vulnerabilities.

The `fixes` table captures remediation recommendations. Like vulnerabilities, each fix is linked to a scan via the `sl_no` foreign key. The table includes `fix_name` (a short title for the remediation), `description` (detailed instructions for implementing the fix), and `priority` (a ranking that helps operators determine which fixes to implement first). This table enables security teams to track not just what was found, but what should be done about it.

The `exploits_attempted` table records any exploit attempts made during the assessment. It links to the scan via `sl_no` and includes `exploit_name` (the name or identifier of the exploit), `result` (a text description of the outcome), and `cve_id` (the CVE identifier if applicable). This table is critical for maintaining an audit trail of what was tested and what the results were.

The `summary` table provides a one-to-one relationship with the history table, containing the overall assessment summary for each scan. It includes `summary_text` (a comprehensive narrative summary of the assessment), `risk_level` (an overall risk rating for the target), and `recommendations` (prioritized action items). This table is what feeds into the final report generation.

```sql
-- The sl_no linking pattern enables cascading queries:
-- Get all findings for scan #5:
SELECT * FROM vulnerabilities WHERE sl_no = 5;
SELECT * FROM fixes WHERE sl_no = 5;
SELECT * FROM exploits_attempted WHERE sl_no = 5;
SELECT * FROM summary WHERE sl_no = 5;

-- Cascading deletes ensure data integrity:
-- When a scan record is deleted, all related findings are removed
```

The `db.py` module provides full CRUD operations for all five tables. Records are created during the assessment as the LLM produces structured output, updated if new information emerges in subsequent loop iterations, and read during report generation. The cascading delete pattern ensures that removing a scan from the history table also removes all associated vulnerabilities, fixes, exploits, and summaries, maintaining referential integrity without orphaned records.

## Recon Pipeline

![METATRON Recon Pipeline](/assets/img/diagrams/metatron/metatron-recon-pipeline.svg)

The reconnaissance pipeline is the first phase of every METATRON assessment, and it is designed with a security-first philosophy. Before any tool is executed, the target passes through a tool allowlist security gate. This gate ensures that only the six approved recon tools -- nmap, whois, whatweb, curl, dig, and nikto -- can be executed. This is a critical safeguard: because the LLM can request additional tool runs through the `[TOOL:]` tag, the allowlist prevents the model from executing potentially destructive or unauthorized commands. Even if the LLM were to emit `[TOOL: rm -rf /]`, the allowlist would reject it.

Each of the six recon tools serves a specific purpose in building a comprehensive picture of the target. **nmap** performs port scanning and service detection, identifying open ports, running services, and their versions. **whois** retrieves domain registration information, including registrant details, name servers, and expiration dates. **whatweb** fingerprints web technologies, detecting CMS platforms, JavaScript frameworks, web servers, and other software running on the target. **curl** captures HTTP response headers and body content, revealing server configuration details and potential misconfigurations. **dig** enumerates DNS records, uncovering subdomains, mail servers, and other DNS entries. **nikto** performs web server vulnerability scanning, checking for known vulnerabilities, misconfigurations, and dangerous default installations.

All six tools run against the target, and their raw output is collected into a single stream. This raw output can be extensive -- a single nmap scan with service detection can produce hundreds of lines, and nikto output is similarly verbose. Rather than feeding this entire output directly into the LLM's context window, METATRON applies a summarization step. The LLM is prompted to compress the raw recon data into 15 or fewer bullet points, extracting only the most operationally significant findings. This summarized recon data then enters the agentic loop, where the LLM can request additional tool runs or web searches to fill in knowledge gaps.

The smart search dispatch in `search.py` adds another layer of intelligence to the pipeline. When the LLM emits a `[SEARCH:]` tag, the search module analyzes the query content and routes it to the most appropriate source. Queries containing CVE identifiers (matching the pattern `CVE-\d{4}-\d{4,}`) are routed directly to vulnerability database lookups. Queries about exploits are directed to exploit databases. Queries about fixes and remediations are routed to security advisory sources. This intelligent routing ensures that each search iteration yields the most relevant results, reducing the number of loop cycles needed to complete an assessment.

```bash
# The six recon tools in the allowlist:
# nmap    - Port scanning and service detection
# whois   - Domain registration lookup
# whatweb - Web technology fingerprinting
# curl    - HTTP response analysis
# dig     - DNS record enumeration
# nikto   - Web server vulnerability scanning

# Example tool execution via [TOOL:] tag:
# [TOOL: nmap -sV -p- 192.168.1.1]
# [TOOL: nikto -h https://target.com]
# [TOOL: dig MX target.com]
```

The pipeline's parallel execution of all six tools at the start of an assessment ensures that the LLM has a broad baseline of information before it begins making decisions about what additional intelligence to gather. This "broad first, deep second" approach mirrors how experienced penetration testers work: start with a wide sweep, then drill into specific areas of interest based on initial findings.

## Key Design Patterns

METATRON incorporates several design patterns that are worth studying for anyone building agentic AI systems, particularly in security-sensitive domains.

**Agentic Loop with Structured Text Tags.** Rather than using OpenAI-style function calling, METATRON uses simple text tags (`[TOOL:]` and `[SEARCH:]`) embedded in the LLM's natural language output. This approach has several advantages. It is model-agnostic -- any LLM that can follow instructions can produce these tags. It is transparent -- you can read the LLM's reasoning alongside its tool requests. And it is debuggable -- if the model produces a malformed tag, the system can log it and continue rather than crashing. The tradeoff is that parsing is less reliable than structured function calls, but METATRON mitigates this with careful prompt engineering and robust tag extraction.

**Context Window Management via Self-Summarization.** One of the most practical challenges in building agentic systems is managing the LLM's context window. Each tool run and search result adds tokens to the conversation, and long-running assessments can easily exceed the model's context limit. METATRON's solution is elegant: instead of truncating history or using a sliding window, it asks the LLM to summarize its own findings into 15 bullet points. This preserves the most important information while freeing up context space. The model effectively compresses its own working memory, retaining key findings and discarding verbose raw output.

**Tool Allowlist for Security.** In any system where an LLM can request tool execution, security is paramount. METATRON's allowlist pattern ensures that only the six predefined recon tools can be executed, regardless of what the LLM requests. This is a defense-in-depth measure: even if the model were to produce a `[TOOL:]` tag requesting a destructive command, the allowlist check in `tools.py` would reject it. This pattern should be adopted by any agentic system that executes commands based on LLM output.

**Smart Search Dispatch Routing.** The `search.py` module implements a routing pattern that analyzes the content of a search query and directs it to the most appropriate backend. CVE lookups, exploit searches, and fix queries each have different optimal sources. By routing intelligently, METATRON reduces the number of search iterations needed and improves the quality of information returned. This pattern generalizes well to any agentic system that needs to query multiple knowledge sources.

**Structured Output Parsing.** The LLM is instructed to format its final assessment using specific tags: `VULN:` for vulnerability names, `SEVERITY:` for severity ratings, `EXPLOIT:` for exploit details, and `RISK_LEVEL:` for overall risk. These tags enable deterministic parsing of the LLM's output into database records. This pattern bridges the gap between unstructured LLM output and structured data storage, and it is essential for any system that needs to persist LLM findings in a relational database.

## Getting Started

Setting up METATRON requires a Linux environment with the appropriate security tools installed. The project is designed to run on Parrot OS, which comes pre-configured with most of the required tools.

### Prerequisites

```bash
# Install Ollama (local LLM runtime)
curl -fsSL https://ollama.com/install.sh | sh

# Pull the METATRON model
ollama pull metatron-qwen

# Install MariaDB
sudo apt install mariadb-server

# Install recon tools (on Parrot OS, most are pre-installed)
sudo apt install nmap whois whatweb curl dnsutils nikto
```

### Installation

```bash
# Clone the repository
git clone https://github.com/MrR0b0t2/METATRON.git
cd METATRON

# Install Python dependencies
pip install -r requirements.txt

# Configure MariaDB connection
# Edit the database configuration in metatron.py
# Default: localhost:3306, database: metatron
```

### Running Your First Scan

```bash
# Run a full assessment against a target
python metatron.py -t 192.168.1.1

# The system will:
# 1. Run all 6 recon tools against the target
# 2. Summarize the output
# 3. Enter the agentic loop (up to 9 iterations)
# 4. Store all findings in MariaDB
# 5. Generate PDF and HTML reports
```

The assessment process is fully automated once initiated. The LLM will decide which additional tools to run and which searches to perform based on its analysis of the initial recon data. You can monitor progress through the CLI output, which shows each tool execution, search query, and LLM decision in real time.

## Conclusion

METATRON represents a practical approach to AI-assisted penetration testing that prioritizes operational security and data sovereignty. By running entirely locally with Ollama and a custom-tuned model, it eliminates the risk of sensitive assessment data being transmitted to cloud APIs. The agentic loop design allows the system to conduct thorough, multi-iteration investigations that mirror the workflow of an experienced penetration tester, while the tool allowlist and structured output parsing provide the safety guardrails needed for production use.

The project's architecture -- with its clean module separation, MariaDB persistence, and professional report generation -- makes it a solid foundation for security teams looking to incorporate AI into their assessment workflows. The six-module design (metatron.py, tools.py, llm.py, search.py, db.py, export.py) keeps concerns well-separated, making the codebase easy to understand, extend, and audit. For the security community, METATRON demonstrates that effective AI-powered pentesting does not require cloud dependencies or proprietary APIs -- it can be built with open-source tools running on local hardware, keeping sensitive data where it belongs.
