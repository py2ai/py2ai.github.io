---
layout: post
title: "LLM Anonymization: Reverse Proxy for AI Privacy in Pentesting"
description: "How a transparent FastAPI proxy anonymizes sensitive pentest data before it reaches Claude, using dual-layer detection with LLM and regex, persistent surrogate mapping, and engagement isolation."
date: 2026-04-20
header-img: "assets/img/diagrams/llm-anonymization/llm-anonymization-proxy-architecture.svg"
permalink: /LLM-Anonymization-Reverse-Proxy-for-AI-Privacy/
featured-img: ai-coding-frameworks/ai-coding-frameworks
tags: [AI, Privacy, Pentesting, Claude, Anonymization, Security, Proxy, LLM]
author: PyShine
---

## Introduction

Pentesters who use AI coding assistants like Claude Code face a fundamental tension: they want the productivity gains of AI-assisted analysis, but the data they work with is among the most sensitive in any industry. Every nmap scan, every credential dump, every lateral-movement log contains real client infrastructure details that cannot leave the pentester's machine. Sending that data to a third-party API is not just a privacy concern -- it is a potential violation of the non-disclosure agreements and engagement contracts that govern professional penetration testing.

The risk is not theoretical. When a pentester pastes the output of crackmapexec or mimikatz into an AI chat, they are transmitting client IP addresses, hostnames, domain usernames, cleartext passwords, and Kerberos tickets to an external service. Even if the API provider promises not to train on the data, the data has already left the perimeter. For security consultants working under strict NDAs, this is a deal-breaker that forces them to choose between AI assistance and contractual compliance.

LLM-anonymization solves this problem by inserting a transparent reverse proxy between the pentester's AI tool and the upstream API. Every request that Claude Code sends to Anthropic is first intercepted by a local FastAPI proxy, scrubbed of all personally identifiable information, and then forwarded with realistic surrogate values in place of the originals. When the response comes back, the proxy reverses the substitution, restoring the real values so the pentester sees their actual data. The entire process is transparent: Claude Code does not know it is talking to a proxy, and Anthropic never sees the real data.

Designed specifically for Claude Code and pentesting workflows, LLM-anonymization understands the unique data formats that security tools produce. It recognizes nmap output, BloodHound data, Kerberos tickets, and credential dumps -- not just generic PII. This domain-specific awareness, combined with a dual-layer detection pipeline and persistent surrogate mapping, makes it a purpose-built privacy gateway for offensive security professionals.

## The Privacy Problem in AI-Assisted Pentesting

Consider what happens during a typical internal penetration test. The pentester runs nmap to discover live hosts, crackmapexec to test credentials across the network, and mimikatz to extract hashes from memory. Each of these tools produces output rich with client-specific identifiers: IP ranges like 10.0.0.0/24, hostnames like DC-ACME-01, domain accounts like ACME\admin, and NTLM hashes. When the pentester asks Claude Code to help analyze this output, write a report, or suggest next steps, all of that data travels to Anthropic's API servers.

This creates a direct conflict with the legal frameworks that govern penetration testing. Engagement contracts typically specify that all client data must remain within agreed-upon boundaries, and NDAs explicitly prohibit sharing client information with unauthorized third parties. An API provider, no matter how trustworthy, is an unauthorized third party under most engagement terms. The pentester who sends raw tool output to Claude is, in effect, breaching their contract with every API call.

The challenge is that AI assistance is genuinely valuable for pentesters. Claude Code can help interpret scan results, suggest exploitation paths, draft findings for reports, and automate repetitive tasks. Giving up that assistance means working slower and potentially missing critical findings. What pentesters need is not to abandon AI tools, but to use them in a way that never exposes real client data. That is exactly the gap LLM-anonymization fills.

![LLM Anonymization Proxy Architecture](/assets/img/diagrams/llm-anonymization/llm-anonymization-proxy-architecture.svg)

The architecture diagram above illustrates the complete request flow through the LLM-anonymization proxy. On the left, Claude Code is configured to send its API requests to a local FastAPI proxy running on port 8080 instead of directly to Anthropic's API endpoint. This redirection is achieved by setting the ANTHROPIC_BASE_URL environment variable, which requires no modification to Claude Code itself -- the tool simply believes it is talking to the real API.

When a request arrives at the proxy, it passes through three internal components in sequence. First, the LLM Detector uses a locally-running Ollama instance (typically with a qwen3 model) to identify contextual entities that regex patterns would miss -- things like bare hostnames without domain suffixes, domain usernames in various formats, cleartext passwords embedded in tool output, organization names, person names, and internal application or project names. Second, the Regex Safety Net applies deterministic pattern matching to catch structured data types like IP addresses, hash values, MAC addresses, email addresses, and API tokens. Third, the PII Vault looks up or creates surrogate mappings for every detected entity, ensuring consistent replacement across the entire engagement.

After anonymization, the proxy forwards the scrubbed request to Anthropic's API. The response travels back through the same proxy, where the PII Vault performs the reverse substitution, replacing surrogate values with the original real data. The pentester sees their actual data in Claude Code's responses, while Anthropic only ever processed surrogate values. This bidirectional flow is the core of the privacy guarantee: sensitive data never leaves the pentester's machine in its original form.

## Dual-Layer Anonymization

A single detection layer is insufficient for pentest data. Regex patterns are excellent at catching structured data like IP addresses and email addresses, but they cannot identify contextual entities. A bare hostname like "WEB-PROD-03" does not match any standard regex pattern, yet it is clearly sensitive client infrastructure data. Conversely, an LLM can identify that "WEB-PROD-03" is a hostname, but it might miss a perfectly valid IPv6 address or a 64-character hex hash that follows a strict format.

This is why LLM-anonymization uses a dual-layer approach. Layer 1, the LLM Detector, leverages a locally-running Ollama instance with a qwen3 model to perform contextual analysis of the text. It identifies six categories of contextual entities: bare hostnames (servers, workstations, domain controllers), domain usernames (DOMAIN\user or user@domain formats), cleartext passwords found in tool output, organization names, person names, and internal application or project names. These are entities that require natural language understanding to detect reliably.

Layer 2, the Regex Safety Net, applies a comprehensive set of deterministic patterns to catch structured data that the LLM might overlook or misidentify. It covers eight pattern types: IPv4 and IPv6 addresses, CIDR notation ranges, hash values (MD5, SHA1, SHA256, NTLM), MAC addresses, email addresses, domain names, cloud provider tokens (AWS, GCP, Azure), and JWT/API key strings. These patterns are unambiguous and match with high precision, making them ideal candidates for regex-based detection.

The two layers complement each other to achieve comprehensive coverage. The LLM catches what regex cannot -- contextual entities that depend on understanding the surrounding text. The regex catches what the LLM might miss -- strictly formatted strings that follow predictable patterns. Together, they ensure that virtually every piece of sensitive information in pentest output is identified and replaced before the data leaves the machine.

![Dual-Layer Anonymization Pipeline](/assets/img/diagrams/llm-anonymization/llm-anonymization-dual-layer-detection.svg)

The diagram above shows the complete anonymization pipeline as data flows from raw pentest output to the anonymized request. Raw text enters the pipeline and first passes through Layer 1, the LLM Detector. This layer sends the text to a locally-running Ollama instance with a structured prompt that asks the model to identify all contextual entities across six categories. The LLM returns a list of detected entities with their types and positions, which the proxy uses to build the initial set of replacements.

Layer 1 catches the following contextual categories: bare hostnames such as DC-ACME-01, WEB-PROD-03, or FILESRV-02 that do not follow any standard regex pattern; domain usernames in formats like ACME\admin, admin@acme.local, or ACME\svc_backup; cleartext passwords that appear in tool output or configuration snippets; organization names like "Acme Corporation" or "Target Corp"; person names of employees, administrators, or contacts found in directory listings; and internal application or project names that would reveal client-specific infrastructure details.

After Layer 1 processing, the text passes through Layer 2, the Regex Safety Net. This layer applies a battery of deterministic patterns to catch structured data types. The eight pattern types are: IPv4 addresses (matching standard dotted-decimal notation), IPv6 addresses (matching full and compressed formats), CIDR notation (matching network ranges like 10.0.0.0/24), hash values (matching MD5, SHA1, SHA256, SHA512, and NTLM formats), MAC addresses (matching colon and hyphen-separated formats), email addresses (matching standard email format), domain names (matching FQDNs with TLD validation), and cloud tokens or API keys (matching AWS, GCP, Azure key formats and JWT structures).

The layers operate sequentially, with Layer 1 running first to catch contextual entities and Layer 2 running second to catch anything the LLM missed. Neither layer alone provides sufficient coverage: Layer 1 without Layer 2 would miss structured patterns like hashes and IP addresses, while Layer 2 without Layer 1 would miss contextual entities like bare hostnames and organization names. The combination ensures that the anonymized output contains no trace of the original sensitive data.

## Surrogate Mapping and the PII Vault

When the proxy detects a piece of sensitive information, it does not simply redact or mask it. Instead, it replaces each original value with a realistic-looking surrogate that preserves the format and type of the original. This is critical because Claude needs to see plausible data to provide useful analysis. If every IP address were replaced with "[REDACTED]", Claude would have no context for network topology analysis. If every hostname were replaced with "[HOSTNAME]", it could not reason about lateral movement paths.

The surrogate format is designed to be both realistic and clearly distinguishable from real data. IP addresses are replaced with values from the RFC 5737 TEST-NET-3 range (203.0.113.0/24), which is reserved for documentation and examples. Domains receive a .pentest.local suffix, making them clearly synthetic while preserving the subdomain structure. Hostnames get numeric suffixes like host-001, host-002 to maintain uniqueness. Usernames receive a user_ prefix like user_admin, user_backup. Credentials are replaced with bracketed tags like [CRED_XK9A2B3C] that clearly indicate a credential was present without revealing its value. Hash values are replaced with synthetic hashes that maintain the correct length and format. Email addresses use the pentest.local domain.

The PII Vault is a SQLite database that stores bidirectional mappings between original values and their surrogates. When the proxy encounters a value it has seen before, it looks up the existing surrogate rather than generating a new one. This consistency is essential: if the IP address 10.0.0.5 is replaced with 203.0.113.5 in one request, it must be replaced with the same surrogate in every subsequent request. Without this consistency, Claude would see the same real host appearing under different surrogate names, making its analysis unreliable.

The ENGAGEMENT_ID variable creates isolated namespaces within the PII Vault. Each client engagement gets its own set of mappings, preventing cross-client data leakage. When a pentester switches from engagement "client-acme-2026" to "client-globex-2026", the surrogate for a given IP address may differ between engagements. This ensures that even if someone gained access to the vault, they could not correlate data across different client engagements.

![Surrogate Mapping and PII Vault](/assets/img/diagrams/llm-anonymization/llm-anonymization-surrogate-mapping.svg)

The diagram above details the seven surrogate mapping types and the role of the PII Vault in maintaining consistent bidirectional mappings. Each mapping type is designed to produce surrogates that are format-preserving and realistic enough for Claude to reason about, while being clearly synthetic to any human reviewer.

The seven mapping types work as follows. First, IP address mapping: an original address like 10.0.0.5 is mapped to 203.0.113.5, preserving the last octet for readability while using the RFC 5737 TEST-NET-3 range. Second, domain mapping: acme-corp.com becomes acme-corp.pentest.local, preserving the subdomain structure while replacing the TLD. Third, hostname mapping: DC-ACME-01 becomes host-001, providing a unique identifier that indicates the original was a hostname. Fourth, username mapping: ACME\admin becomes user_admin, preserving the username portion while removing the domain context. Fifth, credential mapping: P@ssw0rd! becomes [CRED_XK9A2B3C], using a unique alphanumeric tag that clearly indicates a credential was present. Sixth, hash mapping: an NTLM hash like a87f3a33 becomes a synthetic hash of the same length and format. Seventh, email mapping: admin@acme.local becomes admin@pentest.local, preserving the local part while replacing the domain.

The PII Vault stores all of these mappings in a SQLite database with tables indexed by ENGAGEMENT_ID. When the proxy processes a request, it first queries the vault for existing mappings. If a mapping exists, it reuses the same surrogate. If not, it generates a new surrogate and stores the bidirectional mapping. This ensures that across all requests within an engagement, the same original always maps to the same surrogate.

The deanonymization process works in reverse. When a response arrives from Anthropic's API, the proxy scans the response text for surrogate values and looks up the corresponding originals in the PII Vault. Each surrogate is replaced with its real value, so the pentester sees the actual IP addresses, hostnames, and credentials in Claude's responses. The ENGAGEMENT_ID ensures that surrogates from one engagement are never accidentally decoded using mappings from another, preventing cross-client data leakage even in multi-engagement workflows.

## Deployment and Configuration

LLM-anonymization offers three deployment options to suit different operational needs. Option A is the VPS deployment, intended for production pentest engagements. The proxy runs on a virtual private server, and the pentester connects to it via an SSH tunnel. This setup keeps the proxy and Ollama instance running continuously, accessible from any location, and isolated from the pentester's local machine. It is the recommended approach for real engagements where reliability and isolation matter.

Option B is the native Python deployment, intended for local development and testing. The proxy runs directly on the pentester's machine using Python and uvicorn. This is the simplest setup and works well on Apple Silicon machines where Ollama runs natively. It is ideal for testing new anonymization rules, developing custom patterns, or evaluating the tool before deploying it to a VPS.

Option C is the Docker deployment, which provides a quick containerized setup with all dependencies bundled. A single docker compose command brings up both the proxy and the Ollama instance. This is the fastest way to get started and ensures a consistent environment regardless of the host system. It is suitable for both development and production use, though VPS deployment remains the recommended approach for live engagements.

Key configuration variables control the proxy's behavior. ENGAGEMENT_ID sets the namespace for surrogate mappings, ensuring isolation between different client engagements. OLLAMA_HOST specifies the address of the Ollama instance (default: http://localhost:11434). OLLAMA_MODEL selects the model used for contextual detection (default: qwen3:4b). LLM_ENABLED toggles the LLM detection layer on or off. PORT sets the proxy's listening port (default: 8080). The proxy also implements graceful degradation: if the Ollama instance times out or is unavailable, the proxy falls back to regex-only mode, ensuring that requests still get processed even without the LLM layer.

![Deployment Options and Data Flow](/assets/img/diagrams/llm-anonymization/llm-anonymization-deployment-options.svg)

The diagram above illustrates the three deployment options and the data flow for each. Option A shows the VPS deployment path: the pentester's machine connects to the VPS-hosted proxy via an SSH tunnel (ssh -L 8080:localhost:8080 user@vps-ip), which forwards local port 8080 to the VPS. The VPS runs both the FastAPI proxy and the Ollama instance, providing a dedicated and persistent anonymization gateway. This architecture ensures that the proxy is always available and that the Ollama model stays loaded in memory for fast inference.

Option B shows the native Python deployment: the pentester runs the proxy directly on their local machine using Python and uvicorn. The Ollama instance also runs locally, typically on Apple Silicon hardware where it achieves good inference speed. This setup is straightforward -- clone the repository, install dependencies, and run the proxy. It is best suited for development, testing, and situations where a VPS is not available.

Option C shows the Docker deployment: both the proxy and Ollama run inside Docker containers managed by docker compose. The containers share a network, and the proxy connects to Ollama via the internal Docker network. This approach bundles all dependencies and provides a reproducible environment. A single command brings the entire stack up, making it the fastest path to a working setup.

The configuration variables are consistent across all deployment options. ENGAGEMENT_ID defaults to "default" but should always be set to a unique value per client engagement. OLLAMA_HOST defaults to http://localhost:11434, which works for both native and Docker deployments; for VPS deployments, it points to the Ollama instance on the same server. OLLAMA_MODEL defaults to qwen3:4b, which provides a good balance of speed and accuracy for contextual detection. LLM_ENABLED defaults to true, and when set to false, the proxy operates in regex-only mode. PORT defaults to 8080.

The graceful degradation mechanism is a critical reliability feature. When the proxy sends a text to Ollama for contextual analysis, it sets a timeout (typically 30 seconds). If Ollama does not respond within the timeout -- perhaps because the model is still loading, the server is under load, or the network connection is unstable -- the proxy logs a warning and proceeds with regex-only anonymization. This ensures that the pentester's workflow is never blocked by an Ollama outage. The trade-off is reduced coverage during degradation, since the LLM layer catches contextual entities that regex cannot, but partial anonymization is better than no anonymization or a blocked workflow.

## Self-Improvement Loop

One of the most interesting aspects of the LLM-anonymization project is its structured feedback loop for improving anonymization coverage. The project includes a collection of pentest scenario fixtures -- sample outputs from tools like nmap, crackmapexec, mimikatz, and BloodHound -- that serve as test cases for the anonymization pipeline. Each fixture contains known sensitive data, and the tests verify that every piece of sensitive data is correctly identified and replaced.

The improvement workflow follows a clear cycle. First, add new pentest scenario fixtures that represent real-world data patterns the current pipeline might not handle. Second, run the auto_improve script, which processes all fixtures and reports any leaks -- sensitive values that were not anonymized. Third, fix the remaining leaks by updating regex patterns, adjusting the LLM prompt, or adding new detection rules. Fourth, run the full integration test suite to verify that no regressions were introduced and that all fixtures pass with zero leaks.

The coverage progression tells the story. The project started with 16 fixtures covering approximately 85% of common pentest data patterns. Through the self-improvement loop, coverage expanded to 37 fixtures with 100% detection -- zero leaks across all test cases. Further expansion brought the total to 49 fixtures covering 645 individual sensitive items, all detected with 100% accuracy. The project enforces a strict 0% leak policy: if any integration test reveals a sensitive value that was not anonymized, the build fails. This ensures that coverage only improves over time and never regresses.

## Getting Started

Setting up LLM-anonymization with Claude Code is straightforward. The key insight is that Claude Code respects the ANTHROPIC_BASE_URL environment variable, which tells it where to send API requests. By pointing this variable at the local proxy instead of the default Anthropic endpoint, all traffic is automatically routed through the anonymization pipeline. No modifications to Claude Code are required.

For a local development setup, start the proxy and then configure Claude Code:

```bash
export ANTHROPIC_BASE_URL=http://localhost:8080
export ENGAGEMENT_ID=client-acme-2026
claude
```

For a VPS deployment, establish an SSH tunnel first and then configure Claude Code to use the tunneled port:

```bash
# VPS deployment with SSH tunnel
ssh -L 8080:localhost:8080 user@vps-ip
export ANTHROPIC_BASE_URL=http://localhost:8080
export OLLAMA_MODEL=qwen3:4b
claude
```

Once configured, Claude Code operates normally. The pentester can paste tool output, ask for analysis, request report drafts, and use all of Claude's capabilities. The difference is that every request passes through the anonymization proxy, and every response is deanonymized before display. The pentester sees real data; Anthropic sees surrogate data. The privacy guarantee is maintained without any change to the pentester's workflow.

## Conclusion

LLM-anonymization addresses a critical gap in the AI-assisted pentesting workflow. Security professionals no longer have to choose between the productivity gains of AI tools and the contractual obligations of their NDAs. By inserting a transparent reverse proxy between Claude Code and Anthropic's API, the project ensures that sensitive client data never leaves the pentester's machine in its original form.

The dual-layer detection approach is the foundation of the project's effectiveness. The LLM layer catches contextual entities that no regex could identify, while the regex layer catches structured patterns that the LLM might overlook. Together, they provide comprehensive coverage that neither layer could achieve alone. The engagement isolation mechanism adds another layer of protection, ensuring that surrogate mappings from one client engagement cannot leak into or be correlated with mappings from another.

The project is still in its early stages as a specification-only repository, meaning the implementation is being developed from a detailed design document. However, the design itself is thorough and well-considered, addressing the real-world concerns of pentesters who need AI assistance without compromising client confidentiality. As the implementation matures and the self-improvement loop expands coverage, LLM-anonymization has the potential to become an essential tool in every pentester's toolkit.

Check out the project on GitHub: [zeroc00I/LLM-anonymization](https://github.com/zeroc00I/LLM-anonymization)