---
layout: post
title: "Anthropic Cybersecurity Skills: 754 AI Agent Skills Mapped to Security Frameworks"
description: "Learn how Anthropic Cybersecurity Skills provides 754 structured skills for AI agents mapped to MITRE ATT&CK, NIST CSF 2.0, and more. Installation guide, architecture, and real-world examples."
date: 2026-05-28
header-img: "img/post-bg.jpg"
permalink: /Anthropic-Cybersecurity-Skills-AI-Agent-Security-Frameworks/
featured-img: ai-coding-frameworks/ai-coding-frameworks
categories: [Cybersecurity, AI Agents, Python]
tags: [Anthropic, cybersecurity, AI agents, MITRE ATT&CK, NIST CSF, security frameworks, Claude Code, skills, penetration testing, security automation]
keywords: "Anthropic cybersecurity skills tutorial, AI agent security skills, MITRE ATT&CK AI skills, NIST CSF 2.0 AI agents, cybersecurity AI automation, Claude Code security skills, how to use Anthropic cybersecurity skills, AI agent penetration testing, security framework mapping AI, Anthropic skills installation guide"
author: "PyShine"
---

# Anthropic Cybersecurity Skills: 754 AI Agent Skills Mapped to Security Frameworks

Anthropic Cybersecurity Skills is the largest open-source cybersecurity skills library for AI agents, providing 754 production-grade skills spanning 26 security domains, each mapped to five industry frameworks including MITRE ATT&CK, NIST CSF 2.0, MITRE ATLAS, MITRE D3FEND, and NIST AI RMF. Whether you are building security automation pipelines, configuring penetration testing workflows, or mapping compliance controls, this library gives your AI agents the structured decision-making playbooks that turn a generic LLM into a capable security analyst.

The cybersecurity workforce gap hit 4.8 million unfilled roles globally in 2024 according to ISC2. AI agents can help close that gap, but only if they have structured domain knowledge to work from. Anthropic Cybersecurity Skills fills this gap by encoding real practitioner workflows, not generated summaries, into an AI-native format built on the agentskills.io open standard.

![Architecture Diagram](/assets/img/diagrams/anthropic-cybersecurity-skills/anthropic-cybersecurity-skills-architecture.svg)

### Understanding the Architecture

The architecture diagram above illustrates how Anthropic Cybersecurity Skills bridges the gap between AI agent platforms and expert-level security operations. Let us break down each component:

**AI Agent Platforms (Top Layer)**
The library works with 26+ AI platforms including Claude Code, GitHub Copilot, Cursor, OpenAI Codex CLI, Windsurf, Cline, Aider, and autonomous agent frameworks like CrewAI and LangChain. Any platform that supports the agentskills.io standard can load these skills with zero configuration.

**Skill Discovery Layer**
When an AI agent receives a security-related prompt, it scans the YAML frontmatter of all 754 skills. Each skill costs approximately 30 tokens to scan via frontmatter alone, enabling the agent to search the entire library in a single pass without blowing context windows. Once relevant skills are identified, the agent loads the full workflow at 500-2,000 tokens per skill.

**754 Cybersecurity Skills Core**
The core library contains skills organized across 26 security domains, from Cloud Security (60 skills) and Threat Hunting (55 skills) down to Deception Technology (2 skills). Every skill follows the agentskills.io open standard with consistent YAML frontmatter for discovery and structured Markdown for step-by-step execution.

**Skill Anatomy**
Each skill directory contains a SKILL.md file with YAML frontmatter (name, description, domain, subdomain, tags, framework mappings) and a Markdown body with four sections: When to Use, Prerequisites, Workflow, and Verification. Optional references/ and scripts/ directories provide deep technical context and helper scripts.

**Five Framework Mappings**
The library uniquely maps every skill across all five frameworks. A single skill like `analyzing-network-traffic-of-malware` maps to ATT&CK T1071, NIST CSF DE.CM, ATLAS AML.T0047, D3FEND D3-NTA, and AI RMF MEASURE-2.6 simultaneously. This cross-framework coverage is unmatched by any other open-source skills library.

**Key Domains (Sample)**
The diagram shows eight of the 26 domains. Cloud Security leads with 60 skills covering AWS, Azure, and GCP hardening. Threat Hunting provides 55 skills for hypothesis-driven hunts and behavioral analytics. Threat Intelligence offers 50 skills for STIX/TAXII and MISP integration. The full library spans everything from Web Application Security (42 skills) to Ransomware Defense (7 skills).

**Security Operations Output**
The end result is expert-level security guidance delivered in seconds. An AI agent equipped with these skills follows the same playbook a senior DFIR analyst would use, running the right Volatility3 plugins, checking LSASS access patterns, and correlating event log evidence.

> **Key Insight:** Each skill costs only ~30 tokens to scan via frontmatter, enabling AI agents to search all 754 skills in a single pass without exhausting context windows. Full skill loading requires just 500-2,000 tokens.

## Five Frameworks, One Skill Library

No other open-source skills library maps every skill to all five frameworks. One skill, five compliance checkboxes. This unified cross-framework coverage is what makes Anthropic Cybersecurity Skills uniquely valuable for organizations that need to demonstrate compliance across multiple standards simultaneously.

| Framework | Version | Scope | What It Maps |
|---|---|---|---|
| MITRE ATT&CK | v18 | 14 tactics, 200+ techniques | Adversary behaviors and TTPs |
| NIST CSF 2.0 | 2.0 | 6 functions, 22 categories | Organizational security posture |
| MITRE ATLAS | v5.4 | 16 tactics, 84 techniques | AI/ML adversarial threats |
| MITRE D3FEND | v1.3 | 7 categories, 267 techniques | Defensive countermeasures |
| NIST AI RMF | 1.0 | 4 functions, 72 subcategories | AI risk management |

The ATT&CK coverage spans all 14 Enterprise tactics from Reconnaissance (TA0043) through Impact (TA0040), with strong coverage in Execution, Persistence, Privilege Escalation, Defense Evasion, Credential Access, Lateral Movement, Command and Control, and Exfiltration. The NIST CSF 2.0 alignment covers all six functions including the newly added Govern function, with 106 subcategory references across Detect (200+ skills), Respond (160+ skills), and Protect (150+ skills).

> **Amazing:** A single skill like `analyzing-network-traffic-of-malware` maps across all five frameworks simultaneously: ATT&CK T1071, NIST CSF DE.CM, ATLAS AML.T0047, D3FEND D3-NTA, and AI RMF MEASURE-2.6. One skill, five compliance checkboxes.

## How AI Agents Use These Skills

The progressive disclosure architecture is what makes this library practical for AI agents. Here is how it works in practice:

```
User prompt: "Analyze this memory dump for signs of credential theft"

Agent's internal process:

  1. Scans 754 skill frontmatters (~30 tokens each)
     -> identifies 12 relevant skills by matching tags, description, domain

  2. Loads top 3 matches:
     - performing-memory-forensics-with-volatility3
     - hunting-for-credential-dumping-lsass
     - analyzing-windows-event-logs-for-credential-access

  3. Executes the structured Workflow section step-by-step
     -> runs Volatility3 plugins, checks LSASS access patterns,
        correlates with event log evidence

  4. Validates results using the Verification section
     -> confirms IOCs, maps findings to ATT&CK T1003 (Credential Dumping)
```

Without these skills, the agent guesses at tool commands and misses critical steps. With them, it follows the same playbook a senior DFIR analyst would use.

## Skill Anatomy

Every skill follows a consistent directory structure:

```
skills/performing-memory-forensics-with-volatility3/
  SKILL.md              <- Skill definition (YAML frontmatter + Markdown body)
  references/
    standards.md         <- MITRE ATT&CK, ATLAS, D3FEND, NIST mappings
    workflows.md         <- Deep technical procedure reference
  scripts/
    agent.py             <- Working helper scripts
  assets/
    template.md          <- Filled-in checklists and report templates
```

The YAML frontmatter includes structured metadata for agent discovery:

```yaml
---
name: performing-memory-forensics-with-volatility3
description: >-
  Analyze memory dumps to extract running processes, network connections,
  injected code, and malware artifacts using the Volatility3 framework.
domain: cybersecurity
subdomain: digital-forensics
tags: [forensics, memory-analysis, volatility3, incident-response, dfir]
atlas_techniques: [AML.T0047]
d3fend_techniques: [D3-MA, D3-PSMD]
nist_ai_rmf: [MEASURE-2.6]
nist_csf: [DE.CM-01, RS.AN-03]
version: "1.2"
author: mukul975
license: Apache-2.0
---
```

The Markdown body contains four structured sections: When to Use (trigger conditions), Prerequisites (required tools and access), Workflow (step-by-step execution guide with specific commands), and Verification (how to confirm successful execution).

> **Takeaway:** With just `npx skills add mukul975/Anthropic-Cybersecurity-Skills`, Claude Code gains a complete cybersecurity knowledge base covering 26 domains and 5 frameworks. No manual configuration needed.

![Features Diagram](/assets/img/diagrams/anthropic-cybersecurity-skills/anthropic-cybersecurity-skills-features.svg)

### Understanding the Features

The features diagram highlights the key capabilities that make Anthropic Cybersecurity Skills stand out from other security tool repositories:

**Progressive Disclosure Architecture**
The library uses a two-tier loading model. At the frontmatter level, each skill costs approximately 30 tokens to scan, allowing an agent to evaluate all 754 skills in a single context pass. When a skill is selected for execution, the full Markdown workflow loads at 500-2,000 tokens. This design prevents context window exhaustion while maintaining comprehensive coverage.

**agentskills.io Standard**
Every skill follows the agentskills.io open standard with YAML frontmatter for machine-readable discovery and structured Markdown for human-readable execution. The standard defines four required sections: When to Use, Prerequisites, Workflow, and Verification. This consistency enables any compatible platform to parse and execute skills without custom integration code.

**Cross-Framework Mapping**
The five-framework mapping is the library's most distinctive feature. Each skill carries explicit references to ATT&CK techniques, NIST CSF categories, ATLAS techniques, D3FEND techniques, and AI RMF subcategories. This means a single skill execution can satisfy compliance requirements across multiple frameworks simultaneously.

**Validation Pipeline**
The repository includes a `validate-skill.py` tool that enforces required fields (name, description, domain, subdomain, tags) and validates subdomain aliases. Every PR is reviewed for technical accuracy and agentskills.io standard compliance within 48 hours.

**Community-Driven Model**
The project operates under Apache 2.0 license with open contribution guidelines. Domains like Deception Technology (2 skills) and Compliance and Governance (5 skills) actively need contributions. The template in CONTRIBUTING.md makes it straightforward to add new skills.

**Platform Compatibility**
The library works with 26+ AI platforms including Claude Code, GitHub Copilot, Cursor, OpenAI Codex CLI, Gemini CLI, and agent frameworks like CrewAI, LangChain, AutoGen, and Semantic Kernel. Any platform supporting the agentskills.io standard can load these skills with zero configuration.

> **Important:** The validation pipeline enforces required YAML fields and subdomain aliases for every skill, and every PR is reviewed for technical accuracy within 48 hours. This ensures the library maintains consistent quality across all 754 skills.

## Installation

### Option 1: npx (Recommended)

The fastest way to get started is with the npx command:

```bash
npx skills add mukul975/Anthropic-Cybersecurity-Skills
```

This works immediately with Claude Code, GitHub Copilot, OpenAI Codex CLI, Cursor, and any agentskills.io-compatible platform.

### Option 2: Git Clone

For full control and customization:

```bash
git clone https://github.com/mukul975/Anthropic-Cybersecurity-Skills.git
cd Anthropic-Cybersecurity-Skills
```

### Option 3: Using with Specific Platforms

**Claude Code:**
```bash
# After cloning, point Claude Code at the skills directory
npx skills add mukul975/Anthropic-Cybersecurity-Skills
```

**GitHub Copilot:**
The skills are automatically recognized when added via the npx command above.

**Agent Frameworks (CrewAI, LangChain):**
```bash
# Clone and reference in your agent configuration
git clone https://github.com/mukul975/Anthropic-Cybersecurity-Skills.git
# Reference the skills directory in your framework's skill configuration
```

### Validating Skills

The repository includes a validation tool:

```bash
cd Anthropic-Cybersecurity-Skills
python tools/validate-skill.py --all
```

This validates all 754 skills for required fields, subdomain aliases, and agentskills.io standard compliance.

## 26 Security Domains

The library covers 26 security domains with varying depth:

| Domain | Skills | Key Capabilities |
|---|---|---|
| Cloud Security | 60 | AWS, Azure, GCP hardening, CSPM, cloud forensics |
| Threat Hunting | 55 | Hypothesis-driven hunts, LOTL detection, behavioral analytics |
| Threat Intelligence | 50 | STIX/TAXII, MISP, feed integration, actor profiling |
| Web Application Security | 42 | OWASP Top 10, SQLi, XSS, SSRF, deserialization |
| Network Security | 40 | IDS/IPS, firewall rules, VLAN segmentation, traffic analysis |
| Malware Analysis | 39 | Static/dynamic analysis, reverse engineering, sandboxing |
| Digital Forensics | 37 | Disk imaging, memory forensics, timeline reconstruction |
| Security Operations | 36 | SIEM correlation, log analysis, alert triage |
| Identity and Access Management | 35 | IAM policies, PAM, zero trust identity, Okta, SailPoint |
| SOC Operations | 33 | Playbooks, escalation workflows, metrics, tabletop exercises |
| Container Security | 30 | K8s RBAC, image scanning, Falco, container forensics |
| OT/ICS Security | 28 | Modbus, DNP3, IEC 62443, historian defense, SCADA |
| API Security | 28 | GraphQL, REST, OWASP API Top 10, WAF bypass |
| Vulnerability Management | 25 | Nessus, scanning workflows, patch prioritization, CVSS |
| Incident Response | 25 | Breach containment, ransomware response, IR playbooks |
| Red Teaming | 24 | Full-scope engagements, AD attacks, phishing simulation |
| Penetration Testing | 23 | Network, web, cloud, mobile, wireless pentesting |
| Endpoint Security | 17 | EDR, LOTL detection, fileless malware, persistence hunting |
| DevSecOps | 17 | CI/CD security, code signing, Terraform auditing |
| Phishing Defense | 16 | Email authentication, BEC detection, phishing IR |
| Cryptography | 14 | TLS, Ed25519, certificate transparency, key management |
| Zero Trust Architecture | 13 | BeyondCorp, CISA maturity model, microsegmentation |
| Mobile Security | 12 | Android/iOS analysis, mobile pentesting, MDM forensics |
| Ransomware Defense | 7 | Precursor detection, response, recovery, encryption analysis |
| Compliance and Governance | 5 | CIS benchmarks, SOC 2, regulatory frameworks |
| Deception Technology | 2 | Honeytokens, breach detection canaries |

## Example: Memory Forensics Skill

To illustrate the depth of a single skill, here is what the `performing-memory-forensics-with-volatility3` skill provides:

**When to Use:**
- Analyzing a RAM dump from a compromised or suspect system
- During incident response to identify running malware, injected code, or rootkits
- Extracting credentials, encryption keys, or network connections from memory
- Detecting process hollowing, DLL injection, or hidden processes

**Prerequisites:**
- Python 3.7+ installed
- Volatility 3 framework installed (`pip install volatility3`)
- Memory dump in raw, ELF, or crash dump format
- Appropriate symbol tables (ISF files) for the target OS version

**Workflow Steps:**
1. Acquire memory dump and install Volatility 3
2. Identify the operating system profile
3. List running processes and check for hidden processes
4. Extract network connections and DNS cache
5. Scan for injected code and DLL side-loading
6. Extract credentials and encryption keys
7. Generate timeline and correlate findings

**Framework Mappings:**
- ATT&CK: T1071, T1003, T1056, T1569
- NIST CSF: DE.CM-01, RS.AN-03
- ATLAS: AML.T0047
- D3FEND: D3-MA, D3-PSMD
- AI RMF: MEASURE-2.6

## Contributing

The project welcomes community contributions. Domains like Deception Technology (2 skills) and Compliance and Governance (5 skills) need the most help. To add a new skill:

1. Create a new directory: `skills/your-skill-name/`
2. Add a `SKILL.md` file with required YAML frontmatter (name, description, domain, subdomain, tags)
3. Write clear step-by-step instructions using the required sections
4. Optionally add `references/standards.md` for framework mappings and `scripts/` for helper scripts
5. Submit a PR with the title `Add skill: your-skill-name`

Every PR is reviewed for technical accuracy and agentskills.io standard compliance within 48 hours.

## Links

- **GitHub Repository:** [https://github.com/mukul975/Anthropic-Cybersecurity-Skills](https://github.com/mukul975/Anthropic-Cybersecurity-Skills)
- **agentskills.io Standard:** [https://agentskills.io](https://agentskills.io)
- **ATT&CK Navigator Layer:** Available in the v1.0.0 release assets
- **License:** Apache License 2.0