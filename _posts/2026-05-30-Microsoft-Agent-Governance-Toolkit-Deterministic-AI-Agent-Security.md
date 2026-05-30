---
layout: post
title: "Microsoft Agent Governance Toolkit: Deterministic Security for AI Agents"
description: "Learn how Microsoft's Agent Governance Toolkit provides deterministic application-layer interception for AI agents, making misbehavior structurally impossible through policy-as-code, zero-trust identity, and OWASP Agentic Top 10 coverage."
date: 2026-05-30
header-img: "img/post-bg.jpg"
permalink: /Microsoft-Agent-Governance-Toolkit-Deterministic-AI-Agent-Security/
featured-img: ai-coding-frameworks/ai-coding-frameworks
categories: [AI Agents, Security, Developer Tools]
tags: [AI agent governance, agent security, OWASP agentic, MCP security, policy as code, zero trust identity, Microsoft AGT, AI agent safety, deterministic interception, privilege rings]
keywords: "Microsoft agent governance toolkit tutorial, how to secure AI agents with AGT, OWASP agentic top 10 controls, AI agent policy as code, MCP security gateway for agents, deterministic AI agent safety, zero trust identity for AI agents, agent governance vs prompt safety, AI agent privilege rings, Microsoft AGT installation guide"
author: "PyShine"
---

AI agents are moving from prototypes to production systems that call tools, browse the web, query databases, and delegate tasks to other agents. Once deployed, these agents make decisions autonomously, and the consequences of a misbehaving agent can be severe. The Microsoft **agent governance toolkit** (AGT) addresses this gap by providing deterministic application-layer interception that makes agent misbehavior structurally impossible rather than just unlikely. Unlike prompt-level safety measures that politely ask models to follow rules, AGT intercepts every tool call, message send, and delegation in deterministic code before the model's intent reaches the wire. This post walks through AGT's architecture, policy engine, privilege rings, MCP security gateway, and OWASP coverage, with installation instructions and framework integration details.

> **Key Insight:** Research shows that prompt-level safety measures have a 100% attack success rate (OWASP LLM01:2025). AGT's deterministic interception makes misbehavior structurally impossible rather than just unlikely.

## The Problem with Prompt-Level Safety

Current approaches to AI agent safety rely on prompt engineering -- instructing the model to follow rules, avoid harmful actions, and stay within bounds. This is fundamentally flawed because it treats safety as a polite request to a stochastic system. OWASP LLM01:2025 states explicitly that "it is unclear if there are fool-proof methods of prevention for prompt injection." The published research backs this up: Andriushchenko et al. (ICLR 2025) report a 100% attack success rate on GPT-4o, GPT-3.5, Claude 3, and Llama-3 using adaptive attacks evaluated against the JailbreakBench benchmark (Chao et al., NeurIPS 2024).

Microsoft's own AI Red Teaming Agent formalizes Attack Success Rate (ASR) as the canonical metric for this class of failure. Their report on red teaming 100 generative AI products reinforces the point: "mitigations do not eliminate risk entirely" and red teaming must be a continuous process because model-layer defenses are probabilistic by construction.

AGT does not try to win that fight inside the prompt. Every tool call, message send, and delegation is intercepted in deterministic application code before the model's intent reaches the wire. Actions the AGT kernel denies are not "unlikely." They are structurally impossible. That is the difference between asking an agent to behave and making it incapable of misbehaving.

## What is the Agent Governance Toolkit?

The Agent Governance Toolkit (AGT) is Microsoft's open-source framework for policy enforcement, identity management, sandboxing, and site reliability engineering for autonomous AI agents. Currently in Public Preview at version 4.0.0, AGT provides a single `pip install` that works with any agent framework.

AGT is built around three fundamental questions that every production agent system must answer:

**1. Is this action allowed?** An agent with access to `send_email` and `query_database` should not be able to `drop_table`. OAuth scopes and IAM roles control which services an agent can reach, not what it does once connected. AGT's policy engine evaluates every action against configurable YAML policies before execution.

**2. Which agent did this?** In a multi-agent system, five agents might share a single API key. When something goes wrong, "an agent did it" is not an incident response. AGT's zero-trust identity mesh assigns cryptographic identities to every agent using Ed25519 keys, SPIFFE certificates, and DID verification.

**3. Can you prove what happened?** Auditors and regulators need tamper-evident records of every decision: what policy was active, what the agent requested, and why it was allowed or denied. AGT's Merkle chain audit logs provide cryptographically anchored, append-only decision records.

## Architecture Overview

![AGT Architecture Overview](/assets/img/diagrams/agent-governance-toolkit/agt-architecture.svg)

The architecture diagram above illustrates the eight major packages that compose the Agent Governance Toolkit and how they interconnect to form a complete governance layer around any AI agent. The diagram shows the flow of control from the agent's initial action request through the governance stack and out to the actual tool execution, with every intermediate step providing a distinct security function.

Understanding this architecture is essential because it reveals how AGT achieves its core guarantee: no action reaches the wire without passing through every governance layer. The eight packages are not independent modules that can be mixed and matched arbitrarily -- they form a coherent stack where each layer depends on the guarantees provided by the layers below it.

At the center sits the **Agent OS** package, which houses the policy engine and governance gate. This is the core decision-making component that evaluates every action an agent attempts to perform against the loaded policy documents. The policy engine supports three policy formats: YAML for human-readable declarations, OPA (Open Policy Agent) for teams already using Rego-based policies, and Cedar for AWS-compatible authorization policies.

This multi-format support means organizations do not need to rewrite their existing policy infrastructure to adopt AGT. The governance gate is the interception point where all tool calls, message sends, and delegation requests are captured before they reach the wire. It acts as a mandatory checkpoint that no agent can bypass, regardless of the framework it runs on. The governance gate implements a fail-closed design: if the policy engine is unavailable or returns an error, the default action is to deny the request.

The **Agent Mesh** package handles agent discovery, routing, and the zero-trust identity mesh. It provides Ed25519 key generation for agent identity, SPIFFE certificate issuance for workload identity, DID (Decentralized Identifier) verification for cross-mesh trust, and mutual TLS (mTLS) for secure agent-to-agent communication. When an agent registers with the mesh, it receives a cryptographic identity that follows it through every interaction, enabling precise attribution of actions.

This solves the "which agent did this?" problem by ensuring that every action can be traced back to a specific, cryptographically identified agent rather than a shared API key. The mesh also maintains a trust score for each agent, which is updated based on the agent's behavior over time and influences which operations the agent is permitted to perform.

The **Agent Runtime** package implements execution sandboxing through four privilege rings, modeled after operating system protection rings. Ring 0 (kernel) provides full system access for trusted governance operations, Ring 1 (system) allows framework-level operations, Ring 2 (user) is the default for normal agent tool calls, and Ring 3 (untrusted) restricts agents to the narrowest set of capabilities.

This graduated model ensures that agents only access the resources their trust level warrants, preventing a low-trust agent from performing high-privilege operations. The runtime also supports saga orchestration for multi-step agent workflows, ensuring that partial failures are handled gracefully with compensation actions.

The **Agent SRE** package brings site reliability engineering to agents with SLO monitoring, error budgets, circuit breakers, and chaos testing. When an agent begins failing at an unacceptable rate, the circuit breaker trips and prevents further damage. The error budget system tracks how many failures are acceptable within a time window, and when the budget is exhausted, the agent is automatically degraded or suspended. The chaos testing module allows teams to inject failures in a controlled manner, verifying that circuit breakers work as expected before a real incident occurs.

The **Agent Compliance** package provides OWASP verification, policy linting, and integrity checks. It includes the PromptDefense Evaluator, which performs 12-vector prompt injection audits, and the `agt verify` CLI command that generates compliance evidence for auditors. The policy linter catches common policy misconfigurations before they reach production, such as overly permissive default actions or conflicting rules. The integrity check system verifies that installed packages match their expected checksums, detecting supply chain attacks.

The **Agent Marketplace** package governs plugins through trust scoring, signing verification, and schema drift detection. Before a plugin is loaded, its signature is verified against a trusted certificate chain, its trust score is checked against configurable thresholds, and its schema is compared to the expected interface to detect poisoning or typosquatting attacks. This supply chain protection extends AGT's governance boundary to third-party extensions.

The **Agent Lightning** package governs reinforcement learning training with violation penalties. When an agent is trained using RL techniques, Lightning ensures that reward functions do not incentivize policy-violating behavior by applying penalty signals for governance violations during training. This prevents the common failure mode where RL optimization finds reward hacks that violate safety constraints. Lightning integrates with popular RL frameworks and provides a governance-aware reward wrapper.

Finally, the **Agent Hypervisor** package provides execution audit logging through a Merkle chain, a delta engine for tracking state changes, and commitment anchoring for tamper-evident records. Every governance decision is recorded as a leaf in the Merkle tree, and the root hash is periodically anchored to an external commitment store, making it cryptographically impossible to alter historical decisions without detection. The delta engine tracks state changes between decisions, providing a complete replay capability for incident investigation. The commitment anchoring system supports multiple backends, including local files, cloud storage, and distributed ledgers.

## How Policy Evaluation Works

![Policy Evaluation Flow](/assets/img/diagrams/agent-governance-toolkit/agt-policy-flow.svg)

The policy evaluation flow diagram shows the step-by-step process that occurs every time an agent attempts an action. The diagram traces the request from the agent through identity verification, policy evaluation, and audit logging, with three possible outcomes: allow, deny, or require approval.

Understanding this flow is critical because it demonstrates how AGT achieves deterministic governance -- every action follows the same path, and no shortcut exists that would allow an action to bypass any stage. The linear pipeline design is intentional: each stage must complete before the next stage begins, and a failure at any stage immediately terminates the evaluation with a deny decision.

The process begins when an agent issues a tool call, message send, or delegation request. This request is intercepted by the governance gate before it reaches the wire, ensuring that no action bypasses policy evaluation. The interception happens at the application middleware layer, which means it operates in deterministic code rather than relying on the model's compliance with prompt-level instructions.

This is the fundamental architectural decision that separates AGT from prompt-based safety approaches. The governance gate is implemented as a framework adapter that inserts itself into the tool call pipeline, and it works identically across all 15+ supported frameworks.

The intercepted request is first passed to the identity verification stage, where the Agent Mesh confirms the agent's cryptographic identity. The agent's Ed25519 key or SPIFFE certificate is validated, and its DID is resolved to confirm that the agent is registered and in good standing within the trust mesh. The identity check also verifies that the agent's trust score meets the minimum threshold for the requested operation.

If the identity cannot be verified, the request is immediately denied with a `GovernanceDenied` exception, and the denial is recorded in the audit log with the reason "identity verification failed." This immediate denial prevents unregistered or compromised agents from proceeding further in the evaluation pipeline, stopping potential attacks at the earliest possible point.

Once identity is confirmed, the request proceeds to the policy engine. The policy engine loads the active policy document, which is defined in YAML format with an `apiVersion`, `name`, `default_action`, and a list of `rules`. Each rule contains a `name`, a `condition` expression, an `action` (allow, deny, or require_approval), and an optional `description`.

The engine evaluates rules in priority order, checking each rule's condition against the action context, which includes the action type, the agent's identity, the target resource, and any additional parameters. The priority ordering ensures that more specific rules (higher priority) take precedence over general rules, preventing conflicts where a broad allow rule might override a specific deny rule. The condition language supports comparison operators, set membership checks, and logical combinators.

If a rule's condition matches, the corresponding action is taken. An `allow` action permits the request to proceed to execution. A `deny` action blocks the request and raises a `GovernanceDenied` exception with the rule name and description, providing clear feedback about why the action was blocked.

A `require_approval` action pauses the request and routes it to the designated approvers (specified in the rule's `approvers` field) for human review. The approval workflow supports configurable timeouts -- if no approver responds within the timeout window, the request is automatically denied, preventing indefinite pending states.

If no rule matches, the `default_action` from the policy document is applied, which can be set to either `allow` or `deny` depending on the organization's risk posture. A fail-closed default (deny) is recommended for production deployments, as it ensures that any unanticipated action is blocked rather than allowed.

After the policy decision is made, the result is recorded in the Merkle chain audit log. This record includes the policy version, the matched rule (if any), the decision (allow, deny, or pending approval), the agent's identity, the action requested, and a timestamp. The record is hashed and appended to the Merkle tree, making it tamper-evident.

Periodically, the Merkle root is anchored to an external commitment store, providing cryptographic proof that the log has not been altered. This audit trail answers the third governance question -- "Can you prove what happened?" -- with mathematically verifiable evidence rather than trust-based assertions. The Merkle chain structure means that verifying the integrity of the entire log requires only the current root hash, not a full replay of every decision.

The following YAML policy demonstrates the key concepts. It defines a production policy with a default allow action and two rules: one that blocks destructive database operations and another that requires human approval for email sending.

```yaml
# policy.yaml
apiVersion: governance.toolkit/v1
name: production-policy
default_action: allow
rules:
  - name: block-destructive
    condition: "action.type in ['drop', 'delete', 'truncate']"
    action: deny
    description: "Destructive operations require human approval"

  - name: require-approval-for-send
    condition: "action.type == 'send_email'"
    action: require_approval
    approvers: ["security-team"]
```

When an agent attempts a blocked action, the result is immediate and deterministic:

```python
>>> safe_tool(action="read", table="users")
{'table': 'users', 'rows': 42}

>>> safe_tool(action="drop", table="users")
GovernanceDenied: Action denied by policy rule 'block-destructive':
  Destructive operations require human approval
```

> **Takeaway:** With just two lines of code - `govern(agent)` - any AI agent gains a complete governance layer with policy evaluation, identity verification, and tamper-evident audit logging.

## Four Privilege Rings

![Four Privilege Rings](/assets/img/diagrams/agent-governance-toolkit/agt-privilege-rings.svg)

The four privilege rings diagram illustrates AGT's execution sandboxing model, which borrows the protection ring concept from operating system design and applies it to AI agent governance. The diagram shows four concentric rings, with Ring 0 at the center and Ring 3 at the outer edge, each representing a decreasing level of privilege and an increasing level of restriction.

This model provides graduated access levels that constrain what an agent can do based on its trust level and the nature of the operation it is attempting. The concentric layout is intentional: an operation at a given ring can access resources at its own ring and outer rings, but never at an inner ring, mirroring how CPU protection rings prevent user-mode code from accessing kernel-mode memory. This hardware-inspired design provides a proven model for privilege separation that has been battle-tested in operating systems for decades.

**Ring 0 (Kernel)** is the innermost and most privileged ring. It is reserved for governance operations themselves -- policy evaluation, identity verification, audit logging, and the governance gate. Only the AGT kernel operates at Ring 0, ensuring that the governance layer cannot be bypassed or tampered with by the agents it governs.

This separation is critical: if agents could modify the policy engine or audit log, the entire governance model would be undermined. Ring 0 operations include loading policy documents, evaluating rules, recording Merkle chain entries, and managing the trust mesh. The kernel ring is non-negotiable -- no agent, regardless of its trust score, ever executes at Ring 0.

This is the foundation of the entire privilege model: the governance infrastructure must exist at a higher privilege level than the agents it governs, just as the operating system kernel must exist at a higher privilege level than user processes. Without this separation, any agent could potentially modify the governance rules to permit its own malicious actions.

**Ring 1 (System)** provides framework-level access. Operations at this ring include agent lifecycle management (registration, deregistration, heartbeat), inter-agent communication protocols, and framework adapter operations. Agents operating at Ring 1 can coordinate with other agents through the mesh, but they cannot modify governance policies or audit records.

This ring is typically used by agent orchestrators and framework middleware that need to manage agent behavior but should not have unrestricted access to the underlying system. The system ring provides enough privilege to manage agent coordination without exposing the governance infrastructure. Framework adapters that implement the governance gate operate at Ring 1, giving them the ability to intercept and route agent requests without being able to modify the policy evaluation logic itself.

**Ring 2 (User)** is the default ring for normal agent tool calls. When an agent calls `web_search`, `query_database`, or `read_file`, these operations execute at Ring 2. The policy engine evaluates these calls against the loaded policy, and the agent's identity is verified, but the operations themselves are not further restricted beyond what the policy allows.

Most production agents operate primarily at Ring 2, with policy enforcement providing the guardrails for what they can and cannot do. The user ring balances flexibility with safety, allowing agents to perform useful work while preventing them from accessing governance infrastructure. When a new agent is registered with the mesh, it is assigned to Ring 2 by default unless its configuration explicitly specifies a different ring.

**Ring 3 (Untrusted)** is the outermost and most restricted ring. Agents at Ring 3 have the narrowest set of capabilities and are subject to the strictest policy enforcement. This ring is used for agents that handle untrusted input, interact with external services, or have been flagged by the trust scoring system.

Ring 3 agents may be restricted from making outgoing network calls, accessing sensitive data, or delegating to other agents. The Agent Runtime enforces these restrictions at the execution level, making it structurally impossible for a Ring 3 agent to escalate its privileges. An agent can be demoted to Ring 3 dynamically if its trust score drops below the configured threshold, providing an automatic containment mechanism for agents that exhibit suspicious behavior.

This dynamic demotion is a key safety feature: it allows the system to automatically contain a compromised agent without requiring human intervention. The trust scoring system continuously evaluates agent behavior, and when the score crosses the demotion threshold, the runtime immediately restricts the agent's capabilities to Ring 3 levels.

The ring model also defines escalation paths. An agent can request a higher privilege level through a formal escalation process that requires policy approval. For example, a Ring 3 agent that needs to make a specific network call can request a temporary Ring 2 escalation, which the policy engine evaluates against the escalation rules.

If approved, the agent receives a time-limited capability token that allows the specific operation. Once the token expires, the agent returns to its original ring. This prevents privilege escalation attacks while allowing agents to perform their intended functions. The escalation process is logged in the Merkle chain audit log, ensuring that every privilege change is traceable and auditable.

The capability token model is inspired by capability-based security systems, where possession of a token grants specific rights without requiring the holder to have a broader set of permissions. This is a well-studied security pattern that provides fine-grained access control without the complexity of managing broad permission sets.

## MCP Security Gateway

![MCP Security Gateway](/assets/img/diagrams/agent-governance-toolkit/agt-mcp-gateway.svg)

The MCP Security Gateway diagram shows the dual-stage pipeline that governs all Model Context Protocol (MCP) traffic between agents and tool servers. The diagram depicts the two interception points: one before the tool call reaches the MCP server (Stage 1) and one before the response reaches the agent (Stage 2).

Between these two stages, the actual tool execution occurs on the MCP server, but the gateway maintains control by intercepting both the request and the response. MCP is the emerging standard for connecting AI agents to external tools and data sources, but it introduces significant security risks: tool poisoning, schema drift, hidden instructions embedded in tool responses, and typosquatting through malicious tool names.

The MCP Security Gateway addresses these risks through a two-stage interception model that provides bidirectional protection. The gateway is specified in the formal MCP Security Gateway 1.0 specification, which includes 127 conformance tests verifying its behavior.

**Stage 1: Pre-Execution Interception.** Before a tool call is dispatched to the MCP server, the gateway intercepts the request and performs a series of security checks. First, it verifies the tool server's identity using the trust mesh, confirming that the server holds a valid SPIFFE certificate and its trust score meets the configured threshold. This prevents connections to unregistered or low-trust tool servers.

Next, it checks the tool's schema against the expected interface, detecting any schema drift that might indicate a poisoning attack where a tool's interface has been modified to accept malicious parameters. Schema drift detection compares the current tool schema (parameter names, types, required fields) against the previously registered schema, flagging any additions, removals, or type changes.

The gateway also scans the tool call parameters for hidden instructions -- strings that might contain prompt injection payloads designed to manipulate the agent's behavior through the tool response. This scanning uses pattern matching and heuristic analysis to detect common injection patterns, including known jailbreak templates, instruction override sequences, and role-playing prompts.

Finally, it checks the tool name against a typosquatting database to detect tools with names that closely resemble trusted tools but are actually malicious imitations, such as "web_serch" instead of "web_search" or "databse_query" instead of "database_query."

**Stage 2: Post-Response Scanning.** After the MCP server returns a response, the gateway intercepts it before delivering it to the agent. The response scanner checks for hidden instructions embedded in the tool output, which is a common attack vector where a compromised tool server returns data containing prompt injection payloads.

For example, a database query tool might return a row containing "IGNORE ALL PREVIOUS INSTRUCTIONS AND..." which, if delivered to the agent unchecked, could override the agent's safety constraints. The scanner uses both pattern-based detection for known injection templates and semantic analysis for novel injection attempts.

The scanner also checks for data exfiltration patterns, such as responses that contain API keys, credentials, or PII that should not be returned to the agent. PII redaction can automatically mask sensitive fields like social security numbers, credit card numbers, and email addresses before the response reaches the agent.

Content policy filters evaluate the response against organizational content policies, blocking responses that violate data handling rules such as geographic restrictions or classification levels. The response scanner operates on the complete response before any portion is delivered to the agent, ensuring that partial delivery of malicious content is not possible.

The dual-stage model is essential because MCP attacks can originate from either direction. A compromised tool server can inject malicious instructions through its responses (Stage 2 catches this), and a compromised agent can send malicious parameters to a tool server (Stage 1 catches this).

By intercepting traffic in both directions, the MCP Security Gateway provides comprehensive protection that single-direction filtering cannot achieve. This is analogous to how network firewalls inspect both inbound and outbound traffic -- a one-way firewall would leave half the attack surface unprotected. The dual-stage design also means that even if one stage is bypassed through a novel attack technique, the other stage provides a second line of defense, implementing defense in depth at the protocol level.

The gateway also maintains a drift monitoring system that tracks changes in tool schemas and behavior over time. If a tool's response format changes unexpectedly, or if its latency patterns shift significantly, the gateway raises an alert and may temporarily suspend the tool pending investigation.

This continuous monitoring catches slow-drip attacks where a tool is gradually modified over time to avoid detection. The drift monitor maintains a baseline profile for each registered tool, including its expected schema, typical response sizes, normal latency ranges, and historical error rates. Deviations from these baselines trigger alerts at configurable sensitivity levels, allowing teams to balance security strictness against operational noise.

The drift monitor also supports automated quarantine: when a tool's drift score exceeds a critical threshold, the gateway can automatically suspend the tool and route pending requests to alternative tools, preventing service disruption while the suspicious tool is investigated.

> **Important:** The MCP Security Gateway implements a dual-stage pipeline that intercepts tool calls before execution and scans responses before delivery, catching tool poisoning, schema drift, and hidden instruction attacks that bypass prompt-level safety.

## OWASP Agentic Top 10 Coverage

The OWASP Agentic Security Intelligence (ASI) 2026 Top 10 defines the most critical security risks for autonomous AI agent systems. AGT provides deterministic controls for the majority of these risks, making it the most comprehensive agent security framework available.

| OWASP ASI Risk | AGT Control | Coverage |
|---|---|---|
| ASI01: Agent Prompt Injection | Policy engine intercepts before execution | Full |
| ASI02: Sensitive Data Disclosure | PII redaction, data exfiltration checks | Full |
| ASI03: Supply Chain Vulnerability | Plugin signing, trust scoring, schema drift detection | Full |
| ASI04: Insecure Output Handling | Response scanning, content policy filter | Full |
| ASI05: Excessive Agency | Capability model, privilege rings, resource limits | Full |
| ASI06: Prompt Injection in MCP | MCP Security Gateway dual-stage pipeline | Full |
| ASI07: Insecure Identity Management | Zero-trust mesh, Ed25519/SPIFFE/DID | Full |
| ASI08: Unobservable Agent Behavior | Merkle chain audit logs, commitment anchoring | Full |
| ASI09: Insecure Plugin Ecosystem | Marketplace governance, signing, trust scoring | Partial |
| ASI10: Unbounded Resource Consumption | SRE circuit breakers, error budgets | Partial |
| ASI11: Cross-Agent Trust Escalation | Trust scoring, mTLS, identity verification | Partial |

The eight full-coverage risks are addressed by deterministic controls that make violations structurally impossible. For example, ASI01 (Agent Prompt Injection) is fully covered because the policy engine intercepts every action before execution, regardless of whether the agent's prompt has been compromised. Even if an attacker successfully injects a malicious instruction into the agent's prompt, the resulting action will be evaluated against the policy and denied if it violates the rules. The policy engine does not care why the agent is attempting an action -- it only evaluates whether the action is permitted by the current policy.

ASI02 (Sensitive Data Disclosure) is fully covered through a combination of PII redaction in the MCP Security Gateway's response scanner and data exfiltration checks that detect when an agent attempts to access or transmit sensitive data outside of approved channels. The PII redaction system uses pattern matching and named entity recognition to identify and mask sensitive fields before they reach the agent, preventing accidental or intentional data leakage.

ASI05 (Excessive Agency) is fully covered through the capability model and privilege rings. The capability model defines what each agent is allowed to do, and the privilege rings enforce these limits at the execution level. An agent cannot perform operations beyond its assigned ring, and the policy engine evaluates every action against the capability model before allowing it to proceed. This prevents the common failure mode where an agent accumulates excessive permissions over time, a pattern known as "permission creep."

The three partial-coverage risks are addressed by controls that reduce but do not eliminate the risk. ASI09 (Insecure Plugin Ecosystem) is partially covered because while AGT provides plugin signing, trust scoring, and schema drift detection, the broader plugin ecosystem includes distribution channels and dependency chains that extend beyond AGT's governance boundary. A malicious plugin could be introduced through a compromised package registry or a supply chain attack on a dependency, which AGT's signing verification would not detect if the package was signed with a valid key from a compromised publisher.

ASI10 (Unbounded Resource Consumption) is partially covered because SRE circuit breakers and error budgets limit resource usage within the governed boundary, but agents may access external services that are not governed by AGT. ASI11 (Cross-Agent Trust Escalation) is partially covered because trust scoring and mTLS prevent unauthorized escalation within the mesh, but cross-mesh trust relationships require additional governance agreements that are outside AGT's scope.

> **Amazing:** The Agent Governance Toolkit covers 8 out of 11 OWASP Agentic Top 10 risks with full controls and 3 with partial coverage, achieving zero complete gaps in the most comprehensive agent security framework available.

## Installation and Quick Start

AGT is available on PyPI as a consolidated meta-package. The v4.0.0 release merged 45 previously separate packages into 5 top-level distributions for simpler installation and dependency management.

### Python Installation

```bash
# Install the full toolkit (all packages)
pip install agent-governance-toolkit[full]

# Or install individual distributions
pip install agent-governance-toolkit-core      # Policy engine, identity, audit, MCP gateway
pip install agent-governance-toolkit-runtime    # Privilege rings, saga orchestration
pip install agent-governance-toolkit-sre        # SLOs, circuit breakers, chaos testing
pip install agent-governance-toolkit-cli        # agt CLI, OWASP verification, policy linting
```

**Prerequisites:** Python 3.10+

### Other Language SDKs

```bash
# TypeScript / Node.js
npm install @microsoft/agent-governance-sdk

# .NET
dotnet add package Microsoft.AgentGovernance

# Rust
cargo add agent-governance

# Go
go get github.com/microsoft/agent-governance-toolkit/agent-governance-golang
```

### Govern Any Tool in Two Lines

```python
from agentmesh.governance import govern

safe_tool = govern(my_tool, policy="policy.yaml")   # every call checked, logged, enforced
```

That is the entire API surface for basic governance. `safe_tool` evaluates your YAML policy on every call, logs the decision to the Merkle chain audit log, and raises `GovernanceDenied` if the action is blocked.

### Programmatic Policy Evaluation

For more control, use the `PolicyEvaluator` API directly:

```python
from agent_os.policies import (
    PolicyEvaluator, PolicyDocument, PolicyRule,
    PolicyCondition, PolicyAction, PolicyOperator, PolicyDefaults
)

evaluator = PolicyEvaluator(policies=[PolicyDocument(
    name="my-policy", version="1.0",
    defaults=PolicyDefaults(action=PolicyAction.ALLOW),
    rules=[PolicyRule(
        name="block-dangerous-tools",
        condition=PolicyCondition(
            field="tool_name",
            operator=PolicyOperator.IN,
            value=["execute_code", "delete_file"]
        ),
        action=PolicyAction.DENY, priority=100,
    )],
)])

result = evaluator.evaluate({"tool_name": "web_search"})    # Allowed
result = evaluator.evaluate({"tool_name": "delete_file"})   # Blocked
```

### CLI Tools

AGT includes a CLI for common governance operations:

```bash
agt doctor                                        # check installation health
agt verify                                        # OWASP compliance check
agt verify --evidence ./agt-evidence.json --strict # fail CI on weak evidence
agt red-team scan ./prompts/ --min-grade B         # prompt injection audit
agt lint-policy policies/                          # validate policy files
```

### TypeScript Quick Start

```typescript
import { PolicyEngine } from "@microsoft/agent-governance-sdk";

const engine = new PolicyEngine([
  { action: "web_search", effect: "allow" },
  { action: "shell_exec", effect: "deny" },
]);
engine.evaluate("web_search"); // "allow"
engine.evaluate("shell_exec"); // "deny"
```

### .NET Quick Start

```csharp
using AgentGovernance;
using AgentGovernance.Policy;

var kernel = new GovernanceKernel(new GovernanceOptions
{
    PolicyPaths = new() { "policies/default.yaml" },
});
var result = kernel.EvaluateToolCall("did:mesh:agent-1", "web_search",
    new() { ["query"] = "latest AI news" });
```

## Framework Integration

AGT supports 15+ agent frameworks through adapters and middleware. The integration model is consistent across frameworks: wrap the agent with `govern()` and provide a policy document. The framework adapter handles the interception plumbing.

| Framework | Integration Type | Key Feature |
|-----------|-----------------|-------------|
| Microsoft Agent Framework | Native Middleware | Deepest integration, first-class support |
| Semantic Kernel | Native (.NET + Python) | Policy-gated semantic functions |
| AutoGen | Adapter | Multi-agent governance with role-based policies |
| LangGraph / LangChain | Adapter | Policy-gated tool calls with trust tiers |
| CrewAI | Adapter | Role-based policy enforcement per agent |
| OpenAI Agents SDK | Middleware | Policy-gated tool calls with trust tiers |
| Claude Code | Governance plugin | Plugin marketplace installation |
| Google ADK | Adapter | Cross-framework policy enforcement |
| LlamaIndex | Middleware | Query-time governance gates |
| Haystack | Pipeline | Pipeline-level policy injection |
| Mastra | Adapter | TypeScript-native governance |
| Dify | Plugin | Visual workflow governance |
| Azure AI Foundry | Deployment Guide | Cloud-integrated governance |
| GitHub Copilot CLI | Governance installer | CLI-native governance setup |
| HuggingFace smolagents | Adapter | Lightweight agent governance |

Each adapter implements the Framework Adapter Contract specification, which defines the behavioral contract for how governance interceptors integrate with agent frameworks. The specification includes 152 conformance tests that verify adapter behavior, ensuring that governance works consistently regardless of which framework you choose.

The adapter pattern works by inserting the governance gate into the framework's tool call pipeline. When an agent attempts to call a tool, the adapter intercepts the call, evaluates it against the policy, and either allows it to proceed, denies it with a `GovernanceDenied` exception, or routes it to human approvers. This interception happens at the application layer, not at the prompt layer, which is why it provides deterministic guarantees that prompt-level safety cannot.

## Specifications and Testing

AGT is built on a foundation of formal specifications. Every major component has an RFC 2119 specification that defines what implementations MUST, SHOULD, and MAY do. These specs are not documentation -- they are behavioral contracts backed by 992 conformance tests.

| Specification | Scope | Tests |
|---|---|---|
| Agent OS Policy Engine | Policy evaluation, rule merging, fail-closed semantics | 68 |
| AgentMesh Identity and Trust | Credentials, trust scoring, delegation chains | 135 |
| Agent Hypervisor Execution Control | Privilege rings, saga orchestration, kill switch | 80 |
| AgentMesh Trust and Coordination | Peer trust negotiation, mesh-wide policy | 62 |
| Agent SRE Governance | SLOs, error budgets, chaos, circuit breakers | 111 |
| MCP Security Gateway | Tool poisoning, drift detection, hidden instructions | 127 |
| Agent Lightning Fast-Path | RL training governance, violation penalties | 100 |
| Framework Adapter Contract | 10 adapter integrations, interceptor chain | 152 |
| Audit and Compliance | Merkle audit, compliance mapping, Decision BOM | 157 |
| AgentMesh Wire Protocol | Message format, routing, serialization | -- |

The 25 Architecture Decision Records (ADRs) document the reasoning behind major design decisions, providing context for why AGT works the way it does and making it easier for contributors to understand the architectural trade-offs.

## Standards Compliance

Beyond OWASP, AGT maps to several international standards and frameworks:

| Standard | Coverage |
|----------|----------|
| OWASP Agentic AI Top 10 | All ASI risk categories mapped with deterministic controls |
| NIST AI RMF 1.0 | Full GOVERN, MAP, MEASURE, MANAGE alignment |
| EU AI Act | Compliance mapping with automated evidence |
| SOC 2 | Control mapping with audit trail export |

The `agt verify` command generates compliance evidence that maps governance controls to these standards, making it easier to demonstrate compliance during audits. The `--strict` flag causes the command to fail if any control has weak evidence, enabling CI/CD integration where governance compliance is a deployment gate.

## Security Model

AGT enforces governance at the application middleware layer, not at the OS kernel level. The policy engine and agents share the same process boundary, which means that a sufficiently compromised agent could theoretically bypass the governance layer by exploiting process-level vulnerabilities. AGT is transparent about this limitation in its documentation.

The recommended production deployment runs each agent in a separate container for OS-level isolation, combining AGT's application-layer governance with container-level sandboxing for defense in depth. The security tooling includes CodeQL for Python and TypeScript SAST, Gitleaks for secret scanning, ClusterFuzzLite with 7 fuzz targets covering policy evaluation, injection, MCP, sandbox, and trust components, Dependabot across 13 ecosystems, and weekly OpenSSF Scorecard scoring.

## Conclusion

The Microsoft Agent Governance Toolkit represents a fundamental shift in how we approach AI agent safety. Instead of relying on prompt-level instructions that can be bypassed with 100% attack success rates, AGT provides deterministic application-layer interception that makes misbehavior structurally impossible. The three-question framework -- Is this action allowed? Which agent did this? Can you prove what happened? -- provides a clear mental model for reasoning about agent governance.

With 8 major packages covering policy enforcement, zero-trust identity, execution sandboxing, SRE, compliance, plugin governance, RL training governance, and audit logging, AGT addresses the full lifecycle of agent governance. The MCP Security Gateway's dual-stage pipeline provides comprehensive protection for the emerging MCP standard, and the 8/11 full OWASP Agentic Top 10 coverage demonstrates the framework's depth.

The two-line API surface -- `govern(my_tool, policy="policy.yaml")` -- makes it trivially easy to add governance to any agent, while the full `PolicyEvaluator` API and CLI tools provide the depth needed for production deployments. With 992 conformance tests, 10 formal specifications, and 25 ADRs, AGT is built on a rigorous foundation that ensures behavioral correctness and architectural transparency.

**Links:**

- GitHub: [https://github.com/microsoft/agent-governance-toolkit](https://github.com/microsoft/agent-governance-toolkit)
- Documentation: [https://microsoft.github.io/agent-governance-toolkit/](https://microsoft.github.io/agent-governance-toolkit/)
- PyPI: [https://pypi.org/project/agent-governance-toolkit/](https://pypi.org/project/agent-governance-toolkit/)
- npm: [@microsoft/agent-governance-sdk](https://www.npmjs.com/package/@microsoft/agent-governance-sdk)
- NuGet: [Microsoft.AgentGovernance](https://www.nuget.org/packages/Microsoft.AgentGovernance)