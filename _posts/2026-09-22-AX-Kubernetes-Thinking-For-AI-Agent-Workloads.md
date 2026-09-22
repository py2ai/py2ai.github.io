---
layout: post
title: "AX: Kubernetes Thinking for AI Agent Workloads"
description: "AX is Google's open-source, declarative orchestrator that runs AI agent workloads the way Kubernetes runs containers - tasks, workspaces, gateways, and models as YAML, sandboxed on Agent Substrate. We tour the architecture of the 6,000-star Go repository and show how to run your first task."
date: 2026-09-22
header-img: "img/post-bg.jpg"
permalink: /AX-Kubernetes-Thinking-For-AI-Agent-Workloads/
tags:
  - AI
  - Agents
  - Go
  - Kubernetes
  - Open Source
author: "PyShine"
---
# AX: Kubernetes Thinking for AI Agent Workloads

One agent in a terminal is a toy. A thousand agents in a cluster is an operations problem. [AX](https://github.com/google/ax) is Google's answer to that problem: an open-source, high-throughput orchestrator, written in Go and Apache-2.0 licensed, that runs autonomous agent workloads declaratively at cluster scale. If you have used Kubernetes, the repository's own framing lands immediately - declare your task in YAML, apply it, watch it come up, and look over the agent's shoulder. The project has moved fast since it appeared in March 2026, climbing past 6,000 stars with more than 2,300 stars in a single day on GitHub trending at the time of writing. What makes it interesting is not just the pedigree but the design: agents are treated as a genuinely new kind of workload - not stateless microservices, not run-to-completion batch jobs - and the whole control plane is built around that observation. It runs on top of [Agent Substrate](https://github.com/agent-substrate/substrate) for sandboxed execution and, in the project's own words, is built to run billions of tasks per cluster. In this tour we walk the real architecture from the source tree, then go from install to a first running task.

![Architecture overview of the AX repository showing the CLI, control plane, and sandboxed execution layers](/assets/img/diagrams/ax/ax-overview-architecture.svg)

## Why You Need This

Agents break the two workload models we already have. A microservice is long-lived and stateless; an agent accumulates state, calls out to model APIs and tool servers, needs strict isolation, and can burn money in a loop if nobody is watching. A batch job runs once and exits; an agent plans, delegates, retries, and fans work out over its lifetime. AX's response is four small primitives, each a YAML kind under `ax.io/v1alpha1`, documented in the project's [concepts guide](https://github.com/google/ax/blob/main/docs/concepts.md). A `Task` is the smallest unit of isolated execution: container image, command, compute limits, and a reference to a gateway. A `Workspace` pre-wires Git repositories, MCP servers, and skill packages so every agent starts warm instead of spending its first minutes installing toolchains. A `Gateway` is the network boundary: listeners the task exposes plus an explicit egress allowlist of hosts and ports, so an agent can reach your LLM provider and your Git host and nothing else. A `Model` is a named model configuration - provider, model identifier, generation parameters, and a Kubernetes secret reference for the API key - so rotating a credential is one `ax apply`, not a hunt through every agent's environment. That last point matters more than it sounds; the [incident patterns we covered in OpenAI's own postmortem](https://pyshine.com/OpenAI-Listed-Six-Ways-Its-Own-AI-Broke-the-Rules/) show exactly what uncontrolled agent behavior costs. AX makes the boring operational guarantees - isolation, egress fencing, lifecycle, checkpointing - someone else's problem, and it does so declaratively.

## How It Works

The design document in the repository is refreshingly honest about why this is not Kubernetes CRDs: storing millions of short-lived tasks as custom resources would push etcd past its comfort zone, with its single-digit gigabyte storage limits and write-rate bottlenecks. So AX keeps state in Redis and uses Redis Streams as the work queue between the API server and a horizontally scaled pool of controllers. The flow has four stages, and the source tree maps onto them cleanly.

![Detailed architecture diagram of the AX control plane and sandbox from the repository source](/assets/img/diagrams/ax/ax-architecture.svg)

First, the [AX CLI](https://github.com/google/ax/blob/main/DESIGN.md) - deliberately kubectl-shaped with `apply`, `get`, `describe`, `watch`, `delete` - talks gRPC to `ax-server`, a stateless API on port 8080 defined by the `ax.v1alpha1` protobuf service in `pkg/apis/v1alpha1`. The server validates manifests, persists them to the store, and publishes events. Second, the store. `internal/store` defines the interface with two implementations: Redis for production, where task hashes, event streams written with `XADD`, and pubsub live, and an in-memory backend for tests. Third, reconciliation. `ax-controller` replicas consume the stream with `XREADGROUP` worker pools (`internal/controller/worker.go`) and the reconciler (`internal/controller/reconciler.go`) drives each task through phases and conditions - `WorkspaceReady`, `GatewayReady`, and the one you actually wait on, `Ready` - toward the desired state, provisioning atespaces and actors through the Agent Substrate control API (`internal/substrate/client.go`) and applying the gateway's egress policy. Fourth, the sandbox. Every task container starts with `ax-task-runner` as PID 1 (`runner/runner.go`): it loads the task and workspace specs, starts a metadata and guest-management daemon on port 80, prepares each workspace - cloning Git repos, setting up the skills path, and, if the binding carries a `goal`, handing that goal to an agent that finishes environment setup within a default ten-minute budget - and then starts your command as a supervised child process. The [sandbox guide](https://github.com/google/ax/blob/main/docs/sandbox.md) documents the metadata endpoints your code can introspect without any SDK, including full task and workspace YAML served at `AX_METADATA_URL`. Two lifecycle features round it out. Suspending a task checkpoints actor state and pauses it; resuming picks up exactly where it left off. And when `spec.debug` is true, the same port serves Agent Substrate's guest services - process execution and filesystem access over gRPC - which is what powers `ax ssh` into a running sandbox. Networking goes through Agent Substrate's atenet router rather than Kubernetes Services: the router reads a single `ate-target-actor` header, resolves the actor to its worker, resumes it if it was suspended, and proxies the request through, as the [networking guide](https://github.com/google/ax/blob/main/docs/networking.md) explains.

## Advantages

- **Declarative and familiar.** Tasks, workspaces, gateways, and models are multi-document YAML applied with one command; annotated examples for every kind live in the [manifests guide](https://github.com/google/ax/blob/main/docs/manifests.md).
- **Isolation by default.** Untrusted agent code runs in a sandbox with CPU and memory limits, and outbound traffic is fenced to an explicit host allowlist rather than trusted implicitly.
- **Built for scale, not demos.** Redis Streams decouple the API server from horizontally scaled controllers, which is why the project can talk about billions of tasks per cluster without hand-waving.
- **Warm starts.** Workspaces make environment setup declarative and reusable across tasks, including agent-driven bootstrap from a plain-language goal.
- **Operational levers that agents actually need.** Checkpoint-and-suspend, resume-in-place, live `watch` of phase transitions, and shell access for debugging are first-class verbs in the CLI, not afterthoughts.
- **Credentials as resources.** Model configurations live in the control plane with secrets referenced from Kubernetes, so key rotation is centralized.

## Benefits

The practical payoff shows up in how you build agent systems. Because a task is deliberately small and cheap, you stop writing one giant agent and start composing trees of them - a root task that spawns children as the problem breaks down, each with the same sandbox, lifecycle, and tooling. Because workspaces are shared, the second task that needs your monorepo and your MCP servers pays almost nothing to start. Because egress is allowlisted per gateway, an agent that hallucinates a call to the wrong host fails safely instead of exfiltrating anything. Because the platform's own model configuration is a resource, swapping the planning model behind workspace bootstrap is a one-line change. And because everything is inspectable with kubectl reflexes - `ax get tasks`, `ax describe task`, `ax watch task` - your existing operational intuition transfers directly. For teams pairing this with [frameworks that build applications around agents](https://pyshine.com/Agent-Native-BuilderIO-Framework-Builds-Apps-Around-Agents/) or giving agents [shared long-term memory across CLIs](https://pyshine.com/ai-memory-Rust-Long-Term-Memory-For-Coding-Agents/), AX is the layer underneath that runs the resulting workload fleet, and it complements desktop-side environments like [PI-Desktop](https://pyshine.com/PI-Desktop-Open-Source-Desktop-Where-AI-Agents-Get-Their-Own-Workspace/) that focus on the single-developer experience.

## Usage

The [README](https://github.com/google/ax) walks the full path; the short version follows. Install the CLI with `go install github.com/google/ax/cmd/ax@latest`. You need a Kubernetes cluster, the `ko` build tool ([ko.build](https://ko.build)), a container registry your cluster can pull from, and a reachable Agent Substrate Control API; then `make deploy AX_IMAGE_REPO=<your-registry>` deploys Redis first and the control plane images into the `ax-system` namespace. With that up, declare a task in a few lines of YAML:

```yaml
apiVersion: ax.io/v1alpha1
kind: Workspace
metadata:
  name: golang
spec:
  git:
    - repo: https://github.com/golang/go.git
      branch: "my-fix"
---
apiVersion: ax.io/v1alpha1
kind: Task
metadata:
  name: test
spec:
  workspaces:
    - name: golang
      goal: "Ensure that Go tool chain is available and is built from source"
  debug: true   # lets you `ax ssh` into the sandbox
```

Then apply and inspect it with kubectl-shaped verbs:

```bash
ax apply -f task.yaml
ax watch task test
ax ssh test -- ls -al /workspace
ax suspend task test
ax resume task test
```

The repository ships a complete [example manifest](https://github.com/google/ax/blob/main/examples/task.yaml) with a Task, Workspace, Gateway, and Model in one file, and a `demo.sh` script that walks the whole lifecycle end to end - apply, wait for readiness, run commands over `ax ssh`, and suspend.

## Conclusion

AX takes the most tired idea in infrastructure - declare desired state, let controllers reconcile it - and aims it at the newest workload nobody has a good operational story for: autonomous agents. The [source tree](https://github.com/google/ax) backs the ambition with concrete engineering, from Redis Streams as the task queue to a runner that treats sandbox bootstrapping as a solved, declarative problem. The project is early and says so plainly, warning that core concepts, protocols, and specifications may still change before a stable release; the [homepage](https://agentexecutor.io) and design document are the places to track that. But the direction is clear, and the moment your agent experiments stop being one terminal and start being a fleet, this is the shape of the answer. Apply a task and watch what happens.
