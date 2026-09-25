---
layout: post
title: "Terraform: How HashiCorp's Infrastructure-as-Code Engine Plans and Applies Real Infrastructure"
description: "Terraform is a Go-based infrastructure-as-code tool that turns HCL configuration into an execution plan, walks a dependency graph, and drives provider plugins over gRPC. A source-level tour of hashicorp/terraform: the CLI layer, the config loader, the core graph engine, the state manager, the backend system, and the plugin process model."
date: 2026-09-25
header-img: "img/post-bg.jpg"
permalink: /Terraform-Infrastructure-as-Code-How-It-Works/
featured-img: ai-coding-frameworks/ai-coding-frameworks
image: https://pyshine.com/assets/img/diagrams/terraform/hashicorp-terraform-architecture.svg
tags:
  - Infrastructure as Code
  - DevOps
  - Go
  - Open Source
  - HashiCorp
categories: [DevOps, Open Source]
keywords: "Terraform, hashicorp terraform, infrastructure as code, terraform architecture, terraform plan apply, terraform state file, terraform.tfstate, terraform providers plugins, HCL, terraform registry, go-plugin gRPC, terraform core graph engine, terraform backends, terraform lock file"
author: "PyShine"
---

Provisioning a cloud environment by clicking through a console does not scale, does not review well, and does not survive the person who wrote it. [Terraform](https://github.com/hashicorp/terraform), HashiCorp's infrastructure-as-code tool, replaces that workflow with a text file: you describe the infrastructure you want in a high-level configuration language, and a single Go binary figures out how to get there - what to create, what to change, what to destroy, and in what order.

The interesting part is what happens between "you wrote a config" and "your VPC exists". Terraform parses and evaluates the configuration, builds a graph of every resource and its dependencies, compares desired state against the actual state it recorded last time, produces an execution plan you can inspect before anything happens, and then applies that plan by talking to provider plugins - separate processes that translate generic resource operations into calls against AWS, Azure, GCP, Kubernetes, or hundreds of other platforms. This post walks through that machinery as it is actually implemented in the repository: the CLI layer, the configuration loader, the core graph engine, state management, backends, and the plugin protocol.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/terraform/hashicorp-terraform-overview-architecture.svg" alt="Architecture overview of the Terraform core repository" style="max-width:100%;height:auto;" />
</div>

*High-level overview: the operator drives the CLI, the core engine produces plans and reads and writes state through backends, and provider plugins are launched as subprocesses that talk to cloud APIs.*

## Why You Need This

The core problem Terraform solves is that infrastructure changes are easy to make and hard to predict. When you click "create" in a cloud console, nothing tells you what else that change will touch, nothing records what you did, and nothing stops the next person from doing something incompatible. Config files alone do not fix this either - a plain script that calls cloud APIs has no idea what already exists.

Terraform's answer has four parts, each visible in the README as a headline feature and in the source as a subsystem. First, infrastructure is described declaratively in HCL, so a datacenter blueprint can be versioned, reviewed, and shared like any other code. Second, every operation goes through a planning step: `terraform plan` computes an execution plan that shows exactly what will happen when you call apply, so surprises happen at review time, not at 2 a.m. Third, Terraform builds a graph of all your resources and parallelizes work on resources that do not depend on each other, which makes both planning and application efficient and exposes your real dependency structure. Fourth, change automation: complex changesets can be applied with minimal human interaction because the plan and the graph constrain what the tool is allowed to do.

It is worth understanding where the boundary of the tool sits. This repository contains only Terraform core - the command line interface and the main graph engine. Providers are implemented as plugins, published separately, and downloaded automatically from the Terraform Registry. That split is what lets one binary manage AWS one minute and a random SaaS product the next, and it shapes the whole architecture below.

## How It Works

The diagram below maps the real subsystems of the repository and how they connect.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/terraform/hashicorp-terraform-architecture.svg" alt="Detailed architecture of Terraform core, from CLI to plugin processes" style="max-width:100%;height:auto;" />
</div>

*Detailed architecture: CLI and commands, configuration and language, the graph engine, state and backends, the plugin system, and the external services they reach.*

### Understanding the Architecture

**Entry point and CLI layer.** The binary starts at `main.go`, which initializes OpenTelemetry tracing, sets up crash handling, and builds the command list through `commands.go`. The actual commands live in `internal/command`: `init.go`, `plan.go`, `apply.go`, plus `validate`, `fmt`, `console`, `graph`, `import`, `output`, `state`, `workspace`, `login`, and more. The working directory convention is defined here too - `DefaultDataDir` is `.terraform`, and the default state file name is `terraform.tfstate`. One command worth knowing about is `terraform graph`, which exports the internal resource graph in DOT form by default, with a mermaid output type also supported.

**Configuration and language.** Terraform does not execute your config; it parses and evaluates it. The `internal/configs` package loads `.tf` files and turns them into configuration structures, while `internal/lang` provides the expression evaluator - variables, functions, and references get resolved there. During `terraform init`, `internal/initwd` installs the modules your configuration references, using the registry client in `internal/registry` for modules published there. Provider versions are resolved by `internal/getproviders`, checked against the dependency lock file `.terraform.lock.hcl` (the constant `LockFilePath` in `internal/depsfile`), and downloaded into the local plugin cache managed by `internal/providercache`.

**The graph engine.** This is the heart of the tool, in `internal/terraform`. A Terraform context is built over the parsed configuration and the current state (`context.go`), and then walked. The walk operates on a directed acyclic graph: the DAG primitives live in `internal/dag`, and the builders that turn configuration into a plan graph or apply graph are part of the core package (for example `graph_builder_plan.go`). Because the graph encodes dependencies, independent resource nodes are processed in parallel while dependent ones wait. The plan walk (`context_plan.go`) produces a plan - a structured set of changes held by `internal/plans` - which the command layer can serialize to disk through `internal/plans/planfile` as a zip archive whose central member is a file literally named `tfplan`. The apply walk (`context_apply.go`) consumes that plan and drives the changes.

**State and backends.** Every resource Terraform has provisioned is recorded in state, modeled by `internal/states` as snapshots with a lineage and a serial number so stale writes can be detected. Persistence goes through the state manager in `internal/states/statemgr`, which also implements locking - two applies racing on the same state is exactly the kind of accident the lock prevents. Which storage is used is decided by the backend system in `internal/backend`: the local backend writes `terraform.tfstate` on disk, the remote backend delegates operations and state to a hosted service, and the legacy remote-state backends under `internal/backend/remote-state` cover stores like S3, Azure Storage, GCS, Consul, Kubernetes secrets, and Postgres. The backend is also the layer that loads configuration and creates the Terraform context for plan and apply, which is why the commands connect to the engine through it.

**The plugin system.** The engine never talks to a cloud API directly. It works against two contracts, `internal/providers` and `internal/provisioners`, and the implementations arrive as separate plugin processes. `internal/plugin` wraps `hashicorp/go-plugin`: Terraform launches the plugin binary as a subprocess, performs the handshake, and communicates over gRPC, with the wire format defined by the protobuf packages `internal/tfplugin5` and `internal/tfplugin6` (protocol versions 5 and 6). The same wrapper handles both providers and provisioners. This process boundary is deliberate: a crash or memory leak in a provider cannot take down the core, and providers can be written, versioned, and distributed independently of Terraform itself.

**A plan in flight.** Follow a single `terraform plan` and `terraform apply` through the boxes above. Init has already put provider binaries in the cache and modules in place. Plan loads configuration, loads the latest state snapshot through the backend and state manager, builds the plan graph, and during the walk calls each provider plugin - over gRPC, through the go-plugin client - to refresh the current attributes of its resources. The engine diffs desired against actual, records the changes in the plan model, and the command writes the plan file. Apply reads that plan file, builds the apply graph, and executes each change: create, update, or delete calls go to the provider plugin, the provider translates them into cloud API calls, and the resulting new state is serialized through the state manager. Nothing in that chain requires the core to know anything about any specific cloud.

## Advantages

- **Preview before you commit.** The planning step is structural, not optional. Because the plan is a first-class artifact - a serialized zip archive you can save and pass to apply - the tool guarantees that what gets applied is what was reviewed.
- **Dependency-aware parallelism.** The resource graph means Terraform knows that a subnet must exist before an instance attaches to it, while two unrelated buckets can be created simultaneously. The same graph powers the visualization in `terraform graph`.
- **A plugin architecture that scales horizontally.** Providers are separate processes speaking a versioned gRPC protocol (5 and 6 today), so supporting a new platform never requires changing Terraform core, and a misbehaving provider is isolated from the engine.
- **Cloud-agnostic state storage.** The backend interface lets the same workflow store state on local disk, in S3, GCS, Azure, Postgres, Kubernetes, or a hosted service, with locking to prevent concurrent corruption.
- **Reproducible provider versions.** The `.terraform.lock.hcl` dependency lock file pins the exact provider versions and checksums chosen during init, so every team member and CI run installs the same binaries from the provider cache.
- **A mature, single-binary core.** Terraform core is one static Go binary with no runtime dependencies, and the codebase separates CLI, language, engine, state, and plugin concerns cleanly enough that each can be reasoned about - and tested - in isolation.

## Benefits

- **Infrastructure you can review.** Because infrastructure lives in config files, pull requests become the review mechanism for production changes, and git history becomes the audit log.
- **Fewer human errors under pressure.** The plan-apply split plus the constraint graph mean the tool refuses to invent an order of operations. Complex changesets become routine rather than white-knuckle events.
- **Team-safe by default.** State locking and serial numbers on state snapshots mean two people running apply simultaneously get an error, not a corrupted record of reality.
- **Reuse through modules.** The module installer and the registry let you package and consume infrastructure patterns the same way you reuse libraries in application code.
- **Portable across providers.** You learn one workflow - init, plan, apply - and apply it to any platform that has a provider, instead of learning each cloud's bespoke CLI and console.
- **Open and inspectable.** The core is developed in the open under the Business Source License 1.1, and the architecture described here is readable directly in the repository - including the ability to export its own internal graph for inspection.

## Usage

Terraform is distributed as a single binary; install it from the [official installation guide](https://developer.hashicorp.com/terraform/install) or your system package manager. A minimal workflow looks like this. First, describe some infrastructure in a file named `main.tf`:

```hcl
resource "aws_s3_bucket" "logs" {
  bucket = "my-app-access-logs"
}
```

Then initialize the working directory. This installs the referenced modules, resolves the provider versions your config needs, records them in `.terraform.lock.hcl`, and downloads the provider binaries into `.terraform/providers`:

```bash
terraform init
```

Preview the change. Terraform builds the plan graph, refreshes state through the provider, and shows you exactly what it intends to do. Writing the plan to a file makes the apply step consume exactly this reviewed artifact:

```bash
terraform plan -out=tfplan
terraform apply tfplan
```

After the apply, `terraform.tfstate` holds the record of what was created, and `terraform output`, `terraform state list`, and `terraform show` let you inspect it. To see Terraform's own view of your dependency structure, export the graph:

```bash
terraform graph
```

For teams, move the backend from local disk to a remote one - an S3 bucket, Postgres, or a hosted service - so state and locking are shared. And if you want to test infrastructure code the way you test application code, the repository is actively developing `terraform test`, including experimental features like running tests against a shared backend and cleaning up state left by failed test runs.

## Conclusion

Terraform endures because its design matches the problem: infrastructure changes are risky, so the tool makes prediction a first-class step, encodes dependencies in a graph, and confines vendor specifics to isolated plugin processes. Reading the repository makes the whole loop concrete - a Go binary that parses config, walks a DAG, records reality in a versioned state file, and delegates the actual API calls to gRPC subprocesses it can restart, upgrade, and sandbox independently. If you work with cloud infrastructure, clone the repository, run a small plan, and watch the pieces move in the order this post describes. The source is the best documentation of why the tool behaves the way it does.

**Links:**

- Repository: [https://github.com/hashicorp/terraform](https://github.com/hashicorp/terraform)
- Website: [https://developer.hashicorp.com/terraform](https://developer.hashicorp.com/terraform)
- Documentation: [https://developer.hashicorp.com/terraform/docs](https://developer.hashicorp.com/terraform/docs)
- Tutorials: [https://developer.hashicorp.com/terraform/tutorials](https://developer.hashicorp.com/terraform/tutorials)
- Plugin development: [https://developer.hashicorp.com/terraform/plugin](https://developer.hashicorp.com/terraform/plugin)
- Terraform Registry: [https://registry.terraform.io](https://registry.terraform.io)
