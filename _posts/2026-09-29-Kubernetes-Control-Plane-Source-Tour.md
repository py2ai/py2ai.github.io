---
layout: post
title: "Kubernetes: A Source Tour of the Control Plane - Inside kubernetes/kubernetes"
description: "A guided source-tour of the kubernetes/kubernetes repository, walking the real Go code behind kube-apiserver, kube-scheduler, kube-controller-manager, kubelet, and the informer machinery in client-go. Learn how the Kubernetes control plane actually works by reading the directories, entry points, and data structures that power it."
date: 2026-09-29
header-img: "img/post-bg.jpg"
permalink: /Kubernetes-Control-Plane-Source-Tour/
featured-img: "ai-coding-frameworks/ai-coding-frameworks"
image: https://pyshine.com/assets/img/diagrams/kubernetes/kubernetes-kubernetes-architecture.svg
tags:
  - Kubernetes
  - Go
  - Cloud Native
  - Architecture
categories: [AI, Open Source]
keywords: "kubernetes source code, kube-apiserver, kube-scheduler, kubelet, kube-controller-manager, client-go informers, control plane architecture, container orchestration, Go monorepo, kubernetes internals, source tour, reconciliation loop, delta fifo, workqueue"
author: "PyShine"
---

Most engineers meet Kubernetes through YAML and CLI flags, and that interface is polished enough that the machinery behind it can stay invisible for years. The [kubernetes/kubernetes](https://github.com/kubernetes/kubernetes) repository is where that machinery actually lives: a Go monorepo that contains every core component, from the API server that stores your manifests to the agent running on every node. Reading it is the fastest way to stop treating the platform as magic.

Kubernetes, also known as K8s, describes itself in its README as "an open source system for managing containerized applications across multiple hosts," providing basic mechanisms for deployment, maintenance, and scaling. The project grew out of Google's internal experience running production workloads with Borg and is now hosted by the Cloud Native Computing Foundation. What the README does not show is the shape of the codebase: the repository is organized around binaries under `cmd/` and the deep `pkg/` trees behind them, with reusable API machinery published as separate Go modules under `staging/src/k8s.io/`.

The source is worth a tour because Kubernetes is one of the clearest large-scale demonstrations of a specific architectural idea: a declarative control plane built from independent, cooperating loops. Once you can trace a Pod's lifecycle through the actual files — the API server's registry, the scheduler's queue, the kubelet's sync loop — you understand not just how Kubernetes works, but why distributed systems in general get built this way. Every claim in this post points at a real path in the tree, so you can follow along in the repository.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/kubernetes/kubernetes-kubernetes-overview-architecture.svg" alt="Architecture overview of the kubernetes/kubernetes repository" style="max-width:100%;height:auto;" />
</div>

*Overview of the kubernetes/kubernetes source layout: each control-plane binary under `cmd/` fronts its own `pkg/` implementation tree, all of them talk to the cluster state through the generic REST storage and etcd3 store, and every component leans on the shared informer and workqueue machinery published from `staging/src/k8s.io/client-go/`.*

Reading the overview from left to right: clients such as `cmd/kubectl` send REST requests to `cmd/kube-apiserver`, whose assembly code in `pkg/controlplane/instance.go` wires up the REST storage that backs every resource and persists it through the etcd3 store. To the right, the long-running controllers — `cmd/kube-scheduler`, `cmd/kube-controller-manager`, and the node-side `cmd/kubelet` and `cmd/kube-proxy` — never write to etcd themselves; they watch the API server through the shared informer layer and push changes back as ordinary API calls. The informer and workqueue files at the bottom are the connective tissue shared by all of them.

## Why You Need This

The first problem Kubernetes solves is the one every multi-host deployment hits: you want to state what should run, and have reality converge on that statement without a human babysitting servers. The repository's answer is not a single orchestrator process but a set of independent binaries, each owning one concern. The API server owns truth, the scheduler owns placement, the controller manager owns object-level correction, and the kubelet owns the node. Understanding where those boundaries live in code — `cmd/` entry points over `pkg/` implementations — makes the whole platform predictable to reason about.

The second problem is change itself. Kubernetes supports dozens of API groups and a rolling stream of new resources, so adding a resource cannot require rewriting the server. The code solves this with a generic REST storage layer: `pkg/registry/core/rest/storage_core.go` maps core resources onto storage, and a shared `Store` implementation in `staging/src/k8s.io/apiserver/pkg/registry/generic/registry/store.go` provides create, update, watch, and delete semantics for every resource that registers itself. That is why Kubernetes can absorb enormous API surface growth without the server becoming unmaintainable.

The third problem is reliability under partial failure, and the repository is unusually honest about it. Components do not call each other directly; they observe the API server through client-go's informer machinery, which caches state locally and delivers change events. If the API server blips, a controller's caches repopulate and its work queues drain again — no request was in flight to fail. Reading `staging/src/k8s.io/client-go/tools/cache/` teaches a resilience pattern you can reuse in far smaller systems.

Finally, the source matters for anyone building on the platform rather than merely operating it. Operators and CRDs are implemented in the same repository's machinery — the apiextensions-apiserver module under `staging/src/k8s.io/` — and the scheduler itself is explicitly designed for extension, with the reference implementation noting in `cmd/kube-scheduler/app/server.go` that multiple schedulers may run in a cluster. Touring the source turns "it supports extension" from marketing into a set of concrete files you can pattern-match against.

## How It Works

Every control-plane component is a separate binary whose `main()` exists only to assemble options and run the real implementation from `pkg/`.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/kubernetes/kubernetes-kubernetes-architecture.svg" alt="Detailed architecture of the kubernetes/kubernetes repository" style="max-width:100%;height:auto;" />
</div>

*Detailed source map of the control plane: the API server chain builds down to the etcd3 store, the scheduler and controller-manager consume API objects through the client-go informer pipeline on the left, and the kubelet's runtime stack on the right turns watched Pod objects into containers.*

### Understanding the Architecture

**The API server is a chain of servers over one storage layer.** `cmd/kube-apiserver/apiserver.go` is a thin `main()` that calls `app.NewAPIServerCommand()`, and the real assembly happens in `cmd/kube-apiserver/app/server.go`, where `CreateServerChain` builds the apiextensions server (which serves CRDs), then the core kube-apiserver, and finally the aggregator in front of both. Each server in the chain is an instance of the generic API server in `staging/src/k8s.io/apiserver/pkg/server/genericapiserver.go`, and `pkg/controlplane/instance.go` assembles the core instance. The storage it serves is mounted in bulk: `pkg/registry/core/rest/storage_core.go` registers REST storage for every core resource, and the group-specific trees under `pkg/registry/` — apps, batch, networking, rbac, storage, and more — do the same for their APIs.

**Every resource ends up in etcd through one generic Store.** The individual registry trees define resource-specific logic, but the heavy lifting is shared: `staging/src/k8s.io/apiserver/pkg/registry/generic/registry/store.go` implements the standard verbs, and its persistence goes through the etcd3 client in `staging/src/k8s.io/apiserver/pkg/storage/etcd3/store.go`. This is why watch semantics are uniform across the API: whether you watch a ConfigMap or a CustomResource, you are watching the same generic storage interface.

**The informer pipeline is the platform's nervous system.** In `staging/src/k8s.io/client-go/tools/cache/`, `reflector.go` runs the ListAndWatch loop against the API server and streams changes into `delta_fifo.go`, which maintains a local store plus a queue of deltas; `shared_informer.go` fans those events out to every consumer of a resource type. Event handlers do not process work inline — they enqueue keys into the work queues of `staging/src/k8s.io/client-go/util/workqueue/queue.go`, which provide deduplication and rate-limited retries. The scheduler, the controller manager, and large parts of the kubelet are all built on this same pipeline.

**The scheduler is a framework executing one cycle per pod.** `cmd/kube-scheduler/scheduler.go` starts a component whose core loop lives in `pkg/scheduler/scheduler.go`: it pulls a pod from the scheduling queue in `pkg/scheduler/backend/queue/scheduling_queue.go`, snapshots cluster state via the cache in `pkg/scheduler/backend/cache/cache.go`, and enters the per-pod cycle in `pkg/scheduler/schedule_one.go`. That cycle runs the plugin system assembled by `pkg/scheduler/framework/runtime/framework.go` — Filter and Score plugins such as `noderesources/fit.go` or `podtopologyspread` under `pkg/scheduler/framework/plugins/` — and the winning node is bound via the `defaultbinder` plugin, which simply POSTs a Binding object back to the API server. Ingress into the queue comes from event handlers wired in `pkg/scheduler/eventhandlers.go`, fed by the same informers as everything else.

**The controller manager is a fleet of reconciliation loops.** `cmd/kube-controller-manager/app/controllermanager.go` launches dozens of independent controllers from the trees under `pkg/controller/` — deployment, replicaset, job, statefulset, endpointslice, resourcequota, and more. Each follows the same shape as `pkg/controller/deployment/deployment_controller.go`: a `Run` method starts workers, a `worker` drains the queue, and `syncDeployment` fetches the desired object and corrects the world until reality matches. The node lifecycle controller in `pkg/controller/nodelifecycle/` monitors node health, and the garbage collector in `pkg/controller/garbagecollector/` builds a dependency graph (`graph_builder.go`) to delete orphaned objects safely.

**The kubelet turns Pod objects into running containers.** The node agent starts from `cmd/kubelet/kubelet.go` and `cmd/kubelet/app/server.go`, which wire pod sources — the API server, local files, and HTTP — through the mux in `pkg/kubelet/config/config.go` into `pkg/kubelet/kubelet.go`. There, `syncLoop` consumes PodUpdate events, the pod manager in `pkg/kubelet/pod/pod_manager.go` tracks desired versus actual pods, and per-pod workers dispatch each change through `SyncPod`. The sync talks to containers via the CRI-backed runtime manager in `pkg/kubelet/kuberuntime/kuberuntime_manager.go`, while the PLEG in `pkg/kubelet/pleg/pleg.go` relists container state to generate lifecycle events, the prober manager in `pkg/kubelet/prober/prober_manager.go` executes health checks, and the eviction manager in `pkg/kubelet/eviction/eviction_manager.go` reclaims resources under pressure.

Follow one request end to end and the design clicks: `kubectl` POSTs a Pod to the API server chain, which validates it and writes it to etcd through the generic Store; the scheduler's informer delivers the new pod into its scheduling queue, a cycle selects a node, and the binder writes a Binding back; the kubelet on that node observes the assignment through its own informers and drives the container runtime until the pod is live, reporting status upstream all the while. No component called another directly — every hop is an observation of, or a write to, the API server.

## Advantages

- **Declarative convergence.** The controller-manager trees under `pkg/controller/` implement continuous reconciliation, so desired state declared through the API is actively repaired rather than merely recorded.
- **Uniform API machinery.** One generic registry `Store` and one etcd3 backend serve every resource, so custom and built-in APIs behave identically for watch, list, and lifecycle semantics.
- **Decoupled by construction.** Components communicate only through the API server and the informer caches in `staging/src/k8s.io/client-go/tools/cache/`, which keeps failures local and restarts cheap.
- **Extension as a first-class path.** The scheduler's plugin framework under `pkg/scheduler/framework/` and the aggregated, CRD-serving servers assembled in `cmd/kube-apiserver/app/server.go` make extension a documented code path, not an afterthought.
- **Published reusable modules.** The `staging/src/k8s.io/` tree publishes client-go, apiserver, apimachinery, kube-scheduler, kubelet, and more, so the platform's building blocks can be imported into your own Go programs.
- **Battle-tested patterns.** DeltaFIFO, work queues, rate-limited retries, and leader election in the same repository are reference implementations of distributed-systems techniques you can study directly.

## Benefits

- **Read the platform, not the blog posts.** Tracing a Pod through `pkg/scheduler/schedule_one.go` and `pkg/kubelet/kubelet.go` gives you ground truth about scheduling and node behavior without intermediaries.
- **Debug with a mental model.** Knowing that all state flows through the API server's storage layer and informer caches tells you exactly where to look when events, caches, or watches behave unexpectedly.
- **Contribute with confidence.** Because each binary's entry point under `cmd/` cleanly delegates to `pkg/` implementations, new contributors can locate the code that matters for a fix in minutes instead of days.
- **Build your own controllers.** The informer and workqueue patterns are designed to be copied; understanding them from the source makes home-grown operators reliable from day one.
- **Design better distributed systems.** The repository demonstrates, at production scale, how list-and-watch caching, level-triggered reconciliation, and a single source of truth cooperate — lessons that transfer well beyond Kubernetes.
- **Evaluate changes realistically.** Release notes make more sense once you can connect a feature to the actual files and interfaces it touches.

## Usage

The README's from-source path expects a working Go environment:

```
git clone https://github.com/kubernetes/kubernetes
cd kubernetes
make
```

If you prefer not to set up a full Go toolchain, the README offers a Docker-based build:

```
git clone https://github.com/kubernetes/kubernetes
cd kubernetes
make quick-release
```

For day-to-day use, the README points operators at the official documentation on [kubernetes.io](https://kubernetes.io), and developers at the community contributor guides. For library consumers, the README highlights the list of published components under `staging/` and warns that importing the main `k8s.io/kubernetes` module directly is not supported — use the published `k8s.io/*` modules instead.

## Conclusion

Reading kubernetes/kubernetes rewards the effort precisely because the code matches the architecture diagrams. The control plane is four binaries and a storage backend; the glue between them is the informer and workqueue machinery; and every subsystem, from scheduling to node lifecycle, is the same level-triggered reconciliation loop wearing a different hat. Once you have walked `cmd/kube-apiserver`, `pkg/scheduler/schedule_one.go`, and `pkg/kubelet/kubelet.go`, Kubernetes stops being a black box and becomes a well-organized codebase you can navigate on demand.

Links:

- Repository: [github.com/kubernetes/kubernetes](https://github.com/kubernetes/kubernetes)
- Documentation: [kubernetes.io/docs](https://kubernetes.io/docs/home/)
- Contributor and developer documentation: [git.k8s.io/community/contributors/devel](https://git.k8s.io/community/contributors/devel#readme)
