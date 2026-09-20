---
layout: post
title: "Cilium: eBPF Networking, Security, and Observability for Kubernetes"
description: "Cilium is a CNCF-graduated networking, security, and observability platform built on eBPF - this deep dive covers its per-node agent, eBPF datapath, identity-based policies, ClusterMesh, and Hubble flows."
date: 2026-09-20
header-img: "img/post-bg.jpg"
permalink: /Cilium-eBPF-Networking-Security-and-Observability-for-Kubernetes/
featured-img: ai-coding-frameworks/ai-coding-frameworks
tags:
  - Open Source
  - Kubernetes
  - eBPF
  - Networking
  - Security
  - Cloud Native
  - Observability
author: "PyShine"
---
# Cilium: eBPF Networking, Security, and Observability for Kubernetes

Every pod in a Kubernetes cluster needs three things from the network: to reach its neighbors fast, to be protected by policies that survive scale, and to be observable when something breaks. [Cilium](https://github.com/cilium/cilium), a graduated [CNCF](https://www.cncf.io) project with over 25,000 stars, delivers all three with an unusual foundation: an [eBPF](https://ebpf.io)-based dataplane that runs security, networking, and visibility logic inside the Linux kernel itself. Written primarily in Go with C programs for the kernel side and released under Apache-2.0, it provides a flat Layer 3 network that can span clusters in native routing or overlay mode, enforces L3-L7 policies using an identity-based security model decoupled from IP addresses, and exposes deep flow visibility through its integrated Hubble observability layer. If our [Kubernetes tutorial](https://pyshine.com/Learn-Kubernetes-in-One-Post-Complete-Tutorial-Pods-Deployments-Services-Production-Quick-Start/) covered how pods talk to services, this post covers the machinery that makes that conversation fast and safe.

![High-level architecture overview of the Cilium repository](/assets/img/diagrams/cilium/cilium-overview-architecture.svg)

## Why You Need This

The default Kubernetes networking stack ages quickly at scale. kube-proxy traditionally programs service load balancing through iptables, which walks rules linearly per packet; with thousands of services the constant rule updates and per-packet cost become measurable bottlenecks. Container firewalls have a similar scaling problem: filtering on source IPs and ports means touching firewall rules on every server whenever a container starts anywhere in the cluster.

Cilium replaces both mechanisms with eBPF hash tables and identities. East-west service connections are rewritten at the socket level during connect, avoiding per-packet NAT entirely and fully replacing kube-proxy. North-south traffic gets XDP acceleration, Direct Server Return, and Maglev consistent hashing. Security identities attach to groups of containers that share the same policy, so a rule change updates an identity rather than every host firewall, and policies can filter on HTTP methods, URL paths, or gRPC calls rather than just ports. On top of that sits ClusterMesh for multi-cluster connectivity and Hubble for seeing what the network is actually doing - the two features most teams discover they cannot live without after the first production incident.

Visibility is the quieter gap. Traditional networking tools see packets but not context - which pod, which label, which HTTP call. When a connection fails inside a mesh of hundreds of microservices, operators need the answer at the workload identity level, not as a tcpdump firehose they must decode by hand. Cilium was designed with this problem in mind, which is why flow observability is built in from the ground up rather than bolted on after the fact.

## How It Works

The diagram below maps the repository's main components and how they connect.

![Detailed architecture of the Cilium repository](/assets/img/diagrams/cilium/cilium-architecture.svg)

**The agent core.** A deployment through the bundled [Helm chart](https://github.com/cilium/cilium/tree/main/install/kubernetes/cilium) runs one Cilium agent per node as a DaemonSet and one cluster-wide operator. The agent, whose core lives in `daemon/cmd/daemon.go`, watches pods and CRDs through the Kubernetes API, tracks every local endpoint in its endpoint manager, resolves network policy through the policy repository, and assigns each endpoint group a numeric security identity from the identity package. Configuration knobs such as datapath mode and feature toggles flow through the option package that every component shares.

**The eBPF dataplane.** For every endpoint, the agent generates and compiles a per-endpoint datapath program from `bpf/bpf_lxc.c`, linking against the shared helpers in `bpf/lib`, and attaches it to the kernel's network hooks. State - endpoint tables, service entries, policy verdicts - lives in eBPF maps that both the agent and the compiled programs read and write, so the fast path never leaves the kernel. The load balancer package reconciles service backends into these maps, giving pods distributed load balancing with almost unlimited table scale. Where L7 awareness is needed, the agent offloads HTTP, Kafka-style protocol parsing, and gRPC inspection to an integrated Envoy proxy through the Envoy integration package.

**Observability and cluster services.** Each agent embeds a Hubble server that observes flows with identity and label metadata; [Hubble Relay](https://github.com/cilium/hubble) aggregates those per-node streams into a cluster-wide view, and the Hubble CLI queries it for service maps, dropped-flow reasons, and DNS-aware filtering. Metrics export to Prometheus for Grafana dashboards. The operator handles cluster-wide chores such as IPAM and deploying the ClusterMesh apiserver, through which the ClusterMesh package replicates identities and service state across clusters, letting workloads fail over transparently to backends in another cluster. A node health checker probes connectivity between hosts, the CNI plugin wires new pods to the agent, and a bug tool collects diagnostics from a running deployment.

## Advantages

The architecture yields concrete advantages over conventional CNI plugins. Performance is kernel-native: packet processing, load balancing, and policy enforcement happen in eBPF rather than in user-space proxies or linear iptables walks. The security model is identity-based rather than address-based, so policies follow workloads across rescheduling, scaling, and even cluster boundaries. Deployment is flexible: VXLAN or Geneve overlay networks work on almost any infrastructure, native routing integrates with cloud routers, and BGP automates route advertisement across layer 3 boundaries. The service mesh story avoids sidecar sprawl, since L7 handling rides on node-level Envoy integration and eBPF redirects instead of a proxy injected into every pod. And because the Cilium agent is read from a single Go codebase with a debuggable API surface, the CLI can inspect endpoint state, policy verdicts, and BPF maps directly from a running node.

Supply chain hygiene is part of the package too: every release image ships with a Software Bill of Materials in SPDX format, and images are published for both AMD64 and AArch64 architectures. A standalone DNS proxy extends the same FQDN-aware policy enforcement to setups that terminate DNS outside the normal CNI path, keeping egress rules consistent across deployment styles.

## Benefits

In production terms, these advantages translate into fewer moving parts and faster answers. Replacing kube-proxy removes one whole subsystem and its configuration drift. Drop reasons turn a mysterious connection refused into an actionable verdict - policy violation, failed DNS lookup, or port mismatch - which shrinks incident triage time. DNS-based egress policies let teams allow traffic to trusted wildcard domains without pinning IP addresses that third-party services change constantly. Mutual authentication with WireGuard or IPsec encryption can be switched on without redesigning applications. Teams running hybrid or multi-cloud environments get one policy language across all of them, which is precisely why organizations of every size, backed by the team at [Isovalent](https://isovalent.com), contribute to and depend on the project.

## Usage

Getting started is a Helm install away. Ensure your cluster runs a recent Linux kernel, then add the Cilium Helm repository and install the chart from the repository's `install/kubernetes` directory, or use the standalone Cilium CLI which can install and validate the deployment for you. Images for AMD64 and AArch64 ship on [Quay](https://quay.io/repository/cilium/cilium), with the current stable minor release being v1.20, patched most recently on 2026-09-15. Verify the dataplane with `cilium status`, explore live traffic with `hubble observe` and the Hubble UI service map, and enforce your first identity-aware policy by labeling workloads and selecting them in a CiliumNetworkPolicy. Full documentation, from getting-started guides to datapath internals, lives at [docs.cilium.io](https://docs.cilium.io), and our [Docker tutorial](https://pyshine.com/Learn-Docker-in-One-Post-Complete-Tutorial-Dockerfile-Volumes-Compose-Quick-Start/) covers the container fundamentals underneath.

## Conclusion

Cilium is what happens when a hard kernel technology is wrapped in a genuinely operable product: eBPF gives the dataplane its speed and precision, while the agent, operator, Hubble, and ClusterMesh give platform teams something they can actually run. Its core ideas - identity-based security, kernel-level load balancing, and flow observability as a first-class feature - have quietly become the reference design for cloud-native networking. If your cluster is outgrowing its default CNI, the repository is mature, graduated, and ready for production.
