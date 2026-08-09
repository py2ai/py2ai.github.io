---
layout: post
title: "DevOps Interview Guide - Real Questions from 2025 and 2026"
description: "A community-driven collection of 151 real DevOps, SRE, and Cloud engineering interview write-ups from 85 companies. See the exact questions candidates were asked in 2025 and 2026, organized by company for targeted interview prep."
date: 2026-08-08
permalink: /DevOps-Interview-Guide-Real-Questions-2026/
featured-img: ai-coding-frameworks/ai-coding-frameworks
image: /assets/img/diagrams/devops-interview-guide/devops-interview-topics.svg
tags: [DevOps, SRE, Kubernetes, Docker, Terraform, AWS, Azure, GCP, CI/CD, Ansible, Linux, Interview]
categories: [DevOps, SRE, Cloud, Interview, Open Source]
keywords: "DevOps interview questions, SRE interview questions, real DevOps interview 2026, Kubernetes interview questions, Terraform interview, AWS interview, Azure interview, GCP interview, CI/CD interview, Ansible interview, Linux interview, SLI SLO SLA, observability interview, DevOps interview guide, site reliability engineering interview"
author: "PyShine"
---

## Introduction

If you have ever prepared for a DevOps, SRE, or Cloud engineering interview, you already know the problem. Most "interview question" lists online are paraphrased, sanitized, and recycled until they barely resemble what an interviewer actually asks. They read like a textbook table of contents rather than the conversation that happens across the table. The result is that candidates study the wrong things and get blindsided by questions that are routine in real rooms but absent from every "top 50 questions" blog post.

The DevOps Interview Guide takes the opposite approach. It is a community-driven repository of real interview write-ups collected from people who actually sat through DevOps, SRE, and Cloud engineering interviews in 2025 and 2026. There is no paraphrasing and no filler. Each entry is the exact set of questions a candidate was asked, preserved as a single file so you can read the interview the way it actually unfolded. With 151 write-ups spanning 85 named companies plus an `Others/` folder for unnamed interviews, it is one of the most honest interview-prep resources available right now.

This post walks through what the guide contains, how it is organized, the core topics it covers, the companies represented, the SRE fundamentals it emphasizes, and how to use it effectively for your own preparation. The repository is open source and available on GitHub at [litu54/DevOps-Interview-Guide](https://github.com/litu54/DevOps-Interview-Guide).

## What is the DevOps Interview Guide

The DevOps Interview Guide is a GitHub repository that collects real interview experiences from DevOps, SRE, and Cloud engineering candidates. Instead of generic question banks, it stores one file per interview, written from the perspective of the person who went through it. This means you see the actual flow of a round, the follow-up questions interviewers ask when you give an answer, and the depth that a specific company expects on a given topic.

At the time of writing, the repository holds 151 interview write-ups across 85 named companies, with an additional `Others/` folder for submissions where the company could not be named. The topics span the full breadth of modern infrastructure engineering: Kubernetes, Docker, Terraform, AWS, Azure, GCP, CI/CD tooling (Jenkins, GitHub Actions, Azure DevOps), Ansible, Linux, scripting, and SRE fundamentals including SLI/SLO/SLA, observability, and incident response.

The organization is intentionally simple. Each company gets its own folder, and each interview within that folder is a separate Markdown file. When a company was interviewed at multiple times, by different candidates, or for different rounds, those appear as multiple files rather than one merged list. This matters because the same company can ask very different questions depending on the team, the seniority, and the year. The file naming reflects the role when it was stated, defaulting to `DevOps_Engineer.md` when no role was specified.

A typical entry looks like this in the tree:

```text
<Company Name>/
  DevOps_Engineer.md        (default when no specific role was mentioned)
  DevOps_Engineer_2.md      (second interview for the same company)
  SRE_principal.md          (used when a role was explicitly called out)
```

Because every entry is plain Markdown, the repository is fully searchable. You can grep for a company name, a topic, or a specific tool, and land directly on the relevant interview without wading through unrelated content.

## How to Use It

![DevOps Interview Topics Knowledge Map](/assets/img/diagrams/devops-interview-guide/devops-interview-topics.svg)

The diagram above maps the knowledge areas you need to cover for a DevOps, SRE, or Cloud engineering interview and shows how they depend on each other.

Reading it top to bottom, the green Candidate node at the top represents you, the person entering the interview.

It branches into the three target roles the guide covers: DevOps Engineer, SRE, and Cloud Engineer.

On the side, the blue Interview Prep node captures the resource itself, the 151 write-ups across 85 companies that you browse by company to prepare.

A dashed edge feeds back into the Candidate node because preparation is what turns a raw candidate into a confident one.

The middle band of orange nodes is the core tooling layer that almost every interview touches: Kubernetes, Docker, Terraform, CI/CD, Linux, and Ansible.

The edges from each role down into this layer are labeled "must know" because interviewers treat these as baseline expectations rather than nice-to-haves.

A DevOps Engineer is expected to be fluent in Kubernetes and CI/CD pipelines, since those are the daily working tools of the role.

An SRE is grilled on Linux internals and container orchestration, because reliability work lives close to the operating system and the orchestrator.

A Cloud Engineer is tested on Terraform and containerization, because provisioning and packaging are the core of that role.

These are not optional topics; they are the vocabulary of the role, and the edges make that dependency explicit so you know where to spend your study time first.

Below the tooling layer, the diagram splits into two purple backend clusters that represent where the tools actually run and how they are operated.

On the bottom left sit the cloud platforms AWS, Azure, and GCP, because the tools above them have to run somewhere and interviewers will probe whether you can provision and operate infrastructure on a real cloud.

The edges labeled "provisions", "runs on", and "containers on" show that Terraform provisions resources across all three clouds.

Kubernetes clusters run on AWS and GCP, and Docker containers ship onto Azure, so your tool knowledge has to be grounded in at least one platform.

These are the deployment dependencies that turn abstract tool knowledge into production reality, and they are why cloud questions rarely stay theoretical for long.

On the bottom right sit the SRE Fundamentals: SLI/SLO/SLA, Observability, and Incident Response.

These are the reliability and operability concerns that separate a person who can build a system from a person who can keep it running.

The edges "feeds metrics", "configures", "exports logs", and "operated by" trace how the tooling layer feeds the reliability layer with the signals it needs.

Linux exports the metrics and logs that define SLIs, which are the quantitative input to every SLO you negotiate.

Ansible configures the observability stack that consumes those signals, wiring collectors, dashboards, and alerting into the systems you run.

Kubernetes clusters are the systems whose incidents you ultimately respond to, so the orchestrator is both the source of complexity and the thing you must debug under pressure.

The diagram is therefore a dependency graph, not just a topic list, and reading it as a graph tells you the order in which to study.

You cannot credibly discuss incident response without first understanding the Linux and Kubernetes systems that produce the signals you respond to.

And you cannot meaningfully talk about Terraform without a real cloud platform to point it at, which is why the cloud cluster sits directly beneath the tooling layer.

Treat the top-to-bottom flow as a study plan: pick a role, master the orange tools it depends on, ground them in a purple cloud platform, and then layer the purple SRE fundamentals on top so your reliability answers have real tooling to reference.

To use the repository itself, start by cloning it locally so you can search and browse offline:

```bash
git clone https://github.com/litu54/DevOps-Interview-Guide.git
cd DevOps-Interview-Guide
```

If you are preparing for a specific employer, search the repo for that company name. If the company has a folder, open each file in it and read the full interview, paying attention to the follow-up questions because those reveal what the interviewer was actually probing. If you are preparing broadly rather than for one company, skim a handful of folders across different company sizes, product companies, service companies such as TCS, Infosys, or Wipro, and fintech firms, because the range of questions across company types tells you more than any single list. You can also grep across the whole repository for a topic you want to strengthen:

```bash
# Find every interview that mentions Kubernetes
grep -rl "Kubernetes" .

# List all companies that have an SRE interview on file
grep -rl "SRE" . | sort
```

Because each file is one interview experience, you can also compare how the same company asks questions across different candidates and years, which is invaluable for spotting patterns in what a given employer cares about.

## Core Topics Covered

The guide spans the toolchain that modern infrastructure teams actually use. Below are the core topic areas and what you should expect to be asked about in each.

**Kubernetes** is the most frequently covered topic. Expect questions on pod lifecycle, deployments versus stateful sets, services and ingress, ConfigMaps and Secrets, RBAC, namespaces, resource requests and limits, probes (liveness and readiness), scheduling, taints and tolerations, and troubleshooting with `kubectl`. Interviewers also dig into real scenarios such as a CrashLoopBackOff, a pod stuck in Pending, or a service with no endpoints.

**Docker** questions cover image layers and caching, multi-stage builds, volume versus bind mounts, networking modes, the difference between CMD and ENTRYPOINT, image size optimization, and how to debug a container that exits immediately. You should be comfortable explaining what happens between `docker build` and `docker run`.

**Terraform** rounds focus on state management, the difference between `terraform plan`, `apply`, and `destroy`, workspaces, modules, remote state backends and locking, `terraform import`, handling drift, and writing reusable modules with variables and outputs. Expect scenario questions on refactoring a monolithic configuration into modules.

**CI/CD** questions span Jenkins, GitHub Actions, and Azure DevOps. You will be asked about pipeline stages, triggers, artifact management, secret handling, deployment strategies (blue-green, canary, rolling), and how to make pipelines fast and idempotent. Interviewers often ask you to design a pipeline for a sample application end to end.

**Ansible** questions cover playbooks, roles, inventory, variables and facts, modules versus tasks, idempotency, handlers, and templating with Jinja2. You may be asked to write a playbook that installs and configures a service across a group of hosts.

**Linux and scripting** questions cover process management, file permissions, systemd, networking utilities, disk and filesystem management, and shell or Python scripting. Interviewers frequently ask you to write a small script to parse logs, find the top IP addresses, or automate a repetitive task.

A simple shell snippet that reflects the kind of Linux troubleshooting question that appears in the write-ups:

```bash
# Find the top 5 source IPs by request count in an access log
awk '{print $1}' access.log | sort | uniq -c | sort -rn | head -5
```

## Top Companies

The repository organizes interviews by company, so you can prepare against the exact employers you are targeting. The 85 named companies span product companies, global service firms, banks, and specialist consultancies. A representative sample includes large technology and financial employers such as Amazon, JPMorgan, Morgan Stanley, IBM, Oracle, Cisco, SAP, Sony, Verizon, and Akamai; major Indian service and product companies such as TCS, Infosys, Wipro, HCL, LTIMindtree, Persistent Systems, Hexaware, Capgemini, Deloitte, EY, and ZS Associates; and cloud and platform specialists such as F5, Amadeus Labs, Blue Yonder, NatWest Group, Optum, Commonwealth Bank, and Moody's.

Some companies appear more than once because different candidates went through different rounds, sometimes years apart. TCS has multiple DevOps and SRE files, Deloitte and LTIMindtree each have several DevOps Engineer write-ups, and IBM has Cloud Engineer and multiple DevOps Engineer entries. Keeping these as separate files rather than merging them is deliberate, because the questions and interviewer style often differ between rounds. When you prepare for a specific company, read every file in its folder to get the full picture of what that employer tends to ask.

The `Others/` folder holds 21 additional write-ups where the company could not be named. These are useful for broad preparation because they cover the same topic range without being tied to a single employer, and they often include behavioral rounds that named submissions omit.

## SRE Fundamentals

Site Reliability Engineering questions appear both in dedicated SRE interviews and as a reliability thread inside DevOps rounds. The guide covers the three pillars that interviewers return to repeatedly.

**SLI, SLO, and SLA** questions test whether you can define service-level indicators from first principles. You should be able to explain how to choose a good SLI (a proxy for user happiness), how to set an SLO target and an error budget, and how an SLA differs as the external, contractual commitment. Expect scenario questions such as "Our checkout API has a 99.9% SLO; walk me through how you would define the SLI and what you would do when the error budget is exhausted."

**Observability** questions cover the three pillars of metrics, logs, and traces, plus the tooling used to collect and query them. You will be asked about Prometheus and PromQL, Grafana dashboards, the ELK or Loki stack for logs, OpenTelemetry and Jaeger for distributed tracing, cardinality management, and how to instrument an application so that you can debug a latency spike without guessing. The distinction between monitoring (is the system up?) and observability (why is the system slow?) comes up often.

**Incident response** questions assess your operational maturity. Interviewers want to see that you understand incident lifecycle, severity levels, the role of an incident commander, runbooks, blameless postmortems, and the difference between detecting, mitigating, and resolving an incident. You may be given a scenario such as "latency spiked on the checkout service at 2am; walk me through your response" and be expected to talk through detection, triage, mitigation, communication, and the follow-up postmortem.

A small example of the kind of PromQL query that shows up in SRE rounds:

```promql
# Error rate for the checkout service over the last 5 minutes
sum(rate(checkout_requests_total{status="5xx"}[5m]))
/
sum(rate(checkout_requests_total[5m]))
```

## Tips for Interview Prep

Based on the patterns across the 151 write-ups, a few preparation strategies consistently help.

First, prepare by company, not just by topic. Use the repository's per-company folders to study the employers you are actually targeting. Companies have distinctive emphasis areas, and reading their past interviews lets you anticipate the depth and style you will face. If a company has multiple files, read all of them to spot recurring themes.

Second, practice explaining out loud. Many write-ups describe candidates who knew the material but stumbled when asked to explain it verbally. For each topic, rehearse a clear two-minute explanation, a deeper follow-up, and a real-world example. The interview is a conversation, and fluency comes from having said the words before.

Third, prepare scenario answers. Interviewers in this space rarely stop at "what is Kubernetes." They follow up with "a pod is in CrashLoopBackOff, what do you do?" Build a mental library of debugging paths for the tools in the core layer of the diagram, and practice walking through them step by step from observation to root cause.

Fourth, do not neglect Linux and scripting. Even senior candidates get caught out by basic shell questions. Be comfortable with `awk`, `sed`, `grep`, `find`, process inspection with `ps` and `top`, networking with `ss` and `curl`, and writing small automation scripts in bash or Python.

Fifth, treat SRE fundamentals as a differentiator. If you can speak credibly about SLI design, error budgets, and incident response, you stand out from candidates who only know the tooling layer. The dependency edges in the diagram above are a reminder that reliability knowledge builds on top of the tools, so study them in order.

Sixth, contribute back. The guide stays useful because people add their interview experiences. If you sit through an interview, adding a write-up under the matching company folder (or under `Others/` if the company cannot be named) helps the next candidate and keeps the dataset current for 2026 and beyond.

## Conclusion

The DevOps Interview Guide is a refreshingly honest resource in a space crowded with recycled question lists. By preserving real interview write-ups, one file per interview, organized by company, it lets you prepare against the actual questions that candidates faced in 2025 and 2026 rather than a sanitized approximation. With 151 write-ups across 85 companies plus the `Others/` folder, the breadth is enough to prepare for almost any DevOps, SRE, or Cloud engineering role.

The knowledge map in this post makes the structure of that preparation explicit. Start from the role you are targeting, build fluency in the core tooling layer of Kubernetes, Docker, Terraform, CI/CD, Linux, and Ansible, ground that tooling in a real cloud platform, and then layer the SRE fundamentals of SLI/SLO/SLA, observability, and incident response on top. Browse the repository by company to match your preparation to the employers you care about, and practice explaining each topic out loud with scenarios.

If the guide helps your preparation, starring the repository on GitHub is the simplest way to support it and makes it easier for the next candidate to find. The repository is maintained on personal time and only stays current because contributors keep adding their interview experiences. Explore it, use it, and if you have an interview experience worth sharing, add it back so the cycle continues.

The full repository is available at [litu54/DevOps-Interview-Guide](https://github.com/litu54/DevOps-Interview-Guide).
