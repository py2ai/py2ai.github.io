---
layout: post
title: "SpiderFoot: 234 Modules Over One Event Queue - Inside smicallef/spiderfoot"
description: "A source-code tour of SpiderFoot, Steve Micallef's OSINT automation platform that orchestrates hundreds of modules over a single typed event queue, from CLI and web UI through its thread pool to SQLite storage and YAML correlations."
date: 2026-10-07
header-img: "img/post-bg.jpg"
permalink: /spiderfoot/
featured-img: "ai-coding-frameworks/ai-coding-frameworks"
image: https://pyshine.com/assets/img/diagrams/spiderfoot/smicallef-spiderfoot-overview-architecture.svg
tags: [OSINT, Automation, Python, Cybersecurity]
categories: [AI, Open Source]
keywords: "spiderfoot, smicallef, osint automation, python, open source, architecture"
author: "PyShine"
---

Give SpiderFoot a domain, an IP address, an email, or even a bitcoin address, and it does the rest: hundreds of small Python modules wake up, trade findings with each other, and gradually assemble a map of everything publicly connected to that target. Created by Steve Micallef and developed in the open since 2012, the project is one of the veterans of the OSINT world, and the version we walked through, 4.0.0 under the MIT license, is a mature Python 3.7+ codebase where that orchestration happens through a genuinely elegant mechanism: a typed event bus fed by a shared thread pool.

The engineering is what makes this codebase worth a long look. Every module, from DNS resolution to dark-web watching, implements the same plugin contract and declares two lists: what event types it consumes and what it produces. The scanner wires those declarations together at runtime, so a hostname found by one module automatically becomes the input of every module that knows how to handle hostnames, without any central brain deciding what runs next. Around that loop sits a fixed thread pool with per-module concurrency limits, a multiprocessing scanner process that keeps long scans out of the web server, a SQLite store that doubles as the transport between processes, and a YAML correlation layer that re-reads the finished scan for higher-order patterns.

As with every tool in this series, this is an educational tour of source code, not a manual for profiling private individuals. SpiderFoot is powerful enough that its own README points enterprise users at a commercial service, and automation at this scale can easily cross the line from research into harassment. Treat it the way its author intends: for authorized assessments, understanding your own attack surface, and appreciating how a decade of OSINT engineering fits together.

<div style="overflow-x:auto;"><img src="https://pyshine.com/assets/img/diagrams/spiderfoot/smicallef-spiderfoot-overview-architecture.svg" alt="SpiderFoot overview architecture diagram" style="max-width:100%;"></div>
<p><em>SpiderFoot at a glance: CLI and web entry, one scanner process, a plugin base and thread pool, the typed event model, and the module ecosystem feeding two storers, SQLite, and YAML correlations.</em></p>

Reading the overview from left to right:

- The run starts at [sf.py](https://github.com/smicallef/spiderfoot/blob/master/sf.py), which parses the CLI flags, applies default configuration, and either launches the web UI or a scan.
- [sfwebui.py](https://github.com/smicallef/spiderfoot/blob/master/sfwebui.py) serves the CherryPy interface with digest authentication and exposes the scan controls to the browser.
- [sfscan.py](https://github.com/smicallef/spiderfoot/blob/master/sfscan.py) defines SpiderFootScanner, the state machine that loads modules, seeds the first events, and runs the dispatch loop.
- [spiderfoot/plugin.py](https://github.com/smicallef/spiderfoot/blob/master/spiderfoot/plugin.py) is the base class every module inherits, including the notifyListeners fan-out.
- [spiderfoot/threadpool.py](https://github.com/smicallef/spiderfoot/blob/master/spiderfoot/threadpool.py) provides the fixed worker pool with per-module queues and concurrency caps.
- [spiderfoot/event.py](https://github.com/smicallef/spiderfoot/blob/master/spiderfoot/event.py) and [spiderfoot/target.py](https://github.com/smicallef/spiderfoot/blob/master/spiderfoot/target.py) model the flowing data and the target with its aliases.
- The [modules directory](https://github.com/smicallef/spiderfoot/tree/master/modules) holds the sfp_* ecosystem, including the two always-on storers [sfp__stor_db.py](https://github.com/smicallef/spiderfoot/blob/master/modules/sfp__stor_db.py) and [sfp__stor_stdout.py](https://github.com/smicallef/spiderfoot/blob/master/modules/sfp__stor_stdout.py).
- [spiderfoot/db.py](https://github.com/smicallef/spiderfoot/blob/master/spiderfoot/db.py) owns the SQLite schema under ~/.spiderfoot, and [spiderfoot/correlation.py](https://github.com/smicallef/spiderfoot/blob/master/spiderfoot/correlation.py) replays YAML rules over the finished scan.

## Why You Need This

The first reason is that SpiderFoot is the reference implementation of plugin architecture done right. Most automation tools hard-code a fixed sequence: step one, step two, step three. SpiderFoot instead asks every module to declare the data types it produces and consumes, then lets the event bus figure out the wiring. Reading this code teaches you how a system can grow to hundreds of components without a central orchestrator becoming a bottleneck, which is a lesson that transfers to far more than OSINT.

The second reason is that the hard problems of long-running automation are solved here in plain sight. How do you keep one greedy module from starving the rest? Per-module queues with per-module thread caps. How do you keep a browser tab connected to a scan that runs in another process? Poll SQLite once a second. How do you support aborting a scan mid-flight without corrupting state? A status machine with explicit abort-requested states that every module checks between events. These are production-grade answers, written in readable Python.

The third reason is defensive clarity. The module catalog reads like a survey of an organization's public footprint: DNS variants, code repositories, breached credentials, leaked documents, dark-web mentions, paste sites. Studying how the engine correlates those signals, including the YAML rules that combine multiple weak findings into one strong conclusion, is a practical education in how exposure is actually measured, and it runs entirely on your own machine against targets you are entitled to test.

## How It Works

<div style="overflow-x:auto;"><img src="https://pyshine.com/assets/img/diagrams/spiderfoot/smicallef-spiderfoot-architecture.svg" alt="SpiderFoot detailed architecture diagram" style="max-width:100%;"></div>
<p><em>Inside SpiderFoot: CLI defaults, the CherryPy UI, the scanner state machine, module selection, the waitForThreads dispatch loop, the plugin contract on the shared pool, and the SQLite-backed storers and correlations.</em></p>

### Understanding the Architecture

**A launcher with serious defaults.** sf.py starts by building sfConfig, a dictionary of global settings that shows how much operational thinking the project embeds: a maximum of three threads per module by default, a browser-like user agent, a five-second fetch timeout, and a TLD list pulled from publicsuffix.org and cached for seventy-two hours. The argparse surface is wide but honest, covering daemon mode on -l host:port, scans with -s, module lists with -m and -M, type filters with -t, use-case filters with -u, CSV/JSON output with -o, and a strict -x mode that resolves a target using only modules consuming its exact type. Proxy support is equally deliberate, mapping socks4, socks5, HTTP, and TOR schemes onto their conventional default ports.

**A web UI that refuses to be careless.** start_web_server wraps a CherryPy application with CORS support, Mako templating, and hardened response headers, and it guards access with HTTP digest authentication against a password file in ~/.spiderfoot. TLS is optional through a key and certificate pair, and the code warns loudly when it starts without any authentication configured. There is even a migration guard: legacy database and password files sitting in the application directory are refused with instructions to move them into the data directory, a small touch that says a lot about operational maturity.

**A scanner that is a state machine, not a function call.** SpiderFootScanner in sfscan.py walks an explicit status list from INITIALIZING through RUNNING to FINISHED, with dedicated ABORT-REQUESTED and ABORTING states so a cancelled scan ends predictably. The whole scan runs inside a daemon multiprocessing process, and the parent loop polls the database once a second for status updates, which is how the web UI stays live while the heavy lifting happens elsewhere. Ctrl-C is handled by marking the scan ABORTED in the database rather than simply killing the process.

**Module selection driven by type declarations.** Loading starts with SpiderFootHelpers.loadModulesAsDict, which imports every sfp_*.py in the modules directory and collects each one's metadata: name, description, categories, use cases, flags, and crucially the provides and consumes event lists. Selection then composes those lists: use-case and type filters keep modules that produce any requested event type, and the strict -x mode intersects modules consuming the target's own type with modules providing it. Before the scan starts, the code always appends the two storers, sfp__stor_db and sfp__stor_stdout, so persistence and console output are just ordinary subscribers to the event stream.

**One pool, many queues.** SpiderFootThreadPool is a fixed set of long-lived worker threads that round-robin across per-module input queues. Submission blocks when a module's in-flight count reaches its cap, which is what stops a chatty module from monopolizing the pool, and results come back through matching output queues that shutdown() can drain into a results dictionary. Each plugin worker even owns its own database handle, avoiding the classic SQLite cross-thread sharing mistake.

**The plugin contract and the event bus.** SpiderFootPlugin gives every module the same skeleton: a setup call, a watchedEvents list that defaults to everything, a producedEvents list, handleEvent for the work itself, and finish for teardown. The scanner seeds the queue with a ROOT event plus the target's first event type, then waitForThreads loops: it starts every module, pulls events off the shared queue, and delivers copies to interested listeners in priority order. A FINISHED sentinel, emitted only after three consecutive quiet passes, tells modules to run their finish routines, and the same loop finally triggers runCorrelations, which applies the project's thirty-seven YAML rules against the stored events to surface compound findings.

**Events and targets as first-class models.** SpiderFootEvent stamps every finding with an event type, the producing module, a link back to the source event that triggered it, and confidence, visibility, and risk scores validated in the zero-to-one-hundred range, then gives each instance a SHA-256 identity so the whole lineage graph can be recorded. SpiderFootTarget accepts eleven target types, from IP addresses and netblocks to human names and bitcoin addresses, and maintains aliases: an ASN target accumulates its subnets, an IP target accumulates its hostnames, and the matches method decides membership with proper subnet and parent-child domain logic.

**End to end.** A scan begins as a target string in sf.py, becomes a stateful process with a seeded event queue, fans out through a thread-pooled plugin bus where every finding is re-dispatched by type, lands in SQLite through an ordinary subscriber module, and finishes with YAML correlations computed over everything that was stored. The database is simultaneously the audit log, the UI's data source, and the transport between processes, which is why the design holds together at this scale.

## Advantages

- **Declarative plugin ecosystem.** Provides/consumes lists let hundreds of modules compose into scans with no central wiring to maintain.
- **Real concurrency control.** A fixed thread pool with per-module queues and caps keeps scans predictable instead of letting the loudest module win.
- **Process-safe design.** Scans run in a separate process and communicate through SQLite, so the UI stays responsive and crashes stay contained.
- **Honest abort semantics.** The explicit status machine means cancelled scans reach a defined end state rather than leaving orphans.
- **Output is just a module.** CSV, JSON, and tabular console formats come from the same stor_stdout subscriber that other modules rely on.
- **Correlation as data.** YAML rule files express compound findings declaratively, so new detections need no code changes.

## Benefits

- **Automate the boring recon.** One command sweeps dozens of data categories that would otherwise be separate manual lookups.
- **Learn production thread-pool patterns.** Bounded submission, per-task queues, and clean shutdown are implemented here in under three hundred readable lines.
- **Study type-driven architecture.** The event-bus wiring is a transferable template for any system built from many small, independent workers.
- **Understand your exposure.** The module catalog and correlation rules document, concretely, how much public signal one identifier can attract.
- **Audit everything locally.** MIT-licensed and self-contained, scans run against your own storage with no hosted component required.
- **A decade of hardening.** Proxy handling, DNS overrides, user-agent tricks, and TLD caching encode years of operational lessons.

## Usage

Installation follows the README, which also ships a Dockerfile for container deployments:

```bash
pip3 install -r requirements.txt
```

Start the web interface, then drive scans from the browser:

```bash
python sf.py -l 127.0.0.1:5001
```

Command-line scans take a target and a use case, with tab, CSV, or JSON output:

```bash
python sf.py -s example.com -u passive -o csv
python sf.py -s 8.8.8.8 -t IP_ADDRESS -x
```

Module and type exploration flags help you plan a scan before running one:

```bash
python sf.py -M
python sf.py -T
python sf.py -m sfp_dnsresolve
```

Correlations are requested explicitly with -C correlate, and every scan's events land in the SQLite database under ~/.spiderfoot for later inspection.

## Conclusion

SpiderFoot shows what a decade of patient engineering looks like: a plugin contract simple enough for a one-file module, an event bus that scales to hundreds of contributors, and operational details like abort states, per-module thread caps, and digest authentication that most projects never get around to. Nothing in the codebase is exotic, and that is the point; the architecture is a reminder that reliable automation comes from boring, well-composed mechanisms. Study the event flow, borrow the thread-pool discipline, and use the tool, as its author intends, against targets you are authorized to assess.

Links:

- [SpiderFoot on GitHub](https://github.com/smicallef/spiderfoot)
- [SpiderFoot license (MIT)](https://github.com/smicallef/spiderfoot/blob/master/LICENSE)
- [SpiderFoot modules directory](https://github.com/smicallef/spiderfoot/tree/master/modules)
