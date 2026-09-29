---
layout: post
title: "Uptime Kuma: The Self-Hosted Monitoring Engine - Inside louislam/uptime-kuma"
description: "A source-level tour of Uptime Kuma, the self-hosted uptime monitoring tool with 90k+ GitHub stars. We trace the Node.js monitor scheduler loop, the pluggable monitor-type abstraction covering HTTP, DNS, SNMP and Docker, the 100+ notification providers, and the Socket.IO-driven Vue dashboard."
date: 2026-09-29
header-img: "img/post-bg.jpg"
permalink: /Uptime-Kuma-Self-Hosted-Monitoring-Source-Tour/
featured-img: "ai-coding-frameworks/ai-coding-frameworks"
image: https://pyshine.com/assets/img/diagrams/uptime-kuma/louislam-uptime-kuma-architecture.svg
tags:
  - Uptime Kuma
  - Monitoring
  - Node.js
  - Self-Hosted
categories: [AI, Open Source]
keywords: "uptime kuma, self-hosted monitoring, uptime monitoring, status page, node.js monitoring, socket.io dashboard, prometheus metrics, docker container monitoring, heartbeat scheduler, notification providers, open source monitoring, knex migrations, better-sqlite3"
author: "PyShine"
---

Every downtime story starts the same way: a user notices before you do. Uptime Kuma exists to flip that order, and it has done so well enough to gather more than 90,000 GitHub stars. It is a self-hosted monitoring tool that watches HTTP endpoints, TCP ports, DNS records, Docker containers, game servers, and dozens of other targets, then paints the results onto a live dashboard and status pages that you fully control.

Under the hood it is a Node.js application with a clear split: an Express plus Socket.IO server under `server/`, a Vue 3 single-page frontend under `src/`, and persistence handled through knex with better-sqlite3 (plus MariaDB options) in `server/database.js`. There is no heavyweight agent to install, no external dependency on a SaaS control plane; one process, one data directory, and a browser tab.

That combination — a scheduler, a pluggable check system, an alerting fan-out, and a realtime UI — makes the codebase unusually rewarding to read. This post walks the actual source of `louislam/uptime-kuma` (the `master` branch, version 3.0.0-beta.0 at the time of writing) and follows a heartbeat from the moment a monitor fires to the moment a notification lands in your chat client.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/uptime-kuma/louislam-uptime-kuma-overview-architecture.svg" alt="Architecture overview of the louislam/uptime-kuma repository" style="max-width:100%;height:auto;" />
</div>

*Overview of Uptime Kuma's major subsystems: the server entry point boots a singleton server that schedules monitors, dispatches checks to the monitor-type registry, records heartbeats, fans out notifications, and streams everything to the Vue dashboard over Socket.IO.*

Reading the overview from left to right: `server/server.js` boots the `UptimeKumaServer` singleton and connects the database; the scheduler inside `server/model/monitor.js` starts one beat loop per active monitor and dispatches each check through the `MonitorType` base class in `server/monitor-types/`; successful or failed checks become heartbeat beans that are stored, fed to `UptimeCalculator`, and pushed to the browser through the Socket.IO emitters in `server/client.js`; whenever a beat is "important" — a status transition — `server/notification.js` fans the alert out to the provider directory in `server/notification-providers/`; meanwhile cron jobs and the Prometheus exporter round out the background picture.

## Why You Need This

Managed uptime checkers are convenient until they are not: your check intervals are someone else's rate limits, your data lives in someone else's dashboard, and your status page carries someone else's branding. Uptime Kuma takes all three back. Checks can run as frequently as every 20 seconds, heartbeats and response times accumulate in your own database, and multiple status pages can be mapped to your own domains via the domain mapping loaded in `server/model/status_page.js`.

The second problem is breadth of targets. Real infrastructure is more than an HTTPS URL: you may need to confirm an SMTP server answers, that an SNMP OID reports a sane value, that an MQTT broker publishes, that a database actually executes a query, or that a Docker container reports healthy. Uptime Kuma ships handlers for all of these — twenty-six type classes in `server/monitor-types/` plus several core types handled inline — so one tool replaces a pile of single-purpose scripts.

The third problem is alert fatigue. A naive checker pings you on every failed poll. Uptime Kuma's scheduler distinguishes a first failing beat from a confirmed outage: retries are marked PENDING until `maxretries` is exhausted, and only "important" beats — genuine UP/DOWN transitions — trigger notifications, with an optional `resendInterval` to re-alert while a service stays down. That logic, written directly into the beat loop in `server/model/monitor.js`, is the difference between a useful pager and a muted one.

Finally, there is the trust question. Because the whole stack is self-hosted and MIT-licensed, you can read exactly what a "monitor" does, what data leaves your machine, and how credentials such as HTTP basic auth or SNMPv3 settings are handled. The codebase is small enough to audit in an afternoon and active enough to be worth it.

## How It Works

The architecture is easiest to grasp by following one heartbeat end to end, so let's zoom into the detailed diagram and then walk the modules.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/uptime-kuma/louislam-uptime-kuma-architecture.svg" alt="Detailed architecture of the louislam/uptime-kuma repository" style="max-width:100%;height:auto;" />
</div>

*Detailed source map of louislam/uptime-kuma: the runtime layer boots Express, better-auth, and Socket.IO; the scheduler drives monitor types and heartbeats; alerting fans out to providers; the realtime layer connects `server/client.js` to the Vue app; data flows through knex migrations into SQLite or MariaDB.*

### Understanding the Architecture

**The boot sequence.** `server/server.js` is deliberately procedural: it validates the Node.js version against `package.json`'s engines field, loads `.env`, then obtains the `UptimeKumaServer` singleton from `server/uptime-kuma-server.js`. That constructor builds the Express app (or HTTPS variant), reads `dist/index.html` for the embedded frontend, registers every monitor type into the static `UptimeKumaServer.monitorTypeList`, and creates the Socket.IO server with a strict WebSocket origin check (`allowRequest` compares `Origin` against `Host`, with proxy support). Back in `server.js`, the async bootstrap connects the database via `Database.connect()` and `Database.patch()`, mounts the API and status-page routers, and attaches per-connection socket handlers before calling `startMonitors()` once the HTTP listener is up.

**The beat loop.** Each active monitor becomes a `Monitor` bean from `server/model/monitor.js`, and `start(io)` defines a closure named `beat()` wrapped by `safeBeat()`, which catches anything unexpected and reschedules. There is no cron-per-monitor and no setInterval drift: after each beat the loop computes `intervalRemainingMs` as the configured interval minus elapsed time (minimum 1 ms) and re-arms itself with `setTimeout`. Each beat dispenses a fresh heartbeat bean seeded with `status = DOWN`, flips it for upside-down mode, and short-circuits to a MAINTENANCE status if `Monitor.isUnderMaintenance()` says so. On startup, `startMonitors()` in `server.js` staggers each monitor's first run with a random 300–1000 ms sleep so a freshly booted server does not fire every probe simultaneously.

**The monitor-type abstraction.** The base class in `server/monitor-types/monitor-type.js` is small by design: a `check(monitor, heartbeat, server)` method to override, plus `supportsConditions`, `conditionVariables`, and `allowCustomStatus` flags that drive both the edit form and the contract that non-UP statuses must throw inside `check()`. `server/uptime-kuma-server.js` registers twenty-six implementations — including `DnsMonitorType`, `SNMPMonitorType`, `MqttMonitorType`, `TCPMonitorType`, `RealBrowserMonitorType` (driven through Playwright), and database checkers for PostgreSQL, MySQL, MSSQL, MongoDB, Redis, and Oracle. The beat loop dispatches with a single dictionary lookup (`this.type in UptimeKumaServer.monitorTypeList`), while the core HTTP/keyword/json-query family stays inline in `server/model/monitor.js`, where it composes axios agents for mTLS, proxies, OIDC client-credential tokens, and certificate inspection. Several types also plug into the condition evaluator in `server/monitor-conditions/evaluator.js`, which lets users write rules against returned DNS records or SNMP values.

**Notification dispatch.** When a beat finishes, `Monitor.isImportantBeat()` decides whether it represents a transition worth recording loudly. For important beats the scheduler calls `Monitor.sendNotification()`, which loads the monitor's providers through the `monitor_notification` join table and calls `Notification.send()` from `server/notification.js`. That class is a registry: `Notification.init()` instantiates every provider from `server/notification-providers/` — one hundred and ten files covering Telegram, Discord, Slack, SMTP, Gotify, ntfy, PagerDuty, and many more — keyed by a unique `name`, so dispatch is a dictionary hit on `notification.type`. The scheduler also implements re-alerting: while a monitor stays DOWN, `downCount` increments per beat and, once it reaches the monitor's `resendInterval`, the notification fires again. Certificate expiry alerts follow a separate path via `sendCertNotificationByTargetDays()`, gated by a sent-history table so they fire once per threshold.

**The realtime dashboard.** Uptime Kuma's UI is a Vue 3 + Vite SPA (`src/main.js`, `src/App.vue`, `src/router.js`) that talks to the server almost entirely over Socket.IO rather than REST. After login, `afterLogin()` in `server.js` joins the socket to a per-user room and replays the monitor list, heartbeats, notification list, proxy list, and the available monitor types via emitters in `server/client.js`. During operation every beat calls `io.to(this.user_id).emit("heartbeat", bean.toJSON())` straight from `server/model/monitor.js`, so the dashboard updates within milliseconds of a check finishing, and `Monitor.sendStats()` follows with chart aggregates. Administrative operations — adding or pausing monitors, managing proxies, Docker hosts, maintenance windows, API keys, status pages — arrive as socket events handled by the modules in `server/socket-handlers/`, with `checkLogin()` and better-auth sessions (`server/better-auth.ts`) enforcing ownership on each call.

**Storage and housekeeping.** `server/database.js` wraps knex over either better-sqlite3 (the default `kuma.db` file) or MariaDB — including a self-contained embedded MariaDB — and `server/setup-database.js` presents a first-run page when no engine has been chosen yet. Schema evolution uses proper knex migrations in `db/knex_migrations/`, with the historical SQL patch files preserved in `db/old_migrations/`. Two cron jobs scheduled in `server/jobs.js` keep the dataset healthy: `clear-old-data` prunes expired heartbeats daily at 03:14, and `incremental-vacuum` reclaims SQLite space every five minutes. Alongside the database, `server/uptime-calculator.js` maintains rolling 24-hour, 30-day, and 1-year uptime windows, and `server/prometheus.js` exposes the same numbers on the basic-auth-protected `/metrics` endpoint.

Put together, the end-to-end flow looks like this: the server boots, loads monitors, and each one's `safeBeat` loop fires at its interval; the beat dispatches to either inline logic or a `MonitorType.check()` implementation; the result becomes a heartbeat that is stored, calculated into uptime windows, emitted to the user's browser room, and exported to Prometheus; and if the beat marks a real status transition, the notification registry fans the alert out to every provider attached to that monitor — all within one Node.js process.

## Advantages

- **Genuinely self-hosted.** One Node.js process with a local data directory; no cloud dependency, and the MIT license (`LICENSE`) lets you run, modify, and redistribute freely.
- **Broad check coverage out of the box.** Twenty-six pluggable type classes in `server/monitor-types/` plus inline HTTP, keyword, JSON query, ping, push, Docker, RADIUS, and Kafka producer checks in `server/model/monitor.js`.
- **Thoughtful alert semantics.** Retry-then-PENDING logic, importance-based notification, upside-down mode, maintenance windows, and `resendInterval` re-alerting are all encoded in the scheduler rather than bolted on.
- **Realtime by architecture.** Beats are pushed to the browser over Socket.IO rooms per user (`io.to(this.user_id).emit("heartbeat", ...)`), so the dashboard reflects failures in seconds without polling.
- **Observable from external systems.** A Prometheus-compatible `/metrics` endpoint, RSS-style status pages, and per-monitor charts make the same data consumable by other tools.
- **Practical data lifecycle.** knex migrations, a daily prune job, and incremental vacuuming keep the heartbeat table fast even with 20-second check intervals running for months.

## Benefits

- **Downtime caught before customers notice.** Intervals down to 20 seconds with per-monitor retry tuning mean real outages are confirmed quickly and false positives are suppressed.
- **One pane of glass for mixed infrastructure.** Websites, DNS records, containers, brokers, and databases share a dashboard, tags, and group monitors instead of living in separate scripts.
- **Alerts where your team already is.** With one hundred and ten provider integrations in `server/notification-providers/`, the odds your chat or paging tool is covered are high — and webhooks cover the rest.
- **Public transparency on your own domain.** Status pages from `server/model/status_page.js` can be mapped to specific hostnames, giving each service or audience its own branded page.
- **Auditable security posture.** Session handling through better-auth, a WebSocket origin check, optional 2FA, and API keys are all visible in plain JavaScript/TypeScript you can review.
- **Low operational cost.** The embedded SQLite engine and cron-driven housekeeping mean most installations run comfortably on a small VPS or a single Docker container.

## Usage

The README's recommended path is Docker Compose:

```bash
mkdir uptime-kuma
cd uptime-kuma
curl -o compose.yaml https://raw.githubusercontent.com/louislam/uptime-kuma/master/compose.yaml
docker compose up -d
```

Or a plain Docker run, persisting data in the `uptime-kuma` volume:

```bash
docker run -d --restart=always -p 3001:3001 -v uptime-kuma:/app/data --name uptime-kuma louislam/uptime-kuma:2
```

For a non-Docker setup, the README lists Git and PM2 and installs from source:

```bash
git clone https://github.com/louislam/uptime-kuma.git
cd uptime-kuma
npm run setup

# Option 1. Try it
node server/server.js

# (Recommended) Option 2. Run in the background using PM2
# Install PM2 if you don't have it:
npm install pm2 -g && pm2 install pm2-logrotate

# Start Server
pm2 start server/server.js --name uptime-kuma
```

The server then listens on port 3001. Note that the README's stated requirement is Node.js >= 20.4, while the `master` branch checked here (the 3.0.0-beta.0 development line) declares `engines: node >= 26.2.0` in `package.json`, so check the engines field of the release you actually install.

## Conclusion

Uptime Kuma's source is a study in doing one thing well and then widening it cleanly: a self-rescheduling beat loop, a one-method check abstraction that has absorbed twenty-six monitor types, a registry-driven notification fan-out, and a Socket.IO pipeline that keeps the dashboard honest in real time. Because every subsystem lives in an obvious file under `server/`, it is also one of the more approachable large Node.js codebases to learn from — whether you want to add a monitor type, wire up a niche chat provider, or simply understand how a monitoring product is really built. Clone it, read `server/model/monitor.js` top to bottom, and you will come away knowing exactly where your next heartbeat goes.

Links:

- GitHub repository: <https://github.com/louislam/uptime-kuma>
- Official wiki (installation and update docs): <https://github.com/louislam/uptime-kuma/wiki>
- Live demo: <https://demo.kuma.pet/start-demo>
