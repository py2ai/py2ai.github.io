---
layout: post
title: "Learn Traefik in a Single Post: A Complete Tutorial From Dynamic Routing and Middleware to ACME TLS and Production"
description: "A complete Traefik tutorial in one blog post. Covers the whole reverse proxy in 5 stages: entrypoints (web :80, websecure :443), routers (Host/Path rules, priority, dynamic config from Docker/K8s/file providers — no reload), services (load balancing, weighted backends), middleware (rate-limit, auth, compress, retry, headers, circuit breaker, redirect), and production (Let's Encrypt ACME auto-TLS, dashboard, Prometheus metrics, HA, vs Nginx comparison). Five hand-drawn diagrams, runnable config, and a quick-start roadmap."
date: 2026-07-28
header-img: "img/post-bg.jpg"
permalink: /Learn-Traefik-in-One-Post-Complete-Tutorial-Dynamic-Routing-Middleware-ACME-Quick-Start/
featured-img: ai-coding-frameworks/ai-coding-frameworks
tags:
  - Traefik
  - Reverse Proxy
  - Cloud Native
  - Docker
  - Let's Encrypt
  - Tutorial
categories: [Tutorial, DevOps, Infrastructure]
keywords: "Traefik tutorial one post, learn Traefik fast, Traefik entrypoints web websecure, Traefik routers Host Path rule priority, Traefik services load balancing, Traefik middleware rate-limit auth compress retry circuit-breaker, Traefik dynamic config Docker K8s file provider auto-discovery no reload, Traefik vs Nginx comparison, Traefik ACME Let's Encrypt auto TLS, Traefik dashboard metrics Prometheus, Traefik HA, Traefik quick start roadmap"
author: "PyShine"
---

# Learn Traefik in a Single Post: Complete Tutorial From Dynamic Routing and Middleware to ACME TLS

Traefik is the **cloud-native reverse proxy** — a modern alternative to [Nginx](/Learn-Nginx-in-One-Post-Complete-Tutorial-Reverse-Proxy-TLS-Load-Balancing-Quick-Start/) designed for containerized environments. Its defining feature: **dynamic configuration** — it reads routing rules from Docker labels, Kubernetes Ingress, or config files at runtime, and updates routes instantly as containers start and stop, with no config-file edit and no reload. It also has built-in Let's Encrypt ACME for automatic TLS certificates. This single post covers the whole proxy in five stages, with hand-drawn diagrams and runnable config.

## Learning Roadmap

![Traefik Roadmap](/assets/img/diagrams/traefik-tutorial/trk-roadmap.svg)

The roadmap moves from entrypoints (Stage 1), through routers (Stage 2), services (Stage 3), middleware (Stage 4), and production (Stage 5). The [Nginx tutorial](/Learn-Nginx-in-One-Post-Complete-Tutorial-Reverse-Proxy-TLS-Load-Balancing-Quick-Start/) and [Docker tutorial](/Learn-Docker-in-One-Post-Complete-Tutorial-Dockerfile-Volumes-Compose-Quick-Start/) are companions.

---

## Stage 1 — Entrypoints

### Listening on ports

An **entrypoint** is a port Traefik listens on. The defaults are:

```yaml
# traefik.yml (static config)
entryPoints:
  web:
    address: ":80"
  websecure:
    address: ":443"
```

- **`web` (:80)** — HTTP (usually redirected to HTTPS).
- **`websecure` (:443)** — HTTPS (TLS terminated here).

Entrypoints are defined in the **static config** (`traefik.yml` / CLI flags) — they change rarely (a restart is needed to add/remove them). Everything else (routers, services, middleware) is **dynamic** — updated at runtime without restart.

---

## Stage 2 — Routers

### Routing rules

![Traefik Architecture: Dynamic Routing](/assets/img/diagrams/traefik-tutorial/trk-arch.svg)

A **router** matches incoming requests by rule and routes them to a service:

```yaml
# dynamic config (file provider)
http:
  routers:
    api-router:
      rule: "Host(`api.example.com`) && PathPrefix(`/v1`)"
      entryPoints:
        - websecure
      service: api-service
      tls: certResolver: letsencrypt
      middlewares:
        - rate-limit
        - auth
```

### Rule syntax

| Rule | Matches |
|---|---|
| `Host(\`example.com\`)` | requests to this domain |
| `PathPrefix(\`/api\`)` | requests starting with /api |
| `Path(\`/exact/path\`)` | exact path match |
| `HostRegexp(\`{subdomain:[a-z]+}.example.com\`)` | regex host match |
| `Method(\`POST\`)` | HTTP method |
| `Headers(\`X-Custom\`, \`value\`)` | header match |

Rules can be combined with `&&` (AND) and `\|\|` (OR), and grouped with `()`. Priority can be set explicitly (`priority: 100`) — by default, longer/more-specific rules win.

### Dynamic config from Docker labels

```yaml
# docker-compose.yml
services:
  api:
    image: my-api
    labels:
      - "traefik.enable=true"
      - "traefik.http.routers.api.rule=Host(`api.example.com`)"
      - "traefik.http.routers.api.entrypoints=websecure"
      - "traefik.http.routers.api.tls.certresolver=letsencrypt"
      - "traefik.http.services.api.loadbalancer.server.port=8080"
```

When this container starts, Traefik **automatically** creates a router for `api.example.com` → the container's port 8080. When the container stops, the router is removed. **No config file edit, no reload** — this is Traefik's core advantage over Nginx.

---

## Stage 3 — Services

### Load balancing to backends

```yaml
http:
  services:
    api-service:
      loadBalancer:
        healthCheck:
          path: /health
          interval: 10s
        servers:
          - url: http://api1:8080
          - url: http://api2:8080
          - url: http://api3:8080
```

A **service** defines where traffic goes — a pool of backend servers with load balancing. Traefik supports:
- **Round-robin** (default) — distribute evenly.
- **Weighted** — `weight: 3` on one server (canary deployments).
- **Sticky sessions** — `sticky.cookie` for session affinity.
- **Health checks** — periodically check `/health`; remove unhealthy servers.

With the Docker provider, Traefik **auto-discovers** backend containers (it reads their IP + port from Docker's API) — you don't manually list server URLs. The service is created from the container's labels.

---

## Stage 4 — Middleware

### Chain of processing before the service

![Middleware: Chain of Processing Before the Service](/assets/img/diagrams/traefik-tutorial/trk-middleware.svg)

**Middleware** processes requests before they reach the service — and responses on the way back. Multiple middleware can be chained on a router:

```yaml
http:
  routers:
    api-router:
      rule: "Host(`api.example.com`)"
      service: api-service
      middlewares:
        - redirect-to-https
        - rate-limit
        - auth
        - compress
```

### Common middleware

| Middleware | What it does |
|---|---|
| **RateLimit** | requests/sec per source IP — protect backends from spikes |
| **BasicAuth / ForwardAuth** | validate credentials / delegate to an external auth service |
| **Compress** | gzip/brotli response compression |
| **Retry** | retry failed upstream requests (resilience) |
| **Headers** | add/remove headers (HSTS, CORS, security headers) |
| **CircuitBreaker** | stop sending to a failing backend (fail fast, let it recover) |
| **RedirectScheme** | redirect HTTP → HTTPS |
| **StripPrefix / ReplacePath** | rewrite the URL before it reaches the backend |
| **IPAllowList** | only allow specific IPs |

### ForwardAuth — delegate to an external auth service

```yaml
http:
  middlewares:
    auth:
      forwardAuth:
        address: http://auth-service:8080/check
        authResponseHeaders:
          - X-Auth-User
```

`ForwardAuth` sends each request to an auth service first; if the auth service returns 2xx, the request proceeds to the backend; if it returns 401/403, the request is rejected. This is how you integrate OIDC/OAuth ([OAuth tutorial](/Learn-OAuth-2-OIDC-in-One-Post-Complete-Tutorial-Flows-Tokens-PKCE-Security-Quick-Start/)) — the auth service validates the token, and Traefik forwards the authenticated request.

---

## Stage 5 — Production

![Production: TLS ACME, Dashboard, Metrics, HA](/assets/img/diagrams/traefik-tutorial/trk-prod.svg)

### ACME — automatic Let's Encrypt TLS

```yaml
# traefik.yml
certificatesResolvers:
  letsencrypt:
    acme:
      email: admin@example.com
      storage: /acme.json
      httpChallenge:
        entryPoint: web
```

Traefik **automatically obtains and renews** Let's Encrypt TLS certificates. When a router with `tls.certresolver=letsencrypt` receives a request for a new domain, Traefik initiates the ACME challenge, gets the certificate, and serves HTTPS — all without manual `certbot` runs. Certificates are stored in `acme.json` (shared across instances for HA).

**Challenge types**:
- **HTTP-01** — Traefik serves the challenge on port 80 (simplest, works through most firewalls).
- **TLS-01** — challenge via TLS ALPN.
- **DNS-01** — challenge via DNS TXT record (required for wildcard certs `*.example.com`).

### Dashboard

```yaml
# traefik.yml
api:
  dashboard: true
  insecure: false    # serve via a router, not on a public port

# router for the dashboard
http:
  routers:
    dashboard:
      rule: "Host(`traefik.example.com`)"
      service: api@internal
      middlewares:
        - dashboard-auth
```

Traefik has a built-in **web dashboard** at `/dashboard/` showing live routers, services, middleware, and entrypoints. Secure it with BasicAuth or ForwardAuth — never expose it publicly without auth.

### Metrics

```yaml
metrics:
  prometheus:
    addEntryPointsLabels: true
    addServicesLabels: true
    addRoutersLabels: true
```

Traefik exposes a **Prometheus** `/metrics` endpoint — request count, latency, status codes per router/service/entrypoint. This integrates directly with the [Prometheus](/Learn-Prometheus-in-One-Post-Complete-Tutorial-Metrics-PromQL-Alerting-Grafana-Quick-Start/) + [Grafana](/Learn-Loki-in-One-Post-Complete-Tutorial-Labels-LogQL-Promtail-Grafana-Quick-Start/) stack.

### High availability

Run **2+ Traefik instances** behind a load balancer (or with a VIP). Share the ACME storage (`acme.json` on a shared volume or in a KV store like Consul) so instances don't duplicate cert issuance. Use a **KV store** (Consul, Redis, etcd) for shared dynamic config — all instances read the same routing rules.

### Traefik vs Nginx

| | Traefik | Nginx |
|---|---|---|
| **Config** | dynamic (Docker/K8s labels, no reload) | static file + `nginx -s reload` |
| **TLS** | built-in ACME (Let's Encrypt auto) | needs `certbot` externally |
| **Service discovery** | auto (reads Docker/K8s API) | manual (edit upstream list) |
| **Cloud-native** | designed for containers | general-purpose (works everywhere) |
| **Fine control** | less tweakable (opinionated) | more control over edge cases |
| **Performance** | fast (Go, built for it) | fast (C, battle-tested) |
| **Best for** | containerized/Docker/K8s | traditional apps, custom edge cases |

**Use Traefik** when you're in a containerized world (Docker/K8s) and want auto-discovery + auto-TLS with zero manual config. **Use Nginx** when you need maximum control, have non-container backends, or have complex edge requirements Traefik's abstraction can't express.

> **Pitfall:** Traefik's dynamic config is powerful but can be hard to debug — when a route doesn't work, you need to check the dashboard to see if the router was created, what rule it has, and which middleware is attached. Nginx's config file is right there to read. For complex setups, Traefik's abstraction can hide what's happening.

---

## Quick-Start Checklist

1. **Run Traefik in Docker** — `docker run -p 80:80 -p 443:443 -v /var/run/docker.sock:/var/run/docker.sock traefik:v3`.
2. **Open the dashboard** — `http://localhost:8080/dashboard/` (or route it via a router).
3. **Label a container** — `traefik.http.routers.app.rule=Host(\`localhost\`)` + `traefik.http.services.app.loadbalancer.server.port=8080`.
4. **Watch it appear** in the dashboard — the router is created automatically.
5. **Add middleware** — rate-limit, compress, or auth on the router.
6. **Enable ACME** — `certificatesResolvers.letsencrypt.acme...` + `tls.certresolver=letsencrypt`.
7. **Add health checks** — `loadBalancer.healthCheck.path=/health`.
8. **Connect Prometheus** — enable `metrics.prometheus` and scrape `/metrics`.
9. **Run 2 instances** for HA with shared ACME storage.
10. **Compare with Nginx** — if you're container-native, Traefik's auto-discovery saves you from config-file + reload cycles.

## Common Pitfalls

- **Dashboard exposed without auth** — the dashboard shows your entire routing config; secure it with BasicAuth or ForwardAuth.
- **No ACME storage persistence** — `acme.json` must be on a persistent volume; otherwise certs are re-requested on every restart (rate limits).
- **Forgetting `traefik.enable=true`** — without this label, Traefik ignores the container (security: only explicitly-enabled containers are routed).
- **Rule not matching** — check the dashboard; common issues: wrong backtick escaping in YAML, missing `&&` between conditions, port in the Host rule.
- **No health check** — unhealthy backends keep receiving traffic; add `healthCheck.path`.
- **Single instance in prod** — no HA; run 2+ with shared ACME storage.
- **Port not exposed** — the container's port must be exposed (Docker `ports` or `expose`) for Traefik to route to it.
- **Middleware order matters** — they run in the listed order; auth before rate-limit is different from rate-limit before auth.

## Further Reading

- [Traefik Docs](https://doc.traefik.io/traefik/) — the official reference
- [Traefik v3 Migration](https://doc.traefik.io/traefik/v3-migration/) — what's new in v3
- [Traefik Helm Chart](https://github.com/traefik/traefik-helm-chart) — Kubernetes deployment
- [Traefik + Docker Compose](https://doc.traefik.io/traefik/user-guides/docker-compose/) — the quick-start guide
- [Let's Encrypt + Traefik](https://doc.traefik.io/traefik/https/acme/) — ACME configuration

## Related guides

Traefik is the cloud-native edge layer — these PyShine tutorials connect to it:

- **[Learn Nginx in One Post](/Learn-Nginx-in-One-Post-Complete-Tutorial-Reverse-Proxy-TLS-Load-Balancing-Quick-Start/)** — the comparison; Nginx (static, general-purpose) vs Traefik (dynamic, cloud-native).
- **[Learn Docker in One Post](/Learn-Docker-in-One-Post-Complete-Tutorial-Dockerfile-Volumes-Compose-Quick-Start/)** — Traefik reads Docker labels; run it as a container with the Docker socket.
- **[Learn Kubernetes in One Post](/Learn-Kubernetes-in-One-Post-Complete-Tutorial-Pods-Deployments-Services-Production-Quick-Start/)** — the Traefik Helm chart deploys as an IngressController.
- **[Learn Prometheus in One Post](/Learn-Prometheus-in-One-Post-Complete-Tutorial-Metrics-PromQL-Alerting-Grafana-Quick-Start/)** — Traefik's `/metrics` endpoint feeds Prometheus.
- **[Learn OAuth 2.0 + OIDC in One Post](/Learn-OAuth-2-OIDC-in-One-Post-Complete-Tutorial-Flows-Tokens-PKCE-Security-Quick-Start/)** — ForwardAuth middleware delegates to an OIDC auth service.

---

Traefik's value is **dynamic, auto-discovering, auto-TLS configuration for containerized environments** — label a container, and it's routed; stop it, and the route is gone. No config file, no reload, no `certbot`. The five stages here — entrypoints, routers, services, middleware, production — cover everything from a single Docker-labeled container to a multi-instance, ACME-secured, ForwardAuth-gated, Prometheus-monitored, HA production edge. The two habits that pay off: **always secure the dashboard** (it exposes your whole routing config), and **persist `acme.json`** (without it, every restart re-requests certs and hits Let's Encrypt rate limits). Run Traefik with the Docker socket, label a container `traefik.http.routers.app.rule=Host(\`localhost\`)`, and watch the route appear in the dashboard — once you've seen dynamic config work, the Nginx reload cycle feels archaic.