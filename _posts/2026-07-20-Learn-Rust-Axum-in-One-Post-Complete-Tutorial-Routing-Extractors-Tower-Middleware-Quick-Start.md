---
layout: post
title: "Learn Rust and Axum in a Single Post: A Complete Tutorial From Routing and Extractors to Tower Middleware and Deployment"
description: "A complete Rust + Axum web backend tutorial in one blog post. Covers the whole framework in 5 stages: setup (Cargo, tokio, axum, first server), routing (Router, GET/POST, path params, nesting), extractors (State, Path, Query, Json, headers, custom), handlers (async functions, error handling, IntoResponse), and middleware + deploy (Tower layers, from_fn, tracing, sqlx, Docker, shuttle). Five hand-drawn diagrams, runnable Rust, and a quick-start roadmap."
date: 2026-07-20
header-img: "img/post-bg.jpg"
permalink: /Learn-Rust-Axum-in-One-Post-Complete-Tutorial-Routing-Extractors-Tower-Middleware-Quick-Start/
featured-img: ai-coding-frameworks/ai-coding-frameworks
tags:
  - Rust
  - Axum
  - Web Backend
  - Tower
  - Tokio
  - Tutorial
categories: [Tutorial, Backend, Rust]
keywords: "Rust Axum tutorial one post, learn Axum fast, Axum router GET POST path params nest, Axum extractors State Path Query Json, Axum handler async IntoResponse error handling, Tower middleware layers from_fn, tracing tower-http CORS compression, sqlx async SQL compile-time checked, serde JSON, tokio async runtime, Axum Docker shuttle deploy, Axum quick start roadmap"
author: "PyShine"
---

# Learn Rust and Axum in a Single Post: Complete Tutorial From Routing and Extractors to Tower Middleware and Deployment

Axum is the web framework from the Tokio team — a thin routing layer on Tower, on Hyper, on Tokio. It's the Rust backend equivalent of [FastAPI](/Learn-FastAPI-in-One-Post-Complete-Tutorial-Pydantic-Async-Dependency-Injection-Quick-Start/) (Python) or [Express](/Learn-Node-js-Express-in-One-Post-Complete-Tutorial-Event-Loop-Middleware-Quick-Start/), with Rust's compile-time safety and the Tokio async runtime's performance. This single post teaches the whole framework in five stages, with hand-drawn diagrams and runnable Rust.

## Learning Roadmap

![Rust + Axum Roadmap](/assets/img/diagrams/rust-axum-tutorial/axum-roadmap.svg)

The roadmap moves from setup (Stage 1), through routing (Stage 2), the extractors that read requests (Stage 3), writing handlers (Stage 4), and the middleware + deployment layer (Stage 5). You'll want the [Rust tutorial](/Learn-Rust-in-One-Post-Complete-Tutorial-Ownership-Borrow-Async-Quick-Start/) first — ownership, async, and traits are prerequisites.

---

## Stage 1 — Setup

### The stack

![Axum Architecture: Tokio + Tower + Hyper](/assets/img/diagrams/rust-axum-tutorial/axum-arch.svg)

Axum is a **thin layer**:
- **Axum** — routing + extractors (the framework you write against).
- **Tower** — the middleware + `Service` trait abstraction (Axum's middleware is Tower).
- **Hyper** — HTTP/1 + HTTP/2 (the actual HTTP server).
- **Tokio** — the async runtime (I/O, timers, tasks).

Each layer is composable and replaceable — Axum doesn't reinvent the runtime or HTTP; it builds on the Rust async ecosystem.

### A first server

```toml
# Cargo.toml
[dependencies]
axum = "0.7"
tokio = { version = "1", features = ["full"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
```

```rust
use axum::{routing::get, Router};

#[tokio::main]
async fn main() {
    let app = Router::new().route("/", get(|| async { "Hello, world!" }));

    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
```

Run it with `cargo run` and hit `http://localhost:3000`. That's a complete async HTTP server in 8 lines — the `#[tokio::main]` macro sets up the async runtime; `axum::serve` runs the router on the listener.

> **Pitfall:** Forget `#[tokio::main]` (or the `features = ["full"]` on tokio) and you get a cryptic "main function is not async" or "runtime not found" error. The runtime has to be set up before any `await`.

---

## Stage 2 — Routing

### Routes and methods

```rust
use axum::{routing::{get, post, put, delete}, Router};

let app = Router::new()
    .route("/", get(root))
    .route("/users", post(create_user))
    .route("/users/:id", get(get_user).put(update_user).delete(delete_user))
    .route("/users/:id/posts", get(list_user_posts));
```

Each `.route(path, method_router)` maps a path + HTTP method(s) to a handler. Path params use `:name` syntax (`:id`).

### Path params

```rust
async fn get_user(axum::extract::Path(id): axum::extract::Path<u64>) -> String {
    format!("User {}", id)
}
```

`Path(id): Path<u64>` extracts the `:id` from the URL and parses it as a `u64` — if the path segment isn't a valid `u64`, Axum returns a 400 automatically. You can also extract a struct: `Path(params): Path<UserParams>` where `UserParams` derives `Deserialize`.

### Nesting routers

```rust
let api_routes = Router::new()
    .route("/users", get(list_users).post(create_user))
    .route("/users/:id", get(get_user));

let app = Router::new()
    .nest("/api/v1", api_routes);   // -> /api/v1/users, /api/v1/users/:id
```

`nest("/prefix", router)` mounts a sub-router under a prefix — how you split routes into modules. Each module's `Router` composes into the final app.

> **Pitfall:** Axum routes are **statically dispatched** — the path patterns (`/users/:id`) are matched by a generated router, not a runtime regex. This is fast (the router is a state machine), but it means you can't have overlapping patterns (`/users/:id` and `/users/me` conflict unless ordered — Axum resolves this, but be deliberate about path design).

---

## Stage 3 — Extractors

### How Axum reads requests

An **extractor** is a type that implements `FromRequest` (or `FromRequestParts`) — it pulls data out of the request and parses it. You list extractors as handler arguments, in any order.

![Extractors: How Axum Reads Requests](/assets/img/diagrams/rust-axum-tutorial/axum-extractors.svg)

```rust
use axum::extract::{State, Path, Query};
use serde::Deserialize;

#[derive(Deserialize)]
struct Pagination { page: u64, limit: u64 }

async fn list_users(
    State(db): State<DbPool>,
    Query(pagination): Query<Pagination>,
) -> impl IntoResponse {
    let users = db.query("SELECT * FROM users LIMIT $1 OFFSET $2", &[&pagination.limit, &pagination.page]).await?;
    Json(users)
}
```

| Extractor | What it reads |
|---|---|
| `State<T>` | shared app state (DB pool, config) |
| `Path<T>` | URL path params (`/users/:id` → `id`) |
| `Query<T>` | query string (`?page=1&limit=10`) |
| `Json<T>` | deserialize JSON body via serde |
| `Form<T>` | URL-encoded form body |
| `Header<T>` | a specific header |
| `TypedHeader` | typed headers via the `headers` crate |
| `Custom` | implement `FromRequestParts` yourself |

### `State` — shared application state

```rust
#[derive(Clone)]
struct AppState { db: PgPool, config: Config }

let app = Router::new()
    .route("/users", get(list_users))
    .with_state(AppState { db, config });   // inject state

async fn list_users(State(state): State<AppState>) -> impl IntoResponse {
    let users = sqlx::query_as::<_, User>("SELECT * FROM users").fetch_all(&state.db).await?;
    Json(users)
}
```

`State` must be `Clone` (it's cloned per request, so wrap expensive things in `Arc`). The `.with_state(...)` call injects it into the router; handlers that need it add `State<T>` as an argument.

### Custom extractors

```rust
struct ApiKey(String);

impl<S> FromRequestParts<S> for ApiKey where S: Send + Sync {
    type Rejection = StatusCode;
    async fn from_request_parts(parts: &mut Parts, _state: &S) -> Result<Self, Self::Rejection> {
        let key = parts.headers.get("x-api-key").and_then(|v| v.to_str().ok());
        match key {
            Some(k) => Ok(ApiKey(k.to_string())),
            None => Err(StatusCode::UNAUTHORIZED),
        }
    }
}

async fn protected(ApiKey(key): ApiKey) -> String { format!("key: {}", key) }
```

Implement `FromRequestParts` to extract from headers/headers without consuming the body, or `FromRequest` (which consumes the body). Custom extractors are how you build auth, request IDs, and tenant resolution.

> **Pitfall:** Extractor order matters: body-consuming extractors (`Json`, `Form`) must come **last** — once the body is read, you can't re-read it. The `Parts` extractors (`State`, `Path`, `Query`, `Header`) are fine in any order.

---

## Stage 4 — Handlers

### Handlers are just async functions

A handler is any async function whose arguments are extractors and whose return type is `IntoResponse`. Axum generates the dispatch code at compile time.

```rust
use axum::Json;
use serde::Serialize;

#[derive(Serialize)]
struct User { id: u64, name: String }

async fn get_user(Path(id): Path<u64>) -> Json<User> {
    Json(User { id, name: "Ada".into() })
}
```

`Json<T>` is both an extractor (deserialize request body) and a response (serialize + set `Content-Type: application/json`). The return just needs `impl IntoResponse` — `String`, `&'static str`, `Json<T>`, `(StatusCode, Json<T>)`, tuples, and custom types all work.

### Error handling

```rust
use axum::response::{IntoResponse, Response};
use axum::http::StatusCode;

enum AppError {
    NotFound,
    Database(sqlx::Error),
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        match self {
            AppError::NotFound => (StatusCode::NOT_FOUND, "not found").into_response(),
            AppError::Database(e) => {
                tracing::error!("db error: {:?}", e);
                (StatusCode::INTERNAL_SERVER_ERROR, "internal error").into_response()
            }
        }
    }
}

async fn get_user(Path(id): Path<u64>, State(db): State<DbPool>) -> Result<Json<User>, AppError> {
    let user = sqlx::query_as::<_, User>("SELECT * FROM users WHERE id = $1")
        .bind(id as i64)
        .fetch_optional(&db)
        .await
        .map_err(AppError::Database)?
        .ok_or(AppError::NotFound)?;
    Ok(Json(user))
}
```

The pattern: define an error enum, implement `IntoResponse` to map each variant to a status code, and return `Result<T, AppError>` from handlers. The `?` operator propagates errors through your handler; Axum converts them to HTTP responses via your `IntoResponse` impl. Never leak internal error details to the client — log them (`tracing::error!`) and return a generic message.

> **Pitfall:** Returning `500` with the raw `Debug` of an error leaks internals (and sometimes secrets). Always map errors through a custom `IntoResponse` that logs the full error server-side and returns a sanitized message to the client.

---

## Stage 5 — Middleware + Deploy

### Tower middleware

![Middleware: Tower Layers + from_fn](/assets/img/diagrams/rust-axum-tutorial/axum-middleware.svg)

Axum middleware is **Tower layers** — composable wrappers around the `Service` trait. The simplest way to write one is `from_fn`:

```rust
use axum::middleware::from_fn;
use axum::extract::Request;
use axum::middleware::Next;
use axum::response::Response;

async fn auth_middleware(req: Request, next: Next) -> Response {
    if let Some(token) = req.headers().get("authorization") {
        if validate_token(token.to_str().unwrap()) {
            return next.run(req).await;   // pass to the next layer / handler
        }
    }
    (StatusCode::UNAUTHORIZED, "missing or invalid token").into_response()
}

let app = Router::new()
    .route("/protected", get(protected))
    .layer(from_fn(auth_middleware));   // apply to all routes in this router
```

`from_fn` turns an `async fn(req, next) -> Response` into a Tower layer. `next.run(req)` hands the request to the next layer (or the handler); your middleware can transform the request before and the response after. Apply middleware with `.layer(...)` — the order matters: layers added later run first (outermost).

### Common middleware (tower-http)

```rust
use tower_http::{
    trace::TraceLayer,
    cors::CorsLayer,
    compression::CompressionLayer,
    timeout::TimeoutLayer,
};

let app = Router::new()
    .route("/", get(root))
    .layer(TraceLayer::new_for_http())       // structured request logging
    .layer(CompressionLayer::new())          // gzip response
    .layer(CorsLayer::permissive())          // CORS
    .layer(TimeoutLayer::new(Duration::from_secs(30)));  // request timeout
```

`tower-http` provides the standard middleware: tracing, CORS, compression, timeouts, request IDs, sensitive-header logging. These are battle-tested and composable.

### Database: sqlx

```rust
use sqlx::postgres::PgPoolOptions;

#[derive(Clone)]
struct AppState { db: PgPool }

#[tokio::main]
async fn main() {
    let db = PgPoolOptions::new()
        .max_connections(5)
        .connect("postgres://user:pass@localhost/app").await.unwrap();
    let app = Router::new().route("/users", get(list_users)).with_state(AppState { db });
    // ...
}
```

**sqlx** is the standard async SQL crate — it supports **compile-time SQL checking** (`sqlx::query!` macro validates your SQL against the database at build time), [PostgreSQL](/Learn-PostgreSQL-in-One-Post-Complete-Tutorial-Indexes-MVCC-Performance-Quick-Start/)/MySQL/SQLite, and connection pooling. For an ORM on top, **sea-orm** is the async choice.

### Deploy

```dockerfile
# Multi-stage build -> tiny final image
FROM rust:1.80 AS build
WORKDIR /app
COPY . .
RUN cargo build --release

FROM debian:bookworm-slim
COPY --from=build /app/target/release/myapp /usr/local/bin/myapp
EXPOSE 3000
CMD ["myapp"]
```

A release Rust binary is self-contained (no runtime, no dependencies beyond libc) — the final Docker image is tiny and fast to start. Alternatives: **shuttle.rs** (Rust-native deploy, `shuttle deploy`), **fly.io** (edge), or ship the binary directly.

![Rust + Axum Ecosystem](/assets/img/diagrams/rust-axum-tutorial/axum-ecosystem.svg)

| Concern | Crate |
|---|---|
| Database | sqlx (async, compile-time checked), sea-orm, redis |
| Serialization | serde, serde_json, validator |
| Async runtime | tokio, tokio::spawn, async-stream |
| Middleware | tower, tower-http (tracing, CORS, compression, timeout) |
| Deploy | Docker (tiny image), shuttle.rs, fly.io |

---

## Quick-Start Checklist

1. **`cargo new myapi && cd myapi`** — add `axum`, `tokio` (full), `serde`, `serde_json` to `Cargo.toml`.
2. **Write the 8-line hello-world server** — `Router::new().route("/", get(...))` + `axum::serve`.
3. **Add a `/users/:id` route** with a `Path<u64>` extractor.
4. **Add `State<AppState>`** — inject a shared struct via `.with_state(...)`.
5. **Return `Json<T>`** from a handler — derive `Serialize` on your response type.
6. **Accept `Json<T>`** in a POST handler — derive `Deserialize`.
7. **Write a custom error type** — `enum AppError` + `impl IntoResponse`.
8. **Add `from_fn` middleware** — a logging or auth layer.
9. **Add `tower-http`** — `TraceLayer` + `CompressionLayer` + `TimeoutLayer`.
10. **Connect sqlx** — pool, `query_as`, compile-time checked SQL.

## Common Pitfalls

- **Missing `#[tokio::main]`** — async `main` needs the macro + tokio features; without it, a cryptic error.
- **Body extractors not last** — `Json`/`Form` consume the body; put them after `Path`/`Query`/`State`.
- **`State` not `Clone`** — it's cloned per request; wrap expensive fields in `Arc`.
- **Forgetting `.with_state(...)`** — a handler using `State<T>` without the state injected fails to compile (a feature, not a bug).
- **Leaking errors in `IntoResponse`** — return sanitized messages; log the full error server-side.
- **Middleware order** — `.layer(a).layer(b)` means `b` runs first (outermost). Think "onion."
- **Blocking in async** — `std::fs::read`, `thread::sleep`, or sync DB calls block the runtime thread. Use `tokio::fs`, async crates, or `tokio::task::spawn_blocking`.
- **Unbounded channels / `Mutex`** — use `tokio::sync::Mutex` (not `std::sync::Mutex`) across `.await`; use bounded channels to apply backpressure.
- **Debug build in production** — `cargo run` is unoptimized; ship `cargo build --release`.

## Further Reading

- [Axum Docs](https://docs.rs/axum) — the official API reference
- [Tokio Tutorial](https://tokio.rs/tokio/tutorial) — the async runtime Axum runs on
- [Tower Docs](https://docs.rs/tower) — the middleware abstraction
- [sqlx Docs](https://docs.rs/sqlx) — async SQL + compile-time checking
- [Zero To Production In Rust](https://www.zero2prod.com/) by Luca Palmieri — the canonical Axum backend book

## Related guides

Axum is the Rust backend — these PyShine tutorials connect to it:

- **[Learn Rust in One Post](/Learn-Rust-in-One-Post-Complete-Tutorial-Ownership-Borrow-Async-Quick-Start/)** — ownership, lifetimes, async, traits — the prerequisites.
- **[Learn FastAPI in One Post](/Learn-FastAPI-in-One-Post-Complete-Tutorial-Pydantic-Async-Dependency-Injection-Quick-Start/)** — the Python equivalent; extractors ≈ Depends.
- **[Learn Node.js + Express in One Post](/Learn-Node-js-Express-in-One-Post-Complete-Tutorial-Event-Loop-Middleware-Quick-Start/)** — the JS equivalent; Tower middleware ≈ Express middleware.
- **[Learn PostgreSQL in One Post](/Learn-PostgreSQL-in-One-Post-Complete-Tutorial-Indexes-MVCC-Performance-Quick-Start/)** — what sqlx talks to.
- **[Learn Docker in One Post](/Learn-Docker-in-One-Post-Complete-Tutorial-Dockerfile-Volumes-Compose-Quick-Start/)** — the multi-stage Dockerfile above builds a tiny image.

---

Axum's pitch is a **thin, type-safe routing layer on the Tower/Hyper/Tokio stack** — compile-time-checked routing, extractors that parse and validate at the type level, and middleware that composes via Tower. The five stages here — setup, routing, extractors, handlers, middleware + deploy — cover everything from a hello-world server to a SQL-backed, middleware-wrapped, Dockerized API. The two habits that pay off: **let the type system do the work** (extractors + `Result<T, AppError>` catch errors at compile time), and **never block the async runtime** (use async crates or `spawn_blocking`). `cargo new` an Axum app, add a route with a `Path` + `Json` extractor, and watch the compiler enforce your request schema — once you've felt that safety, there's no going back.