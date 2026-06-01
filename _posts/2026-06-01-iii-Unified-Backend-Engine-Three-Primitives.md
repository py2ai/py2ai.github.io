---
layout: post
title: "iii: The Unified Backend Engine -- Three Primitives, Zero Integration Cost"
description: "iii is a Rust-based backend engine that replaces your API framework, task queue, cron scheduler, event bus, state store, WebSocket server, and observability pipeline with a single engine and three primitives: Worker, Function, and Trigger -- supporting Rust, TypeScript, and Python with built-in real-time observability."
date: 2026-06-01
header-img: "img/post-bg.jpg"
permalink: /iii-Unified-Backend-Engine-Three-Primitives/
featured-img: ai-coding-frameworks/ai-coding-frameworks
categories: [Developer Tools, Open Source, Rust]
tags: [iii, backend framework, Rust, worker function trigger, unified backend, event-driven architecture, real-time observability, task queue, cron scheduler, state management]
keywords: "iii backend framework tutorial, Rust unified backend engine, worker function trigger primitives, replace API framework task queue cron, iii engine installation guide, event-driven backend architecture, real-time observability Rust, multi-language backend SDK Rust TypeScript Python, iii vs traditional backend, zero integration cost backend framework"
author: "PyShine"
---

## Introduction

Today, building a typical backend means connecting many separate systems: an API framework for HTTP endpoints, a task queue for async processing, a cron scheduler for periodic jobs, an event bus for pub/sub messaging, a state store for shared data, a WebSocket server for real-time communication, and an observability pipeline for monitoring. Each of these systems comes with its own SDK, configuration, error handling, and integration overhead. The iii backend framework from [iii-hq/iii](https://github.com/iii-hq/iii) (17,395 GitHub stars) collapses all of that into one engine with just three primitives: Worker, Function, and Trigger -- achieving zero integration cost across your entire backend stack.

> **Key Insight:** Today, building a typical backend means connecting many separate systems: API frameworks, a task queue, a cron scheduler, an event bus, a pub/sub layer, a state store, a WebSocket server, and an observability pipeline. iii collapses that into one live system surface with just three primitives: Worker, Function, and Trigger.

iii is a Rust-based unified backend engine that replaces seven separate backend systems with a single coherent architecture. Written in Rust for performance, it provides SDKs in Rust, TypeScript, and Python, allowing you to write workers in whatever language fits the task. A TypeScript API service is a worker. A Python data pipeline is a worker. A Rust microservice is a worker. All communicate through the same engine using the same three primitives, eliminating the integration overhead that plagues traditional multi-tool backend stacks.

![Three Primitives Architecture](/assets/img/diagrams/iii/iii-architecture.svg)

## The Three Primitives

The entire iii system is built on three concepts: **Worker**, **Function**, and **Trigger**. Together, they form a complete mental model for building any backend service.

### Worker

A Worker is a process that registers with the iii engine. It represents a unit of deployment -- a Rust microservice, a TypeScript API service, or a Python data pipeline. Workers initialize by calling `init()`, which establishes a connection to the engine. Once registered, a worker can register functions and triggers. Workers can also create other workers at runtime, enabling dynamic scaling and composition patterns not possible with traditional static backend architectures.

### Function

A Function is a unit of business logic registered by a worker. Functions are the code that actually runs -- processing orders, transforming data, sending notifications, or any other task. The same function can be invoked from many trigger types at once: an HTTP request, a queue message, or a scheduled event can all call the same function without any additional wiring.

### Trigger

A Trigger is an event source that connects functions to the outside world or to other workers. Triggers fire when specific events occur -- an HTTP request arrives, a scheduled time is reached, a message appears on a queue, or another worker sends a signal. When a trigger fires, it invokes the bound function, passing along the event data as input.

> **Amazing:** Worker + Function + Trigger is the entire mental model. A TypeScript API service is a worker. A Python data pipeline is a worker. A Rust microservice is a worker. Any functionality can be transformed into a worker with a few lines of code -- and workers can even create other workers at runtime.

Here is how a Rust worker registers a function and binds it to an HTTP trigger:

```rust
use iii_sdk::prelude::*;

#[iii::main]
async fn main() {
    // Initialize worker and register with the iii engine
    let worker = iii::init("my-rust-worker").await?;
    
    // Register a function
    worker.register_function("greet", |ctx| async move {
        let name = ctx.input::<String>()?;
        Ok(format!("Hello, {}!", name))
    }).await?;
    
    // Register an HTTP trigger for the function
    worker.register_trigger("greet", Trigger::Http {
        path: "/greet".into(),
        method: Method::GET,
    }).await?;
    
    // Keep the worker running
    worker.run().await?;
    Ok(())
}
```

Workers can also trigger functions on other workers, enabling inter-service communication without message brokers:

```rust
// Trigger a function on another worker from within a function
worker.register_function("place_order", |ctx| async move {
    let order = ctx.input::<Order>()?;
    
    // Process the order
    let result = process_order(&order)?;
    
    // Trigger the notification worker
    ctx.trigger("notification-worker", "send_email", EmailPayload {
        to: order.customer_email,
        subject: "Order Confirmed",
        body: format!("Your order #{} has been processed.", order.id),
    }).await?;
    
    Ok(result)
}).await?;
```

And workers can even spawn new workers dynamically at runtime:

```rust
// Spawn a new worker at runtime
let child_worker = worker.spawn("dynamic-processor").await?;
child_worker.register_function("process", |ctx| async move {
    // Dynamic processing logic
    Ok("processed")
}).await?;
child_worker.run().await?;
```

![Worker Lifecycle and Trigger Flow](/assets/img/diagrams/iii/iii-workflow.svg)

## Built-in Service Workers

iii provides six built-in service workers that replace the traditional backend stack. Each service worker is a first-class component of the engine, using the same Worker-Function-Trigger model as your custom workers. This means there is zero integration cost between them -- a function registered with iii-http can be triggered by iii-cron or iii-queue without any additional wiring.

| Service Worker | Replaces | Purpose |
|---------------|----------|---------|
| **iii-http** | Express / FastAPI | API server for HTTP endpoints |
| **iii-cron** | cron / Airflow | Scheduled task execution |
| **iii-queue** | Celery / Bull | Task queue for async processing |
| **iii-state** | Redis / etcd | State management and shared data |
| **iii-stream** | Socket.io / Pusher | Real-time data streams and WebSocket |
| **iii-observability** | OpenTelemetry / Datadog | Logging, tracing, and visual debugging |

> **Takeaway:** iii replaces seven separate backend systems with a single engine. No more wiring up Express for APIs, Celery for queues, cron for scheduling, Redis for state, Socket.io for real-time, and OpenTelemetry for observability. Every capability is built in, connected by the same three primitives.

Setting up a cron trigger is as simple as registering it with the worker:

```rust
// Register a cron trigger for scheduled execution
worker.register_trigger("cleanup", Trigger::Cron {
    schedule: "0 2 * * *".into(), // Run at 2 AM daily
}).await?;
```

Queue triggers enable async processing with configurable concurrency:

```typescript
// Register a queue trigger for async processing
registerTrigger("processPayment", {
  type: "queue",
  source: "payment-queue",
  concurrency: 10, // Process up to 10 items concurrently
});
```

State management is built directly into the worker API:

```rust
// Use built-in state management
let count: i64 = worker.state().get("visit_count").await?;
worker.state().set("visit_count", count + 1).await?;
```

![Built-in Service Workers](/assets/img/diagrams/iii/iii-features.svg)

## Installation and Getting Started

Getting started with iii is straightforward. The engine is installed via Cargo, Rust's package manager:

```bash
# Install iii using Cargo
cargo install iii

# Verify installation
iii --version
```

Once installed, you can start the iii engine and begin creating workers. The engine acts as the central coordinator, managing worker registration, function invocation, and trigger dispatch. Each worker connects to the engine on startup and registers its capabilities.

## Multi-Language SDK

One of iii's most powerful features is its multi-language SDK support. Workers can be written in Rust for high-performance microservices, TypeScript for API services, or Python for data pipelines. All SDKs provide the same core API: `init()`, `register_function()`, `register_trigger()`, and `trigger()`.

### TypeScript SDK

The TypeScript SDK is ideal for building API services and web backends:

```typescript
import { init, registerFunction, registerTrigger } from "@iii/sdk";

// Initialize worker and connect to the iii engine
const worker = await init("my-api-service");

// Register a function
registerFunction("processOrder", async (ctx) => {
  const { orderId, items } = ctx.input;
  
  // Process the order
  const total = items.reduce((sum, item) => sum + item.price, 0);
  
  return { orderId, total, status: "processed" };
});

// Bind to HTTP trigger
registerTrigger("processOrder", {
  type: "http",
  path: "/orders",
  method: "POST",
});

// Keep the worker running
worker.run();
```

### Python SDK

The Python SDK is designed for data pipelines and ML workflows:

```python
import asyncio
from iii_sdk import init, register_function, register_trigger

async def main():
    # Initialize worker and connect to the iii engine
    worker = await init("my-data-pipeline")
    
    # Register a function
    @register_function("transform_data")
    async def transform_data(ctx):
        data = ctx.input
        # Transform the data
        result = {
            "processed": True,
            "count": len(data["records"]),
            "timestamp": data["timestamp"],
        }
        return result
    
    # Bind to queue trigger
    await register_trigger("transform_data", {
        "type": "queue",
        "source": "data-ingestion",
    })
    
    # Keep the worker running
    await worker.run()

asyncio.run(main())
```

The mixed-language support means you can have a Rust worker handling performance-critical operations, a TypeScript worker serving API requests, and a Python worker processing data -- all running on the same iii engine, communicating through the same primitives, with zero integration overhead.

## Real-Time Observability

Unlike traditional backends where observability is bolted on as an afterthought with separate tools, iii provides built-in structured logging, end-to-end link tracking, and a visual debugger as first-class features of the engine itself. The engine maintains a live system surface that shows every worker, every trigger, and every function execution in real time.

> **Important:** Unlike traditional backends where observability is an afterthought bolted on with separate tools, iii provides structured logging, end-to-end link tracking, and a visual debugger as first-class features of the engine itself. You see every service, every trigger, and every function execution in real time on a live system surface.

The observability system is configured through the engine's configuration file:

```toml
# iii.toml - Engine configuration with observability
[engine]
address = "0.0.0.0:8080"

[observability]
logging = true
tracing = true
dashboard = true
dashboard_port = 9090
```

When enabled, the dashboard provides a real-time visual representation of your entire backend -- all workers, their connections, data flow, and health status. You can trace a request from its initial trigger through every function execution to its final result, making debugging and performance optimization significantly easier than with traditional distributed tracing tools.

## Conclusion

iii represents a fundamental shift in how we think about backend architecture. Instead of assembling seven or more separate systems and spending significant effort on integration, iii provides a single engine with three primitives that covers every backend need:

| Traditional Stack | iii |
|-------------------|-----|
| Express/FastAPI + Celery + cron + Redis + Socket.io + OpenTelemetry | iii Engine |
| Multiple SDKs, configs, error handlers | Worker + Function + Trigger |
| Integration overhead between systems | Zero integration cost |
| Separate monitoring tools | Built-in real-time observability |
| Static service composition | Dynamic worker creation at runtime |

With 17,395 GitHub stars and an Apache-2.0 license, iii is production-ready and actively maintained. Whether you are building a new project from scratch or looking to simplify an existing backend, the Worker-Function-Trigger model offers a cleaner, more maintainable alternative to the traditional multi-tool approach.

## Links

- [iii GitHub Repository](https://github.com/iii-hq/iii) -- Source code and documentation
- [iii Documentation](https://iii.dev) -- Official documentation and guides
- [iii Rust SDK](https://iii.dev/docs/sdk-reference/rust-sdk) -- Rust SDK API reference
- [iii Functions](https://iii.dev/docs/using-iii/functions) -- Functions documentation
- [iii Workers SDK](https://github.com/iii-hq/workers) -- TypeScript worker SDK
- [iii Examples](https://github.com/iii-hq/examples) -- Example projects
- [Apache-2.0 License](https://opensource.org/licenses/Apache-2.0) -- License text