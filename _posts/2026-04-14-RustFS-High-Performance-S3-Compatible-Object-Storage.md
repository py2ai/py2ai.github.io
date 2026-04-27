---
layout: post
title: "RustFS: High-Performance S3-Compatible Object Storage"
description: "Discover RustFS, a high-performance S3-compatible object storage system written in Rust that delivers 2.3x faster performance than MinIO with memory safety and Apache 2.0 licensing."
date: 2026-04-14
header-img: "img/post-bg.jpg"
permalink: /RustFS-High-Performance-S3-Compatible-Object-Storage/
featured-img: ai-coding-frameworks/ai-coding-frameworks
tags:
  - Rust
  - Object Storage
  - S3
  - Open Source
  - Performance
  - Distributed Systems
author: "PyShine"
---

# RustFS: High-Performance S3-Compatible Object Storage

In the world of cloud-native applications and data-intensive workloads, object storage has become the backbone of modern infrastructure. From data lakes to AI/ML pipelines, from backup systems to content delivery networks, the need for reliable, high-performance object storage has never been greater. Enter **RustFS** - a revolutionary open-source object storage system written in Rust that combines the simplicity of MinIO with the raw performance and memory safety that only Rust can provide.

## What is RustFS?

RustFS is a high-performance, distributed object storage system built entirely in Rust - one of the most loved programming languages worldwide. It offers full S3 API compatibility, making it a drop-in replacement for applications that already use Amazon S3 or MinIO. What sets RustFS apart is its exceptional performance: benchmark tests show it delivers **2.3x faster throughput** compared to MinIO, while maintaining the same level of functionality and compatibility.

The project is released under the permissive **Apache 2.0 license**, which means organizations can use, modify, and distribute it freely without the restrictions found in AGPL-licensed alternatives like MinIO. This makes RustFS an ideal choice for enterprises that need a business-friendly open-source object storage solution.

### Key Highlights

- **High Performance**: Built with Rust for maximum speed and resource efficiency
- **S3 Compatible**: 100% compatible with Amazon S3 API
- **Distributed Architecture**: Scalable and fault-tolerant design
- **Memory Safe**: Rust's ownership model prevents common memory bugs
- **Apache 2.0 Licensed**: Business-friendly licensing without AGPL restrictions
- **Multi-Protocol Support**: S3 API, OpenStack Swift API, and Keystone authentication

![RustFS Architecture](/assets/img/diagrams/rustfs-architecture.svg)

### Understanding the Architecture

The architecture diagram above illustrates the layered design of RustFS, showing how requests flow from clients through various processing stages to storage backends. Let's examine each component in detail:

**Client Layer**

The client layer consists of two primary interfaces: S3 clients using AWS SDKs and the web-based admin console. S3 clients interact with the storage system through standard S3 API calls on port 9000, while administrators access the web console on port 9001 for management operations. This dual-interface design ensures that both programmatic access and human administration are well-supported.

**API Layer**

The API layer handles incoming requests through two endpoints: the S3 API (port 9000) for object operations and the Admin API (/minio/ prefix) for cluster management, IAM configuration, and metrics collection. Both APIs share the same underlying infrastructure but serve different purposes - the S3 API focuses on data operations while the Admin API handles control plane operations.

**Server Layer**

The server layer is responsible for TLS termination, authentication, request routing, and middleware processing. This layer ensures secure connections through TLS, validates S3 signatures for authentication, and routes requests to appropriate handlers. The middleware stack includes compression, CORS handling, and graceful shutdown capabilities.

**App Layer (Use Cases)**

The application layer orchestrates use cases for object operations, bucket management, and multipart uploads. This is where business logic resides - validating requests, enforcing policies, and managing object lifecycles. The layer acts as an orchestration point between the HTTP interface and the storage engine.

**Storage Layer**

The storage layer translates S3 API calls into storage operations. It handles erasure-coded filesystem operations (ECFS), server-side encryption (SSE), RPC communication for distributed mode, and access control lists (ACL). This layer abstracts the complexity of data storage from the upper layers.

**Core Components**

The core components form the heart of RustFS's performance:

1. **ECFS (Erasure-Coded Filesystem)**: Implements Reed-Solomon erasure coding for data durability. Data is split into shards with parity information, allowing recovery from disk failures without data loss.

2. **Rio Pipeline**: A composable reader chain that processes data through encryption, compression, and hashing stages. This pipeline architecture allows flexible data transformation while maintaining high throughput.

3. **IO Core**: Provides zero-copy I/O operations with buffer pooling and direct I/O capabilities. This minimizes memory copies and context switches, contributing significantly to RustFS's performance advantage.

**Storage Backend**

The storage backend consists of local disks with direct I/O capabilities and remote disks accessed via gRPC RPC. This hybrid approach allows RustFS to operate in both single-node and distributed configurations, with automatic data distribution across available storage resources.

![RustFS Features Overview](/assets/img/diagrams/rustfs-features.svg)

### Feature Breakdown

The features diagram presents RustFS's capabilities organized into five major categories. Each category addresses specific enterprise requirements:

**S3 Compatible API**

RustFS provides 100% S3 API compatibility, ensuring seamless integration with existing applications. Beyond basic S3 operations, it supports the OpenStack Swift API for organizations with Swift-based workloads. Object versioning allows tracking changes and recovering previous versions, making it suitable for compliance requirements and data protection policies.

**Distributed Architecture**

The distributed architecture is built on three pillars: erasure coding for data durability, replication for geographic distribution, and multi-tenancy for secure workload isolation. Erasure coding uses Reed-Solomon algorithms to split data into shards with parity, allowing recovery from multiple disk failures. Replication enables cross-region data synchronization for disaster recovery. Multi-tenancy ensures that different teams or applications can share the same cluster while maintaining isolation.

**High Performance**

Performance is where RustFS truly shines. Zero-copy I/O eliminates unnecessary memory copies, reducing CPU overhead and latency. Benchmark tests demonstrate 2.3x throughput improvement over MinIO under identical conditions. Rust's memory safety guarantees prevent common bugs like buffer overflows, use-after-free errors, and data races - all without garbage collection pauses that can affect Go-based alternatives.

**Security Features**

Security is implemented at multiple levels: TLS/SSL for encrypted communications, IAM/STS for identity and access management, and encryption for data at rest. The IAM system supports fine-grained access policies, temporary credentials via STS, and integration with external identity providers. Server-side encryption (SSE) protects data at rest using industry-standard algorithms.

**Deployment Options**

RustFS offers flexible deployment options to match various infrastructure needs: Docker containers for quick deployment, Kubernetes Helm charts for cloud-native environments, and standalone binaries for traditional server installations. This flexibility ensures that organizations can deploy RustFS in their preferred environment without vendor lock-in.

## Performance Comparison: RustFS vs MinIO

![Performance Comparison](/assets/img/diagrams/rustfs-performance.svg)

### Understanding the Performance Advantage

The performance comparison diagram illustrates how RustFS outperforms MinIO across key metrics. Let's analyze each dimension:

**Throughput**

RustFS achieves significantly higher throughput than MinIO, measured in operations per second for both read and write operations. This improvement stems from Rust's zero-cost abstractions and the absence of garbage collection pauses. In Go-based MinIO, garbage collection can introduce unpredictable latency spikes, especially under high memory pressure. RustFS's ownership model eliminates this concern entirely.

**Latency**

Lower latency is critical for real-time applications and interactive workloads. RustFS's latency advantage comes from several factors: zero-copy I/O reduces memory operations, efficient buffer pooling minimizes allocations, and direct I/O bypasses kernel page caches when appropriate. These optimizations compound to deliver sub-millisecond latency for common operations.

**IOPS (Input/Output Operations Per Second)**

The benchmark environment used Intel Xeon Platinum 8475B processors with 4GB memory and drives rated at 3800 IOPS each. RustFS achieved over 3800 IOPS per drive, effectively saturating the hardware's capabilities. This efficiency comes from the IO Core's scheduling algorithms and backpressure mechanisms that prevent resource exhaustion.

**Memory Efficiency**

Memory efficiency is where Rust's advantages become most apparent. Without a garbage collector, RustFS has predictable memory behavior and no stop-the-world pauses. The buffer pool system reuses memory across operations, reducing allocation overhead. In contrast, Go's garbage collector can introduce latency spikes during collection cycles, particularly problematic for latency-sensitive applications.

### Benchmark Environment Details

The stress test environment used for performance comparison:

| Type | Parameter | Remark |
|------|-----------|--------|
| CPU | 2 Core | Intel Xeon (Sapphire Rapids) Platinum 8475B, 2.7/3.2 GHz |
| Memory | 4GB | |
| Network | 15Gbps | |
| Drive | 40GB x 4 | IOPS 3800 / Drive |

## Data Flow: How PUT Requests Work

![Data Flow Diagram](/assets/img/diagrams/rustfs-dataflow.svg)

### Understanding the PUT Request Pipeline

The data flow diagram shows how a PUT object request travels through RustFS's processing pipeline. This pipeline architecture ensures data integrity, security, and optimal storage efficiency:

**HTTP Request Reception**

The journey begins when an HTTP PUT request arrives at the server. The request contains the object data along with metadata headers. RustFS supports both standard HTTP and HTTPS connections, with TLS termination handled efficiently at the server layer.

**TLS Termination**

For HTTPS connections, TLS termination decrypts the incoming request. RustFS's TLS implementation is optimized for performance, supporting modern cipher suites and TLS 1.3. The termination process adds minimal overhead thanks to Rust's efficient cryptographic primitives.

**Authentication (S3 Signature Verification)**

Every S3 request must be authenticated using AWS Signature Version 4. RustFS validates the signature by recomputing it with the secret key and comparing it to the provided signature. This ensures that only authorized users can access the storage system. The authentication module also handles temporary credentials via STS for fine-grained access control.

**Request Routing**

The routing layer determines which handler should process the request. For PUT operations, this involves identifying the target bucket, checking bucket policies, and routing to the object use case handler. The router also handles request validation, ensuring required headers are present and properly formatted.

**Validation and Policy Check**

Before processing, the request undergoes validation: checking bucket existence, verifying write permissions, and enforcing bucket policies. IAM policies can restrict operations based on user identity, resource patterns, and conditions. This layer ensures that only authorized operations proceed to the storage layer.

**Lifecycle Management**

Object lifecycle management applies rules defined for the bucket. This includes automatic transitions between storage tiers, expiration rules, and non-current version handling. While the actual lifecycle transitions happen asynchronously, the rules are evaluated during the PUT operation to determine initial storage class and metadata.

**Erasure Coding (Reed-Solomon)**

Erasure coding is the heart of RustFS's data durability. Data is split into K data shards and M parity shards using Reed-Solomon coding. The system can tolerate up to M simultaneous disk failures without data loss. For example, with EC:4 (4 data + 2 parity), the system can survive 2 disk failures. This approach provides better storage efficiency than traditional replication while maintaining high durability.

**Encryption (Server-Side Encryption)**

If server-side encryption (SSE) is enabled, the data is encrypted before being written to disk. RustFS supports multiple encryption modes: SSE-S3 (server-managed keys), SSE-C (customer-provided keys), and SSE-KMS (key management service integration). Encryption is performed after erasure coding to minimize the amount of data that needs to be encrypted.

**Checksum Computation**

A checksum (typically SHA-256 or MD5) is computed over the data for integrity verification. This checksum is stored alongside the object and verified during read operations. Bitrot protection uses these checksums to detect silent data corruption, with automatic healing when corruption is detected.

**Compression (Optional)**

For objects that benefit from compression, RustFS can apply lossless compression algorithms before writing. Compression is applied after encryption to ensure security (encrypted data doesn't compress well). The compression algorithm can be configured per-bucket or globally.

**Zero-Copy Write**

The final stage uses zero-copy I/O to write data to disk. Zero-copy techniques minimize memory copies between kernel and user space, reducing CPU overhead and improving throughput. Direct I/O bypasses the page cache for large objects, preventing cache pollution and reducing memory pressure.

**Disk Pool Storage**

Data is distributed across disk pools for load balancing and parallelism. Each pool consists of multiple disks, with erasure coding applied within the pool. For distributed deployments, data may be written to remote disks via gRPC RPC, enabling geographic distribution and disaster recovery.

## Installation

RustFS offers multiple installation methods to suit different environments and use cases.

### Option 1: One-Click Installation

The simplest way to get started:

```bash
curl -O https://rustfs.com/install_rustfs.sh && bash install_rustfs.sh
```

This script downloads and installs RustFS with sensible defaults for single-node deployments.

### Option 2: Docker Quick Start

For containerized deployments:

```bash
# Create data and logs directories
mkdir -p data logs

# Change the owner of these directories
chown -R 10001:10001 data logs

# Using latest version
docker run -d -p 9000:9000 -p 9001:9001 \
  -v $(pwd)/data:/data \
  -v $(pwd)/logs:/logs \
  rustfs/rustfs:latest
```

**Important**: The RustFS container runs as non-root user `rustfs` (UID `10001`). Ensure host directories have the correct ownership.

### Option 3: Docker Compose

For production deployments with observability:

```bash
docker compose --profile observability up -d
```

The docker-compose.yml includes optional services for Grafana, Prometheus, and Jaeger for monitoring and tracing.

### Option 4: Kubernetes with Helm

For cloud-native deployments:

```bash
# Add the RustFS Helm repository
helm repo add rustfs https://charts.rustfs.com

# Install RustFS
helm install rustfs rustfs/rustfs
```

### Option 5: Build from Source

For developers who want to customize RustFS:

```bash
# Clone the repository
git clone https://github.com/rustfs/rustfs.git
cd rustfs

# Build
cargo build --release

# Run
./target/release/rustfs --help
```

## Usage

### Accessing the Console

After starting RustFS, access the web console at `http://localhost:9001`:

- **Default credentials**: `rustfsadmin` / `rustfsadmin`
- Create buckets, upload objects, and manage access policies through the intuitive UI

### Using AWS CLI

Configure AWS CLI to work with RustFS:

```bash
# Configure AWS CLI
aws configure --profile rustfs
# AWS Access Key ID: rustfsadmin
# AWS Secret Access Key: rustfsadmin
# Default region: us-east-1

# Create a bucket
aws --profile rustfs --endpoint-url http://localhost:9000 s3 mb s3://my-bucket

# Upload an object
aws --profile rustfs --endpoint-url http://localhost:9000 s3 cp myfile.txt s3://my-bucket/

# List objects
aws --profile rustfs --endpoint-url http://localhost:9000 s3 ls s3://my-bucket/
```

### Using SDKs

RustFS works with any S3-compatible SDK:

**Python (boto3)**:
```python
import boto3

s3 = boto3.client('s3',
    endpoint_url='http://localhost:9000',
    aws_access_key_id='rustfsadmin',
    aws_secret_access_key='rustfsadmin'
)

# Upload
s3.upload_file('local_file.txt', 'my-bucket', 'remote_file.txt')

# Download
s3.download_file('my-bucket', 'remote_file.txt', 'downloaded_file.txt')
```

**Node.js (AWS SDK v3)**:
```javascript
const { S3Client, PutObjectCommand } = require('@aws-sdk/client-s3');

const client = new S3Client({
    endpoint: 'http://localhost:9000',
    credentials: {
        accessKeyId: 'rustfsadmin',
        secretAccessKey: 'rustfsadmin'
    }
});

await client.send(new PutObjectCommand({
    Bucket: 'my-bucket',
    Key: 'file.txt',
    Body: 'Hello, RustFS!'
}));
```

## Features Comparison

| Feature | RustFS | MinIO | Notes |
|---------|--------|-------|-------|
| **Language** | Rust | Go | Rust provides memory safety without GC |
| **License** | Apache 2.0 | AGPL v3 | Apache 2.0 is business-friendly |
| **S3 API** | 100% Compatible | 100% Compatible | Both are S3-compatible |
| **Swift API** | Yes | No | RustFS supports OpenStack Swift |
| **Performance** | 2.3x faster | Baseline | Benchmark results |
| **Memory Safety** | Guaranteed | GC-dependent | No GC pauses in RustFS |
| **Telemetry** | None | Yes | RustFS respects data sovereignty |
| **Edge Support** | Strong | Limited | RustFS is optimized for edge |

## Troubleshooting

### Common Issues and Solutions

**Permission Denied Errors**

If you encounter permission denied errors when running Docker:
```bash
# Ensure directories are owned by UID 10001
chown -R 10001:10001 data logs
```

**Port Already in Use**

If ports 9000 or 9001 are already in use:
```bash
# Use different ports
docker run -d -p 9002:9000 -p 9003:9001 \
  -v $(pwd)/data:/data \
  -v $(pwd)/logs:/logs \
  rustfs/rustfs:latest
```

**Connection Refused**

If clients cannot connect:
1. Check firewall rules
2. Verify RustFS is running: `docker ps`
3. Check logs: `docker logs <container_id>`

**Slow Performance**

For performance issues:
1. Ensure sufficient disk IOPS
2. Check network bandwidth
3. Review buffer pool configuration
4. Consider direct I/O for large objects

## Why Choose RustFS?

### Advantages Over Alternatives

**Memory Safety Without Garbage Collection**

Rust's ownership model provides memory safety guarantees at compile time, eliminating entire classes of bugs (buffer overflows, use-after-free, data races) without runtime garbage collection. This results in predictable latency and no stop-the-world pauses.

**Business-Friendly Licensing**

The Apache 2.0 license allows unrestricted commercial use, modification, and distribution. Unlike AGPL-licensed alternatives, you don't need to open-source your modifications when using RustFS in your products.

**Data Sovereignty**

RustFS includes no telemetry or data collection features. Your data stays within your infrastructure, ensuring compliance with GDPR, CCPA, and other data protection regulations.

**Edge and IoT Ready**

RustFS's efficient resource usage makes it ideal for edge deployments where hardware resources are limited. The absence of garbage collection ensures consistent performance even on constrained devices.

**Active Development**

With a growing community and regular releases, RustFS is actively developed and maintained. The project welcomes contributions and has clear contribution guidelines.

## Conclusion

RustFS represents a significant advancement in open-source object storage. By combining Rust's performance and safety guarantees with full S3 compatibility, it offers a compelling alternative to existing solutions. Whether you're building a data lake, running AI/ML workloads, or need reliable backup storage, RustFS provides the performance, reliability, and flexibility modern applications require.

The 2.3x performance improvement over MinIO, combined with Apache 2.0 licensing and memory safety guarantees, makes RustFS an excellent choice for organizations seeking a high-performance, business-friendly object storage solution.

## Links

- [RustFS Documentation](https://docs.rustfs.com)
- [GitHub Repository](https://github.com/rustfs/rustfs)
- [Helm Charts](https://charts.rustfs.com)
- [GitHub Discussions](https://github.com/rustfs/rustfs/discussions)
- [Changelog](https://github.com/rustfs/rustfs/releases)
