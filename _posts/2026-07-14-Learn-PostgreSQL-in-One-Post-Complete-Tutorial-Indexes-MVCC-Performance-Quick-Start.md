---
layout: post
title: "Learn PostgreSQL in a Single Post: A Complete Tutorial From Tables and Indexes to MVCC, Performance, and Operations"
description: "A complete PostgreSQL tutorial in one blog post. Covers the whole database in 5 stages: basics (psql, tables, types, CRUD, joins), schema (constraints, indexes, relationships, normalization), performance (EXPLAIN, B-tree, query tuning, connection pooling), advanced features (JSONB, window functions, CTEs, full-text, extensions), and operations (MVCC, isolation levels, VACUUM, backups, replication, monitoring). Five hand-drawn diagrams, runnable SQL, and a quick-start roadmap."
date: 2026-07-14
header-img: "img/post-bg.jpg"
permalink: /Learn-PostgreSQL-in-One-Post-Complete-Tutorial-Indexes-MVCC-Performance-Quick-Start/
featured-img: ai-coding-frameworks/ai-coding-frameworks
tags:
  - PostgreSQL
  - Postgres
  - Database
  - SQL
  - MVCC
  - Performance
  - Tutorial
categories: [Tutorial, Databases, Backend]
keywords: "PostgreSQL tutorial one post, learn Postgres fast, psql client, PostgreSQL data types JSONB, PostgreSQL indexes B-tree GIN GiST BRIN, EXPLAIN ANALYZE query tuning, PostgreSQL MVCC multi version concurrency, isolation levels serializable, VACUUM autovacuum bloat, PostgreSQL replication streaming replica, pgbouncer connection pool, pg_dump backup, PostgreSQL quick start roadmap"
author: "PyShine"
---

# Learn PostgreSQL in a Single Post: Complete Tutorial From Tables and Indexes to MVCC and Operations

PostgreSQL (Postgres) is the database most engineers reach for when they need a real relational database: ACID, rich types, JSON, full-text search, geospatial, and an extension system that lets it do things no other database can. It's free, fast, and battle-tested at scale. This single post teaches the whole database in five stages, with hand-drawn diagrams and runnable SQL.

## Learning Roadmap

![PostgreSQL Learning Roadmap](/assets/img/diagrams/postgresql-tutorial/pg-roadmap.svg)

The roadmap moves from the basics (Stage 1), through schema design (Stage 2), performance (Stage 3), advanced features (Stage 4), and operations (Stage 5). The [SQL tutorial](/Learn-SQL-in-One-Post-Complete-Tutorial-Joins-Window-Functions-Transactions-Quick-Start/) is the prerequisite — this post goes Postgres-specific.

---

## Stage 1 — Basics

### Connect

```bash
psql -h localhost -U postgres -d mydb
# inside psql:
\dt              # list tables
\d users         # describe a table (columns, types, indexes)
\q               # quit
```

`psql` is the CLI; learn its backslash commands (`\d`, `\dt`, `\dx` for extensions, `\l` for databases). It's faster than any GUI for day-to-day work.

### Types

Postgres has a richer type system than most databases:

| Category | Types |
|---|---|
| Numeric | `int`, `bigint`, `numeric` (exact decimal), `serial`/`identity` (auto-increment) |
| Text | `text` (unbounded), `varchar(n)`, `char(n)` |
| Time | `date`, `time`, `timestamp`, `timestamptz` (UTC-aware — prefer this) |
| Boolean | `boolean` |
| JSON | `jsonb` (binary, indexable — always prefer over `json`) |
| Arrays | `int[]`, `text[]` |
| UUID | `uuid` (with `gen_random_uuid()`) |
| Network | `inet`, `cidr` (IP addresses) |

### CRUD + joins (the standard SQL you know)

```sql
CREATE TABLE users (
  id          bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  email       text NOT NULL UNIQUE,
  created_at  timestamptz NOT NULL DEFAULT now()
);

INSERT INTO users (email) VALUES ('ada@example.com') RETURNING id, created_at;

SELECT u.email, count(o.id) AS orders
FROM users u LEFT JOIN orders o ON o.user_id = u.id
GROUP BY u.email
ORDER BY orders DESC;
```

Postgres SQL is standard; see the [SQL tutorial](/Learn-SQL-in-One-Post-Complete-Tutorial-Joins-Window-Functions-Transactions-Quick-Start/) for joins, grouping, and window functions in depth. The rest of this post is what's *distinctive* about Postgres.

---

## Stage 2 — Schema

### Constraints

```sql
CREATE TABLE orders (
  id        bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  user_id   bigint NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  total     numeric(10,2) NOT NULL CHECK (total >= 0),
  status    text NOT NULL DEFAULT 'pending'
             CHECK (status IN ('pending','paid','shipped','cancelled')),
  created_at timestamptz NOT NULL DEFAULT now()
);
```

`PRIMARY KEY`, `FOREIGN KEY` (with `ON DELETE CASCADE` / `SET NULL`), `UNIQUE`, `NOT NULL`, `CHECK`, `DEFAULT` — use them all. Constraints are the database enforcing your invariants; the more you push into the schema, the less application code you write.

### Relationships and normalization

Normalize to 3NF (each non-key column depends on the whole key and nothing but the key) by default; denormalize deliberately (with a measured reason — usually read performance) when you need to. Foreign keys are cheap in Postgres — don't skip them for "performance" without measuring.

### Indexes (intro)

```sql
CREATE INDEX idx_orders_user_id ON orders(user_id);
CREATE INDEX idx_orders_user_status ON orders(user_id, status);  -- composite
```

Index the columns you `WHERE`/`JOIN`/`ORDER BY` on. Every `FOREIGN KEY` column usually wants an index (joins and `ON DELETE CASCADE` get slow otherwise). More on index types in Stage 3.

> **Pitfall:** Indexes speed reads but slow writes (every `INSERT`/`UPDATE` updates them). Don't index everything; index what you query, and verify with `EXPLAIN`.

---

## Stage 3 — Performance

### `EXPLAIN ANALYZE`

The single most important tool. It shows the **query plan** the planner chose, plus actual timings:

```sql
EXPLAIN ANALYZE
SELECT * FROM orders WHERE user_id = 7 AND status = 'paid';
```

- `Seq Scan` on a big table where you filter = **missing index**.
- `Index Scan` / `Index Only Scan` = the index is working.
- `Hash Join` / `Nested Loop` = join strategy; the planner picks based on row estimates.
- `rows` estimates vs actual = if wildly off, `ANALYZE` (update stats).

### Index types — pick the right one

![Index Types + When to Use](/assets/img/diagrams/postgresql-tutorial/pg-indexes.svg)

| Index | When |
|---|---|
| **B-tree** (default) | equality + range (`=`, `<`, `>`, `BETWEEN`, `ORDER BY`) |
| **GIN** | multiple values per row: arrays, `jsonb`, full-text (`tsvector`), trigrams |
| **GiST** | geometric / custom: PostGIS, ranges, KNN nearest-neighbor |
| **BRIN** | huge, naturally-ordered tables (time-series) — tiny, cheap, block-range |
| **Hash** | equality only (rarely worth it; B-tree usually fine) |
| **Partial** | `CREATE INDEX ... WHERE active` — index a subset, small + targeted |
| **Expression** | `CREATE INDEX ON users (lower(email))` — index a function result |

```sql
-- JSONB GIN index for fast key lookups
CREATE INDEX ON events USING gin (data);

-- partial index: only active users
CREATE INDEX ON orders (created_at) WHERE status = 'paid';

-- expression index for case-insensitive email lookup
CREATE INDEX ON users (lower(email));
SELECT * FROM users WHERE lower(email) = 'ada@example.com';
```

> **Pitfall:** `LIKE '%term%'` (leading wildcard) **can't** use a B-tree. Use a `pg_trgm` GIN index (`CREATE INDEX USING gin (name gin_trgm_ops)`) for substring search, or use full-text search for words.

### Connection pooling

Postgres forks a **backend process per connection** (Stage 5) — thousands of idle connections waste memory and a `max_connections` ceiling. Use **PgBouncer** (or Pgcat) to multiplex many app connections onto a small pool of real DB connections. Typical setup: app → PgBouncer (1000 connections) → Postgres (50 connections).

### Query tuning checklist

1. `EXPLAIN ANALYZE` the slow query.
2. A `Seq Scan` on a big filtered table → add an index.
3. Row estimates way off → `VACUUM ANALYZE` (update stats).
4. `ORDER BY ... LIMIT` slow → an index on the sort column.
5. N+1 from the ORM → eager-load or join.

---

## Stage 4 — Advanced Features

### JSONB

`jsonb` is binary JSON, indexable with GIN, queryable with operators:

```sql
CREATE TABLE events (id bigint PRIMARY KEY, data jsonb NOT NULL);
CREATE INDEX ON events USING gin (data);

INSERT INTO events (data) VALUES ('{"type":"click","user":42,"url":"/home"}');

-- find events with type = click
SELECT * FROM events WHERE data @> '{"type":"click"}';
-- extract a key
SELECT data->>'url' FROM events WHERE data->>'user' = '42';
```

`@>` (contains), `->` (json), `->>` (text), `#>` (path). Postgres's JSONB is the reason many apps don't need a separate NoSQL database.

### Window functions

```sql
SELECT user_id, created_at, total,
       sum(total) OVER (PARTITION BY user_id ORDER BY created_at) AS running_total,
       rank()      OVER (PARTITION BY user_id ORDER BY total DESC) AS rnk
FROM orders;
```

Windows compute across a set of rows without collapsing them (unlike `GROUP BY`). See the [SQL tutorial](/Learn-SQL-in-One-Post-Complete-Tutorial-Joins-Window-Functions-Transactions-Quick-Start/) for the full set (`row_number`, `lag`, `lead`, `first_value`).

### CTEs and recursive CTEs

```sql
WITH active AS (SELECT * FROM users WHERE active),
     top AS (SELECT user_id, sum(total) AS spend FROM orders GROUP BY user_id)
SELECT a.email, t.spend FROM active a JOIN top t ON t.user_id = a.id;

-- recursive: org chart
WITH RECURSIVE tree AS (
  SELECT id, name, manager_id, 1 AS depth FROM employees WHERE manager_id IS NULL
  UNION ALL
  SELECT e.id, e.name, e.manager_id, t.depth+1 FROM employees e JOIN tree t ON e.manager_id = t.id
)
SELECT name, depth FROM tree;
```

### Full-text search

```sql
CREATE INDEX ON posts USING gin (to_tsvector('english', body));

SELECT title FROM posts
WHERE to_tsvector('english', body) @@ to_tsquery('english', 'postgres & index');
```

`tsvector` (normalized words), `tsquery` (search terms), `@@` (match), with ranking (`ts_rank`). For serious search, the `pg_trgm` extension adds fuzzy matching; for scale, branch to OpenSearch/Elasticsearch.

### Extensions

```sql
CREATE EXTENSION pg_trgm;       -- trigram fuzzy search
CREATE EXTENSION pgcrypto;       -- encryption functions
CREATE EXTENSION "uuid-ossp";    -- UUID generation
CREATE EXTENSION postgis;        -- geospatial (a whole database on its own)
CREATE EXTENSION pgvector;       -- vector similarity for ML/embeddings
```

Postgres's extension system is its superpower — `PostGIS` (geospatial), `pgvector` (vector search for LLM embeddings), `TimescaleDB` (time-series) all turn Postgres into a specialized database without leaving it.

---

## Stage 5 — Operations

### Architecture: process-per-connection

![PostgreSQL Server Architecture](/assets/img/diagrams/postgresql-tutorial/pg-arch.svg)

Each client connection gets its own **backend process** (forked by the postmaster). All backends share **shared buffers** (the page cache in RAM), **WAL buffers** (the write-ahead log), and catalog caches. Background workers (**checkpointer**, **WAL writer**, **autovacuum**, **stats collector**) run alongside. This is why connection pooling matters — every connection is an OS process.

### MVCC and isolation

![MVCC, Isolation, VACUUM, Locks](/assets/img/diagrams/postgresql-tutorial/pg-tx.svg)

Postgres uses **MVCC** (multi-version concurrency control): each transaction sees a consistent **snapshot** of the data. An `UPDATE` creates a **new row version** and marks the old one dead; readers in flight still see the old version. The payoff: **writes don't block reads, reads don't block writes** — high concurrency without locks for most operations.

The cost: **dead tuples** accumulate until **VACUUM** reclaims them. `autovacuum` runs this automatically; tune it for write-heavy tables.

### Isolation levels

| Level | Prevents |
|---|---|
| **Read Committed** (default) | dirty reads |
| **Repeatable Read** | dirty + non-repeatable reads (snapshot isolation) |
| **Serializable** | all anomalies (via SSI — serializable snapshot isolation) |

Pick the **weakest your app tolerates** — higher isolation costs performance. Most apps use Read Committed; financial code often needs Serializable.

### VACUUM and bloat

- `VACUUM` — mark dead tuples' space reusable (doesn't shrink the file).
- `VACUUM FULL` — rewrites the table to reclaim space (**takes an exclusive lock** — avoid in production).
- `ANALYZE` — update the planner's statistics (so it picks good plans).
- `VACUUM ANALYZE` — both.
- `autovacuum` — does this automatically; tune `autovacuum_vacuum_scale_factor` for hot tables.

> **Pitfall:** "Table bloat" (lots of dead tuples not vacuumed) slows every scan. Watch `pg_stat_user_tables` for `n_dead_tup`; if autovacuum can't keep up with a write-heavy table, lower its threshold or schedule manual `VACUUM` off-peak.

### WAL, backups, replication

- **WAL** (write-ahead log) — every change is logged before it's applied; enables crash recovery and replication.
- **`pg_dump` / `pg_restore`** — logical backups (SQL or custom format). Good for small/medium.
- **`pgBackRest` / `pg_basebackup`** — physical backups + PITR (point-in-time recovery). For production.
- **Streaming replication** — a replica applies WAL in real time; use for read scaling and HA. **Logical replication** for selective table sync or cross-version.

### Monitoring

- `pg_stat_activity` — what queries are running (and what's waiting).
- `pg_stat_statements` — aggregate query stats (slowest queries, most calls). Enable it.
- `pg_stat_user_tables` — dead tuples, seq vs index scans.
- External: **Prometheus** + `postgres_exporter` + Grafana.

### The toolchain

![PostgreSQL Toolchain + Ecosystem](/assets/img/diagrams/postgresql-tutorial/pg-toolchain.svg)

| Concern | Tool |
|---|---|
| Connect | psql, pgAdmin, DBeaver, DataGrip |
| Migrations | pg_dump/restore, Alembic, golang-migrate, Flyway, dbmate |
| ORMs | Prisma, Drizzle, SQLAlchemy, raw `pg`/`node-postgres` |
| Deploy + scale | streaming replica, PgBouncer, pgBackRest, RDS/Cloud SQL/Aurora |

---

## Quick-Start Checklist

1. **Run Postgres** — `docker run -e POSTGRES_PASSWORD=secret -p 5432:5432 postgres` or a managed instance.
2. **Connect with `psql`** and learn `\d`, `\dt`, `\dx`.
3. **Create a table** with `PRIMARY KEY`, `FOREIGN KEY`, `CHECK`, `timestamptz DEFAULT now()`.
4. **Use `jsonb`** for flexible fields, with a GIN index.
5. **Add indexes** on filter/join/sort columns; verify with `EXPLAIN ANALYZE`.
6. **Enable `pg_stat_statements`** to find slow queries.
7. **Put PgBouncer in front** for any app with many connections.
8. **Let autovacuum run**; monitor `n_dead_tup` on hot tables.
9. **Set up backups** (`pg_dump` for dev, `pgBackRest` + PITR for prod) and test a restore.
10. **Add a streaming replica** for read scaling + HA.

## Common Pitfalls

- **`varchar` over `text`** — `text` is unbounded and equally fast; `varchar(n)` only when you need a length cap.
- **Missing FK indexes** — joins and `ON DELETE CASCADE` get slow; index your foreign keys.
- **`LIKE '%term%'`** — can't use B-tree; use `pg_trgm` GIN or full-text.
- **Thousands of connections** — process-per-connection; use PgBouncer.
- **Table bloat** — dead tuples from MVCC; watch and tune autovacuum.
- **`VACUUM FULL` in production** — exclusive lock; use `pg_repack` instead to reclaim without blocking.
- **No backups / untested restore** — a backup you've never restored is not a backup. Test restores regularly.
- **`SERIAL` vs `IDENTITY`** — prefer `GENERATED ALWAYS AS IDENTITY` (SQL-standard, can't be accidentally overwritten).
- **`timestamptz` over `timestamp`** — always store UTC-aware; avoid the timezone-conversion bugs.

## Further Reading

- [PostgreSQL Docs](https://www.postgresql.org/docs/) — the authoritative reference
- [Use The Index, Luke!](https://use-the-index-luke.com/) — indexes, explained
- [PostgreSQL Tutorial](https://www.postgresqltutorial.com/) — hands-on lessons
- [The Art of PostgreSQL](https://theartofpostgresql.com/) by Dimitri Fontaine — query writing for devs
- [Postgres Weekly](https://postgresweekly.com/) — the newsletter

## Related guides

Postgres is the database at the center of the modern stack — these PyShine tutorials connect to it:

- **[Learn SQL in One Post](/Learn-SQL-in-One-Post-Complete-Tutorial-Joins-Window-Functions-Transactions-Quick-Start/)** — the prerequisite; standard SQL that Postgres implements.
- **[Learn Node.js + Express in One Post](/Learn-Node-js-Express-in-One-Post-Complete-Tutorial-Event-Loop-Middleware-Quick-Start/)** — query Postgres from `pg`/Prisma in your handlers.
- **[Learn Python in One Post](/Learn-Python-in-One-Post-Complete-Tutorial-Async-Type-Hints-Quick-Start/)** — SQLAlchemy / asyncpg from Python.
- **[Learn Go in One Post](/Learn-Go-in-One-Post-Complete-Tutorial-Goroutines-Channels-Generics-Quick-Start/)** — `database/sql` + `pgx` with connection pooling.
- **[Learn Docker in One Post](/Learn-Docker-in-One-Post-Complete-Tutorial-Dockerfile-Volumes-Compose-Quick-Start/)** — run Postgres in a container with a named volume for data.
- **[Learn System Design in One Post](/Learn-System-Design-in-One-Post-Complete-Tutorial-Scaling-CAP-Tradeoffs-Interview-Quick-Start/)** — sharding, replication, and CAP choices apply directly here.

---

Postgres is the database that grows with you: it's a perfectly good key-value store (`jsonb`), a relational database, a full-text search engine, a geospatial database (PostGIS), and a vector database (pgvector) — without leaving one engine. The five stages here — basics, schema, performance, advanced features, operations — cover everything from a single-table app to a replicated, pooled, backed-up production system. The two habits that pay off forever: **`EXPLAIN ANALYZE` before you guess**, and **let the constraints + MVCC do the work** the database was designed to do. Start a container, run the SQL above, and watch a plan change when you add an index — that's the moment it clicks.