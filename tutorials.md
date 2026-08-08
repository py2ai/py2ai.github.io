---
layout: page
title: "Evergreen Tutorials"
permalink: /tutorials/
description: "Complete one-post tutorials for 60+ technologies — languages, DevOps, ML, web dev, systems, security."
---

<style>
  .tut-hero {
    text-align: center;
    padding: 48px 20px 36px;
    margin-bottom: 32px;
    background: linear-gradient(135deg, var(--bg-secondary) 0%, var(--card-bg) 100%);
    border-radius: 12px;
    border: 1px solid var(--border-color);
  }
  .tut-hero h1 {
    font-size: 2.6rem;
    margin-bottom: 12px;
    background: linear-gradient(135deg, var(--link-color), var(--link-hover));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
  }
  .tut-hero p {
    font-size: 1.1rem;
    color: var(--text-secondary);
    max-width: 640px;
    margin: 0 auto;
  }
  .tut-hero .tut-stats {
    display: flex;
    justify-content: center;
    gap: 32px;
    margin-top: 24px;
    flex-wrap: wrap;
  }
  .tut-hero .tut-stat {
    text-align: center;
  }
  .tut-hero .tut-stat-num {
    font-size: 2rem;
    font-weight: 700;
    color: var(--link-color);
    display: block;
  }
  .tut-hero .tut-stat-label {
    font-size: 0.85rem;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }

  .tut-search-wrap {
    max-width: 600px;
    margin: 0 auto 36px;
    position: relative;
  }
  .tut-search-wrap input {
    width: 100%;
    padding: 14px 20px 14px 48px;
    font-size: 1.05rem;
    background: var(--input-bg);
    color: var(--input-text);
    border: 2px solid var(--input-border);
    border-radius: 10px;
    transition: border-color 0.2s, box-shadow 0.2s;
  }
  .tut-search-wrap input:focus {
    border-color: var(--link-color);
    outline: none;
    box-shadow: 0 0 0 3px rgba(39, 124, 234, 0.15);
  }
  .tut-search-wrap input::placeholder { color: var(--text-secondary); }
  .tut-search-wrap .search-icon {
    position: absolute;
    left: 16px;
    top: 50%;
    transform: translateY(-50%);
    color: var(--text-secondary);
    font-size: 1.1rem;
    pointer-events: none;
  }
  .tut-search-wrap .search-count {
    position: absolute;
    right: 16px;
    top: 50%;
    transform: translateY(-50%);
    color: var(--text-secondary);
    font-size: 0.85rem;
    pointer-events: none;
  }

  .tut-cat-tabs {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    justify-content: center;
    margin-bottom: 40px;
  }
  .tut-cat-tab {
    padding: 8px 18px;
    border-radius: 999px;
    font-size: 0.9rem;
    text-decoration: none;
    cursor: pointer;
    background: var(--card-bg);
    color: var(--link-color);
    border: 1px solid var(--border-color);
    transition: all 0.2s ease;
  }
  .tut-cat-tab:hover,
  .tut-cat-tab.active {
    background: var(--link-color);
    color: #fff;
    border-color: var(--link-color);
  }

  .tut-section { margin-bottom: 48px; }
  .tut-section-header {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 20px;
    padding-bottom: 12px;
    border-bottom: 2px solid var(--border-color);
  }
  .tut-section-header h2 {
    font-size: 1.5rem;
    margin: 0;
    color: var(--heading-color);
  }
  .tut-section-header .tut-count {
    color: var(--text-secondary);
    font-size: 0.9rem;
    font-weight: 400;
  }
  .tut-section-header .tut-icon {
    font-size: 1.6rem;
  }

  .tut-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
    gap: 20px;
  }

  .tut-card {
    background: var(--card-bg);
    border: 1px solid var(--card-border);
    border-radius: 10px;
    padding: 20px;
    text-decoration: none;
    color: inherit;
    transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
    display: flex;
    flex-direction: column;
    position: relative;
    overflow: hidden;
  }
  .tut-card::before {
    content: "";
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 3px;
    background: linear-gradient(90deg, var(--link-color), var(--link-hover));
    opacity: 0;
    transition: opacity 0.2s ease;
  }
  .tut-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 24px var(--shadow-color);
    border-color: var(--link-color);
    text-decoration: none;
    color: inherit;
  }
  .tut-card:hover::before { opacity: 1; }

  .tut-card .tut-card-title {
    font-size: 1.1rem;
    font-weight: 600;
    color: var(--heading-color);
    margin: 0 0 8px;
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .tut-card .tut-card-icon {
    font-size: 1.3rem;
    flex-shrink: 0;
  }
  .tut-card .tut-card-desc {
    font-size: 0.9rem;
    color: var(--text-secondary);
    margin: 0;
    line-height: 1.5;
    flex-grow: 1;
  }
  .tut-card .tut-card-tag {
    display: inline-block;
    margin-top: 12px;
    padding: 3px 10px;
    border-radius: 999px;
    font-size: 0.72rem;
    background: var(--bg-secondary);
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.3px;
  }
  .tut-card:hover .tut-card-tag {
    background: var(--link-color);
    color: #fff;
  }

  .tut-featured {
    margin-bottom: 48px;
    padding: 28px;
    background: linear-gradient(135deg, rgba(39, 124, 234, 0.08) 0%, var(--card-bg) 100%);
    border-radius: 12px;
    border: 1px solid var(--border-color);
  }
  .tut-featured-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 20px;
  }
  .tut-featured-header h2 {
    font-size: 1.4rem;
    margin: 0;
    color: var(--heading-color);
  }
  .tut-featured-header .tut-badge {
    background: var(--link-color);
    color: #fff;
    padding: 3px 10px;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 600;
  }

  .tut-featured-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(240px, 1fr));
    gap: 16px;
  }
  .tut-featured-card {
    background: var(--card-bg);
    border: 1px solid var(--card-border);
    border-radius: 10px;
    padding: 18px;
    text-decoration: none;
    color: inherit;
    transition: all 0.2s ease;
    position: relative;
    overflow: hidden;
  }
  .tut-featured-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 24px var(--shadow-color);
    border-color: var(--link-color);
    text-decoration: none;
    color: inherit;
  }
  .tut-featured-card .tut-featured-rank {
    position: absolute;
    top: 10px;
    right: 10px;
    width: 28px;
    height: 28px;
    border-radius: 50%;
    background: linear-gradient(135deg, var(--link-color), var(--link-hover));
    color: #fff;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.8rem;
    font-weight: 700;
  }
  .tut-featured-card .tut-featured-icon {
    font-size: 1.8rem;
    margin-bottom: 8px;
  }
  .tut-featured-card .tut-featured-title {
    font-size: 1.05rem;
    font-weight: 600;
    color: var(--heading-color);
    margin: 0 0 6px;
  }
  .tut-featured-card .tut-featured-desc {
    font-size: 0.85rem;
    color: var(--text-secondary);
    margin: 0;
  }

  .tut-no-results {
    display: none;
    text-align: center;
    padding: 60px 20px;
    color: var(--text-secondary);
  }
  .tut-no-results.show { display: block; }
  .tut-no-results .tut-no-results-icon {
    font-size: 3rem;
    margin-bottom: 12px;
  }

  @media (max-width: 600px) {
    .tut-hero h1 { font-size: 1.8rem; }
    .tut-hero .tut-stats { gap: 20px; }
    .tut-grid { grid-template-columns: 1fr; }
    .tut-featured-grid { grid-template-columns: 1fr; }
  }
</style>

<div class="tut-hero">
  <h1>Evergreen Tutorials</h1>
  <p>Master 60+ technologies with complete one-post tutorials — from languages and DevOps to machine learning and systems design. No fluff, just everything you need.</p>
  <div class="tut-stats">
    <div class="tut-stat">
      <span class="tut-stat-num">63+</span>
      <span class="tut-stat-label">Tutorials</span>
    </div>
    <div class="tut-stat">
      <span class="tut-stat-num">7</span>
      <span class="tut-stat-label">Categories</span>
    </div>
    <div class="tut-stat">
      <span class="tut-stat-num">100%</span>
      <span class="tut-stat-label">Free</span>
    </div>
  </div>
</div>

<div class="tut-search-wrap">
  <span class="search-icon">&#128269;</span>
  <input id="tut-search" type="search" placeholder="Search tutorials (e.g., Python, Docker, ML)..." aria-label="Search tutorials" />
  <span class="search-count" id="tut-search-count"></span>
</div>

<div class="tut-cat-tabs" id="tut-cat-tabs">
  <button class="tut-cat-tab active" data-filter="all">All</button>
  <button class="tut-cat-tab" data-filter="featured">Featured</button>
  <button class="tut-cat-tab" data-filter="languages">Languages</button>
  <button class="tut-cat-tab" data-filter="webdev">Web Dev</button>
  <button class="tut-cat-tab" data-filter="databases">Databases</button>
  <button class="tut-cat-tab" data-filter="devops">DevOps</button>
  <button class="tut-cat-tab" data-filter="ml-ai">ML & AI</button>
  <button class="tut-cat-tab" data-filter="systems">Systems</button>
  <button class="tut-cat-tab" data-filter="tools">Tools</button>
</div>

<div class="tut-featured" id="tut-featured" data-cat="featured">
  <div class="tut-featured-header">
    <h2>&#11088; Featured Tutorials</h2>
    <span class="tut-badge">Top 10</span>
  </div>
  <div class="tut-featured-grid" id="tut-featured-grid"></div>
</div>

<p class="tut-no-results" id="tut-no-results">
  <div class="tut-no-results-icon">&#128270;</div>
  <strong>No tutorials found.</strong>
  <p>Try a different search term or browse the categories below.</p>
</p>

{% assign tutorials = site.data.tutorials %}
{% if tutorials %}
  {% assign grouped = tutorials | group_by: 'category' %}
{% endif %}

<div id="tut-categories-container">

<section class="tut-section" data-cat="languages">
  <div class="tut-section-header">
    <span class="tut-icon">&#128218;</span>
    <h2>Languages</h2>
    <span class="tut-count" id="count-languages"></span>
  </div>
  <div class="tut-grid" id="grid-languages">
    <a class="tut-card" href="/Learn-Python-in-One-Post-Complete-Tutorial-Async-Type-Hints-Quick-Start/" data-title="Python" data-desc="Python tutorial async type hints">
      <div class="tut-card-title"><span class="tut-card-icon">&#128013;</span> Python</div>
      <p class="tut-card-desc">Master Python from scratch — syntax, async, type hints, and modern best practices for data science, web dev, and automation.</p>
      <span class="tut-card-tag">Language</span>
    </a>
    <a class="tut-card" href="/Learn-Go-in-One-Post-Complete-Tutorial-Goroutines-Channels-Generics-Quick-Start/" data-title="Go" data-desc="Go tutorial goroutines channels generics">
      <div class="tut-card-title"><span class="tut-card-icon">&#128640;</span> Go</div>
      <p class="tut-card-desc">Learn Go with goroutines, channels, and generics — build fast, concurrent microservices and CLI tools.</p>
      <span class="tut-card-tag">Language</span>
    </a>
    <a class="tut-card" href="/Learn-Rust-in-One-Post-Complete-Tutorial-Ownership-Borrow-Async-Quick-Start/" data-title="Rust" data-desc="Rust tutorial ownership borrow async">
      <div class="tut-card-title"><span class="tut-card-icon">&#9881;</span> Rust</div>
      <p class="tut-card-desc">Master Rust's ownership model, borrowing, lifetimes, and async — write memory-safe systems-level code.</p>
      <span class="tut-card-tag">Language</span>
    </a>
    <a class="tut-card" href="/Learn-Java-in-One-Post-Complete-Tutorial-OOP-Streams-Virtual-Threads-JVM-Quick-Start/" data-title="Java" data-desc="Java tutorial OOP streams virtual threads JVM">
      <div class="tut-card-title"><span class="tut-card-icon">&#9749;</span> Java</div>
      <p class="tut-card-desc">Complete Java guide — OOP, streams, lambdas, virtual threads, and JVM internals for enterprise development.</p>
      <span class="tut-card-tag">Language</span>
    </a>
    <a class="tut-card" href="/Learn-CPP-in-One-Post-Complete-Tutorial-Modern-Cpp-Quick-Start/" data-title="C++" data-desc="C++ tutorial modern C++">
      <div class="tut-card-title"><span class="tut-card-icon">&#128296;</span> C++</div>
      <p class="tut-card-desc">Modern C++ deep dive — smart pointers, templates, lambdas, move semantics, and RAII for high-performance systems.</p>
      <span class="tut-card-tag">Language</span>
    </a>
    <a class="tut-card" href="/Learn-C-Sharp-in-One-Post-Complete-Tutorial-LINQ-Async-Tasks-DotNet-Quick-Start/" data-title="C#" data-desc="CSharp tutorial LINQ async tasks dotnet">
      <div class="tut-card-title"><span class="tut-card-icon">&#127919;</span> C#</div>
      <p class="tut-card-desc">Master C# with LINQ, async/await, delegates, and .NET ecosystem — build modern Windows and cross-platform apps.</p>
      <span class="tut-card-tag">Language</span>
    </a>
    <a class="tut-card" href="/Learn-JavaScript-TypeScript-in-One-Post-Complete-Tutorial-Async-Types-Quick-Start/" data-title="JavaScript TypeScript" data-desc="JavaScript TypeScript tutorial async types">
      <div class="tut-card-title"><span class="tut-card-icon">&#128156;</span> JavaScript / TypeScript</div>
      <p class="tut-card-desc">From ES6+ to TypeScript — learn closures, promises, async/await, generics, and type-safe web development.</p>
      <span class="tut-card-tag">Language</span>
    </a>
    <a class="tut-card" href="/Learn-Ruby-in-One-Post-Complete-Tutorial-Blocks-Mixins-Metaprogramming-Rails-Quick-Start/" data-title="Ruby" data-desc="Ruby tutorial blocks mixins metaprogramming rails">
      <div class="tut-card-title"><span class="tut-card-icon">&#128142;</span> Ruby</div>
      <p class="tut-card-desc">Learn Ruby — blocks, mixins, metaprogramming, and Rails patterns for elegant, productive web development.</p>
      <span class="tut-card-tag">Language</span>
    </a>
    <a class="tut-card" href="/Learn-Kotlin-in-One-Post-Complete-Tutorial-Null-Safety-Coroutines-Compose-Quick-Start/" data-title="Kotlin" data-desc="Kotlin tutorial null safety coroutines compose">
      <div class="tut-card-title"><span class="tut-card-icon">&#128156;</span> Kotlin</div>
      <p class="tut-card-desc">Master Kotlin — null safety, coroutines, data classes, and Jetpack Compose for Android and multiplatform development.</p>
      <span class="tut-card-tag">Language</span>
    </a>
    <a class="tut-card" href="/Learn-Swift-in-One-Post-Complete-Tutorial-Optionals-Protocols-Async-SwiftUI-Quick-Start/" data-title="Swift" data-desc="Swift tutorial optionals protocols async SwiftUI">
      <div class="tut-card-title"><span class="tut-card-icon">&#127822;</span> Swift</div>
      <p class="tut-card-desc">Learn Swift — optionals, protocols, generics, async/await, and SwiftUI for iOS, macOS, and server-side development.</p>
      <span class="tut-card-tag">Language</span>
    </a>
    <a class="tut-card" href="/Learn-Dart-in-One-Post-Complete-Tutorial-Null-Safety-Async-Flutter-Quick-Start/" data-title="Dart" data-desc="Dart tutorial null safety async flutter">
      <div class="tut-card-title"><span class="tut-card-icon">&#127912;</span> Dart</div>
      <p class="tut-card-desc">Master Dart — null safety, sound types, async/await, and the Flutter framework for cross-platform app development.</p>
      <span class="tut-card-tag">Language</span>
    </a>
    <a class="tut-card" href="/Learn-Lua-in-One-Post-Complete-Tutorial-Tables-Metatables-Coroutines-Quick-Start/" data-title="Lua" data-desc="Lua tutorial tables metatables coroutines">
      <div class="tut-card-title"><span class="tut-card-icon">&#127760;</span> Lua</div>
      <p class="tut-card-desc">Learn Lua — tables, metatables, metaprogramming, and coroutines for game scripting and embedded systems.</p>
      <span class="tut-card-tag">Language</span>
    </a>
    <a class="tut-card" href="/Learn-Elixir-in-One-Post-Complete-Tutorial-Pattern-Matching-OTP-Phoenix-Quick-Start/" data-title="Elixir" data-desc="Elixir tutorial pattern matching OTP phoenix">
      <div class="tut-card-title"><span class="tut-card-icon">&#128270;</span> Elixir</div>
      <p class="tut-card-desc">Master Elixir — pattern matching, immutability, OTP, and Phoenix for fault-tolerant, scalable applications.</p>
      <span class="tut-card-tag">Language</span>
    </a>
    <a class="tut-card" href="/Learn-Haskell-in-One-Post-Complete-Tutorial-Purity-Laziness-Monads-Quick-Start/" data-title="Haskell" data-desc="Haskell tutorial purity laziness monads">
      <div class="tut-card-title"><span class="tut-card-icon">&#9581;</span> Haskell</div>
      <p class="tut-card-desc">Learn Haskell — purity, laziness, algebraic data types, typeclasses, and monads for functional programming mastery.</p>
      <span class="tut-card-tag">Language</span>
    </a>
    <a class="tut-card" href="/Learn-Scala-in-One-Post-Complete-Tutorial-Case-Classes-Traits-Cats-Effect-Quick-Start/" data-title="Scala" data-desc="Scala tutorial case classes traits cats effect">
      <div class="tut-card-title"><span class="tut-card-icon">&#127744;</span> Scala</div>
      <p class="tut-card-desc">Master Scala — case classes, traits, pattern matching, Cats Effect, and Akka for hybrid functional OOP on the JVM.</p>
      <span class="tut-card-tag">Language</span>
    </a>
  </div>
</section>

<section class="tut-section" data-cat="webdev">
  <div class="tut-section-header">
    <span class="tut-icon">&#127760;</span>
    <h2>Web Development</h2>
    <span class="tut-count" id="count-webdev"></span>
  </div>
  <div class="tut-grid" id="grid-webdev">
    <a class="tut-card" href="/Learn-HTML-CSS-in-One-Post-Complete-Tutorial-Semantic-Markup-Box-Model-Flexbox-Grid-Quick-Start/" data-title="HTML CSS" data-desc="HTML CSS tutorial semantic markup box model flexbox grid">
      <div class="tut-card-title"><span class="tut-card-icon">&#127760;</span> HTML / CSS</div>
      <p class="tut-card-desc">Master the web foundation — semantic HTML, CSS box model, flexbox, grid, and responsive design.</p>
      <span class="tut-card-tag">Web Dev</span>
    </a>
    <a class="tut-card" href="/Learn-React-Next-js-in-One-Post-Complete-Tutorial-Components-Hooks-Server-Components-Quick-Start/" data-title="React Next.js" data-desc="React Next.js tutorial components hooks server components">
      <div class="tut-card-title"><span class="tut-card-icon">&#9889;</span> React / Next.js</div>
      <p class="tut-card-desc">Master React components, hooks, state management, and Next.js for building production-ready web apps.</p>
      <span class="tut-card-tag">Web Dev</span>
    </a>
    <a class="tut-card" href="/Learn-Next-js-App-Router-in-One-Post-Complete-Tutorial-Server-Components-Caching-Server-Actions-Quick-Start/" data-title="Next.js App Router" data-desc="Next.js App Router tutorial server components caching server actions">
      <div class="tut-card-title"><span class="tut-card-icon">&#128220;</span> Next.js App Router</div>
      <p class="tut-card-desc">Deep dive into Next.js App Router — React Server Components, streaming, caching, and server actions.</p>
      <span class="tut-card-tag">Web Dev</span>
    </a>
    <a class="tut-card" href="/Learn-Node-js-Express-in-One-Post-Complete-Tutorial-Event-Loop-Middleware-Quick-Start/" data-title="Node.js Express" data-desc="Node.js Express tutorial event loop middleware">
      <div class="tut-card-title"><span class="tut-card-icon">&#128994;</span> Node.js / Express</div>
      <p class="tut-card-desc">Learn Node.js event loop, Express middleware, routing, error handling, and production best practices.</p>
      <span class="tut-card-tag">Web Dev</span>
    </a>
    <a class="tut-card" href="/Learn-FastAPI-in-One-Post-Complete-Tutorial-Pydantic-Async-Dependency-Injection-Quick-Start/" data-title="FastAPI" data-desc="FastAPI tutorial pydantic async dependency injection">
      <div class="tut-card-title"><span class="tut-card-icon">&#9889;</span> FastAPI</div>
      <p class="tut-card-desc">Master FastAPI — Pydantic models, async endpoints, dependency injection, and auto-generated OpenAPI docs.</p>
      <span class="tut-card-tag">Web Dev</span>
    </a>
    <a class="tut-card" href="/Learn-Tailwind-CSS-in-One-Post-Complete-Tutorial-Utilities-Responsive-Components-Quick-Start/" data-title="Tailwind CSS" data-desc="Tailwind CSS tutorial utilities responsive components">
      <div class="tut-card-title"><span class="tut-card-icon">&#127912;</span> Tailwind CSS</div>
      <p class="tut-card-desc">Learn Tailwind — utility-first classes, responsive design, dark mode, and building reusable component systems.</p>
      <span class="tut-card-tag">Web Dev</span>
    </a>
    <a class="tut-card" href="/Learn-WebAssembly-in-One-Post-Complete-Tutorial-Modules-WASI-Component-Model-Quick-Start/" data-title="WebAssembly" data-desc="WebAssembly tutorial modules WASI component model">
      <div class="tut-card-title"><span class="tut-card-icon">&#128225;</span> WebAssembly</div>
      <p class="tut-card-desc">Master WebAssembly — modules, WASI, component model, and running near-native code in browsers and edge runtimes.</p>
      <span class="tut-card-tag">Web Dev</span>
    </a>
    <a class="tut-card" href="/Learn-Rust-Axum-in-One-Post-Complete-Tutorial-Routing-Extractors-Tower-Middleware-Quick-Start/" data-title="Rust Axum" data-desc="Rust Axum tutorial routing extractors tower middleware">
      <div class="tut-card-title"><span class="tut-card-icon">&#128296;</span> Rust Axum</div>
      <p class="tut-card-desc">Build web apps with Rust Axum — routing, extractors, Tower middleware, and stateful services.</p>
      <span class="tut-card-tag">Web Dev</span>
    </a>
  </div>
</section>

<section class="tut-section" data-cat="databases">
  <div class="tut-section-header">
    <span class="tut-icon">&#128190;</span>
    <h2>Databases</h2>
    <span class="tut-count" id="count-databases"></span>
  </div>
  <div class="tut-grid" id="grid-databases">
    <a class="tut-card" href="/Learn-SQL-in-One-Post-Complete-Tutorial-Joins-Window-Functions-Transactions-Quick-Start/" data-title="SQL" data-desc="SQL tutorial joins window functions transactions">
      <div class="tut-card-title"><span class="tut-card-icon">&#128190;</span> SQL</div>
      <p class="tut-card-desc">Master SQL — joins, window functions, CTEs, transactions, indexes, and query optimization.</p>
      <span class="tut-card-tag">Database</span>
    </a>
    <a class="tut-card" href="/Learn-PostgreSQL-in-One-Post-Complete-Tutorial-Indexes-MVCC-Performance-Quick-Start/" data-title="PostgreSQL" data-desc="PostgreSQL tutorial indexes MVCC performance">
      <div class="tut-card-title"><span class="tut-card-icon">&#128030;</span> PostgreSQL</div>
      <p class="tut-card-desc">Deep dive PostgreSQL — advanced indexes, MVCC, query planning, performance tuning, and partitioning.</p>
      <span class="tut-card-tag">Database</span>
    </a>
    <a class="tut-card" href="/Learn-Redis-in-One-Post-Complete-Tutorial-Data-Structures-Caching-Persistence-Quick-Start/" data-title="Redis" data-desc="Redis tutorial data structures caching persistence">
      <div class="tut-card-title"><span class="tut-card-icon">&#128293;</span> Redis</div>
      <p class="tut-card-desc">Master Redis — strings, hashes, sorted sets, pub/sub, caching patterns, and persistence strategies.</p>
      <span class="tut-card-tag">Database</span>
    </a>
    <a class="tut-card" href="/Learn-Elasticsearch-OpenSearch-in-One-Post-Complete-Tutorial-Inverted-Index-Shards-Query-DSL-Quick-Start/" data-title="Elasticsearch OpenSearch" data-desc="Elasticsearch OpenSearch tutorial inverted index shards query DSL">
      <div class="tut-card-title"><span class="tut-card-icon">&#128269;</span> Elasticsearch / OpenSearch</div>
      <p class="tut-card-desc">Learn search engines — inverted indexes, shards, query DSL, aggregations, and relevance tuning.</p>
      <span class="tut-card-tag">Database</span>
    </a>
    <a class="tut-card" href="/Learn-RabbitMQ-in-One-Post-Complete-Tutorial-Exchanges-Queues-Reliability-Quick-Start/" data-title="RabbitMQ" data-desc="RabbitMQ tutorial exchanges queues reliability">
      <div class="tut-card-title"><span class="tut-card-icon">&#128007;</span> RabbitMQ</div>
      <p class="tut-card-desc">Master RabbitMQ — exchanges, queues, routing, dead-letter queues, and reliable message delivery.</p>
      <span class="tut-card-tag">Database</span>
    </a>
  </div>
</section>

<section class="tut-section" data-cat="devops">
  <div class="tut-section-header">
    <span class="tut-icon">&#9881;</span>
    <h2>DevOps &amp; Infrastructure</h2>
    <span class="tut-count" id="count-devops"></span>
  </div>
  <div class="tut-grid" id="grid-devops">
    <a class="tut-card" href="/Learn-Docker-in-One-Post-Complete-Tutorial-Dockerfile-Volumes-Compose-Quick-Start/" data-title="Docker" data-desc="Docker tutorial dockerfile volumes compose">
      <div class="tut-card-title"><span class="tut-card-icon">&#128666;</span> Docker</div>
      <p class="tut-card-desc">Master Docker — Dockerfiles, images, volumes, networking, compose, and multi-stage production builds.</p>
      <span class="tut-card-tag">DevOps</span>
    </a>
    <a class="tut-card" href="/Learn-Kubernetes-in-One-Post-Complete-Tutorial-Pods-Deployments-Services-Production-Quick-Start/" data-title="Kubernetes" data-desc="Kubernetes tutorial pods deployments services production">
      <div class="tut-card-title"><span class="tut-card-icon">&#9881;</span> Kubernetes</div>
      <p class="tut-card-desc">Master Kubernetes — pods, deployments, services, ingress, RBAC, and production cluster management.</p>
      <span class="tut-card-tag">DevOps</span>
    </a>
    <a class="tut-card" href="/Learn-Terraform-in-One-Post-Complete-Tutorial-HCL-State-Modules-Providers-Quick-Start/" data-title="Terraform" data-desc="Terraform tutorial HCL state modules providers">
      <div class="tut-card-title"><span class="tut-card-icon">&#127960;</span> Terraform</div>
      <p class="tut-card-desc">Learn Terraform — HCL, state management, modules, providers, and infrastructure as code patterns.</p>
      <span class="tut-card-tag">DevOps</span>
    </a>
    <a class="tut-card" href="/Learn-GitHub-Actions-in-One-Post-Complete-Tutorial-Workflows-Jobs-Runners-Secrets-Quick-Start/" data-title="GitHub Actions" data-desc="GitHub Actions tutorial workflows jobs runners secrets">
      <div class="tut-card-title"><span class="tut-card-icon">&#9881;</span> GitHub Actions</div>
      <p class="tut-card-desc">Master CI/CD with GitHub Actions — workflows, jobs, runners, secrets, matrix builds, and deployment pipelines.</p>
      <span class="tut-card-tag">DevOps</span>
    </a>
    <a class="tut-card" href="/Learn-GitLab-CI-CD-in-One-Post-Complete-Tutorial-Pipelines-Runners-Environments-Quick-Start/" data-title="GitLab CI CD" data-desc="GitLab CI CD tutorial pipelines runners environments">
      <div class="tut-card-title"><span class="tut-card-icon">&#9881;</span> GitLab CI/CD</div>
      <p class="tut-card-desc">Learn GitLab CI/CD — pipelines, runners, environments, Artifacts, and GitOps deployment strategies.</p>
      <span class="tut-card-tag">DevOps</span>
    </a>
    <a class="tut-card" href="/Learn-Ansible-in-One-Post-Complete-Tutorial-Inventory-Playbooks-Roles-Vault-Quick-Start/" data-title="Ansible" data-desc="Ansible tutorial inventory playbooks roles vault">
      <div class="tut-card-title"><span class="tut-card-icon">&#129309;</span> Ansible</div>
      <p class="tut-card-desc">Master Ansible — inventory, playbooks, roles, variables, Ansible Vault, and idempotent automation.</p>
      <span class="tut-card-tag">DevOps</span>
    </a>
    <a class="tut-card" href="/Learn-Nginx-in-One-Post-Complete-Tutorial-Reverse-Proxy-TLS-Load-Balancing-Quick-Start/" data-title="Nginx" data-desc="Nginx tutorial reverse proxy TLS load balancing">
      <div class="tut-card-title"><span class="tut-card-icon">&#127760;</span> Nginx</div>
      <p class="tut-card-desc">Master Nginx — reverse proxy, TLS termination, load balancing, caching, and rate limiting.</p>
      <span class="tut-card-tag">DevOps</span>
    </a>
    <a class="tut-card" href="/Learn-Prometheus-in-One-Post-Complete-Tutorial-Metrics-PromQL-Alerting-Grafana-Quick-Start/" data-title="Prometheus" data-desc="Prometheus tutorial metrics PromQL alerting Grafana">
      <div class="tut-card-title"><span class="tut-card-icon">&#128202;</span> Prometheus</div>
      <p class="tut-card-desc">Master Prometheus — metrics, PromQL, recording rules, alertmanager, and Grafana dashboards.</p>
      <span class="tut-card-tag">DevOps</span>
    </a>
    <a class="tut-card" href="/Learn-Loki-in-One-Post-Complete-Tutorial-Labels-LogQL-Promtail-Grafana-Quick-Start/" data-title="Loki" data-desc="Loki tutorial labels LogQL Promtail Grafana">
      <div class="tut-card-title"><span class="tut-card-icon">&#128221;</span> Loki</div>
      <p class="tut-card-desc">Learn Loki — label-based log management, LogQL queries, Promtail scraping, and Grafana integration.</p>
      <span class="tut-card-tag">DevOps</span>
    </a>
    <a class="tut-card" href="/Learn-Packer-in-One-Post-Complete-Tutorial-Builders-Provisioners-Immutable-Images-Quick-Start/" data-title="Packer" data-desc="Packer tutorial builders provisioners immutable images">
      <div class="tut-card-title"><span class="tut-card-icon">&#128230;</span> Packer</div>
      <p class="tut-card-desc">Master Packer — builders, provisioners, post-processors, and creating immutable machine images.</p>
      <span class="tut-card-tag">DevOps</span>
    </a>
    <a class="tut-card" href="/Learn-Traefik-in-One-Post-Complete-Tutorial-Dynamic-Routing-Middleware-ACME-Quick-Start/" data-title="Traefik" data-desc="Traefik tutorial dynamic routing middleware ACME">
      <div class="tut-card-title"><span class="tut-card-icon">&#128694;</span> Traefik</div>
      <p class="tut-card-desc">Learn Traefik — dynamic routing, middleware, service discovery, and automatic TLS with ACME.</p>
      <span class="tut-card-tag">DevOps</span>
    </a>
    <a class="tut-card" href="/Learn-Apache-Kafka-in-One-Post-Complete-Tutorial-Topics-Partitions-Consumer-Groups-Quick-Start/" data-title="Apache Kafka" data-desc="Apache Kafka tutorial topics partitions consumer groups">
      <div class="tut-card-title"><span class="tut-card-icon">&#128279;</span> Apache Kafka</div>
      <p class="tut-card-desc">Master Kafka — topics, partitions, consumer groups, delivery semantics, and stream processing.</p>
      <span class="tut-card-tag">DevOps</span>
    </a>
  </div>
</section>

<section class="tut-section" data-cat="ml-ai">
  <div class="tut-section-header">
    <span class="tut-icon">&#129302;</span>
    <h2>Machine Learning &amp; AI</h2>
    <span class="tut-count" id="count-ml-ai"></span>
  </div>
  <div class="tut-grid" id="grid-ml-ai">
    <a class="tut-card" href="/Learn-Machine-Learning-in-One-Post-Complete-Tutorial-Supervised-Unsupervised-Deep-Learning-Quick-Start/" data-title="Machine Learning" data-desc="Machine Learning tutorial supervised unsupervised deep learning">
      <div class="tut-card-title"><span class="tut-card-icon">&#129302;</span> Machine Learning</div>
      <p class="tut-card-desc">Master ML — supervised, unsupervised, reinforcement learning, model evaluation, and scikit-learn.</p>
      <span class="tut-card-tag">ML &amp; AI</span>
    </a>
    <a class="tut-card" href="/Learn-Deep-Learning-in-One-Post-Complete-Tutorial-Neural-Networks-CNN-Transformers-PyTorch-Quick-Start/" data-title="Deep Learning" data-desc="Deep Learning tutorial neural networks CNN transformers PyTorch">
      <div class="tut-card-title"><span class="tut-card-icon">&#129516;</span> Deep Learning</div>
      <p class="tut-card-desc">Master deep learning — neural networks, CNNs, RNNs, Transformers, and PyTorch implementation.</p>
      <span class="tut-card-tag">ML &amp; AI</span>
    </a>
    <a class="tut-card" href="/Learn-Linear-Algebra-for-ML-in-One-Post-Complete-Tutorial-Vectors-Matrices-SVD-Eigen-Quick-Start/" data-title="Linear Algebra for ML" data-desc="Linear Algebra for ML tutorial vectors matrices SVD eigen">
      <div class="tut-card-title"><span class="tut-card-icon">&#128208;</span> Linear Algebra for ML</div>
      <p class="tut-card-desc">Essential linear algebra for ML — vectors, matrices, SVD, eigenvalues, and transformations.</p>
      <span class="tut-card-tag">ML &amp; AI</span>
    </a>
    <a class="tut-card" href="/Learn-Probability-and-Statistics-for-ML-in-One-Post-Complete-Tutorial-Distributions-Bayes-Inference-Quick-Start/" data-title="Probability Statistics for ML" data-desc="Probability Statistics for ML tutorial distributions Bayes inference">
      <div class="tut-card-title"><span class="tut-card-icon">&#128202;</span> Probability &amp; Statistics for ML</div>
      <p class="tut-card-desc">Master probability and statistics — distributions, Bayes' theorem, hypothesis testing, and Bayesian inference.</p>
      <span class="tut-card-tag">ML &amp; AI</span>
    </a>
    <a class="tut-card" href="/Learn-Computer-Architecture-in-One-Post-Complete-Tutorial-Pipeline-Memory-Cache-Multicore-Quick-Start/" data-title="Computer Architecture" data-desc="Computer Architecture tutorial pipeline memory cache multicore">
      <div class="tut-card-title"><span class="tut-card-icon">&#128737;</span> Computer Architecture</div>
      <p class="tut-card-desc">Learn computer architecture — pipelines, caching, memory hierarchy, and multicore design.</p>
      <span class="tut-card-tag">ML &amp; AI</span>
    </a>
    <a class="tut-card" href="/Learn-Cryptography-in-One-Post-Complete-Tutorial-Symmetric-Asymmetric-Hashing-TLS-Quick-Start/" data-title="Cryptography" data-desc="Cryptography tutorial symmetric asymmetric hashing TLS">
      <div class="tut-card-title"><span class="tut-card-icon">&#128274;</span> Cryptography</div>
      <p class="tut-card-desc">Master cryptography — symmetric ciphers, asymmetric crypto, hashing, digital signatures, and TLS.</p>
      <span class="tut-card-tag">ML &amp; AI</span>
    </a>
  </div>
</section>

<section class="tut-section" data-cat="systems">
  <div class="tut-section-header">
    <span class="tut-icon">&#128218;</span>
    <h2>Systems &amp; Networking</h2>
    <span class="tut-count" id="count-systems"></span>
  </div>
  <div class="tut-grid" id="grid-systems">
    <a class="tut-card" href="/Learn-Operating-Systems-in-One-Post-Complete-Tutorial-Processes-Memory-Threads-Quick-Start/" data-title="Operating Systems" data-desc="Operating Systems tutorial processes memory threads">
      <div class="tut-card-title"><span class="tut-card-icon">&#128187;</span> Operating Systems</div>
      <p class="tut-card-desc">Master OS concepts — processes, threads, scheduling, memory management, and file systems.</p>
      <span class="tut-card-tag">Systems</span>
    </a>
    <a class="tut-card" href="/Learn-Computer-Networking-in-One-Post-Complete-Tutorial-OSI-TCP-UDP-HTTP-Quick-Start/" data-title="Computer Networking" data-desc="Computer Networking tutorial OSI TCP UDP HTTP">
      <div class="tut-card-title"><span class="tut-card-icon">&#128225;</span> Computer Networking</div>
      <p class="tut-card-desc">Learn networking — OSI model, TCP/UDP, HTTP/HTTPS, DNS, and network security fundamentals.</p>
      <span class="tut-card-tag">Systems</span>
    </a>
    <a class="tut-card" href="/Learn-Data-Structures-and-Algorithms-in-One-Post-Complete-Tutorial-Big-O-Trees-Graphs-DP-Quick-Start/" data-title="Data Structures Algorithms" data-desc="Data Structures Algorithms tutorial Big-O trees graphs DP">
      <div class="tut-card-title"><span class="tut-card-icon">&#128200;</span> Data Structures &amp; Algorithms</div>
      <p class="tut-card-desc">Master DSA — Big-O, arrays, trees, graphs, dynamic programming, and algorithm design patterns.</p>
      <span class="tut-card-tag">Systems</span>
    </a>
    <a class="tut-card" href="/Learn-System-Design-in-One-Post-Complete-Tutorial-Scaling-CAP-Tradeoffs-Interview-Quick-Start/" data-title="System Design" data-desc="System Design tutorial scaling CAP tradeoffs interview">
      <div class="tut-card-title"><span class="tut-card-icon">&#127970;</span> System Design</div>
      <p class="tut-card-desc">Learn system design — scaling, CAP theorem, load balancing, caching, and interview preparation.</p>
      <span class="tut-card-tag">Systems</span>
    </a>
    <a class="tut-card" href="/Learn-Linux-CLI-in-One-Post-Complete-Tutorial-Files-Processes-Permissions-Quick-Start/" data-title="Linux CLI" data-desc="Linux CLI tutorial files processes permissions">
      <div class="tut-card-title"><span class="tut-card-icon">&#128039;</span> Linux CLI</div>
      <p class="tut-card-desc">Master Linux command line — file operations, process management, permissions, and shell scripting.</p>
      <span class="tut-card-tag">Systems</span>
    </a>
    <a class="tut-card" href="/Learn-REST-API-in-One-Post-Complete-Tutorial-Methods-Status-Codes-Production-Quick-Start/" data-title="REST API" data-desc="REST API tutorial methods status codes production">
      <div class="tut-card-title"><span class="tut-card-icon">&#128279;</span> REST API</div>
      <p class="tut-card-desc">Master REST API design — HTTP methods, status codes, resource modeling, and production best practices.</p>
      <span class="tut-card-tag">Systems</span>
    </a>
    <a class="tut-card" href="/Learn-GraphQL-in-One-Post-Complete-Tutorial-Schema-Queries-Mutations-Resolvers-Apollo-Quick-Start/" data-title="GraphQL" data-desc="GraphQL tutorial schema queries mutations resolvers Apollo">
      <div class="tut-card-title"><span class="tut-card-icon">&#128204;</span> GraphQL</div>
      <p class="tut-card-desc">Learn GraphQL — schema design, queries, mutations, resolvers, subscriptions, and Apollo server.</p>
      <span class="tut-card-tag">Systems</span>
    </a>
    <a class="tut-card" href="/Learn-gRPC-Protobuf-in-One-Post-Complete-Tutorial-Proto-Streaming-Interceptors-Quick-Start/" data-title="gRPC Protobuf" data-desc="gRPC Protobuf tutorial proto streaming interceptors">
      <div class="tut-card-title"><span class="tut-card-icon">&#9881;</span> gRPC / Protobuf</div>
      <p class="tut-card-desc">Master gRPC — Protocol Buffers, streaming, interceptors, and high-performance microservice communication.</p>
      <span class="tut-card-tag">Systems</span>
    </a>
    <a class="tut-card" href="/Learn-WebSocket-in-One-Post-Complete-Tutorial-Handshake-Frames-Scaling-Quick-Start/" data-title="WebSocket" data-desc="WebSocket tutorial handshake frames scaling">
      <div class="tut-card-title"><span class="tut-card-icon">&#128279;</span> WebSocket</div>
      <p class="tut-card-desc">Learn WebSocket — handshake, frames, ping/pong, scaling, and building real-time applications.</p>
      <span class="tut-card-tag">Systems</span>
    </a>
    <a class="tut-card" href="/Learn-OAuth-2-OIDC-in-One-Post-Complete-Tutorial-Flows-Tokens-PKCE-Security-Quick-Start/" data-title="OAuth 2.0 OIDC" data-desc="OAuth 2 OIDC tutorial flows tokens PKCE security">
      <div class="tut-card-title"><span class="tut-card-icon">&#128274;</span> OAuth 2.0 / OIDC</div>
      <p class="tut-card-desc">Master OAuth 2.0 and OpenID Connect — flows, tokens, PKCE, SSO, and security best practices.</p>
      <span class="tut-card-tag">Systems</span>
    </a>
    <a class="tut-card" href="/Learn-Compilers-in-One-Post-Complete-Tutorial-Lexing-Parsing-IR-Codegen-Quick-Start/" data-title="Compilers" data-desc="Compilers tutorial lexing parsing IR codegen">
      <div class="tut-card-title"><span class="tut-card-icon">&#128296;</span> Compilers</div>
      <p class="tut-card-desc">Learn compiler construction — lexing, parsing, AST, IR, optimization, and code generation.</p>
      <span class="tut-card-tag">Systems</span>
    </a>
    <a class="tut-card" href="/Learn-Observability-in-One-Post-Complete-Tutorial-Metrics-Logs-Traces-OpenTelemetry-Quick-Start/" data-title="Observability" data-desc="Observability tutorial metrics logs traces OpenTelemetry">
      <div class="tut-card-title"><span class="tut-card-icon">&#128202;</span> Observability</div>
      <p class="tut-card-desc">Master observability — metrics, logs, distributed tracing, OpenTelemetry, and incident response.</p>
      <span class="tut-card-tag">Systems</span>
    </a>
    <a class="tut-card" href="/Learn-WebRTC-in-One-Post-Complete-Tutorial-Signaling-ICE-Media-Data-Channels-Quick-Start/" data-title="WebRTC" data-desc="WebRTC tutorial signaling ICE media data channels">
      <div class="tut-card-title"><span class="tut-card-icon">&#127909;</span> WebRTC</div>
      <p class="tut-card-desc">Learn WebRTC — signaling, ICE, media streams, data channels, and building peer-to-peer applications.</p>
      <span class="tut-card-tag">Systems</span>
    </a>
  </div>
</section>

<section class="tut-section" data-cat="tools">
  <div class="tut-section-header">
    <span class="tut-icon">&#128295;</span>
    <h2>Tools</h2>
    <span class="tut-count" id="count-tools"></span>
  </div>
  <div class="tut-grid" id="grid-tools">
    <a class="tut-card" href="/Learn-Git-in-One-Post-Complete-Tutorial-Branches-Rebase-Workflows-Quick-Start/" data-title="Git" data-desc="Git tutorial branches rebase workflows">
      <div class="tut-card-title"><span class="tut-card-icon">&#128279;</span> Git</div>
      <p class="tut-card-desc">Master Git — branching, rebasing, cherry-picking, conflict resolution, and collaborative workflows.</p>
      <span class="tut-card-tag">Tool</span>
    </a>
    <a class="tut-card" href="/Learn-Regex-in-One-Post-Complete-Tutorial-Anchors-Quantifiers-Lookarounds-Quick-Start/" data-title="Regex" data-desc="Regex tutorial anchors quantifiers lookarounds">
      <div class="tut-card-title"><span class="tut-card-icon">&#128270;</span> Regex</div>
      <p class="tut-card-desc">Master regular expressions — anchors, quantifiers, lookarounds, backreferences, and text processing.</p>
      <span class="tut-card-tag">Tool</span>
    </a>
    <a class="tut-card" href="/Learn-Bash-in-One-Post-Complete-Tutorial-Pipelines-Functions-Scripts-Quick-Start/" data-title="Bash" data-desc="Bash tutorial pipelines functions scripts">
      <div class="tut-card-title"><span class="tut-card-icon">&#128295;</span> Bash</div>
      <p class="tut-card-desc">Master Bash — pipelines, functions, scripting, process substitution, and shell automation.</p>
      <span class="tut-card-tag">Tool</span>
    </a>
  </div>
</section>

</div>

<script>
(function() {
  const searchInput = document.getElementById('tut-search');
  const searchCount = document.getElementById('tut-search-count');
  const noResults = document.getElementById('tut-no-results');
  const tabs = document.querySelectorAll('.tut-cat-tab');
  const sections = document.querySelectorAll('.tut-section');
  const cards = document.querySelectorAll('.tut-card');
  const featuredSection = document.getElementById('tut-featured');
  const featuredGrid = document.getElementById('tut-featured-grid');

  const featuredTutorials = [
    { title: 'Python', icon: '🐍', desc: 'The most popular language for data science, ML, and web dev', slug: '/Learn-Python-in-One-Post-Complete-Tutorial-Async-Type-Hints-Quick-Start/' },
    { title: 'JavaScript / TypeScript', icon: '📜', desc: 'The language of the modern web — from ES6 to type-safe TS', slug: '/Learn-JavaScript-TypeScript-in-One-Post-Complete-Tutorial-Async-Types-Quick-Start/' },
    { title: 'Docker', icon: '🐳', desc: 'Containerize any application with Dockerfiles and Compose', slug: '/Learn-Docker-in-One-Post-Complete-Tutorial-Dockerfile-Volumes-Compose-Quick-Start/' },
    { title: 'Kubernetes', icon: '⚙️', desc: 'Orchestrate containers at scale with production-ready Kubernetes', slug: '/Learn-Kubernetes-in-One-Post-Complete-Tutorial-Pods-Deployments-Services-Production-Quick-Start/' },
    { title: 'Git', icon: '🔧', desc: 'Master version control — branching, rebasing, and collaboration', slug: '/Learn-Git-in-One-Post-Complete-Tutorial-Branches-Rebase-Workflows-Quick-Start/' },
    { title: 'SQL', icon: '📊', desc: 'Query any relational database with confidence — joins to window functions', slug: '/Learn-SQL-in-One-Post-Complete-Tutorial-Joins-Window-Functions-Transactions-Quick-Start/' },
    { title: 'React / Next.js', icon: '⚛️', desc: 'Build modern web apps with components, hooks, and server rendering', slug: '/Learn-React-Next-js-in-One-Post-Complete-Tutorial-Components-Hooks-Server-Components-Quick-Start/' },
    { title: 'Machine Learning', icon: '🤖', desc: 'From linear regression to deep neural networks and beyond', slug: '/Learn-Machine-Learning-in-One-Post-Complete-Tutorial-Supervised-Unsupervised-Deep-Learning-Quick-Start/' },
    { title: 'Linux CLI', icon: '🖥️', desc: 'Navigate and automate Linux like a pro — essential command-line mastery', slug: '/Learn-Linux-CLI-in-One-Post-Complete-Tutorial-Files-Processes-Permissions-Quick-Start/' },
    { title: 'REST API', icon: '🔗', desc: 'Design and build production-ready REST APIs the right way', slug: '/Learn-REST-API-in-One-Post-Complete-Tutorial-Methods-Status-Codes-Production-Quick-Start/' }
  ];

  featuredTutorials.forEach(function(t, i) {
    const card = document.createElement('a');
    card.className = 'tut-featured-card';
    card.href = t.slug;
    card.innerHTML =
      '<div class="tut-featured-rank">' + (i + 1) + '</div>' +
      '<div class="tut-featured-icon">' + t.icon + '</div>' +
      '<div class="tut-featured-title">' + t.title + '</div>' +
      '<div class="tut-featured-desc">' + t.desc + '</div>';
    featuredGrid.appendChild(card);
  });

  function updateCounts() {
    sections.forEach(function(section) {
      const cat = section.getAttribute('data-cat');
      const grid = section.querySelector('.tut-grid');
      if (!grid) return;
      const count = grid.querySelectorAll('.tut-card').length;
      const countEl = document.getElementById('count-' + cat);
      if (countEl) countEl.textContent = count + ' tutorials';
    });
  }

  function filterCards(query) {
    query = query.toLowerCase().trim();
    let totalVisible = 0;

    cards.forEach(function(card) {
      const title = (card.getAttribute('data-title') || '').toLowerCase();
      const desc = (card.getAttribute('data-desc') || '').toLowerCase();
      const text = title + ' ' + desc;
      const match = !query || text.indexOf(query) !== -1;
      card.style.display = match ? '' : 'none';
      if (match) totalVisible++;
    });

    sections.forEach(function(section) {
      const grid = section.querySelector('.tut-grid');
      if (!grid) return;
      const visibleCards = grid.querySelectorAll('.tut-card');
      const anyVisible = Array.from(visibleCards).some(function(c) { return c.style.display !== 'none'; });
      section.style.display = anyVisible ? '' : 'none';
    });

    if (featuredSection) {
      if (!query) {
        featuredSection.style.display = '';
      } else {
        featuredSection.style.display = 'none';
      }
    }

    if (searchCount) {
      searchCount.textContent = totalVisible > 0 ? totalVisible + ' found' : '';
    }

    noResults.classList.toggle('show', totalVisible === 0);
  }

  function setFilter(filter) {
    tabs.forEach(function(t) {
      t.classList.toggle('active', t.getAttribute('data-filter') === filter);
    });

    sections.forEach(function(section) {
      const cat = section.getAttribute('data-cat');
      if (filter === 'all') {
        section.style.display = '';
      } else if (filter === 'featured') {
        section.style.display = 'none';
      } else {
        section.style.display = cat === filter ? '' : 'none';
      }
    });

    if (featuredSection) {
      featuredSection.style.display = filter === 'featured' ? '' : '';
    }
  }

  tabs.forEach(function(tab) {
    tab.addEventListener('click', function() {
      setFilter(tab.getAttribute('data-filter'));
      if (searchInput && searchInput.value) {
        filterCards(searchInput.value);
      } else {
        filterCards('');
      }
    });
  });

  ['input', 'keyup', 'search'].forEach(function(ev) {
    searchInput.addEventListener(ev, function() {
      filterCards(searchInput.value);
    });
  });

  updateCounts();
})();
</script>