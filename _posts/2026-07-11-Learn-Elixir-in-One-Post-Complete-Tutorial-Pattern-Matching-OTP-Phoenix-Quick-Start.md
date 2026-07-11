---
layout: post
title: "Learn Elixir in a Single Post: A Complete Elixir Tutorial from Pattern Matching and Pipes to OTP and Phoenix LiveView"
description: "A complete Elixir tutorial in one blog post. Covers the whole language in 5 stages: fundamentals (iex, atoms, tuples, lists, maps, pattern matching as assignment, immutability), functions + pipe (named/anonymous fns, arity, pattern matching in clauses, guards, the |> pipe operator), modules + structs (defmodule/def, structs, protocols for polymorphism, sigils), concurrency + OTP (processes, send/receive, GenServer, Supervisor, Task/Agent), and Phoenix + ecosystem (mix, hex, ExUnit, Phoenix, LiveView, Ecto, BEAM interop). Five diagrams, runnable code snippets, and a quick-start roadmap."
date: 2026-07-11
header-img: "img/post-bg.jpg"
permalink: /Learn-Elixir-in-One-Post-Complete-Tutorial-Pattern-Matching-OTP-Phoenix-Quick-Start/
featured-img: ai-coding-frameworks/ai-coding-frameworks
tags:
  - Elixir
  - OTP
  - Phoenix
  - Tutorial
  - Programming
  - BEAM
author: "PyShine"
---

# Learn Elixir in a Single Post: A Complete Elixir Tutorial from Pattern Matching and Pipes to OTP and Phoenix LiveView

Elixir is a modern, functional language that runs on the **BEAM** — Erlang's battle-tested virtual machine that powers WhatsApp, Discord, and a lot of telecom infrastructure. The BEAM gives Elixir its superpower: **millions of lightweight processes**, preemptive scheduling, and a "let it crash" supervision model that makes fault-tolerant, distributed systems tractable. Elixir wraps that power in a Ruby-like, approachable syntax and adds a first-class web framework (Phoenix) and real-time UI layer (LiveView).

This post teaches the whole language in five stages with runnable snippets. By the end you'll understand pattern matching and the pipe operator, modules and protocols, processes and the OTP supervision model, and how Phoenix/LiveView fit together.

We target **Elixir 1.16+**. Everything here runs on a current install.

## The Roadmap

![Elixir Roadmap](/assets/img/diagrams/elixir-tutorial/ex-roadmap.svg)

1. **Fundamentals** — `iex`, atoms, tuples, lists, maps, pattern matching, immutability
2. **Functions + Pipe** — named/anonymous functions, arity, pattern matching clauses, guards, `|>`
3. **Modules + Structs** — `defmodule`, structs, protocols (polymorphism), sigils
4. **Concurrency + OTP** — `spawn`, `send`/`receive`, `GenServer`, `Supervisor`, `Task`/`Agent`
5. **Phoenix + Ecosystem** — `mix`, `hex`, `ExUnit`, Phoenix, LiveView, Ecto, BEAM interop

## Stage 1 — Fundamentals

### A program

```elixir
IO.puts("Hello, Elixir!")
```

Run a file with `elixir hello.exs` (scripts use `.exs`; compiled modules use `.ex`). The best way to learn is `iex`, the REPL:

```bash
$ iex
iex> x = 5
5
iex> x + 2
7
iex> "Ada" |> String.upcase()
"ADA"
iex> h Enum.map            # built-in help
```

### Atoms, tuples, lists, maps

```elixir
:ok                  # atom — interned constant, like Ruby symbol / Erlang atom
:error
:active
:true; :false         # booleans are atoms

{:ok, 42}             # tuple — fixed-size, heterogeneous
{:error, "bad input"}
{1, "a", :x}

[1, 2, 3]             # list — linked, heterogeneous
[1 | [2, 3]]          # = [1, 2, 3]  — cons (head | tail)
[head | tail] = [1, 2, 3]   # head=1, tail=[2,3]

%{"name" => "Ada", 30 => "age"}    # map — heterogeneous keys
%{name: "Ada", age: 30}              # map with atom keys (shorthand)
m = %{name: "Ada"}
m.name                                # "Ada" (atom key accessor)
Map.get(m, :name)                     # "Ada"
Map.put(m, :age, 30)                  # new map (immutable)
```

- **Atoms** (`:ok`, `:error`) are interned constants — used for tags and enum-like values. Booleans are atoms.
- **Tuples** (`{:ok, 42}`) are fixed-size, heterogeneous, contiguous in memory — used for tagged results.
- **Lists** are linked lists (head/tail) — prepend (`[h | t]`) is O(1), random access is O(n).
- **Maps** are key-value dictionaries. **Everything is immutable** — `Map.put` returns a new map.

### Pattern matching is assignment

```elixir
# = is a pattern-match, not assignment. It asserts the shape and binds.
{a, b} = {1, 2}        # a = 1, b = 2
[head | tail] = [1, 2, 3]   # head = 1, tail = [2, 3]
%{name: n} = %{name: "Ada", age: 30}   # n = "Ada"

# Pin operator ^ — match against an existing value, don't rebind
x = 1
^x = 1                  # matches (1 == 1), no rebinding
^x = 2                  # MatchError! (1 != 2)

# In function heads and case
case {:ok, 42} do
  {:ok, v} -> v           # matches -> 42
  {:error, _} -> :err     # _ = ignore
end
```

**`=` is pattern matching, not assignment** — it asserts the shape and binds variables. This is the single most important idea in Elixir. The pin operator `^` matches against an existing value rather than rebinding. Pattern matching is used everywhere: function clauses, `case`, `receive`, `with`, destructuring.

### Immutability

```elixir
list = [1, 2, 3]
new = 0 .. list        # [0, 1, 2, 3]  — wait, .. is for ranges; use [0 | list]
new = [0 | list]       # [0, 1, 2, 3]  — list unchanged
list                   # still [1, 2, 3]
```

**All data is immutable.** "Updating" produces a new value; the original is unchanged. Because data is immutable and the BEAM shares it freely, concurrent code doesn't need locks for data — processes own their state and communicate by message passing.

### Control flow

```elixir
# if / else (expression form)
x = if n > 0, do: "pos", else: "neg"

# cond — multi-branch
cond do
  n < 0  -> "neg"
  n == 0 -> "zero"
  true   -> "pos"
end

# case — pattern matching
case file_result do
  {:ok, contents} -> process(contents)
  {:error, reason} -> handle(reason)
end

# with — chained pattern matching (great for pipelines)
with {:ok, user} <- fetch_user(id),
     {:ok, perms} <- fetch_perms(user),
     do: {:ok, perms}
# returns first non-matching clause or the final result
```

`cond` (multi-branch on conditions), `case` (pattern matching), and `with` (chained pattern matching — ideal for fallible pipelines that short-circuit). There's no `for` loop in the imperative sense — you use `Enum.map`/`Enum.each` (below).

## Stage 2 — Functions and the Pipe Operator

![Elixir Core](/assets/img/diagrams/elixir-tutorial/ex-pattern.svg)

### Named and anonymous functions

```elixir
# Anonymous function — fn ... end
square = fn x -> x * x end
square.(5)              # 25 — note the . (called on a variable-bound fn)

# Capture syntax — & shorthand
square2 = &(&1 * &1)     # &1, &2 are positional args
square2.(5)             # 25
add = &(&1 + &2)
add.(1, 2)              # 3

# Named functions live in modules
defmodule Math do
  def add(a, b), do: a + b          # one-liner with do:
  def square(x), do: x * x
  defp helper(x), do: x + 1         # defp = private
end
Math.add(1, 2)          # 3
```

Functions are identified by **name and arity** — `add/2` means "add taking 2 args". Anonymous functions need `.(...)` to call; module functions use `Module.fun(...)`. The capture syntax `&(&1 * &1)` is shorthand for `fn x -> x * x end`.

### Pattern matching in function clauses

```elixir
defmodule Factorial do
  def of(0), do: 1                      # base case clause
  def of(n) when n > 0, do: n * of(n - 1)   # recursive clause
end
Factorial.of(5)    # 120

# Multiple clauses — first matching one wins
defmodule Greet do
  def hello(:morning), do: "Good morning"
  def hello(:evening), do: "Good evening"
  def hello(_), do: "Hello"
end
Greet.hello(:morning)   # "Good morning"
```

Elixir dispatches on **pattern matching in the function head** — clauses are tried in order, first match wins. This replaces `if`/`switch` for many cases and makes function definitions look like specifications. **Guards** (`when n > 0`) add conditions to clauses.

### Default arguments and guards

```elixir
defmodule User do
  def greet(name, greeting \\ "Hi"), do: "#{greeting}, #{name}!"   # \\ = default
end
User.greet("Ada")                 # "Hi, Ada!"
User.greet("Ada", "Hey")          # "Hey, Ada!"

# Guards (limited expressions, must be allowed in guards)
def classify(n) when is_integer(n) and n > 0, do: :positive
def classify(n) when is_integer(n) and n < 0, do: :negative
def classify(0), do: :zero
```

`\\` provides default args. Guards (`when ...`) add predicates — but they're **restricted to a whitelist of safe expressions** (`is_integer`, comparison operators, arithmetic, etc.) so they can't have side effects or arbitrary logic.

### The pipe operator `|>`

```elixir
# Without pipe — nested, reads inside-out
String.split(String.downcase("Hello World"), " ")

# With pipe — linear, reads top-to-bottom
"Hello World" |> String.downcase() |> String.split(" ")
# ["hello", "world"]

# A longer pipeline
"  Hello, World!  "
|> String.trim()
|> String.downcase()
|> String.replace("!", ".")
|> String.split(", ")
```

The **pipe operator `|>`** takes the result of the left expression and passes it as the **first argument** to the right. This is Elixir's signature feature — it turns nested calls into a readable left-to-right pipeline, like a Unix pipe for function calls. **Write every transformation as a pipeline**; it's the idiomatic style and reads beautifully.

### Enum and Stream

```elixir
nums = [1, 2, 3, 4, 5]

Enum.map(nums, fn x -> x * 2 end)        # [2, 4, 6, 8, 10]
Enum.filter(nums, fn x -> x > 2 end)    # [3, 4, 5]
Enum.reduce(nums, 0, fn x, acc -> acc + x end)   # 15
Enum.find(nums, fn x -> x > 3 end)       # 4
Enum.group_by(nums, fn x -> rem(x, 2) end)  # %{0 => [2,4], 1 => [1,3,5]}
Enum.sort(nums)
Enum.chunk_every(nums, 2)               # [[1,2],[3,4],[5]]

# Streams — lazy (good for large/infinite)
1..10_000_000 |> Stream.map(&(&1 * 2)) |> Enum.take(3)   # [2, 4, 6]
```

`Enum` is the eager collection module — works on lists, maps, ranges, anything enumerable. `Stream` is the lazy version — only computes as values are pulled (great for huge or infinite sequences). Almost all collection work goes through `Enum`/`Stream`.

## Stage 3 — Modules, Structs, Protocols, Sigils

![Elixir Features](/assets/img/diagrams/elixir-tutorial/ex-features.svg)

### Modules

```elixir
defmodule Counter do
  @moduledoc "A simple counter module."    # module doc
  @default_step 1                            # module attribute (compile-time const)

  @doc "Increment by the default step."
  @spec inc(integer()) :: integer()
  def inc(n), do: n + @default_step

  defp helper(n), do: n * 2                  # private function
end

Counter.inc(5)    # 6
```

Modules group related functions. `@moduledoc`/`@doc` are doc strings (shown by `h Module.fun` in `iex` and by ExDoc). `@spec` adds type specs (used by Dialyzer for static analysis). Module attributes (`@default_step`) are compile-time constants.

### Structs — typed maps

```elixir
defmodule Point do
  defstruct [:x, :y]            # defines a struct with fields x, y
end

p = %Point{x: 1, y: 2}
p.x                              # 1
%Point{p | y: 5}                 # update — new struct, x unchanged
%Point{} = p                     # pattern match on struct type

# Structs are maps with a __struct__ field
%{__struct__: Point, x: 1, y: 2} === p   # true (structs ARE maps)
```

A **struct** is a map with a fixed set of keys and a `__struct__` field tagging its type. It gives you compile-time key checking and a named type, while remaining a map underneath.

### Protocols — polymorphism

```elixir
defprotocol Size do
  @doc "Calculates the size of a data structure."
  def size(data)
end

defimpl Size, for: List do
  def size(list), do: length(list)
end

defimpl Size, for: Map do
  def size(map), do: map_size(map)
end

defimpl Size, for: Atom do
  def size(_), do: 0
end

Size.size([1, 2, 3])   # 3
Size.size(%{a: 1})      # 1
Size.size(:hello)        # 0
```

A **protocol** is a dispatch mechanism: define a function (`Size.size/1`) and provide implementations for specific types. This is Elixir's polymorphism — `Enum.map` works on lists, maps, ranges because they all implement the `Enumerable` protocol. It's the equivalent of Haskell typeclasses / Rust traits, dispatching on the runtime type.

### Sigils

```elixir
~s(hello)             # "hello"  — string sigil (no escaping needed)
~s(He said "hi")      # "He said \"hi\"" — actually: "He said \"hi\""
~w(a b c)              # ["a", "b", "c"]  — word list
~w(a b c)a            # [:a, :b, :c]  — atom list (suffix 'a')
~r/foo|bar/           # regex
~c"abc"                # charlist
~B"""
multi-line
heredoc
"""                     # binary sigil
```

**Sigils** are `~` prefixed literals — `~s` strings, `~w` word lists, `~r` regex, `~c` charlists, plus modifiers (`a` for atoms). They avoid escaping (`~s(He said "hi")` works without `\"`) and let you define custom sigils via macros.

### Metaprogramming with macros

```elixir
# quote — AST representation
quote do: 1 + 2            # {:+, [], [1, 2]}

# unquote — splice into a quote
n = 5
quote do: 1 + unquote(n)    # {:+, [], [1, 5]}

# defmacro — compile-time code generation
defmodule Unless do
  defmacro unless(condition, do: body) do
    quote do
      if !unquote(condition), do: unquote(body)
    end
  end
end
Unless.unless false do
  IO.puts("runs")
end
```

Elixir is homoiconic — code is represented as nested tuples (`{:+, [], [1, 2]}` is `1 + 2`). `quote` captures code as AST, `unquote` splices values into it, and `defmacro` generates code at compile time. This is how `if`, `unless`, `defmodule`, and many DSLs are built — they're macros, not syntax. Powerful but use sparingly; most code is regular functions.

## Stage 4 — Concurrency and OTP

![Elixir OTP](/assets/img/diagrams/elixir-tutorial/ex-otp.svg)

### Processes — the BEAM's lightweight actors

```elixir
# Spawn a process — runs the given function concurrently
pid = spawn(fn ->
  receive do
    {:hello, from} -> send(from, :hi)
  end
end)

send(pid, {:hello, self()})      # self() = current PID
receive do
  :hi -> IO.puts("got hi")
after
  1000 -> IO.puts("timeout")     # after clause = timeout
end

# BEAM processes are NOT OS threads — millions can run at once
# Each has its own heap and GC (no stop-the-world)
```

Elixir's processes are **BEAM actors** — isolated, communicate by message passing, scheduled preemptively across CPU cores. They're **ultra-lightweight** (~2KB each), so you can have millions. There's no shared mutable state; concurrency is by message passing, which eliminates most data races by construction.

### `send` / `receive`

```elixir
# Every process has a mailbox; receive pattern-matches against it
receive do
  {:ok, v} -> v
  {:error, _} -> :error
  _ -> :unknown
after
  5000 -> :timeout
end
```

`send(pid, msg)` puts a message in a process's mailbox; `receive` pattern-matches messages out of it. Messages are asynchronous. The `after` clause gives a timeout. This is the primitive that GenServer (below) builds on.

### GenServer — the server behavior

```elixir
defmodule Counter do
  use GenServer

  # Client API
  def start_link(initial), do: GenServer.start_link(__MODULE__, initial, name: __MODULE__)
  def inc, do: GenServer.cast(__MODULE__, :inc)            # async
  def get, do: GenServer.call(__MODULE__, :get)              # sync

  # Server callbacks
  @impl true
  def init(initial), do: {:ok, initial}

  @impl true
  def handle_call(:get, _from, state), do: {:reply, state, state}
  @impl true
  def handle_cast(:inc, state), do: {:noreply, state + 1}
end

Counter.start_link(0)
Counter.inc()
Counter.get()   # 1
```

A **GenServer** is the standard stateful server abstraction: a process that receives requests (calls = sync, casts = async), updates its state in `handle_call`/`handle_cast`/`handle_info`, and replies. It standardizes the boilerplate of receive loops, timeouts, and debugging. Use GenServer whenever you need stateful concurrency.

### Supervisor — let it crash

```elixir
children = [
  {Counter, 0},
  {AnotherWorker, []}
]

Supervisor.start_link(children, strategy: :one_for_one)
# If a child crashes, the supervisor restarts it automatically.
# Strategies: :one_for_one, :rest_for_one, :one_for_all
```

The **Supervisor** manages a tree of processes and **restarts them when they crash**. The "let it crash" philosophy: instead of writing defensive code that handles every error, write processes that crash on unexpected inputs and let a supervisor restart them to a known-good state. Combined with supervision trees (supervisors of supervisors), this makes systems self-healing.

### Task and Agent — common patterns

```elixir
# Task — async work with a result
task = Task.async(fn -> slow_computation() end)
result = Task.await(task)        # blocks, returns result

# Agent — simple shared state
{:ok, agent} = Agent.start_link(fn -> 0 end)
Agent.update(agent, fn s -> s + 1 end)
Agent.get(agent, fn s -> s end)   # 1
```

`Task` is for fire-and-await async work (concurrent fetches, parallel maps). `Agent` is a minimal GenServer for plain shared state. These cover most everyday needs without writing a full GenServer.

## Stage 5 — Phoenix and the Ecosystem

![Elixir Toolchain](/assets/img/diagrams/elixir-tutorial/ex-toolchain.svg)

### mix and hex

```bash
mix new my_app           # scaffold a project (mix.exs, lib/, test/)
mix new my_app --sup     # with a supervision tree
mix compile
mix test                 # run ExUnit tests
mix run lib/my_app.exs   # run a script in the project context

mix deps.get             # fetch dependencies from hex.pm
mix deps.compile
mix phx.new my_web       # scaffold a Phoenix web app
```

`mix` is the build tool (scaffold, compile, test, deps, custom tasks). `hex` is the package manager (deps live in `mix.exs`). The BEAM compiles `.ex` files to `.beam` bytecode.

A `mix.exs`:

```elixir
defmodule MyApp.MixProject do
  use Mix.Project

  def project do
    [app: :my_app, version: "0.1.0", elixir: "~> 1.16",
     deps: deps()]
  end

  def application do
    [extra_applications: [:logger], mod: {MyApp.Application, []}]
  end

  defp deps do
    [{:phoenix, "~> 1.7"}, {:ecto_sql, "~> 3.10"}]
  end
end
```

### ExUnit — tests with doctests

```elixir
defmodule MathTest do
  use ExUnit.Case, async: true       # async: tests run concurrently

  test "addition" do
    assert 1 + 1 == 2
    assert {:ok, _} = {:ok, 42}
    refute 1 == 2
  end

  doctest Math                         # runs examples from @doc strings
end
```

ExUnit is the testing framework — `assert`/`refute`/`assert_raise`. **Doctests** run examples from `@doc` strings, so your documentation stays correct (the tests fail if an example stops compiling/returning what it claims).

### Phoenix — the web framework

```elixir
# lib/my_app_web/router.ex
defmodule MyAppWeb.Router do
  use MyAppWeb, :router

  pipeline :browser do
    plug :accepts, ["html"]
  end

  scope "/", MyAppWeb do
    pipe_through :browser
    get "/", PageController, :index
    live "/counter", CounterLive        # LiveView route
    resources "/posts", PostController
  end
end
```

Phoenix is the Rails of Elixir — MVC, routing, plugs (middleware), Ecto (ORM/DB), channels (WebSockets), and **LiveView** (server-rendered, real-time UI without writing JS).

### LiveView — real-time UI without JS

```elixir
defmodule MyAppWeb.CounterLive do
  use MyAppWeb, :live_view

  @impl true
  def mount(_params, _session, socket) do
    {:ok, assign(socket, count: 0)}
  end

  @impl true
  def render(assigns) do
    ~H"""
    <div>
      Count: <%= @count %>
      <button phx-click="inc">+</button>
    </div>
    """
  end

  @impl true
  def handle_event("inc", _, socket) do
    {:noreply, update(socket, :count, &(&1 + 1))}
  end
end
```

**LiveView** renders HTML on the server, then maintains a WebSocket; user events (`phx-click`) go to the server, which re-renders and pushes only the changed DOM back. You get reactive, SPA-like UIs without writing JavaScript — the server is the source of truth.

### Ecto — database

```elixir
defmodule MyApp.Post do
  use Ecto.Schema
  import Ecto.Changeset

  schema "posts" do
    field :title, :string
    field :body, :string
    timestamps()
  end

  def changeset(post, attrs) do
    post
    |> cast(attrs, [:title, :body])
    |> validate_required([:title])
    |> validate_length(:title, max: 100)
  end
end

# Querying
import Ecto.Query
from p in Post, where: p.title == "x", select: p
Repo.all(query)
```

**Ecto** is the database wrapper — schemas map to tables, changesets handle validation, and the query DSL is composable. It's not a traditional ORM (no lazy loading, no automatic associations) — it's more explicit, which the community considers a feature.

### Tooling

- **`iex`** — the REPL; `iex -S mix` loads your project into it.
- **`mix`** — build tool, project generator, task runner.
- **`hex`** — package manager; deps in `mix.exs`.
- **`ExUnit`** — testing with doctests.
- **`Phoenix`** — web framework; **LiveView** for real-time UI.
- **`Ecto`** — database + changesets.
- **`Nx` / `Axon`** — numerical computing / ML on Elixir.
- **`Nerves`** — embedded Elixir (Raspberry Pi, etc.).
- **Dialyzer / Credo / Dialyxir** — static analysis and linting.
- **Erlang interop** — all of Erlang/OTP is callable (`:gen_server`, `:ets`, etc.).

## A Quick-Start Checklist

1. **Install Elixir** (includes Erlang/OTP); use `iex` constantly.
2. **`mix new my_app --sup`** to start; `mix.exs` for deps, `mix test` for tests.
3. **Pattern matching is assignment** — embrace `case`, `with`, multi-clause functions.
4. **`|>` everywhere** — write transformations as pipelines.
5. **`Enum` + `Stream`** for collection work; `Enum` for eager, `Stream` for lazy/large.
6. **Structs for typed data, protocols for polymorphism.**
7. **Processes + GenServer** for stateful concurrency; **Supervisor** for fault tolerance.
8. **"Let it crash"** — don't over-defend; let supervisors restart crashed processes.
9. **ExUnit + doctests** in CI; run `mix test`.
10. **Phoenix + LiveView** for web — real-time UIs without much JS.

## Common Pitfalls

- **Anonymous function call syntax** — `f.(x)`, not `f(x)`. The dot is required for variable-bound functions.
- **List vs tuple performance** — lists are O(n) for length and random access; tuples are O(1). Use the right one.
- **Strings vs charlists** — `"hello"` is a UTF-8 binary; `'hello'` is a charlist (list of integers). Erlang code often wants charlists.
- **Atom exhaustion** — atoms aren't garbage-collected; never convert untrusted user input to atoms (`String.to_atom`). Use `String.to_existing_atom`.
- **Forgetting `handle_call`/`handle_cast` pattern match** — a GenServer with no matching clause crashes the process (which the supervisor restarts — by design, but you probably wanted to handle it).
- **Blocking the VM** — long-running CPU work in a process blocks its scheduler. Use `Task.async` or chunk the work.
- **Shared state via Agent** — fine for simple cases, but a GenServer is better for anything complex.
- **`Enum` on a huge list** — eagerly builds intermediate lists. Use `Stream` for large pipelines.
- **Pin operator confusion** — `^x = y` matches `y` against `x`; without `^`, `x = y` rebinds `x` to `y`.
- **`with` short-circuits on the first non-match** — that's the point, but make sure your error tuples are tagged (`{:error, _}`) so they're distinguishable.

## What to Learn Next

- **Elixir docs / Hexdocs** — [hexdocs.pm/elixir](https://hexdocs.pm/elixir/) the official, comprehensive docs (every Elixir library).
- **Elixir School** — [elixirschool.com](https://elixirschool.com/) free, structured lessons.
- **Programming Elixir ≥ 1.6** by Dave Thomas — the approachable intro (the "Joy" book).
- **Elixir in Action** by Saša Jurić — deeper, including the BEAM and OTP (highly recommended).
- **Designing Elixir Systems with OTP** by Bruce Tate & James Gray — supervision, fault tolerance.
- **Programming Phoenix ≥ 1.4** by Chris McCord (Phoenix's creator) — the web framework.
- **Programming Ecto** — the database library.
- **Erlang docs** — [erlang.org/doc](https://www.erlang.org/doc) Elixir interops with all of Erlang; learn `:gen_server`, `:ets`, `:dets`, `:mnesia`.

Elixir's pitch is the BEAM: a runtime built for concurrent, fault-tolerant, distributed systems, with a syntax approachable enough to feel like Ruby. Pattern matching, the pipe operator, and processes/GenServer are the daily-work tools; supervision and "let it crash" are the architectural payoff. Once you internalize that processes own state and communicate by messages, the language clicks.

Good luck — and `iex -S mix`.

**Resources:**

- Elixir: [https://elixir-lang.org/](https://elixir-lang.org/)
- Hexdocs: [https://hexdocs.pm/](https://hexdocs.pm/)
- Phoenix: [https://phoenixframework.org/](https://phoenixframework.org/)
- Hex: [https://hex.pm/](https://hex.pm/)
- Elixir School: [https://elixirschool.com/](https://elixirschool.com/)