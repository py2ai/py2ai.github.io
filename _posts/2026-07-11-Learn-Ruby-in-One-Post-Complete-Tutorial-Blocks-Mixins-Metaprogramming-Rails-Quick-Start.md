---
layout: post
title: "Learn Ruby in a Single Post: A Complete Ruby Tutorial from Blocks and Mixins to Metaprogramming and Rails"
description: "A complete Ruby tutorial in one blog post. Covers the whole language in 5 stages: fundamentals (everything is an object, variables, symbols, control flow), OOP (classes, modules, mixins, inheritance), blocks + iterators (blocks, yield, Proc/lambda, Enumerable), metaprogramming (open classes, method_missing, define_method, eval), and the ecosystem (irb, gem, bundler, RSpec, Rails, ActiveRecord). Five diagrams, runnable code snippets, and a quick-start roadmap."
date: 2026-07-11
header-img: "img/post-bg.jpg"
permalink: /Learn-Ruby-in-One-Post-Complete-Tutorial-Blocks-Mixins-Metaprogramming-Rails-Quick-Start/
featured-img: ai-coding-frameworks/ai-coding-frameworks
tags:
  - Ruby
  - Rails
  - Tutorial
  - Programming
  - Metaprogramming
  - Learn to Code
author: "PyShine"
---

# Learn Ruby in a Single Post: A Complete Ruby Tutorial from Blocks and Mixins to Metaprogramming and Rails

Ruby's philosophy is **"programmer happiness"** — Matz designed it to read like natural language, favor expressiveness over brevity, and make common things beautiful. The result is a dynamically typed, object-oriented language where everything is an object (even numbers and classes), blocks are first-class, and metaprogramming is routine. Rails turned it into the dominant startup-web stack for a decade.

This post teaches the whole language in five stages with runnable snippets. By the end you'll understand the object model, blocks/procs/lambdas, modules and mixins, metaprogramming, and the Rails/gem ecosystem.

We target **Ruby 3.x** (everything here runs on 3.0+; notes flag 3.0+ features).

## The Roadmap

![Ruby Roadmap](/assets/img/diagrams/ruby-tutorial/rb-roadmap.svg)

1. **Fundamentals** — everything is an object, variables, symbols, control flow, collections
2. **OOP** — classes, `initialize`, `@ivar`, `attr_accessor`, inheritance, modules
3. **Blocks + Iterators** — blocks, `yield`, `Proc`/`lambda`, `Enumerable`
4. **Metaprogramming** — open classes, `method_missing`, `define_method`, `eval`
5. **Ecosystem** — `irb`, `gem`, `bundler`, `RSpec`, `Rails`, `ActiveRecord`

## Stage 1 — Fundamentals

### A program

```ruby
puts "Hello, Ruby!"
```

That's the whole program. Run it with `ruby hello.rb` or in `irb` (the REPL). No `main`, no semicolons, no boilerplate — Ruby reads top to bottom.

### Everything is an object

```ruby
5.class          # Integer
"hi".class       # String
[].class         # Array
nil.class        # NilClass
5.send(:+, 3)    # 8 — even + is a method

5.times { |i| print i }   # 01234 — Integer responds to times
-5.abs            # 5
3.14.round       # 3
"ada".upcase      # "ADA"
[1, 2, 3].length  # 3
```

**Everything is an object** — there are no primitives. `5` is an `Integer` with methods; `nil` is a `NilClass` object; even `+` is a method you can call via `send`. This uniformity is the heart of Ruby's metaprogramming.

### Variables, symbols, nil

```ruby
name = "Ada"        # local variable (snake_case)
@count = 0          # instance variable
@@total = 0         # class variable
$global = 0         # global (avoid)
CONST = 3.14        # constant (ALL_CAPS convention; warning if reassigned)

:symbol             # immutable symbol — "interned string"
status = :active    # often used as enum/identifier
:symbol.object_id == :symbol.object_id  # true — same object

nil                 # "no value" — only nil and false are falsy; 0 and "" are truthy!
```

Symbols (`:foo`) are immutable, interned identifiers — use them for hash keys, enum-like values, and method names. Unlike Python/JS, **only `nil` and `false` are falsy** — `0`, `""`, and `[]` are all truthy. This trips newcomers.

### Collections

```ruby
nums = [1, 2, 3]                    # Array (ordered, index)
nums << 4                          # append (push); returns the array
nums.push(4); nums.first; nums.last
nums.map { |n| n * n }              # [1, 4, 9, 16] — new array
nums.each { |n| puts n }            # iterate
nums.select { |n| n.even? }         # filter

h = { name: "Ada", age: 30 }        # Hash (symbol keys shorthand)
h[:name]                            # "Ada"
h[:missing]                         # nil
h.fetch(:missing, "default")        # "default" — explicit default
h.transform_values { |v| v.to_s }   # transform

# Range
(1..5).to_a    # [1, 2, 3, 4, 5]   inclusive
(1...5).to_a   # [1, 2, 3, 4]      exclusive
```

Arrays and hashes are the workhorses. Methods end with `?` for predicates (`even?`, `empty?`) and `!` for in-place mutation (`map` vs `map!`). Prefer the non-bang version unless you want to mutate.

### Control flow

```ruby
if x > 0
  "positive"
elsif x == 0
  "zero"
else
  "negative"
end

# Modifier form — Ruby's signature conciseness
puts "big" if x > 100
x = 0 unless defined? x

# unless = if not
unless done? then keep_going end

case status
when :active then "online"
when :inactive, :banned then "offline"
else "unknown"
end

# Loops — but prefer iterators
while cond do ... end
until cond do ... end
loop { break if done? }   # infinite
3.times { |i| puts i }    # idiomatic — preferred over for
```

Ruby's `if`/`unless` are expressions (they return values); the modifier form (`statement if condition`) reads like English. **Prefer iterators** (`each`, `times`, `map`) over `for`/`while` — they're idiomatic and avoid loop-variable scoping surprises.

## Stage 2 — OOP

![Ruby Object Model](/assets/img/diagrams/ruby-tutorial/rb-object.svg)

### Classes

```ruby
class Counter
  def initialize(start = 0)   # constructor — invoked by Counter.new
    @count = start             # @count is an instance variable
  end

  def inc
    @count += 1
    self                       # return self for chaining: c.inc.inc
  end

  def count
    @count                     # getter (no parens on call)
  end

  def to_s
    "Counter(#{@count})"
  end
end

c = Counter.new(5)
c.inc.inc
puts c.count   # 7
```

`Counter.new` allocates and calls `initialize`. Instance variables (`@foo`) are private by default — accessed only through methods. Methods with no arguments are called without parens (`c.count`, not `c.count()`), which gives Ruby its natural-language feel.

### attr_accessor / reader / writer

```ruby
class Person
  attr_accessor :name, :age      # generates getter + setter
  attr_reader :id                 # getter only
  attr_writer :email              # setter only

  def initialize(name, age, id)
    @name, @age, @id = name, age, id
  end
end

p = Person.new("Ada", 30, 1)
p.name           # "Ada"  (reader generated)
p.age = 31       #        (writer generated)
p.id             # 1
```

`attr_accessor :name` writes a getter and setter for you — the equivalent of a dozen lines of boilerplate. This is metaprogramming at the most everyday level: `attr_accessor` literally calls `define_method` to generate the methods.

### Inheritance and super

```ruby
class Animal
  def sound = "..."              # endless method (3.0+) — one-liner
  def describe = "#{self.class}: #{sound}"
end

class Dog < Animal               # single inheritance
  def sound = "woof"              # override
end

Dog.new.describe   # "Dog: woof"
Dog.new.sound       # "woof"
```

Ruby has **single inheritance** (one parent, `<`). `super` calls the parent's version; `super` (no parens) forwards all args, `super()` passes none. The endless method (`def f = expr`, 3.0+) makes one-liners elegant.

### Modules, mixins, include vs extend

Ruby solves the single-inheritance limit with **modules** — namespaces that can be mixed into classes:

```ruby
module Walkable
  def walk
    "walking"
  end
end

module Swimmable
  def swim
    "swimming"
  end
end

class Duck < Animal
  include Walkable         # instance methods
  include Swimmable
end

Duck.new.walk   # "walking"
Duck.new.swim   # "swimming"
```

- **`include Module`** — adds the module's methods as **instance methods** of the class.
- **`extend Module`** — adds them as **class methods** (singleton methods on the class).
- **`prepend Module`** — inserts the module *before* the class in the method lookup chain (so the module's `super` calls the class method).

This is how `Enumerable` (gives you `map`/`select`/`reduce` when you define `each`) and `Comparable` (gives you `<`/`<=>` when you define `<=>`) work — they're mixins.

```ruby
class MyCollection
  include Enumerable
  def initialize(items) = @items = items
  def each(&block) = @items.each(&block)
end

MyCollection.new([1, 2, 3]).map { |n| n * 2 }   # [2, 4, 6] — Enumerable gives map
```

## Stage 3 — Blocks and Iterators

Blocks are Ruby's signature feature — they make iteration, callbacks, and DSLs expressive.

![Ruby Features](/assets/img/diagrams/ruby-tutorial/rb-features.svg)

### Blocks

```ruby
# Two syntaxes for blocks
[1, 2, 3].each { |n| puts n }            # inline braces — single line
[1, 2, 3].each do |n|                      # do...end — multi-line
  puts n
end

# Block returns its last evaluated value
summed = [1, 2, 3].map { |n| n * n }       # [1, 4, 9] — last expr is the value
```

A **block** is an anonymous chunk of code attached to a method call. The method decides whether/when to run it via `yield`. Blocks come in two syntaxes: `{ |x| ... }` (inline, conventionally single-line) and `do |x| ... end` (multi-line).

### yield and &block

```ruby
def twice
  yield
  yield
end

twice { puts "hi" }   # hi\nhi

def with_result
  result = yield(5)
  puts "got: #{result}"
end

with_result { |n| n * 2 }   # got: 10 — block takes an arg, returns value

# &block — block as an explicit Proc object
def apply(&block)
  block.call(42)
end
apply { |n| n + 1 }   # 43

# Pass a block onward
def delegate(&b) = other_method(&b)
```

`yield` invokes the block (passing args, receiving the return value). `&block` captures the block as a `Proc` object you can store, pass around, or forward. Most code uses `yield`; reach for `&block` when you need to store or forward the block.

### Proc, lambda, and the difference

```ruby
# Proc — block wrapped in an object
p = Proc.new { |x| x * 2 }
p.call(5)        # 10
p.(5)            # 10 — .() syntax
p[5]             # 10 — [] syntax

# Lambda — stricter Proc
l = lambda { |x| x * 2 }
l = ->(x) { x * 2 }   # stabby lambda literal (->)

# Differences:
# 1. lambda checks arity (extra args -> error); Proc ignores them
p.call(1, 2)   # 2 — Proc ignores the extra arg
->(x){ x }.call(1, 2)  # ArgumentError — lambda is strict

# 2. return inside a lambda returns from the lambda; in a Proc it returns from the ENCLOSING method
def proc_test
  p = Proc.new { return 1 }
  p.call
  return 2   # never reached — Proc's return exits the method
end
proc_test     # 1
```

**`Proc`** is lenient (like a block: ignores extra args, `return` exits the enclosing method). **`lambda`** is strict (checks args, `return` exits only the lambda — behaves like a regular method). Rule of thumb: **use `lambda` when you want method-like semantics; `Proc` when you want block-like.**

### Iterators and Enumerable

```ruby
nums = [1, 2, 3, 4, 5]

nums.each { |n| ... }                # iterate
nums.map { |n| n * n }               # transform -> new array
nums.select { |n| n.even? }          # filter
nums.reject { |n| n.even? }           # inverse filter
nums.reduce(0) { |sum, n| sum + n }  # fold -> 15
nums.find { |n| n > 3 }               # first match -> 4
nums.group_by(&:even?)                # {true=>[2,4], false=>[1,3,5]}
nums.sort                              # sorted
nums.max_by { |n| -n }                # 1 — max by key
nums.tally                             # {1=>1, 2=>1, ...} (3.0+)

# Symbol#to_proc — the & shorthand
nums.map(&:to_s)                      # ["1", "2", ...]  == nums.map { |n| n.to_s }

# Lazy (for infinite or large sequences)
(1..).lazy.map { |n| n * n }.first(5)   # [1, 4, 9, 16, 25] — infinite range, take 5
```

`Enumerable` is the magic mixin: define `each` on your class, `include Enumerable`, and you get `map`/`select`/`reduce`/`sort`/`min`/`max`/`group_by`/`tally`/... for free. The `&:method_name` shorthand (`&:to_s`) is `Symbol#to_proc` — converts `:to_s` to `{ |x| x.to_s }`.

## Stage 4 — Metaprogramming

Ruby's classes are **open** — you can add or redefine methods on any class, including built-ins, at any time. This is powerful and dangerous.

![Ruby Metaprogramming](/assets/img/diagrams/ruby-tutorial/rb-metaprogram.svg)

### Open classes (monkey-patching)

```ruby
class String
  def shout
    upcase + "!"
  end
end

"hello".shout   # "HELLO!" — defined on String itself
```

You can re-open any class and add methods. This is how Rails extends `String` with things like `"path".classify`. The danger: your patch affects *all* code in the process. Use **`refinements`** for scoped patches that only apply where you `using` them:

```ruby
module ShoutRefinement
  refine String do
    def shout = upcase + "!"
  end
end

using ShoutRefinement
"hello".shout   # "HELLO!"
# outside the `using` scope, shout is undefined
```

### method_missing

When you call a method that doesn't exist, Ruby calls `method_missing` before raising `NoMethodError`. Override it to build dynamic APIs:

```ruby
class Dynamic
  def method_missing(name, *args)
    if name.to_s.end_with?("_please")
      "you said #{name}"
    else
      super   # fall back to default — raises NoMethodError
    end
  end

  def respond_to_missing?(name, include_private = false)
    name.to_s.end_with?("_please") || super
  end
end

d = Dynamic.new
d.help_please   # "you said help_please"
d.help          # NoMethodError (super was called)
```

Always override `respond_to_missing?` alongside `method_missing` so `respond_to?` reports the dynamic methods correctly. `method_missing` is how Active Record builds dynamic finders like `find_by_email_and_status`.

### define_method and attr_accessor

```ruby
class Config
  [:host, :port, :timeout].each do |attr|
    define_method(attr) { instance_variable_get("@#{attr}") }
    define_method("#{attr}=") { |v| instance_variable_set("@#{attr}", v) }
  end
end

c = Config.new
c.host = "localhost"
c.host   # "localhost"

# attr_accessor is literally define_method under the hood
class Person
  attr_accessor :name   # generates name + name= via define_method
end
```

`define_method` creates a method programmatically — the foundation of `attr_accessor`, validations, and much of Rails' magic.

### eval and dynamic dispatch

```ruby
# send — call a method by name (string or symbol)
"hi".send(:upcase)   # "HI"
obj.send("method_#{var}=", value)   # dynamic setter

# eval — run a string as Ruby (use sparingly; security risk on user input)
result = eval("1 + 2")   # 3

# class_eval — evaluate code in the context of a class
SomeClass.class_eval do
  define_method(:new_method) { "dynamic" }
end
```

`eval` runs arbitrary strings as Ruby code — powerful but a **security risk** with user input (code injection). Reach for `send` (method dispatch by name) or `define_method` first; reserve `eval` for build-time codegen or trusted configs.

## Stage 5 — Ecosystem and Rails

![Ruby Toolchain](/assets/img/diagrams/ruby-tutorial/rb-toolchain.svg)

### The basics

```bash
ruby file.rb            # run a script
irb                     # REPL (use pry for a better one)
ri Array#map            # built-in docs

gem install rails        # install a gem (package)
gem list                 # list installed gems
gem sources              # registry (default: rubygems.org)

bundle install           # install Gemfile deps
bundle exec rake test    # run a command in the Gemfile context
bundle exec rspec         # run RSpec tests
```

### Bundler and the Gemfile

```ruby
# Gemfile
source "https://rubygems.org"

gem "rails", "~> 7.1"
gem "pg"                          # PostgreSQL adapter
gem "puma"                        # web server
gem "rspec-rails", group: :test

group :development do
  gem "rubocop"
  gem "pry-byebug"
end
```

`bundle install` locks versions in `Gemfile.lock`. **Always run commands via `bundle exec`** so they use the locked versions, not whatever's globally installed.

### RSpec — behavior-driven testing

```ruby
# spec/counter_spec.rb
require "rspec"
require_relative "../lib/counter"

RSpec.describe Counter do
  let(:counter) { Counter.new(5) }    # fresh for each example

  describe "#inc" do
    it "increments the count" do
      expect { counter.inc }.to change { counter.count }.by(1)
    end

    it "returns self for chaining" do
      expect(counter.inc).to be(counter)
    end
  end
end
```

RSpec reads like English — `expect(...).to`, `change { }.by(n)`, `be`, `include`. It's the dominant Ruby test framework; Minitest is the smaller alternative (Rails ships with it by default).

### Rails — the framework

```bash
gem install rails
rails new myapp --database=postgresql
cd myapp
rails server    # http://localhost:3000
rails generate scaffold Post title:string body:text
rails db:migrate
rails test
```

Rails is **convention over configuration** — a full-stack MVC framework with:

- **Active Record** — ORM; models map to tables, with migrations, validations, associations. `Post.where(published: true).order(created_at: :desc)`.
- **Action Pack** — routing (`get "/posts/:id", to: "posts#show"`), controllers, views (ERB / Slim).
- **Asset pipeline** — Sprockets / Propshaft / esbuild for JS/CSS.
- **Hotwire** — modern Turbo + Stimulus for SPA-like apps without an SPA.
- **Active Job** — background jobs; Action Mailer; Action Cable (WebSockets).

A Rails model:

```ruby
# app/models/post.rb
class Post < ApplicationRecord
  has_many :comments, dependent: :destroy
  belongs_to :user

  validates :title, presence: true, length: { maximum: 100 }
  scope :published, -> { where(published: true) }
end

Post.published.recent.limit(10)   # chained scopes -> SQL
```

### Tooling

- **`irb` / `pry`** — REPLs; `pry` has better navigation (`ls`, `show-source`, `binding.pry` breakpoints).
- **`gem` / `bundler`** — package management; `bundle exec` for locked versions.
- **`rake`** — task runner; `rake -T` lists tasks.
- **`rubocop`** — lint + style; run in CI; auto-correct with `-a`.
- **`rspec` / `minitest`** — testing.
- **`standardrb`** — opinionated RuboCop config, zero setup.

## A Quick-Start Checklist

1. **Install Ruby 3.x** via `rbenv` or `rvm` (manage versions per project).
2. **Use `bundler`** — every project has a `Gemfile`; run `bundle exec` for commands.
3. **Prefer iterators** (`each`, `map`, `select`) over `for`/`while`.
4. **Use `attr_accessor`** instead of hand-written getters/setters.
5. **Mixins over deep inheritance** — `include Enumerable` after defining `each`.
6. **Lambdas for method-like, Procs for block-like** — default to `lambda` (stabby `->`).
7. **Test with RSpec** (or Minitest); write one spec per behavior.
8. **Metaprogram with restraint** — reach for `define_method`/`send` before `eval`; use refinements for scoped patches.
9. **Run `rubocop`** in CI; use `binding.pry` for debugging.

## Common Pitfalls

- **`0` and `""` are truthy** — only `nil` and `false` are falsy. `if 0` runs the branch.
- **`!` (bang) mutates in place** — `arr.map { }` returns a new array; `arr.map! { }` mutates. Read the docs.
- **Monkey-patch blast radius** — re-opening `String` affects the whole process, including gems. Use refinements or avoid.
- **`method_missing` without `respond_to_missing?`** — `respond_to? :dynamic_method` returns false; break introspection.
- **`return` inside a Proc** — exits the enclosing method, not just the Proc. Use `lambda` (or `next`) to return from the block.
- **Symbol vs string keys** — `{ name: "x" }` uses `:name`; `h["name"]` is `nil`. Pick one convention; `HashWithIndifferentAccess` (Rails) paper-overs this.
- **Implicit `nil`** — undefined instance vars are `nil` (no error); `@missing` silently returns `nil`.
- **`unless` with `else`** — `unless cond; ... else; ... end` is hard to read; use `if !cond` or flip the branches.
- **Frozen string literals** — strings are mutable by default; `# frozen_string_literal: true` at the top of a file makes them immutable (faster, catches accidental mutation).

## What to Learn Next

- **Ruby docs** — [ruby-doc.org](https://ruby-doc.org/) the standard library reference.
- **Why's (Poignant) Guide to Ruby** — [_why's legendary, whimsical intro](https://poignant.guide/) — a cult classic.
- **Programming Ruby** ("the Pickaxe") by Dave Thomas — the canonical reference.
- **Eloquent Ruby** by Russ Olsen — idioms and style.
- **Metaprogramming Ruby 2** by Paolo Perrotta — the deep dive on the object model and metaprogramming.
- **Rails Guides** — [guides.rubyonrails.org](https://guides.rubyonrails.org/) the official, excellent walkthrough.
- **The Rails 7 Way** by Obie Fernandez — comprehensive Rails reference.
- **Ruby Tapas** — [rubytapas.com](https://www.rubytapas.com/) short screencasts on idioms.

Ruby's design — readable, expressive, everything an object, blocks everywhere — was optimized for developer joy over raw performance. The metaprogramming power means a small amount of code does a lot, and Rails' "magic" is the same power at framework scale. Learn the object model and blocks first; the rest follows.

Good luck — and `bundle exec rspec`.

**Resources:**

- Ruby docs: [https://ruby-doc.org/](https://ruby-doc.org/)
- RubyGems: [https://rubygems.org/](https://rubygems.org/)
- Rails: [https://rubyonrails.org/](https://rubyonrails.org/)
- Bundler: [https://bundler.io/](https://bundler.io/)
- RSpec: [https://rspec.info/](https://rspec.info/)