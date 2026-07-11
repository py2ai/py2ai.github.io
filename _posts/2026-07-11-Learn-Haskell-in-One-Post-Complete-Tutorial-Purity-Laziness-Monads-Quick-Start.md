---
layout: post
title: "Learn Haskell in a Single Post: A Complete Haskell Tutorial from Purity and Laziness to Typeclasses and Monads"
description: "A complete Haskell tutorial in one blog post. Covers the whole language in 5 stages: fundamentals (functions, types, ghci, control flow, lists/tuples), purity + laziness (pure functions, referential transparency, lazy evaluation, infinite lists), ADTs + pattern matching (data, sum/product types, case, newtype, records), typeclasses + polymorphism (Eq, Functor, Applicative, Monad, higher-kinded types, deriving), and monads + IO + ecosystem (IO, do-notation, Maybe/Either/List monads, transformers, cabal/stack, HLS, QuickCheck). Five diagrams, runnable code snippets, and a quick-start roadmap."
date: 2026-07-11
header-img: "img/post-bg.jpg"
permalink: /Learn-Haskell-in-One-Post-Complete-Tutorial-Purity-Laziness-Monads-Quick-Start/
featured-img: ai-coding-frameworks/ai-coding-frameworks
tags:
  - Haskell
  - Functional Programming
  - Tutorial
  - Programming
  - Monads
  - Learn to Code
author: "PyShine"
---

# Learn Haskell in a Single Post: A Complete Haskell Tutorial from Purity and Laziness to Typeclasses and Monads

Haskell is the standard-bearer for pure functional programming: **functions are pure** (same input, same output, no side effects), **evaluation is lazy** (values computed only when needed), and **the type system is among the most expressive in mainstream use** (typeclasses, higher-kinded types, algebraic data types). Haskell is where a lot of modern FP ideas — monads, applicatives, typeclasses — were popularized, and it remains the reference point for "what would a maximally safe, maximally expressive language look like."

This post teaches the whole language in five stages with runnable snippets. By the end you'll understand purity and laziness, ADTs and pattern matching, typeclasses and higher-kinded polymorphism, and how monads make side effects composable without breaking purity.

We target **GHC 9.x** (the de facto standard compiler). Everything here compiles with a current `cabal`/`stack` toolchain.

## The Roadmap

![Haskell Roadmap](/assets/img/diagrams/haskell-tutorial/hs-roadmap.svg)

1. **Fundamentals** — functions, type signatures, `ghci`, control flow, lists/tuples
2. **Purity + Laziness** — pure functions, referential transparency, lazy evaluation, infinite lists
3. **ADTs + Pattern Matching** — `data`, sum/product types, `case`, `newtype`, records
4. **Typeclasses + Polymorphism** — `class`/`instance`, `Functor`/`Applicative`/`Monad`, kinds
5. **Monads + IO + Ecosystem** — `IO`, do-notation, `Maybe`/`Either`/List, transformers, cabal/stack

## Stage 1 — Fundamentals

### A program

```haskell
main :: IO ()
main = putStrLn "Hello, Haskell!"
```

`main :: IO ()` is the entry point — `IO ()` means "an IO action that produces unit". Run with `runhaskell Main.hs` or `cabal run`. For exploration, use `ghci` (the REPL):

```bash
$ ghci
GHCi> let x = 5
GHCi> x * 2
10
GHCi> :t x            -- show type of x
x :: Num a => a       -- polymorphic number
GHCi> :type map
map :: (a -> b) -> [a] -> [b]
```

### Functions and type signatures

```haskell
-- A function with an explicit type signature (read top-to-bottom)
add :: Int -> Int -> Int      -- takes Int, Int, returns Int
add a b = a + b                -- no parens, space-separated args

-- Type inference also works (signature optional but recommended)
double x = x * 2

-- Functions are first-class: pass and return them
apply f x = f x
apply double 5    -- 10

-- Composition: (.) chains functions
f . g = \x -> f (g x)
(show . double) 5   -- "10"
```

**Everything is a function** (or a value). Function application is just a space (`add 1 2`, not `add(1, 2)`). **All functions are curried** — `add :: Int -> Int -> Int` is really `Int -> (Int -> Int)`: take one int, return a function that takes the next. Partial application is natural: `map (add 1) [1,2,3]` (where `add 1` is a function `Int -> Int`).

### Basic types, lists, tuples

```haskell
True, False      :: Bool
'a'              :: Char
"hello"          :: String      -- alias for [Char]
5                :: Int         -- fixed-size
5                :: Integer     -- arbitrary precision
3.14             :: Double

[1, 2, 3]        :: [Int]       -- list (homogeneous, linked)
(1, "a")         :: (Int, String) -- tuple (heterogeneous, fixed size)

[1..5]           -- [1,2,3,4,5]  range
['a'..'e']       -- "abcde"
[1,3..9]         -- [1,3,5,7,9]  step
[10,9..1]        -- countdown

head [1,2,3]     -- 1
tail [1,2,3]     -- [2,3]
length [1,2,3]   -- 3
[1,2,3] ++ [4]   -- [1,2,3,4]    append
1 : [2,3]        -- [1,2,3]       cons (prepend, O(1))
```

Lists are **homogeneous** (all elements same type) and singly-linked — prepend (`:`) is O(1), append (`++`) is O(n). Strings are `[Char]`. Tuples are heterogeneous and fixed-size.

### Control flow — expressions, not statements

```haskell
-- if/then/else is an EXPRESSION (returns a value, both branches required)
absN x = if x < 0 then -x else x

-- Guards
classify n
  | n < 0  = "neg"
  | n == 0 = "zero"
  | otherwise = "pos"

-- case (pattern matching)
describeList xs = case xs of
  []     -> "empty"
  [x]    -> "singleton"
  (x:_)  -> "starts with something"

-- where clauses (local definitions)
area r = pi * r^2
  where pi = 3.14159

-- let .. in (expression-scoped)
result = let sq = x*x in sq + sq
```

There are **no statements, only expressions** — `if`, `case`, `let` all return values. Guards (`|`) are an elegant way to express "test this, then this, else this". `where` and `let ... in` introduce local bindings.

## Stage 2 — Purity and Laziness

![Haskell FP Core](/assets/img/diagrams/haskell-tutorial/hs-fp.svg)

### Pure functions

```haskell
-- Pure: no side effects, output is a pure function of input
square x = x * x                  -- square 5 is always 25

-- Contrast with impure (e.g., a random call in Python):
--   random() returns a different value each call -> NOT pure
-- In Haskell, "random" would return "IO Int" (below), separating purity
```

**Pure functions** always return the same output for the same input and have no side effects. This is the Haskell worldview: the *vast majority* of your code is pure, and side effects (I/O, randomness, mutation) are isolated in `IO` and similar types. The payoff: you can **reason equationally** — `f x + f x` can be replaced with `let y = f x in y + y` — and the compiler does this (via referential transparency).

### Laziness — evaluation on demand

```haskell
-- Take the first 10 squares from an infinite list
take 10 [ x*x | x <- [1..] ]    -- [1,4,9,...,100]

-- The infinite list is never fully built — only the needed parts are computed
primes = filter isPrime [2..]
take 5 primes                    -- [2,3,5,7,11]

-- foldr can terminate early on infinite lists
foldr (||) False (repeat True)   -- True (doesn't evaluate the whole list)

-- Contrast with foldl (strict) which would diverge on infinite input
```

Haskell is **non-strict** (lazy): a value is only computed when it's needed. This lets you define **infinite structures** and process them with `take`/`filter`/`foldr`. The classic example: `primes = sieve [2..]` where `sieve` is defined in terms of itself, and `take 10 primes` only forces the first 10.

**Caution**: laziness can cause **space leaks** (thunks accumulate in memory). Use `foldl'` (strict) instead of `foldl` for arithmetic, and `seq`/`BangPatterns` when you need to force evaluation.

### Immutability

```haskell
-- No mutation. "Updating" a list produces a new one.
nums = [1,2,3]
new = 0 : nums      -- [0,1,2,3]   (nums unchanged)
updated = take 2 nums ++ [9] ++ drop 3 nums   -- recombine, new list
```

There are no variables in the imperative sense — `x = 5` is a binding, not a cell. To "change" a value, you build a new one. This eliminates an entire class of bugs (aliasing, races) and makes parallelism trivial.

## Stage 3 — ADTs and Pattern Matching

![Haskell Types](/assets/img/diagrams/haskell-tutorial/hs-types.svg)

### Algebraic data types — sum and product

```haskell
-- Sum type: a value is one of the variants
data Color = Red | Green | Blue        -- three constructors, no payload

-- Sum + product: variants can carry data
data Shape
  = Circle Double                       -- radius
  | Square Double                       -- side
  | Rectangle Double Double             -- width, height

-- Parameterized: Maybe is built this way
data Maybe a = Nothing | Just a

-- Recursive: a tree
data Tree a = Leaf | Node (Tree a) a (Tree a)
```

An **algebraic data type** is a sum of products: `Shape` is `Circle Double` *or* `Square Double` *or* `Rectangle Double Double`. The "or" makes it a sum type; each variant's fields are the product. This is how Haskell models domains — clearly, exhaustively, type-safely.

### Pattern matching — exhaustive on ADTs

```haskell
area :: Shape -> Double
area (Circle r)        = pi * r^2
area (Square s)        = s^2
area (Rectangle w h)   = w * h

-- On Maybe
describe :: Maybe a -> String
describe Nothing  = "absent"
describe (Just _) = "present"

-- On a recursive type
depth :: Tree a -> Int
depth Leaf         = 0
depth (Node l _ r) = 1 + max (depth l) (depth r)
```

Pattern matching destructures ADTs and is **exhaustive-checked** — miss a case and GHC warns (or errors with `-Wincomplete-patterns`). The `case ... of` expression is pattern matching inline:

```haskell
classify n = case compare n 0 of
  LT -> "neg"
  EQ -> "zero"
  GT -> "pos"
```

### Records and newtype

```haskell
-- Records — named fields, auto-generate accessor functions
data Person = Person { name :: String, age :: Int }
ada = Person { name = "Ada", age = 30 }
name ada             -- "Ada"  (record accessor: a function Person -> String)
ada { age = 31 }     -- record update (new Person with age=31)

-- newtype — zero-cost wrapper (distinct type, same runtime rep)
newtype UserId = UserId Int
mkUser :: Int -> UserId
mkUser = UserId
-- UserId is a distinct type at compile time, no runtime overhead
```

`newtype` creates a **zero-cost distinct type** — `UserId` is an `Int` at runtime but a separate type at compile time, so you can't pass an `Int` where a `UserId` is expected. Use it for type-safety without allocation.

### Deriving

```haskell
data Point = Point Double Double deriving (Eq, Show, Ord)

-- GHC can auto-generate Eq, Show, Read, Ord, Enum, Bounded, and via Generics much more
```

`deriving` auto-generates common typeclass instances (`Eq`, `Show`, etc.) — no boilerplate. For more, enable `DeriveGeneric` / `DeriveAnyClass` for JSON, lenses, etc.

## Stage 4 — Typeclasses and Polymorphism

### Typeclasses — ad-hoc polymorphism

```haskell
-- A typeclass: a set of functions a type can implement
class Eq a where
  (==) :: a -> a -> Bool
  (/=) :: a -> a -> Bool
  x == y = not (x /= y)    -- default impl
  x /= y = not (x == y)

-- Instance: how Int implements Eq
instance Eq Int where
  x == y = intEq x y

-- Use with a constraint: f works for any Eq a
allEqual :: Eq a => a -> a -> a -> Bool
allEqual x y z = x == y && y == z
```

A **typeclass** is an interface (a set of functions) and **instances** are implementations for specific types. This is ad-hoc polymorphism — `f :: Eq a => a -> a -> Bool` means "f works for any `a` that has an `Eq` instance." This is how Haskell does overloading without runtime dispatch (it's resolved at compile time).

### Functor, Applicative, Monad — the hierarchy

```haskell
-- Functor: things you can map over
class Functor f where
  fmap :: (a -> b) -> f a -> f b
-- f must be * -> * (a type constructor)

instance Functor Maybe where
  fmap _ Nothing  = Nothing
  fmap f (Just x) = Just (f x)

fmap (*2) (Just 5)   -- Just 10
fmap (*2) Nothing   -- Nothing

-- Applicative: functions inside a context
class Applicative f where
  pure :: a -> f a
  (<*>) :: f (a -> b) -> f a -> f b

Just (*2) <*> Just 5        -- Just 10
Just (+) <*> Just 1 <*> Just 2   -- Just 3

-- Monad: chainable context
class Monad m where
  return :: a -> m a
  (>>=) :: m a -> (a -> m b) -> m b   -- "bind"
```

The **Functor → Applicative → Monad** hierarchy is the backbone of Haskell abstractions. `Functor` = mappable, `Applicative` = composable effects with pure functions, `Monad` = chainable effects where each step depends on the previous result. All three are **higher-kinded** typeclasses: `f :: * -> *` (a type constructor like `Maybe`, `[]`, `IO`).

### Higher-kinded types and kinds

```haskell
-- Kinds: * is a concrete type, * -> * is a type constructor
-- Maybe :: * -> *           (takes a type, returns a type)
-- Int  :: *                  (concrete type)
-- []   :: * -> *             (list is a type constructor)
-- (->) :: * -> * -> *        (function arrow)

-- Functor has kind (* -> *) -> Constraint:
class Functor (f :: * -> *) where ...   -- f must be a type constructor
```

**Kinds** are "types of types." `Int` has kind `*` (a concrete type); `Maybe` has kind `* -> *` (takes a type to make a type). `Functor` takes a `* -> *` and produces a `Constraint`. Higher-kinded polymorphism — abstracting over `* -> *` — is what lets `Functor`/`Monad` work uniformly across `Maybe`, `[]`, `IO`, etc.

### Deriving and generics

```haskell
{-# LANGUAGE DeriveGeneric #-}
import GHC.Generics

data User = User { id :: Int, name :: String } deriving (Generic, Show, Eq)
-- Generic lets libraries derive JSON, lenses, etc. automatically
```

With `DeriveGeneric`, GHC auto-generates a generic representation, and libraries (aeson for JSON, generic-lens) use it to derive instances without you writing them. This is a major reason Haskell can be ergonomic despite its power.

## Stage 5 — Monads, IO, and the Ecosystem

![Haskell Monads](/assets/img/diagrams/haskell-tutorial/hs-monads.svg)

### IO — side effects as values

```haskell
main :: IO ()
main = do
  putStrLn "What's your name?"
  name <- getLine
  putStrLn ("Hi, " ++ name ++ "!")

-- Desugars to:
-- main = putStrLn "What's your name?" >> getLine >>= \name -> putStrLn ("Hi, " ++ name ++ "!")
```

`IO a` is a **description of an effectful computation** that produces `a` — a pure value until `main` runs it. This is how Haskell keeps purity: side effects aren't performed by your functions, they're *described* as `IO` values and executed by the runtime. `do`-notation is sugar for `(>>=)` chains, making IO code read like imperative code while remaining purely compositional.

### Common monads

```haskell
-- Maybe: short-circuits on Nothing
safeDiv :: Double -> Double -> Maybe Double
safeDiv _ 0 = Nothing
safeDiv x y = Just (x / y)

result = do
  a <- safeDiv 10 2     -- Just 5
  b <- safeDiv a 0      -- Nothing -> whole computation becomes Nothing
  return b
-- result == Nothing

-- Either: typed errors
parse :: String -> Either String Int
parse s | all (`elem` "0123456789") s = Right (read s)
        | otherwise = Left ("bad: " ++ s)

-- List: nondeterminism (multiple results)
pairs = do
  x <- [1,2,3]
  y <- [4,5]
  return (x, y)
-- [(1,4),(1,5),(2,4),(2,5),(3,4),(3,5)]
-- (same as: [(x,y) | x <- [1,2,3], y <- [4,5]])
```

`Maybe`, `Either`, `[]`, `IO` are all monads — they all support `return` and `>>=`, so they all work with `do`-notation. The **same syntax for different effects** is the payoff: `do` over `Maybe` short-circuits on `Nothing`; `do` over `[]` gives nondeterminism; `do` over `IO` sequences effects. The type tells you which.

### Monad transformers — stacking effects

```haskell
-- Stack: ReaderT (config) + StateT (mutable counter) + IO
type App = ReaderT Config (StateT Int IO)

runApp :: Config -> Int -> App a -> IO a
runApp cfg st act = runStateT (runReaderT act cfg) st >>= \(_, final) -> return ()
```

When you need multiple effects (state + errors + IO), you stack monad transformers (`ReaderT`, `StateT`, `ExceptT`). Each adds one capability. The `mtl` library makes this ergonomic with typeclasses (`MonadReader`, `MonadState`, `MonadError`). In modern Haskell, the `effectful` or `polysemy` libraries offer alternatives with better ergonomics.

### The toolchain

![Haskell Toolchain](/assets/img/diagrams/haskell-tutorial/hs-toolchain.svg)

```bash
# Compilers / REPL
ghc Main.hs            # compile
runhaskell Main.hs     # run without compiling to a binary
ghci                   # REPL (:l to load, :t to show type, :r to reload)

# Build tools
cabal init             # scaffold a project
cabal build; cabal run; cabal test
stack new myproj       # alternative, curated snapshot-based
stack build; stack test

# LSP / IDE
# HLS (Haskell Language Server) provides LSP for VS Code / Vim / etc.
```

A minimal `*.cabal`:

```cabal
cabal-version:      3.0
name:               myapp
version:            0.1.0.0
build-type:         Simple

executable myapp
    main-is:          Main.hs
    build-depends:    base, text, aeson
    default-language: Haskell2010
    ghc-options:      -Wall
```

### Testing — property-based with QuickCheck

```haskell
import Test.QuickCheck

prop_reverse :: [Int] -> Bool
prop_reverse xs = reverse (reverse xs) == xs

-- Run: quickCheck prop_reverse  -> tests 100 random inputs
```

**QuickCheck** is Haskell's gift to the programming world — you state a property (`reverse . reverse == id`), and it generates 100 random inputs to try to falsify it. Property-based testing, invented in Haskell, has spread to nearly every language.

### Tooling

- **GHC** — the de facto compiler; lazy, with a huge extension language (`-X LANGUAGE`).
- **ghci** — REPL; essential for exploration.
- **cabal / stack** — build tools (cabal is more flexible; stack gives curated snapshots).
- **HLS** — the Haskell Language Server (LSP); VS Code / Vim integration.
- **Hackage / Stackage** — package repositories; Stackage gives a curated, tested set.
- **QuickCheck** — property-based testing.
- **HUnit / tasty** — unit testing frameworks.
- **servant** — type-level web API DSL; the API types generate clients, servers, docs.
- **lens** — composable getters/setters; powerful but with a learning curve.
- **STM** — software transactional memory; composable concurrency.

## A Quick-Start Checklist

1. **Install GHC + cabal** (or `ghcup` to manage versions).
2. **Type signatures on top-level functions** — they document intent and catch bugs early.
3. **`ghci` for exploration** — `:t` (type), `:i` (info), `:l` (load).
4. **Embrace purity** — keep most code pure; isolate effects in `IO`.
5. **ADTs + pattern matching** to model domains; let exhaustiveness checks find missing cases.
6. **Typeclasses** for ad-hoc polymorphism; `deriving` for common instances.
7. **`do`-notation** for monadic code; learn what `>>=` means underneath.
8. **Use `-Wall`** (all warnings) and `-Werror` in CI.
9. **QuickCheck** for property tests; HUnit/tasty for unit tests.
10. **Avoid space leaks**: `foldl'` for arithmetic, strict fields where appropriate.

## Common Pitfalls

- **Space leaks from laziness** — thunks accumulate. Use `foldl'` (strict), `seq`, or `BangPatterns` when a fold accumulates a large value.
- **Infinite loops from laziness** — `length (repeat 1)` never terminates. Always `take`-bound infinite lists.
- **Pattern match non-exhaustiveness** — runtime `*** Exception: Non-exhaustive patterns`. Enable `-Wincomplete-patterns` (part of `-Wall`).
- **Confusing `Int` and `Integer`** — `Int` is fixed-size (overflows); `Integer` is arbitrary-precision. Default is `Num a => a` (polymorphic).
- **Confusing `String` = `[Char]`** — fine for small strings; use `Text` (strict) or `ByteString` for performance.
- **Monomorphism restriction** — toplevel bindings without signatures can default unexpectedly. Add signatures.
- **`return` is not like other languages** — it's just `pure`, lifts into the monad; it doesn't exit a function.
- **Forgetting `IO`** — a pure function can't print; the type won't allow it. That's the point.
- **Record field name clashes** — two records with `name` collide; use `DuplicateRecordFields` / `OverloadedRecordDot` extensions or a lens library.
- **`seq` abuse** — sprinkling `seq` everywhere doesn't always help and can hurt. Profile first.

## What to Learn Next

- **Learn You a Haskell for Great Good** — [learnyouahaskell.com](http://learnyouahaskell.com/) the friendly, free, illustrated intro. Best first stop.
- **Real World Haskell** — [book.realworldhaskell.org](http://book.realworldhaskell.org/) free, practical (a bit dated but solid).
- **Haskell Programming from First Principles** by Chris Allen & Julie Moronuki — the thorough modern book.
- **Haskell docs / GHC User Guide** — [haskell.org/documentation](https://www.haskell.org/documentation/) and the GHC extension guide.
- **Typeclassopedia** — [wiki.haskell.org/Typeclassopedia](https://wiki.haskell.org/Typeclassopedia) the canonical Functor/Applicative/Monad reference.
- **Stephen Diehl's What I Wish I Knew When Learning Haskell** — [dev.stephendiehl.com/hask](http://dev.stephendiehl.com/hask/) practical guide.
- **Hackage** — [hackage.haskell.org](https://hackage.haskell.org/) the package index.
- **haskell-beginners** mailing list, r/haskell, Functional Programming Slack — the community.

Haskell is the language where you'll *really* learn what "functional programming" means: purity, laziness, and a type system that can encode deep invariants. The learning curve is steep (monads, typeclasses, laziness all take time), but the perspective shift is permanent — you'll write better code in every other language after.

Good luck — and keep `ghci` open.

**Resources:**

- Haskell: [https://www.haskell.org/](https://www.haskell.org/)
- GHC: [https://www.haskell.org/ghc/](https://www.haskell.org/ghc/)
- Hackage: [https://hackage.haskell.org/](https://hackage.haskell.org/)
- cabal: [https://www.haskell.org/cabal/](https://www.haskell.org/cabal/)
- QuickCheck: [https://hackage.haskell.org/package/QuickCheck](https://hackage.haskell.org/package/QuickCheck)