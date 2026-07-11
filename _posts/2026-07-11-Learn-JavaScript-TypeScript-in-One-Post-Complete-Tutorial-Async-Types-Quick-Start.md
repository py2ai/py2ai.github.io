---
layout: post
title: "Learn JavaScript and TypeScript in a Single Post: A Complete Tutorial from Event Loop to Type System"
description: "A complete JavaScript + TypeScript tutorial in one blog post. Covers both languages in 5 stages: JavaScript core (vars, types, this, closures), modern ES6+ (modules, iterators, destructuring), async (callbacks to promises to async/await and the event loop), the TypeScript type system (unions, generics, narrowing, utility types), and the ecosystem + toolchain (Node, npm/pnpm/bun, tsc, eslint, Vitest, bundlers). Five diagrams, runnable code snippets, and a quick-start roadmap."
date: 2026-07-11
header-img: "img/post-bg.jpg"
permalink: /Learn-JavaScript-TypeScript-in-One-Post-Complete-Tutorial-Async-Types-Quick-Start/
featured-img: ai-coding-frameworks/ai-coding-frameworks
tags:
  - JavaScript
  - TypeScript
  - Tutorial
  - Programming
  - Async
  - Learn to Code
author: "PyShine"
---

# Learn JavaScript and TypeScript in a Single Post: A Complete Tutorial from Event Loop to Type System

JavaScript is the language of the web — the only language that runs natively in every browser. TypeScript is JavaScript with a static type system layered on top, compiled away to plain JS. Together they are the default stack for frontend, backend (Node), CLI tools, and increasingly everything else.

This post teaches both in five stages with runnable snippets. By the end you'll understand closures and `this`, the single-threaded event loop and async/await, the module system, and the TypeScript type system — unions, generics, narrowing, utility types. The goal: stop fighting JS quirks and start using TS to write safe, navigable code.

We target **ES2023+ JavaScript** and **TypeScript 5.x**. Everything here runs in a modern browser or Node 20+.

## The Roadmap

![JS/TS Roadmap](/assets/img/diagrams/jsts-tutorial/jsts-roadmap.svg)

1. **JavaScript Core** — variables, types, functions, `this`, objects, arrays, destructuring
2. **ES6+ Modern JS** — modules, template literals, spread/rest, iterables, generators
3. **Async + Event Loop** — callbacks → promises → async/await, microtasks vs macrotasks
4. **TypeScript Types** — primitives, unions, interfaces vs type aliases, generics, narrowing
5. **Ecosystem + Tooling** — Node, npm/pnpm/bun, tsc, eslint/prettier, React/Vue, bundlers, testing

## Stage 1 — JavaScript Core

### A program

```js
console.log("Hello, JS!");
```

Run it in a browser console, in Node (`node file.js`), in Deno, or in Bun. There is no `main` — the file runs top to bottom.

### Variables: let / const / var

```js
let x = 10;        // block-scoped, reassignable
const y = 20;      // block-scoped, not reassignable (but objects inside are mutable)
var z = 30;        // function-scoped, hoisted — legacy, avoid in new code

// const does not make objects immutable:
const obj = { a: 1 };
obj.a = 2;          // OK — const binds the name, not the value
// obj = { b: 3 };  // TypeError — can't reassign the binding
```

Use `const` by default; reach for `let` only when you must reassign. `var` is legacy and has scoping surprises (function scope, not block scope) — don't use it in new code.

### Types and coercion

JavaScript has **8 types**: `number`, `string`, `boolean`, `null`, `undefined`, `symbol`, `bigint`, and `object`. Numbers are all IEEE 754 doubles (no int/float split):

```js
typeof 1        // "number"
typeof NaN      // "number" (NaN is a number, yes)
typeof "a"      // "string"
typeof null     // "object"  ← historical bug, can't be fixed
typeof undefined // "undefined"
typeof {}       // "object"
typeof function(){} // "function"

// Coercion is the famous footgun
"3" + 4   // "34"  + coerces to string
"3" - 4   // -1    - coerces to number
"3" == 3  // true  == coerces
"3" === 3 // false === does not coerce
```

**Always use `===` and `!==`.** They don't coerce, so they say what you mean. The `==` operator has a conversion table no one memorizes.

### Functions, arrows, and `this`

```js
function add(a, b) { return a + b; }
const sub = (a, b) => a - b;            // arrow — concise body, implicit return

// Default and rest params
function greet(name = "world", ...rest) { return `hi ${name} ${rest}`; }

// Closures — inner function captures outer variables
function counter() {
    let n = 0;
    return () => ++n;                   // captures n
}
const c = counter();
c(); c(); c()  // 3
```

`this` is the famous JS puzzle. **Arrows don't bind their own `this`** — they inherit it from the enclosing scope. Regular functions do bind it, and the binding depends on how they're called:

```js
const obj = {
    v: 10,
    reg() { return this.v; },           // this = obj -> 10
    arrow: () => this.v,                // this = enclosing (module) -> undefined
};
obj.reg()   // 10
obj.arrow() // undefined

// Methods lose `this` when detached
const detached = obj.reg;
detached();  // undefined — `this` is whatever called it (global/undefined)
const bound = obj.reg.bind(obj);
bound();     // 10
```

Rule: **use arrows for short callbacks and when you want the surrounding `this`; use regular functions for object methods.** `call`/`apply`/`bind` let you set `this` explicitly when needed.

### Objects, arrays, destructuring

```js
const obj = { a: 1, b: 2, ["c" + "d"]: 3 };  // computed keys
const { a, b } = obj;                         // destructuring
const { a: renamed } = obj;                   // rename
const { e = 0 } = obj;                         // default

const arr = [1, 2, 3];
const [x, y, ...rest] = arr;                  // array destructuring + rest
const [, second] = arr;                        // skip first

// Spread
const copy = { ...obj, f: 9 };
const merged = [...arr1, ...arr2];
```

Arrays are objects with a length property and indexed keys. They're not typed:

```js
arr.map(x => x * x)         // [1, 4, 9]   — new array
arr.filter(x => x > 1)      // [2, 3]
arr.reduce((sum, x) => sum + x, 0)  // 6
arr.forEach(x => console.log(x))    // side effects
arr.find(x => x > 1)        // 2 (first match)
arr.includes(2)             // true
[...arr].sort()            // copy then sort (sort mutates!)
```

**`sort()` mutates the array in place** and sorts lexicographically by default (`[10, 2].sort()` → `[10, 2]`). Pass a comparator: `arr.sort((a, b) => a - b)`.

## Stage 2 — ES6+ Modern JavaScript

### Modules

```js
// math.js
export const PI = 3.14;
export function add(a, b) { return a + b; }
export default function multiply(a, b) { return a * b; }

// main.js
import multiply, { PI, add } from "./math.js";
import * as math from "./math.js";
const mod = await import("./math.js");    // dynamic import — async
```

ESM (ECMAScript Modules) is the standard, replacing CommonJS (`require`). Node supports both; `.mjs` or `"type": "module"` in `package.json` forces ESM. Use ESM in all new code.

### Template literals, spread/rest, optional chaining

```js
const name = "Ada";
`Hello, ${name}! ${1 + 2}`                 // string interpolation
`multi
line`                                      // multi-line strings

const obj = { user: { profile: { age: 30 } } };
obj?.user?.profile?.age                    // 30  optional chaining
obj?.missing?.profile?.age                 // undefined (not a TypeError)
obj.user.profile?.city ?? "unknown"        // nullish coalescing — only null/undefined
```

`?.` short-circuits to `undefined` instead of throwing on a missing property. `??` returns the right side only for `null`/`undefined` (not `0` or `""`), unlike `||` which falsy-coerces.

### Iterables and generators

```js
// for...of works on any iterable (has Symbol.iterator)
for (const x of [1, 2, 3]) { }
for (const [k, v] of new Map([["a", 1]])) { }

// Generators — lazy sequences with yield
function* evens() {
    let i = 0;
    while (true) { yield i; i += 2; }
}
const it = evens();
it.next();  // { value: 0, done: false }
it.next();  // { value: 2, done: false }

// Custom iterable
const range = {
    [Symbol.iterator]() {
        let i = 0;
        return { next: () => i < 3 ? { value: i++, done: false } : { done: true } };
    },
};
[...range]  // [0, 1, 2]
```

### Map/Set and other collections

```js
const m = new Map([["a", 1]]);
m.set("b", 2); m.get("a"); m.has("b"); m.size
const s = new Set([1, 1, 2, 3]);   // {1, 2, 3} — dedup
const wm = new WeakMap();           // keys must be objects, GC-friendly
```

Use `Map` over objects when keys are non-string or you need ordered iteration; `Set` for uniqueness; `WeakMap`/`WeakSet` for metadata that shouldn't prevent GC.

## Stage 3 — Async and the Event Loop

JavaScript is **single-threaded** with an event loop. There's one call stack; I/O is non-blocking because the runtime (browser/Node) does the waiting off-thread and enqueues callbacks when ready.

![JS/TS Async](/assets/img/diagrams/jsts-tutorial/jsts-async.svg)

### Callbacks → Promises → async/await

```js
// Callbacks (old style) — nested, inversion of control
fetch(url, (err, data) => {
    if (err) return handle(err);
    process(data, (err2, r2) => { /* callback hell */ });
});

// Promises — a value that will resolve or reject later
fetch(url)
    .then(r => r.json())
    .then(data => process(data))
    .catch(err => handle(err))
    .finally(() => cleanup());

const p = new Promise((resolve, reject) => {
    setTimeout(() => resolve(42), 100);
});

// async/await — syntax sugar over promises
async function getData(url) {
    try {
        const r = await fetch(url);        // suspends, doesn't block
        const data = await r.json();
        return data;
    } catch (err) {
        handle(err);
        throw err;                          // re-throw to propagate
    }
}

// Parallel — all promises, fail-fast
const [a, b] = await Promise.all([getA(), getB()]);
const first = await Promise.race([fast(), slow()]);
const all = await Promise.allSettled([...]); // never rejects
```

`async` functions always return a Promise. `await` pauses the function until the Promise settles, yielding control to the event loop — it does **not** block the thread. This is why you can have thousands of concurrent fetches on one thread.

### Microtasks vs macrotasks

```js
console.log(1);
setTimeout(() => console.log(4), 0);          // macrotask
Promise.resolve().then(() => console.log(3)); // microtask
console.log(2);
// Output: 1, 2, 3, 4
```

The event loop drains the call stack, then runs **all microtasks** (Promise `.then` callbacks, `queueMicrotask`) before processing one macrotask (`setTimeout`, I/O callbacks, `setImmediate`). Microtasks always run before the next macrotask. This ordering explains many "why did my callback run first?" puzzles.

### Streams

```js
// Read a response body as a stream
const r = await fetch(url);
for await (const chunk of r.body) {
    process(chunk);        // Uint8Array chunks
}
```

Async iteration over streams is the modern way to handle large or chunked data without buffering it all in memory.

## Stage 4 — TypeScript Types

TypeScript adds a static type system that's erased at runtime — it compiles to plain JS. The type system is **structural** (shape-based, like Go interfaces) and quite powerful.

![TS Type System](/assets/img/diagrams/jsts-tutorial/jsts-types.svg)

### Basic types and annotations

```ts
let n: number = 10;
let s: string = "hi";
let b: boolean = true;
let nothing: null = null;
let u: undefined = undefined;
let big: bigint = 100n;
let sym: symbol = Symbol();

// Function signatures
function add(a: number, b: number): number { return a + b; }
const sub = (a: number, b: number): number => a - b;

// Optional and default params
function greet(name: string, greeting = "hi"): string { return `${greeting}, ${name}`; }
function opt(x?: number): number { return x ?? 0; }  // x: number | undefined
```

### Unions and literals

```ts
type Result = "ok" | "error";                // union of literals
type ID = number | string;                   // union of types
type State = "idle" | "loading" | { data: number };

function handle(s: State) {
    if (s === "idle") { /* narrowed to "idle" */ }
    else if (typeof s === "object") { s.data }  // narrowed to object
}
```

### Interfaces vs type aliases

```ts
interface Point { x: number; y: number }
type Point2 = { x: number; y: number };

interface Point { z?: number }     // interfaces can be reopened (declaration merging)
// type Point2 = { z: number }     // error — type aliases can't be reopened

// Extending
interface Point3D extends Point { z: number }
type WithLabel<T> = T & { label: string }
```

Use **interfaces for object shapes** (they can be extended/reopened and show better error messages); use **type aliases for unions, intersections, and computed types**. They're largely interchangeable for plain objects.

### Generics

```ts
function first<T>(xs: T[]): T | undefined { return xs[0] }
const n = first([1, 2, 3]);   // T inferred as number

// Constraints
function len<T extends { length: number }>(x: T): number { return x.length }
len("hi"); len([1, 2]);       // both have .length

// Generic types
class Box<T> { constructor(public value: T) {} }
const b = new Box(42);

// Mapped types
type Readonly<T> = { readonly [K in keyof T]: T[K] };
type Partial_<T> = { [K in keyof T]?: T[K] };

// Conditional types
type IsString<T> = T extends string ? true : false;
type X = IsString<"a">;      // true
```

### Narrowing and type guards

```ts
function f(x: string | number) {
    if (typeof x === "string") { x.toUpperCase() }  // narrowed to string
    else { x.toFixed(2) }                            // narrowed to number
}

// instanceof, in, discriminated unions
interface Circle { kind: "circle"; r: number }
interface Square { kind: "square"; s: number }
type Shape = Circle | Square;

function area(s: Shape): number {
    switch (s.kind) {                                  // discriminant
        case "circle": return Math.PI * s.r ** 2;
        case "square": return s.s ** 2;
    }
}

// User-defined type guards
function isError(x: unknown): x is Error { return x instanceof Error }
if (isError(e)) { e.message }                           // narrowed to Error
```

### Utility types

```ts
Partial<User>          // all fields optional
Required<User>         // all fields required
Pick<User, "id"|"email">   // subset of fields
Omit<User, "password">     // all except specified
Record<string, number> // { [k: string]: number }
Readonly<User>         // all readonly
ReturnType<typeof f>  // the return type of f
Parameters<typeof f>[0]  // first param type of f
```

`unknown` is the safe top type (you must narrow before use); `any` opts out of type checking entirely — avoid it. The built-in utility types (`Partial`, `Pick`, `Omit`, `Record`, `ReturnType`) cover most transformation needs.

## Stage 5 — Ecosystem and Toolchain

![JS/TS Toolchain](/assets/img/diagrams/jsts-tutorial/jsts-toolchain.svg)

### Runtimes and package managers

```bash
# Runtimes
node file.js            # the dominant runtime
deno run file.ts        # secure by default, TS native
bun run file.ts         # fast, all-in-one

# Package managers (npm registry)
npm install express
pnpm install express      # faster, deduped via symlinks
yarn add express
bun add express           # fastest

# Lock files
package-lock.json / pnpm-lock.yaml / yarn.lock / bun.lockb
```

A minimal `package.json`:

```json
{
  "name": "myapp",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "vite build",
    "test": "vitest",
    "lint": "eslint .",
    "typecheck": "tsc --noEmit"
  },
  "dependencies": { "react": "^19.0.0" },
  "devDependencies": { "typescript": "^5.4.0", "vitest": "^2.0.0", "vite": "^6.0.0" }
}
```

A minimal `tsconfig.json`:

```json
{
  "compilerOptions": {
    "target": "ES2022",
    "module": "ESNext",
    "moduleResolution": "bundler",
    "strict": true,
    "noUncheckedIndexedAccess": true,
    "esModuleInterop": true,
    "skipLibCheck": true,
    "verbatimModuleSyntax": true
  },
  "include": ["src"]
}
```

**`strict: true`** turns on all the safety checks — enable it from day one. `noUncheckedIndexedAccess` makes `arr[i]` return `T | undefined`, which catches out-of-bounds reads.

### Frameworks, bundlers, testing

- **Frontend frameworks**: React (dominant), Vue, Svelte, Solid, Astro, Qwik.
- **Bundlers**: Vite (default for new apps), esbuild (raw speed), webpack (legacy), Rollup (libraries), Rspack (Rust webpack).
- **Testing**: Vitest (Jest-compatible, faster), Jest (legacy), Playwright (e2e, browser), Bun test.
- **Lint/format**: ESLint (`eslint`), Prettier (`prettier`), Biome (Rust, all-in-one).
- **Backend**: Node + Express/Fastify/Hono, Deno, Bun.

```bash
# Quality gates
npx tsc --noEmit              # type check without emit
npx eslint .                  # lint
npx prettier --write .        # format
npx vitest                    # test runner
npx playwright test           # e2e

# Build
npx vite build                # bundle for production
```

## A Quick-Start Checklist

1. **`===` only** — never `==`. Read the coercion table once and then forget it.
2. **`const` by default**, `let` only when reassigning, never `var`.
3. **Understand `this` and closures** — they're the two things that surprise newcomers.
4. **Use ESM** (`import`/`export`, `"type": "module"`) in all new code.
5. **Prefer async/await** over raw `.then` chains; wrap with try/catch.
6. **Enable `strict` in tsconfig** from day one; add `noUncheckedIndexedAccess`.
7. **`unknown` over `any`** — narrow with type guards instead of opting out.
8. **Pick a stack**: Vite + Vitest + ESLint + Prettier for most projects.
9. **Run `tsc --noEmit` and `eslint` in CI**, plus tests.

## Common Pitfalls

- **`==` coercion** — `"0" == false` is `true`. Use `===`.
- **`var` hoisting** — `var` is function-scoped and hoisted, leading to "use before declaration" surprises. Use `let`/`const`.
- **`this` in callbacks** — a regular function passed as a callback loses its `this`. Use an arrow or `.bind`.
- **Mutating while iterating** — adding/removing during `forEach`/`for` causes skips. Build a new array instead.
- **`sort()` without comparator** — sorts as strings (`[10, 9, 2]` → `[10, 2, 9]`). Pass `(a, b) => a - b`.
- **`async` without `await`** — an `async` fn with no `await` runs synchronously and wraps the return in a Promise for no reason.
- **Top-level `await`** in a non-ESM file — only works in modules.
- **`any` spreading** — `any` disables checking; one `any` in a chain infects everything downstream. Use `unknown` + narrow.
- **Array/object copy** — `=` copies the reference. Use spread (`[...arr]`, `{...obj}`) for shallow, or a deep-clone library for nested.
- **`null` vs `undefined`** — `null` is explicit "no value"; `undefined` is "not set". Most APIs use `undefined`; `??` handles both.

## What to Learn Next

- **MDN Web Docs** — [developer.mozilla.org](https://developer.mozilla.org/) the authoritative JS/Web API reference.
- **JavaScript.info** — [javascript.info](https://javascript.info/) a thorough modern tutorial.
- **You Don't Know JS** by Kyle Simpson — the deep dive on closures, types, async.
- **Eloquent JavaScript** by Marijn Haverbeke — free online, a gentle intro.
- **TypeScript Handbook** — [typescriptlang.org/docs/handbook](https://www.typescriptlang.org/docs/handbook/) the official type-system guide.
- **Type Challenges** — [typehero.dev](https://typehero.dev/) / [github.com/type-challenges](https://github.com/type-challenges/type-challenges) to practice the type system.
- **Total TypeScript** by Matt Pocock — practical, modern TS.
- **Node.js docs** — [nodejs.org/docs](https://nodejs.org/docs/latest/api/) for the runtime and stdlib.

JavaScript's quirks (coercion, `this`, prototypes) are legacy; TypeScript's type system is modern and powerful. Learn both: the quirks so you don't get bitten, the types so you can refactor a large codebase with confidence. Once `async`/`await` and generics are reflexes, the stack gets out of your way.

Good luck — and turn on `strict`.

**Resources:**

- MDN: [https://developer.mozilla.org/](https://developer.mozilla.org/)
- JavaScript.info: [https://javascript.info/](https://javascript.info/)
- TypeScript: [https://www.typescriptlang.org/](https://www.typescriptlang.org/)
- Node.js: [https://nodejs.org/](https://nodejs.org/)
- Vite: [https://vitejs.dev/](https://vitejs.dev/)
- Vitest: [https://vitest.dev/](https://vitest.dev/)