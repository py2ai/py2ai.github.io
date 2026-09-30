---
layout: post
title: "shadcn-lint: Design System Rules Agents Can Verify - Inside shadcn-ui/lint"
description: "A source tour of shadcn-ui/lint, the agent-first linter for Tailwind design systems. How its six rules, contract policy engine, cn class grammar, and Tailwind oracle turn design-system policy into errors that both humans and AI coding agents can read, follow, and verify."
date: 2026-09-30
header-img: "img/post-bg.jpg"
permalink: /Shadcn-UI-Lint-Agent-First-Design-System-Linter-Source-Tour/
featured-img: "ai-coding-frameworks/ai-coding-frameworks"
image: https://pyshine.com/assets/img/diagrams/shadcn-lint/shadcn-ui-shadcn-lint-architecture.svg
tags:
  - Tailwind
  - Design Systems
  - Linting
  - AI Agents
categories: [AI, Open Source]
keywords: "shadcn-lint, shadcn-ui/lint, agent-first linter, Tailwind design system, ESLint plugin, Oxlint plugin, no-restyle rule, Tailwind v4 linting, design system enforcement, cn class grammar, tailwind-merge classifier, AI coding agents, React Vue Svelte linting, source code architecture"
author: "PyShine"
---

Design systems rarely die in a dramatic rewrite. They erode one `className` at a time: a page adds `p-4` to a Button because the size prop felt inconvenient, another hardcodes `bg-pink-500`, a third reaches for an arbitrary value like `p-[13px]`, and suddenly the variants and theme tokens your system was built around are decorative. TypeScript can push back on some of this, but a type error only says what is forbidden. It rarely says what to use instead, and it says nothing at all to the AI coding agents that now write most of this markup.

`@shadcn/lint` is an open-source, agent-first linter for Tailwind design systems from the shadcn-ui organization. You define what is allowed with ordinary rule options and per-component contracts; when a rule breaks, the error explains what is wrong and suggests a fix drawn from your own components, variants, and theme. The package ships six rules, runs on both ESLint (9.30+) and Oxlint (1.80+), works across React, Vue, and Svelte, targets Tailwind v4, and does not require shadcn/ui at all — it lints any Tailwind project's components and theme.

The source is worth a tour because the interesting problem here is not "write another ESLint rule." It is how to express design-system policy so that two very different readers — a human reviewing a diff and an agent iterating on its own work — can both act on the verdict. That question shapes everything in the repository: a policy engine built around appearance categories, a class grammar that mirrors the tailwind-merge trie, a project model that knows your component names and theme tokens, and an oracle that asks your project's own Tailwind whether a class actually generates CSS.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/shadcn-lint/shadcn-ui-shadcn-lint-overview-architecture.svg" alt="Architecture overview of the shadcn-ui/lint repository" style="max-width:100%;height:auto;" />
</div>

*Overview of the shadcn-ui/lint architecture: a plugin surface, a rule engine driven by contracts, a class-site collector with per-framework readers, the cn class grammar, the project model, and the Tailwind oracle.*

Reading the overview from left to right: the package entry (`packages/lint/src/index.ts`) exports the plugin plus an experimental introspection API into the project model. The plugin (`packages/lint/src/plugin.ts`) registers six rules and wraps each one with the framework readers in `packages/lint/src/sites/readers`, so the same rule code runs over JSX, Vue templates, and Svelte markup. The rules never walk the AST themselves; they ask the class-site collector (`packages/lint/src/sites/collect.ts`) for every place a class string enters the program, resolved to the component it belongs to. Verdicts come from the policy engine in `packages/lint/src/rules/contracts.ts`, which classifies each token through the cn grammar (`packages/lint/src/grammar/classifier.ts`) and the appearance-category table (`packages/lint/src/grammar/categories.ts`). Finally, `no-unknown-classes` consults the Tailwind oracle (`packages/lint/src/tailwind/oracle.ts`), which loads the project's real Tailwind v4 design system to answer what nothing else can.

## Why You Need This

The README opens with a comparison that captures the gap precisely. You can enforce "Button owns its padding" in TypeScript by limiting the `style` prop to a `Pick` of `React.CSSProperties`, and the compiler will reject `<Button style={...}>` with a property error. But that error tells an agent only that padding is not allowed. It does not tell the agent that Button has `sm` and `lg` sizes, that margin is fine, or that a new size belongs in `components/ui/button.tsx` only if the design genuinely calls for one. A lint diagnostic can carry all of that, because the linter knows your component files, your `cva` variants, and your theme tokens.

The second problem is ownership. Most teams do not own every component they style — shadcn/ui components are copied into the project, third-party libraries ship their own primitives, and monorepos share a UI package across apps. Forking or wrapping components to police their usage does not scale, and it breaks on every upstream update. Because `@shadcn/lint` matches components by name through the import graph (`packages/lint/src/project/components.ts` builds an index from your ui directory, barrels included), you can put contracts on a `CardTitle` that came from anywhere, and each project can express its own policy without touching component code.

The third problem is verification for agents. An agent needs a finish line it can check by itself: run one command, read the errors, fix, re-run. That loop only converges if errors are honest. A class Tailwind cannot generate, like `rounded-huge`, produces no CSS and fails silently — `no-unknown-classes` catches it by asking the project's actual Tailwind. Dynamically built classes like `` `bg-${color}` `` are invisible to any static analyzer, so pretending they were checked would be worse than reporting them; `require-static-classes` makes unreadable class code a violation of its own, with a forwarded `className` prop as the one sanctioned escape hatch.

Finally, the rules encode judgments that took real design-system experience to write down. Padding on a button usually means "size," so the error offers the component's sizes and suggests putting space around the component on its parent instead. A raw color gets answered with the nearest theme token. These heuristics live in the rule implementations and message templates, and they are exactly the knowledge an agent cannot guess from a style guide PDF.

## How It Works

Everything in the repository serves one pipeline: find class strings, resolve them to components, classify each token, apply the component's contract, and report a verdict an agent can act on.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/shadcn-lint/shadcn-ui-shadcn-lint-architecture.svg" alt="Architecture of the shadcn-ui/lint repository" style="max-width:100%;height:auto;" />
</div>

*Detailed architecture of shadcn-ui/lint: six rules over a shared contract engine, the class-site collector and its JSX/Vue/Svelte readers, the cn grammar with its trie classifier, the project model, and the worker-based Tailwind oracle.*

### Understanding the Architecture

**The plugin surface.** `packages/lint/src/plugin.ts` is deliberately tiny: a `meta` block with `name: "shadcn"` and the six rules. The comment there explains a subtle decision — the meta name is what Oxlint uses as the rule namespace, so rule ids like `shadcn/no-restyle` are identical across ESLint and Oxlint, and one configuration document serves both runners. Every rule is wrapped by `withTemplates` from `packages/lint/src/sites/readers/index.ts`, which registers the same visitors through Vue's `defineTemplateBodyVisitor` or the Svelte parser services when they are present — and warns once when a `.vue` or `.svelte` file arrives without its template parser, because silently linting only the script block would read as "checked" when it was not. The public entry `packages/lint/src/index.ts` also exposes an experimental `project` object (`projectFor`, `themeFileFor`, `colorTokensFor`, `componentsFor`, and the variant functions) so external tooling can ask the linter what it knows.

**The class-site collector.** `packages/lint/src/sites/collect.ts` is the largest module, and its header states the invariant everything else relies on: Tailwind only generates CSS for class text present in source, so text plus one hop of resolution covers every class that can render. `createComponentTracker` follows imports through `packages/lint/src/project/modules.ts` (export closure) and `packages/lint/src/project/wrappers.ts` (components that forward `className`), resolving each element to a component name and file. The resulting `ClassSite` separates `contextualStrings` (classes on a recognized component), `vocabularyStrings` (classes inside helper calls, checked at their own site), and `unresolved` (what the collector could not read — reported as dynamic, never treated as empty). Framework differences are isolated in `packages/lint/src/sites/readers/jsx.ts`, `vue.ts`, and `svelte.ts`, with `readerFor` picking the right node types from parser services, and `packages/lint/src/project/parser.ts` handling single-file-component specifics.

**The policy engine.** `packages/lint/src/rules/contracts.ts` is where design-system rules are expressed. An option like `allow: ["layout"]` or a contract `{ pattern: "^Button$", allow: ["w-full", "mt-*", "mb-*"] }` compiles into an `EntrySet` of globs, category memberships, and class-group ids. Each token then gets a `Verdict` — `ok`, `denied`, or `not-allowed` — via `decide(component, token)`, memoized per component-and-token pair. The vocabulary of categories comes from `packages/lint/src/grammar/categories.ts`, whose `GROUP_CATEGORY` table maps every class group in cn's config to one of color, typography, spacing, shape, effects, and motion, or to layout — which is why a contract can say "allow layout" and mean it across the whole grammar. Two details stand out. First, compilation is cached per grammar and per option value (`compiledOnce`), so a thousand files compile a policy once. Second, an entry that matches nothing enforces nothing silently, which the module calls the one failure an enforcement tool must not have: near-misses of real names throw a `ContractConfigError` with a "did you mean" hint, and the rule reports the configuration error at the top of the file rather than enforcing a policy the team did not write.

**The class grammar.** `packages/lint/src/grammar/classifier.ts` builds a trie over cn's class groups that mirrors tailwind-merge's class map — exact `-`-split parts first, then validators on the tail, deepest match winning. Crucially, it resolves the project's own installed `cn/config` as the grammar (falling back to the bundled cn 0.3.2 with a one-time warning when the project's copy is older), so classification agrees with the merge behavior the app actually runs. The classifier handles variant prefixes and suffixes, `!important` markers, opacity modifiers, arbitrary properties, and legacy Tailwind 3 names that still generate CSS, and it memoizes the token-to-group lookup because a few thousand distinct tokens repeat across a project.

**The Tailwind oracle.** `no-unknown-classes` cannot answer "does Tailwind know this class" from grammar alone — plugins and `@theme` extensions are project-specific. So `packages/lint/src/tailwind/oracle.ts` resolves the project's own `tailwindcss` v4 and calls its `__unstable__loadDesignSystem` with the project's entry stylesheet, resolving `@import`s the way Tailwind's bundler does (including package `exports` conditions, per `resolveStylesheet`) and loading `@plugin`/`@config` modules with their mtimes in the URL so edits invalidate the module cache. Because the loader is asynchronous and lint rules are synchronous, `packages/lint/src/tailwind/client.ts` bridges through a worker thread with a `SharedArrayBuffer` and `Atomics.wait` — 15 seconds for the cold first question, 5 after that, one restart, and then the rule falls back to the bundled grammar plus `packages/lint/src/project/theme.ts` (which independently parses `@theme` tokens, declared `@utility` names, and custom classes from the project's CSS). "Refusing to judge beats judging against half a theme," as the oracle's own comment puts it.

**End to end.** A file is linted; the collector hands the rule a `ClassSite` with every class string and its resolved component; `no-restyle` splits the string into tokens and asks the compiled contract for a verdict on each; a `not-allowed` spacing token on a Button produces a message that names the token, explains that the component owns its spacing, lists the real sizes from `packages/lint/src/project/variants.ts`, and computes where spacing could go instead by asking the contracts which parents accept the class. `packages/lint/src/rules/messages.ts` interpolates placeholders (a custom message can use slots like <code>&#123;&#123;sizes&#125;&#125;</code> or <code>&#123;&#123;file&#125;&#125;</code>, with a fallback syntax for empty slots) and appends the optional `settings.shadcn.note`. The agent reads one paragraph, applies the fix, and re-runs the same command.

## Advantages

- **Agent-first diagnostics.** Errors say what broke, what to use instead, and where to find it — variant names, size names, theme file paths — with per-category message overrides when your system needs its own words.
- **Two linters, one rule set.** Identical rule ids and options across ESLint and Oxlint, so teams can switch runners without rewriting a single contract.
- **Three frameworks from one engine.** The JSX, Vue, and Svelte readers in `packages/lint/src/sites/readers` give the same rules to template expressions, helper calls like `cn` and `cva`, one-hop variable resolution, and wrapper forwarding.
- **Policies without forks.** Contracts match components by name through the import graph, so rules apply to components you copied, imported, or do not own at all.
- **Real Tailwind, not a guess.** `no-unknown-classes` consults the project's own Tailwind v4 design system, imports and plugins included, with the bundled grammar only as a fallback.
- **Honest failure modes.** Policies that would silently match nothing are configuration errors; unreadable dynamic classes are their own violation; a worker failure degrades with a warning instead of pretending the check passed.

## Benefits

- **Appearance stays in the variants.** `no-restyle` pushes layout to parents and appearance back into `components/ui`, which is where a design system can actually govern it.
- **Theme consistency by default.** `no-raw-colors` and `no-arbitrary-values` keep utilities on your theme's scales, and suggestions name the nearest token so the fix is a mechanical edit.
- **A verifiable loop for agents.** One lint command is the finish line; the README reports that across more than 150 tested task runs, almost every task reached zero violations in a single correction round.
- **Cheaper corrections, by their measurement.** The same README reports 10% to 48% lower fix cost with lint feedback than with rules alone in its Claude control runs — the evals methodology is published in the repository's `docs/evals.md`.
- **Monorepo-ready recognition.** `settings.shadcn` accepts import prefixes, regex `componentImports`, ignore patterns, and custom merge/variant functions; `components.json` projects get discovery for free, and rule overrides can turn rules off inside the UI package itself.
- **Programmable voice.** Placeholders and the shared `note` setting let every diagnostic carry your design system's instructions, not just the rule's default wording.

## Usage

Install for ESLint (React projects; requires ESLint 9.30+ and Node 20.19+):

```bash
npm install -D @shadcn/lint eslint @typescript-eslint/parser
```

Create `eslint.config.mjs`:

```js
import { plugin as shadcn } from "@shadcn/lint"
import tsParser from "@typescript-eslint/parser"
import { defineConfig } from "eslint/config"

export default defineConfig([
  {
    files: ["**/*.{js,jsx,ts,tsx}"],
    languageOptions: {
      parser: tsParser,
      parserOptions: { ecmaFeatures: { jsx: true } },
    },
    plugins: { shadcn },
    rules: {
      "shadcn/no-arbitrary-values": "error",
    },
  },
])
```

Run it:

```bash
npx eslint .
```

For Oxlint (1.80+), create `.oxlintrc.json` instead:

```json
{
  "jsPlugins": ["@shadcn/lint"],
  "rules": {
    "shadcn/no-arbitrary-values": "error"
  }
}
```

```bash
npx oxlint
```

The flagship rule is `no-restyle`, configured with per-component contracts:

```js
"shadcn/no-restyle": ["error", {
  allow: ["layout"],
  contracts: [
    { pattern: "^Button$", allow: ["w-full", "mt-*", "mb-*"] },
  ],
}]
```

Finally, wire the loop into your agent instructions. The README suggests putting this in `AGENTS.md`:

```md
After making changes, run `npm run lint` and fix all errors.
```

## Conclusion

`shadcn-ui/lint` is a small package with a sharp thesis: design-system rules only work if the reader can comply with them, and today that reader is increasingly a machine. The source follows through on the thesis with unusual discipline — a policy engine that refuses to enforce a policy it cannot understand, a grammar that defers to the project's own merge behavior, an oracle that asks the real Tailwind instead of approximating it, and diagnostics that name the fix rather than the crime. If you maintain a Tailwind design system and your contributors include coding agents, reading `packages/lint/src/rules/contracts.ts` alongside `docs/rules.md` is the fastest way to see how "what is allowed" becomes something an agent can verify.

Links:

- GitHub repository: <https://github.com/shadcn-ui/lint>
- Documentation: <https://github.com/shadcn-ui/lint/blob/main/docs/README.md>
- Agent setup guide: <https://github.com/shadcn-ui/lint/blob/main/SETUP.md>
