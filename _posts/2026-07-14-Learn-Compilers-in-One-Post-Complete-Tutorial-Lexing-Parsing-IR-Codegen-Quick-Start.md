---
layout: post
title: "Learn Compilers in a Single Post: A Complete Tutorial From Lexing and Parsing to IR Optimization and Code Generation"
description: "A complete compilers tutorial in one blog post. Covers the whole pipeline in 5 stages: lexing (tokens, regex, finite automata, DFA), parsing (grammars, parse trees, AST, recursive descent), semantics (scope, types, symbol tables, checking), IR + optimization (SSA, constant folding, dead code elimination, inlining), and code generation (instruction selection, register allocation, LLVM). Five hand-drawn diagrams, runnable code, and a quick-start roadmap."
date: 2026-07-14
header-img: "img/post-bg.jpg"
permalink: /Learn-Compilers-in-One-Post-Complete-Tutorial-Lexing-Parsing-IR-Codegen-Quick-Start/
featured-img: ai-coding-frameworks/ai-coding-frameworks
tags:
  - Compilers
  - Lexing
  - Parsing
  - LLVM
  - Computer Science
  - Tutorial
categories: [Tutorial, Computer Science, Compilers]
keywords: "compilers tutorial one post, learn compilers fast, lexing tokens regex finite automata DFA, parsing grammar BNF parse tree AST recursive descent, semantic analysis scope types symbol table, intermediate representation IR SSA form, compiler optimization passes constant folding dead code inlining, code generation instruction selection register allocation, LLVM GCC framework, AOT vs JIT interpreter bytecode, Crafting Interpreters, compilers quick start roadmap"
author: "PyShine"
---

# Learn Compilers in a Single Post: Complete Tutorial From Lexing and Parsing to IR Optimization and Code Generation

A compiler is a translator: it takes source code in one language and produces equivalent code in another — usually machine code. Understanding how a compiler works is the deepest way to understand what code *is*, what your language *does*, and why optimization matters. It's also the bridge between the [operating system](/Learn-Operating-Systems-in-One-Post-Complete-Tutorial-Processes-Memory-Threads-Quick-Start/) (which runs the output) and the language (which produces the input). This single post teaches the whole pipeline in five stages, with hand-drawn diagrams and runnable code.

## Learning Roadmap

![Compilers Learning Roadmap](/assets/img/diagrams/compilers-tutorial/comp-roadmap.svg)

The roadmap moves from the front-end (lexing, parsing, semantics) through the middle (IR + optimization) to the back-end (code generation). You'll want [data structures and algorithms](/Learn-Data-Structures-and-Algorithms-in-One-Post-Complete-Tutorial-Big-O-Trees-Graphs-DP-Quick-Start/) — trees, graphs, recursion, and finite automata are the backbone.

---

## Stage 1 — Lexing

### What lexing does

The compiler's first job is to turn a raw string of characters into a stream of **tokens** — the smallest meaningful units of the language.

```
source:  x = 2 + 3 * y
tokens:  [ID(x), ASSIGN, NUM(2), PLUS, NUM(3), STAR, ID(y), EOF]
```

### Tokens and lexing rules

Each token type is defined by a **pattern** (usually a regular expression):

| Token | Pattern | Example |
|---|---|---|
| `ID` (identifier) | `[a-zA-Z_][a-zA-Z0-9_]*` | `x`, `myVar_2` |
| `NUM` (number) | `[0-9]+` | `42`, `0` |
| `PLUS` | `\+` | `+` |
| `STAR` | `\*` | `*` |
| `WS` (whitespace) | `[ \t\n]+` | (skipped) |

```python
import re

TOKEN_SPEC = [
    ('NUM',   r'\d+'),
    ('ID',    r'[a-zA-Z_][a-zA-Z0-9_]*'),
    ('PLUS',  r'\+'),
    ('STAR',  r'\*'),
    ('ASSIGN',r'='),
    ('WS',    r'[ \t\n]+'),
    ('EOF',   r'$'),
]

def lex(source):
    tokens = []
    pos = 0
    while pos < len(source):
        for kind, pattern in TOKEN_SPEC:
            m = re.match(pattern, source[pos:])
            if m:
                if kind != 'WS':  # skip whitespace
                    tokens.append((kind, m.group()))
                pos += len(m.group())
                break
        else:
            raise SyntaxError(f"unexpected char: {source[pos]}")
    tokens.append(('EOF', ''))
    return tokens

print(lex("x = 2 + 3 * y"))
# [('ID','x'), ('ASSIGN','='), ('NUM','2'), ('PLUS','+'), ('NUM','3'), ('STAR','*'), ('ID','y'), ('EOF','')]
```

### Finite automata

Under the hood, a lexer is a **finite automaton** (DFA): a state machine that reads one character at a time and transitions between states. The regex `\d+` compiles to a DFA that accepts "one or more digits." Lexing is `O(n)` in the input length — it's a single pass, never backtracks.

> **Pitfall:** The lexer must handle the **maximal munch** rule: `>>` in C++ could be a right-shift operator or (in templates) two closing angle brackets. The lexer greedily takes the longest match, which is usually right — but template syntax like `vector<vector<int>>` needed a language-level fix (C++11 allows `>>` as two brackets).

---

## Stage 2 — Parsing

### Grammars and parse trees

A **grammar** defines the structure of the language: which sequences of tokens are valid. It's a set of **production rules** in BNF (Backus-Naur Form):

![Grammar -> Parse Tree -> AST](/assets/img/diagrams/compilers-tutorial/comp-grammar.svg)

```
expr  := term + expr  | term
term  := factor * term | factor
factor := NUM | ( expr )
```

The grammar defines **precedence** (`*` binds tighter than `+`) and **associativity** through its structure: `term + expr` means `*` is parsed first (deeper in the tree).

### Parse tree vs AST

The **parse tree** records every grammar step (every non-terminal expansion). The **AST (abstract syntax tree)** drops the syntax sugar (parentheses, intermediate non-terminals) and keeps only the *meaning*:

```
Parse tree:              AST:
    expr                    Add
   / | \                   / \
 term  +  expr            Num   y
  |          |             2
 factor    term
  |         |
  2       factor
            |
            y
```

The AST is what the rest of the compiler works with — it's the semantic structure of the program.

### Recursive descent — the hand-written parser

**Recursive descent** is the simplest parsing technique: one function per grammar rule, each calling the next:

```python
class Parser:
    def __init__(self, tokens):
        self.tokens = tokens; self.pos = 0
    def peek(self): return self.tokens[self.pos]
    def advance(self): t = self.peek(); self.pos += 1; return t
    def expect(self, kind):
        if self.peek()[0] != kind: raise SyntaxError(f"expected {kind}, got {self.peek()}")
        return self.advance()

    # expr := term (+ expr | - expr | ε)
    def parse_expr(self):
        left = self.parse_term()
        if self.peek()[0] in ('PLUS', 'MINUS'):
            op = self.advance()
            right = self.parse_expr()
            return ('BinOp', op[0], left, right)
        return left

    # term := factor (* term | / term | ε)
    def parse_term(self):
        left = self.parse_factor()
        if self.peek()[0] in ('STAR', 'SLASH'):
            op = self.advance()
            right = self.parse_term()
            return ('BinOp', op[0], left, right)
        return left

    # factor := NUM | ( expr )
    def parse_factor(self):
        if self.peek()[0] == 'NUM':
            return ('Num', int(self.advance()[1]))
        if self.peek()[0] == 'LPAREN':
            self.advance()
            e = self.parse_expr()
            self.expect('RPAREN')
            return e
        raise SyntaxError(f"unexpected: {self.peek()}")
```

Each function matches its grammar rule, calling the others to handle sub-expressions. Precedence comes from the call structure: `parse_term` is called *inside* `parse_expr`, so `*` is parsed before `+`.

### Parser generators and other techniques

| Technique | When | Tools |
|---|---|---|
| **Recursive descent** | hand-written, readable | most production compilers |
| **LL(k)** | top-down, predictive | ANTLR |
| **LALR / LR(1)** | bottom-up, handles more grammars | yacc/bison |
| **PEG** | parsing expression grammar, no ambiguity | pest (Rust) |
| **Pratt parser** | operator precedence, elegant for expressions | many DSLs |
| **tree-sitter** | incremental, error-recovering (for editors) | NeoVim, GitHub |

> **Pitfall:** **Left recursion** (`expr := expr + term`) causes infinite loops in recursive descent (the function calls itself before consuming input). Rewrite to right recursion (`expr := term + expr`) or use iterative parsing. Parser generators (LR/LALR) handle left recursion natively.

---

## Stage 3 — Semantic Analysis

### What the semantic phase does

Parsing only checks *syntax* (structure). **Semantics** checks *meaning*: are the types right? Are variables declared before use? Is the number of arguments to a function correct?

### Scope and symbol tables

A **symbol table** maps names to their declarations (type, scope, memory location). It tracks **scope** — which names are visible where:

```python
class Scope:
    def __init__(self, parent=None):
        self.symbols = {}; self.parent = parent
    def define(self, name, type_):
        self.symbols[name] = type_
    def lookup(self, name):
        if name in self.symbols: return self.symbols[name]
        if self.parent: return self.parent.lookup(name)
        raise NameError(f"undefined: {name}")
```

Nested scopes (function body inside function, block inside function) form a chain; `lookup` walks up to the parent. This is how a variable inside a function sees globals but not vice versa.

### Type checking

```python
def check(node, scope):
    if node[0] == 'Num': return 'int'
    if node[0] == 'BinOp':
        op, left, right = node[1], check(node[2], scope), check(node[3], scope)
        if op in ('PLUS', 'STAR') and left == right: return left
        raise TypeError(f"{op}: {left} and {right} don't match")
    if node[0] == 'Var':
        return scope.lookup(node[1])
    if node[0] == 'Assign':
        vtype = scope.lookup(node[1])
        rtype = check(node[2], scope)
        if vtype != rtype: raise TypeError(f"assign {vtype} = {rtype}")
        return vtype
```

Type checking walks the AST, computing the type of each expression and verifying constraints. A **statically typed** language (C, Rust, Java) catches all type errors at compile time; a **dynamically typed** language (Python, JS) defers them to runtime (the "semantic" phase is lighter).

> **Pitfall:** Type checking with **type inference** (Rust, Haskell, OCaml) is much harder — the compiler deduces types without annotations. This requires a constraint-solving algorithm (Hindley-Milner, or more complex). Most teaching compilers start with explicit annotations.

---

## Stage 4 — IR + Optimization

### The intermediate representation

![The Compiler Pipeline](/assets/img/diagrams/compilers-tutorial/comp-pipeline.svg)

After semantics, the compiler translates the AST into an **intermediate representation (IR)** — a lower-level, machine-independent form. The IR is the shared "middle" of the compiler: the front-end is source-specific (Python, C, Rust), the back-end is target-specific (x86, ARM, RISC-V), but the IR and its optimization passes are **shared**.

```
AST:    Add(Mul(Num 2, Var y), Var x)
IR:     t0 = 2 * y
        t1 = x + t0
```

LLVM's IR is the most widely used: a typed, SSA-based, infinite-register IR that dozens of languages (C, C++, Rust, Swift, Zig) compile to and dozens of backends consume.

### SSA (static single assignment)

In **SSA form**, every variable is assigned exactly **once**. If a variable is reassigned, it gets a new version:

```
before SSA:          after SSA:
x = 1                x1 = 1
x = 2                x2 = 2
y = x + 1            y1 = x2 + 1
```

SSA makes optimization passes dramatically simpler: no aliasing, no "which definition of x is this using?" — each version has one definition. Nearly every modern compiler (LLVM, GCC, V8, Cranelift) uses SSA.

### Optimization passes

![IR + Optimization Passes](/assets/img/diagrams/compilers-tutorial/comp-optim.svg)

| Pass | What it does |
|---|---|
| **Constant folding** | `2 + 3` → `5` at compile time |
| **Dead code elimination** | remove assignments whose result is never used |
| **Common subexpression elimination** | `a+b; a+b` → compute once, reuse |
| **Inlining** | replace a function call with the function body |
| **Loop unrolling** | repeat the loop body N times (fewer branch overheads) |
| **Strength reduction** | `x * 2` → `x + x` or `x << 1` (faster on most CPUs) |
| **Copy propagation** | if `y = x`, use `x` directly, drop the copy |
| **Constant propagation** | if `x = 5`, replace uses of `x` with `5` |

```python
# constant folding example
def fold(node):
    if node[0] == 'BinOp' and node[1] in ('PLUS', 'STAR'):
        left = fold(node[2]); right = fold(node[3])
        if left[0] == 'Num' and right[0] == 'Num':
            if node[1] == 'PLUS': return ('Num', left[1] + right[1])
            if node[1] == 'STAR': return ('Num', left[1] * right[1])
        return ('BinOp', node[1], left, right)
    return node

print(fold(('BinOp', 'PLUS', ('Num', 2), ('Num', 3))))   # ('Num', 5)
```

Optimization passes run **repeatedly** until a fixed point (no more changes). The order matters: inlining enables more constant propagation; constant propagation enables more dead code elimination. Modern compilers run dozens of passes, each `O(n)` or `O(n log n)` in the IR size.

> **Pitfall:** Optimization can change behavior if the compiler assumes things that aren't true — signed integer overflow (undefined behavior in C), floating-point reassociation (changes results), or aliasing (the compiler assumes two pointers don't overlap). These are why `-O2` can make code "break" that worked at `-O0`. Understanding UB is a compiler-writer's daily life.

---

## Stage 5 — Code Generation

### Instruction selection

The back-end translates IR operations into **target machine instructions**:

```
IR:       t0 = 2 * y
x86-64:   mov rax, 2
          imul rax, [y]
```

This is **instruction selection** — choosing the right instructions for the target. A multiply might be one instruction (`imul`) or a shift-add sequence depending on the constant and the ISA.

### Register allocation

The CPU has a fixed number of registers (16 on x86-64). The IR uses infinite "virtual registers"; the back-end must map them to the physical set, **spilling** overflow to the stack. This is modeled as a **graph coloring** problem: variables that are live at the same time "interfere" (can't share a register); color the interference graph with `k` colors (k = number of registers). It's NP-hard, so compilers use heuristics (linear scan, Chaitin's algorithm).

### The LLVM ecosystem

![Compilers Ecosystem + Tools](/assets/img/diagrams/compilers-tutorial/comp-ecosystem.svg)

**LLVM** is the dominant compiler framework: you write a front-end (your language → LLVM IR), and LLVM gives you the optimizer + back-ends for x86, ARM, RISC-V, WebAssembly, and more. Rust, Swift, Zig, Clang (C/C++), and many research languages target LLVM IR — the "write one front-end, get all the back-ends" payoff.

**GCC** is the classic alternative (its own IR, GIMPLE), and **Cranelift** is a fast (less-optimized) IR used by Rust's debug builds and Wasmtime for JIT.

### AOT, JIT, interpreters

| Strategy | When | Examples |
|---|---|---|
| **AOT** (ahead-of-time) | compile before run, fast execution | C, C++, Rust, Go |
| **JIT** (just-in-time) | compile at runtime, profile-guided | V8 (JS), JVM (hotspot), PyPy, LuaJIT |
| **Interpreter** | no compile, walk the AST/bytecode | Python (CPython), Ruby, early JS |
| **Bytecode VM** | compile to bytecode, interpret that | Python (.pyc), JVM (.class), Lua |

---

## Quick-Start Checklist

1. **Write a lexer** — turn `"2 + 3 * x"` into tokens. It's a single loop + regex match.
2. **Write a recursive-descent parser** — one function per grammar rule; produce an AST.
3. **Add a type checker** — walk the AST with a symbol table; catch undefined names and type mismatches.
4. **Translate AST to IR** — flatten the tree into linear three-address code (`t0 = 2 * y`).
5. **Add constant folding** — the simplest optimization: `2 + 3` → `5` at compile time.
6. **Add dead code elimination** — remove assignments whose results are never used.
7. **Generate output** — even just "compile to Python" (transpile the AST back to Python source) is a real compiler.
8. **Read *Crafting Interpreters*** — build a full interpreter + compiler from scratch, in two languages.
9. **Try LLVM's Kaleidoscope tutorial** — write a real front-end that targets LLVM IR and gets x86 output for free.
10. **Study a small real compiler** — `chibicc` (a tiny C compiler in ~1000 lines) is the best "read a real compiler" exercise.

## Common Pitfalls

- **Left recursion in recursive descent** — `expr := expr + term` loops forever; rewrite to right recursion or iterative.
- **Maximal munch in lexing** — `>>` should be one token (right shift) or two (close template brackets); the lexer greedily takes the longest match, which may be wrong for the grammar.
- **Ambiguous grammars** — a grammar where the same string can parse two ways (the classic "dangling else"). Use precedence rules or rewrite the grammar to be unambiguous.
- **SSA at phi nodes** — when control flow merges (if/else), you need **phi nodes** to say "this variable is version A from one branch, version B from the other." This is the hard part of SSA.
- **Register allocation is NP-hard** — the graph-coloring approach is heuristic; compilers approximate. Don't expect optimal allocation.
- **Optimization changes results** — FP reassociation, signed-overflow UB, and pointer aliasing assumptions can make `-O2` produce different (wrong, per the spec) results than `-O0`. Understand what the language spec allows.
- **Error recovery** — a real compiler reports multiple errors per run, not just the first. This requires the parser to recover (skip to a semicolon, synchronize) and continue. Teaching compilers usually stop at the first error.

## Further Reading

- [Crafting Interpreters](https://craftinginterpreters.com/) by Robert Nystrom — build a full language from scratch, free online, the best starting point
- [Compilers: Principles, Techniques, and Tools (the Dragon Book)](https://www.pearson.com/en-us/subject-catalog/p/compilers-principles-techniques-and-tools/P200000003497) by Aho et al — the classic textbook (heavy, thorough)
- [LLVM Kaleidoscope Tutorial](https://llvm.org/docs/tutorial/) — build a language that compiles to LLVM IR and x86
- [chibicc](https://github.com/rui314/chibicc) — a small C compiler in ~1000 lines, readable
- [Engineering a Compiler](https://www.elsevier.com/books/engineering-a-compiler/cooper/978-0-12-815412-1) by Cooper & Torczon — modern, practical
- [Essentials of Compilation](https://www.routledge.com/Essentials-of-Compilation/Keep/9780415745011) by Jeremy Siek — a course-tested, incremental approach

## Related guides

Compilers sit at the intersection of CS theory and systems — these PyShine tutorials connect to it:

- **[Learn Data Structures and Algorithms in One Post](/Learn-Data-Structures-and-Algorithms-in-One-Post-Complete-Tutorial-Big-O-Trees-Graphs-DP-Quick-Start/)** — trees (AST), graphs (register allocation), finite automata (lexing), and graph coloring are all DSA.
- **[Learn Operating Systems in One Post](/Learn-Operating-Systems-in-One-Post-Complete-Tutorial-Processes-Memory-Threads-Quick-Start/)** — the compiled code runs on the OS; understand process memory, registers, and the call stack.
- **[Learn C++ in One Post](/Learn-CPP-in-One-Post-Complete-Tutorial-Modern-Cpp-Quick-Start/)** — C++ is the language most compilers are written *in* and compile *to* (LLVM is C++).
- **[Learn Rust in One Post](/Learn-Rust-in-One-Post-Complete-Tutorial-Ownership-Borrow-Async-Quick-Start/)** — Rust's compiler (rustc) is one of the most sophisticated LLVM front-ends; ownership is a compile-time check.
- **[Learn Python in One Post](/Learn-Python-in-One-Post-Complete-Tutorial-Async-Type-Hints-Quick-Start/)** — CPython is a bytecode VM + interpreter; the snippets above are Python.

---

A compiler is the most complete systems project there is: it touches theory (automata, grammars, graph coloring), engineering (pipelines, optimization passes, IR design), and low-level systems (registers, calling conventions, memory layout). The five stages here — lexing, parsing, semantics, IR + optimization, code generation — are the structure of every compiler from `gcc` to `rustc` to V8. The best way to learn it is to build one: write a lexer, then a parser, then a type checker, then a code generator — even if the "target" is just Python (a transpiler). The *Crafting Interpreters* book walks you through exactly this, and once you've compiled `2 + 3 * x` to `5` at compile time, you'll never look at a compiler error message the same way.