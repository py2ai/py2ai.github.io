---
layout: post
title: "Fauxnix: Bash For Your Windows Agent, No VM Required - Inside 20000419/fauxnix"
description: "Fauxnix is a deterministic bash-to-PowerShell translation layer built for AI agents: your agent keeps writing bash, fauxnix compiles it to native PowerShell and returns GNU-style output. We tour the source behind its 109 translated commands, 253-case differential corpus, and zero-LLM runtime."
date: 2026-10-02
header-img: "img/post-bg.jpg"
permalink: /Fauxnix-Run-Linux-Commands-On-Windows-Without-A-VM/
featured-img: "ai-coding-frameworks/ai-coding-frameworks"
image: https://pyshine.com/assets/img/diagrams/fauxnix/20000419-fauxnix-architecture.svg
tags:
  - AI Agents
  - CLI Tools
  - PowerShell
  - Open Source
categories: [AI, Open Source]
keywords: "fauxnix, bash to powershell translator, windows ai agent, MCP server, claude code windows, codex windows, deterministic shell translation, GNU coreutils output, UTF-8 GBK, no WSL"
author: "PyShine"
---

Ask a coding agent to work on a Windows machine and watch what happens to its shell discipline. The model was trained on decades of bash, so it writes `ls -la | grep foo` and `find . -name '*.ts' | wc -l` with total confidence — and then falls over translating that intent into PowerShell: wrong quoting, a `curl` that is not curl, mojibake from a codepage mismatch, or a `CategoryInfo` error dump where a one-line complaint should be. The usual fixes are heavyweight: install Git Bash and hope the harness detects it, or boot WSL and pay gigabytes for a VM.

Fauxnix takes a third road that the README states plainly: **translate, don't emulate**. It is a TypeScript package that parses a large, high-value subset of bash — file operations, text filters, process management, archives, networking basics — and compiles each command into a self-contained PowerShell script that runs natively on Windows. The output that comes back looks like GNU/Linux: `ls -l` columns, bash-style error messages, coreutils exit codes. There are no LLM calls anywhere in the runtime; translation is deterministic and auditable. The project ships 109 translated commands, more than 400 automated tests, and a 253-case differential corpus verified against real GNU coreutils running in Git Bash.

The source is worth a close look because it is a masterclass in a neglected discipline: making one platform *tell the truth* to software that expects another. Every deviation from bash is documented in an honest list, every unsupported construct fails loudly with a named error instead of silently misbehaving, and the differential-testing setup — using Git Bash itself as the oracle — shows how you can hold a translator to a measurable standard rather than a vibe.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/fauxnix/20000419-fauxnix-overview-architecture.svg" alt="Architecture overview of the 20000419/fauxnix repository" style="max-width:100%;height:auto;" />
</div>

*The system at a glance: a CLI and an MCP server share one translation pipeline — parser to AST to command generators — and hand the compiled script to an executor that speaks a UTF-8 framed protocol with Windows PowerShell; installers and doctor checks wire the whole thing into agent harnesses.*

Reading the overview from left to right: a command enters through the CLI or the MCP server; the parser turns the bash subset into an AST; the translator dispatches to per-command generators and emits a PowerShell script; the executor runs it through the PowerShell host, decoding output through the UTF-8/GBK layer and rewriting errors into bash phrasing; and the installer and doctor make the same machinery available to Claude Code, Codex, OpenCode, Kimi Code and Qwen Code.

## Why You Need This

The core insight is about training data, and the README argues it well: bash dominates the corpus that models learned from, so an agent's bash is fluent while its PowerShell is improvised. Fauxnix removes the translation burden from the model entirely. Your agent keeps writing the shell it already knows, and the Windows box answers like the box the agent was trained on. Nothing about the agent's harness needs to change, because the tool surface — or the MCP tool description — teaches the supported subset itself.

The measured section of the README makes the case with numbers rather than adjectives. Running the same five tasks with the same model in three modes, raw PowerShell needed 14 tool calls and produced 9 unexpected errors in 163 seconds; fauxnix finished in 66 seconds with 7 calls and zero errors; Git Bash itself took 57 seconds with zero errors. Across seven models on one provider's coding plan the gap held for every model tested, with the worst case over three times slower writing PowerShell directly, and fauxnix landing within roughly fifteen percent of the real-bash ceiling with no bash toolchain installed at all.

There is also a positioning argument that many teams will recognize: WSL is a VM measured in gigabytes and a separate filesystem view; Git Bash is a toolchain that can drift or go undetected (the README points to a live issue where Claude Code failed to detect Git Bash on Windows ARM64); CI runners and locked-down machines often allow neither. Fauxnix needs only Node.js and the PowerShell that is already built into Windows. A single `npm install -g` replaces the whole toolchain conversation.

Finally, the project is honest in a way that builds trust. The README's "known deviations" section lists exactly where behavior differs from bash — from `chmod` mapping only the read-only bit to `yes` being capped at 65,536 lines because PowerShell 5.1 pipelines cannot signal upstream producers to stop — and commands that cannot be translated faithfully are rejected with actionable errors rather than returning subtly wrong results. For a tool whose users are autonomous agents, fail-loud semantics are not a nicety; they are the difference between a debuggable system and a haunted one.

## How It Works

The pipeline is a classic compiler shape, and the README's architecture map names it directly: `src/parser.ts` parses a bash subset into an AST, `src/translator.ts` walks that AST into a PowerShell script plus an executor wrapper, and `src/executor.ts` spawns, redirects and keeps sessions alive.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/fauxnix/20000419-fauxnix-architecture.svg" alt="Detailed architecture of the 20000419/fauxnix repository" style="max-width:100%;height:auto;" />
</div>

*The detailed architecture: the translator dispatches to specialized generator modules for files, text filters, archives, networking and system commands; the executor layers encoding, error rewriting and runtime support on top of the PowerShell host; and the differential oracle replays a 253-case corpus against Git Bash.*

### Understanding the Architecture

**A real parser, not string munging.** `src/parser.ts` and `src/ast.ts` implement a genuine bash grammar for the supported subset: pipes, `&&`/`||`/`;`, redirections including `2>&1` and `/dev/null`, quoting, variables and parameters, `${name:-word}` style expansions, arrays, command substitution, `VAR=x cmd` prefixes, and control flow from `if/then` through `case ... esac`. Because the structure is parsed before anything runs, unsupported constructs — heredocs, background `&`, output redirects on a non-last pipeline stage — are rejected at translate time with operation-specific messages, never discovered at runtime as garbage.

**Per-command generators honor a strict contract.** The translator dispatches to modules under `src/commands/`: file operations, text filters (`grep`, `sed`, `awk` with their own parsers), text I/O, archives, networking, and system info. Each generator emits a self-contained PowerShell block that honors what the README calls the fauxnix contract — one string per line on stdout, bash-style stderr through `[Console]::Error.WriteLine`, the exit code in `$script:fx_exit`, stdin via `$input`. A curated set of agent-daily commands carries a `CommandSpec` in `src/registry.ts`, so an unknown flag fails with a GNU-style usage error instead of being silently ignored.

**The executor is the Windows whisperer.** `src/executor.ts` runs scripts via `-EncodedCommand` (UTF-16LE) with a transparent fallback to a temporary `.ps1` file when the 32 KB command-line limit would bite, strips CLIXML serialization and PowerShell noise from stderr, and rewrites common PowerShell errors — including zh-CN locale messages — into bash phrasing. `src/encoding.ts` enforces UTF-8 at the process boundary, decodes native output as UTF-8 or GBK in `ansi` mode, and sniffs file reads per file with UTF-8-strict-to-GBK fallback, which is what lets `grep` and `sed` work over GBK files in either mode.

**The MCP server behaves like a logged-in shell.** `src/mcp.ts` exposes a `bash` tool whose session persists `cwd`, environment variables, `export`/`unset`, `cd -`/OLDPWD and positional parameters across calls — not a stateless `exec`. A companion `bash_batch` tool compiles every step before execution, runs the plan atomically in one session, and returns one structured result per step, stopping on the first nonzero exit. The experimental `src/facade.ts` goes further: it implements the bash process contract (`facade -c "<cmd>"` with bash argv semantics) so a harness can eventually point its built-in Bash tool at fauxnix directly.

**Quality is differential, not aspirational.** `test/differential/` replays a 253-case corpus (`corpus.json`) against real GNU coreutils in Git Bash — the project's own testing oracle — enforcing a 95 percent identity gate; a weekly workflow keeps the check alive in CI. The suite splits into portable unit tests and Windows-only integration tests that exercise the real PowerShell, so the guarantee is about actual runtime behavior, not mocks.

**Onboarding is a product surface.** `src/install.ts` configures Claude Code, Codex, OpenCode, Kimi Code or Qwen Code idempotently, printing exactly what changed; `src/doctor.ts` verifies encoding, harness configuration and an MCP round-trip. `src/powershell.ts` keeps Windows PowerShell 5.1 as the default host with PowerShell 7 as an opt-in, CI-tested tier (`FAUXNIX_PS=pwsh`) — invalid values fail loudly rather than silently falling back.

End to end: the agent writes bash; the parser and translator compile it into a contract-honoring PowerShell script with zero LLM involvement; the executor runs it natively, decoding output and rewriting errors; and GNU-shaped text comes back through MCP or the CLI — with the differential corpus standing guard over the whole promise.

## Advantages

- **Deterministic by design.** No LLM calls at runtime means the same bash always compiles to the same PowerShell — auditable, cacheable, and free of inference cost or latency.
- **GNU-honest output.** `ls -l` columns, bash-style error strings, and coreutils exit codes (including 127 command-not-found and 124 timeout) keep agent reasoning on familiar ground.
- **Fail-loud semantics.** Unknown flags, unsupported constructs and network-guarded addresses produce named, actionable errors — the agent is never handed silently-wrong results.
- **No toolchain to ship.** Node.js plus built-in PowerShell 5.1 is the entire dependency stack; no VM, no Git Bash, no WSL image, nothing for fleet management to drift.
- **Session-real MCP.** Persistent `cwd`, environment, and positional parameters across tool calls, plus an atomic compiled `bash_batch` mode that trades round trips for one structured plan.
- **Encoding traps handled.** UTF-8/GBK detection, CRLF discipline, and zh-CN error-message rewriting address the exact failure class that makes agents miserable on Windows.

## Benefits

- **Agents finish tasks faster.** In the project's own measurements, fauxnix cut a five-task benchmark from 163 seconds of error-prone PowerShell to 66 seconds with zero unexpected errors.
- **A measurable fidelity standard.** The 253-case differential corpus and 95 percent identity gate turn "works like bash" from a claim into a regression-tested contract.
- **Works with the harness you have.** One-command installers for five agent harnesses plus a generic stdio MCP server; the tool description itself teaches the supported subset, so no system-prompt surgery is needed.
- **Safety defaults that fit autonomy.** `curl` and `wget` refuse loopback and private addresses by default; the security model is documented in SECURITY.md rather than implied.
- **An honest upgrade path.** Documented deviations, RFC-driven roadmap, and loud rejection of untranslatable constructs mean surprises are scheduled, not emergent.
- **A template for tool builders.** The parser-translator-executor split, the per-command generator contract, and the differential oracle are reusable patterns for anyone bridging platform gaps for agents.

## Usage

Try it with no install:

```bash
npx fauxnix-cli@latest "ls -la src | head -3"
npx fauxnix-cli@latest translate "find . -name '*.log' -mtime +7 -delete"
```

Install globally and wire it into your agent harness in one command:

```bash
npm install -g fauxnix-cli
fauxnix install --claude     # or --codex / --opencode / --kimi / --qwen
fauxnix doctor               # verifies encoding, harness config, and MCP round-trip
```

Manual MCP registration for Claude Code, or any MCP client:

```bash
claude mcp add fauxnix -- fauxnix mcp
```

```json
{
  "mcpServers": {
    "fauxnix": { "command": "fauxnix", "args": ["mcp"] }
  }
}
```

Development builds run the full unit plus real-PowerShell integration suite:

```powershell
npm install
npm test
npm run build
```

## Conclusion

Fauxnix solves a problem that most agent tooling pretends does not exist: the shell gap between what models know and what Windows offers. Rather than emulating a Linux userland or asking models to improvise in a dialect they handle poorly, it compiles a faithful bash subset into native PowerShell with a strict output contract, fails loudly where fidelity ends, and proves the whole thing with a differential corpus against real GNU coreutils. For agent fleets on Windows machines — CI runners, locked-down desktops, anywhere a bash toolchain is one dependency too many — it is the rare bridge that is both pragmatic and principled.

**Links:**

- Repository: [github.com/20000419/fauxnix](https://github.com/20000419/fauxnix)
- npm package: [npmjs.com/package/fauxnix-cli](https://www.npmjs.com/package/fauxnix-cli)
- Command specs reference: [docs/command-specs.md](https://github.com/20000419/fauxnix/blob/main/docs/command-specs.md)
- Differential testing: [test/differential/README.md](https://github.com/20000419/fauxnix/blob/main/test/differential/README.md)
