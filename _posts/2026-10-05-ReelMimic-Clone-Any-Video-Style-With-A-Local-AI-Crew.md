---
layout: post
title: "ReelMimic: Clone Any Video's Style With A Local AI Crew - Inside edenfunf/reelmimic"
description: "ReelMimic breaks down a reference video's rhythm, shots, palette and camera moves, then a local crew of Claude Code or Codex agents plans, builds and reviews a brand-new 2D animation in the same style. A full source tour of its Node server, React app and agent workspace."
date: 2026-10-05
header-img: "img/post-bg.jpg"
permalink: /reelmimic-clone-any-video-style-with-a-local-ai-crew/
featured-img: "ai-coding-frameworks/ai-coding-frameworks"
image: https://pyshine.com/assets/img/diagrams/reelmimic/edenfunf-reelmimic-architecture.svg
tags: [AI Video, AI Agents, Automation, Open Source]
categories: [AI, Open Source]
keywords: ReelMimic, video style cloning, Claude Code, Codex CLI, multi-agent workflow, AI animation, local AI
author: "PyShine"
---

We have all watched a short film or music video and thought the same thing: I want something of my own that moves, cuts and lands its beats exactly like that. Training a video model is out of reach for most of us, and prompt-only generators give you a lottery ticket, not a style. ReelMimic, an MIT-licensed open-source project from edenfunf, attacks the problem from a completely different angle. You hand it a reference video (a local file, a phone recording, or a YouTube link), say in one line what you want to make, and it studies the reference the way an editor would: shot count, shot lengths, BPM and beat points, transitions, framing, color palette, and camera moves. Then a crew of AI agents running on your own machine plans, builds and reviews a brand-new 2D animation that reproduces the technique, while carrying over none of the footage, characters or assets.

What makes the project unusual is that the whole crew is not one mega-prompt. Up to six builder agents work on different parts of the video at the same time, and every finished shot is handed to a separate reviewer agent that had no part in making it. Fixes must come with before-and-after screenshots that the next review checks first. Everything runs locally through your own Claude Code or Codex account, so there are no new API keys to buy. The README is upfront about the cost in patience: a 30 to 60 second video typically takes one to three and a half hours after you approve the plan, because engines like watercolor paint every frame with brushes.

For engineers, ReelMimic is just as interesting as a blueprint for multi-agent orchestration. The three layers (a React web app, a Node server, and the agent workspace) talk to each other only through files on disk. The server decides whether an agent finished its job by checking modification times of the files it was supposed to write, not by parsing prose. This post walks through the real source tree so you can see how the pieces fit.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/reelmimic/edenfunf-reelmimic-overview-architecture.svg" alt="Architecture overview of the ReelMimic repository" style="min-width:640px;width:100%;">
</div>

*Architecture overview of the ReelMimic repository: the React app, the Node server, the agent workspace and the per-project data folder.*

Reading the overview from left to right: you start in the browser, where `app/web/src/App.tsx` renders the home composer and routes to the project page in `app/web/src/Project.tsx`. Both talk to the Node server's HTTP and Server-Sent Events endpoint in `app/server/index.ts`, which owns the workflow state machine in `app/server/jobs.ts` and dispatches agent turns through the adapters in `app/server/agents/index.ts`, with every step's instructions coming from `app/server/prompts.ts`. On the right side sits the agent workspace: the core `video-clone` skill at `.claude/skills/video-clone/SKILL.md` conducts everything, using the reference analyzer at `.claude/skills/video-clone/scripts/analyze.py`, the Markdown style cards in `.claude/skills/video-clone/styles`, and the production engine skills under `.claude/skills/`. All agents write their outputs into `projects/<id>/` folders, and the server watches those exact files to know when a step is actually done.

## Why You Need This

- **Style cloning without model training.** You get the pacing, transitions, framing and comedic timing of a video you love, rebuilt as original 2D animation. Nothing is trained, nothing is fine-tuned; the "model" here is a disciplined breakdown plus a review culture.
- **Your machine, your account, your files.** All generation runs through your existing Claude Code or Codex login on your own computer. There is no cloud service holding your projects; every video lives in a plain folder you can copy or archive.
- **Reviews that actually catch defects.** Characters are reviewed before shots are built, and every shot is reviewed the moment it is finished, so problems are caught where they are born instead of piling up in the final cut.
- **Evidence-based fixes.** Every claimed fix must attach before-and-after crops of the same moment and region. Reviewers check that evidence first, which kills the classic agent failure mode of confidently declaring something fixed.
- **Extensible by Markdown, not code.** A new visual style is one Markdown file in the style registry; a new drawing engine is one skill folder. Adding a seventh engine took the project no code changes to the server at all.

## How It Works

ReelMimic is three layers that share nothing but the filesystem. The React front end lives in `app/web`, the Node server in `app/server`, and the agent workspace is the repository root itself, where Claude Code or Codex read skills from `.claude/skills/` and write files under `projects/`. The shared TypeScript types that keep the browser and server in agreement live in `app/shared/types.ts`. The next diagram lays out those layers, the per-engine skills, and the Python tooling in the video-clone skill.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/reelmimic/edenfunf-reelmimic-architecture.svg" alt="Detailed architecture of the ReelMimic codebase" style="min-width:720px;width:100%;">
</div>

*Detailed architecture of the ReelMimic codebase, from the React components and server modules down to the engine skills and Python tools.*

### Understanding the Architecture

**The web layer.** `app/web/src/App.tsx` is the entry point: a composer card where you drop a reference video or paste a link, pick Claude Code or Codex, and describe what you want; plus the project list and language menu. Once a project exists, `app/web/src/Project.tsx` gives you four tabs (Final, Production line, Plan, Reference analysis), pause and error cards, required-input prompts, and time-stamped notes you can drop at any second of the rendered video. `app/web/src/Chat.tsx` merges the job's chat history with the live event log: working agents appear as cards showing plain-language steps, timers, and thumbnails of frames they examined, while finished replies collapse into a compact summary. All traffic flows through the thin client in `app/web/src/api.ts`, and `app/web/src/i18n.ts` carries the trilingual interface (Traditional Chinese as the source, English, and Simplified Chinese converted with OpenCC).

**The Node server.** `app/server/index.ts` exposes the REST API, handles multipart uploads, streams Server-Sent Events at `GET /api/projects/:id/events`, and serves files inside a project through `GET /files/:id/*`. At startup `app/server/env.ts` loads `~/.reelmimic/secrets.json`, the out-of-repo home for optional keys and tool paths. The heart of the system is `app/server/jobs.ts`, which owns the stage machine, the production scheduling, and the global agent budget. A nice touch lives at the top of that file: `prompts.ts` is re-imported whenever its modification time changes, so you can edit an agent's instructions mid-project and the next turn picks them up without restarting anything.

**Agent adapters.** `app/server/agents/index.ts` turns two very different CLIs into one event stream of `session`, `text`, `thinking`, `tool`, `done` and `error` events. Claude Code is invoked with `claude -p --output-format stream-json --permission-mode acceptEdits`, Codex with `codex exec --json` and a sandbox configuration; prompts always go through stdin so nothing user-typed lands on a command line. Session discipline matters: the director keeps one conversation for the whole video so context survives, builders keep theirs across fix rounds so they remember details, and reviewers always start fresh so nobody grades their own homework.

**The workflow and the file contract.** A project walks through `new, analyzing, styling, planning, plan_review, producing, critiquing, done`, with `needs_input` pausing for things only you can provide (like lyrics) instead of looping. During `analyzing`, the Python script `.claude/skills/video-clone/scripts/analyze.py` downloads the reference with yt-dlp when needed and measures shot boundaries, average shot length, BPM, palette, per-shot camera moves, and writes contact sheets plus `analysis/report.json`. The styling stage's director agent writes `analysis/STYLE.md` and `analysis/route.json`, which maps the detected style to one of the seven engines. Planning is parallel: the director writes the plan core while one agent per character and one for assets work simultaneously, then everything merges into `plan.json` and `STORYBOARD.md`, with each shot mapped to a reference shot and camera specs an engine can execute. The full naming rules for every file agents must write live in `.claude/skills/video-clone/CONTRACT.md`. Here is the key trick: when the server dispatches a step it records the modification times of the files that step must produce, and after the agent turn it checks whether those files actually changed. If not, the job stops in `error` and the step can be retried. If the server restarts mid-job, `recoverOrphans` flags the interrupted stage honestly, and `autoResume` picks it back up staggered so restarted jobs do not all launch agents at once.

**The production line.** After you approve the plan, `app/server/jobs.ts` runs a three-stage line. First the director sets up shared assets, per-character definitions, and segment assignments in `build/production.json`. While the cast gate runs (a fresh reviewer and a fixer agent loop per character, up to three rounds, with shared-rig problems escalated to the director), the shot line starts: up to six builders, each holding one to four shots, work in parallel. The moment a builder finishes a shot and writes its `<shot>.done.json`, a fresh reviewer is dispatched for it; failing shots go back to the same builder, and only the failed ones get re-reviewed. Review quality is protected by two budgets you can tune: a global `MAX_AGENTS` cap across all projects, and a per-project review concurrency limit, because every review opens Chrome to grab frames and too many at once caused screenshot timeouts. Defects are triaged by severity: only blockers visible at normal viewing send work back, while polish items are recorded and handled along the way. After assembly by the director, a fresh final critic checks only what spans segments (seams, continuity, pacing, caption consistency), verifies that earlier fixes actually landed, and loops with the director up to two rounds.

**The engines and tools.** The seven 2D engines are skill folders under `.claude/skills/`: HyperFrames for vector and motion graphics, painted-animation for hand-painted watercolor, crayon-storybook, pixel-art, paper-cutout, whiteboard, and anime-cel. When you approve a plan, the chosen engine's skill is copied into the project as an engine snapshot, so improving a skill never mutates a job that is already running. Around the engines sit the Python tools in `.claude/skills/video-clone/scripts/`: `clip_strip.py` pulls 12 fps frame strips so agents can verify peak moments against the reference, `compare.py` produces the shot-by-shot side-by-side comparisons used in delivery review, `hf_frames.py` grabs frames for HyperFrames projects with PIL crops, a content-hash cache and a machine-wide concurrency limit, `fetch_assets.py` searches Openverse, Pixabay and Freesound and logs every asset's source, author and license into `assets/ASSETS.md`, `align_lyrics.py` times user-provided lyric text against the audio using faster-whisper purely as a measuring ruler, and `yating_tts.py` and `timeline.py` cover Mandarin narration voices and per-stage production timing reports.

Follow one video end to end and the layers click together: the browser posts your brief to `app/server/index.ts`, which starts the stage machine in `app/server/jobs.ts`; each stage spawns a turn through `app/server/agents/index.ts` with instructions from `app/server/prompts.ts`; the spawned agent reads `.claude/skills/video-clone/SKILL.md`, measures the reference with the Python tools, scores the style registry, dispatches an engine skill, and writes every artifact into `projects/<id>/` following the contract; and the server, watching those same files, advances the job and pushes progress to your browser over Server-Sent Events until `out/video.mp4` exists.

## Advantages

- **Deterministic completion checks.** File modification times, not prose parsing, decide whether an agent finished. It is simple, language-agnostic, and impossible for an agent to talk its way past.
- **Review where defects are born.** Cast before shots, shots the moment they land, film-level checks last. Defects never get built upon, which is why the fix loop stays short.
- **Fresh reviewers by construction.** Reviewers have no stake in the work they grade, and builders keep their own memory for fixes. It is a clean separation of authoring and judgment that most agent systems lack.
- **Honest failure handling.** Missing user input pauses the job with a clear ask, interrupted jobs are flagged on restart and resumed automatically, and each failed step is individually retryable.
- **Budgeted parallelism.** Builder count, review concurrency, and the global agent cap are all explicit configuration, tuned from real runs rather than guessed.
- **Swap-proof layers.** The file contract means you can replace the agent CLI, add an engine, or rewrite the front end independently; the other layers never notice.

## Benefits

- **A repeatable style you own.** Once a reference is broken down, the same technique is available for any subject you brief, with original characters and assets, licensed and logged.
- **No per-video API spend.** The work runs against your existing Claude Code or Codex plan, so experimenting with formats costs usage you already pay for.
- **Full observability.** Every agent thought, command and examined frame is streamed to the UI and archived in the project's event log, so you can audit exactly how your video was made.
- **License-safe by design.** Online assets carry their source, author and license in `assets/ASSETS.md`, unclear licenses are flagged, and lyrics come only from text you supply.
- **Human in the loop at the right moments.** You approve a storyboard with style frames before any production starts, and you review the finished film with second-precision comments instead of rerolling blind.
- **A teachable codebase.** Between `docs/ARCHITECTURE.md`, `docs/EXTENDING.md` and the commented server code, the project doubles as a course in practical multi-agent scheduling.

## Usage

Requirements are Node.js 22.18 or newer, Python 3.10 or newer, FFmpeg, Chrome, and a logged-in Claude Code or Codex CLI. Then:

```bash
git clone https://github.com/edenfunf/reelmimic.git
cd reelmimic
./install.sh        # on Windows, double-click install.bat
./start.sh          # on Windows, double-click start.bat
```

Open `http://localhost:4318`, drop in a reference video or paste a link, describe what you want, and pick your agent. If anything is missing from your setup, the installer tells you, and you can re-check at any time with `cd app && npm run doctor`. Optional keys and tool paths (such as `YATING_KEY`, `PIXABAY_KEY`, `FREESOUND_KEY`, `FFMPEG_DIR`, `CHROME_PATH`, `CODEX_BIN` and `PYTHON`) go in `~/.reelmimic/secrets.json`, following `secrets.example.json`; that file lives outside the repo so it can never be committed.

Behavior is tunable with environment variables read in `app/server/jobs.ts`: `BUILDERS` (parallel builders per project, default 6), `CAST_ROUNDS` and `CHUNK_ROUNDS` (review and fix rounds before the job pauses for you, default 3), `FINAL_ROUNDS` (automatic revise rounds for the final panel, default 2), `MAX_AGENTS` (the cap across all projects, default 12), `MAX_REVIEWS` (simultaneous shot reviews per project, default 4), `OVERLAP_CAST` (build shots during the cast gate; off by default because it cost extra fix rounds in real runs), and `PORT`.

When a project passes final review and you are happy with it, you can publish it to your own social accounts from the `app/` folder with the optional Upload-Post integration:

```bash
npm run post-video -- <project id> --platforms tiktok,instagram,youtube   # dry run
npm run post-video -- <project id> --platforms tiktok --at 2026-10-05T18:00 --tz Europe/Madrid --send
```

Without `--send` nothing leaves your computer, only projects whose final review passed are accepted, and rerunning the same video will not post it twice.

## Conclusion

ReelMimic is a rare thing: a creative AI tool whose most valuable code is not the generation prompt but the factory around it. By making files the only interface between the browser, the server and the agents, the project gets completion detection, crash recovery, hot-reloadable instructions and swappable engines almost for free. By splitting production into builder and reviewer roles with severity triage and mandatory fix evidence, it gets quality control that survives contact with long jobs. And by keeping everything local under your own agent account, it stays a tool you own rather than a subscription you rent. If you want to clone the style of a video you love, or you want to see what a well-run multi-agent production line looks like in about a screenful of TypeScript, clone it and take the tour.

Links:

- Repository: [github.com/edenfunf/reelmimic](https://github.com/edenfunf/reelmimic)
- Architecture notes: [docs/ARCHITECTURE.md](https://github.com/edenfunf/reelmimic/blob/main/docs/ARCHITECTURE.md)
- Extending guide: [docs/EXTENDING.md](https://github.com/edenfunf/reelmimic/blob/main/docs/EXTENDING.md)
- Contributing: [CONTRIBUTING.md](https://github.com/edenfunf/reelmimic/blob/main/CONTRIBUTING.md)
