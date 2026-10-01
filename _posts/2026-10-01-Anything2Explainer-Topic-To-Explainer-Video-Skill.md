---
layout: post
title: "Anything2Explainer: A Skill That Turns Any Topic Into A Coded Explainer Video - Inside Vincentwei1021/anything2explainer"
description: "Anything2Explainer is an agent skill that turns a topic into a narrated motion-graphics explainer video: every frame drawn in code with Remotion, voiceover from local TTS, subtitles and a chapter progress bar included, plus a full multi-agent pipeline with quantified quality checks. We tour the repository to see how the skill, the template, and the QC tooling work together."
date: 2026-10-01
header-img: "img/post-bg.jpg"
permalink: /Anything2Explainer-Topic-To-Explainer-Video-Skill/
featured-img: "ai-coding-frameworks/ai-coding-frameworks"
image: https://pyshine.com/assets/img/diagrams/anything2explainer/vincentwei1021-anything2explainer-architecture.svg
tags:
  - Agent Skills
  - Remotion
  - Video Generation
  - Claude Code
categories: [AI, Open Source]
keywords: "anything2explainer, explainer video skill, Remotion, programmatic video, motion graphics, TTS narration, Claude Code skill, Codex skill, multi-agent workflow, video QC, open source"
author: "PyShine"
---

Explainer videos are the most demanded and most dreaded artifact in technical communication. The visual style is well understood — dark canvas, clean line graphics, a narrator walking through the idea — but producing one means storyboarding, animating, recording, timing subtitles, and repeating all of it every time the content changes. [Anything2Explainer](https://github.com/Vincentwei1021/anything2explainer) attacks that whole pipeline at once, not with a hosted generator, but with something you can read, run, and rewrite: an agent skill for Claude Code and Codex that turns a topic into a finished motion-graphics video where every frame is drawn in code.

The repository is the complete production system behind that claim. Its centerpiece is a compilable Remotion 4 template with a fixed visual language — black canvas with a dot-wave or star-field backdrop, white line graphics with purple accents, bold typography, 44-pixel white-on-outlined subtitles, and a chapter progress bar along the bottom. Around the template sits a full methodology: a research brief that forces every on-screen fact to carry a source URL, a narration writing guide with thirteen principles, a storyboard format with frame-exact tokens, protocols for parallel builder and QC agents, and a set of Python tools that measure whether the finished film actually obeys the motion and composition rules.

The source rewards a close reading because it documents a workflow that is usually trapped in someone's head. The main skill file alone specifies eight production stages, four points where the agent must stop and ask the user, hard rules for flash usage and shot pacing, and quantified acceptance criteria — longest static stretch under three seconds, a settle period of at least thirty frames at the end of every shot. This is craft knowledge converted into something an agent can execute and a machine can check.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/anything2explainer/vincentwei1021-anything2explainer-overview-architecture.svg" alt="Architecture overview of the Vincentwei1021/anything2explainer repository" style="max-width:100%;height:auto;" />
</div>

*High-level architecture overview of the Vincentwei1021/anything2explainer repository, from skill orchestration and standards to the video template and tooling.*

Reading the overview from left to right: the skill's entry file orchestrates everything, scaffolding a fresh project and dispatching a researcher while the narration and storyboard standards shape the script; the TTS builder turns the approved script into audio and a frame-numbered timeline; the Remotion template — visual primitives, light and camera effects, and the overlay chrome of titles, chapter cards, and HUD — hosts the shot groups that builders fill in; and the render script hands its output to motion checks that gate every handoff.

## Why You Need This

If you have ever tried to explain a technical concept to an audience — RAG pipelines, how a protocol works, why a model fails — you know the gap between having the knowledge and having the video. Hiring an motion designer for a five-minute piece is expensive; doing it yourself in After Effects is a month of nights; and the AI video generators that promise a shortcut produce clips you cannot edit fact by fact. Anything2Explainer takes the third path: the video is a codebase, so any sentence, any number, any animation is a diff away.

The second problem it solves is trust. Generated explainers routinely invent numbers, misattribute quotes, and hallucinate terminology. This skill makes factual grounding a hard rule: every digit, English term, year, and name that appears on screen must be traceable to a URL in the research document, and the static selfcheck compares on-screen text literals against the film's fact list. Facts that did not survive research do not get narrated either.

The third problem is consistency at scale. A five-minute film in this pipeline runs to dozens of shots, and the skill's answer is a multi-agent assembly line: the main session writes narration and storyboards, parallel builder agents each construct groups of five to seven shots against written rules, and separate QC agents review per chapter before repair agents fix and re-verify. The rules they follow — one protagonist per shot, light follows the protagonist, flash effects only on the shot's core term, chapter boundaries that connect narratively — are all spelled out in reference documents the agents actually read. You get the same standard on shot forty-four as on shot one.

## How It Works

The repository is an agent skill definition wrapped around a real Remotion project, a set of Python pipeline tools, and the specification documents that govern both agents and aesthetics.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/anything2explainer/vincentwei1021-anything2explainer-architecture.svg" alt="Detailed architecture of the Vincentwei1021/anything2explainer repository" style="max-width:100%;height:auto;" />
</div>

*Detailed architecture of the Vincentwei1021/anything2explainer repository, tracing skill orchestration, standards, the Remotion template, pipeline scripts, and quality control.*

### Understanding the Architecture

**The skill file is the production bible.** SKILL.md defines the agent-facing workflow: stage 0 scaffolds a project from the template with template/scripts/new_project.sh; stage 1 dispatches a researcher following reference/research-brief.md; stage 2 writes narration under reference/narration-guidance.md and runs template/scripts/tts_build.py; stage 3 writes a storyboard source with tokens and renders it via template/scripts/render_storyboard.py. Four confirmation points force the agent to stop and wait for the user — duration and language, final narration, TTS voice, and the first thirty seconds of styled footage — because each is the cheapest place to change direction.

**The visual system is code, not a template grid.** template/src/ui.tsx holds the primitive library and palette, template/src/fx.tsx adds the light effects and camera moves (light sweeps, stage lines, ghost outlines, glows, big numbers, camera operators), and template/src/common/ carries the shared machinery: dot-wave and star-field backgrounds, fog, glitch, subtitles, the progress bar, and the timeline and subtitle data modules that scripts regenerate. template/src/overlay/ provides the title card, chapter cards, the top HUD, the flow track, and the end card, while template/src/shots/ is where the parallel build groups drop their scenes.

**Voiceover is reproducible.** template/scripts/tts_build.py defaults to a language-aware engine choice — the Chinese edge-tts Yunxi voice at natural speed, or the English kokoro am_liam voice — and emits the audio plus src/common/timeline.ts and subs.ts with per-sentence frame numbers. The skill is explicit about what that means: once narration is locked, frame numbers are hardcoded throughout the film, so wording changes are the one intervention that must happen early.

**Quality is measured, not vibes.** template/scripts/motion_check.py reads each shot and reports the longest static stretch and the end-of-shot settle window against the written thresholds; template/scripts/frame_metrics.py measures per-shot object sizes, protagonist glow coverage, stray debris, and stillness; and template/scripts/selfcheck.py statically audits the source against the storyboard for frame coverage holes, flash-count versus the flash whitelist, light-sweep usage versus its whitelist, and on-screen literals versus the fact list. Builders must pass the motion check on their group before handoff, and the finished film is re-measured from final frames.

**The samples are part of the architecture.** examples/rag/ is a complete reference production — research, narration, storyboard, shot sources, QC reports, and frames — used as the quality bar every new film is measured against, and examples/contrast/ pairs bad and good frames for six composition rules so agents can see the failure modes. reference/lessons.md rounds it out with the root causes of past defects, and new experience is written back there after each delivery.

End to end: a topic becomes research with sources, research becomes narration with a chapter structure, narration becomes audio and a frame-locked timeline, the storyboard splits the film into parallelizable shot groups, builders fill template/src/shots/ under the style and motion rules, and the render script produces the film while the QC suite verifies that pacing, composition, and facts all hold.

## Advantages

- **Every frame is editable code.** Because the film is a Remotion project, fixing a wrong number or retiming an animation is a code change with version control, not a re-render of a black box.
- **Facts are auditable by construction.** The source-URL rule and the literal-versus-fact-list selfcheck make hallucinated content a build failure rather than a postmortem.
- **The style system is cohesive.** Fixed palette, typography, backdrops, and chapter chrome mean a film made with this skill looks like one product, not a slideshow of mismatched templates.
- **Parallelism is engineered in.** Shot groups are designed for concurrent builder agents with explicit protocols, which is how the reference production finished in roughly the time of a long meeting.
- **Quality gates are numeric.** Static-time ceilings, settle windows, protagonist sizing, and sweep whitelists turn taste into checkable criteria.
- **The methodology transfers.** Even if you never run the skill, the narration principles, storyboard tokens, and QC metrics are directly reusable in any programmatic video project.

## Benefits

- **A complete, working agent skill.** This is a full production pipeline — scaffolding, research, script, audio, storyboard, parallel builds, render, QC — not a demo of what an agent might do.
- **Deterministic voiceover pipeline.** Language-aware TTS defaults with a documented path to bring your own studio audio by dropping a wav into the assets folder and regenerating the timeline.
- **Human checkpoints where they matter.** The four confirmation points put user judgment at the moments of highest leverage — length, wording, voice, and look — instead of after hours of rendering.
- **Cross-language output.** Chinese and English films come from the same template with documented adjustments for script length, subtitle budgets, and voice speed.
- **Learning material for agent orchestration.** The prompt templates, builder and QC protocols, and the lessons file are a compact education in running multiple coding agents without chaos.
- **Clear licensing position.** Videos you produce belong to you; the toolkit itself is free for noncommercial use under a PolyForm license, with bundled fonts under the SIL Open Font License.

## Usage

The intended entry point is an agent: install the skill into Claude Code or Codex from the repository, then ask for an explainer video on your topic. The skill drives the rest, stopping at its four confirmation points. For a 3 to 5 minute film, expect the reference production's scale — dozens of sentences and shots across three or four chapters.

Under the hood, each film is a fresh project scaffolded from the template:

```bash
template/scripts/new_project.sh <work-directory> <slug>
```

Inside the project, the pipeline scripts drive the stages — the narration-to-audio build, the storyboard render, stills, the preview gate, and the final render:

```bash
python3 scripts/tts_build.py          # audio + timeline + subtitles
python3 scripts/render_storyboard.py  # storyboard source to shot table
scripts/preview.sh 30                 # first 30 seconds for the style gate
npx tsc --noEmit                      # typecheck before rendering
VER=v1 scripts/render.sh              # full film render
```

Quality checks run at every handoff and again on the finished frames:

```bash
python3 scripts/motion_check.py G1            # per build group
python3 scripts/motion_check.py --frames fin_frames   # final film
python3 scripts/frame_metrics.py --out qc/frame_metrics_v1.md
python3 scripts/selfcheck.py
```

Plan disk space too: the skill budgets roughly two gigabytes per film, and the duration table in SKILL.md maps runtime tiers to sentence counts, chapter counts, and build-group sizes.

## Conclusion

Anything2Explainer is a rare artifact: a working definition of taste. It takes the fuzzy craft of motion-graphics storytelling — pacing, emphasis, composition, factual care — and encodes it into template code, written standards, agent protocols, and numeric checks, then proves the system by shipping a reference film inside the same repository. For anyone producing educational video, it is a production line; for anyone building agent skills, it is a masterclass in scoping what an agent should decide and what it must ask. The topic goes in; the explainer comes out; and because everything is code, the next version is only a diff away.

Links:

- GitHub repository: https://github.com/Vincentwei1021/anything2explainer
- Citation file: https://github.com/Vincentwei1021/anything2explainer/blob/main/CITATION.cff
- License: https://github.com/Vincentwei1021/anything2explainer/blob/main/LICENSE
