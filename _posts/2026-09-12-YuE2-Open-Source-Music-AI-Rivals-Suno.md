---
layout: post
title: "YuE2: The Open Source Music AI That Rivals Suno v5"
description: "YuE2 from Multimodal Art Projection (m-a-p) is an open source music generation model that turns lyrics and a style prompt into complete songs with vocals and accompaniment. It uses an AR-NAR Mixture-of-Transformers backbone: an autoregressive transformer writes an editable melody-and-chord score in ABC notation and semantic music tokens, then a non-autoregressive flow matching model generates acoustic latents that a VAE decodes into 48 kHz stereo audio. On WildSongBench (192 prompts), YuE2 best-of-8 achieves 6.9632 SongBench Avg, the highest mean among all 17 evaluated systems including Suno v5/v6, Mureka 9, MiniMax Music, and ACE-Step. Three modes: CREATE (lyrics+style to song), COVER (zero-shot style transfer from transcribed melody), and EDIT (agentic conversation-driven score revision). Staged Python API: plan() to generate_semantic() to synthesize() to decode(). 3B parameters, Apache 2.0 code, CC BY-NC 4.0 weights, 24GB VRAM, runs locally on RTX 4090 (215s song in 71s). MERT2 is SOTA on 14/15 MARBLE metrics. SheetSage2 is SOTA on 10/13 transcription metrics. Includes agent skill for Claude Code, Cursor, and SKILL.md-compatible agents."
date: 2026-09-12
header-img: "img/post-bg.jpg"
permalink: /YuE2-Open-Source-Music-AI-Rivals-Suno/
featured-img: ai-coding-frameworks/ai-coding-frameworks
tags:
  - YuE2
  - Music Generation
  - Open Source
  - AI Music
  - AR-NAR
  - Flow Matching
  - Symbolic Planning
  - m-a-p
author: PyShine
---

Imagine typing a few lines of lyrics and a style prompt, and getting back a complete song with vocals, accompaniment, and studio-quality audio. Now imagine doing that for free, on your own GPU, with a model whose internal composition plan you can read, edit, and re-render.

That is YuE2.

## What is YuE2

[YuE2](https://github.com/multimodal-art-projection/YuE) is an open source music generation model from [Multimodal Art Projection (m-a-p)](https://huggingface.co/m-a-p), a research community spanning HKUST, Tokenwave.AI, NYU, Stanford, MBZUAI, NOIZ, and ACE Studio. Released on September 9, 2026, it turns lyrics and a style prompt into complete songs with vocals and accompaniment at 48 kHz stereo quality.

The headline: **YuE2 best-of-8 achieves 6.9632 SongBench Avg on WildSongBench, the highest mean among all 17 evaluated systems**, including Suno v5 (6.8721), Mureka 9 (6.9377), and Suno v6 (6.5562). This is an open-weights model matching or beating proprietary systems.

The tagline says it best: **Compose in symbols. Create in sound.**

Three things make YuE2 different from every other music AI you have tried:

1. **White-box composition**: YuE2 does not jump straight from text to audio. It first writes a melody-and-chord plan in ABC notation that you can read, play, and edit before rendering. This is the "score" layer.
2. **Zero-shot covers**: Transcribe any recording into a melody score with SheetSage2, then re-render it in a completely different style. Mandarin pop becomes English jazz, no fine-tuning needed.
3. **Agentic editing**: Talk to an agent about the composition. Ask it to change the harmony, add a saxophone solo, or swap the tempo. The agent revises the ABC score, checks musical invariants, and re-renders the song.

## How It Works: One Backbone, Two Modes

YuE2 uses a single AR-NAR Mixture-of-Transformers backbone that works in two stages.

![YuE2 AR-NAR architecture](/assets/img/diagrams/yue2/yue2-architecture.svg)

### Stage 1: Autoregressive (AR) Transformer

The AR transformer takes your lyrics and style prompt and predicts two things autoregressively:

1. **Symbolic score** in ABC notation: a human-readable plan containing melody and chord symbols. This is the white-box layer you can inspect and edit.
2. **Semantic music tokens**: discrete token sequences that capture the musical semantics, separated into vocal and accompaniment tracks using track-decoupled next-token prediction. This technique, inherited from YuE v1, overcomes the challenge of dense mixture signals in music.

### Stage 2: Non-Autoregressive (NAR) Flow Matching

The NAR transformer takes the semantic tokens and generates acoustic latents using flow matching, a continuous generation method that avoids the slowness of token-by-token autoregression. These latents are then decoded by a VAE into 48 kHz stereo audio.

The staged Python API makes this explicit: `plan()` then `generate_semantic()` then `synthesize()` then `decode()`. Each stage produces an artifact you can inspect, reuse, or replace.

## Three Modes, One Model

YuE2 supports three workflows from the same model checkpoint. The difference is where the score comes from.

![YuE2 three modes](/assets/img/diagrams/yue2/yue2-three-modes.svg)

### Mode 1: Create

Give YuE2 lyrics and a style prompt. It writes a melody-and-chord plan, then renders it as a complete song. This is the default mode (`cot="full"`).

```python
from yue2 import YuE2Pipeline

with YuE2Pipeline.from_pretrained("m-a-p/YuE2-3B", device="cuda") as pipe:
    song = pipe(
        style="English, piano-pop, warm vocal, acoustic guitar",
        lyrics="Verse 1...\nChorus...",
        cot="full",  # default: generate editable plan
    )
    song.save_artifacts("outputs/my-song")
```

The output directory contains not just `audio.flac` but also the `score.abc` file, semantic tokens, acoustic latents, generation settings, and model identities. Everything is reproducible.

### Mode 2: Cover

Take a source recording, transcribe it into a melody score with SheetSage2, then re-render it in a new style. Use `cot="melody"` with a score that has no chord symbols, so the accompaniment can adapt freely to the new style.

The [agentic demo](https://map-yue2.github.io/#agentic-music-editing) follows "The Last Train" through 9 steps and 14 versions, from Mandarin pop to English jazz with new harmony and a saxophone solo. On 948 cover works, YuE2 reaches 0.647 CLEWS mAP (versus 0.006 without a score), using the general generator without cover-specific fine-tuning.

### Mode 3: Edit (Agentic)

Export the plan, revise the musical details, and re-render. You can ask an agent to change harmony, melody, tempo, or form. The agent revises the ABC score, checks musical invariants, and generates a new recording.

> Use the yue2-music skill to create an English piano-pop song. Keep the original audio and score. Make a second version with jazz harmony, preserve the vocal melody and lyric order, and give me both versions to compare.

Editing generates a new complete recording. It does not preserve the original waveform outside an edit. The editable score is the white-box interface: you can inspect the intended composition and intervene on it.

## Staged Pipeline and Model Family

YuE2 is not a single model file. It is a family of specialized models, all available on Hugging Face.

![YuE2 pipeline and model family](/assets/img/diagrams/yue2/yue2-pipeline-models.svg)

### The Four-Stage API

| Stage | Function | Input | Output |
|-------|----------|-------|--------|
| 1 | `plan()` | Lyrics + style | Editable ABC score (melody + chords) |
| 2 | `generate_semantic()` | Score + style | Semantic music tokens (AR prediction) |
| 3 | `synthesize()` | Semantic tokens | Acoustic latents (NAR flow matching) |
| 4 | `decode()` | Acoustic latents | 48 kHz stereo audio (VAE decoder) |

Each stage produces a saved artifact. You can reuse an exact plan across multiple renders, swap decoders, or intervene at any stage.

### Model Family

| Model | Purpose | License |
|-------|---------|---------|
| [YuE2-3B](https://huggingface.co/m-a-p/YuE2-3B) | Song generation, symbolic planning, covering, editing | CC BY-NC 4.0 |
| [YuE2-Vae](https://huggingface.co/m-a-p/YuE2-Vae) | Default generation and listening decoder | CC BY-NC 4.0 |
| [YuE2-Vae-legacy](https://huggingface.co/m-a-p/YuE2-Vae-legacy) | Benchmark decoder (reproducible evals) | CC BY-NC 4.0 |
| [SheetSage2](https://huggingface.co/m-a-p/SheetSage2) | Audio-to-score transcription for covers | CC BY-NC 4.0 |
| [MERT-v2-FullSong](https://huggingface.co/m-a-p/MERT-v2-FullSong) | Full-song music representations (SheetSage2's encoder) | CC BY-NC 4.0 |
| [MERT-v2-30s](https://huggingface.co/m-a-p/MERT-v2-30s) | Music representations for short recordings | CC BY-NC 4.0 |
| [WildSongBench](https://huggingface.co/datasets/m-a-p/WildSongBench) | 192 evaluation prompts and benchmark resources | CC BY 4.0 |

### Performance

On an NVIDIA RTX 4090 (24GB VRAM, BF16, no quantization):
- **214.85-second song generated in 71.04 seconds** (faster than real-time)
- **Peak memory: 14.08 GiB**
- **Output: 48 kHz stereo audio**

## WildSongBench: How YuE2 Compares

WildSongBench uses 192 prompts with automatic evaluation across four metrics: SongBench Avg (overall), AudioBox PQ (production quality), MuLan (text-music alignment), and PER (phoneme error rate, lower is better).

![YuE2 benchmark comparison](/assets/img/diagrams/yue2/yue2-benchmark-comparison.svg)

### The Results

| System | SongBench Avg | AudioBox PQ | MuLan | PER |
|--------|--------------|-------------|-------|-----|
| **YuE2 (best-of-8)** | **6.9632** | 8.2714 | 0.5051 | 9.79% |
| Mureka 9 | 6.9377 | 8.0226 | 0.4394 | 11.69% |
| Suno v5 | 6.8721 | 8.1698 | **0.5428** | 8.10% |
| YuE2 | 6.7316 | 8.2598 | 0.5068 | 8.44% |
| Suno v5.5 | 6.7150 | 8.1955 | 0.5089 | 5.96% |
| Suno v4.5 | 6.6995 | 8.2541 | 0.5022 | **5.80%** |
| Suno v6 | 6.5562 | 8.1296 | 0.4916 | 7.58% |
| LeVo 2 | 6.3247 | **8.3966** | 0.3542 | 26.12% |
| MiniMax Music 2.6 | 6.3222 | 8.1711 | 0.4251 | 24.55% |
| ACE-Step 1.5 | 6.0118 | 8.0518 | 0.4372 | 7.46% |
| YuE 1 (previous) | 4.9165 | 7.8683 | 0.2623 | 36.38% |

**Key takeaways:**

- YuE2 best-of-8 has the highest mean score among all 17 evaluated settings.
- YuE2 standard (best-of-2) at 6.7316 is still competitive with Suno v5.5 (6.7150).
- Suno v5 leads on MuLan (text-music alignment) and Suno v4.5 leads on PER (lowest phoneme error).
- Rankings vary by metric; the small gap between the highest means does not establish statistical significance.
- YuE2 jumped from 4.9165 (YuE 1) to 6.7316 (YuE2 standard), a massive generational improvement.

## SOTA Companion Models

YuE2 ships with two companion models that are each state-of-the-art in their own right.

**MERT2** is a music understanding model that is SOTA on **14 of 15 MARBLE metrics**, with **91.72% genre accuracy on GTZAN**. It provides full-song music representations and serves as SheetSage2's encoder. While MERT2 feature extraction is optional for generation, it powers music understanding tasks across the ecosystem.

**SheetSage2** is an audio-to-score transcription model that is SOTA on **10 of 13 benchmark metrics**, with **82.51% vocal melody pitch-class F1 on RWC-Pop**. It runs in a separate environment and loads its MERT2 encoder automatically. SheetSage2 is what makes zero-shot covers possible: it transcribes any recording into an ABC melody score that YuE2 can then re-render in a new style.

## Agent Skill

YuE2 includes a [yue2-music skill](https://github.com/multimodal-art-projection/YuE/blob/main/skills/yue2-music/SKILL.md) that teaches AI agents how to:

- Generate songs from lyrics and style prompts
- Transcribe and cover recordings
- Edit ABC scores
- Check musical invariants
- Organize listening comparisons

The skill works with any agent that supports `SKILL.md` packages, including Claude Code and Cursor. Install the Python runtime separately with `pip install .`, then point your agent at the skill directory.

## Quick Start

**Prerequisites:** Linux, Python 3.12, NVIDIA GPU with BF16 support and 24 GB VRAM.

```bash
git clone https://github.com/multimodal-art-projection/YuE.git
cd YuE
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install .
python examples/generate.py --output outputs/first-song
```

Open `outputs/first-song/audio.flac`. The output directory also contains the score, semantic tokens, acoustic latents, generation settings, and model identities.

### Cot Settings

| Setting | Behavior |
|---------|----------|
| `cot="full"` | Generate an editable melody-and-chord plan (default for new songs) |
| `cot="melody"` | Use a melody plan with free accompaniment (recommended for covers) |
| `cot="off"` | Generate directly from lyrics and style (no plan) |
| `abc=...` | Supply your own score in full or melody mode |

### Cover a Song

Transcribe a source recording with SheetSage2, review its melody ABC, and provide new lyrics or a target style:

```python
from yue2 import YuE2Pipeline

with YuE2Pipeline.from_pretrained("m-a-p/YuE2-3B", device="cuda") as pipe:
    cover = pipe(
        style="English, jazz-funk, warm lead vocal, Rhodes, bass and drums",
        lyrics=open("cover-lyrics.txt").read(),
        abc=open("cover-score/score.abc").read(),
        cot="melody",
        seed=42,
    )
    cover.save_artifacts("outputs/cover")
```

### Edit a Composition

Export a plan, revise the musical details, and render the edited score:

```python
from yue2 import YuE2Pipeline

with YuE2Pipeline.from_pretrained("m-a-p/YuE2-3B", device="cuda") as pipe:
    plan = pipe.plan(
        style="English, piano-pop",
        lyrics=open("lyrics.txt").read(),
    )
    plan.save("outputs/plan")
    # Edit outputs/plan/score.abc, then re-render:
    # python examples/generate.py --request examples/song.json \
    #   --abc-file edited.abc --cot full --output outputs/edited
```

## License

- **Code, agent skill, and documentation**: [Apache 2.0](https://github.com/multimodal-art-projection/YuE/blob/main/LICENSE)
- **Model weights**: [CC BY-NC 4.0](https://github.com/multimodal-art-projection/YuE/blob/main/MODEL_LICENSE) (non-commercial use only)
- **Third-party components**: retain their original licenses

The non-commercial restriction on weights is important: if you want to build a commercial product, you will need to negotiate a separate license. The code itself is permissive Apache 2.0.

## Why YuE2 Matters

Most music AI tools are black boxes. You type a prompt, you get audio, and you have no idea what happened in between. If the melody is wrong, you re-roll. If the chords are off, you re-roll. You are rolling dice.

YuE2 changes this. The symbolic score layer means you can see the plan before the audio exists. You can change a chord, adjust a melody, shift the tempo, and re-render. You can cover a song in a new style without fine-tuning. You can have a conversation with an agent about musical decisions. The model is not just a generator; it is a collaborator with a transparent inner life.

And it does all of this at quality that matches or beats the best proprietary systems, with weights you can download from Hugging Face and run on a single consumer GPU.

That is the open future of music creation.

## Related Posts

- [ACE Step UI: Open Source AI Music Generation](/ACE-Step-UI-Open-Source-AI-Music-Generation/)
- [HyperFrames: Write HTML, Render Video, Built for AI Agents](/HyperFrames-Write-HTML-Render-Video-Built-for-Agents/)
- [Supertonic: On-Device Multilingual TTS](/Supertonic-On-Device-Multilingual-TTS/)
