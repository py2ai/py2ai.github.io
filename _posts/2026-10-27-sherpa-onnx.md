---
layout: post
title: "sherpa-onnx: Streaming Speech Recognition, TTS and VAD on onnxruntime"
description: "A source tour of k2-fsa/sherpa-onnx: the C++ decoding stack for streaming and offline ASR, text-to-speech and voice activity detection, plus the Python bindings."
date: 2026-10-27
header-img: "img/post-bg.jpg"
permalink: /sherpa-onnx/
featured-img: "ai-coding-frameworks/ai-coding-frameworks"
image: https://pyshine.com/assets/img/diagrams/sherpa-onnx/k2-fsa-sherpa-onnx-overview-architecture.svg
tags: [LLM, Python, Open Source]
categories: [AI Inference]
keywords: sherpa-onnx, onnxruntime, speech recognition, streaming ASR, text to speech, voice activity detection, zipformer, whisper, python bindings, C++
author: "PyShine"
---

Most speech toolkits make you choose: a research framework that is flexible but heavy, or an embedded library that is fast but frozen. sherpa-onnx, from the k2-fsa team, is a deliberate third option. It is a C++ library — no Python interpreter required at runtime — that loads ONNX models through onnxruntime and ships the full speech stack: streaming (online) recognition, offline recognition, text-to-speech, voice activity detection, speaker identification, keyword spotting and even speaker diarization. The same binary serves the command line, six or more language bindings, WebSocket servers, and mobile platforms from Android and iOS to HarmonyOS.

We walked through the repository on the `master` branch with one question in mind: how does a single C++ core manage to back so many model families — Zipformer transducers, Paraformer, SenseVoice, Whisper, Moonshine, FireRedASR, Qwen3-ASR, Kokoro, Matcha, VITS — without turning into soup? The answer is a two-layer facade pattern. Every capability has a thin public class (`OnlineRecognizer`, `OfflineRecognizer`, `OfflineTts`) that owns configuration and validation, plus an `...Impl` subclass hierarchy chosen at construction time by peeking at the models you passed in. Everything below that layer is plain onnxruntime session management.

This post tours that structure from the outside in: the facades, the streaming decode loop, the session/provider plumbing, and the Python layer that most users actually touch. All file paths and line numbers below were verified against the current `master` tree. If you have followed our earlier tours of [onnxruntime](/onnxruntime-2/) and the ONNX [format](/onnx/) itself, this is the repo that shows how much mileage a disciplined consumer of that runtime can get.

<div style="overflow-x:auto;"><img src="/assets/img/diagrams/sherpa-onnx/k2-fsa-sherpa-onnx-overview-architecture.svg" alt="Overview architecture of sherpa-onnx: facades, streaming and offline recognition, VAD and the onnxruntime session layer" style="min-width:900px;"></div>

<p><em>The overview graph: public facades dispatch to implementation classes, which own onnxruntime sessions; VAD sits beside recognition as an independent capability.</em></p>

## Reading the overview from left to right:

- Everything starts at construction. [`OnlineRecognizer::OnlineRecognizer, L222`](https://github.com/k2-fsa/sherpa-onnx/blob/master/sherpa-onnx/csrc/online-recognizer.cc#L222) immediately calls [`OnlineRecognizerImpl::Create, L61`](https://github.com/k2-fsa/sherpa-onnx/blob/master/sherpa-onnx/csrc/online-recognizer-impl.cc#L61), the factory that inspects your config and returns a transducer, paraformer, or CTC implementation. It even probes the decoder ONNX to count outputs ([`IsNeMoParakeetUnifiedStreaming, L53`](https://github.com/k2-fsa/sherpa-onnx/blob/master/sherpa-onnx/csrc/online-recognizer-impl.cc#L53)) before deciding which impl to instantiate.
- Audio enters through streams. [`OnlineRecognizer::CreateStream, L232`](https://github.com/k2-fsa/sherpa-onnx/blob/master/sherpa-onnx/csrc/online-recognizer.cc#L232) hands you an `OnlineStream`; feeding it is [`OnlineStream::AcceptWaveform, L237`](https://github.com/k2-fsa/sherpa-onnx/blob/master/sherpa-onnx/csrc/online-stream.cc#L237), which pushes samples into the feature extractor ([`FeatureExtractor::AcceptWaveform, L372`](https://github.com/k2-fsa/sherpa-onnx/blob/master/sherpa-onnx/csrc/features.cc#L372)) where fbank frames are computed.
- The decode loop is polled, not threaded. [`OnlineRecognizer::IsReady, L241`](https://github.com/k2-fsa/sherpa-onnx/blob/master/sherpa-onnx/csrc/online-recognizer.cc#L241) tells you when enough frames have accumulated, then [`OnlineRecognizer::DecodeStreams, L251`](https://github.com/k2-fsa/sherpa-onnx/blob/master/sherpa-onnx/csrc/online-recognizer.cc#L251) batches multiple streams into one encoder call — the key trick for serving many concurrent users cheaply.
- Inside the transducer impl, [`OnlineRecognizerTransducerImpl::DecodeStreams, L288`](https://github.com/k2-fsa/sherpa-onnx/blob/master/sherpa-onnx/csrc/online-recognizer-transducer-impl.h#L288) runs the encoder through [`OnlineTransducerModel::Create, L144`](https://github.com/k2-fsa/sherpa-onnx/blob/master/sherpa-onnx/csrc/online-transducer-model.cc#L144)-produced backends, whose interface ([`RunEncoder, L79`](https://github.com/k2-fsa/sherpa-onnx/blob/master/sherpa-onnx/csrc/online-transducer-model.h#L79), `RunDecoder`, `RunJoiner`) isolates every model architecture behind three calls.
- Endpoints are pure arithmetic. [`Endpoint::IsEndpoint, L76`](https://github.com/k2-fsa/sherpa-onnx/blob/master/sherpa-onnx/csrc/endpoint.cc#L76) evaluates three configurable rules over trailing-silence and utterance length; the impl tracks trailing blanks during decoding and converts them ([`OnlineRecognizerTransducerImpl::IsEndpoint, L375`](https://github.com/k2-fsa/sherpa-onnx/blob/master/sherpa-onnx/csrc/online-recognizer-transducer-impl.h#L375)) into seconds. When a rule fires, `Reset` rewinds the encoder state and the next utterance starts clean.
- Every session goes through one choke point. [`GetSessionOptionsImpl, L209`](https://github.com/k2-fsa/sherpa-onnx/blob/master/sherpa-onnx/csrc/session.cc#L209) parses the provider string (with an optional `key=value` config file), sets thread counts, and appends the right execution provider — CPU, CUDA, TensorRT, CoreML, NNAPI, DirectML, OpenVINO, XNNPACK, SpacemiT — with graceful fallback to CPU.
- The batch side mirrors the streaming side. [`OfflineRecognizer::DecodeStreams, L173`](https://github.com/k2-fsa/sherpa-onnx/blob/master/sherpa-onnx/csrc/offline-recognizer.cc#L173) decodes complete utterances via [`OfflineRecognizerImpl::Create, L81`](https://github.com/k2-fsa/sherpa-onnx/blob/master/sherpa-onnx/csrc/offline-recognizer-impl.cc#L81), which dispatches to a dozen model families; [`OfflineTts::Generate, L271`](https://github.com/k2-fsa/sherpa-onnx/blob/master/sherpa-onnx/csrc/offline-tts.cc#L271) does the same for synthesis; and [`VadModel::Create, L28`](https://github.com/k2-fsa/sherpa-onnx/blob/master/sherpa-onnx/csrc/vad-model.cc#L28) selects Silero or ten-vad for segmentation.

## The decoding stack up close

The detail graph below expands each facade into its implementation subtree. Three things stand out on a careful read.

First, the model zoo is handled by aggressive subclassing, not by configuration flags. `OnlineTransducerModel::Create` alone picks between zipformer, zipformer2, LSTM, ebranchformer, conformer, and NeMo transducer backends; separate impl files handle CTC and paraformer streaming. Each backend only implements state stacking plus the encoder/decoder/joiner trio, so a new architecture is genuinely a leaf addition — nothing upstream changes.

Second, beam search is a small, inspectable algebra. A [`Hypothesis, L25`](https://github.com/k2-fsa/sherpa-onnx/blob/master/sherpa-onnx/csrc/hypothesis.h#L25) is just a token vector with timestamps, per-token acoustic and LM scores, and a context state pointer; its `Key` method serializes the token sequence so hypotheses with identical prefixes collapse in the beam pool. Modified beam search layers on [`ContextGraph`](https://github.com/k2-fsa/sherpa-onnx/blob/master/sherpa-onnx/csrc/context-graph.cc) for hotword biasing and an RNN LM for shallow fusion, while greedy search remains the zero-dependency default — a sensible split given that most embedded deployments never enable the beam at all.

Third, the audio plumbing is humble and therefore portable. There is no dependency on ffmpeg or libsndfile for the common path: [`ReadWave, L346`](https://github.com/k2-fsa/sherpa-onnx/blob/master/sherpa-onnx/csrc/wave-reader.cc#L346) parses RIFF/WAVE headers by hand and returns floats normalized to [-1, 1), handling int8, int16, int32, float32 and multi-channel downmix. The VAD subsystem keeps recent audio in a hand-rolled [`CircularBuffer`](https://github.com/k2-fsa/sherpa-onnx/blob/master/sherpa-onnx/csrc/circular-buffer.cc) and windows it through [`SileroVadModel::IsSpeech, L492`](https://github.com/k2-fsa/sherpa-onnx/blob/master/sherpa-onnx/csrc/silero-vad-model.cc#L492), so segmentation works identically from a microphone, a file, or an ALSA device.

<div style="overflow-x:auto;"><img src="/assets/img/diagrams/sherpa-onnx/k2-fsa-sherpa-onnx-architecture.svg" alt="Detailed architecture of sherpa-onnx: streaming transducer decoding, hypothesis scoring, offline ASR families, TTS backends, VAD and session management" style="min-width:900px;"></div>

<p><em>The detail graph: 34 files across the Python layer, streaming core, decoding and scoring, offline ASR, TTS, VAD and the onnxruntime session layer.</em></p>

## From Install to Your First Model

The Python package is the quickest way in: `pip install sherpa-onnx` drops a prebuilt wheel containing the compiled core and the pure-Python wrappers you can read at `sherpa-onnx/python/sherpa_onnx/`. The wrapper classes are thin — [`class OnlineRecognizer, L34`](https://github.com/k2-fsa/sherpa-onnx/blob/master/sherpa-onnx/python/sherpa_onnx/online_recognizer.py#L34) imports the pybind11 extension and re-exposes it with friendly factory methods.

Streaming recognition is three factory keywords and a loop:

```python
import sherpa_onnx

recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
    tokens="tokens.txt",
    encoder="encoder-epoch-99-avg-1.int8.onnx",
    decoder="decoder-epoch-99-avg-1.onnx",
    joiner="joiner-epoch-99-avg-1.int8.onnx",
    num_threads=2,
    decoding_method="greedy_search",
    enable_endpoint_detection=True,
)

stream = recognizer.create_stream()
stream.accept_waveform(16000, samples)   # float32, [-1, 1)
while recognizer.is_ready(stream):
    recognizer.decode_stream(stream)
if recognizer.is_endpoint(stream):
    print(recognizer.get_result(stream).text)
    recognizer.reset(stream)
```

[`from_transducer, L98`](https://github.com/k2-fsa/sherpa-onnx/blob/master/sherpa-onnx/python/sherpa_onnx/online_recognizer.py#L98) exposes the full C++ option surface — endpoint rules, hotwords, blank penalty, provider selection — as keyword arguments with sane defaults. The offline side is even broader: [`OfflineRecognizer`](https://github.com/k2-fsa/sherpa-onnx/blob/master/sherpa-onnx/python/sherpa_onnx/offline_recognizer.py#L42) has factories like [`from_sense_voice, L267`](https://github.com/k2-fsa/sherpa-onnx/blob/master/sherpa-onnx/python/sherpa_onnx/offline_recognizer.py#L267) and [`from_whisper, L1243`](https://github.com/k2-fsa/sherpa-onnx/blob/master/sherpa-onnx/python/sherpa_onnx/offline_recognizer.py#L1243), each doing the model-name-to-config translation for you. The `python-api-examples/` directory then contains roughly a hundred runnable scripts covering files, microphones, WebSocket servers, VAD+ASR chains and subtitle generation, so there is almost always a template close to your use case.

## Model Loading and Session Setup

Every ONNX file in the repo is opened through one helper family in `session.cc`. [`GetSessionOptions, L603`](https://github.com/k2-fsa/sherpa-onnx/blob/master/sherpa-onnx/csrc/session.cc#L603) takes a model config and returns tuned `SessionOptions`; the provider string syntax `provider[:config_file]` lets you pass provider options without code changes — `tensorrt:trt_config.config` or `cuda:gpu.conf` — and keys prefixed with `SessionConfig.` are forwarded verbatim to onnxruntime's own config entries.

Two subtle decisions deserve attention. Transducer models are three separate graphs, and the code deliberately pins the decoder and joiner to CPU even when the provider is TensorRT or OpenVINO, because those tiny graphs are latency-sensitive and gain nothing from EP round-trips. And the TensorRT case intentionally falls through to the CUDA case, so an unavailable TensorRT EP degrades to GPU execution rather than CPU. This kind of provider plumbing is exactly the layer our [onnxruntime tour](/onnxruntime-2/) covered from the runtime's side; sherpa-onnx is a study in consuming it defensively.

Configuration validation is similarly centralized: `OnlineRecognizerConfig::Validate` checks decoding-method/hotword combinations, verifies that every rule FST for inverse text normalization exists, and only then delegates to model config validation. Fail fast at construction, never mid-stream.

## VAD, TTS and the Rest of the Stack

Voice activity detection is a self-contained capability that composes with, rather than depends on, recognition. [`VoiceActivityDetector::Impl::AcceptWaveform, L50`](https://github.com/k2-fsa/sherpa-onnx/blob/master/sherpa-onnx/csrc/voice-activity-detector.cc#L50) slides a window over incoming audio, queries the Silero or ten-vad model per window, and assembles detected speech into `SpeechSegment`s bounded by configurable min-silence and threshold values. The classic pattern — VAD chunks feeding an offline recognizer — is what powers the repo's `vad-with-non-streaming-asr` examples, and it is the same arrangement used by our previously covered [pyannote-audio](/pyannote-audio/), though here the diarization pieces live in the same binary.

Text-to-speech follows the facade pattern with its own zoo: [`OfflineTtsImpl::Create, L41`](https://github.com/k2-fsa/sherpa-onnx/blob/master/sherpa-onnx/csrc/offline-tts-impl.cc#L41) dispatches to VITS, Matcha, ZipVoice, Kokoro, Kitten, Pocket and Supertonic backends, each pairing a text frontend (lexicon, character, or espeak-ng phonemes) with an acoustic model and often a separate vocoder graph. A nice touch is [`GeneratedAudio::ScaleSilence, L40`](https://github.com/k2-fsa/sherpa-onnx/blob/master/sherpa-onnx/csrc/offline-tts.cc#L40), which shortens or stretches detected pauses in the synthesized waveform — with explicit guards against NaN and overflow — because long silences are the most common complaint about VITS-style models. If your interest is specifically neural TTS serving, [piper](/piper/) is the closely related sibling project that sherpa-onnx can load VITS models from.

## Honest Limits

The breadth has a cost. The `csrc` directory holds several hundred files, and many are near-duplicates differing only in model family (`offline-*-model.cc` for a dozen architectures). If you add a model type today, you will touch the factory in `online-recognizer-impl.cc` or `offline-recognizer-impl.cc`, a config struct, a Validate method, and the Python wrapper — five coordinated edits. The facade/impl split keeps each file simple, but the total surface is large, and the C API that all other bindings consume multiplies review effort accordingly.

Second, the docs-per-model situation is uneven. Pretrained model links are centralized on the project's documentation site rather than in the repo, so picking a model means leaving the code; the `python-api-examples` scripts hardcode their expected filenames, which is fine for templates but means the example and your download rarely match verbatim. Third, endpoint detection, hotword biasing and inverse-text-normalization FSTs are streaming-only or transducer-only in places — the config validation will tell you, but only after you try.

Finally, this is a C++ project with C++ consequences: memory ownership of streams and results crosses the FFI boundary, and the pybind11 wrappers are deliberately conservative, copying small results rather than exposing lifetime hazards. None of this is criticism so much as the tax on the design goal — one static library, zero runtime dependencies beyond onnxruntime, running the same code on a server GPU and a watch. Few speech projects hold that line this well, and the source is unusually rewarding to read once the facade pattern clicks. For a different point on the same trade-off curve — Python-native but heavier — see our [transformers](/transformers/) tour, or [whisperx](/whisperx/) for forced-alignment work built on the Whisper family that sherpa-onnx also exports.

<script type="application/ld+json">{ "@context": "https://schema.org", "@type": "TechArticle", "headline": "sherpa-onnx: Streaming Speech Recognition, TTS and VAD on onnxruntime", "description": "A source tour of the k2-fsa/sherpa-onnx C++ core: streaming and offline ASR facades, the transducer decode loop, session and provider setup, VAD, TTS and the Python bindings.", "author": { "@type": "Person", "name": "PyShine" }, "publisher": { "@type": "Organization", "name": "PyShine" }, "datePublished": "2026-10-27", "keywords": "sherpa-onnx, onnxruntime, speech recognition, streaming ASR, TTS, VAD", "url": "https://pyshine.com/sherpa-onnx/" }</script>
