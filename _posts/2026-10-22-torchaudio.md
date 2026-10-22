---
layout: post
title: "torchaudio Source Tour: TorchCodec I/O, Custom Autograd, and the C++/CUDA Kernels Behind Forced Alignment"
description: "A guided tour of the pytorch/audio source tree: how torchaudio.load hands decoding to TorchCodec, how lfilter dispatches into libtorchaudio CPU/CUDA kernels, and where the maintenance-mode limits are."
date: 2026-10-22
header-img: "img/post-bg.jpg"
permalink: /torchaudio/
featured-img: "ai-coding-frameworks/ai-coding-frameworks"
image: https://pyshine.com/assets/img/diagrams/torchaudio/pytorch-torchaudio-overview-architecture.svg
tags: [LLM, Python, Open Source]
categories: [AI Inference]
keywords: torchaudio, pytorch audio, torchcodec, forced alignment, RNNT loss, CTC, CUDA kernels, lfilter, libtorchaudio, audio decoding, source tour
author: "PyShine"
---

Most people meet torchaudio as a one-liner: `torchaudio.load("clip.wav")` returns a tensor and a sample rate, and that is the end of the story. Under the hood, though, the `pytorch/audio` repository is a fascinating case study in how a PyTorch domain library evolves. It is a smaller sibling to [PyTorch](/pytorch/) itself — same extension philosophy, same tensor-first design — but it has just gone through the most radical simplification of any PyTorch-ecosystem project: the entire decoding stack was ripped out and replaced by a delegation to TorchCodec, while the hand-written CPU and CUDA kernels that actually do the hard math stayed put in a C++ library called `libtorchaudio`.

That combination makes the current main branch unusually readable. There is no more backend-selection maze for sox versus soundfile versus ffmpeg; `load` and `save` now forward to TorchCodec's `AudioDecoder` and `AudioEncoder`. Meanwhile, performance-critical code — Viterbi-style forced alignment, RNN-transducer loss, CUDA IIR filtering — still lives in C++ and CUDA files, registered as custom torch ops through the new stable library ABI. If you enjoyed the image-side twin of this story in [torchvision](/torchvision/), you will recognize the pattern: a thin Python surface, a compact C++ extension, and everything else expressed as composable tensor ops.

This post walks the tree the way an engineer would: the Python entry points first, the extension-loading mechanics second, then the native kernels, and finally the honest trade-offs — because a library in maintenance mode deserves an honest map, not a brochure.

<div style="overflow-x:auto;"><img src="/assets/img/diagrams/torchaudio/pytorch-torchaudio-overview-architecture.svg" alt="torchaudio architecture" style="max-width:100%;"></div>

<p><em>Overview of torchaudio on the main branch: the Python surface delegates I/O to TorchCodec, boots native extensions at import time, and routes specialized math into libtorchaudio's CPU/CUDA kernels.</em></p>

The overview rewards a slow read. Reading the overview from left to right:

- [`load`, L18](https://github.com/pytorch/audio/blob/main/src/torchaudio/__init__.py#L18) — the public entry point. Its docstring now says plainly: as of TorchAudio 2.9 this relies on TorchCodec's decoding under the hood.
- [`load_with_torchcodec`, L9](https://github.com/pytorch/audio/blob/main/src/torchaudio/_torchcodec.py#L9) — the actual implementation, which builds a TorchCodec `AudioDecoder`, fetches metadata, and returns float32 samples.
- [`_load_lib("_torchaudio")`, L30](https://github.com/pytorch/audio/blob/main/src/torchaudio/_extension/__init__.py#L30) — import-time initialization that loads the extension modules before anything else runs.
- [`torch.ops.load_library`, L56](https://github.com/pytorch/audio/blob/main/src/torchaudio/_extension/utils.py#L56) — the glob-based loader that finds `.so` (or `.pyd` on Windows) binaries and registers their ops.
- [`MelSpectrogram`, L515](https://github.com/pytorch/audio/blob/main/src/torchaudio/transforms/_transforms.py#L515) — the `nn.Module` wrapper layer, where every transform stays composable and TorchScript-friendly.
- [`lfilter`, L1032](https://github.com/pytorch/audio/blob/main/src/torchaudio/functional/filtering.py#L1032) — an IIR filter implemented with custom autograd, dispatching into native code.
- [`STABLE_TORCH_LIBRARY_FRAGMENT`, L118](https://github.com/pytorch/audio/blob/main/src/libtorchaudio/lfilter.cpp#L118) — the C++ side that declares the `_lfilter_core_loop` op and binds CPU, CUDA, and composite-fallback implementations.
- [`torch.ops.torchaudio.forced_align`, L72](https://github.com/pytorch/audio/blob/main/src/torchaudio/functional/_alignment.py#L72) — the call that jumps from pure Python straight into the compiled alignment kernel.
- [`falign_cuda_step_kernel`, L26](https://github.com/pytorch/audio/blob/main/src/libtorchaudio/forced_align/gpu/compute.cu#L26) — the CUDA kernel that performs the alignment sweep on GPU.

## From Install to Your First Model

Installation is deliberately boring, because TorchAudio 2.11 rides along with any modern torch build instead of pinning one:

```bash
pip install torchaudio
```

Note that `load` and `save` now require `torchcodec` to be importable (it ships in the CUDA and standard wheels; on some platforms you install it separately). Once it is there, the round trip works exactly as it always did:

```python
import torch, torchaudio

waveform, sr = torchaudio.load("speech.wav")          # [channel, time] float32
waveform_16k = torchaudio.functional.resample(waveform, sr, 16000)

torchaudio.save("speech_16k.wav", waveform_16k, 16000)

mel = torchaudio.transforms.MelSpectrogram(
    sample_rate=16000, n_fft=1024, hop_length=256
)(waveform_16k)
print(mel.shape)   # e.g. torch.Size([1, 64, T])
```

Every call in that snippet was verified against the checked-out source: `resample` lives in the functional layer, and `MelSpectrogram` composes `spectrogram` plus a mel filterbank without ever leaving tensor land, which is exactly what makes it exportable to other runtimes such as [Executorch](/executorch/) for on-device audio.

## One Call Away From the Metal: the lfilter Dispatch Chain

`lfilter` is the best place to watch the Python-to-C++ chain in this library. On the Python side, [`filtering.py`](https://github.com/pytorch/audio/blob/main/src/torchaudio/functional/filtering.py#L1032) pads the waveform, flips the coefficient tensors, and then calls an opaque op named `_lfilter_core_loop`. Three things make this more interesting than a plain loop:

1. **Custom autograd.** `DifferentiableIIR` (at [L983](https://github.com/pytorch/audio/blob/main/src/torchaudio/functional/filtering.py#L983)) and its FIR sibling define forward/backward pairs, so filtering an audio batch is trainable end to end — you can put a biquad before a loss and let gradients flow through the recurrence.
2. **Multi-backend dispatch.** In [`lfilter.cpp`](https://github.com/pytorch/audio/blob/main/src/libtorchaudio/lfilter.cpp#L126), one op schema gets three bindings: a CPU implementation, a CUDA implementation, and a `CompositeExplicitAutograd` fallback for anything else. The CUDA branch is implemented in [`iir_cuda.cu`](https://github.com/pytorch/audio/blob/main/src/libtorchaudio/iir_cuda.cu#L37), whose `iir_cu_kernel` (at [L11](https://github.com/pytorch/audio/blob/main/src/libtorchaudio/iir_cuda.cu#L11)) parallelizes across batch and channel while walking time sequentially — the classic IIR compromise.
3. **The stable library ABI.** These files use `STABLE_TORCH_LIBRARY_FRAGMENT` and `TORCH_BOX` rather than the older C++ extension API. That is the same direction the rest of the PyTorch ecosystem is taking: register ops by schema string, keep ABI churn out of downstream builds.

The same pattern repeats for the effects that came over from sox heritage: [`overdrive`, L1150](https://github.com/pytorch/audio/blob/main/src/torchaudio/functional/filtering.py#L1150) on the Python side has its kernel twin in [`overdrive.cpp`](https://github.com/pytorch/audio/blob/main/src/libtorchaudio/overdrive.cpp#L70), registered under `_overdrive_core_loop`. The biquad family (`lowpass_biquad`, `equalizer_biquad`, `bass_biquad`, and friends) stays in pure torch and only reuses the lfilter machinery — a nice example of paying the native-code toll only where the recurrence demands it.

## Forced Alignment and Transducer Loss, Compiled

The crown jewels of `libtorchaudio` are the sequence models' kernels. Forced alignment — aligning a known transcript to audio emissions, CTC-style — is a dynamic-programming scan that Python would crawl through. The flow here is textbook clean:

- The Python wrapper [`forced_align`, L12](https://github.com/pytorch/audio/blob/main/src/torchaudio/functional/_alignment.py#L12) validates shapes, fills default lengths, guards itself with `fail_if_no_align` (a decorator built from [`fail_with_message`, L61](https://github.com/pytorch/audio/blob/main/src/torchaudio/_internal/module_utils.py#L61)), then calls the op.
- The op signature is declared in [`compute.cpp`, L4](https://github.com/pytorch/audio/blob/main/src/libtorchaudio/forced_align/compute.cpp#L4) — five arguments in, paths and scores out.
- Two implementations exist: the CPU scan in [`forced_align_impl`, L19](https://github.com/pytorch/audio/blob/main/src/libtorchaudio/forced_align/cpu/compute.cpp#L19), and the GPU sweep in [`falign_cuda_step_kernel`, L26](https://github.com/pytorch/audio/blob/main/src/libtorchaudio/forced_align/gpu/compute.cu#L26). Both feed results back as label-per-timestep tensors that `merge_tokens` collapses into [`TokenSpan`, L77](https://github.com/pytorch/audio/blob/main/src/torchaudio/functional/_alignment.py#L77) objects.

RNN-transducer loss follows the same shape but with heavier machinery. The Python `rnnt_loss` (at [L1747](https://github.com/pytorch/audio/blob/main/src/torchaudio/functional/functional.py#L1747)) calls [`torch.ops.torchaudio.rnnt_loss_forward`, L1725](https://github.com/pytorch/audio/blob/main/src/torchaudio/functional/functional.py#L1725); the schema in [`rnnt/compute.cpp`, L4](https://github.com/pytorch/audio/blob/main/src/libtorchaudio/rnnt/compute.cpp#L4) accepts a `clamp` value and a `fused_log_softmax` flag; and the CUDA implementation drives a workspace-based [`GpuTransducer`](https://github.com/pytorch/audio/blob/main/src/libtorchaudio/rnnt/gpu/gpu_transducer.h#L145) whose alpha/beta scans are split into separate cooperative kernels. There is even a third decoding path: the `cuctc` subfolder ships a CUDA prefix-scoring CTC decoder whose ops are registered in [`python_binding.cpp`, L113](https://github.com/pytorch/audio/blob/main/src/libtorchaudio/cuctc/src/python_binding.cpp#L113) and driven by [`CUCTCDecoder`, L42](https://github.com/pytorch/audio/blob/main/src/torchaudio/models/decoder/_cuda_ctc_decoder.py#L42). These kernels pair naturally with the reference [`RNNT`, L453](https://github.com/pytorch/audio/blob/main/src/torchaudio/models/rnnt.py#L453) model, which composes an [`Emformer`](https://github.com/pytorch/audio/blob/main/src/torchaudio/models/emformer.py) encoder with a transducer head — and they complement generic serving stacks like [TorchServe](/torchserve/) when you need aligned, token-level outputs rather than plain logits.

## The Functional Layer and Transform Wrappers

Everything that does not need custom C++ stays in two pure-torch files, and that is most of the library. [`functional.py`](https://github.com/pytorch/audio/blob/main/src/torchaudio/functional/functional.py#L54) opens with `spectrogram` (at [L54](https://github.com/pytorch/audio/blob/main/src/torchaudio/functional/functional.py#L54)), then provides `griffinlim`, `melscale_fbanks` (at [L518](https://github.com/pytorch/audio/blob/main/src/torchaudio/functional/functional.py#L518)), phase-based `pitch_shift`, `fftconvolve`, beamforming utilities, and masking helpers for SpecAugment-style training. The [`_transforms.py`](https://github.com/pytorch/audio/blob/main/src/torchaudio/transforms/_transforms.py#L25) file wraps these as `nn.Module`s — `Resample` (at [L908](https://github.com/pytorch/audio/blob/main/src/torchaudio/transforms/_transforms.py#L908)) caches its kernel, `SpecAugment` (at [L1290](https://github.com/pytorch/audio/blob/main/src/torchaudio/transforms/_transforms.py#L1290)) composes the two masking ops, and `RNNTLoss` (at [L1792](https://github.com/pytorch/audio/blob/main/src/torchaudio/transforms/_transforms.py#L1792)) puts the native loss behind a module call so it slots into ordinary training loops. The split is a good design template: functions are stateless math, modules own configuration and cached state, and native ops are invoked only at the innermost, hot level.

Two more corners of the Python tree are worth knowing about. The `compliance` package reproduces Kaldi's feature extractors (`compliance.kaldi.fbank`, `mfcc`) bit-for-bit, which sounds academic until you have to drop a checkpoint trained in an old speech stack into a torch training loop and get identical features. And the `datasets` and `models` packages carry the reference implementations — WaveRNN, Tacotron2, DeepSpeech, ConvTasNet — kept not as a model zoo to download from, but as executable documentation for how these architectures compose with the transforms above. Reading `conv_tasnet.py` next to `functional.py` teaches mask-based source separation in about four hundred lines.

## Honest Limits

This is a library in maintenance mode, and the README says so directly: features were deprecated from TorchAudio 2.8 and removed in 2.9, with the goal of scoping the project to "processing audio data for ML". Concretely, what you give up and what you should watch for:

- **The backend matrix is gone.** Releases up to 2.8 offered three interchangeable I/O backends with real platform asymmetries: the sox backend was never shipped in Windows wheels, soundfile worked everywhere but could not stream or apply effects, and the ffmpeg backend dlopened versioned shared libraries with an ABI range per release. On main, all of that is replaced by TorchCodec, which is FFmpeg-based — so FFmpeg is still the true dependency, just one layer down and managed by another project. Check TorchCodec's own platform matrix before betting a deployment on it.
- **Compatibility parameters are accepted and ignored.** `load` still takes `normalize`, `buffer_size`, and `backend`, but they do nothing except emit warnings (see the guard clauses at the top of [`load_with_torchcodec`](https://github.com/pytorch/audio/blob/main/src/torchaudio/_torchcodec.py#L89)). Output is always normalized float32; `normalize=False` will not give you int16 samples anymore.
- **Whole-file decode, then slice.** Offset handling is post-hoc: the decoder fetches everything ([`get_all_samples`, L128](https://github.com/pytorch/audio/blob/main/src/torchaudio/_torchcodec.py#L128)) and only afterwards applies `frame_offset`/`num_frames` slicing. For a three-hour audiobook, loading thirty seconds still decodes the whole file into memory first. The docstring points you at `torchcodec.decoders.AudioDecoder` directly for time-based seeking when that matters.
- **Strict native-version checks.** [`_check_cuda_version`, L116](https://github.com/pytorch/audio/blob/main/src/torchaudio/_extension/utils.py#L116) raises at import time if torch and torchaudio were built against different CUDA *majors*, and the Windows loader needs the old conda-PATH workaround ([`_init_dll_path`, L101](https://github.com/pytorch/audio/blob/main/src/torchaudio/_extension/utils.py#L101)) plus `.pyd` globbing — reminders that native wheels still bite hardest on Windows.
- **Alignment constraints.** `forced_align` refuses batches larger than one (the check lives in the CPU impl: ["The batch dimension for targets must be 1", L177](https://github.com/pytorch/audio/blob/main/src/libtorchaudio/forced_align/cpu/compute.cpp#L177)), forbids blank tokens in targets, and requires the alignment extension to have been compiled in at all — the availability flag is probed via [`is_align_available`, L37](https://github.com/pytorch/audio/blob/main/src/torchaudio/_extension/__init__.py#L37) rather than assumed.
- **Export, not extension.** With `save` relying on TorchCodec's `AudioEncoder` (at [L157](https://github.com/pytorch/audio/blob/main/src/torchaudio/_torchcodec.py#L157)), exotic encodings and bit-depth controls that sox once offered have no equivalent; and since the library follows torch's release train without pinning it, CI matrices that assumed one-torch-one-torchaudio need updating.

None of this diminishes the engineering here. What remains is a tight, legible core: TorchCodec for bytes-to-tensor, pure torch for the wide middle, and a small, well-guarded native layer — `libtorchaudio` — for the algorithms that genuinely need compiled code. For interoperability beyond torch, the tensor shapes line up nicely with [ONNX](/onnx/) export paths, and the same thin-Python-over-native-kernels philosophy shows up across the ecosystem, from serving frameworks to this very repository. Read it while it is small; maintenance-mode codebases rarely stay this approachable for long.

<script type="application/ld+json">
{
  "@context": "https://schema.org",
  "@type": "TechArticle",
  "headline": "torchaudio Source Tour: TorchCodec I/O, Custom Autograd, and the C++/CUDA Kernels Behind Forced Alignment",
  "description": "A guided tour of the pytorch/audio source tree: how torchaudio.load hands decoding to TorchCodec, how lfilter dispatches into libtorchaudio CPU/CUDA kernels, and where the maintenance-mode limits are.",
  "author": {"@type": "Person", "name": "PyShine"},
  "datePublished": "2026-10-22",
  "keywords": "torchaudio, pytorch audio, torchcodec, forced alignment, RNNT loss, CTC, CUDA kernels, lfilter, libtorchaudio, audio decoding, source tour",
  "publisher": {"@type": "Organization", "name": "PyShine"}
}
</script>
