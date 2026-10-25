---
layout: post
title: "pyannote.audio source tour: the PyTorch core behind speaker diarization"
description: "A guided tour of the pyannote.audio source on GitHub: how Model, Task, Audio and Inference fit together with PyanNet-style architectures, ArcFace embeddings and TorchMetrics to train speaker diarization, VAD and speaker embedding models."
date: 2026-10-25
header-img: "img/post-bg.jpg"
permalink: /pyannote-audio/
featured-img: "ai-coding-frameworks/ai-coding-frameworks"
image: https://pyshine.com/assets/img/diagrams/pyannote-audio/pyannote-pyannote-audio-overview-architecture.svg
tags: [LLM, Python, Open Source]
categories: [AI Inference]
keywords: pyannote audio, speaker diarization, voice activity detection, speaker embedding, pytorch lightning, audio deep learning, open source
author: "PyShine"
---

If you have ever asked "who spoke when" about a meeting recording, there is a good chance pyannote.audio did the heavy lifting. It is the open-source Python toolkit behind many production diarization systems, a long-running entry in the Hugging Face Hub model registry, and one of the few end-to-end speaker diarization projects built squarely on [PyTorch](/pytorch/) and PyTorch Lightning. The library gives you three capabilities out of the box: speaker diarization (segmenting an audio stream into speaker turns), voice activity detection (VAD, separating speech from silence and noise), and speaker embedding (mapping a voice to a vector you can compare, cluster, or index).

What makes the repository worth a source tour is not the pretrained weights — those you download in one line — but the architecture underneath them. pyannote.audio separates its concerns with unusual discipline. A compact core defines what a model is (`Model`), what a learning problem is (`Task`, `Specifications`), how audio becomes tensors (`Audio`), and how trained weights are executed over long recordings (`Inference`). Everything else — the segmentation architectures, the embedding extractors, the training objectives — plugs into that core as swappable pieces. Once you see the seams, the codebase reads like a small framework rather than a pile of research scripts.

This post walks the source the way we do it in this series: two generated architecture diagrams, and every claim pinned to a real file and line number in the repository. The diagrams were built from the actual working tree of the default branch, so the symbols you see are the symbols that exist today.

<div style="overflow-x:auto;"><img src="https://pyshine.com/assets/img/diagrams/pyannote-audio/pyannote-pyannote-audio-overview-architecture.svg" alt="pyannote.audio architecture overview" style="max-width:100%;"></div>

<p><em>The overview graph groups pyannote.audio into a core framework, audio input and inference, model architectures, training tasks, and the metrics and helpers that glue them together.</em></p>

## Reading the overview from left to right:

The most useful way to absorb the overview diagram is to follow an audio recording through the system, from raw file to trained weights and back out again.

- **Audio decoding.** Everything starts in [Audio, L110](https://github.com/pyannote/pyannote-audio/blob/main/src/pyannote/audio/core/io.py#L110), the class that decodes media files, resamples them to the model's sample rate, and returns fixed-shape tensors. Every consumer in the diagram depends on this single, boring, essential abstraction: if the sample rate is wrong, nothing downstream can be trusted.
- **The model contract.** [Model, L70](https://github.com/pyannote/pyannote-audio/blob/main/src/pyannote/audio/core/model.py#L70) is a `LightningModule` base class that defines how architectures declare their input specs and how weights are saved and restored. Its [from_pretrained, L498](https://github.com/pyannote/pyannote-audio/blob/main/src/pyannote/audio/core/model.py#L498) method pulls checkpoints from the Hugging Face Hub and rebuilds the exact architecture the weights expect — the same pattern that made [Transformers](/transformers/) so easy to adopt.
- **Running over long files.** [Inference, L47](https://github.com/pyannote/pyannote-audio/blob/main/src/pyannote/audio/core/inference.py#L47) wraps a trained model and executes it with a sliding window, exposing `__call__` at [L375](https://github.com/pyannote/pyannote-audio/blob/main/src/pyannote/audio/core/inference.py#L375) and `crop` at [L414](https://github.com/pyannote/pyannote-audio/blob/main/src/pyannote/audio/core/inference.py#L414) so frame-wise scores can be sliced to arbitrary time ranges. This is the bridge between "a model that processes 5-second chunks" and "a report over a 2-hour meeting".
- **Architectures as leaf modules.** On the left of the diagram sit the neural architectures. [PyanNet, L38](https://github.com/pyannote/pyannote-audio/blob/main/src/pyannote/audio/models/segmentation/PyanNet.py#L38) is the classic SincNet-plus-LSTM stack for speech segmentation, and [SSeRiouSS, L42](https://github.com/pyannote/pyannote-audio/blob/main/src/pyannote/audio/models/segmentation/SSeRiouSS.py#L42) is its end-to-end sibling. Both subclass `Model`, which is why the edges flow from architectures to the core.
- **Building blocks.** [SincNet, L40](https://github.com/pyannote/pyannote-audio/blob/main/src/pyannote/audio/models/blocks/sincnet.py#L40) implements the learnable parametrized sinc filter bank that replaces generic mel features with a trainable waveform frontend, and [StatsPool, L63](https://github.com/pyannote/pyannote-audio/blob/main/src/pyannote/audio/models/blocks/pooling.py#L63) provides the statistics pooling head that turns a frame sequence into a fixed-size speaker embedding.
- **Training tasks.** The right side shows the objectives. [SpeakerDiarization, L60](https://github.com/pyannote/pyannote-audio/blob/main/src/pyannote/audio/tasks/segmentation/speaker_diarization.py#L60) is a training task (not the runtime orchestrator) that teaches a segmentation model about multi-speaker overlap, [VoiceActivityDetection, L36](https://github.com/pyannote/pyannote-audio/blob/main/src/pyannote/audio/tasks/segmentation/voice_activity_detection.py#L36) trains the binary speech/non-speech head, and [SupervisedRepresentationLearningWithArcFace, L38](https://github.com/pyannote/pyannote-audio/blob/main/src/pyannote/audio/tasks/embedding/arcface.py#L38) trains embedding models with an additive angular margin.
- **Measurement.** Finally, [DiarizationErrorRate, L34](https://github.com/pyannote/pyannote-audio/blob/main/src/pyannote/audio/torchmetrics/audio/diarization_error_rate.py#L34) is a TorchMetrics family — including confusion, false-alarm and missed-detection variants — that the diarization task imports for validation, so a training run reports the same numbers you would publish.

## The detail graph: inside the training loop

The overview tells you what talks to what; the detail graph shows the mechanics of one training epoch, which is where pyannote.audio spends most of its engineering effort.

<div style="overflow-x:auto;"><img src="https://pyshine.com/assets/img/diagrams/pyannote-audio/pyannote-pyannote-audio-architecture.svg" alt="pyannote.audio detailed architecture" style="max-width:100%;"></div>

<p><em>The detail graph adds the training-time machinery: shared segmentation task mixins, powerset decoding, augmentation hooks, Hugging Face Hub asset loading, and the receptive-field bookkeeping that keeps convolutional architectures honest about timing.</em></p>

Three details deserve your attention. First, code reuse between related objectives is real: [SegmentationTask, L46](https://github.com/pyannote/pyannote-audio/blob/main/src/pyannote/audio/tasks/segmentation/mixins.py#L46) centralizes chunk sampling and batch assembly, and both the diarization and VAD objectives extend it, while the base [Task, L191](https://github.com/pyannote/pyannote-audio/blob/main/src/pyannote/audio/core/task.py#L191) data module owns protocol handling and the train/val dataset plumbing. Second, the multi-speaker overlap problem is solved at the target level: the diarization task imports [Powerset, L36](https://github.com/pyannote/pyannote-audio/blob/main/src/pyannote/audio/utils/powerset.py#L36), which decodes overlapping speakers into a per-class problem a plain classification head can learn — a small idea with an outsized effect on DER. Third, train-time robustness is pluggable: [register_augmentation, L32](https://github.com/pyannote/pyannote-audio/blob/main/src/pyannote/audio/augmentation/registry.py#L32) wraps a model's forward pass with waveform transforms, [MixSpeakerDiarization, L31](https://github.com/pyannote/pyannote-audio/blob/main/src/pyannote/audio/augmentation/mix.py#L31) can synthesize or attenuate speaker overlap inside a batch, and [GraduallyUnfreeze, L31](https://github.com/pyannote/pyannote-audio/blob/main/src/pyannote/audio/core/callback.py#L31) freezes frontend layers early in fine-tuning and unfreezes them as training stabilizes.

Two quieter helpers round out the picture. [create_rng_for_worker, L31](https://github.com/pyannote/pyannote-audio/blob/main/src/pyannote/audio/utils/random.py#L31) gives every data-loading worker its own reproducible random stream so chunk sampling stays consistent across runs, and the receptive-field utilities let architectures compute, from layer parameters alone, exactly how many input samples they need — the kind of bookkeeping that prevents off-by-one errors when windows and strides meet.

## Specifications: one model, many heads

The quiet invention at the center of the core is that models describe themselves. When an architecture is instantiated, it declares a [Problem, L61](https://github.com/pyannote/pyannote-audio/blob/main/src/pyannote/audio/core/task.py#L61) (binary classification, multi-label, or representation learning), a temporal [Resolution, L73](https://github.com/pyannote/pyannote-audio/blob/main/src/pyannote/audio/core/task.py#L73) (frame- or window-level outputs), and a [Specifications, L83](https://github.com/pyannote/pyannote-audio/blob/main/src/pyannote/audio/core/task.py#L83) record tying those to the human-readable speaker and class labels it was trained with. That metadata travels with the checkpoint, and it is why the toolkit can do useful things without you re-explaining the setup every time: the inference layer knows the model's sample rate, its window shape, and what its outputs mean.

This is also the mechanism that makes multi-task training tractable instead of special-cased. A single segmentation model can carry two heads with different resolutions; the helper at [map_with_specifications, L30](https://github.com/pyannote/pyannote-audio/blob/main/src/pyannote/audio/utils/multi_task.py#L30) applies the right function to each head according to its own specifications, so the forward pass, the loss and the metrics all stay generic. The diarization training objective is essentially the first consumer of that machinery, and once it existed, objectives like voice activity detection became thin wrappers rather than new subsystems.

The same philosophy extends past the trainable core. The scoring side of the house includes a [PLDA, L34](https://github.com/pyannote/pyannote-audio/blob/main/src/pyannote/audio/core/plda.py#L34) implementation — probabilistic linear discriminant analysis, the classic statistical back-end for comparing speaker embeddings — sitting in the core next to the neural code because it consumes the same vectors the embedding architectures emit. Whether you take [XVectorSincNet, L210](https://github.com/pyannote/pyannote-audio/blob/main/src/pyannote/audio/models/embedding/xvector.py#L210) vectors through cosine similarity or through PLDA, the interface is the tensor, not the toolkit. Small decision, big consequence: evaluation and downstream indexing never need to know which architecture produced the embedding.

## From Install to Your First Model

Installation is deliberately plain: `pip install pyannote.audio` brings PyTorch, Lightning and the rest with it. On a machine with a working Python 3.10+ environment there is nothing to compile, and the package installs the `torch` variant your platform gives you by default.

The lowest-level entry point bypasses all orchestration and gives you exactly the two classes from the diagram:

```python
from pyannote.audio import Model
from pyannote.audio.core.inference import Inference

# a pretrained speech segmentation checkpoint, downloaded from the Hugging Face Hub
model = Model.from_pretrained("pyannote/segmentation-3.0")

# sliding-window inference over any file Inference can decode
inference = Inference(model)
scores = inference("meeting.wav")
```

`Model.from_pretrained` resolves the checkpoint on the Hub through the asset helper at [download_from_hf_hub, L44](https://github.com/pyannote/pyannote-audio/blob/main/src/pyannote/audio/utils/hf_hub.py#L44), restores the architecture, and returns a `LightningModule` you can also keep training. `Inference` then handles windowing and aggregation; the result is frame-wise speaker probabilities indexed by time, ready for thresholding with the [Binarize, L204](https://github.com/pyannote/pyannote-audio/blob/main/src/pyannote/audio/utils/signal.py#L204) utility or any post-processing you prefer.

For the full "who spoke when" story, the library ships a high-level diarization orchestration module that chains a segmentation model, an embedding extractor and a clustering step into one callable. That module lives outside the trainable core diagrammed above — it consumes `Model` instances and `Inference` output rather than defining them — which is exactly why the core stays testable: the composition layer can change without touching how anything is trained. If you have used [WhisperX](/whisperx/) for transcription alignment, you have already met this composition: it reuses pyannote's diarization entry point for its speaker labels. And if your product needs text-to-speech instead of speaker analysis, [Piper](/piper/) covers the synthesis side of the same audio stack.

Training your own model follows the same shape. Pick an architecture, wrap it in a task with a `pyannote.database` protocol, and hand both to the Lightning trainer. The [training_step, L402](https://github.com/pyannote/pyannote-audio/blob/main/src/pyannote/audio/tasks/segmentation/speaker_diarization.py#L402) implementation in the diarization task shows the whole recipe in one place: sample a chunk, decode its targets with the powerset transform, compute the multi-task loss, and log the metric family from [OptimalDiarizationErrorRate, L273](https://github.com/pyannote/pyannote-audio/blob/main/src/pyannote/audio/torchmetrics/audio/diarization_error_rate.py#L273) that finds the best operating threshold on validation data.

## Honest Limits

The framework tax is real. Training tasks require `pyannote.database` protocol objects, which means assembling a database collection with exact file paths and annotations before the first epoch runs. For datasets with messy folder layouts this is the most time-consuming part of the setup — more so than the model code itself. The reward is reproducibility, but you pay for it up front.

Multi-speaker overlap remains genuinely hard. The powerset decoding trick makes overlap learnable by a standard classification head, but with more than a handful of concurrent speakers the class count grows quickly, and diarization quality still degrades on real meeting audio with crosstalk, distant microphones and reverberant rooms. The pretrained checkpoints are strong on broadcast-style data and noticeably weaker elsewhere.

The metric can flatter you. DER variants that search for an optimal threshold on the validation set, like the ones at [OptimalDiarizationErrorRate, L273](https://github.com/pyannote/pyannote-audio/blob/main/src/pyannote/audio/torchmetrics/audio/diarization_error_rate.py#L273), are convenient for model selection but optimistic relative to a fixed-threshold deployment; the raw [DiarizationErrorRate, L34](https://github.com/pyannote/pyannote-audio/blob/main/src/pyannote/audio/torchmetrics/audio/diarization_error_rate.py#L34) with a fixed decision threshold is the conservative number to quote.

Finally, the abstractions cut both ways. Because architectures subclass `Model` and tasks subclass `Task`, you can replace any piece in isolation — but you inherit Lightning's conventions everywhere: device handling, checkpoint formats, callback timing and data-loader semantics are all Lightning's, not yours. The `Inference` class also processes files window by window, which is memory-friendly but not the fastest path if you only need one short clip; for speech-command or keyword-spotting workloads a lighter toolkit may serve you better than a diarization framework.

None of these limits are accidents. They are the visible edges of a design that treats diarization, VAD and embedding extraction as one learning problem with three heads — and reading the source makes that design, its trade-offs and its extension points all explicit.

<script type="application/ld+json">
{
  "@context": "https://schema.org",
  "@type": "TechArticle",
  "datePublished": "2026-10-25",
  "headline": "pyannote.audio source tour: the PyTorch core behind speaker diarization",
  "description": "A guided tour of the pyannote.audio source: how Model, Task, Audio and Inference fit together with PyanNet-style architectures, ArcFace embeddings and TorchMetrics to train speaker diarization, VAD and speaker embedding models.",
  "keywords": "pyannote audio, speaker diarization, voice activity detection, speaker embedding, pytorch lightning",
  "author": {"@type": "Person", "name": "PyShine"},
  "publisher": {"@type": "Organization", "name": "PyShine"}
}
</script>
