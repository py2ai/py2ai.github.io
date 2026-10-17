---
layout: post
title: "100+ Models, One Command - Inside hiyouga/LLaMA-Factory"
description: "A source tour of LLaMA-Factory, the unified fine-tuning factory: a 2686-line chat template registry, multimodal plugins, stage dispatch from pretraining to PPO, and a v1 engine rewrite."
date: 2026-10-17 00:00:00 +0000
header-img: "img/post-bg.jpg"
permalink: /llama-factory/
featured-img: ai-coding-frameworks/ai-coding-frameworks
image: https://pyshine.com/assets/img/diagrams/llama-factory/hiyouga-llama-factory-overview-architecture.svg
tags: [llama-factory, fine-tuning, LoRA, DPO, PPO, multimodal]
categories: [AI Training]
keywords: "LLaMA-Factory source code, llamafactory-cli, LLaMA Board, chat template registry, multimodal fine-tuning, PPO DPO KTO training, QLoRA, v1 engine"
author: "PyShine"
---

Where axolotl bets on one exhaustive YAML schema, LLaMA-Factory bets on breadth: one CLI that fine-tunes over a hundred model families, one data loader that speaks every open dataset format, and one stage dispatcher that walks from continuous pretraining through supervised fine-tuning into reward modeling, PPO, DPO, and KTO. The project behind the widely cited "Unified Efficient Fine-Tuning of 100+ Language Models" paper has become the default answer to "how do I fine-tune X" for a whole generation of practitioners, and it ships a Gradio web UI so the entire journey runs from a browser. Apache-2.0 licensed, it packs its power into a surprisingly compact src tree of about 180 Python files - most of the surface area lives in a handful of genuinely enormous modules.

Those big modules are the story. A 2686-line template file encodes every supported chat format as data; a 3502-line multimodal plugin file teaches every vision-language family how to emit training rows; a 978-line trainer utilities file carries the callbacks and metrics every stage shares. Reading the repo is really reading three or four epic files plus a lattice of small workflow scripts that glue them together. As always in this series, what follows is an educational tour of published source code.

<div style="text-align:center; margin: 20px 0;">
  <img src="https://pyshine.com/assets/img/diagrams/llama-factory/hiyouga-llama-factory-overview-architecture.svg" alt="llama-factory overview architecture diagram" style="max-width:100%; height:auto;">
</div>
<em>Overview: the CLI routes to a stage dispatcher; argument parsing, the template registry, and dataset processing feed model loading and the workflows; a chat facade serves the result.</em>

Reading the overview from left to right:

- [cli.py](https://github.com/hiyouga/LLaMA-Factory/blob/main/src/llamafactory/cli.py) is a 31-line console entry that forwards every subcommand to the launcher.
- [launcher.py](https://github.com/hiyouga/LLaMA-Factory/blob/main/src/llamafactory/launcher.py) at 185 lines routes train, chat, api, export, webui, and eval to their implementations.
- [hparams/parser.py](https://github.com/hiyouga/LLaMA-Factory/blob/main/src/llamafactory/hparams/parser.py) defines get_train_args at line 395, fusing five argument families into one validated tuple.
- [train/tuner.py](https://github.com/hiyouga/LLaMA-Factory/blob/main/src/llamafactory/train/tuner.py) exposes run_exp at line 163 with the stage dispatch at line 69.
- [data/loader.py](https://github.com/hiyouga/LLaMA-Factory/blob/main/src/llamafactory/data/loader.py) loads and merges datasets via get_dataset at line 276 with caching at line 229.
- [data/template.py](https://github.com/hiyouga/LLaMA-Factory/blob/main/src/llamafactory/data/template.py) is the template registry: register_template at line 681 defines every supported chat format.
- [data/processor/supervised.py](https://github.com/hiyouga/LLaMA-Factory/blob/main/src/llamafactory/data/processor/supervised.py) tokenizes rows in SupervisedDatasetProcessor at line 51.
- [model/loader.py](https://github.com/hiyouga/LLaMA-Factory/blob/main/src/llamafactory/model/loader.py) builds the model while [chat/chat_model.py](https://github.com/hiyouga/LLaMA-Factory/blob/main/src/llamafactory/chat/chat_model.py) serves it afterwards.

## Why You Need This

Every fine-tuning tool solves the same four problems - arguments, data, model prep, training loop - and comparing implementations is the fastest way to understand the trade space. Where the [axolotl](/axolotl/) tour showed a pydantic schema validating one giant YAML, LLaMA-Factory shows the Hugging Face idiom: five dataclass families (model, data, training, finetuning, generating) parsed from CLI or dict and cross-verified in _verify_model_args at line 227 of the parser. Neither is wrong; seeing both teaches you what a config system must guarantee regardless of style. The same comparison holds for adapters - this repo's [model/adapter.py](https://github.com/hiyouga/LLaMA-Factory/blob/main/src/llamafactory/model/adapter.py) implements full, freeze, and LoRA tuning side by side, next to the PEFT internals we covered in the [peft](/peft/) tour.

The template registry deserves particular attention because it quietly carries the entire "100+ models" claim. A chat model fine-tune is useless if the prompt format does not match the pretrained format, and this file turns that compatibility problem into data: each register_template call declares system messages, user and assistant wrappers, stop tokens, and tool-call encodings for one family. New model support lands as a few lines of registration rather than a fork of the trainer - which is exactly why the README's day-0 support table keeps filling in. If you maintain any chat-adjacent system, this is the most complete public catalog of what varies between chat formats.

Multimodal support is the other eye-opener. The [data/mm_plugin.py](https://github.com/hiyouga/LLaMA-Factory/blob/main/src/llamafactory/data/mm_plugin.py) file gives every vision-language family - LLaVA, Gemma 3, InternVL, Kimi-VL and more - a plugin that converts raw rows into the exact interleave of text and image/video tokens that family's model class expects, including pixel-value preprocessing and image token splicing. Rather than one generic VLM loader, the repo embraces per-family reality. And when the current architecture finally hits its limits, the v1 subtree shows the rewrite: a cleaner engine split (data_engine, model_engine, base_trainer) with its own plugin system - worth reading as a case study in how projects migrate themselves.

<div style="text-align:center; margin: 20px 0;">
  <img src="https://pyshine.com/assets/img/diagrams/llama-factory/hiyouga-llama-factory-architecture.svg" alt="llama-factory detailed architecture diagram" style="max-width:100%; height:auto;">
</div>
<em>Detailed view: entry and dispatch, argument families, the data machinery with templates and multimodal plugins, model preparation, the six stage trainers, and the inference surface.</em>

Follow one stage end to end. llamafactory-cli train lands in [launcher.py](https://github.com/hiyouga/LLaMA-Factory/blob/main/src/llamafactory/launcher.py), which calls run_exp at line 163 of [train/tuner.py](https://github.com/hiyouga/LLaMA-Factory/blob/main/src/llamafactory/train/tuner.py); read_args at line 112 of [hparams/parser.py](https://github.com/hiyouga/LLaMA-Factory/blob/main/src/llamafactory/hparams/parser.py) accepts either a dict or command-line tokens, and get_train_args at line 395 returns the five-class tuple after _verify_model_args at line 227 rejects contradictions like a missing template. The _training_function at line 69 then dispatches: stage sft goes to run_sft at line 41 of [train/sft/workflow.py](https://github.com/hiyouga/LLaMA-Factory/blob/main/src/llamafactory/train/sft/workflow.py), while pt, rm, ppo, dpo, and kto each have a sibling workflow - and lines 92 through 136 show optional detours into hyper-parallel, Megatron-Bridge, and Megatron-core backends when those packages are installed.

The data path inside run_sft is the loader at work. [data/loader.py](https://github.com/hiyouga/LLaMA-Factory/blob/main/src/llamafactory/data/loader.py) merges every named dataset at line 164, converts rows via [data/converter.py](https://github.com/hiyouga/LLaMA-Factory/blob/main/src/llamafactory/data/converter.py) (alpaca, sharegpt, openai formats), resolves the chat template through get_template_and_fix_tokenizer at line 819 of [data/template.py](https://github.com/hiyouga/LLaMA-Factory/blob/main/src/llamafactory/data/template.py), and hands rows to the processors: SupervisedDatasetProcessor at line 51 of [data/processor/supervised.py](https://github.com/hiyouga/LLaMA-Factory/blob/main/src/llamafactory/data/processor/supervised.py) builds input_ids with labels masked outside responses, its packed subclass at line 146 concatenates sequences with position-id resets, and the PairwiseDatasetProcessor at line 30 of [data/processor/pairwise.py](https://github.com/hiyouga/LLaMA-Factory/blob/main/src/llamafactory/data/processor/pairwise.py) produces the chosen/rejected pairs that DPO and reward stages consume. Collators from [data/collator.py](https://github.com/hiyouga/LLaMA-Factory/blob/main/src/llamafactory/data/collator.py) finish the batches.

Model preparation is the patcher's show. [model/loader.py](https://github.com/hiyouga/LLaMA-Factory/blob/main/src/llamafactory/model/loader.py) picks quantization through [model/model_utils/quantization.py](https://github.com/hiyouga/LLaMA-Factory/blob/main/src/llamafactory/model/model_utils/quantization.py) - AQLM, AWQ, GPTQ, HQQ, EETQ, and bitsandbytes variants - then [model/patcher.py](https://github.com/hiyouga/LLaMA-Factory/blob/main/src/llamafactory/model/patcher.py) applies patch_config at line 365 (rope scaling, attention impl, flash-attn selection), patch_model at line 437 (longlora shift attention, packed attention, MoE and vision fixes), and unsloth support from [model/model_utils/unsloth.py](https://github.com/hiyouga/LLaMA-Factory/blob/main/src/llamafactory/model/model_utils/unsloth.py) when requested. init_adapter at line 317 of [model/adapter.py](https://github.com/hiyouga/LLaMA-Factory/blob/main/src/llamafactory/model/adapter.py) then branches: full tuning at line 38, freeze tuning at line 57, and _setup_lora_tuning at line 147 covering LoRA, DoRA, PiSSA, and LoftQ variants. The workflow finishes by instantiating CustomSeq2SeqTrainer at line 47 of [train/sft/trainer.py](https://github.com/hiyouga/LLaMA-Factory/blob/main/src/llamafactory/train/sft/trainer.py) with metrics from [train/trainer_utils.py](https://github.com/hiyouga/LLaMA-Factory/blob/main/src/llamafactory/train/trainer_utils.py).

The serving half closes the loop. ChatModel at line 39 of [chat/chat_model.py](https://github.com/hiyouga/LLaMA-Factory/blob/main/src/llamafactory/chat/chat_model.py) wraps three engines - hf, vLLM (VllmEngine at line 46 of [chat/vllm_engine.py](https://github.com/hiyouga/LLaMA-Factory/blob/main/src/llamafactory/chat/vllm_engine.py)), and sglang - behind one astream_chat interface; [api/app.py](https://github.com/hiyouga/LLaMA-Factory/blob/main/src/llamafactory/api/app.py) exposes it as an OpenAI-compatible server; and the LLaMA Board web UI's [webui/runner.py](https://github.com/hiyouga/LLaMA-Factory/blob/main/src/llamafactory/webui/runner.py) submits training jobs straight into the same run_exp entry, so GUI and CLI are one system. Evaluation through [eval/evaluator.py](https://github.com/hiyouga/LLaMA-Factory/blob/main/src/llamafactory/eval/evaluator.py) scores MMLU-style benchmarks with the same template machinery.

Two registries make the 100-model claim hold up. The template registry in data/template.py is a plain dict called TEMPLATES at line 678, filled by register_template at line 681 and consulted by get_template_and_fix_tokenizer at line 819; the base Template class at line 41 carries specializations like Llama2Template at line 454 for the older Llama chat format, and ReasoningTemplate at line 524 with its Qwen38ReasoningTemplate subclass at line 593 for models that emit thinking traces before answers. Multimodal support follows the same shape in mm_plugin.py, a 3,502-line module whose BasePlugin at line 421 defines the default vision interface and whose per-family subclasses - Gemma3Plugin at line 876, InternVLPlugin at line 1180, KimiVLPlugin at line 1330, and LlavaPlugin at line 1429 - override how images and videos expand into placeholder tokens, how processor outputs are cached, and how the collator stitches visual features into the sequence. Adding a new model family usually means registering one template and one plugin instead of touching the trainer at all.

The alignment stages get their own trainers with minimal duplication. CustomDPOTrainer at line 44 of train/dpo/trainer.py extends the TRL DPOTrainer and overrides concatenated_forward at line 219 to route preference batches through the repo's own collators and mm plugin, while CustomPPOTrainer at line 64 of train/ppo/trainer.py inherits from both PPOTrainer and Trainer to blend rollout generation with the standard optimization step. Cross-cutting behavior lands in callbacks rather than trainer edits: LogCallback at line 174 of train/callbacks.py appends each stage's timing and loss to the trainer_log.jsonl you find in every output directory, PissaConvertCallback at line 132 swaps PiSSA adapters between their init and training representations, ReporterCallback at line 433 writes the training report when a run ends, and ModuleProfilerCallback at line 491 measures per-module memory so you can spot which layers are worth offloading or quantizing. When one node is not enough, run_exp checks ray_args.use_ray and re-enters itself through Ray, with the placement-group helpers in trainer_utils.py - get_placement_group, get_ray_head_node_ip, sort_placement_group_by_node_ip - pinning workers across the cluster, so the same YAML scales from a single GPU to a multi-node job without edits.

## From Install to Fine-Tuned Model

The README's zero-code loop:

```sh
uv run llamafactory-cli webui
llamafactory-cli train examples/train_lora/qwen3_lora_sft.yaml
llamafactory-cli chat examples/inference/qwen3_lora_sft.yaml
llamafactory-cli export examples/merge_lora/qwen3_lora_sft.yaml
```

Or programmatically, one dict instead of YAML:

```python
from llamafactory.train.tuner import run_exp

run_exp(args={
    "stage": "sft",
    "model_name_or_path": "Qwen/Qwen3-8B",
    "template": "qwen3",
    "dataset": "alpaca_en_demo",
    "finetuning_type": "lora",
    "output_dir": "qwen3-lora-out",
})
```

Swap stage to dpo with a pairwise dataset and the same entry runs preference tuning; point infer_backend at vllm and the same chat model serves hundreds of concurrent requests.

Honest limits: the breadth comes with weight - the giant template, mm_plugin, and trainer_utils files are dense and occasionally duplicate logic across stage trainers, so tracing one behavior can mean reading three files. The WebUI covers the happy path well but exotic combinations still want the CLI. And because so much support is template registration plus small patches, per-model quirks sometimes surface only in that model's issue tracker, not the docs.

<script type="application/ld+json">
{
  "@context": "https://schema.org",
  "@type": "BlogPosting",
  "headline": "100+ Models, One Command - Inside hiyouga/LLaMA-Factory",
  "description": "A source tour of LLaMA-Factory, the unified fine-tuning factory: a 2686-line chat template registry, multimodal plugins, stage dispatch from pretraining to PPO, and a v1 engine rewrite.",
  "author": {"@type": "Person", "name": "PyShine"},
  "datePublished": "2026-10-17",
  "url": "https://pyshine.com/llama-factory/"
}
</script>
