---
layout: post
title: "The Recipe Book Behind Zephyr and SmolLM3 - Inside huggingface/alignment-handbook"
description: "A source tour of the Alignment Handbook: a thin TRL wrapper, the dataset mixture API, the Zephyr-7B-beta two-stage recipe, SmolLM3's 26-corpus blend, and the release tooling that ships models to the Hub."
date: 2026-10-17 00:00:00 +0000
header-img: "img/post-bg.jpg"
permalink: /alignment-handbook/
featured-img: ai-coding-frameworks/ai-coding-frameworks
image: https://pyshine.com/assets/img/diagrams/alignment-handbook/huggingface-alignment-handbook-overview-architecture.svg
tags: [alignment-handbook, Zephyr, SmolLM3, DPO, ORPO, SFT]
categories: [AI Training]
keywords: "alignment handbook source code, Zephyr 7B recipe, SmolLM3 post-training, dataset mixture API, TRL SFTTrainer DPOTrainer ORPOTrainer, DeepSpeed ZeRO-3 recipe"
author: "PyShine"
---

Some repositories impress with volume, and some with restraint. The Alignment Handbook, from Lewis Tunstall, Edward Beeching, Nathan Lambert, Nazneen Rajani, Shengyi Huang, Kashif Rasul, Alvaro Bartolome, Carlos M. Patino, Alexander M. Rush, and Thomas Wolf at Hugging Face, is emphatically the second kind. It is the codebase that produced Zephyr-7B-beta - one of the most downloaded fine-tuned chat models of its moment on the Hub - and later the post-training recipe behind SmolLM3-3B, yet its entire Python surface is barely 1,250 lines: three launch scripts that mostly wire arguments together, a four-module support package, and one test file. The real payload is the recipes directory: versioned YAML files that record every hyperparameter, dataset split, and accelerate setting behind each released model, with a technical report at arXiv 2310.16944 documenting the results.

As always in this series, what follows is an educational tour of published source code.

<div style="text-align:center; margin: 20px 0;">
  <img src="https://pyshine.com/assets/img/diagrams/alignment-handbook/huggingface-alignment-handbook-overview-architecture.svg" alt="Alignment Handbook overview architecture diagram" style="max-width:100%; height:auto;">
</div>
<em>Overview: accelerate configs and Slurm launchers feed the three training scripts; a thin alignment package validates dataset mixtures and loads models; each script wraps one TRL trainer; recipe YAMLs drive everything; release tooling ships checkpoints to the Hub.</em>

Reading the overview from left to right:

- [scripts/sft.py](https://github.com/huggingface/alignment-handbook/blob/main/scripts/sft.py) is the 174-line supervised fine-tuning entry point: main at line 54 loads data at 87, tokenizer at 91, and model at 96, falls back to ChatML when the tokenizer has no template at 98-100, and builds TRL's SFTTrainer at 105-112.
- [scripts/dpo.py](https://github.com/huggingface/alignment-handbook/blob/main/scripts/dpo.py), 159 lines, builds the policy and a second frozen reference model with two get_model calls at lines 100-101, then hands both to DPOTrainer at 122-130 - the same implicit-reward loss we toured in [direct-preference-optimization](/direct-preference-optimization/), now production-wrapped.
- [scripts/orpo.py](https://github.com/huggingface/alignment-handbook/blob/main/scripts/orpo.py), 158 lines, is the single-stage variant: ORPOTrainer at line 122 takes no reference model, fusing SFT and preference alignment into one run.
- [src/alignment/configs.py](https://github.com/huggingface/alignment-handbook/blob/main/src/alignment/configs.py) extends TRL's argument dataclasses: DatasetConfig at line 37, DatasetMixtureConfig at 48, and ScriptArguments at 57 whose __post_init__ at 88 validates mixtures and enforces consistent column sets at 122-130.
- [src/alignment/data.py](https://github.com/huggingface/alignment-handbook/blob/main/src/alignment/data.py) is the 79-line heart of the mixture API: get_dataset at line 26 loads each entry, subsamples by weight at 52-56, concatenates and reshuffles at 61-62, and optionally carves a test split at 65-68.
- [src/alignment/model_utils.py](https://github.com/huggingface/alignment-handbook/blob/main/src/alignment/model_utils.py) holds get_tokenizer at line 23 - which can override the chat template at 31-32 - and get_model at 37, resolving dtype and quantization at 39-42 and disabling the KV cache under gradient checkpointing at 48.
- [recipes/zephyr-7b-beta/dpo/config_full.yaml](https://github.com/huggingface/alignment-handbook/blob/main/recipes/zephyr-7b-beta/dpo/config_full.yaml) is the artifact that made the repo famous: it starts from the SFT checkpoint, mixes ultrafeedback_binarized preference splits, and sets beta 0.01 with learning rate 5e-7.
- [recipes/smollm3/sft/sft.yaml](https://github.com/huggingface/alignment-handbook/blob/main/recipes/smollm3/sft/sft.yaml) is the modern extreme: 228 lines declaring 26 weighted smoltalk2 sources, packing with the ffd strategy at length 65536, assistant-only loss, and five epochs.

## Why You Need This

Frameworks are places where options go to hide. [LLaMA-Factory](/llama-factory/) and [axolotl](/axolotl/) expose hundreds of switches because they must serve every use case; the Handbook inverts that - it serves exactly one use case, replicating published results, and strips everything else away. Each script is a legible main function you can read top to bottom in two minutes: set seed, load data, load tokenizer, load model, construct trainer, train, save, push. Where [ms-swift](/ms-swift/) routes dozens of model families through registries and [OpenRLHF](/openrlhf/) distributes a controller over Ray, the Handbook's entire distribution story is three accelerate config files and a 176-line Slurm script. Reading it teaches you what a TRL training run actually consists of when no framework is watching.

The mixture API deserves its own study because data blending turned out to be the quiet superpower of every model this repo shipped. The Zephyr SFT recipe mixes ultrachat_200k train and test splits with equal weights; the SmolLM3 recipe blends 26 sources with weights from 1.0 down to 0.02, balancing everyday chats against function calling, long-context tasks, and reasoning traces in both think and no-think variants. The implementation is deliberately simple - shuffle, take a weight fraction, concatenate, reshuffle - and tests/test_data.py pins the arithmetic: three 100-row datasets weighted 0.5, 0.3, and 0.2 yield exactly 90 train and 10 test rows, asserted at lines 44-45. If you have ever wondered how recipe authors express "mostly instruction data, a dash of math," this is the answer, and it transfers directly to your own training runs, whether you fine-tune full weights with [deepspeed](/deepspeed/) ZeRO-3 or bolt [peft](/peft/) adapters on through the same get_peft_config call at sft.py line 111.

Finally, the Handbook is the missing link between the algorithm papers this series has covered and the models you actually download. The DPO recipe here is the industrial version of the reference implementation - same loss, but with checkpoint resumption, Hub push, model cards tagged alignment-handbook at sft.py line 145, and the generation config's eos_token_id synced to the tokenizer at lines 136-137 to prevent unbounded generation. The ORPO script shows the alternative trade - skip the reference model entirely - while the evaluation guidance in scripts/README.md points at MT-Bench, AlpacaEval, and Chatbot Arena, a reminder that [lm-evaluation-harness](/lm-evaluation-harness/) scores are only one lens on a chat model. The repo even ships its own release tooling, because a recipe you cannot re-version is a recipe you cannot trust.

<div style="text-align:center; margin: 20px 0;">
  <img src="https://pyshine.com/assets/img/diagrams/alignment-handbook/huggingface-alignment-handbook-architecture.svg" alt="Alignment Handbook detail architecture diagram" style="max-width:100%; height:auto;">
</div>
<em>Detail: three script mains share the parser, mixture API, and model loaders; each wraps a different TRL trainer; recipe YAMLs parameterize everything from Zephyr's two stages to SmolLM3's 26-corpus blend; release tooling closes the versioning loop.</em>

The scripts are deliberately near-identical, and the diffs are where the interesting decisions live. All three begin with set_seed at line 56 (sft.py), checkpoint detection via get_last_checkpoint at 78-82, and the same three-part TrlParser at the bottom - ScriptArguments, a task config, and TRL's ModelConfig - so every recipe is one YAML deep. sft.py's signature moves: when the tokenizer lacks a chat template, setup_chat_format retrofits the model's embedding matrix for ChatML special tokens at 98-100; before saving, eos_token_id is written into both generation_config and config at 136-137; and on the main process only, use_cache is switched back on at 150-151 since training disabled it for gradient checkpointing. dpo.py instead builds two models - policy at line 100, reference at 101 - falls back to eos as the pad token at 103-104, carries an ignore_bias_buffers switch at 105-109 that tells DDP to skip boolean buffers (a workaround for Flash Attention asserts in distributed DPO), and strips the messages column at 114-117 so the trainer sees only chosen/rejected pairs. orpo.py removes the reference model altogether, matching the method's single-stage design documented in its docstring with max_length 2048.

The mixture chain is three short files acting in concert. ScriptArguments at configs.py line 57 accepts either dataset_name or a dataset_mixture dict; __post_init__ at 88 rejects configs with neither, coerces the raw dict into DatasetConfig entries at 99-112 - each carrying id, config name, split, a columns whitelist, and a weight - and then insists at 122-130 that every entry whitelists the same column names, because concatenation would otherwise fail late and cryptically. get_dataset in data.py takes it from there: the single-dataset branch at lines 35-37 is a plain load_dataset call, while the mixture branch at 38-76 loads each source with its split at 45-49, applies select_columns at 50-51, implements weights as shuffle-then-select(range(int(len(ds) * weight))) at 52-56, concatenates at 61, applies one final seeded shuffle at 62, and either returns a DatasetDict with a train_test_split at 65-68 or wraps everything as a lone train split at 73-74. The tests lock the semantics in at three weight regimes, including the fractional case where 70 plus 40 rows become 99 train and 11 test at lines 87-105.

The model layer is fifty-seven lines that encode real distributed-training judgment. get_tokenizer at model_utils.py line 23 loads with the model revision and trust_remote_code flags from ModelConfig, then applies the recipe's chat_template override at 31-32 - which is how Zephyr's custom template and SmolLM3's reasoning-aware template, both stored as YAML strings inside the recipes, reach the tokenizer without any code changes. get_model at line 37 resolves torch_dtype from the string at 39-41, asks TRL for a bitsandbytes quantization config at 42, and assembles model_kwargs at 43-51: attention implementation passthrough, use_cache forced False whenever gradient_checkpointing is on at 48, and a kbit device map only when quantization is active at 49. That last detail is the difference between a QLoRA run that colocates quantized weights correctly across GPUs and one that silently double-loads.

The recipes are the repo's true deliverables, and comparing them is a history of post-training itself. Zephyr-7B-beta's SFT config points at Mistral-7B-v0.1 with flash_attention_2, a learning rate of 2e-5 for one epoch, sequences of 2048 tokens, and a per-device batch of 16 on the eight-GPU ZeRO-3 node declared in recipes/accelerate_configs/zero3.yaml - which sets zero_stage 3, zero3_init_flag, and zero3_save_16bit_model at lines 7-9; its DPO config then starts from the resulting Hub checkpoint with beta 0.01 and lr 5e-7, the low-beta regime the Zephyr report found best. The fsdp.yaml alternative chooses FULL_SHARD with TRANSFORMER_BASED_WRAP and BACKWARD_PRE prefetch - the same strategy family as the [direct-preference-optimization](/direct-preference-optimization/) reference - plus SHARDED_STATE_DICT and cpu_ram_efficient_loading. SmolLM3's sft.yaml pushes the mixture API to its limit: 26 entries from smoltalk2 with weights like 0.02 for OpenThoughts3 reasoning traces and 0.5 for magpie-ultra, assistant_only_loss true, packing_strategy ffd at 65536 tokens, liger kernel enabled, cosine_with_min_lr at min_lr_rate 0.1, and a ddp_timeout of 18000 seconds because tokenizing that blend takes hours. SmolLM3's DPO stage then runs loss_type apo_zero with beta 0.05 and padding_free true, while smollm2 shows the small-model contrast: beta 0.5, three epochs, tensorboard plus wandb. The Slurm launcher ties it together - it greps gradient_accumulation_steps out of the chosen config at line 90, reserves the last node as a vLLM server when use_vllm is set at 117-130, exports NCCL_ASYNC_ERROR_HANDLING at 133, and sruns the accelerate command across nodes at 140-167.

Release discipline rounds out the design. src/alignment/release.py keeps a REPLACE_PATTERNS table at lines 22-36 that rewrites the version string in four files at once - __init__.py, setup.py, CITATION.cff, and the README citation block - with pre_release_work at 73 refusing to patch from a dev branch at 77-78 and post_release_work at 95 bumping to the next dev0. setup.py carries the pinning that makes recipes reproducible: trl>=0.19.1, transformers>=4.53.3, torch>=2.6.0, peft>=0.16.0, deepspeed>=0.17.2, and liger-kernel>=0.6.0 among the deps at lines 43-73, with the package itself named alignment-handbook at version 0.4.0.dev0. It is a small thing until you try to reproduce a 2023 result on a 2026 stack - then the recipe that pins its world is the only one that still works.

## From Install to a Replicated Zephyr

The README and scripts/README.md give a five-command path from nothing to a replicated Zephyr-7B-beta (commands abridged):

```sh
uv venv handbook --python 3.11 && source handbook/bin/activate
uv pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu126
uv pip install . && uv pip install "flash-attn==2.7.4.post1" --no-build-isolation
huggingface-cli login
# Stage 1 - SFT on 8 GPUs with ZeRO-3
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/zero3.yaml \
  scripts/sft.py --config recipes/zephyr-7b-beta/sft/config_full.yaml
# Stage 2 - DPO from the SFT checkpoint
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/zero3.yaml \
  scripts/dpo.py --config recipes/zephyr-7b-beta/dpo/config_full.yaml
```

Every parameter can be overridden on the command line - the docs suggest scaling batch size with GPU count to keep the global batch, and recipes will push straight to your Hub namespace by default.

Honest limits: the scripts are thin by design, so anything TRL does not expose - custom losses, curriculum scheduling, multi-stage reward modeling - means editing library code, not adding a config; the mixture API loads and concatenates full datasets in memory, which large blends only survive through the streaming-free weight trick it does not implement; the Slurm launcher is tuned for the Hugging Face cluster and says so; chat model evaluation leans on GPT-4 judges (MT-Bench, AlpacaEval) whose biases the README itself flags, including a taste for GPT-distilled models; and the repo is Apache 2.0 code wrapping libraries that move fast, so the exact torch, transformers, and trl pins are not suggestions - the recipes will silently change behavior without them.

<script type="application/ld+json">
{
  "@context": "https://schema.org",
  "@type": "BlogPosting",
  "headline": "The Recipe Book Behind Zephyr and SmolLM3 - Inside huggingface/alignment-handbook",
  "description": "A source tour of the Alignment Handbook: a thin TRL wrapper, the dataset mixture API, the Zephyr-7B-beta two-stage recipe, SmolLM3's 26-corpus blend, and the release tooling that ships models to the Hub.",
  "image": "https://pyshine.com/assets/img/diagrams/alignment-handbook/huggingface-alignment-handbook-overview-architecture.svg",
  "author": {"@type": "Person", "name": "PyShine"},
  "publisher": {"@type": "Organization", "name": "PyShine"},
  "datePublished": "2026-10-17",
  "dateModified": "2026-10-17",
  "mainEntityOfPage": "https://pyshine.com/alignment-handbook/",
  "keywords": "alignment handbook, Zephyr 7B, SmolLM3, DPO, ORPO, SFT recipes"
}
</script>
