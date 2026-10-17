---
layout: post
title: "Your Language Model Is Secretly a Reward Model - Inside eric-mitchell/direct-preference-optimization"
description: "A source tour of the DPO reference implementation: three preference dataset loaders, the concatenated log-probability trick, DPO/IPO/cDPO losses in one 40-line function, and trainer classes from single GPU to FSDP."
date: 2026-10-17 00:00:00 +0000
header-img: "img/post-bg.jpg"
permalink: /direct-preference-optimization/
featured-img: ai-coding-frameworks/ai-coding-frameworks
image: https://pyshine.com/assets/img/diagrams/direct-preference-optimization/eric-mitchell-direct-preference-optimization-overview-architecture.svg
tags: [direct-preference-optimization, DPO, alignment, Pythia, FSDP]
categories: [AI Training]
keywords: "DPO reference implementation, direct preference optimization source code, preference_loss, concatenated forward, FSDP trainer, Anthropic hh-rlhf, Pythia 6.9B"
author: "PyShine"
---

In May 2023 a Stanford team - Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D. Manning, Stefano Ermon, and Chelsea Finn - posted "Direct Preference Optimization: Your Language Model Is Secretly a Reward Model" (arXiv 2305.18290), and the title carried the whole idea: instead of training a separate reward model and then squeezing a policy against it with reinforcement learning, DPO derives a closed-form mapping from an optimal KL-constrained policy back to its implicit reward, and rewrites the entire alignment objective as one classification loss over preference pairs. No PPO rollouts, no reward-model head, no KL penalty coefficients to babysit. This repository is the paper's reference implementation: four Python files totaling about 1,224 lines - no framework, no plugin registry, just the loss, three trainer classes, and loaders for the three preference corpora the experiments used.

As always in this series, what follows is an educational tour of published source code.

<div style="text-align:center; margin: 20px 0;">
  <img src="https://pyshine.com/assets/img/diagrams/direct-preference-optimization/eric-mitchell-direct-preference-optimization-overview-architecture.svg" alt="Direct Preference Optimization overview architecture diagram" style="max-width:100%; height:auto;">
</div>
<em>Overview: a Hydra entry point resolves configs and builds policy plus frozen reference; three dataset loaders normalize their corpora into one dict schema; tokenization builds left-padded chosen/rejected pairs; the loss function branches over DPO, IPO, and cDPO; and three trainers share the concatenated forward while diverging in sharding strategy.</em>

Reading the overview from left to right:

- [train.py](https://github.com/eric-mitchell/direct-preference-optimization/blob/main/train.py) is the 118-line entry point: the hydra main at line 48 validates the composed config with missing_keys at 55, builds the policy at 83-84 and disables its dropout at 85, builds a frozen reference model at 87-92 for DPO and IPO, loads SFT archives at 96-103, and spawns one worker per GPU with mp.spawn at 111.
- [config/config.yaml](https://github.com/eric-mitchell/direct-preference-optimization/blob/main/config/config.yaml) holds the defaults - learning rate 5e-7 with RMSprop, 150 warmup steps, max_length 512 against max_prompt_length 256, batch_size 4, eval_every 20000, and trainer: BasicTrainer - while the loss and model subtrees override it per experiment.
- [preference_datasets.py](https://github.com/eric-mitchell/direct-preference-optimization/blob/main/preference_datasets.py), 386 lines, normalizes three corpora into one schema: get_hh at line 120 for Anthropic/hh-rlhf, get_shp at 85 for stanfordnlp/SHP, and get_se at 46 for StackExchange preferences, each yielding per-prompt dicts of responses, pairs, and sft_target enforced by an assert at 174.
- [tokenize_batch_element](https://github.com/eric-mitchell/direct-preference-optimization/blob/main/preference_datasets.py#L214) turns a pair into tensors: EOS asserts at 229-231 and appends at 233-237, keep_start/keep_end truncation at 242-248, and prompt labels masked to -100 at 258-261.
- [get_batch_iterator](https://github.com/eric-mitchell/direct-preference-optimization/blob/main/preference_datasets.py#L280) is the reproducible DataLoader factory: it shuffles under TemporarilySeededRandom at 315-316 and defaults to keep_end truncation for hh-rlhf at 319.
- [preference_loss](https://github.com/eric-mitchell/direct-preference-optimization/blob/main/trainers.py#L45) is the whole paper in 40 lines: the policy-vs-reference log-ratio gap at lines 70-76, an IPO branch at 78-79, a label-smoothed cDPO branch at 81-82, and the implicit reward margins returned at 84-85.
- [_get_batch_logps](https://github.com/eric-mitchell/direct-preference-optimization/blob/main/trainers.py#L90) converts one forward pass into per-sequence log probabilities - shift at 103, padding mask at 105, gather of log-softmax at 110, token-average at 112-115 - while concatenated_inputs at 118 stacks chosen and rejected rows into a single padded batch.
- The trainer trio: [BasicTrainer](https://github.com/eric-mitchell/direct-preference-optimization/blob/main/trainers.py#L145) runs one process over a naively split model, [FSDPTrainer](https://github.com/eric-mitchell/direct-preference-optimization/blob/main/trainers.py#L430) adds full sharding with activation checkpointing, and [TensorParallelTrainer](https://github.com/eric-mitchell/direct-preference-optimization/blob/main/trainers.py#L523) shards every linear layer through the tensor-parallel package.

## Why You Need This

Every alignment toolbox this series has toured ships a DPO mode, and all of them implement the same closed-form loss that lives in this repo's 545-line trainers.py. [TRL](/trl/) turned DPOTrainer into a one-liner, [OpenRLHF](/openrlhf/) and [LMFlow](/lmflow/) list it among their aligner family, and [ms-swift](/ms-swift/) wires it into dozens of model types - yet the mathematics they configure from YAML is exactly what preference_loss at line 45 writes out: beta, label smoothing, reference-free mode, and the IPO variant are four branches of one function, not four separate systems. If you have ever wondered what actually changes when you flip loss=dpo to loss=ipo, this file answers it in forty lines you can hold in your head.

The repo also teaches the unglamorous mechanics that papers skip and production frameworks hide behind abstractions: the left-padding collate trick that makes right-padded generations scoreable, the EOS-append asserts that catch silent tokenizer bugs before they poison training, the -100 label masking that [stanford-alpaca](/stanford-alpaca/) uses the same way, and the concatenated forward that scores chosen and rejected responses in a single pass - which the comment at trainers.py lines 213-219 notes is faster under FSDP, the same sharding approach [axolotl](/axolotl/) and [deepspeed](/deepspeed/) users run alongside [peft](/peft/) adapters and [accelerate](/accelerate/) launchers. These details are precisely where first attempts at custom preference-training code go wrong, and here each one occupies a handful of auditable lines.

Finally, it is the cleanest available reference for the two-stage recipe - supervised fine-tuning to get the policy on-distribution, then preference learning from the SFT archive - that every downstream toolbox reproduces, from [llama-factory](/llama-factory/) stage configs to [unsloth](/unsloth/) notebooks. Because the repo pins exact dependency versions and documents measured runtimes (about 1h30 for SFT and 2h45 for DPO on Pythia 2.8B across four A100s), it doubles as a reproducibility yardstick in a field where [lm-evaluation-harness](/lm-evaluation-harness/) scores are only as meaningful as the recipe that produced the checkpoint.

<div style="text-align:center; margin: 20px 0;">
  <img src="https://pyshine.com/assets/img/diagrams/direct-preference-optimization/eric-mitchell-direct-preference-optimization-architecture.svg" alt="Direct Preference Optimization detail architecture diagram" style="max-width:100%; height:auto;">
</div>
<em>Detail: loaders converge on the {responses, pairs, sft_target} schema; tokenization and collation build left-padded, label-masked pair batches; preference_loss branches over DPO, IPO, and cDPO; and the three trainers share the concatenated forward while diverging in distribution strategy.</em>

The data half of the repo is a study in normalization. get_hh at line 120 loads Anthropic/hh-rlhf at 142 and splits each chosen/rejected transcript with split_prompt_and_responses at 145, leaning on extract_anthropic_prompt at 14, which rfinds the last "\n\nAssistant:" marker at 17. get_shp at line 85 loads stanfordnlp/SHP at 92 and keeps only candidate pairs whose score ratio is at least 2 (lines 104-106), orienting each pair from labels==1 at 109. get_se at line 46 loads HuggingFaceH4/stack-exchange-preferences at 52, strips HTML with strip_html_tags at 22 - carefully preserving code blocks at lines 34-38 - and enumerates every answer pair ordered by pm_score at 74-77. Each prompt's sft_target is its best response: the chosen reply for hh (line 158), the top-scored answer for shp and se (lines 81 and 114), so the same loaders double as SFT data sources. get_dataset at 163 dispatches by name and hard-asserts the three-key schema at 174.

Tokenization is where pairs become tensors, and tokenize_batch_element at line 214 guards every step. It asserts the chosen and rejected prompts are identical at 229-231, appends the EOS token to both responses at 233-237, truncates with keep_start or keep_end modes at 242-248 (responses get their own pass at 251-253), and masks prompt labels to -100 at 258-261 so the loss only flows through completions. get_collate_fn at 180 implements the left-padding trick: prompt tensors are reversed at line 192, padded, then flipped back at 205-206 so padding sits on the left side of the prompt - the layout autoregressive scoring needs when batch rows must share a right edge. Attention masks pad with 0 at 200 and labels with -100 at 198. get_batch_iterator at 280 seeds each epoch's shuffle inside TemporarilySeededRandom from utils.py at line 156, invoked at 315-316 so the permutation is reproducible without leaking the seed into global state, picks keep_end truncation for hh at 319, and serves either preference pairs or sft-only batches at 341-353.

The loss function reads like the paper's equation section made executable. preference_loss at line 45 takes four log-probability vectors; lines 70-71 form the policy and reference log-ratios, line 76 subtracts them into logits, and the reference_free flag at 73-74 zeroes the reference term for ablations. Plain DPO is negative logsigmoid of beta times logits; the IPO branch at 78-79 swaps the logsigmoid for a squared term, (logits - 1/(2*beta))**2, citing arXiv 2310.12036; the cDPO branch at 81-82 mixes two logsigmoid terms weighted by label_smoothing, citing the conservativeness note hosted at ericmitchell.ai/cdpo.pdf and equation 7 of the DPO paper. Lines 84-85 return the implicit chosen and rejected rewards so training logs can track margins and accuracy. Feeding it, _get_batch_logps at line 90 shifts labels at 103, builds the padding mask at 105, gathers token log-probabilities from log_softmax at 110, and averages per sequence at 112-115. concatenated_inputs at line 118 pads chosen and rejected rows to a shared max_length at 127 with pad values -100 and 0 at 131, then concatenates them at 138-141 - doubling the batch once instead of running two forward passes.

BasicTrainer at line 145 is the single-process reference loop. Its train method at 272 builds the optimizer with getattr(torch.optim, self.config.optimizer) at 276 - RMSprop per the defaults - attaches a warmup LambdaLR at 277, evaluates every 20000 examples at 292, logs sample completions as wandb tables at 299, checkpoints under step-{example_counter} names at 350, accumulates gradients at 360-364, and clips at 369 through clip_gradient at 397. get_batch_samples at 182 wraps FSDP.summon_full_params at 186 so sampling sees unsharded weights. The forward path is concatenated_forward at 210, whose comment at 213-219 explains the single-pass choice is faster for FSDP; get_batch_metrics at 223 branches on the loss name at 229-242 and reports reward_accuracies at 244. save at 415 writes policy.pt, optimizer.pt, and scheduler.pt at 419-427.

FSDPTrainer at line 430 is where the same loop meets distributed state. It asserts a block_name at 439 and builds transformer_auto_wrap_policy at 441-442 so each decoder layer shards independently; it selects ShardingStrategy.FULL_SHARD at 446, CPUOffload(offload_params=False) at 447, and BackwardPrefetch.BACKWARD_PRE at 448; MixedPrecision at 457-459 honors the config dtypes; activation checkpointing wraps each block at 461-484 with CheckpointImpl.NO_REENTRANT at 476. The reference model gets its own FSDP wrapper at 486-488, gradients clip with the FSDP-native norm at 493-495, and save overrides the base to emit FULL_STATE_DICT checkpoints rank-0-only at 497-520. TensorParallelTrainer at 523 is the minimal sibling: it hands the model to tp.tensor_parallel at 533 and inherits everything else. The glue lives in train.py: worker_main at 21 picks the trainer class by name at 40, raises RLIMIT_NOFILE at 108-110 before mp.spawn at 111, and the archive loader at 96-103 restores SFT weights into both policy and reference, which may hold different dtypes since reference_dtype is resolved separately at 87-92.

The config tree itself is a Hydra composition lesson. config/config.yaml fixes trainer-agnostic defaults - one epoch, eval_every 20000, RMSprop, seed - while config/loss/dpo.yaml declares beta with Hydra's required-key placeholder, meaning the launch will refuse to start until you choose a beta on the command line; the README suggests 0.1 to 0.5 as a starting range. config/loss/sft.yaml omits the reference-model semantics entirely, and config/model/pythia69.yaml points at EleutherAI/pythia-6.9b, names GPTNeoXLayer as the FSDP wrap target, and splits dtypes - float32 for the policy, float16 for the reference - halving reference memory at zero code cost.

## From Install to a Preference-Tuned Pythia

The README's complete example trains Pythia 2.8B on Anthropic-HH (commands abridged):

```sh
git clone https://github.com/eric-mitchell/direct-preference-optimization
cd direct-preference-optimization
python3 -m venv env && source env/bin/activate
pip install -r requirements.txt
# Stage 1 - SFT (~1h30 on 4x80GB A100s)
python -u train.py model=pythia28 datasets=[hh] loss=sft exp_name=anthropic_dpo_pythia28 \
  gradient_accumulation_steps=2 batch_size=64 eval_batch_size=32 \
  trainer=FSDPTrainer sample_during_eval=false model.fsdp_policy_mp=bfloat16
# Stage 2 - DPO from the SFT archive (~2h45 on the same machine)
python -u train.py model=pythia28 datasets=[hh] loss=dpo loss.beta=0.1 \
  exp_name=anthropic_dpo_pythia28 gradient_accumulation_steps=2 batch_size=64 eval_batch_size=32 \
  trainer=FSDPTrainer sample_during_eval=false model.fsdp_policy_mp=bfloat16 \
  model.archive=/path/to/archive/from/sft/LATEST/policy.pt
```

The LATEST directory under the SFT run holds the final weights, and the DPO command differs from the SFT one by exactly three flags - loss, beta, and the archive path - which is the recipe in miniature.

Honest limits: the code pins torch 2.0.1 and transformers 4.29.2, and both FSDP internals and generation APIs have moved since; sampling during evaluation is slow enough under sharding that the README recommends sample_during_eval=false; TensorParallelTrainer is flagged experimental; eval_every counts examples rather than optimizer steps; the loaders materialize the full preference dictionary in memory before batching; and FSDP runs need the file-descriptor limit raised (ulimit -n 64000). Prompts are raw "\n\nHuman: ...\n\nAssistant:" strings rather than tokenizer chat templates, so adapting newer instruct models means writing your own prompt formatting in preference_datasets.py.

<script type="application/ld+json">
{
  "@context": "https://schema.org",
  "@type": "BlogPosting",
  "headline": "Your Language Model Is Secretly a Reward Model - Inside eric-mitchell/direct-preference-optimization",
  "description": "A source tour of the DPO reference implementation: three preference dataset loaders, the concatenated log-probability trick, DPO/IPO/cDPO losses in one 40-line function, and trainer classes from single GPU to FSDP.",
  "image": "https://pyshine.com/assets/img/diagrams/direct-preference-optimization/eric-mitchell-direct-preference-optimization-overview-architecture.svg",
  "author": {"@type": "Person", "name": "PyShine"},
  "publisher": {"@type": "Organization", "name": "PyShine"},
  "datePublished": "2026-10-17",
  "dateModified": "2026-10-17",
  "mainEntityOfPage": "https://pyshine.com/direct-preference-optimization/",
  "keywords": "DPO, direct preference optimization, alignment, Pythia, FSDP"
}
</script>
