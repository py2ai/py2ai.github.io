---
layout: post
title: "The 52K Recipe That Started a Thousand Fine-Tunes - Inside tatsu-lab/stanford_alpaca"
description: "A source tour of Stanford Alpaca: batch self-instruct data generation, the 222-line fine-tuning script, FSDP and DeepSpeed recipes, and the weight-diff release trick."
date: 2026-10-17 00:00:00 +0000
header-img: "img/post-bg.jpg"
permalink: /stanford-alpaca/
featured-img: ai-coding-frameworks/ai-coding-frameworks
image: https://pyshine.com/assets/img/diagrams/stanford-alpaca/tatsu-lab-stanford-alpaca-overview-architecture.svg
tags: [stanford-alpaca, instruction-tuning, Self-Instruct, FSDP, DeepSpeed]
categories: [AI Training]
keywords: "Stanford Alpaca source code, alpaca_data.json, self-instruct batch decoding, LLaMA fine-tuning, weight diff, FSDP full shard"
author: "PyShine"
---

Some repositories matter because of their engineering, and some because of what they proved. stanford_alpaca, from five Stanford grad students advised by Tatsunori Hashimoto, is the second kind: fourteen files that showed a 7B LLaMA model fine-tuned on 52K machine-generated instruction examples could behave similarly to text-davinci-003 on the Self-Instruct evaluation suite. The whole data generation cost less than $500, the fine-tuning script is 222 lines, and the release pattern - code under Apache 2.0, data and weight diffs under CC BY NC 4.0 - became the template an entire ecosystem copied. Every toolbox we have toured in this series, from [LLaMA-Factory](/llama-factory/) to [axolotl](/axolotl/) to [ms-swift](/ms-swift/), descends from the recipe in this folder.

As always in this series, what follows is an educational tour of published source code.

<div style="text-align:center; margin: 20px 0;">
  <img src="https://pyshine.com/assets/img/diagrams/stanford-alpaca/tatsu-lab-stanford-alpaca-overview-architecture.svg" alt="Stanford Alpaca overview architecture diagram" style="max-width:100%; height:auto;">
</div>
<em>Overview: seed tasks and the generation template feed the self-instruct loop, whose API client writes the 52K dataset; the fine-tuning script consumes it with FSDP or DeepSpeed, and the weight-diff tool closes the release loop.</em>

Reading the overview from left to right:

- [generate_instruction.py](https://github.com/tatsu-lab/stanford_alpaca/blob/main/generate_instruction.py) is the 217-line generation loop: encode_prompt at line 27, post_process_gpt3_response at 44, and generate_instruction_following_data at 110, with a rouge-L similarity threshold of 0.7 at line 196.
- [prompt.txt](https://github.com/tatsu-lab/stanford_alpaca/blob/main/prompt.txt) is the generation template - seven numbered requirements asking for 20 diverse task instructions per completion, including the famous accidental duplicate numbering the authors flag for future users.
- [seed_tasks.jsonl](https://github.com/tatsu-lab/stanford_alpaca/blob/main/seed_tasks.jsonl) holds 175 human-written seeds, from seed_task_0's breakfast suggestion onward, each with an instruction and one instance.
- [utils.py](https://github.com/tatsu-lab/stanford_alpaca/blob/main/utils.py) is the API client: OpenAIDecodingArguments at line 25, openai_completion at 39 with retry-and-shrink backoff at lines 116-120, and the jdump/jload JSON helpers at 148 and 168.
- [alpaca_data.json](https://github.com/tatsu-lab/stanford_alpaca/blob/main/alpaca_data.json) is the artifact itself: 52,002 entries with instruction, input, and output fields, 20,679 of them carrying a non-empty input.
- [train.py](https://github.com/tatsu-lab/stanford_alpaca/blob/main/train.py) is the entire fine-tuning story in 222 lines: PROMPT_DICT at 31, smart_tokenizer_and_embedding_resize at 65, preprocess at 112, SupervisedDataset at 127, DataCollatorForSupervisedDataset at 157, and train at 182.
- [configs/default_offload_opt_param.json](https://github.com/tatsu-lab/stanford_alpaca/blob/main/configs/default_offload_opt_param.json) is the DeepSpeed escape hatch - ZeRO stage 3 with CPU offload for both optimizer states and parameters.
- [weight_diff.py](https://github.com/tatsu-lab/stanford_alpaca/blob/main/weight_diff.py) is the release mechanism: make_diff at line 25, recover at 71, and a naive-but-effective integrity check at 129 that asserts a checkpoint-wide sum of 50637.1836.

## Why You Need This

Read this repo to see how little a landmark result takes. There is no framework here, no plugin registry, no YAML - just four Python files and one JSON dataset. Where [LLaMA-Factory](/llama-factory/) routes dozens of model families through a template registry and [OpenRLHF](/openrlhf/) distributes a PPO controller over Ray, Alpaca formats 52K strings with two templates, masks the prompt tokens, and hands everything to the stock Hugging Face Trainer. The hyperparameters in the README - global batch 128, learning rate 2e-5 for 7B and 1e-5 for 13B, three epochs, max length 512 - fit in a five-row table that a thousand follow-up papers then cited or tweaked.

The data generation half is the more interesting source lesson: it is Self-Instruct with batch decoding. Instead of one instruction per API call, encode_prompt at line 27 stuffs three sampled seed examples into a template that requests twenty at once, cutting cost by an order of magnitude, and post_process_gpt3_response at line 44 parses all twenty back out with a regex. The filtering gauntlet in between - length limits, a keyword blacklist, and rouge-L deduplication against everything generated so far - is a compact masterclass in keeping a synthetic dataset diverse, and it directly prefigures the synthetic-data machinery inside later toolboxes like [ms-swift](/ms-swift/).

Finally, the weight-diff mechanism is a period piece worth understanding: in March 2023 the LLaMA license forbade redistributing weights, so the repo published only the difference between Alpaca-7B and LLaMA-7B, recoverable in minutes by anyone with the base checkpoint. weight_diff.py makes the whole chain auditable - subtract raw from tuned, add raw back, verify with a checksum - and the licensing split (Apache code, CC BY NC data, CC BY NC diffs) is spelled out in three separate license files. The constraint is gone now, but the pattern of shipping provable artifacts rather than promises shaped how [peft](/peft/) adapters, LoRA deltas, and model diffs get shared to this day.

<div style="text-align:center; margin: 20px 0;">
  <img src="https://pyshine.com/assets/img/diagrams/stanford-alpaca/tatsu-lab-stanford-alpaca-architecture.svg" alt="Stanford Alpaca detailed architecture diagram" style="max-width:100%; height:auto;">
</div>
<em>Detailed view: the self-instruct loop from template to filters, the batched API client with backoff, the dataset, every stage of the 222-line fine-tuning script, the sharding recipes, and the weight-diff tooling.</em>

The generation loop rewards close reading. generate_instruction_following_data at line 110 of [generate_instruction.py](https://github.com/tatsu-lab/stanford_alpaca/blob/main/generate_instruction.py) loads the 175 seed tasks at line 121, resumes from regen.json if one exists at line 132, and then loops: sample three seeds at line 156, encode them via encode_prompt at 27 - which appends a numbered "21. Instruction:" cue after the examples - and fire request_batch_size prompts at the completions endpoint through utils.openai_completion at line 167 with max_tokens hard-coded to 3072 at line 162 and stop sequences tuned to catch the twentieth item. post_process_gpt3_response at line 44 splits the response on "###" markers, discards a truncated final item when the finish reason is "length" at line 52, and runs the gauntlet: instructions must be 4 to 150 words at line 64, must avoid the eighteen-word blacklist spanning lines 67-86 (image, graph, draw, plot, video, and friends - tasks a text model cannot do), must not start with "Write a program" at line 94, punctuation at 97, or a non-ASCII character at 100. Survivors then face the dedup gate: rouge-L F-measure against every prior instruction, computed in a multiprocessing pool at line 187, with anything scoring above 0.7 dropped at line 196. Each keeper is stored with its ten most similar neighbors and an average similarity score at lines 193-201, which is how the dataset release ships its own diversity audit.

The API client in [utils.py](https://github.com/tatsu-lab/stanford_alpaca/blob/main/utils.py) is a period-accurate artifact of the old completions API. OpenAIDecodingArguments at line 25 dataclasses the sampling knobs; openai_completion at line 39 batches prompts at line 87, tags each choice with total token usage at line 110, and - the part still worth copying - handles failures with two distinct branches at lines 113-120: context-length errors shrink max_tokens by twenty percent and retry, everything else sleeps and retries. The logit_bias trick at line 172 of generate_instruction.py, suppressing the end-of-text token so all twenty tasks arrive in one response, shows how much behavior you can steer without touching the model.

The training half of [train.py](https://github.com/tatsu-lab/stanford_alpaca/blob/main/train.py) is deliberately minimal but gets the details right. PROMPT_DICT at line 31 defines the two prompt templates the README quotes; SupervisedDataset at line 127 formats every example, appends the EOS token to each target at line 141, and tokenizes everything up front through preprocess at line 112, which tokenizes concatenated source-plus-target and blanks out the prompt span with IGNORE_INDEX = -100 at line 123 so the loss is computed on responses only. DataCollatorForSupervisedDataset at line 157 right-pads input_ids with the pad token and labels with -100 at lines 164-167, matching the README's note that sequences are right-padded. The most reused function in the file is smart_tokenizer_and_embedding_resize at line 65: when the LLaMA tokenizer lacks pad, bos, or unk tokens, train at line 182 adds them, resizes the embedding matrices, and mean-initializes the new rows from existing embeddings at lines 81-85 - the fix every early LLaMA fine-tune needed, imported by weight_diff.py at its line 21 so the diff math lines up token-for-token.

Distribution is left to recipes rather than code. The README's primary command runs torchrun with FSDP full_shard auto_wrap, wrapping LlamaDecoderLayer on four A100 80G GPUs, with an OPT-6.7B variant swapping in OPTDecoderLayer - and the OOM section does the arithmetic a newcomer needs: 7B parameters times four bytes times four roles is roughly 112GB naive, parameter sharding removes the redundant copies, and CPU offload via the ZeRO-3 config in [configs/default_offload_opt_param.json](https://github.com/tatsu-lab/stanford_alpaca/blob/main/configs/default_offload_opt_param.json) trades runtime for headroom, with LoRA mentioned as the path to about 28GB. Those three rungs - full fine-tune, offload, adapter - are exactly the ladder [deepspeed](/deepspeed/), [accelerate](/accelerate/), and [peft](/peft/) industrialized, and modern toolboxes like [unsloth](/unsloth/) and [axolotl](/axolotl/) still expose the same choices behind friendlier flags.

The weight-diff file closes the loop with unusual transparency. make_diff at line 25 of [weight_diff.py](https://github.com/tatsu-lab/stanford_alpaca/blob/main/weight_diff.py) loads both checkpoints in float32 on one device, subtracts in place with add_ at line 64, and saves the result; recover at line 71 loads the raw LLaMA checkpoint plus the diff, adds it back at line 125, and runs a self-described naive integrity check at line 129 - summing every recovered tensor and asserting the total is 50637.1836 within 1e-2 - before optionally generating a 100-token answer to a built-in instruction at lines 138-148 as a smoke test. The dataset itself is documented in [datasheet.md](https://github.com/tatsu-lab/stanford_alpaca/blob/main/datasheet.md) and [model_card.md](https://github.com/tatsu-lab/stanford_alpaca/blob/main/model_card.md), and the training data's non-commercial license is why the authors kept the live demo rather than the weights. For evaluating instruction-following descendants of this recipe, the natural companion is our tour of [lm-evaluation-harness](/lm-evaluation-harness/); for what came after preference tuning replaced pure SFT, see [openrlhf](/openrlhf/) and [lmflow](/lmflow/).

## From Install to a Fine-Tuned Model

Reproducing the 7B run takes one install and one torchrun command:

```sh
pip install -r requirements.txt
torchrun --nproc_per_node=4 --master_port=<your_random_port> train.py \
    --model_name_or_path <your_path_to_hf_converted_llama_ckpt_and_tokenizer> \
    --data_path ./alpaca_data.json \
    --bf16 True \
    --output_dir <your_output_dir> \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --learning_rate 2e-5 \
    --lr_scheduler_type "cosine" \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --tf32 True
```

Data generation and weight recovery are one-liners with fire:

```sh
python -m generate_instruction generate_instruction_following_data
python weight_diff.py recover --path_raw <raw_dir> --path_diff <diff_dir> --path_tuned <out_dir>
```

Honest limits: the README states the training script is simple and not particularly optimized; the code pins the old-style OpenAI completions API and will not run against current clients unchanged; the dataset is research-only by license; and the authors are explicit that Alpaca was never tuned for safety. But as a reference implementation - one dataset, one script, one diff - it remains the cleanest place to watch instruction tuning happen line by line, before the field's frameworks grew a hundred times around it.

<script type="application/ld+json">
{
  "@context": "https://schema.org",
  "@type": "BlogPosting",
  "headline": "The 52K Recipe That Started a Thousand Fine-Tunes - Inside tatsu-lab/stanford_alpaca",
  "description": "A source tour of Stanford Alpaca: batch self-instruct data generation, the 222-line fine-tuning script, FSDP and DeepSpeed recipes, and the weight-diff release trick.",
  "image": "https://pyshine.com/assets/img/diagrams/stanford-alpaca/tatsu-lab-stanford-alpaca-overview-architecture.svg",
  "author": {"@type": "Person", "name": "PyShine"},
  "publisher": {"@type": "Organization", "name": "PyShine"},
  "datePublished": "2026-10-17",
  "mainEntityOfPage": "https://pyshine.com/stanford-alpaca/"
}
</script>
