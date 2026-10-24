---
layout: post
title: "Stable Diffusion: A Source Tour of the Original Latent Diffusion Code"
description: "Walk through the CompVis source: the latent diffusion objective, the DDIM sampler, the VAE autoencoder, cross-attention text conditioning, and the txt2img entry."
date: 2026-10-24
header-img: "img/post-bg.jpg"
permalink: /stable-diffusion/
featured-img: "ai-coding-frameworks/ai-coding-frameworks"
image: https://pyshine.com/assets/img/diagrams/stable-diffusion/compvis-stable-diffusion-overview-architecture.svg
tags: [LLM, Python, Open Source]
categories: [AI Inference]
keywords: Stable Diffusion, latent diffusion, DDIM sampler, VAE, CLIP text encoder, PyTorch
author: "PyShine"
---

Every image generator you have used in the last three years — the browser apps, the commercial APIs, the open-source forks — descends from one small research repository: CompVis/stable-diffusion. In August 2022 it shipped the weights and the reference code for Stable Diffusion v1, the first text-to-image model good enough, and light enough, to run on a consumer GPU. The paper called the method latent diffusion; the code expresses it in roughly eight thousand lines of PyTorch spread across a `ldm` package. Unlike the giant application built on top of it — which we toured in the [stable-diffusion-webui](/stable-diffusion-webui/) post — this repository is almost pure model: a U-Net, a variational autoencoder, a frozen text encoder, and two samplers, with no interface layer to hide the mechanics.

That directness is exactly why the source is still worth reading. Everything that later became folklore — the 64x64 latent space, the seven-and-a-half guidance scale, the fifty sampling steps, the CLIP ViT-L/14 text tower — is visible here as ordinary Python. The training objective lives in one method, the sampling loop in another, and the trick that made it cheap (running diffusion on a compressed latent instead of raw pixels) is a four-line encode call at the top of the loss function. The project is now archived and its dependency pins have aged, but the architecture it documented became the template that [diffusers](/diffusers/) industrialized and that every subsequent model inherited.

This post walks the actual files: `ldm/models/diffusion/ddpm.py` for the objective, `ldm/models/diffusion/ddim.py` and `plms.py` for sampling, `ldm/models/autoencoder.py` with its encoder and decoder for the first stage, `ldm/modules/attention.py` for cross-attention, and `scripts/txt2img.py` as the entry point. All line references were checked against the current `main` branch. As always in this series, what follows is an educational tour of published source code.

<div style="overflow-x:auto;"><img src="https://pyshine.com/assets/img/diagrams/stable-diffusion/compvis-stable-diffusion-overview-architecture.svg" alt="Stable Diffusion overview architecture diagram showing the txt2img entry point, the DDIM and PLMS samplers, the latent diffusion core with the UNet, the cross-attention modules, the VAE first stage, and the frozen CLIP text encoder" style="min-width:900px;"></div>

<p><em>The overview shows the full text-to-image path: a CLI entry builds the model tree from a YAML config, samplers drive the UNet through latent space, cross-attention injects CLIP text embeddings, and the VAE shuttles between pixels and latents.</em></p>

Reading the overview from left to right:

- The [main entry, txt2img.py L98](https://github.com/CompVis/stable-diffusion/blob/main/scripts/txt2img.py#L98) parses prompts, seeds, and dimensions, then owns the sampling loop that tiles finished images into a grid.
- The [config loader, L49](https://github.com/CompVis/stable-diffusion/blob/main/scripts/txt2img.py#L49) torch-loads the checkpoint and rebuilds the model from its YAML description rather than from pickled objects.
- The [LatentDiffusion class, L424](https://github.com/CompVis/stable-diffusion/blob/main/ldm/models/diffusion/ddpm.py#L424) is the centerpiece: it holds the first stage, the conditioning encoders, and the denoising UNet in one PyTorch Lightning module.
- The [DDIMSampler.sample, L57](https://github.com/CompVis/stable-diffusion/blob/main/ldm/models/diffusion/ddim.py#L57) prepares the noise schedule buffers and runs the deterministic sampling loop over timesteps.
- The [PLMSSampler, L11](https://github.com/CompVis/stable-diffusion/blob/main/ldm/models/diffusion/plms.py#L11) offers the pseudo-linear multistep alternative that the reference script enables with `--plms`.
- The [UNetModel, L413](https://github.com/CompVis/stable-diffusion/blob/main/ldm/modules/diffusionmodules/openaimodel.py#L413) is the 860M-parameter denoiser that predicts the noise of a compressed latent.
- The [AutoencoderKL, L285](https://github.com/CompVis/stable-diffusion/blob/main/ldm/models/autoencoder.py#L285) is the KL-regularized first stage compressing 512x512x3 images into 64x64x4 latents.
- The [FrozenCLIPEmbedder, L137](https://github.com/CompVis/stable-diffusion/blob/main/ldm/modules/encoders/modules.py#L137) turns prompt text into 77x768 embedding sequences, and the [CrossAttention, L152](https://github.com/CompVis/stable-diffusion/blob/main/ldm/modules/attention.py#L152) is where those embeddings steer the UNet.

## The First Stage: Compressing Pixels into Latents

The single most important design decision in this codebase is that diffusion does not happen on pixels. The paper's authors trained a variational autoencoder to squeeze each 512x512 RGB image into a 4x64x64 tensor — an eight-times-per-side downsample — then ran the diffusion model entirely in that compressed space. Everything else in the system orbits this choice: it is why a 4 GB GPU could generate images in 2022, and why later work on speed attacked the first stage rather than the UNet.

The autoencoder itself is conventional and clean. The [Encoder, L368](https://github.com/CompVis/stable-diffusion/blob/main/ldm/modules/diffusionmodules/model.py#L368) is a stack of [ResnetBlocks, L82](https://github.com/CompVis/stable-diffusion/blob/main/ldm/modules/diffusionmodules/model.py#L82) and [AttnBlocks, L150](https://github.com/CompVis/stable-diffusion/blob/main/ldm/modules/diffusionmodules/model.py#L150) that progressively halve the spatial resolution and emit two tensors, mean and logvar; sampling from that diagonal Gaussian gives the latent. The [Decoder, L462](https://github.com/CompVis/stable-diffusion/blob/main/ldm/modules/diffusionmodules/model.py#L462) mirrors the path back to pixels. The wrapper class hides the plumbing: [AutoencoderKL.encode, L324](https://github.com/CompVis/stable-diffusion/blob/main/ldm/models/autoencoder.py#L324) computes the moments and samples the posterior, while [AutoencoderKL.decode, L330](https://github.com/CompVis/stable-diffusion/blob/main/ldm/models/autoencoder.py#L330) multiplies by a learned `scale_factor` before reconstruction — that single scalar matters more than it looks, because it normalizes the latent variance so the diffusion noise schedule behaves consistently. The repository also ships [VQModel, L14](https://github.com/CompVis/stable-diffusion/blob/main/ldm/models/autoencoder.py#L14), the quantized-codebook variant from the original latent-diffusion paper, but Stable Diffusion v1's released weights all use the KL version.

## The Training Objective: p_losses and the Simplified Loss

The diffusion machinery lives in `ldm/models/diffusion/ddpm.py`, a file that starts with the pixel-space DDPM base class and then extends it twice. The base [DDPM class, L44](https://github.com/CompVis/stable-diffusion/blob/main/ldm/models/diffusion/ddpm.py#L44) holds the beta schedule, the [q_sample, L274](https://github.com/CompVis/stable-diffusion/blob/main/ldm/models/diffusion/ddpm.py#L274) forward-noising helper, and an ancestral [p_sample_loop, L253](https://github.com/CompVis/stable-diffusion/blob/main/ldm/models/diffusion/ddpm.py#L253). [LatentDiffusion, L424](https://github.com/CompVis/stable-diffusion/blob/main/ldm/models/diffusion/ddpm.py#L424) subclasses it and quietly swaps the domain: where the base class noises and denoises images, the subclass does it to VAE latents.

The training step is the clearest expression of the method. In [LatentDiffusion.p_losses, L1012](https://github.com/CompVis/stable-diffusion/blob/main/ldm/models/diffusion/ddpm.py#L1012), a batch of images is first encoded by the frozen first stage, then a random timestep is drawn, the latent is noised with `q_sample`, and the UNet is asked to predict the noise that was added. The loss is simply the MSE between the predicted and actual noise — the "simplified objective" from the DDPM paper. A few lines further down, an optional decoder loss lets the model fine-tune the first stage jointly. During sampling, [apply_model, L891](https://github.com/CompVis/stable-diffusion/blob/main/ldm/models/diffusion/ddpm.py#L891) is the funnel every sampler calls: it concatenates the conditional and unconditional batches so one forward pass serves both, then extrapolates between them — that linear extrapolation with a scale of 7.5 is classifier-free guidance in its entirety.

## Cross-Attention Is How Prompts Enter

Text conditioning in this codebase is not a hack bolted onto the UNet; it is wired into the residual blocks themselves. The [SpatialTransformer, L218](https://github.com/CompVis/stable-diffusion/blob/main/ldm/modules/attention.py#L218) is inserted at chosen resolutions of the UNet — the YAML config sets `use_spatial_transformer` and a `context_dim` of 768 — and inside, each [BasicTransformerBlock, L196](https://github.com/CompVis/stable-diffusion/blob/main/ldm/modules/attention.py#L196) applies self-attention over the image's spatial positions, then cross-attention against the text context, then a feed-forward layer. The [CrossAttention.forward, L170](https://github.com/CompVis/stable-diffusion/blob/main/ldm/modules/attention.py#L170) is eleven lines: queries come from image features, keys and values from the 77x768 text embedding, and softmax-weighted retrieval is all the "prompt understanding" there is.

The context itself is produced by the frozen CLIP text tower. [FrozenCLIPEmbedder, L137](https://github.com/CompVis/stable-diffusion/blob/main/ldm/modules/encoders/modules.py#L137) downloads `openai/clip-vit-large-patch14` from Hugging Face, forces it into eval mode in [freeze, L147](https://github.com/CompVis/stable-diffusion/blob/main/ldm/modules/encoders/modules.py#L147), and its [encode, L161](https://github.com/CompVis/stable-diffusion/blob/main/ldm/modules/encoders/modules.py#L161) tokenizes prompts and returns last hidden states. Frozen means no gradients flow into it during training — text understanding is treated as a fixed, pretrained perception organ. If you want to go deeper on the encoder itself, we covered the model family in the [CLIP](/clip/) tour, and the open-source reimplementation in [OpenCLIP](/open-clip/); the tokenizer side is the same family of tools as [transformers](/transformers/).

## Inside the UNet Backbone and the Samplers

The UNet follows the OpenAI guided-diffusion blueprint that the community now simply calls "the SD UNet". [UNetModel, L413](https://github.com/CompVis/stable-diffusion/blob/main/ldm/modules/diffusionmodules/openaimodel.py#L413) builds three block stacks — input, middle, output — wrapped in [TimestepEmbedSequential, L74](https://github.com/CompVis/stable-diffusion/blob/main/ldm/modules/diffusionmodules/openaimodel.py#L74), a small dispatch wrapper that hands each block the right arguments: residuals get the timestep embedding, transformers get the text context. Timesteps enter as sinusoidal vectors from [timestep_embedding, L151](https://github.com/CompVis/stable-diffusion/blob/main/ldm/modules/diffusionmodules/util.py#L151), expanded by a two-layer SiLU MLP at [time_embed, L507](https://github.com/CompVis/stable-diffusion/blob/main/ldm/modules/diffusionmodules/openaimodel.py#L507). The [forward pass, L710](https://github.com/CompVis/stable-diffusion/blob/main/ldm/modules/diffusionmodules/openaimodel.py#L710) pushes data down the input stack while saving skip connections, runs the middle block at the bottleneck, then climbs the output stack fusing those skips — a design [PyTorch](/pytorch/) makes pleasantly explicit because nothing is hidden behind framework magic.

Sampling is two files of elegant arithmetic. The DDIM sampler computes a fresh schedule in [make_schedule, L25](https://github.com/CompVis/stable-diffusion/blob/main/ldm/models/diffusion/ddim.py#L25), then [ddim_sampling, L114](https://github.com/CompVis/stable-diffusion/blob/main/ldm/models/diffusion/ddim.py#L114) walks the reduced timestep list calling [p_sample_ddim, L166](https://github.com/CompVis/stable-diffusion/blob/main/ldm/models/diffusion/ddim.py#L166), which predicts x0 from the UNet's noise estimate and interpolates with the direction term — eta zero gives the deterministic, forty-step-capable sampler that made fast generation practical. The same file's [stochastic_encode, L207](https://github.com/CompVis/stable-diffusion/blob/main/ldm/models/diffusion/ddim.py#L207) powers img2img by noising an input latent to an intermediate step. PLMS, in `plms.py`, reuses the same structure but tracks previous noise predictions for a higher-order update in [p_sample_plms, L173](https://github.com/CompVis/stable-diffusion/blob/main/ldm/models/diffusion/plms.py#L173).

<div style="overflow-x:auto;"><img src="https://pyshine.com/assets/img/diagrams/stable-diffusion/compvis-stable-diffusion-architecture.svg" alt="Stable Diffusion detailed architecture diagram covering the txt2img entry with config loading and the sampling loop, the DDIM and PLMS samplers with their step functions, the latent diffusion core with q_sample and p_losses, the UNet backbone with timestep embedding and residual blocks, cross-attention modules, the VAE encoder and decoder stacks, and the frozen CLIP text encoder" style="min-width:900px;"></div>

<p><em>The detail graph separates seven bands — entry, samplers, diffusion core, UNet backbone, attention, VAE first stage, and text conditioning — with the call edges between them.</em></p>

## From Install to Your First Model

The README's path still works if you accept that the pins are from 2022. Create the environment from the repo's `environment.yaml` or update an existing latent-diffusion one:

```bash
conda env create -f environment.yaml
conda activate ldm
pip install transformers==4.19.2 diffusers invisible-watermark
pip install -e .
```

Then obtain the weights. `sd-v1-4.ckpt` is published on the CompVis Hugging Face organization page, which requires you to accept the CreativeML OpenRAIL-M license while logged in — effectively a Hugging Face token gate — before the download link appears. Link the checkpoint into the expected location and run the reference script:

```bash
mkdir -p models/ldm/stable-diffusion-v1/
ln -s <path/to/model.ckpt> models/ldm/stable-diffusion-v1/model.ckpt
python scripts/txt2img.py --prompt "a photograph of an astronaut riding a horse" --plms
```

That command renders 512x512 images in fifty PLMS steps at guidance scale 7.5 — the defaults the README documents — writing PNGs with an invisible watermark under `outputs/txt2img-samples`. Useful flags from the same file: `--ddim_steps` to trade speed for quality, `--scale` for guidance strength, `--n_samples` for batch size, and `--seed` for reproducibility. On modern PyTorch versions the pinned dependencies can fight the system CUDA, so the smoother route today is loading the same architecture through diffusers and only returning here to read the code.

## Honest Limits

Read this repository as a historical artifact, not a running system. It is explicitly archived: the last real commits are from 2022, issues are closed, and the README itself concedes that community development has moved on. The code targets the PyTorch 1.x era — `torch.load` without `weights_only`, PyTorch Lightning pinned near version 1.4, and attention written long before `scaled_dot_product_attention` — so installing it verbatim on current CUDA stacks is an archaeology exercise. The CLIP weights download through the Hugging Face hub with a license gate, and the checkpoint format it loads predates the safetensors push that the ecosystem adopted for supply-chain safety.

None of that diminishes its value as a reference. The training objective, the guidance splice, and the first-stage compression are stated here more plainly than anywhere else, and every subsequent text-to-image codebase is legible as a set of edits to this one. Active development lives in the community ecosystem: the [stable-diffusion-webui](/stable-diffusion-webui/) application for end users, [diffusers](/diffusers/) for library-grade usage, and the model families that followed. Treat this repo as the primary source — the place where the ideas first appeared in code you can trace line by line.

<script type="application/ld+json">
{
  "@context": "https://schema.org",
  "@type": "TechArticle",
  "headline": "Stable Diffusion: A Source Tour of the Original Latent Diffusion Code",
  "description": "Walk through the CompVis source: the latent diffusion objective, the DDIM sampler, the VAE autoencoder, cross-attention text conditioning, and the txt2img entry.",
  "datePublished": "2026-10-24",
  "author": {"@type": "Person", "name": "PyShine"},
  "keywords": "Stable Diffusion, latent diffusion, DDIM sampler, VAE, CLIP text encoder, PyTorch",
  "about": {"@type": "SoftwareSourceCode", "name": "CompVis/stable-diffusion", "programmingLanguage": "Python"}
}
</script>
