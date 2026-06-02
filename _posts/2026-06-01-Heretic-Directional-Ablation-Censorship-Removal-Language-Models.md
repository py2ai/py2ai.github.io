---
layout: post
title: "Heretic - Directional Ablation for Automatic Censorship Removal in Language Models"
date: 2026-06-01 12:00:00 +0800
categories: ai machine-learning llm safety
tags: [directional-ablation, censorship-removal, llm, optuna, abliteration, huggingface, pytorch]
featured-img: ai-coding-frameworks/ai-coding-frameworks
---

## What is Heretic?

Heretic is a fully automatic censorship removal tool for transformer-based language models. It applies **directional ablation** -- a technique also known as "abliteration" -- to strip safety alignment from LLMs without requiring any manual parameter tuning. The tool uses **Optuna TPE-based Bayesian optimization** to automatically find the best ablation parameters, making the entire process hands-off from start to finish.

The project is available as a PyPI package (`heretic-llm`), can process a model in approximately 45 minutes on an RTX 3090, and has spawned over 1,247 community-published models on HuggingFace. Heretic also has a Codeberg mirror and a website at heretics.fun.

> **Key Insight**: Heretic represents a paradigm shift from manual "jailbreak" prompt engineering to systematic, mathematically grounded removal of refusal behavior at the activation level. Instead of tricking the model into compliance, it surgically removes the refusal direction from the model's internal representations.

## How Directional Ablation Works

Directional ablation is grounded in research by Arditi et al. (2024) and further refined by Lai (2025). The core insight is that safety-aligned language models encode a "refusal direction" in their activation space -- a specific vector along which the model distinguishes between requests it should refuse and those it should fulfill.

The process works by:

1. **Generating contrastive pairs** -- prompts the model with both harmful and benign requests
2. **Collecting activations** -- runs forward passes and captures hidden states at a target layer
3. **Computing the difference** -- subtracts benign activations from harmful activations to isolate the refusal signal
4. **Extracting the refusal direction** -- uses PCA or mean-difference to find the principal refusal vector
5. **Ablating the direction** -- projects out the refusal component from the model's weights, making it incapable of refusing

> **Important**: The ablation operation modifies the model's weight matrices directly. By projecting out the refusal direction from the weight matrix at the target layer, the model loses its ability to represent refusal behavior. This is not a prompt-level bypass -- it is a structural modification to the model's computation graph.

## Architecture Overview

![Heretic Architecture](/assets/img/diagrams/heretic/heretic-architecture.svg)

The architecture diagram illustrates Heretic's four-layer design. At the top, **Entry Points** provide both a command-line interface and a PyPI package (`pip install heretic-llm`). The **Core Modules** layer contains the Model Loader (handles HuggingFace model downloading and quantization), the Directional Ablation Engine (implements the Arditi/Lai ablation algorithms), the Optuna TPE Optimizer (automates hyperparameter search), and the Model Output handler. Below that, the **Ablation Pipeline** shows the four-step internal process: Contrastive Pair Generation, Activation Difference Computation, Refusal Direction Extraction, and the Ablation Operation itself. The **Output and Distribution** layer handles saving the modified model locally and auto-registering it on HuggingFace Hub. A GPU Backend node supports CUDA, MPS, and CPU execution.

## The Abliteration Pipeline

![Heretic Pipeline](/assets/img/diagrams/heretic/heretic-pipeline.svg)

The pipeline diagram traces the six-step abliteration flow from input to output. The process begins with the **Input Model** loaded from HuggingFace. Step 1 generates contrastive prompt pairs -- one set of harmful requests and one set of benign counterparts. Step 2 runs forward passes through the model to collect activations at the target layer for both prompt sets. Step 3 computes the activation difference by subtracting benign activations from harmful activations. Step 4 extracts the refusal direction using PCA or mean-difference on the activation differences. Step 5 enters the Optuna TPE optimization loop -- this is where Heretic differentiates itself from manual approaches. The optimizer evaluates different ablation parameters (target layer, scaling factor, projection method) and uses Bayesian optimization to converge on the best configuration. A feedback arrow shows the iterative nature of this search. Step 6 applies the final ablation operation to the model weights, producing the **Output Model** which is then published to HuggingFace Hub.

## Key Features

![Heretic Features](/assets/img/diagrams/heretic/heretic-features.svg)

The features diagram organizes Heretic's capabilities into five clusters:

**Directional Ablation** -- The core technique implements refusal direction detection through contrastive activation analysis, activation steering to identify the refusal vector, and builds on the Arditi et al. 2024 research foundation.

**Optuna TPE Optimizer** -- Bayesian hyperparameter search replaces manual tuning. The Tree-structured Parzen Estimator (TPE) algorithm efficiently explores the parameter space, and the approach is based on Lai 2025 research. This is what makes Heretic "fully automatic."

**Fully Automatic** -- A single command processes the entire pipeline. No post-training or fine-tuning is required. The entire process takes approximately 45 minutes on an RTX 3090 GPU.

**HuggingFace Integration** -- Automatic model registration on HuggingFace Hub after processing. The community has published over 1,247 Heretic-processed models, creating a growing ecosystem of ablated models.

**Multi-Backend Support** -- CUDA for NVIDIA GPUs, MPS for Apple Silicon, and CPU fallback for environments without GPU access.

> **Takeaway**: The combination of directional ablation with Optuna TPE optimization is what sets Heretic apart from earlier manual abliteration approaches. Instead of requiring researchers to manually identify the target layer, scaling factor, and projection method, Heretic automates the entire search process using Bayesian optimization.

## Code Structure

Heretic follows a modern Python `src` layout pattern with four primary code modules:

- **Model loading and quantization** -- Handles downloading from HuggingFace, applying quantization (4-bit, 8-bit), and preparing the model for inference
- **Contrastive pair generation** -- Creates the harmful/benign prompt pairs used to probe the model's refusal behavior
- **Directional ablation engine** -- Implements the core ablation algorithm: activation collection, difference computation, direction extraction, and weight projection
- **Optuna TPE optimizer** -- Wraps the ablation process in an optimization loop that searches over target layers, scaling factors, and projection methods

The project is installable via `pip install heretic-llm` and provides both a CLI interface and programmatic API.

## Usage

```bash
# Install Heretic
pip install heretic-llm

# Process a model (fully automatic)
heretic --model-name meta-llama/Llama-3.1-8B-Instruct

# The tool will:
# 1. Download the model from HuggingFace
# 2. Generate contrastive prompt pairs
# 3. Run forward passes to collect activations
# 4. Extract the refusal direction
# 5. Optimize ablation parameters with Optuna TPE
# 6. Apply the ablation to model weights
# 7. Save and optionally publish to HuggingFace
```

The entire process runs automatically. No manual configuration of ablation parameters is needed.

## Technical Deep Dive: The Ablation Operation

The ablation operation itself is a projection. Given a weight matrix `W` at the target layer and the extracted refusal direction vector `d`, the ablated weight is computed as:

```python
# Refusal direction extraction
d = mean(harmful_activations) - mean(benign_activations)
d = d / norm(d)  # Normalize

# Ablation: project out the refusal direction
W_ablated = W - (W @ d) @ d.T
```

This operation removes the component of `W` that lies along the refusal direction `d`. After ablation, the model can no longer represent the refusal behavior encoded in that direction. The Optuna TPE optimizer searches over:

- **Target layer** -- Which transformer layer to ablate (typically in the middle-to-late layers)
- **Scaling factor** -- How strongly to apply the ablation (0.0 to 1.0)
- **Projection method** -- PCA-based vs. mean-difference for direction extraction

> **Amazing**: The community has published over 1,247 Heretic-processed models on HuggingFace, demonstrating the tool's reliability and the demand for ablated models across different base architectures.

## Comparison with Manual Abliteration

| Aspect | Manual Abliteration | Heretic (Automatic) |
|--------|-------------------|---------------------|
| Parameter tuning | Manual trial-and-error | Optuna TPE Bayesian search |
| Target layer selection | Manual inspection | Automatic optimization |
| Scaling factor | Guessed or grid-searched | TPE-optimized |
| Time to process | Hours to days | ~45 minutes (RTX 3090) |
| Reproducibility | Low (depends on researcher) | High (deterministic optimization) |
| HuggingFace publishing | Manual upload | Automatic registration |

## Research Foundations

Heretic builds on two key research papers:

1. **Arditi et al. (2024)** -- "Refusal in Language Models is Mediated by a Single Direction" -- Demonstrated that safety-aligned LLMs encode refusal behavior along a single direction in activation space, making it possible to remove refusal by ablating that direction.

2. **Lai (2025)** -- Extended the directional ablation approach with Optuna TPE optimization, showing that automated parameter search significantly outperforms manual tuning for ablation quality.

## Links

- **GitHub**: [p-e-w/heretic](https://github.com/p-e-w/heretic)
- **Codeberg Mirror**: [p-e-w/heretic](https://codeberg.org/p-e-w/heretic)
- **PyPI**: `pip install heretic-llm`
- **Website**: [heretics.fun](https://heretics.fun)
- **HuggingFace Community Models**: 1,247+ published models

## Conclusion

Heretic represents a significant advancement in the study of LLM safety alignment. By automating the directional ablation process with Optuna TPE optimization, it transforms what was previously a labor-intensive manual procedure into a single-command operation. The tool's impact is evident in the 1,247+ community-published models on HuggingFace, and its clean Python package structure makes it accessible to researchers and practitioners alike. Whether used for red-teaming safety evaluations or for studying the geometry of refusal in language models, Heretic provides a rigorous, reproducible, and efficient framework for abliteration research.