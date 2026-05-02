---
layout: post
title: "OpenAI Privacy Filter: Protecting Sensitive Data with Bidirectional Token Classification"
description: "Learn how OpenAI Privacy Filter detects and redacts PII in text using a 1.5B parameter bidirectional transformer model. Explore its architecture, 8 privacy categories, Viterbi decoding, and Python API integration."
date: 2026-04-26
header-img: "img/post-bg.jpg"
permalink: /OpenAI-Privacy-Filter-Protecting-Sensitive-Data/
featured-img: ai-coding-frameworks/ai-coding-frameworks
categories: [AI, Open Source, Python]
tags: [OpenAI, privacy filter, PII detection, data redaction, token classification, NLP, transformer, data privacy, machine learning, open source]
keywords: "OpenAI Privacy Filter tutorial, how to detect PII in text, privacy filter Python API, bidirectional token classification, PII redaction tool, OpenAI privacy filter installation, data sanitization machine learning, Viterbi decoding NER, privacy filter vs alternatives, open source PII detection"
author: "PyShine"
---

# OpenAI Privacy Filter: Protecting Sensitive Data with Bidirectional Token Classification

OpenAI Privacy Filter is a bidirectional token-classification model for personally identifiable information (PII) detection and masking in text. Released under the permissive Apache 2.0 license, it provides a fast, context-aware, and tunable solution for high-throughput data sanitization workflows that teams can run entirely on-premises.

![Architecture Overview](/assets/img/diagrams/privacy-filter/privacy-filter-architecture.svg)

## Understanding the Architecture

The architecture diagram above illustrates the end-to-end data flow of OpenAI Privacy Filter, from raw text input through multiple processing stages to the final redacted output. Let us break down each component:

**Input Sources**

The filter accepts three primary input modalities: raw text strings, documents and files, and data streams via pipes. This flexibility makes it suitable for interactive use, batch processing, and integration into existing data pipelines.

**Tiktoken Tokenizer**

All input text first passes through Tiktoken, OpenAI's byte-pair encoding tokenizer. This is the same tokenizer used across OpenAI's model family, ensuring consistent tokenization behavior. The tokenizer converts raw text into integer token IDs that the transformer can process.

**Context Window Splitter**

With a 128,000-token context window, the splitter handles long documents by breaking them into manageable windows. This is critical for processing lengthy legal documents, medical records, or financial reports without chunking artifacts at window boundaries. Overlapping window aggregation ensures consistent predictions across window boundaries.

**Bidirectional Transformer Encoder**

The core of Privacy Filter is a 1.5 billion parameter transformer with only 50 million active parameters per token thanks to its sparse Mixture-of-Experts (MoE) architecture. Key architectural details include:

- 8 transformer blocks with pre-norm residual connections
- Grouped-query attention (GQA) with 14 query heads and 2 KV heads
- Banded bidirectional attention with 128-token left and right context windows
- Sparse MoE feed-forward blocks with 128 experts and top-4 routing per token
- Rotary positional embeddings (RoPE) with YaRN-style scaling for long contexts

The banded attention mechanism is particularly noteworthy. Unlike causal (autoregressive) models that only look backward, Privacy Filter uses bidirectional attention within a local window of 257 tokens (128 left + self + 128 right). This allows each token to consider both preceding and following context when making classification decisions, which is essential for accurately identifying privacy spans where context determines whether a string is sensitive.

**Token Classification Head and Viterbi Decoder**

The transformer output feeds into a classification head that produces 33 logits per token (1 background class + 8 span categories x 4 BIOES boundary tags). These per-token scores then pass through a constrained Viterbi decoder that enforces valid BIOES transition sequences, ensuring coherent span boundaries rather than fragmented predictions.

## Privacy Detection Pipeline

![Detection Pipeline](/assets/img/diagrams/privacy-filter/privacy-filter-detection-pipeline.svg)

### Understanding the Detection Pipeline

The detection pipeline diagram shows the five-step process that transforms raw input text into privacy-filtered output, along with the eight privacy categories the model can detect.

**Step 1: Tokenize**

Input text is converted to token IDs using Tiktoken encoding. This step handles multilingual text, special characters, and ensures consistent representation across different input formats.

**Step 2: Window Splitting**

For inputs exceeding the context window, the text is split into overlapping windows. Each window is processed independently, and results are aggregated using log-probability log-sum-exp combination across overlapping regions. This ensures seamless handling of documents far longer than the 128K token window.

**Step 3: Bidirectional Encoding**

Each window passes through the 8-layer bidirectional transformer. The banded attention mechanism allows each token to attend to 128 tokens on each side, creating a 257-token effective attention window. This local bidirectional context is crucial for distinguishing, for example, "Alice" as a person name versus "Alice" as part of an email address.

**Step 4: Per-Token Classification**

The classification head produces a probability distribution over 33 token-level classes for each token. These 33 classes derive from the BIOES tagging scheme applied to 8 privacy categories:

| Category | Description | Example |
|----------|-------------|---------|
| `private_person` | Personal names | "Alice Johnson" |
| `private_date` | Dates of birth, events | "1990-01-02" |
| `private_email` | Email addresses | "alice@example.com" |
| `private_phone` | Phone numbers | "+1-555-0123" |
| `private_address` | Physical addresses | "123 Main Street" |
| `private_url` | Personal URLs | "https://alice.example" |
| `account_number` | Account identifiers | "ACC-12345" |
| `secret` | Passwords, API keys, tokens | "sk-abc123..." |

The BIOES scheme (Begin, Inside, Outside, End, Single) tags each token with boundary information. For example, "Alice Johnson" would be tagged as `B-private_person` for "Alice" and `E-private_person` for "Johnson", while a single-token name like "Alice" would receive `S-private_person`.

**Step 5: Constrained Viterbi Decoding**

Rather than taking an independent argmax for each token, Privacy Filter uses a constrained Viterbi decoder with linear-chain transition scoring. This global path optimization ensures that span boundaries are coherent and consistent. The decoder enforces valid BIOES transitions (for example, an `I-` tag must follow a `B-` tag of the same category) and uses six transition-bias parameters that control:

- Background persistence (tendency to stay in the `O` state)
- Span entry (tendency to start a new span)
- Span continuation (tendency to continue an existing span)
- Span closure (tendency to end a span)
- Boundary-to-boundary handoff (transitions between span types)

Users can tune these parameters at runtime to adjust the precision/recall tradeoff for their specific use case.

**Output Modes**

The pipeline supports two output modes:

- **Typed mode**: Preserves the model's category labels (e.g., `<PRIVATE_PERSON>`, `<PRIVATE_DATE>`)
- **Redacted mode**: Collapses all detected spans into a single `<REDACTED>` label

## Integration Options

![Integration Diagram](/assets/img/diagrams/privacy-filter/privacy-filter-integration.svg)

### Understanding Integration Options

The integration diagram shows the four primary ways to use Privacy Filter, along with configuration options, processing internals, and output formats.

**CLI Usage**

The simplest way to get started is through the command-line interface:

```bash
# One-shot redaction
opf "Alice was born on 1990-01-02."

# Redact a file
opf -f /path/to/file

# Pipe input through redaction
cat /path/to/file | grep -e 'some_pattern' | opf

# Interactive mode (no input argument)
opf

# Run on CPU
opf --device cpu "Alice was born on 1990-01-02."

# Use a custom checkpoint
opf --checkpoint /path/to/checkpoint_dir "Alice was born on 1990-01-02."
```

**Python API**

For programmatic integration, the `OPF` class provides a rich, reusable API:

```python
from opf._api import OPF, DecodeOptions

# Create a redactor with default settings
redactor = OPF(output_mode="typed")

# Redact a single text
result = redactor.redact("Alice was born on 1990-01-02.")
print(result.to_json())

# Access structured results
print(f"Text: {result.text}")
print(f"Redacted: {result.redacted_text}")
print(f"Spans: {result.detected_spans}")
print(f"Summary: {result.summary}")

# Switch to redacted mode
redactor.set_output_mode("redacted")
result = redactor.redact("Alice was born on 1990-01-02.")
print(result.redacted_text)  # <REDACTED> was born on <REDACTED>.

# Text-only output
redactor_text_only = OPF(output_text_only=True)
redacted = redactor_text_only.redact("Contact alice@example.com")
print(redacted)  # Contact <PRIVATE_EMAIL>

# Module-level convenience function
from opf._api import redact
redacted = redact("Alice's SSN is 123-45-6789")
```

**Configuration Options**

The `OPF` class supports fluent configuration through setter methods:

```python
redactor = (
    OPF()
    .set_device("cpu")                    # Run on CPU
    .set_output_mode("typed")             # Preserve category labels
    .set_decode_mode("viterbi")            # Use Viterbi decoding
    .trim_whitespace(True)                 # Trim span whitespace
    .set_model_path("/custom/checkpoint")  # Custom model path
)

# Per-call decode overrides
result = redactor.redact(
    "Alice was born on 1990-01-02.",
    decode=DecodeOptions(decode_mode="argmax")  # Override to argmax for this call
)
```

**Structured Output Schema**

The `RedactionResult` object provides a complete, JSON-serializable output:

```json
{
  "schema_version": 1,
  "summary": {
    "output_mode": "typed",
    "span_count": 2,
    "by_label": {
      "private_date": 1,
      "private_person": 1
    },
    "decoded_mismatch": false
  },
  "text": "Alice was born on 1990-01-02.",
  "detected_spans": [
    {
      "label": "private_person",
      "start": 0,
      "end": 5,
      "text": "Alice",
      "placeholder": "<PRIVATE_PERSON>"
    },
    {
      "label": "private_date",
      "start": 14,
      "end": 24,
      "text": "1990-01-02",
      "placeholder": "<PRIVATE_DATE>"
    }
  ],
  "redacted_text": "<PRIVATE_PERSON> was born on <PRIVATE_DATE>."
}
```

## Evaluation and Fine-Tuning

### Running Evaluations

Privacy Filter includes built-in evaluation tooling for measuring detection quality against labeled datasets:

```bash
# Evaluate with typed mode (category-level metrics)
opf eval examples/data/sample_eval_five_examples.jsonl

# Evaluate with untyped mode (span-level matching)
opf eval examples/data/sample_eval_five_examples.jsonl --eval-mode untyped
```

The evaluation system supports two modes:

- **Typed mode**: Compares predicted labels against ground truth using the OPF taxonomy. Reports category-level precision, recall, and F1.
- **Untyped mode**: Ignores category identity and evaluates span detection only. Useful when your ground truth uses a different label taxonomy.

### Fine-Tuning on Custom Data

One of Privacy Filter's most powerful features is the ability to fine-tune the model on domain-specific data:

```bash
# Basic fine-tuning
opf train /path/to/train.jsonl --output-dir /path/to/finetuned_checkpoint

# With validation split
opf train /path/to/train.jsonl \
  --validation-dataset /path/to/validation.jsonl \
  --output-dir /path/to/finetuned_checkpoint

# With custom label space
opf train /path/to/train.jsonl \
  --validation-dataset /path/to/validation.jsonl \
  --label-space-json /path/to/custom_label_space.json \
  --output-dir /path/to/finetuned_checkpoint
```

Custom label spaces allow you to define entirely new privacy categories:

```json
{
  "category_version": "custom_v1",
  "span_class_names": ["O", "custom_account_id", "custom_secret"]
}
```

The fine-tuning output includes:
- `config.json` - Updated model configuration
- `model.safetensors` - Fine-tuned model weights
- `finetune_summary.json` - Training metrics and parameters
- `USAGE.txt` - Usage instructions for the fine-tuned checkpoint

## Key Features and Highlights

| Feature | Details |
|---------|---------|
| License | Apache 2.0 - permissive for commercial use |
| Model Size | 1.5B total parameters, 50M active per token |
| Context Window | 128,000 tokens |
| Architecture | Bidirectional transformer with MoE and GQA |
| Privacy Categories | 8 categories (person, date, email, phone, address, URL, account, secret) |
| Tagging Scheme | BIOES (Begin, Inside, Outside, End, Single) |
| Decoding | Constrained Viterbi with tunable operating points |
| Output Modes | Typed (category labels) and Redacted (generic) |
| Fine-Tuning | Custom label spaces and domain adaptation |
| Deployment | Local-only, on-premises, no API calls needed |
| Hardware | Runs on laptop CPU or GPU; even in web browsers |

## Model Architecture Deep Dive

Privacy Filter's architecture represents an interesting design choice: starting from an autoregressive pretrained checkpoint and converting it to a bidirectional token classifier. This approach combines the strong language understanding from autoregressive pretraining with the efficiency of single-pass classification.

**From Autoregressive to Bidirectional**

The base model follows the gpt-oss architecture pattern but is modified for classification:

1. The language modeling output head is replaced with a token-classification head over 33 privacy labels
2. The causal attention mask is replaced with a banded bidirectional attention pattern (128 tokens left + 128 tokens right)
3. Training switches from next-token prediction to supervised token-level classification

**Mixture-of-Experts Efficiency**

With 128 experts and top-4 routing per token, only 50M of the 1.5B total parameters are active during any single forward pass. This sparse activation pattern provides:

- Faster inference compared to a dense model of equivalent total parameters
- Better capacity utilization through expert specialization
- The ability to run on consumer hardware, including CPUs and even web browsers

**Constrained Sequence Decoding**

The Viterbi decoder with linear-chain transition scoring provides several advantages over independent per-token argmax:

- Enforces valid BIOES boundary transitions (no orphaned `I-` tags, for example)
- Improves span coherence by considering sequence-level structure
- Allows runtime tuning of precision/recall tradeoffs through transition bias parameters
- Produces more stable and consistent span boundaries, especially in noisy or mixed-format text

## Practical Use Cases

**Data Sanitization Pipelines**

Privacy Filter is designed for high-throughput data sanitization workflows. Teams processing large volumes of text data - customer support transcripts, legal documents, medical records, or financial reports - can integrate Privacy Filter to automatically detect and redact PII before data is shared, stored, or used for training.

**AI Training Data Preparation**

Before using text data to train or fine-tune language models, Privacy Filter can strip personal information to prevent models from memorizing and reproducing sensitive data. This is particularly relevant for compliance with data protection regulations like GDPR and CCPA.

**Compliance and Data Minimization**

Privacy Filter serves as one layer in a privacy-by-design approach. By automatically detecting and redacting PII, organizations can implement data minimization principles, ensuring that only necessary information is retained and processed.

**Custom Domain Adaptation**

The fine-tuning capability allows organizations to adapt Privacy Filter to their specific data distributions and privacy policies. A healthcare organization might fine-tune to better detect medical record numbers, while a financial institution might add custom categories for account identifiers.

## Limitations and Considerations

Privacy Filter is explicitly positioned as a redaction and data minimization aid, not an anonymization, compliance, or safety guarantee. Important limitations include:

- **Static label policy**: The model only identifies the 8 trained privacy categories. Changing policies requires fine-tuning.
- **Performance variation**: May underperform on non-English text, non-Latin scripts, uncommon personal names, or domain-specific identifiers.
- **Over-redaction risk**: May redact public entities, organizations, or common nouns when context is ambiguous.
- **Under-detection risk**: May miss uncommon names, regional naming conventions, or novel credential formats.
- **High-sensitivity settings**: Additional caution is warranted in medical, legal, financial, and government workflows where both false negatives and false positives can be costly.

## Installation

```bash
# Clone the repository
git clone https://github.com/openai/privacy-filter.git
cd privacy-filter

# Install the package
pip install -e .

# The model will be automatically downloaded on first use
# Or set OPF_CHECKPOINT environment variable to use a custom path
```

**Dependencies** (from `pyproject.toml`):

- Python 3.10+
- PyTorch
- Tiktoken (OpenAI's tokenizer)
- SafeTensors (model weight format)
- HuggingFace Hub (model downloading)
- NumPy

## Conclusion

OpenAI Privacy Filter represents a significant contribution to the open-source privacy tooling ecosystem. Its combination of bidirectional token classification, constrained Viterbi decoding, and fine-tuning support makes it a versatile tool for any organization that needs to detect and redact PII in text data. The Apache 2.0 license, compact model size, and on-premises deployment capability make it accessible for a wide range of use cases, from small teams to enterprise deployments.

The model's architecture - converting an autoregressive pretrained checkpoint into a bidirectional classifier - demonstrates an innovative approach that leverages strong language understanding while enabling efficient single-pass inference. With 8 privacy categories, tunable operating points, and custom label space support, Privacy Filter provides both out-of-the-box utility and the flexibility to adapt to domain-specific requirements.

## Links

- GitHub Repository: [https://github.com/openai/privacy-filter](https://github.com/openai/privacy-filter)
- Model Weights: [https://huggingface.co/openai/privacy-filter](https://huggingface.co/openai/privacy-filter)
- Interactive Demo: [https://huggingface.co/spaces/openai/privacy-filter](https://huggingface.co/spaces/openai/privacy-filter)
- Model Card: [OpenAI Privacy Filter Model Card (PDF)](https://cdn.openai.com/pdf/c66281ed-b638-456a-8ce1-97e9f5264a90/OpenAI-Privacy-Filter-Model-Card.pdf)
