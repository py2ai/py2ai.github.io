---
layout: post
title: "ESM: Evolutionary Scale Modeling for Protein Language Models and Structure Prediction"
description: "ESM from Chan Zuckerberg Biohub is a protein biology world model featuring ESMC language models up to 6B parameters, ESMFold2 structure prediction surpassing AlphaFold3 on complexes, and the ESM Atlas mapping 6.8 billion proteins with interpretable sparse autoencoder features."
date: 2026-05-30
header-img: "img/post-bg.jpg"
permalink: /ESM-Evolutionary-Scale-Modeling-Protein-Language-Models/
featured-img: ai-coding-frameworks/ai-coding-frameworks
categories: [AI Models, Bioinformatics, Open Source]
tags: [ESM, ESMC, ESMFold2, protein language model, structure prediction, ESM Atlas, sparse autoencoders, protein design, AlphaFold, Chan Zuckerberg Biohub]
keywords: "ESM protein language model tutorial, ESMC 6B parameter model, ESMFold2 structure prediction, protein structure prediction Python, ESM Atlas 6.8 billion proteins, sparse autoencoders protein interpretability, ESM3 multimodal generative model, protein design with language models, ESM vs AlphaFold comparison, evolutionary scale modeling bioinformatics"
author: "PyShine"
---

Protein language models have transformed how we understand and design biological molecules, learning the grammar of life from billions of evolutionary sequences. ESM, from the Chan Zuckerberg Biohub, represents the next frontier: a world model for protein biology that combines ESMC, a state-of-the-art protein language model with up to 6 billion parameters, ESMFold2, a structure prediction system that surpasses AlphaFold3 on protein-protein complexes, and the ESM Atlas, a map of 6.8 billion proteins with over one billion predicted structures -- all open source under the MIT license and published in the journal Science.

The challenge in protein biology has always been scale. Understanding the relationship between a protein's amino acid sequence, its three-dimensional structure, and its biological function requires analyzing patterns across billions of evolutionary sequences. Traditional experimental methods -- X-ray crystallography, cryo-EM, NMR spectroscopy -- are slow, expensive, and limited in throughput. ESM addresses this by training language models on the entire evolutionary record, learning representations that capture structural and functional information directly from sequence data, and making those representations useful for prediction, design, and discovery.

## ESMC -- The Protein Language Model

ESMC is the latest state-of-the-art protein language model from the Chan Zuckerberg Biohub, available in three sizes: 300M, 600M, and 6B parameters. It takes amino acid sequences as input, tokenizes them using a BPE tokenizer with a 64-entry vocabulary, processes the tokens through a TransformerStack with flash attention and SwiGLU activation, and outputs per-residue embeddings and sequence logits that capture evolutionary and structural information learned from billions of protein sequences.

ESMC defines a new scaling frontier relative to ESM2, achieving stronger performance in emergent long-range structural understanding as model scale increases. The 6B parameter model learns protein representations from billions of evolutionary sequences, and its internal representations can be decomposed by Sparse Autoencoders into approximately 16,000 interpretable features that reveal the functional relationships between proteins -- making the model's world model explicitly interpretable rather than a black box.

Two inference modes are available: local inference via HuggingFace Transformers for users with GPU resources, and API inference via the Biohub Platform for those who prefer cloud-based access. The model supports a range of downstream tasks including pseudoperplexity computation, mutation scoring, and feature extraction for downstream classifiers.

![ESM Architecture Overview](/assets/img/diagrams/esm/esm-architecture-overview.svg)

The architecture diagram above illustrates the full ESM ecosystem. At the top, three model pillars define the core capabilities: ESMC for protein representation learning, ESM3 for multimodal generative modeling, and ESMFold2 for structure prediction. Each model leverages a shared infrastructure layer consisting of specialized tokenizers, a TransformerStack with UnifiedTransformerBlocks using SwiGLU activation and optional geometric attention, and output heads that produce predictions for each biological track.

The tokenizers layer converts biological data into discrete tokens: the Sequence tokenizer uses BPE with a 64-vocabulary, the Structure tokenizer uses a VQ-VAE with a 4096-entry codebook, the SS8 tokenizer handles 8 secondary structure classes plus 3 special tokens, the SASA tokenizer discretizes solvent accessibility into 16 bins plus 3 special tokens, the Function tokenizer processes 8 parallel embeddings with 260 vocabulary entries each, and the Residue tokenizer uses an EmbeddingBag with 1478 entries. All tokenized inputs are embedded into a shared d_model-dimensional space and processed through the TransformerStack.

At the bottom of the diagram, the application layer shows how the model outputs feed into three key resources: the ESM Atlas mapping 6.8 billion proteins with over 1 billion predicted structures, Sparse Autoencoders decomposing ESMC representations into approximately 16,000 interpretable features with natural language descriptions, and the Biohub Platform providing API inference and a developer console. The purple dashed arrow from ESMC to ESMFold2 indicates that ESMC 6B embeddings directly feed into ESMFold2's structure prediction pipeline, creating a tight integration between representation learning and structure prediction.

> **Key Insight:** ESMC defines a new scaling frontier relative to ESM2, achieving stronger performance in emergent long-range structural understanding as model scale increases. The 6B parameter model learns protein representations from billions of evolutionary sequences, and its internal representations can be decomposed by Sparse Autoencoders into approximately 16,000 interpretable features that reveal the functional relationships between proteins -- making the model's world model explicitly interpretable rather than a black box.

## ESM3 -- The Multimodal Generative Model

ESM3 is the previous flagship model in the ESM family, and it remains a powerful tool for multimodal protein generation and prediction. What makes ESM3 unique is its ability to jointly reason across six fundamental biological tracks: sequence, structure, secondary structure (SS8), solvent accessibility (SASA), function annotations, and residue annotations. Each track is independently tokenized and embedded, then summed element-wise to create a unified representation per residue.

The model family spans three sizes: esm3-small (1.4B parameters), esm3-medium (7B parameters), and esm3-large (98B parameters). At its largest scale, ESM3 was trained with 1.07 x 10^24 FLOPs on 2.78 billion proteins and 771 billion unique tokens, making it one of the largest protein models ever trained. The training data encompasses the full breadth of known protein diversity, and the model learns to predict masked positions across all six tracks simultaneously.

ESM3 operates as a generative masked language model: you can prompt it with partial inputs across any combination of tracks, and it will iteratively sample masked positions to generate complete predictions. This enables powerful use cases like generating a protein sequence from a desired structure, predicting the structure of a designed sequence, or filling in functional annotations for a partially characterized protein.

![ESM Multi-Track Tokenization Pipeline](/assets/img/diagrams/esm/esm-multi-track-tokenization.svg)

The multi-track tokenization pipeline diagram shows how ESM3 processes its six input tracks. On the left, each track has its own specialized tokenizer and embedding layer. The Sequence Track uses a BPE tokenizer with a 64-entry vocabulary (20 standard amino acids plus special tokens including mask, pad, cls, eos, and chain break), embedding each token into a d_model-dimensional vector via `nn.Embedding(64, d_model)`. The Structure Track encodes 3D atomic coordinates into discrete structure tokens using a VQ-VAE with a codebook of 4096 entries, plus 5 special tokens (MASK, EOS, BOS, PAD, CHAINBREAK), embedded via `nn.Embedding(4096+5, d_model)`. The Secondary Structure Track uses 8 DSSP classes plus 3 special tokens, embedded via `nn.Embedding(8+3, d_model)`. The SASA Track discretizes solvent-accessible surface area into 16 bins plus 3 special tokens, embedded via `nn.Embedding(16+3, d_model)`. The Function Track encodes function annotations as 8 parallel tokens, each with a 260-entry vocabulary, embedded via `nn.Embedding(260, d_model//8)` and concatenated. The Residue Track uses an EmbeddingBag with 1478 entries, summed to produce a d_model-dimensional embedding.

In the center, all six track embeddings are summed element-wise to create a unified d_model-dimensional representation per residue. Two additional mandatory information inputs -- average pLDDT and per-residue pLDDT -- are projected through RBF (16 bins) and linear layers to d_model dimensions and added to the sum. This unified representation then passes through the TransformerStack, which includes optional geometric attention layers for 3D reasoning. On the right, six independent RegressionHead modules produce logits for each output track, enabling prediction across all six modalities simultaneously.

> **Amazing:** ESM3 is a multimodal generative model that jointly reasons across six fundamental biological tracks -- sequence, structure, secondary structure, solvent accessibility, function, and residue annotations -- all represented as discrete tokens. At its largest scale, ESM3 was trained with 1.07 x 10^24 FLOPs on 2.78 billion proteins and 771 billion unique tokens, reaching 98 billion parameters. You can prompt it with partial inputs across any combination of tracks, and it will iteratively sample masked positions to generate complete predictions.

## ESMFold2 -- Structure Prediction

ESMFold2 is the latest structure prediction model from the Chan Zuckerberg Biohub, built on ESMC 6B embeddings and using a diffusion-based architecture to predict all-atom 3D protein structures. It represents a significant advance over previous approaches: ESMFold2 surpasses other models in DockQ pass-rate on Foldbench protein-protein and antibody-antigen complexes, and it has been validated in the lab across five therapeutic targets, producing de novo minibinders and antibody-derived scFvs with nanomolar affinities and functional activity.

What sets ESMFold2 apart from earlier structure prediction tools is its ability to accept not just protein sequences but also DNA, RNA, and ligand inputs. This enables structure prediction for multi-chain molecular assemblies including protein-protein interactions, antibody-antigen complexes, and protein-DNA/RNA complexes. The model uses a `StructurePredictionInput` data class that assembles multiple chains -- `ProteinInput`, `DNAInput`, `RNAInput`, and `LigandInput` -- into a molecular complex representation, with support for post-translational modifications specified using CCD codes.

The core structure prediction uses an iterative diffusion process with configurable parameters: `num_loops` (default 3) controls the number of folding refinement iterations, `num_sampling_steps` (default 50) controls the diffusion sampling steps, and `num_diffusion_samples` controls the number of independent samples. The diffusion process denoises from random coordinates to predicted all-atom 3D structures, producing confidence metrics including pLDDT (per-residue confidence), pTM (predicted TM-score for single chains), and ipTM (inter-chain predicted TM-score for complexes).

A single-sequence mode provides an order-of-magnitude speedup for folding without requiring multiple sequence alignments, making ESMFold2 practical for high-throughput applications where MSA generation would be a bottleneck.

![ESMFold2 Structure Prediction Pipeline](/assets/img/diagrams/esm/esmfold2-structure-prediction.svg)

The ESMFold2 structure prediction pipeline diagram illustrates the five-step process from input to output. Step 1 shows the four input types: ProteinInput for amino acid sequences, DNAInput for nucleotide sequences with optional modifications, RNAInput for RNA sequences, and LigandInput for small molecules specified by CCD codes. An optional MSA input can enhance accuracy on challenging targets.

Step 2 shows the ESMC 6B Embedding stage, where amino acid sequences are tokenized and passed through the ESMC 6B parameter model to produce per-residue embeddings. These embeddings capture evolutionary and structural information learned from billions of protein sequences, providing a rich representation that enables single-sequence folding without requiring external MSA searches.

Step 3 shows the StructurePredictionInput Assembly, where the multi-chain input is assembled into a molecular complex representation. Chain boundaries are marked with special tokens, and covalent bonds, modifications, and ligand interactions are specified. Step 4 shows the Diffusion-Based Folding process, which iteratively denoises from random coordinates to predicted all-atom 3D structures using the configurable parameters described above.

Step 5 shows the output stage, with a decision diamond determining whether the input is a complex. If yes, the model computes ipTM for inter-chain confidence in addition to pLDDT and pTM. If no, only pLDDT and pTM are computed. Results can be exported as CIF (mmCIF) or PDB format files. The dashed arrow from the "Single-Sequence Mode" node to the ESMC embedding stage indicates the fast path where ESMC embeddings alone provide order-of-magnitude speedup without MSA input.

> **Takeaway:** ESMFold2 accepts not just protein sequences but also DNA, RNA, and ligand inputs, enabling structure prediction for protein-protein interactions, antibody-antigen complexes, and multi-chain molecular assemblies. It has been validated in the lab across five therapeutic targets, producing de novo minibinders and antibody-derived scFvs with nanomolar affinities and functional activity. A single-sequence mode provides an order of magnitude speedup for folding without multiple sequence alignments.

## ESM Atlas and Sparse Autoencoders

The ESM Atlas is a map of 6.8 billion proteins covering the full breadth of life's biodiversity, with over one billion predicted structures generated by ESMFold2's folding throughput. The Atlas is organized according to ESMC's internal world model, meaning that proteins are clustered and navigated based on the representations that ESMC has learned from billions of evolutionary sequences. This makes it possible to explore protein space in a biologically meaningful way, discovering relationships between proteins that would be invisible from sequence alone.

Sparse Autoencoders (SAEs) provide a window into ESMC's internal representations, decomposing them into approximately 16,000 interpretable features. Each feature is summarized in natural language using an agentic pipeline that generates descriptions of what the feature responds to -- for example, a feature might correspond to "alpha-helical regions in membrane proteins" or "catalytic residues in serine proteases." SAEs are available for different model scales, layers, and granularity levels, with the primary model being `ESMC-6B-sae-k64-codebook16384` available on HuggingFace.

The combination of the ESM Atlas and SAEs creates a powerful discovery platform: researchers can navigate the Atlas to find proteins of interest, then use SAEs to understand what ESMC has learned about those proteins at a feature-by-feature level. This interpretability is crucial for building trust in model predictions and for generating new biological hypotheses.

## Installation and Quick Start

ESM requires Python 3.12 and can be installed directly from GitHub:

```bash
pip install esm@git+https://github.com/Biohub/esm.git@main
```

### ESMC Local Inference

```python
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
from huggingface_hub import login

# Login with your Hugging Face credentials
login()

# Example GFP sequence
sequences = ["MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK"]

model = AutoModelForMaskedLM.from_pretrained(
    "Biohub/ESMC-6B",
    device_map="auto",
).eval()
tokenizer = AutoTokenizer.from_pretrained("Biohub/ESMC-6B")

inputs = tokenizer(sequences, return_tensors="pt", padding=True)
inputs = {k: v.to(model.device) for k, v in inputs.items()}
with torch.inference_mode():
    output = model(**inputs)

# Return all transformer layer hidden states
output = model(**inputs, output_hidden_states=True)
```

### ESMC Biohub Platform API

```python
from esm.sdk import esmc_client
from esm.sdk.api import ESMProtein, LogitsConfig

# Human carbonic anhydrase II (PDB 2CBA)
protein = ESMProtein(
    sequence=(
        "MSHHWGYGKHNGPEHWHKDFPIAKGERQSPVDIDTHTAKYDPSLKPLSVSYDQATSLRILNNGHAFNVEFDD"
        "SQDKAVLKGGPLDGTYRLIQFHFHWGSLDGQGSEHTVDKKKYAAELHLVHWNTKYGDFGKAVQQPDGLAVL"
        "GIFLKVGSAKPGLQKVVDVLDSIKTKGKSADFTNFDPRGLLPESLDYWTYPGSLTTPPLLECVTWIVLKEP"
        "ISVSSEQVLKFRKLNFNGEGEPEELMVDNWRPAQPLKNRQIKASFK"
    )
)
model = esmc_client(
    model="esmc-600m-2024-12", url="https://biohub.ai", token="<your API token>"
)

protein_tensor = model.encode(protein)
logits_output = model.logits(
    protein_tensor, LogitsConfig(sequence=True, return_embeddings=True)
)

print(logits_output.logits, logits_output.embeddings)
```

### ESMFold2 Local Inference

```python
from esm.models.esmfold2 import (
    DNAInput,
    ESMFold2InputBuilder,
    LigandInput,
    Modification,
    ProteinInput,
    StructurePredictionInput,
)
from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model

HHAI_SEQ = (
    "MIEIKDKQLTGLRFIDLFAGLGGFRLALESCGAECVYSNEWDKYAQEVYEMNFGEKPEGDITQVNEKTIPDH"
    "DILCAGFPCQAFSISGKQKGFEDSRGTLFFDIARIVREKKPKVVFMENVKNFASHDNGNTLEVVKNTMNELD"
    "YSFHAKVLNALDYGIPQKRERIYMICFRNDLNIQNFQFPKPFELNTFVKDLLLPDSEVEHLVIDRKDLVMTN"
    "QEIEQTTPKTVRLGIVGKGGQGERIYSTRGIAITLSAYGGGIFAKTGGYLVNGKTRKLHPRECARVMGYPDS"
    "YKVHPSTSQAYKQFGNSVVINVLQYIAYNIGSSLNFKPY"
)

model = ESMFold2Model.from_pretrained("biohub/ESMFold2").cuda().eval()

spi = StructurePredictionInput(
    sequences=[
        ProteinInput(id="A", sequence=HHAI_SEQ),
        DNAInput(
            id="B",
            sequence="GATAGCGCTATC",
            modifications=[Modification(position=5, ccd="C36")],
        ),
        DNAInput(
            id="C",
            sequence="TGATAGCGCTATC",
            modifications=[Modification(position=6, ccd="C36")],
        ),
        LigandInput(id="L", ccd=["SAH"]),
    ]
)

result = ESMFold2InputBuilder().fold(
    model, spi, num_loops=3, num_sampling_steps=50, num_diffusion_samples=1, seed=0
)

print(f"pLDDT mean: {float(result.plddt.mean()):.3f}, pTM: {float(result.ptm):.3f}, ipTM: {float(result.iptm):.3f}")

with open("1mht_pred.cif", "w") as f:
    f.write(result.complex.to_mmcif())
```

### ESMFold2 Biohub Platform API

```python
from esm.sdk.forge import SequenceStructureForgeInferenceClient
from esm.sdk.api import FoldingConfig
from esm.utils.structure.input_builder import ProteinInput, StructurePredictionInput

client = SequenceStructureForgeInferenceClient(
    model="esmfold2-fast-2026-05", url="https://biohub.ai", token="<your API token>"
)

# Human carbonic anhydrase II (PDB 2CBA)
ca2_sequence = (
    "MSHHWGYGKHNGPEHWHKDFPIAKGERQSPVDIDTHTAKYDPSLKPLSVSYDQATSLRILNNGHAFNVEFDD"
    "SQDKAVLKGGPLDGTYRLIQFHFHWGSLDGQGSEHTVDKKKYAAELHLVHWNTKYGDFGKAVQQPDGLAVL"
    "GIFLKVGSAKPGLQKVVDVLDSIKTKGKSADFTNFDPRGLLPESLDYWTYPGSLTTPPLLECVTWIVLKEP"
    "ISVSSEQVLKFRKLNFNGEGEPEELMVDNWRPAQPLKNRQIKASFK"
)
ca2_input = StructurePredictionInput(
    sequences=[ProteinInput(id="A", sequence=ca2_sequence)]
)

config = FoldingConfig(num_loops=3, num_sampling_steps=32)
result = client.fold_all_atom(ca2_input, config=config)

with open("result.cif", "w") as f:
    f.write(result.complex.to_mmcif())
```

### Sparse Autoencoder Feature Extraction

```python
import torch
from transformers import AutoModel, AutoTokenizer

sequence = "MGSNKSKPKDASQRRRSLEPAENVHGAGGGAFPASQTPSKPASADGHRGPSAAFAPAAAEPKLFGGFNSSDTVTSPQRAGPLAGGVTTFVALYDYESRTETDLSFKKGERLQIVNNTEGDWWLAHSLSTGQTGYIPSNYVAPSDSIQAEEWYFGKITRRESERLLLNAENPRGTFLVRESETTKGAYCLSVSDFDNAKGLNVKHYKIRKLDSGGFYITSRTQFNSLQQLVAYYSKHADGLCHRLTTVCPTSKPQTQGLAKDAWEIPRESLRLEVKLGQGCFGEVWMGTWNGTTRVAIKTLKPGTMSPEAFLQEAQVMKKLRHEKLVQLYAVVSEEPIYIVTEYMSKGSLLDFLKGETGKYLRLPQLVDMAAQIASGMAYVERMNYVHRDLRAANILVGENLVCKVADFGLARLIEDNEYTARQGAKFPIKWTAPEAALYGRFTIKSDVWSFGILLTELTTKGRVPYPGMVNREVLDQVERGYRMPCPPECPESLHDLMCQCWRKEPEERPTFEYLQAFLEDYFTSTEPQYQPGENL"

model = AutoModel.from_pretrained("Biohub/ESMC-6B", device_map="auto").eval()
tokenizer = AutoTokenizer.from_pretrained("Biohub/ESMC-6B")
sae = AutoModel.from_pretrained(
    "Biohub/ESMC-6B-sae-k64-codebook16384",
    allow_patterns=["config.json", "layer_30.safetensors", "layer_60.safetensors"],
    device=model.device,
)
sae.initialize_layers([30, 60])
model.add_sae_models([sae.layers["30"], sae.layers["60"]])

inputs = tokenizer(sequence, return_tensors="pt", padding=True)
inputs = {k: v.to(model.device) for k, v in inputs.items()}

with torch.inference_mode():
    output = model(**inputs)

output["sae_outputs"]["layer60"]  # sparse.coo tensor
print(output["sae_outputs"]["layer60"].shape)
```

### ESM3 Generation

```python
from huggingface_hub import login
from esm.models.esm3 import ESM3
from esm.sdk.api import ESM3InferenceClient, ESMProtein, GenerationConfig

# Will instruct you how to get an API key from Hugging Face Hub
login()

# Download model weights and instantiate on GPU
model = ESM3.from_pretrained("esm3-sm-open-v1").to("cuda")

# Generate a completion for a partial Carbonic Anhydrase (2vvb)
prompt = "___________________________________________________DQATSLRILNNGHAFNVEFDDSQDKAVLKGGPLDGTYRLIQFHFHWGSLDGQGSEHTVDKKKYAAELHLVHWNTKYGDFGKAVQQPDGLAVLGIFLKVGSAKPGLQKVVDVLDSIKTKGKSADFTNFDPRGLLPESLDYWTYPGSLTTPP___________________________________________________________"
protein = ESMProtein(sequence=prompt)

# Generate the sequence, then the structure
protein = model.generate(protein, GenerationConfig(track="sequence", num_steps=8, temperature=0.7))
protein = model.generate(protein, GenerationConfig(track="structure", num_steps=8))
protein.to_pdb("./generation.pdb")

# Round trip: inverse fold sequence, then recompute structure
protein.sequence = None
protein = model.generate(protein, GenerationConfig(track="sequence", num_steps=8))
protein.coordinates = None
protein = model.generate(protein, GenerationConfig(track="structure", num_steps=8))
protein.to_pdb("./round_tripped.pdb")
```

## Conclusion

ESM from the Chan Zuckerberg Biohub provides a complete protein biology world model that spans language understanding, structure prediction, and interpretable feature discovery. ESMC delivers state-of-the-art protein representations at scales up to 6 billion parameters, ESMFold2 predicts all-atom 3D structures for proteins, DNA, RNA, and ligand complexes with accuracy that surpasses AlphaFold3 on protein-protein interactions, and the ESM Atlas maps 6.8 billion proteins with over one billion predicted structures. Sparse Autoencoders make ESMC's internal world model interpretable by decomposing it into approximately 16,000 features with natural language descriptions.

The entire ESM ecosystem is open source under the MIT license, with model weights available on HuggingFace and inference accessible through the Biohub Platform API. Whether you are predicting protein structures, designing novel proteins, exploring the protein universe, or understanding what a language model has learned about biology, ESM provides the tools and representations to make it possible.

> **Important:** The ESM Atlas maps 6.8 billion proteins covering the full breadth of life's biodiversity, with over one billion predicted structures generated by ESMFold2's folding throughput. The Atlas is organized according to ESMC's internal world model, and Sparse Autoencoders make this world model interpretable by decomposing it into approximately 16,000 features with natural language descriptions generated by an agentic pipeline. This combination of scale, structure prediction, and interpretability makes ESM a unique resource for protein biology research.

**Links:**
- GitHub: [https://github.com/Biohub/esm](https://github.com/Biohub/esm)
- Biohub Platform: [https://biohub.ai/](https://biohub.ai/)
- ESMC Product Page: [https://biohub.ai/esm/protein](https://biohub.ai/esm/protein)
- HuggingFace ESMC: [https://huggingface.co/collections/biohub/esmc-model-family](https://huggingface.co/collections/biohub/esmc-model-family)
- HuggingFace ESMFold2: [https://huggingface.co/biohub/ESMFold2](https://huggingface.co/biohub/ESMFold2)
- HuggingFace SAE: [https://huggingface.co/collections/biohub/esmc-saes-for-hidden-states-all-layers](https://huggingface.co/collections/biohub/esmc-saes-for-hidden-states-all-layers)
- HuggingFace ESM3: [https://huggingface.co/biohub/esm3-sm-open-v1](https://huggingface.co/biohub/esm3-sm-open-v1)
- Science Paper: [https://www.science.org/doi/10.1126/science.ads0018](https://www.science.org/doi/10.1126/science.ads0018)