---
layout: post
title: "Scientific Agent Skills: AI-Powered Research, Science, and Finance"
description: "Scientific Agent Skills provides 135 ready-to-use AI agent skills for research, science, and finance with 100+ databases and 70+ Python package integrations for Cursor, Claude Code, and Codex."
date: 2026-05-14
header-img: "img/post-bg.jpg"
permalink: /Scientific-Agent-Skills-AI-Research-Science-Finance/
featured-img: ai-coding-frameworks/ai-coding-frameworks
categories: [AI, Python, Research]
tags: [Scientific Agent Skills, AI agents, research, science, finance, Python, open source, how to use, setup guide, tutorial]
keywords: "how to use Scientific Agent Skills, Scientific Agent Skills tutorial, Scientific Agent Skills AI research, Scientific Agent Skills vs alternatives, Scientific Agent Skills installation guide, open source AI agent skills, Scientific Agent Skills Python setup, best AI research agent tools, Scientific Agent Skills for beginners, AI-powered scientific research"
author: "PyShine"
---

## What Is Scientific Agent Skills?

Scientific Agent Skills is an open-source collection of **135 ready-to-use skills** that transform any AI coding agent -- Cursor, Claude Code, Codex, or Gemini CLI -- into a powerful research assistant capable of executing complex multi-step scientific workflows across biology, chemistry, medicine, finance, and beyond. Created by [K-Dense](https://k-dense.ai/), this repository provides curated documentation, code examples, and best practices for over 70 Python packages and 100+ scientific databases, making it significantly easier for AI agents to work with specialized scientific tools reliably.

The project follows the open [Agent Skills](https://agentskills.io/) standard, meaning these skills work with any compatible agent host. Whether you are running drug discovery pipelines, analyzing single-cell RNA-seq data, querying clinical trial databases, or performing molecular docking simulations, Scientific Agent Skills gives your AI agent the domain expertise it needs to produce production-quality results.

![Scientific Agent Skills Architecture](/assets/img/diagrams/scientific-agent-skills/scientific-agent-skills-architecture.svg)

The architecture diagram above illustrates how Scientific Agent Skills sits between your AI agent (Cursor, Claude Code, Codex, or Gemini CLI) and the scientific ecosystem. The agent discovers relevant skills automatically, then routes tasks through domain-specific skill modules that connect to 100+ databases and 70+ Python packages, ultimately producing research pipelines, publication-ready reports, and drug discovery workflows.

> **Key Insight:** Scientific Agent Skills is not just a collection of API wrappers. Each skill includes comprehensive SKILL.md documentation, practical code examples, reference materials, and integration guides that make your AI agent significantly stronger and more reliable for scientific workflows. The agent can use any Python package, but these explicitly defined skills provide the curated knowledge that turns a general-purpose coding assistant into a domain expert.

## Why Scientific Agent Skills Matters

The gap between what AI coding agents can do in theory and what they can do reliably in practice is especially wide in scientific computing. An agent might know that RDKit exists, but without explicit skill documentation it will struggle with molecular sanitization edge cases, fingerprint parameter selection, or the correct workflow for virtual screening campaigns. Scientific Agent Skills bridges this gap by providing 135 curated skill definitions that encode domain expertise, best practices, and tested code patterns.

**Three core problems this project solves:**

1. **Reliability** -- AI agents without domain-specific skills often produce code that looks correct but contains subtle scientific errors (wrong normalization methods, incorrect statistical tests, misconfigured database queries). Each skill encodes the correct patterns.

2. **Discovery** -- Researchers may not know which of the 100+ databases or 70+ packages is right for their task. The unified database-lookup skill and organized skill categories make it trivial to find the right tool.

3. **Integration** -- Complex multi-step workflows (drug discovery, clinical variant interpretation, multi-omics analysis) require chaining multiple tools and databases. The skills are designed to work together seamlessly.

> **Takeaway:** With 21,562+ GitHub stars and growing at 637+ stars per day, Scientific Agent Skills has clearly struck a nerve in the research community. It is the most comprehensive open-source collection of scientific AI agent skills available today.

## Installation and Setup

Getting started with Scientific Agent Skills is straightforward. The project supports multiple installation methods depending on your preferred agent host.

### Option 1: npx (Recommended)

The official installation method works across all platforms:

```bash
npx skills add K-Dense-AI/scientific-agent-skills
```

This single command installs all 135 skills to your agent's skills directory. Your AI agent will automatically discover and use relevant skills when you mention scientific tasks in your prompts.

### Option 2: GitHub CLI

If you use GitHub CLI v2.90.0+, you can install skills with more control:

```bash
# Install all skills
gh skill install K-Dense-AI/scientific-agent-skills

# Install a specific skill only
gh skill install K-Dense-AI/scientific-agent-skills scanpy

# Target a specific agent host
gh skill install K-Dense-AI/scientific-agent-skills --agent cursor
gh skill install K-Dense-AI/scientific-agent-skills --agent claude-code
gh skill install K-Dense-AI/scientific-agent-skills --agent codex
```

### Prerequisites

- **Python**: 3.11+ (3.12+ recommended)
- **uv**: Python package manager for installing skill dependencies
- **Agent**: Any Agent Skills-compatible host (Cursor, Claude Code, Codex, Gemini CLI)

Install uv on macOS/Linux:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Or on Windows:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

![Scientific Agent Skills Domain Coverage](/assets/img/diagrams/scientific-agent-skills/scientific-agent-skills-features.svg)

The domain coverage diagram above shows how the 135 skills are organized across six major categories: Bioinformatics and Genomics (21+ skills), Cheminformatics and Drug Discovery (10+ skills), Clinical Research and Precision Medicine (8+ skills), Machine Learning and AI (16+ skills), Data Analysis and Visualization (16+ skills), and Scientific Communication (20+ skills). Each category connects to specific tool skills (like Scanpy, RDKit, PyDESeq2) and is tagged with its skill count.

## The 135 Skills: A Deep Dive

### Bioinformatics and Genomics (21+ Skills)

This is the largest domain category, covering the full spectrum of genomic analysis:

- **Scanpy** -- End-to-end single-cell RNA-seq analysis: QC, normalization, dimensionality reduction (PCA, UMAP, t-SNE), clustering (Leiden, Louvain), marker gene identification, and trajectory inference (PAGA)
- **BioPython** -- Sequence manipulation, NCBI database access (38 sub-databases via Entrez), BLAST integration, phylogenetic analysis, and structure parsing
- **PyDESeq2** -- Differential gene expression analysis using negative binomial GLMs, the Python equivalent of the widely-used DESeq2 R package
- **Arboreto** -- Gene regulatory network inference from single-cell data using GRNBoost2 and GENIE3 algorithms
- **Cellxgene Census** -- Access 50M+ cells across 1,000+ datasets from CZ CELLxGENE Discover with standardized annotations
- **gget** -- Unified command-line interface to 20+ genomics databases (Ensembl, UniProt, NCBI, PDB, COSMIC)
- **pysam** -- Read, write, and manipulate SAM/BAM/CRAM alignments and VCF/BCF variant files
- **scVelo** -- RNA velocity analysis for estimating cell state transitions from unspliced/spliced mRNA dynamics
- **scvi-tools** -- 25+ probabilistic deep learning models for single-cell omics (scVI, scANVI, totalVI, MultiVI, and more)
- **deepTools** -- Comprehensive NGS data visualization suite for ChIP-seq, RNA-seq, and ATAC-seq
- **TileDB-VCF** -- Scalable VCF/BCF storage and querying for population genomics with incremental sample addition

### Cheminformatics and Drug Discovery (10+ Skills)

- **RDKit** -- The industry-standard cheminformatics toolkit: molecular I/O, 200+ descriptors, fingerprints (Morgan, MACCS, topological torsions), SMARTS pattern matching, 3D coordinate generation, and molecular drawing
- **DiffDock** -- State-of-the-art diffusion-based molecular docking for predicting protein-ligand binding poses
- **DeepChem** -- Deep learning for molecular property prediction with graph neural networks (GCN, GAT, MPNN, AttentiveFP) and the MoleculeNet benchmark suite
- **Datamol** -- Enhanced RDKit wrapper with optimized workflows for molecular standardization, featurization, and parallel processing
- **Molfeat** -- 100+ molecular featurizers including ECFP, MACCS, pharmacophore fingerprints, and pre-trained models (MolBERT, ChemBERTa, Uni-Mol)
- **MedChem** -- Drug-likeness assessment with Lipinski's Rule of Five, ADMET prediction, and synthetic accessibility scoring
- **PyTDC** -- Access to Therapeutics Data Commons benchmarks for ADMET prediction, drug-target interactions, and molecular generation
- **Rowan** -- Cloud-based quantum chemistry platform with 45+ calculation types including pKa prediction, protein-ligand docking, and AI-powered protein cofolding

### Clinical Research and Precision Medicine (8+ Skills)

- **Database Lookup** -- Unified REST API access to 78 public databases including ClinicalTrials.gov, ClinVar, COSMIC, FDA, cBioPortal, and OMIM
- **DepMap** -- Cancer Dependency Map queries for gene dependency scores, drug sensitivity data, and gene effect profiles
- **PyHealth** -- Healthcare AI toolkit with 10+ datasets (MIMIC-III/IV, eICU), 33+ models, and clinical prediction tasks
- **Clinical Decision Support** -- Generate professional CDS documents with GRADE evidence grading and biomarker integration
- **Clinical Reports** -- Comprehensive clinical report generation following CARE, ICH-E3, and HIPAA standards
- **NeuroKit2** -- Biosignal processing for ECG, EEG, EDA, RSP, PPG, EMG with heart rate variability analysis across 25+ entropy types

### Machine Learning and AI (16+ Skills)

- **PyTorch Lightning** -- Structured deep learning training with distributed training, mixed precision, and checkpointing
- **scikit-learn** -- Classical ML for classification, regression, clustering, dimensionality reduction, and model selection
- **TimesFM** -- Google's zero-shot foundation model for univariate time series forecasting
- **aeon** -- Time series classification, regression, clustering, and anomaly detection with 100+ algorithms
- **PyMC** -- Bayesian statistical modeling with probabilistic programming and MCMC sampling
- **SHAP** -- Model interpretability with SHapley Additive exPlanations for any ML model
- **Torch Geometric** -- Graph neural networks for molecular graphs, protein interaction networks, and social networks
- **UMAP-learn** -- Uniform Manifold Approximation and Projection for dimensionality reduction

### Scientific Databases (100+ Databases)

The unified database-lookup skill provides direct REST API access to 78 public databases across every scientific domain:

| Domain | Key Databases |
|--------|---------------|
| Chemistry and Drugs | PubChem, ChEMBL, DrugBank, FDA, KEGG, ZINC, BindingDB, DailyMed |
| Biology and Genomics | UniProt, STRING, Ensembl, NCBI Gene, GEO, GTEx, AlphaFold, Reactome |
| Disease and Clinical | COSMIC, ClinVar, ClinicalTrials.gov, OMIM, cBioPortal, DisGeNET, GWAS Catalog |
| Physics and Astronomy | NASA, NIST, SIMBAD, SDSS, Exoplanet Archive |
| Economics and Finance | FRED, World Bank, SEC EDGAR, US Treasury, Alpha Vantage, BEA, BLS |
| Materials Science | Materials Project, Crystallography Open Database |
| Regulatory | FDA, USPTO, SEC EDGAR |

> **Amazing:** The database-lookup skill alone gives your AI agent access to 78 public databases through a unified interface. Instead of memorizing 78 different API endpoints, authentication methods, and query formats, your agent simply reads the skill documentation and knows exactly how to query any of them. This is a massive productivity multiplier for research workflows.

### Scientific Communication (20+ Skills)

- **Paper Lookup** -- Search 10 academic databases (PubMed, PMC, bioRxiv, medRxiv, arXiv, OpenAlex, Semantic Scholar, and more)
- **BGPT Paper Search** -- Advanced paper search returning 25+ structured fields per paper including methods, results, and quality scores
- **Literature Review** -- Systematic review workflows with citation tracking and evidence synthesis
- **Citation Management** -- DOI-to-BibTeX conversion, metadata extraction, and citation validation
- **Scientific Writing** -- Publication-quality scientific documents with proper formatting
- **Scientific Slides** -- Presentation generation for academic conferences
- **LaTeX Posters** -- Professional scientific poster creation
- **Scientific Schematics** -- Diagram generation for research figures
- **Infographics** -- AI-powered infographic creation with 10 types and 8 styles

![Scientific Agent Skills Workflow](/assets/img/diagrams/scientific-agent-skills/scientific-agent-skills-workflow.svg)

The workflow diagram above shows the typical user journey: install skills with a single command, write a research prompt, and the agent automatically discovers relevant skills. It then executes multi-step scientific workflows, querying databases as needed, analyzing results, and generating publication-ready output. The decision point ("Need More Data?") shows how the agent can iteratively query 100+ databases before producing final results.

## Real-World Examples

### Drug Discovery Pipeline

Find novel EGFR inhibitors for lung cancer treatment by chaining multiple skills:

```
Use available skills you have access to whenever possible. Query ChEMBL for EGFR inhibitors 
(IC50 < 50nM), analyze structure-activity relationships with RDKit, generate improved analogs 
with datamol, perform virtual screening with DiffDock against AlphaFold EGFR structure, search 
PubMed for resistance mechanisms, check COSMIC for mutations, and create visualizations and 
a comprehensive report.
```

Skills used: ChEMBL, RDKit, Datamol, DiffDock, AlphaFold DB, PubMed, COSMIC, Scientific Visualization

### Single-Cell RNA-seq Analysis

Comprehensive analysis of 10X Genomics data with public data integration:

```
Use available skills you have access to whenever possible. Load 10X dataset with Scanpy, 
perform QC and doublet removal, integrate with Cellxgene Census data, identify cell types 
using NCBI Gene markers, run differential expression with PyDESeq2, infer gene regulatory 
networks with Arboreto, enrich pathways via Reactome/KEGG, and identify therapeutic 
targets with Open Targets.
```

Skills used: Scanpy, Cellxgene Census, NCBI Gene, PyDESeq2, Arboreto, Reactome, KEGG, Open Targets

### Clinical Variant Interpretation

Analyze VCF files for hereditary cancer risk assessment:

```
Use available skills you have access to whenever possible. Parse VCF with pysam, annotate 
variants with Ensembl VEP, query ClinVar for pathogenicity, check COSMIC for cancer 
mutations, retrieve gene info from NCBI Gene, analyze protein impact with UniProt, search 
PubMed for case reports, check ClinPGx for pharmacogenomics, generate clinical report 
with document processing tools, and find matching trials on ClinicalTrials.gov.
```

Skills used: pysam, Ensembl, ClinVar, COSMIC, NCBI Gene, UniProt, PubMed, ClinPGx, Document Skills, ClinicalTrials.gov

### Multi-Omics Biomarker Discovery

Integrate RNA-seq, proteomics, and metabolomics to predict patient outcomes:

```
Use available skills you have access to whenever possible. Analyze RNA-seq with PyDESeq2, 
process mass spec with pyOpenMS, integrate metabolites from HMDB/Metabolomics Workbench, 
map proteins to pathways (UniProt/KEGG), find interactions via STRING, correlate omics 
layers with statsmodels, build predictive model with scikit-learn, and search 
ClinicalTrials.gov for relevant trials.
```

Skills used: PyDESeq2, pyOpenMS, HMDB, Metabolomics Workbench, UniProt, KEGG, STRING, statsmodels, scikit-learn, ClinicalTrials.gov

> **Important:** Each of these examples demonstrates a multi-step workflow that would normally require days of manual API research and integration setup. With Scientific Agent Skills, your AI agent can execute these pipelines from a single prompt because the skill documentation encodes the correct API endpoints, query formats, and data processing patterns.

## Skill Structure and Quality

Every skill in the repository follows a consistent structure defined by the Agent Skills specification:

```
scientific-skills/
  scanpy/
    SKILL.md              # Main skill documentation
    references/
      clustering.md        # Detailed reference for clustering methods
      differential_expression.md
      preprocessing.md
      visualization.md
      ...
  rdkit/
    SKILL.md
    references/
      molecular_io.md
      descriptors.md
      fingerprints.md
      ...
  database-lookup/
    SKILL.md
    references/
      api_endpoints.md
      ...
```

Each `SKILL.md` contains:

- **YAML frontmatter** with name, description, license, and metadata
- **Overview** explaining when to use the skill
- **Quick Start** with working code examples
- **Detailed workflows** for common use cases
- **Best practices** and common pitfalls
- **Integration guides** for combining with other skills

The project also includes security scanning via the [Cisco AI Defense Skill Scanner](https://github.com/cisco-ai-defense/skill-scanner), which checks every skill for prompt injection, data exfiltration, and malicious code patterns. The `scan_skills.py` script runs behavioral and LLM-based analysis on all 135 skills and generates a `SECURITY.md` report.

![Scientific Agent Skills Ecosystem](/assets/img/diagrams/scientific-agent-skills/scientific-agent-skills-ecosystem.svg)

The ecosystem diagram above shows how Scientific Agent Skills connects to the broader scientific computing landscape. At the top, four agent platforms (Cursor, Claude Code, Codex, Gemini CLI) feed into the central skills hub. From there, connections extend to major scientific databases (PubChem, ChEMBL, ClinVar, COSMIC, FRED, UniProt), Python packages (RDKit, Scanpy, scikit-learn, PyTorch Lightning, Astropy, COBRApy), and infrastructure platforms (Modal for cloud GPU, Benchling for LIMS, Opentrons for lab automation, DNAnexus for genomics cloud).

## Finance and Economics Capabilities

While the project name emphasizes science, the finance coverage is substantial. The database-lookup skill provides direct access to:

- **FRED** (Federal Reserve Economic Data) -- 800,000+ US economic time series
- **SEC EDGAR** -- Corporate filings, financial statements, insider transactions
- **US Treasury Fiscal Data** -- 54 datasets covering national debt, Treasury auctions, exchange rates
- **World Bank** -- International development indicators
- **Alpha Vantage** -- Stock quotes, forex, crypto, technical indicators
- **BEA** (Bureau of Economic Analysis) -- GDP, personal income, trade data
- **BLS** (Bureau of Labor Statistics) -- Employment, inflation, productivity data
- **US Census** -- Demographic and economic surveys

This makes Scientific Agent Skills a powerful tool for financial analysis, economic research, and data-driven investment research -- all accessible through natural language prompts to your AI agent.

## K-Dense BYOK: The Desktop AI Co-Scientist

For researchers who want zero-setup execution, [K-Dense BYOK](https://github.com/K-Dense-AI/k-dense-byok) provides a free, open-source desktop application that runs on your local machine. It brings your own API keys, offers 40+ model choices, and provides a full research workspace with web search, file handling, and access to all 135 skills. Your data stays on your computer, and you can optionally scale to cloud compute via [Modal](https://modal.com/) for heavy workloads.

## Contributing and Community

Scientific Agent Skills is MIT-licensed and welcomes contributions. The project has a structured contribution process:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-skill`)
3. Follow the existing directory structure and documentation patterns
4. Ensure all new skills include comprehensive `SKILL.md` files
5. Test examples and workflows thoroughly
6. Submit a pull request with a clear description

All contributions go through a review process, and the Cisco AI Defense Skill Scanner runs on every skill in the repository. Community-contributed skills are reviewed to the best of the team's ability, and the project recommends installing only the skills you actually need rather than everything at once.

## Getting Started Checklist

1. Install uv: `pip install uv` or use the platform-specific installer
2. Install skills: `npx skills add K-Dense-AI/scientific-agent-skills`
3. Open your AI agent (Cursor, Claude Code, Codex, or Gemini CLI)
4. Write a research prompt mentioning the scientific task you want to accomplish
5. The agent automatically discovers and uses relevant skills
6. Review the output and iterate

The project requires Python 3.11+ (3.12+ recommended) and works on macOS, Linux, and Windows with WSL2. Individual skills specify their own Python package dependencies in their `SKILL.md` files, and these are installed automatically via `uv` when the agent needs them.

## Citation

If you use Scientific Agent Skills in your research, cite it as:

```bibtex
@software{scientific_agent_skills_2026,
  author = {{K-Dense Inc.}},
  title = {Scientific Agent Skills: A Comprehensive Collection of Scientific Tools for AI Agents},
  year = {2026},
  url = {https://github.com/K-Dense-AI/scientific-agent-skills},
  note = {135 skills covering databases, packages, integrations, and analysis tools}
}
```

**Repository:** [https://github.com/K-Dense-AI/scientific-agent-skills](https://github.com/K-Dense-AI/scientific-agent-skills)

**License:** MIT (individual skills may have different licenses -- check each `SKILL.md`)

**Stars:** 21,562+ | **Growth:** +637/day | **Version:** 2.38.0 | **Skills:** 135