---
layout: post
title: "Kronos: Foundation Model for Financial Markets Language"
description: "Explore Kronos, the first open-source foundation model designed to understand and process the language of financial markets with state-of-the-art performance on K-line sequences."
date: 2026-04-11
header-img: "img/post-bg.jpg"
permalink: /Kronos-Foundation-Model-Financial-Markets/
featured-img: ai-coding-frameworks/ai-coding-frameworks
tags:
  - AI
  - Machine Learning
  - Finance
  - Foundation Model
  - Python
  - Open Source
author: "PyShine"
---

# Kronos: Foundation Model for Financial Markets Language

Kronos represents a groundbreaking advancement in financial machine learning as the **first open-source foundation model** specifically designed for financial candlesticks (K-lines). Trained on data from over **45 global exchanges**, Kronos introduces a novel approach to understanding the unique "language" of financial markets through hierarchical tokenization and autoregressive modeling.

## Introduction

Financial markets generate vast amounts of time-series data in the form of K-lines (candlesticks), each containing Open, High, Low, Close, Volume, and Amount (OHLCV) information. Unlike general-purpose time-series foundation models (TSFMs), Kronos is specifically architected to handle the unique characteristics of financial data: high noise levels, non-stationary distributions, and complex temporal dependencies.

The model leverages a two-stage framework that treats financial sequences as a language:

1. **Tokenization Stage**: A specialized tokenizer quantizes continuous OHLCV data into hierarchical discrete tokens
2. **Foundation Model Stage**: A large autoregressive Transformer learns to predict future tokens, enabling diverse quantitative tasks

![Kronos Architecture Overview](/assets/img/diagrams/kronos-architecture.svg)

### Understanding the Kronos Architecture

The architecture diagram above illustrates the complete pipeline from raw financial data to market predictions. Let's examine each component in detail:

**Stage 1: Tokenization Pipeline**

The tokenization stage transforms continuous financial data into discrete tokens that can be processed by language models. This approach is revolutionary because it allows us to apply the powerful techniques developed for natural language processing to financial time-series data.

*Data Preprocessing and Normalization*
The preprocessing layer takes raw OHLCV data and applies z-score normalization to standardize the input distribution. This is critical because financial data from different markets and instruments can have vastly different scales and volatilities. The normalization ensures that the model sees consistent input distributions regardless of the underlying asset's price level.

*Encoder Transformer Blocks*
The encoder consists of multiple Transformer blocks that learn to represent the normalized financial data in a latent space. Each block includes:
- Multi-head self-attention with Rotary Positional Embeddings (RoPE)
- Feed-forward networks with SwiGLU activation
- RMSNorm for layer normalization
- Residual connections for gradient flow

The encoder's role is to compress the temporal patterns in financial data into a representation suitable for quantization.

*Binary Spherical Quantizer (BSQ)*
The BSQuantizer is the core innovation that enables hierarchical tokenization. It projects the encoder output onto a hypersphere and quantizes it into binary codes. The quantization process:

1. Normalizes vectors to unit length on the hypersphere
2. Applies binary quantization (sign function) to each dimension
3. Generates two types of tokens: s1 (coarse) and s2 (fine)

This hierarchical approach allows the model to capture both broad market movements (s1 tokens) and fine-grained price changes (s2 tokens).

**Stage 2: Foundation Model**

The foundation model is a decoder-only Transformer that learns to predict future tokens autoregressively.

*Hierarchical Embedding Layer*
The embedding layer combines s1 and s2 token embeddings through a fusion projection. This allows the model to maintain the hierarchical structure throughout the network. Each token type has its own vocabulary:
- s1 vocabulary size: 2^(s1_bits)
- s2 vocabulary size: 2^(s2_bits)

*Temporal Embedding*
Financial markets exhibit strong temporal patterns (intraday, weekly, monthly cycles). The temporal embedding layer encodes:
- Minute of hour (0-59)
- Hour of day (0-23)
- Day of week (0-6)
- Day of month (1-31)
- Month of year (1-12)

These embeddings are added to the token embeddings, providing the model with explicit temporal context.

*Decoder Transformer*
The core of Kronos is a stack of Transformer blocks with:
- Causal self-attention (only attending to past tokens)
- RoPE for position encoding
- SwiGLU feed-forward networks
- Pre-normalization with RMSNorm

The autoregressive nature means each prediction depends only on previous tokens, making it suitable for forecasting.

*Dual Prediction Head*
The model outputs predictions through two heads:
- s1 head: Predicts the next coarse token
- s2 head: Predicts the next fine token, conditioned on s1

This dependency-aware architecture ensures that fine-grained predictions are informed by coarse-grained context.

## Tokenization Pipeline Deep Dive

![Kronos Tokenization Pipeline](/assets/img/diagrams/kronos-tokenization-pipeline.svg)

### Understanding the Tokenization Pipeline

The tokenization pipeline is the critical first stage that enables Kronos to process financial data as a language. Let's break down each step:

**OHLCV Data Input**

The pipeline begins with raw market data in OHLCV format:
- **Open**: Price at the start of the time period
- **High**: Maximum price during the period
- **Low**: Minimum price during the period
- **Close**: Price at the end of the period
- **Volume**: Number of shares/contracts traded
- **Amount**: Total monetary value of trades

This six-dimensional data captures the complete state of the market at each time step. The model can process any frequency from tick data to daily bars.

**Normalization (z-score)**

Z-score normalization transforms each feature to have zero mean and unit variance:

```
x_normalized = (x - mean) / (std + epsilon)
```

This standardization is essential because:
1. Different assets have vastly different price scales (e.g., $0.50 vs $500)
2. Volume and amount are in different units than prices
3. The model needs consistent input distributions for stable training

The epsilon term (typically 1e-5) prevents division by zero for features with near-zero variance.

**Clipping (-5 to 5)**

After normalization, values are clipped to the range [-5, 5]. This serves several purposes:
- Removes extreme outliers that could destabilize training
- Limits the range of values the model needs to learn
- Acts as a form of robustness against data errors

The choice of 5 standard deviations captures approximately 99.99994% of a normal distribution, so legitimate data is rarely affected.

**Linear Embedding**

The clipped values are projected into a higher-dimensional embedding space (d_model). This linear transformation:
- Increases the representational capacity
- Allows the model to learn feature interactions
- Prepares the data for the Transformer encoder

**Encoder Layers**

The encoder consists of multiple Transformer blocks that:
- Learn temporal dependencies through self-attention
- Build hierarchical representations of market patterns
- Compress information for efficient quantization

Each layer applies:
1. RMSNorm for normalization
2. Multi-head attention with RoPE
3. Residual connection
4. Feed-forward network (SwiGLU)
5. Another residual connection

**Binary Spherical Quantizer (BSQ)**

The BSQ is the key innovation that enables discrete tokenization:

*Spherical Projection*
First, the encoder output is L2-normalized to lie on a unit hypersphere. This constrains the representation to a bounded space.

*Binary Quantization*
Each dimension is then quantized to either +1 or -1 based on its sign. This creates a binary code that can be interpreted as a token index.

*Hierarchical Token Generation*
The binary code is split into two parts:
- **s1 (Pre/Coarse)**: The first s1_bits represent coarse-grained information
- **s2 (Post/Fine)**: The remaining s2_bits capture fine-grained details

This hierarchy allows the model to separate broad market movements from specific price fluctuations.

**Entropy Regularization**

During training, the BSQ applies entropy regularization to ensure:
- All codebook entries are used (preventing collapse)
- The distribution remains diverse
- The quantization is efficient

The loss function includes:
- Commitment loss: Distance between input and quantized vectors
- Entropy penalty: Encourages uniform codebook usage

## Prediction Workflow

![Kronos Prediction Workflow](/assets/img/diagrams/kronos-prediction-workflow.svg)

### Understanding the Prediction Workflow

The prediction workflow shows how Kronos generates forecasts from historical data. This process is designed for practical deployment in quantitative trading systems.

**Historical K-line Data (Lookback Window)**

The workflow begins with historical market data. The lookback window typically contains:
- 400-512 time steps for Kronos-small/base
- Up to 2048 time steps for Kronos-mini

The choice of lookback window balances:
- Longer windows capture more historical context
- Shorter windows reduce computational cost
- Financial markets have limited "memory" - very old data may not be relevant

**Timestamps (Historical + Future)**

Temporal information is crucial for financial predictions. The model receives:
- Historical timestamps: When each past K-line occurred
- Future timestamps: When predictions should be made

This allows the model to:
- Learn intraday patterns (market open/close effects)
- Capture weekly cycles (weekend gaps)
- Understand monthly/quarterly patterns (earnings, expirations)

**KronosTokenizer (Encode)**

The tokenizer encodes the historical data into discrete tokens:
1. Applies normalization using statistics from the lookback window
2. Passes through the encoder layers
3. Quantizes to hierarchical tokens (s1, s2)

The encoding process is deterministic at inference time, ensuring reproducible results.

**Kronos Model (Autoregressive Inference)**

The core prediction engine operates autoregressively:
1. Takes the encoded tokens and timestamps
2. Generates predictions one step at a time
3. Each prediction conditions on all previous tokens

The autoregressive process:
- Starts with the historical context
- Generates pred_len future tokens
- Uses causal masking to prevent information leakage

**Nucleus Sampling (Temperature, Top-p)**

To generate diverse and realistic predictions, Kronos uses nucleus sampling:
- **Temperature (T)**: Controls randomness (T=1.0 is standard, lower is more deterministic)
- **Top-p**: Keeps only the top tokens whose cumulative probability exceeds p (typically 0.9)

This sampling strategy:
- Avoids low-probability tokens that are likely errors
- Maintains diversity in predictions
- Allows ensemble forecasting (sample_count > 1)

**KronosTokenizer (Decode)**

After generating token predictions, the tokenizer decodes them back to continuous values:
1. Converts token indices to binary codes
2. Projects back to the embedding space
3. Passes through decoder layers
4. Applies inverse normalization

The decoding process reverses the encoding, producing OHLCV predictions.

**Denormalization (Inverse z-score)**

The final step transforms predictions back to the original scale:
```
prediction = normalized_prediction * std + mean
```

This uses the same statistics computed during encoding, ensuring consistency.

**Forecasted K-lines (OHLCV Predictions)**

The output is a complete K-line forecast containing:
- Predicted Open, High, Low, Close prices
- Predicted Volume and Amount
- Timestamps for each prediction

These predictions can be used for:
- Trading signal generation
- Risk management
- Portfolio optimization
- Market analysis

## Model Components Architecture

![Kronos Model Components](/assets/img/diagrams/kronos-model-components.svg)

### Understanding the Model Components

This diagram details the internal architecture of the Kronos foundation model, showing how hierarchical tokens flow through the network.

**S1 and S2 Token Inputs**

The model accepts two token streams:
- **S1 Token IDs**: Coarse-grained tokens representing broad market movements
- **S2 Token IDs**: Fine-grained tokens capturing detailed price changes

This dual-token system is inspired by multi-scale representations in computer vision, where different resolutions capture different levels of detail.

**S1 and S2 Embeddings**

Each token type has its own embedding table:
- S1 vocabulary size: 2^(s1_bits) entries
- S2 vocabulary size: 2^(s2_bits) entries

The embedding tables learn to represent:
- Price level information (bull/bear markets)
- Volatility regimes (calm/turbulent)
- Trend patterns (momentum/reversal)

**Fusion Projection**

The fusion layer combines s1 and s2 embeddings:
```
combined = Linear(concat(s1_emb, s2_emb))
```

This allows the model to:
- Learn interactions between coarse and fine information
- Create a unified representation for the Transformer
- Maintain the hierarchical structure throughout processing

**Temporal Embedding**

The temporal embedding encodes time information:
- Minute embedding: 60 possible values
- Hour embedding: 24 possible values
- Weekday embedding: 7 possible values
- Day embedding: 31 possible values
- Month embedding: 12 possible values

These are summed together and added to the token embedding, providing explicit temporal context that helps the model understand:
- Market session effects (pre-market, regular, after-hours)
- Day-of-week patterns (Monday effect, Friday effect)
- Monthly patterns (options expiration, rebalancing)

**Transformer Block with RoPE**

The core processing unit uses:
- **Rotary Positional Embeddings (RoPE)**: Encodes relative positions, enabling better extrapolation to longer sequences
- **Multi-head Self-Attention**: Learns dependencies between all positions
- **SwiGLU Feed-Forward Network**: Non-linear transformation with gated activation
- **RMSNorm**: Efficient normalization without bias terms

The Transformer processes the sequence autoregressively, with causal masking ensuring each position only attends to previous positions.

**Dependency-Aware Layer (Cross-Attention)**

A key innovation in Kronos is the dependency-aware layer that conditions s2 predictions on s1:
- Uses cross-attention between s1 embeddings and Transformer context
- Allows fine-grained predictions to incorporate coarse-grained context
- Maintains the hierarchical relationship throughout decoding

This is crucial because:
- s1 tokens represent broader market direction
- s2 tokens should be consistent with the s1 context
- The dependency improves overall prediction coherence

**Dual Prediction Heads**

The model outputs predictions through two specialized heads:
- **S1 Head**: Predicts the next coarse token
- **S2 Head**: Predicts the next fine token (conditioned on s1)

Both heads are linear projections from the hidden dimension to their respective vocabulary sizes. During training, both are trained with cross-entropy loss.

**Hierarchical Token Output**

The final output is a pair of tokens (s1, s2) that together represent the predicted market state. This hierarchical representation:
- Captures multi-scale information
- Enables efficient encoding of complex patterns
- Provides interpretability (s1 shows overall direction, s2 shows details)

## Finetuning Pipeline

![Kronos Finetuning Pipeline](/assets/img/diagrams/kronos-finetuning-pipeline.svg)

### Understanding the Finetuning Pipeline

Kronos supports finetuning on custom datasets, allowing adaptation to specific markets or tasks. The pipeline is designed for production use with proper data handling.

**Raw Market Data (Qlib/CSV)**

The pipeline accepts data from multiple sources:
- **Qlib**: Microsoft's quantitative investment platform with built-in data loaders
- **CSV**: Standard format with columns for OHLCV and timestamps

Supported markets include:
- US equities (NYSE, NASDAQ)
- Chinese A-shares (Shanghai, Shenzhen)
- Cryptocurrency exchanges
- Forex markets
- Commodity futures

**Data Preprocessing (Split: Train/Val/Test)**

Proper data splitting is critical for financial ML:
- **Time-based splitting**: Never use future data for training
- **Train set**: Historical data for model learning
- **Validation set**: For hyperparameter tuning
- **Test set**: Final evaluation, must be chronologically after validation

The preprocessing includes:
- Handling missing values
- Aligning timestamps across instruments
- Computing derived features if needed

**Stage 1: Tokenizer Finetuning**

The tokenizer must be adapted to the target market:
- Different markets have different volatility profiles
- Price scales vary significantly
- Trading patterns differ by market type

The tokenizer training optimizes:
- **BSQ Loss**: Reconstruction quality + entropy regularization
- **Codebook Usage**: Ensures all tokens are utilized
- **Market-Specific Patterns**: Learns relevant quantization levels

Training uses multi-GPU with `torchrun`:
```bash
torchrun --standalone --nproc_per_node=NUM_GPUS finetune/train_tokenizer.py
```

**Stage 2: Predictor Finetuning**

After tokenizer adaptation, the predictor model is finetuned:
- Loads pretrained weights from base Kronos model
- Trains on the target market data
- Optimizes for forecasting accuracy

The predictor training:
- Uses autoregressive language modeling objective
- Applies teacher forcing for stable training
- Supports mixed precision for efficiency

**Backtesting (Strategy Evaluation)**

The finetuned model is evaluated through backtesting:
- Generates predictions on test set
- Converts predictions to trading signals
- Simulates portfolio performance
- Computes risk-adjusted returns

Key metrics include:
- Cumulative returns
- Sharpe ratio
- Maximum drawdown
- Win rate

**Performance Metrics (Returns, Sharpe, etc.)**

The final output includes comprehensive metrics:
- **Total Return**: Overall profit/loss
- **Annualized Return**: Return normalized to yearly rate
- **Sharpe Ratio**: Risk-adjusted return
- **Sortino Ratio**: Downside risk-adjusted return
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Win Rate**: Percentage of profitable trades

## Model Zoo

![Kronos Model Zoo](/assets/img/diagrams/kronos-model-zoo.svg)

### Understanding the Model Zoo

Kronos offers a family of models with varying capacities to suit different computational and application needs. All models are available on the Hugging Face Hub under the NeoQuasar organization.

**Kronos-mini (4.1M parameters)**

The smallest model is ideal for:
- Edge deployment and real-time inference
- Resource-constrained environments
- Rapid prototyping and experimentation
- Educational purposes

Key features:
- Context length: 2048 tokens
- Fastest inference speed
- Lowest memory footprint
- Uses Kronos-Tokenizer-2k

Despite its small size, Kronos-mini achieves competitive performance on many benchmarks, making it suitable for applications where latency is critical.

**Kronos-small (24.7M parameters)**

The balanced option for most use cases:
- Context length: 512 tokens
- Good trade-off between speed and accuracy
- Suitable for production deployment
- Uses Kronos-Tokenizer-base

This model is recommended for:
- General-purpose forecasting
- Research applications
- Medium-scale deployment

**Kronos-base (102.3M parameters)**

The standard model for high-quality predictions:
- Context length: 512 tokens
- Strong performance across benchmarks
- Suitable for demanding applications
- Uses Kronos-Tokenizer-base

Recommended for:
- Professional trading systems
- Risk management applications
- Research requiring high accuracy

**Kronos-large (499.2M parameters)**

The largest model (currently not open-sourced):
- Context length: 512 tokens
- State-of-the-art performance
- Requires significant computational resources
- Uses Kronos-Tokenizer-base

**Tokenizer Compatibility**

The tokenizers are designed for specific model sizes:
- **Kronos-Tokenizer-2k**: Optimized for mini model with 2048 context
- **Kronos-Tokenizer-base**: Standard tokenizer for small/base/large models

Using the correct tokenizer is essential for optimal performance.

**Hugging Face Hub Integration**

All models are accessible through the Hugging Face Hub:
```python
from model import Kronos, KronosTokenizer

# Load tokenizer
tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")

# Load model
model = Kronos.from_pretrained("NeoQuasar/Kronos-small")
```

This integration provides:
- Easy model loading and saving
- Version control for reproducibility
- Community contributions and variants
- Automatic model caching

## Installation

Getting started with Kronos is straightforward. The library requires Python 3.10+ and standard deep learning dependencies.

### Prerequisites

- Python 3.10 or higher
- PyTorch 2.0+ (recommended)
- CUDA-capable GPU (optional but recommended)

### Installation Steps

```bash
# Clone the repository
git clone https://github.com/shiyu-coder/Kronos.git
cd Kronos

# Install dependencies
pip install -r requirements.txt
```

The requirements include:
- `torch` - Deep learning framework
- `transformers` - Hugging Face integration
- `pandas` - Data manipulation
- `numpy` - Numerical operations
- `einops` - Tensor operations

### Quick Start

```python
from model import Kronos, KronosTokenizer, KronosPredictor

# Load from Hugging Face Hub
tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
model = Kronos.from_pretrained("NeoQuasar/Kronos-small")

# Initialize predictor
predictor = KronosPredictor(model, tokenizer, max_context=512)

# Prepare data
import pandas as pd
df = pd.read_csv("your_data.csv")
df['timestamps'] = pd.to_datetime(df['timestamps'])

# Define prediction window
lookback = 400
pred_len = 120

x_df = df.loc[:lookback-1, ['open', 'high', 'low', 'close', 'volume', 'amount']]
x_timestamp = df.loc[:lookback-1, 'timestamps']
y_timestamp = df.loc[lookback:lookback+pred_len-1, 'timestamps']

# Generate predictions
pred_df = predictor.predict(
    df=x_df,
    x_timestamp=x_timestamp,
    y_timestamp=y_timestamp,
    pred_len=pred_len,
    T=1.0,          # Temperature
    top_p=0.9,      # Nucleus sampling
    sample_count=1  # Number of samples
)

print(pred_df.head())
```

## Usage Examples

### Single Asset Prediction

```python
# Predict next 24 hours for BTC/USDT
pred_df = predictor.predict(
    df=historical_data,
    x_timestamp=historical_timestamps,
    y_timestamp=future_timestamps,
    pred_len=24,
    T=1.0,
    top_p=0.9,
    sample_count=1
)
```

### Batch Prediction

For processing multiple assets efficiently:

```python
# Prepare multiple datasets
df_list = [btc_data, eth_data, sol_data]
x_timestamp_list = [btc_ts, eth_ts, sol_ts]
y_timestamp_list = [btc_future, eth_future, sol_future]

# Batch prediction
pred_df_list = predictor.predict_batch(
    df_list=df_list,
    x_timestamp_list=x_timestamp_list,
    y_timestamp_list=y_timestamp_list,
    pred_len=24,
    T=1.0,
    top_p=0.9,
    sample_count=1,
    verbose=True
)
```

### Probabilistic Forecasting

Generate multiple forecast paths for uncertainty quantification:

```python
# Generate 10 forecast paths
pred_df = predictor.predict(
    df=historical_data,
    x_timestamp=historical_timestamps,
    y_timestamp=future_timestamps,
    pred_len=24,
    T=1.0,
    top_p=0.9,
    sample_count=10  # Average over 10 samples
)
```

## Features

| Feature | Description |
|---------|-------------|
| Hierarchical Tokenization | Novel s1/s2 token system captures multi-scale market patterns |
| Autoregressive Modeling | Decoder-only Transformer for sequential prediction |
| Temporal Embeddings | Explicit encoding of time features (minute, hour, day, month) |
| Multi-Exchange Training | Pre-trained on 45+ global exchanges |
| Probabilistic Forecasting | Nucleus sampling for uncertainty quantification |
| Batch Processing | Efficient parallel prediction for multiple assets |
| Finetuning Support | Complete pipeline for custom market adaptation |
| Hugging Face Integration | Easy model loading and sharing |

## Key Advantages

### 1. Financial-Specific Design

Unlike general-purpose time-series models, Kronos is built specifically for financial data:
- Handles OHLCV structure natively
- Captures market microstructure patterns
- Understands temporal dependencies in trading

### 2. Hierarchical Representation

The s1/s2 token system provides:
- Coarse-to-fine information flow
- Better generalization across market regimes
- Interpretable intermediate representations

### 3. Open Source

As the first open-source foundation model for financial K-lines:
- Transparent methodology
- Reproducible results
- Community contributions welcome

### 4. Production Ready

The codebase includes:
- WebUI for visualization
- Batch processing support
- Comprehensive documentation
- Example scripts

## Live Demo

A live demo is available at: [https://shiyu-coder.github.io/Kronos-demo/](https://shiyu-coder.github.io/Kronos-demo/)

The demo showcases:
- BTC/USDT 24-hour forecasts
- Interactive visualization
- Real-time predictions

## Citation

If you use Kronos in your research, please cite:

```bibtex
@misc{shi2025kronos,
      title={Kronos: A Foundation Model for the Language of Financial Markets}, 
      author={Yu Shi and Zongliang Fu and Shuo Chen and Bohan Zhao and Wei Xu and Changshui Zhang and Jian Li},
      year={2025},
      eprint={2508.02739},
      archivePrefix={arXiv},
      primaryClass={q-fin.ST},
      url={https://arxiv.org/abs/2508.02739}, 
}
```

## Conclusion

Kronos represents a significant step forward in applying foundation models to financial markets. By treating K-line sequences as a language and applying hierarchical tokenization, it achieves state-of-the-art performance on forecasting tasks while maintaining interpretability.

The open-source release includes:
- Multiple model sizes for different use cases
- Complete finetuning pipeline
- WebUI for easy exploration
- Comprehensive documentation

Whether you're building trading systems, conducting research, or exploring financial ML, Kronos provides a powerful foundation for your work.

## Related Posts

- [TimesFM: Time Series Foundation Model](/TimesFM-Time-Series-Foundation-Model/)
- [AI Hedge Fund: Multi-Agent Investment Analysis](/AI-Hedge-Fund-Multi-Agent-Investment-Analysis/)
- [TradingAgents: AI-Powered Trading Strategies](/TradingAgents-AI-Powered-Trading-Strategies/)