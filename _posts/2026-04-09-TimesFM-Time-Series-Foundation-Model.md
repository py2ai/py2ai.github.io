---
layout: post
title: "TimesFM: Google's Foundation Model for Time Series Forecasting"
description: "Explore TimesFM, Google's pretrained decoder-only foundation model for zero-shot time-series forecasting with 16K context length and probabilistic predictions."
date: 2026-04-09
header-img: "img/post-bg.jpg"
permalink: /TimesFM-Time-Series-Foundation-Model/
featured-img: ai-coding-frameworks/ai-coding-frameworks
tags:
  - Time Series
  - Foundation Model
  - Machine Learning
  - Forecasting
  - Google Research
author: "PyShine"
---

# TimesFM: Google's Foundation Model for Time Series Forecasting

Time series forecasting is a critical component in numerous domains, from financial market predictions to supply chain optimization, energy demand planning, and weather forecasting. Traditional approaches require extensive domain expertise and model tuning for each specific use case. TimesFM (Time Series Foundation Model), developed by Google Research, represents a paradigm shift in how we approach time series forecasting by introducing a pretrained foundation model capable of zero-shot predictions across diverse domains.

TimesFM is a decoder-only transformer model pretrained on a massive corpus of time series data, enabling it to make accurate forecasts on unseen datasets without any task-specific training. This zero-shot capability is particularly valuable for organizations that need to make predictions on new time series without historical data for training. The model supports context lengths up to 16,000 time steps and can generate probabilistic forecasts with quantile predictions, providing not just point estimates but also uncertainty bounds.

The key innovation of TimesFM lies in its patched decoder-only architecture, which processes time series data in patches similar to how vision transformers process images. This approach allows the model to efficiently capture both local patterns and long-range dependencies in time series data. The model was introduced in the paper "A decoder-only foundation model for time-series forecasting" published at ICML 2024, and has since become one of the most influential open-source time series models with over 15,500 stars on GitHub.

## Architecture Overview

![TimesFM Architecture](/assets/img/diagrams/timesfm-architecture.svg)

### Understanding the TimesFM Architecture

The TimesFM architecture represents a significant departure from traditional time series models by adopting a decoder-only transformer design inspired by large language models. This architecture enables the model to learn universal time series patterns that transfer across domains without requiring task-specific fine-tuning.

**Input Tokenizer and Patching:**

The input processing begins with a sophisticated tokenizer that divides the time series into patches. Unlike traditional approaches that process each time step individually, TimesFM groups consecutive time steps into patches, similar to how vision transformers process image patches. This patching mechanism serves multiple purposes: it reduces the sequence length for computational efficiency, enables the model to capture local patterns within each patch, and provides a natural way to handle varying frequencies in time series data.

The tokenizer applies per-patch normalization, which is crucial for handling time series with different scales and distributions. This normalization ensures that the model focuses on patterns rather than absolute values, making it robust to the wide variety of scales encountered in real-world time series. The patch size is configurable, allowing users to balance between fine-grained temporal resolution and computational efficiency.

**20-Layer Transformer Stack:**

At the core of TimesFM lies a 20-layer transformer decoder stack. Each layer consists of multi-head self-attention mechanisms and feed-forward networks. The self-attention mechanism enables the model to capture long-range dependencies by allowing each position to attend to all previous positions in the sequence. This is particularly important for time series forecasting where patterns from distant history can influence future values.

The transformer uses RoPE (Rotary Position Embedding) for position encoding, which has become the standard for modern transformer architectures. RoPE provides relative position information that naturally extends to longer sequences, making it ideal for handling varying context lengths. Additionally, the architecture incorporates QK (Query-Key) normalization, which stabilizes training and improves convergence by preventing attention scores from becoming too large.

**Output Projections:**

The output layer projects the transformer representations back to the time series space. TimesFM 2.5 introduces an optional continuous quantile head that provides probabilistic forecasts. This head outputs predictions at multiple quantiles (10th, 20th, ..., 90th percentiles), giving users a full distribution of possible outcomes rather than just a point forecast. The quantile predictions are essential for risk assessment and decision-making under uncertainty.

**Key Architectural Innovations:**

The architecture includes several innovations that distinguish it from standard transformer models. First, it removes the frequency indicator present in earlier versions, simplifying the input pipeline while maintaining strong performance across different time series frequencies. Second, it introduces flip invariance for certain types of time series, allowing the model to learn symmetric patterns. Third, it implements quantile crossing fixes to ensure that predicted quantiles are properly ordered (e.g., the 90th percentile is always higher than the 10th percentile).

**Practical Implications:**

This architecture design enables TimesFM to handle a wide variety of time series without domain-specific tuning. The patched approach reduces computational complexity from O(n^2) to O((n/patch_size)^2), making it feasible to process long sequences. The decoder-only design allows for autoregressive generation of forecasts, where predictions are made one step at a time, conditioning on previous predictions.

## Model Versions

![TimesFM Model Versions](/assets/img/diagrams/timesfm-model-versions.svg)

### Evolution of TimesFM: From 1.0 to 2.5

The TimesFM model has undergone significant evolution since its initial release, with each version introducing substantial improvements in capability, efficiency, and usability. Understanding these versions helps practitioners choose the right model for their specific use case and provides insight into the direction of time series foundation model development.

**TimesFM 1.0 - The Foundation:**

The original TimesFM 1.0 introduced the core concept of a pretrained time series foundation model. It demonstrated that a single model trained on diverse time series could generalize to unseen datasets through zero-shot forecasting. The model used a patched decoder architecture with frequency indicators to handle different time series granularities (hourly, daily, weekly, etc.). While groundbreaking, this version had limitations in context length (512 time steps) and required explicit frequency specification.

The 1.0 release established the viability of the foundation model approach for time series, showing competitive results against traditional statistical methods like ARIMA and exponential smoothing, as well as against specialized deep learning models trained on specific datasets. This validation opened the door for subsequent improvements.

**TimesFM 2.0 - Scaling Up:**

TimesFM 2.0 represented a significant scale-up with 500 million parameters, substantially increasing model capacity. This version expanded the context length to 2,048 time steps, enabling the model to capture longer-range dependencies and seasonal patterns. The larger model size improved forecast accuracy across benchmarks but came with increased computational requirements.

Version 2.0 also introduced improved handling of covariates - external variables that influence the time series. The model could incorporate static features (constant over time) and dynamic features (changing over time) to improve forecast accuracy when such information was available. This made the model more practical for real-world applications where domain knowledge often provides additional context.

**TimesFM 2.5 - Efficiency and Capability:**

The latest version, TimesFM 2.5, represents a remarkable achievement in model efficiency. Despite reducing parameters from 500M to 200M, it achieves better performance through architectural improvements and training optimizations. This 60% reduction in model size makes the model more accessible for deployment on standard hardware while improving inference speed.

The most significant advancement in 2.5 is the expansion of context length to 16,000 time steps - an 8x increase from version 2.0. This enables the model to capture very long-range patterns, making it suitable for applications with extensive historical data or long seasonal cycles. The model also introduces an optional 30M parameter quantile head for probabilistic forecasting, providing uncertainty estimates essential for decision-making.

**Key Improvements in 2.5:**

- **Simplified Input Pipeline:** The frequency indicator has been removed, with the model automatically learning to handle different frequencies from the data itself. This simplifies the user experience and reduces potential errors from incorrect frequency specification.

- **Continuous Quantile Head:** The new quantile head provides smooth quantile predictions across the distribution, avoiding the discrete quantization issues of earlier approaches. Users can query any quantile, not just predefined percentiles.

- **Enhanced Normalization:** Improved input normalization handles edge cases better, including time series with zeros, negative values, and extreme outliers.

- **Flip Invariance:** For appropriate time series types, the model can learn symmetric patterns more efficiently by treating forward and backward sequences equivalently.

**Migration Considerations:**

Users migrating from earlier versions should note that TimesFM 1.0 and 2.0 code is archived in the `v1` subdirectory. To use older models, install `timesfm==1.3.0`. The new API in version 2.5 is not backward compatible, requiring code changes for existing implementations. However, the improved capabilities and efficiency make migration worthwhile for most applications.

## Zero-Shot Forecasting Workflow

![Forecasting Workflow](/assets/img/diagrams/timesfm-forecasting-workflow.svg)

### Understanding Zero-Shot Forecasting with TimesFM

Zero-shot forecasting represents the core capability that distinguishes TimesFM from traditional time series models. The ability to make accurate predictions on completely new time series without any training or fine-tuning is transformative for practical applications. This section explains how the zero-shot workflow operates and why it matters for practitioners.

**What is Zero-Shot Forecasting?**

Traditional time series forecasting requires training a model on historical data specific to the target series. This process involves data preparation, model selection, hyperparameter tuning, and validation - often requiring significant time and expertise. Zero-shot forecasting eliminates this entire pipeline by leveraging a pretrained model that has already learned universal time series patterns from massive datasets.

TimesFM achieves zero-shot capability through pretraining on a diverse corpus of time series from multiple domains including finance, retail, energy, weather, and more. During pretraining, the model learns to recognize and extrapolate patterns such as trends, seasonality, cycles, and irregularities that are common across domains. When presented with a new time series, the model applies these learned patterns to generate forecasts.

**The Forecasting Pipeline:**

The zero-shot workflow begins with input preparation. Users provide a time series array and specify the forecast horizon - the number of future time steps to predict. The model automatically handles preprocessing including normalization, patching, and feature extraction. No manual feature engineering or data transformation is required beyond ensuring the input is a numerical array.

The model then processes the input through its transformer layers, generating representations that capture both local patterns (within patches) and global patterns (across the entire context). The decoder autoregressively generates forecasts, with each prediction conditioning on previous predictions. This approach ensures temporal coherence in the output.

**Probabilistic Forecasts with Quantiles:**

A key feature of TimesFM is its ability to provide probabilistic forecasts through quantile predictions. Instead of outputting a single point forecast, the model predicts multiple quantiles (typically the 10th through 90th percentiles in 10% increments). This provides a complete distribution of possible outcomes, essential for risk management and decision-making.

The quantile predictions enable practitioners to:
- Assess forecast uncertainty and confidence intervals
- Plan for best-case and worst-case scenarios
- Calculate risk metrics like Value at Risk (VaR) for financial applications
- Make robust decisions under uncertainty

**Configuration Options:**

TimesFM provides several configuration options to customize forecasts:

- **max_context:** The maximum number of historical time steps to use (up to 16,384). Longer contexts capture more history but require more computation.

- **max_horizon:** The maximum forecast horizon (up to 1,024 with the quantile head). Longer horizons have higher uncertainty.

- **normalize_inputs:** Whether to apply automatic normalization. Recommended for most use cases.

- **use_continuous_quantile_head:** Enable probabilistic forecasts. Set to False for point forecasts only.

- **force_flip_invariance:** Apply flip invariance for symmetric patterns. Useful for certain time series types.

- **infer_is_positive:** Automatically detect and enforce positive-only forecasts. Prevents negative predictions for quantities that cannot be negative.

- **fix_quantile_crossing:** Ensure quantiles are properly ordered. Prevents illogical predictions where lower quantiles exceed higher ones.

**Practical Workflow Example:**

A typical zero-shot forecasting workflow involves:
1. Loading historical time series data
2. Initializing the TimesFM model from pretrained weights
3. Configuring forecast parameters (horizon, context length)
4. Calling the forecast method with the input data
5. Extracting point forecasts and quantile predictions
6. Visualizing results with confidence intervals

This entire process can be completed in minutes, compared to hours or days for traditional model development. The simplicity and speed make TimesFM particularly valuable for organizations with many time series to forecast or for rapid prototyping and experimentation.

## XReg Covariate Support

![XReg Covariates](/assets/img/diagrams/timesfm-xreg-covariates.svg)

### Understanding External Regressors in TimesFM

Time series forecasting often benefits from incorporating external variables that influence the target series. TimesFM's XReg (External Regressor) module provides sophisticated support for covariates, enabling more accurate forecasts when additional contextual information is available. This capability bridges the gap between pure time series extrapolation and multivariate forecasting.

**Types of Covariates:**

TimesFM supports two main categories of covariates, each serving different purposes in the forecasting pipeline:

**Dynamic Numerical Covariates:** These are time-varying numerical features that provide additional context for each time step. Examples include temperature for energy demand forecasting, promotional spend for sales forecasting, or macroeconomic indicators for financial forecasting. The model learns how these external factors correlate with the target series and uses this information to improve predictions.

**Dynamic Categorical Covariates:** These are time-varying categorical features that capture discrete states or conditions. Examples include day of week, month of year, holiday indicators, or event flags. The model learns embeddings for each category, capturing how different states affect the time series behavior.

**Static Categorical Covariates:** These are time-invariant features that describe characteristics of the series itself. Examples include product category, geographic region, store type, or customer segment. Static covariates help the model differentiate between series with different fundamental behaviors.

**Two Integration Modes:**

TimesFM offers two distinct modes for incorporating covariates, each with different computational characteristics:

**"xreg + timesfm" Mode (Pre-processing):**

In this mode, covariates are processed first by a dedicated XReg model that learns representations of the external variables. The output is then combined with the time series input before being fed into TimesFM. This approach is computationally efficient because the covariate processing happens once, and the representations can be cached for repeated forecasts.

The pre-processing mode is suitable when:
- Covariates are available for the entire forecast horizon
- The relationship between covariates and target is relatively stable
- Computational efficiency is important

**"timesfm + xreg" Mode (Post-processing):**

In this mode, TimesFM first generates baseline forecasts without covariates, and then the XReg module adjusts these forecasts based on covariate information. This approach allows for more flexible integration and can handle cases where covariates are only partially available.

The post-processing mode is suitable when:
- Covariates may be missing for some time periods
- The covariate influence varies significantly over time
- You want to compare forecasts with and without covariates

**Implementation Details:**

The XReg module uses a separate neural network to process covariates. For numerical covariates, the network applies normalization and feature extraction. For categorical covariates, it learns embeddings that capture the semantic relationships between categories. The static covariates are broadcast across the time dimension and combined with dynamic features.

The covariate representations are integrated with the time series embeddings through attention mechanisms, allowing the model to learn when and how much to weight different covariates. This attention-based integration is more flexible than simple concatenation and can capture complex interactions between the target series and external factors.

**Practical Applications:**

Consider a retail sales forecasting scenario:
- **Dynamic Numerical:** Price, promotional discount percentage, inventory level
- **Dynamic Categorical:** Day of week, holiday flag, promotion type
- **Static Categorical:** Store format (superstore, convenience, online), region, product category

By incorporating these covariates, the model can:
- Adjust forecasts for upcoming promotions
- Account for holiday effects on sales
- Differentiate between stores with different baseline sales patterns
- Capture price elasticity effects

**Using XReg in Practice:**

To use covariates with TimesFM, users need to:
1. Prepare covariate data aligned with the time series
2. Specify covariate types (numerical/categorical, static/dynamic)
3. Choose the integration mode based on use case
4. Ensure covariates are available for the forecast horizon (for pre-processing mode)

The XReg module handles normalization and embedding internally, requiring minimal preprocessing from users. This makes it practical for real-world applications where covariate data may come from different sources and in different formats.

## Use Cases

![Use Cases](/assets/img/diagrams/timesfm-use-cases.svg)

### Real-World Applications of TimesFM

TimesFM's zero-shot forecasting capability makes it applicable across a wide range of industries and use cases. The model's ability to handle diverse time series without domain-specific training enables rapid deployment and experimentation. This section explores key application areas and how TimesFM addresses their specific challenges.

**Sales and Demand Forecasting:**

Retail and e-commerce businesses rely heavily on accurate demand forecasts for inventory management, staffing, and supply chain optimization. TimesFM excels in this domain by capturing complex patterns including:
- Weekly and seasonal cycles (weekend spikes, holiday peaks)
- Trend components (growth, decline)
- Promotional effects (when combined with covariates)
- External factors (economic conditions, weather)

The zero-shot capability is particularly valuable for new products or seasonal items with limited historical data. The model can leverage patterns learned from similar products to make reasonable forecasts even with minimal history. This addresses a common challenge in retail where new product introductions are frequent.

**Anomaly Detection:**

Time series anomaly detection involves identifying unusual patterns that deviate from expected behavior. TimesFM enables a forecasting-based approach to anomaly detection:
1. Generate forecasts with prediction intervals
2. Compare actual values to predicted ranges
3. Flag points that fall outside expected bounds

The quantile predictions provide natural anomaly thresholds. Points falling outside the 10th-90th percentile range are potential anomalies, with severity determined by how far they deviate from predictions. This approach is more robust than simple threshold-based methods because it accounts for temporal patterns and trends.

**Temperature and Weather Forecasting:**

Weather forecasting is inherently a time series problem, with temperature, humidity, wind speed, and other variables measured over time. TimesFM can model these series, particularly for:
- Short-term temperature predictions
- Precipitation probability
- Wind speed forecasting
- Energy-related weather variables (solar irradiance, heating/cooling degree days)

The model's ability to handle long contexts (up to 16K time steps) allows it to capture multi-year seasonal patterns and climate cycles. When combined with meteorological covariates, it can produce forecasts competitive with specialized weather models for certain applications.

**Sensor Data Prediction:**

Industrial IoT applications generate massive amounts of sensor data that require forecasting for:
- Predictive maintenance (predicting equipment failures)
- Process optimization (predicting quality metrics)
- Resource planning (predicting resource consumption)
- Safety monitoring (predicting hazardous conditions)

TimesFM's zero-shot capability is valuable here because industrial environments often have hundreds or thousands of sensors. Training individual models for each sensor is impractical. With TimesFM, a single model can forecast across all sensors, with covariates capturing equipment-specific characteristics.

**Financial Time Series:**

Financial markets generate numerous time series including:
- Stock prices and returns
- Trading volumes
- Volatility indices
- Economic indicators
- Portfolio metrics

TimesFM can model these series for:
- Price forecasting (with appropriate caution about market efficiency)
- Volatility prediction
- Risk metric forecasting
- Portfolio optimization inputs

The quantile predictions are particularly valuable for risk management, providing Value at Risk (VaR) estimates and confidence intervals for financial decisions. However, users should be aware that financial time series have unique characteristics (non-stationarity, regime changes) that may require additional considerations.

**Healthcare and Epidemiology:**

Time series forecasting has critical applications in healthcare:
- Disease outbreak prediction
- Hospital resource planning
- Patient volume forecasting
- Epidemic trajectory modeling

TimesFM's ability to handle diverse series makes it applicable across different healthcare contexts. The model can forecast patient admissions, resource utilization, or disease spread patterns, supporting proactive healthcare management.

**Energy and Utilities:**

Energy demand forecasting is essential for grid management and utility operations:
- Electricity load forecasting
- Renewable energy generation prediction
- Peak demand planning
- Price forecasting in energy markets

TimesFM captures the complex patterns in energy consumption including daily cycles, weekly patterns, seasonal variations, and weather dependencies. When combined with weather covariates, it can produce highly accurate forecasts for grid planning.

**Key Considerations Across Use Cases:**

When applying TimesFM to any domain, consider:
- **Context Length:** Ensure sufficient historical data for the model to capture relevant patterns
- **Forecast Horizon:** Longer horizons have higher uncertainty; use quantile predictions to assess confidence
- **Covariates:** Identify external factors that influence your series and incorporate them through XReg
- **Evaluation:** Always validate forecasts on held-out data before deployment
- **Domain Knowledge:** While TimesFM is zero-shot, domain expertise still helps in interpreting results and identifying anomalies

## Installation and Usage

Getting started with TimesFM is straightforward. The model is available as a Python package and can be installed with standard package managers.

### Installation

```bash
# Clone the repository
git clone https://github.com/google-research/timesfm.git
cd timesfm

# Create a virtual environment using uv
uv venv

# Activate the environment
source .venv/bin/activate

# Install the package with PyTorch backend
uv pip install -e .[torch]

# Or with Flax backend
uv pip install -e .[flax]

# For covariate support (XReg)
uv pip install -e .[xreg]
```

### Basic Usage

```python
import torch
import numpy as np
import timesfm

# Set precision for better performance
torch.set_float32_matmul_precision("high")

# Load the pretrained model
model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
    "google/timesfm-2.5-200m-pytorch"
)

# Configure the model
model.compile(
    timesfm.ForecastConfig(
        max_context=1024,        # Maximum historical context
        max_horizon=256,         # Maximum forecast horizon
        normalize_inputs=True,   # Automatic normalization
        use_continuous_quantile_head=True,  # Enable quantile predictions
        force_flip_invariance=True,   # Learn symmetric patterns
        infer_is_positive=True,       # Enforce positive forecasts
        fix_quantile_crossing=True,   # Ensure ordered quantiles
    )
)

# Generate forecasts
point_forecast, quantile_forecast = model.forecast(
    horizon=24,  # Predict 24 steps ahead
    inputs=[
        np.sin(np.linspace(0, 20, 200)),  # Example sine wave
        np.linspace(0, 1, 100),           # Example linear trend
    ],
)

# Access results
print(f"Point forecast shape: {point_forecast.shape}")  # (2, 24)
print(f"Quantile forecast shape: {quantile_forecast.shape}")  # (2, 24, 10)
```

### Using with Covariates

```python
# For covariate support, use the XReg-enabled model
import timesfm

model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
    "google/timesfm-2.5-200m-pytorch"
)

# Define covariates
dynamic_numerical = [...]  # Shape: (num_series, time_steps, num_features)
dynamic_categorical = [...]  # Shape: (num_series, time_steps, num_features)
static_categorical = [...]  # Shape: (num_series, num_features)

# Forecast with covariates
point_forecast, quantile_forecast = model.forecast(
    horizon=24,
    inputs=time_series_data,
    dynamic_numerical_covariates=dynamic_numerical,
    dynamic_categorical_covariates=dynamic_categorical,
    static_categorical_covariates=static_categorical,
)
```

## Performance Benchmarks

TimesFM has been extensively evaluated on standard time series benchmarks, demonstrating strong zero-shot performance across diverse datasets.

### GIFT-Eval Benchmark

TimesFM 2.5 achieves competitive results on the GIFT-Eval benchmark, which evaluates generalist forecasting models on diverse time series:

| Model | MASE | CRPS | Parameters |
|-------|------|------|-------------|
| TimesFM 2.5 | 0.89 | 0.42 | 200M |
| TimesFM 2.0 | 0.92 | 0.45 | 500M |
| TimeGPT | 0.95 | 0.48 | - |
| Statistical Baseline | 1.00 | 0.50 | - |

The Mean Absolute Scaled Error (MASE) and Continuous Ranked Probability Score (CRPS) metrics show that TimesFM 2.5 outperforms its larger predecessor (TimesFM 2.0) despite having fewer parameters, demonstrating the efficiency improvements in the new architecture.

### Key Performance Characteristics

- **Zero-Shot Accuracy:** TimesFM achieves within 10% of specialized models trained on target datasets, without any task-specific training.

- **Long Context Advantage:** The 16K context length provides significant improvements for series with long seasonal patterns or slow trends.

- **Probabilistic Calibration:** The quantile predictions are well-calibrated, meaning the 90% prediction interval contains the actual value approximately 90% of the time.

- **Computational Efficiency:** The 200M parameter model can process thousands of time series per second on modern GPUs, making it suitable for large-scale applications.

## Conclusion

TimesFM represents a significant advancement in time series forecasting, demonstrating that foundation models can achieve strong zero-shot performance on diverse forecasting tasks. The model's architecture - combining patched processing, decoder-only transformers, and quantile predictions - provides an effective framework for universal time series modeling.

Key takeaways from TimesFM:

- **Zero-Shot Capability:** Make accurate forecasts on new time series without training, dramatically reducing time-to-deployment.

- **Probabilistic Predictions:** Quantile forecasts provide uncertainty estimates essential for decision-making under uncertainty.

- **Long Context Support:** Up to 16K time steps enable capturing long-range dependencies and seasonal patterns.

- **Covariate Integration:** XReg support allows incorporating external factors for improved forecast accuracy.

- **Efficient Architecture:** 200M parameters deliver strong performance while remaining practical for deployment.

The availability of TimesFM as an open-source model, along with its integration into Google BigQuery, makes it accessible for both research and production applications. As time series foundation models continue to evolve, TimesFM provides a strong baseline and practical tool for practitioners across industries.

## Resources

- [TimesFM GitHub Repository](https://github.com/google-research/timesfm)
- [TimesFM Paper - ICML 2024](https://arxiv.org/abs/2310.10688)
- [Google Research Blog](https://research.google/blog/a-decoder-only-foundation-model-for-time-series-forecasting/)
- [TimesFM Hugging Face Collection](https://huggingface.co/collections/google/timesfm-release-66e4be5fdb56e960c1e482a6)
- [TimesFM in BigQuery](https://cloud.google.com/bigquery/docs/timesfm-model)