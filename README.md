Reimplementation of FUTURIST and Extension with SCMHSA
Overview

This project implements a miniaturized version of FUTURIST, a multimodal visual sequence transformer introduced in CVPR 2025, and evaluates the impact of replacing its standard Multi-Head Self-Attention (MHSA) mechanism with Semantic Concentration Multi-Head Self-Attention (SCMHSA), proposed in a 2025 research paper.

The objective is to study semantic future prediction—forecasting future semantic segmentation masks and depth maps from video—under realistic computational constraints, while isolating and quantifying the effects of semantic dilution in transformer attention mechanisms.

Two architectures are implemented and compared:

Baseline: Mini-FUTURIST with standard MHSA

Experimental: Mini-FUTURIST with SCMHSA replacing MHSA

Both models are trained and evaluated under identical hyperparameters, datasets, and compute environments to ensure a controlled comparison.

Motivation

Predicting future states of an environment is a core requirement for autonomous systems. While many video prediction methods attempt to generate future RGB frames, this approach is computationally expensive and unnecessarily complex.

Instead, this project focuses on semantic future prediction, which directly forecasts:

Semantic segmentation maps

Depth maps

This formulation:

Avoids modeling low-level pixel details

Emphasizes task-relevant scene understanding

Produces outputs directly useful for downstream autonomous decision-making

However, transformer-based video models rely heavily on MHSA, which splits embeddings across attention heads, potentially causing semantic dilution. SCMHSA addresses this by allowing each attention head to process the full embedding, preserving complete semantic context.

Project Objectives

Miniaturize FUTURIST

Implement a computationally feasible version of the FUTURIST architecture

Retain core principles: VAE-free tokenization, multimodal fusion, masked visual modeling

Evaluate SCMHSA vs. MHSA

Replace MHSA blocks with SCMHSA blocks

Quantify the effect of semantic concentration on future prediction performance

Establish a Controlled Benchmark

Ensure identical training configurations for both models

Isolate architectural effects from training noise or data bias

Architecture Overview
Baseline: Mini-FUTURIST

The baseline model is a scaled-down version of FUTURIST designed to operate under memory constraints while preserving its conceptual design.

1. VAE-Free Hierarchical Tokenization

Instead of a VQ-VAE tokenizer, the model uses continuous embeddings:

Per-Pixel Embedding

Each pixel is projected into a low-dimensional embedding space via a trainable matrix

Patch Flattening

Pixel embeddings are grouped into patches

Patches are flattened and linearly projected into the transformer hidden dimension

This produces a sequence of tokens suitable for transformer processing without discrete codebooks.

2. Multimodal Fusion

The model operates on two semantic modalities:

Semantic segmentation

Depth

Fusion strategy:

Early fusion via concatenation

Tokens from both modalities are concatenated along the embedding dimension

Enables immediate cross-modal interaction within self-attention

3. Masked Spatio-Temporal Transformer

Transformer encoder backbone

Operates under the Masked Visual Modeling (MVM) paradigm

Context frames are visible

Future frame tokens are fully masked

Model learns to reconstruct masked future tokens using temporal context

Experimental: FUTURIST + SCMHSA

The experimental architecture replaces all MHSA blocks with SCMHSA blocks.

SCMHSA Mechanism

In standard MHSA:

Embedding dimension d is split across N heads (d/N per head)

Each head processes only partial semantic information

In SCMHSA:

Each attention head receives the full embedding

Prevents semantic fragmentation

Produces a higher-dimensional concatenated output

Dimensionality Projection

Because SCMHSA increases intermediate dimensionality:

A learnable projection matrix Wₒ compresses concatenated outputs back to dimension d

Acts as a semantic filter, retaining relevant aggregated information

Dataset Construction
Source Data

5 First-Person View (FPV) driving videos

Recorded in:

San Francisco

Houston

Washington, D.C.

Ensures geographic and visual diversity

Data Extraction

Sampling rate: 2 frames per second

Total dataset size: 42,000 frames

Semantic Annotation

For each extracted frame:

Semantic segmentation maps generated using Mask2Former

Depth maps generated using DepthAnythingV2

Storage

All processed data stored in Google Cloud Storage

Enables high-throughput loading during training

Training Setup
Compute Environment

Google Colab

NVIDIA A100 GPU

Memory limitations necessitated:

Miniaturized model

Small batch size

Model Configuration
Parameter	Value
Image Resolution	216 × 384
Transformer Layers	4
Attention Heads	4
Patch Size	12
Sequence Length	5 frames
Total Parameters	~3.33M
Optimization

Optimizer: AdamW

Weight Decay: 1e-2

Learning Rate: 2e-4

Scheduler: Cosine Annealing

Tmax = 25

η_min = 1e-6

Training Duration: 25 epochs

Both models use identical hyperparameters.

Evaluation Metrics

Metrics computed only on valid pixels.

Segmentation

mIoU – Mean Intersection over Union

MO-mIoU – Moving Object mIoU

Depth

AbsRel – Absolute Relative Error

δ₁ Accuracy – Percentage of pixels with error < 1.25×

Results
Quantitative Performance
Model	Future Segmentation mIoU
Mini-FUTURIST (MHSA)	0.3044
FUTURIST + SCMHSA	0.2481

Both models converged within 25 epochs, but the baseline consistently outperformed SCMHSA under these constraints.

Qualitative Observations

Baseline FUTURIST

Preserved large structural elements (road, sky, buildings)

Reasonable depth gradients

Struggled with fine object details

SCMHSA Variant

Noisier segmentation outputs

Less distinct class boundaries

Blockier depth transitions

Discussion

The hypothesis that SCMHSA would outperform MHSA was not supported in this experiment.

Three primary factors were identified:

Insufficient Training Horizon

Original FUTURIST trained for thousands of epochs

SCMHSA likely requires longer training to converge

Model Size Constraints

SCMHSA originally evaluated at ~42M parameters

Compression via Wₒ in a 3.3M-parameter model may cause information loss

Head Integrability

Without additional losses (e.g., semantic similarity loss), SCMHSA heads may learn redundant features

Limitations

Severe GPU memory constraints

Reduced image resolution

Short training schedule (25 epochs)

Miniaturized model capacity

Future Work

Scale to larger models (≥40M parameters)

Train for ≥800 epochs

Reduce patch size for finer spatial resolution

Integrate Semantic Similarity Loss

Evaluate on non-driving datasets

Conclusion

This project establishes a complete multimodal semantic future prediction pipeline, from dataset generation to transformer implementation and evaluation.

While SCMHSA did not outperform MHSA under constrained settings, the results highlight the interaction between architectural complexity, training duration, and model capacity, providing a foundation for future exploration of efficient attention mechanisms in video forecasting.
