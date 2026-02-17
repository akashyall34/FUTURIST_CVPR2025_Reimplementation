# Reimplementation of FUTURIST and Extension with SCMHSA

<img width="414" height="230" alt="image" src="https://github.com/user-attachments/assets/b1f07b35-8c56-4c1a-b8e4-253f983ea443" /> <img width="415" height="231" alt="image" src="https://github.com/user-attachments/assets/08848bb3-b52d-48be-aabc-3b3c560f914e" />

## Overview

I built a scaled-down version of **FUTURIST** (a multimodal visual sequence transformer from CVPR 2025) and tested whether replacing standard Multi-Head Self-Attention with **SCMHSA** (Semantic Concentration Multi-Head Self-Attention) improves performance.

The goal: predict future semantic segmentation and depth maps from driving video, while working within tight computational constraints.

**Two models:**
- **Baseline**: Mini-FUTURIST with standard MHSA
- **Experimental**: Mini-FUTURIST with SCMHSA

Both trained with identical settings to isolate the effect of the attention mechanism.

---

## Motivation

Most video prediction models try to generate future RGB frames pixel-by-pixel. This is expensive and kind of overkill for autonomous systems that just need to understand *what's where* and *how far away things are*.

So instead of predicting pixels, I predict:
- **Semantic segmentation** (what each region is: road, car, sidewalk, etc.)
- **Depth maps** (how far away things are)

Why this is better:
- Skip the low-level pixel noise
- Focus on what actually matters for planning and navigation
- Get outputs that downstream systems can use directly

The catch: transformers use Multi-Head Self-Attention (MHSA), which splits embeddings across heads. This might dilute semantic information. SCMHSA tries to fix this by giving each head the *full* embedding instead of a slice.

---

## What I Did

### 1. Built Mini-FUTURIST
- Shrunk down the original architecture to run on Colab
- Kept the important parts: VAE-free tokenization, multimodal fusion, masked visual modeling

### 2. Tested SCMHSA vs. MHSA
- Swapped out all MHSA blocks with SCMHSA
- Measured whether semantic concentration actually helps

### 3. Controlled Comparison
- Same hyperparameters, same data, same training setup
- Only difference: the attention mechanism

---

## Architecture

### Baseline: Mini-FUTURIST

I scaled down FUTURIST to fit in Colab's memory while keeping the core design intact.

#### 1. VAE-Free Tokenization

Instead of using a VQ-VAE (which adds complexity), I went with continuous embeddings:

- Project each pixel into a low-dim embedding
- Group pixels into patches
- Flatten and linearly project into transformer dimension

No discrete codebooks needed—just straight embeddings.

#### 2. Multimodal Fusion

Two inputs: segmentation and depth.

I concatenate them early (along the embedding dimension) so the attention mechanism can immediately learn cross-modal relationships.

#### 3. Masked Spatio-Temporal Transformer

Standard transformer encoder with a twist:
- Past frames are visible
- Future frame tokens are fully masked
- Model has to reconstruct the future using only temporal context

This is Masked Visual Modeling (MVM) for video.

### Experimental: SCMHSA Variant

I replaced all MHSA blocks with SCMHSA blocks.

#### How SCMHSA Works

**Standard MHSA:**
- Split embedding dimension *d* across *N* heads → each head gets *d/N*
- Each head only sees part of the semantic info

**SCMHSA:**
- Give each head the *full* embedding
- No semantic fragmentation
- Concatenate outputs → way higher dimensionality

#### Projection Layer

Since SCMHSA outputs are now *N* times bigger, I add a learnable projection matrix *W*ₒ to compress back to dimension *d*. Think of it as a learned filter that keeps the useful aggregated information.

---

## Dataset

### Video Sources

5 dashcam videos from different cities:
- San Francisco
- Houston
- Washington, D.C.

Geographic diversity helps with generalization.

### Extraction

- Sampled at 2 FPS
- **42,000 total frames**

### Annotation

For every frame, I generated:
- **Semantic segmentation** using Mask2Former
- **Depth maps** using DepthAnythingV2

### Storage

Everything lives in Google Cloud Storage for fast loading during training. However, also stored on google drive for non-cloud storage at https://drive.google.com/drive/folders/13Z_hjaL_ECWwFXgTWSGLgUs_SNSCY2yX?usp=sharing

#
Example of shards before uploading to GCS Bucket:

<img width="150" height="560" alt="image" src="https://github.com/user-attachments/assets/efc270db-df1e-4011-a59f-97df9d574022" />

#
---

## Training

### Setup

- **Platform**: Google Colab
- **GPU**: NVIDIA A100
- **Constraints**: Limited memory → had to miniaturize and use small batches

### Model Config

| Parameter | Value |
|-----------|-------|
| Image Resolution | 216 × 384 |
| Transformer Layers | 4 |
| Attention Heads | 4 |
| Patch Size | 12 |
| Sequence Length | 5 frames |
| Total Parameters | ~3.33M |

### Hyperparameters

- **Optimizer**: AdamW
- **Weight Decay**: 1e-2
- **Learning Rate**: 2e-4
- **Scheduler**: Cosine Annealing (T_max=25, η_min=1e-6)
- **Epochs**: 25

Both models trained with the exact same settings.

---

## Evaluation

All metrics computed only on valid pixels (ignoring masked/invalid regions).

**Segmentation:**
- **mIoU** – Mean Intersection over Union
- **MO-mIoU** – Moving Object mIoU

**Depth:**
- **AbsRel** – Absolute Relative Error
- **δ₁** – % of pixels with error < 1.25×

---

## Results

### Numbers

| Model | Future Segmentation mIoU |
|-------|-------------------------|
| Mini-FUTURIST (MHSA) | **0.3044** |
| FUTURIST + SCMHSA | 0.2481 |

Both converged in 25 epochs. MHSA did better.

---
Mini-FUTURIST (MHSA):

<img width="677" height="481" alt="image" src="https://github.com/user-attachments/assets/c7609aa6-a923-4c96-8a74-46b56c45849e" />

#

FUTURIST + SCMHSA:

<img width="679" height="492" alt="image" src="https://github.com/user-attachments/assets/f596423c-703c-48be-84c0-c2df6840947c" />

---

### What I Saw

**Baseline (MHSA):**
- Kept large structures intact (roads, sky, buildings)
- Depth gradients looked reasonable
- Struggled with small objects

**SCMHSA:**
- Noisier segmentations
- Class boundaries less sharp
- Depth maps blockier

---

## Why SCMHSA Didn't Win

My hypothesis was wrong. SCMHSA didn't beat MHSA. Here's what I think happened:

### 1. Not Enough Training

Original FUTURIST was trained for thousands of epochs. I only did 25. SCMHSA might need way more iterations to actually converge properly.

### 2. Model Too Small

SCMHSA was originally tested on ~42M parameter models. Mine has 3.3M. The projection layer *W*ₒ might be throwing away important information when it compresses the concatenated outputs.

### 3. Heads Doing Redundant Work

Without extra losses (like semantic similarity loss between heads), SCMHSA heads might just learn the same thing multiple times instead of specializing.

---
#
Mini-FUTURIST:

<img width="871" height="296" alt="image" src="https://github.com/user-attachments/assets/dfd2fd1f-1eb4-45f5-bd30-2ac479a6ce3c" />
#

FUTURIST + SCMHSA:

<img width="869" height="274" alt="image" src="https://github.com/user-attachments/assets/f7c33b3e-1e80-4e65-be10-3c7d7a774648" />

#
---

## Limitations

Let's be real about the constraints:
- **GPU memory**: Colab A100 limits forced me to shrink everything
- **Image resolution**: 216×384 is pretty low
- **Training time**: 25 epochs is nothing
- **Model size**: 3.3M parameters vs. 42M in the original work

---

## Next Steps

If I had more resources:
- Train a 40M+ parameter model
- Run for 800+ epochs
- Use smaller patches (finer spatial detail)
- Add semantic similarity loss to encourage head diversity
- Test on non-driving datasets (indoor scenes, robotics, etc.)
