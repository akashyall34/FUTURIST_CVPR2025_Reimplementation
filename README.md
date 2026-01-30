# Reimplementation of FUTURIST and Extension with SCMHSA

## Overview
This project implements a miniaturized version of FUTURIST, a multimodal visual sequence transformer introduced in CVPR 2025, and evaluates the impact of replacing its standard Multi-Head Self-Attention (MHSA) mechanism with Semantic Concentration Multi-Head Self-Attention (SCMHSA), proposed in a 2025 research paper.

The objective is to study semantic future prediction—forecasting future semantic segmentation masks and depth maps from video—under realistic computational constraints, while isolating and quantifying the effects of semantic dilution in transformer attention mechanisms

Two architectures are implemented and compared:
- Baseline: Mini-FUTURIST with standard MHSA
- Experimental: Mini-FUTURIST with SCMHSA replacing MHSA

Both models are trained and evaluated under identical hyperparameters, datasets, and compute environments to ensure a controlled comparison.
---
