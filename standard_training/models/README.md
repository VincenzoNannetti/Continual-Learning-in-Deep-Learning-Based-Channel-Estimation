# Model Architectures for Channel Estimation

## Overview

This directory contains PyTorch implementations of various neural network architectures designed for wireless channel estimation. The models range from standard single-architecture networks to combined models that leverage the strengths of multiple architectures.

## Model Types

### Single Models

- **SRCNN (Super-Resolution CNN)**
  - A lightweight convolutional network designed for enhancing channel matrices
  - Architecture: 3 convolutional layers with ReLU activations
  - Input/Output: Complex-valued channel matrices (2 channels for real and imaginary parts)

- **UNet**
  - A U-shaped network with encoder-decoder architecture and skip connections
  - Features: Effective for preserving both high and low-frequency information
  - Enables multi-scale feature extraction with efficient parameter sharing

- **DnCNN (Denoising CNN)**
  - Specialised for noise removal in channel matrices
  - Architecture: 20 layers including 18 identical Conv-BN-ReLU blocks
  - Uses residual learning to predict noise rather than denoised signal

- **Denoising Autoencoder**
  - Available in two variants:
    - Basic autoencoder: Encoder-bottleneck-decoder architecture
    - Residual autoencoder: Adds skip connections and residual learning
  - Features dropout layers for regularisation

### Combined Models

The `combined_models/` directory contains hybrid architectures that chain multiple networks:

- **UNet + SRCNN**
  - First applies UNet for multi-scale feature extraction and denoising
  - Then passes the result through SRCNN for final enhancement
  - Can be configured to freeze either component during training

- **Autoencoder + SRCNN**
  - Uses either basic or residual autoencoder for initial denoising
  - Feeds denoised output to SRCNN for super-resolution enhancement
  - Selective component freezing for transfer learning

- **SRCNN + DnCNN**
  - Flexible architecture allowing either DnCNN→SRCNN or SRCNN→DnCNN processing
  - DnCNN component provides advanced denoising capabilities
  - SRCNN component enhances resolution and detail

For further information please refer to [Report].