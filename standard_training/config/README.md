# Model Training Configuration Guide

## Overview

This directory contains configuration files for training and evaluating neural network models for wireless channel estimation. The configurations follow a standardised YAML format and support various model architectures, training strategies, and evaluation metrics.

## Configuration File Structure

All configuration files follow a common structure with the following main sections:

- **experiment_name**: A unique identifier for the training run
- **framework**: General settings like random seed
- **model**: Model architecture and parameters
- **data**: Dataset configuration and preprocessing settings
- **training**: Training hyperparameters and optimisation settings
- **evaluation**: Metrics and result saving options
- **logging**: Checkpoint directory settings
- **hardware**: Device and acceleration options

## Available Model Configurations

### Single Model Architectures

- **unet.yaml**: U-Net architecture for channel matrix estimation
- **srcnn.yaml**: Super-Resolution CNN for channel estimation
- **autoencoder.yaml**: Denoising residual autoencoder architecture

### Combined Model Architectures

The `combined/` directory contains configurations for models that combine two architectures:

- **unet_srcnn.yaml**: U-Net followed by SRCNN processing
- **ae_srcnn.yaml**: Autoencoder followed by SRCNN processing
- **srcnn_dncnn.yaml**: SRCNN with DnCNN denoising capabilities

## Template Configuration

The `template.yaml` file provides a comprehensive template with all available configuration options and detailed comments. Use this as a starting point for creating new configurations.

## Common Configuration Parameters

### Data Processing

- **interpolation**: Method for interpolating from pilot positions (`rbf`, `spline`, `linear`, etc.)
- **normalisation**: Data normalisation method (`zscore`, `minmax`, `none`)
- **normalise_before_interp**: Whether to normalise before interpolation

### Training Parameters

- **loss_function**: Loss function to optimise (`mse`, `mae`, `huber`)
- **optimiser**: Optimisation algorithm (`adam`, `sgd`)
- **learning_rate**: Initial learning rate
- **scheduler**: Learning rate scheduler settings
- **early_stopping_patience**: Number of epochs without improvement before stopping

### Evaluation Metrics

Common metrics used for evaluation:
- **nmse**: Normalised Mean Squared Error
- **psnr**: Peak Signal-to-Noise Ratio
- **mse**: Mean Squared Error
- **ssim**: Structural Similarity Index

## Using the Configurations

To use these configurations for training, specify the configuration file path when running the training script:

```bash
python train.py --config standard_training/config/unet.yaml
```

For evaluation:

```bash
python evaluate.py --config standard_training/config/unet.yaml --checkpoint path/to/model.pth
```

## Transfer Learning and Fine-tuning

Many configurations include paths to pretrained models to support transfer learning and fine-tuning scenarios. To use a pretrained model:

1. Set the `pretrained_path` parameter to the checkpoint file path
2. For combined models, specify component model paths in their respective parameters
3. Use the `freeze_*` parameters to control which parts of the model should remain frozen during training
