# Standard Training Framework for Channel Estimation

## Overview

This directory contains a complete PyTorch-based framework for training, evaluating, and deploying neural network models for wireless channel estimation. The framework supports multiple model architectures, data preprocessing strategies, and evaluation metrics.

## Directory Structure

- **`models/`**: Neural network model architectures
  - Single models (SRCNN, UNet, DnCNN, Autoencoder)
  - Combined models (UNet+SRCNN, Autoencoder+SRCNN, SRCNN+DnCNN)

- **`config/`**: YAML configuration files
  - Individual configurations for different models and experiments
  - Template and shared configuration options

- **`datasets/`**: Dataset loading and preprocessing
  - `standard_dataset.py`: Core dataset implementation
  - `dataset_utils.py`: Utilities for data loading and transformations

- **`utils/`**: Helper functions and utilities
  - Metrics calculation
  - Visualisation tools
  - Training/evaluation utilities

- **`checkpoints/`**: Saved model weights (generated during training)

- **`train.py`**: Main training script

- **`evaluate.py`**: Comprehensive evaluation script

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch 1.12+
- Additional dependencies in requirements.txt (numpy, matplotlib, scipy, wandb)

### Training a Model

#### Using PowerShell Script (Recommended)

The easiest way to train a model is using the PowerShell script:

```powershell
# Train and evaluate
.\scripts\run_standard.ps1 -Config standard_training/config/unet.yaml

# Train only with dataset A
.\scripts\run_standard.ps1 -Config standard_training/config/unet.yaml -Mode train -UseDataset a -Suffix _dataset_a
```

See [scripts/README.md](../scripts/README.md) for more options and examples.

#### Direct Python Command

Alternatively, you can use the Python module directly:

```bash
python -m standard_training.train --config config/unet.yaml --experiment_suffix _dataset_a
```

Additional options:
- `--dataset_to_use [a|b]`: Specify which dataset to use if multiple are defined
- `--no_eval`: Skip evaluation after training
- `--no_wandb`: Disable Weights & Biases logging

### Evaluating a Model

#### Using PowerShell Script (Recommended)

```powershell
# Evaluate with a specific checkpoint
.\scripts\run_standard.ps1 -Config standard_training/config/unet.yaml -Mode eval -Checkpoint checkpoints/my_model/best_model.pth
```

#### Direct Python Command

```bash
python -m standard_training.evaluate --config config/unet.yaml --checkpoint checkpoints/my_model/best_model.pth
```

Additional options:
- `--experiment_suffix _evaluation_name`: Add a suffix to the experiment name
- `--results_dir results/my_experiment`: Specify directory for saving results
- `--plot_samples 5`: Number of sample visualisations to generate

## Configuration System

The framework uses a flexible YAML-based configuration system:

```yaml
experiment_name: unet_baseline  # Experiment identifier

model:
  name: unet                    # Model architecture
  params:                       # Model-specific parameters
    in_channels: 2
    base_features: 32

data:
  dataset_type: channel         # Dataset type
  data_name: dataset_a          # Dataset identifier
  interpolation: rbf            # Interpolation method
  normalisation: zscore         # Normalisation method

training:
  epochs: 100
  batch_size: 64
  loss_function: mse
  optimiser: adam
  learning_rate: 1e-4
  # ... additional training parameters
```

## Model Selection Guide

- **For lightweight deployment**: SRCNN
- **For best overall performance**: UNet or combined models
- **For high-noise scenarios**: DnCNN or combined models with DnCNN

Refer to the [models/README.md](models/README.md) for detailed information on each model architecture.

## Citing This Work

If you use this framework in your research, please cite:

```
@misc{nannetti2025channelestimation,
  author = {Nannetti, Vincenzo Riccardo},
  title = {Deep Learning for Wireless Channel Estimation},
  year = {2025},
  publisher = {Imperial College London},
  howpublished = {\url{https://github.com/VincenzoNannetti21/...}}
}
```

## License

? Not sure what license
