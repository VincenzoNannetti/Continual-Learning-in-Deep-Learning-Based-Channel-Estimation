# Standard Training 2.0

## Overview

This module provides a comprehensive framework for training, evaluating, and comparing deep learning models for channel estimation in wireless communications. It includes robust data handling, model training, evaluation, hyperparameter optimisation, and visualisation tools, all tailored for reproducible research and publication-quality results.

## Folder Structure

- `train.py`: Main training script for all supported models.
- `dataset.py`: Data loading, preprocessing, and normalisation utilities.
- `evaluate.py`: Script for evaluating trained models on test data.
- `eval_models.py`: Flexible evaluation for various model architectures.
- `optuna_runner.py`: Hyperparameter optimisation using Optuna and W&B.
- `plot_performance_trajectories.py`: Visualises catastrophic forgetting and adaptation.
- `plot_model_comparison.py`: Compares model performance across metrics.
- `plotting_utils.py`: Helper functions for IEEE-style plots and visualisation.
- `interpolate.py`: Implements RBF-based interpolation for pilot data.
- `noise_generation.py`: Adds complex Gaussian noise to data.
- `models/`: Contains all model definitions (UNet, SRCNN, DnCNN, Autoencoders, etc.).
- `config/`: Example YAML configuration files for experiments.
- `catastrophic_forgetting_v2/`: Scripts and results for catastrophic forgetting analysis.
- `tests/`: Unit tests for core components.

## Key Features

### Data Handling

- **StandardDataset**: Loads and preprocesses channel data, supports both single and sequence-based tasks, and caches processed data for efficiency.
- **NormalisingDatasetWrapper**: Applies Z-score normalisation using statistics computed from the training set, ensuring fair evaluation and stable training.
- **Interpolation**: Uses RBF interpolation to estimate missing channel values from pilot positions.

### Model Training

- **train.py**: 
  - Supports multiple architectures: UNet, SRCNN, DnCNN, Autoencoders, and hybrid models.
  - Handles data loading, normalisation, and augmentation (noise, interpolation).
  - Implements early stopping, learning rate scheduling, and mixed-precision training.
  - Integrates with W&B for experiment tracking.
  - Saves best model checkpoints and training curves.

### Evaluation

- **evaluate.py**: 
  - Loads trained models and evaluates on test data using the same normalisation as training.
  - Computes NMSE, PSNR, and SSIM metrics.
  - Supports all model types and outputs detailed performance logs.

- **eval_models.py**: 
  - Command-line tool for evaluating any supported model.
  - Automatically extracts configuration and normalisation stats from checkpoint files.
  - Can benchmark inference time and plot evaluation samples.

### Hyperparameter Optimisation

- **optuna_runner.py**: 
  - Integrates Optuna for automated hyperparameter search.
  - Supports composite scoring (SSIM, NMSE, overfitting/convergence penalties).
  - Runs training as subprocesses for isolation and logs all outputs.
  - Compatible with W&B sweeps for parameter space definition.

### Visualisation

- **plot_performance_trajectories.py**: 
  - Plots catastrophic forgetting and adaptation trajectories for different models.
  - Generates publication-ready figures (IEEE style) for PSNR, NMSE, and SSIM.

- **plot_model_comparison.py**: 
  - Compares all models across key metrics (MSE, NMSE, SSIM, PSNR, inference time).
  - Produces sorted line plots with clear best/worst shading.

- **plotting_utils.py**: 
  - Helper functions for heatmaps, training curves, and comprehensive evaluation plots.
  - Consistent IEEE-style formatting for all figures.

### Utilities

- **noise_generation.py**: Adds circularly symmetric complex Gaussian noise to tensors at a specified SNR.
- **interpolate.py**: RBF-based interpolation for reconstructing missing channel values from pilots.

## Example Usage

### Training

```bash
python train.py --config config/dncnn_srcnn.yaml
```

### Evaluation

```bash
python evaluate.py --config config/dncnn_srcnn.yaml --weights checkpoints/best_model.pth
```

or

```bash
python eval_models.py --weights_path checkpoints/best_model.pth --model_type srcnn
```

### Hyperparameter Optimisation

```bash
python optuna_runner.py --sweep_config config/sweep.yaml
```

### Plotting

```bash
python plot_performance_trajectories.py --csv_path results/cf_results.csv --target_domain domain_high_snr_fast_linear
python plot_model_comparison.py
```

## Model Architectures Supported

- **UNet**
- **SRCNN**
- **DnCNN**
- **Residual and Non-Residual Autoencoders**
- **Hybrid Models**: UNet+SRCNN, SRCNN+DnCNN, AE+SRCNN

## Metrics

- **NMSE**: Normalised Mean Squared Error
- **PSNR**: Peak Signal-to-Noise Ratio
- **SSIM**: Structural Similarity Index

## Visualisation

All plots are generated in IEEE style, suitable for publication, and saved as SVG files for high quality.

## Reproducibility

- All scripts use YAML configuration files for experiment reproducibility.
- Model checkpoints include configuration and normalisation statistics.
- Caching and deterministic data splits ensure consistent results.

