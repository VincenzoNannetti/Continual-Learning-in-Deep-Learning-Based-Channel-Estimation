# Online Continual Learning & Evaluation (`main_algorithm_v2/online`)

## Overview

This module implements a flexible and extensible pipeline for **online continual learning** in wireless channel estimation, with a focus on LoRA adapters and domain adaptation. It supports real-time data processing, dynamic domain/task switching, online training with EWC, and comprehensive evaluation and ablation studies.

## Key Features

- **Online Continual Learning**: Unified streaming loop for inference and training, supporting LoRA adapters, EWC, and dynamic domain switching.
- **Real-Time Data Pipeline**: Loads raw `.mat` data, adds noise, extracts pilots, interpolates, and normalises on-the-fly (no caching).
- **Dynamic Model Management**: Loads trained offline models, manages LoRA adapters and batch norm stats per domain, and supports replay buffers.
- **Flexible Configuration**: YAML-based config system, with Pydantic models for validation and extension.
- **Ablation & Domain Shift Experiments**: Scripts for running and analysing ablation studies and domain mislabelling/shift scenarios.
- **Comprehensive Metrics**: Tracks NMSE, SSIM, PSNR, timing, adaptation effectiveness, and more.
- **Publication-Ready Visualisation**: Generates detailed plots and CSVs for all experiments.

## Directory Structure

```
online/
├── src/
│   ├── config.py                # Online config management (Pydantic models)
│   ├── data.py                  # Real-time data pipeline (loading, noise, interpolation, normalisation)
│   ├── model_manager.py         # Model loading, LoRA/BN management, domain switching, EWC
│   ├── loss_functions.py        # Losses for masked NMSE, etc.
│   ├── ewc.py                   # Online EWC/Fisher management
│   ├── trigger_manager.py       # Training trigger logic (hybrid, drift, etc.)
│   ├── mixed_batch_manager.py   # Mixed offline/online batch creation
│   ├── online_buffer_manager.py # Online replay buffer management
│   ├── gradient_buffer_manager.py # Gradient diversity buffer
│   ├── comprehensive_metrics.py # Metrics collection and reporting
│   ├── transfer_metrics.py      # Forward/backward transfer metrics
│   └── __init__.py
├── config/
│   └── online_config.yaml       # Example config
├── online_continual_learning.py # Main online continual learning script (Algorithm 2)
├── run_ablation_studies.py      # Script for automated ablation studies
├── run_domain_mislabeling_experiment.py # Script for domain shift/mislabeling experiments
├── eval/                        # Output directory for evaluation results
├── tests/                       # Unit tests
└── README.md
```

## Main Components

### 1. Online Continual Learning (`online_continual_learning.py`)

- Implements a unified streaming loop (Algorithm 2) for online continual learning.
- For each sample:
  - Selects domain (random, sequential, or temporal shift)
  - Loads and processes data (noise, pilots, interpolation, normalisation)
  - Switches model to correct domain (LoRA adapters, BN stats)
  - Runs inference and computes metrics (NMSE, SSIM, PSNR, etc.)
  - Optionally triggers online training (with EWC, early stopping, mixed batches)
  - Logs and saves results, including adaptation effectiveness and timing

### 2. Data Pipeline (`src/data.py`)

- Loads raw `.mat` files for each domain
- Adds complex Gaussian noise at runtime
- Extracts pilot signals and performs interpolation (RBF, etc.)
- Normalises data using statistics from offline training
- Supports domain remapping for mislabelling experiments

### 3. Model Manager (`src/model_manager.py`)

- Loads trained offline models (UNet+SRCNN+LoRA)
- Manages LoRA adapters and batch norm stats per domain
- Handles replay buffers for experience replay
- Supports EWC (Elastic Weight Consolidation) for continual learning
- Provides fast domain switching and inference

### 4. Configuration (`src/config.py`)

- Uses Pydantic models for robust config validation
- Supports both offline and online-specific parameters
- Loads offline config from file or checkpoint
- All experiment settings are YAML-driven

### 5. Ablation Studies (`run_ablation_studies.py`)

- Automates running multiple ablation configurations (e.g., no EWC, no buffer, no LoRA, reduced training, etc.)
- Collects and plots results for comprehensive comparison
- Outputs summary tables and visualisations

### 6. Domain Shift & Mislabeling Experiments (`run_domain_mislabeling_experiment.py`)

- Runs experiments with temporal domain shifts or mislabelled domains
- Analyses adaptation behaviour, performance degradation, and recovery
- Generates detailed plots and CSVs for all phases

## Example Usage

### Online Continual Learning

```bash
python online_continual_learning.py --config config/online_config.yaml
```

### Ablation Studies

```bash
python run_ablation_studies.py
```

### Domain Shift/Mislabeling Experiments

```bash
python run_domain_mislabeling_experiment.py --num_samples 1200 --primary_domain 3 --shift_domain 8 --shift_start 200 --shift_end 800
```

## Configuration

All settings are controlled via YAML files (see `config/online_config.yaml`). Key options include:

- **offline_config_path**: Path to offline config (optional if in checkpoint)
- **offline_checkpoint_path**: Path to trained model checkpoint
- **raw_data**: Paths and mapping for raw data files per domain
- **online_evaluation**: Number of evaluations, domain selection mode, logging, buffer settings
- **online_training**: Enable/disable, EWC settings, learning rate, batch size, trigger type, etc.
- **online_metrics**: Which metrics to track and whether to save detailed results

## Metrics & Output

- **NMSE**: Normalised Mean Squared Error
- **SSIM**: Structural Similarity Index
- **PSNR**: Peak Signal-to-Noise Ratio
- **Timing**: Preprocessing, inference, training, domain switch
- **Adaptation Effectiveness**: Tracks improvement after online updates
- **Comprehensive CSVs and plots**: All results are saved for further analysis

## Notes

- The pipeline is designed for reproducibility and extensibility.
- All timing is measured in milliseconds for real-time analysis.
- Results are deterministic if the same random seed is used.
- The system supports both evaluation-only and online training modes.
