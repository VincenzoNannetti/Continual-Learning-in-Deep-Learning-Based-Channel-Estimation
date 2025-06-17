# Continual Learning for Deep Wireless Channel Estimation

## Project Overview

With the proliferation of connected devices—including smartphones, autonomous vehicles, drones, and Internet of Things (IoT) sensors—and the advancement of 5G networks, reliable and high-performance wireless communication is more critical than ever. A key challenge in this domain is **channel estimation**: recovering the transmitted signal by inverting the effects of the wireless channel, which is often highly dynamic and non-stationary.

While conventional estimation techniques remain widely used, recent advances in deep learning offer promising alternatives, particularly in their ability to model complex, high-dimensional, and correlated data. However, real-world wireless environments are subject to rapid changes due to user mobility, multipath propagation, and varying signal-to-noise ratios (SNR), making robust and adaptive channel estimation a challenging task.

This project investigates the application of **deep learning-based continual learning** to wireless channel estimation, with a focus on maintaining high accuracy across multiple domains and preventing catastrophic forgetting. We propose a novel approach that integrates domain-specific LoRA adapters with batch-normalisation layers, enabling parameter-efficient updates and rapid domain switching. Our method is evaluated in both sequential and online learning scenarios, achieving zero forgetting in sequential domain tests and delivering lower normalised mean square error (NMSE) and reduced variance compared to standard baselines. The compact model demonstrates the feasibility of continual learning for resource-constrained wireless systems and lays the groundwork for future research in adaptive wireless communications.

---

## Repository Structure

This codebase is modular, with each directory supporting a key aspect of the research pipeline:

- **`standard_training_2/`**: Baseline deep learning models and training scripts for standard (non-continual) channel estimation.
- **`main_algorithm_v2/`**: Core continual learning framework, including both offline (sequential) and online (streaming) continual learning, LoRA adapters, EWC, and domain adaptation experiments.
- **`data_generation/`**: Ray-tracing and statistical simulation tools for generating realistic wireless channel datasets, including 3GPP-inspired scenarios.
- **`shared/`**: Common utilities for data processing, metrics, plotting, and training, used across modules.
- **`baseline_cl/`**: Baseline continual learning methods (e.g., EWC, experience replay) for comparison with the proposed approach.

---

## Key Features

- **Deep Learning for Channel Estimation**: UNet, SRCNN, DnCNN, autoencoders, and hybrid models.
- **Continual Learning**: Parameter-efficient LoRA adapters, EWC, replay buffers, and robust domain switching.
- **Online & Sequential Evaluation**: Supports both batch (offline) and streaming (online) continual learning scenarios.
- **Realistic Data Generation**: 3D ray-tracing and 3GPP-inspired channel models for diverse, challenging datasets.
- **Comprehensive Metrics & Visualisation**: NMSE, SSIM, PSNR, and publication-ready plots.
- **Ablation & Domain Shift Studies**: Automated scripts for in-depth analysis and benchmarking.

---

## Getting Started

1. **Install Requirements**  
   See the `requirements.txt` in each submodule or the project root for dependencies (PyTorch, NumPy, SciPy, Matplotlib, etc.).

2. **Generate or Download Data**  
   Use the tools in `data_generation/` to create synthetic channel datasets, or use provided `.mat` files.

3. **Train Baseline Models**  
   Run standard training scripts in `standard_training_2/` to establish baseline performance.

4. **Run Continual Learning Experiments**  
   Use `main_algorithm_v2/` for both offline and online continual learning, including ablation and domain shift experiments.

5. **Analyse Results**  
   Leverage the plotting and analysis scripts in each module for comprehensive evaluation and visualisation.

---

## Module Summaries

### `standard_training_2/`
- Implements standard deep learning models for channel estimation.
- Includes training, evaluation, hyperparameter optimisation (Optuna), and plotting scripts.
- Serves as a baseline for comparison with continual learning methods.

### `main_algorithm_v2/`
- **Offline**: Sequential continual learning, LoRA adapter training, EWC, and replay buffer management.
- **Online**: True streaming continual learning, dynamic domain switching, online adaptation, and comprehensive metrics.
- Includes scripts for ablation studies and domain mislabelling/shift experiments.

### `data_generation/`
- Ray-tracing engine and 3GPP-inspired statistical models for generating realistic wireless channel data.
- Highly configurable via YAML files.
- Includes tools for visualisation and statistical analysis of generated datasets.

### `shared/`
- Utility functions for data formatting, device management, metrics (NMSE, SSIM, PSNR), and plotting.
- Ensures consistency and code reuse across all modules.

### `baseline_cl/`
- Baseline continual learning methods (e.g., EWC, experience replay) for benchmarking.
- Scripts for training, evaluation, and result analysis.

---

## Example Usage

**Standard Training:**
```bash
python standard_training_2/train.py --config standard_training_2/config/dncnn_srcnn.yaml
```

**Continual Learning (Online):**
```bash
python main_algorithm_v2/online/online_continual_learning.py --config main_algorithm_v2/online/config/online_config.yaml
```

**Ablation Study:**
```bash
python main_algorithm_v2/online/run_ablation_studies.py
```

**Data Generation:**
```bash
python data_generation/ray_tracing/examples/channel_environment_example.py --config data_generation/config/dataset_a.yaml
```

---
