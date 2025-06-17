# Shared Utilities (`shared/utils`)

## Overview

This directory contains a collection of utility modules designed to support data processing, model training, evaluation, and visualisation across the project. These utilities ensure code reusability, consistency, and ease of maintenance.

## File Summaries

### `plot_heatmap.py`
- **Purpose:** Provides functions for visualising complex matrices (e.g., channel estimates) as heatmaps, following IEEE publication style.
- **Features:**
  - Plots interpolated, predicted, and ground-truth matrices, as well as error maps.
  - Supports custom colour maps and figure layouts.
  - Saves figures in SVG format for high-quality publication.
  - Handles both real and complex-valued data.

### `interpolation.py`
- **Purpose:** Implements interpolation methods for reconstructing channel estimates from pilot positions.
- **Features:**
  - Supports Radial Basis Function (RBF) and spline interpolation.
  - Handles fallback to nearest-neighbour interpolation if insufficient pilots are present.
  - Designed for 4D tensors: `(samples, subcarriers, symbols, 2)` (real/imag).
  - Includes robust error handling and informative warnings.

### `training_utils.py`
- **Purpose:** Provides helper functions for model training configuration.
- **Features:**
  - Selects loss functions (MSE, MAE/L1, Huber) based on config.
  - Configures optimisers (Adam, SGD) and learning rate schedulers.
  - Ensures only trainable parameters are optimised.
  - Handles scheduler setup with sensible defaults and warnings.

### `get_device.py`
- **Purpose:** Determines the appropriate PyTorch device (CPU or CUDA) based on configuration and hardware availability.
- **Features:**
  - Supports 'auto', 'cpu', and 'cuda' options.
  - Issues warnings if CUDA is requested but unavailable.
  - Returns a `torch.device` object for use in training and evaluation scripts.

### `format_time.py`
- **Purpose:** Formats elapsed time in a human-readable string (hours, minutes, seconds).
- **Features:**
  - Simple utility for logging and progress reporting.

### `metrics.py`
- **Purpose:** Implements standard evaluation metrics for model performance.
- **Features:**
  - **PSNR:** Peak Signal-to-Noise Ratio.
  - **NMSE:** Normalised Mean Squared Error.
  - **SSIM:** Structural Similarity Index (calls `ssim.py`).
  - **MSE:** Mean Squared Error.
  - Provides a dictionary for easy metric function lookup.
  - Handles edge cases (e.g., zero targets) robustly.

### `ssim.py`
- **Purpose:** Implements the Structural Similarity Index (SSIM) for image and channel quality assessment.
- **Features:**
  - Computes SSIM for real or complex-valued data (uses magnitude for complex).
  - Uses a Gaussian window for local statistics.
  - Returns a value between 0 and 1 (1 = perfect similarity).
  - Can average SSIM over spatial dimensions or return per-sample values.

### `__init__.py`
- **Purpose:** Marks the directory as a Python package.

## Example Usage

```python
from shared.utils.plot_heatmap import plot_heatmap
from shared.utils.interpolation import interpolation
from shared.utils.training_utils import get_criterion, get_optimiser, get_scheduler
from shared.utils.get_device import get_device
from shared.utils.format_time import format_time
from shared.utils.metrics import calculate_psnr, calculate_nmse, calculate_ssim
```

## Requirements

- Python 3.8+
- NumPy, SciPy
- PyTorch
- Matplotlib, Seaborn (for plotting)

## Notes

- All plotting functions are designed for publication-quality output.
- Interpolation and metrics modules are robust to edge cases and provide informative warnings.
- The utilities are intended to be imported and used by higher-level scripts throughout the project.

--- 