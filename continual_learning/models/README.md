# Continual Learning Model Architectures

## Overview

This directory contains neural network model implementations designed for continual learning in wireless channel estimation. These models employ supermask techniques to learn multiple tasks sequentially without catastrophic forgetting.

## Core Components

### Layers (`layers/`)

- **MaskConv**: A modified convolutional layer that applies binary masks to weights
  - Supports multiple tasks with separate masks per task
  - Allows mixing between random and pretrained weights
  - Controls sparsity level for efficient parameter usage

### Supermask Models (`supermask/`)

- **SRCNN_Supermask**: A supermask version of the Super-Resolution CNN
  - Lightweight implementation with 3 MaskConv layers
  - Supports multiple tasks with separate binary masks
  - Minimal parameter count for efficient deployment

- **UNet_SRCNN_Supermask**: A combined UNet and SRCNN architecture with supermasks
  - Automatically detects task based on pilot pattern
  - Leverages pretrained UNet and SRCNN weights
  - Applies task-specific binary masks to all convolutional layers

## Supermask Approach

The supermask technique enables continual learning by:

1. **Weight Freezing**: Base model weights are frozen during training
2. **Binary Masks**: Each task learns a binary mask that selects which weights to use
3. **Task-Specific Learning**: Different masks are learned for different tasks
4. **Storage Efficiency**: Only binary masks are stored per task, not entire model copies

## Task Definition

In channel estimation context, tasks are typically defined by:
- Different pilot patterns (low, medium, high density)
- Different channel conditions (SNR levels, multipath characteristics)
- Different antenna configurations

The models automatically detect the current task based on the input data features.

## Getting Started

### Example: Using UNet_SRCNN_Supermask

```python
# Create supermask model with 3 tasks
model = UNet_SRCNN_Supermask(
    pretrained_path="path/to/pretrained_weights.pth",
    num_tasks=3,
    sparsity=0.05,  # 5% of weights active per task
    alpha=0.3       # Mixing coefficient between random and pretrained weights
)

# Forward pass with automatic task detection
output = model(input_tensor)  # Shape: [batch_size, 3, height, width]

# Explicitly set task
model.set_task(task_id=1)
output = model(input_tensor)

# Evaluate in inference mode (weighted mask averaging)
model.set_task(task_id=-1)  # Special value for inference mode
output = model(input_tensor)
```
