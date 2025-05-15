# Continual Learning Framework for Channel Estimation

## Overview

This directory contains a comprehensive framework for implementing and evaluating continual learning approaches for wireless channel estimation. The framework focuses on sequential learning of multiple channel estimation tasks without catastrophic forgetting, enabling models to adapt to new scenarios while retaining performance on previously learned tasks.

## Directory Structure

- **`models/`**: Neural network architectures for continual learning
  - `supermask/`: Models implementing the supermask technique
  - `layers/`: Custom layer implementations (e.g., MaskConv)

- **`trainers/`**: Training implementations
  - `supermask_trainer.py`: Trainer for supermask-based continual learning

- **`evaluators/`**: Evaluation implementations
  - `supermask_evaluator.py`: Evaluator for supermask model performance across tasks

- **`datasets/`**: Dataset handling for continual learning scenarios

- **`utils/`**: Utility functions specific to continual learning

- **`config/`**: Configuration files
  - `supermask/`: Configuration files for supermask models and training

- **`checkpoints/`**: Saved model checkpoints (generated during training)

- **`train.py`**: Main training script

- **`evaluate.py`**: Comprehensive evaluation script

## Continual Learning Approach

The framework currently implements the **supermask** approach to continual learning:

1. A base model has its weights frozen during training
2. Each task learns a binary mask that selects a subset of these weights
3. Different tasks use different masks, allowing for knowledge to be preserved
4. Only the mask parameters are updated, greatly reducing the number of trainable parameters

This enables the model to learn new tasks without forgetting previously learned ones, addressing the catastrophic forgetting problem in neural networks.

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch 1.12+
- Additional dependencies in requirements.txt

### Training a Continual Learning Model

#### Using PowerShell Script (Recommended)

The easiest way to train a model is using the PowerShell script:

```powershell
# Train and evaluate with default task sequence
.\scripts\run_continual.ps1 -Config continual_learning/config/supermask/unet_srcnn.yaml

# Train with custom task sequence
.\scripts\run_continual.ps1 -Config continual_learning/config/supermask/unet_srcnn.yaml -Sequence "0,2,1" -Mode train
```

See [scripts/README.md](../scripts/README.md) for more options and examples.

#### Direct Python Command

Alternatively, you can use the Python module directly:

```bash
# Train and evaluate
python -m continual_learning.train --config continual_learning/config/supermask/unet_srcnn.yaml

# Train with custom task sequence
python -m continual_learning.train --config continual_learning/config/supermask/unet_srcnn.yaml --supermask_sequence "0,2,1"
```

Additional options:
- `--experiment_suffix _suffix`: Add a suffix to the experiment name
- `--dataset_to_use [a|b]`: Specify which dataset to use if multiple are defined
- `--no_eval`: Skip evaluation after training
- `--no_wandb`: Disable Weights & Biases logging

### Evaluating a Continual Learning Model

#### Using PowerShell Script (Recommended)

```powershell
# Evaluate all tasks
.\scripts\run_continual.ps1 -Config continual_learning/config/supermask/unet_srcnn.yaml -Mode eval -CheckpointDir checkpoints/my_run

# Evaluate specific task
.\scripts\run_continual.ps1 -Config continual_learning/config/supermask/unet_srcnn.yaml -Mode eval -CheckpointDir checkpoints/my_run -TaskId 0
```

#### Direct Python Command

```bash
# Evaluate all tasks
python -m continual_learning.evaluate --config continual_learning/config/supermask/unet_srcnn.yaml --checkpoint_dir checkpoints/my_run

# Evaluate specific task
python -m continual_learning.evaluate --config continual_learning/config/supermask/unet_srcnn.yaml --checkpoint_dir checkpoints/my_run --task_id 0
```

## Configuration System

The framework uses a flexible YAML-based configuration system:

```yaml
experiment_name: supermask_unet_srcnn  # Experiment identifier

model:
  name: unet_srcnn_supermask         # Model architecture
  params:
    sparsity: 0.05                   # % of weights to use per task
    num_tasks: 3                     # Number of tasks to support

supermask:
  tasks: 3                           # Number of tasks (must match model.params.num_tasks)
  sequence: [0, 1, 2]                # Task sequence for training
  sparsity: 0.05                     # Sparsity level (% of weights to use)

data:
  dataset_type: channel              # Dataset type
  data_name: dataset_a               # Dataset identifier
  # ... data configuration similar to standard_training ...
```

## Key Concepts

- **Tasks**: In channel estimation, tasks typically correspond to different pilot patterns or channel conditions.
- **Task Sequence**: The order in which tasks are learned during training.
- **Supermask**: A binary mask applied to the frozen weights of the base model.
- **Sparsity**: The percentage of weights that are active (non-zero) in each mask.

## Metrics and Evaluation

The framework evaluates performance using:

- **Task-specific metrics**: NMSE, PSNR, etc. on each individual task
- **Forgetting metrics**: How much performance decreases on previous tasks
- **Backward Transfer**: How learning new tasks affects performance on previous tasks
- **Forward Transfer**: How well the model generalises to unseen tasks

## Visualisations

The framework generates visualisations to help understand continual learning:

- Performance matrices (tasks x training stages)
- Forgetting curves
- Task mask similarity analysis
- Example predictions for each task at different training stages
