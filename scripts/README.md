# PowerShell Workflow Scripts

## Overview

This directory contains PowerShell scripts that simplify the execution of training and evaluation workflows. These scripts provide a convenient command-line interface with parameter validation and helpful error messages, reducing the complexity of running Python commands directly.

## Available Scripts

### `run_standard.ps1`

Manages standard (single-model) training and evaluation workflows.

#### Common Parameters

- `-Config <file>`: Path to configuration YAML file (required)
- `-Mode <mode>`: Workflow mode: train-eval, train, or eval (default: train-eval)
- `-Suffix <suffix>`: Experiment name suffix
- `-NoWandb`: Disable Weights & Biases logging
- `-Python <path>`: Path to Python executable (default: python)

#### Mode-Specific Parameters

For **train** mode:
- `-UseDataset {a,b}`: Which dataset to use (a or b)

For **eval** mode:
- `-Checkpoint <path>`: Path to model checkpoint file
- `-DataConfig <file>`: Path to data configuration override file

#### Examples

```powershell
# Train and evaluate
.\scripts\run_standard.ps1 -Config standard_training/config/unet.yaml

# Train only with dataset A
.\scripts\run_standard.ps1 -Config standard_training/config/unet.yaml -Mode train -UseDataset a -Suffix _dataset_a

# Evaluate only with a specific checkpoint
.\scripts\run_standard.ps1 -Config standard_training/config/unet.yaml -Mode eval -Checkpoint checkpoints/best_model.pth
```

### `run_continual.ps1`

Manages continual learning (supermask) training and evaluation workflows.

#### Common Parameters

- `-Config <file>`: Path to configuration YAML file (required)
- `-Mode <mode>`: Workflow mode: train-eval, train, or eval (default: train-eval)
- `-Suffix <suffix>`: Experiment name suffix
- `-NoWandb`: Disable Weights & Biases logging
- `-Python <path>`: Path to Python executable (default: python)

#### Mode-Specific Parameters

For **train** mode:
- `-UseDataset {a,b}`: Which dataset to use (a or b)
- `-Sequence <seq>`: Supermask task sequence, comma-separated (e.g., "0,1,2")

For **eval** mode:
- `-CheckpointDir <path>`: Path to checkpoint directory
- `-TaskId <id>`: Task ID for evaluation (if omitted, evaluates all tasks)

#### Examples

```powershell
# Train and evaluate
.\scripts\run_continual.ps1 -Config continual_learning/config/supermask_config.yaml

# Train only with dataset A and specific task sequence
.\scripts\run_continual.ps1 -Config continual_learning/config/supermask_config.yaml -Mode train -UseDataset a -Sequence "0,2,1"

# Evaluate only task 0 with a specific checkpoint directory
.\scripts\run_continual.ps1 -Config continual_learning/config/supermask_config.yaml -Mode eval -CheckpointDir results/checkpoints/my_run -TaskId 0
```

## Benefits of Using These Scripts

1. **Simplified Execution**: Run complex workflows with a single command
2. **Parameter Validation**: Automatic validation of file paths and parameter values
3. **Consistent Interface**: Common pattern for both standard and continual learning workflows
4. **Error Handling**: Proper error messages and exit codes
5. **Documentation**: Built-in usage examples and help information

## Usage in Development Workflow

These scripts are particularly useful for:

- **Quick Experimentation**: Rapidly test different configurations
- **Reproducible Research**: Document exact commands for each experiment
- **Batch Processing**: Run multiple experiments from the command line

To see detailed usage information for each script, run:

```powershell
# For standard training
.\scripts\run_standard.ps1 -?

# For continual learning
.\scripts\run_continual.ps1 -?
```
