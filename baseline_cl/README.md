# Baseline Continual Learning Methods

This package implements standard continual learning methods for comparison with the LoRA-based approach.

## Implemented Methods

### ✅ Elastic Weight Consolidation (EWC)
- **Paper**: "Overcoming catastrophic forgetting in neural networks" (Kirkpatrick et al., 2017)
- **Description**: Prevents forgetting by adding regularisation terms based on Fisher Information Matrix
- **Status**: ✅ Implemented and tested
- **Usage**: `python scripts/train_ewc.py --config configs/ewc_config.yaml`

### 🚧 Experience Replay
- **Description**: Basic replay buffer storing samples from previous tasks
- **Status**: 🚧 In progress
- **Usage**: TBD

### 🚧 Learning without Forgetting (LwF)
- **Paper**: "Learning without Forgetting" (Li and Hoiem, 2016)
- **Description**: Knowledge distillation from previous model versions
- **Status**: 🚧 In progress
- **Usage**: TBD

## Directory Structure

```
baseline_cl/
├── methods/               # Core method implementations
│   ├── base_trainer.py   # Base class for all methods
│   ├── ewc.py           # EWC implementation
│   └── ...              # Other methods (TBD)
├── configs/              # Configuration files
│   ├── ewc_config.yaml  # EWC configuration
│   └── ...              # Other configs (TBD)
├── scripts/              # Training scripts
│   ├── train_ewc.py     # EWC training script
│   └── ...              # Other training scripts (TBD)
├── test_ewc.py          # EWC unit tests
└── README.md            # This file
```

## Requirements

- Uses the same infrastructure as the main project (`standard_training_2`)
- Requires pretrained UNet-SRCNN model checkpoint
- Same data pipeline and normalisation as LoRA method

## Configuration

Each method has its own YAML configuration file with method-specific hyperparameters following literature standards:

- **EWC λ**: 1000.0 (literature standard)
- **Fisher samples**: All validation samples (can be limited for speed)
- **Base learning rate**: 1e-4 (same as LoRA method)

## Testing

Run unit tests for implemented methods:

```bash
# Test EWC implementation
python test_ewc.py
```

## Results

Results are saved in `results/` directory with detailed metrics per domain and aggregate statistics for comparison with the LoRA approach. 