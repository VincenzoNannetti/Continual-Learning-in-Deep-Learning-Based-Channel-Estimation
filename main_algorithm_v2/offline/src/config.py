"""
Configuration management for the continual learning LoRA experiment.
Uses Pydantic for type-safe and validated configuration from a YAML file.
"""

import yaml
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

# --- Pydantic Models for Configuration ---

class FrameworkConfig(BaseModel):
    seed: int

class UNetArgs(BaseModel):
    in_channels: int
    base_features: int
    depth: int
    activation: str
    leaky_slope: float
    verbose: bool

class SRCNNParams(BaseModel):
    srcnn_channels: List[int]
    srcnn_kernels: List[int]
    num_tasks_for_model: int

class ModelParameters(BaseModel):
    use_domain_specific_bn: bool = Field(..., alias='use_domain_specific_bn')
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    lora_bias_trainable: str
    task_lora_ranks: Dict[int, int]
    task_lora_alphas: Dict[int, int]
    task_lora_dropouts: Dict[int, float]
    unet_args: UNetArgs
    srcnn_params: SRCNNParams = Field(..., alias='srcnn_params')

class ModelConfig(BaseModel):
    name: str
    pretrained_path: str
    evaluation_path: Optional[str] = None
    params: ModelParameters
    strict_load: bool

class TaskParams(BaseModel):
    data_name: str
    snr: int

class NormStats(BaseModel):
    mean_inputs: List[float]
    std_inputs: List[float]
    mean_targets: List[float]
    std_targets: List[float]

class DataConfig(BaseModel):
    dataset_type: str
    data_dir: str
    preprocessed_dir: str
    tasks: int
    sequence: List[int]
    tasks_params: Dict[int, TaskParams]
    interpolation: str
    normalisation: str
    normalise_target: bool
    norm_stats: Optional[NormStats] = None
    num_workers: int
    validation_split: float
    test_split: float

class SchedulerParams(BaseModel):
    mode: str
    factor: float
    patience: int
    min_lr: float
    verbose: bool

class SchedulerConfig(BaseModel):
    type: str
    params: SchedulerParams

class TrainingConfig(BaseModel):
    epochs_per_task: int
    batch_size: int
    loss_function: str
    optimiser: str
    learning_rate: float
    weight_decay: float
    task_weight_decays: Dict[int, float]
    betas: List[float]
    early_stopping_patience: int
    scheduler: SchedulerConfig

class EvaluationConfig(BaseModel):
    metrics: List[str]
    plot_n_examples: int

class LoggingConfig(BaseModel):
    checkpoint_dir: str

class HardwareConfig(BaseModel):
    device: str
    use_amp: bool

class ExperimentConfig(BaseModel):
    experiment_name: str
    framework: FrameworkConfig
    model: ModelConfig
    data: DataConfig
    training: TrainingConfig
    evaluation: EvaluationConfig
    logging: LoggingConfig
    hardware: HardwareConfig

# --- Helper Function to Load Config ---

def load_config(config_path: str) -> ExperimentConfig:
    """
    Loads a YAML configuration file, parses it using Pydantic models,
    and returns a validated ExperimentConfig object.

    Args:
        config_path (str): The path to the YAML configuration file.

    Returns:
        ExperimentConfig: A Pydantic object representing the config.
    """
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Pydantic doesn't handle nested aliases well, so we manually rename one key
    # before parsing if it exists.
    if 'params' in config_dict.get('model', {}) and 'srcnn_kernels' in config_dict['model']['params']:
        config_dict['model']['params']['srcnn_params'] = {
            'srcnn_channels': config_dict['model']['params'].pop('srcnn_channels'),
            'srcnn_kernels': config_dict['model']['params'].pop('srcnn_kernels'),
            'num_tasks_for_model': config_dict['model']['params'].pop('num_tasks_for_model')
        }

    return ExperimentConfig.model_validate(config_dict)

if __name__ == '__main__':
    # Example of how to load the configuration
    # This block will only run when the script is executed directly
    # Assumes the config file is in the correct relative path
    try:
        config = load_config('../config/unet_srcnn_refactored.yaml')
        print("Configuration loaded successfully!")
        print("\nExperiment Name:", config.experiment_name)
        print("\nModel Config:", config.model.name)
        print("\nUse Domain-Specific BN:", config.model.params.use_domain_specific_bn)
        print("\nTask 0 LoRA Rank:", config.model.params.task_lora_ranks[0])
        print("\nData Directory:", config.data.data_dir)
        print("\nTraining epochs per task:", config.training.epochs_per_task)
    except Exception as e:
        print(f"Error loading or parsing configuration: {e}") 