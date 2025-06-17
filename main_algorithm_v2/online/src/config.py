"""
Configuration management for online continual learning.
Extends the offline configuration with online-specific parameters.
"""

import yaml
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import os

# Import the offline config
from main_algorithm_v2.offline.src.config import ExperimentConfig as OfflineExperimentConfig, load_config as load_offline_config

class OnlineBufferConfig(BaseModel):
    enabled: bool = Field(True, description="Enable online replay buffers")
    size: int = Field(100, description="Size of online buffer per domain")
    strategy: str = Field("fifo", description="Memory management strategy")
    # Gradient diversity parameters
    diversity_threshold: float = Field(0.8, description="Cosine similarity threshold for gradient diversity")
    min_samples_for_diversity: int = Field(5, description="Minimum samples before applying diversity check")

class OnlineEWCConfig(BaseModel):
    enabled: bool = Field(True, description="Enable EWC regularization during online training")
    lambda_ewc: float = Field(1000.0, description="EWC regularization strength")
    load_from_checkpoint: bool = Field(True, description="Load Fisher matrices from offline checkpoint")
    update_fisher_online: bool = Field(False, description="Update Fisher matrices during online training")
    exclude_current_task: bool = Field(True, description="Exclude current task from EWC penalty")

class TriggerConfig(BaseModel):
    type: str = Field("hybrid", description="Trigger type: time, volume, or hybrid")
    max_time: float = Field(60.0, description="Max seconds between updates (hybrid/time)")
    max_samples: int = Field(100, description="Max samples between updates (hybrid/volume)")
    time_weight: float = Field(1.0, description="Weight for time component in hybrid trigger")
    volume_weight: float = Field(1.0, description="Weight for volume component in hybrid trigger")
    time_interval: float = Field(30.0, description="Interval for time-only trigger (seconds)")
    volume_interval: int = Field(50, description="Interval for volume-only trigger (samples)")

class MixedBatchConfig(BaseModel):
    m_offline: int = Field(8, description="Offline samples per batch")
    m_online: int = Field(8, description="Online samples per batch")
    adaptive_weighting: bool = Field(True, description="Enable adaptive offline/online ratio")
    similarity_threshold: float = Field(0.7, description="Threshold for similarity comparison")
    max_offline_ratio: float = Field(0.8, description="Maximum fraction of offline samples")
    min_offline_ratio: float = Field(0.2, description="Minimum fraction of offline samples")

class OnlineEvaluationConfig(BaseModel):
    num_evaluations: int = Field(default=1000, description="Total number of online evaluations")
    domain_selection: str = Field(default="random", description="How to select domains (random, sequential)")
    timing_precision: str = Field(default="millisecond", description="Timing measurement precision")
    log_frequency: int = Field(default=10, description="Log every N evaluations")
    verbose: bool = Field(default=True, description="Enable verbose console output (initialization, detailed status, errors always shown)")
    online_buffer: OnlineBufferConfig = Field(default_factory=OnlineBufferConfig, description="Online buffer configuration")

class OnlineTrainingConfig(BaseModel):
    enabled: bool = Field(False, description="Enable online training")
    ewc: OnlineEWCConfig = Field(default_factory=OnlineEWCConfig, description="EWC configuration for online training")
    learning_rate: float = Field(0.0001, description="Learning rate for online training")
    weight_decay: float = Field(0.0001, description="Weight decay for online training")
    batch_size: int = Field(16, description="Batch size for online training")
    max_epochs_per_trigger: int = Field(5, description="Maximum epochs per training trigger")
    early_stopping_patience: int = Field(3, description="Early stopping patience for online training")
    trigger: TriggerConfig = Field(default_factory=TriggerConfig, description="Trigger configuration for network updates")
    mixed_batch: MixedBatchConfig = Field(default_factory=MixedBatchConfig, description="Mixed batch configuration")
    
class RawDataConfig(BaseModel):
    base_path: str = Field(..., description="Path to raw .mat files")
    domains: List[int] = Field(default=[0, 1, 2, 3, 4, 5, 6, 7, 8], description="Available domains")
    domain_file_mapping: Dict[int, str] = Field(..., description="Mapping from domain ID to .mat filename")

class OnlineMetricsConfig(BaseModel):
    metrics: List[str] = Field(default=["nmse", "inference_time", "interpolation_time"], description="Metrics to track")
    save_detailed_results: bool = Field(default=True, description="Whether to save detailed per-evaluation results")

class OnlineExperimentConfig(BaseModel):
    experiment_name: str
    offline_config_path: Optional[str] = Field(None, description="Path to the offline experiment config (optional if config is in checkpoint)")
    offline_checkpoint_path: str = Field(..., description="Path to the trained offline model checkpoint")
    
    # Online-specific configurations
    online_evaluation: OnlineEvaluationConfig
    online_training: OnlineTrainingConfig = Field(default_factory=OnlineTrainingConfig, description="Online training configuration")
    raw_data: RawDataConfig
    online_metrics: OnlineMetricsConfig
    
    # Reference to loaded offline config (will be populated at runtime)
    offline_config: Optional[OfflineExperimentConfig] = None

def load_online_config(config_path: str) -> OnlineExperimentConfig:
    """
    Loads an online configuration file and the referenced offline configuration.
    The offline config can be loaded either from a separate file or extracted from the checkpoint.
    
    Args:
        config_path (str): Path to the online YAML configuration file
        
    Returns:
        OnlineExperimentConfig: Complete online configuration with offline config loaded
    """
    with open(config_path, 'r') as f:
        online_config_dict = yaml.safe_load(f)
    
    # Parse the online config
    online_config = OnlineExperimentConfig.model_validate(online_config_dict)
    
    # Determine how to load the offline config
    if online_config.offline_config_path is not None:
        # Load config from separate file
        offline_config_path = online_config.offline_config_path
        if not os.path.isabs(offline_config_path):
            # Check if path starts with main_algorithm_v2 (project root relative)
            if offline_config_path.startswith("main_algorithm_v2"):
                # Path is relative to project root, use as-is
                pass
            else:
                # Make path relative to the online config file
                config_dir = os.path.dirname(config_path)
                offline_config_path = os.path.join(config_dir, offline_config_path)
        
        if not os.path.exists(offline_config_path):
            raise FileNotFoundError(f"Offline config file not found: {offline_config_path}")
        
        offline_config = load_offline_config(offline_config_path)
        print(f" Offline config loaded from separate file: {offline_config_path}")
        
    else:
        # Extract config from checkpoint
        checkpoint_path = online_config.offline_checkpoint_path
        if not os.path.isabs(checkpoint_path):
            if checkpoint_path.startswith("main_algorithm_v2"):
                # Path is relative to project root, use as-is
                pass
            else:
                # Make path relative to the online config file
                config_dir = os.path.dirname(config_path)
                checkpoint_path = os.path.join(config_dir, checkpoint_path)
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        
        print(f" Loading offline config from checkpoint: {checkpoint_path}")
        import torch
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            if 'config' not in checkpoint:
                raise KeyError("No 'config' key found in checkpoint. You may need to specify offline_config_path instead.")
            
            config_data = checkpoint['config']
            
            # Handle both dict and object configs
            if isinstance(config_data, dict):
                offline_config = OfflineExperimentConfig.model_validate(config_data)
            else:
                # Assume it's already an OfflineExperimentConfig object
                offline_config = config_data
                
            print(f" Offline config extracted from checkpoint successfully")
            
        except Exception as e:
            raise RuntimeError(f"Error extracting config from checkpoint: {e}")
    
    online_config.offline_config = offline_config
    
    print(f" Online config loaded from: {config_path}")
    print(f" Experiment: {online_config.experiment_name}")
    print(f" Model: {offline_config.model.name}")
    
    return online_config

if __name__ == '__main__':
    # Example usage
    try:
        config = load_online_config('../config/online_config.yaml')
        print("Online configuration loaded successfully!")
        print(f"Experiment: {config.experiment_name}")
        print(f"Domains: {config.raw_data.domains}")
        print(f"Offline model: {config.offline_config.model.name}")
    except Exception as e:
        print(f"Error loading configuration: {e}") 