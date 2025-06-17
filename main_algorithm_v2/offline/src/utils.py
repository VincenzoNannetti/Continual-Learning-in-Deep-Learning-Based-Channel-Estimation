"""
Utility functions for the continual learning experiment, including
checkpointing, device management, and logging setup.
"""
import os
import torch
import torch.nn as nn
from datetime import datetime
import random
import numpy as np
from typing import Dict, Any, Optional

# Assuming these modules are in the same src directory
from .model import UNet_SRCNN_LoRA
from .config import ExperimentConfig

def get_device(device_preference: str = 'auto') -> torch.device:
    """
    Determines the best available device for computation.

    Args:
        device_preference (str): 'auto', 'cpu', 'cuda', or 'mps'

    Returns:
        torch.device: The selected device
    """
    if device_preference == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')
    else:
        return torch.device(device_preference)

def set_seed(seed: int):
    """
    Set random seeds for reproducibility.

    Args:
        seed (int): Random seed to use
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def create_scheduler(optimizer, scheduler_config):
    """
    Create a learning rate scheduler based on config.
    
    Args:
        optimizer: PyTorch optimizer
        scheduler_config: Scheduler configuration
        
    Returns:
        Learning rate scheduler or None if not specified
    """
    if not scheduler_config or scheduler_config.type is None:
        return None
        
    scheduler_type = scheduler_config.type
    params = scheduler_config.params.model_dump() if scheduler_config.params else {}
    
    # Filter out None values from params
    params = {k: v for k, v in params.items() if v is not None}
    
    if scheduler_type == 'ReduceLROnPlateau':
        from torch.optim.lr_scheduler import ReduceLROnPlateau
        return ReduceLROnPlateau(optimizer, **params)
    elif scheduler_type == 'StepLR':
        from torch.optim.lr_scheduler import StepLR
        return StepLR(optimizer, **params)
    elif scheduler_type == 'ExponentialLR':
        from torch.optim.lr_scheduler import ExponentialLR
        return ExponentialLR(optimizer, **params)
    elif scheduler_type == 'CosineAnnealingLR':
        from torch.optim.lr_scheduler import CosineAnnealingLR
        return CosineAnnealingLR(optimizer, **params)
    else:
        print(f"Warning: Unknown scheduler type '{scheduler_type}'. No scheduler will be used.")
        return None

def save_checkpoint(model: UNet_SRCNN_LoRA, config: ExperimentConfig, filename: str):
    """
    Saves the model state, including all adapters and the configuration.

    The state dict of the UNet_SRCNN_LoRA model will contain the frozen
    backbone weights as well as the trained parameters for all LoRA adapters
    and domain-specific Batch Normalization layers across all tasks.

    Args:
        model (UNet_SRCNN_LoRA): The model to save.
        config (ExperimentConfig): The experiment configuration.
        filename (str): The path to the output checkpoint file.
    """
    checkpoint_dir = os.path.dirname(filename)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # We save the state_dict of the entire model.
    # It includes the backbone and all dynamically added modules (LoRA, DomainBN).
    state_to_save = {
        'model_state_dict': model.state_dict(),
        'config': config.model_dump() # Save the config for easy reloading
    }

    torch.save(state_to_save, filename)
    print(f"Checkpoint saved successfully to {filename}")

def save_lora_checkpoint(model: UNet_SRCNN_LoRA, config: ExperimentConfig, 
                        filepath: str, replay_buffers: Optional[Dict] = None):
    """
    Save a comprehensive checkpoint containing the model, config, and replay buffers.

    Args:
        model: The trained LoRA model
        config: Experiment configuration
        filepath: Path to save the checkpoint
        replay_buffers: Dictionary of replay buffers keyed by task_id
    """
    print(f"Saving checkpoint to: {filepath}")
    
    # Prepare checkpoint data
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'config': config.model_dump(),
        'model_class': 'UNet_SRCNN_LoRA',
        'timestamp': datetime.now().isoformat(),
        'training_completed': True
    }
    
    # Add replay buffers if provided
    if replay_buffers:
        print(f"Saving replay buffers for {len(replay_buffers)} tasks")
        replay_buffer_data = {}
        for task_id, buffer in replay_buffers.items():
            replay_buffer_data[task_id] = buffer.state_dict()
        checkpoint['replay_buffers'] = replay_buffer_data
    
    # Count trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    checkpoint['model_info'] = {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'lora_efficiency': f"{(trainable_params/total_params)*100:.2f}%"
    }
    
    # Save the checkpoint
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved successfully!")
    print(f"  - Total parameters: {total_params:,}")
    print(f"  - Trainable parameters: {trainable_params:,} ({(trainable_params/total_params)*100:.2f}%)")
    if replay_buffers:
        total_replay_samples = sum(len(buffer) for buffer in replay_buffers.values())
        print(f"  - Replay buffer samples: {total_replay_samples:,}")

def load_lora_model_for_evaluation(checkpoint_path: str, device: torch.device) -> UNet_SRCNN_LoRA:
    """
    Load a trained LoRA model from checkpoint for evaluation.
    Handles both old and new checkpoint formats.

    Args:
        checkpoint_path: Path to the checkpoint file
        device: Device to load the model on

    Returns:
        Loaded and configured UNet_SRCNN_LoRA model
    """
    print(f"Loading model from checkpoint: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Reconstruct config
    config_dict = checkpoint['config']
    config = ExperimentConfig.model_validate(config_dict)
    
    # Create model
    model = UNet_SRCNN_LoRA(config)
    
    # Handle checkpoint format compatibility
    state_dict = checkpoint['model_state_dict']
    model_state_dict = model.state_dict()
    
    # Check if this is the old format (has lora_A.0, lora_A.1, etc.)
    old_format_keys = [k for k in state_dict.keys() if 'lora_A.' in k and k.split('.')[-1].isdigit()]
    
    if old_format_keys:
        print("Converting old checkpoint format to new format...")
        
        # Convert old format to new format
        converted_state_dict = {}
        
        # Copy non-LoRA parameters as-is
        for key, value in state_dict.items():
            if 'lora_A.' not in key and 'lora_B.' not in key:
                converted_state_dict[key] = value
        
        # Group old LoRA parameters by module and task
        lora_groups = {}
        for key in state_dict.keys():
            if 'lora_A.' in key or 'lora_B.' in key:
                parts = key.split('.')
                task_id = parts[-1]  # Last part is task ID
                param_type = parts[-2]  # lora_A or lora_B
                module_path = '.'.join(parts[:-2])  # Everything else
                
                if module_path not in lora_groups:
                    lora_groups[module_path] = {}
                if task_id not in lora_groups[module_path]:
                    lora_groups[module_path][task_id] = {}
                
                lora_groups[module_path][task_id][param_type] = state_dict[key]
        
        # Convert to new format: module.task_adapters.{task_id}.{A/B}
        for module_path, tasks in lora_groups.items():
            for task_id, params in tasks.items():
                if 'lora_A' in params and 'lora_B' in params:
                    converted_state_dict[f"{module_path}.task_adapters.{task_id}.A"] = params['lora_A']
                    converted_state_dict[f"{module_path}.task_adapters.{task_id}.B"] = params['lora_B']
        
        # Load converted state dict
        missing_keys, unexpected_keys = model.load_state_dict(converted_state_dict, strict=False)
        
        if missing_keys:
            print(f"Missing keys (expected for new parameters): {len(missing_keys)}")
        if unexpected_keys:
            print(f"Unexpected keys: {len(unexpected_keys)}")
            
        # Add tasks from checkpoint to model
        print("Adding tasks to model from checkpoint...")
        tasks_in_checkpoint = set()
        for module_path, tasks in lora_groups.items():
            for task_id in tasks.keys():
                tasks_in_checkpoint.add(task_id)
        
        for task_id in sorted(tasks_in_checkpoint):
            model.add_task(task_id)
            print(f"Added task {task_id}")
    
    else:
        # New format - need to add tasks first, then load
        print("Loading new checkpoint format...")
        
        # Extract tasks from state dict
        new_format_keys = [k for k in state_dict.keys() if 'task_adapters.' in k]
        if new_format_keys:
            # Extract task IDs from the new format keys
            tasks_in_checkpoint = set()
            for key in new_format_keys:
                parts = key.split('.')
                # Find the task_adapters part and get the task ID after it
                for i, part in enumerate(parts):
                    if part == 'task_adapters' and i + 1 < len(parts):
                        task_id = parts[i + 1]
                        tasks_in_checkpoint.add(task_id)
                        break
            
            print(f"Found tasks in checkpoint: {sorted(tasks_in_checkpoint)}")
            
            # Add tasks to model first
            for task_id in sorted(tasks_in_checkpoint):
                model.add_task(task_id)
                print(f"Added task {task_id}")
        
        # Now load the state dict
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        
        if missing_keys:
            print(f"Missing keys: {len(missing_keys)}")
        if unexpected_keys:
            print(f"Unexpected keys: {len(unexpected_keys)}")
    
    model.to(device)
    
    # Attach config to model for easy access
    model.config = config
    
    print(f"Model loaded successfully from {checkpoint_path}")
    return model

def get_checkpoint_filename(config: ExperimentConfig) -> str:
    """
    Generate a checkpoint filename based on the experiment configuration.

    Args:
        config: Experiment configuration

    Returns:
        Generated filename for the checkpoint
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{config.experiment_name}-{timestamp}.pth"
    
    # Ensure the checkpoint directory exists
    os.makedirs(config.logging.checkpoint_dir, exist_ok=True)
    
    return os.path.join(config.logging.checkpoint_dir, filename) 