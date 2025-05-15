"""
Filename: continual_learning/datasets/supermask/dataset_utils.py
Author: Vincenzo Nannetti
Date: 15/05/2025
Description: Dataset utilities for supermask models

Dependencies:
    - PyTorch
    - numpy
"""
import os
import torch
from torch.utils.data import DataLoader

# Import supermask dataset
from continual_learning.datasets.supermask.supermask_dataset import SupermaskDataset

SUPPORTED_DATASETS = {
    "supermask": SupermaskDataset,
}

def load_data(config, mode='train'):
    """
    Loads supermask dataset.
    Returns the raw dataset instance without splitting,
    as splitting will be done per-task during training/evaluation.
    
    Args:
        config (dict): Configuration dictionary
        mode (str): 'train', 'eval', or 'test'
        
    Returns:
        tuple: (dataloaders, norm_info)
    """
    print("\n" + "-"*60)
    print("SUPERMASK DATASET LOADING")
    print("-"*60)
    
    data_config      = config.get('data', {})
    supermask_config = config.get('supermask', {})
    
    # --- Determine Dataset Type ---
    dataset_type = data_config.get('dataset_type', 'supermask')
    
    if dataset_type != 'supermask':
        print(f"Warning: Expected dataset_type 'supermask', got '{dataset_type}'. Forcing to 'supermask'.")
        dataset_type = 'supermask'
    
    if dataset_type not in SUPPORTED_DATASETS:
        raise ValueError(f"Unsupported dataset_type: '{dataset_type}'. Supported: {list(SUPPORTED_DATASETS.keys())}")
    
    # Instantiate the dataset class
    DatasetClass = SUPPORTED_DATASETS[dataset_type]
    print(f"Using dataset type: {dataset_type}")
    
    # --- Instantiate Dataset ---
    try:
        # For supermask dataset, pass both data_config and supermask_config
        dataset = DatasetClass(data_config, supermask_config)
        print(f"Supermask dataset loaded with {supermask_config.get('tasks', 0)} tasks")
        
        # For supermask, we return the raw dataset instance
        # The splitting, normalisation, and dataloader creation is done per-task in the trainer/evaluator
        dataloaders = {'supermask_instance': dataset}
        
        # Return empty normalisation info for supermask
        # Normalisation is calculated and applied per-task in the trainer/evaluator
        norm_info = {
            "type": data_config.get('normalisation', 'none'),
            "params_input": None,
            "params_target": None
        }
        
        print(f"Returning supermask dataset instance. Splitting and normalisation will be done per-task.")
        return dataloaders, norm_info
        
    except Exception as e:
        print(f"Error instantiating dataset: {e}")
        raise 