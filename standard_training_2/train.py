"""
Lightweight training script for Standard Training 2.0
Loads data, adds noise, interpolates, and trains UNet model.
Applies Z-score normalisation based on training set statistics.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset, Subset
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
import torch.amp
import yaml
import os
import sys
from pathlib import Path
import numpy as np
import argparse # For command-line arguments
import wandb # For W&B integration
import time
import matplotlib.pyplot as plt

from standard_training_2.dataset import StandardDataset, NormalisingDatasetWrapper
from standard_training_2.models.unet import UNetModel
from standard_training_2.models.unet_srcnn import UNetCombinedModel
from standard_training_2.models.srcnn import SRCNN
from standard_training_2.models.dncnn import DnCNN
from standard_training_2.models.autoencoder import DenoisingAutoencoder
from standard_training_2.models.residual_autoencoder import DenoisingResAutoencoder
from standard_training_2.models.srcnn_dncnn import CombinedModel_SRCNNDnCNN
from standard_training_2.models.ae_srcnn import CombinedModel_AESRCNN
from standard_training_2.tests.plotting_utils import plot_training_curves, plot_evaluation_samples # Import from new location

def load_config(config_path):
    """Load configuration file safely."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def calculate_channel_wise_stats(dataset_subset):
    """
    Calculates channel-wise mean and std for inputs and targets from a dataset subset.
    Assumes data items are (x, y) where x and y are tensors of shape (H, W, C).
    Returns: (mean_inputs, std_inputs), (mean_targets, std_targets)
             Each stat is a tensor of shape (1, 1, C) or (C,).
    """
    # Use a DataLoader to iterate efficiently, even if batch_size is 1
    # num_workers=0 for this calculation to avoid issues with subprocesses if any
    loader = DataLoader(dataset_subset, batch_size=64, shuffle=False, num_workers=0) 
    
    # For inputs (x)
    all_x = []
    for x_batch, _ in loader:
        all_x.append(x_batch)
    
    if not all_x:
        raise ValueError("Training subset is empty, cannot calculate statistics.")

    # Concatenate all input batches: result shape (N_train, H, W, C)
    # x_batch from loader is (batch_size, H, W, C)
    all_x_tensor = torch.cat(all_x, dim=0)
    
    # Calculate mean and std over N_train, H, W dimensions, keeping C dimension
    # Resulting shape (C,)
    mean_inputs = torch.mean(all_x_tensor, dim=(0, 1, 2))
    std_inputs = torch.std(all_x_tensor, dim=(0, 1, 2))

    # For targets (y)
    all_y = []
    for _, y_batch in loader:
        all_y.append(y_batch)
    
    all_y_tensor = torch.cat(all_y, dim=0)
    mean_targets = torch.mean(all_y_tensor, dim=(0, 1, 2))
    std_targets = torch.std(all_y_tensor, dim=(0, 1, 2))
    
    # Reshape to (1, 1, C) for broadcasting during normalisation
    num_channels = mean_inputs.shape[0]
    mean_inputs = mean_inputs.view(1, 1, num_channels)
    std_inputs = std_inputs.view(1, 1, num_channels)
    mean_targets = mean_targets.view(1, 1, num_channels)
    std_targets = std_targets.view(1, 1, num_channels)

    print(f"Calculated Input Stats: Mean shape {mean_inputs.shape}, Std shape {std_inputs.shape}")
    print(f"Calculated Target Stats: Mean shape {mean_targets.shape}, Std shape {std_targets.shape}")

    return (mean_inputs, std_inputs), (mean_targets, std_targets)

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=10, min_delta=0, verbose=False, checkpoint_path='best_model.pth', config_to_save=None):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 10
            min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                               Default: 0
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            checkpoint_path (Path or str): Path to save the best model checkpoint.
                                         Default: 'best_model.pth'
            config_to_save (dict): The configuration dictionary to save with the checkpoint.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.checkpoint_path = Path(checkpoint_path) # Ensure it's a Path object
        self.config_to_save = config_to_save
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_epoch = -1

    def __call__(self, val_loss, model, optimizer, epoch):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model, optimizer, epoch)
            self.best_epoch = epoch
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                print("Early stopping triggered.")
        else:
            if val_loss < self.best_loss: # Condition corrected: only save if val_loss is strictly better
                if self.verbose:
                    print(f'Validation loss decreased ({self.best_loss:.6f} --> {val_loss:.6f}). Saving model ...')
                self.best_loss = val_loss
                self.save_checkpoint(val_loss, model, optimizer, epoch)
                self.best_epoch = epoch
                self.counter = 0 # Reset counter on improvement

    def save_checkpoint(self, val_loss, model, optimizer, epoch):
        """Saves model checkpoint."""
        # This verbose print is fine, it confirms the save action.
        # The print in __call__ announces the improvement that leads to this save.
        print(f"Saving checkpoint for epoch {epoch+1} with Val Loss: {val_loss:.6f} at {self.checkpoint_path}")
        
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'config': self.config_to_save # Save the full config including norm_stats
        }, self.checkpoint_path)



def train_epoch(model, train_loader, criterion, optimizer, device, use_amp=False, scaler=None):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Debug: Print shapes before permutation
        if batch_idx == 0:
            print(f"DEBUG: Input shape before permute: {inputs.shape}")
            print(f"DEBUG: Target shape before permute: {targets.shape}")
        
        # Only permute if data is in (B, H, W, C) format
        if inputs.dim() == 4 and inputs.shape[-1] == 2:
            inputs = inputs.permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)
        if targets.dim() == 4 and targets.shape[-1] == 2:
            targets = targets.permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)
        
        # Debug: Print shapes after permutation
        if batch_idx == 0:
            print(f"DEBUG: Input shape after permute: {inputs.shape}")
            print(f"DEBUG: Target shape after permute: {targets.shape}")
        
        optimizer.zero_grad()
        
        # Use torch.amp.autocast for device-agnostic autocasting if desired,
        # but here we explicitly use 'cuda' as GradScaler is CUDA-specific.
        with torch.amp.autocast(device_type=device.type, dtype=torch.float16 if device.type == 'cuda' else torch.bfloat16, enabled=use_amp):
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
        if use_amp and scaler: # scaler is only used if use_amp is True and device is cuda
            scaler.scale(loss).backward() 
            scaler.step(optimizer)         
            scaler.update()                
        else: # Handles non-AMP or non-CUDA cases
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
    return total_loss / num_batches

def validate_epoch(model, val_loader, criterion, device, use_amp=False):
    """Validate for one epoch."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Only permute if data is in (B, H, W, C) format
            if inputs.dim() == 4 and inputs.shape[-1] == 2:
                inputs = inputs.permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)
            if targets.dim() == 4 and targets.shape[-1] == 2:
                targets = targets.permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)
            
            # Use torch.amp.autocast here as well for consistency
            with torch.amp.autocast(device_type=device.type, dtype=torch.float16 if device.type == 'cuda' else torch.bfloat16, enabled=use_amp):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches

def load_model(config):
    """
    Load model based on configuration.
    Supports all model types: 'unet', 'srcnn', 'dncnn', 'unet_srcnn', 'combined_srcnn_dncnn', 
    'combined_ae_srcnn', 'denoising_autoencoder', and 'denoising_res_autoencoder'.
    """
    model_name = config['model']['name'].lower()
    model_params = config['model']['params']
    
    # Parse string-based parameters if they exist
    if 'srcnn_channels' in config['model']['params'] and isinstance(config['model']['params']['srcnn_channels'], str):
        # Convert string like "64_32" to list [64, 32]
        channels_str = config['model']['params']['srcnn_channels']
        config['model']['params']['srcnn_channels'] = [int(x) for x in channels_str.split('_')]

    if 'srcnn_kernels' in config['model']['params'] and isinstance(config['model']['params']['srcnn_kernels'], str):
        # Convert string like "9_1_5" to list [9, 1, 5]
        kernels_str = config['model']['params']['srcnn_kernels']
        config['model']['params']['srcnn_kernels'] = [int(x) for x in kernels_str.split('_')]
    
    if model_name == 'unet':
        model = UNetModel(
            in_channels=model_params.get('in_channels', 2),
            base_features=model_params.get('base_features', 16),
            use_batch_norm=model_params.get('use_batch_norm', False),
            depth=model_params.get('depth', 2),
            activation=model_params.get('activation', 'relu'),
            use_leaky_relu=model_params.get('use_leaky_relu', False),  # Backward compatibility
            leaky_slope=model_params.get('leaky_slope', 0.01),
            verbose=model_params.get('verbose', False)
        )
        print(f"[SUCCESS] Created UNet model with {sum(p.numel() for p in model.parameters()):,} parameters")
        print(f"   - Depth: {model_params.get('depth', 2)} layers")
        print(f"   - Activation: {model_params.get('activation', 'relu').upper()}")
        print(f"   - Batch Norm: {'Enabled' if model_params.get('use_batch_norm', False) else 'Disabled'}")
        
    elif model_name == 'unet_srcnn':
        # Extract UNet arguments
        unet_args = {
            'in_channels': model_params.get('in_channels', 2),
            'base_features': model_params.get('base_features', 16),
            'use_batch_norm': model_params.get('use_batch_norm', False),
            'depth': model_params.get('depth', 2),
            'activation': model_params.get('activation', 'relu'),
            'use_leaky_relu': model_params.get('use_leaky_relu', False),  # Backward compatibility
            'leaky_slope': model_params.get('leaky_slope', 0.01),
            'verbose': model_params.get('verbose', False)
        }
        
        # Extract SRCNN arguments
        srcnn_channels = model_params.get('srcnn_channels', [64, 32])
        srcnn_kernels = model_params.get('srcnn_kernels', [9, 1, 5])
        
        # Extract combined model arguments
        pretrained_unet = model_params.get('pretrained_unet', None)
        pretrained_srcnn = model_params.get('pretrained_srcnn', None)
        freeze_unet = model_params.get('freeze_unet', False)
        freeze_srcnn = model_params.get('freeze_srcnn', False)
        
        model = UNetCombinedModel(
            pretrained_unet=pretrained_unet,
            pretrained_srcnn=pretrained_srcnn,
            unet_args=unet_args,
            srcnn_channels=srcnn_channels,
            srcnn_kernels=srcnn_kernels,
        )
        
        # Apply freezing if specified
        if freeze_unet:
            model.freeze_unet()
            print("[LOCKED] UNet weights frozen")
            
        if freeze_srcnn:
            model.freeze_srcnn()
            print("[LOCKED] SRCNN weights frozen")
            
        # Count parameters for the combined model
        param_counts = model.count_submodule_parameters()
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"[SUCCESS] Created UNet+SRCNN combined model:")
        print(f"   - UNet: {param_counts['unet']:,} parameters (Depth: {unet_args['depth']}, Activation: {unet_args['activation'].upper()}, Batch Norm: {'Enabled' if unet_args['use_batch_norm'] else 'Disabled'})")
        print(f"   - SRCNN: {param_counts['srcnn']:,} parameters (Channels: {srcnn_channels}, Kernels: {srcnn_kernels})")
        print(f"   - Total: {param_counts['total']:,} parameters")
        print(f"   - Trainable: {trainable_params:,} parameters")
        
    elif model_name == 'srcnn':
        # SRCNN standalone model with configurable parameters
        channels = model_params.get('channels', [64, 32])
        kernels = model_params.get('kernels', [9, 1, 5])
        
        model = SRCNN(channels=channels, kernels=kernels)
        print(f"[SUCCESS] Created SRCNN model with {sum(p.numel() for p in model.parameters()):,} parameters")
        print(f"   - Channels: {channels}")
        print(f"   - Kernels: {kernels}")
        
    elif model_name == 'dncnn':
        # DnCNN standalone model
        num_channels = model_params.get('num_channels', 2)
        model = DnCNN(num_channels=num_channels)
        print(f"[SUCCESS] Created DnCNN model with {sum(p.numel() for p in model.parameters()):,} parameters")
        
    elif model_name == 'combined_srcnn_dncnn':
        # Combined SRCNN+DnCNN model (order configurable)
        order = model_params.get('order', 'srcnn_first')
        dncnn_args = model_params.get('dncnn_args', {'num_channels': 2})
        srcnn_args = model_params.get('srcnn_args', {})
        pretrained_dncnn = model_params.get('pretrained_dncnn', None)
        pretrained_srcnn = model_params.get('pretrained_srcnn', None)
        freeze_dncnn = model_params.get('freeze_dncnn', False)
        freeze_srcnn = model_params.get('freeze_srcnn', False)
        
        model = CombinedModel_SRCNNDnCNN(
            order=order,
            pretrained_dncnn_path=pretrained_dncnn,
            pretrained_srcnn_path=pretrained_srcnn,
            dncnn_args=dncnn_args,
            srcnn_args=srcnn_args
        )
        
        # Apply freezing if specified
        if freeze_dncnn:
            model.freeze_dncnn()
            print("[LOCKED] DnCNN weights frozen")
            
        if freeze_srcnn:
            model.freeze_srcnn()
            print("[LOCKED] SRCNN weights frozen")
            
        # Count parameters for the combined model
        param_counts = model.count_submodule_parameters()
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"[SUCCESS] Created {order.upper()} combined model:")
        print(f"   - DnCNN: {param_counts['dncnn']:,} parameters")
        print(f"   - SRCNN: {param_counts['srcnn']:,} parameters")
        print(f"   - Total: {param_counts['total']:,} parameters")
        print(f"   - Trainable: {trainable_params:,} parameters")
        
    elif model_name == 'combined_ae_srcnn':
        # Combined Autoencoder+SRCNN model
        autoencoder_type = model_params.get('autoencoder_type', 'residual')
        pretrained_autoencoder = model_params.get('pretrained_autoencoder', None)
        pretrained_srcnn = model_params.get('pretrained_srcnn', None)
        freeze_autoencoder = model_params.get('freeze_autoencoder', False)
        freeze_srcnn = model_params.get('freeze_srcnn', False)
        
        model = CombinedModel_AESRCNN(
            autoencoder_type=autoencoder_type,
            pretrained_autoencoder=pretrained_autoencoder,
            pretrained_srcnn=pretrained_srcnn
        )
        
        # Apply freezing if specified
        if freeze_autoencoder:
            model.freeze_autoencoder()
            print("[LOCKED] Autoencoder weights frozen")
            
        if freeze_srcnn:
            model.freeze_srcnn()
            print("[LOCKED] SRCNN weights frozen")
            
        # Count parameters for the combined model
        param_counts = model.count_submodule_parameters()
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"[SUCCESS] Created {autoencoder_type.title()} Autoencoder+SRCNN combined model:")
        print(f"   - Autoencoder: {param_counts['autoencoder']:,} parameters")
        print(f"   - SRCNN: {param_counts['srcnn']:,} parameters")
        print(f"   - Total: {param_counts['total']:,} parameters")
        print(f"   - Trainable: {trainable_params:,} parameters")
        
    elif model_name == 'denoising_autoencoder':
        # Basic denoising autoencoder
        input_channels = model_params.get('input_channels', 2)
        model = DenoisingAutoencoder(input_channels=input_channels)
        print(f"[SUCCESS] Created Basic Denoising Autoencoder with {sum(p.numel() for p in model.parameters()):,} parameters")
            
    elif model_name == 'denoising_res_autoencoder':
        # Residual denoising autoencoder
        input_channels = model_params.get('input_channels', 2)
        model = DenoisingResAutoencoder(input_channels=input_channels)
        print(f"[SUCCESS] Created Residual Denoising Autoencoder with {sum(p.numel() for p in model.parameters()):,} parameters")
        
    else:
        raise ValueError(f"Unsupported model name: {model_name}. Supported: 'unet', 'srcnn', 'dncnn', 'unet_srcnn', 'combined_srcnn_dncnn', 'combined_ae_srcnn', 'denoising_autoencoder', 'denoising_res_autoencoder'")
    
    return model

def main(args):
    """Main training function."""
    print("=== Standard Training 2.0 - Training Script ===")
    
    # --- W&B Setup (if arguments provided) ---
    wandb_active = False
    if args.wandb_run_id and args.wandb_project:
        try:
            print(f"Initializing W&B: Project='{args.wandb_project}', Entity='{args.wandb_entity}', RunID='{args.wandb_run_id}'")
            
            wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                id=args.wandb_run_id,
                resume="never",  # Never try to resume, always create new
                config=None,
                settings=wandb.Settings(console="off")
            )
            wandb_active = True
            print(f"W&B initialized successfully!")
                    
        except Exception as e:
            print(f"Warning: Failed to initialize W&B: {e}")
            print("Proceeding without W&B logging for this run.")
            wandb_active = False
    else:
        print("W&B arguments not provided, proceeding without W&B logging.")

    # --- Load configuration ---
    script_dir = Path(__file__).resolve().parent
    config_path = Path(args.config_path) if args.config_path else script_dir / "unet.yaml"
    
    if not config_path.exists():
        print(f"Error: Configuration file not found at {config_path}")
        # If W&B is active, log error and exit
        if wandb_active:
            wandb.log({"error": f"Config file not found: {config_path}"})
            wandb.finish(exit_code=1)
        sys.exit(1)
        
    config = load_config(config_path)
    print(f"Configuration loaded from: {config_path}")

    # --- Print key configuration details for visibility ---
    print("\n--- Key Configuration Parameters ---")
    print(f"  Experiment Name: {config.get('experiment_name', 'N/A')}")
    print(f"  Model Name: {config.get('model', {}).get('name', 'N/A')}")
    print("  Model Params:")
    for param, value in config.get('model', {}).get('params', {}).items():
        print(f"    {param}: {value}")
    print("  Training Params:")
    for param, value in config.get('training', {}).items():
        if isinstance(value, dict):
            print(f"    {param}:")
            for k, v in value.items():
                print(f"      {k}: {v}")
        else:
            print(f"    {param}: {value}")
    print("----------------------------------")

    # If W&B is active, log the initial config (before norm_stats are added)
    # The full config (with norm_stats) will be part of the checkpoint and can be logged if needed
    if wandb_active:
        wandb.config.update(config) # Log the loaded config
    
    # Set random seeds
    torch.manual_seed(config['framework']['seed'])
    np.random.seed(config['framework']['seed'])
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # AMP (Automatic Mixed Precision) setup
    use_amp = config.get('hardware', {}).get('use_amp', False) and device.type == 'cuda'
    scaler = None
    if use_amp:
        scaler = torch.cuda.amp.GradScaler()
        print("Automatic Mixed Precision (AMP) enabled.")
    else:
        print("AMP disabled or not available (CUDA required).")
    
    # Create base dataset (without normalisation yet)
    print("Loading base dataset...")
    try:
        # Check if we have sequence configuration
        sequence = config['data'].get('sequence', None)
        tasks_params = config['data'].get('tasks_params', None)
        
        if sequence is not None and not args.all_data:
            print(f"Using sequence-based loading for tasks: {sequence}")
        
        base_dataset = StandardDataset(
            data_dir=config['data']['data_dir'],
            data_name=config['data']['data_name'],
            snr=config['data']['snr'],
            preprocessed_dir=config['data']['preprocessed_dir'],
            all_data=args.all_data,
            sequence=sequence if not args.all_data else None,  # Only use sequence if not loading all data
            tasks_params=tasks_params if not args.all_data else None
        )
        if args.all_data:
            print(f"Combined dataset loaded: {len(base_dataset)} samples from all preprocessed datasets")
        elif sequence is not None:
            print(f"Sequence dataset loaded: {len(base_dataset)} samples from tasks {sequence}")
        else:
            print(f"Base dataset loaded: {len(base_dataset)} samples")
    except Exception as e:
        print(f"Error loading base dataset: {e}")
        return

    # Calculate split sizes
    total_size = len(base_dataset)
    test_split_ratio = config['data']['test_split']
    val_split_ratio = config['data']['validation_split']
    
    if not (0 < test_split_ratio < 1 and 0 < val_split_ratio < 1 and (test_split_ratio + val_split_ratio) < 1):
        raise ValueError("Invalid split ratios. Ensure test_split and validation_split are between 0 and 1, and their sum is less than 1.")

    test_size = int(test_split_ratio * total_size)
    val_size = int(val_split_ratio * total_size)
    train_size = total_size - val_size - test_size
    
    print(f"Dataset split: Train={train_size}, Validation={val_size}, Test={test_size}")

    # Split dataset into train, validation, and test subsets
    train_subset, val_subset, test_subset = random_split(
        base_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(config['framework']['seed'])
    )
    print(f"Train subset size: {len(train_subset)}")
    print(f"Validation subset size: {len(val_subset)}")
    print(f"Test subset size: {len(test_subset)}")

    # Calculate normalisation statistics from the training subset
    print("Calculating normalisation statistics from training set...")
    (train_mean_inputs, train_std_inputs), (train_mean_targets, train_std_targets) = calculate_channel_wise_stats(train_subset)
    
    # Store normalisation stats in config (to be saved with checkpoint)
    # Convert to numpy for YAML serializability if needed, but PyTorch can save tensors in checkpoints.
    # For YAML, it's better to convert to lists of floats.
    config['data']['norm_stats'] = {
        'mean_inputs': train_mean_inputs.cpu().numpy().tolist(), # Ensure on CPU and convert to list
        'std_inputs': train_std_inputs.cpu().numpy().tolist(),
        'mean_targets': train_mean_targets.cpu().numpy().tolist(),
        'std_targets': train_std_targets.cpu().numpy().tolist()
    }
    print("normalisation statistics calculated and stored in config.")
    
    # Create normalising wrappers for each subset using ONLY training statistics
    norm_stats_for_wrapper = ( (train_mean_inputs.to(device), train_std_inputs.to(device)), \
                               (train_mean_targets.to(device), train_std_targets.to(device)) )

    # Move stats to the correct device for normalisation if datasets are on GPU (though subset is on CPU)
    # The normalisingDatasetWrapper will receive tensors; ensure they match device of data if necessary.
    # Data from subset[idx] is on CPU by default. normalisation happens before moving to device in training loop.
    # So, stats for wrapper should be on CPU.
    
    cpu_norm_stats = ( (train_mean_inputs.cpu(), train_std_inputs.cpu()), \
                       (train_mean_targets.cpu(), train_std_targets.cpu()) )

    normalised_train_dataset = NormalisingDatasetWrapper(train_subset, cpu_norm_stats)
    normalised_val_dataset   = NormalisingDatasetWrapper(val_subset, cpu_norm_stats)
    # normalised_test_dataset can be created here for evaluate.py to use the same way
    # config now contains the stats for evaluate.py to load and use.

    # Create data loaders using the normalised datasets
    train_loader = DataLoader(
        normalised_train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True, # Shuffle training data
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory']
    )
    
    val_loader = DataLoader(
        normalised_val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False, # No need to shuffle validation data
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory']
    )
    print(f"Train batches: {len(train_loader)}, Validation batches: {len(val_loader)}")
    
    # Create model using the new model loading function
    model = load_model(config).to(device)
    
    # Loss function and optimizer
    criterion = nn.MSELoss() 
    optimizer = optim.Adam(
        model.parameters(),
        lr=float(config['training']['learning_rate']),
        weight_decay=float(config['training'].get('weight_decay', 0)), # Use .get for weight_decay
        betas=tuple(config['training']['betas']), 
        eps=float(config['training']['eps'])
    )

    # Learning rate scheduler setup
    scheduler_config = config['training'].get('scheduler', {})
    scheduler_type = scheduler_config.get('type', 'ReduceLROnPlateau').lower()
    scheduler_params = scheduler_config.get('params', {})
    scheduler = None

    if scheduler_type == 'reducelronplateau':
        # Ensure default values are provided if not in config for critical params
        # Note: removed deprecated 'verbose' parameter
        plateau_params = {
            'mode': scheduler_params.get('mode', 'min'),
            'factor': float(scheduler_params.get('factor', 0.1)),
            'patience': int(scheduler_params.get('patience', 10)),
            'min_lr': float(scheduler_params.get('min_lr', 1e-7))
        }
        scheduler = ReduceLROnPlateau(optimizer, **plateau_params)
        print(f"Using ReduceLROnPlateau scheduler with params: {plateau_params}")
    elif scheduler_type == 'steplr':
        step_params = {
            'step_size': int(scheduler_params.get('step_size', 30)),
            'gamma': float(scheduler_params.get('gamma', 0.1))
        }
        scheduler = StepLR(optimizer, **step_params)
        print(f"Using StepLR scheduler with params: {step_params}")
    else:
        print("No scheduler or unknown scheduler type specified. Training without scheduler.")

    # Early stopping setup
    early_stopping_config = config['training'].get('early_stopping', {})
    es_patience = int(early_stopping_config.get('patience', 20))
    es_min_delta = float(early_stopping_config.get('min_delta', 0.00001))
    
    checkpoint_dir = Path(config['logging']['checkpoint_dir'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = checkpoint_dir / 'best_model.pth'

    # Pass the full config to EarlyStopping so it can be saved with the checkpoint
    early_stopper = EarlyStopping(
        patience=es_patience, 
        min_delta=es_min_delta, 
        verbose=True, 
        checkpoint_path=best_model_path,
        config_to_save=config # Pass the live config dictionary
    )
    print(f"Early stopping enabled: Patience={es_patience}, Min Delta={es_min_delta}")

    # Training loop
    print("\nStarting training...")
    all_train_losses = [] 
    all_val_losses = []   
    
    # Determine W&B logging frequency
    wandb_log_freq = args.wandb_log_freq if wandb_active else 0

    for epoch in range(config['training']['epochs']):
        epoch_num = epoch + 1
        print(f"\nEpoch {epoch_num}/{config['training']['epochs']}")
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Current learning rate: {current_lr:.2e}")
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, use_amp, scaler)
        
        # Validate
        val_loss = validate_epoch(model, val_loader, criterion, device, use_amp)
        
        print(f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        all_train_losses.append(train_loss)
        all_val_losses.append(val_loss)

        # W&B Logging (if active)
        if wandb_active and (wandb_log_freq > 0 and epoch_num % wandb_log_freq == 0 or epoch_num == config['training']['epochs']):
            log_dict = {
                "epoch": epoch_num,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "learning_rate": current_lr
            }
            # If early stopping has a best loss, log that too.
            if early_stopper.best_loss is not None:
                log_dict["best_val_loss_so_far"] = early_stopper.best_loss
            wandb.log(log_dict)
        
        # Scheduler step (depends on scheduler type)
        if scheduler:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_loss) # ReduceLROnPlateau needs the validation loss
            else:
                scheduler.step() # Other schedulers like StepLR step per epoch

        # Early stopping check (also handles saving the best model)
        early_stopper(val_loss, model, optimizer, epoch)
        if early_stopper.early_stop:
            print("Early stopping condition met. Terminating training.")
            break # Exit training loop
            
    print("\n=== Training Complete ===")
    # The best model is already saved by EarlyStopping at its optimal point.
    # We can retrieve the best validation loss from the early_stopper.
    if early_stopper.best_loss is not None:
        print(f"Best validation loss achieved: {early_stopper.best_loss:.6f} (saved at {early_stopper.checkpoint_path} at epoch {early_stopper.best_epoch +1})")
        if wandb_active:
            wandb.summary["best_val_loss"] = early_stopper.best_loss
            wandb.summary["best_epoch"] = early_stopper.best_epoch + 1
            wandb.save(str(early_stopper.checkpoint_path)) # Save best model to W&B
    else:
        print("Training completed without improvement or early stopping did not trigger a save.")
        if wandb_active:
            wandb.summary["best_val_loss"] = all_val_losses[-1] if all_val_losses else float('nan')
            wandb.summary["best_epoch"] = config['training']['epochs']

    # Optionally, save the model at the very end of training (last epoch), regardless of early stopping
    # This might be useful if early stopping was too aggressive or for other analysis.
    final_model_path = checkpoint_dir / 'model_at_last_epoch.pth' 
    torch.save({
        'epoch': epoch, # epoch will be the last completed epoch
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': all_train_losses[-1] if all_train_losses else float('nan'),
        'val_loss': all_val_losses[-1] if all_val_losses else float('nan'),
        'config': config 
    }, final_model_path)
    print(f"Model from last epoch saved to {final_model_path}")
    if wandb_active:
        wandb.save(str(final_model_path)) # Save final model to W&B

    # Plot training curves
    plot_save_path = checkpoint_dir / 'training_curves.png'
    plot_training_curves(all_train_losses, all_val_losses, plot_save_path)
    if wandb_active and plot_save_path.exists():
        try:
            wandb.log({"training_curves": wandb.Image(str(plot_save_path))})
            print(f"Training curves logged to W&B from {plot_save_path}")
        except Exception as e:
            print(f"Warning: Could not log training_curves.png to W&B: {e}")

    if wandb_active:
        # Save the final populated config (with norm_stats) as an artifact or in summary
        # wandb.config.update({'final_training_config': config}) # This might be too large for config directly.
        # Better to save as an artifact if needed or rely on it being in the checkpoint.
        # For now, the checkpoint contains it.
        # Log final summary metrics
        wandb.summary['final_train_loss'] = all_train_losses[-1] if all_train_losses else float('nan')
        wandb.summary['final_val_loss'] = all_val_losses[-1] if all_val_losses else float('nan')
        
        # Log key hyperparameters to W&B summary for easy viewing
        if config.get('model', {}).get('params'):
            for param, value in config['model']['params'].items():
                # Ensure value is W&B compatible (str, num, bool, or list of them)
                if isinstance(value, (list, tuple)) and any(isinstance(i, dict) for i in value):
                     wandb.summary[f"config_model_param_{param}"] = str(value) # convert complex lists to string
                else:
                    try:
                        wandb.summary[f"config_model_param_{param}"] = value
                    except Exception:
                        wandb.summary[f"config_model_param_{param}"] = str(value)

        if config.get('training'):
            for param, value in config['training'].items():
                if param not in ['scheduler', 'early_stopping', 'betas']: # Avoid logging complex dicts directly to summary
                    if isinstance(value, (list, tuple)) and any(isinstance(i, dict) for i in value):
                        wandb.summary[f"config_training_{param}"] = str(value)
                    else:
                        try:
                            wandb.summary[f"config_training_{param}"] = value
                        except Exception:
                             wandb.summary[f"config_training_{param}"] = str(value)
            # Log specific nested scheduler/early_stopping params if they exist
            if 'scheduler' in config['training'] and isinstance(config['training']['scheduler'], dict):
                for sch_param, sch_value in config['training']['scheduler'].get('params', {}).items():
                    wandb.summary[f"config_scheduler_{sch_param}"] = sch_value
            if 'early_stopping' in config['training'] and isinstance(config['training']['early_stopping'], dict):
                 for es_param, es_value in config['training']['early_stopping'].items():
                    wandb.summary[f"config_earlystop_{es_param}"] = es_value
        
        # Automatic evaluation and logging to the same W&B run
        if len(base_dataset) > 0:
            print("\n=== AUTOMATIC EVALUATION ===")
            try:
                # Get the test split (same way as training)
                total_size = len(base_dataset)
                test_split_ratio = config['data']['test_split']
                val_split_ratio = config['data']['validation_split']
                test_size = int(test_split_ratio * total_size)
                val_size = int(val_split_ratio * total_size)
                train_size = total_size - val_size - test_size
                
                # Recreate the same split
                generator = torch.Generator().manual_seed(config['framework']['seed'])
                _, _, test_subset = random_split(base_dataset, [train_size, val_size, test_size], generator=generator)
                
                # Use the same normalisation stats as training
                cpu_norm_stats = ( (train_mean_inputs.cpu(), train_std_inputs.cpu()), \
                                   (train_mean_targets.cpu(), train_std_targets.cpu()) )
                normalised_test_dataset = NormalisingDatasetWrapper(test_subset, cpu_norm_stats)
                
                test_loader = DataLoader(
                    normalised_test_dataset,
                    batch_size=config['training']['batch_size'],
                    shuffle=False,
                    num_workers=config['data']['num_workers'],
                    pin_memory=config['data']['pin_memory']
                )
                
                print(f"Evaluating on {len(test_subset)} test samples...")
                
                # Evaluation metrics calculation
                model.eval()
                all_preds_denorm_list, all_targets_orig_list = [], []
                avg_test_loss = 0.0
                
                # For denormalisation
                epsilon_denorm = 1e-8
                mean_targets_denorm_device = train_mean_targets.view(1, train_mean_targets.shape[-1], 1, 1).to(device)
                std_targets_denorm_device = train_std_targets.view(1, train_std_targets.shape[-1], 1, 1).to(device)
                
                with torch.no_grad():
                    for inputs_norm, targets_norm in test_loader:
                        inputs_norm_dev = inputs_norm.to(device).permute(0, 3, 1, 2)
                        targets_norm_dev = targets_norm.to(device).permute(0, 3, 1, 2)
                        
                        outputs_norm_dev = model(inputs_norm_dev)
                        loss = criterion(outputs_norm_dev, targets_norm_dev)
                        avg_test_loss += loss.item()
                        
                        # Denormalise predictions
                        preds_denorm_batch = (outputs_norm_dev * (std_targets_denorm_device + epsilon_denorm)) + mean_targets_denorm_device
                        preds_denorm_cpu_hwc = preds_denorm_batch.permute(0, 2, 3, 1).cpu().numpy()
                        
                        # Get original targets (denormalised)
                        targets_denorm_batch = (targets_norm_dev * (std_targets_denorm_device + epsilon_denorm)) + mean_targets_denorm_device
                        targets_denorm_cpu_hwc = targets_denorm_batch.permute(0, 2, 3, 1).cpu().numpy()
                        
                        all_preds_denorm_list.append(preds_denorm_cpu_hwc)
                        all_targets_orig_list.append(targets_denorm_cpu_hwc)

                if all_preds_denorm_list:
                    avg_test_loss /= len(test_loader)
                    final_preds_denorm = np.concatenate(all_preds_denorm_list, axis=0)
                    final_targets_orig = np.concatenate(all_targets_orig_list, axis=0)

                    # Calculate evaluation metrics
                    from standard_training_2.evaluate import nmse, psnr, ssim
                    
                    eval_nmse = float(nmse(final_preds_denorm, final_targets_orig))
                    eval_psnr = float(psnr(final_preds_denorm, final_targets_orig))
                    eval_ssim = float(ssim(final_preds_denorm, final_targets_orig))
                    eval_mse_original_scale = float(np.mean((final_preds_denorm - final_targets_orig)**2))

                    eval_metrics = {
                        'eval_nmse': eval_nmse,
                        'eval_psnr': eval_psnr,
                        'eval_ssim': eval_ssim,
                        'eval_mse_original_scale': eval_mse_original_scale,
                        'eval_avg_loss_normalised': avg_test_loss
                    }
                    
                    print(f"Evaluation Results:")
                    for metric_name, metric_value in eval_metrics.items():
                        print(f"  {metric_name}: {metric_value}")
                    
                    # Log evaluation metrics to W&B summary (CRITICAL for Optuna)
                    wandb.summary.update(eval_metrics)
                    wandb.log(eval_metrics)  # Also log as regular metrics
                    
                    print(f" Logged evaluation metrics to W&B summary: {list(eval_metrics.keys())}")
                    
                    # Give W&B time to sync before Optuna tries to fetch
                    time.sleep(2)
                    print(f" W&B run ready for Optuna: {wandb.run.id}")
                    
                    # Generate evaluation plots if configured
                    plot_examples = config.get('evaluation', {}).get('plot_examples', 0)
                    optuna_plots = config.get('evaluation', {}).get('optuna_wandb_num_plots', 0)
                    num_plots_to_generate = max(plot_examples, optuna_plots)
                    
                    if num_plots_to_generate > 0 and len(final_preds_denorm) > 0:
                        print(f"Generating {num_plots_to_generate} evaluation plots...")
                        
                        # Select random samples for plotting (with fixed seed for reproducibility)
                        plot_rng = np.random.default_rng(42)
                        num_test_samples = len(final_preds_denorm)
                        num_plots_actual = min(num_plots_to_generate, num_test_samples)
                        plot_indices = plot_rng.choice(num_test_samples, size=num_plots_actual, replace=False)
                        
                        # Get base save path from config or use checkpoint directory
                        base_plot_path = config.get('evaluation', {}).get('save_plot_path', str(checkpoint_dir / 'evaluation_sample.png'))
                        base_plot_path = Path(base_plot_path)
                        
                        # Make sure the plot is saved in the checkpoint directory for this trial
                        plot_save_path = checkpoint_dir / base_plot_path.name
                        
                        # Get interpolated data for enhanced plotting (if available)
                        interpolated_data = None
                        try:
                            # Extract interpolated data for the test subset indices
                            test_subset_indices = [test_subset.indices[i] for i in range(len(test_subset))]
                            interpolated_data = base_dataset.inputs[test_subset_indices]
                            print(f"Interpolated data available for enhanced plotting: {interpolated_data.shape}")
                        except Exception as e:
                            print(f"Warning: Could not access interpolated data for enhanced plotting: {e}")
                            print("Proceeding with model prediction plots only.")
                        
                        # Generate evaluation plots
                        try:
                            plot_evaluation_samples(
                                predictions_denorm=final_preds_denorm,
                                targets_original=final_targets_orig, 
                                sample_indices=plot_indices,
                                base_save_path=plot_save_path,
                                display_plots=False,  # Don't display plots in training script
                                interpolated_data=interpolated_data
                            )
                            print(f" Generated {num_plots_actual} evaluation plots in {checkpoint_dir}")
                        except Exception as e:
                            print(f"Warning: Failed to generate evaluation plots: {e}")
                            import traceback
                            traceback.print_exc()
                    
                else:
                    print("Warning: No evaluation data processed")
                    
            except Exception as e:
                print(f"Warning: Automatic evaluation failed: {e}")
                import traceback
                traceback.print_exc()
        
        wandb.finish()
        print("W&B run finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standard Training 2.0 Script")
    parser.add_argument('--config_path', type=str, default=None,
                        help='Path to the YAML configuration file. Defaults to unet.yaml in script directory.')
    parser.add_argument('--wandb_project', type=str, default=None,
                        help='W&B project name. Required for W&B logging if wandb_run_id is set.')
    parser.add_argument('--wandb_entity', type=str, default=None,
                        help='W&B entity (username or team). Optional.')
    parser.add_argument('--wandb_run_id', type=str, default=None,
                        help='W&B run ID for this training run. If provided, W&B logging is enabled.')
    parser.add_argument('--wandb_log_freq', type=int, default=1,
                        help='Log metrics to W&B every N epochs. Default is 1 (every epoch). Set to 0 to disable epoch-wise logging (only final).')
    parser.add_argument('--all_data', action='store_true',
                        help='Load and combine all preprocessed datasets instead of using single dataset from config.')
    
    cli_args = parser.parse_args()

    # Make sure to handle Path objects for os.path functions or use Path methods
    main(cli_args)
