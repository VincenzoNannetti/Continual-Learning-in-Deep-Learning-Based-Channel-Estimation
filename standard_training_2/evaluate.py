#!/usr/bin/env python3
"""
Evaluation script for Standard Training 2.0.
Loads a trained model and evaluates its performance on the test set.
Applies the same normalization used during training for fair comparison.
Supports both UNet and UNet+SRCNN models.
"""

import torch
import torch.nn as nn
import numpy as np
import yaml
from pathlib import Path
from torch.utils.data import DataLoader, random_split
from scipy.signal import convolve2d

from standard_training_2.dataset import StandardDataset, NormalisingDatasetWrapper
from standard_training_2.models.unet import UNetModel
from standard_training_2.models.unet_srcnn import UNetCombinedModel
from standard_training_2.models.srcnn import SRCNN
from standard_training_2.models.dncnn import DnCNN
from standard_training_2.models.autoencoder import DenoisingAutoencoder
from standard_training_2.models.residual_autoencoder import DenoisingResAutoencoder
from standard_training_2.models.srcnn_dncnn import CombinedModel_SRCNNDnCNN
from standard_training_2.models.ae_srcnn import CombinedModel_AESRCNN
from standard_training_2.plotting_utils import plot_evaluation_samples

def load_config_from_yaml(config_path):
    """Load configuration file safely from YAML."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_model_from_config(config):
    """
    Load model based on configuration.
    Supports all model types: 'unet', 'srcnn', 'dncnn', 'unet_srcnn', 'combined_srcnn_dncnn', 
    'combined_ae_srcnn', 'denoising_autoencoder', and 'denoising_res_autoencoder'.
    """
    model_name = config['model']['name'].lower()
    model_params = config['model']['params']
    
    if model_name == 'unet':
        model = UNetModel(
            in_channels=model_params['in_channels'],
            base_features=model_params['base_features']
        )
        print(f"[Sucess] Created UNet model with {sum(p.numel() for p in model.parameters()):,} parameters")
        
    elif model_name == 'unet_srcnn':
        # Extract UNet arguments
        unet_args = {
            'in_channels': model_params['in_channels'],
            'base_features': model_params['base_features']
        }
        
        # Extract combined model arguments (these shouldn't affect evaluation loading)
        pretrained_unet = None  # Don't load pretrained during evaluation
        pretrained_srcnn = None  # Don't load pretrained during evaluation
        
        model = UNetCombinedModel(
            pretrained_unet=pretrained_unet,
            pretrained_srcnn=pretrained_srcnn,
            unet_args=unet_args
        )
        
        # Count parameters for the combined model
        param_counts = model.count_submodule_parameters()
        
        print(f"✅ Created UNet+SRCNN combined model:")
        print(f"   - UNet: {param_counts['unet']:,} parameters")
        print(f"   - SRCNN: {param_counts['srcnn']:,} parameters")
        print(f"   - Total: {param_counts['total']:,} parameters")
        
    elif model_name == 'srcnn':
        # SRCNN standalone model
        model = SRCNN()
        print(f"✅ Created SRCNN model with {sum(p.numel() for p in model.parameters()):,} parameters")
        
    elif model_name == 'dncnn':
        # DnCNN standalone model
        num_channels = model_params.get('num_channels', 2)
        model = DnCNN(num_channels=num_channels)
        print(f"✅ Created DnCNN model with {sum(p.numel() for p in model.parameters()):,} parameters")
        
    elif model_name == 'combined_srcnn_dncnn':
        # Combined SRCNN+DnCNN model (order configurable)
        order = model_params.get('order', 'srcnn_first')
        dncnn_args = model_params.get('dncnn_args', {'num_channels': 2})
        srcnn_args = model_params.get('srcnn_args', {})
        
        model = CombinedModel_SRCNNDnCNN(
            order=order,
            pretrained_dncnn_path=None,  # Don't load pretrained during evaluation
            pretrained_srcnn_path=None,  # Don't load pretrained during evaluation
            dncnn_args=dncnn_args,
            srcnn_args=srcnn_args
        )
        
        # Count parameters for the combined model
        param_counts = model.count_submodule_parameters()
        
        print(f"✅ Created {order.upper()} combined model:")
        print(f"   - DnCNN: {param_counts['dncnn']:,} parameters")
        print(f"   - SRCNN: {param_counts['srcnn']:,} parameters")
        print(f"   - Total: {param_counts['total']:,} parameters")
        
    elif model_name == 'combined_ae_srcnn':
        # Combined Autoencoder+SRCNN model
        autoencoder_type = model_params.get('autoencoder_type', 'residual')
        
        model = CombinedModel_AESRCNN(
            autoencoder_type=autoencoder_type,
            pretrained_autoencoder=None,  # Don't load pretrained during evaluation
            pretrained_srcnn=None  # Don't load pretrained during evaluation
        )
        
        # Count parameters for the combined model
        param_counts = model.count_submodule_parameters()
        
        print(f"✅ Created {autoencoder_type.title()} Autoencoder+SRCNN combined model:")
        print(f"   - Autoencoder: {param_counts['autoencoder']:,} parameters")
        print(f"   - SRCNN: {param_counts['srcnn']:,} parameters")
        print(f"   - Total: {param_counts['total']:,} parameters")
        
    elif model_name == 'denoising_autoencoder':
        # Basic denoising autoencoder
        input_channels = model_params.get('input_channels', 2)
        model = DenoisingAutoencoder(input_channels=input_channels)
        print(f"✅ Created Basic Denoising Autoencoder with {sum(p.numel() for p in model.parameters()):,} parameters")
            
    elif model_name == 'denoising_res_autoencoder':
        # Residual denoising autoencoder
        input_channels = model_params.get('input_channels', 2)
        model = DenoisingResAutoencoder(input_channels=input_channels)
        print(f"✅ Created Residual Denoising Autoencoder with {sum(p.numel() for p in model.parameters()):,} parameters")
        
    else:
        raise ValueError(f"Unsupported model name: {model_name}. Supported: 'unet', 'srcnn', 'dncnn', 'unet_srcnn', 'combined_srcnn_dncnn', 'combined_ae_srcnn', 'denoising_autoencoder', 'denoising_res_autoencoder'")
    
    return model


def nmse(predictions_denorm, targets_original):
    """Calculate Normalised Mean Square Error on original scale data."""
    # Ensure inputs are numpy arrays
    if isinstance(predictions_denorm, torch.Tensor):
        predictions_denorm = predictions_denorm.cpu().numpy()
    if isinstance(targets_original, torch.Tensor):
        targets_original = targets_original.cpu().numpy()

    mse = np.mean((predictions_denorm - targets_original) ** 2)
    power = np.mean(targets_original ** 2)
    if power == 0: # Avoid division by zero if target signal is all zeros
        return float('inf') if mse > 0 else 0.0
    return mse / power

def psnr(predictions_denorm, targets_original, max_val=None):
    """Calculate Peak Signal-to-Noise Ratio on original scale data."""
    # Ensure inputs are numpy arrays
    if isinstance(predictions_denorm, torch.Tensor):
        predictions_denorm = predictions_denorm.cpu().numpy()
    if isinstance(targets_original, torch.Tensor):
        targets_original = targets_original.cpu().numpy()

    if max_val is None:
        max_val = np.max(targets_original)
        if max_val == np.min(targets_original): # Handle case where target is constant
            # If target is constant and non-zero, max_val is fine.
            # If target is constant zero, PSNR is problematic. 
            # Let's ensure max_val is at least 1.0 if everything is zero for a sensible reference for PSNR.
            max_val = 1.0 if max_val == 0 else np.abs(max_val) 

    mse = np.mean((predictions_denorm - targets_original) ** 2)
    if mse == 0:
        return float('inf') # Perfect reconstruction
    # if max_val == 0 and mse > 0: # This case should be handled by max_val being at least 1.0 if data is all zero.
    #     return -float('inf')
    if max_val == 0: # Should ideally not happen if data was all zero due to above adjustment. Safety. Max_val can be also min_val here
        return -float('inf') if mse > 0 else float('inf') # if mse >0 and max_val =0, then psnr is -inf. if mse=0 and max_val=0, it is +inf

    return 20 * np.log10(max_val / np.sqrt(mse))

def ssim(predictions_denorm, targets_original, window_size=11, sigma=1.5, k1=0.01, k2=0.03):
    """
    Efficient SSIM for complex channel estimation data, computed on magnitudes.

    For complex data given as real/imag pairs (N, H, W, 2), we:
      1. Convert to complex, take magnitudes.
      2. Normalize each sample independently to [0, 1].
      3. Compute local means, variances, and covariance via 2D convolution.
      4. Compute SSIM map and average over each sample.
      5. Return the mean SSIM over the batch.

    Args:
        predictions_denorm: np.ndarray or torch.Tensor of shape (N, H, W, 2)
        targets_original:   np.ndarray or torch.Tensor of shape (N, H, W, 2)
        window_size:        size of the Gaussian window (default: 11)
        sigma:              standard deviation of Gaussian (default: 1.5)
        k1, k2:             SSIM constants (default: 0.01, 0.03)

    Returns:
        float: average SSIM over all N samples
    """
    # Convert torch.Tensor → numpy if needed
    if isinstance(predictions_denorm, torch.Tensor):
        predictions_denorm = predictions_denorm.detach().cpu().numpy()
    if isinstance(targets_original, torch.Tensor):
        targets_original = targets_original.detach().cpu().numpy()

    # Compute complex magnitudes:
    # predictions_denorm[..., 0] = real part, [..., 1] = imag part
    pred_mag = np.abs(predictions_denorm[..., 0] + 1j * predictions_denorm[..., 1])  # shape (N, H, W)
    target_mag = np.abs(targets_original[..., 0] + 1j * targets_original[..., 1])

    N, H, W = pred_mag.shape

    # Per-sample normalization to [0, 1]:
    def normalize_per_sample(arr):
        out = np.zeros_like(arr)
        for i in range(arr.shape[0]):
            mn = arr[i].min()
            mx = arr[i].max()
            if mx > mn:
                out[i] = (arr[i] - mn) / (mx - mn)
            else:
                # If all values are equal, map to zeros
                out[i] = np.zeros_like(arr[i])
        return out

    pred_norm = normalize_per_sample(pred_mag)    # (N, H, W)
    target_norm = normalize_per_sample(target_mag)

    # Build 2D Gaussian kernel
    half = window_size // 2
    coords = np.arange(window_size) - half
    g1d = np.exp(- (coords**2) / (2 * sigma**2))
    g1d /= g1d.sum()
    kernel = np.outer(g1d, g1d)  # shape (window_size, window_size)

    # Constants for SSIM
    data_range = 1.0  # after normalization to [0,1]
    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2

    ssim_vals = np.zeros(N, dtype=np.float64)

    for i in range(N):
        x = pred_norm[i]    # shape (H, W)
        y = target_norm[i]  # shape (H, W)

        # Compute local means via 2D convolution (mode='same', boundary='symm' ≈ reflect padding)
        mu_x = convolve2d(x, kernel, mode='same', boundary='symm')
        mu_y = convolve2d(y, kernel, mode='same', boundary='symm')

        # Compute local squares and cross-products
        x_sq = x * x
        y_sq = y * y
        xy   = x * y

        mu_x_sq = mu_x * mu_x
        mu_y_sq = mu_y * mu_y
        mu_xy   = mu_x * mu_y

        # Compute variances and covariance via convolution minus squared means
        sigma_x_sq = convolve2d(x_sq,   kernel, mode='same', boundary='symm') - mu_x_sq
        sigma_y_sq = convolve2d(y_sq,   kernel, mode='same', boundary='symm') - mu_y_sq
        sigma_xy   = convolve2d(xy,     kernel, mode='same', boundary='symm') - mu_xy

        # Compute SSIM map
        numerator   = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
        denominator = (mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2)

        # Avoid division by zero
        ssim_map = np.where(denominator > 0, numerator / denominator, 0.0)

        # Average SSIM over all pixels
        ssim_vals[i] = ssim_map.mean()

    return float(ssim_vals.mean())

def evaluate_model(model, test_loader, device, norm_stats_targets):
    """Evaluate model on test dataset. Predictions are de-normalized."""
    model.eval()
    
    all_predictions_denormalized = []
    all_targets_original = [] # Store original (unnormalized) targets
    total_loss_normalized = 0.0 # Loss is calculated on normalized data
    criterion = nn.MSELoss() # Operates on normalized model output and normalized target
    
    mean_targets, std_targets = norm_stats_targets # Should be (1,1,C) tensors on CPU
    mean_targets = mean_targets.to(device) # Move to device for de-normalization
    std_targets = std_targets.to(device)
    epsilon = 1e-8

    with torch.no_grad():
        for inputs_norm, targets_norm, targets_orig_batch in test_loader:
            inputs_norm, targets_norm = inputs_norm.to(device), targets_norm.to(device)
            # targets_orig_batch is already a tensor from StandardDataset, keep on CPU or move as needed
            
            # Reshape for model: (batch, channels, height, width)
            inputs_norm_permuted = inputs_norm.permute(0, 3, 1, 2)
            targets_norm_permuted = targets_norm.permute(0, 3, 1, 2) # For loss calculation
            
            outputs_norm_permuted = model(inputs_norm_permuted) # Model output is normalized, shape (N, C, H, W)
            loss = criterion(outputs_norm_permuted, targets_norm_permuted)
            total_loss_normalized += loss.item()
            
            # De-normalize predictions
            # outputs_norm_permuted is (N, C, H, W). mean_targets, std_targets need to match for broadcasting.
            # If mean_targets is (1,1,C), permute to (C,1,1) and unsqueeze for batch: (1,C,1,1)
            if mean_targets.ndim == 3 and mean_targets.shape[0] == 1 and mean_targets.shape[1] == 1: # (1,1,C)
                # Permute to (C,1,1) then unsqueeze for batch dimension (1,C,1,1)
                broadcast_mean_targets = mean_targets.permute(2,0,1).unsqueeze(0) 
                broadcast_std_targets = std_targets.permute(2,0,1).unsqueeze(0)
            else: # Assuming it's already (C,) or somehow directly compatible
                # This case might need adjustment based on actual saved shape if not (1,1,C)
                broadcast_mean_targets = mean_targets 
                broadcast_std_targets = std_targets

            predictions_denorm_permuted = (outputs_norm_permuted * (broadcast_std_targets + epsilon)) + broadcast_mean_targets
            
            # Convert de-normalized predictions and original targets to numpy for metrics calculation
            # Reshape back to (N, H, W, C)
            predictions_denorm_np = predictions_denorm_permuted.permute(0, 2, 3, 1).cpu().numpy()
            # targets_orig_batch is (N, H, W, C) from dataloader (originally from StandardDataset)
            targets_original_np = targets_orig_batch.cpu().numpy() 
            
            all_predictions_denormalized.append(predictions_denorm_np)
            all_targets_original.append(targets_original_np)
    
    # Concatenate all batches
    all_predictions_denormalized = np.concatenate(all_predictions_denormalized, axis=0)
    all_targets_original = np.concatenate(all_targets_original, axis=0)
    
    # Calculate metrics using de-normalized predictions and original targets
    mse_val = np.mean((all_predictions_denormalized - all_targets_original) ** 2)
    nmse_score = nmse(all_predictions_denormalized, all_targets_original)
    # For PSNR, max_val should be from original targets
    psnr_max_val = np.max(all_targets_original) 
    psnr_score = psnr(all_predictions_denormalized, all_targets_original, max_val=psnr_max_val)
    # Calculate SSIM
    ssim_score = ssim(all_predictions_denormalized, all_targets_original)
    
    avg_loss_normalized = total_loss_normalized / len(test_loader)
    
    return {
        'mse': mse_val, # This MSE is on the original data scale
        'nmse': nmse_score,
        'psnr': psnr_score,
        'ssim': ssim_score,  # New SSIM metric
        'avg_loss_normalized': avg_loss_normalized # This loss is on the normalized data scale
    }, all_predictions_denormalized, all_targets_original

def main():
    """Main evaluation function."""
    print("=== Standard Training 2.0 - Evaluation Script ===")
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Standard Training 2.0 Evaluation Script")
    parser.add_argument('--all_data', action='store_true',
                        help='Load and combine all preprocessed datasets instead of using single dataset from config.')
    args = parser.parse_args()
    
    # Determine base_config_path relative to this script file for portability
    # This assumes unet.yaml is in the same directory as this script (or train.py)
    script_dir = Path(__file__).resolve().parent
    base_config_path = script_dir / "config/unet.yaml" # Expect unet.yaml in the same folder as evaluate.py
    
    # Load base configuration from YAML (for dataset paths, model structure etc.)
    # This config might be overridden by the one from the checkpoint for some params, but data loading params should be consistent.
    try:
        base_config = load_config_from_yaml(base_config_path)
        print(f"Base configuration loaded from: {base_config_path}")
    except Exception as e:
        print(f"Error loading base YAML configuration from {base_config_path}: {e}")
        print("Ensure unet.yaml is present in the same directory as evaluate.py or update path.")
        return

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- Load Model and Config from Checkpoint ---
    # Use checkpoint_dir from the base_config, as it's where train.py saved it.
    checkpoint_dir = Path(base_config['logging']['checkpoint_dir'])
    checkpoint_path = checkpoint_dir / 'best_model.pth' # Or 'final_model.pth'

    if not checkpoint_path.exists():
        print(f"No trained model found at: {checkpoint_path}")
        print("Please run train.py first to train and save a model.")
        return
    
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # The config from the checkpoint has all settings used during training, including norm_stats
    train_config = checkpoint['config'] 
    model_state_dict = checkpoint['model_state_dict']
    
    print("\n--- Loaded Model Configuration from Checkpoint ---")
    print(f"  Experiment Name: {train_config.get('experiment_name', 'N/A')}")
    print(f"  Model Name: {train_config.get('model', {}).get('name', 'N/A')}")
    print("  Model Params:")
    for param, value in train_config.get('model', {}).get('params', {}).items():
        print(f"    {param}: {value}")
    print("  Training Params (from checkpoint relevant to model structure/data):")
    print(f"    Original Learning Rate: {train_config.get('training', {}).get('learning_rate', 'N/A')}")
    print(f"    Original Batch Size: {train_config.get('training', {}).get('batch_size', 'N/A')}")
    print("  Data Params (from checkpoint):")
    print(f"    Dataset Name: {train_config.get('data', {}).get('data_name', 'N/A')}")
    print(f"    SNR: {train_config.get('data', {}).get('snr', 'N/A')}")
    print("------------------------------------------------")
    
    print(f"Model loaded from epoch {checkpoint.get('epoch', 'N/A')}")
    train_loss = checkpoint.get("train_loss", None)
    if isinstance(train_loss, (float, int)):
        print(f"Original training loss: {train_loss:.6f}")
    else:
        print("Original training loss: N/A")
    val_loss = checkpoint.get("val_loss", None)
    if isinstance(val_loss, (float, int)):
        print(f"Original validation loss: {val_loss:.6f}")
    else:
        print("Original validation loss: N/A")

    # --- Retrieve Normalization Stats ---
    if 'norm_stats' not in train_config['data']:
        print("Error: Normalization statistics not found in the loaded checkpoint's config.")
        print("Ensure the model was trained with the updated train.py that saves these stats.")
        return
        
    norm_stats_loaded_lists = train_config['data']['norm_stats']
    
    # Convert stats from lists back to tensors, ensure they are on CPU for the wrapper
    # And ensure they have the correct shape (1, 1, C) for broadcasting
    try:
        mean_inputs_loaded = torch.tensor(norm_stats_loaded_lists['mean_inputs'], dtype=torch.float32).cpu()
        std_inputs_loaded = torch.tensor(norm_stats_loaded_lists['std_inputs'], dtype=torch.float32).cpu()
        mean_targets_loaded = torch.tensor(norm_stats_loaded_lists['mean_targets'], dtype=torch.float32).cpu()
        std_targets_loaded = torch.tensor(norm_stats_loaded_lists['std_targets'], dtype=torch.float32).cpu()

        # Reshape if they were saved as (C,) or ensure they are (1,1,C) from training script
        # The training script saves them as (1,1,C) after .tolist(), so direct tensor conversion should be fine.
        # Example check, assuming last dim is channel: 
        if not (mean_inputs_loaded.ndim == 3 and mean_inputs_loaded.shape[0]==1 and mean_inputs_loaded.shape[1]==1):
            print(f"Warning: mean_inputs_loaded shape is {mean_inputs_loaded.shape}. Expected (1,1,C).")
            # Attempt to reshape if it was saved as (C,)
            if mean_inputs_loaded.ndim == 1:
                num_channels = mean_inputs_loaded.shape[0]
                mean_inputs_loaded = mean_inputs_loaded.view(1,1,num_channels)
                std_inputs_loaded = std_inputs_loaded.view(1,1,num_channels)
                mean_targets_loaded = mean_targets_loaded.view(1,1,num_channels)
                std_targets_loaded = std_targets_loaded.view(1,1,num_channels)
                print("Reshaped stats to (1,1,C).")

    except KeyError as e:
        print(f"Error: Missing key in norm_stats: {e}. Check how stats were saved.")
        return
    except Exception as e:
        print(f"Error processing normalization stats: {e}")
        return

    # For NormalisingDatasetWrapper, stats should be on CPU.
    # For de-normalization in evaluate_model, target stats will be moved to device.
    norm_stats_for_wrapper = (
        (mean_inputs_loaded.cpu(), std_inputs_loaded.cpu()),
        (mean_targets_loaded.cpu(), std_targets_loaded.cpu())
    )
    # For de-normalization inside evaluate_model, just pass the target stats part
    norm_stats_targets_for_eval = (mean_targets_loaded.cpu(), std_targets_loaded.cpu())

    # --- Load Dataset and Create Test Loader (with normalization) ---
    print("Loading base dataset for evaluation...")
    try:
        # Use data parameters from the *training config* to ensure consistency
        base_dataset = StandardDataset(train_config['data'], all_data=args.all_data)
        if args.all_data:
            print(f"Combined dataset loaded: {len(base_dataset)} samples from all preprocessed datasets")
        else:
            print(f"Base dataset loaded: {len(base_dataset)} samples")
    except Exception as e:
        print(f"Error loading base dataset: {e}")
        return
    
    # Recreate the same train/val/test split as done during training to get the correct test_subset
    total_size = len(base_dataset)
    test_split_ratio = train_config['data']['test_split']
    val_split_ratio = train_config['data']['validation_split']
    test_size = int(test_split_ratio * total_size)
    val_size = int(val_split_ratio * total_size)
    train_size = total_size - val_size - test_size

    # Use the seed from the training config for consistent splitting
    split_seed = train_config['framework']['seed']
    _, _, test_subset = random_split(
        base_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(split_seed)
    )
    print(f"Test subset size: {len(test_subset)}")

    # Wrap the test_subset with normalization using training stats
    normalized_test_dataset = NormalisingDatasetWrapper(test_subset, norm_stats_for_wrapper)
    
    test_loader = DataLoader(
        normalized_test_dataset,
        batch_size=train_config['training']['batch_size'], # Use batch size from training config
        shuffle=False,
        num_workers=train_config['data']['num_workers'],
        pin_memory=train_config['data']['pin_memory']
    )
    print(f"Test batches: {len(test_loader)}")
    
    # --- Create and Load Model State ---
    model = load_model_from_config(train_config).to(device)
    model.load_state_dict(model_state_dict)
    
    print("\n--- Model Architecture to be Evaluated ---")
    print(model)
    print("-------------------------------------------")
    
    # --- Evaluate Model ---
    print("\nEvaluating model...")
    # Pass target norm stats for de-normalization of predictions
    metrics, predictions_denorm, targets_original = evaluate_model(model, test_loader, device, norm_stats_targets_for_eval)
    
    # --- Print and Save Results ---
    print("\n=== Evaluation Results (on original data scale) ===")
    print(f"MSE: {metrics['mse']:.6f}")
    print(f"NMSE: {metrics['nmse']:.6f}")
    print(f"PSNR: {metrics['psnr']:.2f} dB")
    print(f"SSIM: {metrics['ssim']:.6f}")
    print(f"Average Normalized Loss (from model): {metrics['avg_loss_normalized']:.6f}")
    
    if len(predictions_denorm) > 0:
        print("\nGenerating enhanced IEEE-style visualisations...")
        # Define a fixed seed for plot sample selection for reproducibility of plots across runs
        plot_sample_selection_seed = 123 
        rng = np.random.default_rng(plot_sample_selection_seed)
        
        num_test_samples = len(predictions_denorm)
        num_plots_to_generate = min(3, num_test_samples) # Plot up to 3 samples

        if num_test_samples > 0:
            # Ensure selected indices are unique and within bounds
            plot_indices = rng.choice(num_test_samples, size=num_plots_to_generate, replace=False)
            
            # Get interpolated data from the base dataset for enhanced plotting
            interpolated_data = None
            try:
                # Extract interpolated data for the test subset indices
                test_subset_indices = [test_subset.indices[i] for i in range(len(test_subset))]
                interpolated_data = base_dataset.inputs[test_subset_indices]  # This should be the interpolated data
                print(f"Interpolated data available for enhanced plotting: {interpolated_data.shape}")
            except Exception as e:
                print(f"Warning: Could not access interpolated data for enhanced plotting: {e}")
                print("Proceeding with model prediction plots only.")
            
            # Use base_config for general save path, train_config for specific dataset names if needed
            eval_plot_base_name = Path(base_config['evaluation'].get('save_plot_path', './plots/evaluation_sample.svg'))
            
            # Call enhanced plotting function with interpolated data
            plot_evaluation_samples(
                predictions_denorm, 
                targets_original, 
                sample_indices=plot_indices,
                base_save_path=eval_plot_base_name,
                display_plots=True, # For standalone evaluate.py, show the plots
                interpolated_data=interpolated_data  # Pass interpolated data for enhanced plots
            )
        else:
            print("No samples in test set to plot.")
    
    results_dir = Path(base_config['evaluation']['save_results_dir'])
    results_dir.mkdir(parents=True, exist_ok=True)
    
    results_file = results_dir / 'evaluation_metrics.txt'
    with open(results_file, 'w') as f:
        f.write("=== Standard Training 2.0 Evaluation Results ===\n")
        f.write(f"Checkpoint: {checkpoint_path}\n")
        f.write(f"Dataset: {train_config['data']['data_name']}\n")
        f.write(f"SNR: {train_config['data']['snr']} dB\n")
        f.write(f"Test samples: {len(test_subset)}\n")
        f.write("--- Metrics (on original data scale) ---\n")
        f.write(f"MSE: {metrics['mse']:.6f}\n")
        f.write(f"NMSE: {metrics['nmse']:.6f}\n")
        f.write(f"PSNR: {metrics['psnr']:.2f} dB\n")
        f.write(f"SSIM: {metrics['ssim']:.6f}\n")
        f.write(f"Average Normalized Loss (from model): {metrics['avg_loss_normalized']:.6f}\n")
    
    print(f"\nResults saved to: {results_file}")
    print("\n=== Evaluation Complete ===")

if __name__ == "__main__":
    main()
