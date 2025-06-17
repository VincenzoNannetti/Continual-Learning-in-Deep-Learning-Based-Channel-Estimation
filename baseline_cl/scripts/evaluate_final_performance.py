"""
Evaluation Script for Final Performance of Baseline Continual Learning Methods

This script evaluates the final trained models (EWC, Experience Replay) on all domains
to provide comprehensive performance comparisons.
"""

import argparse
import os
import sys
import yaml
import torch
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Add the project root to the path so we can import from standard_training_2
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Import from existing infrastructure
from standard_training_2.dataset import StandardDataset, NormalisingDatasetWrapper
from standard_training_2.models.unet_srcnn import UNetCombinedModel
from torch.utils.data import DataLoader, random_split

# Import correct metrics (using standard_training_2 implementation)
# We'll define the correct metrics below

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Skipping W&B logging.")

# Add correct metric functions from standard_training_2
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
            max_val = 1.0 if max_val == 0 else np.abs(max_val) 

    mse = np.mean((predictions_denorm - targets_original) ** 2)
    if mse == 0:
        return float('inf') # Perfect reconstruction
    if max_val == 0:
        return -float('inf') if mse > 0 else float('inf')

    return 20 * np.log10(max_val / np.sqrt(mse))

def ssim(predictions_denorm, targets_original, window_size=11, sigma=1.5, k1=0.01, k2=0.03):
    """
    Efficient SSIM for complex channel estimation data, computed on magnitudes.
    """
    from scipy.signal import convolve2d
    
    # Convert torch.Tensor → numpy if needed
    if isinstance(predictions_denorm, torch.Tensor):
        predictions_denorm = predictions_denorm.detach().cpu().numpy()
    if isinstance(targets_original, torch.Tensor):
        targets_original = targets_original.detach().cpu().numpy()

    # Convert from (N, C, H, W) to (N, H, W, C) if needed
    if predictions_denorm.ndim == 4 and predictions_denorm.shape[1] == 2:
        predictions_denorm = predictions_denorm.transpose(0, 2, 3, 1)  # (N, H, W, 2)
        targets_original = targets_original.transpose(0, 2, 3, 1)  # (N, H, W, 2)

    # Compute complex magnitudes:
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

        # Compute local means via 2D convolution
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


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_trained_model(checkpoint_path: str, device: torch.device, config: dict):
    """
    Load trained model from checkpoint and get normalisation statistics from original pretrained model.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"Loading trained model from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Get normalisation stats from the original pretrained model (same as used during training)
    pretrained_path = config['model']['pretrained_path']
    if not os.path.exists(pretrained_path):
        raise FileNotFoundError(f"Original pretrained model not found: {pretrained_path}")
    
    print(f"Loading normalisation stats from original pretrained model: {pretrained_path}")
    pretrained_checkpoint = torch.load(pretrained_path, map_location=device, weights_only=False)
    
    # Extract normalisation statistics from pretrained model
    if 'config' in pretrained_checkpoint and 'data' in pretrained_checkpoint['config'] and 'norm_stats' in pretrained_checkpoint['config']['data']:
        norm_stats_lists = pretrained_checkpoint['config']['data']['norm_stats']
        
        # Convert lists to tensors and ensure correct shape for (C, H, W) data
        norm_stats = {
            'mean_inputs': torch.tensor(norm_stats_lists['mean_inputs'], dtype=torch.float32).squeeze().view(-1, 1, 1),
            'std_inputs': torch.tensor(norm_stats_lists['std_inputs'], dtype=torch.float32).squeeze().view(-1, 1, 1),
            'mean_targets': torch.tensor(norm_stats_lists['mean_targets'], dtype=torch.float32).squeeze().view(-1, 1, 1),
            'std_targets': torch.tensor(norm_stats_lists['std_targets'], dtype=torch.float32).squeeze().view(-1, 1, 1)
        }
        print("Normalisation stats loaded from pretrained model and reshaped for (C, H, W) data")
    else:
        raise ValueError("Normalisation statistics not found in pretrained model checkpoint")
    
    # Extract model architecture parameters from pretrained model (same as the trained model should have)
    if 'config' not in pretrained_checkpoint:
        raise ValueError("Pretrained checkpoint does not contain config information")
    
    pretrained_config = pretrained_checkpoint['config']
    if 'model' not in pretrained_config or 'params' not in pretrained_config['model']:
        raise ValueError("Pretrained checkpoint config does not contain model parameters")
    
    # Use the model parameters from the pretrained checkpoint (these match the saved weights)
    model_params = pretrained_config['model']['params']
    
    # Create model
    model = UNetCombinedModel(
        unet_args={
            'in_channels': model_params['in_channels'],
            'base_features': model_params['base_features'],
            'use_batch_norm': model_params['use_batch_norm'],
            'depth': model_params['depth'],
            'activation': model_params['activation'],
            'leaky_slope': model_params['leaky_slope']
        },
        srcnn_channels=model_params['srcnn_channels'],
        srcnn_kernels=model_params['srcnn_kernels']
    )
    
    # Load trained weights
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded successfully")
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    print(f"\nNormalisation stats shapes:")
    for key, value in norm_stats.items():
        print(f"   {key}: {value.shape}")
    
    return model, norm_stats


def create_dataloader(domain_name: str, config: dict, norm_stats: dict, 
                     split: str = 'test') -> DataLoader:
    """
    Create a dataloader for evaluation on a specific domain.
    """
    # Determine SNR based on domain name
    if 'high_snr' in domain_name:
        snr = config['data']['domain_snr_mapping']['high_snr']
    elif 'med_snr' in domain_name:
        snr = config['data']['domain_snr_mapping']['med_snr']
    elif 'low_snr' in domain_name:
        snr = config['data']['domain_snr_mapping']['low_snr']
    else:
        raise ValueError(f"Could not determine SNR for domain: {domain_name}")
    
    # Create dataset
    dataset = StandardDataset(
        data_dir=config['data']['data_dir'],
        data_name=domain_name,
        snr=snr,
        interpolation_kernel='thin_plate_spline',
        preprocessed_dir=config['data']['preprocessed_dir'],
        all_data=False
    )
    
    # Split into train/val/test using same approach as training
    val_split = config['data']['validation_split']
    test_split = config['data']['test_split']
    
    total_size = len(dataset)
    val_size = int(total_size * val_split)
    test_size = int(total_size * test_split)
    train_size = total_size - val_size - test_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)  # Same seed for reproducibility
    )
    
    # Select split
    if split == 'train':
        target_dataset = train_dataset
    elif split == 'val':
        target_dataset = val_dataset
    else:
        target_dataset = test_dataset
    
    # Apply normalisation
    normalised_dataset = NormalisingDatasetWrapper(
        target_dataset, 
        (
            (norm_stats['mean_inputs'].clone().detach(), norm_stats['std_inputs'].clone().detach()),
            (norm_stats['mean_targets'].clone().detach(), norm_stats['std_targets'].clone().detach())
        )
    )
    
    # Create dataloader
    dataloader = DataLoader(
        normalised_dataset,
        batch_size=32,  # Use smaller batch size for evaluation
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    return dataloader


def evaluate_domain(model: torch.nn.Module, dataloader: DataLoader, 
                   domain_name: str, device: torch.device) -> dict:
    """
    Evaluate model on a single domain.
    """
    model.eval()
    
    all_nmse = []
    all_psnr = []
    all_ssim = []
    
    print(f"   Evaluating {domain_name}...")
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            
            # Calculate metrics for each sample in batch using correct implementation
            for i in range(outputs.shape[0]):
                pred = outputs[i:i+1]  # Keep as tensor with batch dimension
                true = targets[i:i+1]  # Keep as tensor with batch dimension
                
                # Calculate metrics using our corrected functions
                nmse_val = nmse(pred, true)
                psnr_val = psnr(pred, true)
                ssim_val = ssim(pred, true)
                
                all_nmse.append(nmse_val)
                all_psnr.append(psnr_val)
                all_ssim.append(ssim_val)
    
    # Calculate mean metrics
    metrics = {
        'nmse': np.mean(all_nmse),
        'psnr': np.mean(all_psnr),
        'ssim': np.mean(all_ssim),
        'nmse_std': np.std(all_nmse),
        'psnr_std': np.std(all_psnr),
        'ssim_std': np.std(all_ssim),
        'num_samples': len(all_nmse)
    }
    
    print(f"     NMSE: {metrics['nmse']:.8f} ± {metrics['nmse_std']:.8f}")
    print(f"     PSNR: {metrics['psnr']:.2f} ± {metrics['psnr_std']:.2f}")
    print(f"     SSIM: {metrics['ssim']:.4f} ± {metrics['ssim_std']:.4f}")
    
    return metrics


def evaluate_all_domains(model: torch.nn.Module, config: dict, norm_stats: dict, 
                        device: torch.device) -> dict:
    """
    Evaluate model on all domains in the sequence.
    """
    all_results = {}
    
    print(f"\nEvaluating on all {len(config['data']['sequence'])} domains...")
    
    for domain_name in config['data']['sequence']:
        try:
            dataloader = create_dataloader(domain_name, config, norm_stats, split='test')
            metrics = evaluate_domain(model, dataloader, domain_name, device)
            all_results[domain_name] = metrics
        except Exception as e:
            print(f"Error evaluating {domain_name}: {e}")
            continue
    
    # Calculate aggregate statistics
    if all_results:
        ssim_values = [r['ssim'] for r in all_results.values()]
        nmse_values = [r['nmse'] for r in all_results.values()]
        psnr_values = [r['psnr'] for r in all_results.values()]
        
        aggregate = {
            'mean_ssim': float(np.mean(ssim_values)),
            'mean_nmse': float(np.mean(nmse_values)),
            'mean_psnr': float(np.mean(psnr_values)),
            'std_ssim': float(np.std(ssim_values)),
            'std_nmse': float(np.std(nmse_values)),
            'std_psnr': float(np.std(psnr_values))
        }
        
        print(f"\nAggregate Results:")
        print(f"   SSIM: {aggregate['mean_ssim']:.4f} ± {aggregate['std_ssim']:.4f}")
        print(f"   NMSE: {aggregate['mean_nmse']:.8f} ± {aggregate['std_nmse']:.8f}")
        print(f"   PSNR: {aggregate['mean_psnr']:.2f} ± {aggregate['std_psnr']:.2f}")
        
        all_results['aggregate'] = aggregate
    
    return all_results


def save_results(results: dict, method_name: str, results_dir: str):
    """Save evaluation results to JSON and CSV files."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Convert numpy types to Python native types for JSON serialization
    def convert_numpy_types(obj):
        if isinstance(obj, dict):
            return {k: convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    # Convert results to JSON-serializable format
    json_results = convert_numpy_types(results)
    
    # Save detailed JSON results
    json_file = os.path.join(results_dir, f"{method_name}_final_evaluation_{timestamp}.json")
    with open(json_file, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    # Save CSV summary for plotting
    csv_data = []
    for domain, metrics in results.items():
        if domain != 'aggregate':
            csv_data.append({
                'Method': method_name,
                'Domain': domain,
                'SSIM': metrics['ssim'],
                'SSIM_std': metrics['ssim_std'],
                'NMSE': metrics['nmse'],
                'NMSE_std': metrics['nmse_std'],
                'PSNR': metrics['psnr'],
                'PSNR_std': metrics['psnr_std'],
                'Num_Samples': metrics['num_samples']
            })
    
    csv_df = pd.DataFrame(csv_data)
    csv_file = os.path.join(results_dir, f"{method_name}_final_evaluation_{timestamp}.csv")
    csv_df.to_csv(csv_file, index=False)
    
    print(f"Results saved:")
    print(f"   JSON: {json_file}")
    print(f"   CSV: {csv_file}")
    
    return json_file, csv_file


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate Final Performance of Baseline Methods")
    parser.add_argument('--method', type=str, required=True, choices=['ewc', 'experience_replay'],
                       help="Method to evaluate")
    parser.add_argument('--config', type=str, required=True,
                       help="Path to configuration file")
    parser.add_argument('--checkpoint', type=str, required=True,
                       help="Path to trained model checkpoint")
    parser.add_argument('--output_dir', type=str, default='baseline_cl/results/final_evaluation',
                       help="Output directory for results")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load configuration
    config = load_config(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Final Performance Evaluation: {args.method.upper()}")
    print(f"   Device: {device}")
    print(f"   Checkpoint: {args.checkpoint}")
    print(f"   Domains: {len(config['data']['sequence'])}")
    
    # Load trained model
    model, norm_stats = load_trained_model(args.checkpoint, device, config)
    
    # Evaluate on all domains
    results = evaluate_all_domains(model, config, norm_stats, device)
    
    # Save results
    json_file, csv_file = save_results(results, args.method, args.output_dir)
    
    print(f"\nEvaluation completed!")
    print(f"   Method: {args.method}")
    print(f"   Domains evaluated: {len(results) - 1}")  # -1 for aggregate
    print(f"   Results saved to: {args.output_dir}")


if __name__ == '__main__':
    main() 