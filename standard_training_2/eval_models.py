"""
Script to evaluate various models on the test dataset.

Usage examples:
1. Evaluate SRCNN model:
    python eval_models.py --weights_path path/to/weights.pth --model_type srcnn

2. Evaluate combined SRCNN-DnCNN model:
    python eval_models.py --weights_path path/to/weights.pth --model_type combined --order dncnn_first

3. Evaluate basic autoencoder:
    python eval_models.py --weights_path path/to/weights.pth --model_type autoencoder

4. Evaluate residual autoencoder:
    python eval_models.py --weights_path path/to/weights.pth --model_type residual_autoencoder

5. Evaluate combined AE-SRCNN model:
    python eval_models.py --weights_path path/to/weights.pth --model_type ae_srcnn --autoencoder_type residual

6. Evaluate standalone UNet model:
    python eval_models.py --weights_path path/to/unet_weights.pth --model_type unet --device cuda

7. Evaluate combined UNet+SRCNN model:
    python eval_models.py --weights_path path/to/unet_srcnn_weights.pth --model_type unet_srcnn --device cuda

8. Calculate average inference time (evaluates all test samples):
    python eval_models.py --weights_path path/to/weights.pth --model_type srcnn --inference_time

9. Specify device (cpu or cuda):
    python eval_models.py --weights_path path/to/weights.pth --model_type srcnn --device cuda

Note: The script automatically extracts configuration and normalization statistics from the weights file.
"""

# script which loads the models and evaluates them on the test set

import torch
from torch.utils.data import random_split, Dataset
import numpy as np
import yaml
import time
import argparse
from pathlib import Path
import tqdm
from scipy.signal import convolve2d

# Import the dataset and plotting utilities
import sys
sys.path.append('standard_training_2')
from standard_training_2.dataset import StandardDataset
from standard_training_2.tests.plotting_utils import plot_evaluation_samples
from standard_training_2.models.srcnn import SRCNN
from standard_training_2.models.srcnn_dncnn import CombinedModel_SRCNNDnCNN
from standard_training_2.models.autoencoder import DenoisingAutoencoder
from standard_training_2.models.residual_autoencoder import DenoisingResAutoencoder
from standard_training_2.models.ae_srcnn import CombinedModel_AESRCNN
from standard_training_2.models.unet import UNetModel
from standard_training_2.models.unet_srcnn import UNetCombinedModel

class NormalizingDatasetWrapper(Dataset):
    """
    A Dataset wrapper that applies Z-score normalization using provided statistics.
    Assumes input x and y are tensors from the underlying dataset.
    """
    def __init__(self, subset, norm_stats_tuple):
        self.subset = subset
        # norm_stats_tuple = ((mean_inputs, std_inputs), (mean_targets, std_targets))
        # All stats should be torch tensors on CPU, shaped for broadcasting (e.g., (1,1,C))
        self.mean_inputs, self.std_inputs = norm_stats_tuple[0]
        self.mean_targets, self.std_targets = norm_stats_tuple[1]
        self.epsilon = 1e-8 # To prevent division by zero

    def __getitem__(self, idx):
        x, y = self.subset[idx] # x, y are already tensors from StandardDataset
        
        # Normalize x
        x_normalized = (x - self.mean_inputs) / (self.std_inputs + self.epsilon)
        # Normalize y (original y is also normalized to be used as target for loss)
        y_normalized = (y - self.mean_targets) / (self.std_targets + self.epsilon)
        
        # Return normalized input, normalized target, and original target for metrics
        return x_normalized, y_normalized, y # y is the original unnormalized target from subset[idx]

    def __len__(self):
        return len(self.subset)

def load_config_from_yaml(config_path):
    """Load configuration file safely from YAML."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_model_weights(model, weights_path, device):
    """Load model weights from .pth file."""
    print(f"Loading model weights from: {weights_path}")
    
    # Load the checkpoint
    checkpoint = torch.load(weights_path, map_location=device, weights_only=True)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    
    # Load the state dict
    model.load_state_dict(state_dict)
    model.eval()
    print("Model weights loaded successfully.")
    return model

def determine_model_type(weights_path, device):
    """Determine the model type from the weights file."""
    try:
        checkpoint = torch.load(weights_path, map_location=device)
        if isinstance(checkpoint, dict):
            if 'config' in checkpoint:
                config = checkpoint['config']
                if 'model' in config:
                    return config['model'].get('type', 'srcnn')  # Default to srcnn if not specified
        return 'srcnn'  # Default to srcnn if no config found
    except Exception as e:
        print(f"Warning: Could not determine model type from weights: {e}")
        return 'srcnn'  # Default to srcnn if error occurs

def evaluate_single_sample(model, interpolated_input, ground_truth_target, device, norm_stats_targets):
    """Evaluate model on a single sample."""
    # Convert to torch tensors and add batch dimension
    if isinstance(interpolated_input, torch.Tensor):
        input_tensor = interpolated_input.unsqueeze(0).to(device)  # (1, H, W, C) -> (1, C, H, W)
    else:
        input_tensor = torch.from_numpy(interpolated_input).unsqueeze(0).to(device)
    
    # Ensure correct channel ordering (C, H, W)
    if input_tensor.dim() == 4 and input_tensor.shape[-1] == 2:
        input_tensor = input_tensor.permute(0, 3, 1, 2)  # (1, H, W, C) -> (1, C, H, W)
    
    # Accurate timing for GPU (CUDA is async)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    inference_start_time = time.time()
    
    with torch.no_grad():
        prediction = model(input_tensor)
        
        # Handle combined model output if needed
        if isinstance(model, CombinedModel_SRCNNDnCNN):
            prediction = prediction
    
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    inference_time = time.time() - inference_start_time
    
    # De-normalize predictions
    mean_targets, std_targets = norm_stats_targets
    mean_targets = mean_targets.to(device)
    std_targets = std_targets.to(device)
    epsilon = 1e-8
    
    # Reshape stats for broadcasting
    if mean_targets.ndim == 3 and mean_targets.shape[0] == 1 and mean_targets.shape[1] == 1:
        broadcast_mean_targets = mean_targets.permute(2,0,1).unsqueeze(0)
        broadcast_std_targets = std_targets.permute(2,0,1).unsqueeze(0)
    else:
        broadcast_mean_targets = mean_targets
        broadcast_std_targets = std_targets
    
    # De-normalize
    prediction_denorm = (prediction * (broadcast_std_targets + epsilon)) + broadcast_mean_targets
    
    # Convert back to numpy and correct format
    prediction_np = prediction_denorm.squeeze(0).permute(1, 2, 0).cpu().numpy()  # (C, H, W) -> (H, W, C)
    
    return prediction_np, inference_time

# --- Metrics functions copied from evaluate.py for consistency ---
def nmse(predictions_denorm, targets_original):
    if isinstance(predictions_denorm, torch.Tensor):
        predictions_denorm = predictions_denorm.cpu().numpy()
    if isinstance(targets_original, torch.Tensor):
        targets_original = targets_original.cpu().numpy()
    mse = np.mean((predictions_denorm - targets_original) ** 2)
    power = np.mean(targets_original ** 2)
    if power == 0:
        return float('inf') if mse > 0 else 0.0
    return mse / power

def psnr(predictions_denorm, targets_original, max_val=None):
    if isinstance(predictions_denorm, torch.Tensor):
        predictions_denorm = predictions_denorm.cpu().numpy()
    if isinstance(targets_original, torch.Tensor):
        targets_original = targets_original.cpu().numpy()
    if max_val is None:
        max_val = np.max(targets_original)
        if max_val == np.min(targets_original):
            max_val = 1.0 if max_val == 0 else np.abs(max_val)
    mse = np.mean((predictions_denorm - targets_original) ** 2)
    if mse == 0:
        return float('inf')
    if max_val == 0:
        return -float('inf') if mse > 0 else float('inf')
    return 20 * np.log10(max_val / np.sqrt(mse))

def ssim(predictions_denorm, targets_original, window_size=11, sigma=1.5, k1=0.01, k2=0.03):
    if isinstance(predictions_denorm, torch.Tensor):
        predictions_denorm = predictions_denorm.detach().cpu().numpy()
    if isinstance(targets_original, torch.Tensor):
        targets_original = targets_original.detach().cpu().numpy()
    pred_mag = np.abs(predictions_denorm[..., 0] + 1j * predictions_denorm[..., 1])
    target_mag = np.abs(targets_original[..., 0] + 1j * targets_original[..., 1])
    N, H, W = pred_mag.shape
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
    pred_norm = normalize_per_sample(pred_mag)
    target_norm = normalize_per_sample(target_mag)
    half = window_size // 2
    coords = np.arange(window_size) - half
    g1d = np.exp(- (coords**2) / (2 * sigma**2))
    g1d /= g1d.sum()
    kernel = np.outer(g1d, g1d)
    data_range = 1.0
    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2
    ssim_vals = np.zeros(N, dtype=np.float64)
    for i in range(N):
        x = pred_norm[i]
        y = target_norm[i]
        mu_x = convolve2d(x, kernel, mode='same', boundary='symm')
        mu_y = convolve2d(y, kernel, mode='same', boundary='symm')
        x_sq = x * x
        y_sq = y * y
        xy   = x * y
        mu_x_sq = mu_x * mu_x
        mu_y_sq = mu_y * mu_y
        mu_xy   = mu_x * mu_y
        sigma_x_sq = convolve2d(x_sq,   kernel, mode='same', boundary='symm') - mu_x_sq
        sigma_y_sq = convolve2d(y_sq,   kernel, mode='same', boundary='symm') - mu_y_sq
        sigma_xy   = convolve2d(xy,     kernel, mode='same', boundary='symm') - mu_xy
        numerator   = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
        denominator = (mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2)
        ssim_map = np.where(denominator > 0, numerator / denominator, 0.0)
        ssim_vals[i] = ssim_map.mean()
    return float(ssim_vals.mean())

def main():
    parser = argparse.ArgumentParser(description="Evaluate various models on test dataset")
    parser.add_argument('--weights_path', type=str, required=True, 
                       help='Path to the .pth file containing model weights')
    parser.add_argument('--model_type', type=str, 
                       choices=['srcnn', 'combined', 'autoencoder', 'residual_autoencoder', 'ae_srcnn', 'unet', 'unet_srcnn'],
                       help='Type of model to evaluate')
    parser.add_argument('--order', type=str, choices=['dncnn_first', 'srcnn_first'],
                       default='dncnn_first',
                       help='Order of operations for combined SRCNN-DnCNN model (dncnn_first or srcnn_first)')
    parser.add_argument('--autoencoder_type', type=str, choices=['basic', 'residual'],
                       default='residual',
                       help='Type of autoencoder to use for ae_srcnn model')
    parser.add_argument('--inference_time', action='store_true',
                       help='Calculate average inference time on all test samples (disables plotting)')
    parser.add_argument('--metrics', action='store_true',
                       help='Calculate and print average evaluation metrics (NMSE, MSE, PSNR, SSIM) over all test samples (disables plotting)')
    parser.add_argument('--device', type=str, default=None, choices=['cpu', 'cuda'],
                       help='Device to use for inference (cpu or cuda). Default: cuda if available, else cpu.')
    
    args = parser.parse_args()
    
    # Select device
    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load checkpoint to get config and normalization stats
    try:
        checkpoint = torch.load(args.weights_path, map_location=device)
        if isinstance(checkpoint, dict) and 'config' in checkpoint:
            config = checkpoint['config']
            print("Successfully loaded configuration from checkpoint")
            
            # Extract data config
            data_config = config.get('data', {})
            
            # Get data path from config
            data_dir = data_config.get('data_dir')
            data_name = data_config.get('data_name')
            
            if not data_dir or not data_name:
                print("Error: Could not find data_dir or data_name in checkpoint config.")
                print("Available config keys:")
                print(f"  Top level: {list(config.keys())}")
                if 'data' in config:
                    print(f"  data section: {list(config['data'].keys())}")
                return
            
            # Create dataset config dictionary
            dataset_config = {
                'data_dir': data_dir,
                'data_name': data_name,
                'preprocessed_dir': data_config.get('preprocessed_dir', 'preprocessed_data'),
                'snr': data_config.get('snr', 20),
                'interpolation_method': data_config.get('interpolation_method', 'rbf'),
                'interpolation_kernel': data_config.get('interpolation_kernel', 'thin_plate_spline')
            }
            
            # Load normalization stats if available
            if 'data' in config and 'norm_stats' in config['data']:
                norm_stats_loaded_lists = config['data']['norm_stats']
                
                # Convert stats from lists to tensors
                mean_inputs_loaded = torch.tensor(norm_stats_loaded_lists['mean_inputs'], dtype=torch.float32).cpu()
                std_inputs_loaded = torch.tensor(norm_stats_loaded_lists['std_inputs'], dtype=torch.float32).cpu()
                mean_targets_loaded = torch.tensor(norm_stats_loaded_lists['mean_targets'], dtype=torch.float32).cpu()
                std_targets_loaded = torch.tensor(norm_stats_loaded_lists['std_targets'], dtype=torch.float32).cpu()
                
                # Ensure correct shape (1,1,C)
                if mean_inputs_loaded.ndim == 1:
                    num_channels = mean_inputs_loaded.shape[0]
                    mean_inputs_loaded = mean_inputs_loaded.view(1,1,num_channels)
                    std_inputs_loaded = std_inputs_loaded.view(1,1,num_channels)
                    mean_targets_loaded = mean_targets_loaded.view(1,1,num_channels)
                    std_targets_loaded = std_targets_loaded.view(1,1,num_channels)
                
                norm_stats_for_wrapper = (
                    (mean_inputs_loaded.cpu(), std_inputs_loaded.cpu()),
                    (mean_targets_loaded.cpu(), std_targets_loaded.cpu())
                )
                norm_stats_targets_for_eval = (mean_targets_loaded.cpu(), std_targets_loaded.cpu())
                print("Successfully loaded normalization statistics from checkpoint")
            else:
                print("Warning: No normalization statistics found in checkpoint config")
                norm_stats_for_wrapper = None
                norm_stats_targets_for_eval = None
        else:
            print("Error: No config found in checkpoint")
            return
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return
    
    # Initialize model based on type
    if args.model_type == 'srcnn':
        model = SRCNN().to(device)
        print(f"SRCNN model created with {model.count_parameters():,} parameters")
        try:
            model = load_model_weights(model, args.weights_path, device)
        except Exception as e:
            print(f"Error loading model weights: {e}")
            return
    elif args.model_type == 'combined':
        # For combined SRCNN-DnCNN model
        model = CombinedModel_SRCNNDnCNN(
            order=args.order,
        ).to(device)
        print(f"Combined SRCNN-DnCNN model created with {model.count_parameters():,} parameters")
        print(f"Submodule parameters: {model.count_submodule_parameters()}")
        print(f"Model order: {args.order}")
        
        # Load the combined weights directly
        try:
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint
                
            # Load the state dict
            model.load_state_dict(state_dict)
            model.eval()
            print("Combined model weights loaded successfully")
        except Exception as e:
            print(f"Error loading combined model weights: {e}")
            return
    elif args.model_type == 'autoencoder':
        # Basic autoencoder
        model = DenoisingAutoencoder(input_channels=2).to(device)
        print(f"Basic Autoencoder model created with {model.count_parameters():,} parameters")
        try:
            model = load_model_weights(model, args.weights_path, device)
        except Exception as e:
            print(f"Error loading model weights: {e}")
            return
    elif args.model_type == 'residual_autoencoder':
        # Residual autoencoder
        model = DenoisingResAutoencoder(input_channels=2).to(device)
        print(f"Residual Autoencoder model created with {model.count_parameters():,} parameters")
        try:
            model = load_model_weights(model, args.weights_path, device)
        except Exception as e:
            print(f"Error loading model weights: {e}")
            return
    elif args.model_type == 'ae_srcnn':
        # Combined AE-SRCNN model
        model = CombinedModel_AESRCNN(
            autoencoder_type=args.autoencoder_type,
            pretrained_autoencoder=args.weights_path
        ).to(device)
        print(f"Combined AE-SRCNN model created with {model.count_parameters():,} parameters")
        print(f"Submodule parameters: {model.count_submodule_parameters()}")
        print(f"Autoencoder type: {args.autoencoder_type}")
    elif args.model_type == 'unet':
        # Standalone UNet model
        model_params_from_config = config.get('model', {}).get('params', {})
        model = UNetModel(
            in_channels=model_params_from_config.get('in_channels', 2),
            base_features=model_params_from_config.get('base_features', 16),
            use_batch_norm=model_params_from_config.get('use_batch_norm', False),
            depth=model_params_from_config.get('depth', 2),
            activation=model_params_from_config.get('activation', 'relu'),
            leaky_slope=model_params_from_config.get('leaky_slope', 0.01),
            verbose=model_params_from_config.get('verbose', False)
        ).to(device)
        print(f"UNet model created with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} parameters")
        try:
            model = load_model_weights(model, args.weights_path, device)
        except Exception as e:
            print(f"Error loading model weights: {e}")
            return
    elif args.model_type == 'unet_srcnn':
        # Combined UNet+SRCNN model
        # config is already loaded from the checkpoint earlier in main()
        model_params_from_config = config.get('model', {}).get('params', {})

        # Prepare unet_args for UNetModel, using defaults from UNetModel if not in config
        unet_args_for_model = {
            'in_channels': model_params_from_config.get('in_channels', 2),
            'base_features': model_params_from_config.get('base_features', 16),
            'use_batch_norm': model_params_from_config.get('use_batch_norm', False),
            'depth': model_params_from_config.get('depth', 2),
            'activation': model_params_from_config.get('activation', 'relu'),
            'leaky_slope': model_params_from_config.get('leaky_slope', 0.01),
            'verbose': model_params_from_config.get('verbose', False)
        }

        # Prepare SRCNN args, using defaults from UNetCombinedModel constructor if not in config
        srcnn_channels_for_model = model_params_from_config.get('srcnn_channels', [64, 32])
        srcnn_kernels_for_model = model_params_from_config.get('srcnn_kernels', [9, 1, 5])

        model = UNetCombinedModel(
            unet_args=unet_args_for_model,
            srcnn_channels=srcnn_channels_for_model,
            srcnn_kernels=srcnn_kernels_for_model
        ).to(device)
        
        print(f"UNet+SRCNN model created with {model.count_parameters():,} parameters")
        print(f"Submodule parameters: {model.count_submodule_parameters()}")
        try:
            checkpoint = torch.load(args.weights_path, map_location=device)
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint
            model.load_state_dict(state_dict)
            model.eval()
            print("UNet+SRCNN model weights loaded successfully")
        except Exception as e:
            print(f"Error loading UNet+SRCNN model weights: {e}")
            return
    
    # Load dataset
    try:
        print("Loading dataset...")
        base_dataset = StandardDataset(config=dataset_config)
        
        print(f"Dataset loaded with {len(base_dataset)} samples")
        
        # Split dataset using the same logic as conventional methods
        total_size = len(base_dataset)
        test_split_ratio = data_config.get('test_split', 0.15)  # Default to 0.15
        val_split_ratio = data_config.get('validation_split', 0.15)  # Default to 0.15
        test_size = int(test_split_ratio * total_size)
        val_size = int(val_split_ratio * total_size)
        train_size = total_size - val_size - test_size
        
        print(f"Dataset split: Train={train_size}, Validation={val_size}, Test={test_size}")
        
        # Split dataset using the same logic as train.py
        framework_seed = config.get('framework', {}).get('seed', 42)  # Default seed
        train_subset, val_subset, test_subset = random_split(
            base_dataset, [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(framework_seed)
        )
        
        # Store test subset indices before wrapping
        test_subset_indices = test_subset.indices
        
        # Apply normalization if stats are available
        if norm_stats_for_wrapper is not None:
            test_subset = NormalizingDatasetWrapper(test_subset, norm_stats_for_wrapper)
            print("Applied normalization to test subset")
        
    except Exception as e:
        print(f"An error occurred loading data: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Process samples
    if args.inference_time:
        print("\nCalculating average inference time on all test samples...")
        inference_times = []  # Times in seconds
        
        for i in tqdm.tqdm(range(len(test_subset)), desc="Evaluating samples"):
            # Get data for this sample
            if isinstance(test_subset, NormalizingDatasetWrapper):
                interpolated_input, _, ground_truth_target = test_subset[i]
            else:
                interpolated_input, ground_truth_target = test_subset[i]
            
            # Evaluate the sample
            _, inference_time = evaluate_single_sample(
                model, interpolated_input, ground_truth_target, device, norm_stats_targets_for_eval
            )
            inference_times.append(inference_time)
        
        # Calculate and print statistics (convert to milliseconds)
        avg_time = np.mean(inference_times) * 1000  # Convert to ms
        std_time = np.std(inference_times) * 1000   # Convert to ms
        min_time = np.min(inference_times) * 1000   # Convert to ms
        max_time = np.max(inference_times) * 1000   # Convert to ms
        
        print(f"\nInference Time Statistics (over {len(inference_times)} samples):")
        print(f"Average: {avg_time:.4f} ms")
        print(f"Std Dev: {std_time:.4f} ms")
        print(f"Min: {min_time:.4f} ms")
        print(f"Max: {max_time:.4f} ms")
        
    elif args.metrics:
        print("\nCalculating evaluation metrics (NMSE, MSE, PSNR, SSIM) on all test samples...")
        predictions = []
        targets = []
        for i in tqdm.tqdm(range(len(test_subset)), desc="Evaluating samples"):
            # Get data for this sample
            if isinstance(test_subset, NormalizingDatasetWrapper):
                interpolated_input, _, ground_truth_target = test_subset[i]
            else:
                interpolated_input, ground_truth_target = test_subset[i]
            pred_sample, _ = evaluate_single_sample(
                model, interpolated_input, ground_truth_target, device, norm_stats_targets_for_eval
            )
            predictions.append(pred_sample)
            if isinstance(ground_truth_target, torch.Tensor):
                targets.append(ground_truth_target.numpy())
            else:
                targets.append(ground_truth_target)
        predictions = np.stack(predictions, axis=0)
        targets = np.stack(targets, axis=0)
        nmse_val = nmse(predictions, targets)
        mse_val = np.mean((predictions - targets) ** 2)
        psnr_val = psnr(predictions, targets)
        try:
            ssim_val = ssim(predictions, targets)
        except Exception:
            ssim_val = float('nan')
        print("\n=== Evaluation Metrics (averaged over all test samples) ===")
        print(f"NMSE: {nmse_val:.6f}")
        print(f"MSE: {mse_val:.15f}")
        print(f"PSNR: {psnr_val:.4f} dB")
        print(f"SSIM: {ssim_val:.6f}")
    else:
        # Generate plots for specific samples
        print(f"\nGenerating evaluation plots...")
        
        # Use hardcoded sample indices that match evaluate.py and conventional methods output
        hardcoded_samples = [267, 1963, 2321]
        
        # Verify all indices are within bounds
        valid_indices = []
        for idx in hardcoded_samples:
            if 0 <= idx < len(test_subset):
                valid_indices.append(idx)
            else:
                print(f"Warning: Hardcoded sample {idx} is out of bounds for test subset (size: {len(test_subset)})")
        
        plot_indices = valid_indices
        print(f"Using sample indices: {plot_indices}")
        if plot_indices:
            print(f"These correspond to test subset global indices: {[test_subset_indices[i] for i in plot_indices]}")
        
        # Create output directory in the same location as model weights
        weights_path = Path(args.weights_path)
        output_dir = weights_path.parent / "evaluation_plots"
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Plots will be saved to: {output_dir}")
        
        # Process and plot each selected sample
        for i, plot_idx in enumerate(plot_indices):
            print(f"\nProcessing plot {i+1}/{len(plot_indices)} for test sample {plot_idx}...")
            
            # Get data for this sample
            if isinstance(test_subset, NormalizingDatasetWrapper):
                interpolated_input, _, ground_truth_target = test_subset[plot_idx]
            else:
                interpolated_input, ground_truth_target = test_subset[plot_idx]
            
            # Evaluate the sample
            pred_sample, inference_time = evaluate_single_sample(
                model, interpolated_input, ground_truth_target, device, norm_stats_targets_for_eval
            )
            
            if pred_sample is not None:
                print(f"  Sample {plot_idx} inference time: {inference_time*1000:.2f} ms")  # Convert to ms
                
                # Format data for plotting (ensure correct shape and format)
                if isinstance(ground_truth_target, torch.Tensor):
                    ground_truth_target = ground_truth_target.numpy()
                if isinstance(interpolated_input, torch.Tensor):
                    interpolated_input = interpolated_input.numpy()
                
                # Add batch dimension for plotting function
                true_H_batch = ground_truth_target[np.newaxis, ...]  # (1, H, W, C)
                pred_H_batch = pred_sample[np.newaxis, ...]  # (1, H, W, C)
                interpolated_input_batch = interpolated_input[np.newaxis, ...]  # (1, H, W, C)
                
                # Create save path with global test index for consistency
                global_test_idx = test_subset_indices[plot_idx]
                model_type_str = args.model_type
                base_save_path = output_dir / f"{model_type_str}_evaluation_sample_{global_test_idx}.svg"
                
                # Use the same plotting function as the conventional methods
                plot_evaluation_samples(
                    predictions_denorm=pred_H_batch, 
                    targets_original=true_H_batch, 
                    sample_indices=[0],  # Since we have batch size 1
                    base_save_path=base_save_path,
                    display_plots=False,  # Only save, don't display
                    interpolated_data=interpolated_input_batch
                )
                print(f"  Plot saved to: {base_save_path} (global test index: {global_test_idx})")
            else:
                print(f"  Failed to process model for test sample {plot_idx}.")
    
    print(f"\n=== {args.model_type.upper()} Evaluation Complete ===")

if __name__ == "__main__":
    main()
