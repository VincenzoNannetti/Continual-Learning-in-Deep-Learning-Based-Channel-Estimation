"""
Main training script for the LoRA-based continual learning refactoring.
"""
import argparse
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Skipping W&B logging.")

# Add shared folder to path for other utilities  
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import from our refactored source directory
from src.config import ExperimentConfig, load_config
from src.model import UNet_SRCNN_LoRA
from src.data import get_dataloaders, get_norm_stats_from_checkpoint
from src.utils import get_device, create_scheduler, set_seed
from src.ewc import FisherInformationManager, EWCLoss
from replay_buffer import calculate_model_difficulty_metrics, perform_difficulty_clustering, create_stratified_replay_buffer

def denormalise(tensor: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """Denormalise a tensor using mean and std."""
    return tensor * (std + 1e-8) + mean

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
    from scipy.signal import convolve2d
    
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

def calculate_metrics(outputs_tensor: torch.Tensor, targets_tensor: torch.Tensor) -> dict:
    """
    Calculate metrics using the standard_training_2 implementation adapted for continual learning.
    
    Args:
        outputs_tensor: Prediction tensor of shape (N, C, H, W) where C=2 (real, imag)
        targets_tensor: Ground truth tensor of shape (N, C, H, W) where C=2 (real, imag)
        
    Returns:
        Dictionary with NMSE, PSNR, and SSIM values
    """
    # Convert from (N, C, H, W) to (N, H, W, C) format expected by standard_training_2 metrics
    if outputs_tensor.dim() == 4 and outputs_tensor.shape[1] == 2:
        outputs_np = outputs_tensor.permute(0, 2, 3, 1).detach().cpu().numpy()  # (N, H, W, 2)
        targets_np = targets_tensor.permute(0, 2, 3, 1).detach().cpu().numpy()  # (N, H, W, 2)
    else:
        # Handle other formats if needed
        outputs_np = outputs_tensor.detach().cpu().numpy()
        targets_np = targets_tensor.detach().cpu().numpy()
    
    # Calculate metrics using the correct standard_training_2 implementation
    nmse_val = nmse(outputs_np, targets_np)
    psnr_val = psnr(outputs_np, targets_np)
    ssim_val = ssim(outputs_np, targets_np)
    
    # Debug: Check for negative or unexpected NMSE
    if nmse_val < 0 or nmse_val > 10:
        print(f"️ DEBUG: Unusual NMSE value detected: {nmse_val}")
        print(f"   MSE: {np.mean((outputs_np - targets_np) ** 2)}")
        print(f"   Signal Power: {np.mean(targets_np ** 2)}")
        print(f"   Output stats: min={np.min(outputs_np):.6f}, max={np.max(outputs_np):.6f}, mean={np.mean(outputs_np):.6f}")
        print(f"   Target stats: min={np.min(targets_np):.6f}, max={np.max(targets_np):.6f}, mean={np.mean(targets_np):.6f}")
    
    return {
        'nmse': float(nmse_val),
        'psnr': float(psnr_val),
        'ssim': float(ssim_val)
    }



def evaluate_single_domain(model, task_id: str, config: ExperimentConfig, device: torch.device) -> dict:
    """Evaluate the model on a single domain."""
    print(f"  Evaluating domain {task_id}...")
    
    # Set model to correct domain
    model.set_active_task(task_id)
    
    # Get validation data loader for the domain
    _, val_loader = get_dataloaders(
        task_id=task_id,
        config=config.data,
        batch_size=config.training.batch_size,
        norm_stats=config.data.norm_stats
    )
    
    all_outputs = []
    all_targets = []
    
    model.eval()
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            
            # Denormalise for metric calculation
            mean_i = torch.tensor(config.data.norm_stats.mean_inputs, device=device).view(1, -1, 1, 1)
            std_i = torch.tensor(config.data.norm_stats.std_inputs, device=device).view(1, -1, 1, 1)
            mean_t = torch.tensor(config.data.norm_stats.mean_targets, device=device).view(1, -1, 1, 1)
            std_t = torch.tensor(config.data.norm_stats.std_targets, device=device).view(1, -1, 1, 1)
            
            outputs_denorm = denormalise(outputs, mean_t, std_t)
            targets_denorm = denormalise(targets, mean_t, std_t)
            
            all_outputs.append(outputs_denorm)
            all_targets.append(targets_denorm)
    
    # Concatenate all batches as tensors
    all_outputs_tensor = torch.cat(all_outputs, dim=0)
    all_targets_tensor = torch.cat(all_targets, dim=0)
    
    # Calculate metrics using shared implementation
    metrics = calculate_metrics(all_outputs_tensor, all_targets_tensor)
    
    print(f"    Domain {task_id}: SSIM={metrics['ssim']:.4f}, NMSE={metrics['nmse']:.8f}, PSNR={metrics['psnr']:.2f}")
    
    return metrics

def evaluate_continual_learning_step(model, config: ExperimentConfig, device: torch.device, 
                                   completed_tasks: list, current_task_step: int) -> dict:
    """
    Evaluate model on all completed tasks for continual learning metrics.
    
    Args:
        model: The LoRA model
        config: Experiment configuration
        device: Device to run on
        completed_tasks: List of task IDs that have been trained
        current_task_step: Current step in the task sequence (0-indexed)
    
    Returns:
        Dictionary with performance on each completed task
    """
    print(f"\n CL Evaluation Step {current_task_step + 1}: Testing on {len(completed_tasks)} completed tasks...")
    
    step_results = {}
    
    # Debug: Track the active task before evaluation
    current_active_task = model.active_task if hasattr(model, 'active_task') else None
    print(f"    DEBUG: Current active task before evaluation: {current_active_task}")
    
    for task_id in completed_tasks:
        try:
            print(f"    Evaluating task {task_id}...")
            task_metrics = evaluate_single_domain(model, task_id, config, device)
            step_results[task_id] = task_metrics
            
            # Debug: Print metrics immediately after evaluation
            print(f"    Task {task_id} metrics: NMSE={task_metrics['nmse']:.8f}, SSIM={task_metrics['ssim']:.4f}, PSNR={task_metrics['psnr']:.2f}")
            
        except Exception as e:
            print(f"   Error evaluating task {task_id}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Debug: Print summary of step results
    print(f"\n    Step {current_task_step + 1} Summary:")
    print(f"   Tasks evaluated: {list(step_results.keys())}")
    for task_id, metrics in step_results.items():
        print(f"     Task {task_id}: NMSE={metrics['nmse']:.8f}")
    
    return step_results

def calculate_continual_learning_metrics(performance_matrix: dict, task_sequence: list) -> dict:
    """
    Calculate BWT, FWT, and other continual learning metrics.
    
    Args:
        performance_matrix: Dict with structure {step: {task_id: {metric: value}}}
        task_sequence: List of task IDs in training order
        
    Returns:
        Dictionary containing BWT, FWT, and other CL metrics
    """
    print("\n Calculating Continual Learning Metrics...")
    
    metrics_to_analyze = ['ssim', 'nmse', 'psnr']
    cl_metrics = {}
    
    for metric_name in metrics_to_analyze:
        print(f"\n   Analyzing {metric_name.upper()}...")
        
        # Extract performance matrix for this metric
        perf_matrix = {}
        for step, step_results in performance_matrix.items():
            perf_matrix[step] = {}
            for task_id, task_metrics in step_results.items():
                if metric_name in task_metrics:
                    perf_matrix[step][task_id] = task_metrics[metric_name]
        
        # Calculate metrics for this performance measure
        metric_results = _calculate_metric_specific_cl_stats(perf_matrix, task_sequence, metric_name)
        cl_metrics[metric_name] = metric_results
    
    # Calculate overall statistics
    cl_metrics['summary'] = _calculate_summary_statistics(cl_metrics, task_sequence)
    
    return cl_metrics

def verify_bwt_fwt_calculations(cl_metrics: dict, performance_matrix: dict, task_sequence: list) -> None:
    """
    Comprehensive verification of BWT and FWT calculations with detailed analysis.
    
    Args:
        cl_metrics: Calculated continual learning metrics
        performance_matrix: Raw performance matrix
        task_sequence: Task sequence
    """
    print("\n" + "="*80)
    print(" VERIFICATION OF BWT/FWT CALCULATIONS")
    print("="*80)
    
    # Focus on NMSE as primary metric
    if 'nmse' not in cl_metrics:
        print("️ NMSE metrics not found in cl_metrics")
        return
    
    nmse_metrics = cl_metrics['nmse']
    print("\n NMSE METRIC ANALYSIS (lower is better):")
    print(f"   Average BWT: {nmse_metrics['bwt']['average']:.6f}")
    print(f"   Average FWT: {nmse_metrics['fwt']['average']:.6f}")
    
    # Verify BWT calculation manually
    print("\n MANUAL BWT VERIFICATION:")
    print("   BWT = (1/(T-1)) * Σ(R_T,i - R_i,i) for i=1 to T-1")
    print("   where R_T,i = performance on task i after learning all T tasks")
    print("   and R_i,i = performance on task i right after learning it")
    
    num_tasks = len(task_sequence)
    if num_tasks > 1:
        final_step = num_tasks - 1
        manual_bwt_sum = 0.0
        bwt_count = 0
        
        print(f"\n   {'Task':<10} | {'R_i,i (Initial)':<15} | {'R_T,i (Final)':<15} | {'Difference':<15} | {'Status':<20}")
        print("   " + "-"*75)
        
        for i, task_id in enumerate(task_sequence[:-1]):
            task_id_str = str(task_id)
            
            # Get initial performance (after learning task)
            initial_perf = None
            if i in performance_matrix and task_id_str in performance_matrix[i]:
                if 'nmse' in performance_matrix[i][task_id_str]:
                    initial_perf = performance_matrix[i][task_id_str]['nmse']
            
            # Get final performance (after all tasks)
            final_perf = None
            if final_step in performance_matrix and task_id_str in performance_matrix[final_step]:
                if 'nmse' in performance_matrix[final_step][task_id_str]:
                    final_perf = performance_matrix[final_step][task_id_str]['nmse']
            
            if initial_perf is not None and final_perf is not None:
                diff = final_perf - initial_perf
                manual_bwt_sum += diff
                bwt_count += 1
                
                if diff > 0.001:  # Threshold for significant forgetting
                    status = "️ Forgetting"
                elif diff < -0.001:  # Threshold for improvement
                    status = " Improved"
                else:
                    status = " Stable"
                
                print(f"   Task {task_id:<6} | {initial_perf:<15.6f} | {final_perf:<15.6f} | {diff:<15.6f} | {status:<20}")
            else:
                print(f"   Task {task_id:<6} | {'Missing data':<15} | {'Missing data':<15} | {'-':<15} | {' Error':<20}")
        
        if bwt_count > 0:
            manual_avg_bwt = manual_bwt_sum / bwt_count
            print(f"\n   Manual Average BWT: {manual_avg_bwt:.6f}")
            print(f"   Calculated BWT: {nmse_metrics['bwt']['average']:.6f}")
            
            if abs(manual_avg_bwt - nmse_metrics['bwt']['average']) < 0.0001:
                print("    BWT calculation VERIFIED!")
            else:
                print("    BWT calculation MISMATCH!")
    
    # Analyze forgetting patterns
    print("\n FORGETTING PATTERN ANALYSIS:")
    if 'per_task' in nmse_metrics['bwt']:
        forgetting_tasks = []
        improving_tasks = []
        stable_tasks = []
        
        for task_id, bwt_val in nmse_metrics['bwt']['per_task'].items():
            if bwt_val > 0.001:
                forgetting_tasks.append((task_id, bwt_val))
            elif bwt_val < -0.001:
                improving_tasks.append((task_id, bwt_val))
            else:
                stable_tasks.append((task_id, bwt_val))
        
        if forgetting_tasks:
            print(f"    Tasks with forgetting ({len(forgetting_tasks)}):")
            for task_id, bwt in sorted(forgetting_tasks, key=lambda x: x[1], reverse=True):
                print(f"      Task {task_id}: +{bwt:.6f} NMSE (worse)")
        
        if improving_tasks:
            print(f"Tasks with improvement ({len(improving_tasks)}):")
            for task_id, bwt in sorted(improving_tasks, key=lambda x: x[1]):
                print(f"      Task {task_id}: {bwt:.6f} NMSE (better)")
        
        if stable_tasks:
            print(f"Stable tasks ({len(stable_tasks)}):")
            for task_id, bwt in stable_tasks:
                print(f"      Task {task_id}: {bwt:.6f} NMSE")
    
    # Expected behavior for LoRA
    print("\n EXPECTED BEHAVIOR FOR LoRA:")
    print("   • BWT should be close to 0 due to parameter isolation")
    print("   • Small positive BWT might indicate:")
    print("     - Shared backbone degradation")
    print("     - Adapter interference (shouldn't happen)")
    print("     - Numerical instability")
    print("   • Performance differences should mainly reflect domain difficulty")
    
    # Summary interpretation
    avg_bwt = nmse_metrics['bwt']['average']
    print(f"\n SUMMARY:")
    if abs(avg_bwt) < 0.001:
        print("    Excellent: Near-zero BWT indicates proper parameter isolation")
    elif avg_bwt > 0.01:
        print("   ️ Warning: Significant forgetting detected - check implementation")
    elif avg_bwt < -0.01:
        print("    Unexpected: Significant improvement - verify evaluation consistency")
    else:
        print("    Good: Minor BWT within expected range")
    
    print("="*80)

def _calculate_metric_specific_cl_stats(perf_matrix: dict, task_sequence: list, metric_name: str) -> dict:
    """Calculate BWT, FWT, and other stats for a specific metric."""
    
    num_tasks = len(task_sequence)
    results = {
        'performance_matrix': perf_matrix,
        'task_sequence': task_sequence
    }
    
    # Debug: Print performance matrix
    print(f"\n DEBUG: Performance Matrix for {metric_name.upper()}:")
    print(f"{'Step':<6} | " + " | ".join([f"Task {t:<4}" for t in task_sequence]))
    print("-" * (8 + 11 * len(task_sequence)))
    
    for step in range(num_tasks):
        if step in perf_matrix:
            row_values = []
            for task in task_sequence:
                task_str = str(task)
                if task_str in perf_matrix[step]:
                    val = perf_matrix[step][task_str]
                    row_values.append(f"{val:>8.4f}")
                else:
                    row_values.append("   -    ")
            print(f"Step {step:<2} | " + " | ".join(row_values))
    print()
    
    # 1. Backward Transfer (BWT)
    # BWT = (1/(T-1)) * Σ(R_T,i - R_i,i) for i=1 to T-1
    # where R_T,i is performance on task i after learning all T tasks
    # and R_i,i is performance on task i immediately after learning it
    bwt_values = []
    task_specific_bwt = {}
    
    if num_tasks > 1:
        final_step = num_tasks - 1  # Last step (0-indexed)
        
        print(f" DEBUG: BWT Calculation Details for {metric_name.upper()}:")
        print(f"{'Task':<6} | {'Initial (R_ii)':<14} | {'Final (R_Ti)':<14} | {'BWT (R_Ti-R_ii)':<16} | {'Interpretation':<20}")
        print("-" * 80)
        
        for i, task_id in enumerate(task_sequence[:-1]):  # Exclude last task
            task_id_str = str(task_id)
            
            # Performance immediately after learning task i (step i)
            if i in perf_matrix and task_id_str in perf_matrix[i]:
                r_ii = perf_matrix[i][task_id_str]
            else:
                continue
                
            # Performance on task i after learning all tasks (final step)
            if final_step in perf_matrix and task_id_str in perf_matrix[final_step]:
                r_ti = perf_matrix[final_step][task_id_str]
            else:
                continue
            
            # BWT for this task - STANDARD DEFINITION: final - initial
            # For NMSE (lower better): Positive BWT = worse performance = MORE forgetting
            # For SSIM/PSNR (higher better): Positive BWT = better performance = LESS forgetting
            task_bwt = r_ti - r_ii  # Standard: final - initial performance
            
            # Interpretation based on metric type
            if metric_name == 'nmse':
                if task_bwt > 0:
                    interpretation = "Forgetting ↓"
                elif task_bwt < 0:
                    interpretation = "Improvement ↑"
                else:
                    interpretation = "No change"
            else:  # SSIM/PSNR - higher is better
                if task_bwt > 0:
                    interpretation = "Improvement ↑"
                elif task_bwt < 0:
                    interpretation = "Forgetting ↓"
                else:
                    interpretation = "No change"
                
            bwt_values.append(task_bwt)
            task_specific_bwt[task_id_str] = task_bwt
            
            print(f"Task {task_id:<2} | {r_ii:<14.6f} | {r_ti:<14.6f} | {task_bwt:<16.6f} | {interpretation:<20}")
    
    avg_bwt = np.mean(bwt_values) if bwt_values else 0.0
    results['bwt'] = {
        'average': float(avg_bwt),
        'per_task': task_specific_bwt,
        'values': bwt_values
    }
    
    print(f"\n Average BWT for {metric_name.upper()}: {avg_bwt:.6f}")
    if metric_name == 'nmse':
        print(f"   Interpretation: {'Forgetting detected' if avg_bwt > 0 else 'No forgetting / Improvement' if avg_bwt <= 0 else 'No change'}")
    
    # 2. Forward Transfer (FWT) 
    # NOTE: Current implementation is non-standard
    # TODO: Implement proper FWT with true baseline comparison
    fwt_values = []
    task_specific_fwt = {}
    
    # For FWT, we need a baseline. We'll use the performance on the first task as reference
    # WARNING: This is not standard FWT - it's comparing to first task performance
    print(f"\n️  WARNING: FWT calculation is using first task as baseline, not true isolated learning baseline")
    
    if num_tasks > 1 and 0 in perf_matrix:
        first_task_perf = list(perf_matrix[0].values())[0] if perf_matrix[0] else None
        
        for i, task_id in enumerate(task_sequence[1:], 1):  # Start from second task
            task_id_str = str(task_id)
            
            if i in perf_matrix and task_id_str in perf_matrix[i]:
                current_task_perf = perf_matrix[i][task_id_str]
                
                if first_task_perf is not None:
                    # FWT: Performance with transfer - baseline performance
                    # For NMSE (lower better): Negative FWT = better than baseline = positive transfer
                    # For SSIM/PSNR (higher better): Positive FWT = better than baseline = positive transfer
                    task_fwt = current_task_perf - first_task_perf
                        
                    fwt_values.append(task_fwt)
                    task_specific_fwt[task_id_str] = task_fwt
                    
                    print(f"    Task {task_id}: Performance={current_task_perf:.4f}, FWT={task_fwt:.4f}")
    
    avg_fwt = np.mean(fwt_values) if fwt_values else 0.0
    results['fwt'] = {
        'average': float(avg_fwt),
        'per_task': task_specific_fwt,
        'values': fwt_values
    }
    
    # 3. Average Performance at end
    if num_tasks - 1 in perf_matrix:
        final_performances = list(perf_matrix[num_tasks - 1].values())
        avg_final_performance = np.mean(final_performances) if final_performances else 0.0
    else:
        avg_final_performance = 0.0
    
    results['average_final_performance'] = float(avg_final_performance)
    
    # 4. Performance matrix statistics
    all_performances = []
    for step_results in perf_matrix.values():
        all_performances.extend(step_results.values())
    
    results['statistics'] = {
        'mean': float(np.mean(all_performances)) if all_performances else 0.0,
        'std': float(np.std(all_performances)) if all_performances else 0.0,
        'min': float(np.min(all_performances)) if all_performances else 0.0,
        'max': float(np.max(all_performances)) if all_performances else 0.0
    }
    
    return results

def _calculate_summary_statistics(cl_metrics: dict, task_sequence: list) -> dict:
    """Calculate summary statistics across all metrics."""
    
    summary = {
        'num_tasks': len(task_sequence),
        'task_sequence': [str(t) for t in task_sequence]
    }
    
    # Average BWT and FWT across metrics
    bwt_values = []
    fwt_values = []
    final_perf_values = []
    
    for metric_name, metric_results in cl_metrics.items():
        if metric_name == 'summary':
            continue
            
        if 'bwt' in metric_results:
            bwt_values.append(metric_results['bwt']['average'])
        if 'fwt' in metric_results:
            fwt_values.append(metric_results['fwt']['average'])
        if 'average_final_performance' in metric_results:
            final_perf_values.append(metric_results['average_final_performance'])
    
    summary['overall_bwt'] = float(np.mean(bwt_values)) if bwt_values else 0.0
    summary['overall_fwt'] = float(np.mean(fwt_values)) if fwt_values else 0.0
    summary['overall_final_performance'] = float(np.mean(final_perf_values)) if final_perf_values else 0.0
    
    return summary

def evaluate_all_domains_direct(model, config: ExperimentConfig, device: torch.device, domain_ids: list) -> dict:
    """
    Evaluate the model on all specified domains directly.
    Returns the results dictionary with all metrics.
    """
    print(f"\n Running Multi-Domain Evaluation on {len(domain_ids)} domains...")
    
    all_domain_results = {}
    aggregate_metrics = {'ssim': [], 'nmse': [], 'psnr': []}
    
    for domain_id in domain_ids:
        if domain_id not in [str(d) for d in config.data.sequence]:
            print(f" Warning: Domain {domain_id} not in model's training sequence")
            continue
            
        try:
            domain_metrics = evaluate_single_domain(model, domain_id, config, device)
            all_domain_results[domain_id] = domain_metrics
            
            # Collect for aggregate statistics
            for metric_name, value in domain_metrics.items():
                if metric_name in aggregate_metrics:
                    aggregate_metrics[metric_name].append(value)
                
        except Exception as e:
            print(f"   Error evaluating domain {domain_id}: {e}")
            continue
    
    # Calculate and log aggregate statistics
    if all_domain_results:
        print(f"\n AGGREGATE RESULTS ACROSS {len(all_domain_results)} DOMAINS:")
        
        aggregate_stats = {}
        for metric_name, values in aggregate_metrics.items():
            if values:
                mean_val = np.mean(values)
                std_val = np.std(values)
                min_val = np.min(values)
                max_val = np.max(values)
                
                aggregate_stats[metric_name] = {
                    'mean': float(mean_val),
                    'std': float(std_val),
                    'min': float(min_val),
                    'max': float(max_val)
                }
                
                print(f"    {metric_name.upper()}: {mean_val:.4f} ± {std_val:.4f} (min: {min_val:.4f}, max: {max_val:.4f})")
        
        # Calculate domain consistency metrics
        if len(all_domain_results) > 1:
            ssim_values = [result['ssim'] for result in all_domain_results.values()]
            nmse_values = [result['nmse'] for result in all_domain_results.values()]
            
            # Coefficient of variation (std/mean) as consistency measure
            ssim_consistency = np.std(ssim_values) / np.mean(ssim_values) if np.mean(ssim_values) > 0 else 1.0
            nmse_consistency = np.std(nmse_values) / np.mean(nmse_values) if np.mean(nmse_values) > 0 else 1.0
            
            print(f"    CONSISTENCY (CoV): SSIM={ssim_consistency:.4f}, NMSE={nmse_consistency:.4f}")
            
            aggregate_stats['consistency'] = {
                'ssim': float(ssim_consistency),
                'nmse': float(nmse_consistency)
            }
    
    # Create results structure
    results_data = {
        'domain_results': all_domain_results,
        'aggregate_stats': aggregate_stats if 'aggregate_stats' in locals() else {},
        'num_domains_evaluated': len(all_domain_results),
        'domain_ids_evaluated': list(all_domain_results.keys())
    }
    
    print(f" Multi-domain evaluation complete! Results for {len(all_domain_results)} domains.")
    
    return results_data

def train_one_epoch(model: UNet_SRCNN_LoRA, loader: torch.utils.data.DataLoader, 
                    criterion: nn.Module, optimiser: optim.Optimizer, 
                    device: torch.device, use_amp: bool, scaler: GradScaler,
                    clip_norm: float = None, wandb_run=None, 
                    ewc_loss_fn=None, current_task_id: str = None) -> tuple:
    """
    Runs a single training epoch with optional EWC regularization.
    Returns tuple of (total_loss, ewc_loss) for logging.
    """
    model.train()
    total_main_loss = 0.0
    total_ewc_loss = 0.0
    total_combined_loss = 0.0
    max_grad_norm = 0.0

    for inputs, targets in tqdm(loader, desc="Training"):
        inputs, targets = inputs.to(device), targets.to(device)

        optimiser.zero_grad()

        with autocast(device_type='cuda', enabled=use_amp):
            outputs = model(inputs)
            main_loss = criterion(outputs, targets)
            
            # Add EWC loss following Algorithm 1
            ewc_loss = torch.tensor(0.0, device=device)
            if ewc_loss_fn is not None and current_task_id is not None:
                ewc_loss = ewc_loss_fn(model, current_task_id, exclude_current_task=True)
            
            # Combined loss: L_main + λ/2 * Σ F_j(θ_j - θ_j*)²
            combined_loss = main_loss + ewc_loss
        
        # Add NaN detection for early debugging
        if not torch.isfinite(combined_loss):
            raise RuntimeError(f"Non-finite combined loss detected. Main: {main_loss.item()}, EWC: {ewc_loss.item()}")
        
        if use_amp:
            scaler.scale(combined_loss).backward()
            scaler.unscale_(optimiser)
            
            # Apply gradient clipping if specified and monitor gradient norms
            if clip_norm is not None:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm).item()
                max_grad_norm = max(max_grad_norm, grad_norm)
                # Check for exploding gradients
                if grad_norm > clip_norm * 10:  # Alert if gradients are very large
                    print(f"Warning: Large gradient norm detected: {grad_norm:.2f}")
                    if wandb_run:
                        wandb_run.log({'gradient_explosion_warning': grad_norm})
            
            scaler.step(optimiser)
            scaler.update()
        else:
            combined_loss.backward()
            
            # Apply gradient clipping if specified and monitor gradient norms
            if clip_norm is not None:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm).item()
                max_grad_norm = max(max_grad_norm, grad_norm)
                # Check for exploding gradients
                if grad_norm > clip_norm * 10:  # Alert if gradients are very large
                    print(f"Warning: Large gradient norm detected: {grad_norm:.2f}")
                    if wandb_run:
                        wandb_run.log({'gradient_explosion_warning': grad_norm})
            
            optimiser.step()

        # Track losses separately for logging
        total_main_loss += main_loss.item()
        total_ewc_loss += ewc_loss.item()
        total_combined_loss += combined_loss.item()

    # Log max gradient norm for the epoch
    if wandb_run and max_grad_norm > 0:
        wandb_run.log({'epoch_max_gradient_norm': max_grad_norm})
    
    return (total_combined_loss / len(loader), total_main_loss / len(loader), total_ewc_loss / len(loader))

def validate_one_epoch(model: UNet_SRCNN_LoRA, loader: torch.utils.data.DataLoader, 
                       criterion: nn.Module, device: torch.device) -> float:
    """
    Runs a single validation epoch.
    """
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for inputs, targets in tqdm(loader, desc="Validating"):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()

    return total_loss / len(loader)

def populate_replay_buffer(model, train_loader, task_id, device, buffer_size=350):
    """
    Populate replay buffer for the current task using difficulty-based sampling.
    
    Args:
        model: Trained model for current task
        train_loader: Training data loader for current task
        task_id: Current task identifier
        device: Device to run on
        buffer_size: Size of replay buffer
        
    Returns:
        ReplayBuffer: Populated replay buffer
    """
    print(f"\n--- Populating Replay Buffer for Task {task_id} ---")
    
    # Collect all training data
    all_inputs = []
    all_targets = []
    
    model.eval()
    with torch.no_grad():
        for inputs, targets in tqdm(train_loader, desc="Collecting training data"):
            all_inputs.append(inputs.cpu())
            all_targets.append(targets.cpu())
    
    # Concatenate all data
    all_inputs = torch.cat(all_inputs, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    print(f"Collected {all_inputs.shape[0]} training samples for buffer population")
    
    # Calculate difficulty metrics using the trained model
    nmse_values, predictions = calculate_model_difficulty_metrics(
        model, all_inputs, all_targets, device, batch_size=32
    )
    
    # Perform clustering based on difficulty
    cluster_labels, kmeans_model = perform_difficulty_clustering(nmse_values, n_clusters=3)
    
    # Create stratified replay buffer (using inputs, not predictions)
    replay_buffer = create_stratified_replay_buffer(
        predictions, all_targets, cluster_labels, nmse_values, buffer_size, inputs=all_inputs
    )
    
    print(f"Created replay buffer with {len(replay_buffer)} samples")
    return replay_buffer

def save_continual_learning_csvs(cl_metrics: dict, performance_matrix: dict, 
                                best_task_val_losses: dict, config: ExperimentConfig,
                                final_evaluation_results: dict, total_training_time: float,
                                memory_footprint_mb: float, output_dir: str) -> None:
    """
    Save continual learning results in CSV formats compatible with plotting scripts.
    
    Args:
        cl_metrics: Continual learning metrics from calculate_continual_learning_metrics
        performance_matrix: Performance evolution matrix
        best_task_val_losses: Best validation losses per task
        config: Experiment configuration
        final_evaluation_results: Results from evaluate_all_domains_direct
        total_training_time: Total training time in seconds
        memory_footprint_mb: Memory footprint in MB
        output_dir: Output directory for CSV files
    """
    if cl_metrics is None:
        print("️ No continual learning metrics to save")
        return
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. EWC SUMMARY CSV (compatible with plot_ewc_results.py)
    summary_data = {
        'Method': ['LoRA_EWC'],
        'Backward_Transfer_BWT': [cl_metrics['summary']['overall_bwt']],
        'Forward_Transfer_FWT': [cl_metrics['summary']['overall_fwt']],
        'Final_Average_Performance_ACC': [cl_metrics['summary']['overall_final_performance']],
        'Learning_Accuracy': [cl_metrics['summary']['overall_final_performance']],  # Use same as final perf
        'Average_Forgetting': [abs(cl_metrics['summary']['overall_bwt']) if cl_metrics['summary']['overall_bwt'] > 0 else 0],
        'Forward_Plasticity': [cl_metrics['summary']['overall_fwt']],
        'Memory_Footprint_MB': [memory_footprint_mb],
        'Total_Training_Time_Seconds': [total_training_time],
        'Average_Training_Time_Seconds': [total_training_time / len(config.data.sequence)],
        'Num_Domains': [len(config.data.sequence)],
        'Timestamp': [timestamp]
    }
    
    # Add NMSE-specific metrics (primary focus)
    if 'nmse' in cl_metrics:
        summary_data.update({
            'NMSE_BWT': [cl_metrics['nmse']['bwt']['average']],
            'NMSE_FWT': [cl_metrics['nmse']['fwt']['average']],
            'NMSE_Final_Performance': [cl_metrics['nmse']['average_final_performance']]
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_file = os.path.join(output_dir, f"lora_ewc_summary_{timestamp}.csv")
    summary_df.to_csv(summary_file, index=False)
    print(f" Saved EWC summary CSV: {summary_file}")
    
         # 2. PERFORMANCE MATRIX CSV (compatible with plot_ewc_results.py)
    matrix_file = None
    if performance_matrix:
        matrix_data = []
        task_sequence = [str(i) for i in config.data.sequence]
        
        for step in range(len(task_sequence)):
            if step in performance_matrix:
                row = {'After_Task': f"T{step+1}_{task_sequence[step]}"}
                
                # Add performance on each task (using NMSE as primary metric)
                for task_idx, task_id in enumerate(task_sequence):
                    col_name = f"Performance_on_T{task_idx+1}_{task_id}"
                    if task_id in performance_matrix[step]:
                        # Use NMSE as the primary performance metric
                        nmse_value = performance_matrix[step][task_id].get('nmse', None)
                        row[col_name] = nmse_value
                    else:
                        row[col_name] = None
                
                matrix_data.append(row)
        
        if matrix_data:
            matrix_df = pd.DataFrame(matrix_data)
            matrix_file = os.path.join(output_dir, f"lora_ewc_performance_matrix_{timestamp}.csv")
            matrix_df.to_csv(matrix_file, index=False)
            print(f" Saved performance matrix CSV: {matrix_file}")
    
         # 3. PER-TASK FORGETTING CSV (compatible with plot_ewc_results.py)
    forgetting_file = None
    if 'nmse' in cl_metrics and 'bwt' in cl_metrics['nmse']:
        forgetting_data = []
        for task_id, bwt_value in cl_metrics['nmse']['bwt']['per_task'].items():
            # Convert task ID to domain name
            task_idx = int(task_id)
            domain_name = f"domain_{config.data.tasks_params[task_idx].data_name.split('.')[0]}"
            
            forgetting_data.append({
                'Domain': domain_name,
                'Task_ID': task_id,
                'Forgetting': max(0.0, bwt_value),  # Only positive forgetting
                'BWT': bwt_value,  # Include raw BWT
                'Method': 'LoRA_EWC',
                'Timestamp': timestamp
            })
        
        if forgetting_data:
            forgetting_df = pd.DataFrame(forgetting_data)
            forgetting_file = os.path.join(output_dir, f"lora_ewc_per_task_forgetting_{timestamp}.csv")
            forgetting_df.to_csv(forgetting_file, index=False)
            print(f" Saved per-task forgetting CSV: {forgetting_file}")
    
    # 4. FINAL EVALUATION CSV (compatible with plot_final_performance.py)
    eval_file = None
    if final_evaluation_results and 'domain_results' in final_evaluation_results:
        eval_data = []
        for domain_id, metrics in final_evaluation_results['domain_results'].items():
            # Convert domain ID to full domain name
            domain_idx = int(domain_id)
            domain_name = f"domain_{config.data.tasks_params[domain_idx].data_name.split('.')[0]}"
            
            eval_data.append({
                'Method': 'LoRA_EWC',
                'Domain': domain_name,
                'Domain_ID': domain_id,
                'SSIM': metrics['ssim'],
                'NMSE': metrics['nmse'],
                'PSNR': metrics['psnr'],
                'SSIM_std': 0.0,  # Would need multiple runs for std
                'NMSE_std': 0.0,
                'PSNR_std': 0.0,
                'Timestamp': timestamp
            })
        
        if eval_data:
            eval_df = pd.DataFrame(eval_data)
            eval_file = os.path.join(output_dir, f"lora_ewc_final_evaluation_{timestamp}.csv")
            eval_df.to_csv(eval_file, index=False)
            print(f" Saved final evaluation CSV: {eval_file}")
    
    # 5. CONTINUAL LEARNING METRICS CSV (for cross-method comparison)
    cl_summary_data = {
        'Method': ['LoRA_EWC'],
        'Backward_Transfer_BWT': [cl_metrics['summary']['overall_bwt']],
        'Forward_Transfer_FWT': [cl_metrics['summary']['overall_fwt']],
        'Forgetting': [abs(cl_metrics['summary']['overall_bwt']) if cl_metrics['summary']['overall_bwt'] > 0 else 0],
        'Final_Average_Performance': [cl_metrics['summary']['overall_final_performance']],
        'Learning_Accuracy': [cl_metrics['summary']['overall_final_performance']],
        'Memory_Footprint_MB': [memory_footprint_mb],
        'Training_Time_Minutes': [total_training_time / 60],
        'Num_Tasks': [len(config.data.sequence)],
        'Timestamp': [timestamp]
    }
    
    # Add metric-specific BWT/FWT
    for metric_name in ['ssim', 'nmse', 'psnr']:
        if metric_name in cl_metrics:
            cl_summary_data[f'{metric_name.upper()}_BWT'] = [cl_metrics[metric_name]['bwt']['average']]
            cl_summary_data[f'{metric_name.upper()}_FWT'] = [cl_metrics[metric_name]['fwt']['average']]
            cl_summary_data[f'{metric_name.upper()}_Final'] = [cl_metrics[metric_name]['average_final_performance']]
    
    cl_summary_df = pd.DataFrame(cl_summary_data)
    cl_summary_file = os.path.join(output_dir, f"lora_ewc_continual_learning_summary_{timestamp}.csv")
    cl_summary_df.to_csv(cl_summary_file, index=False)
    print(f" Saved continual learning summary CSV: {cl_summary_file}")
    
    # Print summary for user
    print(f"\n CSV EXPORT SUMMARY:")
    print(f"    Output directory: {output_dir}")
    print(f"    Files created: 5 CSV files for comprehensive plotting")
    print(f"    Compatible with your plot_ewc_results.py and plot_final_performance.py")
    print(f"    Includes: BWT, FWT, performance matrix, per-domain results, forgetting analysis")
    
    return summary_file, matrix_file, eval_file, forgetting_file, cl_summary_file

def calculate_model_memory_footprint(model, fisher_manager=None) -> float:
    """Calculate total memory footprint in MB."""
    
    # Model parameters
    model_params = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
    
    # Fisher matrices
    fisher_params = 0.0
    if fisher_manager and fisher_manager.fisher_matrices:
        for task_fishers in fisher_manager.fisher_matrices.values():
            if isinstance(task_fishers, dict):
                fisher_params += sum(f.numel() * f.element_size() for f in task_fishers.values()) / (1024 * 1024)
    
    total_memory_mb = model_params + fisher_params
    
    print(f"\n Memory Footprint Analysis:")
    print(f"   Model parameters: {model_params:.2f} MB")
    print(f"   Fisher matrices: {fisher_params:.2f} MB") 
    print(f"   Total: {total_memory_mb:.2f} MB")
    
    return total_memory_mb

def main(config_path: str, wandb_run_id: str = None, wandb_project: str = None, wandb_entity: str = None, eval_results_file: str = None):
    """
    Main function to run the continual learning training process.
    """
    # Track total training time
    training_start_time = datetime.now()
    
    # --- 1. Initialization ---
    config = load_config(config_path)
    set_seed(config.framework.seed)  # Use comprehensive seed setting function
    device = get_device(config.hardware.device)
    print(f"Using device: {device}")
    print(f"Training started at: {training_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Initialize Wandb if available and configured
    wandb_run = None
    if WANDB_AVAILABLE and (wandb_project or hasattr(config, 'wandb')):
        try:
            wandb_config = {
                'project': wandb_project or getattr(config.wandb, 'project', 'lora_continual_learning'),
                'entity': wandb_entity or getattr(config.wandb, 'entity', None),
                'config': config.model_dump(),
                'tags': ['continual_learning', 'lora', 'multi_domain']
            }
            
            if wandb_run_id:
                wandb_config['id'] = wandb_run_id
                wandb_config['resume'] = 'allow'
            
            wandb_run = wandb.init(**wandb_config)
            print(f"W&B initialized: {wandb_run.name}")
        except Exception as e:
            print(f"Warning: Failed to initialize W&B: {e}")
            wandb_run = None

    # --- 2. Data and Model Setup ---
    print("Setting up model and data...")
    # Crucially, get norm_stats from the original backbone checkpoint
    norm_stats = get_norm_stats_from_checkpoint(config.model.pretrained_path)
    if norm_stats is None:
        print("Could not find normalisation stats in backbone checkpoint. Exiting.")
        return
    # Store them in the config for saving in the final checkpoint
    config.data.norm_stats = norm_stats
    
    # Initialize the LoRA-adapted model. This class handles backbone loading.
    model = UNet_SRCNN_LoRA(config).to(device)

    # Initialize EWC (Elastic Weight Consolidation) manager for Fisher computation
    # Always create Fisher manager for computation/saving, but can disable loss with enabled=False or lambda=0
    ewc_enabled = getattr(config.training.ewc, 'enabled', True) if hasattr(config.training, 'ewc') else False
    lambda_ewc = config.training.ewc.lambda_ewc if hasattr(config.training, 'ewc') else 0.0
    fisher_manager = FisherInformationManager(lambda_ewc=lambda_ewc)
    ewc_loss_fn = EWCLoss(fisher_manager) if (ewc_enabled and lambda_ewc > 0) else None
    
    print(" Initializing EWC Fisher Information Manager...")
    print(f"   λ_EWC = {lambda_ewc}")
    print(f"   EWC enabled = {ewc_enabled}")
    if ewc_enabled and lambda_ewc > 0:
        print(f"   EWC loss: ENABLED during training")
    else:
        if not ewc_enabled:
            print(f"   EWC loss: DISABLED (enabled=False), but Fisher still computed for saving")
        elif lambda_ewc == 0:
            print(f"   EWC loss: DISABLED (λ=0), but Fisher still computed for saving")
        else:
            print(f"   EWC loss: DISABLED, but Fisher still computed for saving")
    print(f"   Fisher computation: {'after each task' if hasattr(config.training, 'ewc') and config.training.ewc.compute_fisher_after_task else 'always'}")

    # --- 3. Continual Learning Loop ---
    print("\n--- Starting Continual Learning Training ---")
    
    # Use a single loss function for all tasks as defined in config
    criterion = nn.MSELoss() if config.training.loss_function == 'mse' else nn.HuberLoss()

    # Create a single GradScaler for all epochs to preserve internal statistics
    scaler = GradScaler(enabled=config.hardware.use_amp)
    
    # Get gradient clipping norm if specified
    clip_norm = getattr(config.training, 'gradient_clip_norm', None)
    if clip_norm is not None:
        print(f"Gradient clipping enabled with norm: {clip_norm}")

    # Store replay buffers for each task
    replay_buffers = {}
    
    # Track best validation performance per domain for checkpointing
    best_task_val_losses = {}  # Track best validation loss per task
    checkpoint_save_count = 0
    
    # === CONTINUAL LEARNING METRICS TRACKING ===
    # Track performance evolution for BWT/FWT calculation
    performance_matrix = {}  # {step: {task_id: {metric: value}}}
    completed_tasks = []     # List of task IDs that have been trained
    task_sequence = [str(task_id) for task_id in config.data.sequence]
    
    print(f" Continual Learning Metrics Tracking Enabled")
    print(f"    Will track BWT, FWT, and performance evolution")
    print(f"    Task sequence: {task_sequence}")

    for task_id_int in config.data.sequence:
        task_id = str(task_id_int)
        print(f"\n--- Task {task_id} ---")

        # Add and activate the adapters for the current task
        model.add_task(task_id)
        model.set_active_task(task_id)

        # Get data for the current task
        train_loader, val_loader = get_dataloaders(
            task_id=task_id,
            config=config.data,
            batch_size=config.training.batch_size,
            norm_stats=config.data.norm_stats
        )

        # The optimiser is created for each task, targeting only the newly trainable params
        task_weight_decay = config.training.task_weight_decays.get(task_id_int, config.training.weight_decay)
        
        # Get task-specific learning rate
        task_learning_rate = config.training.task_learning_rates.get(task_id_int, config.training.learning_rate) if config.training.task_learning_rates else config.training.learning_rate
        
        # Support different optimizers
        optimizer_name = getattr(config.training, 'optimizer', config.training.optimiser).lower()
        
        if optimizer_name == 'adam':
            optimiser = optim.Adam(
                model.trainable_parameters(), 
                lr=task_learning_rate,
                weight_decay=task_weight_decay,
                betas=config.training.betas
            )
        elif optimizer_name == 'adamw':
            optimiser = optim.AdamW(
                model.trainable_parameters(), 
                lr=task_learning_rate,
                weight_decay=task_weight_decay,
                betas=config.training.betas
            )
        elif optimizer_name == 'sgd':
            optimiser = optim.SGD(
                model.trainable_parameters(), 
                lr=task_learning_rate,
                weight_decay=task_weight_decay,
                momentum=getattr(config.training, 'momentum', 0.9)
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}. Supported: adam, adamw, sgd")
        
        print(f"Using {optimizer_name.upper()} optimizer with lr={task_learning_rate}, wd={task_weight_decay}")
        
        # Create task-specific scheduler
        task_scheduler_config = config.training.scheduler
        if config.training.task_scheduler_params and task_id_int in config.training.task_scheduler_params:
            # Override scheduler params for this task
            from copy import deepcopy
            task_scheduler_config = deepcopy(config.training.scheduler)
            if task_scheduler_config and task_scheduler_config.params:
                # Update params with task-specific values
                task_specific_params = config.training.task_scheduler_params[task_id_int]
                for param_name, param_value in task_specific_params.items():
                    setattr(task_scheduler_config.params, param_name, param_value)
        
        scheduler = create_scheduler(optimiser, task_scheduler_config)
        if scheduler:
            scheduler_params = task_scheduler_config.params.model_dump() if task_scheduler_config.params else {}
            print(f"Using {task_scheduler_config.type} scheduler with params: {scheduler_params}")
        
        # Get task-specific epoch count
        task_epochs = config.training.task_epochs.get(task_id_int, config.training.epochs_per_task) if config.training.task_epochs else config.training.epochs_per_task
        print(f"Training for {task_epochs} epochs (task-specific)")
        
        # Training sub-loop for the current task
        for epoch in range(task_epochs):
            print(f"Epoch {epoch+1}/{task_epochs}")
            
            # Training with EWC loss (following Algorithm 1)
            train_results = train_one_epoch(model, train_loader, criterion, optimiser, device, 
                                          config.hardware.use_amp, scaler, clip_norm, wandb_run,
                                          ewc_loss_fn, task_id)
            if len(train_results) == 3:
                train_loss_combined, train_loss_main, train_loss_ewc = train_results
            else:
                # Backwards compatibility
                train_loss_combined = train_loss_main = train_results
                train_loss_ewc = 0.0
            
            val_loss = validate_one_epoch(model, val_loader, criterion, device)
            
            # Step scheduler if available
            if scheduler:
                if hasattr(scheduler, 'step'):
                    if 'ReduceLROnPlateau' in str(type(scheduler)):
                        scheduler.step(val_loss)  # ReduceLROnPlateau needs validation loss
                    else:
                        scheduler.step()  # Other schedulers step per epoch
            
            # Get current learning rate for logging
            current_lr = optimiser.param_groups[0]['lr']
            
            # Print epoch summary with EWC info
            if train_loss_ewc > 0:
                print(f"Epoch {epoch+1} Summary: Train Loss: {train_loss_combined:.6f} (Main: {train_loss_main:.6f}, EWC: {train_loss_ewc:.6f}), Val Loss: {val_loss:.6f}, LR: {current_lr:.8f}")
            else:
                print(f"Epoch {epoch+1} Summary: Train Loss: {train_loss_combined:.6f}, Val Loss: {val_loss:.6f}, LR: {current_lr:.8f}")
            
            # Save checkpoint if current domain validation loss improved
            is_best = val_loss < best_task_val_losses.get(task_id, float('inf'))
            if is_best:
                best_task_val_losses[task_id] = val_loss
                checkpoint_save_count += 1
                
                # Save best model checkpoint (accumulates best adapters for each domain)
                best_checkpoint_path = os.path.join(config.logging.checkpoint_dir, "BEST.pth")
                os.makedirs(config.logging.checkpoint_dir, exist_ok=True)
                
                best_checkpoint = {
                    'model_state_dict': model.state_dict(),
                    'config': config.model_dump(),
                    'best_task_val_losses': best_task_val_losses.copy(),
                    'last_saved_info': {
                        'epoch': epoch + 1,
                        'task_id': task_id,
                        'val_loss': val_loss,
                        'train_loss_combined': train_loss_combined,
                        'train_loss_main': train_loss_main,
                        'train_loss_ewc': train_loss_ewc
                    },
                    'replay_buffers': {tid: buf.state_dict() for tid, buf in replay_buffers.items()},
                    'fisher_manager': fisher_manager.state_dict() if fisher_manager is not None else None,
                    'model_class': 'UNet_SRCNN_LoRA',
                    'checkpoint_type': 'best_per_domain',
                    'save_count': checkpoint_save_count
                }
                torch.save(best_checkpoint, best_checkpoint_path)
                print(f" NEW BEST for Domain {task_id}: {val_loss:.6f} (Epoch {epoch+1}) [Save #{checkpoint_save_count}]")
            
            # Log metrics to W&B
            if wandb_run:
                log_dict = {
                    f'task_{task_id}_train_loss_combined': train_loss_combined,
                    f'task_{task_id}_train_loss_main': train_loss_main,
                    f'task_{task_id}_train_loss_ewc': train_loss_ewc,
                    f'task_{task_id}_val_loss': val_loss,
                    f'task_{task_id}_best_val_loss': best_task_val_losses.get(task_id, float('inf')),
                    f'task_{task_id}_learning_rate': current_lr,
                    'epoch': epoch + 1,
                    'task': task_id,
                    'is_best_for_domain': is_best,
                    'checkpoint_save_count': checkpoint_save_count,
                    'ewc_lambda': lambda_ewc
                }
                
                # Add domain-specific best info if this is the best for this domain
                if is_best:
                    log_dict.update({
                        f'new_best_domain_{task_id}_epoch': epoch + 1,
                        f'new_best_domain_{task_id}_val_loss': val_loss
                    })
                
                wandb_run.log(log_dict)

        # === CONTINUAL LEARNING EVALUATION ===
        # Add current task to completed tasks list
        completed_tasks.append(task_id)
        current_task_step = len(completed_tasks) - 1  # 0-indexed step
        
        # Evaluate on all completed tasks to track performance evolution
        print(f"\n Task {task_id} completed! Running CL evaluation...")
        step_performance = evaluate_continual_learning_step(
            model, config, device, completed_tasks, current_task_step
        )
        
        # Store results in performance matrix
        performance_matrix[current_task_step] = step_performance
        
        # Log step performance to W&B
        if wandb_run and step_performance:
            cl_log_dict = {}
            for evaluated_task_id, task_metrics in step_performance.items():
                for metric_name, metric_value in task_metrics.items():
                    cl_log_dict[f'cl_step_{current_task_step}_task_{evaluated_task_id}_{metric_name}'] = metric_value
            
            # Add summary for this step
            cl_log_dict.update({
                f'cl_step_{current_task_step}_num_tasks_evaluated': len(step_performance),
                f'cl_step_{current_task_step}_current_task': task_id,
                'cl_step': current_task_step,
                'cl_completed_tasks': len(completed_tasks)
            })
            
            wandb_run.log(cl_log_dict)
        
        print(f" CL evaluation step {current_task_step + 1} completed!")

        # Populate replay buffer after training this task (when model knows the domain)
        replay_buffer = populate_replay_buffer(model, train_loader, task_id, device)
        replay_buffers[task_id] = replay_buffer

        # Compute Fisher Information for EWC (always compute for saving, following Algorithm 1)
        compute_fisher = True
        if hasattr(config.training, 'ewc') and hasattr(config.training.ewc, 'compute_fisher_after_task'):
            compute_fisher = config.training.ewc.compute_fisher_after_task
        
        if fisher_manager is not None and compute_fisher:
            print(f"\n Computing Fisher Information for Task {task_id}...")
            try:
                # Use validation loader for Fisher computation
                num_fisher_samples = None
                if hasattr(config.training, 'ewc') and hasattr(config.training.ewc, 'fisher_samples'):
                    num_fisher_samples = config.training.ewc.fisher_samples
                
                fisher_manager.compute_fisher_information(
                    model, task_id, val_loader, device, num_samples=num_fisher_samples
                )
                print(f" Fisher Information computed for Task {task_id}")
                
                # Log Fisher statistics to W&B if available
                if wandb_run:
                    fisher_stats = fisher_manager.get_statistics()
                    if task_id in fisher_stats.get('task_statistics', {}):
                        task_fisher_stats = fisher_stats['task_statistics'][task_id]
                        wandb_run.log({
                            f'task_{task_id}_fisher_mean': task_fisher_stats['mean_fisher'],
                            f'task_{task_id}_fisher_std': task_fisher_stats['std_fisher'],
                            f'task_{task_id}_fisher_params': task_fisher_stats['num_parameters'],
                            f'task_{task_id}_fisher_max': task_fisher_stats['max_fisher'],
                            f'task_{task_id}_fisher_min': task_fisher_stats['min_fisher']
                        })
            except Exception as e:
                print(f"️ Warning: Failed to compute Fisher Information for Task {task_id}: {e}")

        # Clear CUDA cache between tasks to avoid memory fragmentation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"CUDA cache cleared after task {task_id}")
        
        print(f"Task {task_id} training completed. Best validation loss: {best_task_val_losses.get(task_id, 'N/A'):.6f}")
        
        # Log final task summary to W&B
        if wandb_run:
            wandb_run.log({
                f'final_task_{task_id}_train_loss_combined': train_loss_combined,
                f'final_task_{task_id}_train_loss_main': train_loss_main,
                f'final_task_{task_id}_train_loss_ewc': train_loss_ewc,
                f'final_task_{task_id}_val_loss': val_loss,
                f'best_task_{task_id}_val_loss': best_task_val_losses.get(task_id, float('inf')),
                f'task_{task_id}_replay_buffer_size': len(replay_buffer),
                'completed_tasks': len(replay_buffers)
            })

    print("\n--- Continual Learning Training Finished ---")
    
    # === CALCULATE FINAL CONTINUAL LEARNING METRICS ===
    cl_metrics = None
    if len(completed_tasks) > 1 and performance_matrix:
        print(f"\n Computing Final Continual Learning Metrics...")
        print(f"    Performance matrix has {len(performance_matrix)} steps")
        print(f"    Completed {len(completed_tasks)} tasks: {completed_tasks}")
        
        try:
            # Calculate BWT, FWT, and other CL metrics
            cl_metrics = calculate_continual_learning_metrics(performance_matrix, task_sequence)
            
            # Verify the calculations
            verify_bwt_fwt_calculations(cl_metrics, performance_matrix, task_sequence)
            
            # Print summary results
            if 'summary' in cl_metrics:
                summary = cl_metrics['summary']
                print(f"\n CONTINUAL LEARNING RESULTS SUMMARY:")
                print(f"    Overall BWT: {summary['overall_bwt']:.4f}")
                print(f"    Overall FWT: {summary['overall_fwt']:.4f}")
                print(f"    Overall Final Performance: {summary['overall_final_performance']:.4f}")
                print(f"    Tasks completed: {summary['num_tasks']}")
            
            # Print detailed results for each metric
            for metric_name in ['ssim', 'nmse', 'psnr']:
                if metric_name in cl_metrics:
                    metric_results = cl_metrics[metric_name]
                    print(f"\n    {metric_name.upper()} Results:")
                    print(f"      BWT: {metric_results['bwt']['average']:.4f}")
                    print(f"      FWT: {metric_results['fwt']['average']:.4f}")
                    print(f"      Final Performance: {metric_results['average_final_performance']:.4f}")
            
            # Log to W&B
            if wandb_run and cl_metrics:
                cl_summary_log = {}
                
                # Overall summary metrics
                if 'summary' in cl_metrics:
                    summary = cl_metrics['summary']
                    cl_summary_log.update({
                        'cl_overall_bwt': summary['overall_bwt'],
                        'cl_overall_fwt': summary['overall_fwt'],
                        'cl_overall_final_performance': summary['overall_final_performance'],
                        'cl_num_tasks_completed': summary['num_tasks']
                    })
                
                # Per-metric results
                for metric_name in ['ssim', 'nmse', 'psnr']:
                    if metric_name in cl_metrics:
                        metric_results = cl_metrics[metric_name]
                        cl_summary_log.update({
                            f'cl_{metric_name}_bwt': metric_results['bwt']['average'],
                            f'cl_{metric_name}_fwt': metric_results['fwt']['average'],
                            f'cl_{metric_name}_final_performance': metric_results['average_final_performance']
                        })
                        
                        # Per-task BWT and FWT
                        for task_id, bwt_val in metric_results['bwt']['per_task'].items():
                            cl_summary_log[f'cl_{metric_name}_bwt_task_{task_id}'] = bwt_val
                        for task_id, fwt_val in metric_results['fwt']['per_task'].items():
                            cl_summary_log[f'cl_{metric_name}_fwt_task_{task_id}'] = fwt_val
                
                wandb_run.log(cl_summary_log)
                print(f" CL metrics logged to W&B")
        
        except Exception as e:
            print(f"️ Error calculating continual learning metrics: {e}")
            cl_metrics = None
    else:
        print(f"️ Insufficient data for CL metrics (need >1 task): {len(completed_tasks)} tasks completed")
    
    # Save final checkpoint with all replay buffers included
    if best_task_val_losses:
        print(f"\n Saving final checkpoint with best adapters + all replay buffers...")
        final_checkpoint_path = os.path.join(config.logging.checkpoint_dir, "FINAL_WITH_REPLAY.pth")
        best_checkpoint_path = os.path.join(config.logging.checkpoint_dir, "BEST.pth")
        os.makedirs(config.logging.checkpoint_dir, exist_ok=True)
        
        # Load the best model state (which contains best adapters for each domain)
        if os.path.exists(best_checkpoint_path):
            print(f"    Loading best model state from: BEST.pth")
            best_checkpoint = torch.load(best_checkpoint_path, map_location=device)
            best_model_state = best_checkpoint['model_state_dict']
        else:
            print(f"   ️  BEST.pth not found, using current model state")
            best_model_state = model.state_dict()
        
        final_checkpoint = {
            'model_state_dict': best_model_state,  # Use BEST adapters, not final ones
            'config': config.model_dump(),
            'best_task_val_losses': best_task_val_losses.copy(),
            'final_training_info': {
                'total_tasks_trained': len(replay_buffers),
                'task_sequence': [str(tid) for tid in config.data.sequence],
                'replay_buffer_sizes': {tid: len(buf) for tid, buf in replay_buffers.items()},
                'uses_best_adapters': True,
                'note': 'Contains best LoRA adapters for each domain + all replay buffers + Fisher matrices + CL metrics'
            },
            'replay_buffers': {tid: buf.state_dict() for tid, buf in replay_buffers.items()},
            'fisher_manager': fisher_manager.state_dict() if fisher_manager is not None else None,
            'continual_learning_metrics': cl_metrics,  # Add CL metrics to checkpoint
            'performance_matrix': performance_matrix,  # Add performance evolution data
            'model_class': 'UNet_SRCNN_LoRA',
            'checkpoint_type': 'best_adapters_with_all_replay_buffers_fisher_and_cl_metrics',
            'save_count': checkpoint_save_count + 1
        }
        torch.save(final_checkpoint, final_checkpoint_path)
        print(f" Final checkpoint saved to: {final_checkpoint_path}")
        print(f"    Uses BEST LoRA adapters for each domain (not final training state)")
        print(f"    Included replay buffers for {len(replay_buffers)} tasks:")
        for task_id, buffer in replay_buffers.items():
            print(f"      Task {task_id}: {len(buffer)} samples")
        
        # Print Fisher Information summary
        if fisher_manager is not None:
            print(f"    Fisher Information matrices computed for {len(fisher_manager.fisher_matrices)} tasks")
            fisher_manager.print_statistics()
    
    # Summary of best model per domain
    if best_task_val_losses:
        print(f"\n BEST MODEL SUMMARY:")
        print(f"    Best checkpoint: {os.path.join(config.logging.checkpoint_dir, 'BEST.pth')}")
        print(f"    Final checkpoint: {os.path.join(config.logging.checkpoint_dir, 'FINAL_WITH_REPLAY.pth')}")
        print(f"    Total saves: {checkpoint_save_count + 1}")
        print(f"    Best validation losses per domain:")
        for domain, val_loss in best_task_val_losses.items():
            print(f"      Domain {domain}: {val_loss:.6f}")
        
        # Log final best model info to W&B
        if wandb_run:
            log_dict = {
                'total_checkpoint_saves': checkpoint_save_count + 1,
                'domains_trained': len(best_task_val_losses),
                'final_checkpoint_path': final_checkpoint_path,
                'total_replay_buffer_samples': sum(len(buf) for buf in replay_buffers.values())
            }
            # Add per-domain best losses
            for domain, val_loss in best_task_val_losses.items():
                log_dict[f'final_best_domain_{domain}_val_loss'] = val_loss
            # Add replay buffer sizes
            for task_id, buffer in replay_buffers.items():
                log_dict[f'replay_buffer_size_task_{task_id}'] = len(buffer)
            
            # Add Fisher Information statistics
            if fisher_manager is not None:
                fisher_stats = fisher_manager.get_statistics()
                log_dict.update({
                    'fisher_tasks_computed': fisher_stats['num_tasks'],
                    'fisher_lambda_ewc': fisher_stats['lambda_ewc']
                })
                if 'global_statistics' in fisher_stats:
                    global_fisher = fisher_stats['global_statistics']
                    log_dict.update({
                        'fisher_total_params': global_fisher['total_parameters'],
                        'fisher_global_mean': global_fisher['mean_fisher'],
                        'fisher_global_std': global_fisher['std_fisher'],
                        'fisher_global_max': global_fisher['max_fisher'],
                        'fisher_global_min': global_fisher['min_fisher']
                    })
            
            wandb_run.log(log_dict)
    
    # === DIRECT MULTI-DOMAIN EVALUATION (W&B Independent) ===
    eval_results = None
    try:
        print("\n=== Running Direct Multi-Domain Evaluation ===")
        
        # Define domains to evaluate
        domain_ids = [str(i) for i in config.data.sequence]
        
        # Run direct evaluation
        eval_results = evaluate_all_domains_direct(model, config, device, domain_ids)
        
        # Save results to file if requested
        if eval_results_file:
            try:
                os.makedirs(os.path.dirname(eval_results_file), exist_ok=True)
                with open(eval_results_file, 'w') as f:
                    json.dump(eval_results, f, indent=2)
                print(f" Evaluation results saved to: {eval_results_file}")
            except Exception as e:
                print(f"️  Warning: Could not save results to file: {e}")
        
        # Log to W&B if available
        if wandb_run and eval_results:
            for domain_id, metrics in eval_results['domain_results'].items():
                for metric_name, value in metrics.items():
                    wandb_run.summary[f"eval_domain_{domain_id}_{metric_name}"] = value
            
            # Also log aggregate stats
            if 'aggregate_stats' in eval_results:
                for metric_name, stats in eval_results['aggregate_stats'].items():
                    if isinstance(stats, dict):
                        for stat_name, stat_value in stats.items():
                            wandb_run.summary[f"aggregate_{metric_name}_{stat_name}"] = stat_value
        
        print(" Direct multi-domain evaluation completed successfully!")
        
    except Exception as e:
        print(f"️ Error in direct multi-domain evaluation: {e}")
        eval_results = None
    
    # === CALCULATE TRAINING STATISTICS ===
    training_end_time = datetime.now()
    total_training_time = (training_end_time - training_start_time).total_seconds()
    
    print(f"\n️ TRAINING TIME ANALYSIS:")
    print(f"   Started: {training_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   Finished: {training_end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   Total time: {total_training_time/3600:.2f} hours ({total_training_time/60:.1f} minutes)")
    print(f"   Average time per task: {total_training_time/len(config.data.sequence)/60:.1f} minutes")
    
    # Calculate memory footprint
    memory_footprint_mb = calculate_model_memory_footprint(model, fisher_manager)
    
    # === EXPORT CSV FILES FOR PLOTTING ===
    if cl_metrics is not None and eval_results is not None:
        try:
            print(f"\n Exporting CSV files for comprehensive plotting...")
            csv_output_dir = os.path.join(config.logging.checkpoint_dir, "csv_results")
            
            save_continual_learning_csvs(
                cl_metrics=cl_metrics,
                performance_matrix=performance_matrix,
                best_task_val_losses=best_task_val_losses,
                config=config,
                final_evaluation_results=eval_results,
                total_training_time=total_training_time,
                memory_footprint_mb=memory_footprint_mb,
                output_dir=csv_output_dir
            )
            
            print(f" CSV export completed! Use these files with your plotting scripts:")
            print(f"    Location: {csv_output_dir}")
            print(f"    Plot with: plot_ewc_results.py --results_dir {csv_output_dir}")
            print(f"    Or with: plot_final_performance.py --results_dir {csv_output_dir}")
            
        except Exception as e:
            print(f"️ Error exporting CSV files: {e}")
    else:
        print(f"️ Cannot export CSV files: Missing continual learning metrics or evaluation results")
    
    # Finish W&B run
    if wandb_run:
        wandb_run.finish()
        print("W&B run finished.")
    
    # Create comprehensive results for external use
    final_results = {
        'direct_evaluation': eval_results,
        'continual_learning_metrics': cl_metrics,
        'performance_matrix': performance_matrix,
        'best_task_val_losses': best_task_val_losses,
        'completed_tasks': completed_tasks,
        'total_training_time_seconds': total_training_time,
        'memory_footprint_mb': memory_footprint_mb
    }
    
    # Return comprehensive results (for Optuna or other external callers)
    return final_results

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run LoRA Continual Learning Training")
    parser.add_argument(
        '--config_path', 
        type=str, 
        required=True,
        help="Path to the experiment configuration YAML file."
    )
    parser.add_argument('--wandb_run_id', type=str, help="W&B run ID for logging")
    parser.add_argument('--wandb_project', type=str, help="W&B project name")
    parser.add_argument('--wandb_entity', type=str, help="W&B entity name")
    parser.add_argument('--eval_results_file', type=str, help="Path to save evaluation results JSON file")
    args = parser.parse_args()
    
    main(args.config_path, args.wandb_run_id, args.wandb_project, args.wandb_entity, args.eval_results_file) 