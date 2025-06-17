"""
Training Script for Elastic Weight Consolidation (EWC) Baseline

This script trains the UNet-SRCNN model using EWC for continual learning
across multiple domains.
"""

import argparse
import os
import sys
import yaml
import torch
import json
from pathlib import Path
from datetime import datetime
import pandas as pd

# Add the project root to the path so we can import from standard_training_2
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Import from existing infrastructure
from standard_training_2.dataset import StandardDataset, NormalisingDatasetWrapper
from standard_training_2.models.unet_srcnn import UNetCombinedModel
from torch.utils.data import DataLoader, random_split

# Import our EWC implementation
from baseline_cl.methods.ewc import EWCTrainer

# Disable wandb for this script
WANDB_AVAILABLE = False


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def get_device(device_config: str) -> torch.device:
    """Get the appropriate device."""
    if device_config == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_config)
    return device


def load_pretrained_model(config: dict, device: torch.device):
    """
    Load pretrained UNet-SRCNN model and extract normalisation statistics.
    """
    pretrained_path = config['model']['pretrained_path']
    
    if not os.path.exists(pretrained_path):
        raise FileNotFoundError(f"Pretrained model not found: {pretrained_path}")
    
    print(f"Loading pretrained model from: {pretrained_path}")
    checkpoint = torch.load(pretrained_path, map_location=device)
    
    # Extract normalisation statistics
    if 'config' in checkpoint and 'data' in checkpoint['config'] and 'norm_stats' in checkpoint['config']['data']:
        norm_stats_lists = checkpoint['config']['data']['norm_stats']
        
        # Convert lists to tensors and ensure correct shape for (C, H, W) data
        # Original stats are in (1, 1, C) format for (H, W, C) data
        # Need to reshape to (C, 1, 1) for (C, H, W) data
        norm_stats = {
            'mean_inputs': torch.tensor(norm_stats_lists['mean_inputs'], dtype=torch.float32).squeeze().view(-1, 1, 1),
            'std_inputs': torch.tensor(norm_stats_lists['std_inputs'], dtype=torch.float32).squeeze().view(-1, 1, 1),
            'mean_targets': torch.tensor(norm_stats_lists['mean_targets'], dtype=torch.float32).squeeze().view(-1, 1, 1),
            'std_targets': torch.tensor(norm_stats_lists['std_targets'], dtype=torch.float32).squeeze().view(-1, 1, 1)
        }
        print("\n Normalisation stats loaded and reshaped for (C, H, W) data:")
        for key, value in norm_stats.items():
            print(f"   {key}: shape {value.shape}")
    elif 'config' in checkpoint and 'norm_stats' in checkpoint['config']:
        norm_stats = checkpoint['config']['norm_stats']
    elif 'norm_stats' in checkpoint:
        norm_stats = checkpoint['norm_stats']
    else:
        raise ValueError("Normalisation statistics not found in checkpoint")
    
    # Extract model architecture from checkpoint config (not from EWC config)
    if 'config' not in checkpoint:
        raise ValueError("Checkpoint does not contain config information")
    
    checkpoint_config = checkpoint['config']
    if 'model' not in checkpoint_config or 'params' not in checkpoint_config['model']:
        raise ValueError("Checkpoint config does not contain model parameters")
    
    # Use the model parameters from the checkpoint (these match the saved weights)
    model_params = checkpoint_config['model']['params']
    print(f"\n️ Creating model with architecture from checkpoint:")
    print(f"   Base features: {model_params['base_features']}")
    print(f"   Depth: {model_params['depth']}")
    print(f"   SRCNN channels: {model_params['srcnn_channels']}")
    print(f"   SRCNN kernels: {model_params['srcnn_kernels']}")
    
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
    
    # Load pretrained weights
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    
    print(f" Pretrained model loaded successfully")
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    return model, norm_stats


def create_dataloader(domain_name: str, config: dict, norm_stats: dict, 
                     split: str = 'train') -> DataLoader:
    """
    Create a dataloader for a specific domain.
    
    Args:
        domain_name: Name of the domain (e.g., 'domain_high_snr_med_linear')
        config: Configuration dictionary
        norm_stats: Normalisation statistics
        split: 'train' or 'val'
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
    
    # Create dataset - this will load from preprocessed cache if available
    print(f" Loading dataset: {domain_name}")
    print(f"   Data dir: {config['data']['data_dir']}")
    print(f"   Preprocessed dir: {config['data']['preprocessed_dir']}")
    print(f"   Using SNR: {snr} dB")
    
    try:
        dataset = StandardDataset(
            data_dir=config['data']['data_dir'],
            data_name=domain_name,
            snr=snr,
            interpolation_kernel='thin_plate_spline',
            preprocessed_dir=config['data']['preprocessed_dir'],
            all_data=False
        )
        print(f"    Successfully loaded {len(dataset)} samples")
    except FileNotFoundError as e:
        print(f"    Dataset not found: {e}")
        print(f"    Expected preprocessed file: {config['data']['preprocessed_dir']}/tester_data/{domain_name}_snr{snr}_thin_plate_spline.mat")
        raise
    
    # Split into train/val/test using the same splits as the main pipeline
    val_split = config['data']['validation_split']
    test_split = config['data']['test_split']
    
    total_size = len(dataset)
    val_size = int(total_size * val_split)
    test_size = int(total_size * test_split)
    train_size = total_size - val_size - test_size
    
    print(f"   Data splits: train={train_size}, val={val_size}, test={test_size}")
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)  # Same seed for reproducibility
    )
    
    # Apply normalisation
    if split == 'train':
        target_dataset = train_dataset
    elif split == 'val':
        target_dataset = val_dataset
    else:
        target_dataset = test_dataset
    
    normalised_dataset = NormalisingDatasetWrapper(
        target_dataset, 
        (
            (norm_stats['mean_inputs'].clone().detach(), norm_stats['std_inputs'].clone().detach()),
            (norm_stats['mean_targets'].clone().detach(), norm_stats['std_targets'].clone().detach())
        )
    )
    
    # Create dataloader with same settings as main pipeline
    dataloader = DataLoader(
        normalised_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=(split == 'train'),
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory'],
        persistent_workers=config['data']['persistent_workers'] and config['data']['num_workers'] > 0
    )
    
    return dataloader


def evaluate_all_previous_domains(trainer: EWCTrainer, config: dict, norm_stats: dict,
                                 completed_tasks: list, current_domain: str = None) -> dict:
    """
    Evaluate the model on all previously learned domains.
    
    Args:
        trainer: EWC trainer instance
        config: Configuration dictionary
        norm_stats: Normalisation statistics
        completed_tasks: List of domains completed so far
        current_domain: Current domain name (for including in evaluation)
    """
    domains_to_evaluate = completed_tasks.copy()
    
    # Also evaluate on current domain if provided
    if current_domain and current_domain not in domains_to_evaluate:
        domains_to_evaluate.append(current_domain)
    
    if not domains_to_evaluate:
        return {}
    
    print(f"\n Evaluating on {len(domains_to_evaluate)} domains...")
    
    results = {}
    for domain_name in domains_to_evaluate:
        print(f"   Evaluating {domain_name}...")
        val_loader = create_dataloader(domain_name, config, norm_stats, split='test')
        metrics = trainer.evaluate_domain(domain_name, val_loader, norm_stats)
        results[domain_name] = metrics
        print(f"      SSIM: {metrics['ssim']:.4f}, NMSE: {metrics['nmse']:.6f}, PSNR: {metrics['psnr']:.2f}")
    
    # Calculate aggregate statistics (excluding current domain if it was added)
    eval_domains = completed_tasks if completed_tasks else domains_to_evaluate
    if eval_domains:
        ssim_values = [results[d]['ssim'] for d in eval_domains if d in results]
        nmse_values = [results[d]['nmse'] for d in eval_domains if d in results]
        psnr_values = [results[d]['psnr'] for d in eval_domains if d in results]
        
        if ssim_values:  # Only compute if we have values
            aggregate = {
                'mean_ssim': float(torch.tensor(ssim_values).mean()),
                'mean_nmse': float(torch.tensor(nmse_values).mean()),
                'mean_psnr': float(torch.tensor(psnr_values).mean()),
                'std_ssim': float(torch.tensor(ssim_values).std()) if len(ssim_values) > 1 else 0.0,
                'std_nmse': float(torch.tensor(nmse_values).std()) if len(nmse_values) > 1 else 0.0,
                'std_psnr': float(torch.tensor(psnr_values).std()) if len(psnr_values) > 1 else 0.0
            }
            
            print(f" Aggregate Results (previous domains):")
            print(f"   SSIM: {aggregate['mean_ssim']:.4f} ± {aggregate['std_ssim']:.4f}")
            print(f"   NMSE: {aggregate['mean_nmse']:.8f} ± {aggregate['std_nmse']:.8f}")
            print(f"   PSNR: {aggregate['mean_psnr']:.2f} ± {aggregate['std_psnr']:.2f}")
            
            results['aggregate'] = aggregate
    
    return results


def calculate_continual_learning_metrics(all_results: dict, domains: list, 
                                        baselines: dict, training_times: dict = None) -> dict:
    """
    Calculate continual learning metrics from results using consistent NMSE throughout.
    
    Args:
        all_results: Dictionary of results from training
        domains: List of domain names in training order
        baselines: Dictionary of baseline performance for each domain
    
    Returns:
        Dictionary containing BWT, FWT, ACC, and other metrics
    """
    num_tasks = len(domains)
    performance_matrix = {}  # R[i][j] = performance on task j after training task i
    pre_training_matrix = {}  # R0[i] = performance on task i before training task i
    
    print(f"\n Building performance matrices for {num_tasks} tasks...")
    
    # Extract performance matrices from results
    for task_idx, current_domain in enumerate(domains):
        if current_domain not in all_results:
            print(f"️ Missing results for {current_domain}")
            continue
            
        performance_matrix[task_idx] = {}
        
        task_results = all_results[current_domain]
        
        # Store pre-training performance for FWT calculation
        if 'pre_training_performance' in task_results:
            pre_training_matrix[task_idx] = task_results['pre_training_performance']['nmse']
        
        # Current task performance (diagonal) - use post-training NMSE
        if 'post_training_performance' in task_results:
            current_performance = task_results['post_training_performance']['nmse']
            performance_matrix[task_idx][task_idx] = current_performance
        
        # Previous tasks performance - use NMSE consistently
        if 'evaluation_on_previous' in task_results:
            prev_results = task_results['evaluation_on_previous']
            for prev_task_idx, prev_domain in enumerate(domains[:task_idx]):
                if prev_domain in prev_results:
                    prev_performance = prev_results[prev_domain]['nmse']
                    performance_matrix[task_idx][prev_task_idx] = prev_performance
    
    # Calculate metrics using correct formulae
    metrics = {}
    
    # 1. Final Average Performance (ACC) - average of final row
    if (num_tasks - 1) in performance_matrix:
        final_performances = []
        for task_idx in range(num_tasks):
            if task_idx in performance_matrix[num_tasks - 1]:
                final_performances.append(performance_matrix[num_tasks - 1][task_idx])
        
        metrics['ACC'] = sum(final_performances) / len(final_performances) if final_performances else 0.0
    else:
        metrics['ACC'] = 0.0
    
    # 2. Backward Transfer (BWT) - for NMSE (lower is better), flip sign for intuitive interpretation
    # BWT = (1/(T-1)) * Σ(R_j,j - R_T,j) for j < T (positive = retention, negative = forgetting)
    if num_tasks > 1 and (num_tasks - 1) in performance_matrix:
        bwt_sum = 0.0
        bwt_count = 0
        
        for j in range(num_tasks - 1):  # For each previous task j
            if (j in performance_matrix and j in performance_matrix[j] and 
                j in performance_matrix[num_tasks - 1]):
                # R_j,j - R_T,j (original performance - final performance on task j)
                # For NMSE: positive = retention (final NMSE not much higher), negative = forgetting
                original_perf_on_j = performance_matrix[j][j]
                final_perf_on_j = performance_matrix[num_tasks - 1][j]
                bwt_sum += (final_perf_on_j - original_perf_on_j)  # positive = forgetting
                bwt_count += 1
        
        metrics['BWT'] = bwt_sum / bwt_count if bwt_count > 0 else 0.0
    else:
        metrics['BWT'] = 0.0
    
    # 3. Forward Transfer (FWT) - for NMSE (lower is better), flip sign for intuitive interpretation
    # FWT = (1/(T-1)) * Σ(baseline_i - R_0,i) for i > 0 (positive = transfer helps, negative = interference)
    if num_tasks > 1:
        fwt_sum = 0.0
        fwt_count = 0
        
        for i in range(1, num_tasks):  # For each task after the first
            domain_name = domains[i]
            if i in pre_training_matrix and domain_name in baselines:
                # Baseline performance - pre-training performance  
                # For NMSE: positive = transfer helps (pre-training NMSE lower than baseline)
                baseline_perf = baselines[domain_name]['nmse']
                pre_training_perf = pre_training_matrix[i]
                fwt_sum += (baseline_perf - pre_training_perf)
                fwt_count += 1
        
        metrics['FWT'] = fwt_sum / fwt_count if fwt_count > 0 else 0.0
    else:
        metrics['FWT'] = 0.0
    
    # 4. Average Forgetting - average performance drop on previous tasks  
    # For NMSE: forgetting = final_perf - original_perf (positive = forgetting, NMSE increased)
    forgetting_values = []
    if num_tasks > 1 and (num_tasks - 1) in performance_matrix:
        for j in range(num_tasks - 1):  # For each task except the last
            if (j in performance_matrix and j in performance_matrix[j] and 
                j in performance_matrix[num_tasks - 1]):
                original_perf = performance_matrix[j][j]
                final_perf = performance_matrix[num_tasks - 1][j]
                forgetting = max(0.0, final_perf - original_perf)  # Only positive forgetting (NMSE increase)
                forgetting_values.append(forgetting)
    
    metrics['Forgetting'] = sum(forgetting_values) / len(forgetting_values) if forgetting_values else 0.0
    
    # 5. Learning Accuracy - average performance when learning each task
    learning_accs = []
    for i in range(num_tasks):
        if i in performance_matrix and i in performance_matrix[i]:
            learning_accs.append(performance_matrix[i][i])
    
    metrics['Learning_ACC'] = sum(learning_accs) / len(learning_accs) if learning_accs else 0.0
    
    # 6. Per-task forgetting for detailed analysis
    per_task_forgetting = {}
    if num_tasks > 1 and (num_tasks - 1) in performance_matrix:
        for j in range(num_tasks - 1):
            domain_name = domains[j]
            if (j in performance_matrix and j in performance_matrix[j] and 
                j in performance_matrix[num_tasks - 1]):
                original_perf = performance_matrix[j][j]
                final_perf = performance_matrix[num_tasks - 1][j]
                per_task_forgetting[domain_name] = final_perf - original_perf  # For NMSE: positive = forgetting
    
    metrics['per_task_forgetting'] = per_task_forgetting
    
    # 7. Forward Plasticity (FP) - improvement on new task during its own training
    forward_plasticity_values = []
    if num_tasks > 1:
        for i in range(1, num_tasks):  # For each task after the first
            if i in pre_training_matrix and i in performance_matrix and i in performance_matrix[i]:
                pre_perf = pre_training_matrix[i]
                post_perf = performance_matrix[i][i]
                fp = post_perf - pre_perf  # How much we improved during training
                forward_plasticity_values.append(fp)
    
    metrics['Forward_Plasticity'] = sum(forward_plasticity_values) / len(forward_plasticity_values) if forward_plasticity_values else 0.0
    
    # 8. Training efficiency metrics
    if training_times:
        total_training_time = sum(training_times.values())
        avg_training_time = total_training_time / len(training_times)
        metrics['total_training_time_seconds'] = total_training_time
        metrics['average_training_time_seconds'] = avg_training_time
        metrics['per_task_training_times'] = training_times.copy()
    
    # 9. Task-specific metrics
    task_specific_metrics = {}
    for i, domain_name in enumerate(domains):
        if domain_name in all_results:
            task_metrics = {
                'learning_improvement': 0.0,  # Post - Pre performance
                'baseline_improvement': 0.0   # Post - Baseline performance
            }
            
            if i in pre_training_matrix and i in performance_matrix and i in performance_matrix[i]:
                pre_perf = pre_training_matrix[i]
                post_perf = performance_matrix[i][i]
                task_metrics['learning_improvement'] = post_perf - pre_perf
                
                if domain_name in baselines:
                    baseline_perf = baselines[domain_name]['nmse']
                    task_metrics['baseline_improvement'] = baseline_perf - post_perf  # For NMSE: lower is better
            
            task_specific_metrics[domain_name] = task_metrics
    
    metrics['task_specific_metrics'] = task_specific_metrics
    
    # Store matrices for detailed analysis
    metrics['performance_matrix'] = performance_matrix
    metrics['pre_training_matrix'] = pre_training_matrix
    
    # Memory footprint (Fisher matrices + model parameters)
    # This will be filled in by the main function
    metrics['memory_footprint_mb'] = 0.0
    
    print(f" Metrics calculated:")
    print(f"   ACC: {metrics['ACC']:.4f}")
    print(f"   BWT: {metrics['BWT']:+.4f}")
    print(f"   FWT: {metrics['FWT']:+.4f}")
    print(f"   Forward Plasticity: {metrics['Forward_Plasticity']:+.4f}")
    print(f"   Forgetting: {metrics['Forgetting']:.4f}")
    if training_times:
        print(f"   Total training time: {metrics.get('total_training_time_seconds', 0):.1f}s")
    
    return metrics


def print_continual_learning_summary(metrics: dict, domains: list):
    """Print a comprehensive summary of continual learning performance."""
    print(f"\n{'='*70}")
    print(f" CONTINUAL LEARNING PERFORMANCE SUMMARY")
    print(f"{'='*70}")
    
    print(f" Final Average Performance (ACC): {metrics.get('ACC', 0.0):.4f}")
    print(f" Backward Transfer (BWT):        {metrics.get('BWT', 0.0):+.4f}")
    print(f"   └─ {'Positive = Forgetting, Negative = Retention' if metrics.get('BWT', 0) > 0 else 'No forgetting detected'}")
    
    print(f" Forward Transfer (FWT):         {metrics.get('FWT', 0.0):+.4f}")
    print(f"   └─ {'Positive = Poor generalisation to new domains' if metrics.get('FWT', 0.0) > 0 else 'Negative = Good zero-shot generalisation'}")
    
    print(f" Average Forgetting:             {metrics.get('Forgetting', 0.0):.4f}")
    print(f"   └─ {'Lower is better (less forgetting)' if metrics.get('Forgetting', 0) > 0 else 'No forgetting detected!'}")
    
    print(f" Learning Accuracy:              {metrics.get('Learning_ACC', 0.0):.4f}")
    
    print(f" Forward Plasticity:             {metrics.get('Forward_Plasticity', 0.0):+.4f}")
    print(f"   └─ How much model improves on new tasks during training")
    
    print(f" Memory Footprint:               {metrics.get('memory_footprint_mb', 0.0):.2f} MB")
    
    # Training efficiency
    if 'total_training_time_seconds' in metrics:
        total_time = metrics['total_training_time_seconds']
        avg_time = metrics['average_training_time_seconds']
        print(f"️ Training Time:                  {total_time:.1f}s total, {avg_time:.1f}s avg/task")
    
    # Per-task forgetting details
    if 'per_task_forgetting' in metrics and metrics['per_task_forgetting']:
        print(f"\n Per-Task Forgetting:")
        for domain, forgetting in metrics['per_task_forgetting'].items():
            status = "" if forgetting > 0.01 else ""
            print(f"   {status} {domain[:20]:20s}: {forgetting:+.4f}")
    
    # Performance matrix visualization
    if 'performance_matrix' in metrics:
        print(f"\n Performance Matrix (rows=after task, cols=on task):")
        print(f"     ", end="")
        for i, domain in enumerate(domains):
            print(f"{domain[:8]:>8}", end=" ")
        print()
        
        perf_matrix = metrics['performance_matrix']
        for i in range(len(domains)):
            print(f"T{i+1:2d}: ", end="")
            for j in range(len(domains)):
                if i in perf_matrix and j in perf_matrix[i]:
                    print(f"{perf_matrix[i][j]:8.4f}", end=" ")
                else:
                    print(f"{'---':>8}", end=" ")
            print()
    
    print(f"{'='*70}")


def save_metrics_to_csv(metrics: dict, domains: list, results_dir: str, method_name: str):
    """Save continual learning metrics to CSV files."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 1. Summary metrics CSV
    summary_data = {
        'Method': [method_name],
        'Final_Average_Performance_ACC': [metrics.get('ACC', 0.0)],
        'Backward_Transfer_BWT': [metrics.get('BWT', 0.0)],
        'Forward_Transfer_FWT': [metrics.get('FWT', 0.0)],
        'Forward_Plasticity': [metrics.get('Forward_Plasticity', 0.0)],
        'Average_Forgetting': [metrics.get('Forgetting', 0.0)],
        'Learning_Accuracy': [metrics.get('Learning_ACC', 0.0)],
        'Memory_Footprint_MB': [metrics.get('memory_footprint_mb', 0.0)],
        'Total_Training_Time_Seconds': [metrics.get('total_training_time_seconds', 0.0)],
        'Average_Training_Time_Seconds': [metrics.get('average_training_time_seconds', 0.0)],
        'Num_Domains': [len(domains)],
        'Timestamp': [timestamp]
    }
    
    summary_df = pd.DataFrame(summary_data)
    summary_file = os.path.join(results_dir, f"{method_name}_summary_{timestamp}.csv")
    summary_df.to_csv(summary_file, index=False)
    
    # 2. Performance matrix CSV
    if 'performance_matrix' in metrics:
        perf_matrix = metrics['performance_matrix']
        
        # Create full matrix with NaN for missing values
        matrix_data = []
        for i in range(len(domains)):
            row = {'After_Task': f"T{i+1}_{domains[i][:12]}"}
            for j in range(len(domains)):
                col_name = f"Performance_on_T{j+1}_{domains[j][:8]}"
                if i in perf_matrix and j in perf_matrix[i]:
                    row[col_name] = perf_matrix[i][j]
                else:
                    row[col_name] = None  # Will become NaN in pandas
            matrix_data.append(row)
        
        matrix_df = pd.DataFrame(matrix_data)
        matrix_file = os.path.join(results_dir, f"{method_name}_performance_matrix_{timestamp}.csv")
        matrix_df.to_csv(matrix_file, index=False)
        
        # 3. Per-task forgetting CSV
        per_task_forgetting_file = None
        if 'per_task_forgetting' in metrics:
            forgetting_data = []
            for domain, forgetting_value in metrics['per_task_forgetting'].items():
                forgetting_data.append({
                    'Domain': domain,
                    'Forgetting': forgetting_value,
                    'Method': method_name,
                    'Timestamp': timestamp
                })
            
            if forgetting_data:
                forgetting_df = pd.DataFrame(forgetting_data)
                per_task_forgetting_file = os.path.join(results_dir, f"{method_name}_per_task_forgetting_{timestamp}.csv")
                forgetting_df.to_csv(per_task_forgetting_file, index=False)
        
        # 4. Task-specific metrics CSV
        task_specific_file = None
        if 'task_specific_metrics' in metrics:
            task_data = []
            for domain, task_metrics in metrics['task_specific_metrics'].items():
                row = {
                    'Domain': domain,
                    'Learning_Improvement': task_metrics.get('learning_improvement', 0.0),
                    'Baseline_Improvement': task_metrics.get('baseline_improvement', 0.0),
                    'Method': method_name,
                    'Timestamp': timestamp
                }
                
                # Add training time if available
                if 'per_task_training_times' in metrics and domain in metrics['per_task_training_times']:
                    row['Training_Time_Seconds'] = metrics['per_task_training_times'][domain]
                
                task_data.append(row)
            
            if task_data:
                task_df = pd.DataFrame(task_data)
                task_specific_file = os.path.join(results_dir, f"{method_name}_task_specific_{timestamp}.csv")
                task_df.to_csv(task_specific_file, index=False)
        
        print(f" Metrics saved:")
        print(f"   Summary: {summary_file}")
        print(f"   Performance Matrix: {matrix_file}")
        if per_task_forgetting_file:
            print(f"   Per-task Forgetting: {per_task_forgetting_file}")
        if task_specific_file:
            print(f"   Task-specific Metrics: {task_specific_file}")
    
    return summary_file, matrix_file


class EarlyStoppingTracker:
    """Early stopping with patience and best model tracking."""
    
    def __init__(self, patience: int = 5, min_delta: float = 1e-6, mode: str = 'min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score = float('inf') if mode == 'min' else float('-inf')
        self.counter = 0
        self.early_stop = False
        self.best_epoch = 0
        
    def __call__(self, score: float, epoch: int) -> bool:
        """
        Check if training should stop early.
        
        Returns:
            bool: True if training should stop
        """
        if self.mode == 'min':
            improved = score < (self.best_score - self.min_delta)
        else:
            improved = score > (self.best_score + self.min_delta)
        
        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            self.early_stop = True
            print(f"️ Early stopping triggered at epoch {epoch}")
            print(f"   Best score: {self.best_score:.6f} at epoch {self.best_epoch}")
            
        return self.early_stop


def calculate_baselines(model: torch.nn.Module, config: dict, norm_stats: dict,
                        domains: list) -> dict:
    """
    Calculate baseline performance using the pretrained model before any CL training.
    
    Args:
        model: Pretrained model before any CL training
        config: Configuration dictionary
        norm_stats: Normalisation statistics
        domains: List of all domain names
        
    Returns:
        Dictionary mapping domain names to baseline performance metrics
    """
    print(f"\n Calculating empirical baselines on pretrained model...")
    
    baselines = {}
    model.eval()
    
    with torch.no_grad():
        for domain_name in domains:
            print(f"   Evaluating baseline for {domain_name}...")
            
            # Create test dataloader for this domain (use test for final evaluation)
            val_loader = create_dataloader(domain_name, config, norm_stats, split='test')
            
            # Calculate metrics on this domain
            device = next(model.parameters()).device
            metrics = evaluate_model_on_loader(model, val_loader, norm_stats, device)
            
            baselines[domain_name] = {
                'ssim': metrics['ssim'],
                'nmse': metrics['nmse'], 
                'psnr': metrics['psnr']
            }
            
            print(f"      SSIM: {metrics['ssim']:.4f}, NMSE: {metrics['nmse']:.8f}, PSNR: {metrics['psnr']:.2f}")
    
    print(f" Baselines calculated for {len(baselines)} domains")
    return baselines


def evaluate_model_on_loader(model: torch.nn.Module, dataloader: DataLoader, 
                           norm_stats: dict, device: torch.device) -> dict:
    """
    Evaluate model on a given dataloader and return metrics using correct standard_training_2 implementation.
    
    Args:
        model: Model to evaluate
        dataloader: DataLoader for evaluation
        norm_stats: Normalisation statistics for denormalisation
        device: Device to run evaluation on
        
    Returns:
        Dictionary containing SSIM, NMSE, and PSNR metrics
    """
    from scipy.signal import convolve2d
    import numpy as np
    
    # Correct metric functions (from standard_training_2)
    def nmse(predictions_denorm, targets_original):
        mse = np.mean((predictions_denorm - targets_original) ** 2)
        power = np.mean(targets_original ** 2)
        if power == 0:
            return float('inf') if mse > 0 else 0.0
        return mse / power

    def psnr(predictions_denorm, targets_original, max_val=None):
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

    def ssim(predictions_denorm, targets_original):
        # Convert from (N, C, H, W) to (N, H, W, C) if needed
        if predictions_denorm.ndim == 4 and predictions_denorm.shape[1] == 2:
            predictions_denorm = predictions_denorm.transpose(0, 2, 3, 1)
            targets_original = targets_original.transpose(0, 2, 3, 1)
        
        # Compute complex magnitudes
        pred_mag = np.abs(predictions_denorm[..., 0] + 1j * predictions_denorm[..., 1])
        target_mag = np.abs(targets_original[..., 0] + 1j * targets_original[..., 1])
        
        N, H, W = pred_mag.shape
        
        # Per-sample normalization to [0, 1]
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
        
        # Build 2D Gaussian kernel
        window_size = 11
        sigma = 1.5
        half = window_size // 2
        coords = np.arange(window_size) - half
        g1d = np.exp(- (coords**2) / (2 * sigma**2))
        g1d /= g1d.sum()
        kernel = np.outer(g1d, g1d)
        
        # Constants for SSIM
        data_range = 1.0
        k1, k2 = 0.01, 0.03
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
            xy = x * y
            
            mu_x_sq = mu_x * mu_x
            mu_y_sq = mu_y * mu_y
            mu_xy = mu_x * mu_y
            
            sigma_x_sq = convolve2d(x_sq, kernel, mode='same', boundary='symm') - mu_x_sq
            sigma_y_sq = convolve2d(y_sq, kernel, mode='same', boundary='symm') - mu_y_sq
            sigma_xy = convolve2d(xy, kernel, mode='same', boundary='symm') - mu_xy
            
            numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
            denominator = (mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2)
            
            ssim_map = np.where(denominator > 0, numerator / denominator, 0.0)
            ssim_vals[i] = ssim_map.mean()
        
        return float(ssim_vals.mean())
    
    model.eval()
    all_outputs = []
    all_targets = []
    
    with torch.no_grad():
        for batch_inputs, batch_targets in dataloader:
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)
            
            # Forward pass
            batch_outputs = model(batch_inputs)
            
            # Denormalise for metric calculation
            mean_targets = norm_stats['mean_targets'].to(device)
            std_targets = norm_stats['std_targets'].to(device)
            
            denorm_outputs = batch_outputs * std_targets + mean_targets
            denorm_targets = batch_targets * std_targets + mean_targets
            
            # Convert to numpy and collect
            all_outputs.append(denorm_outputs.cpu().numpy())
            all_targets.append(denorm_targets.cpu().numpy())
    
    # Concatenate all batches
    all_outputs = np.concatenate(all_outputs, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    # Calculate metrics using correct implementation
    nmse_val = nmse(all_outputs, all_targets)
    psnr_val = psnr(all_outputs, all_targets)
    ssim_val = ssim(all_outputs, all_targets)
    
    # Average across all samples
    return {
        'ssim': float(ssim_val),
        'nmse': float(nmse_val),
        'psnr': float(psnr_val)
    }


def evaluate_before_training(model: torch.nn.Module, domain_name: str, 
                           config: dict, norm_stats: dict) -> dict:
    """
    Evaluate model on a new domain before training starts (for FWT calculation).
    
    Args:
        model: Current model state
        domain_name: Name of the domain to evaluate on
        config: Configuration dictionary
        norm_stats: Normalisation statistics
        
    Returns:
        Dictionary containing evaluation metrics
    """
    print(f"    Pre-training evaluation on {domain_name}...")
    
    # Create test dataloader for evaluation
    val_loader = create_dataloader(domain_name, config, norm_stats, split='test')
    
    # Evaluate model
    device = next(model.parameters()).device
    metrics = evaluate_model_on_loader(model, val_loader, norm_stats, device)
    
    print(f"      Pre-training SSIM: {metrics['ssim']:.4f}")
    
    return metrics


def main(config_path: str):
    """Main training function."""
    
    # Load configuration
    config = load_config(config_path)
    device = get_device(config['hardware']['device'])
    
    # Set seeds for reproducibility
    seed = config.get('seed', 42)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import numpy as np
    import random
    np.random.seed(seed)
    random.seed(seed)
    print(f" Random seeds set to {seed}")
    
    print(f" Starting EWC Training")
    print(f"   Device: {device}")
    print(f"   EWC λ: {config['method']['ewc_lambda']}")
    print(f"   Domains: {len(config['data']['sequence'])}")
    
    # Wandb disabled for this script
    wandb_run = None
    print(" W&B logging disabled")
    
    # Load pretrained model and normalisation stats
    model, norm_stats = load_pretrained_model(config, device)
    
    # Calculate baselines using pretrained model before any CL training
    baselines = calculate_baselines(model, config, norm_stats, config['data']['sequence'])
    
    # Initialize EWC trainer
    trainer = EWCTrainer(
        model=model,
        config=config['training'],
        device=device,
        ewc_lambda=config['method']['ewc_lambda'],
        fisher_samples=config['method']['fisher_samples'],
        wandb_run=wandb_run
    )
    
    # Track all results with new structure
    all_results = {}
    completed_tasks = []
    
    # Track timing and additional metrics
    import time
    training_times = {}  # Task -> training time in seconds
    task_start_time = None
    
    # Main continual learning loop
    for task_idx, domain_name in enumerate(config['data']['sequence']):
        print(f"\n{'='*60}")
        print(f" Task {task_idx + 1}/{len(config['data']['sequence'])}: {domain_name}")
        print(f"{'='*60}")
        
        # Start timing this task
        task_start_time = time.perf_counter()
        
        # Evaluate on new task BEFORE training (for FWT calculation)
        pre_training_performance = evaluate_before_training(model, domain_name, config, norm_stats)
        
        # Prepare trainer for new task
        trainer.prepare_for_task(domain_name, None)  # train_loader not needed for EWC
        
        # Create data loaders for current task
        train_loader = create_dataloader(domain_name, config, norm_stats, split='train')
        val_loader = create_dataloader(domain_name, config, norm_stats, split='val')
        
        print(f" Data loaded:")
        print(f"   Training samples: {len(train_loader.dataset)}")
        print(f"   Validation samples: {len(val_loader.dataset)}")
        
        # Create optimizer for current task
        optimizer = trainer._create_optimizer(model.parameters())
        
        # Training loop for current task with early stopping
        best_val_loss = float('inf')
        early_stopping = EarlyStoppingTracker(
            patience=config['training'].get('early_stopping_patience', 5),
            min_delta=float(config['training'].get('early_stopping_min_delta', 1e-6)),
            mode='min'
        )
        
        for epoch in range(config['training']['epochs_per_task']):
            
            # Train one epoch
            train_loss = trainer.train_epoch(train_loader, optimizer, epoch)
            val_loss = trainer.validate_epoch(val_loader)
            
            print(f"Epoch {epoch+1:2d}/{config['training']['epochs_per_task']}: "
                  f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            
            # Save best model
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                checkpoint_path = os.path.join(
                    config['logging']['checkpoint_dir'], 
                    f"best_{domain_name}.pth"
                )
                trainer.save_checkpoint(domain_name, epoch, val_loss, checkpoint_path, is_best)
                trainer.best_val_losses[domain_name] = val_loss
            
            # Check early stopping
            if early_stopping(val_loss, epoch):
                print(f"   Early stopping at epoch {epoch+1}")
                break
            
            # W&B logging disabled
        
        # Get post-training performance on current task
        post_training_performance = evaluate_model_on_loader(model, val_loader, norm_stats, device)
        
        # Record training time for this task
        task_training_time = time.perf_counter() - task_start_time
        training_times[domain_name] = task_training_time
        print(f"️ Task {domain_name} training time: {task_training_time:.1f} seconds")
        
        # After task completion: compute Fisher matrix
        trainer.after_task_completion(domain_name, val_loader)
        completed_tasks.append(domain_name)
        
        # Evaluate on all domains (previous + current)
        if config['evaluation']['eval_previous_tasks']:
            eval_results = evaluate_all_previous_domains(trainer, config, norm_stats, completed_tasks[:-1], domain_name)
            
            # Store results in new structure
            all_results[domain_name] = {
                'pre_training_performance': pre_training_performance,
                'post_training_performance': post_training_performance,
                'final_train_loss': train_loss,
                'final_val_loss': val_loss,
                'best_val_loss': best_val_loss,
                'training_time_seconds': task_training_time,
                'evaluation_on_previous': {k: v for k, v in eval_results.items() if k != domain_name and k != 'aggregate'}
            }
            
            # W&B logging disabled
        
        print(f" Task {domain_name} completed. Best validation loss: {best_val_loss:.6f}")
        
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Print final EWC statistics
    trainer.print_ewc_statistics()
    
    # Save final results
    if config['evaluation']['save_results']:
        results_dir = config['evaluation']['results_dir']
        os.makedirs(results_dir, exist_ok=True)
        
        results_file = os.path.join(results_dir, f"ewc_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print(f" Results saved to: {results_file}")
    
    # Save final model with all Fisher matrices
    final_checkpoint_path = os.path.join(config['logging']['checkpoint_dir'], "final_ewc_model.pth")
    trainer.save_checkpoint("final", 0, 0.0, final_checkpoint_path)
    
    print(f"\n EWC training completed!")
    print(f"   Domains trained: {len(completed_tasks)}")
    print(f"   Fisher matrices computed: {len(trainer.fisher_matrices)}")
    print(f"   Final model saved: {final_checkpoint_path}")
    
    # W&B logging disabled
    
    # Calculate memory footprint
    model_params = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)  # MB
    
    # Fisher matrices are stored as dictionaries of tensors, so we need to sum over all tensors
    fisher_params = 0.0
    for task_fishers in trainer.fisher_matrices.values():
        if isinstance(task_fishers, dict):
            fisher_params += sum(f.numel() * f.element_size() for f in task_fishers.values()) / (1024 * 1024)
        else:
            # Handle case where it might be a single tensor
            fisher_params += task_fishers.numel() * task_fishers.element_size() / (1024 * 1024)
    
    total_memory_mb = model_params + fisher_params
    
    print(f"\n Memory Footprint:")
    print(f"   Model parameters: {model_params:.2f} MB")
    print(f"   Fisher matrices: {fisher_params:.2f} MB")
    print(f"   Total: {total_memory_mb:.2f} MB")
    
    # Calculate continual learning metrics
    metrics = calculate_continual_learning_metrics(all_results, completed_tasks, baselines, training_times)
    metrics['memory_footprint_mb'] = total_memory_mb
    
    print_continual_learning_summary(metrics, completed_tasks)
    
    # Save metrics to CSV
    if config['evaluation']['save_results']:
        results_dir = config['evaluation']['results_dir']
        os.makedirs(results_dir, exist_ok=True)
        save_metrics_to_csv(metrics, completed_tasks, results_dir, "ewc")
    
    return all_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train EWC Baseline")
    parser.add_argument(
        '--config', 
        type=str, 
        required=True,
        help="Path to EWC configuration file"
    )
    
    args = parser.parse_args()
    
    # Run training
    results = main(args.config)
    print("Training completed successfully!") 