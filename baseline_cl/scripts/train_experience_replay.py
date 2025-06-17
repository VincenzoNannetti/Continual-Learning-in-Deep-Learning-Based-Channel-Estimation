"""
Training Script for Experience Replay Baseline

This script trains the UNet-SRCNN model using Experience Replay for continual learning
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

# Import our Experience Replay implementation
from baseline_cl.methods.experience_replay import ExperienceReplayTrainer

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Skipping W&B logging.")


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
    
    # Print device diagnostics
    print("\n🔍 Device Diagnostics:")
    print(f"   Device config: {device_config}")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA version: {torch.version.cuda}")
    print(f"   Using device: {device}")
    print(f"   Device type: {device.type}")
    
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
    
    # Print checkpoint structure
    print("\nCheckpoint structure:")
    if 'config' in checkpoint:
        print("Config keys:", checkpoint['config'].keys())
        if 'model' in checkpoint['config']:
            print("Model config:", checkpoint['config']['model'])
    
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
        print("\n📊 Normalisation stats loaded and reshaped for (C, H, W) data:")
        for key, value in norm_stats.items():
            print(f"   {key}: shape {value.shape}")
    else:
        raise ValueError("Normalisation statistics not found in checkpoint")
    
    # Extract model architecture from checkpoint config (not from Experience Replay config)
    if 'config' not in checkpoint:
        raise ValueError("Checkpoint does not contain config information")
    
    checkpoint_config = checkpoint['config']
    if 'model' not in checkpoint_config or 'params' not in checkpoint_config['model']:
        raise ValueError("Checkpoint config does not contain model parameters")
    
    # Use the model parameters from the checkpoint (these match the saved weights)
    model_params = checkpoint_config['model']['params']
    print(f"\n🏗️ Creating model with architecture from checkpoint:")
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
    
    # Verify model is on correct device
    print(f"\n🔍 Model Device Check:")
    print(f"   Model device: {next(model.parameters()).device}")
    print(f"   Expected device: {device}")
    
    print(f"✅ Pretrained model loaded successfully")
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    return model, norm_stats


def create_dataloader(domain_name: str, config: dict, norm_stats: dict, 
                     split: str = 'train') -> DataLoader:
    """
    Create a dataloader for a specific domain using the same approach as the main pipeline.
    
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
    print(f"📚 Loading dataset: {domain_name}")
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
        print(f"   ✅ Successfully loaded {len(dataset)} samples")
    except FileNotFoundError as e:
        print(f"   ❌ Dataset not found: {e}")
        print(f"   💡 Expected preprocessed file: {config['data']['preprocessed_dir']}/tester_data/{domain_name}_snr{snr}_thin_plate_spline.mat")
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
    
    # Apply normalisation using the same wrapper as the main pipeline
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


def evaluate_all_previous_domains(trainer: ExperienceReplayTrainer, config: dict, norm_stats: dict,
                                 completed_tasks: list) -> dict:
    """
    Evaluate the model on all previously learned domains.
    """
    if not completed_tasks:
        return {}
    
    print(f"\n🔍 Evaluating on {len(completed_tasks)} previous domains...")
    
    results = {}
    for domain_name in completed_tasks:
        val_loader = create_dataloader(domain_name, config, norm_stats, split='val')
        metrics = trainer.evaluate_domain(domain_name, val_loader, norm_stats)
        results[domain_name] = metrics
    
    # Calculate aggregate statistics
    if results:
        ssim_values = [r['ssim'] for r in results.values()]
        nmse_values = [r['nmse'] for r in results.values()]
        psnr_values = [r['psnr'] for r in results.values()]
        
        aggregate = {
            'mean_ssim': float(torch.tensor(ssim_values).mean()),
            'mean_nmse': float(torch.tensor(nmse_values).mean()),
            'mean_psnr': float(torch.tensor(psnr_values).mean()),
            'std_ssim': float(torch.tensor(ssim_values).std()),
            'std_nmse': float(torch.tensor(nmse_values).std()),
            'std_psnr': float(torch.tensor(psnr_values).std())
        }
        
        print(f"📊 Aggregate Results:")
        print(f"   SSIM: {aggregate['mean_ssim']:.4f} ± {aggregate['std_ssim']:.4f}")
        print(f"   NMSE: {aggregate['mean_nmse']:.8f} ± {aggregate['std_nmse']:.8f}")
        print(f"   PSNR: {aggregate['mean_psnr']:.2f} ± {aggregate['std_psnr']:.2f}")
        
        results['aggregate'] = aggregate
    
    return results


def calculate_continual_learning_metrics(all_results: dict, domains: list) -> dict:
    """
    Calculate continual learning metrics from results.
    
    Args:
        all_results: Dictionary of results from training
        domains: List of domain names in training order
    
    Returns:
        Dictionary containing BWT, FWT, ACC, and other metrics
    """
    # Build performance matrix R[i][j] = performance on task j after training task i
    num_tasks = len(domains)
    performance_matrix = {}
    
    # Extract performance matrix from results
    for task_idx, current_domain in enumerate(domains):
        if current_domain in all_results:
            performance_matrix[task_idx] = {}
            
            # Current task performance (diagonal)
            if 'final_val_loss' in all_results[current_domain]:
                # Convert loss to performance metric (lower loss = higher performance)
                current_performance = 1.0 / (1.0 + all_results[current_domain]['final_val_loss'])
                performance_matrix[task_idx][task_idx] = current_performance
            
            # Previous tasks performance
            if 'evaluation_on_previous' in all_results[current_domain]:
                prev_results = all_results[current_domain]['evaluation_on_previous']
                for prev_task_idx, prev_domain in enumerate(domains[:task_idx]):
                    if prev_domain in prev_results:
                        # Use SSIM as primary performance metric (higher = better)
                        prev_performance = prev_results[prev_domain].get('ssim', 0.0)
                        performance_matrix[task_idx][prev_task_idx] = prev_performance
    
    # Calculate metrics
    metrics = {}
    
    # 1. Final Average Performance (ACC)
    if num_tasks - 1 in performance_matrix:
        final_performances = []
        for task_idx in range(num_tasks):
            if task_idx in performance_matrix[num_tasks - 1]:
                final_performances.append(performance_matrix[num_tasks - 1][task_idx])
        
        if final_performances:
            metrics['ACC'] = sum(final_performances) / len(final_performances)
        else:
            metrics['ACC'] = 0.0
    
    # 2. Backward Transfer (BWT) - measures forgetting
    bwt_sum = 0.0
    bwt_count = 0
    
    for i in range(1, num_tasks):  # Start from task 1
        for j in range(i):  # Previous tasks
            if (num_tasks - 1) in performance_matrix and j in performance_matrix[num_tasks - 1]:
                if j in performance_matrix and j in performance_matrix[j]:
                    # R_T,j - R_j,j (final performance - original performance)
                    final_perf = performance_matrix[num_tasks - 1][j]
                    original_perf = performance_matrix[j][j]
                    bwt_sum += (final_perf - original_perf)
                    bwt_count += 1
    
    metrics['BWT'] = bwt_sum / bwt_count if bwt_count > 0 else 0.0
    
    # 3. Forward Transfer (FWT) - measures knowledge transfer
    fwt_sum = 0.0
    fwt_count = 0
    
    for i in range(1, num_tasks):
        if i in performance_matrix and i in performance_matrix[i]:
            # Compare with random initialization baseline (assume 0.5 for SSIM)
            baseline_performance = 0.5
            actual_performance = performance_matrix[i][i]
            fwt_sum += (actual_performance - baseline_performance)
            fwt_count += 1
    
    metrics['FWT'] = fwt_sum / fwt_count if fwt_count > 0 else 0.0
    
    # 4. Forgetting - average performance drop on previous tasks
    forgetting_sum = 0.0
    forgetting_count = 0
    
    for i in range(num_tasks):
        if i in performance_matrix and i in performance_matrix[i]:
            original_perf = performance_matrix[i][i]
            if (num_tasks - 1) in performance_matrix and i in performance_matrix[num_tasks - 1]:
                final_perf = performance_matrix[num_tasks - 1][i]
                if original_perf > final_perf:  # Only count actual forgetting
                    forgetting_sum += (original_perf - final_perf)
                    forgetting_count += 1
    
    metrics['Forgetting'] = forgetting_sum / forgetting_count if forgetting_count > 0 else 0.0
    
    # 5. Learning Accuracy - average accuracy on new tasks during learning
    learning_accs = []
    for i in range(num_tasks):
        if i in performance_matrix and i in performance_matrix[i]:
            learning_accs.append(performance_matrix[i][i])
    
    metrics['Learning_ACC'] = sum(learning_accs) / len(learning_accs) if learning_accs else 0.0
    
    # Store performance matrix for detailed analysis
    metrics['performance_matrix'] = performance_matrix
    
    return metrics


def print_continual_learning_summary(metrics: dict, domains: list):
    """Print a comprehensive summary of continual learning performance."""
    print(f"\n{'='*60}")
    print(f"📊 CONTINUAL LEARNING PERFORMANCE SUMMARY")
    print(f"{'='*60}")
    
    print(f"🎯 Final Average Performance (ACC): {metrics.get('ACC', 0.0):.4f}")
    print(f"📉 Backward Transfer (BWT):        {metrics.get('BWT', 0.0):+.4f}")
    print(f"   └─ {'Positive = Learning helps, Negative = Forgetting' if metrics.get('BWT', 0) < 0 else 'Positive = Learning helps!'}")
    
    print(f"📈 Forward Transfer (FWT):         {metrics.get('FWT', 0.0):+.4f}")
    print(f"   └─ {'Positive = Knowledge transfer benefit' if metrics.get('FWT', 0) > 0 else 'Negative = Interference'}")
    
    print(f"🧠 Forgetting:                    {metrics.get('Forgetting', 0.0):.4f}")
    print(f"   └─ {'Lower is better (less forgetting)' if metrics.get('Forgetting', 0) > 0 else 'No forgetting detected!'}")
    
    print(f"🎓 Learning Accuracy:              {metrics.get('Learning_ACC', 0.0):.4f}")
    
    # Performance matrix visualization
    if 'performance_matrix' in metrics:
        print(f"\n📋 Performance Matrix (rows=after task, cols=on task):")
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
    
    print(f"{'='*60}")


def save_metrics_to_csv(metrics: dict, domains: list, results_dir: str, method_name: str):
    """Save continual learning metrics to CSV files."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 1. Summary metrics CSV
    summary_data = {
        'Method': [method_name],
        'Final_Average_Performance': [metrics.get('ACC', 0.0)],
        'Backward_Transfer_BWT': [metrics.get('BWT', 0.0)],
        'Forward_Transfer_FWT': [metrics.get('FWT', 0.0)],
        'Forgetting': [metrics.get('Forgetting', 0.0)],
        'Learning_Accuracy': [metrics.get('Learning_ACC', 0.0)],
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
        
        print(f"📊 Metrics saved:")
        print(f"   Summary: {summary_file}")
        print(f"   Performance Matrix: {matrix_file}")
    
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
            print(f"⏹️ Early stopping triggered at epoch {epoch}")
            print(f"   Best score: {self.best_score:.6f} at epoch {self.best_epoch}")
            
        return self.early_stop


def main(config_path: str):
    """Main training function."""
    
    # Load configuration
    config = load_config(config_path)
    device = get_device(config['hardware']['device'])
    
    print(f"🚀 Starting Experience Replay Training")
    print(f"   Device: {device}")
    print(f"   Buffer size: {config['method']['buffer_size']}")
    print(f"   Replay ratio: {config['method']['replay_batch_ratio']}")
    print(f"   Domains: {len(config['data']['sequence'])}")
    
    # Initialize Weights & Biases
    wandb_run = None
    if WANDB_AVAILABLE and config['logging']['wandb']['enabled']:
        try:
            wandb_run = wandb.init(
                project=config['logging']['wandb']['project'],
                entity=config['logging']['wandb']['entity'],
                name=f"replay_buffer{config['method']['buffer_size']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                config=config,
                tags=config['logging']['wandb']['tags']
            )
            print(f"📊 W&B initialized: {wandb_run.name}")
        except Exception as e:
            print(f"⚠️ W&B initialization failed: {e}")
            wandb_run = None
    
    # Load pretrained model and normalisation stats
    model, norm_stats = load_pretrained_model(config, device)
    
    # Initialize Experience Replay trainer
    trainer = ExperienceReplayTrainer(
        model=model,
        config=config['training'],
        device=device,
        buffer_size=config['method']['buffer_size'],
        replay_batch_ratio=config['method']['replay_batch_ratio'],
        wandb_run=wandb_run
    )
    
    # Track all results
    all_results = {}
    completed_tasks = []
    
    # Main continual learning loop
    for task_idx, domain_name in enumerate(config['data']['sequence']):
        print(f"\n{'='*60}")
        print(f"🎯 Task {task_idx + 1}/{len(config['data']['sequence'])}: {domain_name}")
        print(f"{'='*60}")
        
        # Create data loaders for current task
        train_loader = create_dataloader(domain_name, config, norm_stats, split='train')
        val_loader = create_dataloader(domain_name, config, norm_stats, split='val')
        
        print(f"📚 Data loaded:")
        print(f"   Training samples: {len(train_loader.dataset)}")
        print(f"   Validation samples: {len(val_loader.dataset)}")
        
        # Prepare trainer for new task
        trainer.prepare_for_task(domain_name, train_loader)
        
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
            
            # Train one epoch with experience replay
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
            
            # Log to W&B
            if wandb_run:
                log_dict = {
                    f'task_{domain_name}_train_loss': train_loss,
                    f'task_{domain_name}_val_loss': val_loss,
                    f'task_{domain_name}_best_val_loss': best_val_loss,
                    'current_task_idx': task_idx,
                    'epoch': epoch + 1,
                    'is_best': is_best
                }
                
                # Add buffer statistics
                buffer_stats = trainer.get_buffer_statistics()
                if buffer_stats['buffer_initialized']:
                    log_dict.update({
                        'buffer_utilization': buffer_stats['utilization'],
                        'buffer_current_size': buffer_stats['current_size'],
                        'buffer_unique_tasks': buffer_stats['unique_tasks']
                    })
                
                wandb_run.log(log_dict)
        
        # After task completion: add samples to experience buffer
        trainer.after_task_completion(domain_name, val_loader)
        completed_tasks.append(domain_name)
        
        # Evaluate on all previous domains
        if config['evaluation']['eval_previous_tasks']:
            eval_results = evaluate_all_previous_domains(trainer, config, norm_stats, completed_tasks)
            all_results[domain_name] = {
                'final_train_loss': train_loss,
                'final_val_loss': val_loss,
                'best_val_loss': best_val_loss,
                'evaluation_on_previous': eval_results,
                'buffer_stats': trainer.get_buffer_statistics()
            }
            
            # Log evaluation results to W&B
            if wandb_run and eval_results:
                for eval_domain, metrics in eval_results.items():
                    if eval_domain != 'aggregate':
                        for metric_name, value in metrics.items():
                            wandb_run.log({f'eval_{eval_domain}_{metric_name}': value})
                
                # Log aggregate results
                if 'aggregate' in eval_results:
                    for metric_name, value in eval_results['aggregate'].items():
                        wandb_run.log({f'aggregate_{metric_name}': value})
        
        print(f"✅ Task {domain_name} completed. Best validation loss: {best_val_loss:.6f}")
        
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Print final replay statistics
    trainer.print_replay_statistics()
    
    # Save final results
    if config['evaluation']['save_results']:
        results_dir = config['evaluation']['results_dir']
        os.makedirs(results_dir, exist_ok=True)
        
        results_file = os.path.join(results_dir, f"replay_buffer{config['method']['buffer_size']}_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print(f"📁 Results saved to: {results_file}")
    
    # Save final model with experience buffer
    final_checkpoint_path = os.path.join(config['logging']['checkpoint_dir'], f"final_replay_buffer{config['method']['buffer_size']}_model.pth")
    trainer.save_checkpoint("final", 0, 0.0, final_checkpoint_path)
    
    print(f"\n🎉 Experience Replay training completed!")
    print(f"   Domains trained: {len(completed_tasks)}")
    print(f"   Buffer size: {config['method']['buffer_size']}")
    print(f"   Final buffer utilization: {trainer.get_buffer_statistics()['utilization']:.2%}")
    print(f"   Final model saved: {final_checkpoint_path}")
    
    # Finish W&B run
    if wandb_run:
        wandb_run.finish()
    
    # Calculate continual learning metrics
    metrics = calculate_continual_learning_metrics(all_results, completed_tasks)
    print_continual_learning_summary(metrics, completed_tasks)
    
    # Save metrics to CSV
    summary_file, matrix_file = save_metrics_to_csv(metrics, completed_tasks, results_dir, "Experience_Replay")
    
    return all_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Experience Replay Baseline")
    parser.add_argument(
        '--config', 
        type=str, 
        required=True,
        help="Path to Experience Replay configuration file"
    )
    
    args = parser.parse_args()
    
    # Run training
    results = main(args.config)
    print("Training completed successfully!") 