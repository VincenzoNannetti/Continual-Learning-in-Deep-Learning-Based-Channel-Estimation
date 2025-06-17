#!/usr/bin/env python3
"""
Multi-domain evaluation script for LoRA continual learning.
This script evaluates a trained model on ALL domains and logs metrics to W&B.
"""

import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import sys
import wandb
import json
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from typing import Dict, List

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from src.config import ExperimentConfig
from src.data import get_dataloaders
from src.utils import get_device, load_lora_model_for_evaluation

def denormalise(tensor: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """Denormalise a tensor using mean and std."""
    return tensor * (std + 1e-8) + mean

def calculate_metrics(outputs: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    """Calculate NMSE, PSNR, and SSIM for a batch."""
    nmse = np.sum((outputs - targets) ** 2) / np.sum(targets ** 2)
    
    # Reshape for image metrics if necessary
    if outputs.ndim == 4:  # Batch, Channel, H, W
        outputs = outputs.transpose(0, 2, 3, 1)  # Batch, H, W, Channel
        targets = targets.transpose(0, 2, 3, 1)
    
    # Calculate PSNR and SSIM sample by sample and average
    psnr_vals = [psnr(t, o, data_range=t.max() - t.min()) for t, o in zip(targets, outputs)]
    # For SSIM, use first channel if multi-channel
    if targets.shape[-1] > 1:
        ssim_vals = [ssim(t[..., 0], o[..., 0], data_range=t[..., 0].max() - t[..., 0].min()) 
                    for t, o in zip(targets, outputs)]
    else:
        ssim_vals = [ssim(t[..., 0], o[..., 0], data_range=t[..., 0].max() - t[..., 0].min()) 
                    for t, o in zip(targets, outputs)]
    
    return {
        'nmse': nmse,
        'psnr': np.mean(psnr_vals),
        'ssim': np.mean(ssim_vals)
    }

def evaluate_single_domain(model, task_id: str, config: ExperimentConfig, device: torch.device) -> Dict[str, float]:
    """Evaluate the model on a single domain."""
    print(f"Evaluating domain {task_id}...")
    
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
        for inputs, targets in tqdm(val_loader, desc=f"Domain {task_id}", leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            
            # Denormalise for metric calculation
            mean_i = torch.tensor(config.data.norm_stats.mean_inputs, device=device).view(1, -1, 1, 1)
            std_i = torch.tensor(config.data.norm_stats.std_inputs, device=device).view(1, -1, 1, 1)
            mean_t = torch.tensor(config.data.norm_stats.mean_targets, device=device).view(1, -1, 1, 1)
            std_t = torch.tensor(config.data.norm_stats.std_targets, device=device).view(1, -1, 1, 1)
            
            outputs_denorm = denormalise(outputs, mean_t, std_t)
            targets_denorm = denormalise(targets, mean_t, std_t)
            
            all_outputs.append(outputs_denorm.cpu().numpy())
            all_targets.append(targets_denorm.cpu().numpy())
    
    # Concatenate all batches
    all_outputs = np.concatenate(all_outputs, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    # Calculate metrics
    metrics = calculate_metrics(all_outputs, all_targets)
    
    print(f"Domain {task_id} results: SSIM={metrics['ssim']:.4f}, NMSE={metrics['nmse']:.6f}, PSNR={metrics['psnr']:.2f}")
    
    return metrics

def evaluate_all_domains(checkpoint_path: str, domain_ids: List[str], 
                         wandb_run=None, log_to_wandb: bool = True, 
                         save_to_file: str = None) -> Dict[str, Dict[str, float]]:
    """
    Evaluate the model on all specified domains.
    
    Args:
        checkpoint_path: Path to the trained model checkpoint
        domain_ids: List of domain IDs to evaluate
        wandb_run: W&B run object for logging (optional)
        log_to_wandb: Whether to log results to W&B
        save_to_file: Path to save results JSON file (optional)
    
    Returns:
        Dictionary mapping domain_id -> metrics
    """
    device = get_device('auto')
    print(f"Using device: {device}")
    
    # Load the trained model
    try:
        model = load_lora_model_for_evaluation(checkpoint_path, device)
        config = model.config
        print(f"✅ Model loaded successfully from {checkpoint_path}")
        print(f"📋 Model configured for domains: {config.data.sequence}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        raise
    
    # Evaluate each domain
    all_domain_results = {}
    aggregate_metrics = {'ssim': [], 'nmse': [], 'psnr': []}
    
    print(f"\n🔍 Evaluating {len(domain_ids)} domains...")
    
    for domain_id in domain_ids:
        if domain_id not in [str(d) for d in config.data.sequence]:
            print(f"⚠️  Warning: Domain {domain_id} not in model's training sequence")
            continue
            
        try:
            domain_metrics = evaluate_single_domain(model, domain_id, config, device)
            all_domain_results[domain_id] = domain_metrics
            
            # Collect for aggregate statistics
            for metric_name, value in domain_metrics.items():
                if metric_name in aggregate_metrics:
                    aggregate_metrics[metric_name].append(value)
            
            # Log to W&B if enabled
            if log_to_wandb and wandb_run:
                log_dict = {}
                for metric_name, value in domain_metrics.items():
                    log_dict[f"eval_domain_{domain_id}_{metric_name}"] = value
                    # Also log in summary for Optuna wrapper to find
                    wandb_run.summary[f"eval_domain_{domain_id}_{metric_name}"] = value
                wandb_run.log(log_dict)
                
        except Exception as e:
            print(f"❌ Error evaluating domain {domain_id}: {e}")
            continue
    
    # Calculate and log aggregate statistics
    if all_domain_results:
        print(f"\n📊 AGGREGATE RESULTS ACROSS {len(all_domain_results)} DOMAINS:")
        
        aggregate_stats = {}
        for metric_name, values in aggregate_metrics.items():
            if values:
                mean_val = np.mean(values)
                std_val = np.std(values)
                min_val = np.min(values)
                max_val = np.max(values)
                
                aggregate_stats[metric_name] = {
                    'mean': mean_val,
                    'std': std_val,
                    'min': min_val,
                    'max': max_val
                }
                
                print(f"  {metric_name.upper()}: {mean_val:.4f} ± {std_val:.4f} (min: {min_val:.4f}, max: {max_val:.4f})")
                
                # Log aggregate stats to W&B
                if log_to_wandb and wandb_run:
                    wandb_run.log({
                        f"aggregate_{metric_name}_mean": mean_val,
                        f"aggregate_{metric_name}_std": std_val,
                        f"aggregate_{metric_name}_min": min_val,
                        f"aggregate_{metric_name}_max": max_val,
                    })
        
        # Calculate domain consistency metrics
        if len(all_domain_results) > 1:
            ssim_values = [result['ssim'] for result in all_domain_results.values()]
            nmse_values = [result['nmse'] for result in all_domain_results.values()]
            
            # Coefficient of variation (std/mean) as consistency measure
            ssim_consistency = np.std(ssim_values) / np.mean(ssim_values) if np.mean(ssim_values) > 0 else 1.0
            nmse_consistency = np.std(nmse_values) / np.mean(nmse_values) if np.mean(nmse_values) > 0 else 1.0
            
            print(f"  CONSISTENCY (CoV): SSIM={ssim_consistency:.4f}, NMSE={nmse_consistency:.4f}")
            
            if log_to_wandb and wandb_run:
                wandb_run.log({
                    'domain_consistency_ssim': ssim_consistency,
                    'domain_consistency_nmse': nmse_consistency,
                    'num_domains_evaluated': len(all_domain_results)
                })
                
                # Also add to summary for Optuna wrapper
                wandb_run.summary['domain_consistency_ssim'] = ssim_consistency
                wandb_run.summary['domain_consistency_nmse'] = nmse_consistency
                wandb_run.summary['num_domains_evaluated'] = len(all_domain_results)
    
    # Ensure all domain results are in summary for Optuna wrapper
    if log_to_wandb and wandb_run:
        for domain_id, metrics in all_domain_results.items():
            for metric_name, value in metrics.items():
                wandb_run.summary[f"eval_domain_{domain_id}_{metric_name}"] = value
    
    # Save results to file if requested
    if save_to_file:
        results_data = {
            'domain_results': all_domain_results,
            'aggregate_stats': aggregate_stats if 'aggregate_stats' in locals() else {},
            'checkpoint_path': checkpoint_path,
            'domain_ids_evaluated': list(all_domain_results.keys()),
            'num_domains': len(all_domain_results)
        }
        
        try:
            with open(save_to_file, 'w') as f:
                json.dump(results_data, f, indent=2)
            print(f"📁 Results saved to: {save_to_file}")
        except Exception as e:
            print(f"⚠️  Warning: Could not save results to file: {e}")
    
    return all_domain_results

def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Multi-domain evaluation for LoRA continual learning")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained model checkpoint")
    parser.add_argument("--domains", type=str, nargs="+", 
                       default=["0", "1", "2", "3", "4", "5", "6", "7", "8"],
                       help="Domain IDs to evaluate")
    parser.add_argument("--no_wandb", action="store_true", help="Disable W&B logging")
    parser.add_argument("--wandb_project", type=str, default=None, help="W&B project name")
    parser.add_argument("--wandb_entity", type=str, default=None, help="W&B entity name")
    parser.add_argument("--wandb_run_id", type=str, default=None, help="W&B run ID to log to")
    parser.add_argument("--save_results", type=str, default=None, help="Path to save results JSON file")
    
    args = parser.parse_args()
    
    print("🔬 MULTI-DOMAIN LORA EVALUATION")
    print("=" * 60)
    print(f"📁 Checkpoint: {args.checkpoint}")
    print(f"🎯 Domains: {args.domains}")
    print(f"📊 W&B Logging: {'Disabled' if args.no_wandb else 'Enabled'}")
    if args.save_results:
        print(f"💾 Save Results: {args.save_results}")
    print("=" * 60)
    
    # Initialize W&B if enabled
    wandb_run = None
    if not args.no_wandb:
        try:
            if args.wandb_run_id:
                # Resume existing run (for Optuna integration)
                wandb_run = wandb.init(
                    project=args.wandb_project,
                    entity=args.wandb_entity,
                    id=args.wandb_run_id,
                    resume="allow"
                )
            else:
                # Create new run
                wandb_run = wandb.init(
                    project=args.wandb_project or "lora_multi_domain_eval",
                    entity=args.wandb_entity,
                    name=f"eval_{os.path.basename(args.checkpoint)}",
                    tags=["multi_domain_eval", "lora"],
                    config={
                        "checkpoint_path": args.checkpoint,
                        "domains": args.domains,
                        "evaluation_type": "multi_domain"
                    }
                )
            print(f"✅ W&B initialized: {wandb_run.name}")
        except Exception as e:
            print(f"⚠️  W&B initialization failed: {e}")
            wandb_run = None
    
    # Run multi-domain evaluation
    try:
        results = evaluate_all_domains(
            checkpoint_path=args.checkpoint,
            domain_ids=args.domains,
            wandb_run=wandb_run,
            log_to_wandb=(wandb_run is not None),
            save_to_file=args.save_results
        )
        
        print(f"\n✅ Evaluation complete! Results for {len(results)} domains.")
        
        if wandb_run:
            # Ensure summary is synced before finishing
            print(f"📊 Syncing {len(wandb_run.summary)} metrics to W&B summary...")
            
            # Force summary update and sync before finishing
            wandb_run.summary.update({
                f"eval_domain_{d}_{k}": v
                for d, m in results.items()
                for k, v in m.items()
            })
            wandb_run.save()
            import time
            time.sleep(3)  # Give W&B backend time to sync
            
            wandb_run.finish()
            
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        if wandb_run:
            wandb_run.finish()
        raise
    
    return results

if __name__ == "__main__":
    main() 