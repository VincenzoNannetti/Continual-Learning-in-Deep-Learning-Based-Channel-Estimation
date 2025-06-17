"""
Analyse EWC Results - Generate metrics from saved training results
"""

import json
import os
import sys
from pathlib import Path
import yaml
import torch

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Import functions from train_ewc
from baseline_cl.scripts.train_ewc import (
    calculate_continual_learning_metrics, print_continual_learning_summary, 
    save_metrics_to_csv, load_config, get_device, load_pretrained_model,
    calculate_baselines
)

def find_latest_results_file(results_dir: str) -> str:
    """Find the most recent EWC results file."""
    results_files = []
    for file in os.listdir(results_dir):
        if file.startswith('ewc_results_') and file.endswith('.json'):
            results_files.append(os.path.join(results_dir, file))
    
    if not results_files:
        raise FileNotFoundError(f"No EWC results files found in {results_dir}")
    
    # Return the most recent file
    return max(results_files, key=os.path.getmtime)

def load_final_model_and_calculate_memory(config: dict) -> float:
    """Load the final model and calculate memory footprint."""
    try:
        device = get_device(config['hardware']['device'])
        
        # Load the final EWC model
        final_model_path = os.path.join(config['logging']['checkpoint_dir'], "final_ewc_model.pth")
        
        if not os.path.exists(final_model_path):
            print(f"️ Final model not found at {final_model_path}")
            return 0.0
        
        print(f" Loading final model from {final_model_path}")
        checkpoint = torch.load(final_model_path, map_location=device)
        
        # Calculate model memory
        if 'model_state_dict' in checkpoint:
            model_state = checkpoint['model_state_dict']
        else:
            model_state = checkpoint
        
        model_params = sum(p.numel() * 4 for p in model_state.values()) / (1024 * 1024)  # Assume float32
        
        # Calculate Fisher memory from checkpoint
        fisher_params = 0.0
        if 'fisher_matrices' in checkpoint:
            fisher_matrices = checkpoint['fisher_matrices']
            for task_fishers in fisher_matrices.values():
                if isinstance(task_fishers, dict):
                    fisher_params += sum(f.numel() * 4 for f in task_fishers.values()) / (1024 * 1024)
                else:
                    fisher_params += task_fishers.numel() * 4 / (1024 * 1024)
        
        total_memory_mb = model_params + fisher_params
        
        print(f" Memory Footprint:")
        print(f"   Model parameters: {model_params:.2f} MB")
        print(f"   Fisher matrices: {fisher_params:.2f} MB")
        print(f"   Total: {total_memory_mb:.2f} MB")
        
        return total_memory_mb
        
    except Exception as e:
        print(f"️ Could not calculate memory footprint: {e}")
        return 0.0

def main():
    """Main analysis function."""
    
    import argparse
    
    # Parse command line arguments properly
    parser = argparse.ArgumentParser(description="Analyse EWC Results")
    parser.add_argument(
        '--config', 
        type=str, 
        default="baseline_cl/configs/ewc_config.yaml",
        help="Path to EWC configuration file"
    )
    
    args = parser.parse_args()
    config_path = args.config
    
    print(f" Loading config from: {config_path}")
    config = load_config(config_path)
    
    # Find results file
    results_dir = config['evaluation']['results_dir']
    print(f" Looking for results in: {results_dir}")
    
    try:
        results_file = find_latest_results_file(results_dir)
        print(f" Found results file: {results_file}")
    except FileNotFoundError:
        print(f" No results files found in {results_dir}")
        print(f"   Make sure training completed and saved results")
        return
    
    # Load results
    with open(results_file, 'r') as f:
        all_results = json.load(f)
    
    print(f" Loaded results for {len(all_results)} domains")
    
    # Get domain sequence from config
    domains = config['data']['sequence']
    completed_tasks = [d for d in domains if d in all_results]
    
    print(f" Analysing {len(completed_tasks)} completed tasks:")
    for i, domain in enumerate(completed_tasks):
        print(f"   {i+1}. {domain}")
    
    # Extract baselines from pre_training_performance (first task performance = baseline)
    print(f"\n Extracting baselines from pre-training performance...")
    baselines = {}
    
    # Use the first task's pre-training performance as the universal baseline
    # (This represents the pretrained model's performance before any CL)
    first_domain = completed_tasks[0]
    if first_domain in all_results and 'pre_training_performance' in all_results[first_domain]:
        universal_baseline = all_results[first_domain]['pre_training_performance']
        
        # Apply this baseline to all domains
        for domain in domains:
            baselines[domain] = {
                'ssim': universal_baseline['ssim'],
                'nmse': universal_baseline['nmse'], 
                'psnr': universal_baseline['psnr']
            }
        
        print(f"   Using baseline from {first_domain}:")
        print(f"      NMSE: {universal_baseline['nmse']:.6f} (primary metric)")
        print(f"      SSIM: {universal_baseline['ssim']:.4f}")
        print(f"      PSNR: {universal_baseline['psnr']:.2f}")
    else:
        print(f"   ️ Could not extract baseline, using fallback values...")
        baselines = {domain: {'ssim': 0.5, 'nmse': 1.0, 'psnr': 20.0} for domain in domains}
    
    # Extract training times if available
    training_times = {}
    for domain, results in all_results.items():
        if 'training_time_seconds' in results:
            training_times[domain] = results['training_time_seconds']
    
    # Calculate metrics
    print(f"\n Calculating continual learning metrics...")
    metrics = calculate_continual_learning_metrics(all_results, completed_tasks, baselines, training_times)
    
    # Calculate memory footprint (optional - skip if model loading fails)
    print(f"\n Calculating memory footprint...")
    try:
        memory_footprint = load_final_model_and_calculate_memory(config)
        metrics['memory_footprint_mb'] = memory_footprint
    except Exception as e:
        print(f"   ️ Could not calculate memory footprint: {e}")
        print(f"   Using estimated value based on model size...")
        # Rough estimate: 4.7M parameters * 4 bytes + Fisher matrices
        estimated_memory = (4.718692 * 4) + (4.718692 * 4 * 9)  # Model + 9 Fisher matrices
        metrics['memory_footprint_mb'] = estimated_memory
        print(f"   Estimated memory footprint: {estimated_memory:.2f} MB")
    
    # Print summary
    print_continual_learning_summary(metrics, completed_tasks)
    
    # Save metrics
    if config['evaluation']['save_results']:
        print(f"\n Saving metrics to CSV...")
        os.makedirs(results_dir, exist_ok=True)
        save_metrics_to_csv(metrics, completed_tasks, results_dir, "ewc")
    
    print(f"\n Analysis completed successfully!")
    return metrics

if __name__ == '__main__':
    metrics = main() 