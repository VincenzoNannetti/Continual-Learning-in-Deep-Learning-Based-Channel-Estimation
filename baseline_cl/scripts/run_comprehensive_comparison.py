"""
Comprehensive comparison script to evaluate all continual learning methods
(EWC, Experience Replay, and LoRA) on the EXACT SAME DATASETS.

This ensures fair and valid comparison between methods.
"""

import os
import sys
import subprocess
from pathlib import Path
from datetime import datetime
import json

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    print(f"Command: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print("Success!")
        if result.stdout.strip():
            print("Output:")
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        if e.stdout:
            print("Stdout:")
            print(e.stdout)
        if e.stderr:
            print("Stderr:")
            print(e.stderr)
        return False

def check_dataset_consistency():
    """Check which datasets are available and ensure consistency."""
    print(f"\n CHECKING DATASET CONSISTENCY...")
    
    project_root = Path(__file__).parent.parent.parent
    
    # Check for baseline datasets (thin_plate_spline format)
    baseline_data_dir = project_root / "data/preprocessed/tester_data"
    baseline_files = list(baseline_data_dir.glob("*thin_plate_spline.mat")) if baseline_data_dir.exists() else []
    
    # Check for LoRA datasets (_cl format)  
    lora_data_dir = project_root / "data/raw/ray_tracing"
    lora_files = list(lora_data_dir.glob("*_cl.mat")) if lora_data_dir.exists() else []
    
    # Check for cached LoRA datasets
    lora_cached_dir = project_root / "data/preprocessed/tester_data"
    lora_cached_files = list(lora_cached_dir.glob("*_cl_*thin_plate_spline.mat")) if lora_cached_dir.exists() else []
    
    print(f"\n Dataset Analysis:")
    print(f"   Baseline datasets: {len(baseline_files)} files")
    print(f"   LoRA raw datasets: {len(lora_files)} files") 
    print(f"   LoRA cached datasets: {len(lora_cached_files)} files")
    
    # Print available datasets
    if baseline_files:
        print(f"\n Baseline Dataset Names:")
        for f in sorted(baseline_files)[:5]:  # Show first 5
            print(f"   - {f.name}")
        if len(baseline_files) > 5:
            print(f"   ... and {len(baseline_files) - 5} more")
    
    if lora_cached_files:
        print(f"\n LoRA Dataset Names:")
        for f in sorted(lora_cached_files)[:5]:  # Show first 5
            print(f"   - {f.name}")
        if len(lora_cached_files) > 5:
            print(f"   ... and {len(lora_cached_files) - 5} more")
    
    # Determine which dataset format to use for comparison
    if len(lora_cached_files) >= 9 and len(baseline_files) >= 9:
        print(f"\n Both dataset formats available - using LoRA cached format for consistency")
        return "lora_cached", lora_cached_files
    elif len(baseline_files) >= 9:
        print(f"\n️  Using baseline format - will need to retrain LoRA on this format")
        return "baseline", baseline_files
    else:
        print(f"\n Insufficient datasets found for comparison")
        return None, []

def create_unified_config():
    """Create a unified configuration for all methods using the same datasets."""
    dataset_format, available_files = check_dataset_consistency()
    
    if dataset_format is None:
        print("Cannot create unified config - insufficient datasets")
        return None
    
    # Extract domain sequence from available files
    domain_sequence = []
    
    if dataset_format == "lora_cached":
        # Use LoRA cached format for all methods
        domain_names = [
            "domain_high_snr_med_linear_cl",
            "domain_high_snr_slow_linear_cl", 
            "domain_high_snr_fast_linear_cl",
            "domain_med_snr_slow_linear_cl",
            "domain_med_snr_med_linear_cl",
            "domain_med_snr_fast_linear_cl",
            "domain_low_snr_slow_linear_cl",
            "domain_low_snr_med_linear_cl", 
            "domain_low_snr_fast_linear_cl"
        ]
        
        # Check which ones actually exist
        for domain in domain_names:
            matching_files = [f for f in available_files if domain in f.name]
            if matching_files:
                domain_sequence.append(domain)
    
    print(f"\n Unified Domain Sequence ({len(domain_sequence)} domains):")
    for i, domain in enumerate(domain_sequence):
        print(f"   {i}: {domain}")
    
    return {
        "dataset_format": dataset_format,
        "domain_sequence": domain_sequence,
        "num_domains": len(domain_sequence)
    }

def main():
    """Run comprehensive comparison of all continual learning methods."""
    
    print(f" COMPREHENSIVE CONTINUAL LEARNING METHOD COMPARISON")
    print(f"   Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   Methods: EWC, Experience Replay, LoRA")
    print(f"   Goal: Fair comparison on IDENTICAL datasets")
    
    # Define paths
    project_root = Path(__file__).parent.parent.parent
    
    # Create unified configuration
    unified_config = create_unified_config()
    if unified_config is None:
        print(" Cannot proceed without consistent datasets")
        return False
    
    # Output directories
    results_dir = "comprehensive_comparison_results"
    plots_dir = f"{results_dir}/plots"
    
    # Create output directories
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    # Save unified configuration
    config_file = f"{results_dir}/unified_config.json"
    with open(config_file, 'w') as f:
        json.dump(unified_config, f, indent=2)
    print(f"\n Unified configuration saved: {config_file}")
    
    # Change to project root directory
    os.chdir(project_root)
    
    success_count = 0
    total_steps = 4
    
    # Step 1: Evaluate EWC on unified datasets
    if unified_config["dataset_format"] == "lora_cached":
        # Need to modify EWC evaluation to use LoRA datasets
        print(f"\n️  EWC needs to be retrained on LoRA dataset format for fair comparison")
        print(f"   Current EWC models were trained on different datasets")
        print(f"   Skipping EWC evaluation for now...")
    else:
        cmd = f"python baseline_cl/scripts/evaluate_final_performance.py --method ewc --config baseline_cl/configs/ewc_config.yaml --checkpoint baseline_cl/checkpoints/ewc/final_ewc_model.pth --output_dir {results_dir}"
        if run_command(cmd, "Evaluating EWC on Unified Datasets"):
            success_count += 1
    
    # Step 2: Evaluate Experience Replay on unified datasets  
    if unified_config["dataset_format"] == "lora_cached":
        print(f"\n️  Experience Replay needs to be retrained on LoRA dataset format for fair comparison")
        print(f"   Current Experience Replay models were trained on different datasets")
        print(f"   Skipping Experience Replay evaluation for now...")
    else:
        cmd = f"python baseline_cl/scripts/evaluate_final_performance.py --method experience_replay --config baseline_cl/configs/experience_replay_config.yaml --checkpoint baseline_cl/checkpoints/experience_replay/final_replay_buffer1000_model.pth --output_dir {results_dir}"
        if run_command(cmd, "Evaluating Experience Replay on Unified Datasets"):
            success_count += 1
    
    # Step 3: Evaluate LoRA on unified datasets
    lora_checkpoint = "main_algorithm_v2/offline/checkpoints/lora/FINAL_WITH_REPLAY.pth"
    if os.path.exists(lora_checkpoint):
        cmd = f"python main_algorithm_v2/offline/evaluate.py --checkpoint {lora_checkpoint} --output_dir {results_dir}/lora_evaluation"
        if run_command(cmd, "Evaluating LoRA on Unified Datasets"):
            success_count += 1
            
            # Copy LoRA results to main results directory for plotting
            try:
                import shutil
                lora_results = Path(f"{results_dir}/lora_evaluation")
                if lora_results.exists():
                    # Copy CSV files for plotting
                    csv_files = list(lora_results.glob("*.csv"))
                    for csv_file in csv_files:
                        shutil.copy2(csv_file, results_dir)
                    print(f"   Copied {len(csv_files)} LoRA result files")
            except Exception as e:
                print(f"   Warning: Could not copy LoRA results: {e}")
    else:
        print(f" LoRA checkpoint not found: {lora_checkpoint}")
        print(f"   Please train LoRA model first")
    
    # Step 4: Create unified comparison plots
    cmd = f"python baseline_cl/scripts/plot_final_performance.py --results_dir {results_dir} --output_dir {plots_dir}"
    if run_command(cmd, "Creating Unified Comparison Plots"):
        success_count += 1
    
    # Summary
    print(f"\n{'='*60}")
    print(f"COMPREHENSIVE COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"Completed steps: {success_count}/{total_steps}")
    
    if success_count >= 2:  # At least LoRA + plots
        print(f" Comparison analysis available!")
        print(f" Results saved to: {results_dir}")
        print(f" Plots saved to: {plots_dir}")
        
        # Analysis
        print(f"\n DATASET CONSISTENCY ANALYSIS:")
        print(f"   Format used: {unified_config['dataset_format']}")
        print(f"   Domains: {unified_config['num_domains']}")
        
        if unified_config["dataset_format"] == "lora_cached":
            print(f"\n️  IMPORTANT FINDINGS:")
            print(f"    LoRA was trained/evaluated on different dataset format than baselines")
            print(f"    For fair comparison, ALL methods need to use the same datasets")
            print(f"    Recommendation: Retrain baseline methods on LoRA dataset format")
            print(f"    OR: Retrain LoRA on baseline dataset format")
        
        return True
    else:
        print(f"️  Partial results available")
        print(f"   For complete comparison, ensure all models are trained on same datasets")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 