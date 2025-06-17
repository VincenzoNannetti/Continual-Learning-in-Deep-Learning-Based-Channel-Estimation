"""
Convenience script to run complete final evaluation and plotting pipeline
for LoRA-based continual learning method.
"""

import os
import sys
import subprocess
from pathlib import Path
from datetime import datetime

def run_command(command, description, cwd=None):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    print(f"Command: {command}")
    
    try:
        # Set environment variable to handle Unicode output
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True, cwd=cwd, env=env)
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

def main():
    """Run the complete LoRA evaluation pipeline."""
    
    print(f"LORA CONTINUAL LEARNING FINAL EVALUATION PIPELINE")
    print(f"   Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Get current working directory and determine if we're in the right place
    current_dir = Path.cwd()
    
    # If we're running from main_algorithm_v2/offline, adjust paths accordingly
    if current_dir.name == "offline" and current_dir.parent.name == "main_algorithm_v2":
        # We're in the right directory, use relative paths
        project_root = current_dir.parent.parent  # Go up to Project root
        
        # Checkpoint files - relative to current directory
        lora_checkpoint = "checkpoints/lora/FINAL_WITH_REPLAY.pth"
        best_checkpoint = "checkpoints/lora/BEST.pth"
        
        # Output directories - relative to current directory
        results_dir = "results/final_evaluation"
        plots_dir = "results/final_evaluation/plots"
        
        # CSV results from training - relative to current directory
        csv_results_from_training = "checkpoints/lora/csv_results"
        
        # Evaluation script path - relative to current directory since we're already in offline
        evaluate_script = "evaluate.py"
        
        # Keep track of the offline directory for running evaluate.py
        offline_dir = current_dir
        
    else:
        # We're running from project root, use original paths
        project_root = current_dir
        
        lora_checkpoint = "main_algorithm_v2/offline/checkpoints/lora/FINAL_WITH_REPLAY.pth"
        best_checkpoint = "main_algorithm_v2/offline/checkpoints/lora/BEST.pth"
        results_dir = "main_algorithm_v2/offline/results/final_evaluation"
        plots_dir = "main_algorithm_v2/offline/results/final_evaluation/plots"
        csv_results_from_training = "main_algorithm_v2/offline/checkpoints/lora/csv_results"
        evaluate_script = "main_algorithm_v2/offline/evaluate.py"
        
        # Keep track of the offline directory for running evaluate.py
        offline_dir = project_root / "main_algorithm_v2" / "offline"
    
    # Create output directories
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    # Check if checkpoint files exist
    lora_checkpoint_path = Path(lora_checkpoint)
    best_checkpoint_path = Path(best_checkpoint)
    
    # Prefer final checkpoint, fallback to best
    checkpoint_to_use = lora_checkpoint
    checkpoint_path_to_use = lora_checkpoint_path
    
    if not lora_checkpoint_path.exists():
        if best_checkpoint_path.exists():
            print(f"Final checkpoint not found, using best checkpoint instead")
            checkpoint_to_use = best_checkpoint
            checkpoint_path_to_use = best_checkpoint_path
        else:
            print(f"LoRA checkpoint not found: {lora_checkpoint_path}")
            print(f"Best checkpoint also not found: {best_checkpoint_path}")
            print("   Please run LoRA training first!")
            return False
    
    print(f"Found checkpoint:")
    print(f"   LoRA: {checkpoint_to_use}")
    
    success_count = 0
    total_steps = 4
    
    # Step 1: Run comprehensive LoRA evaluation
    cmd = f"python {evaluate_script} --checkpoint {checkpoint_to_use} --output_dir {results_dir} --num_plot_samples 5"
    if run_command(cmd, "Running LoRA Comprehensive Evaluation", cwd=offline_dir):
        success_count += 1
    
    # Step 2: Check if CSV results already exist from training
    csv_results_from_training_path = Path(csv_results_from_training)
    
    if csv_results_from_training_path.exists():
        print(f"\n{'='*60}")
        print(f"Copying CSV Results from Training")
        print(f"{'='*60}")
        try:
            import shutil
            
            # Copy CSV files from training results to evaluation results
            csv_files = list(csv_results_from_training_path.glob("*.csv"))
            if csv_files:
                evaluation_csv_dir = Path(results_dir) / "csv_results"
                evaluation_csv_dir.mkdir(exist_ok=True)
                
                for csv_file in csv_files:
                    shutil.copy2(csv_file, evaluation_csv_dir / csv_file.name)
                    print(f"Copied: {csv_file.name}")
                
                print(f"Successfully copied {len(csv_files)} CSV files for plotting")
                success_count += 1
            else:
                print("No CSV files found in training results")
        except Exception as e:
            print(f"Error copying CSV files: {e}")
    else:
        print(f"No training CSV results found at: {csv_results_from_training_path}")
    
    # Step 3: Create LoRA-specific plots using baseline plotting scripts (if compatible)
    evaluation_csv_dir = Path(results_dir) / "csv_results"
    if evaluation_csv_dir.exists() and list(evaluation_csv_dir.glob("*.csv")):
        # Run final performance plotting without method filter
        cmd = f"python baseline_cl/scripts/plot_final_performance.py --results_dir {evaluation_csv_dir} --output_dir {plots_dir}"
        if run_command(cmd, "Creating LoRA Performance Plots using Baseline Scripts", cwd=project_root):
            success_count += 1
        
        # Try EWC-specific plotting as well
        cmd = f"python baseline_cl/scripts/plot_ewc_results.py --results_dir {evaluation_csv_dir} --output_dir {plots_dir}"
        if run_command(cmd, "Creating LoRA Continual Learning Analysis Plots", cwd=project_root):
            success_count += 1
    else:
        print("No CSV files available for plotting")
        
    # Summary
    print(f"\n{'='*60}")
    print(f"LORA EVALUATION PIPELINE SUMMARY")
    print(f"{'='*60}")
    print(f"Completed steps: {success_count}/{total_steps}")
    
    if success_count >= 2:  # At least evaluation + CSV copying
        print(f"LoRA evaluation completed successfully!")
        print(f"Results saved to: {results_dir}")
        print(f"Plots saved to: {plots_dir}")
        
        # List generated files
        results_path = Path(results_dir)
        plots_path = Path(plots_dir)
        
        if results_path.exists():
            result_files = list(results_path.glob("*.csv")) + list(results_path.glob("*.json"))
            csv_dir = results_path / "csv_results"
            if csv_dir.exists():
                result_files.extend(list(csv_dir.glob("*.csv")))
            
            if result_files:
                print(f"\nGenerated result files:")
                for file in sorted(result_files):
                    print(f"   - {file.name}")
        
        if plots_path.exists():
            plot_files = list(plots_path.glob("*.png")) + list(plots_path.glob("*.pdf")) + list(plots_path.glob("*.svg"))
            if plot_files:
                print(f"\nGenerated plot files:")
                for file in sorted(plot_files):
                    print(f"   - {file.name}")
        
        # Check for individual sample plots
        sample_plots_dir = results_path / "plots"
        if sample_plots_dir.exists():
            sample_plot_files = list(sample_plots_dir.rglob("*.png"))
            if sample_plot_files:
                print(f"\nGenerated sample plots: {len(sample_plot_files)} files in {sample_plots_dir}")
        
        print(f"\nCSV COMPATIBILITY CHECK:")
        evaluation_csv_dir = results_path / "csv_results"
        if evaluation_csv_dir.exists():
            csv_files = list(evaluation_csv_dir.glob("*.csv"))
            print(f"   CSV files ready for baseline plotting scripts: {len(csv_files)} files")
            print(f"   Location: {evaluation_csv_dir}")
            print(f"   Compatible with baseline plot_final_performance.py and plot_ewc_results.py")
        
        return True
    else:
        print(f"Some steps failed. Check the output above for details.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 