"""
Convenience script to run complete final evaluation and plotting pipeline
for both EWC and Experience Replay baseline methods.
"""

import os
import sys
import subprocess
from pathlib import Path
from datetime import datetime

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

def main():
    """Run the complete evaluation pipeline."""
    
    print(f"BASELINE CONTINUAL LEARNING FINAL EVALUATION PIPELINE")
    print(f"   Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Define paths
    project_root = Path(__file__).parent.parent.parent
    
    # Configuration files
    ewc_config = "baseline_cl/configs/ewc_config.yaml"
    replay_config = "baseline_cl/configs/experience_replay_config.yaml"
    
    # Checkpoint files  
    ewc_checkpoint = "baseline_cl/checkpoints/ewc/final_ewc_model.pth"
    replay_checkpoint = "baseline_cl/checkpoints/experience_replay/final_replay_buffer1000_model.pth"
    
    # Output directories
    results_dir = "baseline_cl/results/final_evaluation"
    plots_dir = "baseline_cl/results/final_evaluation/plots"
    
    # Create output directories
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    # Check if checkpoint files exist
    ewc_checkpoint_path = project_root / ewc_checkpoint
    replay_checkpoint_path = project_root / replay_checkpoint
    
    if not ewc_checkpoint_path.exists():
        print(f"EWC checkpoint not found: {ewc_checkpoint_path}")
        print("   Please run EWC training first!")
        return False
    
    if not replay_checkpoint_path.exists():
        print(f"Experience Replay checkpoint not found: {replay_checkpoint_path}")
        print("   Please run Experience Replay training first!")
        return False
    
    print(f"Found checkpoints:")
    print(f"   EWC: {ewc_checkpoint}")
    print(f"   Experience Replay: {replay_checkpoint}")
    
    # Change to project root directory
    os.chdir(project_root)
    
    success_count = 0
    total_steps = 3
    
    # Step 1: Evaluate EWC final performance
    cmd = f"python baseline_cl/scripts/evaluate_final_performance.py --method ewc --config {ewc_config} --checkpoint {ewc_checkpoint} --output_dir {results_dir}"
    if run_command(cmd, "Evaluating EWC Final Performance"):
        success_count += 1
    
    # Step 2: Evaluate Experience Replay final performance
    cmd = f"python baseline_cl/scripts/evaluate_final_performance.py --method experience_replay --config {replay_config} --checkpoint {replay_checkpoint} --output_dir {results_dir}"
    if run_command(cmd, "Evaluating Experience Replay Final Performance"):
        success_count += 1
    
    # Step 3: Create comparison plots
    cmd = f"python baseline_cl/scripts/plot_final_performance.py --results_dir {results_dir} --output_dir {plots_dir}"
    if run_command(cmd, "Creating Performance Comparison Plots"):
        success_count += 1
    
    # Summary
    print(f"\n{'='*60}")
    print(f"EVALUATION PIPELINE SUMMARY")
    print(f"{'='*60}")
    print(f"Completed steps: {success_count}/{total_steps}")
    
    if success_count == total_steps:
        print(f"All evaluations completed successfully!")
        print(f"Results saved to: {results_dir}")
        print(f"Plots saved to: {plots_dir}")
        
        # Copy continual learning metrics for comprehensive analysis
        try:
            import shutil
            results_path = project_root / results_dir
            
            # Copy EWC summary if available
            ewc_results_dir = project_root / "baseline_cl/results/ewc"
            if ewc_results_dir.exists():
                ewc_summaries = list(ewc_results_dir.glob("ewc_summary_*.csv"))
                if ewc_summaries:
                    latest_ewc = max(ewc_summaries, key=lambda x: x.stat().st_mtime)
                    shutil.copy2(latest_ewc, results_path / "ewc_summary.csv")
                    print(f"Copied EWC metrics: {latest_ewc.name}")
            
            # Copy Experience Replay summary if available
            er_results_dir = project_root / "baseline_cl/results/experience_replay"
            if er_results_dir.exists():
                er_summaries = list(er_results_dir.glob("Experience_Replay_summary_*.csv"))
                if er_summaries:
                    latest_er = max(er_summaries, key=lambda x: x.stat().st_mtime)
                    shutil.copy2(latest_er, results_path / "experience_replay_summary.csv")
                    print(f"Copied Experience Replay metrics: {latest_er.name}")
        
        except Exception as e:
            print(f"Warning: Could not copy continual learning metrics: {e}")
        
        # List generated files
        results_path = project_root / results_dir
        plots_path = project_root / plots_dir
        
        if results_path.exists():
            result_files = list(results_path.glob("*.csv")) + list(results_path.glob("*.json"))
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
        
        return True
    else:
        print(f"Some steps failed. Check the output above for details.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 