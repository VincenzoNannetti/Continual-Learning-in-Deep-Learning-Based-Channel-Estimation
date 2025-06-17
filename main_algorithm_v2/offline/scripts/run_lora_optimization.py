#!/usr/bin/env python3
"""
Example script to run LoRA continual learning optimization with Optuna.
This script demonstrates how to set up and run hyperparameter optimization.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_lora_optuna_optimization():
    """Run LoRA continual learning optimization with Optuna."""
    
    # Get current directory
    current_dir = Path(__file__).resolve().parent
    
    # Configuration paths
    sweep_config_path = current_dir / "configs" / "lora_optuna_sweep.yaml"
    
    # Optuna parameters - Systematic domain-by-domain testing
    n_trials = 45  # 9 domains × 5 ranks = 45 trials for complete systematic search
    study_name = "lora_domain_by_domain_systematic_v1"
    
    # W&B configuration for study logging
    wandb_study_project = "lora_continual_learning_optuna"
    wandb_study_entity = None  # Set your W&B entity here
    
    # Results directory
    results_dir = current_dir / "optuna_lora_results"
    
    # Domain IDs to evaluate (all 9 domains)
    domain_ids = ["0", "1", "2", "3", "4", "5", "6", "7", "8"]
    
    print("🚀 STARTING SINGLE-DOMAIN LORA RANK OPTIMIZATION")
    print("=" * 70)
    print(f"📋 Study Name: {study_name}")
    print(f"🔬 Strategy: Independent single-domain training for each rank")
    print(f"🔬 Number of Trials: {n_trials} (9 domains × 5 ranks)")
    print(f"📁 Sweep Config: {sweep_config_path}")
    print(f"🎯 Domains: {domain_ids}")
    print(f"📊 Ranks to test: [2, 4, 8, 12, 16]")
    print(f"📝 Each trial trains ONLY on target domain (not continual learning)")
    print(f"⚡ Alpha = Rank strategy enabled")
    print(f"💾 Results Directory: {results_dir}")
    print(f"📊 W&B Study Project: {wandb_study_project}")
    print("=" * 70)
    
    # Check that sweep config exists
    if not sweep_config_path.exists():
        print(f"❌ Error: Sweep config not found at {sweep_config_path}")
        print("Please create the sweep configuration file first.")
        return False
    
    # Build the command
    cmd = [
        sys.executable,
        str(current_dir / "optuna_lora_runner.py"),
        "--sweep_config", str(sweep_config_path),
        "--n_trials", str(n_trials),
        "--study_name", study_name,
        "--results_root_dir", str(results_dir),
        "--domain_ids"] + domain_ids
    
    # Add W&B configuration
    if wandb_study_project:
        cmd.extend(["--wandb_study_project", wandb_study_project])
    if wandb_study_entity:
        cmd.extend(["--wandb_study_entity", wandb_study_entity])
    
    # Add study tags
    cmd.extend(["--wandb_study_tags", "lora_continual_learning", "optuna_optimization", "multi_domain"])
    
    print(f"🔧 Command: {' '.join(cmd)}")
    print("\n" + "=" * 70)
    print("🏃‍♂️ STARTING OPTIMIZATION...")
    print("This may take several hours depending on your configuration.")
    print("You can monitor progress in W&B and check the results directory.")
    print("=" * 70)
    
    try:
        # Run the optimization
        result = subprocess.run(cmd, check=True, cwd=current_dir)
        
        print("\n" + "=" * 70)
        print("✅ OPTIMIZATION COMPLETED SUCCESSFULLY!")
        print(f"📊 Check your W&B project '{wandb_study_project}' for detailed results.")
        print(f"💾 Local results saved in: {results_dir}")
        print("=" * 70)
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Optimization failed with return code {e.returncode}")
        print("Check the error messages above for details.")
        return False
        
    except KeyboardInterrupt:
        print("\n⚠️ Optimization interrupted by user.")
        print("You can resume the study later using the same study name.")
        return False

def print_configuration_guide():
    """Print a guide for configuring the optimization."""
    print("\n📋 CONFIGURATION GUIDE")
    print("=" * 50)
    print("Before running optimization, please ensure:")
    print()
    print("1. ✅ Update configs/lora_optuna_sweep.yaml:")
    print("   - Set your W&B entity in 'wandb.entity'")
    print("   - Adjust hyperparameter ranges as needed")
    print("   - Verify the base config path")
    print()
    print("2. ✅ Ensure your training script (train.py) supports:")
    print("   - Multi-domain evaluation")
    print("   - W&B logging with domain-specific metrics")
    print("   - LoRA rank and alpha configuration")
    print()
    print("3. ✅ Check computational resources:")
    print("   - Each trial trains a full LoRA model")
    print("   - Consider reducing n_trials for initial testing")
    print("   - Monitor GPU memory usage")
    print()
    print("4. ✅ Set up W&B:")
    print("   - Login with 'wandb login'")
    print("   - Create the project if it doesn't exist")
    print("=" * 50)

def run_domain_chunk(domain_id: int):
    """Run optimization for a specific domain (5 trials)."""
    current_dir = Path(__file__).resolve().parent
    
    cmd = [
        sys.executable,
        str(current_dir / "optuna_lora_runner.py"),
        "--sweep_config", str(current_dir / "configs" / "lora_optuna_sweep.yaml"),
        "--n_trials", "5",
        "--target_domain", str(domain_id),
        "--study_name", f"lora_domain_{domain_id}_rank_search",
        "--results_root_dir", str(current_dir / "optuna_lora_results"),
        "--domain_ids", "0", "1", "2", "3", "4", "5", "6", "7", "8",
        "--wandb_study_project", "lora_domain_chunks",
        "--wandb_study_tags", "lora_continual_learning", "domain_chunk", f"domain_{domain_id}"
    ]
    
    print(f"\n🎯 RUNNING DOMAIN {domain_id} CHUNK (5 trials)")
    print("=" * 50)
    print(f"Study Name: lora_domain_{domain_id}_rank_search")
    print(f"Ranks to test: [2, 4, 8, 12, 16]")
    print("=" * 50)
    
    try:
        result = subprocess.run(cmd, check=True, cwd=current_dir)
        print(f"\n✅ Domain {domain_id} optimization completed!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Domain {domain_id} optimization failed: {e.returncode}")
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run LoRA continual learning optimization")
    parser.add_argument("--configure", action="store_true", 
                       help="Show configuration guide instead of running")
    parser.add_argument("--test_run", action="store_true",
                       help="Run with minimal trials for testing")
    parser.add_argument("--domain_chunk", type=int, choices=range(9), metavar="[0-8]",
                       help="Run optimization for specific domain only (0-8)")
    parser.add_argument("--all_domains_chunked", action="store_true",
                       help="Run all domains sequentially in chunks")
    
    args = parser.parse_args()
    
    if args.configure:
        print_configuration_guide()
    elif args.domain_chunk is not None:
        success = run_domain_chunk(args.domain_chunk)
        sys.exit(0 if success else 1)
    elif args.all_domains_chunked:
        print("🚀 RUNNING ALL DOMAINS IN CHUNKS")
        print("This will run 5 trials per domain and analyze after each domain")
        
        for domain_id in range(9):
            print(f"\n{'='*60}")
            print(f"STARTING DOMAIN {domain_id}")
            print(f"{'='*60}")
            
            success = run_domain_chunk(domain_id)
            if not success:
                print(f"❌ Failed on domain {domain_id}, stopping.")
                sys.exit(1)
                
            print(f"✅ Domain {domain_id} complete! Check results before continuing.")
            if domain_id < 8:  # Don't ask after the last domain
                input("Press Enter to continue to next domain, or Ctrl+C to stop...")
        
        print("\n🎉 ALL DOMAINS COMPLETED!")
        sys.exit(0)
    else:
        if args.test_run:
            print("🧪 RUNNING IN TEST MODE (full 45 trials)")
        
        success = run_lora_optuna_optimization()
        sys.exit(0 if success else 1) 