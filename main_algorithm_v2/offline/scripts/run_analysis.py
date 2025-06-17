#!/usr/bin/env python3
"""
Simple script to analyze your offline continual learning model with the existing checkpoint.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_analysis():
    """Run analysis using the existing FINAL_WITH_REPLAY checkpoint."""
    
    current_dir = Path(__file__).resolve().parent
    
    # Configuration
    checkpoint_path = current_dir / "checkpoints" / "lora" / "FINAL_WITH_REPLAY.pth"
    output_dir = current_dir / "analysis_results"
    
    print("🚀 OFFLINE CONTINUAL LEARNING ANALYSIS")
    print("=" * 50)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Output: {output_dir}")
    print("=" * 50)
    
    # Check if checkpoint exists
    if not checkpoint_path.exists():
        print(f"❌ Error: Checkpoint not found at {checkpoint_path}")
        print("Available files in checkpoints/lora/:")
        lora_dir = current_dir / "checkpoints" / "lora"
        if lora_dir.exists():
            for file in lora_dir.iterdir():
                if file.is_file():
                    print(f"  - {file.name}")
        return False
    
    print(f"✅ Found checkpoint: {checkpoint_path.name}")
    
    # Run evaluation
    cmd = [
        sys.executable, "evaluate.py",
        "--checkpoint", str(checkpoint_path),
        "--output_dir", str(output_dir),
        "--num_plot_samples", "5"
    ]
    
    print(f"\n🔧 Running: {' '.join([cmd[0]] + [cmd[1]] + [cmd[2]] + ['<checkpoint>'] + cmd[4:])}")
    print("\n" + "=" * 50)
    print("🏃‍♂️ STARTING EVALUATION...")
    print("This will evaluate across all 9 domains and generate plots.")
    print("=" * 50)
    
    try:
        result = subprocess.run(cmd, cwd=current_dir, check=True)
        
        print("\n" + "=" * 50)
        print("✅ EVALUATION COMPLETED!")
        print(f"📊 Results saved to: {output_dir}")
        print("📈 Check the comprehensive report and plots!")
        print("=" * 50)
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Evaluation failed with return code {e.returncode}")
        return False
        
    except KeyboardInterrupt:
        print("\n⚠️ Evaluation interrupted by user.")
        return False

if __name__ == "__main__":
    success = run_analysis()
    sys.exit(0 if success else 1) 