"""
Simple script to analyze the offline continual learning algorithm using existing checkpoints.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_domain_analysis():
    """Run comprehensive domain analysis using existing checkpoint."""
    
    current_dir = Path(__file__).resolve().parent
    
    # Configuration parameters
    base_config = current_dir / "configs" / "config_final.yaml"  # Fixed path
    checkpoint_path = current_dir / "checkpoints" / "lora" / "FINAL_WITH_REPLAY.pth"
    domain_ids = ["0", "1", "2", "3", "4", "5", "6", "7", "8"]
    output_dir = current_dir / "analysis_results"  # Simplified path
    
    print(" STARTING OFFLINE CONTINUAL LEARNING ANALYSIS")
    print("=" * 60)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Base Config: {base_config}")
    print(f"Domains: {domain_ids}")
    print(f"Output: {output_dir}")
    print("=" * 60)
    
    # Check if checkpoint exists
    if not checkpoint_path.exists():
        print(f" Error: Checkpoint not found at {checkpoint_path}")
        print("Available files in checkpoints/lora/:")
        lora_dir = current_dir / "checkpoints" / "lora"
        if lora_dir.exists():
            for file in lora_dir.iterdir():
                print(f"  - {file.name}")
        return False
    
    # Check if config exists
    if not base_config.exists():
        print(f" Error: Config not found at {base_config}")
        print("Looking for alternative config files...")
        configs_dir = current_dir / "configs"
        if configs_dir.exists():
            for file in configs_dir.glob("*.yaml"):
                print(f"  Found: {file}")
                base_config = file  # Use the first found config
                break
        else:
            # Look for config files in current directory
            for file in current_dir.glob("*.yaml"):
                print(f"  Found: {file}")
                base_config = file
                break
    
    print(f" Using checkpoint: {checkpoint_path}")
    print(f" Using config: {base_config}")
    
    # Run evaluation using the existing evaluate.py script
    print("\n Running comprehensive evaluation...")
    
    cmd = [
        sys.executable, "evaluate.py",
        "--checkpoint", str(checkpoint_path),
        "--output_dir", str(output_dir),
        "--num_plot_samples", "3"
    ]
    
    print(f"Command: {' '.join(cmd)}")
    print("\n" + "=" * 60)
    print("‍️ STARTING EVALUATION...")
    print("This will evaluate the model on all domains and generate plots.")
    print("=" * 60)
    
    try:
        result = subprocess.run(cmd, cwd=current_dir, check=True)
        
        print("\n" + "=" * 60)
        print(" EVALUATION COMPLETED SUCCESSFULLY!")
        print(f" Check results in: {output_dir}")
        print("=" * 60)
        
        # Also create a domain comparison analysis
        create_domain_comparison_analysis(output_dir)
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n Evaluation failed with return code {e.returncode}")
        return False
        
    except KeyboardInterrupt:
        print("\n️ Evaluation interrupted by user.")
        return False

def create_domain_comparison_analysis(results_dir):
    """Create additional domain comparison plots from the evaluation results."""
    
    print("\n Creating domain comparison analysis...")
    
    # Look for CSV results file
    csv_files = list(Path(results_dir).glob("*evaluation*.csv"))
    
    if not csv_files:
        print("️ No CSV results found for additional analysis")
        return
    
    csv_file = csv_files[0]
    print(f" Using results from: {csv_file}")
    
    # Create a simple analysis script
    analysis_script = f"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Set up plotting
plt.style.use('default')
plt.rcParams.update({{
    'font.size': 12,
    'figure.figsize': (12, 8),
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
}})

# Load results
df = pd.read_csv('{csv_file}')
output_dir = Path('{results_dir}')

print(" Creating domain performance comparison plots...")

# Create domain comparison plots
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Offline Continual Learning: Domain Performance Analysis', fontsize=16, fontweight='bold')

# NMSE by domain
ax1 = axes[0, 0]
domains = df['Domain_ID'].astype(int).sort_values()
nmse_values = [df[df['Domain_ID'] == str(d)]['NMSE'].iloc[0] for d in domains]
ax1.bar(domains, nmse_values, color='lightcoral', alpha=0.7)
ax1.set_xlabel('Domain ID')
ax1.set_ylabel('NMSE (lower is better)')
ax1.set_title('(a) NMSE by Domain')
ax1.set_yscale('log')
ax1.grid(True, alpha=0.3)

# SSIM by domain
ax2 = axes[0, 1]
ssim_values = [df[df['Domain_ID'] == str(d)]['SSIM'].iloc[0] for d in domains]
ax2.bar(domains, ssim_values, color='lightblue', alpha=0.7)
ax2.set_xlabel('Domain ID')
ax2.set_ylabel('SSIM (higher is better)')
ax2.set_title('(b) SSIM by Domain')
ax2.grid(True, alpha=0.3)

# PSNR by domain
ax3 = axes[1, 0]
psnr_values = [df[df['Domain_ID'] == str(d)]['PSNR'].iloc[0] for d in domains]
ax3.bar(domains, psnr_values, color='lightgreen', alpha=0.7)
ax3.set_xlabel('Domain ID')
ax3.set_ylabel('PSNR (dB)')
ax3.set_title('(c) PSNR by Domain')
ax3.grid(True, alpha=0.3)

# Performance summary
ax4 = axes[1, 1]
# Normalize metrics for comparison
nmse_norm = (np.max(nmse_values) - nmse_values) / (np.max(nmse_values) - np.min(nmse_values))
ssim_norm = (ssim_values - np.min(ssim_values)) / (np.max(ssim_values) - np.min(ssim_values))
psnr_norm = (psnr_values - np.min(psnr_values)) / (np.max(psnr_values) - np.min(psnr_values))

x = np.arange(len(domains))
width = 0.25

ax4.bar(x - width, nmse_norm, width, label='NMSE (norm)', alpha=0.7)
ax4.bar(x, ssim_norm, width, label='SSIM (norm)', alpha=0.7)
ax4.bar(x + width, psnr_norm, width, label='PSNR (norm)', alpha=0.7)

ax4.set_xlabel('Domain ID')
ax4.set_ylabel('Normalized Performance')
ax4.set_title('(d) Normalized Performance Comparison')
ax4.set_xticks(x)
ax4.set_xticklabels(domains)
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'domain_performance_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

# Create performance statistics
print("\\n Domain Performance Statistics:")
print(f"NMSE - Mean: {{np.mean(nmse_values):.2e}}, Std: {{np.std(nmse_values):.2e}}")
print(f"SSIM - Mean: {{np.mean(ssim_values):.3f}}, Std: {{np.std(ssim_values):.3f}}")
print(f"PSNR - Mean: {{np.mean(psnr_values):.2f}}, Std: {{np.std(psnr_values):.2f}}")

print("\\n Best/Worst Performing Domains:")
best_ssim_idx = np.argmax(ssim_values)
worst_ssim_idx = np.argmin(ssim_values)
best_nmse_idx = np.argmin(nmse_values)
worst_nmse_idx = np.argmax(nmse_values)

print(f"Best SSIM: Domain {{domains.iloc[best_ssim_idx]}} ({{ssim_values[best_ssim_idx]:.3f}})")
print(f"Worst SSIM: Domain {{domains.iloc[worst_ssim_idx]}} ({{ssim_values[worst_ssim_idx]:.3f}})")
print(f"Best NMSE: Domain {{domains.iloc[best_nmse_idx]}} ({{nmse_values[best_nmse_idx]:.2e}})")
print(f"Worst NMSE: Domain {{domains.iloc[worst_nmse_idx]}} ({{nmse_values[worst_nmse_idx]:.2e}})")

print("\\n Domain analysis plots saved to: domain_performance_analysis.png")
"""
    
    # Write and execute the analysis script
    script_path = Path(results_dir) / "domain_analysis.py"
    with open(script_path, 'w') as f:
        f.write(analysis_script)
    
    try:
        subprocess.run([sys.executable, str(script_path)], cwd=results_dir, check=True)
        print(" Domain comparison analysis completed!")
    except Exception as e:
        print(f"️ Could not create additional analysis: {e}")

if __name__ == "__main__":
    success = run_domain_analysis()
    sys.exit(0 if success else 1) 