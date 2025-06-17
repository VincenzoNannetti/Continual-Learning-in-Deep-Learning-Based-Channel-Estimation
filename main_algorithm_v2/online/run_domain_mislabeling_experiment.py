"""
Dynamic Domain Shift Evaluation for Online Continual Learning

This script tests online adaptation by creating temporal domain shifts:
1. Baseline: Show performance vs samples for all domains
2. Domain shift: Focus on one domain with temporary shifts to test adaptation

Example usage:
    # Run baseline evaluation
    python run_domain_mislabeling_experiment.py --baseline_only --num_samples 1800
    
    # Run domain shift evaluation (domain 3 -> 8 -> 3)
    python run_domain_mislabeling_experiment.py --shift_only --primary_domain 3 --shift_domain 8 --shift_start 200 --shift_end 800 --num_samples 1200
    
    # Run both
    python run_domain_mislabeling_experiment.py --num_samples 1200
"""

import argparse
import yaml
import subprocess
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
from datetime import datetime

# Domain mapping from numeric IDs to letters
DOMAIN_ID_TO_LETTER = {
    0: 'A',
    1: 'C',
    2: 'B',
    3: 'I',
    4: 'G',
    5: 'H',
    6: 'F',
    7: 'D',
    8: 'E'
}

# Reverse mapping for convenience
LETTER_TO_DOMAIN_ID = {v: k for k, v in DOMAIN_ID_TO_LETTER.items()}

def create_single_domain_config(base_config_path, domain_id=3, trigger_type='hybrid'):
    """
    Create a config for single domain streaming with maximum available data.
    
    Args:
        base_config_path: Path to base online config
        domain_id: Domain to stream (default: 3)
        trigger_type: Type of trigger to use
    
    Returns:
        Path to single domain config file
    """
    # Load base config
    with open(base_config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set up single domain experiment
    config['experiment_name'] = f"single_domain_{domain_id}_{trigger_type}"
    config['online_training']['trigger']['type'] = trigger_type
    
    # Configure trigger parameters
    if trigger_type == 'hybrid':
        config['online_training']['trigger']['max_time'] = 30.0
        config['online_training']['trigger']['max_samples'] = 50
        config['online_training']['trigger']['time_weight'] = 1.0
        config['online_training']['trigger']['volume_weight'] = 1.0
    elif trigger_type == 'volume':
        config['online_training']['trigger']['volume_interval'] = 50
    elif trigger_type == 'time':
        config['online_training']['trigger']['time_interval'] = 20.0
    elif trigger_type == 'drift':
        config['online_training']['trigger']['alpha'] = 0.1
        config['online_training']['trigger']['kappa'] = 1.5
        config['online_training']['trigger']['warmup_samples'] = 30
    
    # Use single domain selection
    config['online_evaluation']['domain_selection'] = 'single_domain'
    config['online_evaluation']['target_domain'] = domain_id
    config['online_evaluation']['verbose'] = True
    
    # No domain remapping for single domain
    if 'domain_remapping' not in config['raw_data']:
        config['raw_data']['domain_remapping'] = {}
    
    # Save single domain config
    config_dir = Path(f"main_algorithm_v2/online/config/domain_shift_experiments")
    config_dir.mkdir(parents=True, exist_ok=True)
    
    config_path = config_dir / f"single_domain_{domain_id}_{trigger_type}_config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    return config_path

def create_baseline_config(base_config_path, trigger_type='hybrid', block_size=200):
    """
    Create a baseline config that cycles through all domains sequentially.
    
    Args:
        base_config_path: Path to base online config
        trigger_type: Type of trigger to use
    
    Returns:
        Path to baseline config file
    """
    # Load base config
    with open(base_config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set up baseline experiment
    config['experiment_name'] = f"baseline_all_domains_{trigger_type}"
    config['online_training']['trigger']['type'] = trigger_type
    
    # Configure trigger parameters
    if trigger_type == 'hybrid':
        config['online_training']['trigger']['max_time'] = 30.0
        config['online_training']['trigger']['max_samples'] = 50
        config['online_training']['trigger']['time_weight'] = 1.0
        config['online_training']['trigger']['volume_weight'] = 1.0
    elif trigger_type == 'volume':
        config['online_training']['trigger']['volume_interval'] = 50
    elif trigger_type == 'time':
        config['online_training']['trigger']['time_interval'] = 20.0
    elif trigger_type == 'drift':
        config['online_training']['trigger']['alpha'] = 0.1
        config['online_training']['trigger']['kappa'] = 1.5
        config['online_training']['trigger']['warmup_samples'] = 250
    
    # Use sequential domain cycling for clear baseline
    config['online_evaluation']['domain_selection'] = 'sequential_blocks'
    config['online_evaluation']['domain_block_size'] = block_size  # Samples per domain
    config['online_evaluation']['verbose'] = True
    
    # No domain remapping for baseline
    if 'domain_remapping' not in config['raw_data']:
        config['raw_data']['domain_remapping'] = {}
    
    # Save baseline config
    config_dir = Path(f"main_algorithm_v2/online/config/domain_shift_experiments")
    config_dir.mkdir(parents=True, exist_ok=True)
    
    config_path = config_dir / f"baseline_{trigger_type}_config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    return config_path

def create_domain_shift_config(base_config_path, primary_domain=3, shift_domain=8, 
                               shift_start=200, shift_end=800, trigger_type='hybrid'):
    """
    Create a config for temporal domain shift evaluation.
    
    Args:
        base_config_path: Path to base online config
        primary_domain: Main domain to focus on (e.g., 3)
        shift_domain: Domain to shift to temporarily (e.g., 8)
        shift_start: Sample number to start shift (e.g., 200)
        shift_end: Sample number to end shift (e.g., 800)
        trigger_type: Type of trigger to use
    
    Returns:
        Path to domain shift config file
    """
    # Load base config
    with open(base_config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set up domain shift experiment
    config['experiment_name'] = f"domain_shift_{primary_domain}_to_{shift_domain}_{trigger_type}"
    config['online_training']['trigger']['type'] = trigger_type
    
    # Configure trigger parameters (more sensitive for shift detection)
    if trigger_type == 'hybrid':
        config['online_training']['trigger']['max_time'] = 15.0  # More frequent triggers
        config['online_training']['trigger']['max_samples'] = 20
        config['online_training']['trigger']['time_weight'] = 1.0
        config['online_training']['trigger']['volume_weight'] = 1.0
    elif trigger_type == 'volume':
        config['online_training']['trigger']['volume_interval'] = 30
    elif trigger_type == 'time':
        config['online_training']['trigger']['time_interval'] = 10.0
    elif trigger_type == 'drift':
        config['online_training']['trigger']['alpha'] = 0.15  # More sensitive
        config['online_training']['trigger']['kappa'] = 1.3
        config['online_training']['trigger']['warmup_samples'] = 20
    
    # Add temporal domain shift configuration
    config['raw_data']['temporal_domain_shift'] = {
        'enabled': True,
        'primary_domain': primary_domain,
        'shift_domain': shift_domain,
        'shift_start_sample': shift_start,
        'shift_end_sample': shift_end,
        'description': f"Shift from domain {primary_domain} to {shift_domain} at samples {shift_start}-{shift_end}"
    }
    
    # Use temporal shift domain selection
    config['online_evaluation']['domain_selection'] = 'temporal_shift'
    config['online_evaluation']['verbose'] = True
    
    # No static domain remapping (we'll handle this dynamically)
    if 'domain_remapping' not in config['raw_data']:
        config['raw_data']['domain_remapping'] = {}
    
    # Save domain shift config
    config_dir = Path(f"main_algorithm_v2/online/config/domain_shift_experiments")
    config_dir.mkdir(parents=True, exist_ok=True)
    
    config_path = config_dir / f"domain_shift_{primary_domain}_{shift_domain}_{trigger_type}_config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    return config_path

def run_experiment(config_path, num_samples=1000, experiment_type="baseline"):
    """Run online learning experiment."""
    print(f"\n{'='*60}")
    print(f"RUNNING {experiment_type.upper()} EXPERIMENT")
    print(f"Config: {config_path}")
    print(f"Samples: {num_samples}")
    print(f"{'='*60}")
    
    try:
        cmd = [
            "python", "-m", "main_algorithm_v2.online.online_continual_learning",
            "--config", str(config_path),
            "--num_samples", str(num_samples)
        ]
        
        # Set encoding for Windows
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True,
                               encoding='utf-8', errors='replace', env=env)
        
        print("[SUCCESS] Experiment completed")
        # Print last 500 chars of stdout for debugging
        if result.stdout:
            print("Output (last 500 chars):", result.stdout[-500:])
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Experiment failed:")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return False

def analyze_single_domain_results(experiment_name, domain_id):
    """Analyze single domain streaming results."""
    results_dir = Path(f"main_algorithm_v2/online/eval/{experiment_name}")
    
    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        return None
    
    # Find the most recent results CSV
    csv_files = list(results_dir.glob('online_continual_learning_results_*.csv'))
    if not csv_files:
        print(f"No CSV results found in {results_dir}")
        return None
    
    latest_csv = max(csv_files, key=lambda x: x.stat().st_mtime)
    print(f"Loading single domain results from: {latest_csv}")
    
    # Load results
    df = pd.read_csv(latest_csv)
    
    # Analyze performance trends for the single domain
    analysis = {
        'domain_id': domain_id,
        'total_samples': len(df),
        'initial_nmse': df.iloc[:50]['nmse_post'].mean() if len(df) >= 50 else df['nmse_post'].mean(),
        'final_nmse': df.iloc[-50:]['nmse_post'].mean() if len(df) >= 50 else df['nmse_post'].mean(),
        'mean_nmse': df['nmse_post'].mean(),
        'std_nmse': df['nmse_post'].std(),
        'min_nmse': df['nmse_post'].min(),
        'max_nmse': df['nmse_post'].max(),
        'total_adaptations': df['triggered'].sum(),
        'adaptation_rate': df['triggered'].mean() * 100,  # Percentage of samples that triggered adaptation
        'improvement_trend': df['nmse_post'].iloc[-50:].mean() - df['nmse_post'].iloc[:50].mean() if len(df) >= 100 else 0,
        'performance_stability': df['nmse_post'].std() / df['nmse_post'].mean()  # Coefficient of variation
    }
    
    # Calculate improvement phases
    if len(df) >= 100:
        # Divide into quarters to see learning progression
        quarter_size = len(df) // 4
        analysis['quarters'] = {}
        for i in range(4):
            start_idx = i * quarter_size
            end_idx = (i + 1) * quarter_size if i < 3 else len(df)
            quarter_df = df.iloc[start_idx:end_idx]
            
            analysis['quarters'][f'Q{i+1}'] = {
                'mean_nmse': quarter_df['nmse_post'].mean(),
                'std_nmse': quarter_df['nmse_post'].std(),
                'adaptations': quarter_df['triggered'].sum(),
                'samples': len(quarter_df)
            }
    
    return analysis, df

def analyze_baseline_results(experiment_name):
    """Analyze baseline results showing performance vs samples for all domains."""
    results_dir = Path(f"main_algorithm_v2/online/eval/{experiment_name}")
    
    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        return None
    
    # Find the most recent results CSV
    csv_files = list(results_dir.glob('online_continual_learning_results_*.csv'))
    if not csv_files:
        print(f"No CSV results found in {results_dir}")
        return None
    
    latest_csv = max(csv_files, key=lambda x: x.stat().st_mtime)
    print(f"Loading baseline results from: {latest_csv}")
    
    # Load results
    df = pd.read_csv(latest_csv)
    
    # Analyze per-domain performance trends
    analysis = {
        'total_samples': len(df),
        'domains': {}
    }
    
    for domain_id in df['domain_id'].unique():
        domain_df = df[df['domain_id'] == domain_id]
        
        analysis['domains'][domain_id] = {
            'samples': len(domain_df),
            'initial_nmse': domain_df.iloc[:10]['nmse_post'].mean() if len(domain_df) >= 10 else domain_df['nmse_post'].mean(),
            'final_nmse': domain_df.iloc[-10:]['nmse_post'].mean() if len(domain_df) >= 10 else domain_df['nmse_post'].mean(),
            'mean_nmse': domain_df['nmse_post'].mean(),
            'std_nmse': domain_df['nmse_post'].std(),
            'adaptations': domain_df['triggered'].sum(),
            'improvement_trend': domain_df['nmse_post'].iloc[-10:].mean() - domain_df['nmse_post'].iloc[:10].mean() if len(domain_df) >= 20 else 0
        }
    
    return analysis, df

def analyze_domain_shift_results(experiment_name, shift_start=200, shift_end=800):
    """Analyze domain shift results focusing on adaptation performance."""
    results_dir = Path(f"main_algorithm_v2/online/eval/{experiment_name}")
    
    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        return None
    
    # Find the most recent results CSV
    csv_files = list(results_dir.glob('online_continual_learning_results_*.csv'))
    if not csv_files:
        print(f"No CSV results found in {results_dir}")
        return None
    
    latest_csv = max(csv_files, key=lambda x: x.stat().st_mtime)
    print(f"Loading domain shift results from: {latest_csv}")
    
    # Load results
    df = pd.read_csv(latest_csv)
    
    # Analyze performance in different phases
    pre_shift = df[df['sample_id'] < shift_start]
    during_shift = df[(df['sample_id'] >= shift_start) & (df['sample_id'] < shift_end)]
    post_shift = df[df['sample_id'] >= shift_end]
    
    analysis = {
        'total_samples': len(df),
        'shift_start': shift_start,
        'shift_end': shift_end,
        'phases': {
            'pre_shift': {
                'samples': len(pre_shift),
                'mean_nmse': pre_shift['nmse_post'].mean() if len(pre_shift) > 0 else float('nan'),
                'adaptations': pre_shift['triggered'].sum() if len(pre_shift) > 0 else 0,
                'final_nmse': pre_shift['nmse_post'].iloc[-10:].mean() if len(pre_shift) >= 10 else pre_shift['nmse_post'].mean() if len(pre_shift) > 0 else float('nan')
            },
            'during_shift': {
                'samples': len(during_shift),
                'mean_nmse': during_shift['nmse_post'].mean() if len(during_shift) > 0 else float('nan'),
                'adaptations': during_shift['triggered'].sum() if len(during_shift) > 0 else 0,
                'initial_nmse': during_shift['nmse_post'].iloc[:10].mean() if len(during_shift) >= 10 else during_shift['nmse_post'].mean() if len(during_shift) > 0 else float('nan'),
                'final_nmse': during_shift['nmse_post'].iloc[-10:].mean() if len(during_shift) >= 10 else during_shift['nmse_post'].mean() if len(during_shift) > 0 else float('nan'),
                'adaptation_effectiveness': 0  # Will calculate below
            },
            'post_shift': {
                'samples': len(post_shift),
                'mean_nmse': post_shift['nmse_post'].mean() if len(post_shift) > 0 else float('nan'),
                'adaptations': post_shift['triggered'].sum() if len(post_shift) > 0 else 0,
                'initial_nmse': post_shift['nmse_post'].iloc[:10].mean() if len(post_shift) >= 10 else post_shift['nmse_post'].mean() if len(post_shift) > 0 else float('nan')
            }
        }
    }
    
    # Calculate adaptation effectiveness during shift
    if len(during_shift) > 20:
        initial_shift_nmse = during_shift['nmse_post'].iloc[:10].mean()
        final_shift_nmse = during_shift['nmse_post'].iloc[-10:].mean()
        analysis['phases']['during_shift']['adaptation_effectiveness'] = initial_shift_nmse - final_shift_nmse
    
    return analysis, df

def plot_single_domain_performance(single_analysis, single_df, save_dir, domain_id, trigger_type):
    """Create comprehensive plots for single domain streaming results."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    domain_letter = DOMAIN_ID_TO_LETTER.get(domain_id, f'D{domain_id}')
    
    # Set plotting style
    plt.rcParams.update({
        'font.size': 12,
        'font.family': 'Times New Roman',
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 11,
        'figure.figsize': (14, 10),
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'lines.linewidth': 2,
        'axes.linewidth': 1,
        'grid.linewidth': 0.5,
        'axes.grid': True,
        'grid.alpha': 0.3
    })
    
    # Create comprehensive figure with subplots
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, height_ratios=[2, 1, 1], hspace=0.3, wspace=0.3)
    
    # 1. Main NMSE vs Sample Number Plot (top span)
    ax1 = fig.add_subplot(gs[0, :])
    
    # Plot raw data with transparency
    ax1.plot(single_df['sample_id'], single_df['nmse_post'], 
             alpha=0.4, color='blue', linewidth=1, label='Raw NMSE')
    
    # Plot rolling average for trend
    window = max(50, len(single_df) // 50)  # Adaptive window size
    rolling_nmse = single_df['nmse_post'].rolling(window=window, min_periods=1).mean()
    ax1.plot(single_df['sample_id'], rolling_nmse, 
             color='navy', linewidth=3, label=f'Rolling Average (window={window})')
    
    # Mark adaptations
    adaptations = single_df[single_df['triggered'] == 1]
    if len(adaptations) > 0:
        ax1.scatter(adaptations['sample_id'], adaptations['nmse_post'], 
                   marker='v', s=60, c='red', alpha=0.8, zorder=5, 
                   edgecolors='darkred', linewidth=1, label=f'Adaptations ({len(adaptations)})')
    
    # Add trend line if we have enough data
    if len(single_df) > 100:
        # Fit polynomial trend
        z = np.polyfit(single_df['sample_id'], single_df['nmse_post'], 2)
        p = np.poly1d(z)
        ax1.plot(single_df['sample_id'], p(single_df['sample_id']), 
                 "--", color='orange', linewidth=2, alpha=0.8, label='Polynomial Trend')
    
    ax1.set_xlabel('Sample Number', fontsize=14)
    ax1.set_ylabel('NMSE', fontsize=14)
    ax1.set_title(f'Single Domain Streaming: Domain {domain_letter} Performance Over Time\n'
                  f'Total Samples: {len(single_df):,} | Trigger: {trigger_type.capitalize()}', 
                  fontsize=16, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Add performance statistics as text box
    stats_text = f"""Performance Statistics:
Mean NMSE: {single_analysis['mean_nmse']:.6f}
Min NMSE: {single_analysis['min_nmse']:.6f}
Max NMSE: {single_analysis['max_nmse']:.6f}
Std Dev: {single_analysis['std_nmse']:.6f}
Stability (CV): {single_analysis['performance_stability']:.3f}
Adaptations: {single_analysis['total_adaptations']} ({single_analysis['adaptation_rate']:.1f}%)"""
    
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle="round,pad=0.4", 
             facecolor="lightblue", alpha=0.8))
    
    # 2. NMSE Distribution (bottom left)
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.hist(single_df['nmse_post'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    ax2.axvline(single_analysis['mean_nmse'], color='red', linestyle='--', linewidth=2, label='Mean')
    ax2.axvline(single_analysis['min_nmse'], color='green', linestyle='--', linewidth=2, label='Min')
    ax2.axvline(single_analysis['max_nmse'], color='orange', linestyle='--', linewidth=2, label='Max')
    ax2.set_xlabel('NMSE Value')
    ax2.set_ylabel('Frequency')
    ax2.set_title('NMSE Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Performance Over Time (Quarters) - bottom right
    if 'quarters' in single_analysis:
        ax3 = fig.add_subplot(gs[1, 1])
        quarters = ['Q1', 'Q2', 'Q3', 'Q4']
        quarter_nmse = [single_analysis['quarters'][q]['mean_nmse'] for q in quarters]
        quarter_std = [single_analysis['quarters'][q]['std_nmse'] for q in quarters]
        quarter_adapt = [single_analysis['quarters'][q]['adaptations'] for q in quarters]
        
        bars = ax3.bar(quarters, quarter_nmse, yerr=quarter_std, capsize=5, 
                      alpha=0.7, color=['lightcoral', 'lightblue', 'lightgreen', 'lightyellow'],
                      edgecolor='black')
        ax3.set_ylabel('Mean NMSE')
        ax3.set_title('Performance by Quarter')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Add adaptation counts on bars
        for bar, adapt in zip(bars, quarter_adapt):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2, height + 0.001,
                    f'{adapt} adapt', ha='center', va='bottom', fontsize=9)
    
    # 4. Adaptation Effectiveness (bottom span)
    ax4 = fig.add_subplot(gs[2, :])
    
    # Plot NMSE before and after each adaptation
    if len(adaptations) > 0:
        pre_adapt = []
        post_adapt = []
        improvements = []
        
        for _, adapt_row in adaptations.iterrows():
            if 'nmse_pre' in adapt_row and 'nmse_post' in adapt_row:
                pre_adapt.append(adapt_row['nmse_pre'])
                post_adapt.append(adapt_row['nmse_post'])
                improvements.append(adapt_row['nmse_pre'] - adapt_row['nmse_post'])
        
        if pre_adapt and post_adapt:
            adapt_indices = range(len(pre_adapt))
            
            ax4.plot(adapt_indices, pre_adapt, 'o-', color='red', label='Pre-Adaptation NMSE')
            ax4.plot(adapt_indices, post_adapt, 'o-', color='green', label='Post-Adaptation NMSE')
            
            # Show improvement arrows
            for i, (pre, post) in enumerate(zip(pre_adapt, post_adapt)):
                if pre > post:  # Improvement
                    ax4.annotate('', xy=(i, post), xytext=(i, pre),
                               arrowprops=dict(arrowstyle='↓', color='blue', lw=2))
            
            ax4.set_xlabel('Adaptation Event')
            ax4.set_ylabel('NMSE')
            ax4.set_title('Adaptation Effectiveness: Before vs After Training')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            # Add improvement statistics
            if improvements:
                avg_improvement = np.mean(improvements)
                success_rate = sum(1 for imp in improvements if imp > 0) / len(improvements) * 100
                ax4.text(0.02, 0.98, f'Avg Improvement: {avg_improvement:.6f}\nSuccess Rate: {success_rate:.1f}%',
                        transform=ax4.transAxes, fontsize=10, verticalalignment='top',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8))
    else:
        ax4.text(0.5, 0.5, 'No Adaptations Triggered', ha='center', va='center',
                transform=ax4.transAxes, fontsize=14, color='gray')
        ax4.set_title('No Adaptation Events')
    
    plt.tight_layout()
    plt.savefig(save_dir / f'single_domain_{domain_letter}_{trigger_type}_comprehensive.png', 
                dpi=300, bbox_inches='tight')
    plt.savefig(save_dir / f'single_domain_{domain_letter}_{trigger_type}_comprehensive.svg', 
                format='svg', bbox_inches='tight')
    plt.close()
    
    # Create a clean, simple NMSE vs samples plot for easy analysis
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Only plot the rolling average for clean visualization
    window = max(30, len(single_df) // 100)
    rolling_nmse = single_df['nmse_post'].rolling(window=window, min_periods=1).mean()
    ax.plot(single_df['sample_id'], rolling_nmse, 
            color='blue', linewidth=3, label=f'NMSE (smoothed)')
    
    # Mark adaptations as vertical lines
    if len(adaptations) > 0:
        for _, adapt_row in adaptations.iterrows():
            ax.axvline(x=adapt_row['sample_id'], color='red', alpha=0.6, linewidth=1)
        # Add single legend entry for all adaptations
        ax.axvline(x=adaptations.iloc[0]['sample_id'], color='red', alpha=0.6, linewidth=1, 
                  label=f'Adaptations ({len(adaptations)})')
    
    ax.set_xlabel('Sample Number', fontsize=14)
    ax.set_ylabel('NMSE (Smoothed)', fontsize=14)
    ax.set_title(f'Domain {domain_letter} Streaming Performance\n'
                 f'{len(single_df):,} Samples | {trigger_type.capitalize()} Trigger', 
                 fontsize=16, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / f'single_domain_{domain_letter}_{trigger_type}_clean.png', 
                dpi=300, bbox_inches='tight')
    plt.savefig(save_dir / f'single_domain_{domain_letter}_{trigger_type}_clean.svg', 
                format='svg', bbox_inches='tight')
    plt.close()
    
    print(f"[PLOTS] Single domain plots saved to: {save_dir}")
    print(f"         • single_domain_{domain_letter}_{trigger_type}_comprehensive.png - Full analysis")
    print(f"         • single_domain_{domain_letter}_{trigger_type}_clean.png - Clean NMSE trend")

def plot_baseline_performance(baseline_analysis, baseline_df, save_dir):
    """Create plots for baseline performance across all domains."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Set plotting style
    plt.rcParams.update({
        'font.size': 12,
        'font.family': 'Times New Roman',
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 11,
        'figure.figsize': (12, 8),
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'lines.linewidth': 2,
        'axes.linewidth': 1,
        'grid.linewidth': 0.5,
        'axes.grid': True,
        'grid.alpha': 0.3
    })
    
    # 1. Performance vs Samples for All Domains
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Get domains and sort alphabetically by letter
    domain_ids = list(baseline_analysis['domains'].keys())
    domain_letters = [(DOMAIN_ID_TO_LETTER.get(d, f'D{d}'), d) for d in domain_ids]
    domain_letters.sort(key=lambda x: x[0])  # Sort by letter
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(domain_letters)))
    
    for i, (letter, domain_id) in enumerate(domain_letters):
        domain_df = baseline_df[baseline_df['domain_id'] == domain_id]
        
        if len(domain_df) > 0:
            # Plot raw data with transparency
            ax.plot(domain_df['sample_id'], domain_df['nmse_post'], 
                   alpha=0.3, color=colors[i], linewidth=0.5)
            
            # Plot rolling average
            window = min(20, len(domain_df) // 5)
            if window > 1:
                rolling_nmse = domain_df['nmse_post'].rolling(window=window, min_periods=1).mean()
                ax.plot(domain_df['sample_id'], rolling_nmse, 
                       label=f'Domain {letter}', color=colors[i], linewidth=2)
            
            # Mark adaptations
            adaptations = domain_df[domain_df['triggered'] == 1]
            if len(adaptations) > 0:
                ax.scatter(adaptations['sample_id'], adaptations['nmse_post'], 
                          marker='v', s=30, c=colors[i], alpha=0.8, zorder=5, edgecolors='black')
    
    ax.set_xlabel('Sample Number')
    ax.set_ylabel('NMSE')
    ax.set_title('Baseline Performance: All Domains vs Sample Number')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'baseline_all_domains_performance.png', dpi=300, bbox_inches='tight')
    plt.savefig(save_dir / 'baseline_all_domains_performance.svg', format='svg', bbox_inches='tight')
    plt.close()
    
    # 2. Clean Performance Plot (No Triggers, Smoothed Only) - For Cross-Trigger Comparison
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Use same alphabetical ordering as above
    colors = plt.cm.Set3(np.linspace(0, 1, len(domain_letters)))
    
    for i, (letter, domain_id) in enumerate(domain_letters):
        domain_df = baseline_df[baseline_df['domain_id'] == domain_id]
        
        if len(domain_df) > 0:
            # Only plot rolling average (no raw data, no triggers)
            window = min(50, len(domain_df) // 3)  # Longer smoothing window
            if window > 1:
                rolling_nmse = domain_df['nmse_post'].rolling(window=window, min_periods=1).mean()
                ax.plot(domain_df['sample_id'], rolling_nmse, 
                       label=f'Domain {letter}', color=colors[i], linewidth=2.5)
            else:
                # If too few samples, just plot the mean
                ax.axhline(y=domain_df['nmse_post'].mean(), 
                          color=colors[i], linewidth=2.5, 
                          label=f'Domain {letter}')
    
    ax.set_xlabel('Sample Number', fontsize=14)
    ax.set_ylabel('NMSE (Smoothed)', fontsize=14)
    ax.set_title('Baseline Performance: Smoothed NMSE by Domain\n(For Cross-Trigger Comparison)', 
                 fontsize=16, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Add domain boundaries as subtle vertical lines
    num_domains = len(baseline_analysis['domains'])
    if num_domains > 0:
        block_size = len(baseline_df) // num_domains if num_domains > 0 else 200
        for i in range(1, num_domains):
            ax.axvline(x=i * block_size, color='gray', linestyle=':', alpha=0.5, linewidth=1)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'baseline_clean_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(save_dir / 'baseline_clean_comparison.svg', format='svg', bbox_inches='tight')
    plt.close()
    
    # 3. Domain Performance Summary
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Use alphabetically sorted domains
    sorted_letters_ids = [(DOMAIN_ID_TO_LETTER.get(d, f'D{d}'), d) for d in baseline_analysis['domains'].keys()]
    sorted_letters_ids.sort(key=lambda x: x[0])  # Sort by letter
    
    domain_letters = [x[0] for x in sorted_letters_ids]
    domain_ids = [x[1] for x in sorted_letters_ids]
    
    mean_nmse = [baseline_analysis['domains'][d]['mean_nmse'] for d in domain_ids]
    std_nmse = [baseline_analysis['domains'][d]['std_nmse'] for d in domain_ids]
    adaptations = [baseline_analysis['domains'][d]['adaptations'] for d in domain_ids]
    
    # Mean NMSE per domain
    bars1 = ax1.bar(range(len(domain_ids)), mean_nmse, yerr=std_nmse, 
                    capsize=5, color=colors[:len(domain_ids)], alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Domain')
    ax1.set_ylabel('Mean NMSE')
    ax1.set_title('Baseline: Mean NMSE per Domain')
    ax1.set_xticks(range(len(domain_ids)))
    ax1.set_xticklabels(domain_letters)
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, val, std) in enumerate(zip(bars1, mean_nmse, std_nmse)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, height + std + 0.001,
                f'{val:.4f}', ha='center', va='bottom', fontsize=10)
    
    # Adaptations per domain
    bars2 = ax2.bar(range(len(domain_ids)), adaptations, 
                    color=colors[:len(domain_ids)], alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Domain')
    ax2.set_ylabel('Number of Adaptations')
    ax2.set_title('Baseline: Adaptations per Domain')
    ax2.set_xticks(range(len(domain_ids)))
    ax2.set_xticklabels(domain_letters)
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, val in zip(bars2, adaptations):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, height + 0.5,
                f'{val}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'baseline_domain_summary.png', dpi=300, bbox_inches='tight')
    plt.savefig(save_dir / 'baseline_domain_summary.svg', format='svg', bbox_inches='tight')
    plt.close()
    
    print(f"[PLOTS] Baseline plots saved to: {save_dir}")
    print(f"         • baseline_all_domains_performance.png - Full detail with triggers")
    print(f"         • baseline_clean_comparison.png - Clean smoothed curves for comparison")
    print(f"         • baseline_domain_summary.png - Bar charts of mean performance")

def plot_domain_shift_performance(shift_analysis, shift_df, save_dir, 
                                  primary_domain=3, shift_domain=8):
    """Create plots for domain shift adaptation analysis."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    shift_start = shift_analysis['shift_start']
    shift_end = shift_analysis['shift_end']
    
    # Get letter labels for domains
    primary_letter = DOMAIN_ID_TO_LETTER.get(primary_domain, f'D{primary_domain}')
    shift_letter = DOMAIN_ID_TO_LETTER.get(shift_domain, f'D{shift_domain}')
    
    # 1. Domain Shift Timeline - Main Plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot NMSE timeline
    window = 20
    rolling_nmse = shift_df['nmse_post'].rolling(window=window, min_periods=1).mean()
    
    ax.plot(shift_df['sample_id'], shift_df['nmse_post'], alpha=0.3, color='blue', linewidth=0.5, label='Raw NMSE')
    ax.plot(shift_df['sample_id'], rolling_nmse, color='blue', linewidth=2.5, label='Rolling Average NMSE')
    
    # Mark adaptations with larger, more visible markers
    adaptations = shift_df[shift_df['triggered'] == 1]
    if len(adaptations) > 0:
        ax.scatter(adaptations['sample_id'], adaptations['nmse_post'], 
                  marker='v', s=80, c='red', alpha=0.9, zorder=5, 
                  edgecolors='darkred', linewidth=1.5, label='Adaptation Triggered')
    
    # Add phase regions with better colours and labels
    ax.axvspan(0, shift_start, alpha=0.2, color='lightgreen', 
               label=f'Phase 1: Normal Domain {primary_letter}')
    ax.axvspan(shift_start, shift_end, alpha=0.2, color='lightcoral', 
               label=f'Phase 2: Domain {shift_letter} (mislabeled as {primary_letter})')
    ax.axvspan(shift_end, shift_df['sample_id'].max(), alpha=0.2, color='lightblue', 
               label=f'Phase 3: Back to Domain {primary_letter}')
    
    # Add vertical lines for shift points
    ax.axvline(x=shift_start, color='red', linestyle='--', linewidth=2, alpha=0.8)
    ax.axvline(x=shift_end, color='green', linestyle='--', linewidth=2, alpha=0.8)
    
    # Add text annotations for clarity
    y_top = ax.get_ylim()[1]
    ax.text(shift_start/2, y_top*0.95, f'Normal\nDomain {primary_letter}', 
            ha='center', va='top', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8))
    ax.text((shift_start + shift_end)/2, y_top*0.95, f'MISLABELED\nData: Domain {shift_letter}\nLabel: Domain {primary_letter}', 
            ha='center', va='top', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.8))
    ax.text((shift_end + shift_df['sample_id'].max())/2, y_top*0.95, f'Normal\nDomain {primary_letter}', 
            ha='center', va='top', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
    
    ax.set_xlabel('Sample Number', fontsize=14)
    ax.set_ylabel('NMSE', fontsize=14)
    ax.set_title(f'Online Adaptation to Domain Shift: Domain {primary_letter} → {shift_letter} → {primary_letter}', 
                 fontsize=16, fontweight='bold')
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / f'domain_shift_{primary_letter}_{shift_letter}_timeline.png', 
                dpi=300, bbox_inches='tight')
    plt.savefig(save_dir / f'domain_shift_{primary_letter}_{shift_letter}_timeline.svg', 
                format='svg', bbox_inches='tight')
    plt.close()
    
    # Create a clean plot without individual adaptation markers
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot only rolling average for clean visualization
    window = 50
    rolling_nmse = shift_df['nmse_post'].rolling(window=window, min_periods=1).mean()
    ax.plot(shift_df['sample_id'], rolling_nmse, color='blue', linewidth=3, label='NMSE (Rolling Average)')
    
    # Add phase regions with better colours and labels
    ax.axvspan(0, shift_start, alpha=0.2, color='lightgreen', 
               label=f'Phase 1: Normal Domain {primary_letter}')
    ax.axvspan(shift_start, shift_end, alpha=0.2, color='lightcoral', 
               label=f'Phase 2: Domain {shift_letter} → {primary_letter} Adapter')
    ax.axvspan(shift_end, shift_df['sample_id'].max(), alpha=0.2, color='lightblue', 
               label=f'Phase 3: Back to Domain {primary_letter}')
    
    # Add vertical lines for shift points (cleaner than scatter)
    ax.axvline(x=shift_start, color='red', linestyle='--', linewidth=2, alpha=0.8, label='Domain Injection Start')
    ax.axvline(x=shift_end, color='green', linestyle='--', linewidth=2, alpha=0.8, label='Domain Injection End')
    
    # Add text annotations for clarity
    y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
    y_top = ax.get_ylim()[1] - y_range * 0.05
    ax.text(shift_start/2, y_top, f'Normal\nDomain {primary_letter}', 
            ha='center', va='top', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8))
    ax.text((shift_start + shift_end)/2, y_top, f'INJECTED\nData: Domain {shift_letter}\nAdapter: Domain {primary_letter}', 
            ha='center', va='top', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.8))
    ax.text((shift_end + shift_df['sample_id'].max())/2, y_top, f'Normal\nDomain {primary_letter}', 
            ha='center', va='top', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
    
    ax.set_xlabel('Sample Number', fontsize=14)
    ax.set_ylabel('NMSE (Smoothed)', fontsize=14)
    ax.set_title(f'Domain Injection Analysis: Domain {primary_letter} → {shift_letter} → {primary_letter}\n'
                 f'Clean View (No Individual Adaptation Markers)', 
                 fontsize=16, fontweight='bold')
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Add performance statistics
    pre_mean = shift_analysis['phases']['pre_shift']['mean_nmse']
    during_mean = shift_analysis['phases']['during_shift']['mean_nmse']
    post_mean = shift_analysis['phases']['post_shift']['mean_nmse']
    
    stats_text = f"""Performance Summary:
Pre-injection: {pre_mean:.6f} NMSE
During injection: {during_mean:.6f} NMSE
Post-injection: {post_mean:.6f} NMSE
Degradation: {during_mean - pre_mean:+.6f} NMSE
Recovery: {post_mean - during_mean:+.6f} NMSE"""
    
    ax.text(0.02, 0.02, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', bbox=dict(boxstyle="round,pad=0.4", 
            facecolor="lightyellow", alpha=0.9))
    
    plt.tight_layout()
    plt.savefig(save_dir / f'domain_shift_{primary_letter}_{shift_letter}_clean.png', 
                dpi=300, bbox_inches='tight')
    plt.savefig(save_dir / f'domain_shift_{primary_letter}_{shift_letter}_clean.svg', 
                format='svg', bbox_inches='tight')
    plt.close()
    
    # 2. Phase Analysis
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    phases = ['Pre-Shift\n(Normal)', 'During Shift\n(Mislabeled)', 'Post-Shift\n(Normal)']
    phase_data = shift_analysis['phases']
    
    # NMSE comparison across phases
    mean_nmse_values = [phase_data['pre_shift']['mean_nmse'], 
                        phase_data['during_shift']['mean_nmse'], 
                        phase_data['post_shift']['mean_nmse']]
    
    colors_phase = ['lightgreen', 'lightcoral', 'lightblue']
    bars1 = ax1.bar(phases, mean_nmse_values, color=colors_phase, alpha=0.7, 
                    edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('Mean NMSE', fontsize=14)
    ax1.set_title('Mean NMSE Across Phases', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, val in zip(bars1, mean_nmse_values):
        if not np.isnan(val):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2, height + 0.001,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Adaptations across phases
    adaptations_values = [phase_data['pre_shift']['adaptations'],
                          phase_data['during_shift']['adaptations'],
                          phase_data['post_shift']['adaptations']]
    
    bars2 = ax2.bar(phases, adaptations_values, color=colors_phase, alpha=0.7, 
                    edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('Number of Adaptations', fontsize=14)
    ax2.set_title('Adaptations Across Phases', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, val in zip(bars2, adaptations_values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, height + 0.1,
                f'{val}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_dir / f'domain_shift_{primary_letter}_{shift_letter}_phases.png', 
                dpi=300, bbox_inches='tight')
    plt.savefig(save_dir / f'domain_shift_{primary_letter}_{shift_letter}_phases.svg', 
                format='svg', bbox_inches='tight')
    plt.close()
    
    print(f"[PLOTS] Domain shift plots saved to: {save_dir}")

def generate_trigger_comparison_table(current_trigger):
    """Generate a comparison table across different trigger types."""
    print(f"\n{'='*80}")
    print("CROSS-TRIGGER COMPARISON TABLE GENERATOR")
    print(f"{'='*80}")
    
    # Look for all baseline results across different triggers
    base_dir = Path("main_algorithm_v2/online/eval")
    trigger_types = ['drift', 'hybrid', 'volume', 'time']
    
    comparison_data = {}
    
    for trigger in trigger_types:
        exp_name = f"baseline_all_domains_{trigger}"
        results_dir = base_dir / exp_name
        
        if results_dir.exists():
            csv_files = list(results_dir.glob('online_continual_learning_results_*.csv'))
            if csv_files:
                latest_csv = max(csv_files, key=lambda x: x.stat().st_mtime)
                try:
                    df = pd.read_csv(latest_csv)
                    
                    # Calculate mean NMSE per domain
                    trigger_results = {}
                    for domain_id in sorted(df['domain_id'].unique()):
                        domain_df = df[df['domain_id'] == domain_id]
                        trigger_results[domain_id] = {
                            'mean_nmse': domain_df['nmse_post'].mean(),
                            'std_nmse': domain_df['nmse_post'].std(),
                            'adaptations': domain_df['triggered'].sum(),
                            'samples': len(domain_df)
                        }
                    
                    comparison_data[trigger] = trigger_results
                    print(f"✅ Found results for {trigger} trigger")
                    
                except Exception as e:
                    print(f"❌ Error loading {trigger} results: {e}")
            else:
                print(f"⚠️  No CSV results found for {trigger} trigger")
        else:
            print(f"⚠️  No results directory found for {trigger} trigger")
    
    if len(comparison_data) >= 2:
        print(f"\n📊 GENERATING COMPARISON TABLE")
        print("Run all trigger types to get complete comparison!")
        
        # Create CSV for easy analysis
        table_path = base_dir / "trigger_comparison_table.csv"
        
        # Prepare table data
        table_rows = []
        domains = sorted(list(comparison_data[list(comparison_data.keys())[0]].keys()))
        
        for domain_id in domains:
            domain_letter = DOMAIN_ID_TO_LETTER.get(domain_id, f'D{domain_id}')
            for trigger in sorted(comparison_data.keys()):
                if domain_id in comparison_data[trigger]:
                    stats = comparison_data[trigger][domain_id]
                    table_rows.append({
                        'Domain': domain_letter,
                        'Trigger': trigger,
                        'Mean_NMSE': f"{stats['mean_nmse']:.6f}",
                        'Std_NMSE': f"{stats['std_nmse']:.6f}",
                        'Adaptations': stats['adaptations'],
                        'Samples': stats['samples']
                    })
        
        # Save to CSV
        comparison_df = pd.DataFrame(table_rows)
        comparison_df.to_csv(table_path, index=False)
        
        # Display summary table with letter mapping
        print(f"\n📋 MEAN NMSE COMPARISON TABLE:")
        print("-" * 60)
        header = f"{'Domain':<8}"
        for trigger in sorted(comparison_data.keys()):
            header += f"{trigger.capitalize():<12}"
        print(header)
        print("-" * 60)
        
        # Sort domains alphabetically by letter
        domain_letter_pairs = [(DOMAIN_ID_TO_LETTER.get(d, f'D{d}'), d) for d in domains]
        domain_letter_pairs.sort(key=lambda x: x[0])
        
        for letter, domain_id in domain_letter_pairs:
            row = f"{letter:<8}"
            for trigger in sorted(comparison_data.keys()):
                if domain_id in comparison_data[trigger]:
                    nmse = comparison_data[trigger][domain_id]['mean_nmse']
                    row += f"{nmse:<12.6f}"
                else:
                    row += f"{'N/A':<12}"
            print(row)
        
        print("-" * 60)
        print(f"\n💾 Detailed comparison saved to: {table_path}")
        print("📈 Import this CSV into Excel/Python for further analysis!")
        
        # Show which trigger is best per domain
        print(f"\n🏆 BEST TRIGGER PER DOMAIN:")
        for domain_id in domains:
            best_trigger = None
            best_nmse = float('inf')
            
            for trigger in comparison_data:
                if domain_id in comparison_data[trigger]:
                    nmse = comparison_data[trigger][domain_id]['mean_nmse']
                    if nmse < best_nmse:
                        best_nmse = nmse
                        best_trigger = trigger
            
            if best_trigger:
                domain_letter = DOMAIN_ID_TO_LETTER.get(domain_id, f'D{domain_id}')
                print(f"   Domain {domain_letter}: {best_trigger.capitalize()} ({best_nmse:.6f} NMSE)")
    
    else:
        print(f"\n⚠️  Only {len(comparison_data)} trigger type(s) found.")
        print("Run multiple trigger types to enable comparison:")
        print("   python ... --trigger drift --baseline_only")
        print("   python ... --trigger hybrid --baseline_only")  
        print("   python ... --trigger volume --baseline_only")
        print("   python ... --trigger time --baseline_only")

def diagnose_domain_distribution(experiment_name):
    """Diagnose how samples are distributed across domains."""
    results_dir = Path(f"main_algorithm_v2/online/eval/{experiment_name}")
    
    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        return None
    
    # Find the most recent results CSV
    csv_files = list(results_dir.glob('online_continual_learning_results_*.csv'))
    if not csv_files:
        print(f"No CSV results found in {results_dir}")
        return None
    
    latest_csv = max(csv_files, key=lambda x: x.stat().st_mtime)
    print(f"🔍 DIAGNOSING DATA DISTRIBUTION FROM: {latest_csv}")
    
    # Load results
    df = pd.read_csv(latest_csv)
    
    print(f"\n📊 OVERALL DATA SUMMARY:")
    print(f"   Total samples: {len(df)}")
    print(f"   Sample range: {df['sample_id'].min()} to {df['sample_id'].max()}")
    print(f"   Unique domains: {sorted(df['domain_id'].unique())}")
    
    print(f"\n📋 DOMAIN DISTRIBUTION:")
    print("-" * 80)
    print(f"{'Domain':<8} {'Count':<8} {'First Sample':<12} {'Last Sample':<12} {'Sample Range':<15}")
    print("-" * 80)
    
    # Sort domains alphabetically by letter for display
    domain_ids = sorted(df['domain_id'].unique())
    domain_letter_pairs = [(DOMAIN_ID_TO_LETTER.get(d, f'D{d}'), d) for d in domain_ids]
    domain_letter_pairs.sort(key=lambda x: x[0])
    
    for letter, domain_id in domain_letter_pairs:
        domain_df = df[df['domain_id'] == domain_id]
        first_sample = domain_df['sample_id'].min()
        last_sample = domain_df['sample_id'].max()
        sample_range = f"{first_sample}-{last_sample}"
        
        print(f"{letter:<8} {len(domain_df):<8} {first_sample:<12} {last_sample:<12} {sample_range:<15}")
    
    print("-" * 80)
    
    # Check for overlaps (this should NOT happen with sequential_blocks)
    print(f"\n🔄 CHECKING FOR OVERLAPS (should be NONE):")
    overlaps_found = False
    
    for sample_id in range(0, df['sample_id'].max() + 1, 500):  # Check every 500 samples
        domains_at_sample = df[df['sample_id'] == sample_id]['domain_id'].unique()
        if len(domains_at_sample) > 1:
            print(f"   ⚠️ Sample {sample_id}: Multiple domains {domains_at_sample}")
            overlaps_found = True
    
    if not overlaps_found:
        print("   ✅ No overlaps found - domains are properly sequential")
    
    # Show expected vs actual distribution
    print(f"\n📏 EXPECTED vs ACTUAL (with block_size=2000):")
    expected_blocks = len(df) // 2000 if len(df) >= 2000 else 1
    
    for i, (letter, domain_id) in enumerate(domain_letter_pairs):
        expected_start = i * 2000
        expected_end = (i + 1) * 2000 - 1
        
        domain_df = df[df['domain_id'] == domain_id]
        actual_start = domain_df['sample_id'].min()
        actual_end = domain_df['sample_id'].max()
        
        expected = f"{expected_start}-{expected_end}"
        actual = f"{actual_start}-{actual_end}"
        
        match = "✅" if (actual_start == expected_start and actual_end == expected_end) else "❌"
        print(f"   Domain {letter}: Expected {expected}, Actual {actual} {match}")
    
    return df

def main():
    """Run dynamic domain shift evaluation experiments."""
    parser = argparse.ArgumentParser(description="Run dynamic domain shift evaluation",
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     epilog="""
Examples:
  # Run single domain streaming (recommended for NMSE vs samples analysis)
  python run_domain_mislabeling_experiment.py --single_domain 3 --trigger drift
  
  # Run single domain with limited samples
  python run_domain_mislabeling_experiment.py --single_domain 3 --max_samples 5000
  
  # Run both baseline and domain shift
  python run_domain_mislabeling_experiment.py --num_samples 1200
  
  # Run only baseline
  python run_domain_mislabeling_experiment.py --baseline_only --num_samples 1800
  
  # Run only domain shift (3->8->3)
  python run_domain_mislabeling_experiment.py --shift_only --primary_domain 3 --shift_domain 8
  
  # Quick test with fewer samples
  python run_domain_mislabeling_experiment.py --quick
                                     """)
    parser.add_argument('--base_config', type=str, 
                       default='main_algorithm_v2/online/config/online_config.yaml',
                       help='Path to base online config')
    parser.add_argument('--num_samples', type=int, default=1200,
                       help='Number of samples (default: 1200 for full shift cycle)')
    parser.add_argument('--quick', action='store_true',
                       help='Run quick test with fewer samples')
    parser.add_argument('--trigger', type=str, default='hybrid',
                       choices=['hybrid', 'drift', 'volume', 'time'],
                       help='Trigger type to test (default: hybrid)')
    parser.add_argument('--primary_domain', type=int, default=3,
                       help='Primary domain to focus on (default: 3)')
    parser.add_argument('--shift_domain', type=int, default=8,
                       help='Domain to shift to (default: 8)')
    parser.add_argument('--shift_start', type=int, default=200,
                       help='Sample to start domain shift (default: 200)')
    parser.add_argument('--shift_end', type=int, default=800,
                       help='Sample to end domain shift (default: 800)')
    parser.add_argument('--baseline_only', action='store_true',
                       help='Run only baseline evaluation')
    parser.add_argument('--shift_only', action='store_true',
                       help='Run only domain shift evaluation')
    parser.add_argument('--baseline_block_size', type=int, default=200,
                       help='Samples per domain in baseline (default: 200)')
    parser.add_argument('--single_domain', type=int, default=None,
                       help='Run single domain streaming for specified domain (e.g., --single_domain 3)')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum samples for single domain streaming (default: use all available)')
    args = parser.parse_args()
    
    # Adjust for quick test
    if args.quick:
        args.num_samples = 600
        args.shift_start = 100
        args.shift_end = 400
        print("[QUICK MODE] Using reduced sample counts")
    
    # Handle single domain mode
    if args.single_domain is not None:
        print("="*80)
        print("SINGLE DOMAIN STREAMING EVALUATION")
        print("="*80)
        domain_letter = DOMAIN_ID_TO_LETTER.get(args.single_domain, f'D{args.single_domain}')
        print(f"Domain: {domain_letter} (ID: {args.single_domain})")
        print(f"Trigger type: {args.trigger}")
        max_samples = args.max_samples if args.max_samples else "All available"
        print(f"Max samples: {max_samples}")
        print("="*80)
        print("This mode streams data from a single domain to show NMSE vs samples over time.")
        print("Perfect for understanding adaptation behavior within a single domain.")
        
        # Create single domain config
        single_config = create_single_domain_config(args.base_config, args.single_domain, args.trigger)
        
        # Use all available data unless max_samples specified
        samples_to_use = args.max_samples if args.max_samples else 50000  # Large number to get all data
        
        # Run single domain experiment
        single_success = run_experiment(single_config, samples_to_use, "single_domain")
        
        if single_success:
            single_exp_name = f"single_domain_{args.single_domain}_{args.trigger}"
            single_result = analyze_single_domain_results(single_exp_name, args.single_domain)
            
            if single_result:
                single_analysis, single_df = single_result
                
                print(f"\n[SINGLE DOMAIN ANALYSIS - Domain {domain_letter}]")
                print(f"Total samples processed: {single_analysis['total_samples']:,}")
                print(f"Mean NMSE: {single_analysis['mean_nmse']:.6f} ± {single_analysis['std_nmse']:.6f}")
                print(f"NMSE range: {single_analysis['min_nmse']:.6f} to {single_analysis['max_nmse']:.6f}")
                print(f"Performance stability (CV): {single_analysis['performance_stability']:.3f}")
                print(f"Total adaptations: {single_analysis['total_adaptations']} ({single_analysis['adaptation_rate']:.1f}% of samples)")
                
                if single_analysis['improvement_trend'] != 0:
                    trend_direction = "↗ Improving" if single_analysis['improvement_trend'] < 0 else "↘ Degrading"
                    print(f"Overall trend: {trend_direction} ({single_analysis['improvement_trend']:.6f} NMSE change)")
                else:
                    print("Overall trend: → Stable")
                
                # Show quarter-by-quarter analysis if available
                if 'quarters' in single_analysis:
                    print(f"\nQuarterly Performance Breakdown:")
                    for q in ['Q1', 'Q2', 'Q3', 'Q4']:
                        q_data = single_analysis['quarters'][q]
                        print(f"  {q}: {q_data['mean_nmse']:.6f} NMSE ({q_data['adaptations']} adaptations, {q_data['samples']} samples)")
                
                # Create comprehensive plots
                single_plot_dir = Path(f"main_algorithm_v2/online/eval/domain_shift_analysis/single_domain_{args.single_domain}_{args.trigger}")
                plot_single_domain_performance(single_analysis, single_df, single_plot_dir, args.single_domain, args.trigger)
                
                print(f"\n✅ Single domain streaming analysis completed!")
                print(f"📊 Comprehensive plots saved to: {single_plot_dir}")
                print("📈 Key files:")
                print(f"   • single_domain_{domain_letter}_{args.trigger}_comprehensive.png - Full analysis dashboard")
                print(f"   • single_domain_{domain_letter}_{args.trigger}_clean.png - Clean NMSE vs samples plot")
        else:
            print("❌ Single domain experiment failed!")
        
        return  # Exit after single domain analysis
    
    print("="*80)
    print("DYNAMIC DOMAIN SHIFT EVALUATION")
    print("="*80)
    print(f"Trigger type: {args.trigger}")
    print(f"Primary domain: {args.primary_domain}")
    print(f"Shift domain: {args.shift_domain}")
    print(f"Shift period: samples {args.shift_start}-{args.shift_end}")
    print(f"Total samples: {args.num_samples}")
    print("="*80)
    
    results = {}
    
    # 1. Run Baseline Evaluation
    if not args.shift_only:
        print(f"\n{'-'*60}")
        print("PHASE 1: BASELINE EVALUATION (ALL DOMAINS)")
        print(f"{'-'*60}")
        print("This shows normal performance across all domains to establish baseline behavior.")
        
        baseline_config = create_baseline_config(args.base_config, args.trigger, args.baseline_block_size)
        baseline_success = run_experiment(baseline_config, args.num_samples, "baseline")
        
        if baseline_success:
            baseline_exp_name = f"baseline_all_domains_{args.trigger}"
            baseline_result = analyze_baseline_results(baseline_exp_name)
            
            if baseline_result:
                baseline_analysis, baseline_df = baseline_result
                results['baseline'] = (baseline_analysis, baseline_df)
                
                print(f"\n[BASELINE ANALYSIS]")
                print(f"Total samples: {baseline_analysis['total_samples']}")
                # Sort by letter for consistent display
                domain_letter_pairs = [(DOMAIN_ID_TO_LETTER.get(d, f'D{d}'), d) for d in baseline_analysis['domains'].keys()]
                domain_letter_pairs.sort(key=lambda x: x[0])
                
                for letter, domain_id in domain_letter_pairs:
                    stats = baseline_analysis['domains'][domain_id]
                    trend = "↗" if stats['improvement_trend'] > 0.001 else "↘" if stats['improvement_trend'] < -0.001 else "→"
                    print(f"  Domain {letter}: {stats['mean_nmse']:.6f} ± {stats['std_nmse']:.6f} "
                          f"({stats['adaptations']} adaptations) {trend}")
                
                # Diagnose domain distribution first
                print(f"\n[DOMAIN DISTRIBUTION DIAGNOSIS]")
                diagnose_domain_distribution(baseline_exp_name)
                
                # Create baseline plots
                baseline_plot_dir = Path(f"main_algorithm_v2/online/eval/domain_shift_analysis/baseline_{args.trigger}")
                plot_baseline_performance(baseline_analysis, baseline_df, baseline_plot_dir)
    
    # 2. Run Domain Shift Evaluation
    if not args.baseline_only:
        print(f"\n{'-'*60}")
        print(f"PHASE 2: DOMAIN SHIFT EVALUATION")
        primary_letter = DOMAIN_ID_TO_LETTER.get(args.primary_domain, f'D{args.primary_domain}')
        shift_letter = DOMAIN_ID_TO_LETTER.get(args.shift_domain, f'D{args.shift_domain}')
        
        print(f"PRIMARY: Domain {primary_letter} | SHIFT: Domain {shift_letter}")
        print(f"{'-'*60}")
        print(f"Testing adaptation when domain {shift_letter} data is mislabeled as domain {primary_letter}")
        print(f"Samples 0-{args.shift_start}: Normal domain {primary_letter}")
        print(f"Samples {args.shift_start}-{args.shift_end}: Domain {shift_letter} data labeled as domain {primary_letter}")
        print(f"Samples {args.shift_end}+: Back to normal domain {primary_letter}")
        
        shift_config = create_domain_shift_config(
            args.base_config, args.primary_domain, args.shift_domain,
            args.shift_start, args.shift_end, args.trigger
        )
        shift_success = run_experiment(shift_config, args.num_samples, "domain_shift")
        
        if shift_success:
            shift_exp_name = f"domain_shift_{args.primary_domain}_to_{args.shift_domain}_{args.trigger}"
            shift_result = analyze_domain_shift_results(shift_exp_name, args.shift_start, args.shift_end)
            
            if shift_result:
                shift_analysis, shift_df = shift_result
                results['domain_shift'] = (shift_analysis, shift_df)
                
                print(f"\n[DOMAIN SHIFT ANALYSIS]")
                print(f"Total samples: {shift_analysis['total_samples']}")
                print(f"Shift period: {shift_analysis['shift_start']}-{shift_analysis['shift_end']}")
                
                phases = shift_analysis['phases']
                print(f"\nPhase Analysis:")
                print(f"  Pre-shift:    {phases['pre_shift']['mean_nmse']:.6f} NMSE ({phases['pre_shift']['adaptations']} adaptations)")
                print(f"  During shift: {phases['during_shift']['mean_nmse']:.6f} NMSE ({phases['during_shift']['adaptations']} adaptations)")
                print(f"  Post-shift:   {phases['post_shift']['mean_nmse']:.6f} NMSE ({phases['post_shift']['adaptations']} adaptations)")
                
                # Calculate adaptation effectiveness
                if 'adaptation_effectiveness' in phases['during_shift']:
                    effectiveness = phases['during_shift']['adaptation_effectiveness']
                    print(f"\nAdaptation Effectiveness: {effectiveness:.6f} NMSE improvement during shift")
                    
                    if effectiveness > 0.01:
                        print("  ✅ EXCELLENT: Strong adaptation to domain shift")
                    elif effectiveness > 0.005:
                        print("  👍 GOOD: Moderate adaptation to domain shift")
                    elif effectiveness > 0:
                        print("  ⚠️  WEAK: Limited adaptation to domain shift")
                    else:
                        print("  ❌ POOR: No adaptation or performance degraded")
                
                # Create domain shift plots
                shift_plot_dir = Path(f"main_algorithm_v2/online/eval/domain_shift_analysis/shift_{args.primary_domain}_{args.shift_domain}_{args.trigger}")
                plot_domain_shift_performance(shift_analysis, shift_df, shift_plot_dir, 
                                               args.primary_domain, args.shift_domain)
    
    # 3. Comparative Analysis
    if 'baseline' in results and 'domain_shift' in results:
        print(f"\n{'-'*60}")
        print("PHASE 3: COMPARATIVE ANALYSIS")
        print(f"{'-'*60}")
        
        baseline_analysis, _ = results['baseline']
        shift_analysis, _ = results['domain_shift']
        
        # Compare baseline domain performance with shift experiment
        if args.primary_domain in baseline_analysis['domains']:
            baseline_nmse = baseline_analysis['domains'][args.primary_domain]['mean_nmse']
            shift_pre_nmse = shift_analysis['phases']['pre_shift']['mean_nmse']
            shift_during_nmse = shift_analysis['phases']['during_shift']['mean_nmse']
            
            primary_letter = DOMAIN_ID_TO_LETTER.get(args.primary_domain, f'D{args.primary_domain}')
            print(f"\nDomain {primary_letter} Performance Comparison:")
            print(f"  Baseline (normal):        {baseline_nmse:.6f} NMSE")
            print(f"  Shift exp (pre-shift):    {shift_pre_nmse:.6f} NMSE")
            print(f"  Shift exp (during shift): {shift_during_nmse:.6f} NMSE")
            
            performance_degradation = shift_during_nmse - shift_pre_nmse
            print(f"  Performance degradation:  {performance_degradation:.6f} NMSE")
            
            if performance_degradation > 0.02:
                print("  📊 ANALYSIS: Significant performance drop during shift detected")
            elif performance_degradation > 0.01:
                print("  📊 ANALYSIS: Moderate performance drop during shift")
            else:
                print("  📊 ANALYSIS: Minimal performance impact from shift")
    
    # Final Summary
    print(f"\n{'='*80}")
    print("DYNAMIC DOMAIN SHIFT EVALUATION SUMMARY")
    print(f"{'='*80}")
    
    if 'baseline' in results:
        print("✅ Baseline evaluation completed")
        print(f"   📊 Results: main_algorithm_v2/online/eval/domain_shift_analysis/baseline_{args.trigger}/")
    
    if 'domain_shift' in results:
        print("✅ Domain shift evaluation completed")
        primary_letter = DOMAIN_ID_TO_LETTER.get(args.primary_domain, f'D{args.primary_domain}')
        shift_letter = DOMAIN_ID_TO_LETTER.get(args.shift_domain, f'D{args.shift_domain}')
        print(f"   📊 Results: main_algorithm_v2/online/eval/domain_shift_analysis/shift_{primary_letter}_{shift_letter}_{args.trigger}/")
    
    print(f"\n[SUCCESS] Dynamic domain shift evaluation completed!")
    print("\nKey insights from this evaluation:")
    print("- Baseline shows normal adaptation behavior across all domains")
    print("- Domain shift reveals adaptation capability under mislabeling")
    print("- Temporal shifts test robustness to dynamic domain changes")
    print("- Performance vs samples plots show clear adaptation phases")
    print("\nUse the generated plots to visualize:")
    print("- How quickly the system detects and adapts to domain shifts")
    print("- Whether performance recovers after the shift ends")
    print("- Comparison between normal and mislabeled performance")
    
    # Generate cross-trigger comparison table if we have baseline results
    if 'baseline' in results:
        generate_trigger_comparison_table(args.trigger)

    # Diagnose domain distribution
    diagnose_domain_distribution(f"baseline_all_domains_{args.trigger}")

if __name__ == "__main__":
    main() 