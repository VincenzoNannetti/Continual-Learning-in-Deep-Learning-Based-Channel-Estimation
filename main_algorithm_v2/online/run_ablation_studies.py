import yaml
import subprocess
from pathlib import Path
import json
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import copy
import os

def load_config(config_path):
    """Load the base configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def save_config(config, save_path):
    """Save configuration to file."""
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

def create_ablation_configs(base_config):
    """Create different ablation study configurations."""
    configs = {}
    
    # Baseline - Ensure EWC is enabled for a fair comparison
    configs['baseline'] = copy.deepcopy(base_config)
    configs['baseline']['experiment_name'] = 'ablation_baseline'
    configs['baseline']['online_training']['ewc']['enabled'] = True  # Enable EWC for baseline
    
    # No EWC
    configs['no_ewc'] = copy.deepcopy(base_config)
    configs['no_ewc']['experiment_name'] = 'ablation_no_ewc'
    configs['no_ewc']['online_training']['ewc']['enabled'] = False
    
    # No Buffer (very small buffer)
    configs['no_buffer'] = copy.deepcopy(base_config)
    configs['no_buffer']['experiment_name'] = 'ablation_no_buffer'
    configs['no_buffer']['online_evaluation']['online_buffer']['size'] = 1  # Minimal buffer
    configs['no_buffer']['online_evaluation']['online_buffer']['enabled'] = False
    
    # No LoRA - Use a regular model checkpoint instead
    # NOTE: This requires a non-LoRA checkpoint to be available
    configs['no_lora'] = copy.deepcopy(base_config)
    configs['no_lora']['experiment_name'] = 'ablation_no_lora'
    # Switch to a regular checkpoint if available, otherwise skip this ablation
    regular_checkpoint = "main_algorithm_v2/offline/checkpoints/regular/FINAL_MODEL.pth"
    if os.path.exists(regular_checkpoint):
        configs['no_lora']['offline_checkpoint_path'] = regular_checkpoint
        print(f"    No-LoRA ablation will use regular checkpoint: {regular_checkpoint}")
    else:
        print(f"   ️  No regular checkpoint found at {regular_checkpoint}")
        print(f"    No-LoRA ablation will be skipped unless regular checkpoint is available")
        configs['no_lora']['_skip'] = True  # Mark for skipping
    
    # No EWC and No Buffer (combined ablation)
    configs['no_ewc_no_buffer'] = copy.deepcopy(base_config)
    configs['no_ewc_no_buffer']['experiment_name'] = 'ablation_no_ewc_no_buffer'
    configs['no_ewc_no_buffer']['online_training']['ewc']['enabled'] = False
    configs['no_ewc_no_buffer']['online_evaluation']['online_buffer']['size'] = 1
    configs['no_ewc_no_buffer']['online_evaluation']['online_buffer']['enabled'] = False
    
    # Reduced training (lower trigger rate)
    configs['reduced_training'] = copy.deepcopy(base_config)
    configs['reduced_training']['experiment_name'] = 'ablation_reduced_training'
    # Increase trigger thresholds to reduce training frequency
    configs['reduced_training']['online_training']['trigger']['max_samples'] = 200  # 10x increase
    configs['reduced_training']['online_training']['trigger']['max_time'] = 300.0   # 10x increase
    
    # No mixed batching (only online samples)
    configs['no_mixed_batch'] = copy.deepcopy(base_config)
    configs['no_mixed_batch']['experiment_name'] = 'ablation_no_mixed_batch'
    configs['no_mixed_batch']['online_training']['mixed_batch']['m_offline'] = 0  # No offline samples
    configs['no_mixed_batch']['online_training']['mixed_batch']['m_online'] = 16  # All online samples
    
    # High-frequency training (aggressive updating)
    configs['aggressive_training'] = copy.deepcopy(base_config)
    configs['aggressive_training']['experiment_name'] = 'ablation_aggressive_training'
    configs['aggressive_training']['online_training']['trigger']['max_samples'] = 5   # Very frequent updates
    configs['aggressive_training']['online_training']['trigger']['max_time'] = 5.0   # Very frequent updates
    configs['aggressive_training']['online_training']['max_epochs_per_trigger'] = 5  # More epochs per update
    
    return configs

def run_experiment(config_path, experiment_name, num_samples=1000):
    """Run a single experiment with the given configuration."""
    print(f"\n{'='*60}")
    print(f"RUNNING ABLATION EXPERIMENT: {experiment_name.upper()}")
    print(f"{'='*60}")
    
    try:
        # Run the online learning script with reduced samples for ablation
        cmd = [
            "python", "-m", "main_algorithm_v2.online.online_continual_learning",
            "--config", str(config_path),
            "--num_samples", str(num_samples)
        ]
        
        print(f"Command: {' '.join(cmd)}")
        # Set encoding to handle Unicode characters (emojis) in Windows
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, 
                               encoding='utf-8', errors='replace', env=env)
        
        print(f" Experiment {experiment_name} completed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f" Experiment {experiment_name} failed:")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return False

def collect_results(experiment_name):
    """Collect results from an experiment."""
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
    print(f"Loading results from: {latest_csv}")
    
    try:
        df = pd.read_csv(latest_csv)
        # Ensure required columns exist
        required_columns = ['nmse_post', 'triggered', 'domain_id']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"Warning: Missing required columns {missing_columns} in {latest_csv}")
            print(f"Available columns: {list(df.columns)}")
            return None
            
        return df
    except Exception as e:
        print(f"Error loading {latest_csv}: {e}")
        return None

def plot_ablation_comparison(results_dict, save_dir):
    """Create comparison plots for ablation studies."""
    if not results_dict:
        print("No results to plot for ablation study")
        return
        
    # Create directory for ablation plots
    ablation_dir = save_dir / "ablation_comparison"
    ablation_dir.mkdir(exist_ok=True)
    
    # Filter out None results
    valid_results = {name: df for name, df in results_dict.items() if df is not None}
    
    if not valid_results:
        print("No valid results for ablation comparison")
        return
    
    # Define all possible metrics to analyse
    metrics_to_plot = ['nmse_post']
    if any('ssim_post' in df.columns for df in valid_results.values()):
        metrics_to_plot.append('ssim_post')
    if any('psnr_post' in df.columns for df in valid_results.values()):
        metrics_to_plot.append('psnr_post')
    
    # 1. Overall Performance Comparison (Multiple Metrics)
    fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(6*len(metrics_to_plot), 6))
    if len(metrics_to_plot) == 1:
        axes = [axes]
    
    for idx, metric in enumerate(metrics_to_plot):
        data = []
        for name, df in valid_results.items():
            if metric in df.columns:
                data.append({
                    'Configuration': name.replace('_', ' ').title(),
                    'Value': df[metric].mean(),
                    'Std': df[metric].std(),
                    'Metric': metric.replace('_post', '').upper()
                })
        
        if data:
            comparison_df = pd.DataFrame(data)
            
            # Create bar plot with error bars
            bars = axes[idx].bar(comparison_df['Configuration'], comparison_df['Value'], 
                               yerr=comparison_df['Std'], capsize=5, alpha=0.8)
            
            axes[idx].set_xlabel('Configuration', fontsize=12)
            metric_name = metric.replace('_post', '').upper()
            better_direction = "lower" if metric_name == "NMSE" else "higher"
            axes[idx].set_ylabel(f'Average {metric_name} ({better_direction} = better)', fontsize=12)
            axes[idx].set_title(f'Ablation Study: {metric_name} Performance', fontsize=14)
            axes[idx].tick_params(axis='x', rotation=45)
            axes[idx].grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, val in zip(bars, comparison_df['Value']):
                if metric_name == "NMSE":
                    label = f'{val:.5f}'
                else:
                    label = f'{val:.4f}'
                axes[idx].text(bar.get_x() + bar.get_width()/2, 
                             bar.get_height() + bar.get_height()*0.02,
                             label, ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(ablation_dir / 'overall_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Training Events Analysis
    plt.figure(figsize=(15, 10))
    
    training_data = []
    for name, df in valid_results.items():
        total_samples = len(df)
        training_events = df['triggered'].sum() if 'triggered' in df.columns else 0
        success_rate = 0
        improvement_magnitude = 0
        
        if training_events > 0 and 'nmse_pre' in df.columns and 'nmse_post' in df.columns:
            training_df = df[df['triggered'] == 1].copy()
            improvements = training_df['nmse_pre'] > training_df['nmse_post']
            success_rate = improvements.mean() * 100
            
            # Calculate average improvement magnitude
            relative_improvements = (training_df['nmse_pre'] - training_df['nmse_post']) / training_df['nmse_pre']
            improvement_magnitude = relative_improvements[relative_improvements > 0].mean() * 100
        
        training_data.append({
            'Configuration': name.replace('_', ' ').title(),
            'Training Events': training_events,
            'Training Rate %': (training_events / total_samples) * 100 if total_samples > 0 else 0,
            'Success Rate %': success_rate,
            'Avg Improvement %': improvement_magnitude
        })
    
    training_df = pd.DataFrame(training_data)
    
    # Create subplot layout for training analysis
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Training Events
    bars1 = ax1.bar(training_df['Configuration'], training_df['Training Events'], 
                    alpha=0.8, color='orange')
    ax1.set_ylabel('Number of Training Events', fontsize=12)
    ax1.set_title('Training Events per Configuration', fontsize=14)
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)
    
    for bar, val in zip(bars1, training_df['Training Events']):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(training_df['Training Events'])*0.01,
                f'{int(val)}', ha='center', va='bottom', fontsize=9)
    
    # Training Rate
    bars2 = ax2.bar(training_df['Configuration'], training_df['Training Rate %'], 
                    alpha=0.8, color='blue')
    ax2.set_ylabel('Training Rate (%)', fontsize=12)
    ax2.set_title('Training Frequency per Configuration', fontsize=14)
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    
    for bar, val in zip(bars2, training_df['Training Rate %']):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(training_df['Training Rate %'])*0.01,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
    
    # Success Rate
    bars3 = ax3.bar(training_df['Configuration'], training_df['Success Rate %'], 
                    alpha=0.8, color='green')
    ax3.set_ylabel('Success Rate (%)', fontsize=12)
    ax3.set_title('Training Success Rate per Configuration', fontsize=14)
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3)
    
    for bar, val in zip(bars3, training_df['Success Rate %']):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(training_df['Success Rate %'])*0.01,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
    
    # Improvement Magnitude
    bars4 = ax4.bar(training_df['Configuration'], training_df['Avg Improvement %'], 
                    alpha=0.8, color='purple')
    ax4.set_ylabel('Average Improvement (%)', fontsize=12)
    ax4.set_title('Average Performance Improvement per Configuration', fontsize=14)
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(True, alpha=0.3)
    
    for bar, val in zip(bars4, training_df['Avg Improvement %']):
        if val > 0:  # Only show positive improvements
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(training_df['Avg Improvement %'])*0.01,
                    f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(ablation_dir / 'training_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Per-domain Performance Comparison
    if len(valid_results) > 1:
        plt.figure(figsize=(15, 8))
        
        # Prepare data for grouped bar plot
        all_domains = set()
        for df in valid_results.values():
            if 'domain_id' in df.columns:
                all_domains.update(df['domain_id'].unique())
        all_domains = sorted(all_domains)
        
        if all_domains:
            plot_data = []
            for name, df in valid_results.items():
                for domain in all_domains:
                    if 'domain_id' in df.columns:
                        domain_data = df[df['domain_id'] == domain]
                        if len(domain_data) > 0:
                            plot_data.append({
                                'Configuration': name.replace('_', ' ').title(),
                                'Domain': f'Domain {domain}',
                                'NMSE': domain_data['nmse_post'].mean()
                            })
            
            if plot_data:
                plot_df = pd.DataFrame(plot_data)
                
                # Create grouped bar plot
                sns.barplot(data=plot_df, x='Domain', y='NMSE', hue='Configuration')
                
                plt.xlabel('Domain', fontsize=12)
                plt.ylabel('Average NMSE', fontsize=12)
                plt.title('Ablation Study: Per-domain Performance Comparison', fontsize=14)
                plt.xticks(rotation=45, ha='right')
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                
                plt.savefig(ablation_dir / 'domain_comparison.png', dpi=300, bbox_inches='tight')
                plt.close()
    
    print(f"\n Ablation study plots saved to: {ablation_dir}")
    
    # Print comprehensive summary table
    print(f"\n COMPREHENSIVE ABLATION STUDY SUMMARY:")
    print("=" * 80)
    print(f"{'Configuration':<25} {'Avg NMSE':<12} {'Training':<10} {'Success%':<9} {'Improve%':<9}")
    print("-" * 80)
    
    for _, row in training_df.iterrows():
        config_name = row['Configuration']
        # Find NMSE for this configuration
        nmse_val = "N/A"
        for name, df in valid_results.items():
            if name.replace('_', ' ').title() == config_name:
                nmse_val = f"{df['nmse_post'].mean():.6f}"
                break
        
        training = int(row['Training Events'])
        success = row['Success Rate %']
        improve = row['Avg Improvement %']
        
        print(f"{config_name:<25} {nmse_val:<12} {training:<10} {success:<9.1f} {improve:<9.1f}")
    
    print("=" * 80)

def main():
    """Run ablation studies."""
    print(" STARTING COMPREHENSIVE ABLATION STUDIES")
    print("=" * 60)
    
    # Load base configuration
    base_config_path = "main_algorithm_v2/online/config/online_config.yaml"
    
    if not os.path.exists(base_config_path):
        print(f" Base config not found: {base_config_path}")
        print("Please ensure the config file exists and try again.")
        return
    
    try:
        base_config = load_config(base_config_path)
    except Exception as e:
        print(f" Error loading base config: {e}")
        return
    
    # Create ablation configurations
    configs = create_ablation_configs(base_config)
    
    # Filter out skipped configurations
    configs = {name: config for name, config in configs.items() 
               if not config.get('_skip', False)}
    
    # Create directory for ablation configs
    config_dir = Path("main_algorithm_v2/online/config/ablation_studies")
    config_dir.mkdir(exist_ok=True)
    
    # Run experiments and collect results
    results = {}
    num_ablation_samples = 800  # Increased for more reliable results
    
    print(f"\n Running {len(configs)} ablation experiments with {num_ablation_samples} samples each...")
    
    for name, config in configs.items():
        print(f"\n Preparing experiment: {name}")
        
        # Remove internal flags before saving
        config_clean = {k: v for k, v in config.items() if not k.startswith('_')}
        
        # Save config
        config_path = config_dir / f"{name}_config.yaml"
        save_config(config_clean, config_path)
        print(f"   Config saved: {config_path}")
        
        # Run experiment
        success = run_experiment(config_path, config_clean['experiment_name'], num_ablation_samples)
        
        if success:
            # Collect results
            results[name] = collect_results(config_clean['experiment_name'])
            if results[name] is not None:
                print(f"    Results collected: {len(results[name])} samples")
            else:
                print(f"   ️ No results found for {name}")
        else:
            print(f"    Experiment failed: {name}")
            results[name] = None
    
    # Create comparison plots
    print(f"\n Creating comprehensive comparison plots...")
    plot_ablation_comparison(results, Path("main_algorithm_v2/online/eval"))
    
    # Save summary results
    summary_file = Path("main_algorithm_v2/online/eval/ablation_comparison/ablation_summary.json")
    summary_data = {
        'timestamp': datetime.now().isoformat(),
        'num_samples': num_ablation_samples,
        'configurations': list(configs.keys()),
        'results_summary': {}
    }
    
    for name, df in results.items():
        if df is not None:
            summary_data['results_summary'][name] = {
                'total_samples': len(df),
                'avg_nmse': float(df['nmse_post'].mean()) if 'nmse_post' in df.columns else None,
                'std_nmse': float(df['nmse_post'].std()) if 'nmse_post' in df.columns else None,
                'training_events': int(df['triggered'].sum()) if 'triggered' in df.columns else 0
            }
    
    with open(summary_file, 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    print(f"\n Comprehensive ablation studies completed!")
    print(f" Results available in: main_algorithm_v2/online/eval/ablation_comparison/")
    print(f" Summary saved to: {summary_file}")

if __name__ == "__main__":
    main() 