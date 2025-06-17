"""
Publication-Quality Plotting Script for LoRA Continual Learning Performance Analysis

This script creates comprehensive plots for LoRA-based continual learning results,
compatible with baseline plotting but enhanced for LoRA-specific analysis.
"""

import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import json
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# IEEE-style configuration for publication-quality plots
IEEE_STYLE = {
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
    'axes.titlesize': 10,
    'axes.labelsize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.titlesize': 12,
    'lines.linewidth': 1.0,
    'lines.markersize': 4,
    'axes.grid': True,
    'grid.linestyle': ':',
    'grid.linewidth': 0.5,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.format': 'svg',
    'mathtext.fontset': 'custom',
    'mathtext.rm': 'Times New Roman',
    'mathtext.it': 'Times New Roman:italic',
    'mathtext.bf': 'Times New Roman:bold',
}

# Enhanced color palette including LoRA
METHOD_COLOURS = {
    'ewc': '#2c3e50',  # Professional dark blue-grey
    'EWC': '#2c3e50',
    'experience_replay': '#3498db',  # Professional blue
    'Experience Replay': '#3498db',
    'LoRA_EWC': '#e74c3c',  # Professional red for LoRA
    'LoRA-EWC': '#e74c3c',
    'LoRA': '#e74c3c',
    'L2 Regularisation': '#27ae60',  # Professional green
    'No Regularisation': '#f39c12',  # Professional orange
    'Standard Training': '#8e44ad',  # Professional purple
    'Joint Training': '#16a085',     # Professional teal
}

# Enhanced method name mapping
METHOD_NAME_MAPPING = {
    'ewc': 'EWC',
    'experience_replay': 'Experience Replay',
    'EWC': 'EWC',
    'Experience Replay': 'Experience Replay',
    'Experience_Replay': 'Experience Replay',
    'LoRA_EWC': 'LoRA-EWC',
    'lora_ewc': 'LoRA-EWC',
    'LoRA': 'LoRA-EWC'
}

# Domain name mapping for cleaner labels
DOMAIN_LABELS = {
    'domain_domain_high_snr_med_linear_cl': 'High SNR\nMed Speed',
    'domain_domain_high_snr_slow_linear_cl': 'High SNR\nSlow Speed',
    'domain_domain_high_snr_fast_linear_cl': 'High SNR\nFast Speed',
    'domain_domain_med_snr_med_linear_cl': 'Med SNR\nMed Speed',
    'domain_domain_med_snr_slow_linear_cl': 'Med SNR\nSlow Speed', 
    'domain_domain_med_snr_fast_linear_cl': 'Med SNR\nFast Speed',
    'domain_domain_low_snr_med_linear_cl': 'Low SNR\nMed Speed',
    'domain_domain_low_snr_slow_linear_cl': 'Low SNR\nSlow Speed',
    'domain_domain_low_snr_fast_linear_cl': 'Low SNR\nFast Speed',
    # Original baseline domain names
    'domain_high_snr_med_linear_cl': 'High SNR\nMed Speed',
    'domain_high_snr_slow_linear_cl': 'High SNR\nSlow Speed',
    'domain_high_snr_fast_linear_cl': 'High SNR\nFast Speed',
    'domain_med_snr_med_linear_cl': 'Med SNR\nMed Speed',
    'domain_med_snr_slow_linear_cl': 'Med SNR\nSlow Speed', 
    'domain_med_snr_fast_linear_cl': 'Med SNR\nFast Speed',
    'domain_low_snr_med_linear_cl': 'Low SNR\nMed Speed',
    'domain_low_snr_slow_linear_cl': 'Low SNR\nSlow Speed',
    'domain_low_snr_fast_linear_cl': 'Low SNR\nFast Speed'
}

def load_evaluation_results(results_dir: str) -> pd.DataFrame:
    """
    Load evaluation results from CSV files in the results directory.
    Enhanced to handle LoRA results along with baseline methods.
    
    Returns:
        DataFrame with columns: Method, Domain, SSIM, NMSE, PSNR, etc.
    """
    csv_files = list(Path(results_dir).glob("*_final_evaluation_*.csv"))
    
    if not csv_files:
        raise FileNotFoundError(f"No evaluation CSV files found in {results_dir}")
    
    print(f"Found {len(csv_files)} evaluation files:")
    for file in csv_files:
        print(f"  - {file.name}")
    
    # Enhanced method detection for LoRA
    method_files = {}
    for file in csv_files:
        if 'lora_ewc_final_evaluation' in file.name:
            method = 'lora_ewc'
        elif 'ewc_final_evaluation' in file.name and 'lora' not in file.name:
            method = 'ewc'
        elif 'experience_replay_final_evaluation' in file.name:
            method = 'experience_replay'
        else:
            # Try to detect method from content
            try:
                df_temp = pd.read_csv(file)
                if 'Method' in df_temp.columns:
                    methods_in_file = df_temp['Method'].unique()
                    if any('lora' in m.lower() for m in methods_in_file):
                        method = 'lora_ewc'
                    elif any('ewc' in m.lower() for m in methods_in_file):
                        method = 'ewc'
                    elif any('replay' in m.lower() for m in methods_in_file):
                        method = 'experience_replay'
                    else:
                        print(f"Unknown method in {file.name}: {methods_in_file}")
                        continue
                else:
                    print(f"No Method column in {file.name}")
                    continue
            except Exception as e:
                print(f"Error reading {file.name}: {e}")
                continue
        
        if method not in method_files:
            method_files[method] = []
        method_files[method].append(file)
    
    # Select most recent file for each method
    selected_files = []
    for method, files in method_files.items():
        latest_file = max(files, key=lambda x: x.stat().st_mtime)
        selected_files.append(latest_file)
        print(f"Using most recent {method} file: {latest_file.name}")
    
    # Load and concatenate the selected CSV files
    dfs = []
    for file in selected_files:
        df = pd.read_csv(file)
        dfs.append(df)
    
    if not dfs:
        raise ValueError("No valid CSV files could be loaded")
    
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Apply method name mapping for cleaner display
    combined_df['Method'] = combined_df['Method'].map(METHOD_NAME_MAPPING).fillna(combined_df['Method'])
    
    print(f"\nLoaded data:")
    print(f"  Methods: {combined_df['Method'].unique()}")
    print(f"  Domains: {len(combined_df['Domain'].unique())}")
    print(f"  Total rows: {len(combined_df)}")
    
    return combined_df

def create_domain_order(domains):
    """Create a logical ordering for domains based on their names."""
    # Define ordering by SNR level and speed
    snr_order = ['high_snr', 'med_snr', 'low_snr']
    speed_order = ['slow', 'med', 'fast']
    
    ordered_domains = []
    
    # Group by SNR first, then by speed
    for snr in snr_order:
        for speed in speed_order:
            pattern = f"{snr}_{speed}_linear"
            matching = [d for d in domains if pattern in d]
            ordered_domains.extend(sorted(matching))
    
    # Add any remaining domains
    remaining = [d for d in domains if d not in ordered_domains]
    ordered_domains.extend(sorted(remaining))
    
    return ordered_domains

def create_performance_heatmap(df: pd.DataFrame, metric: str, output_dir: str):
    """Create heatmap showing performance across all domains and methods."""
    # Apply IEEE styling
    plt.style.use('default')
    plt.rcParams.update(IEEE_STYLE)
    
    plt.figure(figsize=(14, 8))
    
    # Create domain ordering
    domains = df['Domain'].unique()
    ordered_domains = create_domain_order(domains)
    ordered_domains = [d for d in ordered_domains if d in domains]
    
    methods = df['Method'].unique()
    
    # Create heatmap data
    heatmap_data = []
    for method in methods:
        row_data = []
        for domain in ordered_domains:
            domain_data = df[(df['Method'] == method) & (df['Domain'] == domain)]
            if len(domain_data) > 0:
                value = domain_data[metric].iloc[0]
                row_data.append(value)
            else:
                row_data.append(np.nan)
        heatmap_data.append(row_data)
    
    # Create heatmap
    domain_display_names = [DOMAIN_LABELS.get(d, d.replace('domain_', '').replace('_linear', '').replace('domain_', '')) 
                          for d in ordered_domains]
    
    heatmap_df = pd.DataFrame(heatmap_data, 
                             index=methods, 
                             columns=domain_display_names)
    
    # Choose colormap based on metric
    if metric.upper() == 'NMSE':
        cmap = 'Reds_r'  # Lower is better for NMSE
        cbar_label = f'{metric.upper()} (lower is better)'
        title = f'Final Performance Heatmap: {metric.upper()} across All Domains\n(Darker colours = better performance)'
    else:
        cmap = 'Blues'  # Higher is better for PSNR/SSIM
        cbar_label = f'{metric.upper()} (higher is better)'
        title = f'Final Performance Heatmap: {metric.upper()} across All Domains\n(Darker colours = better performance)'
    
    # Create heatmap with annotations
    if metric.upper() == 'NMSE':
        # Use scientific notation for NMSE
        annot_data = heatmap_df.map(lambda x: f'{x:.2e}' if not pd.isna(x) else '')
        sns.heatmap(heatmap_df, annot=annot_data, fmt='', cmap=cmap,
                    cbar_kws={'label': cbar_label}, linewidths=0.5)
    else:
        sns.heatmap(heatmap_df, annot=True, fmt='.3f', cmap=cmap,
                    cbar_kws={'label': cbar_label}, linewidths=0.5)
    
    plt.title(title, fontweight='bold', pad=20)
    plt.xlabel('Domain', fontweight='bold', labelpad=10)
    plt.ylabel('Continual Learning Method', fontweight='bold', labelpad=10)
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    
    output_path = Path(output_dir) / f'lora_performance_heatmap_{metric.lower()}.svg'
    plt.savefig(output_path, format='svg')
    plt.close()
    
    print(f"Saved LoRA performance heatmap ({metric}) to {output_path}")

def create_comprehensive_comparison_plot(df: pd.DataFrame, output_dir: str):
    """Create comprehensive subplot comparison of all metrics with LoRA emphasis."""
    
    # Apply IEEE styling
    plt.style.use('default')
    plt.rcParams.update(IEEE_STYLE)
    
    metrics = ['SSIM', 'NMSE', 'PSNR']
    
    # Create domain ordering
    domains = df['Domain'].unique()
    ordered_domains = create_domain_order(domains)
    ordered_domains = [d for d in ordered_domains if d in domains]
    
    methods = sorted(df['Method'].unique())
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    for metric_idx, metric in enumerate(metrics):
        if metric_idx < 2:
            ax = fig.add_subplot(gs[0, metric_idx])
        else:
            ax = fig.add_subplot(gs[1, :])  # PSNR gets full width
        
        # Prepare data for bar plot
        x_pos = np.arange(len(ordered_domains))
        width = 0.8 / len(methods)
        
        for method_idx, method in enumerate(methods):
            values = []
            for domain in ordered_domains:
                domain_data = df[(df['Method'] == method) & (df['Domain'] == domain)]
                if len(domain_data) > 0:
                    values.append(domain_data[metric].iloc[0])
                else:
                    values.append(0)
            
            # Get color for method
            color = METHOD_COLOURS.get(method, '#7f7f7f')
            
            # Create bars
            bars = ax.bar(x_pos + method_idx * width, values, width, 
                         label=method, color=color, alpha=0.8)
        
        # Customization
        domain_display_names = [DOMAIN_LABELS.get(d, d.replace('domain_', '').replace('_linear', '').replace('domain_', ''))
                              for d in ordered_domains]
        
        ax.set_xlabel('Domain', fontweight='bold')
        ax.set_ylabel(f'{metric}', fontweight='bold')
        ax.set_title(f'{metric} Performance Comparison', fontweight='bold')
        ax.set_xticks(x_pos + width * (len(methods) - 1) / 2)
        ax.set_xticklabels(domain_display_names, rotation=45, ha='right')
        ax.legend(loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('LoRA Continual Learning: Comprehensive Performance Analysis', 
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    output_path = Path(output_dir) / 'lora_comprehensive_comparison.svg'
    plt.savefig(output_path, format='svg')
    plt.close()
    
    print(f"Saved comprehensive comparison plot to {output_path}")

def create_method_performance_summary(df: pd.DataFrame, output_dir: str):
    """Create summary statistics table and plot for all methods."""
    
    plt.style.use('default')
    plt.rcParams.update(IEEE_STYLE)
    
    metrics = ['SSIM', 'NMSE', 'PSNR']
    methods = df['Method'].unique()
    
    # Calculate summary statistics
    summary_data = []
    for method in methods:
        method_data = df[df['Method'] == method]
        
        row = {'Method': method}
        for metric in metrics:
            values = method_data[metric].values
            row[f'{metric}_mean'] = np.mean(values)
            row[f'{metric}_std'] = np.std(values)
            row[f'{metric}_min'] = np.min(values)
            row[f'{metric}_max'] = np.max(values)
        
        summary_data.append(row)
    
    summary_df = pd.DataFrame(summary_data)
    
    # Save summary table
    summary_path = Path(output_dir) / 'lora_method_summary.csv'
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved method summary to {summary_path}")
    
    # Create summary plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        means = summary_df[f'{metric}_mean'].values
        stds = summary_df[f'{metric}_std'].values
        method_names = summary_df['Method'].values
        
        colors = [METHOD_COLOURS.get(method, '#7f7f7f') for method in method_names]
        
        bars = ax.bar(method_names, means, yerr=stds, capsize=5, 
                     color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
        
        ax.set_title(f'{metric} Performance Summary', fontweight='bold')
        ax.set_ylabel(f'{metric}', fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Rotate x-axis labels for better readability
        ax.set_xticklabels(method_names, rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, mean, std in zip(bars, means, stds):
            height = bar.get_height()
            if metric == 'NMSE':
                ax.text(bar.get_x() + bar.get_width()/2., height + std,
                       f'{mean:.2e}', ha='center', va='bottom', fontsize=8)
            else:
                ax.text(bar.get_x() + bar.get_width()/2., height + std,
                       f'{mean:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.suptitle('Method Performance Summary with Standard Deviations', 
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    output_path = Path(output_dir) / 'lora_method_summary_plot.svg'
    plt.savefig(output_path, format='svg')
    plt.close()
    
    print(f"Saved method summary plot to {output_path}")
    
    return summary_df

def load_continual_learning_metrics(results_dir: str) -> pd.DataFrame:
    """Load continual learning metrics (BWT, FWT) from CSV files."""
    
    cl_files = list(Path(results_dir).glob("*continual_learning_summary*.csv"))
    
    if not cl_files:
        print("No continual learning summary files found")
        return None
    
    print(f"Found {len(cl_files)} continual learning files:")
    for file in cl_files:
        print(f"  - {file.name}")
    
    # Load most recent file
    latest_file = max(cl_files, key=lambda x: x.stat().st_mtime)
    print(f"Using most recent CL file: {latest_file.name}")
    
    df = pd.read_csv(latest_file)
    
    # Apply method name mapping
    df['Method'] = df['Method'].map(METHOD_NAME_MAPPING).fillna(df['Method'])
    
    return df

def create_continual_learning_analysis(cl_df: pd.DataFrame, output_dir: str):
    """Create continual learning specific analysis plots."""
    
    if cl_df is None or cl_df.empty:
        print("No continual learning data available for analysis")
        return
    
    plt.style.use('default')
    plt.rcParams.update(IEEE_STYLE)
    
    # BWT vs FWT scatter plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    methods = cl_df['Method'].unique()
    
    # Plot 1: BWT vs FWT
    for method in methods:
        method_data = cl_df[cl_df['Method'] == method]
        if len(method_data) > 0:
            bwt = method_data['Backward_Transfer_BWT'].iloc[0]
            fwt = method_data['Forward_Transfer_FWT'].iloc[0]
            color = METHOD_COLOURS.get(method, '#7f7f7f')
            
            ax1.scatter(bwt, fwt, color=color, s=100, alpha=0.8, 
                       edgecolors='black', linewidth=1, label=method)
    
    ax1.axhline(y=0, color='grey', linestyle='--', alpha=0.5)
    ax1.axvline(x=0, color='grey', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Backward Transfer (BWT)', fontweight='bold')
    ax1.set_ylabel('Forward Transfer (FWT)', fontweight='bold')
    ax1.set_title('Continual Learning: BWT vs FWT', fontweight='bold')
    ax1.legend(framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Final Performance vs Forgetting
    for method in methods:
        method_data = cl_df[cl_df['Method'] == method]
        if len(method_data) > 0:
            final_perf = method_data['Final_Average_Performance'].iloc[0]
            forgetting = method_data.get('Average_Forgetting', method_data.get('Forgetting', [0])).iloc[0]
            color = METHOD_COLOURS.get(method, '#7f7f7f')
            
            ax2.scatter(forgetting, final_perf, color=color, s=100, alpha=0.8,
                       edgecolors='black', linewidth=1, label=method)
    
    ax2.set_xlabel('Average Forgetting', fontweight='bold')
    ax2.set_ylabel('Final Average Performance', fontweight='bold')
    ax2.set_title('Performance vs Forgetting Trade-off', fontweight='bold')
    ax2.legend(framealpha=0.9)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = Path(output_dir) / 'lora_continual_learning_analysis.svg'
    plt.savefig(output_path, format='svg')
    plt.close()
    
    print(f"Saved continual learning analysis to {output_path}")

def main():
    """Main function to create all LoRA performance plots."""
    
    parser = argparse.ArgumentParser(description="Create LoRA continual learning performance plots")
    parser.add_argument('--results_dir', type=str, required=True,
                       help='Directory containing CSV result files')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory to save generated plots')
    parser.add_argument('--method_filter', type=str, default=None,
                       help='Filter to specific method (e.g., LoRA_EWC)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"LoRA CONTINUAL LEARNING PLOTTING PIPELINE")
    print(f"Results directory: {args.results_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Load evaluation results
        print("\n--- Loading Evaluation Results ---")
        df = load_evaluation_results(args.results_dir)
        
        # Apply method filter if specified
        if args.method_filter:
            original_len = len(df)
            df = df[df['Method'].str.contains(args.method_filter, case=False, na=False)]
            print(f"Applied method filter '{args.method_filter}': {original_len} -> {len(df)} rows")
        
        if df.empty:
            print("No data available after filtering")
            return
        
        # Create performance plots
        print("\n--- Creating Performance Plots ---")
        
        # Heatmaps for each metric
        for metric in ['SSIM', 'NMSE', 'PSNR']:
            create_performance_heatmap(df, metric, args.output_dir)
        
        # Comprehensive comparison
        create_comprehensive_comparison_plot(df, args.output_dir)
        
        # Method summary
        summary_df = create_method_performance_summary(df, args.output_dir)
        
        # Continual learning analysis
        print("\n--- Loading Continual Learning Metrics ---")
        cl_df = load_continual_learning_metrics(args.results_dir)
        if cl_df is not None:
            create_continual_learning_analysis(cl_df, args.output_dir)
        
        print(f"\n--- Plotting Complete ---")
        print(f"All plots saved to: {args.output_dir}")
        
        # List generated files
        output_path = Path(args.output_dir)
        plot_files = list(output_path.glob("*.svg")) + list(output_path.glob("*.png"))
        csv_files = list(output_path.glob("*.csv"))
        
        if plot_files:
            print(f"\nGenerated plots ({len(plot_files)}):")
            for file in sorted(plot_files):
                print(f"  - {file.name}")
        
        if csv_files:
            print(f"\nGenerated summaries ({len(csv_files)}):")
            for file in sorted(csv_files):
                print(f"  - {file.name}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 