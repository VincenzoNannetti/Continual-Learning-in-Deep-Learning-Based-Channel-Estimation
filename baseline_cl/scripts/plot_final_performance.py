"""
Publication-Quality Plotting Script for Final Performance Comparison of Baseline Continual Learning Methods

This script creates comprehensive plots comparing EWC, Experience Replay across all domains
using publication-quality styling and comprehensive analysis.
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

# Professional color palette - IEEE style
METHOD_COLOURS = {
    'ewc': '#2c3e50',  # Professional dark blue-grey
    'EWC': '#2c3e50',
    'experience_replay': '#3498db',  # Professional blue
    'Experience Replay': '#3498db',
    'L2 Regularisation': '#27ae60',  # Professional green
    'No Regularisation': '#f39c12',  # Professional orange
    'Standard Training': '#8e44ad',  # Professional purple
    'Joint Training': '#e74c3c',     # Professional red
}

# Method name mapping for cleaner display
METHOD_NAME_MAPPING = {
    'ewc': 'EWC',
    'experience_replay': 'Experience Replay',
    'EWC': 'EWC',
    'Experience Replay': 'Experience Replay',
    'Experience_Replay': 'Experience Replay'  # Add mapping for underscore version
}

# Domain name mapping for cleaner labels
DOMAIN_LABELS = {
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
    Uses only the most recent file for each method.
    
    Returns:
        DataFrame with columns: Method, Domain, SSIM, NMSE, PSNR, etc.
    """
    csv_files = list(Path(results_dir).glob("*_final_evaluation_*.csv"))
    
    if not csv_files:
        raise FileNotFoundError(f"No evaluation CSV files found in {results_dir}")
    
    print(f"Found {len(csv_files)} evaluation files:")
    for file in csv_files:
        print(f"  - {file.name}")
    
    # Group files by method and select the most recent for each
    method_files = {}
    for file in csv_files:
        if 'ewc_final_evaluation' in file.name:
            method = 'ewc'
        elif 'experience_replay_final_evaluation' in file.name:
            method = 'experience_replay'
        else:
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
    
    # Load and concatenate only the selected CSV files
    dfs = []
    for file in selected_files:
        df = pd.read_csv(file)
        dfs.append(df)
    
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
    
    plt.figure(figsize=(12, 8))
    
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
    heatmap_df = pd.DataFrame(heatmap_data, 
                             index=methods, 
                             columns=[DOMAIN_LABELS.get(d, d.replace('domain_', '').replace('_linear', '')) 
                                    for d in ordered_domains])
    
    # Choose colormap based on metric
    if metric.upper() == 'NMSE':
        cmap = 'Reds_r'  # Lower is better for NMSE
        cbar_label = f'{metric.upper()} (lower is better)'
        title = f'Final Performance Heatmap: {metric.upper()} across All Domains\n(Darker colours = better performance)'
    else:
        cmap = 'Blues'  # Higher is better for PSNR/SSIM
        cbar_label = f'{metric.upper()} (higher is better)'
        title = f'Final Performance Heatmap: {metric.upper()} across All Domains\n(Darker colours = better performance)'
    
    # Create heatmap with annotations (better formatting for NMSE)
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
    
    output_path = Path(output_dir) / f'performance_heatmap_{metric.lower()}.svg'
    plt.savefig(output_path, format='svg')
    plt.close()
    
    print(f"Saved performance heatmap ({metric}) to {output_path}")

def create_comprehensive_comparison_plot(df: pd.DataFrame, output_dir: str):
    """Create comprehensive subplot comparison of all metrics with publication quality."""
    
    # Apply IEEE styling
    plt.style.use('default')
    plt.rcParams.update(IEEE_STYLE)
    
    metrics = ['SSIM', 'NMSE', 'PSNR']
    
    # Create domain ordering
    domains = df['Domain'].unique()
    ordered_domains = create_domain_order(domains)
    ordered_domains = [d for d in ordered_domains if d in domains]
    
    methods = sorted(df['Method'].unique())
    x_pos = np.arange(len(ordered_domains))
    width = 0.35
    
    for metric_idx, metric in enumerate(metrics):
        # Create individual figure for each metric
        fig, ax = plt.subplots(figsize=(8, 4))
        
        for i, method in enumerate(methods):
            method_data = df[df['Method'] == method]
            
            values = []
            errors = []
            for domain in ordered_domains:
                domain_data = method_data[method_data['Domain'] == domain]
                if len(domain_data) > 0:
                    values.append(domain_data[metric].iloc[0])
                    if f'{metric}_std' in domain_data.columns:
                        errors.append(domain_data[f'{metric}_std'].iloc[0])
                    else:
                        errors.append(0)
                else:
                    values.append(0)
                    errors.append(0)
            
            x_positions = x_pos + i * width
            color = METHOD_COLOURS.get(method, 'gray')
            bars = ax.bar(x_positions, values, width, 
                         label=method, yerr=errors, capsize=2, alpha=0.8,
                         color=color, edgecolor='black', linewidth=0.5)
        
        # Customize subplot
        ax.set_xlabel('Domain', fontsize=10, fontweight='bold')
        ax.set_ylabel(f'{metric}', fontsize=10, fontweight='bold')
        
        # Set x-axis labels with better formatting
        ax.set_xticks(x_pos + width * (len(methods) - 1) / 2)
        domain_labels = [DOMAIN_LABELS.get(d, d.replace('domain_', '').replace('_linear', '')) 
                        for d in ordered_domains]
        ax.set_xticklabels(domain_labels, rotation=45, ha='right', fontsize=8)
        
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # Set appropriate y-axis limits
        if metric.upper() == 'NMSE':
            ax.set_yscale('log')
        
        plt.tight_layout()
        
        # Save individual plot
        output_path = Path(output_dir) / f'{metric.lower()}_comparison.svg'
        plt.savefig(output_path, format='svg')
        plt.close()
        
        print(f"Saved {metric} comparison plot to {output_path}")

def create_individual_metric_plots(df: pd.DataFrame, output_dir: str):
    """Create separate IEEE-style bar charts for each method and each metric."""
    
    # Apply IEEE styling
    plt.style.use('default')
    plt.rcParams.update(IEEE_STYLE)
    
    metrics = ['SSIM', 'NMSE', 'PSNR']
    methods = sorted(df['Method'].unique())
    
    # Create domain ordering
    domains = df['Domain'].unique()
    ordered_domains = create_domain_order(domains)
    ordered_domains = [d for d in ordered_domains if d in domains]
    
    for method in methods:
        # Create method subdirectory
        safe_method_name = method.lower().replace(' ', '_')
        method_dir = Path(output_dir) / safe_method_name
        method_dir.mkdir(parents=True, exist_ok=True)
        
        method_data = df[df['Method'] == method]
        method_color = METHOD_COLOURS.get(method, '#2c3e50')
        
        for metric in metrics:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            values = []
            errors = []
            domain_labels = []
            
            for domain in ordered_domains:
                domain_data = method_data[method_data['Domain'] == domain]
                if len(domain_data) > 0:
                    values.append(domain_data[metric].iloc[0])
                    if f'{metric}_std' in domain_data.columns:
                        errors.append(domain_data[f'{metric}_std'].iloc[0])
                    else:
                        errors.append(0)
                    domain_labels.append(DOMAIN_LABELS.get(domain, domain.replace('domain_', '').replace('_linear', '')))
                else:
                    values.append(0)
                    errors.append(0)
                    domain_labels.append(DOMAIN_LABELS.get(domain, domain.replace('domain_', '').replace('_linear', '')))
            
            # Create professional bars
            bars = ax.bar(range(len(values)), values, 
                         color=method_color, alpha=0.8, 
                         edgecolor='black', linewidth=0.8,
                         yerr=errors, capsize=5, ecolor='black')
            
            # Add professional value labels with error margins
            def autolabel(bars, values, errors):
                for bar, value, error in zip(bars, values, errors):
                    height = bar.get_height()
                    if height > 0:
                        if metric.upper() == 'NMSE':
                            label_text = f'{value:.2e}±{error:.2e}'
                        else:
                            label_text = f'{value:.3f}±{error:.3f}'
                        ax.annotate(label_text,
                                   xy=(bar.get_x() + bar.get_width() / 2, height),
                                   xytext=(0, 4),  
                                   textcoords="offset points",
                                   ha='center', va='bottom', fontsize=7,
                                   bbox=dict(facecolor='white', edgecolor='none', 
                                           alpha=0.8, boxstyle='round,pad=0.2'))
            
            autolabel(bars, values, errors)
            
            # Professional styling
            ax.set_xlabel('Domain', fontweight='bold')
            ax.set_ylabel(f'{metric}', fontweight='bold')
            
            # Set appropriate title based on metric
            if metric.upper() == 'NMSE':
                ax.set_title(f'{method}: Normalised Mean Squared Error Across All Domains')
                ax.set_yscale('log')
            elif metric.upper() == 'PSNR':
                ax.set_title(f'{method}: Peak Signal-to-Noise Ratio Across All Domains')
            else:
                ax.set_title(f'{method}: Structural Similarity Index Across All Domains')
            
            ax.set_xticks(range(len(domain_labels)))
            ax.set_xticklabels(domain_labels, rotation=45, ha='right')
            ax.grid(True, which='major', axis='y')
            
            plt.tight_layout()
            
            # Save as SVG
            output_path = method_dir / f'{metric.lower()}.svg'
            plt.savefig(output_path, format='svg')
            plt.close()
            
            print(f"Saved {method} {metric} plot to {output_path}")

def create_legacy_individual_method_plots(df: pd.DataFrame, output_dir: str):
    """Create the original multi-metric plots for backward compatibility."""
    
    # Apply IEEE styling
    plt.style.use('default')
    plt.rcParams.update(IEEE_STYLE)
    
    metrics = ['SSIM', 'NMSE', 'PSNR']
    methods = sorted(df['Method'].unique())
    
    # Create domain ordering
    domains = df['Domain'].unique()
    ordered_domains = create_domain_order(domains)
    ordered_domains = [d for d in ordered_domains if d in domains]
    
    for method in methods:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(f'{method} Performance Across All Domains', fontweight='bold')
        
        method_data = df[df['Method'] == method]
        method_color = METHOD_COLOURS.get(method, '#2c3e50')
        
        for metric_idx, metric in enumerate(metrics):
            ax = axes[metric_idx]
            
            values = []
            errors = []
            domain_labels = []
            
            for domain in ordered_domains:
                domain_data = method_data[method_data['Domain'] == domain]
                if len(domain_data) > 0:
                    values.append(domain_data[metric].iloc[0])
                    if f'{metric}_std' in domain_data.columns:
                        errors.append(domain_data[f'{metric}_std'].iloc[0])
                    else:
                        errors.append(0)
                    domain_labels.append(DOMAIN_LABELS.get(domain, domain.replace('domain_', '').replace('_linear', '')))
                else:
                    values.append(0)
                    errors.append(0)
                    domain_labels.append(DOMAIN_LABELS.get(domain, domain.replace('domain_', '').replace('_linear', '')))
            
            # Create bars with method-specific colour
            bars = ax.bar(range(len(values)), values, 
                         color=method_color, alpha=0.8, 
                         edgecolor='black', linewidth=0.8,
                         yerr=errors, capsize=3, ecolor='black')
            
            # Customize subplot
            ax.set_xlabel('Domain', fontweight='bold')
            ax.set_ylabel(f'{metric}', fontweight='bold')
            ax.set_title(f'{metric}', fontweight='bold')
            ax.set_xticks(range(len(domain_labels)))
            ax.set_xticklabels(domain_labels, rotation=45, ha='right')
            ax.grid(True, which='major', axis='y')
            
            # Set log scale for NMSE
            if metric.upper() == 'NMSE':
                ax.set_yscale('log')
        
        plt.tight_layout()
        
        # Save plot
        safe_method_name = method.lower().replace(' ', '_')
        output_path = Path(output_dir) / f'{safe_method_name}_performance.svg'
        plt.savefig(output_path, format='svg')
        plt.close()
        
        print(f"Saved {method} performance plot to {output_path}")

def create_method_comparison_summary(df: pd.DataFrame, output_dir: str):
    """Create a summary comparing both methods side-by-side."""
    
    # Apply IEEE styling
    plt.style.use('default')
    plt.rcParams.update(IEEE_STYLE)
    
    metrics = ['SSIM', 'NMSE', 'PSNR']
    methods = sorted(df['Method'].unique())
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Method Comparison: Average Performance Across All Domains', 
                 fontsize=16, fontweight='bold')
    
    for metric_idx, metric in enumerate(metrics):
        ax = axes[metric_idx]
        
        method_means = []
        method_stds = []
        method_names = []
        colors = []
        
        for method in methods:
            method_data = df[df['Method'] == method]
            mean_val = method_data[metric].mean()
            std_val = method_data[metric].std()
            
            method_means.append(mean_val)
            method_stds.append(std_val)
            method_names.append(method)
            colors.append(METHOD_COLOURS.get(method, 'gray'))
        
        bars = ax.bar(range(len(method_means)), method_means, 
                     color=colors, alpha=0.8, edgecolor='black', linewidth=0.5,
                     yerr=method_stds, capsize=5)
        
        # Add value labels
        for i, bar in enumerate(bars):
            height = bar.get_height()
            if metric.upper() == 'NMSE':
                label_text = f'{height:.2e}'
            else:
                label_text = f'{height:.3f}'
            ax.text(bar.get_x() + bar.get_width()/2., height + method_stds[i],
                   label_text, ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        ax.set_xlabel('Method', fontsize=12, fontweight='bold')
        ax.set_ylabel(f'Average {metric}', fontsize=12, fontweight='bold')
        ax.set_title(f'{metric}', fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(method_names)))
        ax.set_xticklabels(method_names, fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # Set log scale for NMSE
        if metric.upper() == 'NMSE':
            ax.set_yscale('log')
    
    plt.tight_layout()
    
    output_path = Path(output_dir) / 'method_comparison_summary.svg'
    plt.savefig(output_path, format='svg')
    plt.close()
    
    print(f"Saved method comparison summary to {output_path}")

def create_domain_difficulty_analysis(df: pd.DataFrame, output_dir: str):
    """Analyse which domains are most challenging across all methods."""
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    metrics = ['SSIM', 'NMSE', 'PSNR']
    domains = df['Domain'].unique()
    ordered_domains = create_domain_order(domains)
    ordered_domains = [d for d in ordered_domains if d in domains]
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        # Calculate average performance for each domain across all methods
        domain_avg_performance = []
        domain_labels = []
        
        for domain in ordered_domains:
            domain_data = df[df['Domain'] == domain]
            avg_perf = domain_data[metric].mean()
            domain_avg_performance.append(avg_perf)
            domain_labels.append(DOMAIN_LABELS.get(domain, domain.replace('domain_', '').replace('_linear', '')))
        
        # Sort domains by difficulty (ascending for PSNR/SSIM, descending for NMSE)
        if metric.upper() == 'NMSE':
            sorted_indices = np.argsort(domain_avg_performance)[::-1]  # Descending for NMSE
            title_suffix = "(Higher NMSE = More Difficult)"
        else:
            sorted_indices = np.argsort(domain_avg_performance)  # Ascending for PSNR/SSIM  
            title_suffix = f"(Lower {metric} = More Difficult)"
        
        sorted_performance = [domain_avg_performance[idx] for idx in sorted_indices]
        sorted_labels = [domain_labels[idx] for idx in sorted_indices]
        
        # Create gradient colours (red = difficult, green = easy)
        colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(sorted_performance)))
        if metric.upper() == 'NMSE':
            colors = colors[::-1]  # Reverse for NMSE
        
        bars = ax.bar(range(len(sorted_performance)), sorted_performance, 
                     color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
        
        # Add value labels
        for j, bar in enumerate(bars):
            height = bar.get_height()
            if metric.upper() == 'NMSE':
                label_text = f'{height:.2e}'
            else:
                label_text = f'{height:.3f}'
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   label_text, ha='center', va='bottom', fontsize=8, rotation=45)
        
        ax.set_xlabel('Domain (Sorted by Difficulty)', fontsize=10, fontweight='bold')
        ax.set_ylabel(f'Average {metric}', fontsize=10, fontweight='bold')
        ax.set_title(f'{metric} Domain Difficulty\n{title_suffix}', fontsize=11, fontweight='bold')
        ax.set_xticks(range(len(sorted_labels)))
        ax.set_xticklabels(sorted_labels, rotation=45, ha='right', fontsize=8)
        ax.grid(True, alpha=0.3)
        
        if metric.upper() == 'NMSE':
            ax.set_yscale('log')
    
    fig.suptitle('Domain Difficulty Analysis: Average Performance Across All Methods', 
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    output_path = Path(output_dir) / 'domain_difficulty_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved domain difficulty analysis to {output_path}")

def generate_performance_table(df: pd.DataFrame, output_dir: str):
    """Generate formatted performance tables."""
    
    # Create domain ordering
    domains = df['Domain'].unique()
    ordered_domains = create_domain_order(domains)
    ordered_domains = [d for d in ordered_domains if d in domains]
    
    methods = sorted(df['Method'].unique())
    
    # Create table for each metric
    for metric in ['SSIM', 'NMSE', 'PSNR']:
        print(f"\n{'='*100}")
        print(f"FINAL PERFORMANCE TABLE: {metric}")
        print(f"{'='*100}")
        
        # Create table
        table_data = []
        for domain in ordered_domains:
            row = {'Domain': DOMAIN_LABELS.get(domain, domain.replace('domain_', '').replace('_linear', ''))}
            for method in methods:
                method_domain_data = df[(df['Method'] == method) & (df['Domain'] == domain)]
                if len(method_domain_data) > 0:
                    mean_val = method_domain_data[metric].iloc[0]
                    if f'{metric}_std' in method_domain_data.columns:
                        std_val = method_domain_data[f'{metric}_std'].iloc[0]
                        if metric.upper() == 'NMSE':
                            row[method] = f"{mean_val:.2e} ± {std_val:.2e}"
                        else:
                            row[method] = f"{mean_val:.4f} ± {std_val:.4f}"
                    else:
                        if metric.upper() == 'NMSE':
                            row[method] = f"{mean_val:.2e}"
                        else:
                            row[method] = f"{mean_val:.4f}"
                else:
                    row[method] = "N/A"
            table_data.append(row)
        
        # Add aggregate row
        agg_row = {'Domain': 'AVERAGE'}
        for method in methods:
            method_data = df[df['Method'] == method]
            mean_val = method_data[metric].mean()
            std_val = method_data[metric].std()
            if metric.upper() == 'NMSE':
                agg_row[method] = f"{mean_val:.2e} ± {std_val:.2e}"
            else:
                agg_row[method] = f"{mean_val:.4f} ± {std_val:.4f}"
        table_data.append(agg_row)
        
        # Convert to DataFrame and print
        table_df = pd.DataFrame(table_data)
        print(table_df.to_string(index=False))
        
        # Save to CSV
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        csv_filename = f"performance_table_{metric.lower()}_{timestamp}.csv"
        csv_filepath = os.path.join(output_dir, csv_filename)
        table_df.to_csv(csv_filepath, index=False)
        print(f"Table saved: {csv_filepath}")

def load_continual_learning_metrics(results_base_dir: str) -> pd.DataFrame:
    """
    Load continual learning metrics from training results.
    Uses only the most recent summary file for each method.
    
    Args:
        results_base_dir: Base results directory containing method subdirectories
    
    Returns:
        DataFrame with continual learning metrics
    """
    base_path = Path(results_base_dir).parent
    eval_path = Path(results_base_dir)
    
    # Look for summary CSV files in all possible locations
    all_summary_files = {
        'ewc': [],
        'experience_replay': []
    }
    
    # Check method subdirectories
    ewc_dir = base_path / "ewc"
    if ewc_dir.exists():
        ewc_files = list(ewc_dir.glob("ewc_summary_*.csv"))
        all_summary_files['ewc'].extend(ewc_files)
    
    er_dir = base_path / "experience_replay"
    if er_dir.exists():
        er_files = list(er_dir.glob("Experience_Replay_summary_*.csv"))
        all_summary_files['experience_replay'].extend(er_files)
    
    # Check evaluation directory
    ewc_summary_files = list(eval_path.glob("ewc_summary*.csv"))
    er_summary_files = list(eval_path.glob("experience_replay_summary*.csv"))
    
    all_summary_files['ewc'].extend(ewc_summary_files)
    all_summary_files['experience_replay'].extend(er_summary_files)
    
    # Select only the most recent file for each method
    cl_data = []
    
    for method, files in all_summary_files.items():
        if files:
            latest_file = max(files, key=lambda x: x.stat().st_mtime)
            df = pd.read_csv(latest_file)
            cl_data.append(df)
            print(f"Using most recent {method} summary: {latest_file.name}")
        else:
            print(f"No summary files found for {method}")
    
    if not cl_data:
        print("Warning: No continual learning metrics found")
        return pd.DataFrame()
    
    combined_cl_df = pd.concat(cl_data, ignore_index=True)
    
    # Apply method name mapping to ensure consistency
    combined_cl_df['Method'] = combined_cl_df['Method'].map(METHOD_NAME_MAPPING).fillna(combined_cl_df['Method'])
    
    print(f"Loaded continual learning metrics for: {', '.join(combined_cl_df['Method'].unique())}")
    
    return combined_cl_df

def create_continual_learning_metrics_plot(cl_df: pd.DataFrame, output_dir: str):
    """Create BWT vs FWT analysis plot with scatter and bar chart panels."""
    
    if cl_df.empty:
        print("No continual learning metrics to plot")
        return
    
    # Apply IEEE styling
    plt.style.use('default')
    plt.rcParams.update(IEEE_STYLE)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Continual Learning Performance Analysis', fontweight='bold', fontsize=14)
    
    methods = cl_df['Method'].tolist()
    bwt_values = cl_df['Backward_Transfer_BWT'].tolist()
    fwt_values = cl_df['Forward_Transfer_FWT'].tolist()
    
    # Enhanced color palette for better visual distinction
    enhanced_colors = ['#2E86C1', '#E74C3C']  # Professional blue and red
    colors = enhanced_colors[:len(methods)]
    
    # Left plot: BWT vs FWT scatter
    for i, (method, bwt, fwt) in enumerate(zip(methods, bwt_values, fwt_values)):
        color = colors[i] if i < len(colors) else '#999999'
        ax1.scatter(bwt, fwt, c=color, s=120, alpha=0.8,
                   edgecolors='black', linewidth=1.5, label=method)
    
    # Add quadrant lines
    ax1.axhline(0, color='gray', linestyle='--', alpha=0.6, linewidth=1.5)
    ax1.axvline(0, color='gray', linestyle='--', alpha=0.6, linewidth=1.5)
    
    # Add quadrant labels - Updated for NMSE interpretation
    ax1.text(0.95, 0.95, 'Forgetting\nPoor Transfer', transform=ax1.transAxes, 
            ha='right', va='top', fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcoral', alpha=0.4))
    ax1.text(0.05, 0.05, 'No Forgetting\nGood Transfer', transform=ax1.transAxes,
            ha='left', va='bottom', fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.4))
    
    ax1.set_xlabel('Backward Transfer (BWT)\nPositive = Forgetting', fontweight='bold', fontsize=11)
    ax1.set_ylabel('Forward Transfer (FWT)\nNegative = Good Generalisation', fontweight='bold', fontsize=11)
    ax1.set_title('Transfer Learning Analysis\nBWT vs FWT (NMSE-based)', fontweight='bold', fontsize=12)
    ax1.legend(loc='best', frameon=True, fancybox=True, fontsize=10)
    ax1.grid(True, alpha=0.4)
    
    # Right plot: Model comparison bars
    x_pos = np.arange(len(methods))
    width = 0.35
    
    bars1 = ax2.bar(x_pos - width/2, bwt_values, width,
                   label='Backward Transfer (BWT)', color='#E74C3C', alpha=0.8,
                   edgecolor='black', linewidth=1)
    bars2 = ax2.bar(x_pos + width/2, fwt_values, width,
                   label='Forward Transfer (FWT)', color='#2E86C1', alpha=0.8,
                   edgecolor='black', linewidth=1)
    
    # Add value labels
    for bar, value in zip(bars1, bwt_values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:+.3f}', ha='center', 
                va='bottom' if height >= 0 else 'top', fontsize=10, fontweight='bold')
    
    for bar, value in zip(bars2, fwt_values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:+.3f}', ha='center',
                va='bottom' if height >= 0 else 'top', fontsize=10, fontweight='bold')
    
    ax2.set_xlabel('Method', fontweight='bold', fontsize=11)
    ax2.set_ylabel('Transfer Score', fontweight='bold', fontsize=11)
    ax2.set_title('Continual Learning Performance\nComparison (NMSE-based)', fontweight='bold', fontsize=12)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(methods, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.axhline(0, color='black', linewidth=1.2)
    ax2.grid(True, alpha=0.4, axis='y')
    
    plt.tight_layout()
    
    output_path = Path(output_dir) / 'continual_learning_metrics.svg'
    plt.savefig(output_path, format='svg')
    plt.close()
    
    print(f"Saved enhanced continual learning metrics plot to {output_path}")

def create_bwt_fwt_table(cl_df: pd.DataFrame, eval_df: pd.DataFrame, output_dir: str):
    """Create comprehensive BWT/FWT table for academic reports."""
    
    if cl_df.empty:
        print("No continual learning metrics for table")
        return
    
    print(f"\n{'='*80}")
    print("CONTINUAL LEARNING METRICS TABLE")
    print(f"{'='*80}")
    
    # Create summary table
    methods = cl_df['Method'].tolist()
    
    # Calculate final performance statistics if available
    final_performance = {}
    if not eval_df.empty:
        for method in methods:
            method_data = eval_df[eval_df['Method'] == method]
            if len(method_data) > 0:
                final_performance[method] = {
                    'SSIM': (method_data['SSIM'].mean(), method_data['SSIM'].std()),
                    'NMSE': (method_data['NMSE'].mean(), method_data['NMSE'].std()),
                    'PSNR': (method_data['PSNR'].mean(), method_data['PSNR'].std())
                }
    
    # Print LaTeX table
    print("\n=== LaTeX Table for Academic Papers ===")
    print("\\begin{table}[htbp]")
    print("\\centering")
    print("\\caption{Continual Learning Performance Analysis}")
    print("\\begin{tabular}{|l|c|c|c|c|c|c|}")
    print("\\hline")
    print("\\textbf{Method} & \\textbf{BWT} & \\textbf{FWT} & \\textbf{Forgetting} & \\textbf{SSIM} & \\textbf{NMSE} & \\textbf{PSNR} \\\\")
    print("& \\textbf{(±std)} & \\textbf{(±std)} & \\textbf{(\\%)} & \\textbf{(mean±std)} & \\textbf{(mean±std)} & \\textbf{(mean±std)} \\\\")
    print("\\hline")
    
    for _, row in cl_df.iterrows():
        method = row['Method'].replace('_', '\\_').replace(' ', '\\ ')
        bwt = f"{row['Backward_Transfer_BWT']:+.3f}"
        fwt = f"{row['Forward_Transfer_FWT']:+.3f}"
        forgetting = f"{row['Forgetting']:.3f}"
        
        # Add final performance if available
        if method.replace('\\_', '_').replace('\\ ', ' ') in final_performance:
            perf = final_performance[method.replace('\\_', '_').replace('\\ ', ' ')]
            ssim_str = f"{perf['SSIM'][0]:.3f}±{perf['SSIM'][1]:.3f}"
            nmse_str = f"{perf['NMSE'][0]:.2e}±{perf['NMSE'][1]:.2e}"
            psnr_str = f"{perf['PSNR'][0]:.1f}±{perf['PSNR'][1]:.1f}"
        else:
            ssim_str = "N/A"
            nmse_str = "N/A"
            psnr_str = "N/A"
        
        print(f"{method} & {bwt} & {fwt} & {forgetting} & {ssim_str} & {nmse_str} & {psnr_str} \\\\")
    
    print("\\hline")
    print("\\end{tabular}")
    print("\\label{tab:continual_learning_analysis}")
    print("\\note{BWT: Backward Transfer, FWT: Forward Transfer. Negative BWT indicates forgetting, positive FWT indicates beneficial knowledge transfer.}")
    print("\\end{table}")
    
    # Print summary statistics
    print(f"\n{'='*80}")
    print("SUMMARY STATISTICS")
    print(f"{'='*80}")
    
    print(f"\nContinual Learning Performance:")
    print(f"{'Method':<20} {'BWT':<12} {'FWT':<12} {'Forgetting (%)':<15}")
    print(f"{'-'*65}")
    
    for _, row in cl_df.iterrows():
        method = row['Method']
        bwt = row['Backward_Transfer_BWT']
        fwt = row['Forward_Transfer_FWT']
        forgetting = row['Forgetting']
        
        print(f"{method:<20} {bwt:+8.3f}    {fwt:+8.3f}    {forgetting:8.1f}%")
    
    # Determine winners
    best_bwt = cl_df.loc[cl_df['Backward_Transfer_BWT'].idxmax(), 'Method']
    best_fwt = cl_df.loc[cl_df['Forward_Transfer_FWT'].idxmax(), 'Method']
    least_forgetting = cl_df.loc[cl_df['Forgetting'].idxmin(), 'Method']
    
    print(f"\nKey Findings:")
    print(f"• Best Backward Transfer (least forgetting): {best_bwt}")
    print(f"• Best Forward Transfer: {best_fwt}")
    print(f"• Least catastrophic forgetting: {least_forgetting}")
    
    if not eval_df.empty:
        # Final performance analysis
        print(f"\nFinal Performance Analysis:")
        for metric in ['SSIM', 'NMSE', 'PSNR']:
            if metric == 'NMSE':
                best_method = eval_df.groupby('Method')[metric].mean().idxmin()
            else:
                best_method = eval_df.groupby('Method')[metric].mean().idxmax()
            print(f"• Best final {metric}: {best_method}")
    
    # Save table to file
    table_path = Path(output_dir) / 'continual_learning_table.txt'
    with open(table_path, 'w') as f:
        f.write("CONTINUAL LEARNING METRICS TABLE\n")
        f.write("="*80 + "\n\n")
        for _, row in cl_df.iterrows():
            f.write(f"Method: {row['Method']}\n")
            f.write(f"  BWT: {row['Backward_Transfer_BWT']:+.3f}\n")
            f.write(f"  FWT: {row['Forward_Transfer_FWT']:+.3f}\n")
            f.write(f"  Forgetting: {row['Forgetting']:.3f}\n\n")
    
    print(f"\nTable also saved to: {table_path}")

def print_detailed_cl_statistics(cl_df: pd.DataFrame, eval_df: pd.DataFrame):
    """Print detailed continual learning statistics with metric-specific BWT/FWT."""
    
    if cl_df.empty:
        print("No continual learning metrics to analyze")
        return
    
    print(f"\n{'='*80}")
    print(f"DETAILED CONTINUAL LEARNING ANALYSIS")
    print(f"{'='*80}")
    
    methods = cl_df['Method'].tolist()
    
    # Debug: Print method names from both DataFrames
    print(f"\nDEBUG INFO:")
    print(f"  CL methods: {methods}")
    print(f"  Eval methods: {eval_df['Method'].unique().tolist() if not eval_df.empty else 'No eval data'}")
    
    # Print overall CL metrics from training
    print(f"\nCONTINUAL LEARNING METRICS (from training process):")
    print(f"{'Method':<20} {'BWT':<12} {'FWT':<12} {'Forgetting':<12}")
    print(f"{'-'*60}")
    
    for _, row in cl_df.iterrows():
        method = row['Method']
        bwt = row['Backward_Transfer_BWT']
        fwt = row['Forward_Transfer_FWT']
        forgetting = row['Forgetting']
        
        print(f"{method:<20} {bwt:+8.3f}    {fwt:+8.3f}    {forgetting:8.3f}")
    
    # Calculate metric-specific statistics if evaluation data is available
    if not eval_df.empty:
        print(f"\nFINAL PERFORMANCE STATISTICS (average across all domains):")
        
        for metric in ['SSIM', 'NMSE', 'PSNR']:
            print(f"\n{metric} Performance:")
            print(f"{'Method':<20} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12}")
            print(f"{'-'*80}")
            
            for method in methods:
                method_data = eval_df[eval_df['Method'] == method]
                if len(method_data) > 0:
                    mean_val = method_data[metric].mean()
                    std_val = method_data[metric].std()
                    min_val = method_data[metric].min()
                    max_val = method_data[metric].max()
                    
                    if metric == 'NMSE':
                        print(f"{method:<20} {mean_val:.2e}   {std_val:.2e}   {min_val:.2e}   {max_val:.2e}")
                    else:
                        print(f"{method:<20} {mean_val:8.4f}    {std_val:8.4f}    {min_val:8.4f}    {max_val:8.4f}")
                else:
                    print(f"{method:<20} {'NO DATA FOUND'}")
        
        # Domain-wise performance analysis
        print(f"\nDOMAIN-WISE PERFORMANCE ANALYSIS:")
        domains = eval_df['Domain'].unique()
        
        for metric in ['SSIM', 'NMSE']:  # Focus on key metrics
            print(f"\n{metric} by Domain:")
            print(f"{'Domain':<25}", end="")
            for method in methods:
                print(f"{method:<15}", end="")
            print()
            print(f"{'-'*90}")
            
            for domain in sorted(domains):
                domain_label = DOMAIN_LABELS.get(domain, domain.replace('domain_', '').replace('_linear', ''))
                print(f"{domain_label:<25}", end="")
                
                for method in methods:
                    method_domain_data = eval_df[(eval_df['Method'] == method) & (eval_df['Domain'] == domain)]
                    if len(method_domain_data) > 0:
                        value = method_domain_data[metric].iloc[0]
                        if metric == 'NMSE':
                            print(f"{value:.2e}    ", end="")
                        else:
                            print(f"{value:8.4f}    ", end="")
                    else:
                        print(f"{'N/A':<15}", end="")
                print()
    
    # Winner summary
    print(f"\nPERFORMANCE SUMMARY:")
    if not cl_df.empty:
        bwt_winner = cl_df.loc[cl_df['Backward_Transfer_BWT'].idxmax(), 'Method']
        fwt_winner = cl_df.loc[cl_df['Forward_Transfer_FWT'].idxmax(), 'Method']
        forgetting_winner = cl_df.loc[cl_df['Forgetting'].idxmin(), 'Method']  # Lower is better
        
        print(f"   Best BWT (less forgetting):     {bwt_winner}")
        print(f"   Best FWT (better transfer):     {fwt_winner}")
        print(f"   Least forgetting:               {forgetting_winner}")
    
    if not eval_df.empty:
        ssim_winner = eval_df.groupby('Method')['SSIM'].mean().idxmax()
        nmse_winner = eval_df.groupby('Method')['NMSE'].mean().idxmin()  # Lower is better
        psnr_winner = eval_df.groupby('Method')['PSNR'].mean().idxmax()
        
        print(f"   Best final SSIM:                {ssim_winner}")
        print(f"   Best final NMSE:                {nmse_winner}")
        print(f"   Best final PSNR:                {psnr_winner}")
    
    print(f"{'='*80}")

def main():
    """Main plotting function."""
    parser = argparse.ArgumentParser(description="Create Publication-Quality Plots for Final Performance Comparison")
    parser.add_argument('--results_dir', type=str, 
                       default='baseline_cl/results/final_evaluation',
                       help="Directory containing evaluation CSV files")
    parser.add_argument('--output_dir', type=str, 
                       default='baseline_cl/results/final_evaluation/plots',
                       help="Output directory for plots")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Creating Publication-Quality Final Performance Plots")
    print(f"   Results directory: {args.results_dir}")
    print(f"   Output directory: {args.output_dir}")
    
    # Load evaluation results
    try:
        df = load_evaluation_results(args.results_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure to run the evaluation script first!")
        return
    
    # Create performance heatmaps for each metric
    for metric in ['SSIM', 'NMSE', 'PSNR']:
        create_performance_heatmap(df, metric, args.output_dir)
    
    # Create comprehensive comparison plot
    create_comprehensive_comparison_plot(df, args.output_dir)
    
    # Create individual metric plots (separate SVG files organized by method)
    create_individual_metric_plots(df, args.output_dir)
    
    # Create legacy combined plots for backward compatibility
    create_legacy_individual_method_plots(df, args.output_dir)
    
    # Create method comparison summary
    create_method_comparison_summary(df, args.output_dir)
    
    # Create domain difficulty analysis
    create_domain_difficulty_analysis(df, args.output_dir)
    
    # Generate performance tables
    generate_performance_table(df, args.output_dir)
    
    # Load continual learning metrics
    cl_df = load_continual_learning_metrics(args.results_dir)
    
    # Create continual learning metrics plot
    create_continual_learning_metrics_plot(cl_df, args.output_dir)
    
    # Create BWT/FWT table
    create_bwt_fwt_table(cl_df, df, args.output_dir)
    
    # Print detailed statistics
    print_detailed_cl_statistics(cl_df, df)
    
    print(f"\nAll publication-quality plots created successfully!")
    print(f"   Output directory: {args.output_dir}")
    print(f"   Generated plots for: {', '.join(df['Method'].unique())}")

if __name__ == '__main__':
    main() 