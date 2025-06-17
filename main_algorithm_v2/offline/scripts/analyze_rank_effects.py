#!/usr/bin/env python3
"""
Analysis script for studying the effect of LoRA rank on performance across domains.
This script creates systematic experiments to understand rank-performance relationships.
"""

import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import sys
import os
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import copy

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from src.config import ExperimentConfig, load_config
from src.model import UNet_SRCNN_LoRA
from src.data import get_dataloaders
from src.utils import get_device, load_lora_model_for_evaluation
from train import evaluate_single_domain

def setup_plotting():
    """Set up matplotlib for publication-quality plots."""
    plt.style.use('default')
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 16,
        'lines.linewidth': 2,
        'lines.markersize': 8,
        'figure.figsize': (12, 8),
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight'
    })

def create_rank_configs(base_config_path: str, ranks: List[int], output_dir: str) -> List[str]:
    """
    Create configuration files for different LoRA ranks.
    
    Args:
        base_config_path: Path to base configuration
        ranks: List of ranks to test
        output_dir: Directory to save configurations
        
    Returns:
        List of paths to created configuration files
    """
    print(f"📋 Creating rank configurations for ranks: {ranks}")
    
    base_config = load_config(base_config_path)
    config_paths = []
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for rank in ranks:
        # Create new config with modified rank
        config_dict = base_config.to_dict()
        
        # Update all task ranks to the specified rank
        for task_id in config_dict['model']['params']['task_lora_ranks']:
            config_dict['model']['params']['task_lora_ranks'][task_id] = rank
            
        # Update LoRA alpha to match rank (common practice)
        for task_id in config_dict['model']['params']['task_lora_alphas']:
            config_dict['model']['params']['task_lora_alphas'][task_id] = rank
            
        # Save config
        config_filename = f"rank_{rank}_config.yaml"
        config_path = output_dir / config_filename
        
        # Write config to file
        import yaml
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
            
        config_paths.append(str(config_path))
        print(f"  ✅ Created config for rank {rank}: {config_path}")
    
    return config_paths

def evaluate_rank_performance(checkpoint_path: str, config_path: str, 
                            domain_ids: List[str], device: torch.device) -> Dict[str, Dict[str, float]]:
    """
    Evaluate model performance across domains for a specific rank.
    
    Args:
        checkpoint_path: Path to model checkpoint
        config_path: Path to configuration file
        domain_ids: List of domain IDs to evaluate
        device: Computing device
        
    Returns:
        Dictionary mapping domain_id -> metrics
    """
    config = load_config(config_path)
    
    try:
        # Load model
        model = load_lora_model_for_evaluation(checkpoint_path, config, device)
        model.eval()
        
        # Evaluate each domain
        results = {}
        for domain_id in domain_ids:
            try:
                domain_metrics = evaluate_single_domain(model, domain_id, config, device)
                results[domain_id] = domain_metrics
                print(f"    Domain {domain_id}: NMSE={domain_metrics.get('nmse', 0):.4f}, "
                      f"SSIM={domain_metrics.get('ssim', 0):.4f}")
            except Exception as e:
                print(f"    ⚠️ Failed to evaluate domain {domain_id}: {e}")
                results[domain_id] = {'nmse': float('inf'), 'ssim': 0.0, 'psnr': 0.0}
                
        return results
        
    except Exception as e:
        print(f"❌ Failed to load model from {checkpoint_path}: {e}")
        return {domain_id: {'nmse': float('inf'), 'ssim': 0.0, 'psnr': 0.0} 
                for domain_id in domain_ids}

def train_single_domain_rank_models(base_config_path: str, domain_ids: List[str], 
                                   ranks: List[int], output_dir: str) -> Dict[Tuple[str, int], str]:
    """
    Train single-domain models for each domain-rank combination.
    
    Args:
        base_config_path: Base configuration path
        domain_ids: List of domain IDs
        ranks: List of ranks to test
        output_dir: Output directory for checkpoints
        
    Returns:
        Dictionary mapping (domain_id, rank) -> checkpoint_path
    """
    import subprocess
    
    checkpoints = {}
    output_dir = Path(output_dir)
    
    for domain_id in domain_ids:
        for rank in ranks:
            print(f"\n🔄 Training single-domain model: Domain {domain_id}, Rank {rank}")
            
            # Create single-domain config
            config = load_config(base_config_path)
            config_dict = config.to_dict()
            
            # Modify for single domain training
            config_dict['data']['sequence'] = [int(domain_id)]
            config_dict['model']['params']['task_lora_ranks'] = {int(domain_id): rank}
            config_dict['model']['params']['task_lora_alphas'] = {int(domain_id): rank}
            
            # Save temporary config
            temp_config_path = output_dir / f"temp_domain_{domain_id}_rank_{rank}_config.yaml"
            import yaml
            with open(temp_config_path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False)
            
            # Define checkpoint path
            checkpoint_dir = output_dir / f"domain_{domain_id}_rank_{rank}"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            # Training command
            cmd = [
                sys.executable, "train.py",
                "--config", str(temp_config_path),
                "--output_dir", str(checkpoint_dir),
                "--wandb_project", f"rank_analysis_domain_{domain_id}",
                "--wandb_run_name", f"rank_{rank}_domain_{domain_id}"
            ]
            
            try:
                result = subprocess.run(cmd, cwd=current_dir, capture_output=True, text=True)
                if result.returncode == 0:
                    # Find the checkpoint file
                    checkpoint_files = list(checkpoint_dir.glob("*.pt"))
                    if checkpoint_files:
                        checkpoints[(domain_id, rank)] = str(checkpoint_files[0])
                        print(f"  ✅ Training completed: {checkpoint_files[0]}")
                    else:
                        print(f"  ⚠️ Training completed but no checkpoint found")
                else:
                    print(f"  ❌ Training failed: {result.stderr}")
                    
            except Exception as e:
                print(f"  ❌ Training error: {e}")
            
            # Clean up temporary config
            temp_config_path.unlink(missing_ok=True)
    
    return checkpoints

def analyze_rank_effects_from_checkpoints(checkpoints: Dict[Tuple[str, int], str],
                                        domain_ids: List[str], ranks: List[int],
                                        base_config_path: str, output_dir: str):
    """
    Analyze rank effects using pre-trained checkpoints.
    
    Args:
        checkpoints: Dictionary mapping (domain_id, rank) -> checkpoint_path
        domain_ids: List of domain IDs
        ranks: List of ranks
        base_config_path: Base configuration path
        output_dir: Output directory for results
    """
    device = get_device()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n🔍 Analyzing rank effects from {len(checkpoints)} checkpoints...")
    
    # Collect results
    results = []
    config_paths = create_rank_configs(base_config_path, ranks, output_dir / "configs")
    
    for domain_id in domain_ids:
        for rank in ranks:
            if (domain_id, rank) not in checkpoints:
                print(f"  ⚠️ Missing checkpoint for domain {domain_id}, rank {rank}")
                continue
                
            checkpoint_path = checkpoints[(domain_id, rank)]
            config_path = output_dir / "configs" / f"rank_{rank}_config.yaml"
            
            print(f"  📊 Evaluating Domain {domain_id}, Rank {rank}...")
            
            # Evaluate this model on ALL domains to see cross-domain effects
            domain_results = evaluate_rank_performance(
                checkpoint_path, str(config_path), domain_ids, device
            )
            
            for eval_domain_id, metrics in domain_results.items():
                results.append({
                    'train_domain': domain_id,
                    'eval_domain': eval_domain_id,
                    'rank': rank,
                    'nmse': metrics.get('nmse', float('inf')),
                    'ssim': metrics.get('ssim', 0.0),
                    'psnr': metrics.get('psnr', 0.0),
                    'is_target_domain': domain_id == eval_domain_id
                })
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Save results
    csv_path = output_dir / f"rank_analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(csv_path, index=False)
    print(f"💾 Results saved to: {csv_path}")
    
    # Create visualizations
    create_rank_analysis_plots(df, output_dir)
    
    return df

def create_rank_analysis_plots(df: pd.DataFrame, output_dir: Path):
    """Create comprehensive plots for rank analysis."""
    setup_plotting()
    
    print("\n📊 Creating rank analysis plots...")
    
    # Filter for target domain performance (training domain == evaluation domain)
    target_df = df[df['is_target_domain'] == True].copy()
    
    # 1. Rank vs Performance for each domain (line plot)
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('LoRA Rank Effects on Domain Performance', fontsize=16, fontweight='bold')
    
    # NMSE vs Rank
    ax1 = axes[0, 0]
    for domain in sorted(target_df['train_domain'].unique()):
        domain_data = target_df[target_df['train_domain'] == domain]
        ax1.plot(domain_data['rank'], domain_data['nmse'], 
                marker='o', label=f'Domain {domain}', linewidth=2)
    ax1.set_xlabel('LoRA Rank')
    ax1.set_ylabel('NMSE (lower is better)')
    ax1.set_title('(a) NMSE vs LoRA Rank')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # SSIM vs Rank
    ax2 = axes[0, 1]
    for domain in sorted(target_df['train_domain'].unique()):
        domain_data = target_df[target_df['train_domain'] == domain]
        ax2.plot(domain_data['rank'], domain_data['ssim'], 
                marker='s', label=f'Domain {domain}', linewidth=2)
    ax2.set_xlabel('LoRA Rank')
    ax2.set_ylabel('SSIM (higher is better)')
    ax2.set_title('(b) SSIM vs LoRA Rank')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    # Rank efficiency: Performance per parameter
    ax3 = axes[1, 0]
    # Approximate parameter count: rank * (in_features + out_features) per layer
    # For visualization, use relative efficiency
    for domain in sorted(target_df['train_domain'].unique()):
        domain_data = target_df[target_df['train_domain'] == domain]
        # Efficiency: SSIM per rank (higher is better)
        efficiency = domain_data['ssim'] / domain_data['rank']
        ax3.plot(domain_data['rank'], efficiency, 
                marker='^', label=f'Domain {domain}', linewidth=2)
    ax3.set_xlabel('LoRA Rank')
    ax3.set_ylabel('SSIM per Rank Unit (efficiency)')
    ax3.set_title('(c) Parameter Efficiency')
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax3.grid(True, alpha=0.3)
    
    # Performance variance across ranks
    ax4 = axes[1, 1]
    rank_variance = target_df.groupby('rank').agg({
        'nmse': ['mean', 'std'],
        'ssim': ['mean', 'std']
    })
    
    ranks = rank_variance.index
    nmse_std = rank_variance[('nmse', 'std')]
    ssim_std = rank_variance[('ssim', 'std')]
    
    ax4_twin = ax4.twinx()
    
    line1 = ax4.plot(ranks, nmse_std, 'r-o', label='NMSE Std', linewidth=2)
    line2 = ax4_twin.plot(ranks, ssim_std, 'b-s', label='SSIM Std', linewidth=2)
    
    ax4.set_xlabel('LoRA Rank')
    ax4.set_ylabel('NMSE Std Dev', color='red')
    ax4_twin.set_ylabel('SSIM Std Dev', color='blue')
    ax4.set_title('(d) Performance Variability Across Domains')
    ax4.grid(True, alpha=0.3)
    
    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax4.legend(lines, labels, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'rank_analysis_overview.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Heatmap: Rank vs Domain performance
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # NMSE heatmap
    pivot_nmse = target_df.pivot(index='train_domain', columns='rank', values='nmse')
    sns.heatmap(pivot_nmse, annot=True, fmt='.2e', cmap='viridis_r', ax=axes[0])
    axes[0].set_title('NMSE by Domain and Rank\n(lower is better)', fontweight='bold')
    axes[0].set_xlabel('LoRA Rank')
    axes[0].set_ylabel('Domain ID')
    
    # SSIM heatmap
    pivot_ssim = target_df.pivot(index='train_domain', columns='rank', values='ssim')
    sns.heatmap(pivot_ssim, annot=True, fmt='.3f', cmap='viridis', ax=axes[1])
    axes[1].set_title('SSIM by Domain and Rank\n(higher is better)', fontweight='bold')
    axes[1].set_xlabel('LoRA Rank')
    axes[1].set_ylabel('Domain ID')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'rank_domain_heatmaps.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Cross-domain transfer analysis
    if not df[df['is_target_domain'] == False].empty:
        create_cross_domain_transfer_plot(df, output_dir)
    
    print("  ✅ All plots created successfully!")

def create_cross_domain_transfer_plot(df: pd.DataFrame, output_dir: Path):
    """Create plots showing cross-domain transfer effects."""
    # Filter for cross-domain evaluation (train_domain != eval_domain)
    cross_df = df[df['is_target_domain'] == False].copy()
    
    if cross_df.empty:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Cross-Domain Transfer Analysis by Rank', fontsize=16, fontweight='bold')
    
    # Average cross-domain NMSE by rank
    ax1 = axes[0, 0]
    cross_summary = cross_df.groupby('rank').agg({
        'nmse': ['mean', 'std'],
        'ssim': ['mean', 'std']
    })
    
    ranks = cross_summary.index
    nmse_mean = cross_summary[('nmse', 'mean')]
    nmse_std = cross_summary[('nmse', 'std')]
    
    ax1.errorbar(ranks, nmse_mean, yerr=nmse_std, marker='o', 
                capsize=5, capthick=2, linewidth=2)
    ax1.set_xlabel('LoRA Rank')
    ax1.set_ylabel('Average Cross-Domain NMSE')
    ax1.set_title('(a) Cross-Domain NMSE vs Rank')
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Average cross-domain SSIM by rank
    ax2 = axes[0, 1]
    ssim_mean = cross_summary[('ssim', 'mean')]
    ssim_std = cross_summary[('ssim', 'std')]
    
    ax2.errorbar(ranks, ssim_mean, yerr=ssim_std, marker='s', 
                capsize=5, capthick=2, linewidth=2)
    ax2.set_xlabel('LoRA Rank')
    ax2.set_ylabel('Average Cross-Domain SSIM')
    ax2.set_title('(b) Cross-Domain SSIM vs Rank')
    ax2.grid(True, alpha=0.3)
    
    # Transfer quality: cross-domain vs target-domain performance ratio
    ax3 = axes[1, 0]
    target_df = df[df['is_target_domain'] == True]
    
    transfer_ratios = []
    for rank in sorted(df['rank'].unique()):
        cross_rank = cross_df[cross_df['rank'] == rank]
        target_rank = target_df[target_df['rank'] == rank]
        
        if not cross_rank.empty and not target_rank.empty:
            # Ratio of cross-domain to target-domain SSIM (higher = better transfer)
            ratio = cross_rank['ssim'].mean() / target_rank['ssim'].mean()
            transfer_ratios.append({'rank': rank, 'transfer_ratio': ratio})
    
    if transfer_ratios:
        transfer_df = pd.DataFrame(transfer_ratios)
        ax3.plot(transfer_df['rank'], transfer_df['transfer_ratio'], 
                marker='^', linewidth=2, markersize=8)
        ax3.set_xlabel('LoRA Rank')
        ax3.set_ylabel('Cross-Domain / Target-Domain SSIM Ratio')
        ax3.set_title('(c) Transfer Quality by Rank')
        ax3.grid(True, alpha=0.3)
        ax3.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Perfect Transfer')
        ax3.legend()
    
    # Domain-specific transfer patterns
    ax4 = axes[1, 1]
    # Show how different source domains transfer to other domains
    pivot_transfer = cross_df.pivot_table(
        index='train_domain', columns='rank', 
        values='ssim', aggfunc='mean'
    )
    
    if not pivot_transfer.empty:
        sns.heatmap(pivot_transfer, annot=True, fmt='.3f', 
                   cmap='coolwarm', ax=ax4, center=0.5)
        ax4.set_title('(d) Cross-Domain Transfer SSIM\nby Source Domain and Rank')
        ax4.set_xlabel('LoRA Rank')
        ax4.set_ylabel('Source Domain')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'cross_domain_transfer_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_rank_recommendation_report(df: pd.DataFrame, output_dir: Path):
    """Generate a comprehensive report with rank recommendations."""
    target_df = df[df['is_target_domain'] == True].copy()
    
    report_path = output_dir / f"rank_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    
    with open(report_path, 'w') as f:
        f.write("LORA RANK ANALYSIS REPORT\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Domains Analyzed: {sorted(target_df['train_domain'].unique())}\n")
        f.write(f"Ranks Tested: {sorted(target_df['rank'].unique())}\n\n")
        
        # Overall performance summary
        f.write("OVERALL PERFORMANCE SUMMARY\n")
        f.write("-" * 30 + "\n")
        
        overall_stats = target_df.groupby('rank').agg({
            'nmse': ['mean', 'std', 'min', 'max'],
            'ssim': ['mean', 'std', 'min', 'max']
        })
        
        for rank in sorted(target_df['rank'].unique()):
            stats = overall_stats.loc[rank]
            f.write(f"\nRank {rank}:\n")
            f.write(f"  NMSE: {stats[('nmse', 'mean')]:.2e} ± {stats[('nmse', 'std')]:.2e} "
                   f"(range: {stats[('nmse', 'min')]:.2e} - {stats[('nmse', 'max')]:.2e})\n")
            f.write(f"  SSIM: {stats[('ssim', 'mean')]:.3f} ± {stats[('ssim', 'std')]:.3f} "
                   f"(range: {stats[('ssim', 'min')]:.3f} - {stats[('ssim', 'max')]:.3f})\n")
        
        # Best rank for each domain
        f.write("\n\nBEST RANK BY DOMAIN\n")
        f.write("-" * 25 + "\n")
        
        for domain in sorted(target_df['train_domain'].unique()):
            domain_data = target_df[target_df['train_domain'] == domain]
            
            # Best for NMSE (lowest)
            best_nmse_idx = domain_data['nmse'].idxmin()
            best_nmse = domain_data.loc[best_nmse_idx]
            
            # Best for SSIM (highest)
            best_ssim_idx = domain_data['ssim'].idxmax()
            best_ssim = domain_data.loc[best_ssim_idx]
            
            f.write(f"\nDomain {domain}:\n")
            f.write(f"  Best NMSE: Rank {best_nmse['rank']} (NMSE: {best_nmse['nmse']:.2e})\n")
            f.write(f"  Best SSIM: Rank {best_ssim['rank']} (SSIM: {best_ssim['ssim']:.3f})\n")
        
        # Rank recommendations
        f.write("\n\nRANK RECOMMENDATIONS\n")
        f.write("-" * 20 + "\n")
        
        # Find rank with best average performance
        avg_performance = target_df.groupby('rank').agg({
            'nmse': 'mean',
            'ssim': 'mean'
        })
        
        # Normalize metrics for comparison (NMSE: lower better, SSIM: higher better)
        avg_performance['nmse_norm'] = (avg_performance['nmse'].max() - avg_performance['nmse']) / \
                                      (avg_performance['nmse'].max() - avg_performance['nmse'].min())
        avg_performance['ssim_norm'] = (avg_performance['ssim'] - avg_performance['ssim'].min()) / \
                                      (avg_performance['ssim'].max() - avg_performance['ssim'].min())
        
        avg_performance['combined_score'] = (avg_performance['nmse_norm'] + avg_performance['ssim_norm']) / 2
        
        best_rank = avg_performance['combined_score'].idxmax()
        best_score = avg_performance.loc[best_rank, 'combined_score']
        
        f.write(f"Recommended Rank: {best_rank}\n")
        f.write(f"Combined Performance Score: {best_score:.3f}\n")
        f.write(f"Average NMSE: {avg_performance.loc[best_rank, 'nmse']:.2e}\n")
        f.write(f"Average SSIM: {avg_performance.loc[best_rank, 'ssim']:.3f}\n\n")
        
        # Efficiency analysis
        f.write("EFFICIENCY ANALYSIS\n")
        f.write("-" * 18 + "\n")
        
        for rank in sorted(target_df['rank'].unique()):
            rank_data = target_df[target_df['rank'] == rank]
            efficiency = rank_data['ssim'].mean() / rank
            f.write(f"Rank {rank}: SSIM per parameter unit = {efficiency:.6f}\n")
    
    print(f"📄 Comprehensive report saved to: {report_path}")

def main():
    parser = argparse.ArgumentParser(description="Analyze LoRA rank effects on domain performance")
    parser.add_argument("--base_config", type=str, required=True,
                       help="Path to base configuration file")
    parser.add_argument("--ranks", nargs="+", type=int, default=[2, 4, 8, 12, 16],
                       help="List of ranks to test")
    parser.add_argument("--domain_ids", nargs="+", type=str, default=None,
                       help="List of domain IDs to test (default: all in config)")
    parser.add_argument("--output_dir", type=str, default="./rank_analysis_results",
                       help="Output directory for results")
    parser.add_argument("--checkpoints_dir", type=str, default=None,
                       help="Directory containing pre-trained checkpoints")
    parser.add_argument("--train_models", action="store_true",
                       help="Train models instead of using existing checkpoints")
    parser.add_argument("--analyze_only", type=str, default=None,
                       help="Path to CSV file with existing results to analyze")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load domain IDs from config if not provided
    if args.domain_ids is None:
        config = load_config(args.base_config)
        args.domain_ids = [str(d) for d in config.data.sequence]
    
    print("🚀 LORA RANK ANALYSIS")
    print("=" * 50)
    print(f"Base Config: {args.base_config}")
    print(f"Ranks to Test: {args.ranks}")
    print(f"Domains: {args.domain_ids}")
    print(f"Output Directory: {output_dir}")
    print("=" * 50)
    
    if args.analyze_only:
        # Analyze existing results
        print(f"📊 Analyzing existing results from: {args.analyze_only}")
        df = pd.read_csv(args.analyze_only)
        create_rank_analysis_plots(df, output_dir)
        create_rank_recommendation_report(df, output_dir)
        
    elif args.train_models:
        # Train new models
        print("🏋️ Training single-domain models for each rank...")
        checkpoints = train_single_domain_rank_models(
            args.base_config, args.domain_ids, args.ranks, output_dir / "checkpoints"
        )
        
        if checkpoints:
            df = analyze_rank_effects_from_checkpoints(
                checkpoints, args.domain_ids, args.ranks, args.base_config, output_dir
            )
            create_rank_recommendation_report(df, output_dir)
        else:
            print("❌ No successful training runs completed.")
    
    elif args.checkpoints_dir:
        # Use existing checkpoints
        print(f"📂 Using existing checkpoints from: {args.checkpoints_dir}")
        
        # Discover checkpoints
        checkpoints = {}
        checkpoints_dir = Path(args.checkpoints_dir)
        
        for domain_id in args.domain_ids:
            for rank in args.ranks:
                # Look for checkpoint pattern: domain_{domain_id}_rank_{rank}/*.pt
                pattern_dir = checkpoints_dir / f"domain_{domain_id}_rank_{rank}"
                if pattern_dir.exists():
                    checkpoint_files = list(pattern_dir.glob("*.pt"))
                    if checkpoint_files:
                        checkpoints[(domain_id, rank)] = str(checkpoint_files[0])
        
        print(f"Found {len(checkpoints)} checkpoints")
        
        if checkpoints:
            df = analyze_rank_effects_from_checkpoints(
                checkpoints, args.domain_ids, args.ranks, args.base_config, output_dir
            )
            create_rank_recommendation_report(df, output_dir)
        else:
            print("❌ No checkpoints found with expected naming pattern.")
    
    else:
        print("❌ Please specify either --train_models, --checkpoints_dir, or --analyze_only")
        return
    
    print("\n✅ Rank analysis completed successfully!")
    print(f"📁 Check results in: {output_dir}")

if __name__ == "__main__":
    main() 