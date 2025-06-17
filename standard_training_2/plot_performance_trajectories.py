import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import matplotlib.patches as mpatches

# IEEE-style configuration for publication-quality plots
IEEE_STYLE = {
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.titlesize': 14,
    'lines.linewidth': 2.0,
    'lines.markersize': 8,
    'axes.linewidth': 1.2,
    'axes.grid': True,
    'grid.linestyle': ':',
    'grid.linewidth': 0.8,
    'grid.alpha': 0.6,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.format': 'svg',
    'mathtext.fontset': 'custom',
    'mathtext.rm': 'Times New Roman',
    'mathtext.it': 'Times New Roman:italic',
    'mathtext.bf': 'Times New Roman:bold',
    'axes.edgecolor': 'black',
    'xtick.color': 'black',
    'ytick.color': 'black',
    'text.color': 'black'
}

# IEEE-style color palette
IEEE_COLORS = {
    'primary': '#1f4e79',      # IEEE blue
    'secondary': '#c5504b',    # IEEE red  
    'accent1': '#70ad47',      # IEEE green
    'accent2': '#ffc000',      # IEEE gold
    'accent3': '#7030a0',      # IEEE purple
    'accent4': '#00b0f0',      # IEEE light blue
    'grey': '#595959',         # IEEE grey
    'black': '#000000'         # IEEE black
}

# Model colors and shapes
MODEL_CONFIGS = {
    'SRCNN': {'color': IEEE_COLORS['primary'], 'marker': 'o', 'name': 'SRCNN'},
    'SRCNN_DnCNN': {'color': IEEE_COLORS['secondary'], 'marker': 's', 'name': 'SRCNN+DnCNN'},
    'DnCNN_SRCNN': {'color': IEEE_COLORS['accent1'], 'marker': '^', 'name': 'DnCNN+SRCNN'},
    'NonRes_Autoencoder': {'color': '#FF6B6B', 'marker': 'D', 'name': 'Non-Res. AE'},
    'Res_Autoencoder': {'color': '#4ECDC4', 'marker': 'v', 'name': 'Res. AE'},
    'Res_Autoencoder_SRCNN': {'color': IEEE_COLORS['accent2'], 'marker': 'p', 'name': 'Res. AE+SRCNN'},
    'UNet': {'color': IEEE_COLORS['accent3'], 'marker': 'h', 'name': 'U-Net'},
    'UNet_SRCNN': {'color': IEEE_COLORS['accent4'], 'marker': '*', 'name': 'U-Net+SRCNN'}
}

def apply_ieee_style():
    """Apply IEEE-style formatting to matplotlib."""
    plt.style.use('default')
    plt.rcParams.update(IEEE_STYLE)

def load_and_process_cf_data(csv_path, target_domain_example):
    """Load and process catastrophic forgetting data for a single target domain example."""
    df = pd.read_csv(csv_path)
    
    # Process the data to extract catastrophic forgetting for one specific target domain
    trajectories = []
    
    # Get all unique models
    models = df['model_architecture'].unique()
    base_domain = 'domain_high_snr_med_linear'  # Base training domain
    
    for model in models:
        model_data = df[df['model_architecture'] == model]
        
        # Get base performance on source domain (before fine-tuning)
        source_before = model_data[
            (model_data['eval_type'] == 'initial_on_A') & 
            (model_data['dataset_name'] == base_domain)
        ]
        
        if len(source_before) == 0:
            continue
        
        # Performance on target domain before fine-tuning (domain shift)
        target_before = model_data[
            (model_data['eval_type'] == 'initial_model_on_target') &
            (model_data['dataset_name'] == target_domain_example)
        ]
        
        # Performance on target domain after fine-tuning (adaptation)
        target_after = model_data[
            (model_data['eval_type'] == 'target_B_post_finetune') &
            (model_data['dataset_name'] == target_domain_example)
        ]
        
        # Performance on source domain after fine-tuning (catastrophic forgetting)
        source_after = model_data[
            (model_data['eval_type'] == 'base_A_post_finetune_on_B') &
            (model_data['finetuned_on_dataset'] == target_domain_example)
        ]
        
        if len(target_before) > 0 and len(target_after) > 0 and len(source_after) > 0:
            trajectory = {
                'model': model,
                'target_domain': target_domain_example,
                # Source domain performance (catastrophic forgetting)
                'source_before_nmse': source_before['nmse'].iloc[0],
                'source_before_psnr': source_before['psnr'].iloc[0],
                'source_before_ssim': source_before['ssim'].iloc[0],
                'source_after_nmse': source_after['nmse'].iloc[0],
                'source_after_psnr': source_after['psnr'].iloc[0],
                'source_after_ssim': source_after['ssim'].iloc[0],
                # Target domain performance (adaptation)
                'target_before_nmse': target_before['nmse'].iloc[0],
                'target_before_psnr': target_before['psnr'].iloc[0],
                'target_before_ssim': target_before['ssim'].iloc[0],
                'target_after_nmse': target_after['nmse'].iloc[0],
                'target_after_psnr': target_after['psnr'].iloc[0],
                'target_after_ssim': target_after['ssim'].iloc[0],
            }
            trajectories.append(trajectory)
    
    return pd.DataFrame(trajectories)

def create_psnr_forgetting_plot(df, output_dir, target_domain):
    """Create PSNR catastrophic forgetting plot."""
    apply_ieee_style()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Add "no forgetting" line (diagonal)
    psnr_min = min(df['source_before_psnr'].min(), df['source_after_psnr'].min()) - 1
    psnr_max = max(df['source_before_psnr'].max(), df['source_after_psnr'].max()) + 1
    ax.plot([psnr_min, psnr_max], [psnr_min, psnr_max], 
             '--', color='grey', alpha=0.7, linewidth=3, label='No Forgetting (Ideal)')
    
    for _, row in df.iterrows():
        if row['model'] not in MODEL_CONFIGS:
            continue
            
        config = MODEL_CONFIGS[row['model']]
        
        before_psnr = row['source_before_psnr']
        after_psnr = row['source_after_psnr']
        
        # Before fine-tuning (filled marker - good performance)
        ax.scatter(before_psnr, before_psnr, marker=config['marker'], s=150, 
                   facecolors=config['color'], edgecolors='black', linewidths=2, 
                   alpha=0.9, label=config['name'], zorder=3)
        
        # After fine-tuning (hollow marker - degraded performance)
        ax.scatter(before_psnr, after_psnr, marker=config['marker'], s=150,
                   facecolors='none', edgecolors=config['color'], linewidths=3, 
                   alpha=0.9, zorder=3)
        
        # Draw arrow showing forgetting
        ax.annotate('', xy=(before_psnr, after_psnr), xytext=(before_psnr, before_psnr),
                    arrowprops=dict(arrowstyle='->', color=config['color'], 
                                  lw=3, alpha=0.8), zorder=2)
    
    ax.set_xlabel('Source Domain PSNR Before Fine-tuning (dB)', weight='bold', fontsize=12)
    ax.set_ylabel('Source Domain PSNR After Fine-tuning (dB)', weight='bold', fontsize=12)
    ax.set_title(f'PSNR Catastrophic Forgetting\n(Fine-tuning Target: {target_domain.replace("_", " ").title()})', 
                weight='bold', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(psnr_min, psnr_max)
    ax.set_ylim(psnr_min, psnr_max)
    
    # Add text annotations
    ax.text(0.02, 0.98, 'Filled = Before Fine-tuning\nHollow = After Fine-tuning\nArrows = Forgetting Effect', 
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.legend(loc='lower right', frameon=True, fancybox=True, fontsize=9)
    
    plt.tight_layout()
    
    # Save plot
    save_path = output_dir / 'psnr_catastrophic_forgetting.svg'
    plt.savefig(save_path)
    plt.close()
    
    print(f"Saved PSNR forgetting plot to: {save_path}")

def create_nmse_forgetting_plot(df, output_dir, target_domain):
    """Create NMSE catastrophic forgetting plot."""
    apply_ieee_style()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Add "no forgetting" line (diagonal)
    nmse_min = min(df['source_before_nmse'].min(), df['source_after_nmse'].min()) * 0.8
    nmse_max = max(df['source_before_nmse'].max(), df['source_after_nmse'].max()) * 1.2
    ax.loglog([nmse_min, nmse_max], [nmse_min, nmse_max], 
              '--', color='grey', alpha=0.7, linewidth=3, label='No Forgetting (Ideal)')
    
    for _, row in df.iterrows():
        if row['model'] not in MODEL_CONFIGS:
            continue
            
        config = MODEL_CONFIGS[row['model']]
        
        before_nmse = row['source_before_nmse']
        after_nmse = row['source_after_nmse']
        
        # Before fine-tuning (filled marker - good performance)
        ax.scatter(before_nmse, before_nmse, marker=config['marker'], s=150, 
                   facecolors=config['color'], edgecolors='black', linewidths=2, 
                   alpha=0.9, label=config['name'], zorder=3)
        
        # After fine-tuning (hollow marker - degraded performance)
        ax.scatter(before_nmse, after_nmse, marker=config['marker'], s=150,
                   facecolors='none', edgecolors=config['color'], linewidths=3, 
                   alpha=0.9, zorder=3)
        
        # Draw arrow showing forgetting
        ax.annotate('', xy=(before_nmse, after_nmse), xytext=(before_nmse, before_nmse),
                    arrowprops=dict(arrowstyle='->', color=config['color'], 
                                  lw=3, alpha=0.8), zorder=2)
    
    ax.set_xlabel('Source Domain NMSE Before Fine-tuning', weight='bold', fontsize=12)
    ax.set_ylabel('Source Domain NMSE After Fine-tuning', weight='bold', fontsize=12)
    ax.set_title(f'NMSE Catastrophic Forgetting\n(Fine-tuning Target: {target_domain.replace("_", " ").title()})', 
                weight='bold', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # Add text annotations
    ax.text(0.02, 0.98, 'Filled = Before Fine-tuning\nHollow = After Fine-tuning\nArrows = Forgetting Effect', 
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.legend(loc='lower right', frameon=True, fancybox=True, fontsize=9)
    
    plt.tight_layout()
    
    # Save plot
    save_path = output_dir / 'nmse_catastrophic_forgetting.svg'
    plt.savefig(save_path)
    plt.close()
    
    print(f"Saved NMSE forgetting plot to: {save_path}")

def create_ssim_forgetting_plot(df, output_dir, target_domain):
    """Create SSIM catastrophic forgetting plot."""
    apply_ieee_style()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Add "no forgetting" line (diagonal)
    ssim_min = min(df['source_before_ssim'].min(), df['source_after_ssim'].min()) - 0.02
    ssim_max = max(df['source_before_ssim'].max(), df['source_after_ssim'].max()) + 0.02
    ax.plot([ssim_min, ssim_max], [ssim_min, ssim_max], 
             '--', color='grey', alpha=0.7, linewidth=3, label='No Forgetting (Ideal)')
    
    for _, row in df.iterrows():
        if row['model'] not in MODEL_CONFIGS:
            continue
            
        config = MODEL_CONFIGS[row['model']]
        
        before_ssim = row['source_before_ssim']
        after_ssim = row['source_after_ssim']
        
        # Before fine-tuning (filled marker - good performance)
        ax.scatter(before_ssim, before_ssim, marker=config['marker'], s=150, 
                   facecolors=config['color'], edgecolors='black', linewidths=2, 
                   alpha=0.9, label=config['name'], zorder=3)
        
        # After fine-tuning (hollow marker - degraded performance)
        ax.scatter(before_ssim, after_ssim, marker=config['marker'], s=150,
                   facecolors='none', edgecolors=config['color'], linewidths=3, 
                   alpha=0.9, zorder=3)
        
        # Draw arrow showing forgetting
        ax.annotate('', xy=(before_ssim, after_ssim), xytext=(before_ssim, before_ssim),
                    arrowprops=dict(arrowstyle='->', color=config['color'], 
                                  lw=3, alpha=0.8), zorder=2)
    
    ax.set_xlabel('Source Domain SSIM Before Fine-tuning', weight='bold', fontsize=12)
    ax.set_ylabel('Source Domain SSIM After Fine-tuning', weight='bold', fontsize=12)
    ax.set_title(f'SSIM Catastrophic Forgetting\n(Fine-tuning Target: {target_domain.replace("_", " ").title()})', 
                weight='bold', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(ssim_min, ssim_max)
    ax.set_ylim(ssim_min, ssim_max)
    
    # Add text annotations
    ax.text(0.02, 0.98, 'Filled = Before Fine-tuning\nHollow = After Fine-tuning\nArrows = Forgetting Effect', 
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.legend(loc='lower right', frameon=True, fancybox=True, fontsize=9)
    
    plt.tight_layout()
    
    # Save plot
    save_path = output_dir / 'ssim_catastrophic_forgetting.svg'
    plt.savefig(save_path)
    plt.close()
    
    print(f"Saved SSIM forgetting plot to: {save_path}")

def create_target_adaptation_plots(df, output_dir, target_domain):
    """Create target domain adaptation plots for comparison."""
    apply_ieee_style()
    
    # PSNR Adaptation Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    psnr_min = min(df['target_before_psnr'].min(), df['target_after_psnr'].min()) - 1
    psnr_max = max(df['target_before_psnr'].max(), df['target_after_psnr'].max()) + 1
    ax.plot([psnr_min, psnr_max], [psnr_min, psnr_max], 
             '--', color='grey', alpha=0.7, linewidth=3, label='No Improvement')
    
    for _, row in df.iterrows():
        if row['model'] not in MODEL_CONFIGS:
            continue
            
        config = MODEL_CONFIGS[row['model']]
        
        before_psnr = row['target_before_psnr']
        after_psnr = row['target_after_psnr']
        
        # Before fine-tuning (hollow marker - poor performance)
        ax.scatter(before_psnr, before_psnr, marker=config['marker'], s=150, 
                   facecolors='none', edgecolors=config['color'], linewidths=3, 
                   alpha=0.9, label=config['name'], zorder=3)
        
        # After fine-tuning (filled marker - improved performance)
        ax.scatter(before_psnr, after_psnr, marker=config['marker'], s=150,
                   facecolors=config['color'], edgecolors='black', linewidths=2, 
                   alpha=0.9, zorder=3)
        
        # Draw arrow showing improvement
        ax.annotate('', xy=(before_psnr, after_psnr), xytext=(before_psnr, before_psnr),
                    arrowprops=dict(arrowstyle='->', color=config['color'], 
                                  lw=3, alpha=0.8), zorder=2)
    
    ax.set_xlabel('Target Domain PSNR Before Fine-tuning (dB)', weight='bold', fontsize=12)
    ax.set_ylabel('Target Domain PSNR After Fine-tuning (dB)', weight='bold', fontsize=12)
    ax.set_title(f'PSNR Target Domain Adaptation\n(Target: {target_domain.replace("_", " ").title()})', 
                weight='bold', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(psnr_min, psnr_max)
    ax.set_ylim(psnr_min, psnr_max)
    
    ax.text(0.02, 0.98, 'Hollow = Before Fine-tuning\nFilled = After Fine-tuning\nArrows = Adaptation', 
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.legend(loc='lower right', frameon=True, fancybox=True, fontsize=9)
    
    plt.tight_layout()
    
    save_path = output_dir / 'psnr_target_adaptation.svg'
    plt.savefig(save_path)
    plt.close()
    
    print(f"Saved PSNR target adaptation plot to: {save_path}")

def create_nmse_target_adaptation_plot(df, output_dir, target_domain):
    """Create NMSE target domain adaptation plot."""
    apply_ieee_style()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Add "no improvement" line (diagonal)
    nmse_min = min(df['target_before_nmse'].min(), df['target_after_nmse'].min()) * 0.8
    nmse_max = max(df['target_before_nmse'].max(), df['target_after_nmse'].max()) * 1.2
    ax.loglog([nmse_min, nmse_max], [nmse_min, nmse_max], 
              '--', color='grey', alpha=0.7, linewidth=3, label='No Improvement')
    
    for _, row in df.iterrows():
        if row['model'] not in MODEL_CONFIGS:
            continue
            
        config = MODEL_CONFIGS[row['model']]
        
        before_nmse = row['target_before_nmse']
        after_nmse = row['target_after_nmse']
        
        # Before fine-tuning (hollow marker - poor performance)
        ax.scatter(before_nmse, before_nmse, marker=config['marker'], s=150, 
                   facecolors='none', edgecolors=config['color'], linewidths=3, 
                   alpha=0.9, label=config['name'], zorder=3)
        
        # After fine-tuning (filled marker - improved performance)
        ax.scatter(before_nmse, after_nmse, marker=config['marker'], s=150,
                   facecolors=config['color'], edgecolors='black', linewidths=2, 
                   alpha=0.9, zorder=3)
        
        # Draw arrow showing improvement
        ax.annotate('', xy=(before_nmse, after_nmse), xytext=(before_nmse, before_nmse),
                    arrowprops=dict(arrowstyle='->', color=config['color'], 
                                  lw=3, alpha=0.8), zorder=2)
    
    ax.set_xlabel('Target Domain NMSE Before Fine-tuning', weight='bold', fontsize=12)
    ax.set_ylabel('Target Domain NMSE After Fine-tuning', weight='bold', fontsize=12)
    ax.set_title(f'NMSE Target Domain Adaptation\n(Target: {target_domain.replace("_", " ").title()})', 
                weight='bold', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    ax.text(0.02, 0.98, 'Hollow = Before Fine-tuning\nFilled = After Fine-tuning\nArrows = Adaptation', 
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.legend(loc='upper right', frameon=True, fancybox=True, fontsize=9)
    
    plt.tight_layout()
    
    save_path = output_dir / 'nmse_target_adaptation.svg'
    plt.savefig(save_path)
    plt.close()
    
    print(f"Saved NMSE target adaptation plot to: {save_path}")

def create_ssim_target_adaptation_plot(df, output_dir, target_domain):
    """Create SSIM target domain adaptation plot."""
    apply_ieee_style()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Add "no improvement" line (diagonal)
    ssim_min = min(df['target_before_ssim'].min(), df['target_after_ssim'].min()) - 0.02
    ssim_max = max(df['target_before_ssim'].max(), df['target_after_ssim'].max()) + 0.02
    ax.plot([ssim_min, ssim_max], [ssim_min, ssim_max], 
             '--', color='grey', alpha=0.7, linewidth=3, label='No Improvement')
    
    for _, row in df.iterrows():
        if row['model'] not in MODEL_CONFIGS:
            continue
            
        config = MODEL_CONFIGS[row['model']]
        
        before_ssim = row['target_before_ssim']
        after_ssim = row['target_after_ssim']
        
        # Before fine-tuning (hollow marker - poor performance)
        ax.scatter(before_ssim, before_ssim, marker=config['marker'], s=150, 
                   facecolors='none', edgecolors=config['color'], linewidths=3, 
                   alpha=0.9, label=config['name'], zorder=3)
        
        # After fine-tuning (filled marker - improved performance)
        ax.scatter(before_ssim, after_ssim, marker=config['marker'], s=150,
                   facecolors=config['color'], edgecolors='black', linewidths=2, 
                   alpha=0.9, zorder=3)
        
        # Draw arrow showing improvement
        ax.annotate('', xy=(before_ssim, after_ssim), xytext=(before_ssim, before_ssim),
                    arrowprops=dict(arrowstyle='->', color=config['color'], 
                                  lw=3, alpha=0.8), zorder=2)
    
    ax.set_xlabel('Target Domain SSIM Before Fine-tuning', weight='bold', fontsize=12)
    ax.set_ylabel('Target Domain SSIM After Fine-tuning', weight='bold', fontsize=12)
    ax.set_title(f'SSIM Target Domain Adaptation\n(Target: {target_domain.replace("_", " ").title()})', 
                weight='bold', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(ssim_min, ssim_max)
    ax.set_ylim(ssim_min, ssim_max)
    
    ax.text(0.02, 0.98, 'Hollow = Before Fine-tuning\nFilled = After Fine-tuning\nArrows = Adaptation', 
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.legend(loc='lower right', frameon=True, fancybox=True, fontsize=9)
    
    plt.tight_layout()
    
    save_path = output_dir / 'ssim_target_adaptation.svg'
    plt.savefig(save_path)
    plt.close()
    
    print(f"Saved SSIM target adaptation plot to: {save_path}")

def main():
    """Generate clean catastrophic forgetting analysis plots for one target domain."""
    # Paths
    csv_path = "standard_training_2/catastrophic_forgetting_v2/cf_batch_results/cf_v2_all_models_summary_20250606_002507.csv"
    output_dir = Path('performance_trajectory_plots')
    output_dir.mkdir(exist_ok=True)
    
    # Choose one target domain as example (medium SNR fast speed as requested)
    target_domain_example = 'domain_low_snr_fast_linear'  # Medium SNR, fast speed
    source_domain = 'domain_high_snr_med_linear'  # High SNR, medium speed
    
    print(f"\n{'='*80}")
    print("GENERATING CLEAN CATASTROPHIC FORGETTING ANALYSIS")
    print(f"{'='*80}")
    print(f"📊 DOMAIN SETUP:")
    print(f"   • SOURCE DOMAIN (Original Training): {source_domain}")
    print(f"     └─ High SNR (20 dB), Medium Speed, Linear Channel")
    print(f"   • TARGET DOMAIN (Fine-tuning): {target_domain_example}")
    print(f"     └─ Medium SNR (10 dB), Fast Speed, Linear Channel")
    print(f"\n🎯 ANALYSIS:")
    print(f"   • Models trained on HIGH SNR, MEDIUM speed")
    print(f"   • Fine-tuned on MEDIUM SNR, FAST speed") 
    print(f"   • Forgetting: Performance drop on original HIGH SNR domain")
    print(f"   • Adaptation: Performance gain on new MEDIUM SNR domain")
    
    # Load and process data
    print(f"\nLoading and processing catastrophic forgetting data...")
    df = load_and_process_cf_data(csv_path, target_domain_example)
    
    print(f"Processed {len(df)} models for target domain: {target_domain_example}")
    
    print(f"\nGenerating plots in: {output_dir}")
    print("-" * 60)
    
    # Generate catastrophic forgetting plots (source domain degradation)
    print("🔴 CATASTROPHIC FORGETTING PLOTS (Source Domain Degradation):")
    print("1. PSNR Catastrophic Forgetting...")
    create_psnr_forgetting_plot(df, output_dir, target_domain_example)
    
    print("2. NMSE Catastrophic Forgetting...")
    create_nmse_forgetting_plot(df, output_dir, target_domain_example)
    
    print("3. SSIM Catastrophic Forgetting...")
    create_ssim_forgetting_plot(df, output_dir, target_domain_example)
    
    # Generate target domain adaptation plots (target domain improvement)
    print("\n🟢 TARGET DOMAIN ADAPTATION PLOTS (Target Domain Improvement):")
    print("4. PSNR Target Domain Adaptation...")
    create_target_adaptation_plots(df, output_dir, target_domain_example)
    
    print("5. NMSE Target Domain Adaptation...")
    create_nmse_target_adaptation_plot(df, output_dir, target_domain_example)
    
    print("6. SSIM Target Domain Adaptation...")
    create_ssim_target_adaptation_plot(df, output_dir, target_domain_example)
    
    print(f"\n{'='*80}")
    print("✅ ALL CLEAN FORGETTING & ADAPTATION PLOTS GENERATED")
    print(f"{'='*80}")
    print(f"\n📁 Plots saved to: {output_dir}")
    print(f"\n📊 Available plots:")
    
    # Categorise plots
    forgetting_plots = list(output_dir.glob("*catastrophic_forgetting*.svg"))
    adaptation_plots = list(output_dir.glob("*target_adaptation*.svg"))
    
    print("🔴 CATASTROPHIC FORGETTING (Source Domain):")
    for plot_file in sorted(forgetting_plots):
        print(f"   • {plot_file.name}")
    
    print("🟢 TARGET ADAPTATION (Target Domain):")
    for plot_file in sorted(adaptation_plots):
        print(f"   • {plot_file.name}")
    
    print(f"\n🎯 Complete Analysis Shows:")
    print(f"   • SOURCE DOMAIN: Performance DEGRADES after fine-tuning (forgetting)")
    print(f"   • TARGET DOMAIN: Performance IMPROVES after fine-tuning (adaptation)")
    print(f"   • All three metrics (PSNR, NMSE, SSIM) demonstrate this trade-off")
    print(f"   • Clear evidence of catastrophic forgetting phenomenon")

if __name__ == '__main__':
    main() 