import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

# IEEE-style configuration for publication-quality plots
IEEE_STYLE = {
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
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

def apply_ieee_style():
    """Apply IEEE-style formatting to matplotlib."""
    plt.style.use('default')
    plt.rcParams.update(IEEE_STYLE)

def extract_evaluation_data():
    """Extract evaluation data from the LaTeX tables."""
    
    # All the models and their performance metrics
    data = {
        'LS': {
            'MSE': 3.4177e-4,
            'NMSE': 0.4627,
            'SSIM': 0.3878,
            'PSNR': 14.6500,
            'Inference_Time': 6.3700
        },
        'MMSE': {
            'MSE': 3.3596e-4,
            'NMSE': 0.4452,
            'SSIM': 0.4223,
            'PSNR': 14.9100,
            'Inference_Time': 6.4000
        },
        'SRCNN': {
            'MSE': 5.9865e-5,
            'NMSE': 0.1333,
            'SSIM': 0.5827,
            'PSNR': 33.5385,
            'Inference_Time': 0.2589
        },
        'SRCNN+DnCNN': {
            'MSE': 3.7998e-5,
            'NMSE': 0.08460,
            'SSIM': 0.7157,
            'PSNR': 35.5126,
            'Inference_Time': 1.6210
        },
        'DnCNN+SRCNN': {
            'MSE': 3.7582e-5,
            'NMSE': 0.08367,
            'SSIM': 0.7139,
            'PSNR': 35.5605,
            'Inference_Time': 1.6210
        },
        'Res. AE+SRCNN': {
            'MSE': 5.0540e-5,
            'NMSE': 0.1125,
            'SSIM': 0.6217,
            'PSNR': 34.2739,
            'Inference_Time': 1.2168
        },
        'U-Net': {
            'MSE': 3.9414e-5,
            'NMSE': 0.0877,
            'SSIM': 0.7240,
            'PSNR': 35.35,
            'Inference_Time': 1.0680
        },
        'U-Net+SRCNN': {
            'MSE': 3.0645e-5,
            'NMSE': 0.0682,
            'SSIM': 0.7876,
            'PSNR': 36.4466,
            'Inference_Time': 1.3198
        },
        'Non-Res. AE': {
            'MSE': 6.9876e-5,
            'NMSE': 0.1556,
            'SSIM': 0.5674,
            'PSNR': 32.8670,
            'Inference_Time': 0.6164
        },
        'Res. AE': {
            'MSE': 4.1749e-5,
            'NMSE': 0.0929,
            'SSIM': 0.6766,
            'PSNR': 35.1037,
            'Inference_Time': 0.8346
        }
    }
    
    return data

def get_model_colors():
    """Define consistent colors for each model type."""
    model_colors = {
        'LS': IEEE_COLORS['grey'],
        'MMSE': IEEE_COLORS['black'],
        'SRCNN': IEEE_COLORS['primary'],
        'SRCNN+DnCNN': IEEE_COLORS['secondary'],
        'DnCNN+SRCNN': IEEE_COLORS['accent1'],
        'Res. AE+SRCNN': IEEE_COLORS['accent2'],
        'U-Net': IEEE_COLORS['accent3'],
        'U-Net+SRCNN': IEEE_COLORS['accent4'],
        'Non-Res. AE': '#FF6B6B',  # Light red
        'Res. AE': '#4ECDC4'       # Teal
    }
    return model_colors

def create_metric_line_plot(data, metric, output_dir):
    """Create line plot for a specific metric."""
    apply_ieee_style()
    
    models = list(data.keys())
    values = [data[model][metric] for model in models]
    colors = [get_model_colors()[model] for model in models]
    
    # Sort by performance (best to worst)
    if metric in ['MSE', 'NMSE', 'Inference_Time']:
        # Lower is better
        sorted_indices = np.argsort(values)
    else:  # SSIM, PSNR
        # Higher is better
        sorted_indices = np.argsort(values)[::-1]
    
    sorted_models = [models[i] for i in sorted_indices]
    sorted_values = [values[i] for i in sorted_indices]
    sorted_colors = [colors[i] for i in sorted_indices]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create line plot with markers
    x_positions = range(len(sorted_models))
    line = ax.plot(x_positions, sorted_values, 
                   color=IEEE_COLORS['primary'], linewidth=3, 
                   marker='o', markersize=10, markerfacecolor='white',
                   markeredgecolor=IEEE_COLORS['primary'], markeredgewidth=2,
                   alpha=0.8)
    
    # Add colored markers for each model
    for i, (x, y, color) in enumerate(zip(x_positions, sorted_values, sorted_colors)):
        ax.scatter(x, y, c=color, s=120, alpha=0.9, edgecolors='black', 
                  linewidth=1.5, zorder=5)
    
    # Customize based on metric
    if metric == 'MSE':
        ax.set_ylabel('Mean Squared Error', weight='bold')
        ax.set_title('Model Comparison: Mean Squared Error\n(Lower is Better)', weight='bold')
        ax.set_yscale('log')
        value_format = '.2e'
    elif metric == 'NMSE':
        ax.set_ylabel('Normalised Mean Squared Error', weight='bold')
        ax.set_title('Model Comparison: Normalised Mean Squared Error\n(Lower is Better)', weight='bold')
        value_format = '.4f'
    elif metric == 'SSIM':
        ax.set_ylabel('Structural Similarity Index', weight='bold')
        ax.set_title('Model Comparison: Structural Similarity Index\n(Higher is Better)', weight='bold')
        value_format = '.4f'
    elif metric == 'PSNR':
        ax.set_ylabel('Peak Signal-to-Noise Ratio (dB)', weight='bold')
        ax.set_title('Model Comparison: Peak Signal-to-Noise Ratio\n(Higher is Better)', weight='bold')
        value_format = '.2f'
    elif metric == 'Inference_Time':
        ax.set_ylabel('Average Inference Time (ms)', weight='bold')
        ax.set_title('Model Comparison: Average Inference Time\n(Lower is Better)', weight='bold')
        value_format = '.4f'
    
    ax.set_xlabel('Model Architecture (Ordered by Performance)', weight='bold')
    ax.set_xticks(x_positions)
    ax.set_xticklabels(sorted_models, rotation=45, ha='right')
    
    # Add value labels next to points
    for i, (x, y, value) in enumerate(zip(x_positions, sorted_values, sorted_values)):
        if metric == 'MSE':
            # For log scale, position text to the right of point
            ax.annotate(f'{value:.2e}', (x, y), xytext=(10, 5), 
                       textcoords='offset points', ha='left', va='bottom',
                       weight='bold', fontsize=9,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        else:
            # Position text above or below point alternately to avoid overlap
            y_offset = 15 if i % 2 == 0 else -15
            va = 'bottom' if i % 2 == 0 else 'top'
            ax.annotate(format(value, value_format), (x, y), xytext=(0, y_offset), 
                       textcoords='offset points', ha='center', va=va,
                       weight='bold', fontsize=9,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # Add subtle background shading to distinguish better/worse regions
    if metric in ['MSE', 'NMSE', 'Inference_Time']:
        # For "lower is better" metrics, shade left side (better) in green
        ax.axvspan(-0.5, len(sorted_models)/3 - 0.5, alpha=0.1, color='green', label='Best Performance')
        ax.axvspan(2*len(sorted_models)/3 - 0.5, len(sorted_models) - 0.5, alpha=0.1, color='red', label='Worst Performance')
    else:
        # For "higher is better" metrics, shade left side (better) in green
        ax.axvspan(-0.5, len(sorted_models)/3 - 0.5, alpha=0.1, color='green', label='Best Performance')
        ax.axvspan(2*len(sorted_models)/3 - 0.5, len(sorted_models) - 0.5, alpha=0.1, color='red', label='Worst Performance')
    
    plt.tight_layout()
    
    # Save plot
    save_path = output_dir / f'metric_comparison_{metric.lower()}.svg'
    plt.savefig(save_path)
    plt.close()
    
    print(f"Saved {metric} line plot to: {save_path}")

def main():
    """Generate metric comparison line plots."""
    # Create output directory
    output_dir = Path('model_comparison_plots')
    output_dir.mkdir(exist_ok=True)
    
    print(f"\n{'='*80}")
    print("GENERATING IEEE-STYLE METRIC COMPARISON LINE PLOTS")
    print(f"{'='*80}")
    
    # Extract evaluation data
    data = extract_evaluation_data()
    
    print(f"Loaded data for {len(data)} models:")
    for model in data.keys():
        print(f"  • {model}")
    
    print(f"\nGenerating line plots in: {output_dir}")
    print("-" * 60)
    
    # Generate individual metric comparison line plots
    print("Generating Metric Comparison Line Plots...")
    metrics = ['MSE', 'NMSE', 'SSIM', 'PSNR', 'Inference_Time']
    for metric in metrics:
        create_metric_line_plot(data, metric, output_dir)
    
    print(f"\n{'='*80}")
    print("✅ ALL IEEE-STYLE LINE PLOTS GENERATED SUCCESSFULLY")
    print(f"{'='*80}")
    print(f"\n📁 Plots saved to: {output_dir}")
    print(f"\n📊 Available plots:")
    plot_files = list(output_dir.glob("metric_comparison_*.svg"))
    for plot_file in sorted(plot_files):
        print(f"   • {plot_file.name}")
    
    print(f"\n🎯 These line plots provide:")
    print("   • Clear metric performance trends across all models")
    print("   • Performance-ordered visualization")
    print("   • Individual model identification with colour coding")
    print("   • IEEE-style formatting for publication quality")

if __name__ == '__main__':
    main() 