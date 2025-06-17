import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os

# IEEE Plot Style Configuration (adapted from plot_heatmap.py)
IEEE_STYLE = {
    # Font settings
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
    'axes.titlesize': 10,
    'axes.labelsize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.titlesize': 12,
    # Line widths and markers
    'lines.linewidth': 1.0,
    'lines.markersize': 4,
    # Grid
    'axes.grid': True,
    'grid.linestyle': ':',
    'grid.linewidth': 0.5,
    # Save settings
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.format': 'svg'
}

def get_pilot_mask():
    """Get the pilot mask used in the dataset (every 12th subcarrier, every 7th symbol)."""
    pilot_mask = np.zeros((72, 70), dtype=bool)
    pilot_mask[::12, ::7] = True
    return pilot_mask

def create_heatmap_with_pilots(ax, data, title, pilot_mask=None, show_pilots=False, is_error=False, custom_cmap='viridis'):
    """Create a heatmap with optional pilot overlay using IEEE style."""
    # Ensure data is numpy array
    data_np = np.asarray(data)
    
    # Plot magnitude for complex, otherwise plot raw data
    if np.iscomplexobj(data_np) and not is_error:
        data_to_plot = np.abs(data_np)
        cbar_label = 'Magnitude'
    else:
        data_to_plot = data_np
        cbar_label = 'Absolute Error' if is_error else 'Value'
    
    n_sc, n_sym = data_to_plot.shape
    x_coords = np.arange(n_sym + 1)
    y_coords = np.arange(n_sc + 1)
    
    # Use pcolormesh
    vmin = np.min(data_to_plot)
    vmax = np.max(data_to_plot)
    cmap = 'hot' if is_error else custom_cmap
    im = ax.pcolormesh(x_coords, y_coords, data_to_plot, cmap=cmap, 
                      shading='flat', vmin=vmin, vmax=vmax)
    
    # Overlay pilots if requested
    if show_pilots and pilot_mask is not None:
        pilot_y, pilot_x = np.where(pilot_mask)
        ax.scatter(pilot_x + 0.5, pilot_y + 0.5, c='white', s=8, marker='x', linewidths=0.8, alpha=0.9)
    
    ax.set_title(title)
    ax.set_xlabel('OFDM Block Index')
    ax.set_ylabel('Subcarrier Index')
    ax.set_xlim(0, n_sym)
    ax.set_ylim(0, n_sc)
    ax.set_aspect('auto')
    
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label)
    cbar.ax.tick_params(labelsize=plt.rcParams['xtick.labelsize'])
    
    return im

def plot_training_curves(train_losses, val_losses, save_path):
    """Plots training and validation loss curves and saves the plot with IEEE style."""
    with plt.style.context(IEEE_STYLE):
        plt.figure(figsize=(6, 4))
        plt.plot(train_losses, label='Training Loss', linewidth=1.5)
        plt.plot(val_losses, label='Validation Loss', linewidth=1.5)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss Curves')
        plt.legend()
        plt.grid(True)
        
        try:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path)
            print(f"Training curves plot saved to: {save_path}")
        except Exception as e:
            print(f"Error saving training curves plot: {e}")
        plt.close()

def plot_evaluation_samples(predictions_denorm, targets_original, sample_indices, base_save_path, 
                          display_plots=False, interpolated_data=None):
    """
    Enhanced plotting function that creates IEEE-style plots showing:
    1. Interpolated input vs Ground Truth vs Error (with and without pilots)
    2. Model prediction vs Ground Truth vs Error (with and without pilots)
    
    Args:
        predictions_denorm: Model predictions (denormalized)
        targets_original: Ground truth targets
        sample_indices: Indices of samples to plot
        base_save_path: Base path for saving plots
        display_plots: Whether to display plots
        interpolated_data: Interpolated input data (if available from dataset)
    """
    if not isinstance(base_save_path, Path):
        base_save_path = Path(base_save_path)
    
    pilot_mask = get_pilot_mask()
    
    for i, sample_idx in enumerate(sample_indices):
        if sample_idx >= len(predictions_denorm):
            print(f"Warning: Sample index {sample_idx} is out of bounds for predictions/targets. Skipping.")
            continue

        target_sample_original = targets_original[sample_idx]  # Shape (H, W, C)
        pred_sample_denorm = predictions_denorm[sample_idx]    # Shape (H, W, C)
        
        # Convert to complex for plotting
        gt_complex = target_sample_original[..., 0] + 1j * target_sample_original[..., 1]
        pred_complex = pred_sample_denorm[..., 0] + 1j * pred_sample_denorm[..., 1]
        
        # Calculate prediction error
        pred_error = np.abs(pred_complex - gt_complex)
        
        with plt.style.context(IEEE_STYLE):
            # --- Plot Set 1: Model Prediction vs Ground Truth ---
            for show_pilots in [False, True]:
                pilot_suffix = "_with_pilots" if show_pilots else "_no_pilots"
                
                # Create prediction comparison plot
                fig_width = 4 * 3  # 3 columns
                fig_height = 4
                fig = plt.figure(figsize=(fig_width, fig_height))
                
                # Plot prediction
                ax1 = fig.add_subplot(1, 3, 1)
                create_heatmap_with_pilots(ax1, pred_complex, 'Model Predicted Channel Gain', 
                                         pilot_mask, show_pilots, False, 'viridis')
                
                # Plot ground truth
                ax2 = fig.add_subplot(1, 3, 2)
                create_heatmap_with_pilots(ax2, gt_complex, 'Ground-Truth Channel Gain', 
                                         pilot_mask, show_pilots, False, 'viridis')
                
                # Plot prediction error
                ax3 = fig.add_subplot(1, 3, 3)
                create_heatmap_with_pilots(ax3, pred_error, 'Prediction Error', 
                                         pilot_mask, show_pilots, True, 'hot')
                
                plt.tight_layout()
                
                # Save prediction plot
                pred_save_path = base_save_path.parent / f"{base_save_path.stem}_sample_{sample_idx}_prediction{pilot_suffix}{base_save_path.suffix}"
                try:
                    pred_save_path.parent.mkdir(parents=True, exist_ok=True)
                    plt.savefig(pred_save_path, dpi=300)
                    print(f"Prediction plot for sample {sample_idx} saved to: {pred_save_path}")
                except Exception as e:
                    print(f"Error saving prediction plot for sample {sample_idx}: {e}")
                
                if display_plots:
                    plt.show()
                plt.close(fig)
                
            # --- Plot Set 2: Interpolated Input vs Ground Truth (if available) ---
            if interpolated_data is not None and sample_idx < len(interpolated_data):
                interp_sample = interpolated_data[sample_idx]  # Shape (H, W, C)
                interp_complex = interp_sample[..., 0] + 1j * interp_sample[..., 1]
                interp_error = np.abs(interp_complex - gt_complex)
                
                for show_pilots in [False, True]:
                    pilot_suffix = "_with_pilots" if show_pilots else "_no_pilots"
                    
                    # Create interpolation comparison plot
                    fig = plt.figure(figsize=(fig_width, fig_height))
                    
                    # Plot interpolated input
                    ax1 = fig.add_subplot(1, 3, 1)
                    create_heatmap_with_pilots(ax1, interp_complex, 'Interpolated Input', 
                                             pilot_mask, show_pilots, False, 'viridis')
                    
                    # Plot ground truth
                    ax2 = fig.add_subplot(1, 3, 2)
                    create_heatmap_with_pilots(ax2, gt_complex, 'Ground Truth', 
                                             pilot_mask, show_pilots, False, 'viridis')
                    
                    # Plot interpolation error
                    ax3 = fig.add_subplot(1, 3, 3)
                    create_heatmap_with_pilots(ax3, interp_error, 'Interpolation Error', 
                                             pilot_mask, show_pilots, True, 'hot')
                    
                    plt.tight_layout()
                    
                    # Save interpolation plot
                    interp_save_path = base_save_path.parent / f"{base_save_path.stem}_sample_{sample_idx}_interpolation{pilot_suffix}{base_save_path.suffix}"
                    try:
                        interp_save_path.parent.mkdir(parents=True, exist_ok=True)
                        plt.savefig(interp_save_path, dpi=300)
                        print(f"Interpolation plot for sample {sample_idx} saved to: {interp_save_path}")
                    except Exception as e:
                        print(f"Error saving interpolation plot for sample {sample_idx}: {e}")
                    
                    if display_plots:
                        plt.show()
                    plt.close(fig)

def plot_comprehensive_evaluation(predictions_denorm, targets_original, interpolated_data, 
                                sample_indices, save_dir, display_plots=False):
    """
    Comprehensive evaluation plotting function that creates all required plots.
    
    Args:
        predictions_denorm: Model predictions (denormalized)
        targets_original: Ground truth targets  
        interpolated_data: Interpolated input data
        sample_indices: Indices of samples to plot
        save_dir: Directory to save plots
        display_plots: Whether to display plots
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    for sample_idx in sample_indices:
        base_save_path = save_dir / f"eval_sample_{sample_idx}.svg"
        plot_evaluation_samples(
            predictions_denorm, 
            targets_original, 
            [sample_idx], 
            base_save_path,
            display_plots=display_plots,
            interpolated_data=interpolated_data
        ) 