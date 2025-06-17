"""
Filename: ./data_generation/utils/plot_heatmaps.py
Author: Vincenzo Nannetti
Date: 09/05/2025 
Description: Plotting functions for heatmaps. The channel matrices are complex;
             this function plots the magnitude.
"""

import matplotlib.pyplot as plt
import numpy as np
import os

# Attempt to import and apply IEEE plot style from ieee_plot_style.py
try:
    from .ieee_plot_style import setup_ieee_style
    setup_ieee_style() # Apply the style
except ImportError:
    print("Warning: Could not import setup_ieee_style from .ieee_plot_style. Using Matplotlib defaults.")
    try:
        import ieee_plot_style # Fallback for direct script run if not as module
        ieee_plot_style.setup_ieee_style()
    except ImportError:
        print("Warning: ieee_plot_style.py not found. Using Matplotlib defaults.")

# Add IEEE style settings at the top
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
    'savefig.format': 'svg'
}

def plot_single_heatmap(matrix_data, ax, title, xlabel="Time Block Index", ylabel="Subcarrier Index", 
                        pilot_mask=None, cmap='viridis', vmin=None, vmax=None):
    """Plots a single magnitude heatmap on the given Matplotlib Axes object."""
    with plt.style.context(IEEE_STYLE):
        n_sc, n_sym = matrix_data.shape
        x_coords = np.arange(n_sym + 1)
        y_coords = np.arange(n_sc + 1)

        im = ax.pcolormesh(x_coords, y_coords, np.abs(matrix_data), cmap=cmap, 
                          shading='flat', vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xlim(0, n_sym)
        ax.set_ylim(0, n_sc)
        ax.set_aspect('auto')

        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Magnitude')
        cbar.ax.tick_params(labelsize=plt.rcParams['xtick.labelsize'])

        # if pilot_mask is not None:
        #     if pilot_mask.shape == matrix_data.shape:
        #         pilot_rows, pilot_cols = np.where(pilot_mask)
        #         ax.scatter(pilot_cols + 0.5, pilot_rows + 0.5, 
        #                   marker='x', color='red', s=20, label='Pilots')
        #     else:
        #         print("Warning: Pilot mask shape mismatch. Cannot overlay pilots.")

def plot_interpolation_comparison(interpolated, perfect, noisy=None, pilot_mask=None, 
                                save_path=None, filename=None, show=True):
    """
    Plot comparison between interpolated, perfect, and error matrices.
    
    Args:
        interpolated: Interpolated channel matrix
        perfect: Perfect/ground truth channel matrix
        noisy: Original noisy matrix (optional)
        pilot_mask: Pilot positions mask (optional)
        save_path: Directory to save the plot
        filename: Filename for saving
        show: Whether to display the plot
    """
    with plt.style.context(IEEE_STYLE):
        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        
        # Plot interpolated result
        im1 = axs[0].pcolormesh(np.abs(interpolated), cmap='viridis', shading='flat')
        axs[0].set_title('Interpolated Channel')
        plt.colorbar(im1, ax=axs[0], fraction=0.046, pad=0.04).set_label('Magnitude')
        
        # Plot perfect channel
        im2 = axs[1].pcolormesh(np.abs(perfect), cmap='viridis', shading='flat')
        axs[1].set_title('Perfect Channel')
        plt.colorbar(im2, ax=axs[1], fraction=0.046, pad=0.04).set_label('Magnitude')
        
        # Plot error
        error = np.abs(interpolated - perfect)
        im3 = axs[2].pcolormesh(error, cmap='hot', shading='flat')
        axs[2].set_title('Error')
        plt.colorbar(im3, ax=axs[2], fraction=0.046, pad=0.04).set_label('Absolute Error')
        
        # Set common labels and adjustments
        for ax in axs:
            ax.set_xlabel('Time Block Index')
            ax.set_ylabel('Subcarrier Index')
            ax.set_aspect('auto')
            
            # Add pilot positions if mask is provided
            # if pilot_mask is not None:
            #     pilot_rows, pilot_cols = np.where(pilot_mask)
            #     ax.scatter(pilot_cols + 0.5, pilot_rows + 0.5, 
            #               marker='x', color='red', s=20, label='Pilots')
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            full_path = os.path.join(save_path, filename or 'interpolation_comparison.svg')
            plt.savefig(full_path)
            print(f"Saved plot to {full_path}")
        
        if show:
            plt.show()
        else:
            plt.close(fig)
        
        return fig

def plot_perfect_vs_noisy_heatmap(perfect_matrix, noisy_matrix, sample_info_str, 
                                  pilot_mask=None, savefig=False, figname="heatmap.svg", 
                                  fig_dir=".", show_fig=True, use_shared_cbar=True):
    """Plots perfect and noisy channel magnitude heatmaps side-by-side for a sample."""
    with plt.style.context(IEEE_STYLE):
        # Increased figure size for better quality
        fig, axs = plt.subplots(1, 2, figsize=(12, 5)) 
        fig.suptitle(f'Channel Magnitude: {sample_info_str}', y=0.95)

        common_vmin, common_vmax = None, None
        if use_shared_cbar:
            abs_values = []
            if perfect_matrix is not None:
                abs_values.append(np.abs(perfect_matrix))
            if noisy_matrix is not None:
                abs_values.append(np.abs(noisy_matrix))
            if abs_values:
                all_abs_values = np.concatenate([val.flatten() for val in abs_values])
                if all_abs_values.size > 0:
                    common_vmin = np.min(all_abs_values)
                    common_vmax = np.max(all_abs_values)
                else:
                    use_shared_cbar = False
            else:
                use_shared_cbar = False

        if perfect_matrix is not None:
            plot_single_heatmap(perfect_matrix, axs[0], title='Perfect Channel', 
                                pilot_mask=pilot_mask, 
                                vmin=common_vmin if use_shared_cbar else None, 
                                vmax=common_vmax if use_shared_cbar else None)
        else:
            axs[0].set_title('Perfect Channel (Not Available)')
            axs[0].text(0.5, 0.5, 'N/A', ha='center', va='center', transform=axs[0].transAxes)
            axs[0].set_xlabel('Time Block Index')
            axs[0].set_ylabel('Subcarrier Index')

        if noisy_matrix is not None:
            plot_single_heatmap(noisy_matrix, axs[1], title='Noisy Channel', 
                                pilot_mask=pilot_mask,
                                vmin=common_vmin if use_shared_cbar else None, 
                                vmax=common_vmax if use_shared_cbar else None)
        else:
            axs[1].set_title('Noisy Channel (Not Available)')
            axs[1].text(0.5, 0.5, 'N/A', ha='center', va='center', transform=axs[1].transAxes)
            axs[1].set_xlabel('Time Block Index')
            axs[1].set_ylabel('Subcarrier Index')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        if savefig:
            if not os.path.exists(fig_dir):
                os.makedirs(fig_dir, exist_ok=True)
            full_figname = os.path.join(fig_dir, figname)
            # Save as SVG with high quality settings
            plt.savefig(full_figname, format='svg', bbox_inches='tight', pad_inches=0.1)
            print(f"Saved high-quality SVG heatmap to {full_figname}")
        
        if show_fig:
            plt.show()
        elif not savefig:
            plt.close(fig)
        elif savefig and not show_fig:
            plt.close(fig)

