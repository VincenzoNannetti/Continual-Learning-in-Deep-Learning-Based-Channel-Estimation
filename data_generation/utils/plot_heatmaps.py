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

def plot_single_heatmap(matrix_data, ax, title, xlabel="OFDM Symbol Index", ylabel="Subcarrier Index", 
                        pilot_mask=None, cmap='viridis', vmin=None, vmax=None):
    """Plots a single magnitude heatmap on the given Matplotlib Axes object."""
    n_sc, n_sym = matrix_data.shape
    x_coords = np.arange(n_sym + 1)
    y_coords = np.arange(n_sc + 1)

    im = ax.pcolormesh(x_coords, y_coords, np.abs(matrix_data), cmap=cmap, shading='flat', vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    x_tick_spacing = max(1, int(round(n_sym / 7)))
    ax.set_xticks(np.arange(0, n_sym + 1, x_tick_spacing))
    ax.set_yticks(np.arange(0, n_sc + 1, max(1, n_sc // 10)))
    ax.set_xlim(0, n_sym)
    ax.set_ylim(0, n_sc)
    ax.set_aspect('auto')

    cbar = plt.gcf().colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Magnitude')
    # Explicitly set colorbar tick label size to match xtick.labelsize from the style
    if 'xtick.labelsize' in plt.rcParams:
        cbar.ax.tick_params(labelsize=plt.rcParams['xtick.labelsize'])

    if pilot_mask is not None:
        if pilot_mask.shape == matrix_data.shape:
            pilot_row_indices, pilot_col_indices = np.where(pilot_mask)
            ax.scatter(pilot_col_indices + 0.5, pilot_row_indices + 0.5, 
                       marker='x', color='red', s=50, label='Pilots')
        else:
            print("Warning: Pilot mask shape mismatch. Cannot overlay pilots.")

def plot_perfect_vs_noisy_heatmap(perfect_matrix, noisy_matrix, sample_info_str, 
                                  pilot_mask=None, savefig=False, figname="heatmap.png", 
                                  fig_dir=".", show_fig=True, use_shared_cbar=True):
    """Plots perfect and noisy channel magnitude heatmaps side-by-side for a sample."""
    # Figure size matches your original plot_example_heatmap for side-by-side
    fig, axs = plt.subplots(1, 2, figsize=(7.5, 3.0)) 
    fig.suptitle(f'Channel Magnitude: {sample_info_str}', y=0.95) # Adjusted y for consistency with your example

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
                use_shared_cbar = False # No data to determine range
        else:
            use_shared_cbar = False # No data

    if perfect_matrix is not None:
        plot_single_heatmap(perfect_matrix, axs[0], title='Perfect Channel', 
                            pilot_mask=pilot_mask, 
                            vmin=common_vmin if use_shared_cbar else None, 
                            vmax=common_vmax if use_shared_cbar else None)
    else:
        axs[0].set_title('Perfect Channel (Not Available)')
        axs[0].text(0.5, 0.5, 'N/A', ha='center', va='center', transform=axs[0].transAxes)
        axs[0].set_xlabel('OFDM Symbol Index')
        axs[0].set_ylabel('Subcarrier Index')

    if noisy_matrix is not None:
        plot_single_heatmap(noisy_matrix, axs[1], title='Noisy Channel', 
                            pilot_mask=pilot_mask,
                            vmin=common_vmin if use_shared_cbar else None, 
                            vmax=common_vmax if use_shared_cbar else None)
    else:
        axs[1].set_title('Noisy Channel (Not Available)')
        axs[1].text(0.5, 0.5, 'N/A', ha='center', va='center', transform=axs[1].transAxes)
        axs[1].set_xlabel('OFDM Symbol Index')
        axs[1].set_ylabel('Subcarrier Index')

    # Adjusted tight_layout rect to match your example
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 

    if savefig:
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir, exist_ok=True)
        full_figname = os.path.join(fig_dir, figname)
        plt.savefig(full_figname) # savefig format is now controlled by rcParams from setup_ieee_style
        print(f"Saved heatmap to {full_figname}")
    
    if show_fig:
        plt.show()
    elif not savefig: # If not showing and not saving, close to free memory
        plt.close(fig)
    elif savefig and not show_fig: # If saved and not showing, also close
        plt.close(fig)

# Example usage (can be commented out or removed)
if __name__ == '__main__':
    n_sc, n_sym = 72, 56
    dummy_perfect = np.random.rand(n_sc, n_sym) + 1j * np.random.rand(n_sc, n_sym)
    dummy_noisy = dummy_perfect + (np.random.randn(n_sc, n_sym) * 0.1 + 1j * np.random.randn(n_sc, n_sym) * 0.1)
    dummy_pilot_mask = np.zeros((n_sc, n_sym), dtype=bool)
    dummy_pilot_mask[::12, ::4] = True

    plot_perfect_vs_noisy_heatmap(dummy_perfect, dummy_noisy, "Sample Test 1 (Shared Cbar)", 
                                  pilot_mask=dummy_pilot_mask, savefig=False, show_fig=True, use_shared_cbar=True)
    plot_perfect_vs_noisy_heatmap(dummy_perfect, dummy_noisy, "Sample Test 2 (Separate Cbar)", 
                                  pilot_mask=dummy_pilot_mask, savefig=False, show_fig=True, use_shared_cbar=False)

    fig_single, ax_single = plt.subplots(1, 1, figsize=(3.5, 2.5)) # Example single plot using style's default size
    plot_single_heatmap(dummy_perfect, ax_single, title="Single Perfect with Pilots", pilot_mask=dummy_pilot_mask)
    plt.tight_layout()
    plt.show()
