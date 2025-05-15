"""
Filename: ./utils/plot_heatmap.py
Author: Vincenzo Nannetti
Date: 26/03/2025
Description: Utility function for plotting heatmaps of complex matrices from channel estimation
             Updated to match IEEE style from data_visualisation.py
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns # Keep for potential future use, but pcolormesh is primary now
import os
import logging

# IEEE Plot Style Configuration (adapted from data_visualisation.py)
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
    # Figure size (will be overridden by dynamic sizing below, but keep for reference)
    # 'figure.figsize': (3.5, 2.5),
    # Line widths and markers (less relevant for heatmaps)
    'lines.linewidth': 1.0,
    'lines.markersize': 4,
    # Grid
    'axes.grid': True,
    'grid.linestyle': ':',
    'grid.linewidth': 0.5,
    # Save settings
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.format': 'svg' # Default to SVG
}

def plot_heatmap(
    interp=None,
    srcnn=None,
    dncnn=None,
    perfect=None,
    ae=None,
    combined=None,
    save_path=None,
    filename=None,
    show=True,
    use_titles=None,
    custom_cmap='viridis',
    figsize_per_plot=(4, 3), # Slightly adjusted default size for pcolormesh layout
    error_plot=False,
    split_interp=True  # New parameter to control split layout
):
    """
    Plot heatmaps of complex matrices for channel estimation visualization using IEEE style.
    Creates two separate figures:
    1. First figure: Interpolation input and its error (side-by-side)
    2. Second figure: Combined/predicted output, perfect/ground truth, and error between them

    Args:
        interp (complex array, optional): Interpolated/noisy input matrix.
        srcnn (complex array, optional): SRCNN model output matrix.
        dncnn (complex array, optional): DnCNN model output matrix.
        perfect (complex array, optional): Perfect/ground truth matrix.
        ae (complex array, optional): Autoencoder output matrix.
        combined (complex array, optional): Combined model output matrix.
        save_path (str, optional): Directory to save the plot. If None, plot won't be saved.
        filename (str, optional): Filename to save the plot (including extension). If None, defaults to .svg.
        show (bool, optional): Whether to show the plot using plt.show(). Default is True.
        use_titles (dict, optional): Custom titles for each plot type. Keys should match parameter names.
        custom_cmap (str, optional): Colormap to use for the main heatmaps. Default is 'viridis'.
        figsize_per_plot (tuple, optional): Figure size per subplot. Default is (4, 3).
        error_plot (bool, optional): Whether to include error plots (magnitude difference). Default is False.
        split_interp (bool, optional): Whether to create a separate figure for interpolation and its error.

    Returns:
        tuple: (interp_fig, model_fig) - Two figures with interp+error and model+perfect+error
    """
    # Convert tensors to numpy arrays if needed
    if hasattr(interp, 'numpy'):
        # Handle case where input is a tensor with multiple channels
        if len(interp.shape) >= 3 and interp.shape[0] >= 2:
            # Extract only the first two channels (real & imag) for complex data
            interp = interp[0] + 1j * interp[1]
        else:
            interp = interp.numpy()
    
    if hasattr(combined, 'numpy'):
        if len(combined.shape) >= 3 and combined.shape[0] >= 2:
            # Extract only the first two channels (real & imag) for complex data
            combined = combined[0] + 1j * combined[1]
        else:
            combined = combined.numpy()
            
    if hasattr(perfect, 'numpy'):
        if len(perfect.shape) >= 3 and perfect.shape[0] >= 2:
            # Extract only the first two channels (real & imag) for complex data
            perfect = perfect[0] + 1j * perfect[1]
        else:
            perfect = perfect.numpy()
    
    # Default titles for each plot type
    default_titles = {
        'interp': 'Interpolated Input',
        'srcnn': 'SRCNN Output',
        'dncnn': 'DnCNN Output',
        'perfect': 'Perfect Matrix',
        'ae': 'Autoencoder Output',
        'combined': 'Combined Model Output'
    }

    # Use custom titles if provided
    titles = default_titles.copy()
    if use_titles is not None:
        titles.update(use_titles)

    # Build a list of the plots to show
    data_arrays = {
        'interp': interp, 'srcnn': srcnn, 'dncnn': dncnn,
        'ae': ae, 'combined': combined, 'perfect': perfect
    }
    
    # Check if we have interpolation data and perfect data for error plot
    has_interp = interp is not None
    has_perfect = perfect is not None
    has_combined = combined is not None
    can_show_error = has_perfect and error_plot

    # Helper to get title with "Error" suffix for error plots
    def get_error_title(plot_type):
        base_key = plot_type.replace('_error', '')
        base_title = titles.get(base_key, base_key)
        return f"{base_title} Error"

    # Helper function to create heatmap on provided axis
    def create_heatmap(ax, data, title, is_error=False):
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
        ax.set_title(title)
        ax.set_xlabel('OFDM Symbol Index')
        ax.set_ylabel('Subcarrier Index')
        ax.set_xlim(0, n_sym)
        ax.set_ylim(0, n_sc)
        ax.set_aspect('auto')
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(cbar_label)
        cbar.ax.tick_params(labelsize=plt.rcParams['xtick.labelsize'])
        return im

    # Create separate figures based on split_interp flag
    with plt.style.context(IEEE_STYLE):
        interp_fig = None
        model_fig = None
        
        # Create interpolation + error figure if requested and possible
        if split_interp and has_interp and has_perfect:
            interp_width = figsize_per_plot[0] * 3  # 3 columns for interp, perfect, error
            interp_height = figsize_per_plot[1]
            interp_fig = plt.figure(figsize=(interp_width, interp_height))
            
            # Three-panel plot with interpolation, perfect, and error
            ax1 = interp_fig.add_subplot(1, 3, 1)
            ax2 = interp_fig.add_subplot(1, 3, 2)
            ax3 = interp_fig.add_subplot(1, 3, 3)
            
            # Plot interpolation
            create_heatmap(ax1, interp, titles['interp'])
            
            # Plot perfect/ground truth
            create_heatmap(ax2, perfect, titles['perfect'])
            
            # Plot interpolation error if we have perfect reference
            if can_show_error:
                interp_error = np.abs(np.asarray(interp) - np.asarray(perfect))
                create_heatmap(ax3, interp_error, get_error_title('interp'), is_error=True)
            else:
                ax3.text(0.5, 0.5, 'Error calculation requires perfect reference',
                        ha='center', va='center', transform=ax3.transAxes)
                ax3.set_title("Error (unavailable)")
            
            plt.tight_layout()
            
            # Save interpolation figure if path is provided
            if save_path is not None:
                os.makedirs(save_path, exist_ok=True)
                interp_filename = "interpolation_perfect_error.svg"
                if filename is not None:
                    base, ext = os.path.splitext(filename)
                    ext = ext if ext else f".{IEEE_STYLE['savefig.format']}" # Use style format
                    interp_filename = f"{base}_interp{ext}"
                interp_fig.savefig(os.path.join(save_path, interp_filename))
                print(f"Saved interpolation, perfect, and error plot to: {os.path.join(save_path, interp_filename)}")
        
        # Create model comparison figure with prediction, perfect, and error
        # Identify which model output to use (prioritizing combined > ae > srcnn > dncnn)
        model_output = None
        model_key = None
        for key in ['combined', 'ae', 'srcnn', 'dncnn']:
            if data_arrays[key] is not None:
                model_output = data_arrays[key]
                model_key = key
                break
        
        if model_output is not None and has_perfect:
            # Create 1x3 figure: prediction, perfect, error
            model_width = figsize_per_plot[0] * 3  # 3 columns
            model_height = figsize_per_plot[1]
            model_fig = plt.figure(figsize=(model_width, model_height))
            
            # Plot prediction (model output)
            ax1 = model_fig.add_subplot(1, 3, 1)
            create_heatmap(ax1, model_output, titles.get(model_key, model_key))
            
            # Plot perfect (ground truth)
            ax2 = model_fig.add_subplot(1, 3, 2)
            create_heatmap(ax2, perfect, titles['perfect'])
            
            # Plot error between prediction and perfect
            ax3 = model_fig.add_subplot(1, 3, 3)
            if can_show_error:
                model_error = np.abs(np.asarray(model_output) - np.asarray(perfect))
                create_heatmap(ax3, model_error, get_error_title(model_key), is_error=True)
            else:
                ax3.text(0.5, 0.5, 'Error calculation requires perfect reference',
                        ha='center', va='center', transform=ax3.transAxes)
                ax3.set_title("Error (unavailable)")
            
            plt.tight_layout()
            
            # Save model figure if path is provided
            if save_path is not None:
                os.makedirs(save_path, exist_ok=True)
                model_filename = "prediction_perfect_error.svg"
                if filename is not None:
                    base, ext = os.path.splitext(filename)
                    ext = ext if ext else f".{IEEE_STYLE['savefig.format']}" # Use style format
                    model_filename = f"{base}_models{ext}"
                model_fig.savefig(os.path.join(save_path, model_filename))
                print(f"Saved prediction, perfect, and error plot to: {os.path.join(save_path, model_filename)}")
        
        # If not using split layout, create a single comprehensive figure
        if not split_interp:
            plots = []
            for key in ['interp'] + ['combined', 'ae', 'srcnn', 'dncnn'] + ['perfect']:
                if data_arrays[key] is not None:
                    plots.append((key, data_arrays[key]))
            
            if not plots:
                print("No data provided to plot")
                return None, None

            # Add error plots if requested and we have perfect reference
            error_plots = []
            if error_plot and perfect is not None:
                for key, data in plots:
                    if key != 'perfect': # Don't plot error against itself
                        # Ensure data and perfect are numpy arrays for subtraction
                        data_np = np.asarray(data)
                        perfect_np = np.asarray(perfect)
                        if data_np.shape == perfect_np.shape:
                            # Error is the magnitude of the difference
                            error_plots.append((f'{key}_error', np.abs(data_np - perfect_np)))
                        else:
                            print(f"Warning: Shape mismatch for error plot between '{key}' {data_np.shape} and perfect {perfect_np.shape}. Skipping error plot.")

            # Determine the number of subplots
            total_main_plots = len(plots)
            
            # Determine rows and columns for the subplot grid
            if error_plots:
                # For error plots, use 2 rows: data on top, errors on bottom
                n_rows = 2
                n_cols = total_main_plots
            else:
                # Without error plots, arrange based on total number
                n_cols = min(3, total_main_plots) # Max 3 columns
                n_rows = (total_main_plots + n_cols - 1) // n_cols # Calculate rows needed
            
            fig_width = n_cols * figsize_per_plot[0]
            fig_height = n_rows * figsize_per_plot[1]
            fig = plt.figure(figsize=(fig_width, fig_height))
            
            # Create grid of subplots
            gs = fig.add_gridspec(n_rows, n_cols)
            
            # Plot the main heatmaps
            for i, (plot_type, data) in enumerate(plots):
                if error_plots:
                    row, col = 0, i
                else:
                    row, col = i // n_cols, i % n_cols
                
                ax = fig.add_subplot(gs[row, col])
                create_heatmap(ax, data, titles.get(plot_type, plot_type))
            
            # Plot the error heatmaps
            if error_plots:
                for i, (plot_type, data) in enumerate(error_plots):
                    base_key = plot_type.replace('_error', '')
                    for j, (key, _) in enumerate(plots):
                        if key == base_key:
                            row, col = 1, j
                            ax = fig.add_subplot(gs[row, col])
                            create_heatmap(ax, data, get_error_title(plot_type), is_error=True)
                            break
            
            plt.tight_layout()
            
            # Save figure if path is provided
            if save_path is not None:
                os.makedirs(save_path, exist_ok=True)
                save_filename = "channel_estimation_comparison.svg"
                if filename is not None:
                    save_filename = filename if os.path.splitext(filename)[1] else f"{filename}.{IEEE_STYLE['savefig.format']}" # Use style format
                
                fig.savefig(os.path.join(save_path, save_filename))
                print(f"Saved heatmap plot to: {os.path.join(save_path, save_filename)}")
            
            # If not splitting, interp_fig holds the single figure, model_fig is None
            interp_fig = fig 
            model_fig = None # Ensure model_fig is None when not split

    # --- Final Steps: Show or Close --- 
    # Show plots ONLY if requested AND figures were generated
    if show and (interp_fig or model_fig):
        plt.show()
    # Otherwise, close figures IF THEY EXIST to free memory, unless shown.
    # If shown, matplotlib handles closing usually.
    else: 
        if interp_fig:
            plt.close(interp_fig)
        if model_fig:
            plt.close(model_fig)

    # Return the figure handles regardless of saving/showing
    return interp_fig, model_fig
