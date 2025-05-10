"""
Filename: ./data_generation/utils/visualise_dataset.py
Author: Vincenzo Nannetti
Date: 09/05/2025 (Refactored Date)
Description: Script to visualise generated channel dataset .mat files.

This script loads a .mat file (output from the ray_tracing data generation or otherwise)
and allows for the visualisation of channel matrix heatmaps (perfect and/or noisy)
with an optional pilot mask overlay.

Running Instructions:
    python -m data_generation.utils.visualise_dataset --filepath <path_to_mat_file> [options]

Parameters & Options:
    --filepath:       Path to the .mat dataset file (required).
    --num_samples:    Number of samples to visualise (default: 1).
    --plot_perfect:   Plot the perfect channel matrices.
    --plot_noisy:     Plot the noisy (received) channel matrices.
    --overlay_mask:   Overlay the pilot mask on the plots.
    --random_select:  Randomly select samples. If False, selects the first --num_samples.
    --save_figs:      Save the figures instead of showing them interactively.
    --fig_dir:        Directory to save figures if --save_figs is used (default: ./visualisations).
    --plot_stats:     Generate statistical plots for the dataset (distributions, correlations, etc.).

Example:
    python -m data_generation.utils.visualise_dataset --filepath ./data/raw/dataset_a/dataset_a_4blocks_100samples.mat --num_samples 3 --overlay_mask --save_figs
"""

import argparse
import os
import numpy as np
import scipy.io
import matplotlib.pyplot as plt

try:
    from .plot_heatmaps import plot_perfect_vs_noisy_heatmap, plot_single_heatmap
except ImportError:
    print("Could not import from .plot_heatmaps, attempting direct import for plot_heatmaps.")
    try:
        import plot_heatmaps
        plot_perfect_vs_noisy_heatmap = plot_heatmaps.plot_perfect_vs_noisy_heatmap
        plot_single_heatmap = plot_heatmaps.plot_single_heatmap
    except ImportError:
        print("ERROR: plot_heatmaps.py not found. Make sure it's in the same directory or accessible in PYTHONPATH.")
        exit()

# Import functions for statistical plots
try:
    from .plot_data_statistics import (
        plot_distribution, plot_correlation, plot_noise_distribution,
        plot_pdp, setup_ieee_style
    )
except ImportError:
    print("Could not import from .plot_data_statistics, attempting direct import.")
    try:
        # Try to import directly if relative import fails
        from plot_data_statistics import (
            plot_distribution, plot_correlation, plot_noise_distribution,
            plot_pdp, setup_ieee_style
        )
    except ImportError:
        print("WARNING: plot_data_statistics.py not found. Statistical plotting will be disabled.")
        # Define placeholder functions that print warnings instead of failing
        def plot_distribution(*args, **kwargs):
            print("Statistical plotting disabled: plot_data_statistics.py not found.")
        def plot_correlation(*args, **kwargs):
            print("Statistical plotting disabled: plot_data_statistics.py not found.")
        def plot_noise_distribution(*args, **kwargs):
            print("Statistical plotting disabled: plot_data_statistics.py not found.")
        def plot_pdp(*args, **kwargs):
            print("Statistical plotting disabled: plot_data_statistics.py not found.")
        def setup_ieee_style():
            pass  # Do nothing for the style


def main():
    parser = argparse.ArgumentParser(description="Visualise channel dataset .mat files.")
    parser.add_argument("--filepath", type=str, required=True,
                        help="Path to the .mat dataset file.")
    parser.add_argument("--num_samples", type=int, default=1,
                        help="Number of samples to visualise from the dataset (default: 1).")
    parser.add_argument("--plot_perfect", action="store_true",
                        help="Plot the perfect channel matrices. If neither --plot_perfect nor --plot_noisy is set, both are plotted side-by-side.")
    parser.add_argument("--plot_noisy", action="store_true",
                        help="Plot the noisy (received) channel matrices. If neither --plot_perfect nor --plot_noisy is set, both are plotted side-by-side.")
    parser.add_argument("--overlay_mask", action="store_true",
                        help="Overlay the pilot mask on the plots.")
    parser.add_argument("--random_select", action="store_true",
                        help="Randomly select samples. If False, selects the first --num_samples (default: False).")
    parser.add_argument("--save_figs", action="store_true",
                        help="Save the figures instead of showing them interactively.")
    parser.add_argument("--fig_dir", type=str, default="./visualisations",
                        help="Directory to save figures if --save_figs is used (default: ./visualisations).")
    parser.add_argument("--plot_stats", action="store_true",
                        help="Generate statistical plots for the dataset (distributions, correlations, etc.).")

    args = parser.parse_args()

    # Determine what to plot based on flags
    plot_both_default = not args.plot_perfect and not args.plot_noisy
    plot_perfect_flag = args.plot_perfect or plot_both_default
    plot_noisy_flag = args.plot_noisy or plot_both_default

    if not plot_perfect_flag and not plot_noisy_flag and not args.plot_stats:
        print("Nothing to plot. Use --plot_perfect, --plot_noisy, or --plot_stats.")
        return

    if not os.path.exists(args.filepath):
        print(f"Error: File not found at {args.filepath}")
        return

    if args.save_figs:
        os.makedirs(args.fig_dir, exist_ok=True)

    try:
        data = scipy.io.loadmat(args.filepath)
        print(f"Successfully loaded {args.filepath}")
        # print("Available keys in .mat file:", list(data.keys()))
    except Exception as e:
        print(f"Error loading .mat file: {e}")
        return

    perfect_matrices_data = data.get('perfect_matrices')
    received_matrices_data = data.get('received_matrices')
    pilot_masks_data = data.get('pilot_mask')

    if pilot_masks_data is not None:
        pilot_masks_data = pilot_masks_data.astype(bool)

    total_available_samples = 0
    if perfect_matrices_data is not None:
        total_available_samples = perfect_matrices_data.shape[0]
    elif received_matrices_data is not None:
        total_available_samples = received_matrices_data.shape[0]
    else:
        print("Error: Neither 'perfect_matrices' nor 'received_matrices' found in the .mat file.")
        return
    
    if args.num_samples > total_available_samples:
        print(f"Warning: Requested {args.num_samples} samples, but only {total_available_samples} are available. Plotting all available samples.")
        args.num_samples = total_available_samples

    if args.num_samples == 0:
        print("Number of samples to plot is zero. Exiting.")
        return

    if args.random_select:
        sample_indices = np.random.choice(total_available_samples, args.num_samples, replace=False)
    else:
        sample_indices = np.arange(args.num_samples)

    # Setup IEEE style if plotting statistics
    if args.plot_stats:
        try:
            setup_ieee_style()
        except Exception as e:
            print(f"Warning: Could not set IEEE plot style: {e}. Using default style.")

    # Create stats directory for saving plots
    # When using --plot_stats, we always need to save the plots since the functions don't support showing interactively
    stats_save_plots = args.plot_stats  # If plotting stats, we need to save them
    temp_dir = None
    
    if stats_save_plots:
        if args.save_figs:
            # Use the user-specified directory
            stats_dir = os.path.join(args.fig_dir, "statistics")
            os.makedirs(stats_dir, exist_ok=True)
            print(f"Statistical plots will be saved to: {stats_dir}")
        else:
            # Create a temporary directory for plots if not saving explicitly
            import tempfile
            temp_dir = tempfile.mkdtemp(prefix="channel_stats_")
            stats_dir = temp_dir
            print(f"Statistical plots will be temporarily saved to: {stats_dir} (Delete manually when done viewing)")

    # Plot statistics if requested (for the entire dataset)
    if args.plot_stats and perfect_matrices_data is not None and received_matrices_data is not None:
        print("\n--- Generating Statistical Plots ---")
        dataset_name = os.path.splitext(os.path.basename(args.filepath))[0]
        
        # Magnitude Distribution
        plot_distribution(np.abs(perfect_matrices_data), np.abs(received_matrices_data),
                        'Perfect Channel', 'Noisy Channel',
                        'Magnitude', stats_dir, 
                        f'magnitude_distributions_{dataset_name}.svg')
        
        # Open the plot files automatically if not saving
        if not args.save_figs and temp_dir:
            import webbrowser
            try:
                magnitude_plot_path = os.path.join(stats_dir, f'magnitude_distributions_{dataset_name}.svg')
                webbrowser.open(magnitude_plot_path)
            except Exception as e:
                print(f"Warning: Could not open plot file: {e}")
                
        # Power Delay Profile (PDP)
        try:
            plot_pdp(perfect_matrices_data, stats_dir)
            # Open the plot if not saving
            if not args.save_figs and temp_dir:
                try:
                    pdp_plot_path = os.path.join(stats_dir, 'average_pdp.svg')
                    webbrowser.open(pdp_plot_path)
                except Exception as e:
                    print(f"Warning: Could not open PDP plot: {e}")
        except Exception as e:
            print(f"Warning: Could not generate PDP plot: {e}")
        
        # Correlation Plot
        try:
            plot_correlation(perfect_matrices_data, received_matrices_data, stats_dir)
            # Open the plot if not saving
            if not args.save_figs and temp_dir:
                try:
                    corr_plot_path = os.path.join(stats_dir, 'average_correlation.svg')
                    webbrowser.open(corr_plot_path)
                except Exception as e:
                    print(f"Warning: Could not open correlation plot: {e}")
        except Exception as e:
            print(f"Warning: Could not generate correlation plot: {e}")
        
        # Noise Distribution
        try:
            plot_noise_distribution(perfect_matrices_data, received_matrices_data, stats_dir)
            # Open the plot if not saving
            if not args.save_figs and temp_dir:
                try:
                    noise_plot_path = os.path.join(stats_dir, 'noise_distribution.svg')
                    webbrowser.open(noise_plot_path)
                except Exception as e:
                    print(f"Warning: Could not open noise distribution plot: {e}")
        except Exception as e:
            print(f"Warning: Could not generate noise distribution plot: {e}")
            
        print("Statistical plots generated.")

    # Plot individual samples
    for i, sample_idx in enumerate(sample_indices):
        print(f"Processing sample {i+1}/{args.num_samples} (index {sample_idx} from dataset)...")
        
        current_pilot_mask = None
        if args.overlay_mask and pilot_masks_data is not None and sample_idx < pilot_masks_data.shape[0]:
            current_pilot_mask = pilot_masks_data[sample_idx]
        elif args.overlay_mask:
            print(f"Warning: Pilot mask not available for sample {sample_idx} or 'pilot_mask' key missing.")

        sample_perfect = perfect_matrices_data[sample_idx] if perfect_matrices_data is not None and sample_idx < perfect_matrices_data.shape[0] else None
        sample_noisy = received_matrices_data[sample_idx] if received_matrices_data is not None and sample_idx < received_matrices_data.shape[0] else None

        sample_info_str = f"Sample {sample_idx}"
        base_figname = f"sample_{sample_idx}"

        if plot_perfect_flag and plot_noisy_flag:
            # Plot side-by-side using the dedicated function
            if sample_perfect is None and sample_noisy is None:
                print(f"Neither perfect nor noisy data available for sample {sample_idx}. Skipping.")
                continue
            plot_perfect_vs_noisy_heatmap(sample_perfect, sample_noisy, sample_info_str,
                                          pilot_mask=current_pilot_mask,
                                          savefig=args.save_figs, 
                                          figname=f"{base_figname}_perfect_vs_noisy.png",
                                          fig_dir=args.fig_dir, 
                                          show_fig=not args.save_figs)
        elif plot_perfect_flag:
            if sample_perfect is not None:
                fig, ax = plt.subplots(1, 1, figsize=(6, 4))
                plot_single_heatmap(sample_perfect, ax, title=f"Perfect Channel: {sample_info_str}",
                                    pilot_mask=current_pilot_mask)
                plt.tight_layout()
                if args.save_figs:
                    fpath = os.path.join(args.fig_dir, f"{base_figname}_perfect.png")
                    plt.savefig(fpath)
                    print(f"Saved figure to {fpath}")
                if not args.save_figs:
                    plt.show()
                else:
                    plt.close(fig)
            else:
                print(f"Perfect matrix not available for sample {sample_idx}.")

        elif plot_noisy_flag:
            if sample_noisy is not None:
                fig, ax = plt.subplots(1, 1, figsize=(6, 4))
                plot_single_heatmap(sample_noisy, ax, title=f"Noisy Channel: {sample_info_str}",
                                    pilot_mask=current_pilot_mask)
                plt.tight_layout()
                if args.save_figs:
                    fpath = os.path.join(args.fig_dir, f"{base_figname}_noisy.png")
                    plt.savefig(fpath)
                    print(f"Saved figure to {fpath}")
                if not args.save_figs:
                    plt.show()
                else:
                    plt.close(fig)
            else:
                print(f"Noisy matrix not available for sample {sample_idx}.")
    
    if args.save_figs:
        if args.plot_stats:
            print(f"Finished processing. Heatmaps saved in {args.fig_dir}, statistics plots saved in {stats_dir}")
        else:
            print(f"Finished processing. Figures saved in {args.fig_dir}")

if __name__ == "__main__":
    main() 