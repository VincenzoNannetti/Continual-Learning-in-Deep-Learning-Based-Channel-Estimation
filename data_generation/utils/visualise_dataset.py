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
    --show_interpolation: Show interpolated channel matrices.
    --sample_idx:     Specific sample index to visualize. Overrides --random_select.
    --interpolation_method: Interpolation method to use (default: rbf).
    --sampling_rate_symbols: Sampling rate of the time blocks in Hz (e.g., 1 / block_duration). A block consists of 14 OFDM symbols. Example default is for 14 symbols of 71.4us each. Needed for Doppler spectrum.

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
        plot_pdp, setup_ieee_style,
        plot_doppler_spectrum
    )
except ImportError:
    print("Could not import from .plot_data_statistics, attempting direct import.")
    try:
        # Try to import directly if relative import fails
        from plot_data_statistics import (
            plot_distribution, plot_correlation, plot_noise_distribution,
            plot_pdp, setup_ieee_style,
            plot_doppler_spectrum
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
                        help="Plot the perfect channel matrices.")
    parser.add_argument("--plot_noisy", action="store_true",
                        help="Plot the noisy (received) channel matrices.")
    parser.add_argument("--overlay_mask", action="store_true",
                        help="Overlay the pilot mask on the plots.")
    parser.add_argument("--random_select", action="store_true",
                        help="Randomly select samples. If False, selects the first --num_samples.")
    parser.add_argument("--save_figs", action="store_true",
                        help="Save the figures instead of showing them interactively.")
    parser.add_argument("--fig_dir", type=str, default="./visualisations",
                        help="Directory to save figures if --save_figs is used.")
    parser.add_argument("--plot_stats", action="store_true",
                        help="Generate statistical plots for the dataset.")
    # Add new arguments for interpolation
    parser.add_argument("--show_interpolation", action="store_true",
                        help="Show interpolated channel matrices.")
    parser.add_argument("--sample_idx", type=int,
                        help="Specific sample index to visualize. Overrides --random_select and --num_samples.")
    parser.add_argument("--sample_indices", type=str,
                        help="Comma-separated list of sample indices to visualize (e.g., '0,5,10'). Overrides other selection methods.")
    parser.add_argument("--interpolation_method", type=str, default="rbf",
                        choices=["rbf", "spline"],
                        help="Interpolation method to use (default: rbf).")
    parser.add_argument("--sampling_rate_symbols", type=float, default=1.0 / (16.67e-6 * 14), 
                        help="Sampling rate of the time blocks in Hz (e.g., 1 / block_duration). A block consists of 14 OFDM symbols. Example default is for 14 symbols of 71.4us each. Needed for Doppler spectrum.")

    args = parser.parse_args()

    # Load the data
    try:
        data = scipy.io.loadmat(args.filepath)
        print(f"Successfully loaded {args.filepath}")
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

    # Handle sample selection
    if args.sample_indices is not None:
        # Parse comma-separated list of indices
        try:
            sample_indices = [int(idx.strip()) for idx in args.sample_indices.split(',')]
            # Validate all indices
            for idx in sample_indices:
                if idx >= total_available_samples:
                    print(f"Error: Sample index {idx} exceeds available samples {total_available_samples}")
                    return
            args.num_samples = len(sample_indices)
            print(f"Plotting {len(sample_indices)} specific samples: {sample_indices}")
        except ValueError as e:
            print(f"Error parsing sample indices '{args.sample_indices}': {e}")
            print("Please provide comma-separated integers (e.g., '0,5,10')")
            return
    elif args.sample_idx is not None:
        if args.sample_idx >= total_available_samples:
            print(f"Error: Requested sample index {args.sample_idx} exceeds available samples {total_available_samples}")
            return
        sample_indices = [args.sample_idx]
        args.num_samples = 1
        print(f"Plotting single sample: {args.sample_idx}")
    else:
        if args.num_samples > total_available_samples:
            print(f"Warning: Requested {args.num_samples} samples, but only {total_available_samples} are available.")
            args.num_samples = total_available_samples

        if args.random_select:
            sample_indices = np.random.choice(total_available_samples, args.num_samples, replace=False)
            print(f"Randomly selected samples: {sample_indices}")
        else:
            sample_indices = np.arange(args.num_samples)
            print(f"Plotting first {args.num_samples} samples: {list(sample_indices)}")

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

        # Phase Distribution
        plot_distribution(np.angle(perfect_matrices_data), np.angle(received_matrices_data),
                        'Perfect Channel', 'Noisy Channel',
                        'Phase (Radians)', stats_dir,
                        f'phase_distributions_{dataset_name}.svg')

        if not args.save_figs and temp_dir:
            import webbrowser # Ensure webbrowser is imported here if not already
            try:
                phase_plot_path = os.path.join(stats_dir, f'phase_distributions_{dataset_name}.svg')
                webbrowser.open(phase_plot_path)
            except Exception as e:
                print(f"Warning: Could not open phase plot file: {e}")
                
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
                    noise_plot_path = os.path.join(stats_dir, 'noise_characteristics_distribution.svg')
                    webbrowser.open(noise_plot_path)
                except Exception as e:
                    print(f"Warning: Could not open noise distribution plot: {e}")
        except Exception as e:
            print(f"Warning: Could not generate noise distribution plot: {e}")

        # Doppler Spectrum
        if args.sampling_rate_symbols > 0:
            try:
                plot_doppler_spectrum(perfect_matrices_data, 
                                        args.sampling_rate_symbols, 
                                        stats_dir, 
                                        filename_suffix="_Perfect")
                if not args.save_figs and temp_dir:
                    try:
                        doppler_perfect_path = os.path.join(stats_dir, 'average_doppler_spectrum_Perfect.svg')
                        webbrowser.open(doppler_perfect_path)
                    except Exception as e:
                        print(f"Warning: Could not open Doppler (Perfect) plot: {e}")

                plot_doppler_spectrum(received_matrices_data, 
                                        args.sampling_rate_symbols, 
                                        stats_dir, 
                                        filename_suffix="_Noisy")
                if not args.save_figs and temp_dir:
                    try:
                        doppler_noisy_path = os.path.join(stats_dir, 'average_doppler_spectrum_Noisy.svg')
                        webbrowser.open(doppler_noisy_path)
                    except Exception as e:
                        print(f"Warning: Could not open Doppler (Noisy) plot: {e}")
            except Exception as e:
                print(f"Warning: Could not generate Doppler spectrum plots: {e}")
        else:
            print("Skipping Doppler spectrum plots as --sampling_rate_symbols is not provided or is zero.")
            
        print("Statistical plots generated.")

    # Process each sample
    for i, sample_idx in enumerate(sample_indices):
        print(f"\nProcessing sample {sample_idx}")
        
        # Default pilot mask to 1's in every 4th column if not provided
        if pilot_masks_data is None:
            n_sc, n_sym = received_matrices_data[sample_idx].shape
            current_pilot_mask = np.zeros((n_sc, n_sym), dtype=bool)
            current_pilot_mask[::12, ::7] = True
        else:
            current_pilot_mask = pilot_masks_data[sample_idx]

        sample_perfect = perfect_matrices_data[sample_idx] if perfect_matrices_data is not None else None
        sample_noisy   = received_matrices_data[sample_idx] if received_matrices_data is not None else None

        # Handle interpolation if requested
        if args.show_interpolation and sample_noisy is not None:
            try:
                from shared.utils.interpolation import interpolation
                # Prepare data for interpolation - handle complex numbers properly
                # Split complex data into real and imaginary parts
                noisy_real = np.real(sample_noisy)
                noisy_imag = np.imag(sample_noisy)
                
                # Stack real and imaginary parts along a new dimension
                # Shape should be (n_samples, n_sc, n_sym, 2) for real/imag
                noisy_reshaped = np.stack([noisy_real, noisy_imag], axis=-1)
                noisy_reshaped = noisy_reshaped[np.newaxis, ...]  # Add batch dimension
                
                # Ensure mask has correct shape (n_samples, n_sc, n_sym)
                mask_reshaped = current_pilot_mask[np.newaxis, ...]  # Add batch dimension
                
                print(f"Input shapes - Noisy: {noisy_reshaped.shape}, Mask: {mask_reshaped.shape}")
                
                # Perform interpolation
                interpolated = interpolation(noisy_reshaped, mask_reshaped, args.interpolation_method)
                
                # Reconstruct complex number from real and imaginary parts
                interpolated_complex = interpolated[0, ..., 0] + 1j * interpolated[0, ..., 1]
                
                # Plot interpolation comparison
                from .plot_heatmaps import plot_interpolation_comparison
                plot_interpolation_comparison(
                    interpolated=interpolated_complex,
                    perfect=sample_perfect,
                    noisy=sample_noisy,
                    pilot_mask=current_pilot_mask,
                    save_path=args.fig_dir if args.save_figs else None,
                    filename=f"sample_{sample_idx}_interpolation.svg",  # Explicitly specify .svg extension
                    show=not args.save_figs
                )
                
                if args.save_figs:
                    print(f"Saved interpolation plot to {os.path.join(args.fig_dir, f'sample_{sample_idx}_interpolation.svg')}")
                
                continue  # Skip other plotting if interpolation is shown
            except Exception as e:
                print(f"Error during interpolation: {e}")
                import traceback
                traceback.print_exc()  # Print full error traceback for debugging

        # Original plotting logic continues here...
        sample_info_str = f"Sample {sample_idx}"
        base_figname = f"sample_{sample_idx}"

        if not args.plot_perfect and not args.plot_noisy:
            # Plot both side by side
            plot_perfect_vs_noisy_heatmap(sample_perfect, sample_noisy, sample_info_str,
                                          pilot_mask=current_pilot_mask,
                                          savefig=args.save_figs, 
                                          figname=f"{base_figname}_perfect_vs_noisy.svg",
                                          fig_dir=args.fig_dir, 
                                          show_fig=not args.save_figs)
        else:
            # Plot individual matrices as requested
            if args.plot_perfect and sample_perfect is not None:
                fig, ax = plt.subplots(1, 1, figsize=(6, 4))
                plot_single_heatmap(sample_perfect, ax, f"Perfect Channel: {sample_info_str}",
                                    pilot_mask=current_pilot_mask)
                plt.tight_layout()
                if args.save_figs:
                    fpath = os.path.join(args.fig_dir, f"{base_figname}_perfect.svg")
                    plt.savefig(fpath)
                    print(f"Saved figure to {fpath}")
                if not args.save_figs:
                    plt.show()
                else:
                    plt.close(fig)

            if args.plot_noisy and sample_noisy is not None:
                fig, ax = plt.subplots(1, 1, figsize=(6, 4))
                plot_single_heatmap(sample_noisy, ax, f"Noisy Channel: {sample_info_str}",
                                    pilot_mask=current_pilot_mask)
                plt.tight_layout()
                if args.save_figs:
                    fpath = os.path.join(args.fig_dir, f"{base_figname}_noisy.svg")
                    plt.savefig(fpath)
                    print(f"Saved figure to {fpath}")
                if not args.save_figs:
                    plt.show()
                else:
                    plt.close(fig)

    if args.save_figs:
        if args.plot_stats:
            print(f"Finished processing. Heatmaps saved in {args.fig_dir}, statistics plots saved in {stats_dir}")
        else:
            print(f"Finished processing. Figures saved in {args.fig_dir}")

if __name__ == "__main__":
    main() 