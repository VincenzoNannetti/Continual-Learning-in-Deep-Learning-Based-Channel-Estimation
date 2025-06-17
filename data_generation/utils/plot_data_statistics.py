import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
from scipy.stats import entropy, wasserstein_distance, ks_2samp # Enhanced imports for additional metrics
from scipy.spatial.distance import jensenshannon
from tqdm import tqdm
import pandas as pd # For creating comparison tables

try:
    from .ieee_plot_style import setup_ieee_style
except ImportError:
    from ieee_plot_style import setup_ieee_style

def load_data(mat_file_path):
    """Loads data from the specified .mat file."""
    try:
        data = scipy.io.loadmat(mat_file_path)
        perfect_matrices = data['perfect_matrices']
        received_matrices = data['received_matrices']
        # pilot_mask = data['pilot_mask'] # Load if needed for specific plots
        print(f"Loaded data from {mat_file_path}")
        print(f"  Perfect matrices shape: {perfect_matrices.shape}")
        print(f"  Received matrices shape: {received_matrices.shape}")
        # print(f"  Pilot mask shape: {pilot_mask.shape}")
        return perfect_matrices, received_matrices #, pilot_mask
    except FileNotFoundError:
        print(f"Error: Data file not found at {mat_file_path}")
        exit()
    except KeyError as e:
        print(f"Error: Missing key {e} in the .mat file.")
        exit()

def plot_example_heatmap(perfect_matrix, noisy_matrix, sample_index, save_path):
    """Plots the magnitude heatmap for a single sample (perfect vs noisy) using pcolormesh."""
    fig, axs = plt.subplots(1, 2, figsize=(7.5, 3.0))
    fig.suptitle(f'Channel Magnitude Heatmap (Sample {sample_index})', fontsize=plt.rcParams['figure.titlesize'], y=0.95)

    cmap = 'viridis'
    n_sc, n_sym = perfect_matrix.shape

    # Calculate dynamic tick spacing for x-axis (aim for ~7 ticks)
    x_tick_spacing = max(1, int(round(n_sym / 7)))

    # Define grid boundaries for pcolormesh
    x_coords = np.arange(n_sym + 1)
    y_coords = np.arange(n_sc + 1)

    # --- Left Plot (Perfect) ---
    vmin_p = np.min(np.abs(perfect_matrix))
    vmax_p = np.max(np.abs(perfect_matrix))
    im1 = axs[0].pcolormesh(x_coords, y_coords, np.abs(perfect_matrix), cmap=cmap, shading='flat', vmin=vmin_p, vmax=vmax_p)
    axs[0].set_title('Perfect Channel')
    axs[0].set_xlabel('Time Block Index')
    axs[0].set_ylabel('Subcarrier Index')
    # Use calculated tick spacing
    axs[0].set_xticks(np.arange(0, n_sym + 1, x_tick_spacing)) # Include end tick if space allows
    axs[0].set_yticks(np.arange(0, n_sc + 1, max(1, n_sc // 10))) # Adjust y-ticks too if needed
    axs[0].set_xlim(0, n_sym)
    axs[0].set_ylim(0, n_sc)
    axs[0].set_aspect('auto') # Adjust aspect ratio if needed

    cbar1 = fig.colorbar(im1, ax=axs[0], fraction=0.046, pad=0.04)
    cbar1.set_label('Magnitude')
    cbar1.ax.tick_params(labelsize=plt.rcParams['xtick.labelsize'])

    # --- Right Plot (Noisy) ---
    vmin_n = np.min(np.abs(noisy_matrix))
    vmax_n = np.max(np.abs(noisy_matrix))
    im2 = axs[1].pcolormesh(x_coords, y_coords, np.abs(noisy_matrix), cmap=cmap, shading='flat', vmin=vmin_n, vmax=vmax_n)
    axs[1].set_title('Noisy Channel')
    axs[1].set_xlabel('Time Block Index')
    axs[1].set_ylabel('Subcarrier Index')
    # Use calculated tick spacing
    axs[1].set_xticks(np.arange(0, n_sym + 1, x_tick_spacing))
    axs[1].set_yticks(np.arange(0, n_sc + 1, max(1, n_sc // 10)))
    axs[1].set_xlim(0, n_sym)
    axs[1].set_ylim(0, n_sc)
    axs[1].set_aspect('auto')

    cbar2 = fig.colorbar(im2, ax=axs[1], fraction=0.046, pad=0.04)
    cbar2.set_label('Magnitude')
    cbar2.ax.tick_params(labelsize=plt.rcParams['xtick.labelsize'])

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(save_path, f'heatmap_sample_{sample_index}.svg')) # Save as SVG
    plt.close(fig)
    print(f"Saved heatmap for sample {sample_index}.")

def plot_distribution(data1, data2, title1, title2, xlabel, save_path, filename, bins=50, max_points=1000000):
    """Plots the distribution (histogram and KDE) side-by-side for two datasets, with subsampling for KDE and Histogram."""
    fig, axs = plt.subplots(1, 2, figsize=(7, 2.8))
    fig.suptitle(f'{xlabel} Distribution', fontsize=plt.rcParams['figure.titlesize'], y=0.95)

    # --- Data Preparation with Subsampling ---
    flat_data1 = data1.flatten()
    flat_data2 = data2.flatten()

    # Subsample if data is too large
    if flat_data1.size > max_points:
        print(f"Subsampling data1 from {flat_data1.size} to {max_points} for Hist/KDE plot.")
        sampled_indices1 = np.random.choice(flat_data1.size, max_points, replace=False)
        plot_data1 = flat_data1[sampled_indices1]
    else:
        plot_data1 = flat_data1 # Use original data if small enough

    if flat_data2.size > max_points:
        print(f"Subsampling data2 from {flat_data2.size} to {max_points} for Hist/KDE plot.")
        sampled_indices2 = np.random.choice(flat_data2.size, max_points, replace=False)
        plot_data2 = flat_data2[sampled_indices2]
    else:
        plot_data2 = flat_data2 # Use original data if small enough
    # ----------------------------------------

    # Plot 1 - Use SUBSAMPLED data for hist AND sampled for KDE
    sns.histplot(plot_data1, kde=False, bins=bins, stat='density', ax=axs[0], edgecolor=None, linewidth=0.5)
    sns.kdeplot(plot_data1, ax=axs[0], color=sns.color_palette()[0]) # Add KDE line separately
    axs[0].set_title(title1)
    axs[0].set_xlabel(xlabel)
    axs[0].set_ylabel('Probability Density')

    # Plot 2 - Use SUBSAMPLED data for hist AND sampled for KDE
    sns.histplot(plot_data2, kde=False, bins=bins, stat='density', ax=axs[1], edgecolor=None, linewidth=0.5)
    sns.kdeplot(plot_data2, ax=axs[1], color=sns.color_palette()[0]) # Add KDE line separately
    axs[1].set_title(title2)
    axs[1].set_xlabel(xlabel)
    axs[1].set_ylabel('') # Remove redundant Y label

    plt.tight_layout(rect=[0, 0.03, 1, 0.93]) # Adjust layout for suptitle
    plt.savefig(os.path.join(save_path, filename))
    plt.close(fig)
    print(f"Saved distribution plot: {filename}")

def plot_pdp(perfect_matrices, save_path):
    """Calculates and plots the average Power Delay Profile (PDP)."""
    # IFFT along the frequency axis (subcarriers) for each symbol and sample
    # Output shape: (n_samples, n_subcarriers, n_symbols) -> (n_samples, n_taps, n_symbols)
    channel_ir = np.fft.ifft(perfect_matrices, axis=1)

    # Calculate power: |h(t, tau)|^2
    power_ir = np.abs(channel_ir)**2

    # Average power over symbols and samples
    # Average over axis 2 (symbols), then axis 0 (samples)
    avg_pdp = np.mean(power_ir, axis=(0, 2))

    # --- Modification: Plot only the first half (causal part) ---
    n_taps = avg_pdp.shape[0]
    taps_to_plot = n_taps // 2
    avg_pdp_causal = avg_pdp[:taps_to_plot]
    delay_indices = np.arange(taps_to_plot)
    # ------------------------------------------------------------

    fig = plt.figure() # Use default figsize from rcParams
    ax = fig.add_subplot(111)
    # Plot in dB, add epsilon for log(0)
    pdp_db = 10 * np.log10(avg_pdp_causal + 1e-12) # Use smaller epsilon

    # Limit y-axis range for better visualisation, relative to the peak in the causal part
    # Avoid plotting extremely low noise floor values if they obscure the shape
    peak_db = np.max(pdp_db)
    floor_db = max(peak_db - 40, np.min(pdp_db)) # Show at least 40dB range or actual min

    ax.plot(delay_indices, pdp_db, marker='o', linestyle='-', markersize=3)
    ax.set_title('Average Power Delay Profile (PDP)')
    ax.set_xlabel('Delay Tap Index (\u03C4)') # Use unicode tau
    ax.set_ylabel('Average Power (dB)')
    ax.set_ylim(bottom=floor_db, top=peak_db + 5) # Set y-limits relative to peak
    ax.set_xlim(left=-0.5, right=taps_to_plot - 0.5) # Adjust x-limits

    # ax.grid(True, linestyle=':', alpha=0.7) # Grid is enabled by default now
    plt.savefig(os.path.join(save_path, 'average_pdp.svg')) # Save as SVG
    plt.close(fig)
    print("Saved Average PDP plot (showing causal part).")

def calculate_correlation(matrices, axis):
    """Calculates average correlation along a given axis."""
    n_samples, n_sc, n_sym = matrices.shape
    correlations = []

    if axis == 1: # Frequency correlation (between adjacent subcarriers)
        for i in range(n_samples):
            for j in range(n_sym):
                sc_series = matrices[i, :, j]
                # Correlation between sc_k and sc_{k+1}
                corr = np.corrcoef(sc_series[:-1], sc_series[1:])[0, 1]
                if not np.isnan(corr):
                    correlations.append(np.abs(corr)) # Use magnitude of correlation
    elif axis == 2: # Temporal correlation (between adjacent symbols)
        for i in range(n_samples):
            for j in range(n_sc):
                sym_series = matrices[i, j, :]
                # Correlation between sym_t and sym_{t+1}
                corr = np.corrcoef(sym_series[:-1], sym_series[1:])[0, 1]
                if not np.isnan(corr):
                    correlations.append(np.abs(corr)) # Use magnitude of correlation
    else:
        raise ValueError("Axis must be 1 (frequency) or 2 (temporal)")

    return np.mean(correlations) if correlations else 0

def plot_correlation(perfect_matrices, noisy_matrices, save_path):
    """Plots the average frequency and temporal correlations."""
    # --- Vectorized Calculation --- #
    print("Calculating vectorized correlations...")
    freq_corr_perfect = vectorized_correlation(perfect_matrices, axis=1)
    temp_corr_perfect = vectorized_correlation(perfect_matrices, axis=2)
    freq_corr_noisy = vectorized_correlation(noisy_matrices, axis=1)
    temp_corr_noisy = vectorized_correlation(noisy_matrices, axis=2)
    print("Correlation calculation finished.")

    labels = ['Freq. (Perfect)', 'Temporal (Perfect)', 'Freq. (Noisy)', 'Temporal (Noisy)']
    values = [freq_corr_perfect, temp_corr_perfect, freq_corr_noisy, temp_corr_noisy]

    fig = plt.figure(figsize=(4.5, 3.0))
    ax = fig.add_subplot(111)

    colors = ['#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78'] # Example: Blue/Orange pairs
    bars = ax.bar(labels, values, color=colors, width=0.6)

    ax.set_ylabel('Avg. Abs. Correlation Coeff.')
    ax.set_title('Average Channel Correlation', y=1.1)
    ax.set_ylim(0, 1)
    ax.tick_params(axis='x', rotation=45)

    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f'{yval:.3f}',
                va='bottom', ha='center', fontsize=plt.rcParams['xtick.labelsize'] - 1)

    # Use subplots_adjust for better control over spacing
    plt.subplots_adjust(top=0.85, bottom=0.20) # Adjust top and bottom spacing
    plt.savefig(os.path.join(save_path, 'average_correlation.svg'))
    plt.close(fig)
    print("Saved Average Correlation plot.")

def plot_noise_distribution(perfect_matrices, received_matrices, save_path, max_points=100000):
    """Plots the distribution of real, imaginary, magnitude, and phase of the noise.
       Subsampling is used for KDE and Histogram if data is too large."""
    noise = received_matrices - perfect_matrices
    noise_real = noise.real.flatten()
    noise_imag = noise.imag.flatten()
    noise_magnitude = np.abs(noise).flatten()
    noise_phase = np.angle(noise).flatten() # Phase will be in radians (-pi to pi)

    data_to_plot = {
        "Real Part of Noise": noise_real,
        "Imaginary Part of Noise": noise_imag,
        "Magnitude of Noise": noise_magnitude,
        "Phase of Noise (Radians)": noise_phase
    }

    # Determine figure layout (2x2 grid)
    fig, axs = plt.subplots(2, 2, figsize=(8, 6)) # Adjusted figsize for 2x2
    axs_flat = axs.flatten() # Flatten for easy iteration

    plot_idx = 0
    for title, data_flat in data_to_plot.items():
        current_ax = axs_flat[plot_idx]
        
        # --- Subsampling --- #
        if data_flat.size > max_points:
            print(f"Subsampling {title.lower()} from {data_flat.size} to {max_points} for Hist/KDE plot.")
            plot_data = data_flat[np.random.choice(data_flat.size, max_points, replace=False)]
        else:
            plot_data = data_flat

        bins = 50
        # Special handling for phase data bins if desired (e.g., ensure full -pi to pi range)
        # if "Phase" in title:
        #     bins = np.linspace(-np.pi, np.pi, 51) # Example: 50 bins covering -pi to pi

        sns.histplot(plot_data, kde=False, bins=bins, stat='density', ax=current_ax, edgecolor=None, linewidth=0.5)
        sns.kdeplot(plot_data, ax=current_ax, color=sns.color_palette()[0]) # Add KDE line separately
        
        current_ax.set_title(title)
        current_ax.set_xlabel("Value") # Generic xlabel, could be more specific if needed
        if plot_idx % 2 == 0: # Only set Y label for left column plots
            current_ax.set_ylabel('Probability Density')
        else:
            current_ax.set_ylabel('')


        plot_idx += 1
    
    fig.suptitle('Noise Characteristics Distribution', fontsize=plt.rcParams['figure.titlesize'], y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout for suptitle
    filename = "noise_characteristics_distribution.svg"
    plt.savefig(os.path.join(save_path, filename))
    plt.close(fig)
    print(f"Saved noise characteristics plot: {filename}")

def plot_doppler_spectrum(channel_matrices, sampling_rate_symbols, save_path, filename_suffix="", max_points_fft=None):
    """Calculates and plots the average Doppler spectrum from channel matrices.

    Args:
        channel_matrices (np.ndarray): Channel matrices (n_samples, n_subcarriers, n_time_blocks).
                                       Each column along the last axis is a time block (e.g., 14 OFDM symbols).
        sampling_rate_symbols (float): Sampling rate of the time blocks in Hz (i.e., 1 / block_duration).
        save_path (str): Directory to save the plot.
        filename_suffix (str): Suffix for the plot filename.
        max_points_fft (int, optional): Max number of time blocks to use for FFT per sample to manage computation.
    """
    if channel_matrices.ndim != 3:
        print("Warning: Doppler spectrum expects 3D channel matrices (n_samples, n_subcarriers, n_time_blocks). Skipping.")
        return
    
    n_samples, n_sc, n_sym = channel_matrices.shape
    if n_sym < 2:
        print("Warning: Not enough symbols to calculate Doppler spectrum. Skipping.")
        return

    all_doppler_spectra = []

    print(f"Calculating Doppler spectrum for {n_samples} samples...")
    for i in tqdm(range(n_samples), desc="Doppler Spectrum Progress"):
        sample_matrix = channel_matrices[i, :, :] # Shape (n_sc, n_sym)
        
        symbols_to_use = n_sym
        if max_points_fft is not None and n_sym > max_points_fft:
            symbols_to_use = max_points_fft
            sample_matrix = sample_matrix[:, :symbols_to_use] # Truncate for this sample

        # IFFT along the time (symbol) axis to get Doppler shifts
        # H(f, t) -> H(f, f_D)
        # We apply FFT, not IFFT, as we want to go from time (blocks) to frequency (Doppler)
        # The input is effectively a time series for each subcarrier, sampled at the block rate.
        channel_doppler_domain = np.fft.fft(sample_matrix, axis=1) # FFT along time block axis
        
        # Power spectrum: |H(f, f_D)|^2
        power_spectrum_sample = np.abs(channel_doppler_domain)**2
        
        # Average over subcarriers
        avg_power_spectrum_sample = np.mean(power_spectrum_sample, axis=0) # Average along subcarrier axis
        all_doppler_spectra.append(avg_power_spectrum_sample)
    
    if not all_doppler_spectra:
        print("No Doppler spectra were calculated.")
        return

    # Average Doppler spectrum over all samples
    avg_doppler_spectrum = np.mean(np.array(all_doppler_spectra), axis=0)
    
    # Create Doppler frequency axis for plotting
    # FFT frequencies range from -fs_block/2 to fs_block/2
    # symbols_to_use here refers to the number of points in FFT (could be truncated from original n_sym/n_time_blocks)
    fft_len = avg_doppler_spectrum.shape[0] # Should match symbols_to_use if truncation happened, else n_time_blocks
    doppler_freq_axis = np.fft.fftshift(np.fft.fftfreq(fft_len, d=1.0/sampling_rate_symbols)) # sampling_rate_symbols is the block rate
    avg_doppler_spectrum_shifted = np.fft.fftshift(avg_doppler_spectrum)

    # Plot in dB
    # Add epsilon for log(0) and normalise by max for relative dB plot
    epsilon = 1e-12
    doppler_db = 10 * np.log10(avg_doppler_spectrum_shifted + epsilon)
    doppler_db_normalized = doppler_db - np.max(doppler_db) # Normalize to 0 dB peak

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(doppler_freq_axis, doppler_db_normalized)
    ax.set_title(f'Average Doppler Power Spectrum')
    ax.set_xlabel('Doppler Frequency (Hz)')
    ax.set_ylabel('Normalised Power (dB)')
    ax.grid(True, linestyle=':', alpha=0.7)
    ax.set_ylim(bottom=max(np.min(doppler_db_normalized), -60)) # Show at least 60dB dynamic range or actual min
    
    filename = f'average_doppler_spectrum{filename_suffix.replace(" ", "_")}.svg'
    plt.savefig(os.path.join(save_path, filename))
    plt.close(fig)
    print(f"Saved Average Doppler Spectrum plot: {filename}")

def plot_overlayed_distribution(data1, data2, label1, label2, xlabel, title, save_path, filename, bins=50, max_points=100000):
    """Plots the overlayed distribution (KDE) for two datasets, with subsampling."""
    fig, ax = plt.subplots(1, 1, figsize=(4.5, 3.0))
    fig.suptitle(title, fontsize=plt.rcParams['figure.titlesize'], y=0.85)

    # --- Subsampling --- #
    flat_data1 = data1.flatten()
    flat_data2 = data2.flatten()

    if flat_data1.size > max_points:
        print(f"Subsampling data1 from {flat_data1.size} to {max_points} for overlayed KDE plot.")
        kde_data1 = flat_data1[np.random.choice(flat_data1.size, max_points, replace=False)]
    else:
        kde_data1 = flat_data1

    if flat_data2.size > max_points:
        print(f"Subsampling data2 from {flat_data2.size} to {max_points} for overlayed KDE plot.")
        kde_data2 = flat_data2[np.random.choice(flat_data2.size, max_points, replace=False)]
    else:
        kde_data2 = flat_data2
    # -------------------

    # Plot KDE for both datasets using sampled data
    sns.kdeplot(kde_data1, ax=ax, label=label1, fill=True, alpha=0.5, linewidth=1.5)
    sns.kdeplot(kde_data2, ax=ax, label=label2, fill=True, alpha=0.5, linewidth=1.5)

    ax.set_xlabel(xlabel)
    ax.set_ylabel('Probability Density')
    ax.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.93]) # Adjust layout for suptitle
    plt.savefig(os.path.join(save_path, filename))
    plt.close(fig)
    print(f"Saved overlayed distribution plot: {filename}")

def plot_complex_distribution(complex_data1, complex_data2, title1, title2, save_path, filename):
    """Plots the 2D distribution (real vs imag) side-by-side."""
    real1, imag1 = complex_data1.real.flatten(), complex_data1.imag.flatten()
    real2, imag2 = complex_data2.real.flatten(), complex_data2.imag.flatten()

    # Determine reasonable limits - use combined data for shared limits or calculate individually
    lim_std = 3
    # Example: Individual limits
    xlim1 = (-lim_std * real1.std(), lim_std * real1.std())
    ylim1 = (-lim_std * imag1.std(), lim_std * imag1.std())
    xlim2 = (-lim_std * real2.std(), lim_std * real2.std())
    ylim2 = (-lim_std * imag2.std(), lim_std * imag2.std())

    fig, axs = plt.subplots(1, 2, figsize=(8, 3.5))
    fig.suptitle('Complex Coefficient Distribution', fontsize=plt.rcParams['figure.titlesize'], y=0.95)
    cmap='viridis'

    # Plot 1
    hb1 = axs[0].hexbin(real1, imag1, gridsize=50, cmap=cmap, mincnt=1, extent=[*xlim1, *ylim1])
    axs[0].set_title(title1)
    axs[0].set_xlabel('Real Part')
    axs[0].set_ylabel('Imaginary Part')
    axs[0].set_aspect('equal', adjustable='box')
    cbar1 = fig.colorbar(hb1, ax=axs[0], label='Count / Density', fraction=0.046, pad=0.04)
    cbar1.ax.tick_params(labelsize=plt.rcParams['xtick.labelsize'])

    # Plot 2
    hb2 = axs[1].hexbin(real2, imag2, gridsize=50, cmap=cmap, mincnt=1, extent=[*xlim2, *ylim2])
    axs[1].set_title(title2)
    axs[1].set_xlabel('Real Part')
    axs[1].set_ylabel('') # Remove redundant label
    axs[1].set_aspect('equal', adjustable='box')
    cbar2 = fig.colorbar(hb2, ax=axs[1], label='Count / Density', fraction=0.046, pad=0.04)
    cbar2.ax.tick_params(labelsize=plt.rcParams['xtick.labelsize'])
    # axs[1].sharex(axs[0]) # Optional: share axes if limits are the same
    # axs[1].sharey(axs[0])

    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    plt.savefig(os.path.join(save_path, filename))
    plt.close(fig)
    print(f"Saved complex distribution plot: {filename}")

# --- Helper function for vectorized correlation --- #
def rowwise_corrcoef_abs(A, B):
    """Calculates the average absolute correlation coefficient between rows of A and B.

    Args:
        A (np.ndarray): First input array, shape (M, K).
        B (np.ndarray): Second input array, shape (M, K).

    Returns:
        float: Average absolute correlation coefficient across rows (ignoring NaNs).
    """
    # A, B shape (M, K) - M features, K observations/batches
    # Center rows (observations)
    A_centered = A - A.mean(axis=1, keepdims=True)
    B_centered = B - B.mean(axis=1, keepdims=True)

    # Covariance of rows
    # np.sum(A_centered * B_centered.conj(), axis=1) / K
    # Using np.mean automatically handles the division by K
    cov_AB = np.mean(A_centered * B_centered.conj(), axis=1) # shape (M,)

    # Variance of rows
    var_A = np.mean(np.abs(A_centered)**2, axis=1) # shape (M,)
    var_B = np.mean(np.abs(B_centered)**2, axis=1) # shape (M,)

    # Standard deviations
    std_A = np.sqrt(var_A)
    std_B = np.sqrt(var_B)

    # Correlation coefficient
    # Add epsilon to denominator to avoid division by zero/NaN
    epsilon = 1e-12
    corr = cov_AB / (std_A * std_B + epsilon)

    # Return average absolute correlation, ignoring NaNs from zero std dev cases
    # NaNs occur if std_A or std_B is zero for a row
    return np.nanmean(np.abs(corr))


def vectorized_correlation(matrices, axis):
    """Calculates average correlation along a given axis using vectorization."""
    n_samples, n_sc, n_sym = matrices.shape
    epsilon = 1e-12 # For stability

    if axis == 1: # Frequency correlation (between adjacent subcarriers)
        if n_sc < 2: return 0.0 # Need at least 2 subcarriers
        # Permute to (S, N, T)
        matrices_permuted = matrices.transpose(1, 0, 2)
        X = matrices_permuted[:-1, :, :] # Shape (S-1, N, T)
        Y = matrices_permuted[1:, :, :]  # Shape (S-1, N, T)
        S_minus_1, N, T = X.shape
        # Flatten batch dimensions (N, T)
        X_flat = X.reshape(S_minus_1, N * T)
        Y_flat = Y.reshape(S_minus_1, N * T)
        return rowwise_corrcoef_abs(X_flat, Y_flat)

    elif axis == 2: # Temporal correlation (between adjacent symbols)
        if n_sym < 2: return 0.0 # Need at least 2 symbols
        # Permute to (T, N, S)
        matrices_permuted = matrices.transpose(2, 0, 1)
        X = matrices_permuted[:-1, :, :] # Shape (T-1, N, S)
        Y = matrices_permuted[1:, :, :]  # Shape (T-1, N, S)
        T_minus_1, N, S = X.shape
        # Flatten batch dimensions (N, S)
        X_flat = X.reshape(T_minus_1, N * S)
        Y_flat = Y.reshape(T_minus_1, N * S)
        return rowwise_corrcoef_abs(X_flat, Y_flat)

    else:
        raise ValueError("Axis must be 1 (frequency) or 2 (temporal)")

# --- Original non-vectorized function (kept for reference/testing if needed) --- #
def calculate_correlation_original(matrices, axis):
    """Calculates average correlation along a given axis (Original Loop Version)."""
    n_samples, n_sc, n_sym = matrices.shape
    correlations = []

    if axis == 1: # Frequency correlation (between adjacent subcarriers)
        if n_sc < 2: return 0.0
        for i in range(n_samples):
            for j in range(n_sym):
                sc_series = matrices[i, :, j]
                # Correlation between sc_k and sc_{k+1}
                # np.corrcoef returns NaN if input is constant
                with np.errstate(invalid='ignore'): # Suppress NaN warnings
                    corr = np.corrcoef(sc_series[:-1], sc_series[1:])[0, 1]
                if not np.isnan(corr):
                    correlations.append(np.abs(corr)) # Use magnitude of correlation
    elif axis == 2: # Temporal correlation (between adjacent symbols)
        if n_sym < 2: return 0.0
        for i in range(n_samples):
            for j in range(n_sc):
                sym_series = matrices[i, j, :]
                # Correlation between sym_t and sym_{t+1}
                with np.errstate(invalid='ignore'): # Suppress NaN warnings
                    corr = np.corrcoef(sym_series[:-1], sym_series[1:])[0, 1]
                if not np.isnan(corr):
                    correlations.append(np.abs(corr)) # Use magnitude of correlation
    else:
        raise ValueError("Axis must be 1 (frequency) or 2 (temporal)")

    return np.mean(correlations) if correlations else 0

# --- Enhanced Statistical Comparison Functions ---
def calculate_kl_divergence(data_a, data_b, num_bins=100, epsilon=1e-10):
    """Calculates KL divergence between two data distributions using histograms.

    Args:
        data_a (np.ndarray): Flattened data array for distribution A.
        data_b (np.ndarray): Flattened data array for distribution B.
        num_bins (int): Number of bins for histogram estimation.
        epsilon (float): Small value to add to probabilities to avoid log(0).

    Returns:
        tuple: (kl_div_ab, kl_div_ba) - KL divergence D(A||B) and D(B||A).
               Returns (np.nan, np.nan) if calculation fails.
    """
    # Determine common range for bins
    combined_data = np.concatenate((data_a, data_b))
    min_val = np.min(combined_data)
    max_val = np.max(combined_data)

    # Handle edge case where min and max are the same
    if np.isclose(min_val, max_val):
        print("Warning: Data appears constant, KL divergence is likely 0 or undefined.")
        return np.nan, np.nan

    bins = np.linspace(min_val, max_val, num_bins + 1)

    # Create histograms and normalise to get probability distributions (PMFs)
    hist_a, _ = np.histogram(data_a, bins=bins, density=False)
    hist_b, _ = np.histogram(data_b, bins=bins, density=False)

    # normalise to sum to 1
    pmf_a = hist_a / np.sum(hist_a) if np.sum(hist_a) > 0 else np.zeros_like(hist_a)
    pmf_b = hist_b / np.sum(hist_b) if np.sum(hist_b) > 0 else np.zeros_like(hist_b)

    # Add epsilon to avoid division by zero or log(0) in entropy calculation
    pmf_a_smooth = pmf_a + epsilon
    pmf_b_smooth = pmf_b + epsilon
    pmf_a_smooth /= np.sum(pmf_a_smooth) # Re-normalise after adding epsilon
    pmf_b_smooth /= np.sum(pmf_b_smooth) # Re-normalise after adding epsilon

    # Calculate KL divergence using scipy.stats.entropy
    try:
        kl_div_ab = entropy(pk=pmf_a_smooth, qk=pmf_b_smooth)
        kl_div_ba = entropy(pk=pmf_b_smooth, qk=pmf_a_smooth)
        return kl_div_ab, kl_div_ba
    except Exception as e:
        print(f"Error calculating KL divergence: {e}")
        return np.nan, np.nan

def calculate_comprehensive_metrics(data_a, data_b, metric_name="", max_samples=50000):
    """Calculate comprehensive statistical metrics between two datasets.
    
    Args:
        data_a, data_b: Input data arrays
        metric_name: Name for logging purposes
        max_samples: Maximum samples for computational efficiency
        
    Returns:
        dict: Dictionary containing all calculated metrics
    """
    # Flatten and subsample if necessary
    flat_a = data_a.flatten()
    flat_b = data_b.flatten()
    
    if len(flat_a) > max_samples:
        flat_a = np.random.choice(flat_a, max_samples, replace=False)
    if len(flat_b) > max_samples:
        flat_b = np.random.choice(flat_b, max_samples, replace=False)
    
    metrics = {}
    
    try:
        # Basic statistics
        metrics['mean_a'] = np.mean(flat_a)
        metrics['mean_b'] = np.mean(flat_b)
        metrics['std_a'] = np.std(flat_a)
        metrics['std_b'] = np.std(flat_b)
        metrics['mean_diff'] = abs(metrics['mean_a'] - metrics['mean_b'])
        metrics['std_ratio'] = metrics['std_a'] / metrics['std_b'] if metrics['std_b'] != 0 else np.inf
        
        # KL Divergence
        kl_ab, kl_ba = calculate_kl_divergence(flat_a, flat_b)
        metrics['kl_divergence_ab'] = kl_ab
        metrics['kl_divergence_ba'] = kl_ba
        metrics['kl_divergence_symmetric'] = (kl_ab + kl_ba) / 2
        
        # Jensen-Shannon Divergence (symmetric)
        # First create probability distributions
        combined_data = np.concatenate((flat_a, flat_b))
        min_val, max_val = np.min(combined_data), np.max(combined_data)
        if not np.isclose(min_val, max_val):
            bins = np.linspace(min_val, max_val, 101)
            hist_a, _ = np.histogram(flat_a, bins=bins, density=True)
            hist_b, _ = np.histogram(flat_b, bins=bins, density=True)
            # Normalise
            hist_a = hist_a / np.sum(hist_a) if np.sum(hist_a) > 0 else hist_a
            hist_b = hist_b / np.sum(hist_b) if np.sum(hist_b) > 0 else hist_b
            # Add small epsilon to avoid log(0)
            hist_a = hist_a + 1e-10
            hist_b = hist_b + 1e-10
            hist_a = hist_a / np.sum(hist_a)
            hist_b = hist_b / np.sum(hist_b)
            metrics['js_divergence'] = jensenshannon(hist_a, hist_b) ** 2  # Squared for distance metric
        else:
            metrics['js_divergence'] = np.nan
            
        # Wasserstein Distance (Earth Mover's Distance)
        metrics['wasserstein_distance'] = wasserstein_distance(flat_a, flat_b)
        
        # Kolmogorov-Smirnov Test
        ks_stat, ks_pvalue = ks_2samp(flat_a, flat_b)
        metrics['ks_statistic'] = ks_stat
        metrics['ks_pvalue'] = ks_pvalue
        
        # Additional distributional metrics
        metrics['range_a'] = np.max(flat_a) - np.min(flat_a)
        metrics['range_b'] = np.max(flat_b) - np.min(flat_b)
        metrics['range_ratio'] = metrics['range_a'] / metrics['range_b'] if metrics['range_b'] != 0 else np.inf
        
        # Percentile differences
        for p in [25, 50, 75, 90, 95]:
            perc_a = np.percentile(flat_a, p)
            perc_b = np.percentile(flat_b, p)
            metrics[f'percentile_{p}_diff'] = abs(perc_a - perc_b)
            
    except Exception as e:
        print(f"Error calculating metrics for {metric_name}: {e}")
        
    return metrics

def plot_comprehensive_comparison(data_a, data_b, label_a, label_b, title, save_path, filename_base, max_points=50000):
    """Create comprehensive comparison plots including overlayed distributions and statistical summaries."""
    
    # Flatten and subsample data
    flat_a = data_a.flatten()
    flat_b = data_b.flatten()
    
    if len(flat_a) > max_points:
        flat_a = np.random.choice(flat_a, max_points, replace=False)
    if len(flat_b) > max_points:
        flat_b = np.random.choice(flat_b, max_points, replace=False)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # 1. Overlayed KDE
    ax1 = fig.add_subplot(gs[0, 0])
    sns.kdeplot(flat_a, ax=ax1, label=label_a, fill=True, alpha=0.6)
    sns.kdeplot(flat_b, ax=ax1, label=label_b, fill=True, alpha=0.6)
    ax1.set_title('Probability Density')
    ax1.set_xlabel('Value')
    ax1.legend()
    
    # 2. Box plots
    ax2 = fig.add_subplot(gs[0, 1])
    box_data = [flat_a, flat_b]
    box_labels = [label_a, label_b]
    bp = ax2.boxplot(box_data, labels=box_labels, patch_artist=True)
    colors = ['lightblue', 'lightcoral']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    ax2.set_title('Box Plot Comparison')
    ax2.tick_params(axis='x', rotation=45)
    
    # 3. Q-Q Plot
    ax3 = fig.add_subplot(gs[0, 2])
    # Sort both datasets for Q-Q plot
    sorted_a = np.sort(flat_a)
    sorted_b = np.sort(flat_b)
    # Interpolate to same length
    min_len = min(len(sorted_a), len(sorted_b))
    if len(sorted_a) != len(sorted_b):
        if len(sorted_a) > min_len:
            indices = np.linspace(0, len(sorted_a)-1, min_len).astype(int)
            sorted_a = sorted_a[indices]
        if len(sorted_b) > min_len:
            indices = np.linspace(0, len(sorted_b)-1, min_len).astype(int)
            sorted_b = sorted_b[indices]
    
    ax3.scatter(sorted_a, sorted_b, alpha=0.5, s=1)
    min_val = min(np.min(sorted_a), np.min(sorted_b))
    max_val = max(np.max(sorted_a), np.max(sorted_b))
    ax3.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
    ax3.set_xlabel(f'{label_a} Quantiles')
    ax3.set_ylabel(f'{label_b} Quantiles')
    ax3.set_title('Q-Q Plot')
    ax3.set_aspect('equal', adjustable='box')
    
    # 4. Histogram comparison
    ax4 = fig.add_subplot(gs[1, :2])
    bins = np.linspace(min(np.min(flat_a), np.min(flat_b)), 
                      max(np.max(flat_a), np.max(flat_b)), 50)
    ax4.hist(flat_a, bins=bins, alpha=0.6, label=label_a, density=True, color='blue')
    ax4.hist(flat_b, bins=bins, alpha=0.6, label=label_b, density=True, color='red')
    ax4.set_title('Histogram Comparison')
    ax4.set_xlabel('Value')
    ax4.set_ylabel('Density')
    ax4.legend()
    
    # 5. Statistical summary
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis('off')
    
    # Calculate key metrics
    metrics = calculate_comprehensive_metrics(flat_a, flat_b)
    
    summary_text = f"""Statistical Summary:
    
Mean: {metrics.get('mean_a', 0):.3f} vs {metrics.get('mean_b', 0):.3f}
Std: {metrics.get('std_a', 0):.3f} vs {metrics.get('std_b', 0):.3f}

Distance Metrics:
KL Div (A||B): {metrics.get('kl_divergence_ab', np.nan):.3f}
KL Div (B||A): {metrics.get('kl_divergence_ba', np.nan):.3f}
JS Divergence: {metrics.get('js_divergence', np.nan):.3f}
Wasserstein: {metrics.get('wasserstein_distance', np.nan):.3f}

Statistical Tests:
KS Statistic: {metrics.get('ks_statistic', np.nan):.3f}
KS p-value: {metrics.get('ks_pvalue', np.nan):.3e}
"""
    
    ax5.text(0.05, 0.95, summary_text, transform=ax5.transAxes, fontsize=9,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    fig.suptitle(title, fontsize=14, y=0.95)
    
    # Save the plot
    plt.savefig(os.path.join(save_path, f'{filename_base}_comprehensive.svg'), 
                bbox_inches='tight', dpi=300)
    plt.close(fig)
    
    return metrics

def create_comparison_report(all_metrics, save_path):
    """Create a comprehensive comparison report with all calculated metrics."""
    
    # Convert metrics to DataFrame for easy handling
    df_data = []
    for metric_name, metrics in all_metrics.items():
        row = {'Metric': metric_name}
        row.update(metrics)
        df_data.append(row)
    
    df = pd.DataFrame(df_data)
    
    # Save as CSV
    csv_path = os.path.join(save_path, 'dataset_comparison_metrics.csv')
    df.to_csv(csv_path, index=False)
    print(f"Saved comprehensive metrics to: {csv_path}")
    
    # Create summary plot of key metrics
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Dataset Comparison Summary', fontsize=16)
    
    # Extract key metrics for plotting
    metrics_to_plot = ['kl_divergence_symmetric', 'js_divergence', 'wasserstein_distance', 'ks_statistic']
    metric_labels = ['KL Divergence\n(Symmetric)', 'Jensen-Shannon\nDivergence', 'Wasserstein\nDistance', 'Kolmogorov-Smirnov\nStatistic']
    
    for i, (metric, label) in enumerate(zip(metrics_to_plot, metric_labels)):
        ax = axes[i//2, i%2]
        
        # Get values for this metric
        values = []
        names = []
        for _, row in df.iterrows():
            if metric in row and not pd.isna(row[metric]):
                values.append(row[metric])
                names.append(row['Metric'])
        
        if values:
            bars = ax.bar(range(len(values)), values, color=plt.cm.Set3(np.linspace(0, 1, len(values))))
            ax.set_title(label)
            ax.set_xticks(range(len(values)))
            ax.set_xticklabels(names, rotation=45, ha='right')
            ax.set_ylabel('Value')
            
            # Add value labels on bars
            for j, (bar, val) in enumerate(zip(bars, values)):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.01,
                       f'{val:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'comparison_summary.svg'), bbox_inches='tight', dpi=300)
    plt.close(fig)
    
    # Print summary to console
    print("\n" + "="*60)
    print("DATASET COMPARISON SUMMARY")
    print("="*60)
    
    for _, row in df.iterrows():
        print(f"\n{row['Metric']}:")
        print("-" * 40)
        
        # Key metrics to highlight
        key_metrics = {
            'Mean Difference': 'mean_diff',
            'KL Divergence (A→B)': 'kl_divergence_ab', 
            'KL Divergence (B→A)': 'kl_divergence_ba',
            'Jensen-Shannon Div': 'js_divergence',
            'Wasserstein Distance': 'wasserstein_distance',
            'KS Test p-value': 'ks_pvalue'
        }
        
        for display_name, col_name in key_metrics.items():
            if col_name in row and not pd.isna(row[col_name]):
                value = row[col_name]
                if col_name == 'ks_pvalue':
                    print(f"  {display_name}: {value:.2e} {'(Significant)' if value < 0.05 else '(Not significant)'}")
                else:
                    print(f"  {display_name}: {value:.4f}")
    
    print("\n" + "="*60)
    print("INTERPRETATION GUIDE:")
    print("• Higher KL/JS divergence = More different distributions")
    print("• Higher Wasserstein distance = More different distributions") 
    print("• KS p-value < 0.05 = Statistically significant difference")
    print("• Values closer to 0 = More similar datasets")
    print("="*60)
    
    return df


if __name__ == "__main__":
    # --- Argument Parsing --- #
    parser = argparse.ArgumentParser(description="Generate visualisations for channel data.")
    parser.add_argument("--plot-a", action="store_true", help="Generate individual plots for Dataset A.")
    parser.add_argument("--plot-b", action="store_true", help="Generate individual plots for Dataset B.")
    parser.add_argument("--compare", action="store_true", help="Generate comparison plots for Dataset A vs B.")
    parser.add_argument("--all", action="store_true", help="Generate all plots (Dataset A, Dataset B, and Comparison). Equivalent to --plot-a --plot-b --compare.")

    args = parser.parse_args()

    # If --all is specified, set the individual flags
    if args.all:
        args.plot_a = True
        args.plot_b = True
        args.compare = True

    # Check if at least one action is requested
    if not args.plot_a and not args.plot_b and not args.compare:
        print("No plotting action specified. Use --plot-a, --plot-b, --compare, or --all.")
        exit()

    # --- Setup Plot Style --- #
    setup_ieee_style()

    # --- Configuration, Directory Creation, Data Loading ---
    # Base directory for the data - Use standard string with escaped backslashes
    DATA_DIR = "C:\\Users\\Vincenzo_DES\\OneDrive - Imperial College London\\Year 4\\ELEC70017 - Individual Project\\Main\\Learning_Algorithms\\data\\raw\\My Data\\"

    # --- Dataset B Configuration ---
    # Define parameters used to generate Dataset B to construct filename
    DATASET_B_SNR = "SNR0" # Assuming this matches the lowest value in the range [5, 10]
    DATASET_B_SCATTER = "with_agg_scatter"
    DATASET_B_BLOCKS = "4blocks"
    DATASET_B_SUFFIX = "datasetB"
    DATA_FILE_PATH_B = os.path.join(DATA_DIR, f"all_test_data_{DATASET_B_SNR}_{DATASET_B_SCATTER}_{DATASET_B_BLOCKS}_{DATASET_B_SUFFIX}.mat")
    PLOT_OUTPUT_DIR_B = os.path.join("Plots", "Dataset B")

    # --- Dataset A Configuration (Modify these based on Dataset A generation params) ---
    # Define parameters used to generate Dataset A to construct filename
    DATASET_A_SNR = "SNR20" # From commented out params
    DATASET_A_SCATTER = "no_agg_scatter" # From commented out params
    DATASET_A_BLOCKS = "4blocks" # Assuming same block count
    DATASET_A_SUFFIX = "datasetA" # Assuming a suffix like this exists
    DATA_FILE_PATH_A = os.path.join(DATA_DIR, f"all_test_data_{DATASET_A_SNR}_{DATASET_A_SCATTER}_{DATASET_A_BLOCKS}_{DATASET_A_SUFFIX}.mat")
    PLOT_OUTPUT_DIR_A = os.path.join("Plots", "Dataset A")

    # --- Comparison Output Directory ---
    PLOT_OUTPUT_DIR_COMPARISON = os.path.join("Plots", "Comparison A vs B")

    # --- Create output directories ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    plot_save_path_a = os.path.join(script_dir, PLOT_OUTPUT_DIR_A)
    plot_save_path_b = os.path.join(script_dir, PLOT_OUTPUT_DIR_B)
    plot_save_path_comparison = os.path.join(script_dir, PLOT_OUTPUT_DIR_COMPARISON)

    if args.plot_a:
        os.makedirs(plot_save_path_a, exist_ok=True)
        print(f"Saving Dataset A plots to: {plot_save_path_a}")
    if args.plot_b:
        os.makedirs(plot_save_path_b, exist_ok=True)
        print(f"Saving Dataset B plots to: {plot_save_path_b}")
    if args.compare:
        os.makedirs(plot_save_path_comparison, exist_ok=True)
        print(f"Saving Comparison plots to: {plot_save_path_comparison}")

    # --- Load Data (Conditionally) ---
    perfect_m_A, received_m_A = None, None
    perfect_m_B, received_m_B = None, None

    if args.plot_a or args.compare:
        print("\n--- Loading Dataset A ---")
        perfect_m_A, received_m_A = load_data(DATA_FILE_PATH_A)

    if args.plot_b or args.compare:
        # Avoid loading B twice if we already loaded A for comparison
        if perfect_m_B is None:
            print("\n--- Loading Dataset B ---")
            perfect_m_B, received_m_B = load_data(DATA_FILE_PATH_B)

    # Determine max samples if comparison is needed
    max_samples = 0
    if args.compare and perfect_m_A is not None and perfect_m_B is not None:
        max_samples = min(perfect_m_A.shape[0], perfect_m_B.shape[0])
    elif args.plot_a and perfect_m_A is not None:
        max_samples = perfect_m_A.shape[0]
    elif args.plot_b and perfect_m_B is not None:
        max_samples = perfect_m_B.shape[0]

    SAMPLE_INDEX_TO_PLOT = 0
    if max_samples > 0:
        SAMPLE_INDEX_TO_PLOT = np.random.randint(0, max_samples)
    else:
        print("Warning: No data loaded or data is empty, cannot select sample index.")

    # --- Generate Plots for Dataset A (Individual) ---
    if args.plot_a and perfect_m_A is not None and received_m_A is not None:
        print("\n--- Generating Plots for Dataset A ---")
        # --- Heatmap Omitted ---
        if perfect_m_A.shape[0] > SAMPLE_INDEX_TO_PLOT:
            plot_example_heatmap(perfect_m_A[SAMPLE_INDEX_TO_PLOT],
                                 received_m_A[SAMPLE_INDEX_TO_PLOT],
                                 SAMPLE_INDEX_TO_PLOT,
                                 plot_save_path_a)
        else:
             print(f"Warning: SAMPLE_INDEX_TO_PLOT ({SAMPLE_INDEX_TO_PLOT}) is out of bounds for Dataset A. Skipping heatmap plot.")

        # Magnitude Distribution
        plot_distribution(np.abs(perfect_m_A), np.abs(received_m_A),
                          'Perfect Channel', 'Noisy Channel',
                          'Magnitude', plot_save_path_a, 'magnitude_distributions_A.svg')
        # --- Phase Distribution Omitted ---
        # plot_distribution(np.angle(perfect_m_A), np.angle(received_m_A),
        #               'Perfect Channel', 'Noisy Channel',
        #                   'Phase (Radians)', plot_save_path_a, 'phase_distributions_A.svg')
        # --- PDP Omitted ---
        # plot_pdp(perfect_m_A, plot_save_path_a)
        # Correlation
        plot_correlation(perfect_m_A, received_m_A, plot_save_path_a)
        # Noise Distribution
        plot_noise_distribution(perfect_m_A, received_m_A, plot_save_path_a)
        # --- Complex Distribution Omitted ---
        # plot_complex_distribution(perfect_m_A, received_m_A,
        #               'Perfect Channel', 'Noisy Channel',
        #                           plot_save_path_a, 'complex_distributions_A.svg')

    # --- Generate Plots for Dataset B (Individual) ---
    if args.plot_b and perfect_m_B is not None and received_m_B is not None:
        print("\n--- Generating Plots for Dataset B ---")
        # Recalculate sample index if only B is plotted
        if not args.compare and not args.plot_a:
             if perfect_m_B.shape[0] > 0:
                  SAMPLE_INDEX_TO_PLOT = np.random.randint(0, perfect_m_B.shape[0])
             else:
                  SAMPLE_INDEX_TO_PLOT = 0 # Keep it 0 if empty

        # --- Heatmap Omitted ---
        if perfect_m_B.shape[0] > SAMPLE_INDEX_TO_PLOT:
            plot_example_heatmap(perfect_m_B[SAMPLE_INDEX_TO_PLOT],
                                 received_m_B[SAMPLE_INDEX_TO_PLOT],
                                 SAMPLE_INDEX_TO_PLOT,
                                 plot_save_path_b)
        else:
            print(f"Warning: SAMPLE_INDEX_TO_PLOT ({SAMPLE_INDEX_TO_PLOT}) is out of bounds for Dataset B. Skipping heatmap plot.")

        # Magnitude Distribution
        plot_distribution(np.abs(perfect_m_B), np.abs(received_m_B),
                          'Perfect Channel', 'Noisy Channel',
                          'Magnitude', plot_save_path_b, 'magnitude_distributions_B.svg')
        # --- Phase Distribution Omitted ---
        # plot_distribution(np.angle(perfect_m_B), np.angle(received_m_B),
        #                   'Perfect Channel', 'Noisy Channel',
        #                   'Phase (Radians)', plot_save_path_b, 'phase_distributions_B.svg')
        # --- PDP Omitted ---
        # plot_pdp(perfect_m_B, plot_save_path_b)
        # Correlation
        plot_correlation(perfect_m_B, received_m_B, plot_save_path_b)
        # Noise Distribution
        plot_noise_distribution(perfect_m_B, received_m_B, plot_save_path_b)
        # --- Complex Distribution Omitted ---
        # plot_complex_distribution(perfect_m_B, received_m_B,
        #                       'Perfect Channel', 'Noisy Channel',
        #                           plot_save_path_b, 'complex_distributions_B.svg')

    # --- Generate Enhanced Comparison Plots --- #
    if args.compare and perfect_m_A is not None and received_m_A is not None and perfect_m_B is not None and received_m_B is not None:
        print("\n--- Generating Enhanced Comparison Analysis (A vs B) ---")
        
        # Calculate noise for both datasets
        noise_A = received_m_A - perfect_m_A
        noise_B = received_m_B - perfect_m_B
        
        # Dictionary to store all metrics for final report
        all_comparison_metrics = {}
        
        # 1. Perfect Channel Magnitude Comparison
        print("  Analysing Perfect Channel Magnitude...")
        metrics = plot_comprehensive_comparison(
            np.abs(perfect_m_A), np.abs(perfect_m_B),
            'Dataset A (Perfect)', 'Dataset B (Perfect)',
            'Perfect Channel Magnitude Comparison',
            plot_save_path_comparison, 'perfect_magnitude'
        )
        all_comparison_metrics['Perfect Channel Magnitude'] = metrics
        
        # 2. Noisy Channel Magnitude Comparison  
        print("  Analysing Noisy Channel Magnitude...")
        metrics = plot_comprehensive_comparison(
            np.abs(received_m_A), np.abs(received_m_B),
            'Dataset A (Noisy)', 'Dataset B (Noisy)', 
            'Noisy Channel Magnitude Comparison',
            plot_save_path_comparison, 'noisy_magnitude'
        )
        all_comparison_metrics['Noisy Channel Magnitude'] = metrics
        
        # 3. Perfect Channel Phase Comparison
        print("  Analysing Perfect Channel Phase...")
        metrics = plot_comprehensive_comparison(
            np.angle(perfect_m_A), np.angle(perfect_m_B),
            'Dataset A (Perfect)', 'Dataset B (Perfect)',
            'Perfect Channel Phase Comparison', 
            plot_save_path_comparison, 'perfect_phase'
        )
        all_comparison_metrics['Perfect Channel Phase'] = metrics
        
        # 4. Noisy Channel Phase Comparison
        print("  Analysing Noisy Channel Phase...")
        metrics = plot_comprehensive_comparison(
            np.angle(received_m_A), np.angle(received_m_B),
            'Dataset A (Noisy)', 'Dataset B (Noisy)',
            'Noisy Channel Phase Comparison',
            plot_save_path_comparison, 'noisy_phase'
        )
        all_comparison_metrics['Noisy Channel Phase'] = metrics
        
        # 5. Noise Real Part Comparison
        print("  Analysing Noise (Real Part)...")
        metrics = plot_comprehensive_comparison(
            noise_A.real, noise_B.real,
            'Dataset A (Noise)', 'Dataset B (Noise)',
            'Noise Real Part Comparison',
            plot_save_path_comparison, 'noise_real'
        )
        all_comparison_metrics['Noise Real Part'] = metrics
        
        # 6. Noise Imaginary Part Comparison
        print("  Analysing Noise (Imaginary Part)...")
        metrics = plot_comprehensive_comparison(
            noise_A.imag, noise_B.imag,
            'Dataset A (Noise)', 'Dataset B (Noise)',
            'Noise Imaginary Part Comparison',
            plot_save_path_comparison, 'noise_imag'
        )
        all_comparison_metrics['Noise Imaginary Part'] = metrics
        
        # 7. Noise Magnitude Comparison
        print("  Analysing Noise Magnitude...")
        metrics = plot_comprehensive_comparison(
            np.abs(noise_A), np.abs(noise_B),
            'Dataset A (Noise)', 'Dataset B (Noise)',
            'Noise Magnitude Comparison',
            plot_save_path_comparison, 'noise_magnitude'
        )
        all_comparison_metrics['Noise Magnitude'] = metrics
        
        # 8. Channel Real Part Comparison (Perfect)
        print("  Analysing Perfect Channel Real Part...")
        metrics = plot_comprehensive_comparison(
            perfect_m_A.real, perfect_m_B.real,
            'Dataset A (Perfect)', 'Dataset B (Perfect)',
            'Perfect Channel Real Part Comparison',
            plot_save_path_comparison, 'perfect_real'
        )
        all_comparison_metrics['Perfect Channel Real Part'] = metrics
        
        # 9. Channel Imaginary Part Comparison (Perfect)
        print("  Analysing Perfect Channel Imaginary Part...")
        metrics = plot_comprehensive_comparison(
            perfect_m_A.imag, perfect_m_B.imag,
            'Dataset A (Perfect)', 'Dataset B (Perfect)',
            'Perfect Channel Imaginary Part Comparison',
            plot_save_path_comparison, 'perfect_imag'
        )
        all_comparison_metrics['Perfect Channel Imaginary Part'] = metrics
        
        # Generate comprehensive comparison report
        print("  Creating comprehensive comparison report...")
        comparison_df = create_comparison_report(all_comparison_metrics, plot_save_path_comparison)
        
        # Create legacy overlayed plots for backward compatibility
        print("  Creating additional overlayed distribution plots...")
        plot_overlayed_distribution(np.abs(perfect_m_A), np.abs(perfect_m_B),
                                  'Dataset A (Perfect)', 'Dataset B (Perfect)',
                                  'Magnitude', 'Overlayed Magnitude Distribution (Perfect)',
                                  plot_save_path_comparison, 'magnitude_comparison_perfect_legacy.svg')
        
        plot_overlayed_distribution(np.abs(received_m_A), np.abs(received_m_B),
                                  'Dataset A (Noisy)', 'Dataset B (Noisy)',
                                  'Magnitude', 'Overlayed Magnitude Distribution (Noisy)',
                                  plot_save_path_comparison, 'magnitude_comparison_noisy_legacy.svg')

    print("\nPlot generation process finished.")

    # --- Generate Plots --- # (The sections below are now profiled)
    # --- Generate Plots for Dataset A (Individual) ---
    # ... (conditional plotting logic for A remains the same) ...

    # --- Generate Plots for Dataset B (Individual) ---
    # ... (conditional plotting logic for B remains the same) ...

    # --- Generate Comparison Plots --- #
    # ... (conditional plotting logic for comparison remains the same) ...

    print("\nPlot generation process finished.") 