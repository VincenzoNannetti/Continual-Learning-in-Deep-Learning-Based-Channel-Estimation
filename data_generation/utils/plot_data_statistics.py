import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
from scipy.stats import entropy # Added for KL divergence

from .ieee_plot_style import setup_ieee_style

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
    axs[0].set_xlabel('OFDM Symbol Index')
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
    axs[1].set_xlabel('OFDM Symbol Index')
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

def plot_distribution(data1, data2, title1, title2, xlabel, save_path, filename, bins=50, max_points=100000):
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

    # Limit y-axis range for better visualization, relative to the peak in the causal part
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
    """Plots the distribution of the real and imaginary parts of the noise, with subsampling for KDE and Histogram."""
    noise = received_matrices - perfect_matrices
    noise_real = noise.real.flatten()
    noise_imag = noise.imag.flatten()

    # --- Subsampling --- #
    if noise_real.size > max_points:
        print(f"Subsampling real noise from {noise_real.size} to {max_points} for Hist/KDE plot.")
        plot_noise_real = noise_real[np.random.choice(noise_real.size, max_points, replace=False)]
    else:
        plot_noise_real = noise_real

    if noise_imag.size > max_points:
        print(f"Subsampling imag noise from {noise_imag.size} to {max_points} for Hist/KDE plot.")
        plot_noise_imag = noise_imag[np.random.choice(noise_imag.size, max_points, replace=False)]
    else:
        plot_noise_imag = noise_imag
    # -------------------

    fig, axs = plt.subplots(1, 2, figsize=(7, 2.8))
    fig.suptitle('Noise Component Distribution', fontsize=plt.rcParams['figure.titlesize'], y=0.95)

    bins = 50
    color_real = '#1f77b4'
    color_imag = '#ff7f0e'

    # Plot Real Part - Use SUBSAMPLED data for hist and KDE
    sns.histplot(plot_noise_real, kde=False, bins=bins, stat='density', ax=axs[0], color=color_real, edgecolor=None, linewidth=0.5)
    sns.kdeplot(plot_noise_real, ax=axs[0], color=color_real)
    axs[0].set_title('Noise (Real Part)')
    axs[0].set_xlabel('Value')
    axs[0].set_ylabel('Probability Density')

    # Plot Imaginary Part - Use SUBSAMPLED data for hist and KDE
    sns.histplot(plot_noise_imag, kde=False, bins=bins, stat='density', ax=axs[1], color=color_imag, edgecolor=None, linewidth=0.5)
    sns.kdeplot(plot_noise_imag, ax=axs[1], color=color_imag)
    axs[1].set_title('Noise (Imaginary Part)')
    axs[1].set_xlabel('Value')
    axs[1].set_ylabel('Probability Density')

    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    plt.savefig(os.path.join(save_path, 'noise_distribution.svg'))
    plt.close(fig)
    print("Saved Noise Distribution plot.")

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

# --- Helper function for KL Divergence ---
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
        # If both datasets are identical constants, KL is 0. If different, it's infinite.
        # We can return 0 as a practical measure here, or NaN. Let's return NaN.
        return np.nan, np.nan

    bins = np.linspace(min_val, max_val, num_bins + 1)

    # Create histograms and normalize to get probability distributions (PMFs)
    hist_a, _ = np.histogram(data_a, bins=bins, density=False)
    hist_b, _ = np.histogram(data_b, bins=bins, density=False)

    # Normalize to sum to 1
    pmf_a = hist_a / np.sum(hist_a) if np.sum(hist_a) > 0 else np.zeros_like(hist_a)
    pmf_b = hist_b / np.sum(hist_b) if np.sum(hist_b) > 0 else np.zeros_like(hist_b)

    # Add epsilon to avoid division by zero or log(0) in entropy calculation
    pmf_a_smooth = pmf_a + epsilon
    pmf_b_smooth = pmf_b + epsilon
    pmf_a_smooth /= np.sum(pmf_a_smooth) # Re-normalize after adding epsilon
    pmf_b_smooth /= np.sum(pmf_b_smooth) # Re-normalize after adding epsilon


    # Calculate KL divergence using scipy.stats.entropy
    try:
        kl_div_ab = entropy(pk=pmf_a_smooth, qk=pmf_b_smooth)
        kl_div_ba = entropy(pk=pmf_b_smooth, qk=pmf_a_smooth)
        return kl_div_ab, kl_div_ba
    except Exception as e:
        print(f"Error calculating KL divergence: {e}")
        return np.nan, np.nan


if __name__ == "__main__":
    # --- Argument Parsing --- #
    parser = argparse.ArgumentParser(description="Generate visualizations for channel data.")
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

    # --- Generate Comparison Plots --- #
    if args.compare and perfect_m_A is not None and received_m_A is not None and perfect_m_B is not None and received_m_B is not None:
        print("\n--- Generating Comparison Plots (A vs B) ---")

        # Overlayed Magnitude Distribution (Perfect Channel)
        plot_overlayed_distribution(np.abs(perfect_m_A), np.abs(perfect_m_B),
                                  'Dataset A (Perfect)', 'Dataset B (Perfect)',
                                  'Magnitude', 'Overlayed Magnitude Distribution (Perfect)',
                                  plot_save_path_comparison, 'magnitude_comparison_perfect.svg')

        # Overlayed Magnitude Distribution (Received Channel)
        plot_overlayed_distribution(np.abs(received_m_A), np.abs(received_m_B),
                                  'Dataset A (Noisy)', 'Dataset B (Noisy)',
                                  'Magnitude', 'Overlayed Magnitude Distribution (Noisy)',
                                  plot_save_path_comparison, 'magnitude_comparison_noisy.svg')

        # --- Overlayed Phase Distribution Omitted ---
        # plot_overlayed_distribution(np.angle(perfect_m_A), np.angle(perfect_m_B),
        #                           'Dataset A (Perfect)', 'Dataset B (Perfect)',
        #                           'Phase (Radians)', 'Overlayed Phase Distribution (Perfect)',
        #                           plot_save_path_comparison, 'phase_comparison_perfect.svg')
        # plot_overlayed_distribution(np.angle(received_m_A), np.angle(received_m_B),
        #                           'Dataset A (Noisy)', 'Dataset B (Noisy)',
        #                           'Phase (Radians)', 'Overlayed Phase Distribution (Noisy)',
        #                           plot_save_path_comparison, 'phase_comparison_noisy.svg')

        # --- Add Overlayed Noise Distributions ---
        noise_A = received_m_A - perfect_m_A
        noise_B = received_m_B - perfect_m_B

        # Overlayed Noise Distribution (Real Part)
        plot_overlayed_distribution(noise_A.real, noise_B.real,
                                  'Dataset A (Noise Real)', 'Dataset B (Noise Real)',
                                  'Noise Value', 'Overlayed Noise Distribution (Real Part)',
                                  plot_save_path_comparison, 'noise_comparison_real.svg')

        # Overlayed Noise Distribution (Imaginary Part)
        plot_overlayed_distribution(noise_A.imag, noise_B.imag,
                                  'Dataset A (Noise Imag)', 'Dataset B (Noise Imag)',
                                  'Noise Value', 'Overlayed Noise Distribution (Imaginary Part)',
                                  plot_save_path_comparison, 'noise_comparison_imag.svg')

        # --- Add Overlayed Noise Magnitude Distribution ---
        noise_magnitude_A = np.abs(noise_A)
        noise_magnitude_B = np.abs(noise_B)

        plot_overlayed_distribution(noise_magnitude_A, noise_magnitude_B,
                                  'Dataset A (Noise Mag)', 'Dataset B (Noise Mag)',
                                  'Noise Magnitude', 'Overlayed Noise Magnitude Distribution',
                                  plot_save_path_comparison, 'noise_comparison_magnitude.svg')

        # --- Add KL Divergence Calculations ---
        print("--- Calculating KL Divergence (A vs B) ---")
        kl_results = {}

        # 1. Perfect Magnitude
        print("  Calculating KL divergence for Perfect Magnitude...")
        kl_ab, kl_ba = calculate_kl_divergence(np.abs(perfect_m_A).flatten(), np.abs(perfect_m_B).flatten())
        kl_results['Perfect Magnitude (A||B)'] = kl_ab
        kl_results['Perfect Magnitude (B||A)'] = kl_ba

        # 2. Noisy Magnitude
        print("  Calculating KL divergence for Noisy Magnitude...")
        kl_ab, kl_ba = calculate_kl_divergence(np.abs(received_m_A).flatten(), np.abs(received_m_B).flatten())
        kl_results['Noisy Magnitude (A||B)'] = kl_ab
        kl_results['Noisy Magnitude (B||A)'] = kl_ba

        # Calculate Noise
        noise_A = received_m_A - perfect_m_A
        noise_B = received_m_B - perfect_m_B

        # 3. Noise Real Part
        print("  Calculating KL divergence for Noise (Real Part)...")
        kl_ab, kl_ba = calculate_kl_divergence(noise_A.real.flatten(), noise_B.real.flatten())
        kl_results['Noise Real (A||B)'] = kl_ab
        kl_results['Noise Real (B||A)'] = kl_ba

        # 4. Noise Imaginary Part
        print("  Calculating KL divergence for Noise (Imaginary Part)...")
        kl_ab, kl_ba = calculate_kl_divergence(noise_A.imag.flatten(), noise_B.imag.flatten())
        kl_results['Noise Imag (A||B)'] = kl_ab
        kl_results['Noise Imag (B||A)'] = kl_ba

        # 5. Noise Magnitude
        print("  Calculating KL divergence for Noise Magnitude...")
        kl_ab, kl_ba = calculate_kl_divergence(np.abs(noise_A).flatten(), np.abs(noise_B).flatten())
        kl_results['Noise Magnitude (A||B)'] = kl_ab
        kl_results['Noise Magnitude (B||A)'] = kl_ba

        # Print results
        print("--- KL Divergence Results ---")
        for key, value in kl_results.items():
            print(f"  {key}: {value:.4f}" if not np.isnan(value) else f"  {key}: Calculation failed/undefined")
        print("-" * 30)

    print("\nPlot generation process finished.")

    # --- Generate Plots --- # (The sections below are now profiled)
    # --- Generate Plots for Dataset A (Individual) ---
    # ... (conditional plotting logic for A remains the same) ...

    # --- Generate Plots for Dataset B (Individual) ---
    # ... (conditional plotting logic for B remains the same) ...

    # --- Generate Comparison Plots --- #
    # ... (conditional plotting logic for comparison remains the same) ...

    print("\nPlot generation process finished.") 