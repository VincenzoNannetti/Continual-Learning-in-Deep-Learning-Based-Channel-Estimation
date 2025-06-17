"""
Filename: ./utils/interpolation.py
Author: Vincenzo Nannetti
Date: 05/03/2025
Description: Function used to interpolate the data

Usage:

Dependencies:
    - numpy
    - scipy
"""
import numpy as np
from scipy import interpolate

def interpolation(noisy, pilot_mask, method="rbf"):
    """Interpolates channel estimates based on pilot locations derived from a mask.

    Args:
        noisy (np.ndarray): The noisy channel estimate tensor with pilots.
                            Expected shape: (n_samples, n_subcarriers, n_symbols, 2) for real/imag.
        pilot_mask (np.ndarray): Boolean or integer mask indicating pilot locations.
                                 Expected shape: (n_samples, n_subcarriers, n_symbols).
                                 Value > 0 indicates a pilot.
        method (str): Interpolation method ('rbf' or 'spline').

    Returns:
        np.ndarray: Interpolated channel estimate tensor.
                    Shape: (n_samples, n_subcarriers, n_symbols, 2).
    """
    n_samples, n_sc, n_sym, n_channels = noisy.shape

    if pilot_mask.shape != (n_samples, n_sc, n_sym):
         raise ValueError(f"Shape mismatch: noisy data ({n_samples}, {n_sc}, {n_sym}) vs pilot_mask ({pilot_mask.shape})")

    if n_channels != 2:
        print(f"Warning: Expected last dimension of noisy tensor to be 2 (real/imag), but got {n_channels}.")

    interpolated_noisy = np.zeros((n_samples, n_sc, n_sym, n_channels), dtype=float)

    # Track interpolation success/failure stats
    successful_samples  = 0
    failed_samples      = 0
    min_pilots_required = {'rbf': 4, 'spline': 16}  # Minimum pilots needed for each method
    
    for i in range(n_samples):
        # Find pilot coordinates for the current sample from the mask
        # np.where returns a tuple of arrays (one for each dimension)
        rows, cols = np.where(pilot_mask[i] > 0) # Find indices where mask is non-zero

        if rows.size == 0:
            print(f"Warning: No pilots found in mask for sample {i}. Skipping interpolation.")
            interpolated_noisy[i] = noisy[i] 
            failed_samples += 1
            continue
        
        # Check if we have enough pilots for the chosen method
        if rows.size < min_pilots_required.get(method, 4):
            print(f"Warning: Only {rows.size} pilots found in sample {i}, but {method} requires at least {min_pilots_required.get(method, 4)}. Using fallback method.")
            # For fallback, use nearest neighbor interpolation which works with any number of pilots
            interpolated_noisy[i] = _nearest_neighbor_interpolation(noisy[i], rows.astype(int), cols.astype(int), n_sc, n_sym, n_channels)
            failed_samples += 1
            continue

        rows = rows.astype(float)
        cols = cols.astype(float)

        sample_success = True  # Track success for this sample
        for j in range(n_channels): 
            # Get pilot values for the current sample and channel using integer indices
            z = noisy[i, rows.astype(int), cols.astype(int), j]

            if method == 'rbf':
                try:
                    # RBF: Use columns (symbols) as x, rows (subcarriers) as y
                    # this method same as base paper.
                    f = interpolate.Rbf(cols, rows, z, function='gaussian')
                    X_eval, Y_eval = np.meshgrid(np.arange(n_sym), np.arange(n_sc))
                    z_intp         = f(X_eval, Y_eval)
                    interpolated_noisy[i, :, :, j] = z_intp
                except Exception as e:
                    print(f"Error during RBF interpolation sample {i}, channel {j}: {e}")
                    # Fallback to nearest neighbor on error
                    interpolated_noisy[i, :, :, j] = _nearest_neighbor_fallback(noisy[i, :, :, j], rows.astype(int), cols.astype(int), z)
                    sample_success = False

            elif method == 'spline':
                try:
                    # Bivariate Spline: x is rows (subcarriers), y is columns (symbols)
                    tck = interpolate.bisplrep(rows, cols, z, kx=min(3, rows.size-1), ky=min(3, cols.size-1))
                    x_range = np.arange(n_sc) # Rows
                    y_range = np.arange(n_sym) # Columns
                    z_intp = interpolate.bisplev(x_range, y_range, tck)
                    interpolated_noisy[i, :, :, j] = z_intp
                except Exception as e:
                    print(f"Error during Spline interpolation sample {i}, channel {j}: {e}")
                    # Fallback to nearest neighbor on error
                    interpolated_noisy[i, :, :, j] = _nearest_neighbor_fallback(noisy[i, :, :, j], rows.astype(int), cols.astype(int), z)
                    sample_success = False
            else:
                 raise ValueError(f"Unsupported interpolation method: {method}")
        
        if sample_success:
            successful_samples += 1
        else:
            failed_samples += 1

    return interpolated_noisy

def _nearest_neighbor_interpolation(sample, rows, cols, n_sc, n_sym, n_channels):
    """Simple nearest neighbor interpolation for samples with too few pilots."""
    result = np.zeros((n_sc, n_sym, n_channels), dtype=float)
    
    # Use a grid of coordinates for all points in the grid
    grid_rows, grid_cols = np.mgrid[0:n_sc, 0:n_sym]
    
    # For each pilot position, calculate distances to all grid points
    for ch in range(n_channels):
        # Initisalise with zeros or the original sample
        channel_result = np.zeros((n_sc, n_sym))
        
        # For each non-pilot position, find the nearest pilot
        for r in range(n_sc):
            for c in range(n_sym):
                # Skip if this is a pilot position
                if any((rows == r) & (cols == c)):
                    # Keep original value at pilot positions
                    channel_result[r, c] = sample[r, c, ch]
                    continue
                
                # Calculate distances to all pilots
                distances = np.sqrt((rows - r)**2 + (cols - c)**2)
                # Find nearest pilot
                nearest_idx = np.argmin(distances)
                # Get value from nearest pilot
                channel_result[r, c] = sample[rows[nearest_idx], cols[nearest_idx], ch]
        
        result[:, :, ch] = channel_result
    
    return result

def _nearest_neighbor_fallback(channel, rows, cols, values):
    """Fallback for individual channel interpolation failures."""
    # Get shape from channel
    n_sc, n_sym = channel.shape
    result = np.zeros((n_sc, n_sym))
    
    # For each non-pilot position, find the nearest pilot
    for r in range(n_sc):
        for c in range(n_sym):
            # Skip if this is a pilot position
            if any((rows == r) & (cols == c)):
                # Find index in the rows/cols arrays
                idx = np.where((rows == r) & (cols == c))[0][0]
                # Use the original pilot value
                result[r, c] = values[idx]
                continue
            
            # Calculate distances to all pilots
            distances = np.sqrt((rows - r)**2 + (cols - c)**2)
            # Find nearest pilot
            nearest_idx = np.argmin(distances)
            # Get value from nearest pilot
            result[r, c] = values[nearest_idx]
    
    return result



# import numpy as np
# from scipy.interpolate import RBFInterpolator
# def interpolation(noisy, pilot_mask, kernel="thin_plate_spline"):
#     """Interpolates channel estimates based on pilot locations derived from a mask.

#     Args:
#         noisy (np.ndarray): The noisy channel estimate tensor with pilots.
#                            Expected shape: (n_samples, n_subcarriers, n_symbols, 2) for real/imag.
#         pilot_mask (np.ndarray): Boolean or integer mask indicating pilot locations.
#                                 Expected shape: (n_samples, n_subcarriers, n_symbols).
#                                 Value > 0 indicates a pilot.
#         kernel (str): The RBF kernel to use. Default is "thin_plate_spline".
#                      Other options: "cubic", "gaussian", "linear", "quintic".

#     Returns:
#         np.ndarray: Interpolated channel estimate tensor.
#                    Shape: (n_samples, n_subcarriers, n_symbols, 2).
#     """
#     n_samples, n_sc, n_sym, n_channels = noisy.shape

#     if pilot_mask.shape != (n_samples, n_sc, n_sym):
#          raise ValueError(f"Shape mismatch: noisy data ({n_samples}, {n_sc}, {n_sym}) vs pilot_mask ({pilot_mask.shape})")

#     if n_channels != 2:
#         raise ValueError(f"Expected last dimension of noisy tensor to be 2 (real/imag), but got {n_channels}.")

#     interpolated_noisy = np.zeros((n_samples, n_sc, n_sym, n_channels), dtype=float)

#     successful_samples = 0
#     failed_samples = 0

#     for i in range(n_samples):
#         # Get pilot positions
#         rows, cols = np.where(pilot_mask[i] > 0)
#         if rows.size == 0:
#             print(f"Warning: No pilots found in sample {i}. Skipping.")
#             interpolated_noisy[i] = noisy[i]
#             failed_samples += 1
#             continue

#         # Prepare grid: (symbol, subcarrier) coordinates for interpolation
#         eval_points = np.stack(np.meshgrid(np.arange(n_sym), np.arange(n_sc)), axis=-1).reshape(-1, 2)
#         train_points = np.column_stack((cols, rows))  # (x, y) = (symbol, subcarrier)

#         sample_success = True
#         for j in range(n_channels):
#             # Get pilot values for current channel (real/imag)
#             pilot_values = noisy[i, rows, cols, j]

#             try:
#                 # Perform RBF interpolation
#                 rbf = RBFInterpolator(
#                     train_points,
#                     pilot_values,
#                     kernel=kernel,
#                     neighbors=None,  # Use all pilots
#                     degree=1  # Include linear term for better extrapolation
#                 )
                
#                 # Interpolate the entire grid
#                 interpolated_values = rbf(eval_points)
                
#                 # Reshape back to matrix form
#                 interpolated_noisy[i, :, :, j] = interpolated_values.reshape(n_sc, n_sym)
                
#                 # Preserve original pilot values
#                 interpolated_noisy[i, rows, cols, j] = pilot_values
                
#             except Exception as e:
#                 print(f"Error during RBF interpolation for sample {i}, channel {j}: {e}")
#                 sample_success = False
#                 break

#         if sample_success:
#             successful_samples += 1
#         else:
#             failed_samples += 1
#             # Preserve pilot positions in case of failure
#             pilot_indices = pilot_mask[i] > 0
#             interpolated_noisy[i, pilot_indices] = noisy[i, pilot_indices]

#     print(f"Interpolation complete: {successful_samples} successful, {failed_samples} failed")
#     return interpolated_noisy




