import numpy as np
from scipy.interpolate import RBFInterpolator

def interpolation(noisy, pilot_mask, kernel="thin_plate_spline"):
    n_samples, n_sc, n_sym, n_channels = noisy.shape

    if pilot_mask.shape != (n_samples, n_sc, n_sym):
         raise ValueError(f"Shape mismatch: noisy data ({n_samples}, {n_sc}, {n_sym}) vs pilot_mask ({pilot_mask.shape})")

    if n_channels != 2:
        raise ValueError(f"Expected last dimension of noisy tensor to be 2 (real/imag), but got {n_channels}.")

    interpolated_noisy = np.zeros((n_samples, n_sc, n_sym, n_channels), dtype=float)

    successful_samples = 0
    failed_samples = 0

    for i in range(n_samples):
        # Get pilot positions
        rows, cols = np.where(pilot_mask[i] > 0)
        if rows.size == 0:
            print(f"Warning: No pilots found in sample {i}. Skipping.")
            interpolated_noisy[i] = noisy[i]
            failed_samples += 1
            continue

        # Prepare grid: (symbol, subcarrier) coordinates for interpolation
        eval_points = np.stack(np.meshgrid(np.arange(n_sym), np.arange(n_sc)), axis=-1).reshape(-1, 2)
        train_points = np.column_stack((cols, rows))  # (x, y) = (symbol, subcarrier)

        sample_success = True
        for j in range(n_channels):
            # Get pilot values for current channel (real/imag)
            pilot_values = noisy[i, rows, cols, j]

            try:
                # Perform RBF interpolation
                rbf = RBFInterpolator(
                    train_points,
                    pilot_values,
                    kernel=kernel,
                    neighbors=None,  # Use all pilots
                    degree=1  # Include linear term for better extrapolation
                )
                
                # Interpolate the entire grid
                interpolated_values = rbf(eval_points)
                
                # Reshape back to matrix form
                interpolated_noisy[i, :, :, j] = interpolated_values.reshape(n_sc, n_sym)
                
                # Preserve original pilot values
                interpolated_noisy[i, rows, cols, j] = pilot_values
                
            except Exception as e:
                print(f"Error during RBF interpolation for sample {i}, channel {j}: {e}")
                sample_success = False
                break

        if sample_success:
            successful_samples += 1
        else:
            failed_samples += 1
            # Preserve pilot positions in case of failure
            pilot_indices = pilot_mask[i] > 0
            interpolated_noisy[i, pilot_indices] = noisy[i, pilot_indices]

    # print(f"Interpolation complete: {successful_samples} successful, {failed_samples} failed")
    return interpolated_noisy




