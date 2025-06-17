import numpy as np

def add_cn_awgn(tensor, snr_db):
    """
    Add circularly symmetric complex Gaussian noise to a complex-valued tensor.

    Args:
        tensor (np.ndarray): Shape (N, H, W, 1), dtype complex64 or complex128
        snr_db (float): Desired signal-to-noise ratio in dB

    Returns:
        np.ndarray: Noisy tensor of same shape and dtype
    """
    if not np.iscomplexobj(tensor):
        raise ValueError("Tensor must be complex-valued (dtype=complex64 or complex128)")

    # Compute signal power
    power_signal = np.mean(np.abs(tensor)**2)

    # Convert SNR from dB to linear
    snr_linear = 10 ** (snr_db / 10)

    # Compute noise power
    power_noise = power_signal / snr_linear
    sigma = np.sqrt(power_noise / 2)

    # Generate complex Gaussian noise: real and imag parts
    noise = sigma * (np.random.randn(*tensor.shape) + 1j * np.random.randn(*tensor.shape))

    return tensor + noise
