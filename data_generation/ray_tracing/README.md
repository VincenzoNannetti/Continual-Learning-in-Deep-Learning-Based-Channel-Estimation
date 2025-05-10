# Ray-Tracing Based OFDM Channel Dataset Generation

## Overview

This directory contains Python scripts designed to generate datasets of Orthogonal Frequency Division Multiplexing (OFDM) channel matrices. The generation process simulates various wireless channel conditions using parameters derived from a simplified ray-tracing concept, allowing for the creation of diverse datasets suitable for research in wireless communications, machine learning for channel estimation, and other related fields.

The core script, `generate_dataset.py`, orchestrates the generation process, leveraging configuration files and helper classes to produce the desired datasets.

## Features

*   **Configurable Environments:** Utilises YAML configuration files (`../config/`) to define a wide range of channel parameters, including:
    *   Antenna distance
    *   Transmit (Tx) and Receive (Rx) antenna gains
    *   Signal-to-Noise Ratio (SNR)
    *   Number of scatterer clusters (simulating multipath richness)
    *   User Equipment (UE) speed (introducing Doppler effects)
    *   OFDM subcarrier spacing
    *   Use of an aggregate scattering model term
    *   Number of OFDM symbol blocks per sample
    *   Specific output directory for each dataset configuration
*   **Dynamic Pilot Patterns:** Implements a system for dynamically selecting pilot patterns (`pattern_low`, `pattern_medium`, `pattern_high`) for each generated sample based on its specific `speed`, `num_clusters`, and `snr` values. This simulates adaptive pilot allocation in real systems.
*   **Parallel Processing:** Employs `concurrent.futures.ProcessPoolExecutor` for efficient parallel generation of data samples, significantly reducing the time required to create large datasets.
*   **Modular Design:** Uses helper classes for distinct functionalities:
    *   `Channel`: Simulates the wireless channel propagation and noise.
    *   `Modulator`: Performs signal modulation (currently QAM-4).
    *   `Multiplexer`: Handles OFDM signal structuring (IFFT operation).
*   **Reproducibility:** Ensures reproducible dataset generation through controlled random number seeding for each sample.
*   **Output Format:** Saves generated datasets as `.mat` files, compatible with MATLAB and Python (`scipy.io.loadmat`). Each file includes:
    *   `perfect_matrices`: Ideal channel frequency response (complex-valued).
    *   `received_matrices`: Noisy channel frequency response (complex-valued).
    *   `pilot_mask`: Integer matrix (0s and 1s) indicating the positions where pilots would be placed.

## File Structure

```
data_generation/
├── ray_tracing/
│   ├── generate_dataset.py     # Main script for dataset generation
│   ├── README.md               # This file
│   └── classes/
│       ├── channel.py          # Channel simulation class
│       ├── modulator.py        # Modulation class
│       └── multiplexer.py      # OFDM signal multiplexing class
└── config/
    ├── dataset_a.yaml          # Example configuration for low multipath
    ├── dataset_b.yaml          # Example configuration for high multipath
    ├── dataset_mixed.yaml      # Example configuration for mixed conditions
    ├── rural_environment.yaml  # Example configuration for rural scenarios
    └── urban_environment.yaml  # Example configuration for urban scenarios
    ...                         # Other custom configuration files
```

## Configuration

Dataset generation is controlled by YAML files located in the `../config/` directory. Each file defines the parameters for a specific dataset. Key parameters include:

*   `antenna_distance`: Distance between Tx and Rx (meters).
*   `tx_gain`, `rx_gain`: Antenna gains (dBi).
*   `snr`: Signal-to-Noise Ratio (dB). Can be a single value or a list `[min, max]` for uniform sampling.
*   `num_clusters`: Number of scatterer clusters. Can be a single value or `[min, max]` for uniform integer sampling.
*   `speed`: UE speed (m/s). Can be a single value or `[min, max]` for uniform sampling.
*   `subcarrier_spacing`: OFDM subcarrier spacing (Hz).
*   `use_aggregate_scattering`: Boolean (`true`/`false`) or a list `[true, false]` to enable/disable an additional scattering term in the channel model.
*   `blocks`: Number of 14-symbol OFDM blocks to generate per sample.
*   `output_dir`: The directory path where the generated `.mat` file will be saved (e.g., `"./data/raw/dataset_a"`).

Refer to the example YAML files for detailed structure.

## Running the Script

The main script `generate_dataset.py` is executed from the command line. It is recommended to run it as a module from the parent directory of `data_generation`.

**Command:**

```bash
python -m data_generation.ray_tracing.generate_dataset --config <config_file_path> --samples <num_samples> [--output <output_directory_override>]
```

**Parameters:**

*   `--config <config_file_path>`: (Required) Path to the YAML configuration file (e.g., `data_generation/config/dataset_a.yaml`).
*   `--samples <num_samples>`: (Required) Number of data samples to generate.
*   `--output <output_directory_override>`: (Optional) Path to an output directory. If provided, this will override the `output_dir` specified in the YAML configuration file.

**Example:**

```bash
python -m data_generation.ray_tracing.generate_dataset --config data_generation/config/urban_environment.yaml --samples 10000
```

This command will generate 10,000 samples using the parameters defined in `urban_environment.yaml` and save the output to the directory specified within that config file (e.g., `./data/raw/urban_environment/`).

## Dependencies

The following Python packages are required:

*   NumPy
*   SciPy
*   PyYAML
*   tqdm

You can typically install them using pip:
`pip install numpy scipy pyyaml tqdm`

## Class Descriptions

*   **`Channel`**: This class is responsible for simulating the effects of the wireless channel on the transmitted signal. It takes into account parameters such as antenna distance, gains, SNR, number of clusters (multipath), UE speed (Doppler), and aggregate scattering. It provides methods to obtain both the perfect (noise-free) and noisy channel matrices.
*   **`Modulator`**: This class handles the modulation of data symbols. Currently, it is configured to use 4-QAM (Quadrature Amplitude Modulation). It takes random data and converts it into complex modulated symbols.
*   **`Multiplexer`**: This class prepares the modulated symbols for OFDM transmission. Its primary role is to arrange the symbols into an OFDM grid (subcarriers vs. time symbols) and then perform the Inverse Fast Fourier Transform (IFFT) to convert the frequency-domain signal into a time-domain signal, ready for the channel. It no longer handles pilot insertion directly, as this is managed by `generate_dataset.py` for dataset recording purposes.

---

