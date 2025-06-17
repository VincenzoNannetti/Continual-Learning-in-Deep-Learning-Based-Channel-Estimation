# Data Generation Framework for Ray-Tracing Based Wireless Channel Simulation

## 1. Overview

This framework generates synthetic wireless channel data, primarily using a 3D ray-tracing engine combined with statistical elements inspired by 3GPP channel models. The goal is to produce realistic and configurable channel matrices suitable for developing and evaluating communication systems and machine learning algorithms for wireless tasks.

The core output is a channel matrix \( \mathbf{H} \) representing the complex gains between transmitter (Tx) and receiver (Rx) antenna elements over a set of OFDM subcarriers and symbols.

## 2. Core Components & Workflow

The data generation process involves several key Python classes and a workflow orchestrated by example/generation scripts:

**Entities:**
*   **`Environment` (`environment.py`):** Manages the 3D simulation space, all physical entities (antennas, buildings, scatterers, clusters), their interactions (visibility, reflections), and their state updates over time.
    *   Procedurally generates scenarios (e.g., street canyons with buildings) based on configuration.
    *   Clips buildings to environment bounds.
    *   Places scatterer clusters in free regions using geometric libraries (Shapely).
*   **`Antenna` (`antenna.py`):** Represents Tx or Rx antennas with properties like gain, 3D position, and movement.
    *   Supports various movement models (static, linear, circular, zigzag, random waypoint) implemented in `ue_movements.py`.
    *   Movement is constrained by environment boundaries, pavement areas, and buildings.
*   **`Building` (`building.py`):** Defines cuboid structures that block rays.
    *   Implements ray-AABB intersection for LoS checks and reflection point calculation.
    *   Can host `Window` objects.
    *   Reflection from building surfaces uses a simple configurable coefficient (not material-based like scatterers).
*   **`Window` (`window.py`):** Rectangular areas on building faces that can reflect/transmit rays.
    *   Calculates its 3D geometry based on parent building and face index.
    *   Implements ray-rectangle intersection.
    *   Reflection uses a configurable coefficient and an angle threshold.
*   **`Scatterer` (`scatterer.py`):** Individual point scatterers.
    *   Reflection coefficient (`self.reflection_coeff`) is complex:
        *   Amplitude can be user-defined or sampled from a material-based model (`_sample_reflection_from_materials`) using Fresnel equations for normal incidence, considering `environment_type` (UMa, UMi etc.) and `frequency_ghz`.
        *   Phase is random.
    *   Supports movement models (static, linear, random_walk, flocking) implemented in `scatterer_movements.py`, with reflecting boundaries.
    *   Can calculate its Doppler frequency contribution (though direct application in channel phase needs confirmation).
*   **Supporting Modules:**
    *   `vector_utils.py`: 3D vector math, Numba-accelerated functions.
    *   `ue_movements.py`: Logic for UE (Antenna) movement and collision handling.
    *   `scatterer_movements.py`: Logic for Scatterer movement.

**Channel Generation (`Channel.py`):**
The `Channel` class computes the channel matrix \( H[k,l] \) for subcarrier \(k\) and symbol \(l\).
1.  **Initialization:** Takes `Environment`, Tx/Rx `Antenna` objects, SNR, center frequency, and 3GPP-inspired parameters (`rms_delay_spread_ns`, `shadow_fading_std_db`).
2.  **Preparation (`prepare_channel`):**
    *   Generates `cluster_delays` for all clusters: drawn from an exponential distribution scaled by `rms_delay_spread_ns` (3GPP TDL characteristic).
3.  **Main Loop (`apply_channel` over OFDM symbols):**
    For each OFDM symbol `i`:
    *   **Environment State:**
        *   Gets current Rx antenna position (Tx is static per call).
        *   `env.get_scatterer_snapshot(tx_pos, rx_pos)`: Provides current positions, reflection coefficients, and a *visibility mask* for all scatterers. Visibility is determined by checking LoS from Tx to scatterer and scatterer to Rx against building blockages using `env.path_blocked`.
        *   `env.get_valid_window_reflections(tx_pos, rx_pos)`: Provides pre-calculated valid reflection/transmission paths through windows, including path distance and effective coefficient.
    *   **Shadow Fading:** A time-correlated (AR1 model if UE moves, related to `symbols_per_block`) or i.i.d. log-normal shadow fading term (`shadowing_factor_linear`) is calculated based on `shadow_fading_std_db`.
    *   **Path Gain Calculation (`calculate_path_gain`):** Uses Friis equation \( \sqrt{G_{tx}G_{rx}} \frac{\lambda}{4\pi d} e^{-j2\pi d/\lambda} \) for complex gain.
    *   **Per-Subcarrier Loop (for subcarrier `j`):**
        *   **Line-of-Sight (LoS):** Calculated if `env.path_blocked(tx_pos, rx_pos)` is false. Gain from `calculate_path_gain`, scaled by `shadowing_factor_linear`.
        *   **Window Paths:** Sum of contributions from `env.get_valid_window_reflections`. Each path uses its distance and coefficient `wp["coeff"]`, scaled by `shadowing_factor_linear`.
        *   **Clustered Scatterer Paths:**
            *   Iterates through *visible* scatterers.
            *   Geometric path gain (Tx-Scatterer-Rx) calculated using `calculate_path_gain`.
            *   This is multiplied by the scatterer's `reflection_coeff`.
            *   **Cluster Delay:** An *additional* phase shift \( e^{-j2\pi f_k \tau_{cluster}} \) is applied, where \( \tau_{cluster} \) is the pre-calculated delay for the scatterer's assigned cluster. This introduces frequency selectivity tied to the cluster's statistical delay.
            *   Contributions from scatterers within the same cluster are summed, then all cluster sums are combined and scaled by `shadowing_factor_linear`.
        *   **Aggregate Scattering (`H_ij`):** An additional complex Gaussian term is added. Its variance is modeled as \( (10^{-3.24}) \cdot (f_k / 1\text{GHz})^{-2} \cdot (\text{LOS\_distance})^{-3} \), providing a diffuse scattering component dependent on frequency and distance.
        *   **Total Channel:** `channel_matrix[j, i]` = LoS + Window + ClusteredScatterers + AggregateScatter.
    *   **Environment Advancement:** `env.advance(dt_effective_block)` is called, where `dt_effective_block = symbols_per_block * symbol_duration`. This updates UE and scatterer positions. The environment state is thus piece-wise constant for `symbols_per_block` durations from the channel's perspective.

**Data Transmission (Simplified for Channel Probing):**
*   The example scripts (e.g., `channel_environment_example.py`) often bypass explicit `Modulator` and `Multiplexer` for channel generation.
*   A dummy signal (e.g., all ones) is used as input to `channel.apply_channel`.
*   `channel.apply_channel` applies the computed `channel_matrix` in the frequency domain.
*   Noise (AWGN) is added based on SNR, where SNR is defined over the average power of the received signal \( E\{|H \cdot X|^2\}/N_0 \).
*   The primary outputs for dataset generation are the `perfect_matrices` (noise-free \( \mathbf{H} \)) and `received_matrices` (noisy \( \mathbf{H} \), or \( \mathbf{Y} \) if a signal \( \mathbf{X} \) was notionally passed).

**Configuration (`dataset_a.yaml`, `3gpp_presets.yaml`):**
*   YAML files define parameters for:
    *   Environment: Dimensions, scenario type (e.g., UMa for scatterer material choices).
    *   Scenario Layout: Building density, road properties (used by procedural generation in examples).
    *   Antennas: Position, gain, movement models (type, speed, bounds like pavement).
    *   Clusters: Number, density, scatterer count, radii, height ranges, movement (e.g., flocking).
    *   Scatterers: Reflection amplitude ranges, if aggregate scattering flag is used per scatterer.
    *   Channel: Carrier frequency, SNR range, `symbols_per_block`, `rms_delay_spread_ns`, `shadow_fading_std_db`.
    *   OFDM: Subcarrier spacing, number of blocks/symbols.

## 3. "Mix of 3GPP" Elements

The ray-tracing model incorporates 3GPP standard concepts in several ways:
1.  **Scenario Definition:** Configuration presets (e.g., "UMa," "UMi" in `3gpp_presets.yaml`) guide the procedural generation of environments (BS/UE heights, typical building densities, `environment_type` for material selection of scatterers).
2.  **Cluster Delays:** Scatterers are grouped into clusters. Each *cluster* is assigned a delay \( \tau_{cluster} \) drawn from an exponential distribution parameterized by a configured Root Mean Square Delay Spread (`rms_delay_spread_ns`), a common 3GPP TDL parameter. This delay introduces an additional, frequency-dependent phase rotation \( e^{-j2\pi f_k \tau_{cluster}} \) to all rays associated with that cluster's scatterers, affecting frequency selectivity.
3.  **Shadow Fading:** Large-scale shadow fading is applied as a log-normal random variable (with configurable standard deviation `shadow_fading_std_db`) that multiplies the sum of geometric path gains. It can be time-correlated based on UE movement and a configured `symbols_per_block` value.
4.  **Aggregate Scattering Component:** An additional statistical (non-geometric) scattering term `H_ij` is added to the channel response. Its variance is modeled based on distance and frequency, contributing to diffuse scattering not captured by discrete scatterers.
5.  **Parameter Values:** Some default parameters (e.g., antenna heights, UE speed for `scatterer_speed_for_clusters`) are chosen with reference to 3GPP TR 38.901.
6.  Angle conventions (`vector_utils.zenith_azimuth`) aim to follow 3GPP.

The model does **not** directly implement full 3GPP statistical channel models like CDL/TDL for the ray-traced paths; rather, it uses ray tracing for the primary path geometry and superimposes specific 3GPP-inspired statistical effects. A separate `tdl/` directory (currently with no Python code) is intended for standard TDL model generation if needed.

## 4. Output Data Structure

Generated datasets are typically saved as `.mat` files (viewable with `visualise_dataset.py` and analyzable with `plot_data_statistics.py`). These files are expected to contain:
*   `perfect_matrices`: `(n_samples, n_subcarriers, n_symbols)` array of complex, noise-free channel matrices.
*   `received_matrices`: `(n_samples, n_subcarriers, n_symbols)` array of complex, noisy channel matrices.
*   `pilot_mask` (optional): `(n_samples, n_subcarriers, n_symbols)` boolean array indicating pilot locations. (Note: pilot insertion logic itself is not detailed in the core channel classes but assumed to be part of the input signal construction or overlaid for visualization).

## 5. Utilities

*   `plot_data_statistics.py`: Calculates and plots PDP, correlation, magnitude/phase distributions, noise characteristics.
*   `visualise_dataset.py`: Visualizes individual channel matrix heatmaps, with options to overlay pilot masks. It uses `plot_heatmaps.py`.
*   `ieee_plot_style.py`: Provides Matplotlib styling for publication-quality plots.

## 6. Running a Simulation

Simulations are typically run via scripts in `data_generation/ray_tracing/examples/` like `channel_environment_example.py`.
These scripts:
1.  Load a base YAML configuration (e.g., `dataset_a.yaml`).
2.  Optionally load and merge a 3GPP preset (e.g., "UMa").
3.  Procedurally create an `Environment` based on the configuration.
4.  Instantiate the `Channel` class.
5.  Call `channel.apply_channel()` to generate the channel matrix.
6.  Save or visualize the results.

Example:
`python -m data_generation.ray_tracing.examples.channel_environment_example --config data_generation/config/dataset_a.yaml --preset UMa`

## 7. Critique and Areas for Clarification/Improvement

Based on the code review:

**Strengths:**
*   **Modular Design:** Clear separation of concerns (Environment, Channel, Antenna, Scatterer, Building, Window, movement modules).
*   **Detailed Geometric Modeling:** Ray-building intersection, window reflections, scatterer visibility checks are implemented.
*   **Material-Aware Scatterers:** Scatterer reflection coefficients are physically grounded using material properties and Fresnel equations.
*   **Sophisticated UE Movement:** Includes various models with collision detection against buildings and pavement boundaries.
*   **Hybrid Channel Model:** The layering of geometric paths with statistical elements (cluster delays, shadow fading, aggregate scattering) is a strong point for realism.
*   **Configurability:** Extensive use of YAML files for scenario and parameter control.
*   **Visualization Tools:** Good support for visualizing the environment and analyzing generated data.

**Areas for Review or Potential Enhancement:**
*   **Doppler Implementation:** While `Scatterer.calculate_doppler_frequency()` exists, its direct application to induce per-path phase rotation in the `Channel.apply_channel()` loop was not explicit. Doppler effects currently arise primarily from the re-computation of geometry (path lengths) at each step due to `env.advance()`. Explicit per-path Doppler shifts could be more precisely integrated. The `Channel.advance_phases` Numba function seems designed for this but isn't clearly used for scatterer paths.
*   **Environment Advancement Timestep (`dt_effective_block`):** The environment advances in steps of `symbols_per_block * symbol_duration`. This means the channel is piece-wise constant over `symbols_per_block` symbols. This is a valid modeling choice if coherence time is long, but it should be clearly understood and documented. If per-symbol channel evolution is desired, `dt` in `env.advance()` would need to be `symbol_duration`.
*   **Building Reflection Model:** Building surface reflections currently use a simple constant coefficient, unlike the material-aware model for scatterers. This could be enhanced for more realism.
*   **Window Transmission:** The `Window` class focuses on reflection. If transmission *through* windows is a key mechanism, its coefficient calculation (distinct from reflection) and handling in `Environment.get_valid_window_reflections` should be clarified or enhanced (e.g., using material properties for transmission loss). The `wp["coeff"]` used by `Channel.py` needs to be clearly defined as reflection or transmission.
*   **`use_aggregate_scattering` Flag in `Scatterer`:** The purpose of this flag on individual scatterers is unclear, as the main aggregate scattering term in `Channel.py` appears to be global.
*   **Initial Signal for `apply_channel`:** The way dummy input data is handled in `channel_environment_example.py` versus the expected time-domain input for `channel.apply_channel` could be streamlined or clarified if the primary goal is just \( \mathbf{H} \) matrix generation.
*   **TDL Module (`tdl/`):** This is currently a placeholder. If it's to be integrated or used for comparison, its development is pending.
*   **Clarity of `cluster_movement_config` vs. Scatterer's `movement_model`:** Ensure the hierarchy and precedence for scatterer movement (cluster-level flocking config vs. individual scatterer's default model if not flocking) is robustly handled.
*   **`min_far_field` in `add_cluster`:** The `min_far_field` parameter is passed to `add_cluster` and then to `create_scatterers`. Its exact use (e.g., ensuring scatterers are in far-field of each other, or relative to antennas) could be documented more explicitly in the code or comments.
