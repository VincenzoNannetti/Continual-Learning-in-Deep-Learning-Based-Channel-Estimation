"""
Filename: ./data_generation/ray_tracing/classes/channel.py
Author: Vincenzo Nannetti
Date: 09/05/2025 (Refactored Date)
Description: Channel Class for the ray tracing model.
             Ground bounce to be added in a later revision.
"""

import math
import numpy as np
from numba import njit, prange

@njit(parallel=True, fastmath=True)
def advance_phases(phases, f_D, dt):
    for n in prange(phases.size):
        phases[n] = (phases[n] + 2*np.pi*f_D[n]*dt) % (2*np.pi)


class Channel:
    def __init__(self, environment, tx_antenna, rx_antenna, snr, center_frequency, pathloss_model="friis",
                 o2i_loss=0, reflection_amplitude=None, cluster_density=0.5, cluster_radius=20, 
                 los_probability_indoor=0.7, rms_delay_spread_ns=100, shadow_fading_std_db=4.0, 
                 rng=None, use_random_walk=True):
        self.environment          = environment                                                                     # Environment object containing scatterers, antennas, etc.
        self.tx_antenna           = tx_antenna                                                                      # TX antenna from environment
        self.rx_antenna           = rx_antenna                                                                      # RX antenna from environment
        self.fc                   = center_frequency                                                                # System center frequency in Hz
        self.snr                  = snr                                                                             # determines power of noise
        self.environment_type     = environment.scatterers[0].environment_type if environment.scatterers else "UMa" # environment type from environment object
        self.pathloss_model       = pathloss_model                                                                  # pathloss model to use (friis or 3gpp)
        self.o2i_loss             = o2i_loss                                                                        # outdoor-to-indoor penetration loss in dB
        self.reflection_amplitude = reflection_amplitude                                                            # reflection amplitude range from config
        self.cluster_density      = cluster_density                                                                 # density of scatterers in clusters
        self.cluster_radius       = cluster_radius                                                                  # radius of clusters
        
        # Extract antenna properties
        self.tx_gain = self.tx_antenna.gain
        self.rx_gain = self.rx_antenna.gain
        
        # Calculate antenna distance
        tx_pos = np.array(self.tx_antenna.get_pos(), dtype=float)
        rx_pos = np.array(self.rx_antenna.get_pos(), dtype=float)
        
        # Extract heights from antenna positions (z is height)
        self.tx_height = tx_pos[2]
        self.rx_height = rx_pos[2]
        
        # 3GPP parameter additions
        self.rms_delay_spread_ns  = rms_delay_spread_ns   # RMS delay spread in nanoseconds
        self.shadow_fading_std_db = shadow_fading_std_db  # Shadow fading standard deviation in dB
        
        # Set r_tau and per-cluster shadowing based on environment
        if self.environment_type == "UMa":
            self.r_tau = 2.5
            self.sigma_zeta_db = 3.0
        elif self.environment_type == "UMi":
            self.r_tau = 2.3
            self.sigma_zeta_db = 3.0
        elif self.environment_type == "RMa":
            self.r_tau = 3.0
            self.sigma_zeta_db = 3.0
        elif self.environment_type.startswith("InF"):
            self.r_tau = 2.1
            self.sigma_zeta_db = 4.0
        elif self.environment_type == "InH":
            self.r_tau = 2.2
            self.sigma_zeta_db = 3.5
        else:
            self.r_tau = 2.5
            self.sigma_zeta_db = 3.5

        self.channel_matrix       = None
        self.channel_matrix_noisy = None
        self.noise_matrix         = None

        # Use aggregate scattering flag from first scatterer (if any)
        self.use_aggregate_scattering = environment.scatterers[0].use_aggregate_scattering if environment.scatterers else True
        
        # Extract cluster information from environment
        self.num_cluster = len(environment.clusters)
        
        # Cluster delay and power arrays (populated in prepare_channel)
        self.cluster_delays = None
        self.cluster_powers = None
        
        # Phase tracking for Doppler evolution (populated in apply_channel)
        self.scatterer_phases = None

        # Initialise random number generator
        if rng is None:
            self.rng = np.random.default_rng()
        elif isinstance(rng, (np.random.RandomState, np.random.Generator)):
            self.rng = rng
        else:
            # If it's not a RandomState or Generator instance, create a new one with the provided seed
            try:
                self.rng = np.random.default_rng(rng)
            except:
                self.rng = np.random.default_rng()
                
        self.los_probability_indoor = los_probability_indoor
        
        # Track the last shadow fading value for each link
        self.previous_shadow_fading = 0.0
        self.last_shadow_fading_position = np.array([0.0, 0.0])
        self.shadow_fading_correlation_distance = self._get_shadow_fading_correlation_distance() # Set based on environment

        # Default average building heights (h in Table 7.4.1-1) - simplified
        self.avg_building_heights = {
            "RMa": 5.0, # Default for RMa
            "UMa": 20.0, # Example, actual UMa h_E is complex
            "UMi": 10.0  # Example, actual UMi h_E is 1m for d_BP calc
        }

        # Set random walk in environment
        self.environment.use_random_walk = use_random_walk

    def _calculate_distances(self, tx_pos, rx_pos):
        """
        Helper method to calculate distances between transmitter and receiver.
        
        Args:
            tx_pos: Transmitter position array [x, y, z]
            rx_pos: Receiver position array [x, y, z]
            
        Returns:
            tuple: (horizontal_distance, height_difference, LOS_distance)
        """
        horizontal_distance = np.sqrt((tx_pos[0] - rx_pos[0])**2 + (tx_pos[1] - rx_pos[1])**2)
        height_difference = abs(tx_pos[2] - rx_pos[2])
        LOS_distance = math.hypot(horizontal_distance, height_difference)
        return horizontal_distance, height_difference, LOS_distance

    def _get_shadow_fading_correlation_distance(self):
        # Placeholder for scenario-specific correlation distances from Table 7.5-6
        # These would ideally be looked up more robustly
        env = self.environment_type
        if env == "UMa": return 50.0 # Example for UMa NLOS, simplified
        if env == "UMi": return 10.0 # Example for UMi LOS
        if env == "RMa": return 120.0 # Example for RMa NLOS
        if env.startswith("InF"): return 10.0
        if env == "InH": return 6.0 # Example for InH NLOS
        return 10.0 # Default

    def prepare_channel(self):
        """Prepare channel parameters before signal processing"""
        # Generate cluster delays and powers using 3GPP power-law model
        if self.num_cluster > 0:
            self.cluster_delays, self.cluster_powers = self.draw_cluster_delays_and_powers(self.num_cluster)
        else:
            self.cluster_delays = np.array([])
            self.cluster_powers = np.array([])

        # Initialise phase tracking for all scatterers
        self.scatterer_phases = np.zeros(len(self.environment.scatterers))

    def apply_channel(self, input_data, subcarrier_spacing, num_subcarriers, cyclic_prefix=0):
        # Ensure channel parameters are prepared
        self.prepare_channel()

        g_tx_lin = 10 ** (self.tx_gain / 10)
        g_rx_lin = 10 ** (self.rx_gain / 10)

        # Get antenna positions and calculate distances
        tx_pos = np.array(self.tx_antenna.get_pos(), dtype=float)
        rx_pos = np.array(self.rx_antenna.get_pos(), dtype=float)
        _, _, LOS_distance = self._calculate_distances(tx_pos, rx_pos)

        # Pre-compute the sub-carrier frequencies for the *whole* frame
        sub_idx           = np.arange(num_subcarriers) - num_subcarriers // 2
        subcarrier_freqs  = self.fc + sub_idx * subcarrier_spacing

        # Calculate the total symbol length including cyclic prefix
        symbol_length = num_subcarriers + cyclic_prefix

        # get the size of the input to ensure they match
        no_ofdm_symbols = int(np.ceil(len(input_data) / symbol_length))
        if len(input_data) % symbol_length != 0:
            padding_needed = symbol_length - (len(input_data) % symbol_length)
            input_data = np.pad(input_data, (0, padding_needed), mode='constant')
        output_length = input_data.shape[0]
        
        # reshape the input accounting for cyclic prefix
        input_data = input_data.reshape((symbol_length, no_ofdm_symbols))
        
        # If we have a cyclic prefix, remove it before processing
        if cyclic_prefix > 0:
            # Extract only the actual data part (after cyclic prefix)
            input_data = input_data[cyclic_prefix:, :]

        # define empty channel matrix
        self.channel_matrix = np.zeros((num_subcarriers, no_ofdm_symbols), dtype=complex)

        # Calculate the OFDM symbol duration (time step for updates)
        symbol_duration = 1.0 / subcarrier_spacing

        # --- Get scatterer data from environment --- #
        num_scatterers = len(self.environment.scatterers)
        if num_scatterers > 0:
            # Get a snapshot of scatterer data from the environment
            scatterer_positions, scatterer_velocities, reflection_coeffs, scatterer_speeds = self.environment.get_scatterer_snapshot()
            
            # Precompute cluster IDs for all scatterers
            cluster_ids = np.array([s.cluster_id for s in self.environment.scatterers], dtype=np.int32)
            
            # Initialise phase tracking for Doppler if not already done
            if self.scatterer_phases is None or len(self.scatterer_phases) != num_scatterers:
                self.scatterer_phases = np.zeros(num_scatterers)
        else:
            # Create empty arrays with correct shapes if no scatterers
            scatterer_positions   = np.empty((0, 3), dtype=float)
            scatterer_velocities  = np.empty((0, 3), dtype=float)
            reflection_coeffs     = np.empty((0,), dtype=complex)
            scatterer_speeds      = np.empty((0,), dtype=float)
            cluster_ids           = np.empty((0,), dtype=np.int32)
            self.scatterer_phases = np.empty((0,), dtype=float)

        # loop through each subcarrier of each symbol
        for i in range(no_ofdm_symbols):
            # Get antenna gains in linear scale
            for j in range(num_subcarriers):
                # calculate the subcarriers frequency
                f_k = self.fc + (j - num_subcarriers // 2) * subcarrier_spacing
                lambda_ = 3e8 / f_k # Wavelength for this subcarrier

                # Determine if path is LOS based on environment
                # 1) compute horizontal distance:
                d2d = np.sqrt((tx_pos[0] - rx_pos[0])**2 + (tx_pos[1] - rx_pos[1])**2)  # Distance in x-y plane

                # 2) evaluate TR38.901 LOS probability:
                if self.environment_type == "UMi":
                    p_los = 1.0 if d2d <= 18 else 18/d2d + np.exp(-d2d/36)*(1 - 18/d2d)
                elif self.environment_type == "UMa":
                    p_los = 1.0 if d2d <= 18 else 18/d2d + np.exp(-d2d/63)*(1 - 18/d2d)
                elif self.environment_type == "RMa":
                    p_los = 1.0 if d2d <= 10 else np.exp(-(d2d-10)/1000)
                else:
                    # for indoor still keep a flat value or parameterise it:
                    p_los = self.los_probability_indoor

                # 3) sample the realisation:
                # Draw one Bernoulli LOS flag for this *symbol* (keeps channel matrix stable)
                is_los_symbol = self.rng.random() < p_los

                # LOS gain for all sub‑carriers at once (returns length Nsc array)
                LOS_gain_vec = self.calculate_path_gain(
                        distance        = LOS_distance,
                        frequency       = subcarrier_freqs,
                        g_tx_lin        = g_tx_lin,
                        g_rx_lin        = g_rx_lin,
                        is_los          = is_los_symbol,
                        rx_position     = rx_pos)

                dominant_gain = LOS_gain_vec[j]

                # # --- Ground specular ---
                # if self.include_ground_bounce:
                #     # image method: reflect Rx in z=0 plane
                #     img_rx = np.array([rx_pos[0], rx_pos[1], -rx_pos[2]])
                #     d_tx_img = np.linalg.norm(img_rx - tx_pos)
                #     d_img_rx = np.linalg.norm(rx_pos - img_rx)
                #     d_gr = d_tx_img + d_img_rx
                #     Γ = fresnel_coefficient(…)
                #     ground_gain = Γ * self.calculate_path_gain(
                #         distance=d_gr,
                #         frequency=f_k,
                #         g_tx_lin=g_tx_lin,
                #         g_rx_lin=g_rx_lin,
                #         is_los=False)
                #     total_dominant_paths_gain += ground_gain


                scatterer_gain = (0 + 0j) # Initialise gain from explicit scatterers

                if num_scatterers > 0:
                    # --- Vectorised Scatterer Calculations --- #

                    # 1. Compute distances (Tx->Scatterer and Scatterer->Rx)
                    vec_tx_sc = scatterer_positions - tx_pos # (N, 3)
                    dist_tx_sc = np.linalg.norm(vec_tx_sc, axis=1) # (N,) Total path distance
                    vec_sc_rx = rx_pos - scatterer_positions # (N, 3)
                    dist_sc_rx = np.linalg.norm(vec_sc_rx, axis=1) # (N,)
                    d_n_array = dist_tx_sc + dist_sc_rx # (N,) Total path distance

                    # Avoid division by zero for coincident points
                    valid_scatterers = (dist_tx_sc > 1e-9) & (dist_sc_rx > 1e-9)
                    valid_idx = np.where(valid_scatterers)[0]  # Get valid indices once
                    
                    if len(valid_idx) > 0:
                        # Filter arrays to only include valid scatterers for calculations
                        _scatterer_positions  = scatterer_positions[valid_idx]
                        _scatterer_velocities = scatterer_velocities[valid_idx]
                        _reflection_coeffs    = reflection_coeffs[valid_idx]
                        _scatterer_speeds     = scatterer_speeds[valid_idx]
                        _vec_tx_sc            = vec_tx_sc[valid_idx]
                        _dist_tx_sc           = dist_tx_sc[valid_idx]
                        _vec_sc_rx            = vec_sc_rx[valid_idx]
                        _dist_sc_rx           = dist_sc_rx[valid_idx]
                        _d_n_array            = d_n_array[valid_idx]
                        _cluster_ids          = cluster_ids[valid_idx]
                        _scatterer_phases     = self.scatterer_phases[valid_idx]
                        _num_valid_scatterers = len(valid_idx)

                        # 2. path-gain for all valid scatterers (vector)
                        scatterer_is_los = np.zeros(_num_valid_scatterers, dtype=bool)  # reflections are NLOS
                        path_gain_vec = self.calculate_path_gain(
                                _d_n_array,          # vector of distances
                                f_k,                 # scalar frequency (broadcasts)
                                g_tx_lin,
                                g_rx_lin,
                                scatterer_is_los,
                                rx_pos)

                        scatterer_contribution_array = path_gain_vec * _reflection_coeffs   # vector

                        # 3) Doppler – fully vectorised
                        dir_tx_sc = _vec_tx_sc / _dist_tx_sc[:, None]        # unit vectors Tx→Sc
                        dir_sc_rx = _vec_sc_rx / _dist_sc_rx[:, None]        # unit vectors Sc→Rx

                        valid_speed        = _scatterer_speeds > 1e-9
                        cos_theta_tx_sc    = np.zeros_like(_scatterer_speeds)
                        cos_theta_sc_rx    = np.zeros_like(_scatterer_speeds)

                        if np.any(valid_speed):
                            cos_theta_tx_sc[valid_speed] = (
                                np.sum(_scatterer_velocities[valid_speed] * dir_tx_sc[valid_speed], axis=1) / _scatterer_speeds[valid_speed])

                            cos_theta_sc_rx[valid_speed] = (
                                np.sum(_scatterer_velocities[valid_speed] * dir_sc_rx[valid_speed], axis=1) / _scatterer_speeds[valid_speed])

                        # Doppler frequency for every scatterer (Hz)
                        f_Dn_array = (_scatterer_speeds / lambda_) * (cos_theta_tx_sc + cos_theta_sc_rx)
                        
                        # 4. apply Doppler phase to the contributions
                        advance_phases(_scatterer_phases, f_Dn_array, symbol_duration)

                        # 5. apply Doppler phase to the contributions
                        scatterer_contribution_array *= np.exp(1j * _scatterer_phases)

                        # keep master phase array in sync when we filtered invalid scatterers
                        self.scatterer_phases[valid_idx] = _scatterer_phases
                        
                        # 7. Apply cluster delay-dependent phase shift for frequency selectivity
                        # Precompute delay phase factors for all clusters
                        delay_phase_factors = np.zeros(len(self.cluster_delays), dtype=complex)
                        for cluster_id in range(len(self.cluster_delays)):
                            delay_ns = self.cluster_delays[cluster_id]
                            delay_s = delay_ns * 1e-9
                            delay_phase_factors[cluster_id] = np.exp(-1j * 2 * np.pi * f_k * delay_s)
                        
                        # Initialise array for cluster contributions
                        cluster_sums = np.zeros(len(self.cluster_delays), dtype=complex)
                        
                        # Group contributions by cluster and apply delays
                        for cluster_id in range(len(self.cluster_delays)):
                            mask = _cluster_ids == cluster_id
                            if np.any(mask):
                                # Sum all scatterer contributions within this cluster
                                sum_within_cluster = np.sum(scatterer_contribution_array[mask] * delay_phase_factors[cluster_id])
                                
                                # Apply cluster power weighting from the power-law model
                                if len(self.cluster_powers) > cluster_id:
                                    # Scale by the cluster power weight (3GPP TR 38.901, Eq 7.5-5)
                                    cluster_power = self.cluster_powers[cluster_id]
                                    # If the sum of contributions is non-zero, normalize by the number of 
                                    # scatterers in the cluster to maintain relative amplitude relationships
                                    if np.abs(sum_within_cluster) > 1e-10 and np.sum(mask) > 0:
                                        # Normalize by scatterer count, then apply cluster power 
                                        cluster_sums[cluster_id] = sum_within_cluster * np.sqrt(cluster_power)
                                    else:
                                        cluster_sums[cluster_id] = 0.0
                                else:
                                    # Fallback if cluster powers aren't available 
                                    cluster_sums[cluster_id] = sum_within_cluster
                        
                        # Sum all cluster contributions
                        scatterer_gain = np.sum(cluster_sums)

                # Combine dominant (LOS) and explicit scatterer paths
                total_dominant_paths_gain = dominant_gain + scatterer_gain

                # Add aggregate scattering term H_ij if enabled
                H_ij = (0 + 0j)
                if self.use_aggregate_scattering:
                    # Calculate Variance based on paper's formula
                    if LOS_distance > 1e-6: # Avoid division by zero if Tx/Rx are coincident
                        scatter_variance_linear = (10**(-3.24)) * ((f_k / 1e9)**(-2)) * (LOS_distance**(-3))
                    else:
                        scatter_variance_linear = 0

                    # Generate complex Gaussian using the calculated linear variance
                    if scatter_variance_linear > 1e-12:
                        std_dev_H = np.sqrt(scatter_variance_linear / 2.0)
                        H_ij = self.rng.normal(0, std_dev_H) + 1j * self.rng.normal(0, std_dev_H)

                # Final gain is sum of dominant paths and aggregate scattering
                final_gain = total_dominant_paths_gain + H_ij
                
                self.channel_matrix[j, i] = final_gain

            # Update scatterer positions through the environment
            self.environment.advance(symbol_duration)
            
            # Update scatterer snapshot for the next symbol
            if num_scatterers > 0:
                scatterer_positions, scatterer_velocities, reflection_coeffs, scatterer_speeds = self.environment.get_scatterer_snapshot()
            
        # input data in time domain so convert to freq so we can multiply instead of convolve
        input_data    = np.fft.fft(input_data, axis=0)
        output_signal = np.multiply(self.channel_matrix, input_data)
        # convert everything back to time domain to be demultiplexed
        output_signal = np.fft.ifft(output_signal, axis=0)

        # add noise to the output signal
        noise = self.generate_noise(output_signal)
        self.noise_matrix = noise
        output_signal = output_signal + noise
        self.channel_matrix_noisy = output_signal # for debugging

        output_signal = np.reshape(output_signal, (output_length,-1))
        return output_signal

    def generate_noise(self, signal):
        """
        Generate noise for the signal based on specified SNR.
        
        This method uses the average power of the channel matrix |H|^2 to determine
        the noise power, setting SNR = E{|H·X|^2}/N_0. This is different from a 
        per-subcarrier Es/N_0 definition.
        
        Parameters:
        signal: Input signal/channel matrix
        
        Returns:
        Complex noise matrix of the same shape as the input signal
        """
        # generate noise for the signal
        signal_power = np.mean(np.abs(signal) ** 2)
        snr_linear = 10 ** (self.snr / 10)
        noise_power = signal_power / snr_linear
        noise_std = np.sqrt(noise_power / 2)
        noise = noise_std * (self.rng.normal(size=signal.shape) + 1j * self.rng.normal(size=signal.shape))
        return noise

    def get_channel_matrix(self):
        return self.channel_matrix
    def get_channel_matrix_noisy(self):
        return self.channel_matrix_noisy
    def get_noise_matrix(self):
        return self.noise_matrix

    def draw_cluster_delays_and_powers(self, num_clusters):
        """
        Generate cluster delays and powers using 3GPP TR 38.901 power-law model.
        
        Args:
            num_clusters: Number of clusters to generate
            
        Returns:
            tuple: (delays_ns, powers) arrays
        """
        # 1. Exponential delays (mu = rms_ds_ns * r_tau as per 3GPP Step 5, Eq 7.5-1 implies scale = DS * r_tau)
        delays_ns = self.rng.exponential(scale=self.rms_delay_spread_ns * self.r_tau, size=num_clusters)
        
        # Sort delays (as per 3GPP specification)
        delays_ns.sort()
        
        # 2. Per-cluster shadowing (Z_n in dB) - STEP 5 OF CLAUSE 7.5 (EQ 7.5-1)
        # Log-normal shadowing with standard deviation sigma_zeta
        shadow_fading_db = self.rng.normal(0, self.sigma_zeta_db, size=num_clusters)
        
        # Convert from dB to linear scale: 10^(-Z_n/10) per 3GPP TR 38.901
        zeta_lin = 10**(-shadow_fading_db / 10)
        
        # 3. Exponential power law (Corrected exponent as per 3GPP Eq 7.5-5)
        if self.r_tau == 0: 
            power_exponent = -delays_ns / self.rms_delay_spread_ns 
        else:
            power_exponent = -(self.r_tau - 1) * delays_ns / (self.r_tau * self.rms_delay_spread_ns)
        
        # Calculate raw power with shadow fading applied
        # P_n ~ exp(-τ_n(r_τ-1)/(r_τDS)) × 10^(-Z_n/10)
        raw_power = np.exp(power_exponent) * zeta_lin
        
        # 4. Normalise so total power sums to 1
        powers = raw_power / raw_power.sum()
        
        return delays_ns, powers

    def advance_phase(self, phase_prev, f_d, symbol_period):
        """
        Update the phase based on Doppler shift for lightweight time evolution.
        
        Args:
            phase_prev: Previous phase value in radians
            f_d: Doppler frequency in Hz
            symbol_period: OFDM symbol duration in seconds
            
        Returns:
            float: Updated phase in radians
        """
        return (phase_prev + 2 * np.pi * f_d * symbol_period) % (2 * np.pi)

    def calculate_path_gain(self, distance, frequency, g_tx_lin, g_rx_lin, is_los=True, rx_position=None):
        """
        Calculate path gain using either Friis or 3GPP model
        Returns linear gain (not loss) value
        
        Parameters:
        distance (float/array): Distance in meters. Can be scalar or array.
        frequency (float/array): Frequency in Hz. Can be scalar or array.
        g_tx_lin (float): Linear transmitter antenna gain
        g_rx_lin (float): Linear receiver antenna gain
        is_los (bool/array): Whether the path is line-of-sight. Can be scalar or array.
        rx_position (array): Receiver position for shadow fading correlation
        
        Note: distance and frequency can differ in rank/shape and will be properly broadcast.
        """
        distance   = np.asarray(distance)
        frequency  = np.asarray(frequency)
        wavelength = 3e8 / frequency
        phase      = np.exp(-1j * 2 * np.pi * distance / wavelength)
        
        if self.pathloss_model.lower() == "friis":
            fs_amp  = wavelength / (4 * np.pi * distance)
            return fs_amp * math.sqrt(g_tx_lin * g_rx_lin) * phase
        else:
            # 3GPP TR 38.901 models
            pl_linear = self.calculate_3gpp_pathloss(distance, frequency / 1e6, is_los, rx_position)
            
            # Apply O2I penetration loss if applicable and environment is indoor
            if self.environment_type.startswith("In") and self.o2i_loss > 0:
                pl_linear *= 10**(-self.o2i_loss/10)
                
            # Combine with antenna gains and phase
            gain = np.sqrt(g_tx_lin * g_rx_lin) * phase * np.sqrt(pl_linear)
            return gain

    def calculate_3gpp_pathloss(self, d_3d, fc_MHz, is_los=True, rx_position=None):
        """
        Calculate pathloss according to 3GPP TR 38.901 models (vectorised is_los handling)
        d_3d: 3D distance in meters (scalar or array)
        fc_MHz: carrier frequency in MHz (scalar or array)
        is_los: whether path is LOS (scalar bool or array of bool)
        rx_position: current receiver position (for shadow fading correlation)
        Returns pathloss in linear scale (gain factor, not dB loss)
        """
        # Convert inputs to arrays for vector operations
        d_3d = np.maximum(np.asarray(d_3d), 1.0)
        fc_GHz = (np.asarray(fc_MHz) / 1000.0)
        fc_Hz = np.asarray(fc_MHz) * 1e6
        c_light = 3e8

        # Heights
        h_BS = self.tx_antenna.get_pos()[2]
        h_UT = self.rx_antenna.get_pos()[2]
        
        # 2D distance - calculate using the Pythagorean theorem
        # d_2d = sqrt(d_3d² - (h_BS - h_UT)²)
        height_diff_squared = (h_BS - h_UT)**2
        d_2d = np.sqrt(np.maximum(d_3d**2 - height_diff_squared, 1.0))

        # Precompute LOS/NLOS pathloss in dB for each scenario
        if self.environment_type == "UMa":
            # Implement UMa h_E calculation as per TR 38.901 V18.0.0, Table 7.4.1-1, Note 1
            # fc_GHz (array of subcarrier frequencies in GHz) is already available for pathloss terms.
            # h_UT is self.rx_height
            # d_2d is available

            # For C(d_2D, h_UT) calculation, use the system's center frequency (fc) in GHz
            # as per 3GPP TR 38.901, which specifies f_c (centre frequency) for this parameter.
            _center_fc_GHz_for_C = self.fc / 1e9 # System center frequency in GHz

            c_threshold_num = 185 * (h_UT / 1.5)**0.4 * (_center_fc_GHz_for_C / 5.0)**-0.25
            
            # Vectorized C_val calculation: d_2d can be an array
            # Create a mask for valid values to avoid invalid power computation
            valid_mask = d_2d > c_threshold_num
            
            # Initialize C_val with zeros
            C_val = np.zeros_like(d_2d)
            
            # Only compute the power for valid elements (where d_2d > c_threshold_num)
            if np.any(valid_mask):
                C_val[valid_mask] = ((d_2d[valid_mask] / c_threshold_num) - 1.0)**1.7
            
            prob_h_E_is_1m = 1.0 / (1.0 + C_val) # prob_h_E_is_1m can now be an array

            # Vectorized h_E determination
            # Ensure prob_h_E_is_1m is treated as an array for consistent shape handling
            _prob_h_E_is_1m_arr = np.atleast_1d(prob_h_E_is_1m)
            # Generate random samples with the same shape as _prob_h_E_is_1m_arr
            random_samples = self.rng.random(size=_prob_h_E_is_1m_arr.shape)
            
            # Initialize h_E as an array with default value 1.0
            h_E = np.ones_like(_prob_h_E_is_1m_arr, dtype=float) 
            
            # Create a mask for elements where h_E should be chosen from the distribution
            needs_dist_choice_mask = random_samples >= _prob_h_E_is_1m_arr
            
            if np.any(needs_dist_choice_mask): # Proceed only if any elements need alternative h_E
                max_h_E_val = math.floor(h_UT - 1.5) # h_UT is scalar
                possible_h_E_values = [h_val for h_val in range(12, int(max_h_E_val) + 1, 3)]
                
                if possible_h_E_values:
                    # Number of choices needed from the distribution
                    num_choices_needed = np.sum(needs_dist_choice_mask)
                    # Draw random choices
                    chosen_h_values = self.rng.choice(possible_h_E_values, size=num_choices_needed)
                    # Assign chosen values to the corresponding elements in h_E
                    h_E[needs_dist_choice_mask] = chosen_h_values
                # If possible_h_E_values is empty, h_E remains 1.0 for those masked elements,
                # which aligns with the original fallback logic.
            
            # h_E is now an array if d_2d was an array, or a 1-element array if d_2d was scalar.
            # This will broadcast correctly in subsequent calculations.
            
            # Original h_E = 1.0 line is now replaced by the logic above
            h_BS_eff = h_BS - h_E
            h_UT_eff = h_UT - h_E

            d_BP = 4 * h_BS_eff * h_UT_eff * (fc_Hz) / c_light
            d_BP = np.maximum(d_BP, 1.0)

            pl_los1 = 28.0 + 22.0 * np.log10(d_3d) + 20.0 * np.log10(fc_GHz)
            pl_los2 = 28.0 + 40.0 * np.log10(d_3d) + 20.0 * np.log10(fc_GHz) \
                      - 9.0 * np.log10(d_BP**2 + (h_BS - h_UT)**2)
            pl_los_dB = np.where(d_2d < d_BP, pl_los1, pl_los2)

            pl_nlos = 13.54 + 39.08 * np.log10(d_3d) + 20.0 * np.log10(fc_GHz) - 0.6 * (h_UT - 1.5)
            pl_nlos_dB = np.maximum(pl_nlos, pl_los_dB)

        elif self.environment_type == "UMi":
            h_E = 1.0
            h_BS_eff = h_BS - h_E
            h_UT_eff = h_UT - h_E
            d_BP = 4 * h_BS_eff * h_UT_eff * (fc_Hz) / c_light
            d_BP = np.maximum(d_BP, 1.0)

            pl_los1 = 32.4 + 21.0 * np.log10(d_3d) + 20.0 * np.log10(fc_GHz)
            pl_los2 = 32.4 + 40.0 * np.log10(d_3d) + 20.0 * np.log10(fc_GHz) \
                      - 9.8 * np.log10(d_BP**2 + (h_BS - h_UT)**2)
            pl_los_dB = np.where(d_2d < d_BP, pl_los1, pl_los2)

            pl_nlos = 22.4 + 35.3 * np.log10(d_3d) + 21.3 * np.log10(fc_GHz) - 0.3 * (h_UT - 1.5)
            pl_nlos_dB = np.maximum(pl_nlos, pl_los_dB)

        elif self.environment_type == "RMa":
            h_build = max(self.avg_building_heights.get("RMa", 5.0), 1.0)
            d_BP = (2 * np.pi * h_BS * h_UT * fc_Hz) / c_light
            d_BP = np.maximum(d_BP, 1.0)
            d_eval = np.maximum(d_3d, 10.0)

            term1 = np.minimum(0.03 * (h_build**1.72), 10.0)
            term2 = np.minimum(0.044 * (h_build**1.72), 14.77)
            term3 = 0.002 * np.log10(h_build) * d_eval

            pl1 = 20.0 * np.log10(4 * np.pi * d_eval * fc_Hz / c_light) + term1 * np.log10(d_eval) - term2 + term3
            pl_bp = 20.0 * np.log10(4 * np.pi * d_BP * fc_Hz / c_light) + term1 * np.log10(d_BP) - term2 + 0.002 * np.log10(h_build) * d_BP
            pl2 = pl_bp + 40.0 * np.log10(d_eval / d_BP)
            pl_los_dB = np.where(d_3d < 10.0, pl1, np.where(d_3d < d_BP, pl1, pl2))

            W = 20.0
            h_avg = max(self.avg_building_heights.get("RMa", 5.0), 1.0)
            log_hBS = np.log10(max(h_BS,1.0))
            pl_nlos = 161.04 - 7.1*np.log10(W) + 7.5*np.log10(h_avg) \
                       - (24.37 - 3.7*(h_avg/h_BS)**2)*log_hBS \
                       + (43.42 - 3.1*log_hBS)*(np.log10(d_3d)-3.0) \
                       + 20*np.log10(fc_GHz) - (3.2*(np.log10(11.75*np.maximum(h_UT,1.0)))**2 - 4.97)
            pl_nlos_dB = np.maximum(pl_nlos, pl_los_dB)

        elif self.environment_type.startswith("InF"):
            pl_los_dB = 31.84 + 21.50*np.log10(d_3d) + 19.00*np.log10(fc_GHz)
            if self.environment_type == "InF-SL":
                pl_nlos = 33.0 + 25.5*np.log10(d_3d) + 20.0*np.log10(fc_GHz)
            elif self.environment_type == "InF-DL":
                sl = 33.0 + 25.5*np.log10(d_3d) + 20.0*np.log10(fc_GHz)
                dl = 18.6 + 35.7*np.log10(d_3d) + 20.0*np.log10(fc_GHz)
                pl_nlos = np.maximum(sl, dl)
            elif self.environment_type == "InF-SH":
                pl_nlos = 32.4 + 23.0*np.log10(d_3d) + 20.0*np.log10(fc_GHz)
            elif self.environment_type == "InF-DH":
                pl_nlos = 33.63 + 21.9*np.log10(d_3d) + 20.0*np.log10(fc_GHz)
            else:
                pl_nlos = 33.0 + 25.5*np.log10(d_3d) + 20.0*np.log10(fc_GHz)
            pl_nlos_dB = np.maximum(pl_nlos, pl_los_dB)

        else:  # InH and default
            pl_los_dB  = 32.4 + 17.3*np.log10(d_3d) + 20.0*np.log10(fc_GHz)
            pl_nlos    = 17.3 + 38.3*np.log10(d_3d) + 24.9*np.log10(fc_GHz)
            pl_nlos_dB = np.maximum(pl_nlos, pl_los_dB)

        # Select per-element pathloss
        pl_dB = np.where(is_los, pl_los_dB, pl_nlos_dB)

        # Apply shadow fading (linear scale) and return
        pl_db_shadow = self.pathloss_with_shadow(pl_dB, is_los, rx_position)
        return 10**(-pl_db_shadow/10)

    def pathloss_with_shadow(self, pathloss_db, is_los, rx_position=None):
        """
        Apply shadow fading to the pathloss value based on 3GPP standards.
        Handles scalar or vector is_los and applies spatial correlation only for scalar main links.
        """
        # Determine std dev based on environment
        if self.environment_type == "UMa": sf_los, sf_nlos = 4.0, 6.0
        elif self.environment_type == "UMi": sf_los, sf_nlos = 3.0, 4.0
        elif self.environment_type == "RMa": sf_los, sf_nlos = 4.0, 6.0
        elif self.environment_type == "InH": sf_los, sf_nlos = 3.0, 4.0
        elif self.environment_type.startswith("InF"): sf_los, sf_nlos = 4.3, 5.7
        else: sf_los = sf_nlos = self.shadow_fading_std_db

        is_array = isinstance(is_los, np.ndarray)
        if is_array:
            sf_std = np.where(is_los, sf_los, sf_nlos)
            # independent sample per element, no spatial correlation
            sf = self.rng.normal(0, sf_std, size=pathloss_db.shape)
            return pathloss_db + sf

        # Scalar branch: correlated fading
        sf_std = sf_los if is_los else sf_nlos
        d_cor = self.shadow_fading_correlation_distance
        current_sf = 0.0
        if rx_position is not None and d_cor > 0:
            rx_pos_2d = rx_position[:2]  # Get x,y coordinates
            last_pos_2d = self.last_shadow_fading_position[:2]
            delta = np.linalg.norm(rx_pos_2d - last_pos_2d)  # Euclidean distance in x-y plane
            if delta == 0 and self.previous_shadow_fading != 0.0:
                current_sf = self.previous_shadow_fading
            else:
                rho = np.exp(-delta / d_cor)
                indep = self.rng.normal(0, sf_std)
                current_sf = rho * self.previous_shadow_fading + np.sqrt(1 - rho**2) * indep
            self.previous_shadow_fading = current_sf
            self.last_shadow_fading_position = np.array([rx_position[0], rx_position[1]])
        else:
            current_sf = self.rng.normal(0, sf_std)
            self.previous_shadow_fading = current_sf
            if rx_position is not None:
                self.last_shadow_fading_position = np.array([rx_position[0], rx_position[1]])

        return pathloss_db + current_sf
 