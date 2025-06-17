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
    def __init__(self, environment, tx_antenna, rx_antenna, snr, center_frequency, symbols_per_block, shadow_fading_std_db=0.0):
        self.environment                      = environment                      # Environment object containing scatterers, antennas, etc.
        self.tx_antenna                       = tx_antenna                       # TX antenna from environment
        self.rx_antenna                       = rx_antenna                       # RX antenna from environment
        self.fc                               = center_frequency                 # System center frequency in Hz
        self.snr                              = snr                              # determines power of noise
        self.shadow_fading_std_db             = shadow_fading_std_db             # Shadow fading standard deviation in dB
        # self.shadow_fading_correlation_time_s = shadow_fading_correlation_time_s # Correlation time for shadow fading in seconds
        self.shadow_fading_correlation_time_s = 0.0
        # Extract antenna properties
        self.tx_gain = self.tx_antenna.gain
        self.rx_gain = self.rx_antenna.gain

        # Initialise random number generator
        self.rng = np.random.default_rng()
                
        # Calculate antenna distance
        tx_pos = np.array(self.tx_antenna.get_pos(), dtype=float)
        rx_pos = np.array(self.rx_antenna.get_pos(), dtype=float)
        
        # Extract heights from antenna positions (z is height)
        self.tx_height = tx_pos[2]
        self.rx_height = rx_pos[2]
        
        # 3GPP parameter additions
        if self.environment.environment_type == "UMa":
            mu_ds_uma_los            = -6.955 - 0.0963 * np.log10(self.fc / 1e9)
            sigma_ds_uma_los         = 0.66
            x_rv_uma_los             = self.rng.normal(0,1)
            log10_ds_uma_los         = mu_ds_uma_los + (sigma_ds_uma_los * x_rv_uma_los)
            self.r_tau               = 2.5
            self.rms_delay_spread_ns = 10**log10_ds_uma_los * 1e9
        elif self.environment.environment_type == "UMi":
            mu_ds_umi_los            = -0.24*np.log10(1+self.fc/1e9) - 7.14
            sigma_ds_umi_los         = 0.38
            x_rv_umi_los             = self.rng.normal(0,1)
            log10_ds_umi_los         = mu_ds_umi_los + (sigma_ds_umi_los * x_rv_umi_los)
            self.r_tau               = 2.2
            self.rms_delay_spread_ns = 10**log10_ds_umi_los * 1e9
        else:
            raise ValueError(f"Environment type {self.environment.environment_type} not supported")
        
        self.symbols_per_block    = symbols_per_block     # Number of OFDM symbols per block
        self.channel_matrix       = None
        self.channel_matrix_noisy = None
        self.noise_matrix         = None
        
        # Extract cluster information from environment
        self.num_cluster = len(environment.clusters)
        # Cluster delay array (populated in prepare_channel)
        self.cluster_delays = None
            
        # Initialise last shadowing value (in dB)
        if self.shadow_fading_std_db > 0.0:
            self.last_shadowing_db = self.rng.normal(loc=0.0, scale=self.shadow_fading_std_db)
        else:
            self.last_shadowing_db = 0.0
                        

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
        height_difference   = abs(tx_pos[2] - rx_pos[2])
        LOS_distance        = math.hypot(horizontal_distance, height_difference)
        return horizontal_distance, height_difference, LOS_distance

    def prepare_channel(self):
        """Prepare channel parameters before signal processing
        It should be noted that the metrics used here are for LOS since there will always be a LOS path.
        """
        self.cluster_delays = np.array([])
        effective_scale = self.r_tau * self.rms_delay_spread_ns
        if self.num_cluster > 0: 
            self.cluster_delays = self.rng.exponential(scale=effective_scale, size=self.num_cluster)
            self.cluster_delays.sort()

    def apply_channel(self, input_data, subcarrier_spacing, num_subcarriers, cyclic_prefix=0):
        # Ensure channel parameters are prepared
        self.prepare_channel()

        g_tx_lin = 10 ** (self.tx_gain / 10)
        g_rx_lin = 10 ** (self.rx_gain / 10)

        # Get TX antenna position 
        tx_pos = np.array(self.tx_antenna.get_pos(), dtype=float)

        # Pre-compute the sub-carrier frequencies for the *whole* frame
        sub_idx           = np.arange(num_subcarriers) - num_subcarriers // 2
        subcarrier_freqs  = self.fc + sub_idx * subcarrier_spacing

        # Calculate the total symbol length including cyclic prefix
        symbol_length = num_subcarriers + cyclic_prefix

        # get the size of the input to ensure they match
        no_ofdm_symbols = int(np.ceil(len(input_data) / symbol_length))
        if len(input_data) % symbol_length != 0:
            padding_needed = symbol_length - (len(input_data) % symbol_length)
            input_data     = np.pad(input_data, (0, padding_needed), mode='constant')
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
        dt_effective_block = self.symbols_per_block * symbol_duration

        # num_scatterers is determined once, assuming it doesn't change during one apply_channel call
        num_scatterers = len(self.environment.scatterers)
        
        # Pre-calculate cluster IDs for all scatterers if they exist
        # This assumes scatterers or their cluster assignments don't change within this apply_channel call
        cluster_ids_all_scatterers = np.empty((0,), dtype=np.int32)
        if num_scatterers > 0:
            cluster_ids_all_scatterers = np.array([s.cluster_id for s in self.environment.scatterers], dtype=np.int32)

        # Pre-calculate alpha for shadow fading if applicable
        current_rx_speed = self.rx_antenna.get_speed()
        shadow_fading_alpha = 1.0 # Default if no shadow fading or no correlation
        if self.shadow_fading_std_db > 0.0:
            if current_rx_speed > 0:
                _shadow_fading_correlation_time_s = 10 / current_rx_speed
            else:
                _shadow_fading_correlation_time_s = 0.0
            
            if _shadow_fading_correlation_time_s > 1e-9:
                shadow_fading_alpha = np.exp(-dt_effective_block / _shadow_fading_correlation_time_s)
            else:
                shadow_fading_alpha = 0.0 # No correlation if correlation time is zero

        # loop through each OFDM symbol
        for i in range(no_ofdm_symbols):
            # Get current RX antenna position for this symbol
            rx_pos             = np.array(self.rx_antenna.get_pos(), dtype=float)
            _, _, LOS_distance = self._calculate_distances(tx_pos, rx_pos)         # Recalculate LOS distance

            # Get current scatterer data from environment for this symbol
            if num_scatterers > 0:
                scatterer_positions, scatterer_velocities, reflection_coeffs, scatterer_speeds, visible_mask = \
                    self.environment.get_scatterer_snapshot(tx_pos, rx_pos) # Use updated rx_pos
                # cluster_ids_all_scatterers is now pre-calculated
            else:
                # Create empty arrays with correct shapes if no scatterers
                scatterer_positions   = np.empty((0, 3), dtype=float)
                reflection_coeffs     = np.empty((0,),   dtype=complex)
                visible_mask          = np.empty((0,),   dtype=bool) 
            
            # Shadow fading update for the current symbol
            current_shadowing_db = 0.0
            if self.shadow_fading_std_db > 0.0:
                # Alpha is pre-calculated
                innovation = self.rng.normal(loc=0.0, scale=self.shadow_fading_std_db)
                current_shadowing_db = shadow_fading_alpha * self.last_shadowing_db + np.sqrt(max(0, 1 - shadow_fading_alpha**2)) * innovation
                self.last_shadowing_db = current_shadowing_db
            
            shadowing_factor_linear = 10**(current_shadowing_db / 20.0)

            # LOS gain for all sub‑carriers at once 
            LOS_gain_vec = self.calculate_path_gain(distance        = LOS_distance,
                                                    frequency       = subcarrier_freqs,
                                                    g_tx_lin        = g_tx_lin,
                                                    g_rx_lin        = g_rx_lin)

            # geometry based terms - they dont change across subcarriers
            vec_tx_sc                   = scatterer_positions - tx_pos              # (N, 3)
            dist_tx_sc                  = np.linalg.norm(vec_tx_sc, axis=1)         # (N,) Total path distance
            vec_sc_rx                   = rx_pos - scatterer_positions              # (N, 3)
            dist_sc_rx                  = np.linalg.norm(vec_sc_rx, axis=1)         # (N,)
            d_n_array                   = dist_tx_sc + dist_sc_rx                   # (N,) Total path distance
            valid_scatterers_dist_check = (dist_tx_sc > 1e-9) & (dist_sc_rx > 1e-9) # Avoid division by zero

            # Combine distance check with visibility mask from environment
            # visible_mask is already the same length as scatterer_positions etc. before any filtering here.
            final_valid_scatterers_mask = valid_scatterers_dist_check & visible_mask
            valid_idx                   = np.where(final_valid_scatterers_mask)[0]  # Get valid indices based on combined criteria

            # Get valid window reflections for this symbol
            self.window_paths = self.environment.get_valid_window_reflections(tx_pos, rx_pos) 

            # loop through each subcarrier
            for j in range(num_subcarriers):
                f_k     = self.fc + (j - num_subcarriers // 2) * subcarrier_spacing # calculate the subcarriers frequency
                lambda_ = 3e8 / f_k                                                 # Wavelength for this subcarrier

                dominant_gain = LOS_gain_vec[j] * shadowing_factor_linear # Apply shadowing to LOS gain
                
                # Window gains
                window_gain = 0+0j
                if hasattr(self, 'window_paths'):
                    for wp in self.window_paths:
                        d_wp        = wp["distance"]
                        fs_amp      = lambda_ / (4 * np.pi * d_wp) if d_wp > 1e-9 else 0 
                        path_gain   = fs_amp * np.sqrt(g_tx_lin * g_rx_lin) * np.exp(-1j * 2 * np.pi * d_wp / lambda_)
                        window_gain += wp["coeff"] * path_gain
                window_gain *= shadowing_factor_linear

                # Scatterer gains
                scatterer_gain = (0 + 0j)
                if num_scatterers > 0:
                    if len(valid_idx) > 0:
                        # Filter arrays to only include valid scatterers for calculations
                        _reflection_coeffs = reflection_coeffs[valid_idx]
                        _d_n_array         = d_n_array[valid_idx]
                        _cluster_ids       = cluster_ids_all_scatterers[valid_idx] 

                        # path-gain for all valid scatterers 
                        path_gain_vec = self.calculate_path_gain(_d_n_array,          
                                                                 f_k,                
                                                                 g_tx_lin,
                                                                 g_rx_lin)             

                        scatterer_contribution_array = path_gain_vec * _reflection_coeffs   # vector

                        # Apply cluster delay-dependent phase shift for frequency selectivity
                        delay_phase_factors = np.zeros(len(self.cluster_delays), dtype=complex)
                        for cluster_id in range(len(self.cluster_delays)):
                            delay_ns = self.cluster_delays[cluster_id]
                            delay_s  = delay_ns * 1e-9
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
                                cluster_sums[cluster_id] = sum_within_cluster
                        
                        # Sum all cluster contributions
                        scatterer_gain = np.sum(cluster_sums) * shadowing_factor_linear

                # Combine dominant (LOS) and explicit scatterer paths
                total_dominant_paths_gain = dominant_gain + scatterer_gain + window_gain

                # Add aggregate scattering term H_ij if enabled
                H_ij = (0 + 0j)
                # Calculate Variance based on paper's formula
                if LOS_distance > 1e-6: 
                    scatter_variance_linear = (10**(-3.24)) * ((f_k / 1e9)**(-2)) * (LOS_distance**(-3))
                else:
                    scatter_variance_linear = 0

                # Generate complex Gaussian using the calculated linear variance
                if scatter_variance_linear > 1e-12:
                    std_dev_H = np.sqrt(scatter_variance_linear / 2.0)
                    H_ij      = self.rng.normal(0, std_dev_H) + 1j * self.rng.normal(0, std_dev_H)

                # Final gain is sum of dominant paths and aggregate scattering
                final_gain                = total_dominant_paths_gain + H_ij
                self.channel_matrix[j, i] = final_gain

            # Update scatterer positions AND UE position through the environment
            self.environment.advance(dt_effective_block)
                                    
        # input data in time domain so convert to freq so we can multiply instead of convolve
        input_data    = np.fft.fft(input_data, axis=0)
        output_signal = np.multiply(self.channel_matrix, input_data)
        # convert everything back to time domain to be demultiplexed
        output_signal = np.fft.ifft(output_signal, axis=0)

        # add noise to the output signal
        noise                     = self.generate_noise(output_signal)
        self.noise_matrix         = noise
        output_signal             = output_signal + noise
        self.channel_matrix_noisy = self.channel_matrix + self.noise_matrix # for debugging

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
        snr_linear   = 10 ** (self.snr / 10)
        noise_power  = signal_power / snr_linear
        noise_std    = np.sqrt(noise_power / 2)
        noise        = noise_std * (self.rng.normal(size=signal.shape) + 1j * self.rng.normal(size=signal.shape))
        return noise

    def get_channel_matrix(self):
        return self.channel_matrix
    
    def get_channel_matrix_noisy(self):
        return self.channel_matrix_noisy
    
    def get_noise_matrix(self):
        return self.noise_matrix

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

    def calculate_path_gain(self, distance, frequency, g_tx_lin, g_rx_lin):
        """
        Calculate path gain using Friis model
        Returns linear gain (not loss) value
        
        Parameters:
        distance (float/array): Distance in meters. Can be scalar or array.
        frequency (float/array): Frequency in Hz. Can be scalar or array.
        g_tx_lin (float): Linear transmitter antenna gain
        g_rx_lin (float): Linear receiver antenna gain

        Note: distance and frequency can differ in rank/shape and will be properly broadcast.
        """
        distance   = np.asarray(distance)
        frequency  = np.asarray(frequency)
        wavelength = 3e8 / frequency
        phase      = np.exp(-1j * 2 * np.pi * distance / wavelength)
        fs_amp     = wavelength / (4 * np.pi * distance)

        return fs_amp * math.sqrt(g_tx_lin * g_rx_lin) * phase
