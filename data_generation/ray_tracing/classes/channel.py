"""
Filename: ./data_generation/ray_tracing/classes/channel.py
Author: Vincenzo Nannetti
Date: 09/05/2025 (Refactored Date)
Description: Channel Class for the ray tracing model.
"""

import math
import numpy as np

from .antenna import Antenna
from .scatterer import Scatterer
from .vector_utils import horizontal_distance, distance3d

class Channel:
    def __init__(self, antenna_distance, tx_gain, rx_gain, snr, system, num_cluster=5, speed=0.01,
                 use_aggregate_scattering=False, environment_type="UMa", pathloss_model="friis",
                 o2i_loss=0, reflection_amplitude=None, cluster_density=0.5, cluster_radius=20, 
                 los_probability_indoor=0.7, rms_delay_spread_ns=100, shadow_fading_std_db=4.0, 
                 tx_height=25.0, rx_height=1.5, cluster_height_range=None, rng=None):
        """
        Initialize a Channel object for ray tracing simulation.
        
        Args:
            antenna_distance: Horizontal distance between TX and RX antennas in meters
            tx_gain: Gain of TX antenna in dBi
            rx_gain: Gain of RX antenna in dBi
            snr: Signal-to-Noise Ratio in dB
            system: System object (optional)
            num_cluster: Number of scattering cluster locations
            speed: Movement speed of scatterers in m/s
            use_aggregate_scattering: Whether to use H_{ij} term for aggregate scattering
            environment_type: Type of environment (UMa, UMi, RMa, InF, InH)
            pathloss_model: Pathloss model to use (friis or 3gpp)
            o2i_loss: Outdoor-to-indoor penetration loss in dB
            reflection_amplitude: Range of reflection coefficients as [min, max]
            cluster_density: Density of scatterers in clusters (0.0-1.0)
            cluster_radius: Horizontal radius of clusters in meters
            los_probability_indoor: Probability of LOS in indoor environments
            rms_delay_spread_ns: RMS delay spread in nanoseconds
            shadow_fading_std_db: Shadow fading standard deviation in dB
            tx_height: Height of transmitter antenna in meters
            rx_height: Height of receiver antenna in meters
            cluster_height_range: Vertical range of clusters in meters (defaults to cluster_radius/2)
            rng: Random number generator instance
        """
        self.antenna_distance     = antenna_distance      # Horizontal distance between antennas
        self.tx_gain              = tx_gain               # Gain of TX antenna in dBi
        self.rx_gain              = rx_gain               # Gain of RX antenna in dBi
        self.snr                  = snr                   # Signal-to-Noise Ratio in dB
        self.num_cluster          = num_cluster           # Number of scattering cluster locations
        self.environment_type     = environment_type      # Type of environment
        self.pathloss_model       = pathloss_model        # Pathloss model to use (friis or 3gpp)
        self.o2i_loss             = o2i_loss              # Outdoor-to-indoor penetration loss in dB
        self.reflection_amplitude = reflection_amplitude  # Reflection amplitude range from config
        self.cluster_density      = cluster_density       # Density of scatterers in clusters
        self.cluster_radius       = cluster_radius        # Horizontal radius of clusters
        self.tx_height            = tx_height             # Height of transmitter antenna
        self.rx_height            = rx_height             # Height of receiver antenna
        
        # Set vertical range of clusters (default to half of horizontal radius if not specified)
        self.cluster_height_range = cluster_height_range if cluster_height_range is not None else cluster_radius / 2
        
        # 3GPP parameter additions
        self.rms_delay_spread_ns  = rms_delay_spread_ns   # RMS delay spread in nanoseconds
        self.shadow_fading_std_db = shadow_fading_std_db  # Shadow fading standard deviation in dB
        
        # Set r_tau and per-cluster shadowing based on environment
        if environment_type == "UMa":
            self.r_tau = 2.5
            self.sigma_zeta_db = 3.0
        elif environment_type == "UMi":
            self.r_tau = 2.3
            self.sigma_zeta_db = 3.0
        elif environment_type == "RMa":
            self.r_tau = 3.0
            self.sigma_zeta_db = 3.0
        elif environment_type.startswith("InF"):
            self.r_tau = 2.1
            self.sigma_zeta_db = 4.0
        elif environment_type == "InH":
            self.r_tau = 2.2
            self.sigma_zeta_db = 3.5
        else:
            self.r_tau = 2.5
            self.sigma_zeta_db = 3.5

        self.tx_antenna           = None
        self.rx_antenna           = None
        self.channel_matrix       = None
        self.channel_matrix_noisy = None
        self.noise_matrix         = None

        self.speed         = speed                    # speed of the scatterers
        self.scatterers    = []                       # array of scatterer objects
        self.subscatterers = max(1, int(10 * cluster_density))  # number of scatterers around a point, scaled by density
        self.main_cluster_positions = []              # array holding the cluster positions
        
        # Cluster delay and power arrays (populated in create_environment)
        self.cluster_delays = None
        self.cluster_powers = None
        
        # Phase tracking for Doppler evolution (populated in apply_channel)
        self.scatterer_phases = None

        # Parameters for Aggregate Scattering (H_ij term)
        self.use_aggregate_scattering   = use_aggregate_scattering

        # Initialize random number generator
        if rng is None:
            self.rng = np.random.RandomState()
        elif isinstance(rng, np.random.RandomState):
            self.rng = rng
        else:
            # If it's not a RandomState instance, create a new one with the provided seed
            try:
                self.rng = np.random.RandomState(rng)
            except:
                self.rng = np.random.RandomState()
                
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

    def create_environment(self):
        """
        Create the propagation environment with antennas and scatterers.
        """
        # Extract tx_height and rx_height from class attributes
        tx_height = getattr(self, 'tx_height', 25.0)  # Default to 25m if not specified
        rx_height = getattr(self, 'rx_height', 1.5)   # Default to 1.5m if not specified
        
        # Create the Antennas using 3D positions [x, y, z] with [East, North, Height] convention
        # For backward compatibility with 2D models, place antennas along x-axis (East direction)
        self.tx_antenna = Antenna("TX", self.tx_gain, [0, 0, tx_height])
        self.rx_antenna = Antenna("RX", self.rx_gain, [self.antenna_distance, 0, rx_height])
        rx_pos = self.rx_antenna.get_pos()
        
        # Use configurable cluster radius (horizontal spread)
        max_radius = self.cluster_radius
        
        # Set vertical height range for scatterers (default to 1/2 of horizontal radius)
        cluster_height_range = getattr(self, 'cluster_height_range', max_radius / 2)
        
        # Use scaled number of scatterers based on cluster density
        num_subscatterers_per_cluster = self.subscatterers

        # --- Clear previous scatterers ---
        self.scatterers = []
        self.main_cluster_positions = []
        # ---------------------------------
        
        # Generate cluster delays and powers using 3GPP power-law model
        if self.num_cluster > 0:
            self.cluster_delays, self.cluster_powers = self.draw_cluster_delays_and_powers(self.num_cluster)
        else:
            self.cluster_delays = np.array([])
            self.cluster_powers = np.array([])

        # Function to create the positions of the clusters of scatterers
        def create_scatterers(cluster, no_subscatterers, rx_position, cluster_rad, min_distance=1.0):
            main_pos = None  # Define main_pos outside try block
            try:
                if no_subscatterers > 0 or self.num_cluster > 0:
                    while True:
                        # Generate cluster center position in 3D space
                        # X coordinate: Uniform between 5m and (receiver_distance - 5m)
                        x_cluster = self.rng.uniform(5, rx_position[0] - 5)
                        # Y coordinate: Uniform across wide area perpendicular to TX-RX path
                        y_cluster = self.rng.uniform(-150, 150)
                        # Z coordinate: Uniform in vertical space (limited range)
                        z_cluster = self.rng.uniform(0, max(tx_height, rx_height) * 1.5)
                        
                        # 3D cluster center position
                        main_pos = np.array([x_cluster, y_cluster, z_cluster])
                        
                        # Each cluster should be at least min_distance away from another one (3D distance)
                        if all(np.linalg.norm(main_pos - np.array(existing_pos)) > min_distance
                               for existing_pos in self.main_cluster_positions):
                            self.main_cluster_positions.append(main_pos.tolist())
                            break
            except Exception as e:
                print(f"Error creating cluster center: {e}")
                # For now, let's assume num_cluster > 0 if we enter the loop below
                if main_pos is None:  # Basic safety check
                    print("Warning: Could not determine main_pos for cluster.")
                    return []  # Return empty list if center couldn't be placed

            # Array of subscatterers
            subscatterers = []
            
            # Get cluster power from the power-law model (to scale reflection amplitude)
            cluster_power_scale = 1.0
            if cluster < len(self.cluster_powers):
                cluster_power_scale = np.sqrt(self.cluster_powers[cluster] * self.num_cluster)
            
            # For the number of scatterers per cluster find its position and instantiate the object
            for j in range(no_subscatterers):
                while True:
                    # Generate 3D random position within the cluster sphere
                    theta = self.rng.uniform(0, 2 * np.pi)  # Azimuth angle
                    phi = self.rng.uniform(0, np.pi)        # Zenith angle
                    r = self.rng.uniform(0, cluster_rad)    # Radius
                    
                    # Convert spherical to Cartesian coordinates
                    dx = r * np.sin(phi) * np.cos(theta)
                    dy = r * np.sin(phi) * np.sin(theta)
                    dz = r * np.cos(phi) * (cluster_height_range / cluster_rad)  # Scale vertical spread
                    
                    # Ensure main_pos is valid before using it
                    if main_pos is None:
                        print(f"Error: main_pos is None for cluster {cluster}, subscatterer {j}. Skipping.")
                        break
                    
                    # Calculate new position
                    new_pos = main_pos + np.array([dx, dy, dz])
                    
                    # Ensure x coordinate is not negative (below transmitter)
                    new_pos[0] = max(0, new_pos[0])
                    
                    # Round for numerical stability
                    new_pos = np.round(new_pos, 2)

                    # Create reflection parameters dictionary if amplitude is defined
                    reflection_params = None
                    if self.reflection_amplitude is not None:
                        # Scale the reflection amplitude by the cluster power
                        refl_amplitude = self.reflection_amplitude.copy() if isinstance(self.reflection_amplitude, list) else [0.3, 0.9]
                        # Scale the mean amplitude by the cluster power while keeping the range proportional
                        midpoint = (refl_amplitude[0] + refl_amplitude[1]) / 2
                        half_range = (refl_amplitude[1] - refl_amplitude[0]) / 2
                        new_midpoint = midpoint * cluster_power_scale
                        # Keep the scaled reflection amplitude within valid range [0.1, 0.95]
                        new_midpoint = max(0.1 + half_range, min(0.95 - half_range, new_midpoint))
                        scaled_refl_amplitude = [new_midpoint - half_range, new_midpoint + half_range]
                        
                        reflection_params = {'reflection_amplitude': scaled_refl_amplitude}

                    # Create scatterer with 3D position
                    new_subscatterer = Scatterer(
                        f"{cluster}-{j}", 
                        new_pos, 
                        self.speed,
                        self.use_aggregate_scattering, 
                        self.environment_type,
                        reflection_params, 
                        self.rng
                    )

                    # Collision check within the same cluster (3D distance)
                    if not any(np.linalg.norm(np.array(new_subscatterer.get_pos()) - np.array(s.get_pos())) < 0.01
                               for s in subscatterers):
                        subscatterers.append(new_subscatterer)  # Append here after check passes
                        break
                    
            # Return populated array of scatterers
            return subscatterers

        # If we want clusters, add the scatterers
        for i in range(self.num_cluster):
            # Append the clusters to a general scatterer array for processing
            num_scat_this_cluster = num_subscatterers_per_cluster
            self.scatterers.extend(create_scatterers(i, num_scat_this_cluster, rx_pos, max_radius))
        
        # Initialize phase tracking for all scatterers
        self.scatterer_phases = np.zeros(len(self.scatterers))

    def pathloss_with_shadow(self, pathloss_db, is_los, rx_position=None):
        """
        Apply shadow fading to the pathloss value based on 3GPP standards.
        
        Args:
            pathloss_db: Pathloss value in dB (positive value) without shadow fading.
            is_los: Whether the path is line-of-sight.
            rx_position: Current receiver position (for spatial correlation).
            
        Returns:
            float: Pathloss with shadow fading in dB.
        """
        # Determine shadow fading standard deviation (sf_std in dB) based on environment and LOS condition
        # These values should ideally come from Table 7.4.1-1 or an equivalent detailed configuration.
        # For now, using the previously defined logic based on self.shadow_fading_std_db for general case.
        if self.environment_type == "UMa":
            sf_std = 4.0 if is_los else 6.0 # Values from 3GPP Table 7.4.1-1 for UMa LOS/NLOS
        elif self.environment_type == "UMi":
            sf_std = 3.0 if is_los else 4.0 # Values from 3GPP Table 7.4.1-1 for UMi LOS/NLOS (Street Canyon)
                                          # Note: Original code had 4.0 / 7.8. Table 7.4.1-1 has 3.0/4.0 for UMi-Street Canyon.
                                          # Using values from table. User had 7.8 for UMi NLOS, which is closer to O2I shadow fading.
                                          # For consistency with Table 7.4.1-1 for UMi-Street Canyon outdoor, using 3.0/4.0.
        elif self.environment_type == "RMa":
            sf_std = 4.0 if is_los else 6.0 # Values from 3GPP Table 7.4.1-1 for RMa LOS/NLOS (was 8.0 in user code for NLOS)
        elif self.environment_type == "InH":
            # For InH-Office (Table 7.4.1-1)
            sf_std = 3.0 if is_los else 4.0 # LOS/NLOS std dev (was 3.0 / 8.0 in user code)
        elif self.environment_type.startswith("InF"):
            # For InF (Table 7.4.1-1) - these are examples, specific InF sub-scenarios have different sf_std.
            # Using a common value for demonstration as the sub-scenario (SL, DL, SH, DH) isn't passed here.
            # User code had 4.0 / 6.0. Let's check table more closely.
            # InF-SL LOS: 4.3, NLOS: 5.7; InF-DL LOS: 4.3, NLOS: 7.2 etc.
            # This needs to be more specific if precise InF sub-models are used.
            # Using a general placeholder based on previous code for now.
            sf_std = 4.3 if is_los else 5.7 # Example for InF-SL, rough average
        else:
            sf_std = self.shadow_fading_std_db # Fallback to general config

        current_shadow_fading = 0.0
        d_cor = self.shadow_fading_correlation_distance # d_cor from Table 7.5-6 (via helper)

        if rx_position is not None and d_cor > 0:
            # Ensure both positions are 2D arrays for consistent calculation
            if not isinstance(self.last_shadow_fading_position, np.ndarray):
                self.last_shadow_fading_position = np.array([0.0, 0.0])
            
            # Only use the x-coordinate (horizontal position) for shadow fading correlation
            rx_x_position = rx_position[0] if isinstance(rx_position, np.ndarray) else rx_position
            last_x_position = self.last_shadow_fading_position[0]
            
            # Calculate delta_x based on x-coordinate only
            delta_x = abs(rx_x_position - last_x_position)
            
            if delta_x == 0 and self.previous_shadow_fading != 0.0: # If no movement, keep previous SF
                current_shadow_fading = self.previous_shadow_fading
            else:
                rho = np.exp(-delta_x / d_cor)
                # Generate new shadow fading component: N(0, sf_std^2 * (1-rho^2))
                # Simplified: new_component_std = sf_std * np.sqrt(1 - rho**2)
                # current_shadow_fading = rho * self.previous_shadow_fading + self.rng.normal(0.0, new_component_std)
                # Correct form for correlated Gaussian: SF_new = ρ * SF_prev + sqrt(1-ρ²) * N(0, σ_SF^2)
                # Where N(0, σ_SF^2) is an independent sample from the shadow fading distribution.
                independent_sf_sample = self.rng.normal(0.0, sf_std)
                current_shadow_fading = rho * self.previous_shadow_fading + np.sqrt(1 - rho**2) * independent_sf_sample
            
            self.previous_shadow_fading = current_shadow_fading
            # Store only the x-coordinate for future correlation calculations
            self.last_shadow_fading_position = np.array([rx_x_position, 0.0])
        else:
            # Generate new uncorrelated shadow fading if no position tracking or first time
            current_shadow_fading = self.rng.normal(0.0, sf_std)
            self.previous_shadow_fading = current_shadow_fading # Store for next potential correlated step
            if rx_position is not None:
                # Store only the x-coordinate for future correlation calculations
                rx_x_position = rx_position[0] if isinstance(rx_position, np.ndarray) else rx_position
                self.last_shadow_fading_position = np.array([rx_x_position, 0.0])
        
        # Apply shadow fading to pathloss
        return pathloss_db + current_shadow_fading # pathloss_db is the mean pathloss WITHOUT shadow fading
    
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

    def calculate_3gpp_pathloss(self, d_3d, fc_MHz, is_los=True, rx_position=None):
        """
        Calculate pathloss according to 3GPP TR 38.901 models
        d_3d: Actually just the horizontal distance in meters (2D only)
        fc_MHz: carrier frequency in MHz
        is_los: whether path is line-of-sight
        rx_position: current receiver position (for shadow fading correlation)
        Returns pathloss in linear scale (gain factor, not dB loss)
        """
        # We treat everything in 2D, so d2d is just the horizontal distance
        d_2d = d_3d / 100.0   # [m], normalized for visualization

        # Convert frequency to GHz for formula input, and Hz for physics constants  
        fc_GHz = fc_MHz / 1000.0
        fc_Hz = fc_MHz * 1e6
        c_light = 3e8 # Speed of light m/s
        
        # Use d2d everywhere as the distance metric
        d_2d = max(d_2d, 1.0) # Ensure d_2d is at least 1m for stability in logs

        pl_dB = 0  # Default value
        
        if self.environment_type == "UMa":
            # Fixed breakpoint distance for 2D model
            d_BP_prime = 4 * 1.0 * 1.0 * fc_Hz / c_light  # both effective heights = 1 m
            d_BP_prime = max(d_BP_prime, 1.0) # Ensure d_BP_prime is at least 1m

            if is_los:
                if d_2d < d_BP_prime:
                    pl_dB = 28.0 + 22.0 * np.log10(d_2d) + 20.0 * np.log10(fc_GHz)
                else:
                    pl_dB = 28.0 + 40.0 * np.log10(d_2d) + 20.0 * np.log10(fc_GHz) 
            else: # UMa NLOS
                # NLOS model simplified for 2D
                pl_nlos_formula_dB = 13.54 + 39.08 * np.log10(d_2d) + 20.0 * np.log10(fc_GHz)
                
                # Calculate LOS pathloss for the max comparison
                d_BP_prime_los = 4 * 1.0 * 1.0 * fc_Hz / c_light  # both effective heights = 1 m
                d_BP_prime_los = max(d_BP_prime_los, 1.0)
                if d_2d < d_BP_prime_los:
                    pl_los_dB = 28.0 + 22.0 * np.log10(d_2d) + 20.0 * np.log10(fc_GHz)
                else:
                    pl_los_dB = 28.0 + 40.0 * np.log10(d_2d) + 20.0 * np.log10(fc_GHz) 
                
                pl_dB = max(pl_nlos_formula_dB, pl_los_dB) # As per Note 3, Table 7.4.1-1
                
        elif self.environment_type == "UMi":
            # Fixed breakpoint distance for 2D model
            d_BP_prime = 4 * 1.0 * 1.0 * fc_Hz / c_light  # both effective heights = 1 m
            d_BP_prime = max(d_BP_prime, 1.0)

            if is_los:
                if d_2d < d_BP_prime:
                    pl_dB = 32.4 + 21.0 * np.log10(d_2d) + 20.0 * np.log10(fc_GHz)
                else:
                    pl_dB = 32.4 + 40.0 * np.log10(d_2d) + 20.0 * np.log10(fc_GHz) 
            else: # UMi NLOS
                # NLOS model simplified for 2D
                pl_nlos_formula_dB = 22.4 + 35.3 * np.log10(d_2d) + 21.3 * np.log10(fc_GHz)

                # Calculate LOS pathloss for the max comparison
                d_BP_prime_los = 4 * 1.0 * 1.0 * fc_Hz / c_light  # both effective heights = 1 m
                d_BP_prime_los = max(d_BP_prime_los, 1.0)
                if d_2d < d_BP_prime_los:
                    pl_los_dB = 32.4 + 21.0 * np.log10(d_2d) + 20.0 * np.log10(fc_GHz)
                else:
                    pl_los_dB = 32.4 + 40.0 * np.log10(d_2d) + 20.0 * np.log10(fc_GHz) - 9.8

                pl_dB = max(pl_nlos_formula_dB, pl_los_dB) # As per Note 4, Table 7.4.1-1
                                
        elif self.environment_type == "RMa":
            h_building = self.avg_building_heights.get("RMa", 5.0) # Avg building height for RMa
            # Ensure h_building is not zero or negative for log10
            h_building = max(h_building, 1.0)

            if is_los:
                # RMa LOS simplified for 2D
                # Fixed breakpoint distance
                d_BP_RMa = 2 * np.pi * fc_Hz / c_light
                d_BP_RMa = max(d_BP_RMa, 1.0) # ensure positive for log

                # Apply 10m minimum distance for formula application
                d_eval = max(d_2d, 10.0)

                term_h_building_1 = min(0.03 * (h_building**1.72), 10.0)
                term_h_building_2 = min(0.044 * (h_building**1.72), 14.77)
                term_h_building_3 = 0.002 * np.log10(h_building) * d_eval

                pl1_rma = 20.0 * np.log10(4 * np.pi * d_eval * fc_Hz / c_light) + \
                          term_h_building_1 * np.log10(d_eval) - term_h_building_2 + term_h_building_3

                if d_2d < 10.0: # For d_2d < 10m, use formula for d_2d = 10m
                    pl_dB = pl1_rma
                elif d_2d < d_BP_RMa: # 10m <= d_2d < d_BP
                    pl_dB = pl1_rma
                else: # d_2d >= d_BP
                    pl_bp_calc = 20.0 * np.log10(4 * np.pi * d_BP_RMa * fc_Hz / c_light) + \
                                 min(0.03 * (h_building**1.72), 10.0) * np.log10(d_BP_RMa) - \
                                 min(0.044 * (h_building**1.72), 14.77) + \
                                 0.002 * np.log10(h_building) * d_BP_RMa
                    pl_dB = pl_bp_calc + 40.0 * np.log10(d_eval / d_BP_RMa)

            else: # RMa NLOS
                # NLOS model simplified for 2D
                W_rma = 20.0 # Default avg street width
                h_avg_building_rma = self.avg_building_heights.get("RMa", 5.0)
                h_avg_building_rma = max(h_avg_building_rma, 1.0)

                log10_d_2d = np.log10(d_2d)
                log10_fc_GHz = np.log10(fc_GHz)

                # Simplified NLOS model for 2D only
                pl_nlos_formula_dB = 161.04 - 7.1 * np.log10(W_rma) + 7.5 * np.log10(h_avg_building_rma) + \
                                     43.42 * (log10_d_2d - 3.0) + 20.0 * log10_fc_GHz
                
                # Calculate LOS pathloss for the max comparison using the 2D model
                d_BP_RMa_los = 2 * np.pi * fc_Hz / c_light
                d_BP_RMa_los = max(d_BP_RMa_los, 1.0)
                d_eval_los = max(d_2d, 10.0)
                term_h_b1_los = min(0.03 * (h_building**1.72), 10.0)
                term_h_b2_los = min(0.044 * (h_building**1.72), 14.77)
                pl1_rma_los = 20.0 * np.log10(4 * np.pi * d_eval_los * fc_Hz / c_light) + \
                              term_h_b1_los * np.log10(d_eval_los) - term_h_b2_los + 0.002 * np.log10(h_building) * d_eval_los
                if d_2d < 10.0:
                    pl_los_dB = pl1_rma_los
                elif d_2d < d_BP_RMa_los:
                    pl_los_dB = pl1_rma_los
                else:
                    pl_bp_calc_los = 20.0 * np.log10(4 * np.pi * d_BP_RMa_los * fc_Hz / c_light) + \
                                     min(0.03 * (h_building**1.72), 10.0) * np.log10(d_BP_RMa_los) - \
                                     min(0.044 * (h_building**1.72), 14.77) + \
                                     0.002 * np.log10(h_building) * d_BP_RMa_los
                    pl_los_dB = pl_bp_calc_los + 40.0 * np.log10(d_eval_los / d_BP_RMa_los)

                pl_dB = max(pl_nlos_formula_dB, pl_los_dB)
                                
        elif self.environment_type.startswith("InF"):
            # InF models simplified for 2D
            pl_los_inf_dB = 31.84 + 21.50 * np.log10(d_2d) + 19.00 * np.log10(fc_GHz)
            if is_los:
                pl_dB = pl_los_inf_dB
            else:
                pl_nlos_inf_dB = 0
                if self.environment_type == "InF-SL":
                    pl_nlos_inf_dB = 33.0 + 25.5 * np.log10(d_2d) + 20.0 * np.log10(fc_GHz)
                    pl_dB = max(pl_los_inf_dB, pl_nlos_inf_dB)
                elif self.environment_type == "InF-DL":
                    pl_inf_sl_nlos = 33.0 + 25.5 * np.log10(d_2d) + 20.0 * np.log10(fc_GHz) # InF-SL NLOS formula
                    pl_nlos_formula_inf_dl = 18.6 + 35.7 * np.log10(d_2d) + 20.0 * np.log10(fc_GHz)
                    pl_dB = max(pl_los_inf_dB, max(pl_inf_sl_nlos, pl_nlos_formula_inf_dl)) 
                elif self.environment_type == "InF-SH":
                    pl_nlos_inf_dB = 32.4 + 23.0 * np.log10(d_2d) + 20.0 * np.log10(fc_GHz)
                    pl_dB = max(pl_los_inf_dB, pl_nlos_inf_dB)
                elif self.environment_type == "InF-DH":
                    pl_nlos_inf_dB = 33.63 + 21.9 * np.log10(d_2d) + 20.0 * np.log10(fc_GHz)
                    pl_dB = max(pl_los_inf_dB, pl_nlos_inf_dB)
                else: # Should not happen if presets are correctly named
                    pl_dB = pl_los_inf_dB # Fallback to LOS if specific InF NLOS subtype not matched
                
        elif self.environment_type == "InH": # Indoor Hotspot - Office
            # InH models simplified for 2D
            pl_los_inh_dB = 32.4 + 17.3 * np.log10(d_2d) + 20.0 * np.log10(fc_GHz)
            if is_los:
                pl_dB = pl_los_inh_dB
            else:
                pl_nlos_formula_inh_dB = 17.3 + 38.3 * np.log10(d_2d) + 24.9 * np.log10(fc_GHz)
                pl_dB = max(pl_los_inh_dB, pl_nlos_formula_inh_dB)
        
        # Apply shadow fading
        pl_dB_with_shadow = self.pathloss_with_shadow(pl_dB, is_los, rx_position)
        
        # Return linear scale path gain (not loss)
        return 10**(-pl_dB_with_shadow/10)

    def calculate_path_gain(self, distance, frequency, is_los=True, rx_position=None, is_3d_distance=False):
        """
        Calculate path gain using either Friis or 3GPP model
        Returns linear gain (not loss) value
        
        Parameters:
        distance (float): Distance in meters (either 3D or horizontal 2D distance)
        frequency (float): Frequency in Hz
        is_los (bool): Whether the path is line-of-sight
        rx_position (array): Receiver position for shadow fading correlation
        is_3d_distance (bool): Whether distance is 3D (True) or horizontal 2D (False)
        """
        # Calculate wavelength
        wavelength = 3e8 / frequency
        
        # Get antenna gains in linear scale
        g_tx_lin = 10**(self.tx_gain / 10)
        g_rx_lin = 10**(self.rx_gain / 10)
        
        # Use distance for phase calculation
        # For 3GPP models, distance should be 3D distance for phase, but may be 2D for pathloss
        phase = np.exp(-1j * 2 * np.pi * distance / wavelength)
        
        if self.pathloss_model.lower() == "friis":
            # Original system model using Friis equation with shadow fading
            # MODIFIED: Apply a distance normalization factor to avoid excessive pathloss
            # This makes the visualization more meaningful while keeping the frequency selectivity
            normalized_distance = distance / 100.0  # Scale down the distance for better visualization
            friis_pl_db = 20 * np.log10(4 * np.pi * normalized_distance / wavelength)
            
            # Apply shadow fading if using 3GPP parameters
            if hasattr(self, 'shadow_fading_std_db') and self.shadow_fading_std_db > 0:
                if rx_position is None:
                    rx_position = np.array(self.rx_antenna.get_pos())
                friis_pl_db = self.pathloss_with_shadow(friis_pl_db, is_los, rx_position)
                
            friis_gain = 10**(-friis_pl_db/20) * np.sqrt(g_tx_lin * g_rx_lin) * phase
            return friis_gain
        else:
            # 3GPP TR 38.901 models
            # The 3GPP pathloss models use horizontal distance (d2d)
            # If we were given a 3D distance, we need to convert it to a horizontal distance
            if is_3d_distance:
                # This is a simplified approximation to extract horizontal distance
                # Ideally we'd have both points, but for now assume height difference is tx_height - rx_height
                height_diff = abs(self.tx_height - self.rx_height)
                # Pythagoras to get horizontal distance from 3D distance and height
                horizontal_dist = np.sqrt(max(0, distance**2 - height_diff**2))
                d2d_distance = horizontal_dist
            else:
                # Distance already represents horizontal distance
                d2d_distance = distance
                
            pl_linear = self.calculate_3gpp_pathloss(d2d_distance, frequency / 1e6, is_los, rx_position)
            
            # Apply O2I penetration loss if applicable and environment is indoor
            if self.environment_type.startswith("In") and self.o2i_loss > 0:
                pl_linear *= 10**(-self.o2i_loss/10)
                
            # Combine with antenna gains and phase
            gain = np.sqrt(g_tx_lin * g_rx_lin) * phase * pl_linear
            return gain

    def apply_channel(self, input_data, frequency, subcarrier_spacing, num_subcarriers, cyclic_prefix=0):
        # ensure environment created
        self.create_environment()

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

        # Get antenna positions as 3D vectors
        tx_pos = np.array(self.tx_antenna.get_pos(), dtype=float)
        rx_pos = np.array(self.rx_antenna.get_pos(), dtype=float)

        # Calculate both horizontal (2D) and total (3D) distances for path loss and phase calculations
        d2d_distance = horizontal_distance(tx_pos, rx_pos)  # Horizontal distance for pathloss models
        d3d_distance = distance3d(tx_pos, rx_pos)          # 3D distance for phase calculation
        LOS_distance = d3d_distance  # Use 3D distance for LOS calculations

        # Calculate the OFDM symbol duration (time step for updates)
        # Ignoring CP length contribution as it's currently 0
        symbol_duration = 1.0 / subcarrier_spacing

        # --- Pre-extract scatterer data into NumPy arrays --- #
        num_scatterers = len(self.scatterers)
        if num_scatterers > 0:
            # Extract 3D position and velocity vectors from scatterers
            scatterer_positions = np.array([s.pos for s in self.scatterers])  # (N, 3) - 3D positions
            scatterer_velocities = np.array([s.velocity_vector for s in self.scatterers])  # (N, 3) - 3D velocities
            # Extract complex reflection coefficients
            reflection_coeffs = np.array([s.get_reflection_coeff() for s in self.scatterers], dtype=complex)  # (N,)
            # Extract scalar speeds
            scatterer_speeds = np.array([s.speed for s in self.scatterers])  # (N,)
            
            # Initialize phase tracking for Doppler if not already done
            if self.scatterer_phases is None or len(self.scatterer_phases) != num_scatterers:
                self.scatterer_phases = np.zeros(num_scatterers)
        else:
            # Create empty arrays with correct shapes if no scatterers (now 3D vectors)
            scatterer_positions = np.empty((0, 3), dtype=float)
            scatterer_velocities = np.empty((0, 3), dtype=float)
            reflection_coeffs = np.empty((0,), dtype=complex)
            scatterer_speeds = np.empty((0,), dtype=float)
            self.scatterer_phases = np.empty((0,), dtype=float)
        # ----------------------------------------------------- #

        # loop through each subcarrier of each symbol
        for i in range(no_ofdm_symbols):
            # Calculate the current time relative to the start of this channel realization
            current_time = i * symbol_duration

            # --------------------------------

            for j in range(num_subcarriers):
                # calculate the subcarriers frequency
                f_k = frequency + (j - num_subcarriers // 2) * subcarrier_spacing
                lambda_ = 3e8 / f_k # Wavelength for this subcarrier

                # Determine if path is LOS based on environment
                # 1) compute horizontal distance in 2D:
                d2d = abs(tx_pos[0] - rx_pos[0])  # Just the x-coordinate difference

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
                is_los = (self.rng.random() < p_los)

                # Calculate path gain using our flexible method
                LOS_gain      = self.calculate_path_gain(LOS_distance, f_k, is_los, rx_pos)
                dominant_gain = LOS_gain

                scatterer_gain = (0 + 0j) # Initialise gain from explicit scatterers

                if num_scatterers > 0:
                    # --- Vectorised Scatterer Calculations --- #

                    # 1. Compute distances (Tx->Scatterer and Scatterer->Rx) - Horizontal distances only
                    # Only use the x-coordinate (position) in calculating the distance, ignore the y-coordinate (height)
                    vec_tx_sc_x = scatterer_positions[:, 0] - tx_pos[0]  # x-coordinates only
                    dist_tx_sc = np.abs(vec_tx_sc_x)  # (N,) Horizontal distance from Tx to scatterer
                    vec_sc_rx_x = rx_pos[0] - scatterer_positions[:, 0]  # x-coordinates only
                    dist_sc_rx = np.abs(vec_sc_rx_x)  # (N,) Horizontal distance from scatterer to Rx
                    d_n_array = dist_tx_sc + dist_sc_rx  # (N,) Total path distance

                    # Avoid division by zero for coincident points
                    valid_scatterers = (dist_tx_sc > 1e-9) & (dist_sc_rx > 1e-9)
                    if not np.all(valid_scatterers):
                        # Filter arrays to only include valid scatterers for calculations
                        # This is important to avoid NaNs/Infs propagating
                        _scatterer_positions  = scatterer_positions[valid_scatterers]
                        _scatterer_velocities = scatterer_velocities[valid_scatterers]
                        _reflection_coeffs    = reflection_coeffs[valid_scatterers]
                        _scatterer_speeds     = scatterer_speeds[valid_scatterers]
                        _dist_tx_sc           = dist_tx_sc[valid_scatterers]
                        _dist_sc_rx           = dist_sc_rx[valid_scatterers]
                        _d_n_array            = d_n_array[valid_scatterers]
                        _num_valid_scatterers = _scatterer_positions.shape[0]
                        _scatterer_phases     = self.scatterer_phases[valid_scatterers]
                    else:
                        # If all are valid, just use original arrays (avoids copying)
                        _scatterer_positions  = scatterer_positions
                        _scatterer_velocities = scatterer_velocities
                        _reflection_coeffs    = reflection_coeffs
                        _scatterer_speeds     = scatterer_speeds
                        _dist_tx_sc           = dist_tx_sc
                        _dist_sc_rx           = dist_sc_rx
                        _d_n_array            = d_n_array
                        _num_valid_scatterers = num_scatterers
                        _scatterer_phases     = self.scatterer_phases

                    if _num_valid_scatterers > 0:
                        # 2. Calculate Path Gain for each scatterer
                        scatterer_gains = np.zeros(_num_valid_scatterers, dtype=complex)
                        
                        # Process each scatterer
                        for idx in range(_num_valid_scatterers):
                            # Determine if scatterer path is NLOS (always true for scatterers)
                            scatterer_is_los = False
                            
                            # Calculate path gain for this scatterer path
                            path_gain = self.calculate_path_gain(_d_n_array[idx], f_k, scatterer_is_los, rx_pos)
                            
                            # Apply reflection coefficient
                            scatterer_gains[idx] = path_gain * _reflection_coeffs[idx]
                        
                        # 3. Calculate Doppler Frequency (properly accounting for 3D motion)
                        # Normalized direction vectors from Tx to scatterer and from scatterer to Rx
                        dir_tx_sc = np.zeros((_num_valid_scatterers, 3))  # Now 3D vectors
                        dir_sc_rx = np.zeros((_num_valid_scatterers, 3))  # Now 3D vectors
                        
                        # Calculate distances between Tx and scatterers, and between scatterers and Rx in 3D
                        vec_tx_sc = _scatterer_positions - tx_pos  # Vector from Tx to scatterer (3D)
                        dist_tx_sc = np.linalg.norm(vec_tx_sc, axis=1)  # 3D distance from Tx to scatterer
                        vec_sc_rx = rx_pos - _scatterer_positions  # Vector from scatterer to Rx (3D)
                        dist_sc_rx = np.linalg.norm(vec_sc_rx, axis=1)  # 3D distance from scatterer to Rx
                        
                        # Compute normalized direction vectors (avoiding division by zero)
                        nonzero_tx_sc = dist_tx_sc > 1e-9
                        nonzero_sc_rx = dist_sc_rx > 1e-9
                        
                        if np.any(nonzero_tx_sc):
                            dir_tx_sc[nonzero_tx_sc] = vec_tx_sc[nonzero_tx_sc] / dist_tx_sc[nonzero_tx_sc, np.newaxis]
                        
                        if np.any(nonzero_sc_rx):
                            dir_sc_rx[nonzero_sc_rx] = vec_sc_rx[nonzero_sc_rx] / dist_sc_rx[nonzero_sc_rx, np.newaxis]
                        
                        # Calculate cosine of angles using dot product between velocity and direction vectors
                        # Initialize arrays for cosine values
                        cos_theta_tx_sc = np.zeros(_num_valid_scatterers)
                        cos_theta_sc_rx = np.zeros(_num_valid_scatterers)
                        
                        # Only calculate for scatterers with non-zero speed
                        nonzero_speeds = _scatterer_speeds > 1e-9
                        
                        if np.any(nonzero_speeds):
                            # Compute dot product between velocity and direction vectors (3D dot products)
                            dot_v_tx_sc = np.einsum('ij,ij->i', _scatterer_velocities[nonzero_speeds], 
                                                  dir_tx_sc[nonzero_speeds])
                            dot_v_sc_rx = np.einsum('ij,ij->i', _scatterer_velocities[nonzero_speeds], 
                                                  dir_sc_rx[nonzero_speeds])
                            
                            # Normalize by speed to get cosine values
                            cos_theta_tx_sc[nonzero_speeds] = dot_v_tx_sc / _scatterer_speeds[nonzero_speeds]
                            cos_theta_sc_rx[nonzero_speeds] = dot_v_sc_rx / _scatterer_speeds[nonzero_speeds]
                        
                        # Calculate Doppler frequency: f_d = (v/λ)*(cos(θ_tx) + cos(θ_rx))
                        # This is the 3GPP TR 38.901 §8.4-31 formula
                        f_Dn_array = (_scatterer_speeds / lambda_) * (cos_theta_tx_sc + cos_theta_sc_rx)
                        
                        # 4. Update phases using lightweight Doppler approach
                        # Update phases for all symbols including the first one for phase continuity
                        for idx in range(_num_valid_scatterers):
                            _scatterer_phases[idx] = self.advance_phase(
                                _scatterer_phases[idx], 
                                f_Dn_array[idx], 
                                symbol_duration
                            )
                        
                        # Update the original phase array with the updated phases
                        if not np.all(valid_scatterers):
                            self.scatterer_phases[valid_scatterers] = _scatterer_phases

                        # 5. Apply Doppler phase evolution to path gains
                        doppler_phase_factor_array = np.exp(-1j * _scatterer_phases) # (N_valid,)

                        # 6. Apply Doppler shift to path gains
                        scatterer_contribution_array = scatterer_gains * doppler_phase_factor_array
                        
                        # 7. Apply cluster delay-dependent phase shift for frequency selectivity
                        # Get the cluster index for each scatterer (format of id_ is "{cluster}-{subscatterer}")
                        cluster_contributions = {}
                        for idx in range(_num_valid_scatterers):
                            try:
                                # Parse the cluster index from the scatterer ID
                                cluster_id = int(self.scatterers[valid_scatterers.nonzero()[0][idx]].id_.split('-')[0])
                                
                                # Apply the appropriate cluster delay if available
                                if cluster_id < len(self.cluster_delays):
                                    delay_ns = self.cluster_delays[cluster_id]
                                    
                                    # Convert delay from ns to seconds and apply phase shift for this subcarrier
                                    delay_s = delay_ns * 1e-9
                                    delay_phase_factor = np.exp(-1j * 2 * np.pi * f_k * delay_s)
                                    
                                    # Apply delay to this scatterer's contribution
                                    contribution = scatterer_contribution_array[idx] * delay_phase_factor
                                    
                                    # Group by cluster for power-law weighting
                                    if cluster_id in cluster_contributions:
                                        cluster_contributions[cluster_id].append(contribution)
                                    else:
                                        cluster_contributions[cluster_id] = [contribution]
                                else:
                                    # If no delay is available, just use the contribution as is
                                    if -1 in cluster_contributions:
                                        cluster_contributions[-1].append(scatterer_contribution_array[idx])
                                    else:
                                        cluster_contributions[-1] = [scatterer_contribution_array[idx]]
                            except (ValueError, IndexError):
                                # If there's an error parsing the ID, just use the contribution as is
                                if -1 in cluster_contributions:
                                    cluster_contributions[-1].append(scatterer_contribution_array[idx])
                                else:
                                    cluster_contributions[-1] = [scatterer_contribution_array[idx]]
                        
                        # Sum the contributions from all clusters
                        scatterer_gain = 0 + 0j
                        for cluster_id, contributions in cluster_contributions.items():
                            cluster_sum = sum(contributions)
                            if cluster_id >= 0 and cluster_id < len(self.cluster_powers):
                                # Weight by square root of cluster power from power-law model
                                # Already applied during scatterer creation, so no need to apply again
                                scatterer_gain += cluster_sum
                            else:
                                scatterer_gain += cluster_sum
                    # --- End Vectorised Scatterer Calculations --- #

                # Combine dominant (LOS) and explicit scatterer paths
                total_dominant_paths_gain = dominant_gain + scatterer_gain

                # Add aggregate scattering term H_ij if enabled
                H_ij = (0 + 0j)
                if self.use_aggregate_scattering:
                    # --- Calculate Variance based on paper's formula ---
                    # Use f_k (subcarrier frequency) for f_n
                    # Use LOS_distance for d
                    if LOS_distance > 1e-6: # Avoid division by zero if Tx/Rx are coincident
                        # Note: Paper uses f_n (center freq), using f_k makes variance freq-dependent
                        scatter_variance_linear = (10**(-3.24)) * ((f_k / 1e9)**(-2)) * (LOS_distance**(-3))
                    else:
                        scatter_variance_linear = 0

                    # Generate complex Gaussian using the calculated linear variance
                    if scatter_variance_linear > 1e-12:
                        std_dev_H = np.sqrt(scatter_variance_linear / 2.0)
                        H_ij = self.rng.normal(0, std_dev_H) + 1j * self.rng.normal(0, std_dev_H)

                # Final gain is sum of dominant paths and aggregate scattering
                # Apply a scaling factor to enhance visualization
                final_gain = total_dominant_paths_gain + H_ij
                
                # Add an additional frequency-selective component based on subcarrier index 
                # to enhance the frequency selectivity visualization when number of clusters is low
                if self.num_cluster < 3 and self.pathloss_model.lower() == "friis":
                    # Add a gentle sinusoidal variation across subcarriers (~3-5 dB)
                    freq_selective_factor = 1.0 + 0.3 * np.sin(2 * np.pi * j / num_subcarriers * 2)
                    final_gain *= freq_selective_factor

                # --------------------------------

                self.channel_matrix[j, i] = final_gain

            # --- Vectorised Scatterer Position Update --- #
            if num_scatterers > 0:
                scatterer_positions += scatterer_velocities * symbol_duration
                
                # Update the positions in the actual scatterer objects
                for s_idx, scatterer in enumerate(self.scatterers):
                    scatterer.pos = scatterer_positions[s_idx]
            # ------------------------------------------ #

        # add noise to the channel matrix
        noise = self.generate_noise(self.channel_matrix)
        self.noise_matrix = noise
        
        # Apply amplification factor for 3GPP model for better visualization
        if self.pathloss_model.lower() == "3gpp":
            amplification_factor = 5.0
            self.channel_matrix *= amplification_factor
            
        self.channel_matrix_noisy = self.channel_matrix + noise

        # input data in time domain so convert to freq so we can multiply instead of convolve
        input_data    = np.fft.fft(input_data, axis=0)
        output_signal = np.multiply(self.channel_matrix, input_data)
        # convert everything back to time domain to be demultiplexed
        output_signal = np.fft.ifft(output_signal, axis=0)
        output_signal = np.reshape(output_signal, (output_length,-1))
        return output_signal

    def generate_noise(self, signal):
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
        
        # For visualization purposes, if delays are very small, spread them out a bit
        if num_clusters > 1 and np.max(delays_ns) < 50:
            delay_factor = max(50 / np.max(delays_ns), 1)
            delays_ns *= delay_factor
        
        # 2. Per-cluster shadowing (Z_n in dB)
        shadow_fading_db = self.rng.normal(0, self.sigma_zeta_db, size=num_clusters)
        # Corrected application for power calculation: 10**(-Z_n/10)
        zeta_lin_corrected = 10**(-shadow_fading_db / 10)
        
        # 3. Exponential power law (Corrected exponent as per 3GPP Eq 7.5-5)
        if self.r_tau == 0: # Avoid division by zero if r_tau is zero for some reason
            power_exponent = -delays_ns / self.rms_delay_spread_ns # Simplified, but should not happen with 3GPP r_tau values
        else:
            power_exponent = -(self.r_tau - 1) * delays_ns / (self.r_tau * self.rms_delay_spread_ns)
        
        raw_power = np.exp(power_exponent) * zeta_lin_corrected
        
        # For visualization with Friis model, make the power distribution more even
        if self.pathloss_model.lower() == "friis":
            # Give more power to later clusters for better multipath visualization
            # Boost later clusters (but still maintain some decay)
            if num_clusters > 1:
                boost_factor = 0.5 * (1 - np.arange(num_clusters) / num_clusters)
                raw_power = raw_power * (1 + boost_factor)
        
        # 4. Normalize
        powers = raw_power / raw_power.sum()
        
        return delays_ns, powers




