"""
Filename: ./data_generation/ray_tracing/classes/environment.py
Author: Vincenzo Nannetti
Date: 10/05/2025 (Refactored Date)
Description: Environment Class for the ray tracing model.
"""

import numpy as np
import pyvista as pv
import warnings
from .vector_utils import ensure_vector3d
from .scatterer.scatterer import Scatterer
from .antenna import Antenna
from typing import Optional
from scipy.spatial import cKDTree


class Environment:
    def __init__(self, dimensions, movement_type="random_walk"):
        self.dimensions = dimensions        # (x, y, z) dimensions of the space
        self.scatterers = []                # list of scatterers in the environment
        self.rx_antennas = []               # list of rx antennas in the environment
        self.tx_antennas = []               # list of tx antennas in the environment
        self.clusters = []                  # list of cluster centers and properties
        self.rng = np.random.default_rng()  # random number generator
        self.movement_type = movement_type  # Global movement type for scatterers

    def place_tx(self, name, position, gain_dbi):
        """Place a transmitter antenna in the environment"""
        # Check if position is within bounds
        if not self._is_within_bounds(position):
            raise ValueError(f"TX antenna position {position} is outside environment bounds {self.dimensions}")
        
        antenna = Antenna(name, "TX", gain_dbi, position)
        self.tx_antennas.append(antenna)
        return antenna
    
    def place_rx(self, name, position, gain_dbi):
        """Place a receiver antenna in the environment"""
        # Check if position is within bounds
        if not self._is_within_bounds(position):
            raise ValueError(f"RX antenna position {position} is outside environment bounds {self.dimensions}")
        
        antenna = Antenna(name, "RX", gain_dbi, position)
        self.rx_antennas.append(antenna)
        return antenna

    def add_cluster(self,
                    center,
                    num_scatterers,
                    radius,
                    height_range=None,
                    scatterer_speed_range: Optional[list] = None,
                    use_aggregate_scattering_for_scatterer: bool = False,
                    environment_type_for_scatterer: str = "UMa",
                    reflection_params_for_scatterer: Optional[dict] = None,
                    carrier_freq_ghz_for_scatterer: float = 2.49,
                    min_far_field: float = 5.0,
                    cluster_movement_config: Optional[dict] = None):
        """
        Add a cluster of scatterers around a centre point, ensuring that the
        centre is at least (min_far_field + radius) away from any antenna.
        """
        center = np.array(ensure_vector3d(center), dtype=float)
        effective_vertical_extent_from_center = height_range if height_range is not None else radius

        if not self._is_within_bounds(np.array([center[0], center[1], center[2]])) :
            raise ValueError(f"Initial cluster centre {center} is outside environment bounds {self.dimensions}")

        # --- Scenario-specific adjustment for input 'center[2]' using environment_type_for_scatterer ---
        if environment_type_for_scatterer == "RMa": 
            bs_height_rma               = self.tx_antennas[0].get_pos()[2] if self.tx_antennas else 35.0
            min_realistic_center_z      = effective_vertical_extent_from_center
            max_allowed_cluster_top_rma = bs_height_rma + 20.0
            max_realistic_center_z      = max_allowed_cluster_top_rma - effective_vertical_extent_from_center
            max_realistic_center_z      = max(min_realistic_center_z, max_realistic_center_z)

            if center[2] > max_realistic_center_z:
                center[2] = max_realistic_center_z
            if center[2] < min_realistic_center_z:
                center[2] = min_realistic_center_z
        
        elif environment_type_for_scatterer == "UMa":
            bs_height_uma               = self.tx_antennas[0].get_pos()[2] if self.tx_antennas else 25.0
            min_realistic_center_z      = effective_vertical_extent_from_center
            max_allowed_cluster_top_uma = bs_height_uma + 30.0 # Allow clusters somewhat above BS
            max_realistic_center_z      = max_allowed_cluster_top_uma - effective_vertical_extent_from_center
            max_realistic_center_z      = max(min_realistic_center_z, max_realistic_center_z)

            if center[2] > max_realistic_center_z:
                center[2] = max_realistic_center_z
            if center[2] < min_realistic_center_z:
                center[2] = min_realistic_center_z

        elif environment_type_for_scatterer == "UMi":
            bs_height_umi               = self.tx_antennas[0].get_pos()[2] if self.tx_antennas else 10.0
            min_realistic_center_z      = effective_vertical_extent_from_center
            # Allow clusters above UMi BS, up to a reasonable building height, capped by env dimensions
            max_allowed_cluster_top_umi = min(bs_height_umi + 25.0, self.dimensions[2] - effective_vertical_extent_from_center)
            max_realistic_center_z      = max_allowed_cluster_top_umi # Center itself can go up to this if extent is 0
            if effective_vertical_extent_from_center > 0 : # ensure subtraction is valid if extent is from center
                 max_realistic_center_z = max_allowed_cluster_top_umi # Top of cluster at this height
            # Re-evaluate max_realistic_center_z correctly based on its definition
            max_realistic_center_z      = (bs_height_umi + 25.0) - effective_vertical_extent_from_center
            max_realistic_center_z      = min(max_realistic_center_z, self.dimensions[2] - effective_vertical_extent_from_center)
            max_realistic_center_z      = max(min_realistic_center_z, max_realistic_center_z)
            
            if center[2] > max_realistic_center_z:
                center[2] = max_realistic_center_z
            if center[2] < min_realistic_center_z:
                center[2] = min_realistic_center_z
        
        elif "InF" in environment_type_for_scatterer: # General rule for Indoor Factory types
            # Determine default BS height based on InF sub-type if possible
            if self.tx_antennas:
                bs_height_inf = self.tx_antennas[0].get_pos()[2]
            elif "-SH" in environment_type_for_scatterer or "-DH" in environment_type_for_scatterer:
                bs_height_inf = 6.0 # Typical for high BS InF
            else: # SL or DL or generic InF
                bs_height_inf = 3.0 # Typical for low BS InF
            
            min_realistic_center_z      = effective_vertical_extent_from_center
            # Allow clusters some height above BS, but capped by hall ceiling
            max_allowed_cluster_top_inf = min(bs_height_inf + 10.0, self.dimensions[2]) 
            max_realistic_center_z      = max_allowed_cluster_top_inf - effective_vertical_extent_from_center
            max_realistic_center_z      = max(min_realistic_center_z, max_realistic_center_z)

            if center[2] > max_realistic_center_z:
                center[2] = max_realistic_center_z
            if center[2] < min_realistic_center_z:
                center[2] = min_realistic_center_z
        
        # For InH (Indoor Hotspot), the existing generic clamping in Step 3 against 
        # self.dimensions should be sufficient if self.dimensions[2] is set to the room height.

        margin_spherical_ant = min_far_field + radius
        for ant in self.tx_antennas + self.rx_antennas:
            ant_pos = np.array(ant.get_pos(), dtype=float)
            vec = center - ant_pos
            dist = np.linalg.norm(vec)
            if dist < margin_spherical_ant:
                if dist == 0: 
                    direction_candidate = self.rng.standard_normal(3)
                    norm_dir = np.linalg.norm(direction_candidate)
                    if norm_dir < 1e-9: direction = np.array([1.0,0.0,0.0]) 
                    else: direction = direction_candidate / norm_dir
                else:
                    direction = vec / dist
                center = ant_pos + direction * margin_spherical_ant

        # 3) clamp centre back into environment bounds using appropriate margins
        final_margin_xy = radius 
        final_margin_z = effective_vertical_extent_from_center

        low_clip_bounds = np.array([
            final_margin_xy,
            final_margin_xy,
            final_margin_z
        ])
        high_clip_bounds = np.array([
            self.dimensions[0] - final_margin_xy,
            self.dimensions[1] - final_margin_xy,
            self.dimensions[2] - final_margin_z
        ])
        
        high_clip_bounds = np.maximum(low_clip_bounds, high_clip_bounds)
        
        center = np.clip(center, low_clip_bounds, high_clip_bounds)

        # Default height_range for storing (used by _create_scatterers if passed as None)
        height_range_for_scatterers = height_range if height_range is not None else radius

        # Store cluster parameters
        cluster_data_entry = {
            'center':         center,
            'num_scatterers': num_scatterers,
            'radius':         radius,
            'height_range':   height_range_for_scatterers,
            'shared_flock_velocity': None 
        }
        self.clusters.append(cluster_data_entry)

        # Generate scatterers
        new_scatterers = self._create_scatterers(
            center,
            num_scatterers,
            radius,
            height_range_for_scatterers,
            speed_range=scatterer_speed_range,
            aggregate_scattering=use_aggregate_scattering_for_scatterer,
            environment_type=environment_type_for_scatterer,
            reflection_params_scatterer=reflection_params_for_scatterer,
            carrier_freq_ghz=carrier_freq_ghz_for_scatterer,
            min_far_field=min_far_field,
            cluster_movement_config=cluster_movement_config
        )

        # Add to environment
        self.scatterers.extend(new_scatterers)
        return new_scatterers

    def add_cluster_near_los(
            self, tx_idx:int, rx_idx:int,
            frac_along:float,
            los_clearance:float,
            num_scatterers:int,
            radius:float,
            height_range:Optional[float]=None,
            min_far_field:float=5.0,
            **scatterer_kwargs):
        """
        Place a cluster centre on the LOS line at `frac_along` (0→at BS, 1→at UE)
        then push it perpendicularly by >= los_clearance metres.
        No part of the cluster is allowed within `min_far_field` of *any* antenna.
        
        Args:
            tx_idx: Index of the transmitter antenna
            rx_idx: Index of the receiver antenna
            frac_along: Fraction along LOS path (0=at TX, 1=at RX)
            los_clearance: Minimum perpendicular distance from LOS
            num_scatterers: Number of scatterers in the cluster
            radius: Radius of the cluster
            height_range: Vertical range of the cluster. If None, defaults to radius.
            min_far_field: Minimum distance between cluster/scatterers and antennas
            **scatterer_kwargs: Additional arguments passed to add_cluster
        """
        tx_pos = np.array(self.tx_antennas[tx_idx].get_pos(), float)
        rx_pos = np.array(self.rx_antennas[rx_idx].get_pos(), float)

        # 1. choose centre on LOS
        centre_los = tx_pos + frac_along * (rx_pos - tx_pos)

        # 2. random perpendicular displacement
        e_los, e1, e2 = self._los_basis(tx_pos, rx_pos)
        phi = self.rng.uniform(0, 2*np.pi)
        centre = (centre_los
                + (los_clearance + self.rng.exponential(radius/4))  # ≥ clearance
                * (np.cos(phi)*e1 + np.sin(phi)*e2))

        # 3. keep inside box & above ground
        centre[0] = np.clip(centre[0], 0+radius, self.dimensions[0]-radius)
        centre[1] = np.clip(centre[1], 0+radius, self.dimensions[1]-radius)
        centre[2] = np.clip(centre[2], 0+(height_range or radius), self.dimensions[2]-(height_range or radius))

        # 4. enforce far-field for the *centre* itself
        for ant in self.tx_antennas + self.rx_antennas:
            if np.linalg.norm(centre - ant.get_pos()) < min_far_field+radius:
                raise ValueError("Cannot place cluster: far-field margin violated")

        # 5. pass through to the normal cluster routine
        return self.add_cluster(centre, num_scatterers, radius,
                               height_range=height_range,
                               min_far_field=min_far_field,
                               **scatterer_kwargs)

    def _create_scatterers(self, center, num_scatterers, radius, height_range, 
                      min_distance=0.5, speed_range=None, aggregate_scattering=False, 
                      environment_type="UMa", reflection_params_scatterer=None,
                      carrier_freq_ghz=2.49, min_far_field=5.0,
                      cluster_movement_config: Optional[dict] = None):
        """
        Create a cluster of scatterers around a center point, using a KD-tree
        to enforce a minimum inter-scatterer spacing.
        """
        if speed_range is None:
            speed_range = [0, 0]

        new_scatterers = []
        points = []        # list of accepted positions
        tree = None        # will become our KD-tree
        attempts = 0
        max_attempts = num_scatterers * 10

        # Determine the cluster_id for the scatterers being created
        # This assumes add_cluster has just appended the cluster_data_entry
        cluster_id_for_current_op = len(self.clusters) - 1

        # For flocking movement, generate a single shared velocity for this cluster
        # This velocity is generated ONCE for all scatterers in this cluster call.
        generated_shared_flock_velocity_for_cluster = None
        flock_speed_for_this_cluster = 0.0 # Used to set s.speed for flock members

        if self.movement_type == "flocking":
            # Use the general speed_range for the flock's speed magnitude
            flock_speed_for_this_cluster = self.rng.uniform(speed_range[0], speed_range[1]) if speed_range else 0.0
            
            generated_shared_flock_velocity_for_cluster = np.zeros(3, dtype=float) # Initialize
            flock_direction = None

            if flock_speed_for_this_cluster > 1e-9: # Only determine direction if speed is meaningful
                user_specified_direction = None
                if cluster_movement_config:
                    user_specified_direction = cluster_movement_config.get("flock_movement_direction")

                if user_specified_direction and isinstance(user_specified_direction, (list, tuple)) and len(user_specified_direction) == 3:
                    direction_vec = np.array(user_specified_direction, dtype=float)
                    norm = np.linalg.norm(direction_vec)
                    if norm > 1e-9:
                        flock_direction = direction_vec / norm
                    else:
                        warnings.warn(
                            f"Cluster {cluster_id_for_current_op}: flock_movement_direction was a zero vector "
                            f"or invalid ({user_specified_direction}). Defaulting to random direction."
                        )
                        # Let it fall through to random generation below

                if flock_direction is None: # No valid user direction, so generate random
                    flock_direction_candidate = self.rng.standard_normal(3)
                    norm = np.linalg.norm(flock_direction_candidate)
                    if norm > 1e-9:
                        flock_direction = flock_direction_candidate / norm
                    else: # Default to a fixed direction if random vector is zero (rare)
                        flock_direction = np.array([1.0, 0.0, 0.0])
                
                generated_shared_flock_velocity_for_cluster = flock_direction * flock_speed_for_this_cluster
            # else: speed is negligible, so velocity remains zeros(3)
            
            # Store this shared velocity in the cluster's data structure
            if cluster_id_for_current_op < len(self.clusters):
                self.clusters[cluster_id_for_current_op]['shared_flock_velocity'] = generated_shared_flock_velocity_for_cluster.copy()
            else:
                warnings.warn(f"Flocking: Cluster index {cluster_id_for_current_op} out of bounds when trying to set shared_flock_velocity.")

        while len(new_scatterers) < num_scatterers and attempts < max_attempts:
            attempts += 1

            u = self.rng.random()
            r = radius * u**(1/3)
            cos_phi = self.rng.uniform(-1, 1)
            phi = np.arccos(cos_phi)
            theta = self.rng.uniform(0, 2*np.pi)

            dx = r * np.sin(phi) * np.cos(theta)
            dy = r * np.sin(phi) * np.sin(theta)
            dz = r * np.cos(phi) * (height_range / radius)
            new_pos = np.array([center[0] + dx, center[1] + dy, center[2] + dz])

            # 1) ensure above ground and inside bounds
            if new_pos[2] <= 0 or not self._is_within_bounds(new_pos):
                continue

            # 2) far-field margin from antennas
            if any(self._distance3d(new_pos, ant.get_pos()) < min_far_field
                for ant in self.tx_antennas + self.rx_antennas):
                continue

            # 3) inter-scatterer spacing via KD-tree
            if tree is not None:
                # if any existing point is within min_distance, reject
                if tree.query_ball_point(new_pos, r=min_distance):
                    continue

            # --- accept this position ---
            points.append(new_pos)
            tree = cKDTree(points)  # rebuild tree each insertion

            # create Scatterer object
            scatterer_id = f"{cluster_id_for_current_op}-{len(new_scatterers)}"
            
            # For flocking, speed is determined by the flock, not individually sampled here
            # For other types, it's individually sampled.
            current_scatterer_speed = flock_speed_for_this_cluster if self.movement_type == "flocking" else self.rng.uniform(speed_range[0], speed_range[1])
            
            # This dictionary will hold the specific, sampled parameters for the scatterer
            individual_scatterer_movement_params = {}

            if self.movement_type == "sinusoidal" and cluster_movement_config:
                amp_range    = cluster_movement_config.get("amplitude_range", [0.0, 0.0])
                period_range = cluster_movement_config.get("period_range", [1.0, 1.0])
                axis_config  = cluster_movement_config.get("axis", "random") 

                individual_scatterer_movement_params['sinusoidal_amplitude'] = self.rng.uniform(amp_range[0], amp_range[1])
                individual_scatterer_movement_params['sinusoidal_period']    = self.rng.uniform(period_range[0], period_range[1])
                individual_scatterer_movement_params['sinusoidal_axis']      = axis_config
            
            elif self.movement_type == "brownian" and cluster_movement_config:
                sigma_range = cluster_movement_config.get("sigma_brownian_range", [0.005, 0.015])
                individual_scatterer_movement_params['sigma_brownian'] = self.rng.uniform(sigma_range[0], sigma_range[1])

            elif self.movement_type == "gauss_markov" and cluster_movement_config:
                alpha_range = cluster_movement_config.get("alpha_gm_range", [0.3, 0.7])
                noise_std_dev_range = cluster_movement_config.get("noise_std_dev_gm_range", [0.05, 0.15])
                mean_vel_config = cluster_movement_config.get("mean_velocity_gm_config", [0.,0.,0.]) 

                individual_scatterer_movement_params['alpha_gm'] = self.rng.uniform(alpha_range[0], alpha_range[1])
                individual_scatterer_movement_params['noise_std_dev_gm'] = self.rng.uniform(noise_std_dev_range[0], noise_std_dev_range[1])
                individual_scatterer_movement_params['mean_velocity_gm_config'] = mean_vel_config
            
            # For "linear" or "random_walk", no extra params are typically needed from cluster_movement_config 
            # beyond what Scatterer.__init__ defaults for its own attributes if movement_specific_params is empty.

            s = Scatterer(
                scatterer_id,
                new_pos.tolist(),
                current_scatterer_speed, # Use flock speed if flocking, else individual
                use_aggregate_scattering=aggregate_scattering,
                environment_type=environment_type,
                reflection_params=reflection_params_scatterer,
                frequency_ghz=carrier_freq_ghz,
                movement_type=self.movement_type, # Pass the environment's movement type
                movement_specific_params=individual_scatterer_movement_params # Pass the prepared dict
            )

            # If flocking, override the scatterer's velocity_vector with the shared one
            if self.movement_type == "flocking" and generated_shared_flock_velocity_for_cluster is not None:
                s.velocity_vector = generated_shared_flock_velocity_for_cluster.copy()
                # s.speed is already set to flock_speed_for_this_cluster

            new_scatterers.append(s)

        if len(new_scatterers) < num_scatterers:
            warnings.warn(
                f"Cluster {len(self.clusters)-1}: placed {len(new_scatterers)} "
                f"of {num_scatterers} requested scatterers "
                f"(radius {radius}, min_distance {min_distance}, min_far_field {min_far_field})."
            )

        return new_scatterers

    def advance(self, dt):
        """Advance the environment by a time step dt"""

        if self.movement_type == "flocking":
            # Phase 1: Synchronize flock scatterers to their cluster's shared velocity
            for cluster_idx, cluster_data in enumerate(self.clusters):
                shared_vel = cluster_data.get('shared_flock_velocity')
                if shared_vel is not None: # This cluster is a flock
                    for s in self.scatterers:
                        if s.cluster_id == cluster_idx:
                            s.velocity_vector = shared_vel.copy() # Ensure they all start with the same velocity for this step
        
        # Phase 2: All scatterers update their position based on their current velocity_vector
        # For flocking scatterers, this velocity_vector was just set to the shared one.
        # If an individual flocking scatterer hits a boundary, its s.velocity_vector will be modified by flocking_move.
        for s in self.scatterers:
            s.update_pos(dt, movement_type=self.movement_type, environment_dims=self.dimensions)
        
        if self.movement_type == "flocking":
            # Phase 3: Reconcile flock velocities. If any member of a flock reflected, the flock reflects.
            for cluster_idx, cluster_data in enumerate(self.clusters):
                original_flock_velocity = cluster_data.get('shared_flock_velocity')
                if original_flock_velocity is None: # Not a flocking cluster or not initialized
                    continue

                new_collective_flock_velocity = original_flock_velocity.copy()
                reflection_occurred_on_axis = [False, False, False]

                for s in self.scatterers:
                    if s.cluster_id == cluster_idx:
                        for axis_i in range(3):
                            # Check if velocity component sign flipped, indicating reflection by _handle_boundaries
                            # (and original component was not zero, to avoid issues with stationary axes)
                            if original_flock_velocity[axis_i] != 0 and \
                               (s.velocity_vector[axis_i] * original_flock_velocity[axis_i] < -1e-9): # Check for sign flip robustly
                                reflection_occurred_on_axis[axis_i] = True
                
                for axis_i in range(3):
                    if reflection_occurred_on_axis[axis_i]:
                        new_collective_flock_velocity[axis_i] *= -1
                
                self.clusters[cluster_idx]['shared_flock_velocity'] = new_collective_flock_velocity
            
    def visualise_movement(self, time_period=1.0, movement_to_visualise="linear", show_bounds=True, background='white'):
        """
        Visualise scatterer positions before and after movement over a time period.
        
        Args:
            time_period (float): Time in seconds to simulate movement
            movement_to_visualise (str): Movement type to apply ("linear", "random_walk", "sinusoidal")
            show_bounds (bool): Whether to draw the environment cube
            background (str): Background colour
        """
        # Store original positions
        original_positions = np.array([s.get_pos() for s in self.scatterers])
        
        # Make copies of scatterers to restore them later
        import copy
        scatterer_backup = copy.deepcopy(self.scatterers)
        
        # Save the current environment movement type and set to the type for visualisation
        original_env_movement_type = self.movement_type
        self.movement_type = movement_to_visualise
        
        # Advance environment by specified time
        # For some movement types like random_walk, multiple small steps can be smoother
        if self.movement_type == "random_walk": 
            num_steps = 10
            dt_step = time_period / num_steps
            for _ in range(num_steps):
                self.advance(dt_step) # advance uses the new self.movement_type
        else:
            self.advance(time_period) # advance uses the new self.movement_type
        
        # Get new positions
        new_positions = np.array([s.get_pos() for s in self.scatterers])
        
        # Visualise both sets
        plotter = pv.Plotter(window_size=[1200, 800])
        plotter.set_background(background)
        
        # Draw ground plane
        ground_size_x = self.dimensions[0]
        ground_size_y = self.dimensions[1]
        ground_center_x = self.dimensions[0] / 2
        ground_center_y = self.dimensions[1] / 2
        ground_plane = pv.Plane(
            center=(ground_center_x, ground_center_y, 0),
            direction=(0, 0, 1),
            i_size=ground_size_x,
            j_size=ground_size_y,
            i_resolution=10,
            j_resolution=10
        )
        plotter.add_mesh(ground_plane, color='lightgrey', ambient=0.2, diffuse=0.8, specular=0.1, roughness=0.7)
        
        # Draw TX/RX antennas
        for tx in self.tx_antennas:
            pos = np.array(tx.get_pos())
            antenna_base = pv.Cylinder(
                center=[pos[0], pos[1], pos[2]/2], 
                direction=[0, 0, 1],
                radius=0.5,
                height=pos[2] 
            )
            antenna_top = pv.Cone(
                center=[pos[0], pos[1], pos[2]], 
                direction=[0, 0, 1],
                radius=0.7,
                height=1.2 
            )
            plotter.add_mesh(antenna_base, color='red', metallic=0.8, roughness=0.2)
            plotter.add_mesh(antenna_top, color='red', metallic=0.8, roughness=0.2)
            plotter.add_point_labels([pos[0],pos[1],pos[2]+1.5], [tx.name], point_size=0, font_size=12, 
                                    text_color='red', shape_opacity=0)
        
        for rx in self.rx_antennas:
            pos = np.array(rx.get_pos())
            antenna_base = pv.Cylinder(
                center=[pos[0], pos[1], pos[2]/2], 
                direction=[0, 0, 1],
                radius=0.3,
                height=pos[2]
            )
            antenna_top = pv.Cone(
                center=[pos[0], pos[1], pos[2]], 
                direction=[0, 0, 1],
                radius=0.5,
                height=0.8
            )
            plotter.add_mesh(antenna_base, color='blue', metallic=0.8, roughness=0.2)
            plotter.add_mesh(antenna_top, color='blue', metallic=0.8, roughness=0.2)
            plotter.add_point_labels([pos[0],pos[1],pos[2]+1.0], [rx.name], point_size=0, font_size=12, 
                                    text_color='blue', shape_opacity=0)
        
        # Add LOS ray between TX and RX
        if self.tx_antennas and self.rx_antennas:
            for tx in self.tx_antennas:
                tx_pos = np.array(tx.get_pos())
                for rx in self.rx_antennas:
                    rx_pos = np.array(rx.get_pos())
                    los_line = pv.Line(tx_pos, rx_pos)
                    plotter.add_mesh(los_line, color='yellow', line_width=3, opacity=1.0)
        
        # Draw original positions (transparent green)
        if len(original_positions) > 0:
            plotter.add_points(original_positions, color='lightgreen', point_size=8, 
                              render_points_as_spheres=True, opacity=0.5)
        
        # Draw movement paths as lines
        for i in range(len(original_positions)):
            line = pv.Line(original_positions[i], new_positions[i])
            plotter.add_mesh(line, color='gray', line_width=1, opacity=0.3)
        
        # Draw new positions (solid green)
        if len(new_positions) > 0:
            plotter.add_points(new_positions, color='green', point_size=8, 
                              render_points_as_spheres=True)
        
        # Draw environment bounds
        if show_bounds:
            cube = pv.Cube(center=(
                self.dimensions[0] / 2,
                self.dimensions[1] / 2,
                self.dimensions[2] / 2
            ), x_length=self.dimensions[0], y_length=self.dimensions[1], z_length=self.dimensions[2])
            plotter.add_mesh(cube, style='wireframe', color='dimgray', line_width=1, opacity=0.2)
        
        # Add legend
        plotter.add_legend([
            ('Original Positions', 'lightgreen'),
            ('New Positions', 'green'),
            ('Movement Path', 'gray'),
            ('Transmitter', 'red'),
            ('Receiver', 'blue'),
            ('LOS Path', 'yellow')
        ], size=(0.15, 0.15), loc='lower right')
        
        movement_type_str = self.movement_type.replace("_", " ").title()
        plotter.add_text(f"Movement over {time_period} seconds ({movement_type_str})", 
                        font_size=12, position='upper_left')
        
        # Set camera position
        x_mid = self.dimensions[0] / 2
        y_mid = self.dimensions[1] / 2
        z_mid = self.dimensions[2] / 2
        initial_camera_position = [
            (x_mid - self.dimensions[0]*0.7, y_mid - self.dimensions[1]*1.5, z_mid + self.dimensions[2]*2),
            (x_mid, y_mid, z_mid),
            (0, 0, 1)
        ]
        
        plotter.camera_position = initial_camera_position
        plotter.camera.zoom(1.2)
        
        # Add a home button to reset the camera
        def reset_camera():
            plotter.camera_position = initial_camera_position
            plotter.camera.zoom(1.2)
            plotter.render()
            
        plotter.add_key_event('h', reset_camera)
        plotter.add_text("Press 'h' to reset view", font_size=10, position='upper_right')
        
        # Show the plot
        plotter.show()
        
        # Restore original scatterers and environment movement type after visualisation
        self.scatterers = scatterer_backup
        self.movement_type = original_env_movement_type

    def get_scatterer_snapshot(self):
        """Return positions, velocities, reflection_coeffs, speeds as NumPy arrays."""
        return (
            np.array([s.pos               for s in self.scatterers]),
            np.array([s.velocity_vector   for s in self.scatterers]),
            np.array([s.get_reflection_coeff() for s in self.scatterers], dtype=complex),
            np.array([s.speed             for s in self.scatterers])
        )
    
    def _is_within_bounds(self, position):
        """Check if position is within the environment bounds"""
        x, y, z = ensure_vector3d(position)
            
        return (0 <= x <= self.dimensions[0] and
                0 <= y <= self.dimensions[1] and
                0 <= z <= self.dimensions[2])
       
    def _distance3d(self, pos1, pos2):
        """Calculate 3D distance between two positions"""
        p1 = ensure_vector3d(pos1)
        p2 = ensure_vector3d(pos2)
        return np.linalg.norm(p2 - p1)

    def _los_basis(self, tx_pos, rx_pos):
        """Return unit vector along BS→UE line and two orthogonal unit vectors spanning its normal plane."""
        v = rx_pos - tx_pos
        e_los = v / np.linalg.norm(v)
        # Find a vector not parallel to e_los, use it to build an orthonormal basis
        tmp = np.array([1,0,0]) if abs(e_los[0]) < 0.9 else np.array([0,1,0])
        e1 = np.cross(e_los, tmp);  e1 /= np.linalg.norm(e1)
        e2 = np.cross(e_los, e1)
        return e_los, e1, e2

    def visualise_environment(self, show_bounds=True, background='white', 
                              show_rays=True, show_movement=True, movement_time=2.0, 
                              max_rays=100, show_grid=True, 
                              show_cluster_radii_initially=True):
        """
        Visualise the 3D environment using PyVista.
        
        Args:
            show_bounds (bool): Whether to draw the environment cube.
            background (str): Background colour (e.g., 'white', 'black').
            show_rays (bool): Whether to show ray paths from TX to scatterers to RX.
            show_movement (bool): Whether to show movement paths.
            movement_time (float): Time in seconds to simulate for movement paths.
            max_rays (int): Maximum number of ray paths to draw (for performance).
            show_grid (bool): Whether to show grid lines on each plane.
            show_cluster_radii_initially (bool): Whether to show cluster radii spheres initially.
        """
        plotter = pv.Plotter(window_size=[1200, 800]) 
        plotter.set_background(background)
        plotter.enable_lightkit() 

        # 1. Add a solid ground plane at z=0 (top view)
        ground_size_x = self.dimensions[0]
        ground_size_y = self.dimensions[1]
        ground_center_x = self.dimensions[0] / 2
        ground_center_y = self.dimensions[1] / 2
        ground_plane = pv.Plane(
            center=(ground_center_x, ground_center_y, 0),
            direction=(0, 0, 1),  # Normal to the plane is Z-axis
            i_size=ground_size_x,
            j_size=ground_size_y,
            i_resolution=5,  # Reduced for performance
            j_resolution=5   # Reduced for performance
        )
        plotter.add_mesh(ground_plane, color='lightgrey', ambient=0.2, diffuse=0.8, specular=0.1, roughness=0.7, show_edges=False)
        
        # 2. Plot TX antennas (Base Station)
        tx_positions = []
        for tx in self.tx_antennas:
            pos = np.array(tx.get_pos())
            tx_positions.append(pos)
            antenna_base = pv.Cylinder(
                center=[pos[0], pos[1], pos[2]/2], 
                direction=[0, 0, 1],
                radius=0.5,
                height=pos[2] 
            )
            antenna_top = pv.Cone(
                center=[pos[0], pos[1], pos[2]], 
                direction=[0, 0, 1],
                radius=0.7,
                height=1.2 
            )
            plotter.add_mesh(antenna_base, color='red', name='TX', metallic=0.8, roughness=0.2)
            plotter.add_mesh(antenna_top, color='red', name='TX_Top', metallic=0.8, roughness=0.2)
            plotter.add_point_labels([pos[0],pos[1],pos[2]+1.5], [tx.name], point_size=0, font_size=12, text_color='red', shape_opacity=0)

        # 3. Plot RX antennas (User Equipment)
        rx_positions = []
        for rx in self.rx_antennas:
            pos = np.array(rx.get_pos())
            rx_positions.append(pos)
            antenna_base = pv.Cylinder(
                center=[pos[0], pos[1], pos[2]/2], 
                direction=[0, 0, 1],
                radius=0.3,
                height=pos[2]
            )
            antenna_top = pv.Cone(
                center=[pos[0], pos[1], pos[2]], 
                direction=[0, 0, 1],
                radius=0.5,
                height=0.8
            )
            plotter.add_mesh(antenna_base, color='blue', name='RX', metallic=0.8, roughness=0.2)
            plotter.add_mesh(antenna_top, color='blue', name='RX_Top', metallic=0.8, roughness=0.2)
            plotter.add_point_labels([pos[0],pos[1],pos[2]+1.0], [rx.name], point_size=0, font_size=12, text_color='blue', shape_opacity=0)

        # 4. Plot scatterers
        if self.scatterers:
            scatterer_positions = np.array([s.get_pos() for s in self.scatterers])
            points = plotter.add_points(
                scatterer_positions, color='green', point_size=8, 
                render_points_as_spheres=True, name='scatterers'
            )
            
            # 4b. Plot ray paths if requested
            if show_rays and tx_positions and rx_positions:
                # Add LOS ray between TX and RX
                for tx_pos_single in tx_positions: # Renamed to avoid conflict
                    for rx_pos_single in rx_positions: # Renamed to avoid conflict
                        los_line = pv.Line(tx_pos_single, rx_pos_single)
                        plotter.add_mesh(los_line, color='yellow', line_width=3, opacity=1.0, name='los_ray')
                
                # Use the first TX and RX for simplicity for scatterer rays
                # Ensure there's at least one TX and RX before accessing them
                if tx_positions and rx_positions:
                    tx_pos_for_rays = tx_positions[0]
                    rx_pos_for_rays = rx_positions[0]
                
                    # Limit the number of rays for performance
                    num_rays_to_draw = min(len(scatterer_positions), max_rays) # Use a different variable name
                    # Ensure num_rays_to_draw is not zero to avoid division by zero
                    step = max(1, len(scatterer_positions) // num_rays_to_draw if num_rays_to_draw > 0 else len(scatterer_positions) + 1)

                    # Create a line for each path: TX → Scatterer → RX (with stride for performance)
                    for i in range(0, len(scatterer_positions), step):
                        s_pos = scatterer_positions[i]
                        # Line from TX to scatterer
                        line1 = pv.Line(tx_pos_for_rays, s_pos)
                        
                        # Line from scatterer to RX
                        line2 = pv.Line(s_pos, rx_pos_for_rays)
                        
                        # Add lines with different opacity based on scatterer index (to visualize multiple paths)
                        alpha = max(0.1, 1.0 - i / len(scatterer_positions))
                        plotter.add_mesh(line1, color='orange', line_width=1, opacity=alpha, name=f'ray_tx_sc_{i}')
                        plotter.add_mesh(line2, color='cyan', line_width=1, opacity=alpha, name=f'ray_sc_rx_{i}')
            
            # 4c. Show movement paths if requested
            if show_movement and self.scatterers:
                # Save original scatterers
                import copy
                scatterer_backup = copy.deepcopy(self.scatterers)
                original_movement_type = self.movement_type 
                
                # Create future positions after movement time
                self.advance(movement_time)
                future_positions = np.array([s.get_pos() for s in self.scatterers])
                
                # Limit the number of movement paths for performance
                num_paths_to_draw = min(len(scatterer_positions), max_rays) 
                # Ensure num_paths_to_draw is not zero to avoid division by zero
                path_step = max(1, len(scatterer_positions) // num_paths_to_draw if num_paths_to_draw > 0 else len(scatterer_positions) + 1)

                # Draw movement paths as lines (with stride for performance)
                movement_actors = []
                for i in range(0, len(scatterer_positions), path_step):
                    if i < len(future_positions):
                        line = pv.Line(scatterer_positions[i], future_positions[i])
                        actor = plotter.add_mesh(line, color='magenta', line_width=1, opacity=0.5, name=f'movement_{i}')
                        movement_actors.append(actor)
                
                # Draw future points (slightly transparent)
                if len(future_positions) > 0: 
                    future_points = plotter.add_points(
                        future_positions, color='lightgreen', point_size=6, 
                        render_points_as_spheres=True, opacity=0.5, name='future_points'
                    )
                
                # Restore original scatterers
                self.scatterers = scatterer_backup
                self.movement_type = original_movement_type 

        # 5. Plot clusters
        cluster_sphere_actors = [] # To store cluster sphere actors for toggling
        for i, cluster_info in enumerate(self.clusters):
            c = cluster_info['center']
            radius = cluster_info['radius']
            
            # Create the full sphere mesh with lower resolution for better performance
            sphere_mesh = pv.Sphere(radius=radius, center=[c[0], c[1], c[2]], theta_resolution=15, phi_resolution=15)
            
            # Don't perform clipping if cluster is fully within bounds
            needs_clipping = (
                c[2] - radius < 0 or
                c[2] + radius > self.dimensions[2] or
                c[0] - radius < 0 or
                c[0] + radius > self.dimensions[0] or
                c[1] - radius < 0 or
                c[1] + radius > self.dimensions[1]
            )
            
            if needs_clipping:
                # Min Z (Ground plane)
                if c[2] - radius < 0:
                    sphere_mesh = sphere_mesh.clip(normal=(0, 0, 1), origin=(0, 0, 0), invert=False)
                # Max Z
                if c[2] + radius > self.dimensions[2] and sphere_mesh.n_points > 0:
                    sphere_mesh = sphere_mesh.clip(normal=(0, 0, -1), origin=(0, 0, self.dimensions[2]), invert=False)
                # Min X
                if c[0] - radius < 0 and sphere_mesh.n_points > 0:
                    sphere_mesh = sphere_mesh.clip(normal=(1, 0, 0), origin=(0, 0, 0), invert=False)
                # Max X
                if c[0] + radius > self.dimensions[0] and sphere_mesh.n_points > 0:
                    sphere_mesh = sphere_mesh.clip(normal=(-1, 0, 0), origin=(self.dimensions[0], 0, 0), invert=False)
                # Min Y
                if c[1] - radius < 0 and sphere_mesh.n_points > 0:
                    sphere_mesh = sphere_mesh.clip(normal=(0, 1, 0), origin=(0, 0, 0), invert=False)
                # Max Y
                if c[1] + radius > self.dimensions[1] and sphere_mesh.n_points > 0:
                    sphere_mesh = sphere_mesh.clip(normal=(0, -1, 0), origin=(0, self.dimensions[1], 0), invert=False)
            
            if sphere_mesh and sphere_mesh.n_points > 0: 
                actor = plotter.add_mesh(sphere_mesh, color='lightcoral', 
                                         opacity=0.25, name=f'cluster_sphere_{i}', 
                                         show_edges=False, 
                                         pickable=False) # Make it not pickable if not needed
                actor.SetVisibility(show_cluster_radii_initially)
                cluster_sphere_actors.append(actor)

        # 6. Plot bounding box (full cube wireframe)
        if show_bounds:
            # Add a wireframe cube for the environment bounds
            cube = pv.Cube(center=(
                self.dimensions[0] / 2,
                self.dimensions[1] / 2,
                self.dimensions[2] / 2
            ), x_length=self.dimensions[0], y_length=self.dimensions[1], z_length=self.dimensions[2])
            plotter.add_mesh(cube, style='wireframe', color='dimgray', line_width=1, opacity=0.2, name='bounds')

        # 7. Add grid lines (without coordinate labels)
        if show_grid:
            # Define custom bounds for the grid, extending Z to double height
            grid_bounds = (
                0, self.dimensions[0],  # X min, X max
                0, self.dimensions[1],  # Y min, Y max
                0, self.dimensions[2] * 2  # Z min, Z max (double height)
            )
            plotter.show_grid(bounds=grid_bounds, color='lightgray', fmt="%.0f")

        # 8. Set camera position for optimal top and side view
        x_mid = self.dimensions[0] / 2
        y_mid = self.dimensions[1] / 2
        z_mid = self.dimensions[2] / 2
        
        # Position camera to show top and side view
        initial_camera_position = [
            (-self.dimensions[0] * 0.5, y_mid * 0.3, self.dimensions[2] * 1.2),  # Camera position
            (x_mid * 0.2, y_mid, z_mid * 0.5),                                # Focal point
            (0, 0, 1)                                                          # View-up vector (Z is up)
        ]
        
        plotter.camera_position = initial_camera_position
        plotter.camera.zoom(1.2)
        
        # Add a home button to reset the camera
        def reset_camera():
            plotter.camera_position = initial_camera_position
            plotter.camera.zoom(1.2)
            plotter.render()
            
        plotter.add_key_event('h', reset_camera)

        # Add key events to toggle ray visibility and movement paths
        instruction_text = "Press 'h' to reset view"
        
        if show_rays:
            def toggle_rays():
                # Toggle visibility of all ray lines 
                for actor_name, actor in plotter.renderer.actors.items(): # Iterate through items
                    if actor_name.startswith('ray_') or actor_name == 'los_ray':
                        visibility = not actor.GetVisibility()
                        actor.SetVisibility(visibility)
                plotter.render()
                
            plotter.add_key_event('r', toggle_rays)
            instruction_text += ", 'r' to toggle ray paths"
        
        if show_movement:
            def toggle_movement():
                # Toggle visibility of movement lines and future points
                for actor_name, actor in plotter.renderer.actors.items(): # Iterate through items
                    if actor_name.startswith('movement_') or actor_name == 'future_points':
                        visibility = not actor.GetVisibility()
                        actor.SetVisibility(visibility)
                plotter.render()
            
            plotter.add_key_event('m', toggle_movement)
            instruction_text += ", 'm' to toggle movement paths"
                    
        def toggle_grid():
            if hasattr(plotter, 'grid_actor'):
                plotter.grid_actor.SetVisibility(not plotter.grid_actor.GetVisibility())
            plotter.render()
            
        plotter.add_key_event('g', toggle_grid)
        instruction_text += ", 'g' to toggle grid"
        
        # Add key event for toggling cluster radii
        def toggle_cluster_radii():
            for actor in cluster_sphere_actors:
                actor.SetVisibility(not actor.GetVisibility())
            plotter.render()
        plotter.add_key_event('c', toggle_cluster_radii)
        instruction_text += ", 'c' to toggle cluster radii"
        
        plotter.add_text(instruction_text, font_size=10, position='upper_left')

        # Add environment information
        env_info = (
            f"Environment: {self.dimensions[0]}m × {self.dimensions[1]}m × {self.dimensions[2]}m\n"
            f"Movement: {self.movement_type}\n"
            f"Scatterers: {len(self.scatterers)}, Clusters: {len(self.clusters)}"
        )
        plotter.add_text(env_info, font_size=10, position='lower_left')

        # Ensure default trackball interactor for navigation
        plotter.enable_trackball_style()

        # Add an orientation marker
        plotter.add_camera_orientation_widget()

        plotter.show()


if __name__ == "__main__":
    env = Environment(ensure_vector3d([100, 100, 110]))
    env.place_tx("BS", ensure_vector3d([50, 50, 105]), gain_dbi=15) # Centered BS, high up
    env.place_rx("UE1", ensure_vector3d([10, 10, 1.5]), gain_dbi=0) # UE near a corner, on ground
    env.place_rx("UE2", ensure_vector3d([90, 80, 1.5]), gain_dbi=0) # Another UE

    # Clusters for a more complex scene
    env.add_cluster(ensure_vector3d([20, 20, 15]), num_scatterers=150, radius=10, height_range=20) # Near UE1
    env.add_cluster(ensure_vector3d([70, 70, 30]), num_scatterers=200, radius=20, height_range=40) # Mid-field cluster
    env.add_cluster(ensure_vector3d([50, 50, 5]), num_scatterers=100, radius=15, height_range=10)  # Ground-level central cluster
    env.add_cluster(ensure_vector3d([10, 85, 90]), num_scatterers=250, radius=25, height_range=30) # High cluster near an edge
    env.add_cluster(ensure_vector3d([85, 15, 60]), num_scatterers=180, radius=18, height_range=50) # Another mid-high cluster
    
    # A cluster that might go out of bounds to test clipping
    env.add_cluster(ensure_vector3d([5, 5, 5]), num_scatterers=50, radius=15, height_range=10) # Near origin, radius might exceed x=0, y=0, z=0
    env.add_cluster(ensure_vector3d([95, 95, 105]), num_scatterers=50, radius=15, height_range=10) # Near max corner

