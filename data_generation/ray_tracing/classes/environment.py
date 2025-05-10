"""
Filename: ./data_generation/ray_tracing/classes/environment.py
Author: Vincenzo Nannetti
Date: 10/05/2025 (Refactored Date)
Description: Environment Class for the ray tracing model.
"""

import numpy as np
import pyvista as pv
from .vector_utils import ensure_vector3d
from .scatterer import Scatterer
from .antenna import Antenna


class Environment:
    def __init__(self, dimensions):
        self.dimensions = dimensions        # (x, y, z) dimensions of the space
        self.scatterers = []                # list of scatterers in the environment
        self.rx_antennas = []               # list of rx antennas in the environment
        self.tx_antennas = []               # list of tx antennas in the environment
        self.clusters = []                  # list of cluster centers and properties
        self.rng = np.random.RandomState()  # random number generator

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

    def add_cluster(self, center, num_scatterers, radius, height_range=None):
        """
        Add a cluster of scatterers around a center point.
        By default, if height_range is not specified, it will be equal to the radius,
        resulting in a spherical cluster.
        
        Args:
            center: list/tuple with (x, y, z) coordinates
            num_scatterers: number of scatterers in the cluster
            radius: horizontal and vertical radius of the cluster sphere (if height_range is not specified)
            height_range: vertical range of the cluster. If None, defaults to `radius` for a spherical cluster.
        """
        
        center = ensure_vector3d(center)
            
        # Check if center is within bounds
        if not self._is_within_bounds(center):
            raise ValueError(f"Cluster center {center} is outside environment bounds {self.dimensions}")
            
        # Default height_range to radius if not specified, for a spherical cluster
        if height_range is None:
            height_range = radius # Changed from radius / 2
            
        # Store cluster parameters for future reference
        self.clusters.append({
            'center':         center,
            'num_scatterers': num_scatterers,
            'radius':         radius,
            'height_range':   height_range
        })
        
        # Create the scatterers within the cluster
        new_scatterers = self._create_scatterers(
            center, 
            num_scatterers, 
            radius, 
            height_range
        )
        
        # Add the new scatterers to the environment
        self.scatterers.extend(new_scatterers)
        
        return new_scatterers

    def _create_scatterers(self, center, num_scatterers, radius, height_range, min_distance=0.5, speed_range=None, aggregate_scattering=False, environment_type="UMa"):
        """
        Create a cluster of scatterers around a center point
        
        Args:
            center: list/tuple with (x, y, z) coordinates
            num_scatterers: number of scatterers to create
            radius: horizontal radius of the cluster sphere
            height_range: vertical range of the cluster sphere
            min_distance: minimum distance between scatterers
            speed_range: range of speeds for the scatterers
            aggregate_scattering: whether to use aggregate scattering
            environment_type: type of environment
        Returns:
            list of Scatterer objects
        """

        if speed_range is None:
            speed_range = [0, 0]

        new_scatterers = []
        cluster_id = len(self.clusters) - 1  # Use current cluster index
        
        attempts = 0
        max_attempts = num_scatterers * 10  # Prevent infinite loops
        
        while len(new_scatterers) < num_scatterers and attempts < max_attempts:
            attempts += 1
            
            # Generate random spherical coordinates
            theta = np.random.uniform(0, 2 * np.pi)  # Azimuth angle
            phi   = np.random.uniform(0, np.pi)      # Zenith angle
            r     = np.random.uniform(0, radius)     # Radius
            
            # Convert spherical to Cartesian coordinates
            dx = r * np.sin(phi) * np.cos(theta)
            dy = r * np.sin(phi) * np.sin(theta)
            dz = r * np.cos(phi) * (height_range / radius)  
            
            # Calculate new position
            new_pos = np.array([center[0] + dx, center[1] + dy, center[2] + dz])

            # Ensure scatterer is above ground (z > 0)
            if new_pos[2] <= 0:
                continue # Skip this attempt and try generating another position

            # Check if position is within bounds
            if not self._is_within_bounds(new_pos):
                continue
                
            # Check minimum distance to other scatterers in this cluster
            too_close = False
            for existing in new_scatterers:
                dist = self._distance3d(new_pos, existing.pos)
                if dist < min_distance:
                    too_close = True
                    break
                    
            if too_close:
                continue
                
            # Create the scatterer
            scatterer_id = f"{cluster_id}-{len(new_scatterers)}"
            
            # Default parameters for the scatterer - can be extended with more parameters
            speed = np.random.uniform(speed_range[0], speed_range[1])  
                        
            new_scatterer = Scatterer(
                scatterer_id,
                new_pos.tolist(),
                speed,
                aggregate_scattering,
                environment_type,
                None,  # reflection_params
                self.rng
            )
            
            new_scatterers.append(new_scatterer)
            
        return new_scatterers

    def to_channel(self, snr, system=None, pathloss_model="friis", environment_type="UMa", 
                  use_aggregate_scattering=False, reflection_amplitude=None, 
                  rms_delay_spread_ns=100, shadow_fading_std_db=4.0, o2i_loss=0):
        """
        Convert this environment to a Channel object
        
        Args:
            snr: Signal-to-Noise Ratio in dB
            system: System object (optional)
            pathloss_model: Pathloss model to use (friis or 3gpp)
            environment_type: Type of environment (UMa, UMi, RMa, InF, InH)
            use_aggregate_scattering: Whether to use H_{ij} term for aggregate scattering
            reflection_amplitude: Range of reflection coefficients as [min, max]
            rms_delay_spread_ns: RMS delay spread in nanoseconds
            shadow_fading_std_db: Shadow fading standard deviation in dB
            o2i_loss: Outdoor-to-indoor penetration loss in dB
            
        Returns:
            Channel object initialised with this environment
        """
        # Ensure we have at least one TX and one RX antenna
        if not self.tx_antennas:
            raise ValueError("No TX antennas in environment")
        if not self.rx_antennas:
            raise ValueError("No RX antennas in environment")
            
        # For now, just use the first TX and RX antenna
        tx_antenna = self.tx_antennas[0]
        rx_antenna = self.rx_antennas[0]
        
        # Calculate the horizontal distance between antennas
        tx_pos = tx_antenna.get_pos()
        rx_pos = rx_antenna.get_pos()
        antenna_distance = np.sqrt((rx_pos[0] - tx_pos[0])**2 + (rx_pos[1] - tx_pos[1])**2)
        
        # Import here to avoid circular imports
        from .channel import Channel
        
        # Create a new Channel with all the environment parameters
        channel = Channel(
            antenna_distance=antenna_distance,
            tx_gain=tx_antenna.gain,
            rx_gain=rx_antenna.gain,
            snr=snr,
            system=system,
            num_cluster=len(self.clusters),  # Number of cluster locations
            speed=0.01,  # Default speed for scatterers
            use_aggregate_scattering=use_aggregate_scattering,
            environment_type=environment_type,
            pathloss_model=pathloss_model,
            o2i_loss=o2i_loss,
            reflection_amplitude=reflection_amplitude,
            cluster_density=1.0,  # We specify exact number of scatterers
            cluster_radius=max([cluster['radius'] for cluster in self.clusters]) if self.clusters else 20,
            rms_delay_spread_ns=rms_delay_spread_ns,
            shadow_fading_std_db=shadow_fading_std_db,
            tx_height=tx_pos[2],
            rx_height=rx_pos[2],
            rng=self.rng
        )
        
        # Set the antennas
        channel.tx_antenna = tx_antenna
        channel.rx_antenna = rx_antenna
        
        # Set the scatterers
        channel.scatterers = self.scatterers.copy()
        
        # Set the main cluster positions
        channel.main_cluster_positions = [
            cluster['center'].tolist()
            for cluster in self.clusters
        ]

        # initialise phase tracking for all scatterers
        channel.scatterer_phases = np.zeros(len(self.scatterers))
        
        # Generate cluster delays and powers using 3GPP power-law model
        if self.clusters:
            channel.cluster_delays, channel.cluster_powers = channel.draw_cluster_delays_and_powers(len(self.clusters))
        else:
            channel.cluster_delays = np.array([])
            channel.cluster_powers = np.array([])
            
        return channel

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

    
    def visualise_environment(self, show_bounds=True, background='white'):
        """
        Visualise the 3D environment using PyVista.
        
        Args:
            show_bounds (bool): Whether to draw the environment cube.
            background (str): Background colour (e.g., 'white', 'black').
        """
        plotter = pv.Plotter(window_size=[1200, 800]) 
        plotter.set_background(background)
        plotter.enable_lightkit() 

        # 1. Add a solid ground plane at z=0, matching environment dimensions
        ground_size_x = self.dimensions[0]
        ground_size_y = self.dimensions[1]
        ground_center_x = self.dimensions[0] / 2
        ground_center_y = self.dimensions[1] / 2
        ground_plane = pv.Plane(
            center=(ground_center_x, ground_center_y, 0),
            direction=(0, 0, 1), # Normal to the plane is Z-axis
            i_size=ground_size_x,
            j_size=ground_size_y,
            i_resolution=10,
            j_resolution=10
        )
        plotter.add_mesh(ground_plane, color='lightgrey', ambient=0.2, diffuse=0.8, specular=0.1, roughness=0.7, show_edges=False)

        # 2. Plot TX antennas (Base Station)
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
            plotter.add_mesh(antenna_base, color='red', name='TX', metallic=0.8, roughness=0.2)
            plotter.add_mesh(antenna_top, color='red', name='TX_Top', metallic=0.8, roughness=0.2)
            plotter.add_point_labels([pos[0],pos[1],pos[2]+1.5], [tx.name], point_size=0, font_size=12, text_color='red', shape_opacity=0)

        # 3. Plot RX antennas (User Equipment)
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
            plotter.add_mesh(antenna_base, color='blue', name='RX', metallic=0.8, roughness=0.2)
            plotter.add_mesh(antenna_top, color='blue', name='RX_Top', metallic=0.8, roughness=0.2)
            plotter.add_point_labels([pos[0],pos[1],pos[2]+1.0], [rx.name], point_size=0, font_size=12, text_color='blue', shape_opacity=0)

        # 4. Plot scatterers
        if self.scatterers:
            scatterer_positions = np.array([s.get_pos() for s in self.scatterers])
            plotter.add_points(
                scatterer_positions, color='green', point_size=8, render_points_as_spheres=True
            )

        # 5. Plot clusters
        for cluster in self.clusters:
            c = cluster['center']
            radius = cluster['radius']
            
            # Create the full sphere mesh
            sphere_mesh = pv.Sphere(radius=radius, center=[c[0], c[1], c[2]], theta_resolution=30, phi_resolution=30)
            
            # Clip against environment boundaries
            # Min Z (Ground plane)
            if c[2] - radius < 0:
                sphere_mesh = sphere_mesh.clip(normal=(0, 0, 1), origin=(0, 0, 0), invert=False)
            # Max Z
            if c[2] + radius > self.dimensions[2]:
                sphere_mesh = sphere_mesh.clip(normal=(0, 0, -1), origin=(0, 0, self.dimensions[2]), invert=False)
            # Min X
            if c[0] - radius < 0:
                sphere_mesh = sphere_mesh.clip(normal=(1, 0, 0), origin=(0, 0, 0), invert=False)
            # Max X
            if c[0] + radius > self.dimensions[0]:
                sphere_mesh = sphere_mesh.clip(normal=(-1, 0, 0), origin=(self.dimensions[0], 0, 0), invert=False)
            # Min Y
            if c[1] - radius < 0:
                sphere_mesh = sphere_mesh.clip(normal=(0, 1, 0), origin=(0, 0, 0), invert=False)
            # Max Y
            if c[1] + radius > self.dimensions[1]:
                sphere_mesh = sphere_mesh.clip(normal=(0, -1, 0), origin=(0, self.dimensions[1], 0), invert=False)
            
            if sphere_mesh and sphere_mesh.n_points > 0: # Ensure mesh is valid after clipping
                plotter.add_mesh(sphere_mesh, color='lightcoral', opacity=0.25)

        # 6. plot bounding box
        if show_bounds:
            cube = pv.Cube(center=(
                self.dimensions[0] / 2,
                self.dimensions[1] / 2,
                self.dimensions[2] / 2
            ), x_length=self.dimensions[0], y_length=self.dimensions[1], z_length=self.dimensions[2])
            plotter.add_mesh(cube, style='wireframe', color='dimgray', line_width=1, opacity=0.2)

        # 7. Set initial camera position for a good overview (home view)
        x_mid = self.dimensions[0] / 2
        y_mid = self.dimensions[1] / 2
        z_mid = self.dimensions[2] / 2
        initial_camera_position = [
            (x_mid - self.dimensions[0]*0.7, y_mid - self.dimensions[1]*1.5, z_mid + self.dimensions[2]*2), # Camera position
            (x_mid, y_mid, z_mid),                                                                          # Focal point (center of scene)
            (0, 0, 1)                                                                                       # View-up vector (Z is up)
        ]
        
        plotter.camera_position = initial_camera_position
        plotter.camera.zoom(1.2)
        
        # Add a home button to reset the camera
        def reset_camera():
            plotter.camera_position = initial_camera_position
            plotter.camera.zoom(1.2)
            plotter.render()
            
        plotter.add_key_event('h', reset_camera)

        # Ensure default trackball interactor for navigation
        plotter.enable_trackball_style()

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

    env.visualise_environment(show_bounds=True)
