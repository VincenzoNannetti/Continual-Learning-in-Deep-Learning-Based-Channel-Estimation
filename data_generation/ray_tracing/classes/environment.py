"""
Filename: ./data_generation/ray_tracing/classes/environment.py
Author: Vincenzo Nannetti
Date: 10/05/2025 (Refactored Date)
Description: Environment Class for the ray tracing model.
"""

import numpy as np
import pyvista as pv
import warnings
import os
import datetime
from .vector_utils import ensure_vector3d
from .scatterer.scatterer import Scatterer
from .antenna import Antenna
from .objects.building import Building
from typing import Optional, List, Dict
from scipy.spatial import cKDTree
import numpy.linalg as LA
import data_generation.ray_tracing.classes.ue_movements as ue_movements
from shapely.geometry import box, Point, Polygon
from shapely.ops import unary_union
import collections # Added for defaultdict


class Environment:
    def __init__(self, config, movement_type="random_walk"):
        self.dimensions         = ensure_vector3d(config.get('environment_dimensions', [200, 100, 100]))
        self.scatterers         = []                      # list of scatterers in the environment
        self.rx_antennas        = []                      # list of rx antennas in the environment
        self.tx_antennas        = []                      # list of tx antennas in the environment
        self.clusters           = []                      # list of cluster centers and properties
        self.buildings          = []                      # list of buildings in the environment
        self.rng                = np.random.default_rng() # random number generator
        self.movement_type      = movement_type           # Global movement type for scatterers
        self.ue_movement_config = None                    # Added to store UE movement config from main script
        self.environment_type   = config.get('environment_type', 'UMa')

        # Spatial grid for building culling
        self.building_grid_cell_size = config.get('building_grid_cell_size', 20.0) # e.g., 20m cell size
        self.building_grid = collections.defaultdict(list)
        self.grid_num_cells_x = 0
        self.grid_num_cells_y = 0
        self._initialize_building_grid_parameters()

    def _initialize_building_grid_parameters(self):
        if self.building_grid_cell_size <= 0:
            # Disable grid if cell size is invalid
            self.grid_num_cells_x = 0
            self.grid_num_cells_y = 0
            return

        self.grid_num_cells_x = int(np.ceil(self.dimensions[0] / self.building_grid_cell_size))
        self.grid_num_cells_y = int(np.ceil(self.dimensions[1] / self.building_grid_cell_size))
        self.building_grid.clear() # Ensure it's empty before repopulating if called again
        # print(f"Initialized building grid: {self.grid_num_cells_x}x{self.grid_num_cells_y} cells, cell size: {self.building_grid_cell_size}m")


    def _get_grid_cells_for_aabb(self, min_bounds_xy, max_bounds_xy):
        """Get all grid cell indices (ix, iy) that an AABB overlaps."""
        cells = set()
        if self.grid_num_cells_x == 0 or self.grid_num_cells_y == 0: # Grid disabled
            return cells

        start_ix = max(0, int(min_bounds_xy[0] / self.building_grid_cell_size))
        end_ix   = min(self.grid_num_cells_x - 1, int(max_bounds_xy[0] / self.building_grid_cell_size))
        start_iy = max(0, int(min_bounds_xy[1] / self.building_grid_cell_size))
        end_iy   = min(self.grid_num_cells_y - 1, int(max_bounds_xy[1] / self.building_grid_cell_size))

        for ix in range(start_ix, end_ix + 1):
            for iy in range(start_iy, end_iy + 1):
                cells.add((ix, iy))
        return cells

    def _add_building_to_grid(self, building_obj):
        if self.grid_num_cells_x == 0 or self.grid_num_cells_y == 0: # Grid disabled
            return

        min_b, max_b = building_obj.get_bounds()
        cells = self._get_grid_cells_for_aabb(min_b[:2], max_b[:2])
        for cell_coords in cells:
            if building_obj not in self.building_grid[cell_coords]: # Avoid duplicates
                self.building_grid[cell_coords].append(building_obj)
        # print(f"Added building {building_obj.id} to {len(cells)} grid cells. Example cell: {list(cells)[0] if cells else 'N/A'}, building count in cell: {len(self.building_grid[list(cells)[0]]) if cells else 'N/A'}")


    def _rebuild_building_grid(self):
        """Rebuilds the entire spatial grid for buildings. Call if buildings move or are added/removed."""
        self._initialize_building_grid_parameters() # Re-init params and clears grid
        if self.grid_num_cells_x == 0 or self.grid_num_cells_y == 0: # Grid disabled
            # print("Building grid is disabled or dimensions are zero. Skipping rebuild.")
            return

        for building_obj in self.buildings:
            self._add_building_to_grid(building_obj)
        # print(f"Rebuilt building grid. Total cells with buildings: {len(self.building_grid)}")


    def place_tx(self, name, position, gain_dbi):
        """Place a transmitter antenna in the environment"""
        # Check if position is within bounds
        if not self.is_within_bounds(position):
            raise ValueError(f"TX antenna position {position} is outside environment bounds {self.dimensions}")
        
        antenna = Antenna(name, "TX", gain_dbi, position)
        self.tx_antennas.append(antenna)
        return antenna
    
    def place_rx(self, name, position, gain_dbi, movement_config: Optional[Dict] = None, pavement_bounds: Optional[Dict] = None): # Added movement_config and pavement_bounds
        """Place a receiver antenna in the environment"""
        # Check if position is within bounds
        if not self.is_within_bounds(position):
            raise ValueError(f"RX antenna position {position} is outside environment bounds {self.dimensions}")
        
        antenna = Antenna(name, "RX", gain_dbi, position, movement_config=movement_config, pavement_bounds=pavement_bounds) 
        self.rx_antennas.append(antenna)
        return antenna
    
    def get_rx_antenna(self):
        return self.rx_antennas[0]

    def add_cluster(
        self,
        num_scatterers,
        radius,
        cluster_category: str, 
        height_range=None,
        scatterer_speed_range: Optional[List[float]] = None,
        use_aggregate_scattering: bool = False,
        environment_type_for_scatterer: str = "UMa",
        reflection_params_for_scatterer: Optional[dict] = None,
        carrier_freq_ghz_for_scatterer: float = 2.49,
        min_far_field: float = 5.0,
        cluster_movement_config: Optional[dict] = None
    ):
        """
        Add a cluster of scatterers around a centre point, ensuring that the
        centre is at least (min_far_field + radius) away from any antenna
        and that the cluster sphere does not intersect with buildings.
        """

        # 1) Choose one centre in the free region
        centre_xy = self.sample_cluster_centers(
            num_centers=1,
            radius=radius,
        )[0]
        
        # 2) Clamp the Z of that centre according to environment_type rules
        effective_vert_extent = height_range if height_range is not None else radius
        H = effective_vert_extent 

        # Define Z constraints based on category
        GROUND_CLUSTER_MAX_TOP_Z    = 4.0
        TYPICAL_AERIAL_MIN_Z_CENTER = 10.0
        TYPICAL_AERIAL_MAX_Z_CENTER = 50.0

        # Absolute min/max for centre_z to fit cluster in environment
        abs_min_allowable_center_z = H / 2.0
        abs_max_allowable_center_z = self.dimensions[2] - H / 2.0

        if cluster_category == "ground":
            target_min_center_z = H / 2.0 
            target_max_center_z = GROUND_CLUSTER_MAX_TOP_Z - H / 2.0 
        elif cluster_category == "aerial":
            target_min_center_z = TYPICAL_AERIAL_MIN_Z_CENTER
            target_max_center_z = TYPICAL_AERIAL_MAX_Z_CENTER
        else:
            raise ValueError(f"Unknown cluster_category: {cluster_category}. Must be 'aerial' or 'ground'.")

        # Intersect target sampling range with absolute allowable range
        actual_sample_min_z = max(target_min_center_z, abs_min_allowable_center_z)
        actual_sample_max_z = min(target_max_center_z, abs_max_allowable_center_z)

        if actual_sample_max_z < actual_sample_min_z:
            print(f"Warning: Z-constraints for cluster category '{cluster_category}' with height {H:.2f}m are conflicting or out of environment bounds.")
            print(f"  Target range for center_z: [{target_min_center_z:.2f}, {target_max_center_z:.2f}]")
            print(f"  Absolute allowable range for center_z: [{abs_min_allowable_center_z:.2f}, {abs_max_allowable_center_z:.2f}]")
            if abs_max_allowable_center_z < abs_min_allowable_center_z: 
                print("  Cluster height exceeds environment Z dimension. Placing center at environment midpoint.")
                centre_z = self.dimensions[2] / 2.0
            else:
                print("  Placing cluster at midpoint of allowable Z range.")
                centre_z = (abs_min_allowable_center_z + abs_max_allowable_center_z) / 2.0
        else:
            centre_z = float(np.random.uniform(actual_sample_min_z, actual_sample_max_z))
        
        centre = np.array([centre_xy[0], centre_xy[1], centre_z], dtype=float)

        # 3) Record the cluster meta‐data
        height_range_for_scatterers = height_range if height_range is not None else radius
        self.clusters.append({
            'center':         centre,
            'num_scatterers': num_scatterers,
            'radius':         radius,
            'height_range':   height_range_for_scatterers,
            'shared_flock_velocity': None
        })

        # 4) Generate scatterers around that sampled centre
        new_scatterers = self.create_scatterers(
            centre,
            num_scatterers,
            radius,
            height_range_for_scatterers,
            min_distance                = getattr(self, 'min_distance_between_scatterers', 0.5),
            speed_range                 = scatterer_speed_range,
            aggregate_scattering        = use_aggregate_scattering,
            environment_type            = environment_type_for_scatterer,
            reflection_params_scatterer = reflection_params_for_scatterer,
            carrier_freq_ghz            = carrier_freq_ghz_for_scatterer,
            min_far_field               = min_far_field,
            cluster_movement_config     = cluster_movement_config
        )

        self.scatterers.extend(new_scatterers)
        return new_scatterers

    def add_building(self, 
                     id_prefix: str,
                     position, 
                     dimensions, 
                     reflection_coefficient=0.0, 
                     material="concrete"):
        """
        Add a building to the environment.
        
        Args:
            id_prefix: Prefix for the building ID (e.g., "building")
            position: [x, y, z] center position of the building
            dimensions: [width, length, height] dimensions of the building
            reflection_coefficient: How much signal is reflected (0-1)
            material: Building material (affects reflection properties)
            
        Returns:
            The created Building object
        """
        position   = ensure_vector3d(position)
        dimensions = ensure_vector3d(dimensions)
                
        # Ensure building doesn't extend beyond environment bounds 
        building_min_bounds = np.array(position) - np.array(dimensions) / 2
        building_max_bounds = np.array(position) + np.array(dimensions) / 2
        
        # Check if the building is entirely outside the environment dimensions
        if (building_max_bounds[0] < 0 or building_min_bounds[0] > self.dimensions[0] or
            building_max_bounds[1] < 0 or building_min_bounds[1] > self.dimensions[1] or
            building_max_bounds[2] < 0 or building_min_bounds[2] > self.dimensions[2]):
            warnings.warn(f"Building at {position} with dimensions {dimensions} is entirely outside environment bounds and will not be added.")
            return None

        # Clip building to environment bounds if it extends beyond
        clipped_min_bounds = np.maximum(building_min_bounds, np.array([0,0,0]))
        clipped_max_bounds = np.minimum(building_max_bounds, self.dimensions)

        # Recalculate position and dimensions based on clipped bounds
        clipped_position   = (clipped_min_bounds + clipped_max_bounds) / 2
        clipped_dimensions =  clipped_max_bounds - clipped_min_bounds

        if not np.allclose(clipped_dimensions, dimensions):
             warnings.warn(f"Building at {position} with dimensions {dimensions} extends beyond environment bounds and has been clipped to fit. New dimensions: {clipped_dimensions}, new position: {clipped_position}")
             position   = clipped_position
             dimensions = clipped_dimensions

        # Ensure dimensions are positive after clipping
        if not np.all(dimensions > 1e-6):
            warnings.warn(f"Building at {position} with dimensions {dimensions} resulted in non-positive dimensions after clipping and will not be added.")
            return None
            
        # Create building ID
        building_id = f"{id_prefix}-{len(self.buildings)}"
        
        # Create the building
        building = Building(
            building_id,
            position.tolist(),
            dimensions.tolist(), 
            reflection_coefficient=reflection_coefficient,
            material=material
        )
        
        self.buildings.append(building)
        self._add_building_to_grid(building) # Add to spatial grid
        return building

    def check_building_intersection(self, ray_origin, ray_direction, max_distance=float('inf'), exclude_building_id: Optional[str] = None):
        """
        Check if a ray intersects with any building in the environment.
        
        Args:
            ray_origin: Origin point of the ray
            ray_direction: Direction vector of the ray
            max_distance: Maximum distance to check for intersections
            exclude_building_id: Optional ID of a building to exclude from the check.
            
        Returns:
            Tuple of (intersects, distance, building, normal)
            - intersects: True if the ray intersects any building
            - distance: Distance to closest intersection point
            - building: The building that was intersected
            - normal: Normal vector at intersection point
        """
        closest_distance = float('inf')
        closest_building = None
        closest_normal   = None
        
        _ray_origin    = np.array(ensure_vector3d(ray_origin),    dtype=float)
        _ray_direction = np.array(ensure_vector3d(ray_direction), dtype=float)
        
        norm_direction = np.linalg.norm(_ray_direction)
        if norm_direction > 1e-9:
            _ray_direction_normalized = _ray_direction / norm_direction
        else:
            return False, None, None, None

        candidate_buildings = []
        if self.grid_num_cells_x > 0 and self.grid_num_cells_y > 0: # Grid is enabled
            ray_end_point = _ray_origin + _ray_direction_normalized * max_distance
            
            # Determine AABB of the ray segment for grid cell lookup
            ray_min_xy = np.minimum(_ray_origin[:2], ray_end_point[:2])
            ray_max_xy = np.maximum(_ray_origin[:2], ray_end_point[:2])
            
            grid_cells_to_check = self._get_grid_cells_for_aabb(ray_min_xy, ray_max_xy)
            
            # More accurate: DDA or Bresenham to find cells ray actually passes through
            # For simplicity, using AABB of ray for now. This might over-select cells for diagonal rays.
            # A more precise method would trace the ray through the grid.
            # Example of a simple DDA-like cell traversal (can be improved):
            # This part needs a proper line-grid traversal algorithm for accuracy.
            # For now, let's stick to AABB of ray segment, it's a common broad-phase.
            
            seen_buildings = set()
            if not grid_cells_to_check: # Ray AABB is outside grid or grid empty
                # This case should ideally not happen if ray is within env_dims.
                # Fallback to checking all buildings if ray doesn't map to any cell,
                # or if grid is small / ray is large.
                # print(f"Warning: Ray AABB {ray_min_xy} to {ray_max_xy} did not map to any grid cells. Falling back to all buildings.")
                candidate_buildings = self.buildings
            else:
                for cell_coords in grid_cells_to_check:
                    for building_obj in self.building_grid.get(cell_coords, []):
                        if building_obj not in seen_buildings:
                            candidate_buildings.append(building_obj)
                            seen_buildings.add(building_obj)
                if not candidate_buildings: # No buildings in the ray's path cells
                    return False, None, None, None
            # print(f"Ray check: origin={_ray_origin[:2]}, end={ray_end_point[:2]}. Cells checked: {len(grid_cells_to_check)}. Candidate buildings: {len(candidate_buildings)} out of {len(self.buildings)}")
        else: # Grid is disabled, check all buildings
            candidate_buildings = self.buildings


        for building_obj in candidate_buildings:  # Iterate over potentially smaller list
            if exclude_building_id and building_obj.id == exclude_building_id:
                continue

            intersects, distance, normal = building_obj.intersects_ray(_ray_origin, _ray_direction_normalized) # Pass normalized
            if intersects and distance is not None and distance < closest_distance and distance <= max_distance:
                closest_distance = distance
                closest_building = building_obj
                closest_normal = normal
                
        if closest_building is None:
            return False, None, None, None
            
        return True, closest_distance, closest_building, closest_normal
    
    def sample_cluster_centers(
        self,
        radius: float,
        num_centers: int,
    ) -> List[np.ndarray]:
        """
        Sample cluster centers within the environment, avoiding buildings,
        with a preference for the street canyon area.
        """
        # Build 2D env rectangle
        env_dims = (self.dimensions[0], self.dimensions[1])
        env_rect = box(0, 0, *env_dims)

        # Buffer every building footprint by radius
        buffered = []
        for b in self.buildings:
            (xmin, ymin, _), (xmax, ymax, _) = b.get_bounds()
            buffered.append(box(xmin, ymin, xmax, ymax).buffer(radius))

        forbidden   = unary_union(buffered) if buffered else Polygon()
        free_region = env_rect.difference(forbidden)

        # Sample until we have num_centers
        centers, attempts = [], 0
        
        # Define the street canyon for sampling cluster centers
        # X-dimension: full environment width, ensuring cluster fits
        canyon_intended_x_min = 0.0 
        canyon_intended_x_max = self.dimensions[0] 
        
        # Y-dimension: centered in the environment, with a defined half-width
        street_canyon_y_center   = self.dimensions[1] / 2.0
        street_canyon_half_width = 10.0 

        # Effective sampling bounds, ensuring the cluster of 'radius' fits within the canyon & environment
        effective_sample_x_min = max(canyon_intended_x_min + radius, radius)
        effective_sample_x_max = min(canyon_intended_x_max - radius, self.dimensions[0] - radius)

        effective_sample_y_min = max(street_canyon_y_center - street_canyon_half_width + radius, radius) 
        effective_sample_y_max = min(street_canyon_y_center + street_canyon_half_width - radius, self.dimensions[1] - radius) 

        # Increase attempts if the sampling region is smaller or more constrained
        max_attempts_multiplier = 20 

        while len(centers) < num_centers and attempts < num_centers * max_attempts_multiplier:
            attempts += 1
            
            x_candidate, y_candidate = -1.0, -1.0 

            # Ensure valid sampling range 
            if (effective_sample_x_max <= effective_sample_x_min or
                effective_sample_y_max <= effective_sample_y_min):
                fb_minx, fb_miny, fb_maxx, fb_maxy = free_region.bounds
                if fb_maxx <= fb_minx or fb_maxy <= fb_miny:
                    warnings.warn(f"Cannot sample cluster centers: free_region has no valid span. Check building layout, cluster radius, and environment dimensions.")
                    break 
                x_candidate = np.random.uniform(fb_minx, fb_maxx)
                y_candidate = np.random.uniform(fb_miny, fb_maxy)
            else:
                # Sample within the defined effective canyon area
                x_candidate = np.random.uniform(effective_sample_x_min, effective_sample_x_max)
                y_candidate = np.random.uniform(effective_sample_y_min, effective_sample_y_max)
            
            if free_region.contains(Point(x_candidate, y_candidate)):
                intended_canyon_y_min = street_canyon_y_center - street_canyon_half_width
                intended_canyon_y_max = street_canyon_y_center + street_canyon_half_width
                if intended_canyon_y_min <= y_candidate <= intended_canyon_y_max:
                    centers.append(np.array([x_candidate, y_candidate, 0.0])) # z=0.0 initially, add_cluster handles final Z

        if len(centers) < num_centers:
            raise RuntimeError(f"Only got {len(centers)} / {num_centers} cluster centers. Try reducing cluster radius, number of clusters, or check building density within the canyon.")
        return centers

    def create_scatterers(self, center, num_scatterers, radius, height_range, min_distance=0.5, speed_range=None, aggregate_scattering=False, 
                           environment_type="UMa", reflection_params_scatterer=None, carrier_freq_ghz=2.49, min_far_field=5.0,
                           cluster_movement_config: Optional[dict] = None):
        """
        Create a cluster of scatterers around a center point, using a KD-tree
        to enforce a minimum inter-scatterer spacing.
        """
        if speed_range is None:
            speed_range = [0, 0]

        new_scatterers = []
        points         = []        # list of accepted positions
        tree           = None      # will become our KD-tree
        attempts       = 0
        max_attempts   = num_scatterers * 10

        # Determine the cluster_id for the scatterers being created
        cluster_id_for_current_op = len(self.clusters) - 1

        generated_shared_flock_velocity_for_cluster = None
        flock_speed_for_this_cluster = 0.0 

        # Get movement parameters from config
        movement_params = {}
        if cluster_movement_config and 'movement_params' in cluster_movement_config:
            movement_params = cluster_movement_config['movement_params'].get(self.movement_type, {})

        if self.movement_type == "flocking":
            # Use the general speed_range for the flock's speed magnitude
            flock_speed_for_this_cluster = self.rng.uniform(speed_range[0], speed_range[1]) if speed_range else 0.0
            
            generated_shared_flock_velocity_for_cluster = np.zeros(3, dtype=float) 
            flock_direction = None

            if flock_speed_for_this_cluster > 1e-9: 
                # Get direction from movement_params if available
                user_specified_direction = movement_params.get("flock_movement_direction")

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

                if flock_direction is None: 
                    flock_direction_candidate = self.rng.standard_normal(3)
                    norm = np.linalg.norm(flock_direction_candidate)
                    if norm > 1e-9:
                        flock_direction = flock_direction_candidate / norm
                    else: 
                        flock_direction = np.array([1.0, 0.0, 0.0])
                
                generated_shared_flock_velocity_for_cluster = flock_direction * flock_speed_for_this_cluster
            
            # Store this shared velocity in the cluster's data structure
            if cluster_id_for_current_op < len(self.clusters):
                self.clusters[cluster_id_for_current_op]['shared_flock_velocity'] = generated_shared_flock_velocity_for_cluster.copy()
            else:
                warnings.warn(f"Flocking: Cluster index {cluster_id_for_current_op} out of bounds when trying to set shared_flock_velocity.")

        while len(new_scatterers) < num_scatterers and attempts < max_attempts:
            attempts += 1

            u       = self.rng.random()
            r       = radius * u**(1/3)
            cos_phi = self.rng.uniform(-1, 1)
            phi     = np.arccos(cos_phi)
            theta   = self.rng.uniform(0, 2*np.pi)

            dx      = r * np.sin(phi) * np.cos(theta)
            dy      = r * np.sin(phi) * np.sin(theta)
            dz      = r * np.cos(phi) * (height_range / radius)
            new_pos = np.array([center[0] + dx, center[1] + dy, center[2] + dz])

            # 1) ensure above ground and inside bounds
            if new_pos[2] <= 0 or not self.is_within_bounds(new_pos):
                continue

            # 2) far-field margin from antennas
            if any(self.distance3d(new_pos, ant.get_pos()) < min_far_field
                for ant in self.tx_antennas + self.rx_antennas):
                continue

            is_inside_building = False
            if self.buildings: 
                for building_obj in self.buildings:
                    b_min, b_max = building_obj.get_bounds()
                    # A simple AABB check for the point:
                    if (b_min[0] <= new_pos[0] <= b_max[0] and
                        b_min[1] <= new_pos[1] <= b_max[1] and
                        b_min[2] <= new_pos[2] <= b_max[2]):
                        is_inside_building = True
                        break 
            
            if is_inside_building:
                continue 

            # 3) inter-scatterer spacing via KD-tree
            if tree is not None:
                # if any existing point is within min_distance, reject
                if tree.query_ball_point(new_pos, r=min_distance):
                    continue

            # accept this position 
            points.append(new_pos)
            tree         = cKDTree(points) 
            scatterer_id = f"{cluster_id_for_current_op}-{len(new_scatterers)}"
            current_scatterer_speed = flock_speed_for_this_cluster if self.movement_type == "flocking" else self.rng.uniform(speed_range[0], speed_range[1])
            
            # Set up movement-specific parameters based on movement type
            individual_scatterer_movement_params = {}
            
            if self.movement_type == "random_walk":
                individual_scatterer_movement_params = {
                    'direction_change_prob': movement_params.get('direction_change_prob', 0.4),
                    'max_angle_change': np.radians(movement_params.get('max_angle_change', 45))
                }
            elif self.movement_type == "linear":
                if 'direction' in movement_params:
                    individual_scatterer_movement_params = {
                        'direction': movement_params['direction']
                    }
            
            s = Scatterer(
                scatterer_id,
                new_pos.tolist(),
                current_scatterer_speed, 
                use_aggregate_scattering = aggregate_scattering,
                environment_type         = environment_type,
                reflection_params        = reflection_params_scatterer,
                frequency_ghz            = carrier_freq_ghz,
                movement_type            = self.movement_type, 
                movement_specific_params = individual_scatterer_movement_params 
            )

            # If flocking, override the scatterer's velocity_vector with the shared one
            if self.movement_type == "flocking" and generated_shared_flock_velocity_for_cluster is not None:
                s.velocity_vector = generated_shared_flock_velocity_for_cluster.copy()

            new_scatterers.append(s)
        return new_scatterers

    def advance(self, dt):
        """Advance the environment by a time step dt"""

        if self.movement_type == "flocking":
            # Phase 1: Synchronise flock scatterers to their cluster's shared velocity
            for cluster_idx, cluster_data in enumerate(self.clusters):
                shared_vel = cluster_data.get('shared_flock_velocity')
                if shared_vel is not None: # This cluster is a flock
                    for s in self.scatterers:
                        if s.cluster_id == cluster_idx:
                            s.velocity_vector = shared_vel.copy() 
        
        # Phase 2: All scatterers update their position based on their current velocity_vector
        for s in self.scatterers:
            s.update_pos(dt, movement_type=self.movement_type, environment_dims=self.dimensions)
        
        for rx_ant in self.rx_antennas:
            if hasattr(rx_ant, 'update_pos') and callable(getattr(rx_ant, 'update_pos')):
                # Prepare building data for the UE movement module
                building_bboxes = []
                if self.buildings:
                    # Pass only 2D bounding boxes for horizontal collision
                    building_bboxes = [{'xmin': b.min_bounds[0], 'xmax': b.max_bounds[0],
                                        'ymin': b.min_bounds[1], 'ymax': b.max_bounds[1]}
                                       for b in self.buildings]
                rx_ant.update_pos(dt, self.dimensions, ue_movements, buildings=building_bboxes)

        if self.movement_type == "flocking":
            for cluster_idx, cluster_data in enumerate(self.clusters):
                original_flock_velocity = cluster_data.get('shared_flock_velocity')
                if original_flock_velocity is None: 
                    continue

                new_collective_flock_velocity = original_flock_velocity.copy()
                reflection_occurred_on_axis = [False, False, False]

                for s in self.scatterers:
                    if s.cluster_id == cluster_idx:
                        for axis_i in range(3):
                            if original_flock_velocity[axis_i] != 0 and \
                               (s.velocity_vector[axis_i] * original_flock_velocity[axis_i] < -1e-9): 
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
        
        if self.movement_type == "random_walk": 
            num_steps = 10
            dt_step = time_period / num_steps
            for _ in range(num_steps):
                self.advance(dt_step) 
        else:
            self.advance(time_period) 
        
        # Get new positions
        new_positions = np.array([s.get_pos() for s in self.scatterers])
        
        # Visualise both sets
        plotter = pv.Plotter(window_size=[1200, 800])
        plotter.set_background(background)
        
        # Draw ground plane
        ground_size_x   = self.dimensions[0]
        ground_size_y   = self.dimensions[1]
        ground_center_x = self.dimensions[0] / 2
        ground_center_y = self.dimensions[1] / 2
        ground_plane    = pv.Plane(
                                center=(ground_center_x, ground_center_y, 0),
                                direction=(0, 0, 1),
                                i_size=ground_size_x,
                                j_size=ground_size_y,
                                i_resolution=10,
                                j_resolution=10)
        plotter.add_mesh(ground_plane, color='lightgrey', ambient=0.2, diffuse=0.8, specular=0.1, roughness=0.7)
        
        # Draw TX/RX antennas
        tx_positions = []
        for tx in self.tx_antennas:
            log_pos = np.array(tx.get_pos()) 
            tx_positions.append(log_pos) 

            tx_mast_height = 3.0
            tx_mast_radius = 0.2 
            visual_y_offset = tx_mast_radius

            drawn_pos_x = log_pos[0]
            drawn_pos_y = log_pos[1] + visual_y_offset
            drawn_pos_z = log_pos[2]

            # Mast (cylinder) 
            mast_center_z = drawn_pos_z - tx_mast_height / 2.0
            antenna_base_mast = pv.Cylinder(
                center=[drawn_pos_x, drawn_pos_y, mast_center_z],
                direction=[0, 0, 1],
                radius=tx_mast_radius,
                height=tx_mast_height
            )
            plotter.add_mesh(antenna_base_mast, color='darkred', name=f'TX_Mast_{tx.name}', metallic=0.7, roughness=0.3)

            # Cone (top part)
            antenna_top = pv.Cone(
                center=[drawn_pos_x, drawn_pos_y, drawn_pos_z], 
                direction=[0, 0, 1],
                radius=0.7, 
                height=1.2 
            )
            plotter.add_mesh(antenna_top, color='red', name=f'TX_Top_{tx.name}', metallic=0.8, roughness=0.2)
            plotter.add_point_labels([drawn_pos_x, drawn_pos_y, drawn_pos_z + 1.5], [tx.name], point_size=0, font_size=12, text_color='red', shape_opacity=0)

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
            plotter.add_mesh(antenna_base, color='blue', name=f'RX_base_{rx.name}', metallic=0.8, roughness=0.2)
            plotter.add_mesh(antenna_top, color='blue', name=f'RX_Top_{rx.name}', metallic=0.8, roughness=0.2)
            plotter.add_point_labels([pos[0],pos[1],pos[2]+1.0], [rx.name], point_size=0, font_size=12, text_color='blue', shape_opacity=0)
        
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

    def get_scatterer_snapshot(self, tx_pos: Optional[np.ndarray] = None, rx_pos: Optional[np.ndarray] = None):
        """Return positions, velocities, reflection_coeffs, speeds, and visibility mask as NumPy arrays."""
        num_scatterers_env = len(self.scatterers)
        if num_scatterers_env == 0:
            # Return empty arrays with correct shapes if no scatterers
            return (
                np.empty((0, 3), dtype=float),
                np.empty((0, 3), dtype=float),
                np.empty((0,), dtype=complex),
                np.empty((0,), dtype=float),
                np.empty((0,), dtype=bool), # Empty visibility mask
            )

        positions         = np.array([s.pos for s in self.scatterers], dtype=float)
        velocities        = np.array([s.velocity_vector for s in self.scatterers], dtype=float)
        reflection_coeffs = np.array([s.get_reflection_coeff() for s in self.scatterers], dtype=complex)
        speeds            = np.array([s.speed for s in self.scatterers], dtype=float)

        if tx_pos is not None and rx_pos is not None:
            _tx_pos = np.asarray(tx_pos, dtype=float)
            _rx_pos = np.asarray(rx_pos, dtype=float)
            
            visible_mask = np.ones(num_scatterers_env, dtype=bool) # Start assuming all are visible
            for i in range(num_scatterers_env):
                s_pos = positions[i] # Use already fetched position
                
                # Check path Tx -> Scatterer
                blocked_tx_s = self.path_blocked(_tx_pos, s_pos)
                if blocked_tx_s:
                    visible_mask[i] = False
                    continue # No need to check second leg if first is blocked
                
                # Check path Scatterer -> Rx
                blocked_s_rx = self.path_blocked(s_pos, _rx_pos)
                if blocked_s_rx:
                    visible_mask[i] = False
        else:
            # If tx_pos or rx_pos is not provided, assume all are visible
            visible_mask = np.ones(num_scatterers_env, dtype=bool)
            
        return positions, velocities, reflection_coeffs, speeds, visible_mask
    
    def is_within_bounds(self, position):
        """Check if position is within the environment bounds"""
        x, y, z = ensure_vector3d(position)
            
        return (0 <= x <= self.dimensions[0] and
                0 <= y <= self.dimensions[1] and
                0 <= z <= self.dimensions[2])
       
    def distance3d(self, pos1, pos2):
        """Calculate 3D distance between two positions"""
        p1 = ensure_vector3d(pos1)
        p2 = ensure_vector3d(pos2)
        return np.linalg.norm(p2 - p1)

    def los_basis(self, tx_pos, rx_pos):
        """Return unit vector along BS->UE line and two orthogonal unit vectors spanning its normal plane."""
        v     = rx_pos - tx_pos
        e_los = v / np.linalg.norm(v)
        # Find a vector not parallel to e_los, use it to build an orthonormal basis
        tmp = np.array([1,0,0]) if abs(e_los[0]) < 0.9 else np.array([0,1,0])
        e1  = np.cross(e_los, tmp);  e1 /= np.linalg.norm(e1)
        e2  = np.cross(e_los, e1)
        return e_los, e1, e2

    def get_valid_window_reflections(self, tx: np.ndarray, rx: np.ndarray):
        """
        Return a list of dicts {pos, coeff, distance} for every first-order
        specular reflection Tx -> window -> Rx that is
          - inside the pane rectangle,
          - below the pane's angle threshold,
          - not blocked by any other building.
        """
        reflections = []
        for b in self.buildings:
            for w in b.get_windows():
                # Ensure window normal is a unit vector for mirror reflection calculation
                window_normal_unnormalised = w.get_normal()
                norm_window_normal = LA.norm(window_normal_unnormalised)
                if norm_window_normal < 1e-9: # Skip if window normal is a zero vector
                    continue
                n = window_normal_unnormalised / norm_window_normal
                p0  = w.get_center()

                # mirror method – reflect Tx through the pane
                tx_m = tx - 2 * np.dot(tx - p0, n) * n
                dir_line = rx - tx_m
                denom = np.dot(dir_line, n)
                if abs(denom) < 1e-9:          # line parallel to plane
                    continue
                t = np.dot(p0 - tx_m, n) / denom
                if not (0.0 < t < 1.0):        # intersection outside segment tx_m --- rx
                    continue
                p_ref = tx_m + t * dir_line    # reflection point on the infinite plane of the window

                # inside rectangle?
                if not w.contains_point(p_ref):
                    continue

                # incidence angle OK?
                vec_tx_pref = p_ref - tx
                norm_vec_tx_pref = LA.norm(vec_tx_pref)
                if norm_vec_tx_pref < 1e-9: 
                    continue

                d_in = vec_tx_pref / norm_vec_tx_pref 
                
                current_reflection_coeff = w.get_reflection_coefficient(d_in)
                if np.abs(current_reflection_coeff) < 1e-12: 
                    continue

                dist_tx_pref = norm_vec_tx_pref 
                seg1_block = False 
                if dist_tx_pref > 1e-6: 
                    seg1_block = self.check_building_intersection(tx, d_in, 
                                                              max_distance=dist_tx_pref - 1e-6, 
                                                              exclude_building_id=b.id)[0]
                vec_pref_rx = rx - p_ref
                dist_pref_rx = LA.norm(vec_pref_rx)
                seg2_block = False 
                if dist_pref_rx > 1e-6: 
                    dir_pref_rx = vec_pref_rx / dist_pref_rx
                    seg2_block = self.check_building_intersection(p_ref, dir_pref_rx, 
                                                              max_distance=dist_pref_rx - 1e-6, 
                                                              exclude_building_id=b.id)[0]
                                                              
                if seg1_block or seg2_block:
                    continue

                reflections.append({
                    "pos":      p_ref,
                    "coeff":    current_reflection_coeff,
                    "distance": dist_tx_pref + dist_pref_rx
                })
        return reflections

    def add_path(self, plotter, start, end, colour='yellow',
                  name='ray', line_width=1.5, opacity=1.0, show_blockage_visuals=True):
        start = np.array(start, dtype=float)
        end   = np.array(end, dtype=float)

        # Direction and total distance
        vec   = end - start
        dist  = np.linalg.norm(vec)
        if dist < 1e-9:
            return  
        d_hat = vec / dist

        # Check for blockage IF show_blockage_visuals is enabled
        if show_blockage_visuals:
            check_max_distance = max(0.0, dist - 1e-6) 
            is_blocked_vis, _t_hit, _building_obj, _normal_hit = self.check_building_intersection(
                start, d_hat, max_distance=check_max_distance)
            
            if is_blocked_vis:
                return 
        line_full = pv.Line(start, end)
        plotter.add_mesh(line_full, color=colour, line_width=line_width, opacity=opacity,
                         name=name)

    def path_blocked(self, p1: np.ndarray, p2: np.ndarray, exclude_bldg_id: Optional[str] = None) -> bool:
        """Check if the path between p1 and p2 is blocked by a building.
        Assumes p1 and p2 are already NumPy arrays.
        """
        vec  = p2 - p1 
        dist = np.linalg.norm(vec)

        if dist < 1e-6:
            return False  
        
        direction = vec / dist
        # Check slightly less than full distance to avoid issues with endpoint being on a surface
        is_blocked, _intersect_dist, _building, _normal = self.check_building_intersection(
            p1, direction, max_distance=dist - 1e-6, exclude_building_id=exclude_bldg_id
        )
        return is_blocked

    def visualise_environment(self, show_bounds=True, background='white', 
                              show_rays=True, show_movement=True, movement_time=2.0, 
                              max_rays=100, show_grid=True, 
                              show_cluster_radii_initially=True,
                              show_blockage_visuals: bool = True):
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
            show_blockage_visuals (bool): Whether to visually indicate ray blockage by buildings.
        """
        plotter = pv.Plotter(window_size=[1200, 800]) 
        plotter.set_background(background)
        plotter.enable_lightkit() 

        # Add road
        road_width      = 12.0
        main_street_y   = self.dimensions[1] / 2
        main_road_start = [0, main_street_y, 0.01]
        main_road_end   = [self.dimensions[0], main_street_y, 0.01]
        main_road_mesh  = self.create_road_mesh(main_road_start, main_road_end, road_width)
        if main_road_mesh:
            plotter.add_mesh(main_road_mesh, color='dimgray', name='main_street', ambient=0.3, diffuse=0.7)

            # Add road markings 
            marking_color  = 'white'
            dash_length    = 4.0
            gap_length     = 6.0
            segment_length = dash_length + gap_length
            line_z_offset  = 0.015 
            num_segments   = int(self.dimensions[0] / segment_length)

            for i in range(num_segments):
                dash_start_x = i * segment_length
                dash_end_x   = dash_start_x + dash_length
                if dash_end_x > self.dimensions[0]:
                    dash_end_x = self.dimensions[0]
                    if dash_start_x >= self.dimensions[0]:
                        break 
                p1 = [dash_start_x, main_street_y, line_z_offset]
                p2 = [dash_end_x, main_street_y, line_z_offset]
                
                if np.linalg.norm(np.array(p2) - np.array(p1)) > 1e-3:
                    dash_line = pv.Line(p1, p2)
                    plotter.add_mesh(dash_line, color=marking_color, line_width=5, name=f'center_dash_{i}')

        # Initialise lists for antenna positions before the loops
        tx_positions = []
        rx_positions = []

        # Add a solid ground plane at z=0 (top view)
        ground_size_x   = self.dimensions[0]
        ground_size_y   = self.dimensions[1]
        ground_center_x = self.dimensions[0] / 2
        ground_center_y = self.dimensions[1] / 2
        ground_plane    = pv.Plane(
            center=(ground_center_x, ground_center_y, 0),
            direction=(0, 0, 1),
            i_size=ground_size_x,
            j_size=ground_size_y,
            i_resolution=5,
            j_resolution=5
        )
        plotter.add_mesh(ground_plane, color='lightgrey', ambient=0.2, diffuse=0.8, specular=0.1, roughness=0.7, show_edges=False)
        
        # Plot TX antennas (Base Station)
        for tx in self.tx_antennas:
            tip  = np.array(tx.get_pos())  
            base = tip - np.array([0,0,3.0])
            tx_positions.append(tip)
            mast_radius = 0.2
            mast_height = 3.0
            mast_center = base + np.array([0,0,mast_height/2])
            antenna_base_mast = pv.Cylinder(
                center=mast_center.tolist(),
                direction=[0,0,1],
                radius=mast_radius,
                height=mast_height
            )
            plotter.add_mesh(antenna_base_mast, color='darkred', name=f'TX_Mast_{tx.name}')

            # top cone at `tip`
            antenna_top = pv.Cone(
                center=tip.tolist(),
                direction=[0,0,1],
                radius=0.7,
                height=1.2
            )
            plotter.add_mesh(antenna_top, color='red', name=f'TX_Top_{tx.name}')

        # Plot RX antennas (User Equipment)
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
            plotter.add_mesh(antenna_base, color='blue', name=f'RX_base_{rx.name}', metallic=0.8, roughness=0.2)
            plotter.add_mesh(antenna_top, color='blue', name=f'RX_Top_{rx.name}', metallic=0.8, roughness=0.2)
            plotter.add_point_labels([pos[0],pos[1],pos[2]+1.0], [rx.name], point_size=0, font_size=12, text_color='blue', shape_opacity=0)
        
        # Plot ray paths
        if show_rays and tx_positions and rx_positions:
            # Add LOS ray between TX and RX
            for tx_pos_single in tx_positions:
                for rx_pos_single in rx_positions:
                    self.add_path(plotter, tx_pos_single, rx_pos_single, 
                                   colour='yellow', name='los_ray', line_width=3,
                                   show_blockage_visuals=show_blockage_visuals)
            
            if tx_positions and rx_positions: 
                tx_pos_for_rays = tx_positions[0]
                rx_pos_for_rays = rx_positions[0]
            
                # Window Reflection Rays (always check for these, regardless of scatterers)
                if self.buildings:
                    valid_window_reflections = self.get_valid_window_reflections(tx_pos_for_rays, rx_pos_for_rays)
                    num_window_rays_to_draw = min(len(valid_window_reflections), max_rays)
                    window_ray_step = max(1, len(valid_window_reflections) // num_window_rays_to_draw if num_window_rays_to_draw > 0 else len(valid_window_reflections) + 1)
                    for i in range(0, len(valid_window_reflections), window_ray_step):
                        reflection_data = valid_window_reflections[i]
                        p_ref = reflection_data["pos"]
                        self.add_path(plotter, tx_pos_for_rays, p_ref, 
                                       colour='purple', name=f'ray_tx_win_{i}', line_width=1.5, opacity=0.7,
                                       show_blockage_visuals=show_blockage_visuals)
                        self.add_path(plotter, p_ref, rx_pos_for_rays, 
                                       colour='deeppink', name=f'ray_win_rx_{i}', line_width=1.5, opacity=0.7,
                                       show_blockage_visuals=show_blockage_visuals)
        
        # Plot scatterers
        if self.scatterers:
            scatterer_positions = np.array([s.get_pos() for s in self.scatterers])
            points = plotter.add_points(
                scatterer_positions, color='green', point_size=8, 
                render_points_as_spheres=True, name='scatterers'
            )
            
            # Plot scatterer ray paths
            if show_rays and tx_positions and rx_positions:
                if tx_positions and rx_positions: 
                    tx_pos_for_rays = tx_positions[0]
                    rx_pos_for_rays = rx_positions[0]
                
                    # Scatterer Rays
                    if scatterer_positions.any(): 
                        num_rays_to_draw = min(len(scatterer_positions), max_rays)
                        step = max(1, len(scatterer_positions) // num_rays_to_draw if num_rays_to_draw > 0 else len(scatterer_positions) + 1)
                        for i in range(0, len(scatterer_positions), step):
                            s_pos = scatterer_positions[i]
                            alpha = max(0.1, 1.0 - i / len(scatterer_positions))
                            self.add_path(plotter, tx_pos_for_rays, s_pos, 
                                           colour='orange', name=f'ray_tx_sc_{i}', line_width=1, opacity=alpha,
                                           show_blockage_visuals=show_blockage_visuals)
                            self.add_path(plotter, s_pos, rx_pos_for_rays, 
                                           colour='cyan', name=f'ray_sc_rx_{i}', line_width=1, opacity=alpha,
                                           show_blockage_visuals=show_blockage_visuals)
            
            # Show movement paths if requested
            if show_movement and self.scatterers:
                # Save original scatterers
                import copy
                scatterer_backup = copy.deepcopy(self.scatterers)
                # Save original RX antenna states if they are to be moved and restored
                rx_antennas_backup = copy.deepcopy(self.rx_antennas)
                original_movement_type = self.movement_type 
                
                # Store initial positions of RX antennas if they are mobile
                initial_rx_positions = []
                if self.rx_antennas:
                    initial_rx_positions = [rx.get_pos() for rx in self.rx_antennas]

                # Create future positions after movement time
                self.advance(movement_time)
                future_scatterer_positions = np.array([s.get_pos() for s in self.scatterers])
                
                future_rx_positions = []
                if self.rx_antennas:
                    future_rx_positions = [rx.get_pos() for rx in self.rx_antennas]
                
                # Limit the number of movement paths for performance
                num_scatterer_paths_to_draw = min(len(scatterer_positions), max_rays) 
                path_step = max(1, len(scatterer_positions) // num_scatterer_paths_to_draw if num_scatterer_paths_to_draw > 0 else len(scatterer_positions) + 1)

                # Draw scatterer movement paths as lines 
                movement_actors = []
                for i in range(0, len(scatterer_positions), path_step):
                    if i < len(future_scatterer_positions):
                        line = pv.Line(scatterer_positions[i], future_scatterer_positions[i])
                        actor = plotter.add_mesh(line, color='magenta', line_width=1, opacity=0.5, name=f'scatterer_movement_{i}')
                        movement_actors.append(actor)
                
                # Draw UE (RX antenna) movement paths
                for i in range(len(initial_rx_positions)):
                    if i < len(future_rx_positions):
                        # Check if the antenna actually moved to avoid drawing zero-length lines if static
                        if np.linalg.norm(np.array(initial_rx_positions[i]) - np.array(future_rx_positions[i])) > 1e-6:
                            line = pv.Line(np.array(initial_rx_positions[i]), np.array(future_rx_positions[i]))
                            actor = plotter.add_mesh(line, color='dodgerblue', line_width=2, opacity=0.7, name=f'ue_movement_{i}')
                            movement_actors.append(actor)

                # Draw future scatterer points (slightly transparent)
                if len(future_scatterer_positions) > 0: 
                    future_points_scatterers = plotter.add_points(
                        future_scatterer_positions, color='lightgreen', point_size=6, 
                        render_points_as_spheres=True, opacity=0.5, name='future_scatterer_points'
                    )
                
                # Draw future UE (RX antenna) points
                if len(future_rx_positions) > 0:
                    # Filter to only draw points for UEs that actually moved
                    moved_future_rx_pos = [pos for idx, pos in enumerate(future_rx_positions) 
                                           if np.linalg.norm(np.array(initial_rx_positions[idx]) - np.array(pos)) > 1e-6]
                    if moved_future_rx_pos:
                        future_points_ue = plotter.add_points(
                            np.array(moved_future_rx_pos), color='cyan', point_size=7, 
                            render_points_as_spheres=True, opacity=0.6, name='future_ue_points'
                        )

                # Restore original scatterers and RX antennas
                self.scatterers = scatterer_backup
                self.rx_antennas = rx_antennas_backup 
                self.movement_type = original_movement_type 

        # Plot buildings
        if self.buildings:
            for i, building_obj in enumerate(self.buildings): 
                pos = building_obj.get_position()

                min_b, max_b    = building_obj.get_bounds()
                building_box_pv = pv.Box(bounds=(min_b[0], max_b[0],
                                                 min_b[1], max_b[1],
                                                 min_b[2], max_b[2]))
                # Add building with a gray/blue color and some transparency
                plotter.add_mesh(
                    building_box_pv, 
                    color='slategray', 
                    opacity=1.0, 
                    show_edges=True,
                    edge_color='darkslategray', 
                    name=f'building_{building_obj.id}' 
                )

                # Plot windows for this building
                for win_idx, window_obj in enumerate(building_obj.get_windows()): 
                    win_center   = window_obj.get_center()
                    dims_on_face = window_obj.get_dimensions_on_face()
                    win_opacity  = window_obj.get_opacity()
                    face_idx     = window_obj.face_index 
                    
                    epsilon = 0.05 
                    window_box_dims_vector = np.zeros(3)

                    if face_idx == 0 or face_idx == 1: 
                        window_box_dims_vector = np.array([epsilon, dims_on_face[0], dims_on_face[1]])
                    elif face_idx == 2 or face_idx == 3: 
                        window_box_dims_vector = np.array([dims_on_face[0], epsilon, dims_on_face[1]])
                    elif face_idx == 4 or face_idx == 5: 
                        window_box_dims_vector = np.array([dims_on_face[0], dims_on_face[1], epsilon])
                    else:
                        continue
                    
                    half_win_box_dims = window_box_dims_vector / 2.0
                    min_win_bounds = win_center - half_win_box_dims
                    max_win_bounds = win_center + half_win_box_dims
                    
                    window_mesh = pv.Box(bounds=(
                        min_win_bounds[0], max_win_bounds[0],
                        min_win_bounds[1], max_win_bounds[1],
                        min_win_bounds[2], max_win_bounds[2]
                    ))
                    plotter.add_mesh(
                        window_mesh, 
                        color='lightblue', 
                        opacity=win_opacity, 
                        name=f'building_{building_obj.id}_win_{win_idx}'
                    )

        # Plot clusters
        cluster_sphere_actors = [] 
        for i, cluster_info in enumerate(self.clusters):
            c      = cluster_info['center']
            radius = cluster_info['radius']
            
            sphere_mesh = pv.Sphere(radius=radius, center=[c[0], c[1], c[2]], theta_resolution=15, phi_resolution=15)
            
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
                                         pickable=False) 
                actor.SetVisibility(show_cluster_radii_initially)
                cluster_sphere_actors.append(actor)

        if show_bounds:
            cube = pv.Cube(center=(
                self.dimensions[0] / 2,
                self.dimensions[1] / 2,
                self.dimensions[2] / 2
            ), x_length=self.dimensions[0], y_length=self.dimensions[1], z_length=self.dimensions[2])
            plotter.add_mesh(cube, style='wireframe', color='dimgray', line_width=1, opacity=0.2, name='bounds')

        if show_grid:
            grid_bounds = (
                0, self.dimensions[0], 
                0, self.dimensions[1], 
                0, self.dimensions[2] * 2 
            )
            plotter.show_grid(bounds=grid_bounds, color='lightgray', fmt="%.0f")

        x_mid = self.dimensions[0] / 2
        y_mid = self.dimensions[1] / 2
        z_mid = self.dimensions[2] / 2
        
        initial_camera_position = [
            (-self.dimensions[0] * 0.5, y_mid * 0.3, self.dimensions[2] * 1.2), 
            (x_mid * 0.2, y_mid, z_mid * 0.5),                                
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

        # Add key events to toggle ray visibility and movement paths
        instruction_text = "Press 'h' to reset view"
        
        if show_rays:
            def toggle_rays():
                processed_base_rays = set()
                for actor_name, actor in plotter.renderer.actors.items():
                    base_name_candidate = actor_name
                    is_primary_ray_actor = False

                    if actor_name.endswith('_blocked_segment'):
                        base_name_candidate = actor_name.replace('_blocked_segment', '')
                    if base_name_candidate == 'los_ray' or base_name_candidate.startswith('ray_tx_sc_') or base_name_candidate.startswith('ray_sc_rx_'):
                        is_primary_ray_actor = True

                    if is_primary_ray_actor and base_name_candidate not in processed_base_rays:
                        main_ray_actor_name = base_name_candidate
                        visibility_to_set = True 
                        if main_ray_actor_name in plotter.renderer.actors:
                            visibility_to_set = not plotter.renderer.actors[main_ray_actor_name].GetVisibility()
                            plotter.renderer.actors[main_ray_actor_name].SetVisibility(visibility_to_set)
                        elif actor_name.endswith('_blocked_segment') and actor_name == (base_name_candidate + '_blocked_segment'):
                            visibility_to_set = not plotter.renderer.actors[actor_name].GetVisibility()
                            plotter.renderer.actors[actor_name].SetVisibility(visibility_to_set)
                        blocked_segment_name = base_name_candidate + '_blocked_segment'
                        if blocked_segment_name in plotter.renderer.actors and blocked_segment_name != main_ray_actor_name:
                            plotter.renderer.actors[blocked_segment_name].SetVisibility(visibility_to_set)
                        processed_base_rays.add(base_name_candidate)
                plotter.render()
            plotter.add_key_event('r', toggle_rays)
            instruction_text += ", 'r' to toggle ray paths"
        
        if show_movement:
            def toggle_movement():
                for actor_name, actor in plotter.renderer.actors.items(): 
                    if actor_name.startswith('movement_') or actor_name == 'future_points':
                        visibility = not actor.GetVisibility()
                        actor.SetVisibility(visibility)
                plotter.render()
            plotter.add_key_event('m', toggle_movement)
            instruction_text += ", 'm' to toggle movement paths"
        
        if show_rays:
            def toggle_window_rays():
                ray_window_actors = [actor_name for actor_name in plotter.renderer.actors.keys() 
                                     if actor_name.startswith('ray_tx_win_') or actor_name.startswith('ray_win_rx_')]
                if len(ray_window_actors) == 0:
                    if self.tx_antennas and self.rx_antennas:
                        tx_pos = self.tx_antennas[0].get_pos()
                        rx_pos = self.rx_antennas[0].get_pos()
                        valid_reflections = self.get_valid_window_reflections(tx_pos, rx_pos)
                        window_count = sum(len(b.get_windows()) for b in self.buildings)
                        if window_count > 0 and len(valid_reflections) == 0:
                            print("No valid reflection paths found. See --debug-reflections for details.")
                        elif len(valid_reflections) > 0:
                            print("ERROR: Valid reflections exist but rays weren't drawn!")
                for actor_name, actor in plotter.renderer.actors.items():
                    if actor_name.startswith('ray_tx_win_') or actor_name.startswith('ray_win_rx_'):
                        blockage_marker_name_hit = actor_name + '_hit_blockage_marker'
                        blocked_segment_name = actor_name + '_blocked_segment'
                        visibility = not actor.GetVisibility()
                        actor.SetVisibility(visibility) 
                        print(f"Toggled visibility of {actor_name} to {visibility}")
                        if blockage_marker_name_hit in plotter.renderer.actors:
                            plotter.renderer.actors[blockage_marker_name_hit].SetVisibility(visibility)
                        if blocked_segment_name in plotter.renderer.actors:
                            plotter.renderer.actors[blocked_segment_name].SetVisibility(visibility)
                plotter.render()
            plotter.add_key_event('p', toggle_window_rays)   
            instruction_text += ", 'p' to toggle window rays"
                    
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
        
        # Add screenshot functionality
        def take_screenshot():
            # Create screenshots directory if it doesn't exist
            screenshot_dir = "C:\\Users\\vrnan\\OneDrive - Imperial College London\\Year 4\\ELEC70017 - Individual Project\\Project\\data_generation\\utils\\figures\\environment"
            os.makedirs(screenshot_dir, exist_ok=True)
            
            # Generate timestamp for unique filename
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"environment_screenshot_{timestamp}.png"
            filepath = os.path.join(screenshot_dir, filename)
            
            # Store original visibility states
            hidden_elements = []
            
            # Hide text elements by removing them temporarily
            text_widgets = []
            if hasattr(plotter, 'textActor') and plotter.textActor:
                text_widgets.append(('textActor', plotter.textActor.GetVisibility()))
                plotter.textActor.SetVisibility(False)
            
            # Hide corner annotation (instruction text)
            if hasattr(plotter.renderer, 'GetTextActors'):
                for actor in plotter.renderer.GetTextActors():
                    if actor:
                        hidden_elements.append((actor, actor.GetVisibility()))
                        actor.SetVisibility(False)
            
            # Hide grid if visible
            grid_was_visible = False
            if hasattr(plotter, 'grid_actor') and plotter.grid_actor:
                grid_was_visible = plotter.grid_actor.GetVisibility()
                plotter.grid_actor.SetVisibility(False)
            
            # Hide camera orientation widget/gimbal
            widget_was_enabled = False
            if hasattr(plotter, 'camera_widget') and plotter.camera_widget:
                widget_was_enabled = plotter.camera_widget.GetEnabled()
                plotter.camera_widget.SetEnabled(False)
            
            # Note: Keeping point labels visible for screenshots as requested
            
            # Remove all text from renderer temporarily (but keep point labels)
            renderer_texts = []
            if hasattr(plotter.renderer, 'GetActors2D'):
                actors_2d = plotter.renderer.GetActors2D()
                actors_2d.InitTraversal()
                actor = actors_2d.GetNextItem()
                while actor:
                    if hasattr(actor, 'GetMapper') and actor.GetMapper():
                        mapper_type = type(actor.GetMapper()).__name__
                        # Hide text but not point labels/antenna names
                        if 'Text' in mapper_type and not hasattr(actor, '_is_point_label'):
                            renderer_texts.append((actor, actor.GetVisibility()))
                            actor.SetVisibility(False)
                    actor = actors_2d.GetNextItem()
            
            # Force render to update the scene
            plotter.render()
            
            # Take the screenshot
            try:
                plotter.screenshot(filepath, transparent_background=False, return_img=False)
                print(f"Screenshot saved to: {filepath}")
            except Exception as e:
                print(f"Failed to save screenshot: {e}")
            
            # Restore all hidden elements
            for actor, original_visibility in hidden_elements:
                if actor:
                    actor.SetVisibility(original_visibility)
            
            for actor, original_visibility in renderer_texts:
                if actor:
                    actor.SetVisibility(original_visibility)
            
            for widget_name, original_visibility in text_widgets:
                widget = getattr(plotter, widget_name, None)
                if widget:
                    widget.SetVisibility(original_visibility)
            
            # Restore grid visibility
            if hasattr(plotter, 'grid_actor') and plotter.grid_actor:
                plotter.grid_actor.SetVisibility(grid_was_visible)
            
            # Restore camera orientation widget
            if hasattr(plotter, 'camera_widget') and plotter.camera_widget:
                plotter.camera_widget.SetEnabled(widget_was_enabled)
            
            # Force render to restore the scene
            plotter.render()
        
        plotter.add_key_event('s', take_screenshot)
        # Removed instruction text and environment info text as requested
        # plotter.add_text(instruction_text, font_size=10, position='upper_left')

        # Add environment information - commented out as requested
        # env_info = (
        #     f"Environment: {self.dimensions[0]}m × {self.dimensions[1]}m × {self.dimensions[2]}m\n"
        #     f"Scatterer Movement: {self.movement_type}\n"
        #     f"UE Movement: {self.rx_antennas[0].movement_type if self.rx_antennas else 'N/A'} (Speed: {round(self.rx_antennas[0].speed, 3) if self.rx_antennas else 'N/A'} m/s)\n"
        #     f"Scatterers: {len(self.scatterers)}, Clusters: {len(self.clusters)}\n"
        #     f"Buildings: {len(self.buildings)}"
        # )
        # plotter.add_text(env_info, font_size=10, position='lower_left')
        plotter.enable_trackball_style()
        plotter.add_camera_orientation_widget()
        plotter.show()

    def create_road_mesh(self, start_point, end_point, width):
        """Creates a PyVista mesh for a road segment."""
        start_point = np.array(start_point, dtype=float)
        end_point   = np.array(end_point, dtype=float)
        direction   = end_point - start_point
        length      = np.linalg.norm(direction)
        if length < 1e-6: 
            return None
        direction_norm = direction / length
        perp_direction = np.array([-direction_norm[1], direction_norm[0], 0])
        half_width_vec = perp_direction * (width / 2)
        corner1 = start_point - half_width_vec
        corner2 = start_point + half_width_vec
        corner3 = end_point + half_width_vec
        corner4 = end_point - half_width_vec
        points  = np.array([corner1, corner2, corner3, corner4])
        faces   = np.hstack([[4], np.arange(4)]).tolist() 
        road_mesh = pv.PolyData(points, faces=faces)
        return road_mesh


if __name__ == "__main__":
    pass

