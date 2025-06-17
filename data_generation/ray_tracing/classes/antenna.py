"""
Filename: ./data_generation/ray_tracing/classes/antenna.py
Author: Vincenzo Nannetti
Date: 09/05/2025 (Refactored Date)
Description: Antenna Class.
"""
import numpy as np
from .vector_utils import generate_random_3d_direction, unit_vector3d
from typing import Optional, Dict, List

class Antenna:
    def __init__(self, name, antenna_type, gain, pos, movement_config: dict = None, pavement_bounds: Optional[Dict] = None):
        """
        Initialise an antenna.
        
        Args:
            name: Name of the antenna (BS or UE etc.)
            antenna_type: Type of antenna ("TX" or "RX")
            gain: Antenna gain in dBi
            pos: 3D position as [x, y, z]
            movement_config: Dictionary defining movement parameters for the antenna (optional)
            pavement_bounds: Dictionary defining pavement boundaries for UE movement (optional)
        """
        # Basic attributes
        self.name         = name
        self.gain         = gain
        self.antenna_type = antenna_type
        
        # Position handling
        self.pos = np.array(pos, dtype=float) if not isinstance(pos, np.ndarray) else pos
        if len(self.pos) != 3:
            raise ValueError(f"Position must be 3D [x, y, z], got {len(self.pos)} elements")
        self.initial_pos = self.pos.copy()
        
        # Movement setup
        self.rng                       = np.random.default_rng()
        self.movement_cfg              = movement_config
        self.movement_type             = movement_config.get("type", "static")    if movement_config else "static"
        self.speed_range               = movement_config.get("speed", [1.3, 1.5]) if movement_config else [0, 0]
        self.velocity_vector           = np.zeros(3, dtype=float)
        self.movement_params           = {}
        self.time_elapsed_for_movement = 0.0
        self.movement_state            = {}
        self.pavement_bounds           = pavement_bounds
        self.speed                     = float(self.rng.uniform(self.speed_range[0], self.speed_range[1]))

        # Initialise movement if configured
        if movement_config and self.movement_type != "static":
            self._initialise_movement()
    
    def _initialise_movement(self):
        """Helper method to set up movement parameters"""
        if self.speed <= 0:
            self._set_static()
            return
        if self.movement_type == "linear":
            self._setup_linear_movement()
        elif self.movement_type == "forward_back":
            self._setup_forward_back_movement()
        elif self.movement_type == "circular":
            self._setup_circular_movement()
        elif self.movement_type == "zigzag":
            self._setup_zigzag_movement()
        elif self.movement_type == "random_waypoint":
            self._setup_random_waypoint_movement()
        else:
            self._set_static()
    
    def _set_static(self):
        """Helper method to set antenna as static"""
        self.movement_type   = "static"
        self.speed           = 0.0
        self.velocity_vector = np.zeros(3, dtype=float)
    
    def _setup_linear_movement(self):
        """Helper method to set up linear movement"""
        params = self.movement_cfg.get("linear_params", {})
        direction = np.array(params.get("direction", [0,0,0]), dtype=float)
        if self.antenna_type == "RX" and np.count_nonzero(direction) > 0:
            if len(direction) == 3 and direction[2] != 0 and not params.get("allow_z_movement", False):
                direction[2] = 0
            norm_dir = np.linalg.norm(direction)
            if norm_dir > 1e-9:
                self.velocity_vector = (direction / norm_dir) * self.speed
            else:
                self._set_static()
        elif np.count_nonzero(direction) == 0:
            self._set_static()
        else:
            self.velocity_vector = generate_random_3d_direction(self.rng) * self.speed
    
    def _setup_forward_back_movement(self):
        """Helper method to set up forward-back movement"""
        params = self.movement_cfg.get("forward_back_params", {})
        direction = np.array(params.get("direction", [1,0,0]), dtype=float)
        if self.antenna_type == "RX":
            direction[2] = 0
            
        norm_dir = np.linalg.norm(direction)
        if norm_dir > 1e-9:
            self.velocity_vector = (direction / norm_dir) * self.speed
            self.movement_params = params
            self.movement_state  = {"phase": "forward", "distance_moved_current_leg": 0.0}
        else:
            self._set_static()
    
    def _setup_circular_movement(self):
        """Helper method to set up circular movement"""
        params = self.movement_cfg.get("circular_params", {})
        self.movement_params = params.copy()
        self.movement_state = {}
        
        # Get the configured radius (this should be respected, not overridden)
        configured_radius = params.get('radius', 1.0)
        
        # Set up center
        if "center_abs" in params:
            # Absolute center position provided
            self.movement_state["center"] = np.array(params["center_abs"], dtype=float)
            # Calculate starting angle based on initial position relative to this center
            center = self.movement_state["center"]
            dx = self.initial_pos[0] - center[0]
            dy = self.initial_pos[1] - center[1]
            distance_to_center = np.sqrt(dx*dx + dy*dy)
            
            if distance_to_center < 1e-6:
                # UE is at the center, start at angle 0 (east direction)
                self.movement_state['angle'] = 0.0
            else:
                # Use the angle from center to initial position
                self.movement_state['angle'] = np.arctan2(dy, dx)
                
        elif "center_offset" in params:
            # Center offset from initial position provided
            offset = np.array(params["center_offset"], dtype=float)
            if len(offset) not in [2, 3]:
                raise ValueError("center_offset must be 2D or 3D array")
            
            # Normalize the offset direction to create a center at the correct distance
            offset_2d = offset[:2]
            offset_magnitude = np.linalg.norm(offset_2d)
            
            if offset_magnitude < 1e-6:
                # Zero offset, center at initial position
                self.movement_state["center"] = self.initial_pos.copy()
                self.movement_state['angle'] = 0.0
            else:
                # Calculate center position so UE is on circumference of desired radius
                offset_direction = offset_2d / offset_magnitude
                center_offset_distance = configured_radius  # Distance from UE to center = radius
                center_position_2d = self.initial_pos[:2] + offset_direction * center_offset_distance
                self.movement_state["center"] = np.array([center_position_2d[0], center_position_2d[1], self.initial_pos[2]])
                
                # Starting angle: from center to current UE position (opposite to offset direction)
                self.movement_state['angle'] = np.arctan2(-offset_direction[1], -offset_direction[0])
            
        else:
            # No center specified, use initial position as center
            self.movement_state["center"] = self.initial_pos.copy()
            self.movement_state['angle'] = 0.0
            
        # Always use the configured radius, don't override it
        self.movement_params['radius'] = configured_radius
        self.movement_state['effective_clockwise'] = self.movement_params.get('clockwise', True)
        
        # Verify that the UE will start at its current position
        center = self.movement_state["center"]
        angle = self.movement_state['angle']
        radius = self.movement_params['radius']
        expected_start_pos = center + np.array([radius * np.cos(angle), radius * np.sin(angle), 0])
        
        # Check if the expected position matches the initial position (within tolerance)
        position_error = np.linalg.norm(expected_start_pos[:2] - self.initial_pos[:2])
        if position_error > 0.1:  # 10cm tolerance
            print(f"INFO: Antenna '{self.name}' circular motion will start from current position.")
            print(f"      Initial pos: [{self.initial_pos[0]:.2f}, {self.initial_pos[1]:.2f}]")
            print(f"      Circle center: [{center[0]:.2f}, {center[1]:.2f}], radius: {radius:.2f}m")
            print(f"      The UE will move smoothly from its starting position in a {radius:.2f}m radius circle.")

    def _setup_zigzag_movement(self):
        """Helper method to set up zigzag movement"""
        params = self.movement_cfg.get("zigzag_params", {})
        # Ensure there are parameters for zigzag, otherwise, it's static.
        if not params:
            self._set_static()
            return
        self.movement_params = params.copy()
        # Initial state for zigzag can be prepared here if needed,
        # but ue_movements.py already handles it if 'current_leg' is not in state.
        # For example, ensuring main_direction is present or defaulting it.
        main_dir_param = np.array(self.movement_params.get('main_direction', [1,0,0]), dtype=float)
        if np.linalg.norm(main_dir_param) < 1e-9: # Avoid zero vector for direction
            main_dir_param = np.array([1,0,0])
        main_dir_param[2] = 0 # Ensure movement is in XY plane
        self.movement_params['main_direction'] = (main_dir_param / np.linalg.norm(main_dir_param)).tolist() if np.linalg.norm(main_dir_param) > 1e-9 else [1,0,0]
        
        # The velocity_vector will be set dynamically by the zigzag logic in ue_movements.py
        # based on the current leg's direction. So, no need to set self.velocity_vector here
        # other than ensuring it's a zero vector initially if speed is zero.
        if self.speed <= 1e-9:
            self.velocity_vector = np.zeros(3, dtype=float)
        # else, it will be managed by the update_zigzag function

    def _setup_random_waypoint_movement(self):
        """Helper method to set up random waypoint movement."""
        params = self.movement_cfg.get("random_waypoint_params", {})
        self.movement_params = params.copy()
        if self.speed <= 1e-9:
            self.velocity_vector = np.zeros(3, dtype=float)

    def get_name(self):
        return self.name

    def get_gain(self):
        return self.gain

    def get_pos(self):
        return self.pos.copy()

    def get_speed(self):
        return self.speed

    def get_antenna_type(self):
        return self.antenna_type

    def update_pos(self, dt: float, environment_dims: np.ndarray, ue_movement_module, buildings: Optional[List[Dict[str, float]]] = None):
        """
        Update antenna position based on its movement type and parameters.
        
        Args:
            dt (float): Time step in seconds.
            environment_dims (np.ndarray): Dimensions of the environment [x_max, y_max, z_max].
            ue_movement_module: The module containing UE movement functions.
            buildings: List of dictionaries defining building boundaries (optional)
        """
        if self.movement_type == "static" or self.speed <= 1e-9:
            if self.movement_type == "circular":
                if not self.movement_params.get("radius", 0) > 1e-9 : return
            else:
                return 

        self.time_elapsed_for_movement += dt
        new_pos                = self.pos
        new_velocity           = self.velocity_vector
        updated_movement_state = self.movement_state.copy()

        # Store the original z-coordinate
        original_z = self.pos[2]

        new_pos, new_velocity, updated_movement_state = ue_movement_module.update_movement(
            move_type=self.movement_type,
            pos=self.pos,
            vel=self.velocity_vector,
            speed=self.speed,
            dt=dt,
            dims=environment_dims,
            params=self.movement_params,
            state=self.movement_state,
            pavement=self.pavement_bounds,
            buildings=buildings 
        )

        self.pos             = new_pos
        self.pos[2]          = original_z
        self.velocity_vector = new_velocity
        self.movement_state  = updated_movement_state




