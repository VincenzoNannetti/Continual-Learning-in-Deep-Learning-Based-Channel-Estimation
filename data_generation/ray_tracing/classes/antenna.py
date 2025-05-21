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
        
        # Initialise movement if configured
        if movement_config and self.movement_type != "static":
            self._initialise_movement()
    
    def _initialise_movement(self):
        """Helper method to set up movement parameters"""
        self.speed = float(self.rng.uniform(self.speed_range[0], self.speed_range[1]))
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
        else:
            self._set_static()
    
    def _set_static(self):
        """Helper method to set antenna as static"""
        self.movement_type = "static"
        self.speed = 0.0
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
        
        # Set up center
        if "center_abs" in params:
            self.movement_state["center"] = np.array(params["center_abs"], dtype=float)
        elif "center_offset" in params:
            offset = np.array(params["center_offset"], dtype=float)
            if len(offset) not in [2, 3]:
                raise ValueError("center_offset must be 2D or 3D array")
            center_xy = self.initial_pos[:2] + offset[:2]
            self.movement_state["center"] = np.array([center_xy[0], center_xy[1], self.initial_pos[2]])
        else:
            self.movement_state["center"] = self.initial_pos.copy()
            
        # Set up radius and angle
        center = self.movement_state["center"]
        if np.allclose(self.initial_pos[:2], center[:2], atol=1e-6):
            self.movement_params.setdefault('radius', 10.0)
            self.movement_state['angle'] = 0.0
        else:
            dx = self.initial_pos[0] - center[0]
            dy = self.initial_pos[1] - center[1]
            implied_radius = np.sqrt(dx*dx + dy*dy)
            
            if 'radius' in self.movement_params:
                if not np.isclose(implied_radius, self.movement_params['radius'], rtol=1e-3, atol=1e-3):
                    print(f"INFO: Antenna '{self.name}' circular motion: Using implied radius {implied_radius:.2f} from initial position")
            self.movement_params['radius'] = implied_radius
            self.movement_state['angle'] = np.arctan2(dy, dx)
            
        self.movement_state['effective_clockwise'] = self.movement_params.get('clockwise', True)

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

    def get_name(self):
        return self.name

    def get_gain(self):
        return self.gain

    def get_pos(self):
        return self.pos.copy()

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
        new_pos = self.pos
        new_velocity = self.velocity_vector
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




