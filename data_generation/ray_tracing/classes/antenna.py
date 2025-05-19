"""
Filename: ./data_generation/ray_tracing/classes/antenna.py
Author: Vincenzo Nannetti
Date: 09/05/2025 (Refactored Date)
Description: Antenna Class.
"""
import numpy as np
from .vector_utils import generate_random_3d_direction, unit_vector3d
from typing import Optional, Dict

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
        self.name         = name
        self.antenna_type = antenna_type
        self.gain         = gain
        
        # Convert position to numpy array if it isn't already
        if not isinstance(pos, np.ndarray):
            pos = np.array(pos, dtype=float)
            
        # Validate position is 3D
        if len(pos) != 3:
            raise ValueError(f"Position must be 3D [x, y, z], got {len(pos)} elements")
        
        self.pos = pos
        self.initial_pos = self.pos.copy()
        self.rng = np.random.default_rng() # For movements that might need RNG

        # Movement attributes
        self.movement_type             = "static"
        self.speed                     = 0.0
        self.velocity_vector           = np.zeros(3, dtype=float)
        self.movement_params           = {}
        self.time_elapsed_for_movement = 0.0
        self.movement_state            = {}                      # To store state for complex movements 
        self.pavement_bounds           = pavement_bounds         # Store pavement boundaries

        if movement_config:
            self.movement_type = movement_config.get("type", "static")
            self.speed         = float(movement_config.get("speed", 0.0))

            if self.movement_type != "static" and self.speed > 0:
                if self.movement_type == "linear":
                    params = movement_config.get("linear_params", {})
                    direction = np.array(params.get("direction", [0,0,0]), dtype=float)
                    if self.antenna_type == "RX" and np.count_nonzero(direction) > 0 : 
                        if len(direction) == 3 and direction[2] != 0 and not params.get("allow_z_movement", False):
                            direction[2] = 0
                        
                        norm_dir = np.linalg.norm(direction)
                        if norm_dir > 1e-9:
                            self.velocity_vector = (direction / norm_dir) * self.speed
                        else: 
                            self.velocity_vector = np.zeros(3, dtype=float)
                            self.speed = 0.0
                            self.movement_type = "static"
                    elif np.count_nonzero(direction) == 0: 
                         self.velocity_vector = np.zeros(3, dtype=float)
                         self.speed = 0.0
                         self.movement_type = "static"
                    else: 
                        self.velocity_vector = generate_random_3d_direction(self.rng) * self.speed
                elif self.movement_type == "forward_back":
                    params    = movement_config.get("forward_back_params", {})
                    direction = np.array(params.get("direction", [1,0,0]), dtype=float)
                    if self.antenna_type == "RX": direction[2] = 0 
                    
                    norm_dir = np.linalg.norm(direction)
                    if norm_dir > 1e-9:
                        self.velocity_vector = (direction / norm_dir) * self.speed
                    else: 
                        self.velocity_vector = np.zeros(3, dtype=float)
                        self.speed = 0.0
                        self.movement_type = "static"
                    
                    self.movement_params = params
                    self.movement_state = {"phase": "forward", "distance_moved_current_leg": 0.0}

                elif self.movement_type == "circular":
                    params               = movement_config.get("circular_params", {})
                    self.movement_params = params
                    self.movement_state  = {"current_angle_rad": 0.0} 
                    if "center_abs" in params:
                        self.movement_state["center"] = np.array(params["center_abs"], dtype=float)
                    elif "center_offset" in params:
                        center_xy                     = self.initial_pos[:2] + np.array(params["center_offset"], dtype=float)
                        self.movement_state["center"] = np.array([center_xy[0], center_xy[1], self.initial_pos[2]])
                    else: 
                         self.movement_state["center"] = self.initial_pos.copy()
                else: 
                    self.movement_type = "static"
                    self.speed = 0.0
                    self.velocity_vector = np.zeros(3, dtype=float)
            else: 
                self.movement_type = "static"
                self.speed = 0.0
                self.velocity_vector = np.zeros(3, dtype=float)


    def get_name(self):
        return self.name

    def get_gain(self):
        return self.gain

    def get_pos(self):
        return self.pos.copy()

    def get_antenna_type(self):
        return self.antenna_type

    def update_pos(self, dt: float, environment_dims: np.ndarray, ue_movement_module):
        """
        Update antenna position based on its movement type and parameters.
        
        Args:
            dt (float): Time step in seconds.
            environment_dims (np.ndarray): Dimensions of the environment [x_max, y_max, z_max].
            ue_movement_module: The module containing UE movement functions.
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
            buildings=None 
        )

        self.pos             = new_pos
        self.pos[2]          = original_z
        self.velocity_vector = new_velocity
        self.movement_state  = updated_movement_state




