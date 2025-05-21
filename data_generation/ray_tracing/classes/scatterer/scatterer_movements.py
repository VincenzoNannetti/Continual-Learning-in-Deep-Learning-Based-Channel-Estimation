"""
Filename: ./data_generation/ray_tracing/classes/scatterer/scatterer_movements.py
Author: Vincenzo Nannetti 
Date: 13/05/2025
Description: Script to define the movement types for the scatterers.
"""

import numpy as np
from enum import Enum
from ..vector_utils import generate_random_3d_direction
from typing import Optional, Tuple

class MovementType(Enum):
    LINEAR       = "linear"
    RANDOM_WALK  = "random_walk"
    FLOCKING     = "flocking"
    STATIC       = "static"

def _handle_boundaries(pos: np.ndarray, 
                       velocity: np.ndarray, 
                       dims: Optional[np.ndarray],
                       movement_type: MovementType = MovementType.LINEAR):
    """
    Internal helper to apply boundary conditions for linear, random walk, and flocking movements.
    
    Returns:
        adjusted_position - 3-vector [m]
        adjusted_velocity - 3-vector [m/s]
    """
    if dims is None or movement_type == MovementType.STATIC:
        return pos, velocity

    adj_pos = pos.copy()
    adj_velocity = velocity.copy()

    for i in range(3):
        if adj_pos[i] < 0:
            adj_pos[i] = 0
            adj_velocity[i] *= -1
        elif adj_pos[i] > dims[i]:
            adj_pos[i] = dims[i]
            adj_velocity[i] *= -1

    return adj_pos, adj_velocity

def linear_move(current_pos: np.ndarray, 
                current_velocity: np.ndarray, 
                time_step: float, 
                speed: float,
                environment_dims: Optional[np.ndarray]):
    """
    Simple straight-line movement based on current velocity.
    
    Returns:
        new_pos       - 3-vector [m]
        new_velocity  - 3-vector [m/s]
    """
    new_pos_ideal = current_pos.copy()
    new_velocity_ideal = current_velocity.copy()

    if speed > 1e-9:
        new_pos_ideal = current_pos + current_velocity * time_step
    
    # Velocity doesn't change due to movement type itself, only by boundaries
    final_pos, final_velocity = _handle_boundaries(new_pos_ideal, new_velocity_ideal, environment_dims, MovementType.LINEAR)
    return final_pos, final_velocity

def random_walk_move(current_pos: np.ndarray, 
                     current_velocity: np.ndarray, 
                     time_step: float, 
                     speed: float,
                     rng: np.random.Generator,
                     direction_change_prob: float, 
                     max_angle_change: float,
                     environment_dims: Optional[np.ndarray]):
    """
    Random walk movement with probability-based direction changes.
    
    Returns:
        new_pos       - 3-vector [m]
        new_velocity  - 3-vector [m/s]
    """
    new_velocity_ideal = current_velocity.copy()
    new_pos_ideal = current_pos.copy()

    if speed > 0:
        if rng.random() < direction_change_prob:
            current_dir_norm = np.linalg.norm(new_velocity_ideal)
            if current_dir_norm > 0:
                current_dir = new_velocity_ideal / current_dir_norm
                random_axis = rng.standard_normal(3)
                if np.linalg.norm(random_axis) > 0:
                    random_axis = random_axis / np.linalg.norm(random_axis)
                    angle = rng.uniform(-max_angle_change, max_angle_change)
                    cos_angle, sin_angle = np.cos(angle), np.sin(angle)
                    K = np.array([[0, -random_axis[2], random_axis[1]],
                                  [random_axis[2], 0, -random_axis[0]],
                                  [-random_axis[1], random_axis[0], 0]])
                    rotation_matrix = np.eye(3) + sin_angle * K + (1 - cos_angle) * (K @ K)
                    new_dir_rotated = np.dot(rotation_matrix, current_dir)
                    new_velocity_ideal = new_dir_rotated * speed
            else:
                new_velocity_ideal = generate_random_3d_direction(rng) * speed
        
        new_pos_ideal = current_pos + new_velocity_ideal * time_step
    
    # Apply boundaries
    final_pos, final_velocity = _handle_boundaries(new_pos_ideal, new_velocity_ideal, environment_dims, MovementType.RANDOM_WALK)
    return final_pos, final_velocity

def flocking_move(current_pos: np.ndarray,
                  current_flock_velocity: np.ndarray, 
                  time_step: float,
                  environment_dims: Optional[np.ndarray]):
    """
    Moves the scatterer as part of a flock using a shared velocity.

    Args:
        current_pos (np.ndarray): Current position of the scatterer.
        current_flock_velocity (np.ndarray): Shared velocity of the flock.
        time_step (float): Time duration for the movement.
        environment_dims (Optional[np.ndarray]): Dimensions of the environment for boundary handling.

    Returns:
        Tuple[np.ndarray, np.ndarray]: The new position and new velocity of the scatterer.
    """
    new_pos_ideal = current_pos + current_flock_velocity * time_step
    
    # The velocity passed to _handle_boundaries is the flock's velocity.
    # If a boundary is hit, _handle_boundaries will reflect this velocity for the individual scatterer.
    final_pos, final_velocity = _handle_boundaries(
        new_pos_ideal, 
        current_flock_velocity.copy(), 
        environment_dims, 
        MovementType.FLOCKING 
    )
    return final_pos, final_velocity

def static_move(current_pos: np.ndarray,
                current_velocity: np.ndarray, 
                time_step: float,   
                speed: float,   
                environment_dims: Optional[np.ndarray]):
    """
    Static movement: position and velocity do not change.

    - Params are there for consistency with other movement functions.
    
    Returns:
        new_pos       - 3-vector [m] (same as current_pos)
        new_velocity  - 3-vector [m/s] (zero vector)
    """
    # For static, position remains the same, and velocity is zero.
    # _handle_boundaries is effectively bypassed for MovementType.STATIC earlier.
    return current_pos.copy(), np.zeros_like(current_velocity)



