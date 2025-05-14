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
    LINEAR = "linear"
    RANDOM_WALK  = "random_walk"
    SINUSOIDAL   = "sinusoidal"
    BROWNIAN     = "brownian"
    GAUSS_MARKOV = "gauss_markov"
    FLOCKING     = "flocking"

def _handle_boundaries(
    pos: np.ndarray, 
    velocity: np.ndarray, 
    dims: Optional[np.ndarray],
    movement_type: MovementType = MovementType.LINEAR, 
    oscillation_center: Optional[np.ndarray] = None, 
    oscillation_amplitude: Optional[float] = None, 
    oscillation_axis: Optional[np.ndarray] = None, 
    oscillation_phase: Optional[float] = None 
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[float]]:
    """
    Internal helper to apply boundary conditions.
    
    Returns:
        adjusted_position    - 3-vector [m]
        adjusted_velocity    - 3-vector [m/s]
        adjusted_osc_center  - 3-vector [m] (or None if not applicable)
        adjusted_phase       - float [rad] (or None if not applicable)
    """
    if dims is None:
        return pos, velocity, oscillation_center, oscillation_phase

    adj_pos = pos.copy()
    adj_velocity = velocity.copy()
    
    if oscillation_center is not None:
        adj_osc_center = oscillation_center.copy()
    else:
        adj_osc_center = None
        
    adj_osc_phase = oscillation_phase # Start with original phase

    for i in range(3):
        hit_boundary = False
        if adj_pos[i] < 0:
            adj_pos[i] = 0
            adj_velocity[i] *= -1
            hit_boundary = True
            if movement_type == MovementType.SINUSOIDAL and adj_osc_center is not None and oscillation_amplitude is not None and oscillation_axis is not None:
                # Option 1: Pinning center (current logic, simplified)
                # adj_osc_center[i] = oscillation_amplitude * abs(oscillation_axis[i]) if oscillation_axis[i] != 0 else 0
                pass # Phase reflection handles the "bounce"

        elif adj_pos[i] > dims[i]:
            adj_pos[i] = dims[i]
            adj_velocity[i] *= -1
            hit_boundary = True
            if movement_type == MovementType.SINUSOIDAL and adj_osc_center is not None and oscillation_amplitude is not None and oscillation_axis is not None:
                # Option 1: Pinning center (current logic, simplified)
                # adj_osc_center[i] = dims[i] - (oscillation_amplitude * abs(oscillation_axis[i])) if oscillation_axis[i] != 0 else dims[i]
                pass # Phase reflection handles the "bounce"
        
        if hit_boundary and movement_type == MovementType.SINUSOIDAL and adj_osc_phase is not None:
            # Reflect phase for the component of oscillation normal to the wall if axis has component along i
            # This is a simplification; true reflection is more complex. For now, reflect total phase.
            # A more accurate reflection would only flip the phase contribution from the normal direction.
            if oscillation_axis is not None and abs(oscillation_axis[i]) > 1e-6: # If movement along this axis contributed
                 adj_osc_phase = np.pi - adj_osc_phase 
                 # Normalize phase to [0, 2*pi) or (-pi, pi] if desired, though not strictly necessary for np.sin
                 # adj_osc_phase = adj_osc_phase % (2 * np.pi)

    return adj_pos, adj_velocity, adj_osc_center, adj_osc_phase


def linear_move(
    current_pos: np.ndarray, 
    current_velocity: np.ndarray, 
    time_step: float, 
    speed: float,
    environment_dims: Optional[np.ndarray]
) -> Tuple[np.ndarray, np.ndarray]:
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
    final_pos, final_velocity, _, _ = _handle_boundaries(new_pos_ideal, new_velocity_ideal, environment_dims, MovementType.LINEAR)
    return final_pos, final_velocity

def random_walk_move(
    current_pos: np.ndarray, 
    current_velocity: np.ndarray, 
    time_step: float, 
    speed: float,
    rng: np.random.Generator,
    direction_change_prob: float, 
    max_angle_change: float,
    environment_dims: Optional[np.ndarray]
) -> Tuple[np.ndarray, np.ndarray]:
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
    final_pos, final_velocity, _, _ = _handle_boundaries(new_pos_ideal, new_velocity_ideal, environment_dims, MovementType.RANDOM_WALK)
    return final_pos, final_velocity

def sinusoidal_move(
    current_pos: np.ndarray, 
    time_elapsed: float,
    oscillation_center: np.ndarray, 
    oscillation_axis: np.ndarray,
    oscillation_amplitude: float,
    oscillation_angular_frequency: float,
    oscillation_phase: float,
    speed: float, # Fallback
    current_velocity: np.ndarray, # Fallback
    time_step: float, # Fallback & potentially for velocity calculation if needed
    environment_dims: Optional[np.ndarray]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]: 
    """
    Sinusoidal oscillation around a center point.
    
    Returns:
        new_pos           - 3-vector [m]
        new_velocity      - 3-vector [m/s]
        updated_osc_center - 3-vector [m]
        updated_phase     - float [rad]
    """
    new_pos_ideal = current_pos.copy()
    new_velocity_ideal = current_velocity.copy()
    updated_osc_center = oscillation_center.copy()

    if oscillation_amplitude > 0 and oscillation_angular_frequency > 0:
        offset = oscillation_axis * oscillation_amplitude * \
                 np.sin(oscillation_angular_frequency * time_elapsed + oscillation_phase)
        new_pos_ideal = updated_osc_center + offset # Use the (potentially modified) center
        
        new_velocity_ideal = oscillation_axis * oscillation_amplitude * \
                             oscillation_angular_frequency * \
                             np.cos(oscillation_angular_frequency * time_elapsed + oscillation_phase)
    elif speed > 0:  # Fallback to linear
        new_pos_ideal = current_pos + current_velocity * time_step
        # new_velocity_ideal remains current_velocity for linear fallback
    # else: no movement, pos and vel remain as they were

    # Pass all relevant sinusoidal params to _handle_boundaries
    final_pos, final_velocity, final_osc_center, final_phase = _handle_boundaries(
        new_pos_ideal, 
        new_velocity_ideal, 
        environment_dims, 
        MovementType.SINUSOIDAL,
        updated_osc_center, # Pass the current oscillation center
        oscillation_amplitude,
        oscillation_axis,
        oscillation_phase # Pass current phase to be potentially modified
    )
    return final_pos, final_velocity, \
           (final_osc_center if final_osc_center is not None else updated_osc_center), \
           (final_phase if final_phase is not None else oscillation_phase)

def brownian_move(
    current_pos: np.ndarray,
    current_velocity: np.ndarray,
    sigma_brownian: float,
    rng: np.random.Generator,
    time_step: float,
    environment_dims: Optional[np.ndarray]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates the new position based on Brownian motion (Gaussian steps).
    
    Returns:
        new_pos       - 3-vector [m]
        new_velocity  - 3-vector [m/s] (calculated as delta/time_step for Doppler)
    """
    if sigma_brownian <= 1e-9: # No Brownian component
        new_pos_ideal = current_pos.copy()
        new_velocity_ideal = current_velocity.copy()
    else:
        delta = rng.normal(scale=sigma_brownian, size=3)
        new_pos_ideal = current_pos + delta
        # Calculate velocity from position change for Doppler effects
        new_velocity_ideal = delta / time_step if time_step > 0 else current_velocity.copy()
    
    final_pos, final_velocity, _, _ = _handle_boundaries(new_pos_ideal, new_velocity_ideal, environment_dims, MovementType.BROWNIAN)
    return final_pos, final_velocity

def gauss_markov_move(
    current_pos: np.ndarray,
    current_velocity: np.ndarray,
    time_step: float,
    alpha_gm: float,
    mean_velocity_gm: np.ndarray,
    noise_std_dev_gm: float,
    rng: np.random.Generator,
    environment_dims: Optional[np.ndarray]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates new position and velocity based on the Gauss-Markov mobility model.
    
    Gauss-Markov equation:
    v_t = α·v_{t-1} + (1-α)·mean_v + sqrt(1-α²)·η
    
    Returns:
        new_pos       - 3-vector [m]
        new_velocity  - 3-vector [m/s]
    """
    if alpha_gm < 0 or alpha_gm > 1:
        raise ValueError("Gauss-Markov alpha must be between 0 and 1.")
    if noise_std_dev_gm < 0:
        raise ValueError("Gauss-Markov noise_std_dev_gm cannot be negative.")

    # Ensure mean_velocity_gm is a 3D vector
    if not isinstance(mean_velocity_gm, np.ndarray) or mean_velocity_gm.shape != (3,):
        actual_mean_velocity_gm = np.zeros(3)
    else:
        actual_mean_velocity_gm = mean_velocity_gm 

    eta_t = rng.normal(scale=noise_std_dev_gm, size=3)
    
    # Gauss-Markov equation components
    term1 = alpha_gm * current_velocity
    term2 = (1 - alpha_gm) * actual_mean_velocity_gm
    
    # Handle edge case for alpha_gm ≈ 1.0
    sqrt_val = max(0, 1 - alpha_gm**2)  # Ensure non-negative
    term3 = np.sqrt(sqrt_val) * eta_t

    new_velocity_ideal = term1 + term2 + term3
    
    # Update position
    new_pos_ideal = current_pos + new_velocity_ideal * time_step
    
    final_pos, final_velocity, _, _ = _handle_boundaries(
        new_pos_ideal, new_velocity_ideal, environment_dims, MovementType.GAUSS_MARKOV
    )
    return final_pos, final_velocity

def flocking_move(
    current_pos: np.ndarray,
    current_flock_velocity: np.ndarray, # shared velocity for the entire flock
    time_step: float,
    environment_dims: Optional[np.ndarray]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Moves the scatterer as part of a flock using a shared velocity.
    'current_flock_velocity' is interpreted as the shared velocity of the entire flock.
    The magnitude of this vector defines the flock's speed.
    Boundary handling adjusts the individual scatterer's position and may
    return an adjusted velocity for this scatterer if a boundary is hit.
    Maintaining true flock cohesion (e.g., all scatterers perfectly mirroring
    velocity changes due to one member's boundary interaction, or the flock
    itself changing direction) would require higher-level management of the
    'current_flock_velocity' outside this function.

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
    final_pos, final_velocity, _, _ = _handle_boundaries(
        new_pos_ideal, 
        current_flock_velocity.copy(), # Pass a copy to avoid unintended modification
        environment_dims, 
        MovementType.FLOCKING # Pass the movement type for context, though _handle_boundaries may not use it uniquely here
    )
    return final_pos, final_velocity



