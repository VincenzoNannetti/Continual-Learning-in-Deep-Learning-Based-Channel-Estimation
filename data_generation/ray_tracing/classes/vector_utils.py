"""
Filename: ./data_generation/ray_tracing/classes/vector_utils.py
Author: Vincenzo Nannetti
Date: 09/05/2025 (Created Date)
Description: Helper functions for 3D vector operations in the ray tracing model.
"""

import numpy as np
from typing import Union, Tuple, Optional

Vector = np.ndarray

def ensure_vector3d(v: Union[list, tuple, np.ndarray]) -> Vector:
    if not isinstance(v, np.ndarray):
        v = np.array(v, dtype=float)
    if len(v) == 2:
        return np.array([v[0], 0.0, v[1]], dtype=float)
    elif len(v) == 3:
        return v
    else:
        raise ValueError(f"Expected 2 or 3 values, got {len(v)}")

def distance3d(p: Vector, q: Vector) -> float:
    p = ensure_vector3d(p)
    q = ensure_vector3d(q)
    return np.linalg.norm(p - q)

def horizontal_distance(p: Vector, q: Vector) -> float:
    p = ensure_vector3d(p)
    q = ensure_vector3d(q)
    return np.linalg.norm(p[:2] - q[:2])

def unit_vector3d(p: Vector, q: Vector) -> Vector:
    """
    Calculate the unit vector pointing from p to q.
    
    Args:
        p: Origin point as 3D vector [x, y, z]
        q: Destination point as 3D vector [x, y, z]
        
    Returns:
        Unit vector from p to q
        
    Raises:
        ValueError: If p and q are too close (distance < 1e-9)
    """
    p = ensure_vector3d(p)
    q = ensure_vector3d(q)
    
    direction = q - p
    distance = np.linalg.norm(direction)
    
    if distance < 1e-9:
        raise ValueError("Points are too close to define a direction")
        
    return direction / distance

def zenith_azimuth(p: Vector, q: Vector) -> Tuple[float, float]:
    """
    Calculate the zenith and azimuth angles from p to q.
    
    Args:
        p: Origin point as 3D vector [x, y, z]
        q: Destination point as 3D vector [x, y, z]
        
    Returns:
        Tuple of (zenith, azimuth) in radians:
        - zenith: angle from +z axis (0 to π)
        - azimuth: angle from +x axis in x-y plane (0 to 2π)
        
    Note:
        This follows 3GPP TR 38.901 convention where:
        - x = East
        - y = North
        - z = Height
    """
    p = ensure_vector3d(p)
    q = ensure_vector3d(q)
    
    direction = q - p
    
    # Calculate zenith angle (θ) from +z axis
    zenith = np.arccos(direction[2] / np.linalg.norm(direction))
    
    # Calculate azimuth angle (φ) from +x axis in x-y plane
    azimuth = np.arctan2(direction[1], direction[0])
    
    # Ensure azimuth is in [0, 2π)
    if azimuth < 0:
        azimuth += 2 * np.pi
        
    return zenith, azimuth

def generate_random_3d_direction(rng: Optional[np.random.RandomState] = None) -> Vector:
    """
    Generate a random 3D unit vector with uniform distribution.
    
    Args:
        rng: Random number generator instance
        
    Returns:
        Random 3D unit vector
    """
    if rng is None:
        # Use numpy's random directly for non-reproducibility
        # Generate random azimuth (0 to 2π) and zenith (0 to π)
        azimuth = np.random.uniform(0, 2 * np.pi)
        zenith = np.arccos(2 * np.random.random() - 1)  # Uniform on a sphere
    else:
        # Use provided random number generator for reproducibility
        azimuth = rng.uniform(0, 2 * np.pi)
        zenith = np.arccos(2 * rng.random() - 1)  # Uniform on a sphere
    
    # Convert to Cartesian coordinates
    x = np.sin(zenith) * np.cos(azimuth)
    y = np.sin(zenith) * np.sin(azimuth)
    z = np.cos(zenith)
    
    return np.array([x, y, z])
