"""
Building class for the ray tracing model.
Represents a solid structure that blocks rays completely.
Can also reflect rays from its surfaces.
"""

import numpy as np
from ..vector_utils import ensure_vector3d
from typing import List, Tuple, Optional, Dict
from .window import Window

class Building:
    def __init__(self, 
                 id_: str,
                 position: List[float],  
                 dimensions: List[float],  
                 reflection_coefficient: float = 0.0, 
                 material: str = "concrete"):
        """
        Initialise a building in the environment.
        
        Args:
            id_: Unique identifier for the building
            position: [x, y, z] center position of the building
            dimensions: [width, length, height] dimensions of the building
            reflection_coefficient: How much signal is reflected (0-1)
            material: Building material (affects reflection properties)
        """
        self.id = id_
        self.position = np.array(ensure_vector3d(position), dtype=float)
        self.dimensions = np.array(ensure_vector3d(dimensions), dtype=float)
        self.reflection_coefficient = reflection_coefficient
        self.material = material
        
        # Calculate corners for collision detection
        self._calculate_bounds()
        self.windows = [] 
        
    def _calculate_bounds(self):
        """Calculate the min/max bounds of the building."""
        half_dims = self.dimensions / 2
        self.min_bounds = self.position - half_dims
        self.max_bounds = self.position + half_dims
        
    def intersects_ray(self, ray_origin, ray_direction) -> Tuple[bool, Optional[float], Optional[np.ndarray]]:
        """
        Check if a ray intersects with the building.
        
        Args:
            ray_origin: Origin point of the ray
            ray_direction: Direction vector of the ray (normalized)
            
        Returns:
            Tuple of (intersects, distance, normal_vector)
            - intersects: True if the ray intersects the building
            - distance: Distance to intersection point (or None)
            - normal_vector: Normal vector at intersection point (or None)
        """
        # Implementation of ray-box intersection algorithm
        ray_origin = np.array(ensure_vector3d(ray_origin), dtype=float)
        ray_direction = np.array(ensure_vector3d(ray_direction), dtype=float)
        
        # Normalize ray direction
        ray_direction = ray_direction / np.linalg.norm(ray_direction)
        
        t_min = -np.inf
        t_max = np.inf
        normal = np.zeros(3)
        
        for i in range(3):
            if abs(ray_direction[i]) < 1e-6:
                # Ray is parallel to slab - no hit if origin is outside box
                if ray_origin[i] < self.min_bounds[i] or ray_origin[i] > self.max_bounds[i]:
                    return False, None, None
            else:
                # Calculate intersections with the slabs
                t1 = (self.min_bounds[i] - ray_origin[i]) / ray_direction[i]
                t2 = (self.max_bounds[i] - ray_origin[i]) / ray_direction[i]
                
                # Ensure t1 <= t2
                if t1 > t2:
                    t1, t2 = t2, t1
                
                # Update closest intersection
                if t1 > t_min:
                    t_min = t1
                    normal = np.zeros(3)
                    normal[i] = -1 if ray_direction[i] > 0 else 1
                
                t_max = min(t_max, t2)
                
                # Exit early if no intersection possible
                if t_min > t_max:
                    return False, None, None
        
        # Check if intersection is in front of the ray origin
        if t_min < 0:
            if t_max < 0: 
                 return False, None, None
            return False, None, None 
            
        return True, t_min, normal
    
    def get_id(self):
        """Return the building's ID."""
        return self.id
    
    def get_position(self):
        """Return the building's center position."""
        return self.position.copy()
    
    def get_dimensions(self):
        """Return the building's dimensions."""
        return self.dimensions.copy()
    
    def get_bounds(self):
        """Return the building's min and max bounds."""
        return self.min_bounds.copy(), self.max_bounds.copy()
    
    def get_reflection_coeff(self, angle_of_incidence=0.0, frequency_ghz=2.5):
        """
        Return the reflection coefficient.
        
        Can be extended to handle angle-dependent and frequency-dependent reflection.
        """
        # Simple reflection model for now - can be extended
        return self.reflection_coefficient 

    def add_window(self, face_index: int, position_on_face: List[float], 
                   window_dimensions: List[float], material: str = "glass", 
                   opacity: float = 0.5, reflection_amplitude: float = 0.6):
        """
        Adds a window to a specific face of the building.

        Creates a Window object and stores it.

        Args:
            face_index (int): Index of the face the window is on.
            position_on_face (List[float]): 2D coordinates [u, v] of the window's bottom-left.
            window_dimensions (List[float]): 2D dimensions [width, height] of the window.
            material (str): Material of the window.
            opacity (float): Visual opacity.
            reflection_amplitude (float): Amplitude of the reflection coefficient.
        """
        if not (0 <= face_index <= 5):
            raise ValueError("face_index must be between 0 and 5.")
        if len(position_on_face) != 2 or len(window_dimensions) != 2:
            raise ValueError("position_on_face and window_dimensions must be 2D.")
        
        # Create a Window object, passing self (this Building instance)
        new_window = Window(
            building             = self,
            face_index           = face_index,
            position_on_face     = position_on_face,
            dimensions_on_face   = window_dimensions,
            material             = material,
            opacity              = opacity,
            reflection_amplitude = reflection_amplitude
        )
        self.windows.append(new_window)

    def get_windows(self) -> List[Window]:
        """Returns the list of Window objects for this building."""
        return self.windows 
