"""
Filename: ./data_generation/ray_tracing/classes/objects/window.py
Author: Vincenzo Nannetti
Date: 17/05/2025 (Refactored Date)
Description: Window Class for the ray tracing model.
             Represents a window on a building face, acting as a potential reflector.
"""
import numpy as np
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from .building import Building 

class Window:
    def __init__(self,
                 building: 'Building', # Reference to the parent building
                 face_index: int,
                 position_on_face: List[float], # 2D [u,v] of bottom-left on face
                 dimensions_on_face: List[float], # 2D [width, height] on face
                 material: str = "glass",
                 opacity: float = 0.5,
                 reflection_amplitude: float = 0.6,
                 reflection_angle_threshold: float = np.pi/2):
        """
        Initialise a window on a building face.

        Args:
            building: The parent Building object.
            face_index: Index of the face the window is on. Convention:
                        0: +X face (Building's local +X direction)
                        1: -X face
                        2: +Y face
                        3: -Y face
                        4: +Z face (roof)
                        5: -Z face (floor/ground-facing)
            position_on_face: [u,v] of window's bottom-left corner on the specified face.
                              The (u,v) axes are 2D local coordinates on the face plane.
                              Origin (0,0) for u,v is typically the min-corner of the face.
                              - For +/-X faces: u is along global Y, v is along global Z.
                              - For +/-Y faces: u is along global X, v is along global Z.
                              - For +/-Z faces: u is along global X, v is along global Y.
                              The `_calculate_world_geometry` method implements the precise mapping.
            dimensions_on_face: [width, height] of window on face, corresponding to (u,v) directions.
            material: Window material.
            opacity: Visual opacity (0 fully transparent, 1 fully opaque).
            reflection_amplitude: Amplitude of the reflection coefficient (0-1).
            reflection_angle_threshold: Max angle (radians) from normal for reflection to occur.
        """
        self.parent_building = building
        self.face_index = face_index
        self.position_on_face = np.array(position_on_face, dtype=float)
        self.dimensions_on_face = np.array(dimensions_on_face, dtype=float) # [width, height]
        self.material = material
        self.opacity = opacity
        self.reflection_amplitude = reflection_amplitude
        self.reflection_coeff = self.reflection_amplitude * np.exp(1j * np.random.uniform(0, 2 * np.pi))
        self.reflection_angle_threshold = reflection_angle_threshold

        # Calculate and store world geometry upon initialization
        self.center: np.ndarray
        self.normal: np.ndarray
        self._calculate_world_geometry()

    def _calculate_world_geometry(self):
        """Calculates and stores the 3D world center and normal vector for this window."""
        building_pos = self.parent_building.get_position() # Access parent building's position
        building_dims = self.parent_building.get_dimensions() # Access parent building's dimensions
        half_dims = building_dims / 2.0
        
        pos_on_face = self.position_on_face
        dims_on_face = self.dimensions_on_face

        win_world_center = np.zeros(3)
        win_world_normal = np.zeros(3)

        if self.face_index == 0: # +X face
            win_world_normal = np.array([1, 0, 0])
            win_world_center[0] = building_pos[0] + half_dims[0]
            win_world_center[1] = building_pos[1] - half_dims[1] + pos_on_face[0] + dims_on_face[0] / 2
            win_world_center[2] = building_pos[2] - half_dims[2] + pos_on_face[1] + dims_on_face[1] / 2
        elif self.face_index == 1: # -X face
            win_world_normal = np.array([-1, 0, 0])
            win_world_center[0] = building_pos[0] - half_dims[0]
            win_world_center[1] = building_pos[1] - half_dims[1] + pos_on_face[0] + dims_on_face[0] / 2
            win_world_center[2] = building_pos[2] - half_dims[2] + pos_on_face[1] + dims_on_face[1] / 2
        elif self.face_index == 2: # +Y face
            win_world_normal = np.array([0, 1, 0])
            win_world_center[0] = building_pos[0] - half_dims[0] + pos_on_face[0] + dims_on_face[0]/2
            win_world_center[1] = building_pos[1] + half_dims[1]
            win_world_center[2] = building_pos[2] - half_dims[2] + pos_on_face[1] + dims_on_face[1] / 2
        elif self.face_index == 3: # -Y face
            win_world_normal = np.array([0, -1, 0])
            win_world_center[0] = building_pos[0] - half_dims[0] + pos_on_face[0] + dims_on_face[0]/2
            win_world_center[1] = building_pos[1] - half_dims[1]
            win_world_center[2] = building_pos[2] - half_dims[2] + pos_on_face[1] + dims_on_face[1] / 2
        elif self.face_index == 4: # +Z face (roof)
            win_world_normal = np.array([0,0,1])
            win_world_center[0] = building_pos[0] - half_dims[0] + pos_on_face[0] + dims_on_face[0]/2
            win_world_center[1] = building_pos[1] - half_dims[1] + pos_on_face[1] + dims_on_face[1]/2
            win_world_center[2] = building_pos[2] + half_dims[2]
        elif self.face_index == 5: # -Z face (floor)
            win_world_normal = np.array([0,0,-1])
            win_world_center[0] = building_pos[0] - half_dims[0] + pos_on_face[0] + dims_on_face[0]/2
            win_world_center[1] = building_pos[1] - half_dims[1] + pos_on_face[1] + dims_on_face[1]/2
            win_world_center[2] = building_pos[2] - half_dims[2]
        else:
            # This case should ideally not be reached if face_index is validated beforehand
            raise ValueError(f"Invalid face_index: {self.face_index} in Window._calculate_world_geometry")

        self.center = win_world_center
        self.normal = win_world_normal


    def intersects_ray(self, ray_origin: np.ndarray, ray_direction: np.ndarray):
        """
        Ray-plane intersection against this window pane.
        Returns (hit: bool, t: float, normal: np.ndarray).
        """
        # 1. Normalize inputs
        ray_origin = np.array(ray_origin, dtype=float)
        ray_dir = ray_direction / np.linalg.norm(ray_direction)

        # 2. Compute denominator for plane intersection
        denom = np.dot(ray_dir, self.normal)
        if abs(denom) < 1e-6:
            # Ray is parallel to window plane
            return False, None, None

        # 3. Distance along ray to plane
        t = np.dot(self.center - ray_origin, self.normal) / denom
        if t < 0:
            # Intersection is behind the ray origin
            return False, None, None

        # 4. Compute hit point
        hit_pt = ray_origin + t * ray_dir

        # 5. Choose in-plane axes (u, v) depending on face
        if self.face_index in (0, 1):        # ±X faces
            u_vec, v_vec = np.array([0,1,0]), np.array([0,0,1])
        elif self.face_index in (2, 3):      # ±Y faces
            u_vec, v_vec = np.array([1,0,0]), np.array([0,0,1])
        else:                                # ±Z faces
            u_vec, v_vec = np.array([1,0,0]), np.array([0,1,0])

        # 6. Project hit point onto (u,v)
        local_u = np.dot(hit_pt - self.center, u_vec)
        local_v = np.dot(hit_pt - self.center, v_vec)
        half_w, half_h = self.dimensions_on_face / 2

        # 7. Check if within rectangle
        if abs(local_u) <= half_w and abs(local_v) <= half_h:
            return True, t, self.normal

        return False, None, None


    def get_reflection_coefficient(self, ray_direction: np.ndarray) -> complex:
        """
        Get the reflection coefficient for a given ray direction.
        Only return it if the ray is incident at an angle less than the threshold (critical angle).
        """
        cos_theta = abs(np.dot(ray_direction, self.normal))
        theta = np.arccos(cos_theta)
        if theta <= self.reflection_angle_threshold:
            return self.reflection_coeff
        return 0+0j
    
    def contains_point(self, p: np.ndarray) -> bool:
        """
        Is the point p on the window rectangle?
        """
        # choose local (u,v) axes exactly as in intersects_ray()
        if self.face_index in (0, 1):  u_vec, v_vec = np.array([0,1,0]), np.array([0,0,1])
        elif self.face_index in (2, 3): u_vec, v_vec = np.array([1,0,0]), np.array([0,0,1])
        else:                           u_vec, v_vec = np.array([1,0,0]), np.array([0,1,0])

        disp   = p - self.center
        local  = np.array([np.dot(disp, u_vec), np.dot(disp, v_vec)])
        half_w = self.dimensions_on_face[0] / 2
        half_h = self.dimensions_on_face[1] / 2
        return (-half_w <= local[0] <= half_w) and (-half_h <= local[1] <= half_h)

    def get_center(self) -> np.ndarray:
        return self.center.copy()

    def get_normal(self) -> np.ndarray:
        return self.normal.copy()
        
    def get_dimensions_on_face(self) -> np.ndarray:
        return self.dimensions_on_face.copy()

    def get_opacity(self) -> float:
        return self.opacity
        
    def get_material(self) -> str:
        return self.material

