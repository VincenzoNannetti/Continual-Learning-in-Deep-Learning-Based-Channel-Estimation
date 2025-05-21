"""
Filename: ./data_generation/ray_tracing/classes/scatterer/scatterer.py
Author: Vincenzo Nannetti 
Date: 13/05/2025
Description: Scatterer class for the ray-tracing PoC - now with
             material-aware reflection coefficients & slightly faster Doppler.
"""
import math
from typing import List, Optional, Union
import numpy as np

from ..vector_utils import (
    Vector,
    distance3d,
    ensure_vector3d,
    generate_random_3d_direction,
    unit_vector3d,
    zenith_azimuth,
)

# Import all movement functions and the MovementType enum
from .scatterer_movements import (
    linear_move, 
    random_walk_move, 
    flocking_move,
    static_move,  
    MovementType  
)

# physical constants & material DB – values taken from ITU‑R P.2040/527 tables
# (band validity in GHz) 
EPS0 = 8.854_187_817e-12  # vacuum permittivity [F/m]

MATERIAL_PROPERTIES: dict[str, tuple[float, float, tuple[float, float]]] = {
    # material           ε_rʹ    σ [S/m]   (f_min, f_max) in GHz
    "vacuum":             (1.00, 0.0000, (0.001, 100)),
    "concrete":           (5.24, 0.0462, (1, 100)),
    "brick":              (3.91, 0.0238, (1, 40)),
    "plasterboard":       (2.73, 0.0085, (1, 100)),
    "wood":               (1.99, 0.0047, (0.001, 100)),
    "glass":              (6.31, 0.0036, (0.1, 100)),
    "metal":              (1.00, 1e7,    (0.1, 100)),  # treated as PEC
    "very_dry_ground":    (3.0,  0.00015,(1, 10)),
    "medium_dry_ground":  (15.0, 0.035,  (1, 10)),
    "wet_ground":         (30.0, 0.15,   (1, 10)),
}

# Fresnel helper 
def fresnel_reflection_normal_incidence(eps_r: float, sigma: float, freq_hz: float) -> float:
    """Return **linear** |Γ| for normal incidence air->material interface.

    Parameters
    ----------
    eps_r : float
        Real part of the relative permittivity (ε_r').
    sigma : float
        Conductivity [S/m].
    freq_hz : float
        RF frequency [Hz].
    """
    eps_r_dblprime = sigma / (2 * math.pi * freq_hz * EPS0)
    eps_r_complex = eps_r - 1j * eps_r_dblprime
    n = np.sqrt(eps_r_complex)  # complex refractive index
    gamma = (1 - n) / (1 + n)
    return abs(gamma)


class Scatterer:
    """A single scatterer (point reflector) in the ray-tracing scene."""

    def __init__(self, id_: str,pos: Union[List[float], np.ndarray], speed: float, 
                 use_aggregate_scattering: bool, 
                 environment_type: str = "UMa",
                 reflection_params: Optional[dict] = None,
                 frequency_ghz: float = 2.49,
                 movement_type: str = "linear",
                 movement_specific_params: Optional[dict] = None,
                 **kwargs):
        
        # RNG & identifiers 
        self.id_        = id_
        self.cluster_id = int(id_.split('-')[0])  
        self.rng        = np.random.default_rng()

        # position & velocity 
        self.pos         = ensure_vector3d(pos)
        self.initial_pos = self.pos.copy() 
        self.speed       = float(speed)
        if self.speed > 0:
            self.velocity_vector = generate_random_3d_direction(self.rng) * self.speed
        else:
            self.velocity_vector = np.zeros(3, dtype=float)

        # environment & settings 
        self.use_aggregate_scattering = bool(use_aggregate_scattering)
        self.environment_type         = environment_type
        self.frequency_ghz            = float(frequency_ghz)

        # reflection coefficient 
        self.reflection_amplitude: float
        if reflection_params and "reflection_amplitude" in reflection_params:
            lo, hi = reflection_params["reflection_amplitude"]
            self.reflection_amplitude = self.rng.uniform(lo, hi)
        else:
            self.reflection_amplitude = self._sample_reflection_from_materials()

        self.reflection_phase = self.rng.uniform(0, 2 * math.pi)
        self.reflection_coeff = self.reflection_amplitude * np.exp(1j * self.reflection_phase)
        
        # Movement-specific initialisation
        self.time_elapsed_for_movement = 0.0
        self.movement_specific_params = movement_specific_params or {}

        # Convert movement_type string to MovementType enum member
        try:
            self.movement_model = MovementType(movement_type.lower())
        except ValueError:
            print(f"Warning: Invalid movement type '{movement_type}'. Defaulting to linear.")
            self.movement_model = MovementType.LINEAR

    def _sample_reflection_from_materials(self):
        """Pick a realistic |Γ| for the current environment & freq."""
        env2mats = {
            "UMa": ["concrete", "glass", "brick", "metal"],
            "UMi": ["concrete", "glass", "brick", "plasterboard"],
            "RMa": ["very_dry_ground", "medium_dry_ground", "wood", "wet_ground"],
            "InF": ["metal", "concrete", "glass"],
            "InH": ["plasterboard", "wood", "glass"],
        }
        mats = env2mats.get(self.environment_type, ["concrete", "glass"])

        gammas: list[float] = []
        for m in mats:
            if m not in MATERIAL_PROPERTIES:
                continue
            eps_r, sigma, (fmin, fmax) = MATERIAL_PROPERTIES[m]
            # Clamp requested freq to validity band
            f_ghz = np.clip(self.frequency_ghz, fmin, fmax)
            gammas.append(fresnel_reflection_normal_incidence(eps_r, sigma, f_ghz * 1e9))

        if not gammas:
            return 0.5 

        g_min, g_max = min(gammas), max(gammas)
        g_mid = 0.5 * (g_min + g_max)
        # triangular distribution – peak at mid
        return self.rng.triangular(g_min, g_mid, g_max)

    def get_id(self):
        return self.id_

    def get_pos(self):
        return self.pos.tolist()

    def get_velocity_vector(self):
        return self.velocity_vector

    def update_pos(self, time_step: float = 1.0, movement_type: str = "linear", environment_dims=None):
        """
        Update position based on the specified movement type by dispatching to helper functions.
        
        Args:
            time_step: Time step in seconds
            movement_type: "linear", "random_walk", or "sinusoidal"
            environment_dims: Dimensions of the environment [x_max, y_max, z_max]
            direction_change_prob: Probability of changing direction (for random_walk)
            max_angle_change: Maximum angle change in radians (for random_walk)
        """
        # Use the enum member for dispatch
        current_movement_type_enum = MovementType(movement_type.lower()) 

        # Check for negligible movement conditions first
        if self.speed <= 1e-9 and not current_movement_type_enum == MovementType.STATIC:
            return 

        self.time_elapsed_for_movement += time_step
        
        new_pos      = self.pos
        new_velocity = self.velocity_vector

        if current_movement_type_enum == MovementType.LINEAR:
            new_pos, new_velocity = linear_move(
                current_pos=self.pos,
                current_velocity=self.velocity_vector,
                time_step=time_step,
                speed=self.speed,
                environment_dims=environment_dims
            )
        elif current_movement_type_enum == MovementType.RANDOM_WALK:
            # Use movement-specific parameters if available, otherwise use defaults
            direction_change_prob = self.movement_specific_params.get('direction_change_prob', 0.4)
            max_angle_change      = self.movement_specific_params.get('max_angle_change', np.pi/6)
            
            new_pos, new_velocity = random_walk_move(
                current_pos=self.pos,
                current_velocity=self.velocity_vector,
                time_step=time_step,
                speed=self.speed,
                rng=self.rng,
                direction_change_prob=direction_change_prob,
                max_angle_change=max_angle_change,
                environment_dims=environment_dims
            )
        elif current_movement_type_enum == MovementType.FLOCKING:
            new_pos, new_velocity = flocking_move(
                current_pos=self.pos,
                current_flock_velocity=self.velocity_vector, 
                time_step=time_step,
                environment_dims=environment_dims
            )
        elif current_movement_type_enum == MovementType.STATIC: 
            new_pos, new_velocity = static_move(
                current_pos=self.pos,
                current_velocity=self.velocity_vector, 
                time_step=time_step, 
                speed=self.speed, 
                environment_dims=environment_dims 
            )
        
        self.pos             = new_pos
        self.velocity_vector = new_velocity

    def get_speed(self):
        return self.speed

    def compute_distance(self, Rx_pos: Vector, Tx_pos: Vector):
        tx3d, rx3d = ensure_vector3d(Tx_pos), ensure_vector3d(Rx_pos)
        return distance3d(self.pos, tx3d) + distance3d(self.pos, rx3d)

    def compute_angle(self, Rx_pos: Vector, Tx_pos: Vector):
        tx3d, rx3d = ensure_vector3d(Tx_pos), ensure_vector3d(Rx_pos)
        AOD_zen, AOD_az = zenith_azimuth(tx3d, self.pos)
        AOA_zen, AOA_az = zenith_azimuth(self.pos, rx3d)
        AOA_az = (AOA_az + math.pi) % (2 * math.pi)
        return (AOD_az, AOD_zen), (AOA_az, AOA_zen)

    def calculate_doppler_frequency(self, tx_pos: Vector, rx_pos: Vector, wavelength: float):
        if self.speed < 1e-9 or wavelength == 0:
            return 0.0
        tx3d, rx3d = ensure_vector3d(tx_pos), ensure_vector3d(rx_pos)
        v_unit = self.velocity_vector / self.speed
        # direction unit vectors
        dir_tx_sc = unit_vector3d(tx3d, self.pos)
        dir_sc_rx = unit_vector3d(self.pos, rx3d)
        cos_theta_tx = np.dot(v_unit, dir_tx_sc)
        cos_theta_rx = np.dot(v_unit, dir_sc_rx)
        return (self.speed / wavelength) * (cos_theta_tx + cos_theta_rx)

    def get_reflection_coeff(self):
        return self.reflection_coeff
