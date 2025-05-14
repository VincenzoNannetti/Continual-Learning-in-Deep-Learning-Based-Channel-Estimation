"""
Filename: ./data_generation/ray_tracing/classes/antenna.py
Author: Vincenzo Nannetti
Date: 09/05/2025 (Refactored Date)
Description: Antenna Class.
"""
import numpy as np

class Antenna:
    def __init__(self, name, antenna_type, gain, pos):
        """
        Initialise an antenna.
        
        Args:
            name: Name of the antenna (BS or UE etc.)
            antenna_type: Type of antenna ("TX" or "RX")
            gain: Antenna gain in dBi
            pos: 3D position as [x, y, z] where:
                 x = East coordinate (m)
                 y = North coordinate (m)
                 z = Height above ground (m)
        """
        self.name = name
        self.antenna_type = antenna_type
        self.gain = gain
        
        # Convert position to numpy array if it isn't already
        if not isinstance(pos, np.ndarray):
            pos = np.array(pos, dtype=float)
            
        # Validate position is 3D
        if len(pos) != 3:
            raise ValueError(f"Position must be 3D [x, y, z], got {len(pos)} elements")
        
        self.pos = pos

    def get_name(self):
        return self.name

    def get_gain(self):
        return self.gain

    def get_pos(self):
        return self.pos.copy()

    def get_antenna_type(self):
        return self.antenna_type




