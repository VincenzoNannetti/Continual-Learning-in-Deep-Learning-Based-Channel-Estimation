"""
Filename: data_generation/ray_tracing/examples/visualise_street_environment.py
Author: Vincenzo Nannetti
Date: May 16, 2025
Description: Script to generate and visualise a simple "main street"
             with buildings lining both sides of a single road, irregular spacing,
             and the base station on the 2nd tallest rooftop.
Usage:
    python -m data_generation.ray_tracing.examples.visualise_street_environment
"""

import numpy as np
import random
from data_generation.ray_tracing.classes.environment import Environment
from data_generation.ray_tracing.classes.objects.building import Building
from data_generation.ray_tracing.classes.vector_utils import ensure_vector3d

def main():
    # Environment: 200 m long in X, 100 m wide in Y, 100 m tall in Z
    env_dimensions = ensure_vector3d([200, 100, 100])
    env = Environment(dimensions=env_dimensions)

    # Road parameters
    road_y = 50                  # Y-coordinate of the road centre line
    road_width = 10              # road width (m)
    setback = road_width/2 + 5   # distance from road centre to building face
    num_per_side = 10            # buildings on each side
    
    # Build up a list of along-street positions with random widths, gaps & heights
    floor_height = 3.0           # metres per storey
    positions = []
    x_offset = 10.0              # start 10 m in from the X=0 edge
    
    for _ in range(num_per_side):
        # frontage width and depth
        width = random.uniform(10, 15)
        depth = random.uniform(8, 20)
        # gap after previous building (skip for first)
        if positions:
            gap = random.uniform(3, 7)
            x_offset += gap
        
        # centre‐line X
        x_centre = x_offset + width/2.0
        
        # choose realistic floor count
        r = random.random()
        if   r < 0.6: floors = random.randint(2, 6)
        elif r < 0.9: floors = random.randint(7, 15)
        else:         floors = random.randint(16, 30)
        height = floors * floor_height
        
        positions.append({
            "x":      x_centre,
            "width":  width,
            "depth":  depth,
            "height": height
        })
        
        # move offset to the end of this building
        x_offset += width

    # Now place buildings on both sides of the road
    rooftops = []   # will hold (x, y, height) for each building
    id_counter = 0
    for side in (-1, +1):
        for pos in positions:
            x, width, depth, height = pos["x"], pos["width"], pos["depth"], pos["height"]
            y = road_y + side * (setback + depth/2.0)
            z = height/2.0
            
            env.add_building(
                id_prefix=f"bldg{id_counter}",
                position=[x, y, z],
                dimensions=[width, depth, height],
                material="concrete",
                reflection_coefficient=0.7
            )
            rooftops.append((x, y, height))
            id_counter += 1

    # Pick the 2nd tallest building for BS1
    heights = [h for (_, _, h) in rooftops]
    sorted_idxs = sorted(range(len(heights)), key=lambda i: heights[i], reverse=True)
    bs_idx = sorted_idxs[1] if len(sorted_idxs) > 1 else sorted_idxs[0]
    tx_x, tx_y, tx_roof_z = rooftops[bs_idx]
    tx_z = tx_roof_z + 3.0   # 3 m above roof
    
    env.place_tx("BS1", ensure_vector3d([tx_x, tx_y, tx_z]), gain_dbi=15)
    # Place UE1 on the pavement
    # Pavement is between road_edge (road_y - road_width/2) and building_face (road_y - setback)
    # Setback = road_width/2 + 5. Pavement width = 5m.
    # Middle of pavement on one side: road_y - (road_width/2 + 2.5) = road_y - 7.5
    ue1_y_position = road_y - 7.5 
    env.place_rx("UE1", ensure_vector3d([190, ue1_y_position, 1.5]), gain_dbi=0)

    # Add ground-level clusters (cars/pedestrians)
    env.add_cluster(
        center=ensure_vector3d([50, road_y, 1.5]), 
        num_scatterers=30, 
        radius=5, 
        height_range=2
    )
    env.add_cluster(
        center=ensure_vector3d([150, road_y + 5, 1.5]), # Slightly off center of road
        num_scatterers=30, 
        radius=5, 
        height_range=2
    )

    # Add more street-level clusters
    env.add_cluster(
        center=ensure_vector3d([25, road_y -2, 1.5]), # Near one end of the road
        num_scatterers=25,
        radius=4,
        height_range=2
    )
    env.add_cluster(
        center=ensure_vector3d([100, road_y + 2, 1.5]), # Mid-road, slightly offset
        num_scatterers=25,
        radius=4,
        height_range=2
    )

    # Add aerial clusters (birds)
    env.add_cluster(
        center=ensure_vector3d([30, 30, 60]), 
        num_scatterers=20, 
        radius=8, 
        height_range=8 # Effectively radius for Z spread if not specified differently
    )
    env.add_cluster(
        center=ensure_vector3d([170, 70, 70]), 
        num_scatterers=20, 
        radius=8, 
        height_range=8
    )

    print(
        f"Visualising {len(env.buildings)} buildings and {len(env.scatterers)} scatterers in {len(env.clusters)} clusters, "
        f"BS1 on 2nd tallest ({tx_roof_z:.1f} m) at bldg{bs_idx}, antenna at {tx_z:.1f} m."
    )
    env.visualise_environment(
        show_bounds=True,
        show_rays=False,
        show_movement=False,
        show_grid=True
    )

if __name__ == "__main__":
    main()
