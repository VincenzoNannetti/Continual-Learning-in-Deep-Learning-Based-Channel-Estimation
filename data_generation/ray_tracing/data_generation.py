# main script for data generation
# python -m data_generation.ray_tracing.data_generation --config "C:\Users\vrnan\OneDrive - Imperial College London\Year 4\ELEC70017 - Individual Project\Project\data_generation\config\dataset_a.yaml" --num_samples 1        

import argparse
import numpy as np
import os
import scipy.io
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import random
import time
import yaml 
import copy 

from data_generation.ray_tracing.classes.environment import Environment
from data_generation.ray_tracing.classes.vector_utils import ensure_vector3d
from data_generation.ray_tracing.classes.channel import Channel

def sanitize_dict_for_matlab(d):
    """Recursively sanitizes a dictionary for MATLAB .mat file saving.
    Replaces None with empty strings.
    Converts other problematic types if necessary (not implemented here yet).
    """
    if not isinstance(d, dict):
        # If it's a list, sanitize its elements
        if isinstance(d, list):
            return [sanitize_dict_for_matlab(item) for item in d]
        return d # Return as is if not a dict or list (e.g. string, number)

    new_dict = {}
    for k, v in d.items():
        if isinstance(v, dict):
            new_dict[k] = sanitize_dict_for_matlab(v)
        elif isinstance(v, list):
            new_dict[k] = [sanitize_dict_for_matlab(item) for item in v] # Sanitize list elements
        elif v is None:
            new_dict[k] = ""  # Replace None with an empty string
        else:
            new_dict[k] = v
    return new_dict

def load_config_from_path(config_path): # Helper function
    """Load configuration from YAML file"""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def create_default_environment(config_dict): # Expects a config dictionary
    """
    Creates a predefined default environment, similar to visualise_building_example.py.
    Uses some parameters from the global config where appropriate.
    """
    env_cfg         = config_dict.get('environment', {})
    clusters_cfg    = config_dict.get('clusters', {})
    antennas_cfg    = config_dict.get('antennas', {})
    ue_movement_cfg = antennas_cfg.get('rx_antenna', {}).get('movement', {})

    FIXED_SEED = 42 # A constant seed value
    original_random_state    = None
    original_np_random_state = None
    use_const_env            = env_cfg.get('use_constant_environment', False)

    if use_const_env:
        original_random_state    = random.getstate()
        original_np_random_state = np.random.get_state()
        random.seed(FIXED_SEED)
        np.random.seed(FIXED_SEED)

    # Environment
    scatterer_movement_type = clusters_cfg.get('movement_type', 'random_walk')
    env_params_for_class    = env_cfg if env_cfg else config_dict 
    env                     = Environment(config=env_params_for_class, movement_type=scatterer_movement_type)

    # Road parameters
    road_y         = 50                                    # Y-coordinate of the road centre line
    road_width     = env_cfg.get('road_width', 20)         # road width (m)
    pavement_width = env_cfg.get('pavement_width', 3.5)    # pavement width (m)
    setback        = road_width/2 + pavement_width         # distance from road centre to building face
    num_per_side   = env_cfg.get('buildings_per_side', 10) # buildings on each side
    floor_height   = env_cfg.get('floor_height', 3.0)      # metres per storey

    rooftops       = []
    id_counter     = 0

    # Add the buildings along the road 
    for side in (-1, +1):
        positions_this_side = [] # Generate positions independently for this side
        x_offset = 10.0          # Reset x_offset for each side
        
        for _ in range(num_per_side):
            width = random.uniform(10, 15)
            depth = random.uniform(10, 15)
            if positions_this_side: # Check current side's list for gap
                gap = random.uniform(3, 7)
                x_offset += gap
            x_centre = x_offset + width/2.0
            
            r = random.random()
            if   r < 0.6: floors = random.randint(2, 6)
            elif r < 0.9: floors = random.randint(7, 15)
            else:         floors = random.randint(16, 30)
            height = np.clip(floors * floor_height, 0.8*25, 1.2*25)
            
            positions_this_side.append({
                "x":      x_centre,
                "width":  width,
                "depth":  depth,
                "height": height
            })
            x_offset += width

        # Now place buildings for this specific side using its generated positions
        for pos_data in positions_this_side:
            x, width, depth, height = pos_data["x"], pos_data["width"], pos_data["depth"], pos_data["height"]
            y = road_y + side * (setback + depth/2.0)
            z = height/2.0
            
            building_id = f"bldg{id_counter}"
            current_building = env.add_building(
                id_prefix=building_id, 
                position=[x, y, z],
                dimensions=[width, depth, height],
                material="concrete",
                reflection_coefficient=config_dict.get('building_reflection_coefficient', 0.7)
            )
            rooftops.append({"x": x, "y": y, "z_roof": height, "width": width, "depth": depth, "id": building_id})
            id_counter += 1

            if current_building: # If building was successfully added
                num_floors = int(height / floor_height)
                gap_size = 0.1 # Define a small gap, e.g., 10 cm

                actual_building_face_width = current_building.dimensions[0]
                win_width = actual_building_face_width - (2 * gap_size)
                win_height = floor_height - (2 * gap_size)
                win_opacity = 0.6

                if win_width <= 0 or win_height <= 0:
                    continue 

                street_face_idx = -1
                if side == -1: 
                    street_face_idx = 2 
                else: 
                    street_face_idx = 3 

                if street_face_idx != -1:
                    u_pos_on_face = gap_size
                    for floor_num in range(num_floors):
                        v_pos_on_face = (floor_num * floor_height) + gap_size
                        if (floor_num * floor_height) + floor_height > height + 1e-6: 
                            continue
                        current_building.add_window(
                            face_index=street_face_idx,
                            position_on_face=[u_pos_on_face, v_pos_on_face],
                            window_dimensions=[win_width, win_height],
                            material="glass", 
                            reflection_amplitude=0.8, 
                            opacity=win_opacity
                        )

    # Antennas
    tx_gain = antennas_cfg.get('tx_antenna', {}).get('gain', 15)
    rx_gain = antennas_cfg.get('rx_antenna', {}).get('gain', 0)
 
    h_bs_target = antennas_cfg.get('tx_antenna', {}).get('target_height', 25.0) 
    if not rooftops:
        tx_antenna_pos = ensure_vector3d([env.dimensions[0] / 2, env.dimensions[1] / 4, h_bs_target])
    else:
        best_bldg      = min(rooftops, key=lambda b: abs(b["z_roof"] - h_bs_target))
        bx,  by        = best_bldg["x"], best_bldg["y"]
        bd,  bh        = best_bldg["depth"], best_bldg["z_roof"]    
        road_facing    = +1 if by < road_y else -1 
        inset          = bd/2 - 0.3                      
        mast_base      = ensure_vector3d([bx, by + road_facing * inset, bh])
        ANTENNA_ELEV   = 3.0                                                 
        tx_antenna_pos = mast_base + np.array([0.0, 0.0, ANTENNA_ELEV]) 
    tx_antenna         = env.place_tx("BS1", tx_antenna_pos, gain_dbi=tx_gain)
    
    if use_const_env and original_random_state is not None and original_np_random_state is not None:
        random.setstate(original_random_state)
        np.random.set_state(original_np_random_state)

    # UE Placement
    x_offset_fallback = env.dimensions[0] - 10.0
    if rooftops: 
        pass 
    elif num_per_side > 0 : 
        x_offset_fallback = 10.0 + num_per_side * 15 

    min_x_street_ue = 10.0 
    max_x_ue_bound  = x_offset if rooftops else x_offset_fallback 
    max_x_street_ue = max(min_x_street_ue + 1.0, max_x_ue_bound) 

    ue_z = antennas_cfg.get('rx_antenna', {}).get('target_height', 1.5)
    
    # Check if circular movement is configured and use safe placement
    movement_type = ue_movement_cfg.get('type', 'static')
    if movement_type == 'circular':
        circular_params = ue_movement_cfg.get('circular_params', {})
        center_offset = circular_params.get('center_offset', [0, 0])
        radius = circular_params.get('radius', 1.0)
        safety_margin = circular_params.get('safety_margin', 0.5)
        
        # Try to find a safe position for circular movement
        safe_ue_pos = find_safe_ue_position_for_circular_movement(
            env, road_y, setback, min_x_street_ue, max_x_street_ue, 
            ue_z, center_offset, radius, safety_margin
        )
        
        if safe_ue_pos is not None:
            ue_x, ue_y, ue_z = safe_ue_pos
        else:
            # Fallback to default placement if no safe position found
            print("Using fallback UE placement - circular movement may encounter buildings")
            ue_x = random.uniform(min_x_street_ue, max_x_street_ue)
            ue_y = random.uniform(road_y - setback, road_y + setback)
    else:
        # Default placement for non-circular movement
        ue_x = random.uniform(min_x_street_ue, max_x_street_ue)
        ue_y = random.uniform(road_y - setback, road_y + setback)

    min_x_pavement = min_x_street_ue
    max_x_pavement = max_x_street_ue 
    min_y_pavement = road_y - setback
    max_y_pavement = road_y + setback
    pavement_bounds = {
        "x_min": min_x_pavement, "x_max": max_x_pavement,
        "y_min": min_y_pavement, "y_max": max_y_pavement,
        "z_min": 0.0, "z_max": ue_z + 0.1 
    }
    rx_antenna = env.place_rx("UE1", ensure_vector3d([ue_x, ue_y, ue_z]), gain_dbi=rx_gain, movement_config=ue_movement_cfg, pavement_bounds=pavement_bounds)
    
    ue_speed_param = ue_movement_cfg.get('speed', [0.5, 2.0])
    if isinstance(ue_speed_param, list) and len(ue_speed_param) == 2:
        scatterer_speed_for_clusters = ue_speed_param
    elif isinstance(ue_speed_param, (int, float)):
        scatterer_speed_for_clusters = [0, ue_speed_param]
    else: 
        scatterer_speed_for_clusters = [0.5, 2.0]
    scatterer_speed_for_clusters = clusters_cfg.get('speed_range', scatterer_speed_for_clusters)

    cluster_density_cfg          = clusters_cfg.get('density', 0.5)
    number_scatteres_per_cluster = clusters_cfg.get('num_scatterers_per_cluster', 50) 
    num_scatterers_default       = int(number_scatteres_per_cluster * cluster_density_cfg) 
    
    use_agg_scattering_cfg = config_dict.get('use_aggregate_scattering', True)
    environment_type_cfg   = env_cfg.get('environment_type', 'UMa') 
    far_field_margin_cfg   = float(config_dict.get('far_field_margin', 6e-2))
    carrier_freq_ghz_cfg   = config_dict.get('carrier_frequency', 2900)/1000
    
    reflection_amplitude_cfg = clusters_cfg.get('reflection_amplitude', None)
    reflection_params_for_scatterer_cfg = None
    if reflection_amplitude_cfg is not None:
        if isinstance(reflection_amplitude_cfg, list) and len(reflection_amplitude_cfg) == 2:
            reflection_params_for_scatterer_cfg = {'reflection_amplitude': reflection_amplitude_cfg} 
        else: 
            reflection_params_for_scatterer_cfg = {'reflection_amplitude': [reflection_amplitude_cfg, reflection_amplitude_cfg]}

    cluster_movement_params_from_config = clusters_cfg.get("movement_params", None)

    num_aerial_clusters_rng = clusters_cfg.get('num_aerial_clusters_rng', [2, 4])
    num_ground_clusters_rng = clusters_cfg.get('num_ground_clusters_rng', [5, 12])
    num_aerial_clusters = random.randint(num_aerial_clusters_rng[0], num_aerial_clusters_rng[1])
    num_ground_clusters = random.randint(num_ground_clusters_rng[0], num_ground_clusters_rng[1])

    for _ in range(num_aerial_clusters):
        aerial_radius = np.clip(np.random.lognormal(mean=0.3, sigma=0.5), 0.1, 6.0)
        num_sc        = max(1, int(num_scatterers_default * random.uniform(0.7, 1.5)))
        env.add_cluster(
            num_scatterers=num_sc, radius=aerial_radius, cluster_category="aerial",
            height_range=random.uniform(aerial_radius*0.5, aerial_radius*1.2),
            scatterer_speed_range=scatterer_speed_for_clusters,
            use_aggregate_scattering=use_agg_scattering_cfg,
            environment_type_for_scatterer=environment_type_cfg,
            reflection_params_for_scatterer=reflection_params_for_scatterer_cfg,
            carrier_freq_ghz_for_scatterer=carrier_freq_ghz_cfg,
            min_far_field=far_field_margin_cfg,
            cluster_movement_config=cluster_movement_params_from_config
        )

    for _ in range(num_ground_clusters):
        ground_radius = np.clip(np.random.lognormal(mean=0.2, sigma=0.4), 0.1, 5.0)
        num_sc        = max(1, int(num_scatterers_default * random.uniform(0.5, 1.0)))
        env.add_cluster(
            num_scatterers=num_sc, radius=ground_radius, cluster_category="ground",
            height_range=random.uniform(clusters_cfg.get('ground_cluster_height_range', [1, 3])[0], clusters_cfg.get('ground_cluster_height_range', [1, 3])[1]),
            scatterer_speed_range=scatterer_speed_for_clusters,
            use_aggregate_scattering=use_agg_scattering_cfg,
            environment_type_for_scatterer=environment_type_cfg,
            reflection_params_for_scatterer=reflection_params_for_scatterer_cfg,
            carrier_freq_ghz_for_scatterer=carrier_freq_ghz_cfg,
            min_far_field=far_field_margin_cfg,
            cluster_movement_config=cluster_movement_params_from_config
        )

    # env.visualise_environment(show_rays=True, show_movement=True, movement_time=0.0)

    
    return env, config_dict

def process_sample(sample_idx, base_config_dict):
    time.sleep(0.001)

    # Create a fresh environment for each sample
    env, _ = create_default_environment(base_config_dict)

    snr_range = base_config_dict.get('environment', {}).get('snr', [15,15])
    snr = round(np.random.uniform(snr_range[0], snr_range[1]),3)

    channel = Channel(
        environment          = env,
        tx_antenna           = env.tx_antennas[0],
        rx_antenna           = env.rx_antennas[0],
        center_frequency     = base_config_dict.get('carrier_frequency', 2490) * 1e6,
        snr                  = snr,
        symbols_per_block    = base_config_dict.get('symbols_per_block', 14),
        shadow_fading_std_db = base_config_dict.get('shadow_fading_std_db', 4.0),
    )

    subcarrier_spacing    = base_config_dict.get('subcarrier_spacing', 60000)
    num_blocks            = base_config_dict.get('blocks', 4)
    num_subcarriers       = base_config_dict.get('num_subcarriers', 72)
    num_symbols_per_block = base_config_dict.get('symbols_per_block', 14)
    num_symbols           = num_blocks * num_symbols_per_block
    cyclic_prefix         = base_config_dict.get('cyclic_prefix', 0)

    input_data = np.ones(num_subcarriers * num_symbols, dtype=complex)

    output_signal = channel.apply_channel(
        input_data,
        subcarrier_spacing,
        num_subcarriers,
        cyclic_prefix
    )

    perfect_matrix  = channel.get_channel_matrix()
    received_matrix = channel.get_channel_matrix_noisy()

    return sample_idx, perfect_matrix, received_matrix

def check_circular_path_safety(center_pos, radius, buildings, safety_margin=0.5, num_check_points=16):
    """
    Check if a circular path around center_pos with given radius would intersect any buildings.
    
    Args:
        center_pos: [x, y, z] center of the circular path
        radius: radius of the circular path
        buildings: list of building objects
        safety_margin: additional clearance from buildings (metres)
        num_check_points: number of points to check around the circle
        
    Returns:
        bool: True if path is safe (no building intersections), False otherwise
    """
    if not buildings:
        return True
    
    # Check points around the circle
    for i in range(num_check_points):
        angle = 2 * np.pi * i / num_check_points
        x = center_pos[0] + radius * np.cos(angle)
        y = center_pos[1] + radius * np.sin(angle)
        z = center_pos[2]  # Keep same height
        
        # Check if this point is too close to any building (including safety margin)
        for building in buildings:
            b_min, b_max = building.get_bounds()
            # Expand building bounds by safety margin
            expanded_min = [b_min[0] - safety_margin, b_min[1] - safety_margin, b_min[2]]
            expanded_max = [b_max[0] + safety_margin, b_max[1] + safety_margin, b_max[2]]
            
            if (expanded_min[0] <= x <= expanded_max[0] and 
                expanded_min[1] <= y <= expanded_max[1] and 
                expanded_min[2] <= z <= expanded_max[2]):
                return False
    
    return True

def find_safe_ue_position_for_circular_movement(env, road_y, setback, min_x_street_ue, max_x_street_ue, ue_z,         center_offset, radius, safety_margin=0.5, max_attempts=100):
    """
    Find a safe UE position that allows circular movement without building collisions.
    
    Args:
        env: Environment object
        road_y: Y-coordinate of road center
        setback: Distance from road center to building face
        min_x_street_ue, max_x_street_ue: X bounds for UE placement
        ue_z: UE height
        center_offset: [x_offset, y_offset] for circle center
        radius: circular movement radius
        safety_margin: additional clearance from buildings (metres)
        max_attempts: maximum attempts to find safe position
        
    Returns:
        np.ndarray: Safe UE position [x, y, z] or None if not found
    """
    for attempt in range(max_attempts):
        # Generate candidate UE position
        ue_x = random.uniform(min_x_street_ue, max_x_street_ue)
        ue_y = random.uniform(road_y - setback, road_y + setback)
        ue_pos = np.array([ue_x, ue_y, ue_z])
        
        # Calculate circle center based on offset
        center_pos = ue_pos + np.array([center_offset[0], center_offset[1], 0])
        
        # Check if the circular path is safe
        if check_circular_path_safety(center_pos, radius, env.buildings, safety_margin):
            # Also check that the UE starting position isn't in a building
            ue_in_building = False
            for building in env.buildings:
                b_min, b_max = building.get_bounds()
                if (b_min[0] <= ue_x <= b_max[0] and 
                    b_min[1] <= ue_y <= b_max[1] and 
                    b_min[2] <= ue_z <= b_max[2]):
                    ue_in_building = True
                    break
            
            if not ue_in_building:
                return ue_pos
    
    print(f"Warning: Could not find safe UE position for circular movement after {max_attempts} attempts")
    return None

def main():
    parser = argparse.ArgumentParser(description='Run channel simulation for data generation')
    parser.add_argument('--config', '-c', type=str, default="data_generation/config/dataset_a.yaml",
                        help='Path to the configuration file')
    parser.add_argument('--num_samples', type=int, default=2500,
                        help='Number of samples to generate')
    
    args = parser.parse_args()
    
    config_dict = load_config_from_path(args.config)

    output_dir  = config_dict.get('output_dir', "./data/raw/ray_tracing/default_domain")
    domain_name = config_dict.get('domain_name', 'domain_data') 
    os.makedirs(output_dir, exist_ok=True)

    num_samples       = args.num_samples
    NUM_SUBCARRIERS   = config_dict.get('num_subcarriers', 72) 
    SYMBOLS_PER_BLOCK = config_dict.get('symbols_per_block', 14)
    NUM_BLOCKS        = config_dict.get('blocks', 5)
    TOTAL_SYMBOLS     = SYMBOLS_PER_BLOCK * NUM_BLOCKS
    
    perfect_matrices  = np.zeros((num_samples, NUM_SUBCARRIERS, TOTAL_SYMBOLS), dtype=np.complex64)
    received_matrices = np.zeros((num_samples, NUM_SUBCARRIERS, TOTAL_SYMBOLS), dtype=np.complex64)
    
    num_workers = os.cpu_count()
    print(f"Starting data generation for domain: {domain_name}")
    print(f"Configuration loaded from: {args.config}")
    print(f"Number of samples: {num_samples}")
    print(f"Output directory: {output_dir}")
    print(f"Using {num_workers} workers for data generation.")

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_sample, i, config_dict) for i in range(num_samples)]

        for future in tqdm(as_completed(futures), total=num_samples, desc=f"Processing samples for {domain_name}"):
            try:
                i, perfect_matrix_sample, received_matrix_sample = future.result()
                perfect_matrices[i]  = perfect_matrix_sample
                received_matrices[i] = received_matrix_sample
            except Exception as e:
                print(f"Error processing sample index {i}: {e}")

    file_name = f"{domain_name}.mat" 
    full_output_path = os.path.join(output_dir, file_name)
    
    sanitised_config = sanitize_dict_for_matlab(copy.deepcopy(config_dict))

    scipy.io.savemat(full_output_path, {
        "perfect_matrices": perfect_matrices, 
        "received_matrices": received_matrices,
        "config": sanitised_config 
    })
    print(f"Data generation complete. Output saved to: {full_output_path}")

if __name__ == "__main__":
    main()
