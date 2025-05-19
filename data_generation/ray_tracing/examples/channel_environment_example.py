"""
Filename: data_generation/ray_tracing/examples/channel_environment_example.py
Author: Vincenzo Nannetti
Date: 15/07/2024
Description: Example demonstrating the integration of Environment and Channel classes using a config file.
Usage:
    List presets:
        python -m data_generation.ray_tracing.examples.channel_environment_example --list-presets
    Run with preset:
        python -m data_generation.ray_tracing.examples.channel_environment_example --config data_generation/config/dataset_a.yaml --preset UMa
    Run with config:
        python -m data_generation.ray_tracing.examples.channel_environment_example --config data_generation/config/dataset_a.yaml
    Run with default:
        python -m data_generation.ray_tracing.examples.channel_environment_example
    Run with config and preset:
        python -m data_generation.ray_tracing.examples.channel_environment_example --config data_generation/config/dataset_a.yaml --preset UMa
    Run with just a preset (no custom config):
        python -m data_generation.ray_tracing.examples.channel_environment_example --preset-only UMa
"""

import numpy as np
import matplotlib.pyplot as plt
import yaml
import random
import time
import argparse
from pathlib import Path

from data_generation.ray_tracing.classes.environment  import Environment
from data_generation.ray_tracing.classes.channel      import Channel
from data_generation.ray_tracing.classes.vector_utils import ensure_vector3d


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


def merge_configs(base_config, preset_config):
    """Merge preset config with base config, with base config taking precedence"""
    merged = preset_config.copy()
    for key, value in base_config.items():
        # Special handling for nested dictionaries
        if isinstance(value, dict) and key in merged and isinstance(merged[key], dict):
            merged[key] = {**merged[key], **value}
        else:
            merged[key] = value
    return merged


def create_default_environment(config):
    """
    Creates a predefined default environment, similar to visualise_building_example.py.
    Uses some parameters from the global config where appropriate.
    """
    # Environment: 200 m long in X, 100 m wide in Y, 100 m tall in Z
    env_dimensions          = ensure_vector3d([200, 100, 100])
    scatterer_movement_type = config.get('scatterer_movement_type', 'random_walk')
    env                     = Environment(dimensions=env_dimensions, movement_type=scatterer_movement_type)

    # Road parameters
    road_y       = 50               # Y-coordinate of the road centre line
    road_width   = 10               # road width (m)
    setback      = road_width/2 + 5 # distance from road centre to building face
    num_per_side = 10               # buildings on each side
    floor_height = 3.0              # metres per storey

    rooftops   = []
    id_counter = 0

    # Add the buildings along the road 
    for side in (-1, +1):
        positions_this_side = [] # Generate positions independently for this side
        x_offset = 10.0          # Reset x_offset for each side
        
        for _ in range(num_per_side):
            width = random.uniform(10, 15)
            depth = random.uniform(8, 20)
            if positions_this_side: # Check current side's list for gap
                gap = random.uniform(3, 7)
                x_offset += gap
            x_centre = x_offset + width/2.0
            
            r = random.random()
            if   r < 0.6: floors = random.randint(2, 6)
            elif r < 0.9: floors = random.randint(7, 15)
            else:         floors = random.randint(16, 30)
            height = floors * floor_height
            
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
                reflection_coefficient=config.get('building_reflection_coefficient', 0.7)
            )
            rooftops.append({"x": x, "y": y, "z_roof": height, "width": width, "depth": depth, "id": building_id})
            id_counter += 1

            if current_building: # If building was successfully added
                num_floors = int(height / floor_height)
                gap_size = 0.1 # Define a small gap, e.g., 10 cm

                # Full width of the building's X-dimension (which is the width of Y-facing faces)
                actual_building_face_width = current_building.dimensions[0]
                
                # Adjust window width for gaps on left and right
                win_width = actual_building_face_width - (2 * gap_size)
                
                # Adjust window height for gaps on top and bottom within the floor height
                win_height = floor_height - (2 * gap_size)
                
                win_opacity = 0.6

                # Ensure dimensions are positive after applying gaps
                if win_width <= 0 or win_height <= 0:
                    # If gaps make dimensions non-positive, skip adding windows for this building/config
                    # Or, could add a single, smaller window as a fallback, but let's skip for now.
                    continue 

                street_face_idx = -1
                if side == -1: 
                    street_face_idx = 2 
                else: 
                    street_face_idx = 3 

                if street_face_idx != -1:
                    # Horizontal start position for the window, accounting for the gap
                    u_pos_on_face = gap_size

                    for floor_num in range(num_floors):
                        # Vertical start position for the window, accounting for the bottom gap
                        v_pos_on_face = (floor_num * floor_height) + gap_size
                        
                        # Check if the window (including its top gap) would exceed total building height
                        # Total space taken by window + its gaps within this floor slot = win_height + 2*gap_size = floor_height
                        # So, top of this window slot is (floor_num * floor_height) + floor_height
                        if (floor_num * floor_height) + floor_height > height + 1e-6: # Check against actual building height
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
    tx_gain = config.get('tx_gain', 15)
    rx_gain = config.get('rx_gain', 0)

    # Select a random building for BS placement
    bs_building_info     = random.choice(rooftops)     
    tx_building_center_x = bs_building_info["x"]
    tx_building_center_y = bs_building_info["y"]
    tx_building_roof_z   = bs_building_info["z_roof"]
    tx_building_depth    = bs_building_info["depth"]
    
    # Place BS on the middle of the building's inner edge (facing the road)
    bs_on_edge_x = tx_building_center_x 
    if tx_building_center_y < road_y: 
        bs_on_edge_y = tx_building_center_y + tx_building_depth / 2 
    else: 
        bs_on_edge_y = tx_building_center_y - tx_building_depth / 2 

    tx_antenna_height_above_roof = 3.0
    tx_final_z                   = tx_building_roof_z + tx_antenna_height_above_roof
    tx_antenna_pos               = ensure_vector3d([bs_on_edge_x, bs_on_edge_y, tx_final_z])
    tx_antenna                   = env.place_tx("BS1", tx_antenna_pos, gain_dbi=tx_gain)
    
    # UE Placement: Randomly within the street canyon
    min_x_street_ue = 10.0 
    max_x_street_ue = x_offset 
    ue_x            = random.uniform(min_x_street_ue, max_x_street_ue)
    # Y-coordinate: within the street canyon (anywhere within road or pavements)
    ue_y             = random.uniform(road_y - setback, road_y + setback)
    ue_z             = 1.5 # Standard UE height (defined in the spec to be this as well)
    ue_movement_type = config.get('ue_movement', 'static')

    # Define pavement boundaries
    min_x_pavement = min_x_street_ue
    max_x_pavement = max_x_street_ue 
    min_y_pavement = road_y - setback
    max_y_pavement = road_y + setback
    pavement_bounds = {
        "x_min": min_x_pavement, "x_max": max_x_pavement,
        "y_min": min_y_pavement, "y_max": max_y_pavement,
        "z_min": 0.0, "z_max": ue_z + 0.1 
    }

    rx_antenna = env.place_rx("UE1", ensure_vector3d([ue_x, ue_y, ue_z]), gain_dbi=rx_gain, movement_config=ue_movement_type, pavement_bounds=pavement_bounds)

    speed_range_cfg = config.get('speed', [0.5, 2])
    if isinstance(speed_range_cfg, list) and len(speed_range_cfg) == 2:
        scatterer_speed_for_clusters = speed_range_cfg 
    else: 
        scatterer_speed_for_clusters = [0, speed_range_cfg] 

    cluster_density_cfg    = config.get('cluster_density', 0.5)
    num_scatterers_default = int(50 * cluster_density_cfg) 
    
    # Cluster shared parameters from config
    use_agg_scattering_cfg = config.get('use_aggregate_scattering', False)
    environment_type_cfg   = config.get('environment_type', 'UMa') 
    far_field_margin_cfg   = float(config.get('far_field_margin', 5.0))
    carrier_freq_ghz_cfg   = config.get('carrier_frequency', 2900)/1000
    
    reflection_amplitude_cfg = config.get('reflection_amplitude', None)
    reflection_params_for_scatterer_cfg = None
    if reflection_amplitude_cfg is not None:
        if isinstance(reflection_amplitude_cfg, list) and len(reflection_amplitude_cfg) == 2:
            reflection_params_for_scatterer_cfg = {'reflection_amplitude': reflection_amplitude_cfg} 
        else: 
            reflection_params_for_scatterer_cfg = {'reflection_amplitude': [reflection_amplitude_cfg, reflection_amplitude_cfg]}

    cluster_movement_params_from_config = config.get("cluster_movement_params", None)

    # Define number of clusters
    num_aerial_clusters = random.randint(2, 4)
    num_ground_clusters = random.randint(4, 8)

    # Add Aerial Clusters
    print(f"Adding {num_aerial_clusters} aerial clusters.")
    for _ in range(num_aerial_clusters):
        # 1) sample an (x,y) at least radius away from every building footprint
        aerial_radius = random.uniform(0.1, 3.5)
        num_sc = max(1, int(num_scatterers_default * random.uniform(0.7, 1.5)))
        env.add_cluster(
            num_scatterers                  = num_sc,
            radius                          = aerial_radius,
            cluster_category                = "aerial",
            height_range                    = random.uniform(aerial_radius*0.5, aerial_radius*1.2),
            scatterer_speed_range           = scatterer_speed_for_clusters,
            use_aggregate_scattering        = use_agg_scattering_cfg,
            environment_type_for_scatterer  = environment_type_cfg,
            reflection_params_for_scatterer = reflection_params_for_scatterer_cfg,
            carrier_freq_ghz_for_scatterer  = carrier_freq_ghz_cfg,
            min_far_field                   = far_field_margin_cfg,
            cluster_movement_config         = cluster_movement_params_from_config
        )

    print(f"Adding {num_ground_clusters} ground-level clusters.")
    for _ in range(num_ground_clusters):
        ground_radius = random.uniform(0.1, 3.5)
        num_sc        = max(1, int(num_scatterers_default * random.uniform(0.5, 1.0)))

        env.add_cluster(
            num_scatterers                  = num_sc,
            radius                          = ground_radius,
            cluster_category                = "ground",
            height_range                    = random.uniform(1.0, 3.0),
            scatterer_speed_range           = scatterer_speed_for_clusters,
            use_aggregate_scattering        = use_agg_scattering_cfg,
            environment_type_for_scatterer  = environment_type_cfg,
            reflection_params_for_scatterer = reflection_params_for_scatterer_cfg,
            carrier_freq_ghz_for_scatterer  = carrier_freq_ghz_cfg,
            min_far_field                   = far_field_margin_cfg,
            cluster_movement_config         = cluster_movement_params_from_config
        )
    
    print(f"Using default environment: {len(env.buildings)} buildings, {len(env.scatterers)} scatterers in {len(env.clusters)} clusters.")
    print(f"BS1 at [{tx_antenna.pos[0]:.1f}, {tx_antenna.pos[1]:.1f}, {tx_antenna.pos[2]:.1f}]m, UE1 at [{rx_antenna.pos[0]:.1f}, {rx_antenna.pos[1]:.1f}, {rx_antenna.pos[2]:.1f}]m.")

    return env, config


def run_simulation(config_path=None, preset_name=None, preset_only=False, use_default_env=False):
    """Main simulation function to be profiled"""
    base_config   = {}
    config_to_use = {} 

    # Load base configuration from --config if provided and not overridden by --preset-only
    if config_path and not preset_only:
        base_config = load_config(config_path)
        config_to_use = base_config.copy() 

    if preset_name: 
        current_script_dir = Path(__file__).resolve().parent
        presets_path = current_script_dir.parent / "config" / "3gpp_presets.yaml"        
        try:
            presets = load_config(presets_path)
            if preset_name not in presets:
                print(f"Warning: Preset '{preset_name}' not found in {presets_path}. Using previously loaded config (if any).")
            else:
                preset_config_data = presets[preset_name]
                print(f"Using preset '{preset_name}': {preset_config_data.get('description', '')}")
                
                if preset_only:
                    config_to_use = preset_config_data 
                else:                    
                    config_to_use = merge_configs(config_to_use, preset_config_data)
        except FileNotFoundError:
            print(f"Error: Presets file not found at {presets_path}. Make sure it exists.")
        except yaml.YAMLError as e:
            print(f"Error parsing presets file {presets_path}: {e}")
            
    ue_movement_config = config_to_use.get('ue_movement')

    # --- Create Environment (Default or Empty) 
    if use_default_env:
        env, config_to_use = create_default_environment(config_to_use) 
    else:
        env_dims_config = config_to_use.get('environment_dimensions', [200, 200, 50])
        env_dims = ensure_vector3d(env_dims_config)
        scatterer_movement_type_cfg = config_to_use.get('scatterer_movement_type', 'random_walk')
        env = Environment(dimensions=env_dims, movement_type=scatterer_movement_type_cfg)
        env.ue_movement_config = ue_movement_config 
        custom_pavement_bounds = None 

    if not use_default_env:
        tx_pos_cfg  = config_to_use.get('tx_antenna_pos', [env.dimensions[0]*0.1, env.dimensions[1]/2, 25])
        tx_gain_cfg = config_to_use.get('tx_gain', 15)
        env.place_tx("BS0", ensure_vector3d(tx_pos_cfg), gain_dbi=tx_gain_cfg)

        rx_pos_cfg  = config_to_use.get('rx_antenna_pos', [env.dimensions[0]*0.8, env.dimensions[1]/2, 1.5])
        rx_gain_cfg = config_to_use.get('rx_gain', 0)
        env.place_rx("UE0", ensure_vector3d(rx_pos_cfg), gain_dbi=rx_gain_cfg, movement_config=ue_movement_config, pavement_bounds=custom_pavement_bounds) 

        num_clusters_cfg = config_to_use.get('num_clusters', [3, 7])
        if isinstance(num_clusters_cfg, list) and len(num_clusters_cfg) == 2:
            num_clusters_val = random.randint(num_clusters_cfg[0], num_clusters_cfg[1])
        else:
            num_clusters_val = num_clusters_cfg
        
        speed_range = config_to_use.get('speed', [0.5, 2])
        if isinstance(speed_range, list) and len(speed_range) == 2:
            speed_val = random.uniform(speed_range[0], speed_range[1]) 
            scatterer_speed_range_for_cluster = speed_range 
        else:
            speed_val = speed_range
            scatterer_speed_range_for_cluster = [0, speed_val]


        cluster_density      = config_to_use.get('cluster_density', 0.5)
        cluster_radius       = float(config_to_use.get('cluster_radius', 15))
        cluster_height_range = config_to_use.get('cluster_height_range', cluster_radius)
        use_agg_scattering   = config_to_use.get('use_aggregate_scattering', False)
        far_field_margin     = float(config_to_use.get('far_field_margin', 5.0))
        
        # Handle reflection amplitude
        reflection_amplitude = config_to_use.get('reflection_amplitude', None)
        reflection_params = None # Initialise to None
        if reflection_amplitude is not None:
            if isinstance(reflection_amplitude, list) and len(reflection_amplitude) == 2:
                # If it's a range, add_cluster will handle sampling if Scatterer expects a range
                reflection_params = {'reflection_amplitude': reflection_amplitude}
            else: # Single value
                reflection_params = {'reflection_amplitude': [reflection_amplitude, reflection_amplitude]}


        # Time the cluster creation process
        start_time_cluster = time.time()
        
        # Add clusters randomly within the environment
        for _ in range(num_clusters_val):
            # Simple uniform placement within the central 80% of each dimension
            cluster_x = random.uniform(env.dimensions[0] * 0.1, env.dimensions[0] * 0.9)
            cluster_y = random.uniform(env.dimensions[1] * 0.1, env.dimensions[1] * 0.9)
            cluster_z = random.uniform(env.dimensions[2] * 0.1, env.dimensions[2] * 0.9)
            num_scatterers_in_cluster = int(50 * cluster_density)

            # Get cluster-specific movement parameters from config if they exist
            cluster_movement_params_from_config = config_to_use.get("cluster_movement_params", None)

            env.add_cluster(
                [cluster_x, cluster_y, cluster_z],
                num_scatterers=num_scatterers_in_cluster,
                radius=cluster_radius,
                height_range=cluster_height_range,
                scatterer_speed_range=scatterer_speed_range_for_cluster, # Use the range here
                use_aggregate_scattering_for_scatterer=use_agg_scattering,
                environment_type_for_scatterer=config_to_use.get('environment_type', 'UMa'),
                reflection_params_for_scatterer=reflection_params,
                carrier_freq_ghz_for_scatterer=config_to_use.get('carrier_frequency', 2900)/1000,
                min_far_field=far_field_margin,
                cluster_movement_config=cluster_movement_params_from_config 
            )
        
        cluster_time = time.time() - start_time_cluster
        print(f"Time to create {num_clusters_val} clusters: {cluster_time:.2f} seconds")
        if num_clusters_val > 0:
             print(f"Total scatterers: {len(env.scatterers)} (avg {len(env.scatterers)/num_clusters_val:.1f} per cluster)")
        else:
            print(f"Total scatterers: {len(env.scatterers)}")
    
    
    # Common part of simulation (Channel, OFDM, etc.) 
    snr = round(np.random.uniform(config_to_use.get('snr', [15,15])[0], config_to_use.get('snr', [15,15])[1]),3)
    
    # Create channel using parameters from config
    channel = Channel(
        environment          = env,
        tx_antenna           = env.tx_antennas[0],
        rx_antenna           = env.rx_antennas[0],
        center_frequency     = config_to_use.get('carrier_frequency', 2490) * 1e6, 
        snr                  = snr,
        pathloss_model       = config_to_use.get('pathloss_model', 'friis'),
        rms_delay_spread_ns  = config_to_use.get('rms_delay_spread_ns', 100),
        shadow_fading_std_db = config_to_use.get('shadow_fading_std_db', 4.0),
        o2i_loss             = config_to_use.get('o2i_loss', 0),
    )
   
    # Time the channel simulation
    start_time = time.time()
    channel.prepare_channel()
    
    # OFDM setup
    subcarrier_spacing    = config_to_use.get('subcarrier_spacing', 60000)
    num_blocks            = config_to_use.get('blocks', 4)
    num_subcarriers       = 72
    num_symbols_per_block = 14
    num_symbols           = num_blocks * num_symbols_per_block
    cyclic_prefix         = 0
    input_data            = np.ones(num_subcarriers * num_symbols, dtype=complex)
    
    output_signal = channel.apply_channel(
        input_data,
        subcarrier_spacing,
        num_subcarriers,
        cyclic_prefix
    )
    
    channel_time = time.time() - start_time
    print(f"Time to simulate channel: {channel_time:.2f} seconds")
    
    # Extract and display
    perfect_ch = channel.get_channel_matrix()
    noisy_ch   = channel.get_channel_matrix_noisy()
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.subplot(121)
    title_text = f'Perfect Channel ({config_to_use.get("environment_type", "UMa")})'
    if preset_name:
        title_text += f' [{preset_name}]'
    plt.title(title_text)
    plt.pcolormesh(np.abs(perfect_ch), cmap='viridis')
    plt.xlabel('Symbol'); plt.ylabel('Subcarrier')
    
    plt.subplot(122)
    plt.title(f'Noisy Channel (SNR={snr}dB)')
    plt.pcolormesh(np.abs(noisy_ch), cmap='viridis')
    plt.xlabel('Symbol'); plt.ylabel('Subcarrier')
    plt.tight_layout(); plt.show()
    
    symbol_duration = 1.0 / subcarrier_spacing
    total_simulation_time = num_symbols * symbol_duration
    print(f"Total simulated physical time: {total_simulation_time:.4f} seconds")
    
    return env, channel, perfect_ch, noisy_ch, total_simulation_time


def main():
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Run channel simulation with optional 3GPP preset')
    parser.add_argument('--config', '-c', type=str, default="data_generation/config/dataset_a.yaml",
                        help='Path to the configuration file')
    parser.add_argument('--preset', '-p', type=str, default=None,
                        help='3GPP preset to use (UMa, UMi, RMa, InH, etc.)')
    parser.add_argument('--preset-only', type=str, default=None,
                        help='Use only the specified preset with no custom config')
    parser.add_argument('--list-presets', '-l', action='store_true',
                        help='List available 3GPP presets and exit')
    parser.add_argument('--use-default-env', action='store_true',
                        help='Use a predefined default environment instead of config/preset for geometry')
    parser.add_argument('--debug-reflections', action='store_true',
                        help='Print detailed debug information about window reflections')
    
    args = parser.parse_args()
    
    # List presets if requested
    if args.list_presets:
        # Path to presets file - should be relative to this script's location for robustness
        current_script_dir = Path(__file__).resolve().parent
        presets_path = current_script_dir.parent / "config" / "3gpp_presets.yaml"
        try:
            presets = load_config(presets_path)
            print("Available 3GPP presets:")
            for preset, details in presets.items():
                print(f"  {preset}: {details.get('description', '')}")
        except FileNotFoundError:
            print(f"Error: Presets file not found at {presets_path}. Check the path.")
            return # Exit if presets can't be listed
        except yaml.YAMLError as e:
            print(f"Error parsing presets file: {e}")
            return # Exit
        return
    
    # Determine preset_name for run_simulation based on args
    sim_preset_name = args.preset_only if args.preset_only else args.preset

    # Run the simulation with the specified config and preset
    env, channel, perfect_ch, noisy_ch, total_sim_time = run_simulation(
        config_path     = args.config if not args.preset_only else None, 
        preset_name     = sim_preset_name,
        preset_only     = bool(args.preset_only), 
        use_default_env = args.use_default_env,
    )
   
    # Debug window reflections if requested
    if args.debug_reflections:
        print("\n--- Window Reflection Debug Info ---")
        tx_pos = np.array(env.tx_antennas[0].get_pos())
        rx_pos = np.array(env.rx_antennas[0].get_pos())
        
        # Count total windows
        window_count = 0
        for b in env.buildings:
            window_count += len(b.get_windows())
        print(f"Total buildings: {len(env.buildings)}")
        print(f"Total windows: {window_count}")
        
        # Get and analyze valid reflections
        valid_reflections = env.get_valid_window_reflections(tx_pos, rx_pos)
        print(f"Valid window reflections found: {len(valid_reflections)}")
        
        # If no valid reflections, let's analyze why
        if len(valid_reflections) == 0 and window_count > 0:
            print("\nWhy no reflections were found:")
            for b_idx, b in enumerate(env.buildings):
                for w_idx, w in enumerate(b.get_windows()):
                    # Get window properties
                    window_normal = w.get_normal()
                    window_center = w.get_center()
                    
                    # 1. Check incident angle against threshold
                    vec_tx_win = window_center - tx_pos
                    if np.linalg.norm(vec_tx_win) < 1e-6:
                        print(f"  Building {b_idx}, Window {w_idx}: TX coincides with window center")
                        continue
                        
                    d_in          = vec_tx_win / np.linalg.norm(vec_tx_win)
                    cos_theta     = np.abs(np.dot(d_in, window_normal))
                    angle         = np.arccos(cos_theta)
                    angle_deg     = np.degrees(angle)
                    max_angle_deg = np.degrees(w.reflection_angle_threshold)
                    
                    if angle > w.reflection_angle_threshold:
                        print(f"  Building {b_idx}, Window {w_idx}: Angle too large ({angle_deg:.1f}° > {max_angle_deg:.1f}°)")
                        continue
                        
                    # 2. Try the mirror method and check if reflection point is on window
                    n = window_normal / np.linalg.norm(window_normal)
                    p0 = window_center
                    tx_m = tx_pos - 2 * np.dot(tx_pos - p0, n) * n
                    dir_line = rx_pos - tx_m
                    denom = np.dot(dir_line, n)
                    
                    if abs(denom) < 1e-9:
                        print(f"  Building {b_idx}, Window {w_idx}: Ray parallel to window plane")
                        continue
                        
                    t = np.dot(p0 - tx_m, n) / denom
                    if not (0.0 < t < 1.0):
                        print(f"  Building {b_idx}, Window {w_idx}: Reflection outside TX-RX path")
                        continue
                        
                    p_ref = tx_m + t * dir_line
                    
                    if not w.contains_point(p_ref):
                        print(f"  Building {b_idx}, Window {w_idx}: Reflection point not on window pane")
                        continue
                        
                    # 3. Check for blockages
                    vec_tx_pref = p_ref - tx_pos
                    dist_tx_pref = np.linalg.norm(vec_tx_pref)
                    d_in = vec_tx_pref / dist_tx_pref
                    
                    if env.check_building_intersection(tx_pos, d_in, max_distance=dist_tx_pref-1e-6, exclude_building_id=b.id)[0]:
                        print(f"  Building {b_idx}, Window {w_idx}: Path TX→window blocked by another building")
                        continue
                        
                    vec_pref_rx = rx_pos - p_ref
                    dist_pref_rx = np.linalg.norm(vec_pref_rx)
                    dir_pref_rx = vec_pref_rx / dist_pref_rx
                    
                    if env.check_building_intersection(p_ref, dir_pref_rx, max_distance=dist_pref_rx-1e-6, exclude_building_id=b.id)[0]:
                        print(f"  Building {b_idx}, Window {w_idx}: Path window→RX blocked by another building")
                        continue
                        
                    print(f"  Building {b_idx}, Window {w_idx}: Should have been valid! Check code for errors")
    
    # Ensure rays are shown, including window reflections
    env.visualise_environment(show_rays=True, show_movement=True, movement_time=total_sim_time)


if __name__ == "__main__":
    main()