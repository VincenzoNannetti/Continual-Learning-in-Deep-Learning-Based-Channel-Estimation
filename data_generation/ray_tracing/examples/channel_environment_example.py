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
    env_cfg         = config.get('environment', {})
    clusters_cfg    = config.get('clusters', {})
    antennas_cfg    = config.get('antennas', {})
    ue_movement_cfg = antennas_cfg.get('rx_antenna', {}).get('movement', {})

    # Environment
    env_dimensions          = ensure_vector3d(env_cfg.get('environment_dimensions', [200, 100, 100]))
    scatterer_movement_type = clusters_cfg.get('movement_type', 'random_walk')
    env                     = Environment(dimensions=env_dimensions, movement_type=scatterer_movement_type)

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
    tx_gain = antennas_cfg.get('tx_antenna', {}).get('gain', 15)
    rx_gain = antennas_cfg.get('rx_antenna', {}).get('gain', 0)

    # Determine target BS height based on environment type 
    h_bs_target = antennas_cfg.get('target_height', 25.0)  # Typical for UMa (e.g., 3GPP TR 38.901 UMa)
    # Select building for BS placement based on proximity to target height
    if not rooftops:
        print("Warning: No buildings available for BS placement. Placing BS at a default fallback location.")
        # Fallback BS position: e.g., mid-point of one side of the road at target height.
        tx_antenna_pos = ensure_vector3d([env_dimensions[0] * 0.25, road_y, h_bs_target]) 
    else:
        best_bldg      = min(rooftops, key=lambda b: abs(b["z_roof"] - h_bs_target))
        bx,  by        = best_bldg["x"], best_bldg["y"]
        bd,  bh        = best_bldg["depth"], best_bldg["z_roof"]    
        road_facing    = +1 if by < road_y else -1 
        inset          = bd/2 - 0.3                      
        mast_base      = ensure_vector3d([bx, by + road_facing * inset, bh])
        ANTENNA_ELEV   = 3.0                                                 # 3GPP macro default
        tx_antenna_pos = mast_base + np.array([0.0, 0.0, ANTENNA_ELEV]) 
    tx_antenna         = env.place_tx("BS1", tx_antenna_pos, gain_dbi=tx_gain)
    
    # UE Placement: Randomly within the street canyon
    min_x_street_ue = 10.0 
    max_x_street_ue = x_offset 
    ue_x            = random.uniform(min_x_street_ue, max_x_street_ue)
    # Y-coordinate: within the street canyon (anywhere within road or pavements)
    ue_y             = random.uniform(road_y - setback, road_y + setback)
    ue_z             = antennas_cfg.get('target_height', 1.5)               # Standard UE height (defined in the spec to be this as well)

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

    rx_antenna = env.place_rx("UE1", ensure_vector3d([ue_x, ue_y, ue_z]), gain_dbi=rx_gain, movement_config=ue_movement_cfg, pavement_bounds=pavement_bounds)

    speed_range_cfg = ue_movement_cfg.get('speed', [0.5, 2])
    if isinstance(speed_range_cfg, list) and len(speed_range_cfg) == 2:
        scatterer_speed_for_clusters = speed_range_cfg 
    else: 
        scatterer_speed_for_clusters = [0, speed_range_cfg] 

    cluster_density_cfg    = clusters_cfg.get('density', 0.5)
    num_scatterers_default = int(50 * cluster_density_cfg) 
    
    # Cluster shared parameters from config
    use_agg_scattering_cfg = config.get('use_aggregate_scattering', True)
    environment_type_cfg   = env_cfg.get('environment_type', 'UMa') 
    far_field_margin_cfg   = float(config.get('far_field_margin', 6e-2))
    carrier_freq_ghz_cfg   = config.get('carrier_frequency', 2900)/1000
    
    reflection_amplitude_cfg = clusters_cfg.get('reflection_amplitude', None)
    reflection_params_for_scatterer_cfg = None
    if reflection_amplitude_cfg is not None:
        if isinstance(reflection_amplitude_cfg, list) and len(reflection_amplitude_cfg) == 2:
            reflection_params_for_scatterer_cfg = {'reflection_amplitude': reflection_amplitude_cfg} 
        else: 
            reflection_params_for_scatterer_cfg = {'reflection_amplitude': [reflection_amplitude_cfg, reflection_amplitude_cfg]}

    cluster_movement_params_from_config = clusters_cfg.get("cluster_movement_params", None)

    # Define number of clusters
    num_aerial_clusters = random.randint(clusters_cfg.get('num_aerial_clusters_rng', [2, 4])[0],  clusters_cfg.get('num_aerial_clusters_rng', [2, 4])[1])
    num_ground_clusters = random.randint(clusters_cfg.get('num_ground_clusters_rng', [5, 12])[0], clusters_cfg.get('num_ground_clusters_rng', [5, 12])[1])

    # Add Aerial Clusters
    print(f"Adding {num_aerial_clusters} aerial clusters.")
    for _ in range(num_aerial_clusters):
        # 1) sample an (x,y) at least radius away from every building footprint
        aerial_radius = np.clip(np.random.lognormal(mean=0.3, sigma=0.5), 0.1, 6.0)
        num_sc        = max(1, int(num_scatterers_default * random.uniform(0.7, 1.5)))
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
        ground_radius = np.clip(np.random.lognormal(mean=0.2, sigma=0.4), 0.1, 5.0)
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


def run_simulation(config_path=None, preset_name=None, preset_only=False):
    """Main simulation function to be profiled"""
    base_config   = {}
    config_to_use = {} 

    if config_path and not preset_only:
        base_config = load_config(config_path)
        config_to_use = base_config.copy() 

    if preset_name: 
        presets_path = "C:/Users/Vincenzo_DES/OneDrive - Imperial College London/Year 4/ELEC70017 - Individual Project/Project/data_generation/config/3gpp_presets.yaml"        
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
            
    env, config_to_use = create_default_environment(config_to_use) 
    # Common part of simulation (Channel, OFDM, etc.) 
    snr = round(np.random.uniform(config_to_use.get('snr', [15,15])[0], config_to_use.get('snr', [15,15])[1]),3)
    
    # Create channel using parameters from config
    channel = Channel(
        environment          = env,
        tx_antenna           = env.tx_antennas[0],
        rx_antenna           = env.rx_antennas[0],
        center_frequency     = config_to_use.get('carrier_frequency', 2490) * 1e6, 
        snr                  = snr,
        rms_delay_spread_ns  = config_to_use.get('rms_delay_spread_ns', 100),
        shadow_fading_std_db = config_to_use.get('shadow_fading_std_db', 4.0),
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
    
    # Plot magnitude
    plt.figure(figsize=(10, 6))
    plt.subplot(121)
    title_text = f'Perfect Channel Magnitude ({config_to_use.get("environment_type", "UMa")})'
    if preset_name:
        title_text += f' [{preset_name}]'
    plt.title(title_text)
    plt.pcolormesh(np.abs(perfect_ch), cmap='viridis')
    plt.colorbar(label='Magnitude')
    plt.xlabel('Symbol'); plt.ylabel('Subcarrier')
    
    plt.subplot(122)
    plt.title(f'Noisy Channel Magnitude (SNR={snr}dB)')
    plt.pcolormesh(np.abs(noisy_ch), cmap='viridis')
    plt.colorbar(label='Magnitude')
    plt.xlabel('Symbol'); plt.ylabel('Subcarrier')
    plt.tight_layout(); plt.show()
    
    symbol_duration = 1.0 / subcarrier_spacing
    total_simulation_time = num_symbols * symbol_duration
    print(f"Total simulated physical time: {total_simulation_time:.4f} seconds")

    # Plot phase
    plt.figure(figsize=(10, 6))
    plt.subplot(121)
    title_text = f'Perfect Channel Phase ({config_to_use.get("environment_type", "UMa")})'
    if preset_name:
        title_text += f' [{preset_name}]'
    plt.title(title_text)
    phase_plot = plt.pcolormesh(np.angle(perfect_ch), cmap='twilight')
    plt.colorbar(phase_plot, label='Phase (rad)')
    plt.xlabel('Symbol'); plt.ylabel('Subcarrier')
    
    plt.subplot(122)
    plt.title(f'Noisy Channel Phase (SNR={snr}dB)')
    phase_plot = plt.pcolormesh(np.angle(noisy_ch), cmap='twilight')
    plt.colorbar(phase_plot, label='Phase (rad)')
    plt.xlabel('Symbol'); plt.ylabel('Subcarrier')
    plt.tight_layout(); plt.show()
    
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
    parser.add_argument('--debug-reflections', action='store_true',
                        help='Print detailed debug information about window reflections')
    
    args = parser.parse_args()
    
    # List presets if requested
    if args.list_presets:
        current_script_dir = Path(__file__).resolve().parent
        presets_path       = current_script_dir.parent / "config" / "3gpp_presets.yaml"
        try:
            presets = load_config(presets_path)
            print("Available 3GPP presets:")
            for preset, details in presets.items():
                print(f"  {preset}: {details.get('description', '')}")
        except FileNotFoundError:
            print(f"Error: Presets file not found at {presets_path}. Check the path.")
            return
        except yaml.YAMLError as e:
            print(f"Error parsing presets file: {e}")
            return 
        return
    
    # Determine preset_name for run_simulation based on args
    sim_preset_name = args.preset_only if args.preset_only else args.preset

    # Run the simulation with the specified config and preset
    env, _, _, _, total_sim_time = run_simulation(
        config_path     = args.config if not args.preset_only else None, 
        preset_name     = sim_preset_name,
        preset_only     = bool(args.preset_only), 
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
        
        # Get and analyse valid reflections
        valid_reflections = env.get_valid_window_reflections(tx_pos, rx_pos)
        print(f"Valid window reflections found: {len(valid_reflections)}")
        
        # If no valid reflections then see why 
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
                        
                    # 2 Try mirror method and check if reflection point is on window
                    n        = window_normal / np.linalg.norm(window_normal)
                    p0       = window_center
                    tx_m     = tx_pos - 2 * np.dot(tx_pos - p0, n) * n
                    dir_line = rx_pos - tx_m
                    denom    = np.dot(dir_line, n)
                    
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
                    vec_tx_pref  = p_ref - tx_pos
                    dist_tx_pref = np.linalg.norm(vec_tx_pref)
                    d_in         = vec_tx_pref / dist_tx_pref
                    
                    if env.check_building_intersection(tx_pos, d_in, max_distance=dist_tx_pref-1e-6, exclude_building_id=b.id)[0]:
                        print(f"  Building {b_idx}, Window {w_idx}: Path TX->window blocked by another building")
                        continue
                        
                    vec_pref_rx  = rx_pos - p_ref
                    dist_pref_rx = np.linalg.norm(vec_pref_rx)
                    dir_pref_rx  = vec_pref_rx / dist_pref_rx
                    
                    if env.check_building_intersection(p_ref, dir_pref_rx, max_distance=dist_pref_rx-1e-6, exclude_building_id=b.id)[0]:
                        print(f"  Building {b_idx}, Window {w_idx}: Path window->RX blocked by another building")
                        continue    
    # Ensure rays are shown, including window reflections
    # env.visualise_environment(show_rays=True, show_movement=True, movement_time=total_sim_time)


if __name__ == "__main__":
    main()