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

from data_generation.ray_tracing.classes.environment import Environment
from data_generation.ray_tracing.classes.channel import Channel
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


def run_simulation(config_path=None, preset_name=None, preset_only=False):
    """Main simulation function to be profiled"""
    # Load configuration
    config = {}
    
    if config_path and not preset_only:
        config = load_config(config_path)
    
    # If a preset is specified, load and apply preset config
    if preset_name:
        presets_path = Path("data_generation/config/3gpp_presets.yaml")
        presets = load_config(presets_path)
        
        if preset_name not in presets:
            print(f"Warning: Preset '{preset_name}' not found. Using base config only.")
        else:
            preset_config = presets[preset_name]
            print(f"Using preset '{preset_name}': {preset_config.get('description', '')}")
            
            if preset_only:
                # Use only the preset config
                config = preset_config
            else:
                # Merge preset with base config
                config = merge_configs(config, preset_config)
    
    # Create environment with dimensions from config - ensure they're large enough
    env_dimensions = config.get('environment_dimensions', [100, 100, 100])
    
    # Make sure environment is large enough for antenna distance
    antenna_distance = config.get('antenna_distance', 50)
    min_env_size = antenna_distance * 1.2  # Add some margin
    
    env_dimensions = [
        max(env_dimensions[0], min_env_size), 
        max(env_dimensions[1], min_env_size), 
        env_dimensions[2]
    ]
    
    # Extract movement model parameter from config
    default_movement_type = config.get('scatterer_movement_type', 'random_walk') # Default to random_walk if not specified
    print(f"Using movement type: {default_movement_type}")
    
    env = Environment(ensure_vector3d(env_dimensions), movement_type=default_movement_type)
    
    # Extract antenna parameters
    tx_height = config.get('tx_height', 25)
    rx_height = config.get('rx_height', 1.5)
    tx_gain   = config.get('tx_gain', 15)
    rx_gain   = config.get('rx_gain', 0)
    
    # Place transmitter at 1/4 of environment and receiver within bounds
    tx_pos = [env_dimensions[0]/4, env_dimensions[1]/2, tx_height]
    rx_pos = [tx_pos[0] + antenna_distance, tx_pos[1], rx_height]
    
    # Verify positions are within bounds
    if rx_pos[0] >= env_dimensions[0]:
        print(f"Warning: Adjusted RX position to fit within environment bounds")
        tx_pos[0] = env_dimensions[0] * 0.1
        rx_pos[0] = env_dimensions[0] * 0.8
    
    tx_antenna = env.place_tx("BS", tx_pos, gain_dbi=tx_gain)
    rx_antenna = env.place_rx("UE", rx_pos, gain_dbi=rx_gain)
    
    # Extract cluster parameters
    num_clusters_range = config.get('num_clusters', [1, 3])
    if isinstance(num_clusters_range, list) and len(num_clusters_range) == 2:
        num_clusters = random.randint(num_clusters_range[0], num_clusters_range[1])
    else:
        num_clusters = num_clusters_range
    
    speed_range = config.get('speed', [0.5, 2])
    if isinstance(speed_range, list) and len(speed_range) == 2:
        speed = random.uniform(speed_range[0], speed_range[1])
    else:
        speed = speed_range
    
    cluster_density      = config.get('cluster_density', 0.5)
    cluster_radius       = float(config.get('cluster_radius', 15))
    cluster_height_range = config.get('cluster_height_range', cluster_radius)
    use_agg_scattering   = config.get('use_aggregate_scattering', False)
    environment_type     = config.get('environment_type', 'UMa')
    far_field_margin     = float(config.get('far_field_margin', 5.0))
    
    # Handle reflection amplitude
    reflection_amplitude = config.get('reflection_amplitude', None)
    if reflection_amplitude is not None:
        if isinstance(reflection_amplitude, list) and len(reflection_amplitude) == 2:
            # If it's a range, choose a random value in the range
            reflection_amplitude = random.uniform(reflection_amplitude[0], reflection_amplitude[1])
        reflection_params = {'reflection_amplitude': reflection_amplitude}
    else:
        reflection_params = None

    # Time the cluster creation process
    start_time = time.time()
    
    # Add clusters randomly within the environment
    for _ in range(num_clusters):
        # Simple uniform placement within the central 80% of each dimension
        cluster_x = random.uniform(env_dimensions[0] * 0.1, env_dimensions[0] * 0.9)
        cluster_y = random.uniform(env_dimensions[1] * 0.1, env_dimensions[1] * 0.9)
        cluster_z = random.uniform(env_dimensions[2] * 0.1, env_dimensions[2] * 0.9)
        num_scatterers = int(50 * cluster_density)

        # Get cluster-specific movement parameters from config if they exist
        cluster_movement_params_from_config = config.get("cluster_movement_params", None)

        env.add_cluster(
            [cluster_x, cluster_y, cluster_z],
            num_scatterers=num_scatterers,
            radius=cluster_radius,
            height_range=cluster_height_range,
            scatterer_speed_range=[0, speed],
            use_aggregate_scattering_for_scatterer=use_agg_scattering,
            environment_type_for_scatterer=environment_type,
            reflection_params_for_scatterer=reflection_params,
            carrier_freq_ghz_for_scatterer=config.get('carrier_frequency', 2900)/1000,
            min_far_field=far_field_margin,
            cluster_movement_config=cluster_movement_params_from_config # Pass the dict here
        )
    
    cluster_time = time.time() - start_time
    print(f"Time to create {num_clusters} clusters: {cluster_time:.2f} seconds")
    print(f"Total scatterers: {len(env.scatterers)} (avg {len(env.scatterers)/num_clusters:.1f} per cluster)")

    snr = round(np.random.uniform(config.get('snr', 15)[0], config.get('snr', 15)[1]),3)
    
    # Create channel using parameters from config
    channel = Channel(
        environment=env,
        tx_antenna=tx_antenna,
        rx_antenna=rx_antenna,
        center_frequency = config.get('carrier_frequency', 2490) * 1e6, # frequency in Hz
        snr=snr,
        pathloss_model=config.get('pathloss_model', 'friis'),
        rms_delay_spread_ns=config.get('rms_delay_spread_ns', 100),
        shadow_fading_std_db=config.get('shadow_fading_std_db', 4.0),
        o2i_loss=config.get('o2i_loss', 0),
    )
    
    # Time the channel simulation
    start_time = time.time()
    channel.prepare_channel()
    
    # OFDM setup
    subcarrier_spacing    = config.get('subcarrier_spacing', 60000)
    num_blocks            = config.get('blocks', 4)
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
    title_text = f'Perfect Channel ({environment_type})'
    if preset_name:
        title_text += f' [{preset_name}]'
    plt.title(title_text)
    plt.pcolormesh(np.abs(perfect_ch), cmap='viridis')
    plt.xlabel('Symbol'); plt.ylabel('Subcarrier')
    
    plt.subplot(122)
    plt.title(f'Received Signal |y(t)| (SNR={snr}dB)')
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
    
    args = parser.parse_args()
    
    # List presets if requested
    if args.list_presets:
        presets_path = Path("data_generation/config/3gpp_presets.yaml")
        presets = load_config(presets_path)
        print("Available 3GPP presets:")
        for preset, details in presets.items():
            print(f"  {preset}: {details.get('description', '')}")
        return
    
    # Run the simulation with the specified config and preset
    if args.preset_only:
        env, channel, perfect_ch, noisy_ch, total_sim_time = run_simulation(preset_name=args.preset_only, preset_only=True)
    else:
        env, channel, perfect_ch, noisy_ch, total_sim_time = run_simulation(args.config, args.preset)
    
    env.visualise_environment(show_movement=True, movement_time=total_sim_time)


if __name__ == "__main__":
    main()