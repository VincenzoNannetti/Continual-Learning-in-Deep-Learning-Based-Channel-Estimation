"""
Filename: ./data_generation/ray_tracing/generate_dataset.py
Author: Vincenzo Nannetti
Date: 09/05/2025 (Refactored Date)
Description: Script to generate a dataset of channel matrices of varying channel parameters.

Running Instructions:
    python -m data_generation.ray_tracing.generate_dataset --config CONFIG_PATH --samples NUM_SAMPLES [--output OUTPUT_PATH] [--preset PRESET_NAME] [--pathloss MODEL]
    
    Parameters:
        --config:   Path to YAML configuration file (default: ../config/dataset_a.yaml)
        --samples:  Number of samples to generate (default: 100000)
        --output:   Optional directory to override the output path in config file
        --preset:   3GPP TR 38.901 preset to apply (UMa, UMi, RMa, InH, InF-SL, etc.)
        --pathloss: Pathloss model to use: 'friis' (original model) or '3gpp' (TR 38.901 models)
    
    Configuration Parameters:
        # Antenna Parameters
        antenna_distance: Distance between TX and RX antennas in meters
        tx_gain:          Transmitter antenna gain in dBi
        rx_gain:          Receiver antenna gain in dBi
        
        # Channel Parameters
        snr:                    Signal-to-Noise Ratio in dB
        num_clusters:           Number of scattering cluster locations
        speed:                  Movement speed of scatterers in m/s
        use_aggregate_scattering: Boolean to enable/disable H_{ij} term
        pathloss_model:         "friis" (original) or "3gpp" (TR 38.901)
        cluster_density:        Density of scatterers in clusters (0.0-1.0)
        cluster_radius:         Radius where scatterers are placed relative to cluster center
        reflection_amplitude:   Range of reflection coefficients [min, max]
        
        # Environment Parameters
        environment_type: Type of environment from 3GPP TR 38.901:
            "UMa":   Urban Macrocell (BS above rooftops)
            "UMi":   Urban Microcell (street canyons)
            "RMa":   Rural Macrocell (sparse buildings)
            "InH":   Indoor Hotspot (offices, malls)
            "InF-SL": Indoor Factory (sparse, low BS)
            "InF-DL": Indoor Factory (dense, low BS)
            "InF-SH": Indoor Factory (sparse, high BS)
            "InF-DH": Indoor Factory (dense, high BS)
            
        # System Parameters
        subcarrier_spacing: Spacing between OFDM subcarriers (Hz)
        blocks:            Number of 14-symbol blocks to generate per sample
        carrier_frequency: Center frequency in MHz
        
        # Output Settings
        output_dir:        Output directory for generated dataset

Usage Examples:
    # Basic usage with default parameters
    python -m data_generation.ray_tracing.generate_dataset --config ../config/dataset_a.yaml --samples 1000
    
    # Using 3GPP presets
    python -m data_generation.ray_tracing.generate_dataset --preset UMa --samples 1000   # Urban Macrocell
    python -m data_generation.ray_tracing.generate_dataset --preset RMa --samples 1000   # Rural Macrocell
    python -m data_generation.ray_tracing.generate_dataset --preset InF-DH --samples 1000  # Indoor Factory
    
    # Changing pathloss model
    python -m data_generation.ray_tracing.generate_dataset --preset UMi --pathloss 3gpp --samples 1000
    
    # Custom output directory
    python -m data_generation.ray_tracing.generate_dataset --preset InH --output "./data/indoor_dataset" --samples 1000
    
    # Complex example with multiple parameters
    python -m data_generation.ray_tracing.generate_dataset --preset UMi --pathloss 3gpp --samples 5000 --output "./data/urban_microcell_3gpp"

Output:
    The script generates a .mat file containing:
    - perfect_matrices: Perfect channel matrices without noise
    - received_matrices: Channel matrices with noise
    - pilot_mask: Boolean masks indicating pilot positions
"""
import os
import time
import scipy.io
from tqdm import tqdm
import numpy as np
import yaml
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

from .classes.channel     import Channel
from .classes.modulator   import Modulator
from .classes.multiplexer import Multiplexer

def load_config(config_path):
    """Load configuration from a YAML file."""
    try:
        # First try to load the config using the path as provided
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
            return config
    except FileNotFoundError:
        # If that fails, try to resolve the path relative to the script location
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(script_dir))
        
        # Try different relative paths
        possible_paths = [
            os.path.join(script_dir, config_path),  # Relative to script
            os.path.join(project_root, config_path.lstrip('/')),  # Relative to project root
            os.path.join(script_dir, '..', 'config', os.path.basename(config_path)),  # config dir relative to script
            os.path.join(os.path.dirname(script_dir), 'config', os.path.basename(config_path))  # config dir at same level as ray_tracing
        ]
        
        for path in possible_paths:
            try:
                with open(path, 'r') as file:
                    config = yaml.safe_load(file)
                    print(f"Loaded config from: {path}")
                    return config
            except FileNotFoundError:
                continue
        
        # If we get here, none of the paths worked
        raise FileNotFoundError(f"Could not find configuration file: {config_path}. Tried paths: {possible_paths}")

# --- Helper Functions ---
def select_random_value(param_value, rng):
    """Select a random value from the parameter configuration."""
    if isinstance(param_value, list):
        # Use uniform sampling ONLY if it's a 2-element list of ACTUAL int/float (NOT bool)
        if len(param_value) == 2 and type(param_value[0]) in (int, float) and type(param_value[1]) in (int, float):
            # Ensure low <= high
            if param_value[1] < param_value[0]:
                print(f"Warning: Reversed range: {param_value}.")
            return round(rng.uniform(param_value[0], param_value[1]), 2)
        else:
            return rng.choice(param_value)
    return param_value

# --- Pilot Pattern Generation Class ---
class PilotPatternGenerator:
    """Handles the generation of pilot patterns based on channel conditions."""
    
    def __init__(self):
        # Define thresholds for pattern selection
        self.LOW_SPEED_THRESH     = 5
        self.HIGH_SPEED_THRESH    = 20
        self.LOW_CLUSTERS_THRESH  = 5
        self.HIGH_CLUSTERS_THRESH = 15

    def select_pattern_type(self, speed, num_clusters, snr):
        """Selects appropriate pilot pattern based on channel conditions."""
        if speed < self.LOW_SPEED_THRESH and num_clusters < self.LOW_CLUSTERS_THRESH:
            # Low speed and low multipath -> Use the sparsest pattern
            pattern_type = 'pattern_low'
        elif speed >= self.HIGH_SPEED_THRESH or num_clusters >= self.HIGH_CLUSTERS_THRESH:
            # High speed OR high multipath -> Use the densest pattern
            pattern_type = 'pattern_high'
        else:
            # Moderate conditions -> Use the medium pattern
            pattern_type = 'pattern_medium'
            
        # If SNR is poor, upgrade the pilot pattern for better estimation
        if snr < 3:
            if pattern_type == 'pattern_low':
                pattern_type = 'pattern_medium'
            elif pattern_type == 'pattern_medium':
                pattern_type = 'pattern_high'
                
        return pattern_type
    
    def generate_mask(self, pattern_type, num_subcarriers, total_ofdm_symbols):
        """
        Returns a boolean mask where True indicates pilot positions.
        """
        pilot_mask = np.zeros((num_subcarriers, total_ofdm_symbols), dtype=bool)
        
        if pattern_type == 'pattern_low':
            # Low density pilot pattern (symbols 13 of slots 0 and 2)
            base_rows = np.arange(0, num_subcarriers, 12)
            for block_start_slot in range(0, total_ofdm_symbols // 14, 4):
                col_idx_slot0 = 13 + (block_start_slot * 14)
                col_idx_slot2 = 13 + ((block_start_slot + 2) * 14)
                
                if col_idx_slot0 < total_ofdm_symbols:
                    pilot_mask[base_rows, col_idx_slot0] = True
                if col_idx_slot2 < total_ofdm_symbols:
                    pilot_mask[base_rows, col_idx_slot2] = True
                    
        elif pattern_type == 'pattern_medium':
            # Medium density (symbol 13 of every slot)
            base_rows = np.arange(0, num_subcarriers, 12)
            for slot in range(0, total_ofdm_symbols // 14):
                col_idx = 13 + (slot * 14)
                if col_idx < total_ofdm_symbols:
                    pilot_mask[base_rows, col_idx] = True
                    
        elif pattern_type == 'pattern_high':
            # High density (symbols 6 and 13 of every slot)
            base_rows = np.arange(0, num_subcarriers, 12)
            for slot in range(0, total_ofdm_symbols // 14):
                col_idx_1 = 6 + (slot * 14)
                col_idx_2 = 13 + (slot * 14)
                if col_idx_1 < total_ofdm_symbols:
                    pilot_mask[base_rows, col_idx_1] = True
                if col_idx_2 < total_ofdm_symbols:
                    pilot_mask[base_rows, col_idx_2] = True
        
        return pilot_mask

def load_preset(preset_name):
    """Load a 3GPP TR 38.901 preset configuration."""
    preset_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                              "config", "3gpp_presets.yaml")
    
    try:
        with open(preset_path, 'r') as file:
            presets = yaml.safe_load(file)
    except FileNotFoundError:
        print(f"Error: Presets file not found: {preset_path}")
        return None
    except yaml.YAMLError as e:
        print(f"Error parsing presets file: {e}")
        return None
    
    if preset_name not in presets:
        print(f"Warning: Preset '{preset_name}' not found. Available presets: {list(presets.keys())}")
        return None
    
    return presets[preset_name]

# --- Apply 3GPP Preset to Config ---
def apply_preset_to_config(config, preset, preset_name):
    """Apply a 3GPP preset to the base configuration."""
    if preset is None:
        return config
    
    # Make a copy of the config to avoid modifying the original
    modified_config = config.copy()
    
    # Apply preset parameters (overriding any existing values)
    for key, value in preset.items():
        # Skip description field
        if key == "description":
            continue
        modified_config[key] = value
    
    # Set the environment type to match the preset
    modified_config['environment_type'] = preset_name
    
    return modified_config

def process_sample(i, config):
    """Process a single sample with the given configuration."""
    time.sleep(0.001)
    
    # Create a random generator with no specific seed
    rng = np.random.RandomState()

    # Extract constants from config
    SYMBOLS_PER_BLOCK      = 14
    NUM_SUBCARRIERS        = 72
    BLOCKS_TO_GENERATE     = config['blocks']
    TOTAL_OFDM_SYMBOLS     = SYMBOLS_PER_BLOCK * BLOCKS_TO_GENERATE
    TOTAL_SYMS_FOR_GRID    = NUM_SUBCARRIERS * TOTAL_OFDM_SYMBOLS

    # --- Generate antenna parameters ---
    antenna_distance   = select_random_value(config['antenna_distance'], rng)
    tx_gain            = select_random_value(config['tx_gain'], rng)
    rx_gain            = select_random_value(config['rx_gain'], rng)
    tx_height          = select_random_value(config.get('tx_height', 25.0), rng)  # Default to 25m
    rx_height          = select_random_value(config.get('rx_height', 1.5), rng)   # Default to 1.5m
    
    # --- Generate propagation parameters ---
    snr                = select_random_value(config['snr'], rng)
    num_clusters       = int(select_random_value(config['num_clusters'], rng))
    speed              = select_random_value(config['speed'], rng)
    subcarrier_spacing = select_random_value(config['subcarrier_spacing'], rng)
    environment_type   = config.get('environment_type', 'UMa')
    
    # --- Get additional configurable parameters ---
    pathloss_model         = config.get('pathloss_model', 'friis')          # Default to 'friis' for backward compatibility
    cluster_density        = config.get('cluster_density', 0.5)             # Default to 0.5 if not specified
    cluster_radius         = config.get('cluster_radius', 20)               # Default to 20m if not specified
    cluster_height_range   = config.get('cluster_height_range', cluster_radius / 2)  # Default to half of radius
    los_probability_indoor = config.get('los_probability_indoor', 0.7)
    rms_delay_spread_ns    = select_random_value(config.get('rms_delay_spread_ns', 100), rng)
    shadow_fading_std_db   = select_random_value(config.get('shadow_fading_std_db', 4.0), rng)
    reflection_amplitude   = config.get('reflection_amplitude', None)
    carrier_frequency      = select_random_value(config.get('carrier_frequency', 2490), rng)
    enable_3d_model        = config.get('enable_3d_model', True)  # Default to 3D model
    
    # --- Handle O2I (outdoor-to-indoor) penetration loss for indoor environments ---
    o2i_loss = 0
    if environment_type.startswith('In') and 'o2i_penetration_loss' in config:
        o2i_loss_config = config['o2i_penetration_loss']
        if isinstance(o2i_loss_config, dict):
            # Complex o2i model with probability
            if rng.random() < o2i_loss_config.get('low_loss_probability', 0.5):
                o2i_loss = select_random_value(o2i_loss_config.get('low_loss', 20), rng)
            else:
                o2i_loss = select_random_value(o2i_loss_config.get('high_loss', 40), rng)
        else:
            # Simple o2i loss value
            o2i_loss = select_random_value(o2i_loss_config, rng)
    
    # --- Determine use_agg_scatter based on config and number of clusters ---
    use_agg_scatter_param = config.get('use_aggregate_scattering', False)
    if isinstance(use_agg_scatter_param, list):
        use_agg_scatter = select_random_value(use_agg_scatter_param, rng)
    else:
        use_agg_scatter = use_agg_scatter_param
    
    # Override use_agg_scatter for rural environments
    if num_clusters < 5:
        use_agg_scatter = False  # No aggregate scattering in rural areas

    # --- Use the PilotPatternGenerator for pilot pattern determination ---
    pilot_generator    = PilotPatternGenerator()
    dynamic_pilot_type = pilot_generator.select_pattern_type(speed, num_clusters, snr)

    # Create channel instance with all parameters
    channel = Channel(
        antenna_distance=antenna_distance, 
        tx_gain=tx_gain, 
        rx_gain=rx_gain, 
        snr=snr, 
        system=None, 
        num_cluster=num_clusters, 
        speed=speed,
        use_aggregate_scattering=use_agg_scatter, 
        environment_type=environment_type,
        pathloss_model=pathloss_model,
        o2i_loss=o2i_loss,
        reflection_amplitude=reflection_amplitude,
        cluster_density=cluster_density,
        cluster_radius=cluster_radius,
        los_probability_indoor=los_probability_indoor,
        rms_delay_spread_ns=rms_delay_spread_ns,
        shadow_fading_std_db=shadow_fading_std_db,
        tx_height=tx_height,
        rx_height=rx_height,
        cluster_height_range=cluster_height_range,
        rng=rng
    )

    # Generate modulator and multiplexer instances
    modulator = Modulator("QAM", 4)
    # Modulate enough symbols to fill the entire NUM_SUBCARRIERS x TOTAL_OFDM_SYMBOLS grid
    modulated_message = modulator.modulate("rand", num_symbols=TOTAL_SYMS_FOR_GRID)

    # Create Multiplexer and multiplex the message
    multiplexer         = Multiplexer(NUM_SUBCARRIERS, 0) 
    multiplexed_message = multiplexer.multiplex(modulated_message)

    # Apply channel - necessary to run for each symbol
    output_signal = channel.apply_channel(
        multiplexed_message,
        carrier_frequency * 1e6,  # Convert MHz to Hz
        subcarrier_spacing,
        NUM_SUBCARRIERS,
        0
    )

    perfect_grid = channel.get_channel_matrix()
    noisy_grid   = channel.get_channel_matrix_noisy()

    # --- Generate the boolean pilot mask for *this* sample ---
    pilot_mask_boolean_sample = pilot_generator.generate_mask(dynamic_pilot_type, NUM_SUBCARRIERS, TOTAL_OFDM_SYMBOLS)

    # Return the sample-specific mask along with the channel grids
    return i, noisy_grid, perfect_grid, pilot_mask_boolean_sample

def main():
    parser = argparse.ArgumentParser(description="Generate channel matrix dataset")
    parser.add_argument("--config", type=str, default="config/dataset_a.yaml", 
                        help="Path to the configuration YAML file")
    parser.add_argument("--output", type=str, default=None, 
                        help="Optional directory to override the output path in config file")
    parser.add_argument("--samples", type=int, default=100000, 
                        help="Number of samples to generate")
    parser.add_argument("--preset", type=str, default=None, 
                        help="3GPP TR 38.901 preset to apply to the configuration (UMa, UMi, RMa, InH, InF-SL, etc.)")
    parser.add_argument("--pathloss", type=str, default=None,
                        help="Pathloss model to use: 'friis' (original model) or '3gpp' (TR 38.901 models)")
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)

    # Apply 3GPP preset if specified
    if args.preset:
        preset = load_preset(args.preset)
        if preset:
            print(f"Applying 3GPP preset: {args.preset} - {preset.get('description', '')}")
            config = apply_preset_to_config(config, preset, args.preset)
        else:
            print(f"Continuing with default configuration from {args.config}")
    
    # Override pathloss model if specified
    if args.pathloss:
        config['pathloss_model'] = args.pathloss
        print(f"Using pathloss model: {args.pathloss}")
    
    # Determine output directory (command line overrides config)
    output_dir = args.output if args.output is not None else config.get('output_dir', '../output')
    
    # Define constants
    SYMBOLS_PER_BLOCK  = 14
    NUM_SUBCARRIERS    = 72
    BLOCKS_TO_GENERATE = config['blocks']
    TOTAL_OFDM_SYMBOLS = SYMBOLS_PER_BLOCK * BLOCKS_TO_GENERATE
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    NUM_SAMPLES       = args.samples
    perfect_matrices  = np.zeros((NUM_SAMPLES, NUM_SUBCARRIERS, TOTAL_OFDM_SYMBOLS), dtype=np.complex64)
    received_matrices = np.zeros((NUM_SAMPLES, NUM_SUBCARRIERS, TOTAL_OFDM_SYMBOLS), dtype=np.complex64)
    pilot_masks       = np.zeros((NUM_SAMPLES, NUM_SUBCARRIERS, TOTAL_OFDM_SYMBOLS), dtype=bool)
    
    num_workers = os.cpu_count()
    print(f"Using {num_workers} worker processes...")
    print(f"Generating {NUM_SAMPLES} samples with configuration from {args.config}")
    if args.preset:
        print(f"Using 3GPP TR 38.901 preset: {args.preset}")
    print(f"Pathloss model: {config.get('pathloss_model', 'friis')}")
    print(f"Output will be saved to {output_dir}")
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_sample, i, config) for i in range(NUM_SAMPLES)]
        
        for future in tqdm(as_completed(futures), total=NUM_SAMPLES, desc="Processing samples"):
            i, received_grid, perfect_grid, pilot_mask_sample = future.result()
            received_matrices[i] = received_grid
            perfect_matrices[i]  = perfect_grid
            pilot_masks[i]       = pilot_mask_sample
    
    # Extract config filename without extension for the output filename
    config_name = os.path.splitext(os.path.basename(args.config))[0]
    filename = f"{config_name}_dataset_{BLOCKS_TO_GENERATE}blocks_{NUM_SAMPLES}samples.mat"
    
    # Save the matrices, pilot masks, and metadata
    scipy.io.savemat(os.path.join(output_dir, filename), {
        "perfect_matrices": perfect_matrices,
        "received_matrices": received_matrices,
        "pilot_mask": pilot_masks.astype(int),  # Save the array of boolean masks as int
    })
    
    print(f"All test data saved successfully to {os.path.join(output_dir, filename)}")

if __name__ == "__main__":
    main()

