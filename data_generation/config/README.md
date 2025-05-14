# Channel Simulation Configuration Guide

This directory contains configuration files for the wireless channel simulation framework. The system uses a two-tier configuration approach:

1. **3GPP Standard Presets**: Base configurations according to 3GPP TR 38.901 standards
2. **Custom Configurations**: Your specific parameter adjustments that override preset values

## Quick Start

```bash
# List available presets
python -m data_generation.ray_tracing.examples.channel_environment_example --list-presets

# Run with just a 3GPP preset (no customisations)
python -m data_generation.ray_tracing.examples.channel_environment_example --preset-only UMa

# Run with custom config and preset (config overrides preset values)
python -m data_generation.ray_tracing.examples.channel_environment_example --config data_generation/config/dataset_a.yaml --preset RMa

# Run with just custom config (no preset)
python -m data_generation.ray_tracing.examples.channel_environment_example --config data_generation/config/dataset_a.yaml
```

## Configuration Files

### 3GPP Presets (`3gpp_presets.yaml`)

Standard environment configurations based on 3GPP TR 38.901 specifications:

- **UMa** (Urban Macrocell): Dense urban areas with BS height (25m) above surrounding buildings
- **UMi** (Urban Microcell): Urban street canyons with BS height (10m) below surrounding buildings
- **RMa** (Rural Macrocell): Rural areas with sparse buildings and large continuous coverage
- **InH** (Indoor Hotspot): Office environments and shopping malls with dense user distribution
- **InF-SL/DL/SH/DH** (Indoor Factory): Various factory environments with different clutter and BS heights

Each preset includes standardised parameters for:
- Antenna configurations
- Scatterer clusters
- Propagation characteristics
- Channel model parameters

### Custom Configs

Custom configurations (like `dataset_a.yaml`) should include only:
1. Which preset to use as base (`environment_type`)
2. Parameters you want to override from the preset
3. Custom parameters not defined in presets

Example:
```yaml
# Specify which preset to use as base
environment_type: "RMa"

# Override specific preset parameters
num_clusters: [1, 5]  # Changed from default [8, 11] in RMa

# Add custom parameters not in presets
scatterer_movement_type: "flocking"
```

## Available Custom Parameters

Here's a comprehensive list of custom parameters that can be used in your configuration files but are not part of the standard 3GPP presets:

### Environment Setup
| Parameter | Description | Default | Example |
|-----------|-------------|---------|---------|
| `environment_dimensions` | 3D dimensions [x,y,z] in meters | [100, 100, 50] | `[200, 200, 100]` |
| `max_cluster_distance_from_antennas` | Maximum distance for clusters from antennas (m) | None | `100` |
| `far_field_margin` | Minimum distance for far-field approximations (m) | 5.0 | `6e-2` |
| `scatterer_movement_type` | Movement model for scatterers | "random_walk" | `"flocking"` |

### Movement-Specific Parameters
These parameters are set within the `cluster_movement_params` dictionary in your configuration:

#### Linear Movement
```yaml
scatterer_movement_type: "linear"
cluster_movement_params:
  direction: [1, 0, 0]  # Direction vector [x,y,z]
```

#### Sinusoidal Movement
```yaml
scatterer_movement_type: "sinusoidal"
cluster_movement_params:
  amplitude_range: [0.5, 2.0]  # Oscillation amplitude range (m)
  period_range: [1.0, 5.0]     # Oscillation period range (s)
  axis: "random"               # "random", "vertical", "horizontal_random", or [x,y,z]
```

#### Brownian Movement
```yaml
scatterer_movement_type: "brownian"
cluster_movement_params:
  sigma_brownian_range: [0.01, 0.05]  # Standard deviation for step size
```

#### Gauss-Markov Movement
```yaml
scatterer_movement_type: "gauss_markov"
cluster_movement_params:
  alpha_gm_range: [0.1, 0.9]           # Memory factor (0-1)
  noise_std_dev_gm_range: [0.05, 0.2]  # Step noise standard deviation
  mean_velocity_gm_config: "random"    # Target velocity: "random" or [x,y,z]
```

#### Flocking Movement
```yaml
scatterer_movement_type: "flocking"
cluster_movement_params:
  flock_movement_direction: [1, 0, 0]  # Overall flock direction [x,y,z]
  # Future parameters (not yet implemented):
  # flock_cohesion_factor: 0.5         # Attraction to center strength
  # flock_alignment_factor: 0.3        # Alignment with neighbors strength
  # flock_separation_factor: 0.2       # Collision avoidance strength
```

### Advanced Channel Parameters
| Parameter | Description | Default | Example |
|-----------|-------------|---------|---------|
| `reflection_params` | Custom reflection coefficient configuration | None | `{"reflection_amplitude": 0.7}` |
| `o2i_penetration_loss` | Outdoor-to-indoor penetration loss (dB) | 0 | `15` |
| `los_probability_indoor` | Probability of LOS in indoor environments | 0.7 | `0.5` |

### Visualisation Parameters
These parameters affect how visualisations appear but don't impact simulation results:
```yaml
visualisation:
  show_bounds: true            # Show environment boundaries
  show_rays: true              # Show ray paths
  show_movement: true          # Show movement paths
  movement_time: 2.0           # Time to simulate for movement paths
  max_rays: 100                # Maximum number of rays to draw
```

### Output Settings
| Parameter | Description | Default | Example |
|-----------|-------------|---------|---------|
| `output_dir` | Directory for saving output data | "./data/raw/ray_tracing/dataset" | `"./data/raw/ray_tracing/my_dataset"` |

## Movement Models

The simulation supports various movement models for scatterers, configurable via `scatterer_movement_type`:

### 1. Random Walk
Simple Brownian motion where scatterers move in random directions each step.

```yaml
scatterer_movement_type: "random_walk"
```

### 2. Linear
Straight-line movement in a fixed direction.

```yaml
scatterer_movement_type: "linear"
cluster_movement_params:
  direction: [1, 0, 0]  # [x, y, z] direction vector
```

### 3. Sinusoidal
Oscillating movement with configurable amplitude and period.

```yaml
scatterer_movement_type: "sinusoidal"
cluster_movement_params:
  amplitude_range: [0.5, 2.0]  # meters
  period_range: [1.0, 5.0]     # seconds
  axis: "random"               # "random", "vertical", "horizontal_random", or [x,y,z]
```

### 4. Brownian
Random movement following a Gaussian distribution.

```yaml
scatterer_movement_type: "brownian"
cluster_movement_params:
  sigma_brownian_range: [0.01, 0.05]  # standard deviation for step size
```

### 5. Gauss-Markov
Time-correlated random movement model.

```yaml
scatterer_movement_type: "gauss_markov"
cluster_movement_params:
  alpha_gm_range: [0.1, 0.9]           # memory factor (0-1)
  noise_std_dev_gm_range: [0.05, 0.2]  # std dev for noise term
  mean_velocity_gm_config: "random"    # target mean velocity
```

### 6. Flocking
Group movement model where scatterers move like a flock of birds.

```yaml
scatterer_movement_type: "flocking"
cluster_movement_params:
  flock_movement_direction: [1, 0, 0]  # overall direction of the flock
  # Optional parameters (to be implemented):
  # flock_cohesion_factor: 0.5         # how strongly scatterers are attracted to the center
  # flock_alignment_factor: 0.3        # how strongly scatterers align with neighbors
  # flock_separation_factor: 0.2       # how strongly scatterers avoid collisions
```

## Advanced Configuration

### Environment Dimensions
```yaml
environment_dimensions: [100, 100, 50]  # [x, y, z] in meters
```

### Far Field Margin
Minimum distance for far-field approximations:
```yaml
far_field_margin: 6e-2  # 6cm
```

### Cluster Parameters
```yaml
num_clusters: [1, 5]       # Range or single value
cluster_density: 1.0       # Density of scatterers in each cluster
cluster_radius: 6          # Radius in meters
```

## Output Settings
Configure the output directory for generated data:
```yaml
output_dir: "./data/raw/ray_tracing/my_dataset"
```
