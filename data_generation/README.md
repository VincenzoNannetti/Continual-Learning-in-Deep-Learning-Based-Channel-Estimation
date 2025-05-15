# Data Generation Framework

## Overview

This directory contains multiple implementations for generating wireless channel datasets used for training and evaluating machine learning models for channel estimation. Each approach simulates different aspects of wireless propagation to create diverse and realistic datasets.

## Directory Structure

- **config/**: Configuration files for all data generation methods, including 3GPP standard presets
- **ray_tracing/**: Implementation of ray-tracing based OFDM channel dataset generation
- **tdl/**: Implementation of Tapped Delay Line channel model generation
- **utils/**: Utility scripts for data processing, visualisation and analysis

## Available Data Generation Methods

### Ray Tracing Based Generation

The `ray_tracing/` directory contains a simplified ray-tracing implementation that models wireless channels by simulating direct paths and scattered paths with configurable parameters including delays, attenuations, and angles of arrival/departure.

For detailed usage instructions, refer to the [Ray Tracing README](ray_tracing/README.md).

### Tapped Delay Line (TDL) Model

The `tdl/` directory implements the standard 3GPP Tapped Delay Line channel models for generating datasets with specific delay profiles and Doppler characteristics.

For detailed usage instructions, refer to the [TDL README](tdl/README.md).

## Configuration System

All data generation methods share a common configuration system based in the `config/` directory:

- **3GPP Standard Presets**: Base configurations according to 3GPP TR 38.901 standards
- **Custom Configurations**: Parameter adjustments that override preset values

For detailed configuration options, refer to the [Configuration README](config/README.md).

## Utilities

The `utils/` directory contains scripts for:

- **Data Visualisation**: Plotting channel matrices, heatmaps, and statistical distributions
- **Data Analysis**: Computing and visualising dataset statistics
- **Data Processing**: Helper functions for data formatting and transformation

## Getting Started

To generate datasets using any of the implementations:

1. Review the README in the specific implementation directory (e.g., `ray_tracing/README.md`)
2. Choose or create an appropriate configuration file in the `config/` directory
3. Run the generation script with the desired parameters

Example:

```bash
# Generate a dataset using ray-tracing with Urban Macrocell configuration
python -m data_generation.ray_tracing.generate_dataset --config data_generation/config/dataset_a.yaml --samples 1000

# Generate a dataset using TDL-A model
python -m data_generation.tdl.generate_dataset --model TDL-A --delay-spread 100 --samples 1000
```

## Common Dataset Format

All generation methods produce datasets in a standardised format:
- `.mat` files containing complex-valued channel matrices
- Each file includes perfect (noise-free) and noisy channel responses
- Each file includes pilot masks which indicate where the pilot signal is in the channel matrix.

## Reference Data

TO DO
