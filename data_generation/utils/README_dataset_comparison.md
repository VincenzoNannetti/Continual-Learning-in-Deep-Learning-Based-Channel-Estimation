# Enhanced Dataset Comparison Tools

This directory contains enhanced tools for comparing two datasets with comprehensive statistical analysis and visualisation capabilities. The tools are designed to help you prove differences between similar datasets using multiple statistical metrics.

## 🚀 Quick Start

### Option 1: Using the Standalone Script (Recommended)

```bash
# Compare two datasets with all metrics
python compare_datasets.py --dataset-a path/to/dataset_a.mat --dataset-b path/to/dataset_b.mat --output-dir comparison_results

# Compare only specific aspects
python compare_datasets.py --dataset-a data_a.mat --dataset-b data_b.mat --output-dir results --aspects magnitude noise

# Use custom labels
python compare_datasets.py --dataset-a data_a.mat --dataset-b data_b.mat --output-dir results --label-a "High SNR" --label-b "Low SNR"
```

### Option 2: Using the Enhanced Original Script

```bash
# Use the enhanced plot_data_statistics.py with comparison mode
python plot_data_statistics.py --compare
```

## 📊 What You Get

### 1. Comprehensive Statistical Metrics

For each comparison aspect, you'll get:

- **KL Divergence** (A→B and B→A): Measures how different the probability distributions are
- **Jensen-Shannon Divergence**: Symmetric version of KL divergence (0 = identical, 1 = completely different)
- **Wasserstein Distance**: Earth Mover's Distance between distributions
- **Kolmogorov-Smirnov Test**: Statistical test for distribution differences (p < 0.05 = significant difference)
- **Basic Statistics**: Mean, standard deviation, percentile differences

### 2. Rich Visualisations

Each comparison generates a comprehensive plot with:
- **Overlayed KDE plots**: Shows probability density distributions
- **Box plots**: Compares quartiles and outliers
- **Q-Q plots**: Shows how quantiles compare (points on diagonal = similar distributions)
- **Histograms**: Direct comparison of value distributions
- **Statistical summary**: Key metrics displayed on the plot

### 3. Detailed Reports

- **CSV file**: All metrics for further analysis
- **Summary plot**: Key metrics across all comparisons
- **Console output**: Interpretation guide and key findings

## 🔍 Comparison Aspects

The tool can compare multiple aspects of your datasets:

### Channel Characteristics
- **Magnitude**: `|H(f,t)|` - Channel gain
- **Phase**: `∠H(f,t)` - Channel phase response
- **Real Part**: `Re{H(f,t)}` - Real component
- **Imaginary Part**: `Im{H(f,t)}` - Imaginary component

### Noise Characteristics
- **Noise Real/Imaginary**: Components of `received - perfect`
- **Noise Magnitude**: `|noise|` - Noise power

## 📈 Interpreting Results

### Key Metrics to Look For

1. **KL Divergence**
   - `0`: Identical distributions
   - `> 0.1`: Noticeable differences
   - `> 1.0`: Significant differences
   - `> 5.0`: Very different distributions

2. **Jensen-Shannon Divergence**
   - `0`: Identical distributions
   - `0.1-0.3`: Moderate differences
   - `> 0.5`: Large differences

3. **Wasserstein Distance**
   - Depends on data scale, but larger = more different
   - Good for comparing magnitude of differences

4. **Kolmogorov-Smirnov p-value**
   - `< 0.05`: Statistically significant difference
   - `< 0.001`: Highly significant difference

### Visual Interpretation

- **KDE plots**: Non-overlapping curves = different distributions
- **Box plots**: Different medians/quartiles = different central tendencies
- **Q-Q plots**: Points off diagonal = different distributions
- **Histograms**: Different shapes = different distributions

## 🛠 Advanced Usage

### Customising the Analysis

```python
from plot_data_statistics import plot_comprehensive_comparison, calculate_comprehensive_metrics

# Calculate metrics only (no plots)
metrics = calculate_comprehensive_metrics(data_a, data_b, metric_name="My Comparison")

# Custom comparison with specific parameters
metrics = plot_comprehensive_comparison(
    data_a, data_b,
    "Dataset A", "Dataset B", 
    "My Custom Comparison",
    save_path="./results",
    filename_base="custom_comparison",
    max_points=100000  # Use more samples for higher precision
)
```

### Batch Comparisons

```python
# Compare multiple dataset pairs
dataset_pairs = [
    ("dataset_1a.mat", "dataset_1b.mat", "Condition 1"),
    ("dataset_2a.mat", "dataset_2b.mat", "Condition 2"),
    # ... more pairs
]

all_results = {}
for file_a, file_b, condition in dataset_pairs:
    # Load and compare datasets
    # Store results in all_results[condition]
    pass

# Generate combined report
create_comparison_report(all_results, "batch_comparison_results")
```

## 📋 Output Files

After running the comparison, you'll find:

```
comparison_results/
├── perfect_magnitude_comprehensive.svg    # Perfect channel magnitude comparison
├── noisy_magnitude_comprehensive.svg      # Noisy channel magnitude comparison
├── perfect_phase_comprehensive.svg        # Perfect channel phase comparison
├── noise_real_comprehensive.svg           # Noise real part comparison
├── noise_magnitude_comprehensive.svg      # Noise magnitude comparison
├── comparison_summary.svg                 # Summary of all metrics
├── dataset_comparison_metrics.csv         # Detailed metrics (Excel-compatible)
└── magnitude_comparison_perfect_legacy.svg # Legacy overlayed plots
```

## 🔧 Requirements

```bash
pip install numpy matplotlib seaborn scipy pandas tqdm
```

## 💡 Tips for Proving Dataset Differences

1. **Use Multiple Metrics**: Don't rely on just one metric. KL divergence might be high while Wasserstein distance is low, indicating different types of differences.

2. **Check Statistical Significance**: The KS test p-value tells you if differences are statistically significant, not just due to random variation.

3. **Look at Different Aspects**: Datasets might be similar in magnitude but different in phase, or similar in perfect channels but different in noise characteristics.

4. **Visual Inspection**: Sometimes the plots reveal patterns that metrics miss. Look for:
   - Bimodal vs unimodal distributions
   - Different tail behaviours
   - Systematic shifts vs spread differences

5. **Consider Data Scale**: Normalise your data if comparing datasets with very different scales.

## 🐛 Troubleshooting

### Common Issues

1. **"Data appears constant" warning**: One dataset has no variation. Check your data loading.

2. **NaN values in metrics**: Usually indicates insufficient data or numerical issues. Try increasing `max_samples`.

3. **Memory issues**: Reduce `max_samples` parameter for large datasets.

4. **Import errors**: Ensure you're running from the correct directory and have all dependencies installed.

### Performance Tips

- Use `max_samples=10000` for quick exploratory analysis
- Use `max_samples=100000` for publication-quality results
- For very large datasets, consider pre-sampling your data

## 📚 References

- **KL Divergence**: Kullback-Leibler divergence measures information loss
- **Jensen-Shannon Divergence**: Symmetric version of KL divergence
- **Wasserstein Distance**: Optimal transport distance between distributions
- **Kolmogorov-Smirnov Test**: Non-parametric test for distribution equality

## 🤝 Contributing

Feel free to extend the comparison tools with additional metrics or visualisation types. The modular design makes it easy to add new comparison functions. 