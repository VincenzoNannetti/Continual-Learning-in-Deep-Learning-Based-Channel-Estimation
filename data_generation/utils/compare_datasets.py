"""
Enhanced Dataset Comparison Script

This script provides a simplified interface for comparing two datasets using
comprehensive statistical metrics and visualisations.

Usage:
    python compare_datasets.py --dataset-a path/to/dataset_a.mat --dataset-b path/to/dataset_b.mat --output-dir comparison_results

Features:
- KL Divergence, Jensen-Shannon Divergence, Wasserstein Distance
- Kolmogorov-Smirnov statistical tests
- Comprehensive visualisations (KDE, box plots, Q-Q plots, histograms)
- Detailed statistical reports (CSV + console output)
- Multiple comparison aspects (magnitude, phase, real/imaginary parts, noise characteristics)
"""

import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add the utils directory to the path so we can import our functions
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from plot_data_statistics import (
        load_data, 
        plot_comprehensive_comparison, 
        create_comparison_report,
        setup_ieee_style
    )
except ImportError:
    # If running from different directory
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from plot_data_statistics import (
        load_data, 
        plot_comprehensive_comparison, 
        create_comparison_report,
        setup_ieee_style
    )

def main():
    parser = argparse.ArgumentParser(
        description="Compare two datasets with comprehensive statistical analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare two datasets with all metrics
  python compare_datasets.py --dataset-a data_a.mat --dataset-b data_b.mat --output-dir results
  
  # Compare only specific aspects
  python compare_datasets.py --dataset-a data_a.mat --dataset-b data_b.mat --output-dir results --aspects magnitude phase
  
  # Use custom labels for datasets
  python compare_datasets.py --dataset-a data_a.mat --dataset-b data_b.mat --output-dir results --label-a "High SNR" --label-b "Low SNR"
        """
    )
    
    parser.add_argument('--dataset-a', required=True, help='Path to first dataset (.mat file)')
    parser.add_argument('--dataset-b', required=True, help='Path to second dataset (.mat file)')
    parser.add_argument('--output-dir', required=True, help='Directory to save comparison results')
    parser.add_argument('--label-a', default='Dataset A', help='Label for first dataset in plots')
    parser.add_argument('--label-b', default='Dataset B', help='Label for second dataset in plots')
    parser.add_argument('--aspects', nargs='+', 
                       choices=['magnitude', 'phase', 'real', 'imag', 'noise', 'all'],
                       default=['all'],
                       help='Which aspects to compare (default: all)')
    parser.add_argument('--max-samples', type=int, default=50000,
                       help='Maximum samples to use for analysis (for computational efficiency)')
    
    args = parser.parse_args()
    
    # Setup plotting style
    setup_ieee_style()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Results will be saved to: {args.output_dir}")
    
    # Load datasets
    print(f"\nLoading Dataset A from: {args.dataset_a}")
    try:
        perfect_a, received_a = load_data(args.dataset_a)
    except Exception as e:
        print(f"Error loading Dataset A: {e}")
        return 1
        
    print(f"Loading Dataset B from: {args.dataset_b}")
    try:
        perfect_b, received_b = load_data(args.dataset_b)
    except Exception as e:
        print(f"Error loading Dataset B: {e}")
        return 1
    
    # Calculate noise
    noise_a = received_a - perfect_a
    noise_b = received_b - perfect_b
    
    print(f"\nDataset A shape: {perfect_a.shape}")
    print(f"Dataset B shape: {perfect_b.shape}")
    
    # Determine which aspects to compare
    if 'all' in args.aspects:
        aspects_to_compare = ['magnitude', 'phase', 'real', 'imag', 'noise']
    else:
        aspects_to_compare = args.aspects
    
    print(f"Comparing aspects: {', '.join(aspects_to_compare)}")
    
    # Dictionary to store all comparison metrics
    all_metrics = {}
    
    # Perform comparisons based on selected aspects
    if 'magnitude' in aspects_to_compare:
        print("\n--- Comparing Channel Magnitude ---")
        
        # Perfect channel magnitude
        print("  Perfect channel magnitude...")
        metrics = plot_comprehensive_comparison(
            np.abs(perfect_a), np.abs(perfect_b),
            f'{args.label_a} (Perfect)', f'{args.label_b} (Perfect)',
            'Perfect Channel Magnitude Comparison',
            args.output_dir, 'perfect_magnitude',
            max_points=args.max_samples
        )
        all_metrics['Perfect Channel Magnitude'] = metrics
        
        # Noisy channel magnitude
        print("  Noisy channel magnitude...")
        metrics = plot_comprehensive_comparison(
            np.abs(received_a), np.abs(received_b),
            f'{args.label_a} (Noisy)', f'{args.label_b} (Noisy)',
            'Noisy Channel Magnitude Comparison',
            args.output_dir, 'noisy_magnitude',
            max_points=args.max_samples
        )
        all_metrics['Noisy Channel Magnitude'] = metrics
    
    if 'phase' in aspects_to_compare:
        print("\n--- Comparing Channel Phase ---")
        
        # Perfect channel phase
        print("  Perfect channel phase...")
        metrics = plot_comprehensive_comparison(
            np.angle(perfect_a), np.angle(perfect_b),
            f'{args.label_a} (Perfect)', f'{args.label_b} (Perfect)',
            'Perfect Channel Phase Comparison',
            args.output_dir, 'perfect_phase',
            max_points=args.max_samples
        )
        all_metrics['Perfect Channel Phase'] = metrics
        
        # Noisy channel phase
        print("  Noisy channel phase...")
        metrics = plot_comprehensive_comparison(
            np.angle(received_a), np.angle(received_b),
            f'{args.label_a} (Noisy)', f'{args.label_b} (Noisy)',
            'Noisy Channel Phase Comparison',
            args.output_dir, 'noisy_phase',
            max_points=args.max_samples
        )
        all_metrics['Noisy Channel Phase'] = metrics
    
    if 'real' in aspects_to_compare:
        print("\n--- Comparing Real Parts ---")
        
        print("  Perfect channel real part...")
        metrics = plot_comprehensive_comparison(
            perfect_a.real, perfect_b.real,
            f'{args.label_a} (Perfect)', f'{args.label_b} (Perfect)',
            'Perfect Channel Real Part Comparison',
            args.output_dir, 'perfect_real',
            max_points=args.max_samples
        )
        all_metrics['Perfect Channel Real Part'] = metrics
        
        print("  Noisy channel real part...")
        metrics = plot_comprehensive_comparison(
            received_a.real, received_b.real,
            f'{args.label_a} (Noisy)', f'{args.label_b} (Noisy)',
            'Noisy Channel Real Part Comparison',
            args.output_dir, 'noisy_real',
            max_points=args.max_samples
        )
        all_metrics['Noisy Channel Real Part'] = metrics
    
    if 'imag' in aspects_to_compare:
        print("\n--- Comparing Imaginary Parts ---")
        
        print("  Perfect channel imaginary part...")
        metrics = plot_comprehensive_comparison(
            perfect_a.imag, perfect_b.imag,
            f'{args.label_a} (Perfect)', f'{args.label_b} (Perfect)',
            'Perfect Channel Imaginary Part Comparison',
            args.output_dir, 'perfect_imag',
            max_points=args.max_samples
        )
        all_metrics['Perfect Channel Imaginary Part'] = metrics
        
        print("  Noisy channel imaginary part...")
        metrics = plot_comprehensive_comparison(
            received_a.imag, received_b.imag,
            f'{args.label_a} (Noisy)', f'{args.label_b} (Noisy)',
            'Noisy Channel Imaginary Part Comparison',
            args.output_dir, 'noisy_imag',
            max_points=args.max_samples
        )
        all_metrics['Noisy Channel Imaginary Part'] = metrics
    
    if 'noise' in aspects_to_compare:
        print("\n--- Comparing Noise Characteristics ---")
        
        print("  Noise real part...")
        metrics = plot_comprehensive_comparison(
            noise_a.real, noise_b.real,
            f'{args.label_a} (Noise)', f'{args.label_b} (Noise)',
            'Noise Real Part Comparison',
            args.output_dir, 'noise_real',
            max_points=args.max_samples
        )
        all_metrics['Noise Real Part'] = metrics
        
        print("  Noise imaginary part...")
        metrics = plot_comprehensive_comparison(
            noise_a.imag, noise_b.imag,
            f'{args.label_a} (Noise)', f'{args.label_b} (Noise)',
            'Noise Imaginary Part Comparison',
            args.output_dir, 'noise_imag',
            max_points=args.max_samples
        )
        all_metrics['Noise Imaginary Part'] = metrics
        
        print("  Noise magnitude...")
        metrics = plot_comprehensive_comparison(
            np.abs(noise_a), np.abs(noise_b),
            f'{args.label_a} (Noise)', f'{args.label_b} (Noise)',
            'Noise Magnitude Comparison',
            args.output_dir, 'noise_magnitude',
            max_points=args.max_samples
        )
        all_metrics['Noise Magnitude'] = metrics
    
    # Generate comprehensive report
    print("\n--- Generating Comprehensive Report ---")
    comparison_df = create_comparison_report(all_metrics, args.output_dir)
    
    print(f"\n Analysis complete! Results saved to: {args.output_dir}")
    print("\nGenerated files:")
    print("   Individual comparison plots: *_comprehensive.svg")
    print("   Summary plot: comparison_summary.svg") 
    print("   Detailed metrics: dataset_comparison_metrics.csv")
    print("   Console output above shows key findings")
    
    return 0

if __name__ == "__main__":
    exit(main()) 