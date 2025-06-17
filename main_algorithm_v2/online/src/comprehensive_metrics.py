"""
Comprehensive metrics module for online continual learning evaluation.

This module provides:
1. Core performance metrics (NMSE, SSIM, PSNR)
2. Timing instrumentation 
3. Memory usage tracking
4. Per-channel analysis
5. Buffer monitoring utilities
6. LoRA adapter tracking
"""

import time
import torch
import torch.nn as nn
import numpy as np
import psutil
import os
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict, deque
from datetime import datetime
import json

# Import existing SSIM/PSNR implementations
try:
    from shared.utils.ssim import compute_ssim
    from shared.utils.metrics import calculate_psnr, calculate_nmse
except ImportError:
    print("Warning: Could not import shared metrics, using fallback implementations")
    compute_ssim = None
    calculate_psnr = None
    calculate_nmse = None


class TimingProfiler:
    """High-precision timing profiler for online learning operations."""
    
    def __init__(self):
        self.timings = defaultdict(list)
        self.active_timers = {}
        self.total_samples = 0
        
    def start_timer(self, operation: str) -> None:
        """Start timing an operation."""
        self.active_timers[operation] = time.perf_counter()
    
    def end_timer(self, operation: str) -> float:
        """End timing an operation and return elapsed time."""
        if operation not in self.active_timers:
            return 0.0
        
        elapsed = time.perf_counter() - self.active_timers[operation]
        self.timings[operation].append(elapsed)
        del self.active_timers[operation]
        return elapsed
    
    def get_stats(self) -> Dict[str, Dict[str, float]]:
        """Get timing statistics for all operations."""
        stats = {}
        for operation, times in self.timings.items():
            if times:
                stats[operation] = {
                    'mean': np.mean(times),
                    'std': np.std(times),
                    'min': np.min(times),
                    'max': np.max(times),
                    'total': np.sum(times),
                    'count': len(times),
                    'per_sample': np.sum(times) / max(1, self.total_samples)
                }
        return stats
    
    def reset(self) -> None:
        """Reset all timing data."""
        self.timings.clear()
        self.active_timers.clear()
        self.total_samples = 0


class MemoryProfiler:
    """Memory usage profiler for tracking system resources."""
    
    def __init__(self):
        self.memory_snapshots = []
        self.gpu_memory_snapshots = []
        self.baseline_memory = None
        
    def take_snapshot(self, label: str = "") -> Dict[str, float]:
        """Take a memory usage snapshot."""
        # System memory
        memory = psutil.virtual_memory()
        process = psutil.Process(os.getpid())
        process_memory = process.memory_info()
        
        snapshot = {
            'timestamp': time.time(),
            'label': label,
            'system_total_gb': memory.total / (1024**3),
            'system_available_gb': memory.available / (1024**3),
            'system_used_percent': memory.percent,
            'process_rss_gb': process_memory.rss / (1024**3),
            'process_vms_gb': process_memory.vms / (1024**3)
        }
        
        # GPU memory if available
        if torch.cuda.is_available():
            # Handle parallel runs where CUDA might be disabled
            if os.getenv("CUDA_VISIBLE_DEVICES", "") == "":
                torch.cuda.empty_cache()
            
            gpu_mem = torch.cuda.memory_stats()
            snapshot.update({
                'gpu_allocated_gb': torch.cuda.memory_allocated() / (1024**3),
                'gpu_reserved_gb': torch.cuda.memory_reserved() / (1024**3),
                'gpu_max_allocated_gb': torch.cuda.max_memory_allocated() / (1024**3),
                'gpu_max_reserved_gb': torch.cuda.max_memory_reserved() / (1024**3)
            })
        
        self.memory_snapshots.append(snapshot)
        return snapshot
    
    def get_memory_growth(self) -> Dict[str, float]:
        """Calculate memory growth since baseline."""
        if not self.memory_snapshots or self.baseline_memory is None:
            return {}
        
        current = self.memory_snapshots[-1]
        growth = {}
        
        for key in ['process_rss_gb', 'gpu_allocated_gb', 'gpu_reserved_gb']:
            if key in current and key in self.baseline_memory:
                growth[f'{key}_growth'] = current[key] - self.baseline_memory[key]
        
        return growth
    
    def set_baseline(self) -> None:
        """Set current memory state as baseline."""
        self.baseline_memory = self.take_snapshot("baseline")


class ImageQualityMetrics:
    """Image quality metrics calculator for channel estimation."""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def calculate_all_metrics(self, predictions: torch.Tensor, targets: torch.Tensor, 
                            denormalize_fn: Optional[callable] = None) -> Dict[str, float]:
        """
        Calculate all image quality metrics.
        
        Args:
            predictions: Model predictions [batch, channels, height, width]
            targets: Ground truth targets [batch, channels, height, width]  
            denormalize_fn: Function to denormalize tensors for metric calculation
            
        Returns:
            Dictionary of calculated metrics
        """
        # Denormalize if function provided
        if denormalize_fn is not None:
            pred_denorm = denormalize_fn(predictions)
            target_denorm = denormalize_fn(targets)
        else:
            pred_denorm = predictions
            target_denorm = targets
        
        metrics = {}
        
        # Convert to numpy for calculations
        pred_np = pred_denorm.detach().cpu().numpy()
        target_np = target_denorm.detach().cpu().numpy()
        
        # NMSE calculation
        metrics['nmse'] = self._calculate_nmse(pred_np, target_np)
        
        # PSNR calculation
        metrics['psnr'] = self._calculate_psnr(pred_np, target_np)
        
        # SSIM calculation
        metrics['ssim'] = self._calculate_ssim(pred_np, target_np)
        
        # Per-channel metrics if multi-channel
        if pred_np.shape[1] > 1:
            for ch in range(pred_np.shape[1]):
                pred_ch = pred_np[:, ch:ch+1, :, :]
                target_ch = target_np[:, ch:ch+1, :, :]
                
                metrics[f'nmse_ch{ch}'] = self._calculate_nmse(pred_ch, target_ch)
                metrics[f'psnr_ch{ch}'] = self._calculate_psnr(pred_ch, target_ch)
                metrics[f'ssim_ch{ch}'] = self._calculate_ssim(pred_ch, target_ch)
        
        return metrics
    
    def _calculate_nmse(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """Calculate Normalized Mean Squared Error."""
        if calculate_nmse is not None:
            # Use shared implementation if available
            pred_tensor = torch.from_numpy(predictions)
            target_tensor = torch.from_numpy(targets)
            return calculate_nmse(pred_tensor, target_tensor).item()
        else:
            # Fallback implementation
            mse = np.mean((predictions - targets) ** 2)
            signal_power = np.mean(targets ** 2)
            return mse / (signal_power + 1e-8)
    
    def _calculate_psnr(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """Calculate Peak Signal-to-Noise Ratio."""
        if calculate_psnr is not None:
            # Use shared implementation if available
            pred_tensor = torch.from_numpy(predictions)
            target_tensor = torch.from_numpy(targets)
            return calculate_psnr(pred_tensor, target_tensor).item()
        else:
            # Fallback implementation
            mse = np.mean((predictions - targets) ** 2)
            if mse == 0:
                return float('inf')
            
            max_val = np.max(np.abs(targets))
            if max_val == 0:
                return 0.0
            
            return 20 * np.log10(max_val / np.sqrt(mse))
    
    def _calculate_ssim(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """Calculate Structural Similarity Index."""
        if compute_ssim is not None:
            # Use shared implementation if available
            return compute_ssim(predictions, targets)
        else:
            # Simple fallback - just use correlation coefficient
            pred_flat = predictions.flatten()
            target_flat = targets.flatten()
            
            if np.std(pred_flat) == 0 or np.std(target_flat) == 0:
                return 0.0
            
            correlation = np.corrcoef(pred_flat, target_flat)[0, 1]
            return max(0.0, correlation)  # Clamp to [0, 1]


class BufferMonitor:
    """Monitor for replay buffer statistics and utilization."""
    
    def __init__(self):
        self.buffer_stats = defaultdict(dict)
        self.utilization_history = defaultdict(list)
        
    def update_buffer_stats(self, domain_id: str, buffer_type: str, 
                          current_size: int, max_size: int, 
                          sample_ages: Optional[List[float]] = None) -> None:
        """
        Update buffer statistics.
        
        Args:
            domain_id: Domain identifier
            buffer_type: 'offline' or 'online'
            current_size: Current number of samples in buffer
            max_size: Maximum buffer capacity
            sample_ages: Optional list of sample ages (time since added)
        """
        buffer_key = f"{domain_id}_{buffer_type}"
        
        stats = {
            'current_size': current_size,
            'max_size': max_size,
            'utilization': current_size / max_size if max_size > 0 else 0.0,
            'timestamp': time.time()
        }
        
        if sample_ages is not None:
            stats.update({
                'mean_age': np.mean(sample_ages) if sample_ages else 0.0,
                'max_age': np.max(sample_ages) if sample_ages else 0.0,
                'age_std': np.std(sample_ages) if sample_ages else 0.0
            })
        
        self.buffer_stats[buffer_key] = stats
        self.utilization_history[buffer_key].append(stats['utilization'])
    
    def get_buffer_summary(self) -> Dict[str, Any]:
        """Get summary of all buffer statistics."""
        summary = {
            'current_stats': dict(self.buffer_stats),
            'utilization_trends': {}
        }
        
        for buffer_key, history in self.utilization_history.items():
            if history:
                summary['utilization_trends'][buffer_key] = {
                    'mean': np.mean(history),
                    'std': np.std(history),
                    'min': np.min(history),
                    'max': np.max(history),
                    'current': history[-1],
                    'trend': np.polyfit(range(len(history)), history, 1)[0] if len(history) > 1 else 0.0
                }
        
        return summary


class LoRAAdapterTracker:
    """Track LoRA adapter weights and specialization over time."""
    
    def __init__(self):
        self.weight_history = defaultdict(list)
        self.similarity_matrix = {}
        
    def capture_adapter_weights(self, model: nn.Module, domain_id: str) -> None:
        """Capture current LoRA adapter weights for a domain."""
        weights = {}
        
        # Extract LoRA weights from model
        for name, module in model.named_modules():
            if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                # Capture LoRA A and B matrices
                if hasattr(module.lora_A, domain_id):
                    lora_a = getattr(module.lora_A, domain_id).weight.data.clone()
                    lora_b = getattr(module.lora_B, domain_id).weight.data.clone()
                    
                    weights[f"{name}_A"] = {
                        'mean': torch.mean(torch.abs(lora_a)).item(),
                        'std': torch.std(lora_a).item(),
                        'frobenius_norm': torch.norm(lora_a, 'fro').item()
                    }
                    weights[f"{name}_B"] = {
                        'mean': torch.mean(torch.abs(lora_b)).item(),
                        'std': torch.std(lora_b).item(),
                        'frobenius_norm': torch.norm(lora_b, 'fro').item()
                    }
        
        if weights:
            weights['timestamp'] = time.time()
            self.weight_history[domain_id].append(weights)
    
    def calculate_adapter_similarity(self, model: nn.Module, 
                                   domain1: str, domain2: str) -> float:
        """Calculate similarity between two domain adapters."""
        similarities = []
        
        for name, module in model.named_modules():
            if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                if (hasattr(module.lora_A, domain1) and hasattr(module.lora_A, domain2) and
                    hasattr(module.lora_B, domain1) and hasattr(module.lora_B, domain2)):
                    
                    # Get weights for both domains
                    a1 = getattr(module.lora_A, domain1).weight.data.flatten()
                    a2 = getattr(module.lora_A, domain2).weight.data.flatten()
                    b1 = getattr(module.lora_B, domain1).weight.data.flatten()
                    b2 = getattr(module.lora_B, domain2).weight.data.flatten()
                    
                    # Calculate cosine similarity
                    sim_a = torch.cosine_similarity(a1, a2, dim=0).item()
                    sim_b = torch.cosine_similarity(b1, b2, dim=0).item()
                    
                    similarities.extend([sim_a, sim_b])
        
        return np.mean(similarities) if similarities else 0.0
    
    def get_specialization_metrics(self) -> Dict[str, Any]:
        """Get adapter specialization metrics."""
        metrics = {}
        
        # Weight evolution analysis
        for domain_id, history in self.weight_history.items():
            if len(history) > 1:
                # Track weight magnitude changes over time
                norms = [h.get('total_norm', 0) for h in history]
                metrics[f"{domain_id}_weight_trend"] = {
                    'initial_norm': norms[0] if norms else 0,
                    'final_norm': norms[-1] if norms else 0,
                    'growth_rate': (norms[-1] - norms[0]) / len(norms) if len(norms) > 1 else 0
                }
        
        return metrics


class ComprehensiveMetricsCollector:
    """Main collector that orchestrates all metrics collection."""
    
    def __init__(self, save_dir: str = "metrics_logs"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Initialize all metric collectors
        self.timing_profiler = TimingProfiler()
        self.memory_profiler = MemoryProfiler()
        self.image_metrics = ImageQualityMetrics()
        self.buffer_monitor = BufferMonitor()
        self.lora_tracker = LoRAAdapterTracker()
        
        # Results storage
        self.sample_results = []
        self.domain_performance_history = defaultdict(list)
        self.trigger_stats = defaultdict(int)
        
        # Optional transfer metrics (set externally)
        self.transfer_tracker = None
        
        # Set memory baseline
        self.memory_profiler.set_baseline()
        
    def record_sample_result(self, sample_data: Dict[str, Any]) -> None:
        """Record results from processing a single sample."""
        # Add timestamp
        sample_data['timestamp'] = time.time()
        sample_data['sample_id'] = len(self.sample_results)
        
        self.sample_results.append(sample_data)
        
        # Update domain performance history
        domain_id = sample_data.get('domain_id')
        if domain_id is not None:
            metrics = {k: v for k, v in sample_data.items() 
                      if k in ['nmse', 'psnr', 'ssim', 'inference_time']}
            metrics['timestamp'] = sample_data['timestamp']
            self.domain_performance_history[domain_id].append(metrics)
    
    def record_training_trigger(self, trigger_type: str, domain_id: str) -> None:
        """Record a training trigger event."""
        self.trigger_stats[f"{trigger_type}_{domain_id}"] += 1
        self.trigger_stats['total_triggers'] += 1
    
    def get_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive evaluation report."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_samples_processed': len(self.sample_results),
            'timing_stats': self.timing_profiler.get_stats(),
            'memory_stats': {
                'snapshots': self.memory_profiler.memory_snapshots[-5:],  # Last 5 snapshots
                'growth': self.memory_profiler.get_memory_growth()
            },
            'buffer_stats': self.buffer_monitor.get_buffer_summary(),
            'lora_specialization': self.lora_tracker.get_specialization_metrics(),
            'trigger_stats': dict(self.trigger_stats),
            'domain_performance': self._analyze_domain_performance(),
            'forgetting_analysis': self._analyze_forgetting(),
            'system_health': self._check_system_health()
        }
        
        # Add transfer metrics if available
        if self.transfer_tracker is not None:
            report['transfer_metrics'] = self.transfer_tracker.get_transfer_summary()
        
        return report
    
    def _analyze_domain_performance(self) -> Dict[str, Any]:
        """Analyze performance trends per domain."""
        analysis = {}
        
        for domain_id, history in self.domain_performance_history.items():
            if not history:
                continue
                
            # Extract metrics over time
            nmse_values = [h.get('nmse', float('nan')) for h in history]
            psnr_values = [h.get('psnr', float('nan')) for h in history]
            ssim_values = [h.get('ssim', float('nan')) for h in history]
            
            # Filter out NaN values
            nmse_values = [x for x in nmse_values if not np.isnan(x)]
            psnr_values = [x for x in psnr_values if not np.isnan(x)]
            ssim_values = [x for x in ssim_values if not np.isnan(x)]
            
            if nmse_values:
                analysis[domain_id] = {
                    'sample_count': len(history),
                    'nmse': {
                        'mean': np.mean(nmse_values),
                        'std': np.std(nmse_values),
                        'trend': np.polyfit(range(len(nmse_values)), nmse_values, 1)[0] if len(nmse_values) > 1 else 0.0,
                        'latest': nmse_values[-1] if nmse_values else None
                    },
                    'psnr': {
                        'mean': np.mean(psnr_values) if psnr_values else None,
                        'std': np.std(psnr_values) if psnr_values else None,
                        'trend': np.polyfit(range(len(psnr_values)), psnr_values, 1)[0] if len(psnr_values) > 1 else 0.0,
                        'latest': psnr_values[-1] if psnr_values else None
                    },
                    'ssim': {
                        'mean': np.mean(ssim_values) if ssim_values else None,
                        'std': np.std(ssim_values) if ssim_values else None,
                        'trend': np.polyfit(range(len(ssim_values)), ssim_values, 1)[0] if len(ssim_values) > 1 else 0.0,
                        'latest': ssim_values[-1] if ssim_values else None
                    }
                }
        
        return analysis
    
    def _analyze_forgetting(self) -> Dict[str, Any]:
        """Analyze catastrophic forgetting across domains."""
        forgetting_analysis = {}
        
        # For each domain, analyze performance degradation when other domains are learned
        for domain_id, history in self.domain_performance_history.items():
            if len(history) < 2:
                continue
            
            # Get performance at different time points
            early_performance = np.mean([h.get('nmse', float('nan')) for h in history[:len(history)//3]])
            late_performance = np.mean([h.get('nmse', float('nan')) for h in history[-len(history)//3:]])
            
            if not (np.isnan(early_performance) or np.isnan(late_performance)):
                forgetting_score = late_performance - early_performance  # Higher = more forgetting
                
                forgetting_analysis[domain_id] = {
                    'early_nmse': early_performance,
                    'late_nmse': late_performance,
                    'forgetting_score': forgetting_score,
                    'relative_forgetting': forgetting_score / early_performance if early_performance > 0 else 0.0
                }
        
        return forgetting_analysis
    
    def _check_system_health(self) -> Dict[str, Any]:
        """Check overall system health and performance."""
        health = {
            'status': 'healthy',
            'warnings': [],
            'errors': []
        }
        
        # Check memory growth
        memory_growth = self.memory_profiler.get_memory_growth()
        if 'process_rss_gb_growth' in memory_growth and memory_growth['process_rss_gb_growth'] > 2.0:
            health['warnings'].append(f"High memory growth: {memory_growth['process_rss_gb_growth']:.2f} GB")
        
        # Check timing performance (adjust threshold for CPU vs GPU)
        timing_stats = self.timing_profiler.get_stats()
        inference_threshold = 0.25 if not torch.cuda.is_available() else 0.1  # Higher tolerance for CPU
        if 'inference' in timing_stats and timing_stats['inference']['mean'] > inference_threshold:
            device_type = "CPU" if not torch.cuda.is_available() else "GPU"
            health['warnings'].append(f"Slow inference on {device_type}: {timing_stats['inference']['mean']:.3f}s per sample")
        
        # Check for NaN values in recent samples
        recent_samples = self.sample_results[-100:] if len(self.sample_results) > 100 else self.sample_results
        nan_count = sum(1 for s in recent_samples if any(np.isnan(v) for v in s.values() if isinstance(v, (int, float))))
        if nan_count > 0:
            health['errors'].append(f"Found {nan_count} samples with NaN values in last {len(recent_samples)} samples")
        
        # Set overall status
        if health['errors']:
            health['status'] = 'error'
        elif health['warnings']:
            health['status'] = 'warning'
        
        return health
    
    def save_results(self, prefix: str = "comprehensive_metrics") -> str:
        """Save all collected metrics to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save comprehensive report
        report = self.get_comprehensive_report()
        report_file = os.path.join(self.save_dir, f"{prefix}_report_{timestamp}.json")
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Save raw sample data
        samples_file = os.path.join(self.save_dir, f"{prefix}_samples_{timestamp}.json")
        with open(samples_file, 'w') as f:
            json.dump(self.sample_results, f, indent=2, default=str)
        
        print(f"Comprehensive metrics saved to {report_file}")
        return report_file


# Context manager for automatic timing
class TimeOperation:
    """Context manager for automatic timing of operations."""
    
    def __init__(self, profiler: TimingProfiler, operation: str):
        self.profiler = profiler
        self.operation = operation
    
    def __enter__(self):
        self.profiler.start_timer(self.operation)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.profiler.end_timer(self.operation) 