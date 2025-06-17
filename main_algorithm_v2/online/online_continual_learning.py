"""
Online Continual Learning Script

This script implements the true online continual learning algorithm (Algorithm 2) where:
- Samples arrive in a streaming fashion
- Each sample is evaluated (inference)
- Training updates are triggered based on conditions
- Everything happens in a single unified loop

Usage:
    python online_continual_learning.py --config config/online_config.yaml
"""

import argparse
import random
import time
import torch
import numpy as np
from typing import Dict, Any
import os
from datetime import datetime

# Import offline utilities
from main_algorithm_v2.offline.src.utils import get_device, set_seed

# Import online components  
from main_algorithm_v2.online.src.config import load_online_config
from main_algorithm_v2.online.src.data import OnlineDataPipeline
from main_algorithm_v2.online.src.model_manager import OnlineModelManager
from main_algorithm_v2.online.src.online_buffer_manager import OnlineBufferManager
from main_algorithm_v2.online.src.gradient_buffer_manager import GradientDiversityBufferManager
from main_algorithm_v2.online.src.trigger_manager import TriggerManager
from main_algorithm_v2.online.src.mixed_batch_manager import MixedBatchManager
from main_algorithm_v2.online.src.loss_functions import (
    masked_nmse, masked_nmse_tensor, extract_pilot_ground_truth,
    exponential_moving_average
)
from main_algorithm_v2.online.src.comprehensive_metrics import (
    ComprehensiveMetricsCollector, TimeOperation
)
from main_algorithm_v2.online.src.transfer_metrics import TransferMetricsTracker


class OnlineContinualLearner:
    """
    Unified online continual learning system implementing Algorithm 2.
    Combines evaluation and training in a single streaming loop.
    """
    
    def __init__(self, config_path: str):
        """Initialize the online continual learning system."""
        print("=" * 80)
        print("ONLINE CONTINUAL LEARNING SYSTEM")
        print("=" * 80)
        
        # Load configuration
        print("\n[CONFIG] Loading configuration...")
        self.config = load_online_config(config_path)
        print(f"Experiment: {self.config.experiment_name}")
        
        # Set random seed
        if hasattr(self.config.offline_config.framework, 'seed'):
            set_seed(self.config.offline_config.framework.seed)
            print(f"Random seed: {self.config.offline_config.framework.seed}")
        
        # Get device
        self.device = get_device(self.config.offline_config.hardware.device)
        print(f"Device: {self.device}")
        
        # Initialize model manager with EWC support
        print("\n[MODEL] Initialising model manager...")
        self.model_manager = OnlineModelManager(
            self.config.offline_checkpoint_path,
            self.device,
            enable_ewc=self.config.online_training.ewc.enabled,
            ewc_lambda=self.config.online_training.ewc.lambda_ewc
        )
        
        # Print model info
        model_info = self.model_manager.get_model_info()
        print(f"Available domains: {model_info['available_domains']}")
        print(f"Model: {model_info['model_name']}")
        
        # Initialize data pipeline
        print("\n[DATA] Initialising data pipeline...")
        norm_stats = self.model_manager.get_normalisation_stats()
        raw_data_config = self.config.raw_data.model_dump()
        domain_remapping = raw_data_config.get('domain_remapping', {})
        
        self.data_pipeline = OnlineDataPipeline(
            raw_data_config=raw_data_config,
            norm_stats=norm_stats,
            interpolation_kernel=self.config.offline_config.data.interpolation,
            domain_remapping=domain_remapping
        )
        
        # Initialize online buffer manager
        if self.config.online_evaluation.verbose:
            print("\n[BUFFER] Initialising buffer manager...")
        buffer_config = self.config.online_evaluation.online_buffer.model_dump()
        if buffer_config.get('strategy') == 'gradient_diversity':
            self.online_buffer_manager = GradientDiversityBufferManager(buffer_config, verbose=self.config.online_evaluation.verbose)
        else:
            self.online_buffer_manager = OnlineBufferManager(buffer_config)
        
        # Initialize training components
        if self.config.online_evaluation.verbose:
            print("\n[TRAINING] Initialising training components...")
        self.trigger_manager = TriggerManager(self.config.online_training.trigger.model_dump())
        self.mixed_batch_manager = MixedBatchManager(self.config.online_training.mixed_batch.model_dump())
        
        # Training parameters
        self.learning_rate = self.config.online_training.learning_rate
        self.weight_decay  = self.config.online_training.weight_decay
        self.max_epochs_per_trigger = self.config.online_training.max_epochs_per_trigger
        self.early_stopping_patience = self.config.online_training.early_stopping_patience
        
        # Per-domain components
        self.optimizers = {}  # Domain-specific optimizers
        self.ema_losses = {}  # Exponential moving average losses per domain
        self.ema_alpha = 0.1  # Smoothing factor
        
        # EWC monitoring (for analysis only - no auto-adjustment)
        self.domain_ewc_ratios = {}  # Track EWC/MSE ratios per domain for analysis
        
        # Statistics tracking
        self.total_samples = 0
        self.total_updates = 0
        self.domain_counts = {d: 0 for d in self.config.raw_data.domains}
        self.domain_update_counts = {str(d): 0 for d in self.config.raw_data.domains}
        self.results_history = []
        
        # Initialize comprehensive metrics collector
        if self.config.online_evaluation.verbose:
            print("\n[METRICS] Initialising comprehensive metrics collection...")
        metrics_dir = os.path.join("main_algorithm_v2", "online", "eval", self.config.experiment_name)
        self.metrics_collector = ComprehensiveMetricsCollector(save_dir=metrics_dir)
        if self.config.online_evaluation.verbose:
            print(f"Metrics will be saved to: {metrics_dir}")
        
        # Initialize transfer metrics tracker (academic standard)
        self.transfer_tracker = TransferMetricsTracker(
            domains=self.config.raw_data.domains,
            evaluation_frequency=self.config.online_evaluation.log_frequency * 5  # Every 5x log frequency
        )
        if self.config.online_evaluation.verbose:
            print(f"Transfer metrics (FWT/BWT) will be evaluated every {self.transfer_tracker.evaluation_frequency} samples")
        
        # Link transfer tracker to comprehensive metrics
        self.metrics_collector.transfer_tracker = self.transfer_tracker
        
        # Add proof-of-concept metrics tracking
        self.adaptation_tracker = {
            'total_adaptations': 0,
            'successful_adaptations': 0,  # Where NMSE improved
            'domain_adaptation_history': {str(d): [] for d in self.config.raw_data.domains},
            'baseline_performance': {},  # Initial performance per domain
            'adaptation_magnitudes': [],  # Track how much improvement we get
            'failed_adaptations': 0,
            'domain_specialization_scores': {str(d): [] for d in self.config.raw_data.domains}
        }
        
        print("\n[SUCCESS] Online continual learning system initialized!")
        print(f"   EWC: {'Enabled' if self.model_manager.is_ewc_enabled() else 'Disabled'}")
        if self.model_manager.is_ewc_enabled():
            print(f"   EWC λ: {self.config.online_training.ewc.lambda_ewc} (fixed)")
        print(f"   Learning rate: {self.learning_rate}")
        print(f"   Trigger: {self.config.online_training.trigger.type}")
        print(f"   Proof-of-concept tracking: Enabled")

    def _get_temporal_shift_domain(self, sample_id: int) -> int:
        """
        Get domain ID based on temporal domain shift configuration.
        Used for dynamic domain shift evaluation.
        """
        # Get temporal shift configuration
        shift_config = getattr(self.config.raw_data, 'temporal_domain_shift', {})
        
        if not shift_config.get('enabled', False):
            # Fallback to primary domain if no shift config
            return shift_config.get('primary_domain', self.config.raw_data.domains[0])
        
        primary_domain = shift_config['primary_domain']
        shift_domain = shift_config['shift_domain']
        shift_start = shift_config['shift_start_sample']
        shift_end = shift_config['shift_end_sample']
        
        # Determine which domain to use based on sample position
        if shift_start <= sample_id < shift_end:
            # During shift period: use shift_domain data but labeled as primary_domain
            actual_domain = shift_domain
            if self.config.online_evaluation.verbose and sample_id == shift_start:
                print(f"[DOMAIN SHIFT] Sample {sample_id}: Starting shift to domain {shift_domain} (labeled as {primary_domain})")
        else:
            # Normal operation: use primary_domain
            actual_domain = primary_domain
            if self.config.online_evaluation.verbose and sample_id == shift_end:
                print(f"[DOMAIN SHIFT] Sample {sample_id}: Ending shift, back to domain {primary_domain}")
        
        return actual_domain

    def _log_ewc_penalty_ratio(self, domain_id_str: str, mse_loss: float, ewc_loss: float):
        """Log the ratio between EWC penalty and MSE loss for analysis."""
        import math
        
        # Guard against NaN values
        if math.isnan(mse_loss) or math.isnan(ewc_loss):
            print(f"[EWC] Domain {domain_id_str}: NaN detected in losses (MSE={mse_loss}, EWC={ewc_loss})")
            return
            
        if ewc_loss > 0 and mse_loss > 0:
            ratio = ewc_loss / mse_loss
            
            # Store for analysis
            if domain_id_str not in self.domain_ewc_ratios:
                self.domain_ewc_ratios[domain_id_str] = []
            self.domain_ewc_ratios[domain_id_str].append(ratio)
            
            # Log current values (only if verbose)
            if self.config.online_evaluation.verbose:
                lambda_val = self.config.online_training.ewc.lambda_ewc
                print(f"[EWC] Domain {domain_id_str}: MSE={mse_loss:.6f}, EWC={ewc_loss:.6f}, Ratio={ratio:.3f}, λ={lambda_val}")
            
        else:
            if self.config.online_evaluation.verbose:
                print(f"[EWC] Domain {domain_id_str}: No valid EWC penalty (ewc_loss={ewc_loss:.6f}, mse_loss={mse_loss:.6f})")

    
    def create_domain_optimizer(self, domain_id: str) -> torch.optim.Optimizer:
        """Create optimizer for domain-specific parameters."""
        params_to_optimize = []
        
        # Get LoRA parameters for this domain
        for name, module in self.model_manager.model.named_modules():
            if hasattr(module, 'task_adapters') and domain_id in module.task_adapters:
                adapter = module.task_adapters[domain_id]
                params_to_optimize.extend([adapter.A, adapter.B])
        
        # Get BatchNorm parameters for this domain
        for name, module in self.model_manager.model.named_modules():
            if hasattr(module, 'bns') and domain_id in module.bns:
                bn = module.bns[domain_id]
                params_to_optimize.extend([bn.weight, bn.bias])
        
        return torch.optim.AdamW(params_to_optimize, lr=self.learning_rate, weight_decay=self.weight_decay)
    
    def process_single_sample(self, sample_id: int) -> Dict[str, Any]:
        """
        Process a single online sample following Algorithm 2.
        This is the core online learning loop.
        """
        # Start total sample timing
        with TimeOperation(self.metrics_collector.timing_profiler, "total_sample"):
            # Algorithm 2, Line 1: Get streaming sample
            with TimeOperation(self.metrics_collector.timing_profiler, "domain_selection"):
                if self.config.online_evaluation.domain_selection == "random":
                    domain_id = random.choice(self.config.raw_data.domains)
                elif self.config.online_evaluation.domain_selection == "temporal_shift":
                    # Handle temporal domain shifting for adaptation testing
                    domain_id = self._get_temporal_shift_domain(sample_id)
                elif self.config.online_evaluation.domain_selection == "sequential_blocks":
                    # Sequential blocks of domains - simple incremental approach
                    block_size = getattr(self.config.online_evaluation, 'domain_block_size', 2000)
                    domain_idx = sample_id // block_size
                    
                    # Clamp to valid domain indices (don't cycle)
                    if domain_idx >= len(self.config.raw_data.domains):
                        domain_idx = len(self.config.raw_data.domains) - 1
                    
                    domain_id = self.config.raw_data.domains[domain_idx]
                    
                    # Verbose logging for sequential blocks
                    if self.config.online_evaluation.verbose and (sample_id % block_size == 0 or sample_id == 0):
                        print(f"[DOMAIN] Sample {sample_id}: Starting block for Domain {domain_id} (samples {sample_id}-{sample_id + block_size - 1})")
                else:
                    # Default: cycle through domains one by one
                    domain_idx = sample_id % len(self.config.raw_data.domains)
                    domain_id = self.config.raw_data.domains[domain_idx]
            
            domain_id_str = str(domain_id)
            self.domain_counts[domain_id] += 1
            
            # For temporal shift experiments, we need to handle mislabeling
            adapter_domain_id = domain_id  # Default: use same domain for adapter
            if (self.config.online_evaluation.domain_selection == "temporal_shift" and 
                hasattr(self.config.raw_data, 'temporal_domain_shift')):
                shift_config = self.config.raw_data.temporal_domain_shift
                if (shift_config.get('enabled', False) and 
                    shift_config['shift_start_sample'] <= sample_id < shift_config['shift_end_sample']):
                    # During shift: use primary domain adapter but shift domain data
                    adapter_domain_id = shift_config['primary_domain']
                    if self.config.online_evaluation.verbose and sample_id % 50 == 0:
                        print(f"[MISLABEL] Sample {sample_id}: Using domain {domain_id} data with domain {adapter_domain_id} adapter")
            
            # Switch to the adapter domain (which may be different from data domain during shift)
            with TimeOperation(self.metrics_collector.timing_profiler, "domain_switch"):
                self.model_manager.set_active_domain(adapter_domain_id)
            
            # Get buffers
            online_buffer = self.online_buffer_manager.get_or_create_buffer(domain_id)
            offline_buffer = self.model_manager.get_replay_buffer(domain_id)
            
            # Algorithm 2, Line 1-3: Get sample and preprocess
            with TimeOperation(self.metrics_collector.timing_profiler, "preprocessing"):
                model_input, ground_truth, pilot_mask, metadata = self.data_pipeline.process_sample_online(domain_id)
                
                # Move to device
                model_input = model_input.to(self.device)
                ground_truth = ground_truth.to(self.device)
                pilot_mask = pilot_mask.to(self.device)
                
                # Algorithm 2, Line 2: Extract pilot-only ground truth
                pilot_ground_truth = extract_pilot_ground_truth(ground_truth, pilot_mask)
            
            # Algorithm 2, Line 4-5: Forward pass and calculate loss
            self.model_manager.model.eval()
            
            with TimeOperation(self.metrics_collector.timing_profiler, "inference"):
                with torch.no_grad():
                    prediction        = self.model_manager.model(model_input.unsqueeze(0))
                    prediction        = prediction.squeeze(0)
                    masked_nmse_value = masked_nmse(prediction, pilot_ground_truth, pilot_mask)
            
            # Calculate comprehensive image quality metrics
            with TimeOperation(self.metrics_collector.timing_profiler, "quality_metrics"):
                # Create denormalization function if available
                denorm_fn = None
                if hasattr(self.model_manager, 'get_denormalization_fn'):
                    denorm_fn = self.model_manager.get_denormalization_fn()
                
                # Calculate comprehensive metrics on the prediction vs ground truth
                quality_metrics = self.metrics_collector.image_metrics.calculate_all_metrics(
                    prediction.unsqueeze(0), ground_truth.unsqueeze(0), denorm_fn
                )
            
            # Extract timing for backward compatibility
            inference_time = self.metrics_collector.timing_profiler.timings['inference'][-1] if 'inference' in self.metrics_collector.timing_profiler.timings else 0.0
        
            # Algorithm 2, Line 7: Add to online buffer
            with TimeOperation(self.metrics_collector.timing_profiler, "buffer_management"):
                if online_buffer is not None:
                    if hasattr(self.online_buffer_manager, 'add_sample'):
                        # Gradient diversity buffer
                        was_added, reason = self.online_buffer_manager.add_sample(
                            domain_id, self.model_manager, (model_input, pilot_ground_truth),
                            pilot_mask, masked_nmse_value
                        )
                    else:
                        # Standard buffer
                        online_buffer.add((model_input, pilot_ground_truth))
                        was_added, reason = True, "added_to_buffer"
                else:
                    was_added, reason = False, "no_buffer"
                
                # Update buffer monitoring
                if online_buffer is not None:
                    self.metrics_collector.buffer_monitor.update_buffer_stats(
                        domain_id_str, "online", 
                        len(online_buffer) if hasattr(online_buffer, '__len__') else 0,
                        getattr(online_buffer, 'capacity', 100)
                    )
                
                if offline_buffer is not None:
                    self.metrics_collector.buffer_monitor.update_buffer_stats(
                        domain_id_str, "offline",
                        len(offline_buffer) if hasattr(offline_buffer, '__len__') else 0,
                        getattr(offline_buffer, 'capacity', 1000)
                    )
            
            # Algorithm 2, Line 8: Update EMA loss
            if domain_id_str not in self.ema_losses:
                self.ema_losses[domain_id_str] = masked_nmse_value
            else:
                self.ema_losses[domain_id_str] = exponential_moving_average(
                    self.ema_losses[domain_id_str], masked_nmse_value, self.ema_alpha
                )
            
            # Algorithm 2, Line 9: Check trigger (only if training is enabled)
            if self.config.online_training.enabled:
                # Pass current_loss for drift trigger compatibility
                should_update = self.trigger_manager.should_trigger_update(current_loss=masked_nmse_value)
            else:
                should_update = False
            # print(f"Should update: {should_update}")
        
            # Extract preprocessing time for backward compatibility
            preprocessing_time = self.metrics_collector.timing_profiler.timings['preprocessing'][-1] if 'preprocessing' in self.metrics_collector.timing_profiler.timings else 0.0
            
            # Initialize result with comprehensive metrics
            result = {
                'sample_id': sample_id,
                'domain_id': domain_id,
                'masked_nmse_pre': masked_nmse_value,
                'inference_time': inference_time,  # Extracted from timing profiler
                'preprocessing_time': preprocessing_time,  # From timing profiler
                'ema_loss': self.ema_losses[domain_id_str],
                'buffer_added': was_added,
                'buffer_reason': reason,
                'triggered_update': should_update,
                'training_time': 0.0,  # Will be updated if training occurs
                'masked_nmse_post': masked_nmse_value,  # Same as pre if no training
                **quality_metrics  # Add SSIM, PSNR, per-channel metrics
            }
        
            # Algorithm 2, Lines 10-20: Perform update if triggered
            if should_update:
                with TimeOperation(self.metrics_collector.timing_profiler, "training"):
                    # Record trigger
                    self.metrics_collector.record_training_trigger("time_volume", domain_id_str)
                    
                    # Capture LoRA weights before training
                    self.metrics_collector.lora_tracker.capture_adapter_weights(
                        self.model_manager.model, domain_id_str
                    )
                    
                    # Store baseline performance if first time seeing this domain
                    if domain_id_str not in self.adaptation_tracker['baseline_performance']:
                        self.adaptation_tracker['baseline_performance'][domain_id_str] = masked_nmse_value
                        if self.config.online_evaluation.verbose:
                            print(f"[ADAPTATION] Baseline performance for Domain {domain_id_str}: {masked_nmse_value:.6f}")
                    
                    # Get or create optimizer
                    if domain_id_str not in self.optimizers:
                        self.optimizers[domain_id_str] = self.create_domain_optimizer(domain_id_str)
                    optimizer = self.optimizers[domain_id_str]
                
                    # Training loop
                    self.model_manager.model.train()
                    best_loss        = float('inf')
                    patience_counter = 0
                    epochs_run = 0
                    
                    for epoch in range(self.max_epochs_per_trigger):
                        epochs_run += 1
                        # Algorithm 2, Lines 11-15: Create mixed batch
                        batch_inputs, batch_targets, batch_masks = self.mixed_batch_manager.create_mixed_batch_tensors(
                            (model_input, pilot_ground_truth, pilot_mask),
                            offline_buffer, online_buffer, self.device
                        )
                        
                        # Forward pass
                        predictions = self.model_manager.model(batch_inputs)
                        
                        # Algorithm 2, Line 17: Calculate masked loss
                        total_loss = 0.0
                        for i in range(batch_inputs.size(0)):
                            loss = masked_nmse_tensor(predictions[i], batch_targets[i], batch_masks[i])
                            total_loss += loss
                        main_loss = total_loss / batch_inputs.size(0)
                        
                        # Algorithm 2, Line 18: Add EWC penalty
                        if self.model_manager.is_ewc_enabled():
                            ewc_loss = self.model_manager.get_ewc_loss(
                                domain_id_str,
                                exclude_current_task=self.config.online_training.ewc.exclude_current_task
                            )
                            
                            # Log EWC penalty ratio for analysis
                            self._log_ewc_penalty_ratio(domain_id_str, main_loss.item(), ewc_loss.item())
                            
                            total_loss = main_loss + ewc_loss
                        else:
                            total_loss = main_loss
                        
                        # Algorithm 2, Line 19: Backward and optimize
                        optimizer.zero_grad()
                        total_loss.backward()
                        optimizer.step()
                        
                        # Early stopping
                        if main_loss.item() < best_loss:
                            best_loss = main_loss.item()
                            patience_counter = 0
                        else:
                            patience_counter += 1
                            if patience_counter >= self.early_stopping_patience:
                                if self.config.online_evaluation.verbose:
                                    print(f"[TRAINING] Early stopping after {epochs_run} epochs (patience={patience_counter})")
                                break
                    
                    # Log training completion
                    if self.config.online_evaluation.verbose:
                        print(f"[TRAINING] Completed {epochs_run}/{self.max_epochs_per_trigger} epochs, best loss: {best_loss:.6f}")
                    
                    # Re-evaluate after training
                    self.model_manager.model.eval()
                    with torch.no_grad():
                        prediction_post = self.model_manager.model(model_input.unsqueeze(0)).squeeze(0)
                        masked_nmse_post = masked_nmse(prediction_post, pilot_ground_truth, pilot_mask)
                    
                    result['masked_nmse_post'] = masked_nmse_post
                    improvement = masked_nmse_value - masked_nmse_post
                    result['improvement'] = improvement
                    result['epochs_run'] = epochs_run
                    
                    # Update adaptation tracking (proof-of-concept metrics)
                    self.adaptation_tracker['total_adaptations'] += 1
                    
                    if improvement > 0:
                        self.adaptation_tracker['successful_adaptations'] += 1
                        self.adaptation_tracker['adaptation_magnitudes'].append(improvement)
                        
                        # Track domain specialization improvement
                        baseline = self.adaptation_tracker['baseline_performance'].get(domain_id_str, masked_nmse_value)
                        specialization_score = (baseline - masked_nmse_post) / baseline * 100  # % improvement from baseline
                        self.adaptation_tracker['domain_specialization_scores'][domain_id_str].append(specialization_score)
                        
                        if self.config.online_evaluation.verbose:
                            print(f"[ADAPTATION]  Success! Domain {domain_id_str}: {improvement:.6f} improvement")
                            print(f"             Specialization: {specialization_score:.2f}% better than baseline")
                    else:
                        self.adaptation_tracker['failed_adaptations'] += 1
                        if self.config.online_evaluation.verbose:
                            print(f"[ADAPTATION]  Failed. Domain {domain_id_str}: {improvement:.6f} (degraded)")
                    
                    # Store adaptation history
                    self.adaptation_tracker['domain_adaptation_history'][domain_id_str].append({
                        'sample_id': sample_id,
                        'nmse_before': masked_nmse_value,
                        'nmse_after': masked_nmse_post,
                        'improvement': improvement,
                        'epochs': epochs_run,
                        'timestamp': time.time()
                    })
                    
                    # Capture LoRA weights after training
                    self.metrics_collector.lora_tracker.capture_adapter_weights(
                        self.model_manager.model, domain_id_str
                    )
                    
                    # Update statistics
                    self.total_updates += 1
                    self.domain_update_counts[domain_id_str] += 1
                    
                    # Reset trigger
                    self.trigger_manager.reset_trigger()
                
                    # Extract training time and update result
                    training_time = self.metrics_collector.timing_profiler.timings['training'][-1] if 'training' in self.metrics_collector.timing_profiler.timings else 0.0
                    result['training_time'] = training_time
            
            # Update timing profiler sample count
            self.metrics_collector.timing_profiler.total_samples += 1
            
            # Take memory snapshot periodically
            if sample_id % 100 == 0:
                self.metrics_collector.memory_profiler.take_snapshot(f"sample_{sample_id}")
            
            # Record comprehensive sample result
            self.metrics_collector.record_sample_result(result)
            
            # Update transfer metrics tracking
            self.transfer_tracker.increment_sample_count()
            
            # Record transfer snapshot if needed
            if self.transfer_tracker.should_evaluate_transfer():
                if self.config.online_evaluation.verbose:
                    print(f"[TRANSFER] Evaluating transfer metrics at sample {sample_id}...")
                with TimeOperation(self.metrics_collector.timing_profiler, "transfer_evaluation"):
                    snapshot = self.transfer_tracker.record_transfer_snapshot(
                        self.model_manager, self.data_pipeline, self.device, domain_id_str
                    )
                    if snapshot and self.config.online_evaluation.verbose:
                        print(f"[TRANSFER] Snapshot recorded: {len(snapshot.domain_performances)} domains evaluated")
            
            # Store result
            self.results_history.append(result)
            self.total_samples += 1
            
            return result
    
    def run(self, num_samples: int):
        """Run the online continual learning process for specified number of samples."""
        print(f"\n[START] Starting online continual learning for {num_samples} samples...")
        print("-" * 80)
        
        start_time = time.time()
        
        try:
            for sample_id in range(num_samples):
                # Process single sample
                result = self.process_single_sample(sample_id)
                
                # Log periodically
                if (sample_id + 1) % self.config.online_evaluation.log_frequency == 0:
                    self.log_progress(sample_id + 1, result)
        
        except KeyboardInterrupt:
            print("\n[WARNING] Interrupted by user")
        except Exception as e:
            print(f"\n[ERROR] Error: {e}")
            import traceback
            traceback.print_exc()
        
        # Final summary
        total_time = time.time() - start_time
        self.print_final_summary(total_time)
    
    def log_progress(self, sample_num: int, latest_result: Dict[str, Any]):
        """Log progress during online learning."""
        if self.config.online_evaluation.verbose:
            # Full verbose output
            marker = " [TRAINED]" if latest_result['triggered_update'] else ""
            
            # Get timing info from metrics collector
            timing_stats = self.metrics_collector.timing_profiler.get_stats()
            total_time = timing_stats.get('total_sample', {}).get('mean', 0) * 1000
            
            # Enhanced progress display with quality metrics
            print(f"Sample {sample_num:4d} | Domain {latest_result['domain_id']} | "
                  f"NMSE: {latest_result['masked_nmse_pre']:.6f} → {latest_result['masked_nmse_post']:.6f}")
            
            # Add SSIM and PSNR if available
            if 'ssim' in latest_result:
                print(f"         SSIM: {latest_result['ssim']:.4f} | PSNR: {latest_result.get('psnr', 0):.2f}dB | "
                      f"Time: {total_time:.1f}ms{marker}")
            else:
                print(f"         Time: {total_time:.1f}ms{marker}")
            
            # Print recent statistics
            recent_results = self.results_history[-self.config.online_evaluation.log_frequency:]
            avg_nmse = np.mean([r['masked_nmse_post'] for r in recent_results])
            avg_ssim = np.mean([r.get('ssim', 0) for r in recent_results])
            num_updates = sum(1 for r in recent_results if r['triggered_update'])
            
            print(f"         Recent {len(recent_results)}: Avg NMSE={avg_nmse:.6f}, "
                  f"Avg SSIM={avg_ssim:.4f}, Updates={num_updates}")
        else:
            # Minimal output - just basic progress
            if latest_result['triggered_update']:
                epochs_info = f" [TRAINED {latest_result.get('epochs_run', '?')}ep]"
            else:
                epochs_info = ""
            print(f"Sample {sample_num:4d} | Domain {latest_result['domain_id']} | "
                  f"NMSE: {latest_result['masked_nmse_post']:.6f}{epochs_info}")
        
        # Periodically print detailed status and save metrics (verbose only)
        if self.config.online_evaluation.verbose and sample_num % (self.config.online_evaluation.log_frequency * 10) == 0:
            self.print_detailed_status()
            
            # Save intermediate results
            if sample_num % (self.config.online_evaluation.log_frequency * 50) == 0:
                intermediate_file = self.metrics_collector.save_results(f"intermediate_{sample_num}")
                print(f"          Intermediate results saved: {intermediate_file}")
    
    def print_detailed_status(self):
        """Print detailed status of the online learning system."""
        print("\n" + "=" * 60)
        print("DETAILED STATUS")
        print("=" * 60)
        
        # Trigger status
        self.trigger_manager.print_status()
        
        # Buffer status
        self.online_buffer_manager.print_status()
        
        # Training statistics
        print(f"\n[TRAINING] TRAINING STATISTICS:")
        print(f"   Total updates: {self.total_updates}")
        print(f"   Updates per domain:")
        for domain_id in sorted(self.domain_update_counts.keys()):
            count = self.domain_update_counts[domain_id]
            ema = self.ema_losses.get(domain_id, 0.0)
            print(f"     Domain {domain_id}: {count} updates, EMA loss: {ema:.6f}")
        
        # EWC statistics
        if self.model_manager.is_ewc_enabled():
            self._print_ewc_statistics()
        
        print("=" * 60 + "\n")
    
    def _print_ewc_statistics(self):
        """Print EWC penalty statistics."""
        print(f"\n[EWC] EWC PENALTY STATISTICS:")
        
        if not self.domain_ewc_ratios:
            print("   No EWC ratios recorded yet")
            return
        
        print(f"   Fixed λ = {self.config.online_training.ewc.lambda_ewc}")
        
        for domain_id in sorted(self.domain_ewc_ratios.keys()):
            ratios = self.domain_ewc_ratios[domain_id]
            if ratios:
                mean_ratio = np.mean(ratios)
                std_ratio = np.std(ratios)
                min_ratio = np.min(ratios)
                max_ratio = np.max(ratios)
                
                print(f"   Domain {domain_id}:")
                print(f"     EWC/MSE ratio: {mean_ratio:.3f} ± {std_ratio:.3f} (min: {min_ratio:.3f}, max: {max_ratio:.3f})")
                print(f"     Samples: {len(ratios)}")
                
                # Provide analysis (no recommendations for changes)
                if mean_ratio < 0.05:
                    print(f"      Analysis: EWC penalty is very low ({mean_ratio:.3f})")
                elif mean_ratio > 0.5:
                    print(f"      Analysis: EWC penalty is very high ({mean_ratio:.3f})")
                elif 0.1 <= mean_ratio <= 0.2:
                    print(f"      Analysis: EWC penalty ratio is in typical range ({mean_ratio:.3f})")
                else:
                    print(f"      Analysis: EWC penalty ratio is moderate ({mean_ratio:.3f})")
    
    def print_final_summary(self, total_time: float):
        """Print final summary of the online learning session."""
        print("\n" + "=" * 80)
        print("ONLINE CONTINUAL LEARNING COMPLETE")
        print("=" * 80)
        
        if not self.results_history:
            print("No samples processed.")
            return
        
        # Overall statistics
        all_nmse_pre = [r['masked_nmse_pre'] for r in self.results_history]
        all_nmse_post = [r['masked_nmse_post'] for r in self.results_history]
        all_inference_times = [r['inference_time'] for r in self.results_history]
        all_training_times = [r['training_time'] for r in self.results_history]
        
        print(f"\n[STATS] OVERALL STATISTICS ({len(self.results_history)} samples):")
        print(f"   Initial NMSE : Mean={np.mean(all_nmse_pre):.6f}, Std={np.std(all_nmse_pre):.6f}")
        print(f"   Final NMSE   : Mean={np.mean(all_nmse_post):.6f}, Std={np.std(all_nmse_post):.6f}")
        print(f"   Improvement  : {(np.mean(all_nmse_pre) - np.mean(all_nmse_post)):.6f}")
        print(f"   Inference    : Mean={np.mean(all_inference_times)*1000:.2f}ms")
        print(f"   Training     : Mean={np.mean(all_training_times)*1000:.2f}ms")
        
        # Proof-of-concept adaptation effectiveness
        print(f"\n[ADAPTATION] PROOF-OF-CONCEPT METRICS:")
        total_adaptations = self.adaptation_tracker['total_adaptations']
        successful_adaptations = self.adaptation_tracker['successful_adaptations']
        success_rate = (successful_adaptations / max(1, total_adaptations)) * 100
        
        print(f"   Total adaptation attempts: {total_adaptations}")
        print(f"   Successful adaptations: {successful_adaptations}")
        print(f"   Success rate: {success_rate:.1f}%")
        print(f"   Failed adaptations: {self.adaptation_tracker['failed_adaptations']}")
        
        if self.adaptation_tracker['adaptation_magnitudes']:
            avg_improvement = np.mean(self.adaptation_tracker['adaptation_magnitudes'])
            max_improvement = np.max(self.adaptation_tracker['adaptation_magnitudes'])
            print(f"   Average improvement: {avg_improvement:.6f} NMSE reduction")
            print(f"   Best single improvement: {max_improvement:.6f} NMSE reduction")
        
        # Domain specialization summary
        print(f"\n[SPECIALIZATION] DOMAIN LEARNING EFFECTIVENESS:")
        for domain_id, scores in self.adaptation_tracker['domain_specialization_scores'].items():
            if scores:
                avg_specialization = np.mean(scores)
                max_specialization = np.max(scores)
                baseline_nmse = self.adaptation_tracker['baseline_performance'].get(domain_id, 0)
                print(f"   Domain {domain_id}: {avg_specialization:.2f}% avg improvement from baseline")
                print(f"             Best: {max_specialization:.2f}%, Baseline: {baseline_nmse:.6f}")
        
        # Learning curve insights
        print(f"\n[LEARNING] ADAPTATION LEARNING CURVES:")
        for domain_id, history in self.adaptation_tracker['domain_adaptation_history'].items():
            if len(history) > 1:
                improvements = [h['improvement'] for h in history]
                positive_improvements = [i for i in improvements if i > 0]
                
                if positive_improvements:
                    trend = "" if improvements[-1] > improvements[0] else ""
                    print(f"   Domain {domain_id}: {len(history)} adaptations, "
                          f"avg improvement: {np.mean(positive_improvements):.6f} {trend}")
        
        # Training events
        num_trained = sum(1 for r in self.results_history if r['triggered_update'])
        print(f"\n[TRAINING] TRAINING EVENTS:")
        print(f"   Total triggers: {num_trained} / {len(self.results_history)} ({num_trained/len(self.results_history)*100:.1f}%)")
        print(f"   Trigger frequency: every {len(self.results_history)/max(1, num_trained):.1f} samples")
        
        # Validate our proof-of-concept
        if success_rate > 60:
            print(f"\n PROOF-OF-CONCEPT VALIDATION: SUCCESS!")
            print(f"   Online LoRA adaptation is working effectively ({success_rate:.1f}% success)")
        elif success_rate > 30:
            print(f"\n️  PROOF-OF-CONCEPT VALIDATION: PARTIALLY WORKING")
            print(f"   Online LoRA adaptation shows some benefit ({success_rate:.1f}% success)")
        else:
            print(f"\n PROOF-OF-CONCEPT VALIDATION: NEEDS INVESTIGATION")
            print(f"   Online LoRA adaptation success rate is low ({success_rate:.1f}%)")
        
        # Per-domain statistics
        print(f"\n[STATS] PER-DOMAIN STATISTICS:")
        for domain in sorted(self.config.raw_data.domains):
            domain_results = [r for r in self.results_history if r['domain_id'] == domain]
            if domain_results:
                domain_nmse_post = [r['masked_nmse_post'] for r in domain_results]
                domain_psnr = [r.get('psnr', float('nan')) for r in domain_results]
                domain_ssim = [r.get('ssim', float('nan')) for r in domain_results]
                domain_updates = sum(1 for r in domain_results if r['triggered_update'])
                
                # Filter out NaN values
                domain_psnr_clean = [x for x in domain_psnr if not np.isnan(x)]
                domain_ssim_clean = [x for x in domain_ssim if not np.isnan(x)]
                
                psnr_str = f"{np.mean(domain_psnr_clean):.2f}" if domain_psnr_clean else "N/A"
                ssim_str = f"{np.mean(domain_ssim_clean):.4f}" if domain_ssim_clean else "N/A"
                
                print(f"   Domain {domain}: {len(domain_results):3d} samples | "
                      f"NMSE: {np.mean(domain_nmse_post):.6f} ± {np.std(domain_nmse_post):.6f} | "
                      f"PSNR: {psnr_str}dB | SSIM: {ssim_str} | Updates: {domain_updates}")
        
        # Performance
        print(f"\n[PERFORMANCE] PERFORMANCE METRICS:")
        print(f"   Total runtime: {total_time:.2f} seconds")
        print(f"   Throughput: {len(self.results_history)/total_time:.2f} samples/second")
        
        # Final status
        self.print_detailed_status()
        
        # Generate and save comprehensive metrics report
        print(f"\n[METRICS] COMPREHENSIVE METRICS ANALYSIS:")
        comprehensive_report = self.metrics_collector.get_comprehensive_report()
        
        # Print key insights from comprehensive analysis
        if 'timing_stats' in comprehensive_report:
            timing_stats = comprehensive_report['timing_stats']
            print(f"   Timing breakdown:")
            for operation, stats in timing_stats.items():
                print(f"     {operation}: {stats['mean']*1000:.2f}ms avg ({stats['count']} samples)")
        
        if 'domain_performance' in comprehensive_report:
            domain_perf = comprehensive_report['domain_performance']
            print(f"   Domain performance trends:")
            for domain_id, metrics in domain_perf.items():
                if 'nmse' in metrics:
                    trend = "↗" if metrics['nmse']['trend'] > 0.001 else "↘" if metrics['nmse']['trend'] < -0.001 else "→"
                    print(f"     Domain {domain_id}: NMSE {metrics['nmse']['mean']:.6f} {trend}")
        
        if 'forgetting_analysis' in comprehensive_report:
            forgetting = comprehensive_report['forgetting_analysis']
            print(f"   Forgetting analysis:")
            for domain_id, analysis in forgetting.items():
                forgetting_score = analysis['forgetting_score']
                status = "stable" if abs(forgetting_score) < 0.01 else "degraded" if forgetting_score > 0 else "improved"
                print(f"     Domain {domain_id}: {status} (score: {forgetting_score:.6f})")
        
        if 'system_health' in comprehensive_report:
            health = comprehensive_report['system_health']
            print(f"   System health: {health['status']}")
            if health['warnings']:
                for warning in health['warnings']:
                    print(f"     ️  {warning}")
            if health['errors']:
                for error in health['errors']:
                    print(f"      {error}")
        
        # Generate transfer metrics report
        if self.transfer_tracker.domain_snapshots:
            self.transfer_tracker.print_transfer_report()
        
        # Save comprehensive results
        report_file = self.metrics_collector.save_results("online_continual_learning")
        print(f"\n    Comprehensive metrics saved to: {report_file}")
        
        # Save results
        if self.config.online_metrics.save_detailed_results:
            self.save_results()
    
    def save_results(self):
        """Save detailed results to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save to eval/ directory
        eval_dir = os.path.join("main_algorithm_v2", "online", "eval", self.config.experiment_name)
        os.makedirs(eval_dir, exist_ok=True)
        filename = os.path.join(eval_dir, f"online_continual_learning_results_{timestamp}.csv")
        
        print(f"\n[SAVE] Saving results to: {filename}")
        
        with open(filename, 'w') as f:
            # Header
            f.write("sample_id,domain_id,nmse_pre,nmse_post,improvement,inference_ms,training_ms,triggered,epochs_run,ema_loss\n")
            
            # Data
            for r in self.results_history:
                epochs_run = r.get('epochs_run', 0) if r['triggered_update'] else 0
                f.write(f"{r['sample_id']},{r['domain_id']},{r['masked_nmse_pre']:.6f},"
                       f"{r['masked_nmse_post']:.6f},{r.get('improvement', 0.0):.6f},"
                       f"{r['inference_time']*1000:.2f},{r['training_time']*1000:.2f},"
                       f"{int(r['triggered_update'])},{epochs_run},{r['ema_loss']:.6f}\n")


def main():
    """Entry point for online continual learning."""
    parser = argparse.ArgumentParser(description="Run online continual learning")
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help="Path to the online configuration YAML file"
    )
    parser.add_argument(
        '--num_samples',
        type=int,
        default=None,
        help="Number of samples to process (overrides config)"
    )
    args = parser.parse_args()
    
    # Create online continual learner
    learner = OnlineContinualLearner(args.config)
    
    # Get number of samples
    num_samples = args.num_samples or learner.config.online_evaluation.num_evaluations
    
    # Run online learning
    learner.run(num_samples)
    
    print("\n[SUCCESS] Online continual learning completed!")


if __name__ == '__main__':
    main() 