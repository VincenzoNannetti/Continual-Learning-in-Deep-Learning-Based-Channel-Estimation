"""
Forward and Backward Transfer Metrics for Continual Learning Evaluation.

This module implements standard continual learning metrics:
- Forward Transfer (FWT): How much learning new tasks helps with future tasks
- Backward Transfer (BWT): How much learning new tasks hurts previous tasks  
- Average Accuracy over Time (AAoT): Overall performance across all tasks over time

References:
- Lopez-Paz & Ranzato (2017): "Gradient Episodic Memory for Continual Learning"
- Chaudhry et al. (2018): "Efficient Lifelong Learning with A-GEM"
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Any
import time
from dataclasses import dataclass


@dataclass
class TransferSnapshot:
    """A snapshot of performance across all domains at a specific time."""
    timestamp: float
    domain_performances: Dict[str, float]  # domain_id -> NMSE
    current_training_domain: Optional[str]
    sample_count: int


class TransferMetricsTracker:
    """Track Forward and Backward Transfer metrics during online continual learning."""
    
    def __init__(self, domains: List[str], evaluation_frequency: int = 100):
        """
        Initialize transfer metrics tracker.
        
        Args:
            domains: List of domain identifiers
            evaluation_frequency: How often to evaluate all domains (in samples)
        """
        self.domains = [str(d) for d in domains]
        self.evaluation_frequency = evaluation_frequency
        
        # Performance tracking
        self.domain_snapshots: List[TransferSnapshot] = []
        self.baseline_performances: Dict[str, float] = {}  # Initial performance per domain
        self.domain_first_encounter: Dict[str, int] = {}  # When each domain was first seen
        
        # Internal state
        self.sample_count = 0
        self.last_evaluation_sample = 0
        
    def should_evaluate_transfer(self) -> bool:
        """Check if it's time to evaluate transfer metrics."""
        return (self.sample_count - self.last_evaluation_sample) >= self.evaluation_frequency
    
    def evaluate_all_domains(self, model_manager, data_pipeline, device, 
                           current_domain: str, num_eval_samples: int = 50) -> Dict[str, float]:
        """
        Evaluate the model on all domains without training.
        
        Args:
            model_manager: The model manager instance
            data_pipeline: The data pipeline instance  
            device: PyTorch device
            current_domain: Currently active domain
            num_eval_samples: Number of samples to evaluate per domain
            
        Returns:
            Dictionary mapping domain_id -> average NMSE
        """
        from main_algorithm_v2.online.src.loss_functions import masked_nmse, extract_pilot_ground_truth
        
        model_manager.model.eval()
        domain_performances = {}
        
        original_domain = current_domain
        
        with torch.no_grad():
            for domain_id in self.domains:
                domain_id_int = int(domain_id)
                
                # Switch to this domain
                model_manager.set_active_domain(domain_id_int)
                
                nmse_values = []
                for _ in range(num_eval_samples):
                    try:
                        # Get a sample from this domain
                        model_input, ground_truth, pilot_mask, metadata = data_pipeline.process_sample_online(domain_id_int)
                        
                        # Move to device
                        model_input = model_input.to(device)
                        ground_truth = ground_truth.to(device)
                        pilot_mask = pilot_mask.to(device)
                        
                        # Extract pilot ground truth
                        pilot_ground_truth = extract_pilot_ground_truth(ground_truth, pilot_mask)
                        
                        # Forward pass
                        prediction = model_manager.model(model_input.unsqueeze(0))
                        prediction = prediction.squeeze(0)
                        
                        # Calculate NMSE
                        nmse_value = masked_nmse(prediction, pilot_ground_truth, pilot_mask)
                        nmse_values.append(nmse_value)
                        
                    except Exception as e:
                        print(f"Warning: Failed to evaluate domain {domain_id}: {e}")
                        continue
                
                if nmse_values:
                    domain_performances[domain_id] = np.mean(nmse_values)
                else:
                    domain_performances[domain_id] = float('nan')
        
        # Restore original domain
        model_manager.set_active_domain(int(original_domain))
        
        return domain_performances
    
    def record_transfer_snapshot(self, model_manager, data_pipeline, device, 
                               current_domain: str) -> TransferSnapshot:
        """Record a transfer metrics snapshot."""
        if not self.should_evaluate_transfer():
            return None
            
        # Evaluate all domains
        domain_performances = self.evaluate_all_domains(
            model_manager, data_pipeline, device, current_domain
        )
        
        # Create snapshot
        snapshot = TransferSnapshot(
            timestamp=time.time(),
            domain_performances=domain_performances,
            current_training_domain=current_domain,
            sample_count=self.sample_count
        )
        
        self.domain_snapshots.append(snapshot)
        self.last_evaluation_sample = self.sample_count
        
        # Record baselines on first encounter
        for domain_id, performance in domain_performances.items():
            if domain_id not in self.baseline_performances and not np.isnan(performance):
                self.baseline_performances[domain_id] = performance
                self.domain_first_encounter[domain_id] = self.sample_count
        
        return snapshot
    
    def increment_sample_count(self):
        """Increment the sample counter."""
        self.sample_count += 1
    
    def compute_forward_transfer(self) -> Dict[str, float]:
        """
        Compute Forward Transfer (FWT) for each domain.
        
        FWT_i = performance_at_first_encounter_i - random_baseline_i
        
        Returns:
            Dictionary mapping domain_id -> FWT score
        """
        fwt_scores = {}
        
        for domain_id in self.domains:
            if domain_id not in self.baseline_performances:
                fwt_scores[domain_id] = float('nan')
                continue
            
            # Find first encounter performance
            first_encounter_sample = self.domain_first_encounter.get(domain_id, 0)
            
            # Find snapshot closest to first encounter
            first_snapshot = None
            for snapshot in self.domain_snapshots:
                if snapshot.sample_count >= first_encounter_sample:
                    first_snapshot = snapshot
                    break
            
            if first_snapshot and domain_id in first_snapshot.domain_performances:
                first_performance = first_snapshot.domain_performances[domain_id]
                baseline_performance = self.baseline_performances[domain_id]
                
                # FWT = baseline - first_encounter (negative is better for NMSE)
                fwt_scores[domain_id] = baseline_performance - first_performance
            else:
                fwt_scores[domain_id] = float('nan')
        
        return fwt_scores
    
    def compute_backward_transfer(self) -> Dict[str, float]:
        """
        Compute Backward Transfer (BWT) for each domain.
        
        BWT_i = final_performance_i - performance_after_training_i
        
        Returns:
            Dictionary mapping domain_id -> BWT score
        """
        bwt_scores = {}
        
        if len(self.domain_snapshots) < 2:
            return {domain_id: float('nan') for domain_id in self.domains}
        
        final_snapshot = self.domain_snapshots[-1]
        
        for domain_id in self.domains:
            if domain_id not in final_snapshot.domain_performances:
                bwt_scores[domain_id] = float('nan')
                continue
            
            final_performance = final_snapshot.domain_performances[domain_id]
            
            # Find the performance right after this domain was last trained
            after_training_performance = None
            for snapshot in reversed(self.domain_snapshots[:-1]):
                if (snapshot.current_training_domain == domain_id and 
                    domain_id in snapshot.domain_performances):
                    after_training_performance = snapshot.domain_performances[domain_id]
                    break
            
            if after_training_performance is not None:
                # BWT = after_training - final (negative is worse for NMSE)
                bwt_scores[domain_id] = after_training_performance - final_performance
            else:
                bwt_scores[domain_id] = float('nan')
        
        return bwt_scores
    
    def compute_average_accuracy_over_time(self) -> float:
        """
        Compute Average Accuracy over Time (AAoT).
        
        AAoT = average of all domain performances across all snapshots
        
        Returns:
            AAoT score (lower is better for NMSE)
        """
        if not self.domain_snapshots:
            return float('nan')
        
        all_performances = []
        for snapshot in self.domain_snapshots:
            for domain_id, performance in snapshot.domain_performances.items():
                if not np.isnan(performance):
                    all_performances.append(performance)
        
        return np.mean(all_performances) if all_performances else float('nan')
    
    def compute_area_under_curve(self) -> Dict[str, float]:
        """
        Compute Area Under Curve (AUC) of NMSE vs time for each domain.
        
        Returns:
            Dictionary mapping domain_id -> AUC score
        """
        auc_scores = {}
        
        for domain_id in self.domains:
            performances = []
            timestamps = []
            
            for snapshot in self.domain_snapshots:
                if domain_id in snapshot.domain_performances:
                    performance = snapshot.domain_performances[domain_id]
                    if not np.isnan(performance):
                        performances.append(performance)
                        timestamps.append(snapshot.sample_count)
            
            if len(performances) > 1:
                # Compute AUC using trapezoidal rule
                auc = np.trapz(performances, timestamps)
                # Normalize by time span
                time_span = timestamps[-1] - timestamps[0]
                auc_scores[domain_id] = auc / time_span if time_span > 0 else float('nan')
            else:
                auc_scores[domain_id] = float('nan')
        
        return auc_scores
    
    def get_transfer_summary(self) -> Dict[str, Any]:
        """Get comprehensive transfer metrics summary."""
        fwt_scores = self.compute_forward_transfer()
        bwt_scores = self.compute_backward_transfer()
        aaot_score = self.compute_average_accuracy_over_time()
        auc_scores = self.compute_area_under_curve()
        
        # Filter out NaN values for aggregation
        fwt_clean = [v for v in fwt_scores.values() if not np.isnan(v)]
        bwt_clean = [v for v in bwt_scores.values() if not np.isnan(v)]
        auc_clean = [v for v in auc_scores.values() if not np.isnan(v)]
        
        summary = {
            'forward_transfer': {
                'per_domain': fwt_scores,
                'average': np.mean(fwt_clean) if fwt_clean else float('nan'),
                'std': np.std(fwt_clean) if fwt_clean else float('nan')
            },
            'backward_transfer': {
                'per_domain': bwt_scores,
                'average': np.mean(bwt_clean) if bwt_clean else float('nan'),
                'std': np.std(bwt_clean) if bwt_clean else float('nan')
            },
            'average_accuracy_over_time': aaot_score,
            'area_under_curve': {
                'per_domain': auc_scores,
                'average': np.mean(auc_clean) if auc_clean else float('nan'),
                'std': np.std(auc_clean) if auc_clean else float('nan')
            },
            'evaluation_snapshots': len(self.domain_snapshots),
            'domains_evaluated': len([d for d in self.domains if d in self.baseline_performances])
        }
        
        return summary
    
    def print_transfer_report(self):
        """Print a formatted transfer metrics report."""
        summary = self.get_transfer_summary()
        
        print("\n" + "=" * 80)
        print("TRANSFER METRICS ANALYSIS")
        print("=" * 80)
        
        # Forward Transfer
        print(f"\n[FWT] FORWARD TRANSFER (how much learning helps future tasks):")
        fwt = summary['forward_transfer']
        if not np.isnan(fwt['average']):
            print(f"   Average FWT: {fwt['average']:.6f} ± {fwt['std']:.6f}")
            print(f"   Per-domain:")
            for domain_id, score in fwt['per_domain'].items():
                if not np.isnan(score):
                    status = "positive" if score > 0 else "negative" if score < 0 else "neutral"
                    print(f"     Domain {domain_id}: {score:.6f} ({status})")
        else:
            print("   No FWT data available")
        
        # Backward Transfer  
        print(f"\n[BWT] BACKWARD TRANSFER (how much new learning hurts old tasks):")
        bwt = summary['backward_transfer']
        if not np.isnan(bwt['average']):
            print(f"   Average BWT: {bwt['average']:.6f} ± {bwt['std']:.6f}")
            print(f"   Per-domain:")
            for domain_id, score in bwt['per_domain'].items():
                if not np.isnan(score):
                    status = "forgetting" if score < 0 else "improvement" if score > 0 else "stable"
                    print(f"     Domain {domain_id}: {score:.6f} ({status})")
        else:
            print("   No BWT data available")
        
        # AAoT
        aaot = summary['average_accuracy_over_time']
        if not np.isnan(aaot):
            print(f"\n[AAoT] AVERAGE ACCURACY OVER TIME: {aaot:.6f}")
        
        # AUC
        print(f"\n[AUC] AREA UNDER CURVE:")
        auc = summary['area_under_curve']
        if not np.isnan(auc['average']):
            print(f"   Average AUC: {auc['average']:.6f} ± {auc['std']:.6f}")
        else:
            print("   No AUC data available")
        
        print(f"\n[INFO] Transfer metrics based on {summary['evaluation_snapshots']} snapshots")
        print(f"       Domains evaluated: {summary['domains_evaluated']}/{len(self.domains)}")
        print("=" * 80) 