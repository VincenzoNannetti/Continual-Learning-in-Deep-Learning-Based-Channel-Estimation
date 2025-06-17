"""
Mixed Batch Manager for Online Continual Learning

This module creates mixed batches combining:
1. Current sample (streaming)
2. Offline buffer samples (full labels)
3. Online buffer samples (pilot labels)

The ratio between offline and online samples is dynamically adjusted based on
similarity between the distributions.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, List, Optional, Any
import random


class MixedBatchManager:
    """
    Creates mixed batches for online continual learning with adaptive weighting.
    """
    
    def __init__(self, batch_config: Dict[str, Any]):
        """
        Initialize mixed batch manager.
        
        Args:
            batch_config: Configuration dict with keys:
                - m_offline: int (offline samples per batch)
                - m_online: int (online samples per batch)
                - adaptive_weighting: bool (enable adaptive offline/online ratio)
                - similarity_threshold: float (threshold for similarity comparison)
                - max_offline_ratio: float (max fraction of offline samples)
                - min_offline_ratio: float (min fraction of offline samples)
        """
        self.m_offline = batch_config.get('m_offline', 8)
        self.m_online = batch_config.get('m_online', 8)
        self.adaptive_weighting = batch_config.get('adaptive_weighting', True)
        self.similarity_threshold = batch_config.get('similarity_threshold', 0.7)
        self.max_offline_ratio = batch_config.get('max_offline_ratio', 0.8)
        self.min_offline_ratio = batch_config.get('min_offline_ratio', 0.2)
        
        # Statistics
        self.batches_created = 0
        self.total_offline_samples = 0
        self.total_online_samples = 0
        self.similarity_history = []
        
        print(f"🔀 MixedBatchManager initialized:")
        print(f"   Offline samples per batch: {self.m_offline}")
        print(f"   Online samples per batch: {self.m_online}")
        print(f"   Adaptive weighting: {self.adaptive_weighting}")
        if self.adaptive_weighting:
            print(f"   Similarity threshold: {self.similarity_threshold}")
            print(f"   Offline ratio range: [{self.min_offline_ratio:.1%}, {self.max_offline_ratio:.1%}]")
    
    def compute_distribution_similarity(self, offline_samples: List[Tuple[torch.Tensor, torch.Tensor]], 
                                       online_samples: List[Tuple[torch.Tensor, torch.Tensor]]) -> float:
        """
        Compute similarity between offline and online sample distributions.
        
        Args:
            offline_samples: List of (input, target) tuples from offline buffer
            online_samples: List of (input, target) tuples from online buffer
            
        Returns:
            float: Similarity score between 0 and 1 (1 = very similar)
        """
        if not offline_samples or not online_samples:
            return 0.0
        
        # Strategy 1: Compare input statistics (mean, std, etc.)
        # Extract inputs and compute statistics
        offline_inputs = torch.stack([sample[0] for sample in offline_samples])
        online_inputs = torch.stack([sample[0] for sample in online_samples])
        
        # Flatten for easier computation
        offline_flat = offline_inputs.flatten(start_dim=1)  # Keep batch dimension
        online_flat = online_inputs.flatten(start_dim=1)
        
        # Compute statistics - use mean across all data
        offline_global_mean = offline_flat.mean()
        offline_global_std = offline_flat.std()
        online_global_mean = online_flat.mean()
        online_global_std = online_flat.std()
        
        # Create feature vectors from global statistics
        offline_features = torch.tensor([offline_global_mean, offline_global_std])
        online_features = torch.tensor([online_global_mean, online_global_std])
        
        # Compute cosine similarity between feature vectors
        similarity = F.cosine_similarity(offline_features.unsqueeze(0), 
                                       online_features.unsqueeze(0)).item()
        
        # Ensure similarity is between 0 and 1 (cosine similarity is in [-1, 1])
        similarity = max(0.0, min(1.0, (similarity + 1.0) / 2.0))
        
        return similarity
    
    def compute_adaptive_ratios(self, offline_samples: List[Tuple[torch.Tensor, torch.Tensor]], 
                               online_samples: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[float, float]:
        """
        Compute adaptive ratios for offline and online samples based on similarity.
        
        Args:
            offline_samples: Offline buffer samples
            online_samples: Online buffer samples
            
        Returns:
            Tuple of (offline_ratio, online_ratio) that sum to 1.0
        """
        if not self.adaptive_weighting:
            # Fixed 50/50 split if adaptive weighting is disabled
            return 0.5, 0.5
        
        # Compute similarity between distributions
        similarity = self.compute_distribution_similarity(offline_samples, online_samples)
        self.similarity_history.append(similarity)
        
        # Adaptive logic:
        # - High similarity: Use balanced approach (50/50)
        # - Low similarity: Prioritise online samples (they represent new distribution)
        if similarity >= self.similarity_threshold:
            # Similar distributions - use balanced approach
            offline_ratio = 0.5
        else:
            # Different distributions - prioritise online samples
            # Map similarity to offline ratio: lower similarity = lower offline ratio
            offline_ratio = self.min_offline_ratio + (similarity / self.similarity_threshold) * (0.5 - self.min_offline_ratio)
        
        # Ensure ratios are within bounds
        offline_ratio = max(self.min_offline_ratio, min(self.max_offline_ratio, offline_ratio))
        online_ratio = 1.0 - offline_ratio
        
        return offline_ratio, online_ratio
    
    def create_mixed_batch(self, current_sample: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                          offline_buffer, online_buffer, 
                          device: torch.device) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """
        Create a mixed batch combining current sample, offline buffer, and online buffer samples.
        
        Args:
            current_sample: Tuple of (input, target, pilot_mask) for current streaming sample
            offline_buffer: Offline replay buffer (with full labels)
            online_buffer: Online replay buffer (with pilot labels)
            device: Device to move tensors to
            
        Returns:
            Tuple of (inputs, targets, masks) as lists of tensors
        """
        current_input, current_target, current_mask = current_sample
        
        # Lists to store batch components
        batch_inputs = []
        batch_targets = []
        batch_masks = []
        
        # Add current sample
        batch_inputs.append(current_input.to(device))
        batch_targets.append(current_target.to(device))
        batch_masks.append(current_mask.to(device))
        
        # Get samples from buffers
        offline_samples = offline_buffer.get_random_batch(self.m_offline) if offline_buffer else []
        online_samples = online_buffer.get_random_batch(self.m_online) if online_buffer else []
        
        # Compute adaptive ratios
        offline_ratio, online_ratio = self.compute_adaptive_ratios(offline_samples, online_samples)
        
        # Calculate actual number of samples to use
        total_buffer_samples = len(offline_samples) + len(online_samples)
        if total_buffer_samples > 0:
            n_offline = int(offline_ratio * total_buffer_samples)
            n_online = total_buffer_samples - n_offline
            
            # Adjust if we don't have enough samples
            n_offline = min(n_offline, len(offline_samples))
            n_online = min(n_online, len(online_samples))
        else:
            n_offline = n_online = 0
        
        # Add offline samples (full labels, mask = all ones)
        if n_offline > 0:
            selected_offline = random.sample(offline_samples, n_offline)
            for off_input, off_target in selected_offline:
                batch_inputs.append(off_input.to(device))
                batch_targets.append(off_target.to(device))
                # Offline samples have full labels (mask = all ones)
                # Create mask with shape (72, 70) - same as current_mask
                full_mask = torch.ones((72, 70), dtype=torch.bool).to(device)
                batch_masks.append(full_mask)
            
            self.total_offline_samples += n_offline
        
        # Add online samples (pilot labels only)
        if n_online > 0:
            selected_online = random.sample(online_samples, n_online)
            for on_input, on_target in selected_online:
                batch_inputs.append(on_input.to(device))
                batch_targets.append(on_target.to(device))
                # Online samples need their pilot masks - we'll use the current mask as approximation
                # In practice, you might want to store masks with online samples
                batch_masks.append(current_mask.to(device))
            
            self.total_online_samples += n_online
        
        self.batches_created += 1
        
        return batch_inputs, batch_targets, batch_masks
    
    def create_mixed_batch_tensors(self, current_sample: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                                  offline_buffer, online_buffer, 
                                  device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Create mixed batch and return as stacked tensors (compatible with algorithm).
        
        Args:
            current_sample: Current streaming sample (input, target, pilot_mask)
            offline_buffer: Offline replay buffer
            online_buffer: Online replay buffer
            device: Device to move tensors to
            
        Returns:
            Tuple of (inputs_tensor, targets_tensor, masks_tensor) as stacked tensors
        """
        batch_inputs, batch_targets, batch_masks = self.create_mixed_batch(
            current_sample, offline_buffer, online_buffer, device
        )
        
        # Stack tensors
        inputs_tensor = torch.stack(batch_inputs, dim=0)
        targets_tensor = torch.stack(batch_targets, dim=0)
        masks_tensor = torch.stack(batch_masks, dim=0)
        
        return inputs_tensor, targets_tensor, masks_tensor
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics about batch creation."""
        stats = {
            'batches_created': self.batches_created,
            'total_offline_samples': self.total_offline_samples,
            'total_online_samples': self.total_online_samples,
            'config': {
                'm_offline': self.m_offline,
                'm_online': self.m_online,
                'adaptive_weighting': self.adaptive_weighting,
                'similarity_threshold': self.similarity_threshold
            }
        }
        
        if self.batches_created > 0:
            stats['avg_offline_per_batch'] = self.total_offline_samples / self.batches_created
            stats['avg_online_per_batch'] = self.total_online_samples / self.batches_created
            
        if self.similarity_history:
            stats['similarity_stats'] = {
                'avg_similarity': np.mean(self.similarity_history),
                'min_similarity': np.min(self.similarity_history),
                'max_similarity': np.max(self.similarity_history),
                'recent_similarity': self.similarity_history[-1]
            }
            
        return stats
    
    def print_status(self):
        """Print current status of mixed batch creation."""
        stats = self.get_stats()
        print(f"\n🔀 MIXED BATCH MANAGER STATUS:")
        print("-" * 45)
        print(f"   Batches created: {stats['batches_created']}")
        print(f"   Total samples used: {stats['total_offline_samples'] + stats['total_online_samples']}")
        print(f"   Offline: {stats['total_offline_samples']} | Online: {stats['total_online_samples']}")
        
        if stats['batches_created'] > 0:
            print(f"   Avg per batch: {stats['avg_offline_per_batch']:.1f} offline, "
                  f"{stats['avg_online_per_batch']:.1f} online")
            
        if 'similarity_stats' in stats:
            sim_stats = stats['similarity_stats']
            print(f"   Similarity: avg={sim_stats['avg_similarity']:.3f}, "
                  f"recent={sim_stats['recent_similarity']:.3f}")
            
        print(f"   Adaptive weighting: {self.adaptive_weighting}") 