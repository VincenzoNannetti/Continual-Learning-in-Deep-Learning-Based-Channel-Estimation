"""
Gradient Diversity Buffer Manager for Online Continual Learning

This module implements gradient-based sample selection for online replay buffers.
Samples are selected based on gradient diversity to ensure each buffer sample
provides unique learning signals to the LoRA adapters.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple, List
from collections import deque
import time


class GradientDiversityBuffer:
    """
    Online replay buffer that uses gradient diversity for sample selection.
    
    Each sample is associated with its gradient w.r.t. LoRA parameters.
    New samples are only added if they provide sufficiently diverse gradients
    compared to existing buffer samples.
    """
    
    def __init__(self, buffer_size: int, diversity_threshold: float = 0.8, 
                 min_samples_for_diversity: int = 5, verbose: bool = False):
        """
        Initialize gradient diversity buffer.
        
        Args:
            buffer_size: Maximum number of samples to store
            diversity_threshold: Cosine similarity threshold (0-1, higher = more selective)
            min_samples_for_diversity: Minimum samples before applying diversity check
            verbose: Enable verbose debug output
        """
        self.buffer_size = buffer_size
        self.diversity_threshold = diversity_threshold
        self.min_samples_for_diversity = min_samples_for_diversity
        self.verbose = verbose
        
        # Storage for samples and their gradients
        self.samples = deque(maxlen=buffer_size)
        self.gradients = deque(maxlen=buffer_size)  # Flattened gradient vectors
        self.losses = deque(maxlen=buffer_size)     # Sample losses
        
        # Statistics
        self.creation_time = time.time()
        self.total_samples_seen = 0
        self.samples_added = 0
        self.samples_rejected = 0
        
    def compute_sample_gradient(self, model, sample_input: torch.Tensor, 
                              target: torch.Tensor, pilot_mask: torch.Tensor, 
                              domain_id: int) -> torch.Tensor:
        """
        Compute gradient of masked loss w.r.t. LoRA parameters for a sample.
        
        Args:
            model: Model manager with LoRA adapters
            sample_input: Input tensor to recompute prediction with gradients enabled
            target: Target tensor (pilot-only ground truth)
            pilot_mask: Boolean mask for pilot positions
            domain_id: Current domain ID
            
        Returns:
            Flattened gradient tensor
        """
        # Ensure tensors are on correct device
        if hasattr(model, 'device'):
            sample_input = sample_input.to(model.device)
            target = target.to(model.device)
            pilot_mask = pilot_mask.to(model.device)
        
        # Get LoRA parameters for this domain
        lora_params = self._get_lora_parameters(model, domain_id)
        
        # Enable gradients for LoRA parameters only
        for param in lora_params.values():
            param.requires_grad_(True)
        
        # Zero gradients
        model.model.zero_grad()
        
        # Set model to training mode temporarily for gradient computation
        model.model.train()
        
        # Recompute prediction with gradients enabled
        # Add batch dimension if needed (model expects 4D: batch, channels, height, width)
        if sample_input.dim() == 3:
            sample_input_batch = sample_input.unsqueeze(0)  # Add batch dimension
        else:
            sample_input_batch = sample_input
            
        with torch.enable_grad():
            prediction = model.model(sample_input_batch)
            
        # Remove batch dimension from prediction if it was added
        if sample_input.dim() == 3 and prediction.dim() == 4:
            prediction = prediction.squeeze(0)
        
        # Compute masked loss as a tensor (not as a float)
        from .loss_functions import masked_nmse_tensor
        loss_tensor = masked_nmse_tensor(prediction, target, pilot_mask)
        
        # Backward pass to get gradients
        loss_tensor.backward()
        
        # Set model back to eval mode
        model.model.eval()
        
        # Collect gradients from domain-specific LoRA parameters
        gradients = []
        gradient_info = []
        
        for name, param in lora_params.items():
            if param.grad is not None:
                grad_flat = param.grad.flatten()
                gradients.append(grad_flat)
                gradient_info.append(f"{name}: {grad_flat.shape[0]} elements")
            else:
                gradient_info.append(f"{name}: No gradient computed")
        
        # Debug information (only print occasionally to avoid spam)
        if len(gradients) > 0 and self.verbose:
            total_grad_elements = sum(g.shape[0] for g in gradients)
            # Only print debug info for first few samples
            if hasattr(self, '_debug_count'):
                self._debug_count += 1
            else:
                self._debug_count = 1
                
            if self._debug_count <= 3:  # Only print for first 3 samples
                print(f"Domain {domain_id} gradient computation:")
                for info in gradient_info:
                    print(f"  {info}")
                print(f"  Total gradient elements: {total_grad_elements}")
        
        # Flatten all gradients into single vector
        if gradients:
            gradient_vector = torch.cat(gradients)
        else:
            raise RuntimeError(f"No gradients computed for domain {domain_id} LoRA parameters. "
                             f"Found {len(lora_params)} parameters but none had gradients. "
                             f"Check if loss computation and backward pass are working correctly.")
        
        # Clean up
        model.model.zero_grad()
        for param in lora_params.values():
            param.requires_grad_(False)
        
        return gradient_vector.detach().cpu()
    
    def _get_lora_parameters(self, model, domain_id: int) -> Dict[str, torch.Tensor]:
        """
        Extract LoRA parameters ONLY for the current domain by traversing the model 
        and finding all LoRAConv2d instances, then getting their task-specific adapters.
        
        Args:
            model: Model manager with loaded model
            domain_id: Domain ID
            
        Returns:
            Dictionary of domain-specific LoRA parameter tensors
            
        Raises:
            ValueError: If no LoRA parameters found for the domain
        """
        lora_params = {}
        domain_str = str(domain_id)
        
        # Get the actual model (handle different model wrapper structures)
        actual_model = getattr(model, 'model', model)
        
        # Traverse the model to find all LoRAConv2d instances
        def find_lora_layers(module, prefix=""):
            """Recursively find all LoRAConv2d instances in the model."""
            for name, child in module.named_children():
                current_path = f"{prefix}.{name}" if prefix else name
                
                # Check if this is a LoRAConv2d layer
                if hasattr(child, 'task_adapters') and hasattr(child, 'lora_scaling'):
                    # This is a LoRAConv2d instance
                    if domain_str in child.task_adapters:
                        adapter = child.task_adapters[domain_str]
                        # Add the A and B parameters for this domain
                        lora_params[f"{current_path}.task_adapters.{domain_str}.A"] = adapter.A
                        lora_params[f"{current_path}.task_adapters.{domain_str}.B"] = adapter.B
                
                # Check if this is a Sequential containing LoRAConv2d
                elif isinstance(child, torch.nn.Sequential):
                    for idx, seq_child in enumerate(child):
                        if hasattr(seq_child, 'task_adapters') and hasattr(seq_child, 'lora_scaling'):
                            if domain_str in seq_child.task_adapters:
                                adapter = seq_child.task_adapters[domain_str]
                                lora_params[f"{current_path}.{idx}.task_adapters.{domain_str}.A"] = adapter.A
                                lora_params[f"{current_path}.{idx}.task_adapters.{domain_str}.B"] = adapter.B
                
                # Recurse into child modules
                find_lora_layers(child, current_path)
        
        # Find all LoRA parameters for this domain
        find_lora_layers(actual_model)
        
        if not lora_params:
            raise ValueError(f"No LoRA parameters found for domain {domain_id}. "
                           f"Make sure the model has been properly loaded and the task adapter "
                           f"for domain {domain_id} has been added to the model.")
        
        if self.verbose:
            print(f"Found {len(lora_params)} LoRA parameters for domain {domain_id}")
        return lora_params
    
    def compute_gradient_similarity(self, gradient1: torch.Tensor, 
                                   gradient2: torch.Tensor) -> float:
        """
        Compute cosine similarity between two gradient vectors.
        
        Args:
            gradient1: First gradient vector
            gradient2: Second gradient vector
            
        Returns:
            Cosine similarity [-1, 1]
        """
        # Ensure both gradients are 1D
        grad1_flat = gradient1.flatten()
        grad2_flat = gradient2.flatten()
        
        # Handle different sizes (shouldn't happen but safety check)
        min_size = min(len(grad1_flat), len(grad2_flat))
        grad1_flat = grad1_flat[:min_size]
        grad2_flat = grad2_flat[:min_size]
        
        # Compute cosine similarity
        similarity = F.cosine_similarity(grad1_flat.unsqueeze(0), 
                                       grad2_flat.unsqueeze(0))
        return similarity.item()
    
    def should_add_sample(self, new_gradient: torch.Tensor, 
                         new_loss: float) -> Tuple[bool, str]:
        """
        Decide whether to add a new sample based on gradient diversity.
        
        Args:
            new_gradient: Gradient vector for the new sample
            new_loss: Loss value for the new sample
            
        Returns:
            Tuple of (should_add: bool, reason: str)
        """
        # Always add if buffer not full
        if len(self.samples) < self.buffer_size:
            return True, "buffer_not_full"
        
        # Always add if we don't have enough samples for diversity check
        if len(self.gradients) < self.min_samples_for_diversity:
            return True, "insufficient_samples_for_diversity"
        
        # Compute similarities with existing gradients
        similarities = []
        for existing_gradient in self.gradients:
            similarity = self.compute_gradient_similarity(new_gradient, existing_gradient)
            similarities.append(similarity)
        
        max_similarity = max(similarities)
        
        # Add if sufficiently different (below threshold)
        if max_similarity < self.diversity_threshold:
            return True, f"diverse_gradient_max_sim_{max_similarity:.3f}"
        
        # Check if new sample has higher loss than least diverse existing sample
        # (Replace similar sample if new one is harder)
        if max_similarity >= self.diversity_threshold:
            most_similar_idx = similarities.index(max_similarity)
            existing_loss = self.losses[most_similar_idx]
            
            if new_loss > existing_loss * 1.1:  # 10% higher loss threshold
                # Replace the most similar sample
                self._replace_sample(most_similar_idx, new_gradient, new_loss)
                return False, f"replaced_similar_sample_idx_{most_similar_idx}"
        
        return False, f"too_similar_max_sim_{max_similarity:.3f}"
    
    def add_sample(self, model, sample: Tuple[torch.Tensor, torch.Tensor], 
                  pilot_mask: torch.Tensor, domain_id: int, loss: float) -> Tuple[bool, str]:
        """
        Add a sample to the buffer using gradient diversity criterion.
        
        Args:
            model: Model manager for gradient computation
            sample: (input, target) tensors
            pilot_mask: Pilot mask for loss computation
            domain_id: Current domain
            loss: Pre-computed loss for this sample
            
        Returns:
            Tuple of (was_added: bool, reason: str)
        """
        self.total_samples_seen += 1
        
        # Compute gradient by recomputing prediction with gradients enabled
        gradient = self.compute_sample_gradient(model, sample[0], sample[1], pilot_mask, domain_id)
        
        # Decide whether to add
        should_add, reason = self.should_add_sample(gradient, loss)
        
        if should_add:
            # Add to buffer
            input_cpu = sample[0].detach().clone().cpu()
            target_cpu = sample[1].detach().clone().cpu()
            
            self.samples.append((input_cpu, target_cpu))
            self.gradients.append(gradient)
            self.losses.append(loss)
            self.samples_added += 1
            
            return True, reason
        else:
            self.samples_rejected += 1
            return False, reason
    
    def _replace_sample(self, index: int, new_gradient: torch.Tensor, new_loss: float):
        """Replace sample at given index with new gradient and loss."""
        # Convert deque to list for indexing, then back to deque
        gradients_list = list(self.gradients)
        losses_list = list(self.losses)
        
        gradients_list[index] = new_gradient
        losses_list[index] = new_loss
        
        self.gradients = deque(gradients_list, maxlen=self.buffer_size)
        self.losses = deque(losses_list, maxlen=self.buffer_size)
    
    def get_all_samples(self) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Get all samples currently in the buffer."""
        return list(self.samples)
    
    def get_random_batch(self, batch_size: int) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Get a random batch of samples from the buffer."""
        import random
        available_samples = list(self.samples)
        if len(available_samples) <= batch_size:
            return available_samples
        return random.sample(available_samples, batch_size)
    
    def __len__(self) -> int:
        """Return current number of samples in buffer."""
        return len(self.samples)
    
    def get_stats(self) -> Dict[str, any]:
        """Get buffer statistics including gradient diversity metrics."""
        stats = {
            'current_size': len(self.samples),
            'max_size': self.buffer_size,
            'total_samples_seen': self.total_samples_seen,
            'samples_added': self.samples_added,
            'samples_rejected': self.samples_rejected,
            'utilization': len(self.samples) / self.buffer_size,
            'acceptance_rate': self.samples_added / max(1, self.total_samples_seen),
            'diversity_threshold': self.diversity_threshold,
            'age_seconds': time.time() - self.creation_time
        }
        
        # Add gradient diversity statistics
        if len(self.gradients) > 1:
            similarities = []
            gradients_list = list(self.gradients)
            for i in range(len(gradients_list)):
                for j in range(i + 1, len(gradients_list)):
                    sim = self.compute_gradient_similarity(gradients_list[i], gradients_list[j])
                    similarities.append(sim)
            
            if similarities:
                stats.update({
                    'avg_gradient_similarity': np.mean(similarities),
                    'max_gradient_similarity': np.max(similarities),
                    'min_gradient_similarity': np.min(similarities),
                    'gradient_diversity_score': 1.0 - np.mean(similarities)
                })
        
        return stats


class GradientDiversityBufferManager:
    """
    Manages multiple gradient diversity buffers, one per domain.
    """
    
    def __init__(self, buffer_config: Dict[str, any], verbose: bool = False):
        """
        Initialize the gradient diversity buffer manager.
        
        Args:
            buffer_config: Configuration dict with keys:
                - enabled: bool
                - size: int (buffer size per domain)
                - diversity_threshold: float (0-1, cosine similarity threshold)
                - min_samples_for_diversity: int
            verbose: Enable verbose debug output
        """
        self.enabled = buffer_config.get('enabled', True)
        self.buffer_size = buffer_config.get('size', 100)
        self.diversity_threshold = buffer_config.get('diversity_threshold', 0.8)
        self.min_samples_for_diversity = buffer_config.get('min_samples_for_diversity', 5)
        self.verbose = verbose
        
        # Domain-specific buffers (created on-demand)
        self.domain_buffers: Dict[int, GradientDiversityBuffer] = {}
        
        if self.verbose:
            print(f"GradientDiversityBufferManager initialised:")
            print(f"   Enabled: {self.enabled}")
            print(f"   Buffer size per domain: {self.buffer_size}")
            print(f"   Diversity threshold: {self.diversity_threshold}")
            print(f"   Min samples for diversity: {self.min_samples_for_diversity}")
    
    def get_or_create_buffer(self, domain_id: int) -> Optional[GradientDiversityBuffer]:
        """Get existing buffer for domain or create new one."""
        if not self.enabled:
            return None
            
        if domain_id not in self.domain_buffers:
            self.domain_buffers[domain_id] = GradientDiversityBuffer(
                buffer_size=self.buffer_size,
                diversity_threshold=self.diversity_threshold,
                min_samples_for_diversity=self.min_samples_for_diversity,
                verbose=self.verbose
            )
            if self.verbose:
                print(f"Created new gradient diversity buffer for domain {domain_id}")
        
        return self.domain_buffers[domain_id]
    
    def add_sample(self, domain_id: int, model, sample: Tuple[torch.Tensor, torch.Tensor], 
                  pilot_mask: torch.Tensor, loss: float) -> Tuple[bool, str]:
        """Add a sample using gradient diversity criterion."""
        buffer = self.get_or_create_buffer(domain_id)
        if buffer is not None:
            return buffer.add_sample(model, sample, pilot_mask, domain_id, loss)
        return False, "buffer_disabled"
    
    def get_buffer(self, domain_id: int) -> Optional[GradientDiversityBuffer]:
        """Get buffer for a specific domain."""
        if not self.enabled:
            return None
        return self.domain_buffers.get(domain_id, None)
    
    def print_status(self):
        """Print status of all buffers including gradient diversity metrics."""
        if self.verbose:
            print("\nGRADIENT DIVERSITY BUFFER STATUS:")
            print("-" * 50)
            
            if not self.domain_buffers:
                print("   No buffers created yet")
                return
            
            for domain_id in sorted(self.domain_buffers.keys()):
                buffer = self.domain_buffers[domain_id]
                stats = buffer.get_stats()
                
                print(f"   Domain {domain_id}: {stats['current_size']:3d}/{stats['max_size']:3d} samples | "
                      f"Accept: {stats['acceptance_rate']:.1%} | "
                      f"Diversity: {stats.get('gradient_diversity_score', 'N/A')}")
                
                if 'avg_gradient_similarity' in stats:
                    print(f"             Avg Similarity: {stats['avg_gradient_similarity']:.3f} | "
                          f"Range: [{stats['min_gradient_similarity']:.3f}, {stats['max_gradient_similarity']:.3f}]") 