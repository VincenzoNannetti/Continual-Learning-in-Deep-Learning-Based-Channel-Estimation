"""
Online Buffer Manager for Continual Learning

Manages domain-specific online replay buffers that are instantiated during 
online evaluation and used for dynamic experience replay.
"""

import torch
from typing import Dict, Optional, Tuple, List
from collections import deque
import time

class OnlineReplayBuffer:
    """
    Online replay buffer for a single domain with configurable memory management.
    """
    
    def __init__(self, buffer_size: int, strategy: str = "fifo"):
        """
        Initialize online replay buffer for a domain.
        
        Args:
            buffer_size: Maximum number of samples to store
            strategy: Memory management strategy ("fifo" for now)
        """
        self.buffer_size = buffer_size
        self.strategy = strategy
        self.buffer = deque(maxlen=buffer_size)
        self.creation_time = time.time()
        self.total_samples_seen = 0
        
    def add(self, sample: Tuple[torch.Tensor, torch.Tensor]):
        """
        Add a sample to the online buffer.
        
        Args:
            sample: Tuple of (input_tensor, target_tensor)
        """
        # Detach and move to CPU for storage
        input_data, target_data = sample
        input_cpu = input_data.detach().clone().cpu()
        target_cpu = target_data.detach().clone().cpu()
        
        #  MEMORY MANAGEMENT STRATEGY POINT 
        # This is where you can implement your own sample selection logic!
        self._add_with_strategy((input_cpu, target_cpu))
        
        self.total_samples_seen += 1
    
    def _add_with_strategy(self, sample: Tuple[torch.Tensor, torch.Tensor]):
        """
         CUSTOMIZE THIS METHOD FOR YOUR MEMORY MANAGEMENT STRATEGY! 
        
        Current implementation: Simple FIFO (First In, First Out)
        
        Args:
            sample: CPU tensors tuple (input, target)
        """
        # FIFO strategy: Just append (deque handles max length automatically)
        if self.strategy == "fifo":
            self.buffer.append(sample)
        else:
            # TODO: Implement other strategies here
            # e.g., difficulty-based, uncertainty-based, etc.
            self.buffer.append(sample)  # Fallback to FIFO
    
    def get_all_samples(self) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Get all samples currently in the buffer.
        
        Returns:
            List of (input, target) tuples
        """
        return list(self.buffer)
    
    def get_random_batch(self, batch_size: int) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Get a random batch of samples from the buffer.
        
        Args:
            batch_size: Number of samples to return
            
        Returns:
            List of randomly selected samples
        """
        import random
        available_samples = list(self.buffer)
        if len(available_samples) <= batch_size:
            return available_samples
        return random.sample(available_samples, batch_size)
    
    def __len__(self) -> int:
        """Return current number of samples in buffer."""
        return len(self.buffer)
    
    def get_stats(self) -> Dict[str, any]:
        """Get buffer statistics."""
        return {
            'current_size': len(self.buffer),
            'max_size': self.buffer_size,
            'total_samples_seen': self.total_samples_seen,
            'utilization': len(self.buffer) / self.buffer_size,
            'strategy': self.strategy,
            'age_seconds': time.time() - self.creation_time
        }


class OnlineBufferManager:
    """
    Manages multiple online replay buffers, one per domain.
    """
    
    def __init__(self, buffer_config: Dict[str, any]):
        """
        Initialize the online buffer manager.
        
        Args:
            buffer_config: Configuration dict with keys:
                - enabled: bool
                - size: int (buffer size per domain)
                - strategy: str (memory management strategy)
        """
        self.enabled = buffer_config.get('enabled', True)
        self.buffer_size = buffer_config.get('size', 100)
        self.strategy = buffer_config.get('strategy', 'fifo')
        
        # Domain-specific buffers (created on-demand)
        self.domain_buffers: Dict[int, OnlineReplayBuffer] = {}
        
        print(f" OnlineBufferManager initialized:")
        print(f"   Enabled: {self.enabled}")
        print(f"   Buffer size per domain: {self.buffer_size}")
        print(f"   Memory strategy: {self.strategy}")
    
    def get_or_create_buffer(self, domain_id: int) -> Optional[OnlineReplayBuffer]:
        """
        Get existing buffer for domain or create new one if first time.
        
        Args:
            domain_id: Domain identifier
            
        Returns:
            OnlineReplayBuffer for the domain, or None if disabled
        """
        if not self.enabled:
            return None
            
        if domain_id not in self.domain_buffers:
            # First time seeing this domain - create new buffer
            self.domain_buffers[domain_id] = OnlineReplayBuffer(
                buffer_size=self.buffer_size,
                strategy=self.strategy
            )
            print(f" Created new online buffer for domain {domain_id} (size: {self.buffer_size})")
        
        return self.domain_buffers[domain_id]
    
    def add_sample(self, domain_id: int, sample: Tuple[torch.Tensor, torch.Tensor]):
        """
        Add a sample to the online buffer for the specified domain.
        
        Args:
            domain_id: Domain identifier
            sample: Tuple of (input, target) tensors
        """
        buffer = self.get_or_create_buffer(domain_id)
        if buffer is not None:
            buffer.add(sample)
    
    def get_buffer(self, domain_id: int) -> Optional[OnlineReplayBuffer]:
        """
        Get buffer for a specific domain (without creating if doesn't exist).
        
        Args:
            domain_id: Domain identifier
            
        Returns:
            OnlineReplayBuffer or None if doesn't exist or disabled
        """
        if not self.enabled:
            return None
        return self.domain_buffers.get(domain_id, None)
    
    def get_active_domains(self) -> List[int]:
        """
        Get list of domains that have active buffers.
        
        Returns:
            List of domain IDs with instantiated buffers
        """
        return list(self.domain_buffers.keys())
    
    def get_total_samples(self) -> int:
        """
        Get total number of samples across all domain buffers.
        
        Returns:
            Total sample count
        """
        return sum(len(buffer) for buffer in self.domain_buffers.values())
    
    def get_manager_stats(self) -> Dict[str, any]:
        """
        Get comprehensive statistics about all buffers.
        
        Returns:
            Dictionary with manager and per-domain statistics
        """
        stats = {
            'enabled': self.enabled,
            'active_domains': len(self.domain_buffers),
            'total_samples': self.get_total_samples(),
            'buffer_config': {
                'size': self.buffer_size,
                'strategy': self.strategy
            },
            'domain_stats': {}
        }
        
        for domain_id, buffer in self.domain_buffers.items():
            stats['domain_stats'][domain_id] = buffer.get_stats()
            
        return stats
    
    def print_status(self):
        """Print current status of all buffers."""
        print(f"\n ONLINE BUFFER MANAGER STATUS:")
        print(f"   Active domains: {len(self.domain_buffers)}")
        print(f"   Total samples: {self.get_total_samples()}")
        
        for domain_id in sorted(self.domain_buffers.keys()):
            buffer = self.domain_buffers[domain_id]
            stats = buffer.get_stats()
            print(f"   Domain {domain_id}: {stats['current_size']}/{stats['max_size']} samples "
                  f"({stats['utilization']:.1%} full, seen {stats['total_samples_seen']} total)") 