"""
Experience Replay for Continual Learning

This module implements Experience Replay (ER) as a continual learning baseline.
Experience Replay maintains a buffer of past experiences and replays them 
during training on new tasks to prevent catastrophic forgetting.

Reference: Lopez-Paz, D., & Ranzato, M. A. (2017). Gradient episodic memory 
for continual learning. NIPS.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import random

from .base_trainer import BaseContinualTrainer


class ExperienceBuffer:
    """
    Experience buffer for storing samples from previous tasks.
    Implements reservoir sampling for efficient memory management.
    """
    
    def __init__(self, buffer_size: int, input_shape: Tuple[int, ...], target_shape: Tuple[int, ...]):
        """
        Initialize experience buffer.
        
        Args:
            buffer_size: Maximum number of samples to store
            input_shape: Shape of input samples (C, H, W)
            target_shape: Shape of target samples (C, H, W)
        """
        self.buffer_size = buffer_size
        self.current_size = 0
        self.insertion_index = 0
        
        # Pre-allocate buffers for efficiency
        self.inputs = torch.zeros((buffer_size, *input_shape))
        self.targets = torch.zeros((buffer_size, *target_shape))
        self.task_labels = torch.zeros(buffer_size, dtype=torch.long)
        
        # Track samples per task for balanced sampling
        self.samples_per_task = defaultdict(int)
        
    def add_samples(self, inputs: torch.Tensor, targets: torch.Tensor, task_id: int):
        """
        Add samples to the buffer using reservoir sampling.
        
        Args:
            inputs: Input tensor (batch_size, C, H, W)
            targets: Target tensor (batch_size, C, H, W)
            task_id: Task identifier
        """
        batch_size = inputs.size(0)
        
        for i in range(batch_size):
            if self.current_size < self.buffer_size:
                # Buffer not full, just add
                idx = self.current_size
                self.inputs[idx] = inputs[i].cpu()
                self.targets[idx] = targets[i].cpu()
                self.task_labels[idx] = task_id
                self.samples_per_task[task_id] += 1
                self.current_size += 1
            else:
                # Buffer full, use reservoir sampling
                j = random.randint(0, self.insertion_index)
                if j < self.buffer_size:
                    # Replace existing sample
                    old_task = self.task_labels[j].item()
                    self.samples_per_task[old_task] -= 1
                    
                    self.inputs[j] = inputs[i].cpu()
                    self.targets[j] = targets[i].cpu()
                    self.task_labels[j] = task_id
                    self.samples_per_task[task_id] += 1
            
            self.insertion_index += 1
    
    def sample_batch(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample a batch from the buffer.
        
        Args:
            batch_size: Number of samples to return
            device: Device to move tensors to
            
        Returns:
            Tuple of (inputs, targets, task_labels)
        """
        if self.current_size == 0:
            return None, None, None
        
        # Sample indices
        actual_batch_size = min(batch_size, self.current_size)
        indices = torch.randperm(self.current_size)[:actual_batch_size]
        
        return (
            self.inputs[indices].to(device),
            self.targets[indices].to(device),
            self.task_labels[indices].to(device)
        )
    
    def get_statistics(self) -> Dict:
        """Get buffer statistics."""
        return {
            'current_size': self.current_size,
            'buffer_size': self.buffer_size,
            'utilization': self.current_size / self.buffer_size,
            'samples_per_task': dict(self.samples_per_task),
            'unique_tasks': len(self.samples_per_task)
        }


class ExperienceReplayTrainer(BaseContinualTrainer):
    """
    Experience Replay trainer for continual learning.
    """
    
    def __init__(self, model: nn.Module, config: dict, device: torch.device, 
                 buffer_size: int = 1000, replay_batch_ratio: float = 0.5, 
                 wandb_run=None):
        """
        Initialize Experience Replay trainer.
        
        Args:
            model: Neural network model
            config: Training configuration
            device: Device to train on
            buffer_size: Size of experience buffer
            replay_batch_ratio: Ratio of replay samples in each batch
            wandb_run: Weights & Biases run object
        """
        super().__init__(model, config, device, wandb_run)
        
        self.buffer_size = buffer_size
        self.replay_batch_ratio = replay_batch_ratio
        self.current_task_id = 0
        self.task_names = []
        
        # Initialize buffer (will be created when we see first data)
        self.buffer = None
        
        print(f"🔄 Experience Replay initialized")
        print(f"   Buffer size: {buffer_size}")
        print(f"   Replay ratio: {replay_batch_ratio}")
        
    def prepare_for_task(self, task_name: str, train_loader=None):
        """
        Prepare trainer for a new task.
        
        Args:
            task_name: Name of the new task
            train_loader: Training data loader (used to initialize buffer)
        """
        print(f"🎯 ER preparing for task {task_name}")
        
        self.task_names.append(task_name)
        
        # Initialize buffer with first batch if not done yet
        if self.buffer is None and train_loader is not None:
            sample_batch = next(iter(train_loader))
            inputs, targets = sample_batch
            input_shape = inputs.shape[1:]  # (C, H, W)
            target_shape = targets.shape[1:]  # (C, H, W)
            
            self.buffer = ExperienceBuffer(
                buffer_size=self.buffer_size,
                input_shape=input_shape,
                target_shape=target_shape
            )
            print(f"   Buffer initialized with shapes: input={input_shape}, target={target_shape}")
    
    def train_epoch(self, train_loader, optimizer, epoch: int) -> float:
        """
        Train for one epoch with experience replay.
        
        Args:
            train_loader: Training data loader
            optimizer: Optimizer
            epoch: Current epoch number
            
        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        total_samples = 0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            current_batch_size = inputs.size(0)
            
            # Calculate replay batch size
            if self.buffer is not None and self.buffer.current_size > 0:
                replay_batch_size = int(current_batch_size * self.replay_batch_ratio)
                current_task_size = current_batch_size - replay_batch_size
                
                # Get replay samples
                replay_inputs, replay_targets, _ = self.buffer.sample_batch(
                    replay_batch_size, self.device
                )
                
                if replay_inputs is not None:
                    # Combine current task and replay samples
                    combined_inputs = torch.cat([
                        inputs[:current_task_size], 
                        replay_inputs
                    ], dim=0)
                    combined_targets = torch.cat([
                        targets[:current_task_size], 
                        replay_targets
                    ], dim=0)
                else:
                    combined_inputs, combined_targets = inputs, targets
            else:
                combined_inputs, combined_targets = inputs, targets
            
            # Forward pass
            optimizer.zero_grad()
            outputs = self.model(combined_inputs)
            loss = self.criterion(outputs, combined_targets)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * combined_inputs.size(0)
            total_samples += combined_inputs.size(0)
            
            # Log batch-level metrics
            if self.wandb_run and batch_idx % 10 == 0:
                self.wandb_run.log({
                    'batch_loss': loss.item(),
                    'batch_size': combined_inputs.size(0),
                    'replay_ratio': replay_batch_size / combined_inputs.size(0) if self.buffer and self.buffer.current_size > 0 else 0.0
                })
        
        return total_loss / total_samples if total_samples > 0 else 0.0
    
    def after_task_completion(self, task_name: str, val_loader):
        """
        Operations to perform after completing a task.
        
        Args:
            task_name: Name of completed task
            val_loader: Validation data loader for the task
        """
        print(f"🧠 Adding samples to experience buffer for task {task_name}...")
        
        if self.buffer is None:
            print("   ⚠️ Buffer not initialized, skipping sample addition")
            return
        
        # Add samples from validation set to buffer
        # (using validation to avoid overfitting to training data)
        samples_added = 0
        max_samples_per_task = min(1000, len(val_loader.dataset))  # Limit samples per task
        
        self.model.eval()
        with torch.no_grad():
            for inputs, targets in val_loader:
                if samples_added >= max_samples_per_task:
                    break
                
                # Add to buffer
                self.buffer.add_samples(inputs, targets, self.current_task_id)
                samples_added += inputs.size(0)
        
        self.current_task_id += 1
        
        # Print buffer statistics
        stats = self.buffer.get_statistics()
        print(f"   ✅ Buffer update complete")
        print(f"      Samples added: {samples_added}")
        print(f"      Buffer utilization: {stats['utilization']:.2%}")
        print(f"      Samples per task: {stats['samples_per_task']}")
    

    
    def add_regularization_loss(self, base_loss: torch.Tensor, inputs: torch.Tensor, 
                               targets: torch.Tensor, outputs: torch.Tensor) -> torch.Tensor:
        """
        Add regularization loss. For Experience Replay, no additional regularization is needed.
        The replay mechanism is handled in train_epoch().
        
        Args:
            base_loss: Base task loss
            inputs: Input tensor
            targets: Target tensor
            outputs: Model outputs
            
        Returns:
            The base loss (no additional regularization for Experience Replay)
        """
        return base_loss
    
    def get_method_state(self) -> Dict:
        """Get Experience Replay-specific state for checkpointing."""
        state = {
            'buffer_size': self.buffer_size,
            'replay_batch_ratio': self.replay_batch_ratio,
            'current_task_id': self.current_task_id,
            'task_names': self.task_names.copy(),
            'completed_tasks': self.completed_tasks.copy()
        }
        
        # Save buffer state if initialized
        if self.buffer is not None:
            buffer_stats = self.buffer.get_statistics()
            state.update({
                'buffer_initialized': True,
                'buffer_current_size': buffer_stats['current_size'],
                'buffer_inputs': self.buffer.inputs[:buffer_stats['current_size']].clone(),
                'buffer_targets': self.buffer.targets[:buffer_stats['current_size']].clone(),
                'buffer_task_labels': self.buffer.task_labels[:buffer_stats['current_size']].clone(),
                'buffer_samples_per_task': dict(self.buffer.samples_per_task),
                'buffer_insertion_index': self.buffer.insertion_index
            })
        else:
            state['buffer_initialized'] = False
        
        return state
    
    def load_method_state(self, state: Dict):
        """Load Experience Replay-specific state from checkpoint."""
        self.buffer_size = state.get('buffer_size', self.buffer_size)
        self.replay_batch_ratio = state.get('replay_batch_ratio', self.replay_batch_ratio)
        self.current_task_id = state.get('current_task_id', 0)
        self.task_names = state.get('task_names', [])
        self.completed_tasks = state.get('completed_tasks', [])
        
        # Restore buffer if it was saved
        if state.get('buffer_initialized', False):
            buffer_current_size = state['buffer_current_size']
            
            # Get input/target shapes from saved data
            saved_inputs = state['buffer_inputs']
            saved_targets = state['buffer_targets']
            input_shape = saved_inputs.shape[1:]
            target_shape = saved_targets.shape[1:]
            
            # Recreate buffer
            self.buffer = ExperienceBuffer(
                buffer_size=self.buffer_size,
                input_shape=input_shape,
                target_shape=target_shape
            )
            
            # Restore buffer contents
            self.buffer.current_size = buffer_current_size
            self.buffer.inputs[:buffer_current_size] = saved_inputs
            self.buffer.targets[:buffer_current_size] = saved_targets
            self.buffer.task_labels[:buffer_current_size] = state['buffer_task_labels']
            self.buffer.samples_per_task = defaultdict(int, state['buffer_samples_per_task'])
            self.buffer.insertion_index = state['buffer_insertion_index']
            
            print(f"🔄 Experience Replay buffer restored:")
            print(f"   Buffer size: {self.buffer.current_size}/{self.buffer.buffer_size}")
            print(f"   Samples per task: {dict(self.buffer.samples_per_task)}")
        
        print(f"🔄 Experience Replay state loaded:")
        print(f"   Buffer size: {self.buffer_size}")
        print(f"   Replay ratio: {self.replay_batch_ratio}")
        print(f"   Completed tasks: {self.completed_tasks}")
    
    def get_buffer_statistics(self) -> Dict:
        """Get experience buffer statistics."""
        if self.buffer is None:
            return {'buffer_initialized': False}
        
        stats = self.buffer.get_statistics()
        stats['buffer_initialized'] = True
        return stats
    
    def print_replay_statistics(self):
        """Print experience replay statistics."""
        print(f"\n📊 Experience Replay Statistics:")
        print(f"   Buffer size: {self.buffer_size}")
        print(f"   Replay ratio: {self.replay_batch_ratio}")
        print(f"   Tasks completed: {len(self.task_names)}")
        
        if self.buffer is not None:
            stats = self.buffer.get_statistics()
            print(f"   Buffer utilization: {stats['utilization']:.2%}")
            print(f"   Current samples: {stats['current_size']}")
            print(f"   Unique tasks in buffer: {stats['unique_tasks']}")
            
            for task_id, count in stats['samples_per_task'].items():
                task_name = self.task_names[task_id] if task_id < len(self.task_names) else f"Task_{task_id}"
                print(f"   {task_name}: {count} samples")
        else:
            print(f"   Buffer: Not initialized") 