"""
Custom LoRA (Low-Rank Adaptation) layer implementation that supports
managing multiple adapters for continual learning scenarios.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Union
import logging

logger = logging.getLogger(__name__)

def _compute_effective_rank(conv: nn.Conv2d, requested_r: int,
                            min_rank: int = 1, warn: bool = True) -> Optional[int]:
    """
    Compute the effective LoRA rank for a Conv2d layer, respecting mathematical constraints.
    
    Args:
        conv: The Conv2d layer to analyze
        requested_r: The originally requested rank
        min_rank: Minimum rank to allow (default: 1)
        warn: Whether to emit warnings for adjustments
        
    Returns:
        Effective rank to use, or None if layer should be skipped
    """
    in_dim = conv.in_channels * conv.kernel_size[0] * conv.kernel_size[1]
    out_dim = conv.out_channels
    max_valid = min(out_dim, in_dim)

    if max_valid < min_rank:
        if warn:
            logger.info(f"[LoRA] Skipping {conv}: max_valid_rank={max_valid} < {min_rank}")
        return None  # signal to skip layer

    eff_r = max(min_rank, min(requested_r, max_valid))
    if eff_r != requested_r and warn:
        logger.warning(f"[LoRA] {conv}: r reduced {requested_r} → {eff_r} (max={max_valid})")
    return eff_r


class TaskAdapter(nn.Module):
    """
    Individual task adapter containing A and B matrices for a specific task.
    """
    def __init__(self, A: nn.Parameter, B: nn.Parameter):
        super().__init__()
        self.A = A
        self.B = B

class LoRAConv2d(nn.Module):
    """
    A LoRA-adapted 2D convolutional layer capable of managing multiple,
    task-specific adapter pairs.

    This layer wraps a standard nn.Conv2d layer. The weights of the original
    convolutional layer are frozen. Task-specific, trainable low-rank adapter
    matrices (A and B) are added and can be activated on a per-task basis.
    This allows for efficient continual learning where a base model is adapted
    for a sequence of tasks.

    Attributes:
        conv (nn.Conv2d): The original, frozen 2D convolutional layer.
        task_adapters (nn.ModuleDict): A dictionary of TaskAdapter modules, keyed by task_id.
        lora_dropout (nn.Dropout): Dropout layer applied to the LoRA path.
        lora_scaling (Dict[str, float]): Dictionary to store scaling factor for each task.
        active_task_id (Optional[str]): The key of the currently active adapter.
    """
    def __init__(self, conv_layer: nn.Conv2d, lora_dropout: float = 0.0, 
                 min_rank: int = 1, quiet: bool = False):
        """
        Initialises the LoRAConv2d layer.

        Args:
            conv_layer (nn.Conv2d): The original nn.Conv2d layer to be adapted.
            lora_dropout (float): The dropout rate to be used in the LoRA path.
            min_rank (int): Minimum rank to allow for LoRA adapters (default: 1).
            quiet (bool): If True, suppress rank adjustment warnings.
        """
        super().__init__()
        self.conv = conv_layer
        self.conv.weight.requires_grad = False # Freeze original weights
        if self.conv.bias is not None:
            self.conv.bias.requires_grad = False # Freeze original bias too

        self.task_adapters = nn.ModuleDict()
        self.min_rank = min_rank
        self.quiet = quiet

        self.lora_dropout = nn.Dropout(lora_dropout)
        self.lora_scaling = {}
        self.active_task_id = None
        self.merged_task_ids = set()  # Track which tasks have been merged

    def add_task_adapters(self, task_id: str, r: int, lora_alpha: int):
        """
        Adds a new pair of LoRA adapters for a specific task.

        Args:
            task_id (str): A unique identifier for the task.
            r (int): The requested rank of the LoRA adapter.
            lora_alpha (int): The scaling factor for the LoRA adapter.
        """
        if task_id in self.task_adapters:
            # Adapters for this task already exist
            return

        # Compute effective rank respecting mathematical constraints
        effective_r = _compute_effective_rank(self.conv, r, self.min_rank, warn=not self.quiet)
        
        if effective_r is None:
            # Layer should be skipped - store a placeholder to avoid breaking later code
            self.task_adapters[task_id] = None
            self.lora_scaling[task_id] = 0.0
            return

        # Create A and B parameters on the same device as the backbone
        device = self.conv.weight.device
        param_A = nn.Parameter(
            torch.zeros(effective_r, self.conv.in_channels * self.conv.kernel_size[0] * self.conv.kernel_size[1], device=device)
        )
        param_B = nn.Parameter(
            torch.zeros(self.conv.out_channels, effective_r, device=device)
        )
        
        # Initialise weights following LoRA paper: A ~ N(0, 0.01), B = 0
        nn.init.normal_(param_A, mean=0.0, std=0.01)
        nn.init.zeros_(param_B)

        # Create and register the task adapter
        self.task_adapters[task_id] = TaskAdapter(param_A, param_B)

        self.lora_scaling[task_id] = lora_alpha / effective_r
        
        # Log success with effective rank info
        if not self.quiet:
            if effective_r == r:
                logger.info(f"[LoRA] Added adapters for task '{task_id}' with r={effective_r} to {self.conv}")
            else:
                logger.info(f"[LoRA] Added adapters for task '{task_id}' with r={effective_r} (requested {r}) to {self.conv}")

    def set_active_task(self, task_id: Optional[str]):
        """
        Sets the active LoRA adapter for the forward pass.

        Args:
            task_id (Optional[str]): The ID of the task to activate. If None,
                                     only the backbone will be used.
        """
        if task_id is not None and task_id not in self.task_adapters:
            raise ValueError(f"Task ID '{task_id}' not found in LoRA adapters.")
        self.active_task_id = task_id

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the LoRA-adapted convolutional layer.
        Implements W = W₀ + B @ A * scaling like the official LoRA implementation.
        """
        if self.active_task_id is None:
            # No LoRA adaptation, just backbone
            return self.conv(x)
        
        task_id = self.active_task_id
        if task_id not in self.task_adapters:
            # This can happen if we are evaluating a task that has no adapter
            return self.conv(x)

        # Get LoRA weights from task adapter (already on correct device)
        adapter = self.task_adapters[task_id]
        
        # Handle skipped layers (placeholder adapters)
        if adapter is None:
            return self.conv(x)
            
        lora_A = adapter.A  # (r, in_channels*k*k)
        lora_B = adapter.B  # (out_channels, r)
        scaling = self.lora_scaling[task_id]
        
        # Compute backbone output (frozen path - no dropout)
        backbone_output = self.conv(x)
        
        # Compute LoRA adaptation with dropout applied only to LoRA path
        dropped_x = self.lora_dropout(x)
        lora_weight = (lora_B @ lora_A).view(self.conv.weight.shape) * scaling  # (out_channels, in_channels, k, k)
        lora_weight = lora_weight.to(dtype=x.dtype, device=x.device)  # Fix AMP dtype and device mismatch
        lora_output = F.conv2d(
            dropped_x,
            lora_weight,
            None,  # No bias for LoRA path
            self.conv.stride,
            self.conv.padding,
            self.conv.dilation,
            self.conv.groups
        )
        
        # Combine backbone and LoRA outputs
        return backbone_output + lora_output

    def train(self, mode: bool = True):
        """
        Override the default train method to ensure backbone remains frozen.
        Only LoRA parameters and bias (if trainable) are set to training mode.
        """
        super().train(mode)
        self.conv.train(False) # Keep backbone conv in eval mode
        return self

    def merge_weights(self, task_id: str = None):
        """
        Merge LoRA weights into the base convolution weights for zero-latency inference.
        Warning: This modifies the original conv weights permanently until unmerge is called.
        
        Args:
            task_id: Task to merge. If None, uses active_task_id.
        """
        if task_id is None:
            task_id = self.active_task_id
            
        if task_id is None or task_id not in self.task_adapters:
            return
            
        if task_id in self.merged_task_ids:
            logger.warning(f"Task '{task_id}' weights already merged. Use unmerge() first.")
            return
            
        adapter = self.task_adapters[task_id]
        
        # Handle skipped layers
        if adapter is None:
            return
            
        lora_A = adapter.A
        lora_B = adapter.B
        scaling = self.lora_scaling[task_id]
        
        # Compute LoRA weight update and add to original weights
        lora_weight = (lora_B @ lora_A).view(self.conv.weight.shape) * scaling
        self.conv.weight.data += lora_weight
        self.merged_task_ids.add(task_id)
        
        if not self.quiet:
            logger.info(f"Merged LoRA weights for task '{task_id}' into base conv layer")
    
    def unmerge_weights(self, task_id: str = None):
        """
        Remove previously merged LoRA weights from the base convolution weights.
        
        Args:
            task_id: Task to unmerge. If None, uses active_task_id.
        """
        if task_id is None:
            task_id = self.active_task_id
            
        if task_id is None or task_id not in self.task_adapters:
            return
            
        if task_id not in self.merged_task_ids:
            logger.warning(f"Task '{task_id}' weights not currently merged.")
            return
            
        adapter = self.task_adapters[task_id]
        
        # Handle skipped layers
        if adapter is None:
            return
            
        lora_A = adapter.A
        lora_B = adapter.B
        scaling = self.lora_scaling[task_id]
        
        # Compute LoRA weight update and subtract from original weights
        lora_weight = (lora_B @ lora_A).view(self.conv.weight.shape) * scaling
        self.conv.weight.data -= lora_weight
        self.merged_task_ids.discard(task_id)
        
        if not self.quiet:
            logger.info(f"Unmerged LoRA weights for task '{task_id}' from base conv layer")

    def extra_repr(self) -> str:
        return f'active_task_id={self.active_task_id}, tasks={[key for key in self.task_adapters.keys()]}' 