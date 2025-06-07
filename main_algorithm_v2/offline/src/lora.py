"""
Custom LoRA (Low-Rank Adaptation) layer implementation that supports
managing multiple adapters for continual learning scenarios.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

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
    def __init__(self, conv_layer: nn.Conv2d, lora_dropout: float = 0.0):
        """
        Initialises the LoRAConv2d layer.

        Args:
            conv_layer (nn.Conv2d): The original nn.Conv2d layer to be adapted.
            lora_dropout (float): The dropout rate to be used in the LoRA path.
        """
        super().__init__()
        self.conv = conv_layer
        self.conv.weight.requires_grad = False # Freeze original weights

        self.task_adapters = nn.ModuleDict()

        self.lora_dropout = nn.Dropout(lora_dropout)
        self.lora_scaling = {}
        self.active_task_id = None

    def add_task_adapters(self, task_id: str, r: int, lora_alpha: int):
        """
        Adds a new pair of LoRA adapters for a specific task.

        Args:
            task_id (str): A unique identifier for the task.
            r (int): The rank of the LoRA adapter.
            lora_alpha (int): The scaling factor for the LoRA adapter.
        """
        if task_id in self.task_adapters:
            # Adapters for this task already exist
            return

        # Create A and B parameters
        param_A = nn.Parameter(
            torch.zeros(r, self.conv.in_channels * self.conv.kernel_size[0] * self.conv.kernel_size[1])
        )
        param_B = nn.Parameter(
            torch.zeros(self.conv.out_channels, r)
        )
        
        # Initialise weights
        nn.init.kaiming_uniform_(param_A, a=math.sqrt(5))
        nn.init.zeros_(param_B)

        # Create and register the task adapter
        self.task_adapters[task_id] = TaskAdapter(param_A, param_B)

        self.lora_scaling[task_id] = lora_alpha / r
        print(f"Added LoRA adapters for task '{task_id}' with r={r} to layer {self.conv}")

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

        # Get LoRA weights from task adapter on correct device
        adapter = self.task_adapters[task_id]
        lora_A = adapter.A.to(x.device)  # (r, in_channels*k*k)
        lora_B = adapter.B.to(x.device)  # (out_channels, r)
        scaling = self.lora_scaling[task_id]
        
        # Compute LoRA weight update: B @ A
        # Reshape to match conv weight dimensions
        lora_weight = (lora_B @ lora_A).view(self.conv.weight.shape) * scaling  # (out_channels, in_channels, k, k)
        
        # Apply convolution with modified weights: W₀ + ΔW
        return F.conv2d(
            x,
            self.conv.weight + lora_weight,
            self.conv.bias,
            self.conv.stride,
            self.conv.padding,
            self.conv.dilation,
            self.conv.groups
        )

    def train(self, mode: bool = True):
        """
        Override the default train method to ensure backbone remains frozen.
        Only LoRA parameters and bias (if trainable) are set to training mode.
        """
        super().train(mode)
        self.conv.train(False) # Keep backbone conv in eval mode
        return self

    def extra_repr(self) -> str:
        return f'active_task_id={self.active_task_id}, tasks={[key for key in self.task_adapters.keys()]}' 