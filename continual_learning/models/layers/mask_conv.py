"""
Filename: ./Supermasks/layers/MaskConv.py
Author: Vincenzo Nannetti
Date: 14/03/2025
Description: Modification of standard convolutional layer which applies a mask to the weights.
             Code initially from What's Hidden in a Randomly Weighted Neural Network?

Usage:

Dependencies:
    - PyTorch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from continual_learning.utils import supermask_utils as utils

class MaskConv(nn.Conv2d):
    def __init__(self, *args, sparsity=0.05, num_tasks=3, alpha=0.3, mask_pretrained=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_tasks = num_tasks
        self.sparsity = sparsity
        # Alpha controls mixing between random and pretrained weights
        # alpha=1 means all random, alpha=0 means all pretrained
        # Make alpha a learnable parameter as per original design
        self.alpha = nn.Parameter(torch.tensor(alpha))
        # Whether to apply the mask to pretrained weights or not
        self.mask_pretrained = mask_pretrained
        
        # --- Safe attribute initialization --- 
        self.task   = 0 # Default to task 0
        self.alphas = torch.ones(self.num_tasks) / self.num_tasks 
        self.num_tasks_learned = 1 # Default to 1 task learned initially
        # --- End Safe init ---

        # Separate score tensors for each task
        self.scores = nn.ParameterList(
            [
                nn.Parameter(utils.mask_init(self))
                for _ in range(num_tasks)
            ]
        )

        # Initisalise the random weights (these are the base self.weight)
        nn.init.kaiming_normal_(self.weight, mode="fan_out", nonlinearity="relu")
        # Freeze the random weights - only scores are trained
        self.weight.requires_grad = False 

        # Buffer for pretrained weights will be registered IF they are provided
        # self.register_buffer('pretrained', None) # Don't register yet
        
        # Buffer for cached masks will be registered by cache_masks()
        self.register_buffer("stacked", None) 

    # method which precomputes binary masks for all tasks and stores them in a buffer.
    def cache_masks(self):
        self.register_buffer(
            "stacked",
            torch.stack(
                [
                    utils.get_subnet(self.scores[j].abs(), self.sparsity)
                    for j in range(self.num_tasks)
                ]
            ),
        )

    # method which clears the cache
    def clear_masks(self):
        self.register_buffer("stacked", None)

    def forward(self, x):
        # Determine the subnet mask based on current task or inference mode
        if self.task < 0: # Inference mode (using averaged masks)
            if self.stacked is None:
                # Masks not cached yet - cache them now
                self.cache_masks()
                if self.stacked is None:
                    # If still None after attempting to cache, something is wrong
                    raise RuntimeError(f"Cannot perform inference: Failed to cache masks for {self.num_tasks} tasks")
            
            if self.num_tasks_learned == 0:
                # No tasks learned yet, use default task 0
                print(f"Warning: Inference mode called but no tasks marked as learned. Using task 0 mask.")
                subnet = self.stacked[0]
            else:
                # Average the masks according to alphas
                subnet = torch.zeros_like(self.stacked[0])
                # Only use the tasks that have been learned
                for i in range(self.num_tasks_learned):
                    if i < len(self.alphas):  # Safety check
                        subnet += self.alphas[i] * self.stacked[i]
                subnet = (subnet > 0.5).float()

        else: # Training/evaluation mode for a specific task
            if self.task >= len(self.scores):
                 raise IndexError(f"Task index {self.task} out of range for scores (num_tasks={self.num_tasks})")
            subnet_scores = self.scores[self.task].abs()
            subnet = utils.GetSubnet.apply(subnet_scores, self.sparsity)

        # --- Ensure mask is on correct device and dtype --- 
        subnet_mask = subnet.to(self.weight.dtype).to(self.weight.device)
        # --- 

        # Apply the subnet mask to the random weights
        masked_random_weight = self.weight * subnet_mask

        # Check if pretrained weights exist and apply the SAME mask
        if hasattr(self, "pretrained") and self.pretrained is not None:
            # Ensure pretrained is also on the correct device/dtype before masking
            pretrained_weights = self.pretrained.to(self.weight.dtype).to(self.weight.device)
            if pretrained_weights.shape != self.weight.shape:
                 raise ValueError(f"Shape mismatch between pretrained {pretrained_weights.shape} and random {self.weight.shape}")
            
            # Apply mask to pretrained weights only if mask_pretrained is True
            if self.mask_pretrained:
                masked_pretrained_weight = pretrained_weights * subnet_mask  # Apply same mask to pretrained weights
            else:
                # Use pretrained weights as-is without masking
                masked_pretrained_weight = pretrained_weights
            
            # Mix the masked weights using alpha
            final_weight = self.alpha * masked_random_weight + (1.0 - self.alpha) * masked_pretrained_weight
        else:
            # No pretrained weights, just use the masked random weights
            final_weight = masked_random_weight

        # Perform the convolution using the final masked and mixed weights
        x = F.conv2d(
            x, final_weight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        return x

    def __repr__(self):
        # Corrected representation string
        return f"MaskConv({self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}, num_tasks={self.num_tasks}, sparsity={self.sparsity})"