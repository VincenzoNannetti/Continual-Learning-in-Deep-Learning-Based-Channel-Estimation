"""
Filename: ./Supermasks/utils/supermask_utils.py
Author: Vincenzo Nannetti
Date: 14/03/2025
Description: File which contains helper functions for the supermask algorithm

Usage:


Dependencies:
    - PyTorch
"""
import torch
import torch.nn as nn
import torch.autograd as autograd
import numpy as np

# creates a tensor of the same size as the weights of the layer
def mask_init(module):
    scores = torch.Tensor(module.weight.size())
    # kaiming uniform distribution
    nn.init.kaiming_uniform_(scores, a=np.sqrt(5))
    return scores

# custom autograd function - like the image version
class GetSubnet(autograd.Function):
    @staticmethod
    # receives scores and a sparsity level - fraction of elements that should be kept in the final mask
    def forward(ctx, scores, k):
        # Get the supermask by sorting the scores and using the top k%
        # sorts the flattened scores and finds the cut off index j such that the top k% of scores are kept.
        out = scores.clone()
        _, idx = scores.flatten().sort()
        j = int((1 - k) * scores.numel())

        # flat_out and out access the same memory.
        # the lowest (1-k)% of values are set to 0 and the highest k% are set to 1
        flat_out = out.flatten()
        flat_out[idx[:j]] = 0
        flat_out[idx[j:]] = 1

        return out

    @staticmethod
    def backward(ctx, g):
        # send the gradient g straight-through on the backward pass.
        return g, None

# performs the same operation as GetSubnet but as a function
def get_subnet(scores, k):
    out = scores.clone()
    _, idx = scores.flatten().sort()
    j = int((1 - k) * scores.numel())

    # flat_out and out access the same memory.
    flat_out = out.flatten()
    flat_out[idx[:j]] = 0
    flat_out[idx[j:]] = 1

    return out

# function to ensure that the weights have a fixed magnitude while preserving their sign.
def signed_constant(module):
    # compute the number of input connections
    fan = nn.init._calculate_correct_fan(module.weight, 'fan_in')
    # determine the gain value for ReLU activations
    gain = nn.init.calculate_gain('relu')
    # compute the standard deviation
    std = gain / np.sqrt(fan)
    # update the weight matrix to the sign of the orignal value * std (i.e., contains -std or +std)
    module.weight.data = module.weight.data.sign() * std

# set the task of each layer so it can fetch the mask from cache
def set_model_task(model, task):
    from continual_learning.models.layers.mask_conv import MaskConv
    for n,m in model.named_modules():
        if isinstance(m, MaskConv):
            m.task = task

# called to cache the masks of each layer for the given task
def cache_masks(model):
    from continual_learning.models.layers.mask_conv import MaskConv
    for n,m in model.named_modules():
        if isinstance(m, MaskConv):
            m.cache_masks()

# update the alpha weights of each layer
def set_alphas(model, alphas):
    from continual_learning.models.layers.mask_conv import MaskConv
    for n,m in model.named_modules():
        if isinstance(m, MaskConv):
            m.alphas = alphas

# update the number of tasks that have been learned
def set_num_tasks_learned(model, num_tasks):
    from continual_learning.models.layers.mask_conv import MaskConv
    for n,m in model.named_modules():
        if isinstance(m, MaskConv):
            m.num_tasks_learned = num_tasks