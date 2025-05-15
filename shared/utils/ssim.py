"""
Filename: ./utils/SSIM.py
Author: Vincenzo Nannetti
Date: 30/03/2025
Description: Implementation of the Structural Similarity Index (SSIM) for image/channel quality assessment

Usage:
    from utils.SSIM import compute_ssim

Dependencies:
    - PyTorch
    - NumPy
"""

import numpy as np
import torch

def compute_ssim(img1, img2, window_size=11, size_average=True):
    """
    Calculate SSIM (Structural Similarity Index) between two images/channels
    
    Args:
        img1 (numpy.ndarray or torch.Tensor): First image/channel (can be complex)
        img2 (numpy.ndarray or torch.Tensor): Second image/channel (can be complex)
        window_size (int): Size of the Gaussian window
        size_average (bool): If True, average the SSIM across the spatial dimensions
        
    Returns:
        float: SSIM value between 0 and 1 (1 means identical images)
    """
    # Convert complex values to magnitude if needed
    if np.iscomplexobj(img1):
        img1 = np.abs(img1)
    if np.iscomplexobj(img2):
        img2 = np.abs(img2)
    
    # Convert numpy arrays to torch tensors if needed
    if isinstance(img1, np.ndarray):
        img1 = torch.from_numpy(img1).float()
    if isinstance(img2, np.ndarray):
        img2 = torch.from_numpy(img2).float()
    
    # Ensure shapes are compatible
    if img1.dim() == 2:
        img1 = img1.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
    elif img1.dim() == 3:
        img1 = img1.unsqueeze(0)  # Add batch dimension
    
    if img2.dim() == 2:
        img2 = img2.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
    elif img2.dim() == 3:
        img2 = img2.unsqueeze(0)  # Add batch dimension
    
    # Constants for stability
    C1 = (0.01 * 1.0) ** 2
    C2 = (0.03 * 1.0) ** 2
    
    # Gaussian window
    window = _gaussian_window(window_size, 1.5).repeat(img1.shape[1], 1, 1, 1)
    
    # Mean calculations with Gaussian window
    mu1 = torch.nn.functional.conv2d(img1, window, padding=window_size//2, groups=img1.shape[1])
    mu2 = torch.nn.functional.conv2d(img2, window, padding=window_size//2, groups=img2.shape[1])
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    
    # Variance calculations with Gaussian window
    sigma1_sq = torch.nn.functional.conv2d(img1 * img1, window, padding=window_size//2, groups=img1.shape[1]) - mu1_sq
    sigma2_sq = torch.nn.functional.conv2d(img2 * img2, window, padding=window_size//2, groups=img2.shape[1]) - mu2_sq
    sigma12 = torch.nn.functional.conv2d(img1 * img2, window, padding=window_size//2, groups=img1.shape[1]) - mu1_mu2
    
    # SSIM formula
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    if size_average:
        return ssim_map.mean().item()
    else:
        return ssim_map.mean(1).mean(1).mean(1).item()

def _gaussian_window(window_size, sigma):
    """
    Create a Gaussian window for SSIM calculation
    
    Args:
        window_size (int): Size of the window
        sigma (float): Standard deviation of the Gaussian
        
    Returns:
        torch.Tensor: Gaussian window
    """
    coords = torch.arange(window_size, dtype=torch.float)
    coords -= window_size // 2
    
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    
    window = g.unsqueeze(0) * g.unsqueeze(1)
    window = window.unsqueeze(0).unsqueeze(0)
    
    return window 