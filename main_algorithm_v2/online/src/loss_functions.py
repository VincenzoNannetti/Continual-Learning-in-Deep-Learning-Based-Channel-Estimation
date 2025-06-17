"""
Loss Functions for Online Continual Learning

This module implements various loss functions used in the online learning algorithm,
particularly the masked NMSE that operates only on pilot positions.
"""

import torch
import numpy as np


def masked_nmse(prediction: torch.Tensor, 
                ground_truth: torch.Tensor, 
                pilot_mask: torch.Tensor) -> float:
    """
    Calculate Normalised Mean Squared Error (NMSE) only at pilot positions.
    
    This is the core loss function for online continual learning (Algorithm 2, line 9):
    L₀ = MaskedNmse(Ĥ, Y_pilot, P)
    
    Args:
        prediction: Model prediction tensor, shape (2, 72, 70) or (batch, 2, 72, 70)
        ground_truth: Ground truth tensor at pilot positions, same shape as prediction
        pilot_mask: Boolean mask indicating pilot positions, shape (72, 70) 
        
    Returns:
        NMSE value as float
    """
    # Handle batch dimension
    if prediction.dim() == 4:
        # Batch mode: (batch, 2, 72, 70)
        prediction = prediction.squeeze(0)  # Remove batch dim
        ground_truth = ground_truth.squeeze(0)
    
    if prediction.dim() != 3 or prediction.shape[0] != 2:
        raise ValueError(f"Expected prediction shape (2, 72, 70), got {prediction.shape}")
    
    # Ensure pilot_mask is a torch tensor on the same device
    if isinstance(pilot_mask, np.ndarray):
        pilot_mask = torch.from_numpy(pilot_mask)
    pilot_mask = pilot_mask.to(prediction.device)
    
    # Apply pilot mask to both channels
    # pilot_mask shape: (72, 70) -> expand to (2, 72, 70)
    pilot_mask_expanded = pilot_mask.unsqueeze(0).expand(2, -1, -1)
    
    # Extract values only at pilot positions
    masked_prediction = prediction[pilot_mask_expanded]  # Shape: (num_pilots * 2,)
    masked_ground_truth = ground_truth[pilot_mask_expanded]  # Shape: (num_pilots * 2,)
    
    # Calculate MSE at pilot positions
    mse = torch.mean((masked_prediction - masked_ground_truth) ** 2)
    
    # Calculate signal power (for normalisation)
    signal_power = torch.mean(masked_ground_truth ** 2)
    
    # Calculate NMSE
    if signal_power > 0:
        nmse = mse / signal_power
    else:
        nmse = torch.tensor(float('inf'))
    
    return nmse.item()


def masked_nmse_tensor(prediction: torch.Tensor, 
                      ground_truth: torch.Tensor, 
                      pilot_mask: torch.Tensor) -> torch.Tensor:
    """
    Calculate Normalised Mean Squared Error (NMSE) only at pilot positions as a tensor.
    
    Same as masked_nmse() but returns a tensor for gradient computation.
    
    Args:
        prediction: Model prediction tensor, shape (2, 72, 70) or (batch, 2, 72, 70)
        ground_truth: Ground truth tensor at pilot positions, same shape as prediction
        pilot_mask: Boolean mask indicating pilot positions, shape (72, 70) 
        
    Returns:
        NMSE value as tensor (maintains computational graph)
    """
    # Handle batch dimension
    if prediction.dim() == 4:
        # Batch mode: (batch, 2, 72, 70)
        prediction = prediction.squeeze(0)  # Remove batch dim
        ground_truth = ground_truth.squeeze(0)
    
    if prediction.dim() != 3 or prediction.shape[0] != 2:
        raise ValueError(f"Expected prediction shape (2, 72, 70), got {prediction.shape}")
    
    # Ensure pilot_mask is a torch tensor on the same device
    if isinstance(pilot_mask, np.ndarray):
        pilot_mask = torch.from_numpy(pilot_mask)
    pilot_mask = pilot_mask.to(prediction.device)
    
    # Apply pilot mask to both channels
    # pilot_mask shape: (72, 70) -> expand to (2, 72, 70)
    pilot_mask_expanded = pilot_mask.unsqueeze(0).expand(2, -1, -1)
    
    # Extract values only at pilot positions
    masked_prediction = prediction[pilot_mask_expanded]  # Shape: (num_pilots * 2,)
    masked_ground_truth = ground_truth[pilot_mask_expanded]  # Shape: (num_pilots * 2,)
    
    # Calculate MSE at pilot positions
    mse = torch.mean((masked_prediction - masked_ground_truth) ** 2)
    
    # Calculate signal power (for normalisation)
    signal_power = torch.mean(masked_ground_truth ** 2)
    
    # Calculate NMSE (keep as tensor for gradient computation)
    if signal_power > 0:
        nmse = mse / signal_power
    else:
        nmse = torch.tensor(float('inf'), device=prediction.device)
    
    return nmse


def extract_pilot_ground_truth(ground_truth: torch.Tensor, 
                              pilot_mask: torch.Tensor) -> torch.Tensor:
    """
    Extract ground truth values only at pilot positions (Algorithm 2, line 2).
    
    Implements: Y_pilot = P ⊙ Y
    
    Args:
        ground_truth: Full ground truth tensor, shape (2, 72, 70)
        pilot_mask: Boolean mask indicating pilot positions, shape (72, 70)
        
    Returns:
        Ground truth with non-pilot positions zeroed out
    """
    # Handle batch dimension
    if ground_truth.dim() == 4:
        ground_truth = ground_truth.squeeze(0)
    
    # Ensure pilot_mask is a torch tensor on the same device
    if isinstance(pilot_mask, np.ndarray):
        pilot_mask = torch.from_numpy(pilot_mask)
    pilot_mask = pilot_mask.to(ground_truth.device)
    
    # Create pilot-only ground truth
    pilot_ground_truth = ground_truth.clone()
    
    # Zero out non-pilot positions
    pilot_mask_expanded = pilot_mask.unsqueeze(0).expand(2, -1, -1)
    pilot_ground_truth[~pilot_mask_expanded] = 0.0
    
    return pilot_ground_truth


def calculate_loss_statistics(losses: list, window_size: int = 10) -> dict:
    """
    Calculate rolling statistics for loss values for drift detection.
    
    Args:
        losses: List of recent loss values
        window_size: Window size for rolling statistics
        
    Returns:
        Dictionary with loss statistics
    """
    if len(losses) == 0:
        return {'mean': 0.0, 'std': 0.0, 'trend': 0.0}
    
    recent_losses = losses[-window_size:]
    
    stats = {
        'mean': np.mean(recent_losses),
        'std': np.std(recent_losses),
        'latest': losses[-1] if losses else 0.0,
        'count': len(recent_losses)
    }
    
    # Calculate trend (simple linear slope)
    if len(recent_losses) >= 3:
        x = np.arange(len(recent_losses))
        y = np.array(recent_losses)
        # Simple linear regression slope
        trend = np.polyfit(x, y, 1)[0]
        stats['trend'] = trend
    else:
        stats['trend'] = 0.0
    
    return stats


def exponential_moving_average(current_ema: float, 
                              new_value: float, 
                              alpha: float = 0.1) -> float:
    """
    Update exponential moving average (Algorithm 2, line 12).
    
    Implements: ℓ̄_d = (1-α)ℓ̄_d + αL₀
    
    Args:
        current_ema: Current EMA value
        new_value: New observation
        alpha: Smoothing factor (0 < α < 1)
        
    Returns:
        Updated EMA value
    """
    return (1 - alpha) * current_ema + alpha * new_value 