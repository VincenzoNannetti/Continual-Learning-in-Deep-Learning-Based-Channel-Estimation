import torch
import torch.nn.functional as F
from math import log10
from shared.utils.ssim import compute_ssim

# Based on Learning_Algorithms/utils/PSNR.py and placeholders in engine.py
def calculate_psnr(output, target, max_val=None):
    """Calculates Peak Signal-to-Noise Ratio.
       Uses dynamic range from target if max_val is not provided.

    Args:
        output (torch.Tensor): Predicted tensor.
        target (torch.Tensor): Ground truth tensor.
        max_val (float, optional): Maximum possible value. If None, uses max(abs(target)).

    Returns:
        torch.Tensor: PSNR value.
    """
    mse = F.mse_loss(output, target)
    if mse == 0:
        return torch.tensor(float('inf')).to(output.device)

    if max_val is None:
        # Calculate max absolute value from target tensor
        data_range = torch.max(torch.abs(target))
        if data_range == 0:
            # Target is all zeros. Since mse > 0, output is non-zero. PSNR is arguably -inf.
            # Returning 0.0 is a safer practical value.
            return torch.tensor(0.0).to(output.device)
    else:
        # Use provided max_val, ensure it's a tensor on the correct device
        data_range = torch.tensor(float(max_val)).to(target.device)

    # Use torch.log10 for tensor operations
    psnr = 20 * torch.log10(data_range / torch.sqrt(mse))
    return psnr

# Based on placeholder in engine.py
def calculate_nmse(output, target, reduction='mean'):
    """Calculates normalised Mean Squared Error.
    
    Args:
        output (torch.Tensor): Predicted tensor.
        target (torch.Tensor): Ground truth tensor.
        reduction (str): 'mean' or 'sum' for how to aggregate MSE.

    Returns:
        torch.Tensor: NMSE value.
    """
    if target.shape != output.shape:
         raise ValueError(f"Output shape {output.shape} must match Target shape {target.shape}")

    target_power = torch.sum(torch.abs(target)**2)
    if target_power == 0:
        # Handle case where target is all zeros
        if torch.all(output == 0):
            return torch.tensor(0.0).to(output.device) # Perfect prediction
        else:
             return torch.tensor(float('inf')).to(output.device) # Non-zero output for zero target
    
    if reduction == 'mean':
         mse = F.mse_loss(output.float(), target.float(), reduction='mean')
    elif reduction == 'sum':
         mse = F.mse_loss(output.float(), target.float(), reduction='sum')
    else:
         raise ValueError(f"Unsupported reduction type: {reduction}")

    nmse = mse / target_power
    return nmse

# Based on Learning_Algorithms/utils/SSIM.py (requires implementation or porting)
# This is a complex metric, might need external library like piqa or skimage
def calculate_ssim(output, target, window_size=11, size_average=True):
    """Calculates Structural Similarity Index (SSIM) using the implementation from SSIM.py.
    
    Note: This implementation calculates SSIM on the magnitude of complex inputs.

    Args:
        output (torch.Tensor): Predicted tensor.
        target (torch.Tensor): Ground truth tensor.
        window_size (int): Size of the Gaussian window (passed to compute_ssim).
        size_average (bool): If True, average SSIM map (passed to compute_ssim).
        
    Returns:
        float: SSIM value.
    """
    # Convert tensors to NumPy arrays for compute_ssim
    output_np = output.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    
    # Call the imported compute_ssim function
    ssim_value = compute_ssim(output_np, target_np, window_size=window_size, size_average=size_average)
    
    return torch.tensor(ssim_value).to(output.device)

# --- Add other metrics as needed --- 

def calculate_mse(output, target):
    """Calculates Mean Squared Error."""
    return F.mse_loss(output, target)

# Dictionary mapping metric names (lowercase) to functions
METRICS_MAP = {
    "psnr": calculate_psnr,
    "nmse": calculate_nmse,
    "ssim": calculate_ssim,
    "mse": calculate_mse
}

def get_metric_function(name):
     """Returns the metric calculation function based on name."""
     name_lower = name.lower()
     if name_lower not in METRICS_MAP:
         raise ValueError(f"Unsupported metric: {name}. Supported: {list(METRICS_MAP.keys())}")
     return METRICS_MAP[name_lower]
