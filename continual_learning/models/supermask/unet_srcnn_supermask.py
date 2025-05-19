"""
Filename: ./Supermasks/models/UNet_SRCNN_Supermask.py
Author: Vincenzo Nannetti
Date: 03/05/2025
Description: UNet combined with SRCNN Supermask.

Usage:
    Combined model with UNet and SRCNN architectures using supermasks for continual learning.
    Takes input with 3 channels (real, imag, pilot mask) and determines the task based on 
    the pilot mask pattern.

Dependencies:
    - PyTorch
    - MaskConv
    - UNetModel
    - SRCNN
    - supermask_utils
"""
import torch
import torch.nn as nn
import numpy as np
from standard_training.models.unet              import UNetModel
from standard_training.models.srcnn             import SRCNN
from continual_learning.models.layers.mask_conv import MaskConv
from continual_learning.utils.supermask_utils   import set_model_task, cache_masks, set_num_tasks_learned


def replace_conv_with_maskconv(module, num_tasks, sparsity, mask_pretrained=False):
    """
    Recursively replace all nn.Conv2d layers in a module with MaskConv layers.
    Copies the original Conv2d weights into the 'pretrained' buffer of MaskConv.
    
    Args:
        module: The module to modify
        num_tasks: Number of tasks for MaskConv layers
        sparsity: Sparsity level for MaskConv layers
        mask_pretrained: Whether to apply masks to pretrained weights
    """
    for name, child_module in module.named_children():
        if isinstance(child_module, nn.Conv2d):
            # Create MaskConv with same parameters as original Conv2d
            mask_conv = MaskConv(
                in_channels=child_module.in_channels,
                out_channels=child_module.out_channels,
                kernel_size=child_module.kernel_size,
                stride=child_module.stride,
                padding=child_module.padding,
                dilation=child_module.dilation,
                groups=child_module.groups,
                bias=(child_module.bias is not None),
                num_tasks=num_tasks,
                sparsity=sparsity,
                mask_pretrained=mask_pretrained
            )
            
            # --- Copy original weights to pretrained buffer --- 
            pretrained_weights = child_module.weight.data.clone()
            mask_conv.register_buffer('pretrained', pretrained_weights)
            
            # Copy bias if it exists
            if child_module.bias is not None:
                mask_conv.bias.data.copy_(child_module.bias.data)
            
            # Set the new MaskConv layer, replacing the old Conv2d
            setattr(module, name, mask_conv)
        else:
            # Recursively apply to children
            replace_conv_with_maskconv(child_module, num_tasks, sparsity, mask_pretrained)


class UNet_SRCNN_Supermask(nn.Module):
    def __init__(self, pretrained_path=None, num_tasks=3, sparsity=0.05, mask_pretrained=False, unet_args={}):
        """
        initialise the UNet SRCNN Supermask model.
        
        Args:
            pretrained_path: Path to pretrained UNetCombinedModel weights 
            num_tasks: Number of tasks to support (default: 3 for low/medium/high pilot patterns)
            sparsity: Sparsity level for supermasks (default: 0.05)
            mask_pretrained: Whether to apply masks to pretrained weights (default: False)
            unet_args: Arguments to pass to UNetModel constructor
        """
        super(UNet_SRCNN_Supermask, self).__init__()
                
        # initialise standard UNet and SRCNN models first
        # Default to 2 input channels for UNet (real + imag)
        unet_constructor_args = {'in_channels': 2}
        unet_constructor_args.update(unet_args)
        self.unet = UNetModel(**unet_constructor_args)
        self.srcnn = SRCNN()
        
        # Replace all Conv2d layers with MaskConv layers
        replace_conv_with_maskconv(self.unet, num_tasks, sparsity, mask_pretrained)
        replace_conv_with_maskconv(self.srcnn, num_tasks, sparsity, mask_pretrained)
        
        # Register number of tasks
        self.num_tasks = num_tasks
        self.sparsity  = sparsity
        
        # Register canonical pilot masks (these will be defined during forward pass)
        self.canonical_masks = {
            'low':    None,
            'medium': None,
            'high':   None
        }
        self.canonical_masks_initialised = False
        
        # Load pretrained weights if provided
        if pretrained_path:
            self._load_pretrained_weights(pretrained_path)
            
        # Cache masks after initialisation (if pretrained weights are loaded)
        cache_masks(self)
    
    def _load_pretrained_weights(self, pretrained_model_path):
        """
        Load weights from a pretrained UNetCombinedModel into the MaskConv layers.
        
        Args:
            pretrained_model_path: Path to the pretrained model checkpoint
        """
        try:
            # Load the state dict from the pretrained model
            checkpoint = torch.load(pretrained_model_path, map_location='cpu', weights_only=True)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                pretrained_state_dict = checkpoint['model_state_dict']
            else:
                pretrained_state_dict = checkpoint
                
            print(f"Loaded pretrained weights from {pretrained_model_path}")
            
            # Iterate through each MaskConv layer and set its pretrained weights
            mask_conv_count = 0
            for name, module in self.named_modules():
                if isinstance(module, MaskConv):
                    mask_conv_count += 1
                    # Find the corresponding Conv2d weights in the pretrained model
                    # Convert MaskConv path to expected Conv2d path in the original model
                    conv_name = name.replace('.', '.') + '.weight'
                    
                    if conv_name in pretrained_state_dict:
                        pretrained_weight = pretrained_state_dict[conv_name]
                        # Register the pretrained weight as a buffer
                        module.register_buffer('pretrained', pretrained_weight)
                        
                        # Reinitialise scores for all tasks based on the pretrained weights
                        # This gives a better starting point than random initialisation
                        for task_id in range(module.num_tasks):
                            # initialise with slightly perturbed version of pretrained weights
                            # to give each task a different starting point
                            perturbation = torch.randn_like(pretrained_weight) * 0.01
                            module.scores[task_id].data = pretrained_weight.abs() + perturbation
                            
                        print(f"Set pretrained weights and initialised scores for {name}")
                    else:
                        print(f"Warning: Could not find pretrained weights for {name} at {conv_name}")
                        
            # Also load the bias terms directly into the MaskConv layers
            bias_count = 0
            for name, module in self.named_modules():
                if isinstance(module, MaskConv) and module.bias is not None:
                    bias_name = name.replace('.', '.') + '.bias'
                    if bias_name in pretrained_state_dict:
                        module.bias.data.copy_(pretrained_state_dict[bias_name])
                        bias_count += 1
                        
            print(f"Successfully updated {mask_conv_count} MaskConv layers and {bias_count} bias terms")
                        
        except Exception as e:
            print(f"Error loading pretrained weights: {e}")
            import traceback
            traceback.print_exc()
    
    def _initialise_canonical_masks(self, height, width, device):
        """
        initialise the canonical pilot masks for the three patterns (low, medium, high).
        
        Args:
            height: Height of the channel grid
            width: Width of the channel grid
            device: Device to create masks on
        """
        if self.canonical_masks_initialised:
            return
        
        print(f"\n--- initialising Canonical Masks ---")
        print(f"Grid dimensions: {height}x{width}")
        
        # following patterns are the same as data generation script

        # Create low density pattern mask 
        mask_low = torch.zeros((height, width), dtype=torch.bool, device=device)
        # Set pattern: Symbol 13 in slots 0 and 2, on SCs 0, 12, 24, 36, 48, 60
        sc_indices = np.arange(0, height, 12)
        for block_start in range(0, width, 56):  # Assuming 4 slots of 14 symbols each
            if block_start + 13 < width:  # Slot 0, symbol 13
                mask_low[sc_indices, block_start + 13] = True
            if block_start + 41 < width:  # Slot 2, symbol 13
                mask_low[sc_indices, block_start + 41] = True
        
        # Create medium density pattern mask 
        mask_medium = torch.zeros((height, width), dtype=torch.bool, device=device)
        # Set pattern: Symbol 13 of every slot, on SCs 0, 12, 24, 36, 48, 60
        for block_start in range(0, width, 14):  # Every slot (14 symbols)
            if block_start + 13 < width:  # Symbol 13 in each slot
                mask_medium[sc_indices, block_start + 13] = True
        
        # Create high density pattern mask 
        mask_high = torch.zeros((height, width), dtype=torch.bool, device=device)
        # Set pattern: Symbols 6 and 13 of every slot, on SCs 0, 12, 24, 36, 48, 60
        for block_start in range(0, width, 14):  # Every slot (14 symbols)
            if block_start + 6 < width:  # Symbol 6 in each slot
                mask_high[sc_indices, block_start + 6] = True
            if block_start + 13 < width:  # Symbol 13 in each slot
                mask_high[sc_indices, block_start + 13] = True
        
        # Print debug info about masks
        print(f"Low density mask: {torch.sum(mask_low).item()} pilot points")
        print(f"Medium density mask: {torch.sum(mask_medium).item()} pilot points")
        print(f"High density mask: {torch.sum(mask_high).item()} pilot points")
        
        # Check for overlap between patterns (sanity check)
        low_medium_overlap = torch.sum(mask_low & mask_medium).item()
        low_high_overlap = torch.sum(mask_low & mask_high).item()
        medium_high_overlap = torch.sum(mask_medium & mask_high).item()
        
        print(f"Pattern overlaps: Low-Medium: {low_medium_overlap}, Low-High: {low_high_overlap}, Medium-High: {medium_high_overlap}")
        
        # Register the masks
        self.canonical_masks['low']      = mask_low
        self.canonical_masks['medium']   = mask_medium
        self.canonical_masks['high']     = mask_high
        self.canonical_masks_initialised = True
        print("Canonical masks initialised successfully.")
            
    def _determine_task_from_mask(self, input_mask):
        # Ensure input_mask is boolean
        input_mask_bool = input_mask > 0.5 if not input_mask.dtype == torch.bool else input_mask
        
        # Validate canonical masks initialisation
        if not self.canonical_masks_initialised:
            print("WARNING: Canonical masks not initialised yet! initialising with input mask dimensions...")
            height, width = input_mask.shape
            self._initialise_canonical_masks(height, width, input_mask.device)
        
        # --- Count total active pilot points in input mask ---
        input_points = torch.sum(input_mask_bool).item()
        
        # --- Calculate match metrics for each pattern ---
        patterns = {'low': 0, 'medium': 1, 'high': 2}
        match_metrics = {}
        
        for pattern_name, task_id in patterns.items():
            pattern_mask = self.canonical_masks[pattern_name]
            pattern_total = torch.sum(pattern_mask).item()
            
            # Calculate overlap (true positives)
            true_positives = torch.sum(input_mask_bool & pattern_mask).item()
            
            # Calculate precision: What percentage of input points match this pattern?
            precision = (true_positives / input_points) if input_points > 0 else 0
            
            # Calculate recall: What percentage of this pattern's points are in the input?
            recall = (true_positives / pattern_total) if pattern_total > 0 else 0
            
            # Calculate F1 score: harmonic mean of precision and recall
            f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            # Exact match: input contains exactly the pattern's points (no more, no less)
            exact_match = (true_positives == pattern_total) and (input_points == pattern_total)
            
            match_metrics[pattern_name] = {
                'task_id': task_id,
                'true_positives': true_positives,
                'pattern_total': pattern_total,
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score,
                'exact_match': exact_match
            }
        
        # --- Remove detailed logging for debugging ---
        # Only print minimally needed information, not detailed metrics
        
        # --- Make determination based on metrics ---
        
        # First check for exact matches
        for pattern, metrics in match_metrics.items():
            if metrics['exact_match']:
                return metrics['task_id']
        
        # Then check for high F1 scores (strong matches)
        best_f1 = 0
        best_pattern = None
        for pattern, metrics in match_metrics.items():
            if metrics['f1_score'] > best_f1:
                best_f1 = metrics['f1_score']
                best_pattern = pattern
        
        # If we have a strong match with F1 > 0.9, use it
        if best_f1 > 0.9:
            return match_metrics[best_pattern]['task_id']
        
        # If no strong match, look at recall (which pattern is most completely represented)
        best_recall = 0
        best_pattern = None
        for pattern, metrics in match_metrics.items():
            if metrics['recall'] > best_recall:
                best_recall = metrics['recall']
                best_pattern = pattern
                
        if best_recall > 0.9:
            return match_metrics[best_pattern]['task_id']
        
        # Fallback: use the pattern with highest F1 score regardless of threshold
        if best_pattern:
            return match_metrics[best_pattern]['task_id']
            
        # Last resort: return task 0 (low density) as default
        print("WARNING: Could not determine task from pilot mask! Defaulting to task 0.")
        return 0

            
    def forward(self, x):
        """
        Forward pass through the UNet SRCNN Supermask model.
        
        Args:
            x: Input tensor of shape [batch_size, channels, height, width]
               where channels=3 (real, imag, pilot mask)
        
        Returns:
            Output tensor after passing through UNet and SRCNN with appropriate masks
        """
        if x.shape[1] != 3:
            raise ValueError(f"Expected input with 3 channels (real, imag, pilot mask), got {x.shape[1]} channels")
        
        # Extract pilot mask and data channels
        data       = x[:, :2, :, :]  # Channels 0,1 are real/imag
        pilot_mask = x[:,  2, :, :]  # Channel 2 is the pilot mask
        
        # Initialise canonical masks if not initialised yet
        if not self.canonical_masks_initialised:
            height, width = pilot_mask.shape[1], pilot_mask.shape[2]
            # Get device from input tensor for consistency
            device = x.device
            print(f"Initialising canonical masks on device: {device}")
            self._initialise_canonical_masks(height, width, device)
            
        # Determine task from the first sample's mask
        # Assuming all samples in batch have same mask pattern
        first_mask = pilot_mask[0]
        task_id = self._determine_task_from_mask(first_mask)
        
        # Fallback if detection fails
        if task_id < 0:
            print(f"WARNING: Unable to determine task from pilot mask! Using default task 0.")
            task_id = 0
        
        # Set task ID for all MaskConv layers
        set_model_task(self, task_id)
        
        # Pass data through UNet
        denoised = self.unet(data)
        
        # Pass denoised output through SRCNN
        output = self.srcnn(denoised)
        
        return output
    
    def cache_all_task_masks(self):
        """Cache the masks for all tasks"""
        cache_masks(self)
    
    def set_num_tasks_learned(self, num_tasks):
        """Set the number of tasks learned"""
        set_num_tasks_learned(self, num_tasks)
    
    def count_parameters(self):
        """Count the number of trainable parameters in the model"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)