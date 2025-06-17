"""
Main model definitions for the LoRA-based continual learning framework.

This module contains:
- The backbone model architecture (UNet, SRCNN, and their combination).
- Domain-specific Batch Normalization layer.
- The final LoRA-integrated model (`UNet_SRCNN_LoRA`) that wraps the
  backbone and injects trainable adapters.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Any, List

# Assuming config.py and lora.py are in the same directory
from .lora import LoRAConv2d
from .config import ExperimentConfig


# --- Domain-Specific Batch Normalization ---

class DomainBatchNorm2d(nn.Module):
    """
    A Batch Normalization layer that maintains separate running statistics
    and trainable parameters for each domain/task.
    """
    def __init__(self, num_features: int, num_tasks: int, default_task_id: str = None):
        super().__init__()
        self.num_features = num_features
        self.num_tasks = num_tasks
        self.bns = nn.ModuleDict({
            str(i): nn.BatchNorm2d(num_features) for i in range(num_tasks)
        })
        # Set active task to first available task or provided default
        if default_task_id is not None and default_task_id in self.bns:
            self.active_task_id = default_task_id
        else:
            self.active_task_id = str(min(range(num_tasks)))  # First available task

    def set_active_task(self, task_id: str):
        if task_id not in self.bns:
            raise ValueError(f"Task ID '{task_id}' not found in DomainBatchNorm2d layers.")
        self.active_task_id = task_id

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bns[self.active_task_id](x)

    def extra_repr(self) -> str:
        return f'num_features={self.num_features}, num_tasks={self.num_tasks}, active_task_id={self.active_task_id}'


# --- Backbone Model Definitions (Copied from standard_training_2) ---

class UNetModel(nn.Module):
    """
    UNet model - exact copy from standard_training_2/models/unet.py
    """
    def __init__(self, in_channels=2, base_features=16, use_batch_norm=False, 
                 depth=3, activation='leakyrelu', use_leaky_relu=False, leaky_slope=0.01, verbose=False):
        super(UNetModel, self).__init__()
        
        # Validation
        assert depth >= 1, "UNetModel depth must be at least 1"
        assert base_features > 0, "base_features must be positive"
        
        self.use_batch_norm = use_batch_norm
        self.depth = depth
        self.in_channels = in_channels
        self.base_features = base_features
                    
        self.activation_type = activation.lower()
        
        # Determine activation function
        if self.activation_type == 'leakyrelu':
            self.activation = nn.LeakyReLU(negative_slope=leaky_slope)
        elif self.activation_type == 'gelu':
            self.activation = nn.GELU()
        elif self.activation_type == 'swish' or self.activation_type == 'silu':
            self.activation = nn.SiLU()
        elif self.activation_type == 'relu':
            self.activation = nn.ReLU()
        else:
            self.activation = nn.ReLU()  # Default fallback
            self.activation_type = 'relu'
            
        if verbose:
            print(f"UNet using {self.activation_type.upper()} activation with depth {depth}")
        
        # Build encoder layers with proper powers of 2
        self.encoder_convs = nn.ModuleList()
        self.encoder_bns = nn.ModuleList() if use_batch_norm else None
        
        current_channels = in_channels
        self.encoder_channels = []  # Track encoder output channels for skip connections
        
        for i in range(depth):
            out_channels = base_features * (2 ** i)  # 16, 32, 64, 128, ...
            conv = nn.Conv2d(current_channels, out_channels, 3, padding=1)
            self.encoder_convs.append(conv)
            self.encoder_channels.append(out_channels)
            
            if use_batch_norm:
                self.encoder_bns.append(nn.BatchNorm2d(out_channels))
                
            current_channels = out_channels
        
        # Bottleneck - one level deeper in channel dimension
        bottleneck_channels = base_features * (2 ** depth)  # Next power of 2
        self.bottleneck = nn.Conv2d(current_channels, bottleneck_channels, 3, padding=1)
        if use_batch_norm:
            self.bottleneck_bn = nn.BatchNorm2d(bottleneck_channels)
        
        # Build decoder layers with symmetric powers of 2
        self.decoder_convs = nn.ModuleList()
        self.decoder_reduce_convs = nn.ModuleList()  # To reduce concatenated channels back to powers of 2
        self.decoder_bns = nn.ModuleList() if use_batch_norm else None
        self.decoder_reduce_bns = nn.ModuleList() if use_batch_norm else None
        
        current_channels = bottleneck_channels
        
        for i in range(depth):
            # Calculate skip connection channels from corresponding encoder
            skip_channels = self.encoder_channels[-(i+1)]  # Reverse order
            concat_channels = current_channels + skip_channels  # After concatenation
            
            # Target output channels - symmetric powers of 2
            if i == depth - 1:  # Final layer
                target_channels = base_features  # Will be reduced to in_channels separately
            else:
                target_channels = base_features * (2 ** (depth - 1 - i))
            
            # First conv: reduce concatenated channels to target channels (maintaining powers of 2)
            reduce_conv = nn.Conv2d(concat_channels, target_channels, 3, padding=1)
            self.decoder_reduce_convs.append(reduce_conv)
            
            # Second conv: refine features (channel count stays the same)
            refine_conv = nn.Conv2d(target_channels, target_channels, 3, padding=1)
            self.decoder_convs.append(refine_conv)
            
            # Batch norm for both convs (except final output)
            if use_batch_norm:
                self.decoder_reduce_bns.append(nn.BatchNorm2d(target_channels))
                if i < depth - 1:  # No BN on final layer
                    self.decoder_bns.append(nn.BatchNorm2d(target_channels))
                else:
                    self.decoder_bns.append(None)  # Placeholder
                
            current_channels = target_channels
        
        # Final 1x1 conv to map base_features → in_channels
        self.final_conv = nn.Conv2d(base_features, in_channels, 1, padding=0)
    
    def forward(self, x):
        # Get original dimensions
        _, _, h, w = x.size()
        
        # Encoder path - store features for skip connections
        encoder_features = []
        current = x
        
        for i, conv in enumerate(self.encoder_convs):
            current = conv(current)
            
            if self.use_batch_norm and self.encoder_bns is not None:
                current = self.encoder_bns[i](current)
                
            current = self.activation(current)
            encoder_features.append(current)  # Store for skip connection
            
            # Apply pooling except for the last encoder layer
            if i < len(self.encoder_convs) - 1:
                current = F.max_pool2d(current, 2, 2)
        
        # Bottleneck
        current = self.bottleneck(current)
        if self.use_batch_norm and hasattr(self, 'bottleneck_bn'):
            current = self.bottleneck_bn(current)
        current = self.activation(current)
        
        # Decoder path with skip connections and symmetric channel reduction
        for i in range(self.depth):
            # Upsample to match the corresponding encoder feature size
            encoder_feature = encoder_features[-(i+1)]  # Get corresponding encoder feature
            current = F.interpolate(current, size=encoder_feature.size()[2:], mode='bilinear', align_corners=True)
            
            # Concatenate with skip connection
            current = torch.cat([current, encoder_feature], dim=1)
            
            # First conv: reduce concatenated channels to target power-of-2
            current = self.decoder_reduce_convs[i](current)
            if self.use_batch_norm and self.decoder_reduce_bns is not None:
                current = self.decoder_reduce_bns[i](current)
            current = self.activation(current)
            
            # Second conv: refine features
            current = self.decoder_convs[i](current)
            
            # Apply batch norm and activation (except for final layer)
            if i < self.depth - 1:
                if self.use_batch_norm and self.decoder_bns is not None and self.decoder_bns[i] is not None:
                    current = self.decoder_bns[i](current)
                current = self.activation(current)
        
        # Final 1x1 conv to map to output channels
        current = self.final_conv(current)
        
        # Ensure output has the same dimensions as input
        if current.size()[2:] != x.size()[2:]:
            current = F.interpolate(current, size=(h, w), mode='bilinear', align_corners=True)
        
        return current

class SRCNN(nn.Module):
    """
    SRCNN model - exact copy from standard_training_2/models/srcnn.py
    """
    def __init__(self, channels=[64, 32], kernels=[9, 1, 5]):
        super(SRCNN, self).__init__()
        
        # Ensure we have the right number of channels and kernels
        assert len(channels) == 2, "channels should contain exactly 2 values [conv1_out, conv2_out]"
        assert len(kernels) == 3, "kernels should contain exactly 3 values [conv1_kernel, conv2_kernel, conv3_kernel]"
        
        # Calculate padding to maintain spatial dimensions
        # For odd kernel sizes: padding = (kernel_size - 1) // 2
        pad1 = (kernels[0] - 1) // 2
        pad2 = (kernels[1] - 1) // 2
        pad3 = (kernels[2] - 1) // 2
        
        self.conv1 = nn.Conv2d(2, channels[0], kernel_size=kernels[0], padding=pad1)
        self.conv2 = nn.Conv2d(channels[0], channels[1], kernel_size=kernels[1], padding=pad2)
        self.conv3 = nn.Conv2d(channels[1], 2, kernel_size=kernels[2], padding=pad3)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x
        
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

class Backbone_UNet_SRCNN(nn.Module):
    """
    Combined UNet+SRCNN backbone model.
    """
    def __init__(self, unet_args: Dict[str, Any], srcnn_params: Dict[str, Any]):
        super().__init__()
        self.unet = UNetModel(**unet_args)
        
        # Extract and map SRCNN parameters correctly
        srcnn_channels = srcnn_params['srcnn_channels']
        srcnn_kernels = srcnn_params['srcnn_kernels']
        self.srcnn = SRCNN(channels=srcnn_channels, kernels=srcnn_kernels)

    def forward(self, x):
        denoised = self.unet(x)
        return self.srcnn(denoised)

# --- LoRA-Integrated Model ---

class UNet_SRCNN_LoRA(nn.Module):
    """
    The main model for the continual learning experiment. It wraps the
    backbone model and injects LoRA and domain-specific BN layers.
    """
    def __init__(self, config: ExperimentConfig):
        super().__init__()
        self.config = config

        # 1. Initialize backbone
        self.backbone = Backbone_UNet_SRCNN(
            unet_args=config.model.params.unet_args.model_dump(),
            srcnn_params=config.model.params.srcnn_params.model_dump()
        )

        # 2. Load pretrained weights for the entire backbone
        print(f"Loading pretrained backbone weights from: {config.model.pretrained_path}")
        try:
            checkpoint = torch.load(config.model.pretrained_path, map_location='cpu')
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            self.backbone.load_state_dict(state_dict, strict=config.model.strict_load)
            print("Backbone weights loaded successfully.")
        except Exception as e:
            print(f"Error loading backbone weights: {e}")
            raise

        # 3. Inject LoRA and Domain-Specific BN layers
        self._inject_adapters(self.backbone)
        
        # 4. Freeze all original backbone parameters
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # 5. Unfreeze LoRA and Domain-Specific BN parameters
        for name, param in self.named_parameters():
            # Unfreeze LoRA parameters (task_adapters)
            if 'task_adapters.' in name:
                param.requires_grad = True
            # Unfreeze Domain-Specific BN parameters
            elif 'bns.' in name:  # DomainBatchNorm2d parameters
                param.requires_grad = True
            # Unfreeze biases if configured (but respect LoRA layer's bias freezing)
            elif (config.model.params.lora_bias_trainable != 'none' and 'bias' in name and 
                  'task_adapters' not in name):  # Don't unfreeze conv biases in LoRA layers
                param.requires_grad = True

        print("Finished model setup. Trainable parameters are now LoRA adapters and BN layers.")

    def _inject_adapters(self, module: nn.Module, module_path: str = ""):
        """
        Recursively traverse the model and replace Conv2d layers with their
        LoRA-adapted versions, optionally with Domain-Specific BN.
        Skips layers where LoRA would be problematic (min(out_channels, in_channels*k*k) too small).
        """
        for name, child_module in module.named_children():
            current_path = f"{module_path}.{name}" if module_path else name
            
            if isinstance(child_module, nn.Conv2d):
                # This is a Conv2d layer we might want to adapt
                conv_layer = child_module
                
                # Check if LoRA is viable for this layer
                out_channels = conv_layer.out_channels
                in_channels = conv_layer.in_channels
                k_h, k_w = conv_layer.kernel_size
                k = in_channels * k_h * k_w
                min_dk = min(out_channels, k)
                
                # # Skip problematic output layers with very few output channels
                # skip_layer = False
                # if 'final_conv' in current_path or 'conv3' in current_path:
                #     # These are typically output layers with only 2 channels
                #     if out_channels <= 2:
                #         print(f"⚠️  Skipping LoRA for {current_path}: output layer with {out_channels} channels")
                #         skip_layer = True
                
                # # Additional constraint: skip if rank would be problematic
                # # For now, assume we want at least min_dk >= 4 for meaningful LoRA
                # if min_dk < 4 and not skip_layer:
                #     print(f"⚠️  Skipping LoRA for {current_path}: min(d,k)={min_dk} too small")
                #     skip_layer = True
                skip_layer = False
                if not skip_layer:
                    # Create the LoRA wrapper for the conv layer
                    lora_dropout = getattr(self.config.model.params, 'lora_dropout', 0.0)
                    lora_conv    = LoRAConv2d(conv_layer, lora_dropout=lora_dropout)
                    
                    # Check if we need to add domain-specific BN
                    if self.config.model.params.use_domain_specific_bn:
                        dbn = DomainBatchNorm2d(conv_layer.out_channels, self.config.data.tasks)
                        # Replace the original conv with a sequence of (LoRA-Conv -> DBN)
                        setattr(module, name, nn.Sequential(lora_conv, dbn))
                    else:
                        # Just replace with the LoRA-Conv layer
                        setattr(module, name, lora_conv)
                    
                    print(f"✅ Added LoRA to {current_path}: {out_channels}x{in_channels}x{k_h}x{k_w} (min_dk={min_dk})")
                else:
                    print(f"🔄 Keeping standard Conv2d for {current_path}")
            else:
                # Recurse into submodules
                self._inject_adapters(child_module, current_path)
    
    def add_task(self, task_id: str):
        """
        Adds adapters for a new task to all LoRA and Domain BN layers.
        """
        task_idx = int(task_id)
        r = self.config.model.params.task_lora_ranks[task_idx]
        alpha = self.config.model.params.task_lora_alphas[task_idx]
        
        print(f"\nAdding adapters for task {task_id} with r={r}, alpha={alpha}")
        for module in self.modules():
            if isinstance(module, LoRAConv2d):
                module.add_task_adapters(task_id, r, alpha)

    def set_active_task(self, task_id: Optional[str]):
        """
        Activates the adapters and BN layers for a specific task.
        """
        # print(f"Setting active task to: {task_id}")
        for module in self.modules():
            if isinstance(module, (LoRAConv2d, DomainBatchNorm2d)):
                module.set_active_task(task_id)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def trainable_parameters(self):
        """
        Returns an iterator over the trainable parameters of the model,
        which are the LoRA adapters and domain-specific BN parameters.
        """
        return filter(lambda p: p.requires_grad, self.parameters()) 