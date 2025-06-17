import torch
import torch.nn as nn
import os # Added for path existence check

# Import the sub-models
# Ensure these relative paths are correct based on your file structure
from .dncnn import DnCNN
from .srcnn import SRCNN

class CombinedModel_SRCNNDnCNN(nn.Module):
    """
    Combined model using DnCNN for denoising and SRCNN for super-resolution.
    The order of operations (DnCNN first or SRCNN first) can be configured.
    """
    def __init__(self,
                 order='dncnn_first',
                 pretrained_dncnn_path=None,
                 pretrained_srcnn_path=None,
                 dncnn_args={}, # Pass args if DnCNN needs them, e.g., num_channels
                 srcnn_args={}): # Pass args if SRCNN needs them
        """
        Initialise the combined SRCNN-DnCNN model.

        Args:
            order (str): Order of operations. Either 'dncnn_first' or 'srcnn_first'.
            pretrained_dncnn_path (str, optional): Path to pretrained DnCNN weights (.pth file).
            pretrained_srcnn_path (str, optional): Path to pretrained SRCNN weights (.pth file).
            dncnn_args (dict): Arguments to pass to the DnCNN constructor.
            srcnn_args (dict): Arguments to pass to the SRCNN constructor.
        """
        super().__init__()

        if order not in ['dncnn_first', 'srcnn_first']:
            raise ValueError("Order must be either 'dncnn_first' or 'srcnn_first'")
        self.order = order

        # Initialise the sub-models
        # Assumes DnCNN and SRCNN constructors are compatible with defaults or passed args
        # Ensure channel counts match between models based on order
        self.dncnn = DnCNN(**dncnn_args)
        self.srcnn = SRCNN(**srcnn_args)

        # Load pretrained weights if provided
        self._load_pretrained(self.dncnn, pretrained_dncnn_path, "DnCNN")
        self._load_pretrained(self.srcnn, pretrained_srcnn_path, "SRCNN")

    def _load_pretrained(self, model, path, model_name):
        """Helper function to load state dict."""
        if path:
            if not os.path.exists(path):
                 print(f"Warning: Pretrained {model_name} weights not found at {path}. Skipping load.")
                 return
            print(f"Loading pretrained {model_name} weights from: {path}")
            try:
                # Load to CPU first to avoid GPU memory issues if the checkpoint was saved on GPU
                state_dict = torch.load(path, map_location='cpu') 

                # Handle checkpoints saved with 'model_state_dict' key (common practice)
                if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
                    state_dict = state_dict['model_state_dict']

                # Handle 'module.' prefix if saved from DataParallel/DistributedDataParallel
                if all(key.startswith('module.') for key in state_dict.keys()):
                    print(f"  Detected 'module.' prefix in {model_name} state_dict keys. Removing it.")
                    state_dict = {k.partition('module.')[2]: v for k, v in state_dict.items()}

                # Load the state dict. Using strict=True is generally recommended first.
                # If it fails, you might consider strict=False, but investigate mismatches.
                load_result = model.load_state_dict(state_dict, strict=True) 
                print(f"Successfully loaded pretrained {model_name} weights. Load result: {load_result}")
            except FileNotFoundError:
                 print(f"Error: Pretrained {model_name} weights file not found at {path}.")
            except Exception as e:
                 print(f"Error loading {model_name} state dict from {path}: {e}")
                 print("  Consider checking the checkpoint structure and keys.")


    def forward(self, x):
        """
        Forward pass through the combined model based on the specified order,
        correctly applying the DnCNN residual learning principle.
        """
        if self.order == 'dncnn_first':
            # Input -> Predict Noise (DnCNN) -> Denoise -> SRCNN -> Output
            predicted_noise = self.dncnn(x)
            denoised_x = x - predicted_noise # Apply DnCNN residual subtraction
            output = self.srcnn(denoised_x)
        elif self.order == 'srcnn_first':
            # Input -> SRCNN -> Predict Noise (DnCNN) -> Denoise -> Output
            processed = self.srcnn(x)
            predicted_noise_on_processed = self.dncnn(processed)
            output = processed - predicted_noise_on_processed # Apply DnCNN residual subtraction
        else:
             # This case should not be reached due to __init__ check
             raise ValueError(f"Invalid order specified: {self.order}")

        return output

    def freeze_dncnn(self):
        """Freeze the DnCNN weights."""
        print("Freezing DnCNN parameters.")
        for param in self.dncnn.parameters():
            param.requires_grad = False

    def freeze_srcnn(self):
        """Freeze the SRCNN weights."""
        print("Freezing SRCNN parameters.")
        for param in self.srcnn.parameters():
            param.requires_grad = False

    def unfreeze_all(self):
        """Unfreeze all weights."""
        print("Unfreezing all parameters.")
        for param in self.parameters():
            param.requires_grad = True

    def count_parameters(self):
        """Count the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_submodule_parameters(self):
        """Count parameters for each submodule separately."""
        dncnn_params = sum(p.numel() for p in self.dncnn.parameters())
        srcnn_params = sum(p.numel() for p in self.srcnn.parameters())

        return {
            'dncnn': dncnn_params,
            'srcnn': srcnn_params,
            'total': dncnn_params + srcnn_params
        }