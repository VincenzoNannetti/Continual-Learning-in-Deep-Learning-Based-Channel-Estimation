import torch
import numpy as np
from torch.utils.data import Dataset

class PilotMaskInjectorWrapper(Dataset):
    def __init__(self, base_dataset, pilot_mask_numpy):
        """
        Wrapper to inject a pilot mask as the third channel to the input from a base dataset.

        Args:
            base_dataset (Dataset): The base dataset that returns (input_2ch, target_2ch) or 
                                    (input_2ch, target_2ch, original_target) from NormalisingDatasetWrapper.
                                    input_2ch is expected to be (2, H, W) as torch tensor or numpy array.
            pilot_mask_numpy (np.ndarray): The pilot mask to inject, shape (H, W), boolean or float.
        """
        self.base_dataset = base_dataset
        # Ensure pilot_mask is (H, W) and then convert to (1, H, W) tensor
        if pilot_mask_numpy.ndim != 2:
            raise ValueError(f"Pilot mask must be 2D (H, W), got shape {pilot_mask_numpy.shape}")
        self.pilot_mask_tensor = torch.from_numpy(pilot_mask_numpy.astype(np.float32)).unsqueeze(0) # Shape (1, H, W)

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        # Handle both 2-value and 3-value returns from base dataset
        base_data = self.base_dataset[idx]
        
        if len(base_data) == 2:
            # Standard case: (input, target)
            input_data, target_data = base_data
        elif len(base_data) == 3:
            # Handle NormalisingDatasetWrapper with return_original_target=True
            # Returns: (normalized_input, normalized_target, original_target)
            input_data, target_data, _ = base_data  # Use normalized target, ignore original target
        else:
            raise ValueError(f"PilotMaskInjectorWrapper expects base dataset to return 2 or 3 values, got {len(base_data)}")

        # Ensure input_data is a tensor and has 2 channels
        if not isinstance(input_data, torch.Tensor):
            input_data = torch.from_numpy(input_data.astype(np.float32))
        
        if input_data.ndim == 3 and input_data.shape[0] == 2: # Expected (2, H, W)
            # Ensure pilot mask matches spatial dimensions
            if self.pilot_mask_tensor.shape[1:] != input_data.shape[1:]:
                raise ValueError(f"Pilot mask spatial dimensions {self.pilot_mask_tensor.shape[1:]} "
                                 f"do not match input data dimensions {input_data.shape[1:]}")
            input_3ch = torch.cat((input_data, self.pilot_mask_tensor), dim=0) # Shape (3, H, W)
        elif input_data.ndim == 2 and self.pilot_mask_tensor.shape[1:] == input_data.shape: # Case: input is (H,W), target is (H,W) from some datasets
             # This case might occur if base_dataset returns single channel images.
             # For now, assuming base_dataset (StandardDataset) gives (2, H, W) for input.
             # If StandardDataset could return (H,W) for input/target and needs to be 2-channeled first,
             # that logic should be in StandardDataset or a wrapper before this one.
             # This wrapper strictly expects 2-channel input to make it 3-channel.
            raise ValueError(f"PilotMaskInjectorWrapper expects 2-channel input (2, H, W), got {input_data.shape}")
        else:
            raise ValueError(f"PilotMaskInjectorWrapper received unexpected input shape: {input_data.shape}. Expected (2, H, W).")

        # Ensure target is also a tensor
        if not isinstance(target_data, torch.Tensor):
            target_data = torch.from_numpy(target_data.astype(np.float32))
            
        return input_3ch, target_data 