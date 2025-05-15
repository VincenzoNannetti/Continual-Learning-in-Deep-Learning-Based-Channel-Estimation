"""
Filename: standard_training/datasets/dataset_utils.py
Author: Vincenzo Nannetti
Date: 04/03/2025
Description: Dataset utilities for loading, normalisation, and preprocessing

Dependencies:
    - PyTorch
    - numpy
"""
import torch
from torch.utils.data import DataLoader, random_split, Subset, Dataset

# Import dataset classes
from .standard_dataset import StandardDataset

SUPPORTED_DATASETS = {
    "standard": StandardDataset,
    "channel": StandardDataset  # Alias for backward compatibility
}

# --- Normalisation Helper Functions ---

def calculate_zscore_params(dataset):
    """Calculates mean and std for z-score normalisation across the entire dataset.
    Assumes dataset __getitem__ returns (input, target) tensors with shape (C, H, W).
    Calculates stats per channel based on the input tensors.
    """
    print(f"Calculating Z-score parameters from {len(dataset)} samples...")
    # Important: Use num_workers=0 here to avoid multiprocessing issues with stats calculation
    temp_loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)
    
    all_inputs = []
    # only need the inputs for normalisation
    for inputs, _ in temp_loader: 
        # Always use just the first 2 channels (real & imaginary) if more are present
        if inputs.shape[1] > 2:
            inputs = inputs[:, :2, :, :]
        all_inputs.append(inputs)
        
    if not all_inputs:
        raise ValueError("No data found to calculate Z-score parameters.")
        
    all_inputs_tensor = torch.cat(all_inputs, dim=0)
    mean = torch.mean(all_inputs_tensor, dim=(0, 2, 3)) # Mean per channel
    std  = torch.std(all_inputs_tensor, dim=(0, 2, 3))  # Std per channel
    
    # Avoid division by zero
    std[std == 0] = 1e-8
    return mean, std

def calculate_minmax_params(dataset):
    """Calculates min and max for min-max normalization across the entire dataset.
    Assumes dataset __getitem__ returns (input, target) tensors with shape (C, H, W).
    Calculates stats per channel based on the input tensors.
    """
    print(f"Calculating MinMax parameters from {len(dataset)} samples...")
    # Important: Use num_workers=0 here to avoid multiprocessing issues with stats calculation
    temp_loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)
    
    min_val_channel = None
    max_val_channel = None

    first_batch = True
    for inputs, _ in temp_loader:
        # Always use just the first 2 channels (real & imaginary) if more are present
        if inputs.shape[1] > 2:
            inputs = inputs[:, :2, :, :]
            
        # inputs shape (B, C, H, W)
        # Find min/max across batch, height, width dimensions for each channel
        batch_min = torch.amin(inputs, dim=(0, 2, 3)) # Shape (C)
        batch_max = torch.amax(inputs, dim=(0, 2, 3)) # Shape (C)

        if first_batch:
            min_val_channel = batch_min
            max_val_channel = batch_max
            first_batch = False
        else:
            min_val_channel = torch.minimum(min_val_channel, batch_min)
            max_val_channel = torch.maximum(max_val_channel, batch_max)
            
    if min_val_channel is None or max_val_channel is None:
        raise ValueError("No data found to calculate MinMax parameters.")

    return min_val_channel, max_val_channel

def apply_zscore(tensor, mean_v, std_v):
    """Applies z-score normalisation to the first 2 channels.

    For inputs with more than 2 channels, only normalises the first 2 channels
    and leaves any additional channels untouched.
    Assumes tensor shape is (C, H, W) and mean_v, std_v have shape (2,).
    """
    result = tensor.clone()
    num_channels = tensor.shape[0]

    # Ensure mean/std are correctly shaped (2, 1, 1) for broadcasting
    mean_reshaped = mean_v.view(2, 1, 1)
    std_safe = torch.max(std_v.view(2, 1, 1), torch.tensor(1e-6, device=tensor.device))

    if num_channels >= 2:
        # Apply normalization only to the first 2 channels
        result[:2] = (tensor[:2] - mean_reshaped) / std_safe
        # Channels from index 2 onwards remain unchanged in the cloned 'result'
    elif num_channels == 1:
        # Handle 1-channel case: Use first element of mean/std
        # Note: This assumes the first element of mean/std is appropriate for a 1-channel input.
        # If mean/std were calculated differently for 1-channel, this needs adjustment.
        print(f"Warning: Applying z-score to 1-channel tensor using first element of 2-channel stats.")
        mean_reshaped_1 = mean_v[0].view(1, 1, 1)
        std_safe_1 = torch.max(std_v[0].view(1, 1, 1), torch.tensor(1e-6, device=tensor.device))
        result = (tensor - mean_reshaped_1) / std_safe_1
    # else: num_channels == 0 - do nothing

    return result

def apply_minmax(tensor, min_val, max_val):
    """Applies min-max normalisation using pre-calculated min and max.
    
    For inputs with more than 2 channels, only normalises the first 2 channels
    and leaves any additional channels untouched.
    Assumes tensor shape is (C, H, W) and min_val, max_val have shape (2,).
    """
    result = tensor.clone()
    num_channels = tensor.shape[0]

    # Ensure min/max are correctly shaped (2, 1, 1) for broadcasting
    min_reshaped = min_val.view(2, 1, 1)
    max_reshaped = max_val.view(2, 1, 1)
    
    denominator = max_reshaped - min_reshaped
    # Add small epsilon where denominator is zero to avoid NaN/Inf
    denominator = torch.max(denominator, torch.tensor(1e-6, device=tensor.device)) 

    if num_channels >= 2:
        # Apply normalization only to the first 2 channels
        result[:2] = (tensor[:2] - min_reshaped) / denominator
        # Channels from index 2 onwards remain unchanged
    elif num_channels == 1:
        # Handle 1-channel case: Use first element of min/max
        print(f"Warning: Applying min-max to 1-channel tensor using first element of 2-channel stats.")
        min_reshaped_1 = min_val[0].view(1, 1, 1)
        max_reshaped_1 = max_val[0].view(1, 1, 1)
        denominator_1 = torch.max(max_reshaped_1 - min_reshaped_1, torch.tensor(1e-6, device=tensor.device))
        result = (tensor - min_reshaped_1) / denominator_1
    # else: num_channels == 0 - do nothing

    return result

# --- Dataset Wrapper for Normalisation --- 
class NormalisedDatasetWrapper(Dataset):
    """Applies pre-calculated normalisation to a raw dataset subset."""
    def __init__(self, raw_dataset_subset, norm_type, norm_params_input, norm_params_target=None):
        self.raw_dataset_subset = raw_dataset_subset
        self.norm_type          = norm_type
        self.norm_params_input  = norm_params_input  # e.g., (mean, std) or (min, max)
        self.norm_params_target = norm_params_target # e.g., (mean, std) or (min, max) or None

    def __len__(self):
        return len(self.raw_dataset_subset)

    def __getitem__(self, idx):
        # Get raw input and target tensors
        raw_input, raw_target = self.raw_dataset_subset[idx]
        
        normalised_input  = raw_input #  Default if no normalisation
        normalised_target = raw_target

        # Apply normalisation to input
        if self.norm_params_input is not None:
            if self.norm_type == "zscore":
                mean_in, std_in = self.norm_params_input
                normalised_input = apply_zscore(raw_input, mean_in, std_in)
            elif self.norm_type == "minmax":
                min_in, max_in = self.norm_params_input
                normalised_input = apply_minmax(raw_input, min_in, max_in)
        
        # Apply normalisation to target (optional, based on norm_params_target)
        if self.norm_params_target is not None:
            if self.norm_type == "zscore":
                mean_tgt, std_tgt = self.norm_params_target
                normalised_target = apply_zscore(raw_target, mean_tgt, std_tgt)
            elif self.norm_type == "minmax":
                min_tgt, max_tgt = self.norm_params_target
                normalised_target = apply_minmax(raw_target, min_tgt, max_tgt)
        
        return normalised_input, normalised_target

# --- Main Data Loading Function --- 
def load_data(config, mode='train'):
    """Loads datasets, performs train/val/test split, calculates normalisation
       params from training set, applies normalisation, and creates dataloaders."""
    data_config     = config.get('data', {})
    training_config = config.get('training', {})

    # --- Determine Dataset Type ---
    print("\n" + "-"*60)
    print("DATASET TYPE")
    print("-"*60)
    dataset_type = data_config.get('dataset_type', 'standard')

    if dataset_type not in SUPPORTED_DATASETS:
        raise ValueError(f"Unsupported dataset_type: '{dataset_type}'. Supported: {list(SUPPORTED_DATASETS.keys())}")

    # Instantiate the dataset class
    DatasetClass = SUPPORTED_DATASETS[dataset_type]
    print(f"Using dataset type: {dataset_type}")

    # --- Determine Data Directory --- 
    print("\n" + "-"*60)
    print("DATA DIRECTORY")
    print("-"*60)
    
    selected_data_dir = data_config.get('data_dir', data_config.get('data_dir_self'))
    if not selected_data_dir: 
        raise ValueError("'data_dir' or 'data_dir_self' must be specified in config.")
    
    data_config['data_dir'] = selected_data_dir
    print(f"Selected data directory: {selected_data_dir}")

    # --- Instantiate FULL Dataset --- 
    print("\n" + "-"*60)
    print("DATASET INSTANTIATION")
    print("-"*60)
    try:
        full_raw_dataset = DatasetClass(data_config)
        print(f"Raw dataset loaded. Total samples: {len(full_raw_dataset)}")
    except Exception as e:
        print(f"Error instantiating raw dataset '{dataset_type}': {e}")
        raise

    # --- Split Dataset ---
    print("\n" + "-"*60)
    print("DATASET SPLITTING")
    print("-"*60)
    split_seed = config.get('framework', {}).get('seed', 42)
    total_size = len(full_raw_dataset)
    val_split  = data_config.get('validation_split', 0.15)
    test_split = data_config.get('test_split', 0.15)
    
    if not (0 <= val_split < 1 and 0 <= test_split < 1 and (val_split + test_split) < 1):
        raise ValueError("Invalid train/validation/test splits.")
    
    val_size   = int(val_split * total_size)
    test_size  = int(test_split * total_size)
    train_size = total_size - val_size - test_size
    print(f"Splitting raw data: Train={train_size}, Val={val_size}, Test={test_size} (Seed: {split_seed})")
    
    # Use Subset to avoid loading data multiple times if dataset loads all data at once
    indices       = list(range(total_size))
    train_indices = indices[:train_size]
    val_indices   = indices[train_size:train_size + val_size]
    test_indices  = indices[train_size + val_size:]

    train_dataset_raw = Subset(full_raw_dataset, train_indices)
    val_dataset_raw   = Subset(full_raw_dataset, val_indices)
    test_dataset_raw  = Subset(full_raw_dataset, test_indices) if test_size > 0 else None

    # --- Calculate Normalisation Params from Training Set --- 
    print("\n" + "-"*60)
    print("NORMALIZATION")
    print("-"*60)
    norm_type = data_config.get('normalisation', 'none').lower()
    norm_params_input  = None
    normalise_target   = data_config.get('normalise_target', True) 
    norm_params_target = None

    if train_dataset_raw is None or len(train_dataset_raw) == 0:
        print("Warning: No training data available to calculate normalisation parameters. Skipping normalisation.")
        norm_type = 'none' 
    
    elif norm_type == "zscore":
        print("Calculating Z-score parameters from training set...")
        mean_in, std_in = calculate_zscore_params(train_dataset_raw) 
        norm_params_input = (mean_in, std_in)
        if normalise_target:
            print("  Normalising target using noisy Z-score statistics.")
            norm_params_target = (mean_in, std_in) 
    elif norm_type == "minmax":
        print("Calculating MinMax parameters from training set...")
        min_in, max_in = calculate_minmax_params(train_dataset_raw)
        norm_params_input = (min_in, max_in)
        if normalise_target:
            print("  Normalising target using noisy MinMax statistics.")
            norm_params_target = (min_in, max_in)
    elif norm_type == "none":
        print("Normalisation set to 'none'. Skipping.")
    else:
        print(f"Warning: Unsupported normalisation type '{norm_type}'. Skipping.")
        norm_type = 'none' # Treat unsupported as none

    # --- Wrap Datasets for Normalisation ---
    train_dataset_norm = NormalisedDatasetWrapper(train_dataset_raw, norm_type, norm_params_input, norm_params_target)
    val_dataset_norm   = NormalisedDatasetWrapper(val_dataset_raw, norm_type, norm_params_input, norm_params_target)
    test_dataset_norm  = NormalisedDatasetWrapper(test_dataset_raw, norm_type, norm_params_input, norm_params_target) if test_dataset_raw else None

    # --- Create DataLoaders ---
    print("\n" + "-"*60)
    print("DATALOADER CREATION")
    print("-"*60)
    batch_size  = training_config.get('batch_size', 64)
    num_workers = data_config.get('num_workers', 0)
    pin_memory  = config.get('hardware', {}).get('device', 'cpu') == 'cuda'

    dataloaders = {}
    dataloaders['train'] = DataLoader(train_dataset_norm, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory, drop_last=True)
    dataloaders['val']   = DataLoader(val_dataset_norm, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    
    if test_dataset_norm:
        dataloaders['test'] = DataLoader(test_dataset_norm, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    else:
        print("No test split defined, test dataloader will not be created.")

    print(f"Final dataloader keys created: {list(dataloaders.keys())}")
    
    # Return normalisation info along with dataloaders
    norm_info = {
        "type": norm_type,
        "params_input": norm_params_input,
        "params_target": norm_params_target
    }

    print(f"Configuring dataloaders specifically for mode '{mode}'")

    return dataloaders, norm_info
