"""
Data loading and preprocessing for the continual learning experiment.
"""
import os
import sys
import torch
import numpy as np
import scipy.io as sio
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, Any, Optional
import time
from pydantic import BaseModel

from .config import DataConfig, NormStats

# Add the project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.append(project_root)

# Now we can import from standard_training_2
from standard_training_2.noise_generation import add_cn_awgn
from standard_training_2.interpolate import interpolation

class ChannelDataset(Dataset):
    """
    Custom PyTorch Dataset for loading and preprocessing channel estimation data
    from .mat files. Follows the same pattern as the original StandardDataset.
    """
    def __init__(self, data_path: str, norm_stats: NormStats, snr: int = 20, 
                 normalise_target: bool = True, use_cache: bool = True, 
                 preprocessed_dir: str = "./data/preprocessed/", 
                 interpolation_kernel: str = "thin_plate_spline"):
        """
        Args:
            data_path (str): Path to the .mat file.
            norm_stats (NormStats): Pydantic model with mean/std for inputs and targets.
            snr (int): SNR for noise addition.
            normalise_target (bool): Whether to apply normalisation to the target.
            use_cache (bool): Whether to use cached preprocessed data.
            preprocessed_dir (str): Directory for preprocessed/cached data.
            interpolation_kernel (str): Interpolation kernel type.
        """
        super().__init__()
        
        self.data_path = data_path
        self.norm_stats = norm_stats
        self.normalise_target = normalise_target
        self.snr = snr
        self.preprocessed_dir = preprocessed_dir
        self.interpolation_kernel = interpolation_kernel
        
        # Create pilot mask (same as original)
        self.pilot_mask = np.zeros((72, 70), dtype=bool)
        self.pilot_mask[::12, ::7] = True
        
        # Load and process data
        self.inputs, self.targets = self.load_and_process_data(use_cache)

    def _get_cache_filename(self):
        """Generate cache filename based on dataset name, SNR, and interpolation settings."""
        base_name = os.path.splitext(os.path.basename(self.data_path))[0]
        cache_name = f"{base_name}_snr{self.snr}_{self.interpolation_kernel}.mat"
        cache_dir = os.path.join(self.preprocessed_dir, "tester_data")
        os.makedirs(cache_dir, exist_ok=True)
        return os.path.join(cache_dir, cache_name)

    def _load_cached_data(self):
        """Load cached processed data if available."""
        cache_path = self._get_cache_filename()
        
        print(f"[CACHE_LOAD] Attempting to load cache from: {cache_path}")
        if not os.path.exists(cache_path):
            print(f"[CACHE_LOAD] No cached data found at: {cache_path}")
            
            # Try to find existing preprocessed data that matches
            base_name = os.path.splitext(os.path.basename(self.data_path))[0]
            cache_dir = os.path.join(self.preprocessed_dir, "tester_data")
            
            # Look for files that match the pattern
            if os.path.exists(cache_dir):
                try:
                    files = os.listdir(cache_dir)
                    print(f"[CACHE_LOAD] Searching {len(files)} files in cache directory...")
                    
                    for filename in files:
                        if (filename.startswith(base_name) and 
                            f"snr{self.snr}" in filename and 
                            self.interpolation_kernel in filename and
                            filename.endswith('.mat')):
                            
                            alternative_path = os.path.join(cache_dir, filename)
                            print(f"[CACHE_LOAD] Found matching cache file: {alternative_path}")
                            try:
                                cached = sio.loadmat(alternative_path)
                                # For existing preprocessed files, use simpler extraction
                                if 'interpolated_data' in cached and 'ground_truth' in cached:
                                    interpolated_data = cached['interpolated_data']
                                    gt_2ch = cached['ground_truth']
                                    print(f"[CACHE_LOAD] Successfully loaded from alternative cache with {interpolated_data.shape[0]} samples!")
                                    return interpolated_data, gt_2ch
                            except Exception as e:
                                print(f"[CACHE_LOAD] Error loading alternative cache {filename}: {e}")
                                continue
                    print(f"[CACHE_LOAD] No matching cache files found for base_name='{base_name}', snr={self.snr}, kernel='{self.interpolation_kernel}'")
                except PermissionError as e:
                    print(f"[CACHE_LOAD] Permission error listing cache directory: {e}")
                except Exception as e:
                    print(f"[CACHE_LOAD] Error accessing cache directory: {e}")
            else:
                print(f"[CACHE_LOAD] Cache directory does not exist: {cache_dir}")
            
            return None
        
        print(f"[CACHE_LOAD] Loading cached data from: {cache_path}")
        try:
            cached = sio.loadmat(cache_path)
            
            # Verify cache integrity (same as original)
            if (cached['snr'][0][0] != self.snr or 
                str(cached['interpolation_kernel'][0]) != self.interpolation_kernel or
                str(cached['data_name'][0]) != os.path.basename(self.data_path)):
                print("Cache parameters don't match current settings. Regenerating...")
                return None
            
            # Extract data
            interpolated_data = cached['interpolated_data']
            gt_2ch = cached['ground_truth']
            
            print(f"[CACHE_LOAD] Successfully loaded cached data with {interpolated_data.shape[0]} samples!")
            return interpolated_data, gt_2ch
            
        except Exception as e:
            print(f"[CACHE_LOAD] Error loading cached data: {e}. Regenerating...")
            return None

    def _save_processed_data(self, interpolated_data, gt_2ch, noisy_before_interp):
        """Save processed data to cache."""
        cache_path = self._get_cache_filename()
        print(f"Saving processed data to cache: {cache_path}")
        
        # Save as .mat file for consistency (same format as original)
        cache_data = {
            'interpolated_data': interpolated_data,
            'ground_truth': gt_2ch,
            'noisy_before_interp': noisy_before_interp,
            'pilot_mask': self.pilot_mask,
            'snr': self.snr,
            'interpolation_kernel': self.interpolation_kernel,
            'data_name': os.path.basename(self.data_path)
        }
        
        sio.savemat(cache_path, cache_data)
        print(f"Cache saved successfully!")

    def load_and_process_data(self, use_cache: bool = True):
        """
        Load and process data following the original StandardDataset pattern.
        """
        # First, try to load from cache
        if use_cache:
            cached_result = self._load_cached_data()
            if cached_result is not None:
                print("[LOAD_DATA] Successfully loaded data from cache.")
                return cached_result
        
        # If no cache, process from scratch
        print("[LOAD_DATA] No valid cache found. Processing data from scratch...")
        
        # Load raw data
        print(f"Loading and processing data from: {self.data_path}")
        try:
            mat_data = sio.loadmat(self.data_path)
            if 'perfect_matrices' in mat_data:
                raw_data = mat_data['perfect_matrices']  # Shape: (N, 72, 70)
            else:
                raise KeyError("Expected key 'perfect_matrices' in the .mat file")
        except Exception as e:
            print(f"Error loading data from {self.data_path}: {e}")
            raise

        # Ensure correct shape and convert to complex
        if raw_data.ndim != 3 or raw_data.shape[1:] != (72, 70):
            raise ValueError("Expected raw data shape (N, 72, 70)")
        raw_data = raw_data.astype(np.complex64)

        # Add noise
        print(f"Adding AWGN noise with SNR = {self.snr} dB...")
        noisy_data = add_cn_awgn(raw_data, self.snr)
        
        # Verify SNR immediately after noise addition (before interpolation)
        signal_power = np.mean(np.abs(raw_data) ** 2)
        noise_power = np.mean(np.abs(noisy_data - raw_data) ** 2)
        actual_snr = 10 * np.log10(signal_power / noise_power)
        print(f"Actual SNR after noise addition: {actual_snr:.2f} dB (Target: {self.snr} dB)")

        # Convert to 2-channel: real and imaginary parts
        noisy_2ch = np.stack([np.real(noisy_data), np.imag(noisy_data)], axis=-1)  # (N, 72, 70, 2)
        gt_2ch = np.stack([np.real(raw_data), np.imag(raw_data)], axis=-1)

        # Create batch-wise pilot mask
        pilot_mask_batch = np.broadcast_to(self.pilot_mask, (noisy_2ch.shape[0], 72, 70))

        # Apply interpolation (this is the expensive part)
        print(f"[LOAD_DATA] Starting interpolation for {noisy_2ch.shape[0]} samples... This might take a while.")
        start_time = time.time()
        interpolated_data = interpolation(noisy_2ch, pilot_mask_batch, kernel=self.interpolation_kernel)
        end_time = time.time()
        print(f"[LOAD_DATA] Interpolation finished. Time taken: {end_time - start_time:.2f} seconds.")

        # Save to cache for next time
        if use_cache:
            self._save_processed_data(interpolated_data, gt_2ch, noisy_2ch)

        return interpolated_data, gt_2ch

    def __len__(self) -> int:
        return self.inputs.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves and preprocesses a single data sample.
        """
        # Get data: inputs are (72, 70, 2), targets are (72, 70, 2)
        x_np = self.inputs[idx].astype(np.float32)  # (72, 70, 2)
        y_np = self.targets[idx].astype(np.float32)  # (72, 70, 2)

        # Convert to tensor and permute from (H, W, C) to (C, H, W)
        inputs = torch.tensor(x_np, dtype=torch.float32).permute(2, 0, 1)  # (2, 72, 70)
        targets = torch.tensor(y_np, dtype=torch.float32).permute(2, 0, 1)  # (2, 72, 70)
        
        # Apply normalisation
        mean_i = torch.tensor(self.norm_stats.mean_inputs).view(2, 1, 1)
        std_i = torch.tensor(self.norm_stats.std_inputs).view(2, 1, 1)
        inputs = (inputs - mean_i) / (std_i + 1e-8)

        if self.normalise_target:
            mean_t = torch.tensor(self.norm_stats.mean_targets).view(2, 1, 1)
            std_t = torch.tensor(self.norm_stats.std_targets).view(2, 1, 1)
            targets = (targets - mean_t) / (std_t + 1e-8)
            
        return inputs, targets

def get_dataloaders(
    task_id: str,
    config: DataConfig,
    batch_size: int,
    norm_stats: Optional[NormStats] = None
) -> Tuple[DataLoader, DataLoader]:
    """
    Creates training and validation DataLoaders for a specific task.

    Args:
        task_id (str): The ID of the task to load data for.
        config (DataConfig): The data configuration object.
        batch_size (int): The batch size for the DataLoaders.
        norm_stats (Optional[NormStats]): Normalisation statistics. If None,
            normalisation is skipped (e.g., for initial calculation).

    Returns:
        A tuple containing the training and validation DataLoaders.
    """
    if norm_stats is None:
        raise ValueError("Normalisation statistics must be provided to the data loader.")

    task_idx = int(task_id)
    task_params = config.tasks_params[task_idx]
    file_path = os.path.join(config.data_dir, task_params.data_name)

    print(f"Loading data for task {task_id} from: {file_path}")

    # Create the full dataset
    full_dataset = ChannelDataset(
        data_path=file_path,
        norm_stats=norm_stats,
        snr=task_params.snr,
        normalise_target=config.normalise_target,
        use_cache=True,
        preprocessed_dir=config.preprocessed_dir,
        interpolation_kernel=config.interpolation
    )

    # Split dataset into training and validation
    indices = list(range(len(full_dataset)))
    train_indices, val_indices = train_test_split(
        indices,
        test_size=config.validation_split,
        random_state=42 # Use a fixed random state for reproducibility
    )

    train_subset = Subset(full_dataset, train_indices)
    val_subset = Subset(full_dataset, val_indices)

    # Create DataLoaders
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )

    print(f"Created DataLoaders for task {task_id}:")
    print(f"  - Training samples: {len(train_subset)}")
    print(f"  - Validation samples: {len(val_subset)}")
    
    return train_loader, val_loader

def get_norm_stats_from_checkpoint(checkpoint_path: str) -> Optional[NormStats]:
    """
    Extracts normalisation statistics from a model checkpoint file.
    It expects the stats to be under the 'config' -> 'data' -> 'norm_stats' keys.
    This version includes a robust fix for stats saved in various nested formats.
    """
    try:
        # The warning is unavoidable with this loading method, but it's safe for this project.
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        if 'config' in checkpoint and 'data' in checkpoint['config'] and 'norm_stats' in checkpoint['config']['data']:
            stats_dict = checkpoint['config']['data']['norm_stats']
            print(f"Found normalisation stats in checkpoint: {checkpoint_path}")

            # Create a new dictionary with the corrected structure.
            corrected_stats = {}
            for key, value in stats_dict.items():
                # Recursively flatten until we get to the actual list of numbers
                flattened = value
                while isinstance(flattened, list) and len(flattened) == 1 and isinstance(flattened[0], list):
                    flattened = flattened[0]
                corrected_stats[key] = flattened
            
            return NormStats.model_validate(corrected_stats)
    except Exception as e:
        print(f"Could not load or parse norm_stats from checkpoint {checkpoint_path}: {e}")
    
    return None 