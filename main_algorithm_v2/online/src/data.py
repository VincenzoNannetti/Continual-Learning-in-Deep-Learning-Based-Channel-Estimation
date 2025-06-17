"""
Online data loading and preprocessing for continual learning.
Loads raw data and performs real-time interpolation without caching.
"""

import os
import torch
import numpy as np
import scipy.io as sio
from typing import Tuple, Dict, Any
import random
from pathlib import Path

# Import interpolation and noise functions
from standard_training_2.noise_generation import add_cn_awgn
from standard_training_2.interpolate import interpolation

# Import offline config for NormStats
from main_algorithm_v2.offline.src.config import NormStats

class OnlineDataPipeline:
    """
    Online data pipeline that loads raw data and performs real-time processing.
    No caching is used to simulate true online conditions.
    """
    
    def __init__(self, raw_data_config: Dict[str, Any], norm_stats: Dict[str, Any], 
                 interpolation_kernel: str = 'thin_plate_spline', domain_remapping: Dict[int, int] = None):
        """
        Initialize the online data pipeline.
        
        Args:
            raw_data_config: Configuration for raw data loading
            norm_stats: Normalization statistics from offline training
            interpolation_kernel: Kernel for interpolation
            domain_remapping: Optional dict to remap domains (e.g., {1: 5, 5: 1} to swap domains 1 and 5)
        """
        self.config = raw_data_config
        self.norm_stats = norm_stats
        self.interpolation_kernel = interpolation_kernel
        self.domain_remapping = domain_remapping or {}
        
        # Convert norm stats to tensors
        # Handle both dict and NormStats object formats
        if hasattr(norm_stats, 'mean_inputs'):
            # NormStats object (Pydantic model)
            self.mean_inputs = torch.tensor(norm_stats.mean_inputs).view(1, -1, 1, 1)
            self.std_inputs = torch.tensor(norm_stats.std_inputs).view(1, -1, 1, 1)
            self.mean_targets = torch.tensor(norm_stats.mean_targets).view(1, -1, 1, 1)
            self.std_targets = torch.tensor(norm_stats.std_targets).view(1, -1, 1, 1)
        else:
            # Dictionary format
            self.mean_inputs = torch.tensor(norm_stats['mean_inputs']).view(1, -1, 1, 1)
            self.std_inputs = torch.tensor(norm_stats['std_inputs']).view(1, -1, 1, 1)
            self.mean_targets = torch.tensor(norm_stats['mean_targets']).view(1, -1, 1, 1)
            self.std_targets = torch.tensor(norm_stats['std_targets']).view(1, -1, 1, 1)
        
        # Load raw data for all domains
        self.domain_data = {}
        self.domain_indices = {}
        
        print(f"[DATA] Loading raw data for domains: {self.config['domains']}")
        if self.domain_remapping:
            print(f"[DATA] Domain remapping active: {self.domain_remapping}")
            print(f"[DATA] Domain remapping keys: {list(self.domain_remapping.keys())} (types: {[type(k) for k in self.domain_remapping.keys()]})")
        else:
            print(f"[DATA] No domain remapping configured")
        
        for domain_id in self.config['domains']:
            # Load data for this domain
            # Handle both old format (data_dir/domain_files) and new format (base_path/domain_file_mapping)
            if 'data_dir' in self.config and 'domain_files' in self.config:
                data_path = Path(self.config['data_dir']) / self.config['domain_files'][str(domain_id)]
            elif 'base_path' in self.config and 'domain_file_mapping' in self.config:
                data_path = Path(self.config['base_path']) / self.config['domain_file_mapping'][domain_id]
            else:
                raise KeyError(f"Config must have either 'data_dir'/'domain_files' or 'base_path'/'domain_file_mapping' keys")
            
            if not data_path.exists():
                raise FileNotFoundError(f"Data file not found: {data_path}")
            
            # Load the .mat file
            mat_data = sio.loadmat(str(data_path))
            
            # Extract perfect matrices
            if 'perfect_matrices' in mat_data:
                perfect_data = mat_data['perfect_matrices']
            else:
                raise KeyError(f"'perfect_matrices' not found in {data_path}")
            
            # Store the data
            self.domain_data[domain_id] = perfect_data
            self.domain_indices[domain_id] = 0  # Track current index for each domain
            
            print(f"  Domain {domain_id}: Loaded {perfect_data.shape[0]} samples from {data_path.name}")
        
        # Create pilot mask (same as offline)
        self.pilot_mask = self._create_pilot_mask()
        
        print(f"[DATA] Online data pipeline initialized with {len(self.domain_data)} domains")
        
    def _create_pilot_mask(self):
        """Create pilot mask with pilots at regular intervals."""
        pilot_mask = np.zeros((72, 70), dtype=bool)
        pilot_mask[::12, ::7] = True  # Every 12th row, every 7th column
        return pilot_mask
    
    def _load_domain_data(self, domain_id: int) -> np.ndarray:
        """
        Load raw data for a specific domain.
        
        Args:
            domain_id: Domain identifier (0-8)
            
        Returns:
            Raw complex channel data of shape (N, 72, 70)
        """
        if domain_id in self.domain_data:
            return self.domain_data[domain_id]
            
        if domain_id not in self.config['domain_file_mapping']:
            raise ValueError(f"Domain {domain_id} not found in domain file mapping")
            
        filename = self.config['domain_file_mapping'][domain_id]
        file_path = os.path.join(self.config['base_path'], filename)
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Domain data file not found: {file_path}")
            
        print(f"Loading domain {domain_id} data from: {file_path}")
        
        try:
            mat_data = sio.loadmat(file_path)
            if 'perfect_matrices' in mat_data:
                raw_data = mat_data['perfect_matrices']
            else:
                raise KeyError("Expected key 'perfect_matrices' in the .mat file")
                
            # Ensure correct shape and convert to complex
            if raw_data.ndim != 3 or raw_data.shape[1:] != (72, 70):
                raise ValueError(f"Expected raw data shape (N, 72, 70), got {raw_data.shape}")
                
            raw_data = raw_data.astype(np.complex64)
            
            # Cache the loaded data
            self.domain_data[domain_id] = raw_data
            print(f"Domain {domain_id} loaded: {raw_data.shape[0]} samples")
            
            return raw_data
            
        except Exception as e:
            raise RuntimeError(f"Error loading domain {domain_id} data: {e}")
    
    def get_random_sample(self, domain_id: int, snr: int = 20) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get a random sample from the specified domain with noise added.
        
        Args:
            domain_id: Domain to sample from (0-8)
            snr: Signal-to-noise ratio in dB
            
        Returns:
            Tuple of (noisy_sample, ground_truth) both of shape (72, 70, 2)
        """
        # Apply domain remapping if configured (for mislabeling experiments)
        actual_domain_id = domain_id
        
        # Check both int and str keys to handle type mismatches
        if domain_id in self.domain_remapping:
            actual_domain_id = self.domain_remapping[domain_id]
            print(f"[MISLABEL] Domain {domain_id} → using data from Domain {actual_domain_id}")
        elif str(domain_id) in self.domain_remapping:
            actual_domain_id = self.domain_remapping[str(domain_id)]
            print(f"[MISLABEL] Domain {domain_id} (str key) → using data from Domain {actual_domain_id}")
        elif int(domain_id) in self.domain_remapping:
            actual_domain_id = self.domain_remapping[int(domain_id)]
            print(f"[MISLABEL] Domain {domain_id} (int key) → using data from Domain {actual_domain_id}")
        
        # Load domain data (either original or remapped)
        if actual_domain_id in self.domain_data:
            raw_data = self.domain_data[actual_domain_id]
        else:
            raw_data = self._load_domain_data(actual_domain_id)
        
        # Select random sample
        sample_idx = random.randint(0, raw_data.shape[0] - 1)
        clean_sample = raw_data[sample_idx]  # Shape: (72, 70)
        
        # Add noise
        noisy_sample = add_cn_awgn(clean_sample[np.newaxis, :, :], snr)[0]  # Add batch dim, then remove
        
        # Convert to 2-channel format (real, imaginary)
        noisy_2ch = np.stack([np.real(noisy_sample), np.imag(noisy_sample)], axis=-1)
        ground_truth_2ch = np.stack([np.real(clean_sample), np.imag(clean_sample)], axis=-1)
        
        return noisy_2ch, ground_truth_2ch
    
    def extract_pilot_signal(self, noisy_sample: np.ndarray) -> np.ndarray:
        """
        Extract pilot signal from noisy sample using pilot mask.
        
        Args:
            noisy_sample: Shape (72, 70, 2) - real/imag channels
            
        Returns:
            Pilot signal with non-pilot positions set to zero
        """
        pilot_signal = noisy_sample.copy()
        
        # Zero out non-pilot positions
        pilot_signal[~self.pilot_mask] = 0.0
        
        return pilot_signal
    
    def interpolate_from_pilots(self, pilot_signal: np.ndarray) -> np.ndarray:
        """
        Perform interpolation from pilot signal to full grid.
        
        Args:
            pilot_signal: Shape (72, 70, 2) with pilots at mask positions
            
        Returns:
            Interpolated signal of shape (72, 70, 2)
        """
        # Add batch dimension for interpolation function
        pilot_batch      = pilot_signal[np.newaxis, :, :, :]  # Shape: (1, 72, 70, 2)
        pilot_mask_batch = self.pilot_mask[np.newaxis, :, :]  # Shape: (1, 72, 70)
        
        # Perform interpolation
        interpolated_batch = interpolation(
            pilot_batch, 
            pilot_mask_batch, 
            kernel=self.interpolation_kernel
        )
        
        # Remove batch dimension
        interpolated_signal = interpolated_batch[0]  # Shape: (72, 70, 2)
        
        return interpolated_signal
    
    def normalise_data(self, data: np.ndarray, is_target: bool = False) -> torch.Tensor:
        """
        Normalise data using the norm stats from the trained model.
        
        Args:
            data: Input data of shape (72, 70, 2)
            is_target: Whether this is target data (affects normalisation choice)
            
        Returns:
            Normalised tensor ready for model input
        """
        if is_target:
            mean = self.mean_targets.squeeze().numpy()  # Shape: (2,)
            std  = self.std_targets.squeeze().numpy()
        else:
            mean = self.mean_inputs.squeeze().numpy()   # Shape: (2,)
            std  = self.std_inputs.squeeze().numpy()
        
        # Normalise: (data - mean) / std
        # Broadcasting: (72, 70, 2) - (2,) -> (72, 70, 2)
        normalised_data = (data - mean) / std
        
        # Convert to tensor and rearrange to (channels, height, width)
        tensor = torch.from_numpy(normalised_data).float()
        tensor = tensor.permute(2, 0, 1)  # (2, 72, 70)
        
        return tensor
    
    def process_sample_online(self, domain_id: int, snr: int = 20) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Complete online processing pipeline for a single sample.
        
        Args:
            domain_id: Domain to sample from
            snr: Signal-to-noise ratio
            
        Returns:
            Tuple of (model_input, ground_truth_target, pilot_mask, metadata)
        """
        import time
        
        # Step 1: Get raw sample with noise
        start_time                 = time.time()
        noisy_sample, ground_truth = self.get_random_sample(domain_id, snr)
        sampling_time              = time.time() - start_time
        
        # Step 2: Extract pilot signal
        start_time            = time.time()
        pilot_signal          = self.extract_pilot_signal(noisy_sample)
        pilot_extraction_time = time.time() - start_time
        
        # Step 3: Interpolate from pilots
        start_time          = time.time()
        interpolated_signal = self.interpolate_from_pilots(pilot_signal)
        interpolation_time  = time.time() - start_time
        
        # Step 4: Normalise for model input
        start_time          = time.time()
        model_input         = self.normalise_data(interpolated_signal, is_target=False)
        ground_truth_tensor = self.normalise_data(ground_truth, is_target=True)
        normalisation_time  = time.time() - start_time
        
        # Convert pilot mask to tensor
        pilot_mask_tensor = torch.from_numpy(self.pilot_mask).bool()
        
        # Metadata for timing analysis
        metadata = {
            'domain_id': domain_id,
            'snr': snr,
            'sampling_time': sampling_time,
            'pilot_extraction_time': pilot_extraction_time,
            'interpolation_time': interpolation_time,
            'normalisation_time': normalisation_time,
            'total_preprocessing_time': sampling_time + pilot_extraction_time + interpolation_time + normalisation_time
        }
        
        return model_input, ground_truth_tensor, pilot_mask_tensor, metadata 