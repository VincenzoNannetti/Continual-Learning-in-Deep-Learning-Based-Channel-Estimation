import os
import time
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.io import loadmat, savemat
from pathlib import Path
from standard_training_2.noise_generation import add_cn_awgn
from standard_training_2.interpolate import interpolation

class NormalisingDatasetWrapper(Dataset):
    """
    A Dataset wrapper that applies Z-score normalisation using provided statistics.
    """
    def __init__(self, subset, norm_stats, return_original_target=False):
        self.subset = subset
        self.mean_inputs, self.std_inputs = norm_stats[0]
        self.mean_targets, self.std_targets = norm_stats[1]
        self.epsilon = 1e-8 # To prevent division by zero
        self.return_original_target = return_original_target

    def __getitem__(self, idx):
        x, y = self.subset[idx] # x, y are already tensors from StandardDataset
        
        # normalise x
        x_normalised = (x - self.mean_inputs) / (self.std_inputs + self.epsilon)
        # normalise y
        y_normalised = (y - self.mean_targets) / (self.std_targets + self.epsilon)
        
        if self.return_original_target:
            return x_normalised, y_normalised, y # Return original y as well
        else:
            return x_normalised, y_normalised

    def __len__(self):
        return len(self.subset)
    

class StandardDataset(Dataset):
    def __init__(self, data_dir, data_name, snr, interpolation_kernel='thin_plate_spline', 
                 preprocessed_dir=None, all_data=False, sequence=None, tasks_params=None):
        self.data_dir = data_dir
        self.data_name = data_name
        self.snr = snr
        self.interpolation_kernel = interpolation_kernel
        self.preprocessed_dir = preprocessed_dir
        self.all_data = all_data
        self.sequence = sequence
        self.tasks_params = tasks_params
        
        # If sequence is provided, override data_name and snr with sequence-based loading
        if self.sequence is not None and self.tasks_params is not None:
            print(f"[DATASET] Loading data based on sequence: {self.sequence}")
            self.use_sequence = True
        else:
            self.use_sequence = False
            
            # Extract SNR level from data name for validation (only for single dataset mode)
            if 'high_snr' in data_name:
                expected_snr = 20
            elif 'med_snr' in data_name:
                expected_snr = 10
            elif 'low_snr' in data_name:
                expected_snr = 3
            else:
                expected_snr = None
                
            if expected_snr is not None and self.snr != expected_snr:
                print(f"Warning: SNR mismatch for {data_name}. Expected {expected_snr}dB but got {self.snr}dB")
                self.snr = expected_snr  # Override with correct SNR
        
        # Create pilot mask BEFORE loading data (since load_data uses it)
        self.pilot_mask = np.zeros((72, 70), dtype=bool)
        # Set pilot locations - using [::12, ::7] as default pattern
        self.pilot_mask[::12, ::7] = True  # Every 12th row, every 7th column
            
        # Load and process data
        self.interpolated_data, self.gt_2ch = self.load_data(all_data)
        
        # Store raw data for analysis
        self.noisy_before_interp = None
        self.perfect_data = None

    def __len__(self):
        return self.interpolated_data.shape[0]
    
    def __getitem__(self, idx):
        # Get data in (H, W, C) format
        x_np = self.interpolated_data[idx].astype(np.float32)  # (72, 70, 2)
        y_np = self.gt_2ch[idx].astype(np.float32)  # (72, 70, 2)

        # Convert to tensor - keep in (H, W, C) format for now
        # The training script will handle the permutation if needed
        x = torch.tensor(x_np, dtype=torch.float32)  # (72, 70, 2)
        y = torch.tensor(y_np, dtype=torch.float32)  # (72, 70, 2)
            
        return x, y
    
    def _get_cache_filename(self):
        """Generate cache filename based on dataset name, SNR, and interpolation settings."""
        base_name, _ = os.path.splitext(self.data_name)
        cache_name = f"{base_name}_snr{self.snr}_{self.interpolation_kernel}.mat"
        cache_dir = Path(self.preprocessed_dir) / "tester_data"
        cache_dir.mkdir(parents=True, exist_ok=True)
        return str(cache_dir / cache_name)
    
    def _save_processed_data(self, interpolated_data, gt_2ch, noisy_before_interp):
        """Save processed data to cache."""
        cache_path = self._get_cache_filename()
        print(f"Saving processed data to cache: {cache_path}")
        
        # Save as .mat file for consistency
        cache_data = {
            'interpolated_data': interpolated_data,
            'ground_truth': gt_2ch,
            'noisy_before_interp': noisy_before_interp,
            'pilot_mask': self.pilot_mask,
            'snr': self.snr,
            'interpolation_kernel': self.interpolation_kernel,
            'data_name': self.data_name
        }
        
        savemat(cache_path, cache_data)
        print(f"Cache saved successfully!")
    
    def _load_cached_data(self):
        """Load cached processed data if available."""
        cache_path = self._get_cache_filename()
        
        print(f"[CACHE_LOAD] Attempting to load cache from: {cache_path}")
        if not os.path.exists(cache_path):
            print(f"[CACHE_LOAD] No cached data found at: {cache_path}")
            return None
        
        print(f"[CACHE_LOAD] Loading cached data from: {cache_path}")
        try:
            cached = loadmat(cache_path)
            
            # Verify cache integrity - handle MATLAB's array wrapping
            cached_snr = float(cached['snr'].flat[0]) if hasattr(cached['snr'], 'flat') else float(cached['snr'])
            cached_kernel = str(cached['interpolation_kernel'].flat[0]) if hasattr(cached['interpolation_kernel'], 'flat') else str(cached['interpolation_kernel'])
            cached_data_name = str(cached['data_name'].flat[0]) if hasattr(cached['data_name'], 'flat') else str(cached['data_name'])
            
            # Normalize data names for comparison (remove .mat extension if present)
            def normalize_data_name(name):
                return name.replace('.mat', '') if name.endswith('.mat') else name
            
            normalized_cached_name = normalize_data_name(cached_data_name)
            normalized_current_name = normalize_data_name(self.data_name)
            
            if (cached_snr != self.snr or 
                cached_kernel != self.interpolation_kernel or
                normalized_cached_name != normalized_current_name):
                print(f"Cache parameters don't match current settings:")
                print(f"  SNR: cached={cached_snr}, current={self.snr}")
                print(f"  Kernel: cached='{cached_kernel}', current='{self.interpolation_kernel}'")
                print(f"  Data name: cached='{normalized_cached_name}', current='{normalized_current_name}'")
                print("Regenerating...")
                return None
            
            # Extract data
            interpolated_data = cached['interpolated_data']
            gt_2ch = cached['ground_truth']
            noisy_before_interp = cached['noisy_before_interp']
            
            # Store for analysis
            self.noisy_before_interp = noisy_before_interp
            self.perfect_data = gt_2ch
            
            print(f"[CACHE_LOAD] Successfully loaded cached data with {interpolated_data.shape[0]} samples!")
            return interpolated_data, gt_2ch
            
        except Exception as e:
            print(f"[CACHE_LOAD] Error loading cached data: {e}. Regenerating...")
            return None
    
    def load_data(self, all_data=False):
        print("[LOAD_DATA] Attempting to load data...")

        # Handle sequence-based loading
        if self.use_sequence:
            print(f"[LOAD_DATA] Loading sequence-based data for tasks: {self.sequence}")
            return self._load_sequence_data()

        if all_data:
            print("[LOAD_DATA] Loading ALL preprocessed datasets...")
            # Load all data from the preprocessed tester_data directory
            tester_data_dir = Path(self.preprocessed_dir) / "tester_data"
            if not tester_data_dir.exists():
                raise FileNotFoundError(f"Preprocessed data directory not found: {tester_data_dir}")
            
            # Find all .mat files in the tester_data directory
            all_files = list(tester_data_dir.glob("*.mat"))
            
            if not all_files:
                raise FileNotFoundError(f"No preprocessed datasets found in {tester_data_dir}")
            
            print(f"[LOAD_DATA] Found {len(all_files)} preprocessed datasets to combine:")
            for f in sorted(all_files):
                print(f"  - {f.name}")
            
            # Load and combine all datasets
            all_interpolated_data = []
            all_ground_truth = []
            all_noisy_before_interp = []
            
            for file_path in sorted(all_files):
                print(f"[LOAD_DATA] Loading {file_path.name}...")
                
                try:
                    cached = loadmat(str(file_path))
                    
                    # Extract data (same format as _load_cached_data)
                    interpolated_data = cached['interpolated_data']
                    gt_2ch = cached['ground_truth']
                    noisy_before_interp = cached['noisy_before_interp']
                    
                    all_interpolated_data.append(interpolated_data)
                    all_ground_truth.append(gt_2ch)
                    all_noisy_before_interp.append(noisy_before_interp)
                    
                    print(f"  Loaded {interpolated_data.shape[0]} samples from {file_path.name}")
                    
                except Exception as e:
                    print(f"  Warning: Failed to load {file_path.name}: {e}")
                    continue
            
            if not all_interpolated_data:
                raise RuntimeError("No datasets were successfully loaded")
            
            # Concatenate all datasets
            print("[LOAD_DATA] Combining all datasets...")
            combined_interpolated = np.concatenate(all_interpolated_data, axis=0)
            combined_ground_truth = np.concatenate(all_ground_truth, axis=0)
            combined_noisy_before_interp = np.concatenate(all_noisy_before_interp, axis=0)
            
            # Store for analysis
            self.noisy_before_interp = combined_noisy_before_interp
            self.perfect_data = combined_ground_truth
            
            print(f"[LOAD_DATA] Successfully combined all datasets: {combined_interpolated.shape[0]} total samples")
            print(f"  Shape: {combined_interpolated.shape}")
            
            return combined_interpolated, combined_ground_truth

        # First, try to load from cache
        cached_result = self._load_cached_data()
        if cached_result is not None:
            print("[LOAD_DATA] Successfully loaded data from cache.")
            return cached_result
        
        # If no cache, process from scratch
        print("[LOAD_DATA] No valid cache found. Processing data from scratch...")
        # Handle case where data_name might already include .mat extension
        data_path = Path(self.data_dir)
        if self.data_name.endswith('.mat'):
            mat_path = data_path / self.data_name
        else:
            mat_path = data_path / f"{self.data_name}.mat"
            
        if not mat_path.exists():
            raise FileNotFoundError(f"Dataset not found at path: {mat_path}")

        try:
            mat = loadmat(str(mat_path))
            if 'perfect_matrices' in mat:
                raw_data = mat['perfect_matrices']
            else:
                raise KeyError("Expected key 'perfect_matrices' in the .mat file")

            # Ensure correct shape: (N, H, W)
            if raw_data.ndim != 3 or raw_data.shape[1:] != (72, 70):
                raise ValueError("Expected raw data shape (N, 72, 70)")

            # Convert to complex
            raw_data = raw_data.astype(np.complex64)

            # Add noise
            print("Adding noise...")
            noisy_data = add_cn_awgn(raw_data, self.snr)
            
            # Verify SNR immediately after noise addition (before interpolation)
            signal_power = np.mean(np.abs(raw_data) ** 2)
            noise_power = np.mean(np.abs(noisy_data - raw_data) ** 2)
            actual_snr = 10 * np.log10(signal_power / noise_power)
            print(f"Actual SNR after noise addition: {actual_snr:.2f} dB (Target: {self.snr} dB)")

            # Convert to 2-channel: real and imaginary parts
            noisy_2ch = np.stack([np.real(noisy_data), np.imag(noisy_data)], axis=-1)  # (N, 72, 70, 2)
            gt_2ch    = np.stack([np.real(raw_data), np.imag(raw_data)], axis=-1)

            # Store raw noisy data before interpolation for SNR analysis
            self.noisy_before_interp = noisy_2ch.copy()
            self.perfect_data = gt_2ch.copy()

            # Create batch-wise pilot mask
            pilot_mask_batch  = np.broadcast_to(self.pilot_mask, (noisy_2ch.shape[0], 72, 70))

            # Interpolate (this is the expensive part)
            print(f"[LOAD_DATA] Starting interpolation for {noisy_2ch.shape[0]} samples... This might take a while.")
            start_time = time.time()
            interpolated_data = interpolation(noisy_2ch, pilot_mask_batch, kernel=self.interpolation_kernel)
            end_time = time.time()
            print(f"[LOAD_DATA] Interpolation finished. Time taken: {end_time - start_time:.2f} seconds.")

            # Save to cache for next time
            self._save_processed_data(interpolated_data, gt_2ch, noisy_2ch)

            return interpolated_data, gt_2ch

        except Exception as e:
            print(f"Failed to load and process dataset: {e}")
            raise

    def _load_sequence_data(self):
        """Load data for specific tasks defined in the sequence."""
        tester_data_dir = Path(self.preprocessed_dir) / "tester_data"
        if not tester_data_dir.exists():
            raise FileNotFoundError(f"Preprocessed data directory not found: {tester_data_dir}")
        
        all_interpolated_data = []
        all_ground_truth = []
        all_noisy_before_interp = []
        
        print(f"[SEQUENCE_LOAD] Loading {len(self.sequence)} tasks from sequence")
        
        for task_id in self.sequence:
            # Get task parameters - handle both string and integer keys
            task_key = task_id  # Try integer key first
            if task_key not in self.tasks_params:
                task_key = str(task_id)  # Try string key
                if task_key not in self.tasks_params:
                    raise ValueError(f"Task {task_id} not found in tasks_params (tried both {task_id} and '{task_id}')")
            
            task_params = self.tasks_params[task_key]
            task_data_name = task_params['data_name']
            task_snr = task_params['snr']
            
            # Construct cache filename for this task
            cache_name = f"{task_data_name}_snr{task_snr}_{self.interpolation_kernel}.mat"
            cache_path = tester_data_dir / cache_name
            
            print(f"[SEQUENCE_LOAD] Task {task_id}: Loading {task_data_name} (SNR={task_snr}dB)")
            
            if not cache_path.exists():
                print(f"  WARNING: Cache file not found: {cache_path}")
                print(f"  Attempting to process from raw data...")
                
                # Process this task from raw data
                interpolated, ground_truth = self._process_single_task(task_data_name, task_snr)
                if interpolated is not None:
                    all_interpolated_data.append(interpolated)
                    all_ground_truth.append(ground_truth)
                    # Note: noisy_before_interp would need to be returned from _process_single_task
                else:
                    print(f"  ERROR: Failed to process task {task_id}")
                    continue
            else:
                # Load from cache
                try:
                    cached = loadmat(str(cache_path))
                    
                    interpolated_data = cached['interpolated_data']
                    gt_2ch = cached['ground_truth']
                    noisy_before_interp = cached['noisy_before_interp']
                    
                    all_interpolated_data.append(interpolated_data)
                    all_ground_truth.append(gt_2ch)
                    all_noisy_before_interp.append(noisy_before_interp)
                    
                    print(f"  Loaded {interpolated_data.shape[0]} samples from cache")
                    
                except Exception as e:
                    print(f"  ERROR: Failed to load cache for task {task_id}: {e}")
                    continue
        
        if not all_interpolated_data:
            raise RuntimeError(f"No datasets were successfully loaded from sequence {self.sequence}")
        
        # Concatenate all datasets
        print("[SEQUENCE_LOAD] Combining sequence datasets...")
        combined_interpolated = np.concatenate(all_interpolated_data, axis=0)
        combined_ground_truth = np.concatenate(all_ground_truth, axis=0)
        
        if all_noisy_before_interp:
            combined_noisy_before_interp = np.concatenate(all_noisy_before_interp, axis=0)
            self.noisy_before_interp = combined_noisy_before_interp
        
        self.perfect_data = combined_ground_truth
        
        print(f"[SEQUENCE_LOAD] Successfully loaded sequence data: {combined_interpolated.shape[0]} total samples")
        print(f"  Tasks included: {self.sequence}")
        print(f"  Shape: {combined_interpolated.shape}")
        
        return combined_interpolated, combined_ground_truth
    
    def _process_single_task(self, data_name, snr):
        """Process a single task from raw data (fallback when cache doesn't exist)."""
        # Handle case where data_name might already include .mat extension
        data_path = Path(self.data_dir)
        if data_name.endswith('.mat'):
            mat_path = data_path / data_name
        else:
            mat_path = data_path / f"{data_name}.mat"
            
        if not mat_path.exists():
            print(f"  ERROR: Raw data not found at: {mat_path}")
            return None, None
        
        try:
            mat = loadmat(str(mat_path))
            if 'perfect_matrices' in mat:
                raw_data = mat['perfect_matrices']
            else:
                raise KeyError("Expected key 'perfect_matrices' in the .mat file")

            # Ensure correct shape: (N, H, W)
            if raw_data.ndim != 3 or raw_data.shape[1:] != (72, 70):
                raise ValueError("Expected raw data shape (N, 72, 70)")

            # Convert to complex
            raw_data = raw_data.astype(np.complex64)

            # Add noise
            print(f"  Adding noise (SNR={snr}dB)...")
            noisy_data = add_cn_awgn(raw_data, snr)

            # Convert to 2-channel: real and imaginary parts
            noisy_2ch = np.stack([np.real(noisy_data), np.imag(noisy_data)], axis=-1)
            gt_2ch = np.stack([np.real(raw_data), np.imag(raw_data)], axis=-1)

            # Create batch-wise pilot mask
            pilot_mask_batch = np.broadcast_to(self.pilot_mask, (noisy_2ch.shape[0], 72, 70))

            # Interpolate
            print(f"  Interpolating {noisy_2ch.shape[0]} samples...")
            interpolated_data = interpolation(noisy_2ch, pilot_mask_batch, kernel=self.interpolation_kernel)

            # Save to cache for next time
            cache_name = f"{data_name}_snr{snr}_{self.interpolation_kernel}.mat"
            cache_dir = Path(self.preprocessed_dir) / "tester_data"
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path = cache_dir / cache_name
            
            cache_data = {
                'interpolated_data': interpolated_data,
                'ground_truth': gt_2ch,
                'noisy_before_interp': noisy_2ch,
                'pilot_mask': self.pilot_mask,
                'snr': snr,
                'interpolation_kernel': self.interpolation_kernel,
                'data_name': data_name
            }
            
            savemat(str(cache_path), cache_data)
            print(f"  Saved processed data to cache: {cache_path}")

            return interpolated_data, gt_2ch

        except Exception as e:
            print(f"  ERROR: Failed to process task: {e}")
            return None, None
