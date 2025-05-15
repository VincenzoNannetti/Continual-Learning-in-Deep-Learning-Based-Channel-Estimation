"""
Filename: ./Supermaks/dataset/dataloader.py
Author: Vincenzo Nannetti
Date: 14/03/2025
Description: Supermask network dataloader

Usage:


Dependencies:
    - PyTorch
    - numpy
    - os
    - scipy
    - src.utils.interpolation
"""
import os
import torch
import numpy as np
from scipy.io import loadmat
from torch.utils.data import Dataset
from shared.utils.interpolation import interpolation

class SupermaskDataset(Dataset):
    def __init__(self, data_config, supermask_config, transform=None):
        """Initialises the Supermask Task-Based Dataset.

        Args:
            data_config (dict): Dictionary containing data configuration parameters like:
                - data_dir (str): Path to the raw data directory (containing task_*.mat files).
                - preprocessed_dir (str): Path to save/load preprocessed task .npy files.
                - tasks (int): Number of tasks (supermasks) to load.
                - interpolation (str): Interpolation method.
                - normalisation (str): normalisation method.
                - normalise_before_interp (bool): normalise before interpolation.
                - input_channels (int): 2 or 3 (if including pilot mask).
            transform (callable, optional): Optional transform to be applied.
        """
        self.data_config             = data_config
        self.transform               = transform
        self.data_dir                = data_config['data_dir']
        self.preprocessed_dir        = data_config.get('preprocessed_dir', "Learning_Algorithms/data/preprocessed/supermasks")
        self.interp_method           = data_config['interpolation']
        self.normalisation           = data_config['normalisation']
        self.normalise_before_interp = data_config['normalise_before_interp']
        self.dataset_names           = [data_config['dataset_a_name'], data_config['dataset_b_name'], data_config['dataset_c_name']]

        self.tasks                   = supermask_config['tasks']
        self.sequence                = supermask_config['sequence']

        # Define precomputed file paths base
        self.norm_type = "before" if self.normalise_before_interp else "after"
        os.makedirs(self.preprocessed_dir, exist_ok=True)

        # Load data for all tasks
        self.all_perfect, self.all_noisy = self.load_all_task_data()

        self.current_task_id     = 0 
        self.current_dataset_len = len(self.all_perfect[0]) if self.all_perfect else 0

    def load_data(self, task_num):
        print("Loading Data...")

        # create the path to the current task precomputed files
        noisy_path   = os.path.join(self.preprocessed_dir, f"noisy_task_{task_num}_{self.interp_method}.npy")
        perfect_path = os.path.join(self.preprocessed_dir, f"perfect_task_{task_num}.npy")

        if os.path.exists(noisy_path) and os.path.exists(perfect_path):
            print(f"Loading precomputed data for task {task_num}...")

            # load the precomputed interpolated data to save time. 
            # the noisy input should have 3 channels (interpolated real, interpolated imag, pilot mask)
            noisy_input_interpolated = np.load(noisy_path)
            perfect_input_processed  = np.load(perfect_path) # still 2 channels (real, imag)
        else:
            print(f"Precomputed data for task {task_num} not found. Processing now...")

            # load raw perfect, raw noisy and pilot mask
            perfect_input, noisy_input_raw, pilot_mask = self.load_raw_data(task_num, load_matrices=True)

            print("Processing raw data (complex to 2 or 3 channel)...")
            perfect_input_processed = self.process_data(perfect_input)               # 2 channels (real, imag)
            noisy_input_processed   = self.process_data(noisy_input_raw, pilot_mask) # 3 channels (interpolated real, interpolated imag, pilot mask)

            # Separate real/imag channels from the pilot mask channel
            noisy_data_to_interp = noisy_input_processed[..., :2]  # Shape (N, H, W, 2)
            pilot_mask_channel   = noisy_input_processed[..., 2:]  # Shape (N, H, W, 1)

            # Apply interpolation ONLY to the real/imag channels
            print(f"Interpolating noisy data (real/imag channels) using {self.interp_method}...")
            if noisy_data_to_interp.shape[:-1] != pilot_mask.shape:
                raise ValueError(f"Shape mismatch: Noisy data channels {noisy_data_to_interp.shape[:-1]} vs Pilot mask {pilot_mask.shape}")
            
            # Pass the boolean mask directly. Assume interpolation handles the 2 channels.
            interpolated_real_imag = interpolation(noisy_data_to_interp, pilot_mask, self.interp_method)

            # Combine interpolated data with the original pilot mask channel
            noisy_input_interpolated = np.concatenate((interpolated_real_imag, pilot_mask_channel), axis=-1)
            # Resulting shape should be (N, H, W, 3)
            
            # save precomputed data (perfect is processed, noisy is processed and interpolated)
            print(f"Saving precomputed and processed data to: {self.preprocessed_dir}")
            try:
                np.save(noisy_path, noisy_input_interpolated)
                np.save(perfect_path, perfect_input_processed)
                print("Processing complete. Processed data saved.")
            except Exception as e:
                print(f"Error saving precomputed data: {e}")

        # return the processed and interpolated data
        return perfect_input_processed, noisy_input_interpolated
                
    def load_raw_data(self, task_num, load_matrices=True):
        """Loads raw data and pilot mask from .mat file."""
        mat_path = os.path.join(self.data_dir, self.dataset_names[task_num])
        try:
            main_data = loadmat(mat_path)
            # if load matrices is true then data not precomputed hence load it 
            if load_matrices:
                perfect_input   = main_data["perfect_matrices"]
                noisy_input_raw = main_data["received_matrices"]
            # either way we need the pilot mask
            pilot_mask          = main_data["pilot_mask"]
        except FileNotFoundError as e:
            print(f"Error loading raw .mat file: {mat_path} - {e}")
            raise
        except KeyError as e:
            print(f"Error: Key {e} not found in {mat_path}. Check .mat file structure.")
            print(f"Available keys: {list(main_data.keys())}")
            raise

        # error check for pilot mask since this is required
        if pilot_mask is None:
            raise RuntimeError(f"Pilot mask could not be loaded from {mat_path}.")
        elif load_matrices:
            if perfect_input is not None and perfect_input.shape != pilot_mask.shape:
                print(f"Warning: Shape mismatch between perfect matrices {perfect_input.shape} and pilot mask {pilot_mask.shape}")
            if noisy_input_raw is not None and noisy_input_raw.shape != pilot_mask.shape:
                print(f"Warning: Shape mismatch between noisy matrices {noisy_input_raw.shape} and pilot mask {pilot_mask.shape}")

        if load_matrices:
            # necessary to return all data in this case
            return perfect_input, noisy_input_raw, pilot_mask
        else:
            # even if precomputed, we still need the pilot mask
            return None, None, pilot_mask
        
    def process_data(self, matrix_input, pilot_mask=None):
        """Converts complex matrix to 3-channel real/imaginary + pilot mask numpy array."""
        if matrix_input is None:
             return None 
        if matrix_input.ndim != 3:
             raise ValueError(f"Expected 3D input (samples, sc, sym), got {matrix_input.ndim}D")

        # get the shape of the input
        n_samples, n_sc, n_sym = matrix_input.shape
        matrix_output = np.zeros((n_samples, n_sc, n_sym, 2 + (1 if pilot_mask is not None else 0)), dtype=np.float32)

        # Separate real and imaginary parts
        matrix_output[..., 0] = np.real(matrix_input)
        matrix_output[..., 1] = np.imag(matrix_input)

        # in the supermask case, the pilot mask is the third channel since needed to work out the task
        # only for the noisy input
        if pilot_mask is not None:
            matrix_output[..., 2] = pilot_mask

        return matrix_output

    def load_all_task_data(self):
        """Loads all task data from precomputed files."""
        all_perfect     = []
        all_noisy       = []

        for i in range(self.tasks):
            perfect, noisy = self.load_data(i)
            all_perfect.append(perfect)
            all_noisy.append(noisy)

        return all_perfect, all_noisy

    # function called by the dataloader to set the current task and load the relevant data.
    def set_task(self, task_id):
        """Sets the active task for __getitem__."""
        if not 0 <= task_id < self.tasks:
            raise IndexError(f"Task ID {task_id} out of range (0-{self.tasks-1})")
        print(f"Setting dataset to Task {task_id}")
        self.current_task_id     = task_id
        self.current_dataset_len = len(self.all_perfect[task_id])

    def __len__(self):
        return self.current_dataset_len

    def __getitem__(self, idx):
        # Get data for the currently active task
        perfect_sample = self.all_perfect[self.current_task_id][idx].astype(np.float32)
        noisy_sample   = self.all_noisy[self.current_task_id][idx].astype(np.float32)

        # Convert to PyTorch tensors: (H, W, C) -> (C, H, W)
        perfect_sample = torch.from_numpy(perfect_sample).permute(2, 0, 1)
        noisy_sample   = torch.from_numpy(noisy_sample).permute(2, 0, 1)

        # Apply transformations if any
        if self.transform:
            noisy_sample = self.transform(noisy_sample)

        return noisy_sample, perfect_sample
