"""
Filename: standard_training/datasets/standard_dataset.py
Author: Vincenzo Nannetti
Date: 04/03/2025
Description: General Dataset Loader (Handles Raw, Interpolation)
             normalisation is now handled externally after data splitting.

Dependencies:
    - PyTorch
    - numpy
    - os
    - scipy
    - src.utils.interpolation 
"""
import torch
import numpy as np
import os
from scipy.io import loadmat
from torch.utils.data import Dataset
from shared.utils.interpolation import interpolation

class StandardDataset(Dataset):
    def __init__(self, data_config, transform=None):
        """Initialises the Standard Dataset. Loads raw data and performs interpolation if specified.

        Args:
            data_config (dict): Dictionary containing data configuration parameters like:
                - data_dir (str): Path to the raw data directory.
                - preprocessed_dir (str): Path to save/load preprocessed .npy files.
                - data_name (str): Specific name identifier for dataset.
                - interpolation (str): Interpolation method, e.g., "rbf".
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_config        = data_config
        self.transform          = transform 
        self.data_dir           = data_config['data_dir']
        self.preprocessed_dir   = data_config.get('preprocessed_dir', "Learning_Algorithms/data/preprocessed")
        self.data_name          = data_config['data_name']
        self.interp_method      = data_config['interpolation']
        self.pilot_mask         = None 

        # Define precomputed file paths 
        base_filename = f"{self.interp_method}_{self.data_name}"
        os.makedirs(self.preprocessed_dir, exist_ok=True)

        # Paths for potentially interpolated noisy data and raw perfect data
        self.precomputed_path_noisy   = os.path.join(self.preprocessed_dir, f"interpolated_{base_filename}.npy")
        self.precomputed_path_perfect = os.path.join(self.preprocessed_dir, f"perfect_{self.data_name}.npy")

        # load the data from the precomputed files or raw data.
        # self.perfect should be a 2 channel matrix
        # self.noisy_input should likewise be a 2 channel matrix but interpolated using the pilot mask.
        self.perfect, self.noisy_input = self.load_data()

    def load_data(self):
        print("\n" + "*"*60)
        print("DATASET LOADING")
        print("*"*60)

        # Check if precomputed *unnormalised* dataset exists
        if os.path.exists(self.precomputed_path_noisy) and os.path.exists(self.precomputed_path_perfect):
            print(f"Loading precomputed PROCESSED dataset from: {self.preprocessed_dir}")
            print(f"  Interpolated Input file: {os.path.basename(self.precomputed_path_noisy)}")
            print(f"  Perfect file: {os.path.basename(self.precomputed_path_perfect)}")

            # load the precomputed interpolated data to save time. 
            noisy_input_interpolated  = np.load(self.precomputed_path_noisy)
            perfect_input_processed   = np.load(self.precomputed_path_perfect)
            # Load pilot mask even if data is precomputed
            _, _, self.pilot_mask = self.load_raw_data(load_matrices=False)
        else:
            print("\n" + "*"*60)
            print("PROCESSING RAW DATA")
            print("*"*60)
            print("Precomputed dataset not found. Processing from raw data now...")
            print(f"  Raw data directory: {self.data_dir}")
            # Load raw perfect, raw noisy, and pilot mask
            perfect_input_raw, noisy_input_raw, self.pilot_mask = self.load_raw_data()

            # Process raw data (complex data to 2-channel array/matrix)
            print("\n" + "*"*60)
            print("COMPLEX DATA CONVERSION")
            print("*"*60)
            print("Processing raw data (complex to 2-channel)...")
            perfect_input_processed = self.process_data(perfect_input_raw)
            noisy_input_processed   = self.process_data(noisy_input_raw)

            # Apply interpolation using the loaded boolean pilot mask
            print("\n" + "*"*60)
            print("INTERPOLATION")
            print("*"*60)
            print(f"Interpolating processed noisy data using {self.interp_method}...")
            # Ensure loaded mask shape matches noisy data shape (excluding real/imag dim)
            if noisy_input_processed.shape[:-1] != self.pilot_mask.shape:
                 raise ValueError(f"Shape mismatch: Noisy data {noisy_input_processed.shape[:-1]} vs Pilot mask {self.pilot_mask.shape}")

            # Pass the boolean mask directly into external interpolation function.
            noisy_input_interpolated = interpolation(noisy_input_processed, self.pilot_mask, self.interp_method)

            # Save precomputed data (perfect is processed, noisy is processed and interpolated)
            print("\n" + "*"*60)
            print("SAVING PROCESSED DATA")
            print("*"*60)
            print(f"Saving precomputed and processed data to: {self.preprocessed_dir}")
            try:
                np.save(self.precomputed_path_noisy, noisy_input_interpolated)
                np.save(self.precomputed_path_perfect, perfect_input_processed) 
                print("Processing complete. Processed data saved.")
            except Exception as e:
                print(f"Error saving precomputed data: {e}")

        # return the processed and interpolated data
        return perfect_input_processed, noisy_input_interpolated 

    def load_raw_data(self, load_matrices=True):
        """Loads raw data and pilot mask from .mat file.

        Args:
            load_matrices (bool): If False, only loads pilot mask.

        Returns:
            tuple: (perfect_input, noisy_input_raw, pilot_mask) or (None, None, pilot_mask)
        """
        perfect_input   = None
        noisy_input_raw = None
        pilot_mask      = None 
        mat_path        = None

        # Create the path to the .mat file
        mat_path = os.path.join(self.data_dir, f"{self.data_name}.mat") 
        try:
            # load the .mat file
            main_data = loadmat(mat_path)
            if load_matrices:
                # load the perfect and noisy matrices
                perfect_input   = main_data["perfect_matrices"]
                noisy_input_raw = main_data["received_matrices"]
            # load the pilot mask
            pilot_mask = main_data["pilot_mask"]

        except FileNotFoundError as e:
            print(f"Error loading raw .mat file: {mat_path} - {e}")
            raise
        except KeyError as e:
            # error messages
            print(f"Error: Key {e} not found in .mat file ({mat_path}). Check file structure.")
            print(f"Available keys: {list(main_data.keys())}")
            raise

        if pilot_mask is None:
            print(f"Warning: Pilot mask could not be loaded from {mat_path}.")
        elif load_matrices: 
             if perfect_input is not None and perfect_input.shape != pilot_mask.shape:
                  print(f"Warning: Shape mismatch between perfect matrices {perfect_input.shape} and pilot mask {pilot_mask.shape}")
             if noisy_input_raw is not None and noisy_input_raw.shape != pilot_mask.shape:
                  print(f"Warning: Shape mismatch between noisy matrices {noisy_input_raw.shape} and pilot mask {pilot_mask.shape}")


        if load_matrices:
            # return all data
            return perfect_input, noisy_input_raw, pilot_mask
        else:
            # even if precomputed, we still need the pilot mask
            return None, None, pilot_mask 

    def process_data(self, matrix_input):
        """Converts complex matrix to 2-channel real/imaginary numpy array."""
        if matrix_input is None:
             return None 
        if matrix_input.ndim != 3:
             raise ValueError(f"Expected 3D input (samples, sc, sym), got {matrix_input.ndim}D")

        # get the shape of the input
        n_samples, n_sc, n_sym = matrix_input.shape
        matrix_output = np.zeros((n_samples, n_sc, n_sym, 2), dtype=np.float32)

        # Separate real and imaginary parts
        matrix_output[..., 0] = np.real(matrix_input)
        matrix_output[..., 1] = np.imag(matrix_input)

        return matrix_output

    def __len__(self):
        # Use perfect shape as reference
        return self.perfect.shape[0]

    def __getitem__(self, idx):
        # Get the interpolated noisy data
        perfect_sample = self.perfect[idx].astype(np.float32)
        noisy_sample   = self.noisy_input[idx].astype(np.float32)

        # Convert to PyTorch tensors: (H, W, C) -> (C, H, W)
        perfect_tensor = torch.from_numpy(perfect_sample).permute(2, 0, 1)
        noisy_tensor   = torch.from_numpy(noisy_sample).permute(2, 0, 1)

        # Apply transformations if any (applied before external normalisation)
        if self.transform:
            noisy_tensor = self.transform(noisy_tensor)

        # Return the raw tensors - to be normalised externally.
        return noisy_tensor, perfect_tensor

