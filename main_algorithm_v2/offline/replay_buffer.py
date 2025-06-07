import torch
from collections import deque
from typing import Tuple, List, Dict
import scipy.io
import numpy as np
from sklearn.cluster import KMeans
import os
from standard_training_2.interpolate import interpolation

# Fix memory leak on Windows with MKL
os.environ['OMP_NUM_THREADS'] = '2'

class ReplayBuffer:
    def __init__(self, buffer_size: int):
        """
        Initialises the replay buffer.

        Args:
            buffer_size: The maximum number of samples the buffer can hold.
        """
        self.buffer_size = buffer_size
        self.buffer      = deque(maxlen=buffer_size)

    def add(self, sample: Tuple[torch.Tensor, torch.Tensor], **kwargs):
        """

        MAKE BETTER SAMPLE SELECTION LOGIC HERE
        
        Adds a sample to the buffer.
        If the buffer is full, the oldest sample is removed (FIFO).
        Samples are detached and moved to CPU before storing.

        Args:
            sample: A tuple containing (input_data, target_data).
        """
        input_data, target_data = sample
        # Detach, clone, and move to CPU to save memory and ensure serializability
        input_data_cpu = input_data.detach().clone().cpu()
        target_data_cpu = target_data.detach().clone().cpu()
        
        self.buffer.append((input_data_cpu, target_data_cpu))

    def get_all_samples(self) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Returns all samples currently stored in the buffer.

        Returns:
            A list of tuples, where each tuple is (input_data, target_data).
        """
        return list(self.buffer)

    def state_dict(self) -> Dict:
        """
        Returns a dictionary containing the buffer's samples.
        This is useful for saving the buffer's state.

        Returns:
            A dictionary with a single key 'samples' holding the list of samples.
        """
        return {'samples': self.get_all_samples()}

    def load_state_dict(self, state_dict: Dict):
        """
        Loads samples into the buffer from a state dictionary.

        Args:
            state_dict: A dictionary, expected to have a 'samples' key
                        with a list of (input_data, target_data) tuples.
        """
        samples = state_dict.get('samples', [])
        self.buffer.clear()
        for input_data, target_data in samples:
            # Ensure they are on CPU, though they should be if saved correctly
            self.buffer.append((input_data.cpu(), target_data.cpu()))
            if len(self.buffer) >= self.buffer_size:
                break # Stop if buffer size is reached during loading

    def __len__(self) -> int:
        """
        Returns the current number of samples in the buffer.
        """
        return len(self.buffer) 

def calculate_model_difficulty_metrics(model: torch.nn.Module, inputs: torch.Tensor, targets: torch.Tensor, 
                                     device: str = 'cpu', batch_size: int = 32) -> np.ndarray:
    """
    Calculate difficulty metrics (NMSE) for each sample using the model's predictions.
    
    Args:
        model: Trained model to evaluate samples with
        inputs: Input tensor of shape (N, C, H, W)
        targets: Target tensor of shape (N, C, H, W)  
        device: Device to run inference on
        batch_size: Batch size for inference
        
    Returns:
        Tuple of (NMSE values array, predictions tensor)
    """
    print(f"Calculating model difficulty metrics (NMSE) for {inputs.shape[0]} samples...")
    
    model.eval()
    model.to(device)
    nmse_values = []
    all_outputs = []
    
    with torch.no_grad():
        for i in range(0, inputs.shape[0], batch_size):
            batch_inputs = inputs[i:i+batch_size].to(device)
            batch_targets = targets[i:i+batch_size].to(device)
            
            # Run model prediction
            batch_outputs = model(batch_inputs)  # UNet_SRCNN_LoRA returns output tensor
            all_outputs.append(batch_outputs.cpu())
            
            # Calculate NMSE for each sample in the batch
            for j in range(batch_outputs.shape[0]):
                target_sample = batch_targets[j]
                output_sample = batch_outputs[j]
                
                # NMSE calculation: ||y - y_hat||^2 / ||y||^2
                mse = torch.mean((output_sample - target_sample)**2)
                target_power = torch.mean(target_sample**2)
                
                if target_power > 0:
                    nmse = mse / target_power
                else:
                    nmse = torch.tensor(0.0)  # Perfect case when target is zero
                    
                nmse_values.append(nmse.item())
    
    nmse_data = np.array(nmse_values)
    predictions = torch.cat(all_outputs, dim=0)
    print(f"NMSE range: {np.min(nmse_data):.6f} to {np.max(nmse_data):.6f}")
    return nmse_data, predictions


def perform_difficulty_clustering(difficulty_data: np.ndarray, n_clusters: int = 3) -> Tuple[np.ndarray, KMeans]:
    """
    Perform K-means clustering on difficulty metrics (NMSE).
    
    Args:
        difficulty_data: Array of NMSE values for each sample
        n_clusters: Number of clusters for K-means
        
    Returns:
        Tuple of (cluster_labels, fitted_kmeans_model)
    """
    print(f"Performing K-means clustering with {n_clusters} clusters on NMSE values...")
    
    difficulty_reshaped = difficulty_data.reshape(-1, 1)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(difficulty_reshaped)
    
    # Show clustering results
    print(f"NMSE-based clustering results (K={n_clusters}):")
    for cluster_id in range(n_clusters):
        cluster_mask = cluster_labels == cluster_id
        cluster_nmse = difficulty_data[cluster_mask]
        print(f"  Cluster {cluster_id}: {np.sum(cluster_mask)} samples, "
              f"NMSE {np.min(cluster_nmse):.6f}-{np.max(cluster_nmse):.6f}")
    
    return cluster_labels, kmeans


def create_stratified_replay_buffer(predictions: torch.Tensor, targets: torch.Tensor, 
                                   cluster_labels: np.ndarray, difficulty_data: np.ndarray,
                                   buffer_size: int = 350) -> ReplayBuffer:
    """
    Create a replay buffer with stratified sampling based on cluster labels.
    
    Args:
        predictions: Predictions tensor of shape (N, C, H, W)
        targets: Target tensor of shape (N, C, H, W)
        cluster_labels: Cluster assignment for each sample
        difficulty_data: NMSE values for each sample
        buffer_size: Target buffer size
        
    Returns:
        ReplayBuffer with optimally selected samples
    """
    print(f"Creating stratified replay buffer with {buffer_size} samples...")
    
    # Calculate proportional allocation based on natural distribution
    unique_labels, counts = np.unique(cluster_labels, return_counts=True)
    total_samples = len(cluster_labels)
    
    allocation = {}
    cluster_info = {}
    
    print(f"  Proportional allocation for {buffer_size} samples:")
    
    # Sort clusters by difficulty (NMSE) - lower NMSE = easier, higher NMSE = harder
    cluster_means = []
    for cluster_id in unique_labels:
        cluster_mask = cluster_labels == cluster_id
        cluster_nmse = difficulty_data[cluster_mask]
        cluster_means.append((cluster_id, np.mean(cluster_nmse)))
    
    # Sort by mean NMSE: lowest = easiest, highest = hardest
    cluster_means.sort(key=lambda x: x[1])
    difficulty_names = ['Easy', 'Medium', 'Hard']
    
    for i, (cluster_id, count) in enumerate(zip(unique_labels, counts)):
        proportion = count / total_samples
        allocated_samples = int(proportion * buffer_size)
        allocation[cluster_id] = allocated_samples
        
        # Get cluster NMSE info
        cluster_mask = cluster_labels == cluster_id
        cluster_nmse = difficulty_data[cluster_mask]
        cluster_info[cluster_id] = {
            'indices': np.where(cluster_mask)[0],
            'nmse_values': cluster_nmse,
            'min_nmse': np.min(cluster_nmse),
            'max_nmse': np.max(cluster_nmse)
        }
        
        # Determine difficulty level based on sorted clusters
        difficulty_idx = next((j for j, (cid, _) in enumerate(cluster_means) if cid == cluster_id), 0)
        difficulty_idx = min(difficulty_idx, len(difficulty_names) - 1)
        difficulty = difficulty_names[difficulty_idx]
        
        print(f"    {difficulty} cluster {cluster_id}: {allocated_samples} samples "
              f"(NMSE {cluster_info[cluster_id]['min_nmse']:.6f}-{cluster_info[cluster_id]['max_nmse']:.6f})")
    
    # Create the replay buffer
    replay_buffer = ReplayBuffer(buffer_size)
    
    print(f"\n  Selecting diverse samples from each cluster:")
    
    total_selected = 0
    for cluster_id in unique_labels:
        target_samples = allocation[cluster_id]
        cluster_indices = cluster_info[cluster_id]['indices']
        cluster_nmse = cluster_info[cluster_id]['nmse_values']
        
        if len(cluster_indices) >= target_samples and target_samples > 0:
            # Sort by NMSE and select evenly distributed samples
            sorted_idx = np.argsort(cluster_nmse)
            sorted_cluster_indices = cluster_indices[sorted_idx]
            
            # Select samples evenly across NMSE range
            step = len(sorted_cluster_indices) // target_samples
            if step < 1:
                step = 1
            selected_indices = sorted_cluster_indices[::step][:target_samples]
        else:
            # Take all samples if cluster is smaller than target
            selected_indices = cluster_indices
        
        # Add selected samples to buffer
        for idx in selected_indices:
            # Storing predictions and targets in the buffer
            replay_buffer.add((predictions[idx], targets[idx]))
            total_selected += 1
        
        selected_nmse_range = cluster_nmse[np.argsort(cluster_nmse)][::step][:target_samples] if len(selected_indices) > 0 else cluster_nmse
        
        if len(selected_nmse_range) == 0:
            print(f"    Cluster {cluster_id}: Skipped (0 samples selected)")
        else:
            print(f"    Cluster {cluster_id}: Selected {len(selected_indices)} samples "
                  f"spanning NMSE range {np.min(selected_nmse_range):.6f}-{np.max(selected_nmse_range):.6f}")
    
    print(f"\n [Success] Replay buffer created: {len(replay_buffer)}/{buffer_size} samples")
    
    # Verify buffer quality
    stored_samples = replay_buffer.get_all_samples()
    if stored_samples:
        stored_nmse = []
        for pred_sample, target_sample in stored_samples:
            # Calculate NMSE for verification using the stored prediction
            mse = torch.mean((pred_sample - target_sample)**2)
            target_power = torch.mean(target_sample**2)
            if target_power > 0:
                nmse = mse / target_power
                stored_nmse.append(nmse.item())
        
        print(f"\n  📊 Final buffer statistics:")
        print(f"    NMSE coverage: {np.min(stored_nmse):.6f} to {np.max(stored_nmse):.6f}")
        print(f"    NMSE diversity (std): {np.std(stored_nmse):.6f}")
        print(f"    Memory usage: ~{len(replay_buffer) * 2 * predictions.shape[2] * predictions.shape[3] * 4 / (1024**2):.1f} MB")
    
    return replay_buffer


if __name__ == "__main__":
    # Example usage of the extracted replay buffer functions
    print("="*80)
    print("REPLAY BUFFER FUNCTIONS - DEMONSTRATION")
    print("="*80)
    print("")
    print("This file now contains reusable functions for replay buffer population:")
    print("1. preprocess_mat_data() - Preprocesses .mat files for analysis")  
    print("2. calculate_model_difficulty_metrics() - Calculates NMSE using model predictions")
    print("3. perform_difficulty_clustering() - Performs K-means clustering on NMSE values")
    print("4. create_stratified_replay_buffer() - Creates stratified replay buffer")
    print("")
    print("To populate replay buffers for a trained LoRA model, use:")
    print("python populate_replay_buffers.py --checkpoint_path path/to/lora_run_checkpoint.pth")
    print("")
    print("="*80)
    
    # Demonstrate basic ReplayBuffer functionality
    print("Demonstrating basic ReplayBuffer functionality:")
    buffer = ReplayBuffer(buffer_size=5)
    
    # Create some dummy samples
    for i in range(3):
        dummy_input = torch.randn(2, 72, 70)  # (channels, height, width)
        dummy_target = torch.randn(2, 72, 70)
        buffer.add((dummy_input, dummy_target))
    
    print(f"Buffer size after adding 3 samples: {len(buffer)}")
    
    # Test state dict functionality
    state = buffer.state_dict()
    print(f"State dict contains {len(state['samples'])} samples")
    
    # Test loading
    new_buffer = ReplayBuffer(buffer_size=10)
    new_buffer.load_state_dict(state)
    print(f"New buffer size after loading state: {len(new_buffer)}")
    
    print("✅ Basic ReplayBuffer functionality verified")
