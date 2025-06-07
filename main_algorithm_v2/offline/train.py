"""
Main training script for the LoRA-based continual learning refactoring.
"""
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import os
from datetime import datetime

# Import from our refactored source directory
from src.config import ExperimentConfig, load_config
from src.model import UNet_SRCNN_LoRA
from src.data import get_dataloaders, get_norm_stats_from_checkpoint
from src.utils import get_device, save_checkpoint, get_checkpoint_filename, create_scheduler, save_lora_checkpoint
from replay_buffer import ReplayBuffer, calculate_model_difficulty_metrics, perform_difficulty_clustering, create_stratified_replay_buffer

def train_one_epoch(model: UNet_SRCNN_LoRA, loader: torch.utils.data.DataLoader, 
                    criterion: nn.Module, optimiser: optim.Optimizer, 
                    device: torch.device, use_amp: bool) -> float:
    """
    Runs a single training epoch.
    """
    model.train()
    total_loss = 0.0
    scaler = GradScaler('cuda', enabled=use_amp)

    for inputs, targets in tqdm(loader, desc="Training"):
        inputs, targets = inputs.to(device), targets.to(device)

        optimiser.zero_grad()

        with autocast('cuda', enabled=use_amp):
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        
        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimiser)
            scaler.update()
        else:
            loss.backward()
            optimiser.step()

        total_loss += loss.item()

    return total_loss / len(loader)

def validate_one_epoch(model: UNet_SRCNN_LoRA, loader: torch.utils.data.DataLoader, 
                       criterion: nn.Module, device: torch.device) -> float:
    """
    Runs a single validation epoch.
    """
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for inputs, targets in tqdm(loader, desc="Validating"):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()

    return total_loss / len(loader)

def populate_replay_buffer(model, train_loader, task_id, device, buffer_size=350):
    """
    Populate replay buffer for the current task using difficulty-based sampling.
    
    Args:
        model: Trained model for current task
        train_loader: Training data loader for current task
        task_id: Current task identifier
        device: Device to run on
        buffer_size: Size of replay buffer
        
    Returns:
        ReplayBuffer: Populated replay buffer
    """
    print(f"\n--- Populating Replay Buffer for Task {task_id} ---")
    
    # Collect all training data
    all_inputs = []
    all_targets = []
    
    model.eval()
    with torch.no_grad():
        for inputs, targets in tqdm(train_loader, desc="Collecting training data"):
            all_inputs.append(inputs.cpu())
            all_targets.append(targets.cpu())
    
    # Concatenate all data
    all_inputs = torch.cat(all_inputs, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    print(f"Collected {all_inputs.shape[0]} training samples for buffer population")
    
    # Calculate difficulty metrics using the trained model
    nmse_values, predictions = calculate_model_difficulty_metrics(
        model, all_inputs, all_targets, device, batch_size=32
    )
    
    # Perform clustering based on difficulty
    cluster_labels, kmeans_model = perform_difficulty_clustering(nmse_values, n_clusters=3)
    
    # Create stratified replay buffer
    replay_buffer = create_stratified_replay_buffer(
        predictions, all_targets, cluster_labels, nmse_values, buffer_size
    )
    
    print(f"Created replay buffer with {len(replay_buffer)} samples")
    return replay_buffer

def main(config_path: str):
    """
    Main function to run the continual learning training process.
    """
    # --- 1. Initialization ---
    config = load_config(config_path)
    torch.manual_seed(config.framework.seed)
    device = get_device(config.hardware.device)
    print(f"Using device: {device}")

    # --- 2. Data and Model Setup ---
    print("Setting up model and data...")
    # Crucially, get norm_stats from the original backbone checkpoint
    norm_stats = get_norm_stats_from_checkpoint(config.model.pretrained_path)
    if norm_stats is None:
        print("Could not find normalisation stats in backbone checkpoint. Exiting.")
        return
    # Store them in the config for saving in the final checkpoint
    config.data.norm_stats = norm_stats
    
    # Initialize the LoRA-adapted model. This class handles backbone loading.
    model = UNet_SRCNN_LoRA(config).to(device)

    # --- 3. Continual Learning Loop ---
    print("\n--- Starting Continual Learning Training ---")
    
    # Use a single loss function for all tasks as defined in config
    criterion = nn.MSELoss() if config.training.loss_function == 'mse' else nn.HuberLoss()

    # Store replay buffers for each task
    replay_buffers = {}

    for task_id_int in config.data.sequence:
        task_id = str(task_id_int)
        print(f"\n--- Task {task_id} ---")

        # Add and activate the adapters for the current task
        model.add_task(task_id)
        model.set_active_task(task_id)

        # Get data for the current task
        train_loader, val_loader = get_dataloaders(
            task_id=task_id,
            config=config.data,
            batch_size=config.training.batch_size,
            norm_stats=config.data.norm_stats
        )

        # The optimiser is created for each task, targeting only the newly trainable params
        optimiser = optim.Adam(
            model.trainable_parameters(), 
            lr=config.training.learning_rate,
            weight_decay=config.training.task_weight_decays.get(task_id_int, config.training.weight_decay)
        )
        
        # Training sub-loop for the current task
        for epoch in range(config.training.epochs_per_task):
            print(f"Epoch {epoch+1}/{config.training.epochs_per_task}")
            
            train_loss = train_one_epoch(model, train_loader, criterion, optimiser, device, config.hardware.use_amp)
            val_loss = validate_one_epoch(model, val_loader, criterion, device)
            
            print(f"Epoch {epoch+1} Summary: Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

        # Populate replay buffer after training this task
        replay_buffer = populate_replay_buffer(model, train_loader, task_id, device)
        replay_buffers[task_id] = replay_buffer
        
        print(f"Task {task_id} training completed. Best validation loss: {val_loss:.6f}")

    print("\n--- Continual Learning Training Finished ---")

    # --- 4. Final Checkpoint ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_filename = f"{config.experiment_name}-{timestamp}.pth"
    checkpoint_path = os.path.join(config.logging.checkpoint_dir, checkpoint_filename)
    
    # Ensure checkpoint directory exists
    os.makedirs(config.logging.checkpoint_dir, exist_ok=True)
    
    save_lora_checkpoint(model, config, checkpoint_path, replay_buffers)
    print(f"Training completed! Checkpoint saved: {checkpoint_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run LoRA Continual Learning Training")
    parser.add_argument(
        '--config', 
        type=str, 
        required=True,
        help="Path to the experiment configuration YAML file."
    )
    args = parser.parse_args()
    
    # Example command:
    # python main_algorithm_v2/offline/train.py --config main_algorithm_v2/offline/config/unet_srcnn_refactored.yaml
    main(args.config) 