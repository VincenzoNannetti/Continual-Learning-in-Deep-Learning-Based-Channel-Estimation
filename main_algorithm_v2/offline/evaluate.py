"""
Evaluation script for the LoRA-based continual learning model.
"""
import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import sys
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from src.config import ExperimentConfig
from src.data import get_dataloaders, get_norm_stats_from_checkpoint
from src.utils import get_device, load_lora_model_for_evaluation

def denormalise(tensor: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """Denormalise a tensor using mean and std."""
    return tensor * (std + 1e-8) + mean

def calculate_metrics(outputs: np.ndarray, targets: np.ndarray) -> dict:
    """Calculates NMSE, PSNR, and SSIM for a batch."""
    nmse = np.sum((outputs - targets) ** 2) / np.sum(targets ** 2)
    
    # Reshape for image metrics if necessary
    if outputs.ndim == 4: # Batch, Channel, H, W
        outputs = outputs.transpose(0, 2, 3, 1) # Batch, H, W, Channel
        targets = targets.transpose(0, 2, 3, 1)
    
    # Calculate PSNR and SSIM sample by sample and average
    psnr_vals = [psnr(t, o, data_range=t.max() - t.min()) for t, o in zip(targets, outputs)]
    # For SSIM, we need a single channel
    ssim_vals = [ssim(t[..., 0], o[..., 0], data_range=t[..., 0].max() - t[..., 0].min()) for t, o in zip(targets, outputs)]
    
    return {
        'nmse': nmse,
        'psnr': np.mean(psnr_vals),
        'ssim': np.mean(ssim_vals)
    }

def evaluate_task(model, task_id, config, device):
    """Evaluate the model on a single task."""
    print(f"\n--- Evaluating Task {task_id} ---")

    # Activate the adapters for the current task
    model.set_active_task(task_id)

    # Get validation data loader for the task
    # We use the validation set for evaluation as a proxy for a test set
    _, val_loader = get_dataloaders(
        task_id=task_id,
        config=config.data,
        batch_size=config.training.batch_size,
        norm_stats=config.data.norm_stats
    )

    all_outputs = []
    all_targets = []

    model.eval()
    with torch.no_grad():
        for inputs, targets in tqdm(val_loader, desc=f"Evaluating Task {task_id}"):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            
            # Denormalise for metric calculation
            mean_i = torch.tensor(config.data.norm_stats.mean_inputs, device=device).view(1, -1, 1, 1)
            std_i = torch.tensor(config.data.norm_stats.std_inputs, device=device).view(1, -1, 1, 1)
            mean_t = torch.tensor(config.data.norm_stats.mean_targets, device=device).view(1, -1, 1, 1)
            std_t = torch.tensor(config.data.norm_stats.std_targets, device=device).view(1, -1, 1, 1)
            
            outputs_denorm = denormalise(outputs, mean_t, std_t)
            targets_denorm = denormalise(targets, mean_t, std_t)

            all_outputs.append(outputs_denorm.cpu().numpy())
            all_targets.append(targets_denorm.cpu().numpy())

    all_outputs = np.concatenate(all_outputs, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Calculate overall metrics for the task
    metrics = calculate_metrics(all_outputs, all_targets)
    print(f"Results for Task {task_id}:")
    for key, value in metrics.items():
        print(f"  - {key.upper()}: {value:.4f}")
    
    return metrics

def main(checkpoint_path: str):
    """Main evaluation function."""
    device = get_device('auto')
    print(f"Using device: {device}")

    # Load the trained model and its config from the checkpoint
    # This utility function handles model reconstruction and state loading
    model = load_lora_model_for_evaluation(checkpoint_path, device)
    config = model.config # The config is attached to the loaded model

    # --- Evaluation Loop ---
    all_task_metrics = {}
    for task_id_int in config.data.sequence:
        task_id = str(task_id_int)
        task_metrics = evaluate_task(model, task_id, config, device)
        all_task_metrics[task_id] = task_metrics

    # --- Final Report ---
    print("\n--- Final Evaluation Report ---")
    for task_id, metrics in all_task_metrics.items():
        print(f"\nTask {task_id}:")
        for key, value in metrics.items():
            print(f"  - {key.upper()}: {value:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate a trained LoRA CL model.")
    parser.add_argument(
        '--checkpoint', 
        type=str, 
        required=True,
        help="Path to the trained model checkpoint (.pth file)."
    )
    args = parser.parse_args()

    # Example command:
    # python main_algorithm_v2/offline/evaluate.py --checkpoint main_algorithm_v2/offline/checkpoints/lora/your_checkpoint.pth
    main(args.checkpoint) 