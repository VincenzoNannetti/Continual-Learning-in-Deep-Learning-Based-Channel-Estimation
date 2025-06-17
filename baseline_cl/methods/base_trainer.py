"""
Base Trainer Class for Continual Learning Methods

This provides common functionality for all baseline continual learning approaches.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import os
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class BaseContinualTrainer(ABC):
    """
    Base class for all continual learning trainers.
    Provides common functionality and defines the interface.
    """
    
    def __init__(self, model: nn.Module, config: dict, device: torch.device, wandb_run=None):
        """
        Initialize the base trainer.
        
        Args:
            model: The neural network model
            config: Configuration dictionary
            device: Device to run on
            wandb_run: Weights & Biases run object (optional)
        """
        self.model = model
        self.config = config
        self.device = device
        self.wandb_run = wandb_run
        
        # Training components
        self.criterion = self._create_criterion()
        self.scaler = GradScaler(enabled=config.get('use_amp', False))
        
        # Tracking
        self.current_task_id = None
        self.completed_tasks = []
        self.best_val_losses = {}
        self.training_history = {}
        
    def _create_criterion(self) -> nn.Module:
        """Create loss criterion based on config."""
        loss_fn = self.config.get('loss_function', 'mse').lower()
        if loss_fn == 'mse':
            return nn.MSELoss()
        elif loss_fn == 'huber':
            return nn.HuberLoss()
        else:
            raise ValueError(f"Unsupported loss function: {loss_fn}")
    
    def _create_optimizer(self, parameters):
        """Create optimizer based on config."""
        lr = float(self.config.get('learning_rate', 1e-4))
        weight_decay = float(self.config.get('weight_decay', 0.0))
        
        if self.config.get('optimizer', 'adam').lower() == 'sgd':
            momentum = float(self.config.get('momentum', 0.9))
            return optim.SGD(parameters, lr=lr, momentum=momentum, weight_decay=weight_decay)
        else:
            return optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
    
    def denormalise(self, tensor: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        """Denormalise a tensor using mean and std."""
        return tensor * (std + 1e-8) + mean
    
    def calculate_metrics(self, outputs: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
        """Calculate NMSE, PSNR, and SSIM for a batch using correct standard_training_2 implementation."""
        # Import correct metrics here to avoid import errors
        from scipy.signal import convolve2d
        
        # Convert from (N, C, H, W) to (N, H, W, C) if needed for complex data handling
        if outputs.ndim == 4 and outputs.shape[1] == 2:
            outputs = outputs.transpose(0, 2, 3, 1)  # (N, H, W, 2)
            targets = targets.transpose(0, 2, 3, 1)  # (N, H, W, 2)

        # Calculate NMSE using standard_training_2 implementation
        mse = np.mean((outputs - targets) ** 2)
        power = np.mean(targets ** 2)
        if power == 0:
            nmse_val = float('inf') if mse > 0 else 0.0
        else:
            nmse_val = mse / power
        
        # Calculate PSNR using standard_training_2 implementation
        max_val = np.max(targets)
        if max_val == np.min(targets):
            max_val = 1.0 if max_val == 0 else np.abs(max_val)
        
        if mse == 0:
            psnr_val = float('inf')
        elif max_val == 0:
            psnr_val = -float('inf') if mse > 0 else float('inf')
        else:
            psnr_val = 20 * np.log10(max_val / np.sqrt(mse))
        
        # Calculate SSIM on complex magnitude using standard_training_2 implementation
        if outputs.shape[-1] >= 2:  # Complex data with real/imag channels
            # Compute complex magnitudes
            pred_mag = np.abs(outputs[..., 0] + 1j * outputs[..., 1])  # shape (N, H, W)
            target_mag = np.abs(targets[..., 0] + 1j * targets[..., 1])
            
            N, H, W = pred_mag.shape
            
            # Per-sample normalization to [0, 1]
            def normalize_per_sample(arr):
                out = np.zeros_like(arr)
                for i in range(arr.shape[0]):
                    mn = arr[i].min()
                    mx = arr[i].max()
                    if mx > mn:
                        out[i] = (arr[i] - mn) / (mx - mn)
                    else:
                        out[i] = np.zeros_like(arr[i])
                return out
            
            pred_norm = normalize_per_sample(pred_mag)
            target_norm = normalize_per_sample(target_mag)
            
            # Build 2D Gaussian kernel
            window_size = 11
            sigma = 1.5
            half = window_size // 2
            coords = np.arange(window_size) - half
            g1d = np.exp(- (coords**2) / (2 * sigma**2))
            g1d /= g1d.sum()
            kernel = np.outer(g1d, g1d)
            
            # Constants for SSIM
            data_range = 1.0
            k1, k2 = 0.01, 0.03
            c1 = (k1 * data_range) ** 2
            c2 = (k2 * data_range) ** 2
            
            ssim_vals = np.zeros(N, dtype=np.float64)
            
            for i in range(N):
                x = pred_norm[i]
                y = target_norm[i]
                
                # Compute local means via 2D convolution
                mu_x = convolve2d(x, kernel, mode='same', boundary='symm')
                mu_y = convolve2d(y, kernel, mode='same', boundary='symm')
                
                # Compute local squares and cross-products
                x_sq = x * x
                y_sq = y * y
                xy = x * y
                
                mu_x_sq = mu_x * mu_x
                mu_y_sq = mu_y * mu_y
                mu_xy = mu_x * mu_y
                
                # Compute variances and covariance
                sigma_x_sq = convolve2d(x_sq, kernel, mode='same', boundary='symm') - mu_x_sq
                sigma_y_sq = convolve2d(y_sq, kernel, mode='same', boundary='symm') - mu_y_sq
                sigma_xy = convolve2d(xy, kernel, mode='same', boundary='symm') - mu_xy
                
                # Compute SSIM map
                numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
                denominator = (mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2)
                
                ssim_map = np.where(denominator > 0, numerator / denominator, 0.0)
                ssim_vals[i] = ssim_map.mean()
            
            ssim_val = float(ssim_vals.mean())
        else:
            # Fallback for single channel data
            ssim_val = 0.5  # Placeholder value
        
        return {
            'nmse': float(nmse_val),
            'psnr': float(psnr_val),
            'ssim': float(ssim_val)
        }
    
    def train_epoch(self, train_loader: torch.utils.data.DataLoader, 
                   optimizer: optim.Optimizer, epoch: int) -> float:
        """
        Train for one epoch with base functionality.
        Can be overridden by subclasses for method-specific training.
        """
        self.model.train()
        total_loss = 0.0
        use_amp = self.config.get('use_amp', False)
        
        for inputs, targets in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            optimizer.zero_grad()
            
            with autocast(device_type='cuda', enabled=use_amp):
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                # Add method-specific loss terms
                loss = self.add_regularization_loss(loss, inputs, targets, outputs)
            
            if use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                loss.backward()
                optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def validate_epoch(self, val_loader: torch.utils.data.DataLoader) -> float:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        use_amp = self.config.get('use_amp', False)
        
        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc="Validating"):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                with autocast(device_type='cuda', enabled=use_amp):
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                
                total_loss += loss.item()
        
        return total_loss / len(val_loader)
    
    def evaluate_domain(self, task_id: str, val_loader: torch.utils.data.DataLoader, 
                       norm_stats: Dict) -> Dict[str, float]:
        """
        Evaluate the model on a single domain.
        
        Args:
            task_id: Domain identifier
            val_loader: Validation data loader
            norm_stats: Normalisation statistics
            
        Returns:
            Dictionary of metrics
        """
        self.model.eval()
        all_outputs = []
        all_targets = []
        
        mean_t = torch.tensor(norm_stats['mean_targets'], device=self.device).view(1, -1, 1, 1)
        std_t = torch.tensor(norm_stats['std_targets'], device=self.device).view(1, -1, 1, 1)
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                
                # Denormalise for metric calculation
                outputs_denorm = self.denormalise(outputs, mean_t, std_t)
                targets_denorm = self.denormalise(targets, mean_t, std_t)
                
                all_outputs.append(outputs_denorm.cpu().numpy())
                all_targets.append(targets_denorm.cpu().numpy())
        
        # Concatenate all batches
        all_outputs = np.concatenate(all_outputs, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        
        # Calculate metrics
        metrics = self.calculate_metrics(all_outputs, all_targets)
        
        print(f"    Domain {task_id}: SSIM={metrics['ssim']:.4f}, NMSE={metrics['nmse']:.8f}, PSNR={metrics['psnr']:.2f}")
        
        return metrics
    
    @abstractmethod
    def add_regularization_loss(self, base_loss: torch.Tensor, inputs: torch.Tensor, 
                               targets: torch.Tensor, outputs: torch.Tensor) -> torch.Tensor:
        """
        Add method-specific regularisation to the base loss.
        Must be implemented by subclasses.
        """
        pass
    
    @abstractmethod
    def prepare_for_task(self, task_id: str, train_loader: torch.utils.data.DataLoader):
        """
        Prepare the method for training on a new task.
        Must be implemented by subclasses.
        """
        pass
    
    @abstractmethod
    def after_task_completion(self, task_id: str, train_loader: torch.utils.data.DataLoader):
        """
        Perform any necessary operations after completing a task.
        Must be implemented by subclasses.
        """
        pass
    
    def save_checkpoint(self, task_id: str, epoch: int, val_loss: float, 
                       save_path: str, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'task_id': task_id,
            'epoch': epoch,
            'val_loss': val_loss,
            'completed_tasks': self.completed_tasks.copy(),
            'best_val_losses': self.best_val_losses.copy(),
            'method_specific_state': self.get_method_state()
        }
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(checkpoint, save_path)
        
        if is_best:
            print(f" NEW BEST for Domain {task_id}: {val_loss:.6f} (Epoch {epoch+1})")
    
    @abstractmethod
    def get_method_state(self) -> Dict:
        """
        Get method-specific state for checkpointing.
        Must be implemented by subclasses.
        """
        pass
    
    @abstractmethod
    def load_method_state(self, state: Dict):
        """
        Load method-specific state from checkpoint.
        Must be implemented by subclasses.
        """
        pass 