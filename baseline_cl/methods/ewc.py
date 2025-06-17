"""
Elastic Weight Consolidation (EWC) Implementation

Based on the paper:
"Overcoming catastrophic forgetting in neural networks"
by Kirkpatrick et al., 2017

This is a clean, literature-standard implementation for baseline comparison.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, List, Optional
import numpy as np
from .base_trainer import BaseContinualTrainer


class EWCTrainer(BaseContinualTrainer):
    """
    Elastic Weight Consolidation (EWC) trainer.
    
    EWC prevents catastrophic forgetting by adding a regularisation term that
    penalises changes to parameters that are important for previous tasks.
    
    The importance of parameters is measured by the Fisher Information Matrix.
    """
    
    def __init__(self, model: nn.Module, config: dict, device: torch.device, 
                 ewc_lambda: float = 1000.0, fisher_samples: Optional[int] = None, wandb_run=None):
        """
        Initialize EWC trainer.
        
        Args:
            model: Neural network model
            config: Configuration dictionary
            device: Device to run on
            ewc_lambda: EWC regularisation strength (typical: 1000-10000)
            fisher_samples: Number of samples to use for Fisher computation (None = use all)
            wandb_run: Weights & Biases run object
        """
        super().__init__(model, config, device, wandb_run)
        
        self.ewc_lambda = ewc_lambda
        self.fisher_samples = fisher_samples
        
        # EWC-specific state
        self.fisher_matrices = {}  # Fisher information for each task
        self.optimal_params = {}   # Optimal parameters for each task
        self.task_importance = {}  # Overall importance score per task
        
        print(f" EWC initialized with λ = {ewc_lambda}")
        if fisher_samples:
            print(f"   Fisher computation will use {fisher_samples} samples per task")
        else:
            print(f"   Fisher computation will use all available validation samples")
    
    def add_regularization_loss(self, base_loss: torch.Tensor, inputs: torch.Tensor, 
                               targets: torch.Tensor, outputs: torch.Tensor) -> torch.Tensor:
        """
        Add EWC regularisation term to the base loss.
        
        EWC Loss = Base Loss + λ * Σ_previous_tasks F_i * (θ_i - θ*_i)^2
        """
        if not self.fisher_matrices:
            # No previous tasks, return base loss
            return base_loss
        
        ewc_loss = 0.0
        current_params = dict(self.model.named_parameters())
        
        for task_id in self.fisher_matrices:
            fisher = self.fisher_matrices[task_id]
            optimal = self.optimal_params[task_id]
            
            task_loss = 0.0
            for param_name, param in current_params.items():
                if param_name in fisher and param.requires_grad:
                    # Calculate (θ - θ*)^2 * F
                    param_diff = param - optimal[param_name]
                    fisher_weight = fisher[param_name]
                    task_loss += (fisher_weight * param_diff.pow(2)).sum()
            
            ewc_loss += task_loss
        
        # Scale by lambda and add to base loss
        total_loss = base_loss + (self.ewc_lambda * ewc_loss)
        
        # Log EWC loss components for debugging
        if self.wandb_run and len(self.fisher_matrices) > 0:
            self.wandb_run.log({
                'ewc_regularization_loss': float(ewc_loss.item()) if isinstance(ewc_loss, torch.Tensor) else float(ewc_loss),
                'base_task_loss': float(base_loss.item()),
                'total_loss_with_ewc': float(total_loss.item())
            })
        
        return total_loss
    
    def prepare_for_task(self, task_id: str, train_loader: DataLoader):
        """
        Prepare EWC for a new task.
        For EWC, this just involves tracking the new task.
        """
        self.current_task_id = task_id
        print(f" EWC preparing for task {task_id}")
        
        if len(self.fisher_matrices) > 0:
            print(f"   Previous tasks: {list(self.fisher_matrices.keys())}")
            print(f"   EWC regularisation will be applied with λ = {self.ewc_lambda}")
    
    def after_task_completion(self, task_id: str, val_loader: DataLoader):
        """
        Compute Fisher Information Matrix after completing a task.
        
        This is the core of EWC - we estimate the importance of each parameter
        for the task we just learned.
        """
        print(f"\n Computing Fisher Information Matrix for task {task_id}...")
        
        # Set model to evaluation mode for Fisher computation
        self.model.eval()
        
        # Initialize Fisher matrix and parameter storage
        fisher_matrix = {}
        optimal_params = {}
        
        # Store optimal parameters (current parameters after training)
        for param_name, param in self.model.named_parameters():
            if param.requires_grad:
                optimal_params[param_name] = param.data.clone()
                fisher_matrix[param_name] = torch.zeros_like(param.data)
        
        # Compute Fisher Information using validation data
        sample_count = 0
        max_samples = self.fisher_samples if self.fisher_samples else float('inf')
        
        print(f"   Computing Fisher using validation data...")
        with torch.enable_grad():
            for batch_idx, (inputs, targets) in enumerate(val_loader):
                if sample_count >= max_samples:
                    break
                
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                batch_size = inputs.size(0)
                
                # Zero gradients
                self.model.zero_grad()
                
                # Forward pass
                outputs = self.model(inputs)
                
                # Compute loss for each sample individually (required for proper Fisher computation)
                for i in range(batch_size):
                    if sample_count >= max_samples:
                        break
                    
                    # Single sample loss
                    sample_loss = self.criterion(outputs[i:i+1], targets[i:i+1])
                    
                    # Compute gradients
                    self.model.zero_grad()
                    sample_loss.backward(retain_graph=True)
                    
                    # Accumulate squared gradients (Fisher Information)
                    for param_name, param in self.model.named_parameters():
                        if param.requires_grad and param.grad is not None:
                            fisher_matrix[param_name] += param.grad.data.pow(2)
                    
                    sample_count += 1
                
                if (batch_idx + 1) % 10 == 0:
                    print(f"      Processed {sample_count} samples...")
        
        # Normalize by number of samples
        if sample_count > 0:
            for param_name in fisher_matrix:
                fisher_matrix[param_name] /= sample_count
        
        # Store Fisher matrix and optimal parameters
        self.fisher_matrices[task_id] = fisher_matrix
        self.optimal_params[task_id] = optimal_params
        
        # Calculate importance statistics
        total_fisher = sum(fisher.sum().item() for fisher in fisher_matrix.values())
        num_params = sum(fisher.numel() for fisher in fisher_matrix.values())
        mean_fisher = total_fisher / num_params if num_params > 0 else 0.0
        
        self.task_importance[task_id] = {
            'total_fisher': total_fisher,
            'mean_fisher': mean_fisher,
            'num_parameters': num_params,
            'samples_used': sample_count
        }
        
        print(f"    Fisher computation complete for task {task_id}")
        print(f"      Samples used: {sample_count}")
        print(f"      Parameters tracked: {num_params:,}")
        print(f"      Mean Fisher value: {mean_fisher:.2e}")
        print(f"      Total Fisher magnitude: {total_fisher:.2e}")
        
        # Log Fisher statistics to W&B
        if self.wandb_run:
            self.wandb_run.log({
                f'fisher_task_{task_id}_total': total_fisher,
                f'fisher_task_{task_id}_mean': mean_fisher,
                f'fisher_task_{task_id}_params': num_params,
                f'fisher_task_{task_id}_samples': sample_count,
                'ewc_lambda': self.ewc_lambda,
                'fisher_tasks_computed': len(self.fisher_matrices)
            })
        
        # Add to completed tasks
        if task_id not in self.completed_tasks:
            self.completed_tasks.append(task_id)
        
        # Set model back to training mode
        self.model.train()
    
    def get_method_state(self) -> Dict:
        """Get EWC-specific state for checkpointing."""
        return {
            'ewc_lambda': self.ewc_lambda,
            'fisher_samples': self.fisher_samples,
            'fisher_matrices': self.fisher_matrices,
            'optimal_params': self.optimal_params,
            'task_importance': self.task_importance,
            'completed_tasks': self.completed_tasks
        }
    
    def load_method_state(self, state: Dict):
        """Load EWC-specific state from checkpoint."""
        self.ewc_lambda = state.get('ewc_lambda', self.ewc_lambda)
        self.fisher_samples = state.get('fisher_samples', self.fisher_samples)
        self.fisher_matrices = state.get('fisher_matrices', {})
        self.optimal_params = state.get('optimal_params', {})
        self.task_importance = state.get('task_importance', {})
        self.completed_tasks = state.get('completed_tasks', [])
        
        print(f" EWC state loaded:")
        print(f"   Fisher matrices for {len(self.fisher_matrices)} tasks")
        print(f"   Completed tasks: {self.completed_tasks}")
        print(f"   λ = {self.ewc_lambda}")
    
    def print_ewc_statistics(self):
        """Print detailed EWC statistics."""
        if not self.task_importance:
            print("No EWC statistics available (no tasks completed)")
            return
        
        print(f"\n EWC Statistics:")
        print(f"   λ_EWC = {self.ewc_lambda}")
        print(f"   Tasks with Fisher matrices: {len(self.fisher_matrices)}")
        
        for task_id, stats in self.task_importance.items():
            print(f"   Task {task_id}:")
            print(f"      Parameters: {stats['num_parameters']:,}")
            print(f"      Samples used: {stats['samples_used']}")
            print(f"      Mean Fisher: {stats['mean_fisher']:.2e}")
            print(f"      Total Fisher: {stats['total_fisher']:.2e}")
    
    def get_current_ewc_loss(self) -> float:
        """
        Calculate current EWC regularisation loss.
        Useful for monitoring during training.
        """
        if not self.fisher_matrices:
            return 0.0
        
        ewc_loss = 0.0
        current_params = dict(self.model.named_parameters())
        
        for task_id in self.fisher_matrices:
            fisher = self.fisher_matrices[task_id]
            optimal = self.optimal_params[task_id]
            
            for param_name, param in current_params.items():
                if param_name in fisher and param.requires_grad:
                    param_diff = param - optimal[param_name]
                    fisher_weight = fisher[param_name]
                    ewc_loss += (fisher_weight * param_diff.pow(2)).sum().item()
        
        return float(ewc_loss) 