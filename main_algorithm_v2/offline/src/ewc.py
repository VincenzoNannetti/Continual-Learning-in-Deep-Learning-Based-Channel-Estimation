"""
Elastic Weight Consolidation (EWC) for Continual Learning with LoRA + BatchNorm

This module implements EWC for preserving important parameters from previous tasks
during continual learning. It computes Fisher Information matrices for task-specific
LoRA adapters and domain-specific BatchNorm parameters.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
from tqdm import tqdm
import copy

from .lora import LoRAConv2d
from .model import DomainBatchNorm2d


class FisherInformationManager:
    """
    Manages computation and storage of Fisher Information matrices for EWC.
    
    Computes diagonal Fisher Information approximation for task-specific parameters:
    - LoRA adapter parameters (A, B matrices)
    - Domain-specific BatchNorm parameters (weight, bias)
    """
    
    def __init__(self, lambda_ewc: float = 1000.0):
        """
        Initialize Fisher Information Manager.
        
        Args:
            lambda_ewc: EWC regularization strength
        """
        self.lambda_ewc = lambda_ewc
        self.fisher_matrices = {}  # {task_id: {param_name: fisher_diagonal}}
        self.important_params = {}  # {task_id: {param_name: param_values}}
        self.task_param_counts = {}  # Track parameter counts per task
        
        print(f"🧠 FisherInformationManager initialized with λ_EWC = {lambda_ewc}")
    
    def get_task_parameters(self, model, task_id: str) -> Dict[str, torch.Tensor]:
        """
        Extract all task-specific trainable parameters.
        
        Args:
            model: UNet_SRCNN_LoRA model
            task_id: Task identifier
            
        Returns:
            Dictionary of {parameter_name: parameter_tensor}
        """
        task_params = {}
        
        # Extract LoRA parameters for this task
        for name, module in model.named_modules():
            if isinstance(module, LoRAConv2d):
                if task_id in module.task_adapters:
                    adapter = module.task_adapters[task_id]
                    task_params[f"{name}.task_adapters.{task_id}.A"] = adapter.A
                    task_params[f"{name}.task_adapters.{task_id}.B"] = adapter.B
        
        # Extract Domain-specific BatchNorm parameters for this task
        for name, module in model.named_modules():
            if isinstance(module, DomainBatchNorm2d):
                if task_id in module.bns:
                    bn = module.bns[task_id]
                    task_params[f"{name}.bns.{task_id}.weight"] = bn.weight
                    task_params[f"{name}.bns.{task_id}.bias"] = bn.bias
        
        return task_params
    
    def compute_fisher_information(self, model, task_id: str, dataloader, 
                                 device: torch.device, num_samples: Optional[int] = None) -> None:
        """
        Compute Fisher Information diagonal for task-specific parameters.
        
        Uses the diagonal Fisher approximation: F_ii = E[∇log p(y|x)_i²]
        
        Args:
            model: Trained model
            task_id: Current task ID
            dataloader: Validation dataloader for the task
            device: Device to compute on
            num_samples: Limit number of samples (None = use all)
        """
        print(f"\n🔍 Computing Fisher Information for Task {task_id}...")
        
        # Set model to evaluation mode and activate current task
        model.eval()
        model.set_active_task(task_id)
        
        # Get task-specific parameters
        task_params = self.get_task_parameters(model, task_id)
        if not task_params:
            print(f"⚠️  No task-specific parameters found for task {task_id}")
            return
        
        print(f"   Found {len(task_params)} task-specific parameters:")
        param_counts = {'lora_A': 0, 'lora_B': 0, 'bn_weight': 0, 'bn_bias': 0}
        for param_name in task_params.keys():
            if '.A' in param_name:
                param_counts['lora_A'] += 1
            elif '.B' in param_name:
                param_counts['lora_B'] += 1
            elif '.weight' in param_name:
                param_counts['bn_weight'] += 1
            elif '.bias' in param_name:
                param_counts['bn_bias'] += 1
        
        for param_type, count in param_counts.items():
            if count > 0:
                print(f"     {param_type}: {count} parameters")
        
        # Initialize Fisher diagonal accumulators
        fisher_diagonals = {}
        for param_name, param in task_params.items():
            fisher_diagonals[param_name] = torch.zeros_like(param.data)
        
        # Store important parameter values (current optimal values)
        important_params = {}
        for param_name, param in task_params.items():
            important_params[param_name] = param.data.clone()
        
        # Compute Fisher diagonal
        num_samples_processed = 0
        
        # Handle both actual DataLoader and list of tuples (for testing)
        if hasattr(dataloader, 'dataset'):
            total_samples = len(dataloader.dataset)
            data_iterator = enumerate(tqdm(dataloader, desc="Computing Fisher"))
        else:
            # Assume it's a list of (input, target) tuples
            total_samples = sum(inputs.size(0) for inputs, targets in dataloader)
            data_iterator = enumerate(tqdm(dataloader, desc="Computing Fisher"))
        
        criterion = nn.MSELoss()
        
        for batch_idx, (inputs, targets) in data_iterator:
            if num_samples is not None and num_samples_processed >= num_samples:
                break
                
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Zero gradients
            model.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass to compute gradients
            loss.backward()
            
            # Accumulate squared gradients (Fisher diagonal approximation)
            for param_name, param in task_params.items():
                if param.grad is not None:
                    fisher_diagonals[param_name] += param.grad.data ** 2
                else:
                    print(f"⚠️  No gradient for parameter {param_name}")
            
            num_samples_processed += inputs.size(0)
        
        # Average Fisher over samples
        for param_name in fisher_diagonals.keys():
            fisher_diagonals[param_name] /= num_samples_processed
        
        # Store results
        self.fisher_matrices[task_id] = fisher_diagonals
        self.important_params[task_id] = important_params
        self.task_param_counts[task_id] = param_counts
        
        # Statistics
        total_fisher_params = sum(f.numel() for f in fisher_diagonals.values())
        avg_fisher_value = torch.mean(torch.cat([f.flatten() for f in fisher_diagonals.values()])).item()
        
        print(f"✅ Fisher computation complete for Task {task_id}:")
        print(f"   Processed {num_samples_processed} samples")
        print(f"   Total Fisher parameters: {total_fisher_params:,}")
        print(f"   Average Fisher value: {avg_fisher_value:.6f}")
    
    def get_ewc_loss(self, model, current_task_id: str, exclude_current_task: bool = True) -> torch.Tensor:
        """
        Compute EWC regularization loss for previous tasks.
        
        Args:
            model: Current model
            current_task_id: Current task being trained
            exclude_current_task: Whether to exclude current task from penalty
            
        Returns:
            EWC loss tensor
        """
        ewc_loss = torch.tensor(0.0, device=next(model.parameters()).device)
        
        for task_id, fisher_dict in self.fisher_matrices.items():
            # Skip current task if requested
            if exclude_current_task and task_id == current_task_id:
                continue
            
            if task_id not in self.important_params:
                continue
            
            important_dict = self.important_params[task_id]
            current_params = self.get_task_parameters(model, task_id)
            
            task_penalty = torch.tensor(0.0, device=ewc_loss.device)
            
            for param_name in fisher_dict.keys():
                if param_name in important_dict and param_name in current_params:
                    fisher_diag = fisher_dict[param_name].to(ewc_loss.device)
                    important_param = important_dict[param_name].to(ewc_loss.device)
                    current_param = current_params[param_name]
                    
                    # EWC penalty: 0.5 * F * (θ - θ*)²
                    param_penalty = 0.5 * fisher_diag * (current_param - important_param) ** 2
                    task_penalty += param_penalty.sum()
            
            ewc_loss += task_penalty
        
        return self.lambda_ewc * ewc_loss
    
    def state_dict(self) -> Dict[str, Any]:
        """Get state dictionary for saving."""
        return {
            'lambda_ewc': self.lambda_ewc,
            'fisher_matrices': self.fisher_matrices,
            'important_params': self.important_params,
            'task_param_counts': self.task_param_counts
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load state dictionary."""
        self.lambda_ewc = state_dict['lambda_ewc']
        self.fisher_matrices = state_dict['fisher_matrices']
        self.important_params = state_dict['important_params']
        self.task_param_counts = state_dict.get('task_param_counts', {})
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about Fisher information."""
        stats = {
            'num_tasks': len(self.fisher_matrices),
            'lambda_ewc': self.lambda_ewc,
            'tasks': list(self.fisher_matrices.keys())
        }
        
        if self.fisher_matrices:
            # Per-task statistics
            task_stats = {}
            for task_id in self.fisher_matrices.keys():
                fisher_dict = self.fisher_matrices[task_id]
                
                # Calculate statistics
                all_fisher_values = torch.cat([f.flatten() for f in fisher_dict.values()])
                task_stats[task_id] = {
                    'num_parameters': sum(f.numel() for f in fisher_dict.values()),
                    'mean_fisher': all_fisher_values.mean().item(),
                    'std_fisher': all_fisher_values.std().item(),
                    'min_fisher': all_fisher_values.min().item(),
                    'max_fisher': all_fisher_values.max().item(),
                    'param_counts': self.task_param_counts.get(task_id, {})
                }
            
            stats['task_statistics'] = task_stats
            
            # Global statistics
            all_tasks_fisher = torch.cat([
                torch.cat([f.flatten() for f in fisher_dict.values()])
                for fisher_dict in self.fisher_matrices.values()
            ])
            
            stats['global_statistics'] = {
                'total_parameters': all_tasks_fisher.numel(),
                'mean_fisher': all_tasks_fisher.mean().item(),
                'std_fisher': all_tasks_fisher.std().item(),
                'min_fisher': all_tasks_fisher.min().item(),
                'max_fisher': all_tasks_fisher.max().item()
            }
        
        return stats
    
    def print_statistics(self) -> None:
        """Print comprehensive Fisher information statistics."""
        stats = self.get_statistics()
        
        print(f"\n🧠 FISHER INFORMATION STATISTICS:")
        print("-" * 50)
        print(f"   Tasks with Fisher info: {stats['num_tasks']}")
        print(f"   EWC lambda: {stats['lambda_ewc']}")
        
        if 'global_statistics' in stats:
            global_stats = stats['global_statistics']
            print(f"   Total parameters: {global_stats['total_parameters']:,}")
            print(f"   Global Fisher mean: {global_stats['mean_fisher']:.6f}")
            print(f"   Global Fisher std: {global_stats['std_fisher']:.6f}")
            print(f"   Global Fisher range: [{global_stats['min_fisher']:.6f}, {global_stats['max_fisher']:.6f}]")
        
        if 'task_statistics' in stats:
            print(f"\n   Per-task breakdown:")
            for task_id, task_stats in stats['task_statistics'].items():
                print(f"     Task {task_id}: {task_stats['num_parameters']:,} params, "
                      f"Fisher={task_stats['mean_fisher']:.6f}±{task_stats['std_fisher']:.6f}")
                if task_stats['param_counts']:
                    param_info = []
                    for ptype, count in task_stats['param_counts'].items():
                        if count > 0:
                            param_info.append(f"{ptype}:{count}")
                    if param_info:
                        print(f"                {', '.join(param_info)}")


class EWCLoss(nn.Module):
    """
    EWC Loss module for continual learning regularization.
    
    Computes the Elastic Weight Consolidation penalty to prevent forgetting
    of important parameters from previous tasks.
    """
    
    def __init__(self, fisher_manager: FisherInformationManager):
        """
        Initialize EWC Loss.
        
        Args:
            fisher_manager: FisherInformationManager with computed Fisher matrices
        """
        super().__init__()
        self.fisher_manager = fisher_manager
        self.enabled = True
    
    def forward(self, model, current_task_id: str, exclude_current_task: bool = True) -> torch.Tensor:
        """
        Compute EWC loss.
        
        Args:
            model: Model being trained
            current_task_id: Current task ID
            exclude_current_task: Whether to exclude current task from penalty
            
        Returns:
            EWC regularization loss
        """
        if not self.enabled or not self.fisher_manager.fisher_matrices:
            return torch.tensor(0.0, device=next(model.parameters()).device)
        
        return self.fisher_manager.get_ewc_loss(model, current_task_id, exclude_current_task)
    
    def enable(self) -> None:
        """Enable EWC loss computation."""
        self.enabled = True
    
    def disable(self) -> None:
        """Disable EWC loss computation."""
        self.enabled = False
    
    def get_info(self) -> Dict[str, Any]:
        """Get information about EWC configuration."""
        return {
            'enabled': self.enabled,
            'lambda_ewc': self.fisher_manager.lambda_ewc,
            'num_tasks_with_fisher': len(self.fisher_manager.fisher_matrices),
            'tasks': list(self.fisher_manager.fisher_matrices.keys())
        }


def create_ewc_manager(config, lambda_ewc: Optional[float] = None) -> FisherInformationManager:
    """
    Factory function to create EWC manager from config.
    
    Args:
        config: Experiment configuration
        lambda_ewc: Override lambda from config
        
    Returns:
        Configured FisherInformationManager
    """
    # Get lambda from config or use default
    if lambda_ewc is None:
        lambda_ewc = getattr(config.training, 'ewc_lambda', 1000.0)
    
    return FisherInformationManager(lambda_ewc=lambda_ewc) 