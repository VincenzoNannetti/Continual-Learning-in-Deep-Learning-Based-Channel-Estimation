"""
Elastic Weight Consolidation (EWC) for Online Continual Learning with LoRA + BatchNorm

This module implements EWC for preserving important parameters from previous tasks
during online continual learning. It computes Fisher Information matrices for task-specific
LoRA adapters and domain-specific BatchNorm parameters.

Adapted for online training environment.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Union
from tqdm import tqdm
import copy

# Import offline components (using absolute imports)
from main_algorithm_v2.offline.src.lora import LoRAConv2d
from main_algorithm_v2.offline.src.model import DomainBatchNorm2d


class OnlineFisherInformationManager:
    """
    Manages computation and storage of Fisher Information matrices for online EWC.
    
    Computes diagonal Fisher Information approximation for task-specific parameters:
    - LoRA adapter parameters (A, B matrices)
    - Domain-specific BatchNorm parameters (weight, bias)
    
    Optimized for online training with:
    - Loading pre-computed Fisher matrices from offline training
    - Computing Fisher information from small online batches
    - Managing Fisher updates during online learning
    """
    
    def __init__(self, lambda_ewc: float = 1000.0):
        """
        Initialize Online Fisher Information Manager.
        
        Args:
            lambda_ewc: EWC regularization strength
        """
        self.lambda_ewc = lambda_ewc
        self.fisher_matrices = {}  # {task_id: {param_name: fisher_diagonal}}
        self.important_params = {}  # {task_id: {param_name: param_values}}
        self.task_param_counts = {}  # Track parameter counts per task
        
        print(f"🧠 OnlineFisherInformationManager initialized with λ_EWC = {lambda_ewc}")
    
    def load_from_offline_checkpoint(self, checkpoint_path: str) -> bool:
        """
        Load pre-computed Fisher matrices from offline training checkpoint.
        
        Args:
            checkpoint_path: Path to offline checkpoint with Fisher matrices
            
        Returns:
            True if successfully loaded, False otherwise
        """
        try:
            print(f"🔄 Loading Fisher matrices from offline checkpoint...")
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            
            if 'fisher_manager' not in checkpoint or checkpoint['fisher_manager'] is None:
                print("❌ No Fisher manager found in checkpoint")
                return False
            
            fisher_data = checkpoint['fisher_manager']
            
            # Load Fisher manager state
            self.load_state_dict(fisher_data)
            
            print(f"✅ Fisher matrices loaded successfully:")
            print(f"   Tasks: {list(self.fisher_matrices.keys())}")
            print(f"   λ_EWC: {self.lambda_ewc}")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to load Fisher matrices: {e}")
            return False
    
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
    
    def compute_fisher_information_online(self, model, task_id: str, 
                                        batch_data: List[Tuple[torch.Tensor, torch.Tensor]], 
                                        device: torch.device) -> None:
        """
        Compute Fisher Information diagonal for task-specific parameters from online batch.
        
        Optimized for online learning with small batches.
        
        Args:
            model: Trained model
            task_id: Current task ID
            batch_data: List of (input, target) tuples for Fisher computation
            device: Device to compute on
        """
        print(f"🔍 Computing online Fisher Information for Task {task_id}...")
        
        # Set model to evaluation mode and activate current task
        model.eval()
        model.set_active_task(task_id)
        
        # Get task-specific parameters
        task_params = self.get_task_parameters(model, task_id)
        if not task_params:
            print(f"⚠️  No task-specific parameters found for task {task_id}")
            return
        
        # Initialize Fisher diagonal accumulators
        fisher_diagonals = {}
        for param_name, param in task_params.items():
            fisher_diagonals[param_name] = torch.zeros_like(param.data)
        
        # Store important parameter values (current optimal values)
        important_params = {}
        for param_name, param in task_params.items():
            important_params[param_name] = param.data.clone()
        
        # Compute Fisher diagonal from batch
        num_samples_processed = 0
        criterion = nn.MSELoss()
        
        for inputs, targets in batch_data:
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
            
            num_samples_processed += inputs.size(0)
        
        # Average Fisher over samples
        if num_samples_processed > 0:
            for param_name in fisher_diagonals.keys():
                fisher_diagonals[param_name] /= num_samples_processed
        
        # Store or update results
        self.fisher_matrices[task_id] = fisher_diagonals
        self.important_params[task_id] = important_params
        
        print(f"✅ Online Fisher computation complete for Task {task_id}")
        print(f"   Processed {num_samples_processed} samples")
    
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
            
            # Get current parameters for this task
            current_params = self.get_task_parameters(model, task_id)
            important_params = self.important_params[task_id]
            
            # Compute EWC loss for this task
            task_loss = torch.tensor(0.0, device=ewc_loss.device)
            
            for param_name in fisher_dict.keys():
                if param_name in current_params and param_name in important_params:
                    fisher_info = fisher_dict[param_name]
                    current_param = current_params[param_name]
                    important_param = important_params[param_name]
                    
                    # EWC penalty: F_i * (θ_i - θ_i*)^2
                    # Ensure all tensors are on the same device
                    fisher_info = fisher_info.to(current_param.device)
                    important_param = important_param.to(current_param.device)
                    penalty = fisher_info * (current_param - important_param) ** 2
                    task_loss += penalty.sum()
            
            ewc_loss += task_loss
        
        # Apply EWC weight
        ewc_loss *= (self.lambda_ewc / 2.0)
        
        return ewc_loss
    
    def update_important_params(self, model, task_id: str) -> None:
        """
        Update the stored important parameters for a task after online training.
        
        Args:
            model: Model with updated parameters
            task_id: Task whose parameters to update
        """
        if task_id in self.important_params:
            current_params = self.get_task_parameters(model, task_id)
            for param_name, param in current_params.items():
                self.important_params[task_id][param_name] = param.data.clone()
            print(f"✅ Updated important parameters for Task {task_id}")
    
    def state_dict(self) -> Dict[str, Any]:
        """
        Return state dictionary for checkpointing.
        """
        return {
            'lambda_ewc': self.lambda_ewc,
            'fisher_matrices': self.fisher_matrices,
            'important_params': self.important_params,
            'task_param_counts': self.task_param_counts
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """
        Load state from dictionary.
        """
        self.lambda_ewc = state_dict.get('lambda_ewc', self.lambda_ewc)
        self.fisher_matrices = state_dict.get('fisher_matrices', {})
        self.important_params = state_dict.get('important_params', {}) 
        self.task_param_counts = state_dict.get('task_param_counts', {})
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about Fisher Information.
        """
        if not self.fisher_matrices:
            return {"error": "No Fisher matrices computed"}
        
        stats = {
            'lambda_ewc': self.lambda_ewc,
            'num_tasks': len(self.fisher_matrices),
            'tasks': list(self.fisher_matrices.keys()),
            'total_parameters': 0,
            'global_fisher_stats': {},
            'per_task_stats': {}
        }
        
        # Collect all Fisher values
        all_fisher_values = []
        
        for task_id, fisher_dict in self.fisher_matrices.items():
            task_stats = {
                'num_parameters': 0,
                'parameter_types': {},
                'fisher_mean': 0.0,
                'fisher_std': 0.0,
                'fisher_min': float('inf'),
                'fisher_max': float('-inf')
            }
            
            task_fisher_values = []
            
            for param_name, fisher_tensor in fisher_dict.items():
                task_stats['num_parameters'] += fisher_tensor.numel()
                
                # Categorize parameter type
                if '.A' in param_name:
                    param_type = 'lora_A'
                elif '.B' in param_name:
                    param_type = 'lora_B'
                elif '.weight' in param_name:
                    param_type = 'bn_weight'
                elif '.bias' in param_name:
                    param_type = 'bn_bias'
                else:
                    param_type = 'other'
                
                task_stats['parameter_types'][param_type] = task_stats['parameter_types'].get(param_type, 0) + 1
                
                # Collect Fisher values
                fisher_vals = fisher_tensor.flatten().tolist()
                task_fisher_values.extend(fisher_vals)
                all_fisher_values.extend(fisher_vals)
            
            if task_fisher_values:
                task_fisher_tensor = torch.tensor(task_fisher_values)
                task_stats['fisher_mean'] = task_fisher_tensor.mean().item()
                task_stats['fisher_std'] = task_fisher_tensor.std().item()
                task_stats['fisher_min'] = task_fisher_tensor.min().item()
                task_stats['fisher_max'] = task_fisher_tensor.max().item()
            
            stats['per_task_stats'][task_id] = task_stats
            stats['total_parameters'] += task_stats['num_parameters']
        
        # Global statistics
        if all_fisher_values:
            global_fisher_tensor = torch.tensor(all_fisher_values)
            stats['global_fisher_stats'] = {
                'mean': global_fisher_tensor.mean().item(),
                'std': global_fisher_tensor.std().item(),
                'min': global_fisher_tensor.min().item(),
                'max': global_fisher_tensor.max().item()
            }
        
        return stats
    
    def print_statistics(self) -> None:
        """
        Print comprehensive Fisher Information statistics.
        """
        stats = self.get_statistics()
        
        if 'error' in stats:
            print(f"❌ {stats['error']}")
            return
        
        print("\n🧠 FISHER INFORMATION STATISTICS:")
        print("-" * 50)
        print(f"   Tasks with Fisher info: {stats['num_tasks']}")
        print(f"   EWC lambda: {stats['lambda_ewc']}")
        print(f"   Total parameters: {stats['total_parameters']:,}")
        
        if 'global_fisher_stats' in stats and stats['global_fisher_stats']:
            global_stats = stats['global_fisher_stats']
            print(f"   Global Fisher mean: {global_stats['mean']:.6f}")
            print(f"   Global Fisher std: {global_stats['std']:.6f}")
            print(f"   Global Fisher range: [{global_stats['min']:.6f}, {global_stats['max']:.6f}]")
        
        print(f"\n   Per-task breakdown:")
        for task_id, task_stats in stats['per_task_stats'].items():
            print(f"     Task {task_id}: {task_stats['num_parameters']:,} params, "
                  f"Fisher={task_stats['fisher_mean']:.6f}±{task_stats['fisher_std']:.6f}")
            
            param_types = task_stats['parameter_types']
            type_str = ", ".join([f"{k}:{v}" for k, v in param_types.items()])
            print(f"                {type_str}")


class OnlineEWCLoss(nn.Module):
    """
    Online EWC Loss module for computing regularization penalty during online training.
    
    Optimized for online continual learning scenarios.
    """
    
    def __init__(self, fisher_manager: OnlineFisherInformationManager):
        """
        Initialize EWC Loss module.
        
        Args:
            fisher_manager: Fisher Information manager with pre-computed matrices
        """
        super().__init__()
        self.fisher_manager = fisher_manager
        self.enabled = True
    
    def forward(self, model, current_task_id: str, exclude_current_task: bool = True) -> torch.Tensor:
        """
        Compute EWC regularization loss.
        
        Args:
            model: Current model being trained
            current_task_id: Current task ID
            exclude_current_task: Whether to exclude current task from penalty
            
        Returns:
            EWC loss tensor
        """
        if not self.enabled:
            return torch.tensor(0.0, device=next(model.parameters()).device)
        
        return self.fisher_manager.get_ewc_loss(model, current_task_id, exclude_current_task)
    
    def enable(self) -> None:
        """Enable EWC loss computation."""
        self.enabled = True
    
    def disable(self) -> None:
        """Disable EWC loss computation."""
        self.enabled = False
    
    def get_info(self) -> Dict[str, Any]:
        """Get information about EWC loss state."""
        return {
            'enabled': self.enabled,
            'lambda_ewc': self.fisher_manager.lambda_ewc,
            'num_tasks_with_fisher': len(self.fisher_manager.fisher_matrices)
        }


def create_online_ewc_manager(lambda_ewc: float = 1000.0, 
                             offline_checkpoint_path: Optional[str] = None) -> OnlineFisherInformationManager:
    """
    Create and initialize an Online EWC Fisher Information Manager.
    
    Args:
        lambda_ewc: EWC regularization strength
        offline_checkpoint_path: Path to offline checkpoint with pre-computed Fisher matrices
        
    Returns:
        Initialized OnlineFisherInformationManager
    """
    fisher_manager = OnlineFisherInformationManager(lambda_ewc=lambda_ewc)
    
    if offline_checkpoint_path:
        success = fisher_manager.load_from_offline_checkpoint(offline_checkpoint_path)
        if not success:
            print("⚠️  Failed to load offline Fisher matrices, starting with empty manager")
    
    return fisher_manager 