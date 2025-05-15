"""
Filename: continual_learning/evaluators/supermask_evaluator.py
Author: Vincenzo Nannetti
Date: 15/05/2025
Description: Supermask Evaluation Class - for evaluating supermask models on different tasks

Dependencies:
    - PyTorch
    - numpy
    - matplotlib
    - wandb
"""

import os
import time
import torch
import wandb
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from datetime import datetime

from shared.utils.get_device import get_device
from shared.utils.format_time import format_time
from shared.utils.training_utils import get_criterion
from continual_learning.utils.supermask_utils import set_model_task
from standard_training.datasets.dataset_utils import NormalisedDatasetWrapper

class SupermaskEvaluator:
    def __init__(self, config, checkpoint_dir=None):
        """
        Initialise the supermask evaluator.
        
        Args:
            config (dict): Configuration dictionary
            checkpoint_dir (str, optional): Directory where checkpoints are stored.
                If None, will use the checkpoint directory from the config.
        """
        self.config = config
        self.device = get_device(config)
        
        # Extract configs
        self.supermask_config = config.get('supermask', {})
        self.data_config      = config.get('data', {})
        self.eval_config      = config.get('evaluation', {})
        self.hardware_config  = config.get('hardware', {})
        self.framework_config = config.get('framework', {})
        self.logging_config   = config.get('logging', {})
        
        # Set up parameters
        self.num_tasks        = self.supermask_config.get('tasks', None)
        self.val_split_ratio  = self.data_config.get('validation_split', 0.15)
        self.test_split_ratio = self.data_config.get('test_split', 0.15)
        self.split_seed       = self.framework_config.get('seed', 42)
        self.batch_size       = self.config.get('training', {}).get('batch_size', 64)
        self.num_workers      = self.data_config.get('num_workers', 0)
        self.pin_memory       = self.hardware_config.get('device', 'cpu') == 'cuda'
        self.plot_n_examples  = self.eval_config.get('plot_n_examples', 3)
        
        # Setup directories
        self.checkpoint_dir = checkpoint_dir
        if self.checkpoint_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = config.get('experiment_name', 'supermask_eval')
            run_name = f"{experiment_name}-{timestamp}"
            base_eval_dir = self.eval_config.get('save_results_dir', './results/evaluations')
            self.results_dir = os.path.join(base_eval_dir, run_name)
        else:
            self.results_dir = os.path.join(self.checkpoint_dir, 'evaluation')
        
        if not os.path.exists(self.results_dir) and not (wandb.run and wandb.run.sweep_id):
            os.makedirs(self.results_dir, exist_ok=True)
            
        # Set up AMP
        self.use_amp = self.hardware_config.get('use_amp', False) and self.device.type == 'cuda'
        
        # Initisalise model and criterion
        self.model = None
        self.criterion = get_criterion(config)
        
        # Load data
        self._load_data()
        
    def _load_data(self):
        """Load the supermask dataset."""
        from continual_learning.datasets.supermask.dataset_utils import load_data
        
        # Ensure config has required keys
        if 'supermask' not in self.config:
            self.config['supermask'] = {}
        if 'tasks' not in self.config['supermask']:
            self.config['supermask']['tasks'] = self.num_tasks
        
        # Ensure dataset_type is set to 'supermask'
        if 'data' not in self.config:
            self.config['data'] = {}
        self.config['data']['dataset_type'] = 'supermask'
        
        # Load data
        dataloaders, norm_info = load_data(self.config, mode='eval')
        
        # Verify supermask_instance is present in the returned dataloaders
        if 'supermask_instance' not in dataloaders:
            raise ValueError(
                "The load_data function did not return a 'supermask_instance' key in dataloaders."
            )
        
        self.supermask_instance = dataloaders['supermask_instance']
        self.norm_info = norm_info
        
    def load_model(self, model_path=None):
        """
        Load the model from a checkpoint.
        
        Args:
            model_path (str, optional): Path to the model checkpoint. 
                If None, will look for 'final_model_all_tasks.pth' or 'best_overall_model.pth'
                in the checkpoint directory.
        """
        from continual_learning.models.supermask.srcnn_supermask import SRCNN_Supermask
        from continual_learning.models.supermask.unet_srcnn_supermask import UNet_SRCNN_Supermask
        
        SUPPORTED_MODELS = {
            'srcnn_supermask': SRCNN_Supermask,
            'unet_srcnn_supermask': UNet_SRCNN_Supermask,
        }
        
        # Get model configuration
        model_config = self.config.get('model', {})
        model_name = model_config.get('name', '')
        model_name_lower = model_name.lower()
        
        if model_name_lower not in SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model: {model_name}. Supported models: {list(SUPPORTED_MODELS.keys())}")
        
        # Instantiate model
        print(f"Loading {model_name} model...")
        ModelClass = SUPPORTED_MODELS[model_name_lower]
        
        # Make sure model params include num_tasks, sparsity, and alpha
        model_params = model_config.get('params', {})
        model_params['num_tasks'] = self.num_tasks
        if 'sparsity' not in model_params:
            model_params['sparsity'] = model_config.get('sparsity', 0.95)
        if 'alpha' not in model_params:
            model_params['alpha'] = model_config.get('alpha', 0.1)
        
        # Create model instance
        self.model = ModelClass(**model_params)
        
        # Load weights
        if model_path is not None:
            checkpoint_path = model_path
        else:
            # Try to find model in checkpoint directory
            if self.checkpoint_dir is None:
                raise ValueError("No checkpoint directory or model path provided.")
            
            # Try final model first, then best overall model
            final_model_path = os.path.join(self.checkpoint_dir, 'final_model_all_tasks.pth')
            best_model_path = os.path.join(self.checkpoint_dir, 'best_overall_model.pth')
            
            if os.path.exists(final_model_path):
                checkpoint_path = final_model_path
                print(f"Using final model: {checkpoint_path}")
            elif os.path.exists(best_model_path):
                checkpoint_path = best_model_path
                print(f"Using best overall model: {checkpoint_path}")
            else:
                raise ValueError(f"No model checkpoint found in {self.checkpoint_dir}")
        
        # Load the state dict
        try:
            state_dict = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            print(f"Successfully loaded model weights from {checkpoint_path}")
        except Exception as e:
            print(f"Error loading model weights: {e}")
            raise
        
        # Move model to device
        self.model.to(self.device)
        self.model.eval()
        
    def _prepare_task_data(self, task_id):
        """
        Prepare data for a specific task.
        
        Args:
            task_id (int): Task ID to prepare data for
            
        Returns:
            tuple: (test_loader, task_norm_info)
        """
        print(f"\n--- Preparing data for Task {task_id} ---")
        
        # Set task in dataset
        self.supermask_instance.set_task(task_id)
        
        # Get total dataset size
        task_total_size = len(self.supermask_instance)
        if task_total_size == 0:
            print(f"  Warning: No data found for task {task_id}.")
            return None, None
        
        # Calculate split sizes
        val_size = int(self.val_split_ratio * task_total_size)
        test_size = int(self.test_split_ratio * task_total_size)
        train_size = task_total_size - val_size - test_size
        
        if test_size <= 0:
            print(f"  Warning: Test split for task {task_id} is empty.")
            return None, None
        
        print(f"  Splitting task {task_id} data: Train={train_size}, Val={val_size}, Test={test_size}")
        
        # Generate indices
        generator = torch.Generator().manual_seed(self.split_seed)
        indices = torch.randperm(task_total_size, generator=generator).tolist()
        
        # Split dataset
        train_indices = indices[:train_size]
        test_indices = indices[train_size + val_size:]
        
        # Create subsets
        train_subset_raw = Subset(self.supermask_instance, train_indices)
        test_subset_raw = Subset(self.supermask_instance, test_indices)
        
        # Calculate normalisation parameters from training subset
        norm_type = self.data_config.get('normalisation', 'none').lower()
        normalise_target = self.data_config.get('normalise_target', True)
        
        norm_params_input = None
        norm_params_target = None
        task_norm_info = {'type': norm_type, 'params_input': None, 'params_target': None}
        
        if norm_type != 'none' and len(train_subset_raw) > 0:
            print(f"  Calculating {norm_type} normalisation parameters for task {task_id} from training data...")
            if norm_type == "zscore":
                from standard_training.datasets.dataset_utils import calculate_zscore_params
                mean_in, std_in = calculate_zscore_params(train_subset_raw)
                norm_params_input = (mean_in, std_in)
                if normalise_target:
                    norm_params_target = (mean_in, std_in)
            elif norm_type == "minmax":
                from standard_training.datasets.dataset_utils import calculate_minmax_params
                min_in, max_in = calculate_minmax_params(train_subset_raw)
                norm_params_input = (min_in, max_in)
                if normalise_target:
                    norm_params_target = (min_in, max_in)
            
            task_norm_info = {
                'type': norm_type,
                'params_input': norm_params_input,
                'params_target': norm_params_target
            }
        
        # Create normalised dataset for test set
        test_dataset_norm = NormalisedDatasetWrapper(
            test_subset_raw, 
            task_norm_info['type'],
            task_norm_info['params_input'],
            task_norm_info['params_target']
        )
        
        # Create dataloader
        test_loader = DataLoader(
            test_dataset_norm,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
        
        return test_loader, task_norm_info
    
    def evaluate_task(self, task_id):
        """
        Evaluate a specific task.
        
        Args:
            task_id (int): Task ID to evaluate
            
        Returns:
            dict: Evaluation results
        """
        print(f"\n=== Evaluating Task {task_id} ===")
        
        # Set task in model
        set_model_task(self.model, task_id)
        
        # Prepare data
        test_loader, task_norm_info = self._prepare_task_data(task_id)
        if test_loader is None:
            print(f"Skipping evaluation for task {task_id}: No data available.")
            return None
        
        # Evaluation metrics
        eval_metrics = self.eval_config.get('metrics', ['nmse', 'psnr'])
        
        # Set up metrics
        metrics = {}
        for name in eval_metrics:
            try:
                if name.lower() == 'mse':
                    metrics[name.lower()] = lambda o, t: torch.nn.functional.mse_loss(o, t)
                else:
                    # Import get_metric_function from wherever it's defined in your project
                    from shared.utils.metrics import get_metric_function
                    metrics[name.lower()] = get_metric_function(name)
            except Exception as e:
                print(f"Warning: Could not set up metric '{name}': {e}")
        
        # Initisalise metric values
        results = {name: 0.0 for name in metrics.keys()}
        running_loss = 0.0
        num_samples = 0
        
        # Set up diagnostic directory
        diagnostic_dir = None
        if self.plot_n_examples > 0:
            diagnostic_dir = os.path.join(self.results_dir, f"diagnostics_task_{task_id}")
            if not os.path.exists(diagnostic_dir) and not (wandb.run and wandb.run.sweep_id):
                os.makedirs(diagnostic_dir, exist_ok=True)
        
        # Store examples for visualisation
        diagnostic_examples = []
        
        # Start evaluation
        eval_start_time = time.time()
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                # Move to device
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # Forward pass
                with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                    outputs = self.model(inputs)
                    # Handle tuple output from some models
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]
                    loss = self.criterion(outputs, targets)
                
                # Accumulate loss
                batch_size = inputs.size(0)
                running_loss += loss.item() * batch_size
                num_samples += batch_size
                
                # Collect samples for visualisation
                if len(diagnostic_examples) < self.plot_n_examples and diagnostic_dir:
                    for i in range(min(self.plot_n_examples - len(diagnostic_examples), batch_size)):
                        diagnostic_examples.append({
                            'input': inputs[i].cpu().numpy(),
                            'output': outputs[i].cpu().numpy(),
                            'target': targets[i].cpu().numpy()
                        })
                
                # Denormalize outputs and targets for metrics
                outputs_denorm = outputs.detach()
                targets_denorm = targets.detach()
                
                norm_type = task_norm_info.get('type', 'none')
                params_in = task_norm_info.get('params_input')
                params_tgt = task_norm_info.get('params_target')
                
                # Apply denormalization
                if norm_type == "zscore" and params_in is not None:
                    mean_in, std_in = params_in
                    if not isinstance(mean_in, torch.Tensor): mean_in = torch.tensor(mean_in)
                    if not isinstance(std_in, torch.Tensor): std_in = torch.tensor(std_in)
                    mean_in = mean_in.to(outputs_denorm.device)
                    std_in = std_in.to(outputs_denorm.device)
                    outputs_denorm = (outputs_denorm * std_in.view(1, -1, 1, 1)) + mean_in.view(1, -1, 1, 1)
                    
                    if params_tgt is not None:
                        mean_tgt, std_tgt = params_tgt
                        if not isinstance(mean_tgt, torch.Tensor): mean_tgt = torch.tensor(mean_tgt)
                        if not isinstance(std_tgt, torch.Tensor): std_tgt = torch.tensor(std_tgt)
                        mean_tgt = mean_tgt.to(targets_denorm.device)
                        std_tgt = std_tgt.to(targets_denorm.device)
                        targets_denorm = (targets_denorm * std_tgt.view(1, -1, 1, 1)) + mean_tgt.view(1, -1, 1, 1)
                
                elif norm_type == "minmax" and params_in is not None:
                    min_in, max_in = params_in
                    if not isinstance(min_in, torch.Tensor): min_in = torch.tensor(min_in)
                    if not isinstance(max_in, torch.Tensor): max_in = torch.tensor(max_in)
                    min_in = min_in.to(outputs_denorm.device)
                    max_in = max_in.to(outputs_denorm.device)
                    denominator = max_in - min_in
                    denominator[denominator == 0] = 1e-8
                    outputs_denorm = (outputs_denorm * denominator.view(1, -1, 1, 1)) + min_in.view(1, -1, 1, 1)
                    
                    if params_tgt is not None:
                        min_tgt, max_tgt = params_tgt
                        if not isinstance(min_tgt, torch.Tensor): min_tgt = torch.tensor(min_tgt)
                        if not isinstance(max_tgt, torch.Tensor): max_tgt = torch.tensor(max_tgt)
                        min_tgt = min_tgt.to(targets_denorm.device)
                        max_tgt = max_tgt.to(targets_denorm.device)
                        denominator = max_tgt - min_tgt
                        denominator[denominator == 0] = 1e-8
                        targets_denorm = (targets_denorm * denominator.view(1, -1, 1, 1)) + min_tgt.view(1, -1, 1, 1)
                
                # Calculate metrics on denormalized data
                for metric_name, metric_func in metrics.items():
                    try:
                        metric_val = metric_func(outputs_denorm, targets_denorm)
                        results[metric_name] += metric_val.item() * batch_size
                    except Exception as e:
                        print(f"Error calculating metric '{metric_name}': {e}")
        
        # Calculate average metrics
        eval_time = time.time() - eval_start_time
        avg_loss = running_loss / num_samples if num_samples > 0 else float('nan')
        
        # Format results
        task_results = {}
        task_results[f'task_{task_id}/loss'] = avg_loss
        
        for metric_name in results:
            avg_metric = results[metric_name] / num_samples if num_samples > 0 else float('nan')
            task_results[f'task_{task_id}/{metric_name}'] = avg_metric
        
        # Print results
        print(f"  Evaluation completed in {format_time(eval_time)}")
        print(f"  Loss: {avg_loss:.6f}")
        for metric_name in results:
            avg_metric = results[metric_name] / num_samples if num_samples > 0 else float('nan')
            print(f"  {metric_name.upper()}: {avg_metric:.6f}")
        
        # Save visualisation
        if diagnostic_examples and diagnostic_dir:
            self._save_visualisations(diagnostic_examples, diagnostic_dir, task_id)
        
        # Log to wandb
        if wandb.run:
            wandb.log(task_results)
            
            # Also log to summary for permanent record
            for key, value in task_results.items():
                wandb.summary[key] = value
        
        return task_results
    
    def _save_visualisations(self, examples, save_dir, task_id):
        """
        Save visualisations of the examples.
        
        Args:
            examples (list): List of example dictionaries
            save_dir (str): Directory to save visualisations
            task_id (int): Task ID
        """
        print(f"Saving {len(examples)} visualisations to {save_dir}...")
        
        try:
            for idx, example in enumerate(examples):
                input_data = example['input']
                output_data = example['output']
                target_data = example['target']
                
                # Create a figure
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                
                # Plot input
                if input_data.shape[0] > 2:  # If input has more than 2 channels
                    # If 3 channels, third is likely a mask. Plot real/imag only
                    axes[0].imshow(np.abs(input_data[0] + 1j * input_data[1]), cmap='viridis')
                else:
                    axes[0].imshow(np.abs(input_data[0] + 1j * input_data[1]), cmap='viridis')
                axes[0].set_title("Input")
                
                # Plot output
                axes[1].imshow(np.abs(output_data[0] + 1j * output_data[1]), cmap='viridis')
                axes[1].set_title("Output")
                
                # Plot target
                axes[2].imshow(np.abs(target_data[0] + 1j * target_data[1]), cmap='viridis')
                axes[2].set_title("Target")
                
                # Add task info
                plt.suptitle(f"Task {task_id} - Example {idx+1}")
                
                # Save figure
                plt.tight_layout()
                save_path = os.path.join(save_dir, f"example_{idx+1}.png")
                plt.savefig(save_path)
                plt.close()
                
                # Also log to wandb
                if wandb.run:
                    wandb.log({f"task_{task_id}/example_{idx+1}": wandb.Image(fig)})
        except Exception as e:
            print(f"Error saving visualisations: {e}")
    
    def evaluate_all_tasks(self):
        """
        Evaluate all tasks.
        
        Returns:
            dict: Aggregated evaluation results
        """
        if self.model is None:
            self.load_model()
        
        overall_results = {}
        task_results = []
        
        # Evaluate each task
        for task_id in range(self.num_tasks):
            result = self.evaluate_task(task_id)
            if result is not None:
                task_results.append(result)
        
        # Calculate aggregated metrics
        if task_results:
            # Initisalise metrics
            metrics_dict = {}
            
            # Collect all metric keys (they'll be like task_X/metric_name)
            all_keys = set()
            for result in task_results:
                all_keys.update(result.keys())
            
            # Extract base metrics (loss, nmse, psnr, etc.) from keys
            base_metrics = set()
            for key in all_keys:
                # Extract the metric name after the slash
                if '/' in key:
                    base_metrics.add(key.split('/')[1])
            
            # Initisalise the aggregated results
            for metric in base_metrics:
                metrics_dict[metric] = []
            
            # Collect metrics across tasks
            for result in task_results:
                for key, value in result.items():
                    if '/' in key:
                        metric = key.split('/')[1]
                        metrics_dict[metric].append(value)
            
            # Calculate average for each metric
            for metric, values in metrics_dict.items():
                if values:
                    avg_value = sum(values) / len(values)
                    overall_results[f'avg/{metric}'] = avg_value
                    print(f"Average {metric.upper()}: {avg_value:.6f}")
            
            # Log to wandb
            if wandb.run:
                wandb.log(overall_results)
                
                # Also log to summary
                for key, value in overall_results.items():
                    wandb.summary[key] = value
        
        return overall_results 