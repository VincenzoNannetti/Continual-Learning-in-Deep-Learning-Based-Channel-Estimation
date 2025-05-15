"""
Filename: continual_learning/trainers/supermask_trainer.py
Author: Vincenzo Nannetti
Date: 15/05/2025
Description: Supermask Training Script - since not part of the standard training pipeline, it is not included in the standard training script

Usage: 

Dependencies:
    
"""

import os
import yaml
import time
import torch
import wandb
from datetime import datetime
from torch.utils.data import Subset, DataLoader
from shared.utils.get_device import get_device
from shared.utils.format_time import format_time
from shared.utils.training_utils import get_criterion, get_optimiser, get_scheduler
from standard_training.datasets.dataset_utils import calculate_zscore_params, calculate_minmax_params, NormalisedDatasetWrapper
from standard_training.train import train_epoch, validate_epoch
from continual_learning.utils.supermask_utils import cache_masks, set_num_tasks_learned

from continual_learning.models.supermask.srcnn_supermask import SRCNN_Supermask
from continual_learning.models.supermask.unet_srcnn_supermask import UNet_SRCNN_Supermask

SUPPORTED_SUPERMASK_MODELS = {
    'srcnn_supermask': SRCNN_Supermask,
    'unet_srcnn_supermask': UNet_SRCNN_Supermask,
}

class SupermaskTrainer:
    def __init__(self, config):
        self.device = get_device(config)

        # extract the configs
        self.config           = config
        self.supermask_config = config.get('supermask', {})
        self.data_config      = config.get('data', {})
        self.training_config  = config.get('training', {})
        self.hardware_config  = config.get('hardware', {})
        self.framework_config = config.get('framework', {})
        self.logging_config   = config.get('logging', {})

        self.num_tasks        = self.supermask_config.get('tasks', None)
        self.sequence         = self.supermask_config.get('sequence', list(range(self.num_tasks)))
        
        # Initisalise counters and parameters
        self.tasks_trained    = 0
        self.epochs           = self.training_config.get('epochs', 100)
        self.patience         = self.training_config.get('early_stopping_patience', 10)
        self.batch_size       = self.training_config.get('batch_size', 64)
        self.num_workers      = self.training_config.get('num_workers', 0)
        self.pin_memory       = self.hardware_config.get('device', 'cpu') == 'cuda'
        self.val_split_ratio  = self.data_config.get('validation_split', 0.15)
        self.split_seed       = self.framework_config.get('seed', 42)
        self.norm_type        = self.data_config.get('normalisation', 'none').lower()
        self.normalise_target = self.data_config.get('normalise_target', True)
        
        # Setup for saving models
        self.save_task_specific_models = self.supermask_config.get('save_task_specific_models', False)
        self.early_stopping_metric = self.training_config.get('early_stopping_metric', 'val_loss').lower()
        
        # Setup directories
        self.base_checkpoint_dir = self.logging_config.get('checkpoint_dir', './checkpoints')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = config.get('experiment_name', None)
        if experiment_name:
            run_name = f"{experiment_name}-{timestamp}"
        else:
            model_name_short = self.model_name.replace('_', '').replace('denoising','den').replace('res','r')[:15] if hasattr(self, 'model_name') else 'supermask'
            data_name_short = self.data_config.get('data_name', 'data')[:20]
            run_name = f"{model_name_short}-{data_name_short}-{timestamp}"
        
        self.run_checkpoint_dir = os.path.join(self.base_checkpoint_dir, run_name)
        
        # AMP setup
        self.use_amp = self.hardware_config.get('use_amp', False) and self.device.type == 'cuda'
        self.scaler = torch.amp.GradScaler() if self.use_amp else None

        # load the model 
        print("Loading Model...")
        self.model_name            = None
        self.model_params          = {}
        self.mode                  = None
        self.pretrained_path       = None
        self.evaluation_model_path = None
        self.model                 = self.load_model(config).to(self.device)

        # load the data
        print("Loading Data...")
        dataloaders, self.norm_info = self._load_data(config)
        self.supermask_instance = dataloaders.get('supermask_instance')
        
        if self.supermask_instance is None:
            raise ValueError("Supermask instance not found in dataloaders. Make sure your data loading function returns a 'supermask_instance' key.")
        
        # Get training components
        self.criterion = get_criterion(config)
        self.optimiser = get_optimiser(config, self.model)
        self.scheduler = get_scheduler(config, self.optimiser)
        
        print(f"Using Loss: {self.criterion.__class__.__name__}, Optimiser: {self.optimiser.__class__.__name__}, AMP: {self.use_amp}")


    def _load_data(self, config):
        """
        Load data for supermask training using the continual learning data loading function.
        This function returns a raw supermask dataset instance that can be used with task-specific operations.
        """
        # Import the data loading function from our continual learning module
        from continual_learning.datasets.supermask.dataset_utils import load_data
        
        # Make sure supermask settings are in the config
        if 'supermask' not in config:
            config['supermask'] = {}
        if 'tasks' not in config['supermask']:
            config['supermask']['tasks'] = self.num_tasks
        
        # Ensure dataset_type is set to 'supermask'
        if 'data' not in config:
            config['data'] = {}
        config['data']['dataset_type'] = 'supermask'
        
        # Call the load_data function which returns a supermask dataset instance
        dataloaders, norm_info = load_data(config, self.mode)
        
        # Verify supermask_instance is present in the returned dataloaders
        if 'supermask_instance' not in dataloaders:
            raise ValueError(
                "The load_data function did not return a 'supermask_instance' key in dataloaders. "
                "Make sure your load_data function properly handles dataset_type='supermask'."
            )
        
        return dataloaders, norm_info

    def load_model(self, config):
        model_config = config.get('model', {})
        self.model_name   = model_config.get('name', None)
        self.model_params = model_config.get('params', {})
        self.mode         = config.get('mode', 'train')
        
        if self.mode == "train":
            self.pretrained_path = model_config.get('pretrained_path', None)
            print(f"Using Pretrained Model: {self.pretrained_path}") if self.pretrained_path else print("No pretrained model provided")
        else:
            self.evaluation_model_path = model_config.get('evaluation_path', None)
            print(f"Using Evaluation Model: {self.evaluation_model_path}") if self.evaluation_model_path else print("No evaluation model provided")

        if self.model_name is None:
            raise ValueError("Model name is required")
        
        model_name_lower = self.model_name.lower()
        if model_name_lower not in SUPPORTED_SUPERMASK_MODELS:
            raise ValueError(f"Unsupported model: {self.model_name}. Supported models are: {list(SUPPORTED_SUPERMASK_MODELS.keys())}")
        
        print(f"Instantiating model: {self.model_name}")
        ModelClass = SUPPORTED_SUPERMASK_MODELS[model_name_lower]

        model = None

        try:
            sm_config = config.get('supermask', {})
            if 'tasks' not in sm_config:
                raise ValueError(f"Supermask model {self.model_name} requires a 'tasks' key in the config")
            self.model_params['num_tasks'] = sm_config['tasks']
            
            if 'sparsity' not in model_config:
                raise ValueError(f"Supermask model {self.model_name} requires a 'sparsity' key in the config")
            self.model_params['sparsity'] = model_config['sparsity']

            if 'alpha' not in model_config:
                raise ValueError(f"Supermask model {self.model_name} requires an 'alpha' key in the config")
            self.model_params['alpha'] = model_config['alpha']

            if self.pretrained_path:
                self.model_params['pretrained_path'] = self.pretrained_path

            model = ModelClass(**self.model_params)
            print(f"Model {self.model_name} instantiated with params: {self.model_params}")

        except Exception as e:
            raise ValueError(f"Error loading model: {e}")
        
        return model
        
    def train(self):
        print("Training all tasks sequentially...")

        # Create checkpoint directory if not a sweep run
        if not (wandb.run and wandb.run.sweep_id):
            os.makedirs(self.run_checkpoint_dir, exist_ok=True)
            print(f"Checkpoints and results for this run will be saved in: {self.run_checkpoint_dir}")
        else:
            print(f"Checkpoints directory ({self.run_checkpoint_dir}) will NOT be created locally (Sweep run).")

        # Save configuration locally if W&B is not active
        if not wandb.run:
            try:
                # Convert config to dict if it's a wandb object
                if isinstance(self.config, wandb.sdk.wandb_config.Config):
                    config_to_save = dict(self.config)
                else:
                    config_to_save = self.config
                
                if self.run_checkpoint_dir:
                    os.makedirs(self.run_checkpoint_dir, exist_ok=True)
                    config_save_path = os.path.join(self.run_checkpoint_dir, 'config.yaml')
                    with open(config_save_path, 'w') as f:
                        yaml.dump(config_to_save, f, default_flow_style=False)
                    print(f"Saved run configuration locally to {config_save_path}")
                else:
                    print("Warning: run_checkpoint_dir is None, cannot save config.yaml locally.")
            except Exception as e:
                print(f"Warning: Could not save config.yaml locally: {e}")
        else:
            print("Skipping local save of run configuration (W&B is active).")

        # W&B model watching
        if wandb.run:
            try:
                wandb.watch(self.model, self.criterion, log="gradients", log_freq=100)
                print("W&B watching model parameters and gradients.")
            except Exception as e:
                print(f"Warning: Failed to initiate wandb.watch: {e}")

        all_task_results = [] 
        task_norm_params = {}
        task_sequence = []
        if all(isinstance(item, str) for item in self.sequence):
            # Map alphabetical sequence (a,b,c) to task IDs (0,1,2)
            # Assuming a=0, b=1, c=2, etc.
            letter_to_id = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5}
            task_sequence = [letter_to_id.get(s.lower(), idx) for idx, s in enumerate(self.sequence)]
            print(f"Using task sequence from config: {self.sequence} → Task IDs: {task_sequence}")
        else:
            # Use numerical sequence directly
            task_sequence = self.sequence
            print(f"Using task sequence from config: {task_sequence}")

        best_overall_model_state = None
        best_overall_val_loss    = float('inf')
        best_overall_task_id     = None

        for task_index, task_id in enumerate(task_sequence):
            task_start_time = time.time()
            print(f"\n==== Starting Supermask Task {self.tasks_trained + 1}/{len(task_sequence)} (Task ID: {task_id}) ====")

            # 1. Set Task in Dataset
            print(f"  Setting dataset to Task {task_id}...")
            self.supermask_instance.set_task(task_id)

            # 2. Per-Task Split
            task_total_size = len(self.supermask_instance)
            if task_total_size == 0:
                print(f"  Warning: No data found for task {task_id}. Skipping...")
                continue
            task_val_size   = int(self.val_split_ratio * task_total_size)
            task_train_size = task_total_size - task_val_size
            print(f"  Splitting Task {task_id} data: Train={task_train_size}, Val={task_val_size}, Seed={self.split_seed}")

            generator = torch.Generator().manual_seed(self.split_seed)
            indices   = torch.randperm(task_total_size, generator=generator).tolist()

            train_indices = indices[:task_train_size]
            val_indices   = indices[task_train_size:]

            train_subset_raw = Subset(self.supermask_instance, train_indices)
            val_subset_raw   = Subset(self.supermask_instance, val_indices)

            # 3. Per-Task Normalisation
            norm_params_input_task  = None
            norm_params_target_task = None
            current_task_norm_info = {'input': None, 'target': None} # Use temp dict
            if self.norm_type != 'none' and len(train_subset_raw) > 0:
                print(f"  Calculating {self.norm_type} normalisation parameters for task {task_id} from its training data...")
                if self.norm_type == "zscore":
                    mean_in, std_in = calculate_zscore_params(train_subset_raw)
                    norm_params_input_task = (mean_in, std_in)
                    if self.normalise_target:
                        print("    Normalising target using INPUT Z-score statistics for this task.")
                        norm_params_target_task = (mean_in, std_in)
                elif self.norm_type == "minmax":
                    min_in, max_in = calculate_minmax_params(train_subset_raw)
                    norm_params_input_task = (min_in, max_in)
                    if self.normalise_target:
                        print("    Normalising target using INPUT MinMax statistics for this task.")
                        norm_params_target_task = (min_in, max_in)
                current_task_norm_info = {'input': norm_params_input_task, 'target': norm_params_target_task}
            elif len(train_subset_raw) == 0:
                print(f"  Warning: No training data for task {task_id} to calculate normalisation parameters. Skipping normalisation for this task.")
            else: # norm_type is 'none'
                print(f"  Skipping normalisation calculation for task {task_id} (type: {self.norm_type}).")
            task_norm_params[task_id] = current_task_norm_info # Store in main dict

            task_norm_params[task_id]['type'] = self.norm_type

            # 4. Wrap Task Subsets with NormalisationWrapper (using task-specific params)
            print(f"  Wrapping task {task_id} datasets with normalisation ({self.norm_type})...")
            train_dataset_norm = NormalisedDatasetWrapper(train_subset_raw, self.norm_type,
                                                        task_norm_params[task_id]['input'], task_norm_params[task_id]['target'])
            val_dataset_norm   = NormalisedDatasetWrapper(val_subset_raw, self.norm_type,
                                                        task_norm_params[task_id]['input'], task_norm_params[task_id]['target'])

            # 5. Create Task-Specific DataLoaders
            print(f"  Creating DataLoaders for task {task_id}...")
            train_loader_task = DataLoader(train_dataset_norm, batch_size=self.batch_size, shuffle=True,
                                            num_workers=self.num_workers, pin_memory=self.pin_memory, drop_last=True)
            val_loader_task   = DataLoader(val_dataset_norm, batch_size=self.batch_size, shuffle=False,
                                            num_workers=self.num_workers, pin_memory=self.pin_memory)

            # 6. Epoch Loop for Current Task
            best_val_loss_task = float('inf')
            patience_counter   = 0
            task_train_losses  = []
            task_val_losses    = []
            best_task_model_state = None

            for epoch in range(self.epochs):
                epoch_start_time = time.time()

                # Pass the task-specific train loader
                train_loss = train_epoch(self.config, self.model, train_loader_task, self.criterion, self.optimiser, self.device, self.scaler)
                task_train_losses.append(train_loss)

                # Pass the task-specific val loader
                val_loss = validate_epoch(self.config, self.model, val_loader_task, self.criterion, self.device)
                task_val_losses.append(val_loss)

                epoch_duration = time.time() - epoch_start_time

                # --- Log Metrics to W&B (with task prefix) ---
                log_dict = {
                    f'task_{task_id}/train_loss': train_loss,
                    f'task_{task_id}/val_loss': val_loss,
                    f'task_{task_id}/epoch': epoch + 1, # Log epoch within task
                    'global_epoch': (task_index * self.epochs) + epoch + 1, # Accurate global epoch count
                    f'task_{task_id}/lr': self.optimiser.param_groups[0]['lr']
                }
                
                # Add global metrics tracking
                log_dict['global/train_loss'] = train_loss
                log_dict['global/val_loss'] = val_loss
                log_dict['global/active_task'] = task_id
                
                if wandb.run:
                    wandb.log(log_dict)

                # --- Learning Rate Scheduler Step (based on task's val_loss) ---
                lr_string = f"| LR: {self.optimiser.param_groups[0]['lr']:.2e}" if self.scheduler else ""
                print(f"  Task {task_id} Epoch {epoch + 1}/{self.epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Time: {format_time(epoch_duration)} {lr_string}")
                if self.scheduler:
                    if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(val_loss)
                    # Add elif for other schedulers if needed

                # --- Checkpointing & Early Stopping (per task) ---
                current_metric = val_loss # Using task's val_loss
                is_better      = current_metric < best_val_loss_task

                if is_better:
                    best_val_loss_task = current_metric
                    patience_counter = 0
                    best_epoch_for_task = epoch + 1
                    # Save the current model state in memory
                    best_task_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

                    # Save best model checkpoint for this task if configured to save task-specific models
                    if self.save_task_specific_models:
                        checkpoint_name = f'best_model_task_{task_id}.pth'
                        final_save_path = os.path.join(self.run_checkpoint_dir, checkpoint_name)

                        if not wandb.run: # Save locally
                            print(f"    Task {task_id} validation metric improved to {best_val_loss_task:.6f}. Saving task-specific model to: {final_save_path}")
                            try:
                                os.makedirs(self.run_checkpoint_dir, exist_ok=True)
                                torch.save(self.model.state_dict(), final_save_path)
                            except Exception as e:
                                print(f"    Error saving task-specific checkpoint: {e}")

                        # Optional: Log task-specific model as W&B artifact
                        if wandb.run:
                            try:
                                if self.save_task_specific_models:
                                    model_artifact = wandb.Artifact(f'model-task{task_id}-{wandb.run.id}', type='model',
                                                                  metadata={'task_id': task_id, 'epoch': best_epoch_for_task, 
                                                                           f'task_{task_id}_best_{self.early_stopping_metric}': best_val_loss_task})
                                    # Create temp path for W&B artifact if needed
                                    temp_save_path = None
                                    if final_save_path and os.path.exists(final_save_path):
                                        temp_save_path = final_save_path
                                    else:
                                        import tempfile
                                        temp_dir = tempfile.mkdtemp()
                                        temp_save_path = os.path.join(temp_dir, checkpoint_name)
                                        os.makedirs(os.path.dirname(temp_save_path), exist_ok=True)
                                        torch.save(self.model.state_dict(), temp_save_path)
                                    
                                    model_artifact.add_file(temp_save_path)
                                    wandb.log_artifact(model_artifact, aliases=[f'task_{task_id}_best'])
                                    print(f"    Logged task-specific model to W&B as artifact.")
                                    
                                    # Clean up temp file if created
                                    if temp_save_path != final_save_path and os.path.exists(temp_save_path):
                                        os.remove(temp_save_path)
                            except Exception as e:
                                print(f"    Warning: Failed to log task-specific model to W&B: {e}")

                else: # Metric did not improve
                    patience_counter += 1

                # --- Early Stopping Check ---
                if patience_counter >= self.patience:
                    print(f"  Early stopping triggered for Task {task_id} after {self.patience} epochs without improvement on {self.early_stopping_metric}.")
                    break # Stop training this task
            # --- End of Epoch Loop for Task ---

            # 7. Cache Masks after Task Training
            print(f"  Caching masks for task {task_id}...")
            cache_masks(self.model) # Use the utility function

            # 8. Update number of tasks learned
            self.tasks_trained += 1
            set_num_tasks_learned(self.model, self.tasks_trained)
            
            # 9. Update best overall model if this task's model is better than previous best
            if best_val_loss_task < best_overall_val_loss:
                best_overall_val_loss = best_val_loss_task
                best_overall_task_id = task_id
                best_overall_model_state = best_task_model_state # Use the saved state dict
                print(f"  New best overall model from task {task_id} with val_loss: {best_overall_val_loss:.6f}")

            task_duration = time.time() - task_start_time
            print(f"==== Supermask Task {self.tasks_trained} Finished | Best Val Loss: {best_val_loss_task:.6f} | Time: {format_time(task_duration)} ====")
            
            # Sequential learning rate reduction for next task
            if task_index < len(task_sequence) - 1:  # If not the last task
                reduce_lr_sequential = self.training_config.get('reduce_lr_for_sequential', False)
                sequential_lr_factor = self.training_config.get('sequential_lr_factor', 1.0)
                
                if reduce_lr_sequential:
                    current_lr = self.optimiser.param_groups[0]['lr']
                    new_lr = current_lr * sequential_lr_factor
                    print(f"  Reducing learning rate for next task: {current_lr:.2e} → {new_lr:.2e}")
                    
                    for param_group in self.optimiser.param_groups:
                        param_group['lr'] = new_lr
            
            # Store task results
            all_task_results.append({
                "task_id": task_id,
                "best_val_loss": best_val_loss_task,
                "train_losses": task_train_losses,
                "val_losses": task_val_losses
            })
        # --- End of Task Loop (Supermask) ---
        
        # --- Save the final model with ALL cached masks ---
        final_model_path = os.path.join(self.run_checkpoint_dir, 'final_model_all_tasks.pth')
        print(f"\nSaving final model with all cached masks to: {final_model_path}")
        
        try:
            os.makedirs(os.path.dirname(final_model_path), exist_ok=True)
            torch.save(self.model.state_dict(), final_model_path)
            
            # Log final model to W&B
            if wandb.run:
                try:
                    model_artifact = wandb.Artifact(f'model-final-{wandb.run.id}', type='model',
                                               metadata={'tasks_trained': self.tasks_trained})
                    model_artifact.add_file(final_model_path)
                    wandb.log_artifact(model_artifact, aliases=['final_model'])
                    print(f"Logged final model with all cached masks to W&B.")
                except Exception as e:
                    print(f"Warning: Failed to log final model to W&B: {e}")
        except Exception as e:
            print(f"Error saving final model: {e}")
        
        # --- Also save the best overall model ---
        if best_overall_model_state is not None:
            best_model_path = os.path.join(self.run_checkpoint_dir, 'best_overall_model.pth')
            print(f"Saving best overall model (task {best_overall_task_id}, val_loss: {best_overall_val_loss:.6f}) to: {best_model_path}")
            
            try:
                torch.save(best_overall_model_state, best_model_path)
                
                # Log best overall model to W&B
                if wandb.run:
                    try:
                        model_artifact = wandb.Artifact(f'model-best-{wandb.run.id}', type='model',
                                                   metadata={'best_task_id': best_overall_task_id, 
                                                             'best_val_loss': best_overall_val_loss})
                        model_artifact.add_file(best_model_path)
                        wandb.log_artifact(model_artifact, aliases=['best_overall'])
                        print(f"Logged best overall model to W&B.")
                    except Exception as e:
                        print(f"Warning: Failed to log best overall model to W&B: {e}")
            except Exception as e:
                print(f"Error saving best overall model: {e}")

        return all_task_results, task_norm_params, self.run_checkpoint_dir

                
