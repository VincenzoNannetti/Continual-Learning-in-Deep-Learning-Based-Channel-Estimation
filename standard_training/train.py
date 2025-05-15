"""
Filename: standard_training/train.py
Author: Vincenzo Nannetti
Date: 15/05/2025
Description: Standard Training Script

Usage: python train.py --config configs/my_config.yaml --experiment_suffix _dataset_a --dataset_to_use a

    - to not run evaluation after training: python train.py --config configs/my_config.yaml --experiment_suffix _dataset_a --dataset_to_use a --no_eval

Dependencies:
    - PyTorch
    - numpy
    - matplotlib
    - wandb
    - utils.metrics
"""

import torch
import torch.optim as optim
import time
import os
import yaml
from datetime import datetime
import matplotlib.pyplot as plt
import wandb
import argparse
import sys

from shared.utils.format_time import format_time
from shared.utils.get_device import get_device
from shared.utils.training_utils import get_criterion, get_optimiser, get_scheduler

def train_epoch(config, model, dataloader, criterion, optimiser, device, scaler=None):
    """Train the model for one epoch."""
    model.train()
    running_loss = 0.0
    num_samples  = 0
    use_amp      = scaler is not None

    for batch_idx, (inputs, targets) in enumerate(dataloader):
        # Move inputs and targets to device
        inputs, targets = inputs.to(device), targets.to(device)

        # Zero the parameter gradients
        optimiser.zero_grad()

        # Forward + backward + optimise
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            outputs = model(inputs)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            loss = criterion(outputs, targets)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimiser)
            scaler.update()
        else:
            loss.backward()
            optimiser.step()

        running_loss += loss.item() * inputs.size(0)
        num_samples  += inputs.size(0)

    epoch_loss = running_loss / num_samples if num_samples > 0 else float('nan')
    return epoch_loss


def validate_epoch(config, model, dataloader, criterion, device):
    """Validate the model for one epoch."""
    model.eval()
    running_loss = 0.0
    num_samples  = 0
    use_amp      = config.get('hardware', {}).get('use_amp', False) and device.type == 'cuda'

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(inputs)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)
            num_samples  += inputs.size(0)

    epoch_loss = running_loss / num_samples if num_samples > 0 else float('nan')
    return epoch_loss


def train_model(config, model, dataloaders, device, norm_info: dict, config_path: str | None = None):
    """Train the model."""
    print("\n" + "#"*70)
    print("TRAINING SETUP")
    print("#"*70)
    print("Setting up training components...")
    training_config = config.get('training', {})
    logging_config  = config.get('logging', {})
    model_cfg       = config.get('model', {})
    data_cfg        = config.get('data', {})
    model_name      = model_cfg.get('name', 'model')

    # Setup Criterion, Optimiser, Scheduler, AMP Scaler
    criterion = get_criterion(config)
    optimiser = get_optimiser(config, model)
    scheduler = get_scheduler(config, optimiser)
    use_amp   = config.get('hardware', {}).get('use_amp', False) and device.type == 'cuda'
    scaler    = torch.amp.GradScaler() if use_amp else None 
    print(f"Using Loss: {criterion.__class__.__name__}, Optimiser: {optimiser.__class__.__name__}, AMP: {use_amp}")

    # Create Unique Run Directory
    base_checkpoint_dir = logging_config.get('checkpoint_dir', './checkpoints')
    timestamp           = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name_short    = model_name.replace('_', '').replace('denoising','den').replace('res','r')[:15]
    experiment_name     = config.get('experiment_name', None)
    
    if experiment_name:
        run_name = f"{experiment_name}-{timestamp}"
    else:
        data_name_short = data_cfg.get('data_name', 'data')[:20]
        run_name = f"{model_name_short}-{data_name_short}-{timestamp}"

    run_checkpoint_dir = os.path.join(base_checkpoint_dir, run_name)
    
    # Create directory only if not a sweep run
    if not (wandb.run and wandb.run.sweep_id):
        os.makedirs(run_checkpoint_dir, exist_ok=True)
        print(f"Checkpoints and results for this run will be saved in: {run_checkpoint_dir}")
    else:
        print(f"Checkpoints directory ({run_checkpoint_dir}) will NOT be created locally (Sweep run).")

    # Save configuration locally if W&B is not active
    if not wandb.run:
        try:
            # Convert config to dict if it's a wandb object
            if isinstance(config, wandb.sdk.wandb_config.Config):
                config_to_save = dict(config)
            else:
                config_to_save = config
            
            if run_checkpoint_dir:
                os.makedirs(run_checkpoint_dir, exist_ok=True)
                config_save_path = os.path.join(run_checkpoint_dir, 'config.yaml')
                with open(config_save_path, 'w') as f:
                    yaml.dump(config_to_save, f, default_flow_style=False)
                print(f"Saved run configuration locally to {config_save_path}")
            else:
                print("Warning: run_checkpoint_dir is None, cannot save config.yaml locally.")
        except Exception as e:
            print(f"Warning: Could not save config.yaml locally: {e}")
    else:
        print("Skipping local save of run configuration (W&B is active).")

    # Log original base config path to W&B as artifact
    if wandb.run and config_path and os.path.exists(config_path):
        try:
            config_artifact = wandb.Artifact(f'config-{wandb.run.id}', type='config')
            config_artifact.add_file(config_path)
            wandb.log_artifact(config_artifact)
            print(f"Logged base config file '{os.path.basename(config_path)}' to W&B Artifacts")
        except Exception as e:
            print(f"Warning: Failed to log base config artifact to W&B: {e}")

    # W&B Model Watching
    if wandb.run:
        try:
            wandb.watch(model, criterion, log="gradients", log_freq=100)
            print("W&B watching model parameters and gradients.")
        except Exception as e:
            print(f"Warning: Failed to initiate wandb.watch: {e}")

    # Training Loop Setup
    epochs      = training_config.get('epochs', 100)
    patience    = training_config.get('early_stopping_patience', 10)
    metric_mode = 'min'  # Default mode
    early_stopping_metric = training_config.get('early_stopping_metric', 'val_loss').lower()
    print(f"Early stopping based on: {early_stopping_metric} (mode: {metric_mode})")

    # Baseline Training
    print("\n" + "#"*70)
    print("TRAINING EXECUTION")
    print("#"*70)
    print("Starting training...")
    if 'train' not in dataloaders or 'val' not in dataloaders:
        raise RuntimeError("Missing train or val dataloader for training.")
    
    train_loader = dataloaders['train']
    val_loader   = dataloaders['val']
    
    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'lr': []}
    overall_start_time = time.time()
    best_model_path = None

    for epoch in range(epochs):
        epoch_start_time = time.time()
        
        train_loss = train_epoch(config, model, train_loader, criterion, optimiser, device, scaler)
        val_loss   = validate_epoch(config, model, val_loader, criterion, device)
        
        epoch_duration = time.time() - epoch_start_time
        current_lr = optimiser.param_groups[0]['lr']
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['lr'].append(current_lr)

        # Log Metrics to W&B
        log_dict = {
            'train_loss': train_loss,
            'val_loss': val_loss,
            'epoch': epoch + 1,
            'lr': current_lr
        }
        if wandb.run:
            wandb.log(log_dict)

        # Learning Rate Scheduler Step
        lr_string = f"| LR: {current_lr:.2e}" if scheduler else ""
        print(f"Epoch {epoch + 1}/{epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Time: {format_time(epoch_duration)} {lr_string}")
        if scheduler:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)

        # Checkpointing & Early Stopping
        current_metric = val_loss
        is_better = current_metric < best_val_loss

        if is_better:
            best_val_loss = current_metric
            patience_counter = 0
            best_epoch = epoch + 1
            
            # Save Best Model Checkpoint
            checkpoint_name = 'best_model.pth'
            temp_save_path = None
            final_save_path = os.path.join(run_checkpoint_dir, checkpoint_name) if run_checkpoint_dir else None
            best_model_path = final_save_path  

            # Never save locally during sweep runs
            if not wandb.run or (wandb.run and not wandb.run.sweep_id):  # Save locally for non-sweep runs
                if final_save_path:
                    print(f"  Validation metric improved to {best_val_loss:.6f}. Saving model locally to: {final_save_path}")
                    try:
                        os.makedirs(run_checkpoint_dir, exist_ok=True)
                        torch.save({
                            'epoch': best_epoch,
                            'model_state_dict': model.state_dict(),
                            'optimiser_state_dict': optimiser.state_dict(),
                            'best_val_loss': best_val_loss,
                            'config': config,
                        }, final_save_path)
                    except Exception as e:
                        print(f"  Error saving checkpoint locally: {e}")

            # Log artifact to W&B
            if wandb.run:
                try:
                    # Determine path for artifact
                    if final_save_path and os.path.exists(final_save_path):
                        temp_save_path = final_save_path
                    else:
                        import tempfile
                        temp_dir = tempfile.mkdtemp()
                        temp_save_path = os.path.join(temp_dir, checkpoint_name)
                        print(f"  Saving temporary checkpoint for W&B artifact: {temp_save_path}")
                        if not os.path.exists(os.path.dirname(temp_save_path)):
                            os.makedirs(os.path.dirname(temp_save_path))
                        torch.save({
                            'epoch': best_epoch,
                            'model_state_dict': model.state_dict(),
                            'optimiser_state_dict': optimiser.state_dict(),
                            'best_val_loss': best_val_loss,
                            'config': config,
                        }, temp_save_path)

                    # Create and log artifact
                    model_artifact = wandb.Artifact(f'model-{wandb.run.id}', type='model',
                                                   metadata={'epoch': best_epoch, 
                                                            f'best_{early_stopping_metric}': best_val_loss})
                    model_artifact.add_file(temp_save_path, name=checkpoint_name)
                    wandb.log_artifact(model_artifact, aliases=['latest', 'best'])
                    print(f"  Logged best model checkpoint to W&B Artifacts.")

                    # Cleanup temp file if created
                    if not final_save_path or not os.path.exists(final_save_path):
                        if temp_save_path and os.path.exists(temp_save_path):
                            os.remove(temp_save_path)

                except Exception as e:
                    print(f"  Warning: Failed to log model artifact to W&B: {e}")
                    if (not final_save_path or not os.path.exists(final_save_path)) and temp_save_path and os.path.exists(temp_save_path):
                        os.remove(temp_save_path)

        else:  # Metric did not improve
            patience_counter += 1

        # Early Stopping Check
        if patience_counter >= patience:
            print(f"Early stopping triggered after {patience} epochs without improvement on {early_stopping_metric}.")
            break  # Stop training

    # Training complete
    print("\n" + "#"*70)
    print("TRAINING COMPLETION")
    print("#"*70)
    total_time = time.time() - overall_start_time
    print(f"Training completed in {format_time(total_time)}. Best validation loss: {best_val_loss:.6f} at epoch {best_epoch}.")
    
    # Plot and save learning curves if not using W&B and not in a sweep
    if (not wandb.run or (wandb.run and not wandb.run.sweep_id)) and run_checkpoint_dir:
        try:
            plt.figure(figsize=(10, 6))
            plt.plot(range(1, len(history['train_loss']) + 1), history['train_loss'], label='Train Loss')
            plt.plot(range(1, len(history['val_loss']) + 1), history['val_loss'], label='Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Learning Curve')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            
            curve_path = os.path.join(run_checkpoint_dir, 'learning_curve.png')
            plt.savefig(curve_path)
            plt.close()
            print(f"Learning curve saved to {curve_path}")
        except Exception as e:
            print(f"Error saving learning curve: {e}")

    return history, norm_info, run_checkpoint_dir, best_model_path


def main(config_path):
    """Main function to run training from a config file."""
    # Parse additional arguments
    parser = argparse.ArgumentParser(description='Train a model using a config file')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--experiment_suffix', type=str, default="",
                        help='Suffix to append to the experiment_name (e.g., _train_a, _finetune_b)')
    parser.add_argument('--no_wandb', action='store_true',
                        help='Disable Weights & Biases logging')
    parser.add_argument('--dataset_to_use', type=str, default=None, choices=['a', 'b'],
                        help='Specify which dataset to use based on dataset_a_name or dataset_b_name in config')
    parser.add_argument('--no_eval', action='store_true',
                        help='Do not run evaluation after training')
    args = parser.parse_args()
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Apply command-line overrides
    if args.experiment_suffix:
        original_name = config.get('experiment_name', 'experiment')
        new_name = original_name + args.experiment_suffix
        print(f"Appending suffix '{args.experiment_suffix}' to experiment name. New name: '{new_name}'")
        config['experiment_name'] = new_name
        if 'logging' in config and 'checkpoint_dir' in config['logging']:
            base_checkpoint_dir = os.path.dirname(config['logging']['checkpoint_dir'].rstrip('/\\'))
            config['logging']['checkpoint_dir'] = os.path.join(base_checkpoint_dir, new_name)
    
    if args.dataset_to_use:
        target_data_key = f"dataset_{args.dataset_to_use}_name"
        if target_data_key not in config.get('data', {}):
            print(f"Error: Config file does not contain the key 'data.{target_data_key}'.")
            sys.exit(1)
        effective_data_name = config['data'][target_data_key]
        print(f"Overriding 'data.data_name' to use '{effective_data_name}' (from '{target_data_key}')")
        config['data']['data_name'] = effective_data_name
    
    from standard_training.utils.utils import load_model
    from standard_training.datasets.dataset_utils import load_data
    
    # Setup device
    print("\n" + "="*80)
    print("HARDWARE SETUP")
    print("="*80)
    device = get_device(config)
    print(f"Using device: {device}")
    
    # Load data
    print("\n" + "="*80)
    print("DATA LOADING")
    print("="*80)
    dataloaders, norm_info = load_data(config)
    
    # Load model
    print("\n" + "="*80)
    print("MODEL SETUP")
    print("="*80)
    model = load_model(config)
    model = model.to(device)
    
    # Initialise W&B if configured and not disabled via command line
    use_wandb = config.get('logging', {}).get('use_wandb', False) and not args.no_wandb
    if use_wandb:
        print("\n" + "="*80)
        print("WANDB INITIALIsATION")
        print("="*80)
        wandb_config = config.get('wandb', {})
        wandb_project = wandb_config.get('project', 'standard_training')
        print(f"Saving to W&B project: {wandb_project}")
        wandb_entity = wandb_config.get('entity', None)
        print(f"Saving to W&B entity: {wandb_entity}")
        wandb_name = wandb_config.get('name', None)
        print(f"Saving to W&B name: {wandb_name}")
        wandb.init(project=wandb_project, entity=wandb_entity, name=wandb_name, config=config)
    
    # Train model
    print("\n" + "="*80)
    print("TRAINING")
    print("="*80)
    history, norm_info, checkpoint_dir, best_model_path = train_model(config, model, dataloaders, device, norm_info, config_path)
    
    # Automatically run evaluation if not disabled
    if not args.no_eval:
        print("\n" + "="*80)
        print("EVALUATION")
        print("="*80)
        try:
            # Import the evaluate module
            from standard_training.evaluate import evaluate_model
            
            # Check if we have a test dataloader
            if 'test' in dataloaders:
                test_loader = dataloaders['test']
                
                # Load the best model for evaluation
                if best_model_path and os.path.exists(best_model_path):
                    print(f"Loading best model from {best_model_path} for evaluation")
                    # Load the model weights
                    checkpoint = torch.load(best_model_path, map_location=device)
                    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
                else:
                    print("Using current model state for evaluation (best model checkpoint not found)")
                
                # Run evaluation
                eval_results = evaluate_model(
                    config=config,
                    model=model,
                    dataloader=test_loader,
                    device=device,
                    norm_info=norm_info,
                    plot_n_examples=config.get('evaluation', {}).get('plot_examples', 3),
                    training_history=history,
                    results_dir=checkpoint_dir if not (wandb.run and wandb.run.sweep_id) else None,
                    config_path=config_path
                )
                
                print("Evaluation complete.")
            else:
                print("No test dataloader found. Skipping evaluation.")
                
        except Exception as e:
            print(f"Error during automatic evaluation: {e}")
    
    # Finish W&B run
    if wandb.run:
        wandb.finish()
    
    return history, norm_info, checkpoint_dir


if __name__ == "__main__":
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description='Train a model using a config file')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--experiment_suffix', type=str, default="",
                        help='Suffix to append to the experiment_name (e.g., _train_a, _finetune_b)')
    parser.add_argument('--no_wandb', action='store_true',
                        help='Disable Weights & Biases logging')
    parser.add_argument('--dataset_to_use', type=str, default=None, choices=['a', 'b'],
                        help='Specify which dataset to use based on dataset_a_name or dataset_b_name in config')
    parser.add_argument('--no_eval', action='store_true',
                        help='Do not run evaluation after training')
    args = parser.parse_args()
    
    main(args.config)
