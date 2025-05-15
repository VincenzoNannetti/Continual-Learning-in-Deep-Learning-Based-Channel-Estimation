"""
Filename: standard_training/evaluate.py
Author: Vincenzo Nannetti
Date: 15/05/2025
Description: Standard Training Evaluation Script

Usage: python evaluate.py --config configs/my_config.yaml --checkpoint checkpoints/best_model.pth --experiment_suffix _eval_test --no_wandb

Dependencies:
    - PyTorch
    - numpy
    - matplotlib
    - wandb
    - utils.metrics
    - utils.plot_heatmap
"""

import torch
import torch.nn as nn
import time
import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
import wandb
from shared.utils.metrics import get_metric_function
from shared.utils.plot_heatmap import plot_heatmap


def format_time(seconds):
    """Format time in hours, minutes, seconds."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours}h {minutes}m {seconds}s"


def get_criterion(config):
    """Get the loss function based on the config."""
    loss_fn_name = config.get('training', {}).get('loss_function', 'mse').lower()
    if loss_fn_name == 'mse':
        return nn.MSELoss()
    elif loss_fn_name == 'mae' or loss_fn_name == 'l1':
        return nn.L1Loss()
    elif loss_fn_name == 'huber':
        delta = float(config.get('training', {}).get('huber_delta', 1.0))
        return nn.HuberLoss(delta=delta)
    else:
        raise ValueError(f"Unsupported loss function: {loss_fn_name}. Supported: 'mse', 'mae', 'l1', 'huber'")


def evaluate_model(
    config, 
    model, 
    dataloader, 
    device, 
    norm_info: dict, 
    plot_n_examples: int = 0, 
    training_history: list | dict | None = None,
    results_dir: str | None = None, 
    config_path: str | None = None 
):
    """Evaluate the model and return evaluation metrics."""
    print("\n" + "="*80)
    print("EVALUATION SETUP")
    print("="*80)
    print("Starting evaluation process...")
    
    # Ensure model is in evaluation mode
    model.eval()
    print("Model set to evaluation mode")
    
    # Setup evaluation configs
    eval_config = config.get('evaluation', {})
    model_cfg = config.get('model', {})
    data_cfg = config.get('data', {})
    criterion = get_criterion(config)
    metric_names = eval_config.get('metrics', ['nmse', 'psnr'])
    use_amp = config.get('hardware', {}).get('use_amp', False) and device.type == 'cuda'
    
    print("\n" + "="*80)
    print("RESULTS DIRECTORY")
    print("="*80)
    
    # Determine Evaluation Results Directory
    save_directory = None
    
    # If explicit results_dir is provided, use it
    if results_dir:
        if os.path.isdir(results_dir):
            save_directory = results_dir
            print(f"Using existing results directory: {save_directory}")
        else:
            try:
                os.makedirs(results_dir, exist_ok=True)
                save_directory = results_dir
                print(f"Created specified results directory: {save_directory}")
            except Exception as e:
                print(f"Warning: Could not create specified results directory: {e}")
    
    # If we still don't have a save directory, use the one from config
    if save_directory is None:
        config_results_dir = eval_config.get('save_results_dir')
        if config_results_dir and not (wandb.run and wandb.run.sweep_id):
            try:
                os.makedirs(config_results_dir, exist_ok=True)
                save_directory = config_results_dir
                print(f"Using results directory from config: {save_directory}")
            except Exception as e:
                print(f"Warning: Could not create config results directory: {e}")
    
    # If still no directory, create a timestamped one as fallback
    if save_directory is None and not (wandb.run and wandb.run.sweep_id):
        base_eval_dir = './evaluations'
        model_name = model_cfg.get('name', 'model').replace('/', '_')
        data_name = data_cfg.get('data_name', 'data').replace('/', '_')
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        run_name = f"{model_name}-{data_name}-eval_{timestamp}"
        save_directory = os.path.join(base_eval_dir, run_name)
        
        try:
            os.makedirs(save_directory, exist_ok=True)
            print(f"Created fallback evaluation directory: {save_directory}")
        except Exception as e:
            print(f"Warning: Could not create fallback directory: {e}")
            save_directory = None
    
    # Skip local saving for sweep runs
    if wandb.run and wandb.run.sweep_id:
        print("W&B sweep run detected - skipping local result storage")
        save_directory = None
    
    # Create diagnostic directory
    diagnostic_dir = None
    if save_directory:
        diagnostic_dir = os.path.join(save_directory, "diagnostics")
        os.makedirs(diagnostic_dir, exist_ok=True)
        print(f"Created diagnostic directory: {diagnostic_dir}")
    else:
        print("Diagnostic visualisations will not be saved (no save directory available)")
    
    print("\n" + "="*80)
    print("NORMALISATION INFO")
    print("="*80)
    print(f"Type: {norm_info.get('type', 'none')}")
    if norm_info.get('type') == 'zscore' and norm_info.get('input') is not None:
        mean_in, std_in = norm_info.get('input')
        print(f"Input mean: {mean_in.min().item():.6f} to {mean_in.max().item():.6f}")
        print(f"Input std: {std_in.min().item():.6f} to {std_in.max().item():.6f}")
    elif norm_info.get('type') == 'minmax' and norm_info.get('input') is not None:
        min_in, max_in = norm_info.get('input')
        print(f"Input min: {min_in.min().item():.6f} to {min_in.max().item():.6f}")
        print(f"Input max: {max_in.min().item():.6f} to {max_in.max().item():.6f}")

    print("\n" + "="*80)
    print("EVALUATION METRICS")
    print("="*80)
    
    # Print normalisation info for debugging
    print(f"Normalisation info:")
    print(f"  Type: {norm_info.get('type', 'none')}")
    if norm_info.get('type') == 'zscore' and norm_info.get('input') is not None:
        mean_in, std_in = norm_info.get('input')
        print(f"  Input mean: {mean_in.min().item():.6f} to {mean_in.max().item():.6f}")
        print(f"  Input std: {std_in.min().item():.6f} to {std_in.max().item():.6f}")
    elif norm_info.get('type') == 'minmax' and norm_info.get('input') is not None:
        min_in, max_in = norm_info.get('input')
        print(f"  Input min: {min_in.min().item():.6f} to {min_in.max().item():.6f}")
        print(f"  Input max: {max_in.min().item():.6f} to {max_in.max().item():.6f}")
    
    # Setup Metrics
    results = {name: 0.0 for name in metric_names}
    running_loss = 0.0
    num_samples = 0
    plots_saved = 0
    total_inference_time = 0.0

    print(f"Evaluating on {len(dataloader.dataset)} samples...")
    print(f"Using metrics: {', '.join(metric_names)}")
    evaluation_start_time = time.time()
    
    # Store some examples for diagnostic visualisation
    diagnostic_examples = []
    
    print("\n" + "="*80)
    print("EVALUATION PROGRESS")
    print("="*80)
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            # DIAGNOSTIC: Save first batch raw tensors
            if batch_idx == 0 and diagnostic_dir:
                try:
                    print(f"Diagnostic: Analyzing raw input/target tensors from batch {batch_idx}")
                    print(f"Raw input shape: {inputs.shape}, dtype: {inputs.dtype}")
                    print(f"Raw input range: {inputs.min().item():.6f} to {inputs.max().item():.6f}")
                    print(f"Raw target range: {targets.min().item():.6f} to {targets.max().item():.6f}")
                except Exception as e:
                    print(f"Error analyzing raw diagnostic tensors: {e}")
            
            inputs, targets = inputs.to(device), targets.to(device)

            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                # Time the model inference step specifically
                inference_start_time = time.time()
                outputs = model(inputs)
                inference_end_time = time.time()
                total_inference_time += (inference_end_time - inference_start_time)
                
                if isinstance(outputs, tuple): 
                    outputs = outputs[0]
                loss = criterion(outputs, targets)

            batch_size = inputs.size(0)
            running_loss += loss.item() * batch_size
            
            # DIAGNOSTIC: Analyse first batch processed tensors
            if batch_idx == 0 and diagnostic_dir:
                try:
                    print(f"Diagnostic: Analyzing processed tensors from batch {batch_idx}")
                    print(f"Processed input range: {inputs.min().item():.6f} to {inputs.max().item():.6f}")
                    print(f"Processed output range: {outputs.min().item():.6f} to {outputs.max().item():.6f}")
                    print(f"Processed target range: {targets.min().item():.6f} to {targets.max().item():.6f}")
                except Exception as e:
                    print(f"Error analyzing processed diagnostic tensors: {e}")
            
            if metric_names:
                # Denormalise outputs and targets before metric calculation
                outputs_denorm = outputs.detach()
                targets_denorm = targets.detach()
                
                # Use the provided norm_info
                norm_type = norm_info.get('type', 'none')
                params_in = norm_info.get('input')  
                params_tgt = norm_info.get('target') 

                # Denormalise outputs using input stats
                if norm_type == "zscore" and params_in is not None:
                    mean_in, std_in = params_in
                    # Ensure params are tensors and on correct device for broadcast
                    if not isinstance(mean_in, torch.Tensor): mean_in = torch.tensor(mean_in)
                    if not isinstance(std_in, torch.Tensor): std_in = torch.tensor(std_in)
                    # Move parameters to the same device as outputs_denorm
                    mean_in = mean_in.to(outputs_denorm.device)
                    std_in = std_in.to(outputs_denorm.device)
                    outputs_denorm = (outputs_denorm * std_in.view(1, -1, 1, 1)) + mean_in.view(1, -1, 1, 1)
                elif norm_type == "minmax" and params_in is not None:
                    min_in, max_in = params_in
                    if not isinstance(min_in, torch.Tensor): min_in = torch.tensor(min_in)
                    if not isinstance(max_in, torch.Tensor): max_in = torch.tensor(max_in)
                    # Move parameters to the same device as outputs_denorm
                    min_in = min_in.to(outputs_denorm.device)
                    max_in = max_in.to(outputs_denorm.device)
                    denominator = max_in - min_in
                    denominator[denominator == 0] = 1e-8
                    outputs_denorm = (outputs_denorm * denominator.view(1, -1, 1, 1)) + min_in.view(1, -1, 1, 1)

                # Denormalise targets using target stats (or input stats if target used them)
                if norm_type == "zscore" and params_tgt is not None:
                    mean_tgt, std_tgt = params_tgt
                    if not isinstance(mean_tgt, torch.Tensor): mean_tgt = torch.tensor(mean_tgt)
                    if not isinstance(std_tgt, torch.Tensor): std_tgt = torch.tensor(std_tgt)
                    # Move parameters to the same device as targets_denorm
                    mean_tgt = mean_tgt.to(targets_denorm.device)
                    std_tgt = std_tgt.to(targets_denorm.device)
                    targets_denorm = (targets_denorm * std_tgt.view(1, -1, 1, 1)) + mean_tgt.view(1, -1, 1, 1)
                elif norm_type == "minmax" and params_tgt is not None:
                    min_tgt, max_tgt = params_tgt
                    if not isinstance(min_tgt, torch.Tensor): min_tgt = torch.tensor(min_tgt)
                    if not isinstance(max_tgt, torch.Tensor): max_tgt = torch.tensor(max_tgt)
                    # Move parameters to the same device as targets_denorm
                    min_tgt = min_tgt.to(targets_denorm.device)
                    max_tgt = max_tgt.to(targets_denorm.device)
                    denominator = max_tgt - min_tgt
                    denominator[denominator == 0] = 1e-8
                    targets_denorm = (targets_denorm * denominator.view(1, -1, 1, 1)) + min_tgt.view(1, -1, 1, 1)

                # DIAGNOSTIC: Analyse first batch denormalised tensors
                if batch_idx == 0 and diagnostic_dir:
                    try:
                        print(f"Diagnostic: Analyzing denormalised tensors from batch {batch_idx}")
                        print(f"Denormalised output range: {outputs_denorm.min().item():.6f} to {outputs_denorm.max().item():.6f}")
                        print(f"Denormalised target range: {targets_denorm.min().item():.6f} to {targets_denorm.max().item():.6f}")
                    except Exception as e:
                        print(f"Error analyzing denormalised diagnostic tensors: {e}")

                # Calculate metrics on DENORMALISED data
                batch_metrics = {}
                for metric_name in metric_names:
                    try:
                        metric_func = get_metric_function(metric_name)
                        metric_val = metric_func(outputs_denorm, targets_denorm)
                        results[metric_name] += metric_val.item() * batch_size
                        batch_metrics[metric_name] = metric_val.item()
                    except Exception as e:
                        print(f"Error calculating metric '{metric_name}' on denormalised data (batch {batch_idx}): {e}")
                        results[metric_name] += float('nan') * batch_size
                        batch_metrics[metric_name] = float('nan')
                
                # Print batch metrics for first few batches
                if batch_idx < 3:
                    print(f"Batch {batch_idx} metrics: {', '.join([f'{k}={v:.6f}' for k, v in batch_metrics.items()])}")
                
                # Save first few examples for later visualisation
                if len(diagnostic_examples) < min(plot_n_examples if plot_n_examples > 0 else 3, batch_size):
                    for i in range(min(plot_n_examples if plot_n_examples > 0 else 3, batch_size) - len(diagnostic_examples)):
                        diagnostic_examples.append({
                            'input': inputs[i].cpu().numpy(),
                            'output': outputs_denorm[i].cpu().numpy(),
                            'target': targets_denorm[i].cpu().numpy(),
                            'metrics': {k: v for k, v in batch_metrics.items()}
                        })
            
            num_samples += batch_size
    
    # Generate diagnostic visualisations
    if diagnostic_dir and diagnostic_examples:
        print("\n" + "="*80)
        print("DIAGNOSTIC VISUALISATIONS")
        print("="*80)
        try:
            print(f"Generating diagnostic visualisations in {diagnostic_dir}...")
            
            for idx, example in enumerate(diagnostic_examples):
                # Create complex data if possible (assuming first two channels are real and imaginary)
                if example['input'].shape[0] >= 2:
                    input_complex  = example['input'][0]  + 1j * example['input'][1]
                    output_complex = example['output'][0] + 1j * example['output'][1]
                    target_complex = example['target'][0] + 1j * example['target'][1]
                    
                    # Use plot_heatmap function to generate visualisations
                    metric_str = ", ".join([f"{k}={v:.6f}" for k, v in example['metrics'].items()])
                    custom_titles = {
                        'interp': f"Input\n{metric_str}",
                        'combined': "Model Output", 
                        'perfect': "Ground Truth"
                    }
                    
                    # Generate plots using shared plot_heatmap utility
                    interp_fig, model_fig = plot_heatmap(
                        interp=input_complex,
                        combined=output_complex,
                        perfect=target_complex,
                        save_path=diagnostic_dir,
                        filename=f"example_{idx}.svg",
                        show=False, 
                        use_titles=custom_titles,
                        error_plot=True,  
                        split_interp=True  
                    )
                    
                    print(f"Saved visualisation for example {idx}")
                else:
                    print(f"Skipping complex visualisation for example {idx} - insufficient channels")
                
            print("Diagnostic visualisations complete.")
        except Exception as e:
            print(f"Error generating diagnostic visualisations: {e}")
    
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    evaluation_time = time.time() - evaluation_start_time
    print(f"Evaluation finished in {format_time(evaluation_time)}:")

    # Aggregate and Log Results
    final_results = {}
    avg_loss = running_loss / num_samples if num_samples > 0 else float('nan')
    print(f"Average Loss: {avg_loss:.6f}")
    final_results['eval/loss'] = avg_loss 

    avg_inference_time_per_sample_ms = (total_inference_time / num_samples * 1000) if num_samples > 0 else float('nan')
    print(f"Average Inference Time per Sample: {avg_inference_time_per_sample_ms:.4f} ms")
    final_results['eval/avg_inference_time_ms'] = avg_inference_time_per_sample_ms

    for metric_name in results:
        avg_metric = results[metric_name] / num_samples if num_samples > 0 else float('nan')
        final_results[f"eval/{metric_name}"] = avg_metric 
        print(f"Average {metric_name.upper()}: {avg_metric:.6f}")


    # Log Final Metrics to W&B Summary
    if wandb.run:
        print("\n" + "="*80)
        print("WANDB LOGGING")
        print("="*80)
        wandb.log(final_results, commit=True)
        wandb.summary.update(final_results)
        wandb.summary['evaluation_time_seconds'] = evaluation_time
        print("Logged final evaluation metrics to W&B.")


    # Save Local Results YAML if W&B is not active
    if not wandb.run and save_directory:
        print("\n" + "="*80)
        print("SAVING RESULTS")
        print("="*80)
        results_save_path = os.path.join(save_directory, 'evaluation_metrics.yaml')
        try:
            results_to_save = {
                 'config_path_used': config_path if config_path else 'N/A',
                 'evaluation_results': {k.replace('eval/',''): v for k, v in final_results.items()},
                 'evaluation_time_seconds': evaluation_time,
                 'evaluated_samples': num_samples,
                 'local_results_directory': save_directory
            }
            with open(results_save_path, 'w') as f:
                 yaml.dump(results_to_save, f, default_flow_style=False)
            print(f"Local evaluation results summary saved to {results_save_path}")
        except Exception as e:
            print(f"Error saving local evaluation results YAML: {e}")
    elif not save_directory:
        print("Skipping local evaluation results YAML save (no save directory defined).")
    else:
        print("Skipping local evaluation results YAML save (W&B is active).")


    # Plot Learning Curves & Log to W&B if training history is provided
    if training_history and not wandb.run and save_directory:
        print("\n" + "="*80)
        print("LEARNING CURVES")
        print("="*80)
        print("Plotting and saving learning curves locally...")
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Handle both dictionary (baseline) and list (multi-task) formats
            if isinstance(training_history, dict):
                # Baseline history format
                if 'train_loss' in training_history and 'val_loss' in training_history:
                    epochs_ran = len(training_history['train_loss'])
                    if epochs_ran > 0:
                        ax.plot(range(1, epochs_ran + 1), training_history['train_loss'], label='Train Loss')
                        if len(training_history['val_loss']) == epochs_ran:
                            ax.plot(range(1, epochs_ran + 1), training_history['val_loss'], label='Validation Loss')
                        ax.set_xlabel('Epoch')
                        ax.set_ylabel('Loss')
                        ax.set_title('Learning Curve')
                        ax.legend()
                        ax.grid(True)
                    else:
                        ax.set_title('Learning Curve (No data)')
                        ax.text(0.5, 0.5, 'No training data recorded.', ha='center', va='center')
            else:
                # Handle list format (multiple tasks)
                num_tasks = len(training_history)
                fig, axes = plt.subplots(num_tasks, 1, figsize=(10, 6 * num_tasks), squeeze=False)
                
                for task_id, task_results in enumerate(training_history):
                    ax = axes[task_id, 0]
                    epochs_ran = len(task_results.get('train_losses', []))
                    if epochs_ran > 0:
                        ax.plot(range(1, epochs_ran + 1), task_results['train_losses'], label='Train Loss')
                        if 'val_losses' in task_results and len(task_results['val_losses']) == epochs_ran:
                            ax.plot(range(1, epochs_ran + 1), task_results['val_losses'], label='Validation Loss')
                        ax.set_xlabel('Epoch')
                        ax.set_ylabel('Loss')
                        ax.set_title(f'Learning Curve - Task {task_id}')
                        ax.legend()
                        ax.grid(True)
                    else:
                        ax.set_title(f'Learning Curve - Task {task_id} (No data)')
                        ax.text(0.5, 0.5, 'No training data recorded for this task.', ha='center', va='center')

            plt.tight_layout()
            learning_curve_path = os.path.join(save_directory, 'learning_curves.png')
            fig.savefig(learning_curve_path)
            print(f"Learning curves saved locally to {learning_curve_path}")
            plt.close(fig)

        except Exception as e:
            print(f"Error plotting/saving learning curves locally: {e}")
            if 'fig' in locals() and plt.fignum_exists(fig.number): plt.close(fig)
            
    elif not save_directory and training_history:
        print("Skipping local learning curve saving (no save directory).")
    elif training_history:
        print("Skipping local learning curve saving (W&B is active).")

    # Log Learning Curve Plot to W&B (if W&B active and history exists)
    if wandb.run and training_history:
        print("\n" + "="*80)
        print("WANDB VISUALISATION")
        print("="*80)
        print("Generating learning curves for W&B logging...")
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Handle both dictionary (baseline) and list (multi-task) formats
            if isinstance(training_history, dict):
                # Baseline history format
                if 'train_loss' in training_history and 'val_loss' in training_history:
                    epochs_ran = len(training_history['train_loss'])
                    if epochs_ran > 0:
                        ax.plot(range(1, epochs_ran + 1), training_history['train_loss'], label='Train Loss')
                        if len(training_history['val_loss']) == epochs_ran:
                            ax.plot(range(1, epochs_ran + 1), training_history['val_loss'], label='Validation Loss')
                        ax.set_xlabel('Epoch')
                        ax.set_ylabel('Loss')
                        ax.set_title('Learning Curve')
                        ax.legend()
                        ax.grid(True)
                    else:
                        ax.set_title('Learning Curve (No data)')
                        ax.text(0.5, 0.5, 'No training data recorded.', ha='center', va='center')
            else:
                # Handle list format (multiple tasks)
                num_tasks = len(training_history)
                fig, axes = plt.subplots(num_tasks, 1, figsize=(10, 6 * num_tasks), squeeze=False)
                
                for task_id, task_results in enumerate(training_history):
                    ax = axes[task_id, 0]
                    epochs_ran = len(task_results.get('train_losses', []))
                    if epochs_ran > 0:
                        ax.plot(range(1, epochs_ran + 1), task_results['train_losses'], label='Train Loss')
                        if 'val_losses' in task_results and len(task_results['val_losses']) == epochs_ran:
                            ax.plot(range(1, epochs_ran + 1), task_results['val_losses'], label='Validation Loss')
                        ax.set_xlabel('Epoch')
                        ax.set_ylabel('Loss')
                        ax.set_title(f'Learning Curve - Task {task_id}')
                        ax.legend()
                        ax.grid(True)
                    else:
                        ax.set_title(f'Learning Curve - Task {task_id} (No data)')
                        ax.text(0.5, 0.5, 'No training data recorded for this task.', ha='center', va='center')

            plt.tight_layout()
            wandb.log({"learning_curves": wandb.Image(fig)}, commit=True)
            print(f"Logged learning curves plot to W&B.")
            plt.close(fig)
        except Exception as e:
            print(f"Error plotting/logging learning curves to W&B: {e}")
            if 'fig' in locals() and plt.fignum_exists(fig.number): plt.close(fig)

    return final_results


def main(config_path, checkpoint_path=None, data_config=None):
    """Main function to run evaluation from a config file."""
    # Parse additional arguments
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate a model using a config file')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to model checkpoint (optional)')
    parser.add_argument('--data_config', type=str, default=None, help='Path to data config override (optional)')
    parser.add_argument('--experiment_suffix', type=str, default="",
                        help='Suffix to append to the experiment_name (e.g., _eval_a, _test_b)')
    parser.add_argument('--no_wandb', action='store_true',
                        help='Disable Weights & Biases logging')
    args = parser.parse_args()
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print("\n" + "="*80)
    print("CONFIGURATION")
    print("="*80)
    
    # Apply command-line overrides
    if args.experiment_suffix:
        original_name = config.get('experiment_name', 'experiment')
        new_name = original_name + args.experiment_suffix
        print(f"Appending suffix '{args.experiment_suffix}' to experiment name. New name: '{new_name}'")
        config['experiment_name'] = new_name
        if 'logging' in config and 'checkpoint_dir' in config['logging']:
            base_checkpoint_dir = os.path.dirname(config['logging']['checkpoint_dir'].rstrip('/\\'))
            config['logging']['checkpoint_dir'] = os.path.join(base_checkpoint_dir, new_name)
    
    # Override data configuration if provided
    if data_config is not None:
        print(f"Loading data configuration override from: {data_config}")
        with open(data_config, 'r') as f:
            data_override = yaml.safe_load(f)
            # Update data section of config
            if 'data' in data_override:
                config['data'] = data_override['data']
                print("Data configuration overridden.")
    
    print("\n" + "="*80)
    print("HARDWARE SETUP")
    print("="*80)
    
    # Import necessary modules here to avoid circular imports
    from standard_training.utils.utils import load_model
    from standard_training.datasets.dataset_utils import load_data
    
    # Setup device
    device_name = config.get('hardware', {}).get('device', 'auto').lower()
    if device_name == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif device_name == 'cuda':
        if not torch.cuda.is_available():
            print("Warning: CUDA requested but not available. Using CPU.")
            device = torch.device("cpu")
        else:
            device = torch.device("cuda")
    elif device_name == 'cpu':
        device = torch.device("cpu")
    else:
        print(f"Warning: Invalid device name '{device_name}'. Using auto-detection.")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print("\n" + "="*80)
    print("DATA LOADING")
    print("="*80)
    
    # Load evaluation data
    dataloaders, norm_info = load_data(config, mode='evaluate')
    
    print("\n" + "="*80)
    print("MODEL LOADING")
    print("="*80)
    
    # Load model
    if checkpoint_path:
        # Override the model path in config if provided
        if 'model' not in config:
            config['model'] = {}
        print(f"Loading model from checkpoint: {checkpoint_path}")
        config['model']['evaluation_path'] = checkpoint_path
    
    model = load_model(config)
    model = model.to(device)
    
    print("\n" + "="*80)
    print("WANDB INITIALIZATION")
    print("="*80)
    
    # Initialise W&B if configured and not disabled via command line
    use_wandb = config.get('logging', {}).get('use_wandb', False) and not args.no_wandb
    if use_wandb:
        wandb_config = config.get('wandb', {})
        wandb_project = wandb_config.get('project', 'standard_evaluation')
        wandb_entity = wandb_config.get('entity', None)
        wandb_name = wandb_config.get('name', None)
        print(f"Initializing W&B - Project: {wandb_project}, Entity: {wandb_entity}")
        wandb.init(project=wandb_project, entity=wandb_entity, name=wandb_name, config=config)
    else:
        print("W&B logging disabled.")
    
    # Ensure we have an evaluation dataloader
    if 'eval' not in dataloaders and 'test' not in dataloaders:
        raise ValueError("No evaluation dataloader found. Make sure your data configuration includes 'eval' or 'test' sets.")
    
    eval_loader = dataloaders.get('eval', dataloaders.get('test'))
    
    # Run evaluation
    print("\n" + "="*80)
    print("STARTING EVALUATION")
    print("="*80)
    
    plot_examples = config.get('evaluation', {}).get('plot_examples', 3)
    
    # Get results directory from config
    results_dir = config.get('evaluation', {}).get('save_results_dir')
    
    # Don't create results directory for sweep runs
    if wandb.run and wandb.run.sweep_id:
        print(f"Skipping results directory creation (sweep run)")
        results_dir = None
    elif results_dir:
        print(f"Will use results directory from config: {results_dir}")
    else:
        print("No results directory specified in config.")
    
    results = evaluate_model(
        config=config,
        model=model,
        dataloader=eval_loader,
        device=device,
        norm_info=norm_info,
        plot_n_examples=plot_examples,
        results_dir=results_dir,
        config_path=config_path
    )
    
    # Finish W&B run
    if wandb.run:
        print("\n" + "="*80)
        print("FINISHING UP")
        print("="*80)
        wandb.finish()
        print("W&B run completed.")
    
    print("\nEvaluation complete!")
    return results


if __name__ == "__main__":
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description='Evaluate a model using a config file')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to model checkpoint (optional)')
    parser.add_argument('--data_config', type=str, default=None, help='Path to data config override (optional)')
    parser.add_argument('--experiment_suffix', type=str, default="",
                        help='Suffix to append to the experiment_name (e.g., _eval_a, _test_b)')
    parser.add_argument('--no_wandb', action='store_true',
                        help='Disable Weights & Biases logging')
    args = parser.parse_args()
    
    main(args.config, args.checkpoint, args.data_config)
