"""
Filename: continual_learning/train.py
Author: Vincenzo Nannetti
Date: 15/05/2025
Description: Continual Learning Training Script - Entry point for supermask training

Usage: python -m continual_learning.train --config configs/supermask_config.yaml

Dependencies:
    - PyTorch
    - wandb
    - continual_learning.trainers.supermask_trainer
"""

import torch
import yaml
import os
import time
import wandb
import argparse
import sys

from continual_learning.trainers.supermask_trainer import SupermaskTrainer
from shared.utils.format_time import format_time
from continual_learning.evaluate import evaluate_model

def main():
    """Main function to run continual learning training from a config file."""
    parser = argparse.ArgumentParser(description='Train a continual learning model using a config file')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--experiment_suffix', type=str, default="",
                        help='Suffix to append to the experiment_name (e.g., _train_a, _finetune_b)')
    parser.add_argument('--no_wandb', action='store_true',
                        help='Disable Weights & Biases logging')
    parser.add_argument('--dataset_to_use', type=str, default=None, choices=['a', 'b'],
                        help='Specify which dataset to use based on dataset_a_name or dataset_b_name in config')
    parser.add_argument('--no_eval', action='store_true',
                        help='Do not run evaluation after training')
    parser.add_argument('--supermask_sequence', type=str, default=None,
                        help='Override supermask training sequence (comma-separated list of task IDs)')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
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

    if args.supermask_sequence:
        sequence = args.supermask_sequence.split(',')
        # Convert numeric values to integers
        processed_sequence = []
        for item in sequence:
            if item.isdigit():
                processed_sequence.append(int(item))
            else:
                processed_sequence.append(item)
        
        print(f"Overriding supermask.sequence with: {processed_sequence}")
        if 'supermask' not in config:
            config['supermask'] = {}
        config['supermask']['sequence'] = processed_sequence
    
    # Initialise W&B if configured and not disabled via command line
    use_wandb = config.get('logging', {}).get('use_wandb', False) and not args.no_wandb
    if use_wandb:
        print("\n" + "="*80)
        print("WANDB INITIALISATION")
        print("="*80)
        wandb_config = config.get('wandb', {})
        wandb_project = wandb_config.get('project', 'continual_learning')
        print(f"Saving to W&B project: {wandb_project}")
        wandb_entity = wandb_config.get('entity', None)
        print(f"Saving to W&B entity: {wandb_entity}")
        wandb_name = wandb_config.get('name', None)
        print(f"Saving to W&B name: {wandb_name}")
        wandb.init(project=wandb_project, entity=wandb_entity, name=wandb_name, config=config)
    
    # Create and run SupermaskTrainer
    print("\n" + "="*80)
    print("SUPERMASK TRAINING")
    print("="*80)
    
    start_time = time.time()
    trainer = SupermaskTrainer(config)
    results, norm_params, checkpoint_dir = trainer.train()
    training_time = time.time() - start_time
    
    print(f"Training completed in {format_time(training_time)}")
    
    # Automatically run evaluation if not disabled
    if not args.no_eval and checkpoint_dir:
        print("\n" + "="*80)
        print("EVALUATION")
        print("="*80)
        try:
            # Import the evaluate module to run evaluation
            print(f"Starting post-training evaluation...")
            
            # The best model should be loaded and set to the appropriate task
            # Based on the supermask.sequence in the config
            num_tasks = config.get('supermask', {}).get('tasks', 1)
            
            # Handle evaluation for each task
            for task_id in range(num_tasks):
                print(f"\nEvaluating on task {task_id}...")
                
                # Set up model and dataset for the current task
                # This should be done in the evaluate_model function
                results = evaluate_model(
                    config=config,
                    checkpoint_dir=checkpoint_dir,
                    task_id=task_id,
                    norm_params=norm_params.get(task_id, {})
                )
                
                print(f"Task {task_id} evaluation complete.")
                
        except Exception as e:
            print(f"Error during automatic evaluation: {e}")
    
    # Finish W&B run
    if wandb.run:
        wandb.finish()
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    return results, norm_params, checkpoint_dir


if __name__ == "__main__":
    main()
