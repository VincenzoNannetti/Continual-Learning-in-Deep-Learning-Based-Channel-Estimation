"""
Filename: continual_learning/evaluate.py
Author: Vincenzo Nannetti
Date: 15/05/2025
Description: Continual Learning Evaluation Script - Entry point for supermask evaluation

Usage: python -m continual_learning.evaluate --config configs/supermask_config.yaml --checkpoint_dir checkpoints/my_run

Dependencies:
    - wandb
    - continual_learning.evaluators.supermask_evaluator
"""

import yaml
import os
import time
import wandb
import argparse
import sys
from datetime import datetime

from continual_learning.evaluators.supermask_evaluator import SupermaskEvaluator
from shared.utils.format_time import format_time


def evaluate_model(config, checkpoint_dir=None, task_id=None, norm_params=None):
    """
    Evaluate a trained model on a specific task or all tasks.
    
    Args:
        config (dict): Configuration dictionary
        checkpoint_dir (str): Directory containing model checkpoints
        task_id (int, optional): Specific task ID to evaluate. If None, evaluates all tasks.
        norm_params (dict, optional): Normalisation parameters for the task
        
    Returns:
        dict: Evaluation results
    """
    # Create evaluator
    evaluator = SupermaskEvaluator(config, checkpoint_dir)
    
    # Load model
    evaluator.load_model()
    
    # Evaluate model
    if task_id is not None:
        print(f"\nEvaluating task {task_id}...")
        results = evaluator.evaluate_task(task_id)
    else:
        print("\nEvaluating all tasks...")
        results = evaluator.evaluate_all_tasks()
        
    return results


def main():
    """Main function to run continual learning evaluation from a config file."""
    parser = argparse.ArgumentParser(description='Evaluate a continual learning model using a config file')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--checkpoint_dir', type=str, required=True, help='Path to checkpoint directory')
    parser.add_argument('--task_id', type=int, default=None, help='Specific task ID to evaluate')
    parser.add_argument('--no_wandb', action='store_true', help='Disable Weights & Biases logging')
    parser.add_argument('--plot_examples', type=int, default=None, help='Number of examples to plot')
    args = parser.parse_args()
    
    # Load config
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading config file: {e}")
        sys.exit(1)
    
    # Update config with command-line options
    if args.plot_examples is not None:
        config['evaluation'] = config.get('evaluation', {})
        config['evaluation']['plot_n_examples'] = args.plot_examples
    
    # Initialise W&B if configured and not disabled via command line
    use_wandb = config.get('logging', {}).get('use_wandb', False) and not args.no_wandb
    if use_wandb:
        print("\n" + "="*80)
        print("WANDB INITIALISATION")
        print("="*80)
        wandb_config = config.get('wandb', {})
        wandb_project = wandb_config.get('project', 'continual_learning')
        wandb_entity = wandb_config.get('entity', None)
        wandb_name = f"{wandb_config.get('name', 'eval')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        wandb.init(project=wandb_project, entity=wandb_entity, name=wandb_name, config=config, job_type='evaluation')
    
    # Evaluate model
    print("\n" + "="*80)
    print("SUPERMASK EVALUATION")
    print("="*80)
    
    start_time = time.time()
    results = evaluate_model(config, args.checkpoint_dir, args.task_id)
    eval_time = time.time() - start_time
    
    print(f"\nEvaluation completed in {format_time(eval_time)}")
    
    # Save results locally
    if results and not (wandb.run and wandb.run.sweep_id):
        try:
            results_dir = os.path.join(args.checkpoint_dir, 'evaluation_results')
            os.makedirs(results_dir, exist_ok=True)
            
            # Save as YAML
            results_path = os.path.join(results_dir, f"eval_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml")
            with open(results_path, 'w') as f:
                yaml.dump(results, f, default_flow_style=False)
            print(f"Results saved to {results_path}")
        except Exception as e:
            print(f"Error saving results: {e}")
    
    # Finish W&B run
    if wandb.run:
        wandb.finish()
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    return results


if __name__ == "__main__":
    main()
