"""
Optuna hyperparameter optimisation for LoRA-based continual learning.
This differs from standard training by evaluating across ALL domains and optimizing LoRA-specific parameters.
"""

import optuna
import yaml
import sys
import os
import argparse
import wandb
import numpy as np
import time
from pathlib import Path
from typing import Dict, List

# Ensure the script can find modules
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))
if str(SCRIPT_DIR.parent) not in sys.path:
    sys.path.append(str(SCRIPT_DIR.parent))

def calculate_multi_domain_composite_score_from_results(results_data: Dict, trial_number: int, domain_ids: List[str]) -> float:
    """
    Calculate composite score from evaluation results data (from file or W&B).
    
    Args:
        results_data: Dictionary containing domain results
        trial_number: Trial number for logging
        domain_ids: List of domain IDs to evaluate
    
    Returns:
        Composite score to be MAXIMIZED by Optuna
    """
    print(f"[Optuna T{trial_number}] Multi-Domain Evaluation:")
    
    # Extract domain metrics from results
    domain_metrics = {}
    missing_domains = []
    
    # Results can come from file (domain_results key) or W&B (direct access)
    if 'domain_results' in results_data:
        # From file
        for domain_id in domain_ids:
            if domain_id in results_data['domain_results']:
                domain_metrics[domain_id] = results_data['domain_results'][domain_id]
                print(f"  Domain {domain_id}: SSIM={domain_metrics[domain_id]['ssim']:.4f}, NMSE={domain_metrics[domain_id]['nmse']:.6f}")
            else:
                missing_domains.append(domain_id)
    else:
        # From W&B (backward compatibility)
        for domain_id in domain_ids:
            ssim_key = f"eval_domain_{domain_id}_ssim"
            nmse_key = f"eval_domain_{domain_id}_nmse"
            
            if ssim_key in results_data and nmse_key in results_data:
                domain_metrics[domain_id] = {
                    'ssim': float(results_data[ssim_key]),
                    'nmse': float(results_data[nmse_key])
                }
                print(f"  Domain {domain_id}: SSIM={domain_metrics[domain_id]['ssim']:.4f}, NMSE={domain_metrics[domain_id]['nmse']:.6f}")
            else:
                missing_domains.append(domain_id)
    
    if missing_domains:
        print(f"  Missing metrics for domains: {missing_domains}")
        if len(domain_metrics) == 0:
            raise optuna.exceptions.TrialPruned("No domain evaluation metrics found")
    
    # Calculate aggregate metrics
    ssim_values = [metrics['ssim'] for metrics in domain_metrics.values()]
    nmse_values = [metrics['nmse'] for metrics in domain_metrics.values()]
    
    # 1. Average performance
    avg_ssim = np.mean(ssim_values)
    avg_nmse = np.mean(nmse_values)
    
    # 2. Stability across domains (lower variance is better)
    ssim_std = np.std(ssim_values)
    nmse_std = np.std(nmse_values)
    
    # 3. Worst domain performance (to penalize catastrophic forgetting)
    min_ssim = np.min(ssim_values)
    max_nmse = np.max(nmse_values)
    
    print(f"  Aggregate Metrics:")
    print(f"    Avg SSIM: {avg_ssim:.4f} ± {ssim_std:.4f} (min: {min_ssim:.4f})")
    print(f"    Avg NMSE: {avg_nmse:.6f} ± {nmse_std:.6f} (max: {max_nmse:.6f})")
    
    # === WORST-DOMAIN FIRST COMPOSITE SCORE ===
    
    # Extract individual domain performance
    ssim_values = [metrics['ssim'] for metrics in domain_metrics.values()]
    nmse_values = [metrics['nmse'] for metrics in domain_metrics.values()]
    
    # Worst-domain first: focus on the weakest link
    min_ssim = np.min(ssim_values)  # Worst SSIM performance
    max_nmse = np.max(nmse_values)  # Worst NMSE performance
    
    # Simple worst-domain objective (higher is better)
    lambda_log = 0.4  # Weight for NMSE term
    final_score = min_ssim - lambda_log * np.log10(max_nmse + 1e-6)
    
    print(f"  Score Breakdown (Worst-Domain First):")
    print(f"    Min SSIM (worst domain): {min_ssim:.4f}")
    print(f"    Max NMSE (worst domain): {max_nmse:.6f}")
    print(f"    SSIM contribution: +{min_ssim:.4f}")
    print(f"    NMSE penalty: -{lambda_log * np.log10(max_nmse + 1e-6):.4f}")
    print(f"    FINAL SCORE: {final_score:.4f}")
    
    # Additional logging for debugging
    domain_breakdown = []
    for domain_id, metrics in domain_metrics.items():
        domain_breakdown.append(f"D{domain_id}: SSIM={metrics['ssim']:.3f}")
    print(f"    Domain breakdown: {', '.join(domain_breakdown)}")
    
    return final_score

def calculate_multi_domain_composite_score(wandb_run, trial_number: int, domain_ids: List[str]) -> float:
    """
    Calculate composite score across ALL domains for LoRA continual learning.
    
    Aggregates performance across domains with penalties for:
    1. Poor average performance
    2. High variance between domains (catastrophic forgetting)
    3. Training instability
    
    Returns a score to be MAXIMIZED by Optuna.
    """
    print(f"[Optuna T{trial_number}] Multi-Domain Evaluation:")
    
    # Collect metrics for each domain
    domain_metrics = {}
    missing_domains = []
    
    for domain_id in domain_ids:
        # Try to get evaluation metrics for this domain
        ssim_key = f"eval_domain_{domain_id}_ssim"
        nmse_key = f"eval_domain_{domain_id}_nmse"
        
        if ssim_key in wandb_run.summary and nmse_key in wandb_run.summary:
            domain_metrics[domain_id] = {
                'ssim': float(wandb_run.summary[ssim_key]),
                'nmse': float(wandb_run.summary[nmse_key])
            }
            print(f"  Domain {domain_id}: SSIM={domain_metrics[domain_id]['ssim']:.4f}, NMSE={domain_metrics[domain_id]['nmse']:.6f}")
        else:
            missing_domains.append(domain_id)
    
    if missing_domains:
        print(f"  Missing metrics for domains: {missing_domains}")
        if len(domain_metrics) == 0:
            raise optuna.exceptions.TrialPruned("No domain evaluation metrics found")
    
    # Calculate aggregate metrics
    ssim_values = [metrics['ssim'] for metrics in domain_metrics.values()]
    nmse_values = [metrics['nmse'] for metrics in domain_metrics.values()]
    
    # 1. Average performance
    avg_ssim = np.mean(ssim_values)
    avg_nmse = np.mean(nmse_values)
    
    # 2. Stability across domains (lower variance is better)
    ssim_std = np.std(ssim_values)
    nmse_std = np.std(nmse_values)
    
    # 3. Worst domain performance (to penalize catastrophic forgetting)
    min_ssim = np.min(ssim_values)
    max_nmse = np.max(nmse_values)
    
    print(f"  Aggregate Metrics:")
    print(f"    Avg SSIM: {avg_ssim:.4f} ± {ssim_std:.4f} (min: {min_ssim:.4f})")
    print(f"    Avg NMSE: {avg_nmse:.6f} ± {nmse_std:.6f} (max: {max_nmse:.6f})")
    
    # === COMPOSITE SCORE CALCULATION ===
    
    # Base score from average performance
    ssim_score = float(avg_ssim)
    nmse_score = 1.0 - np.clip(np.log10(avg_nmse + 1e-6) / np.log10(0.5 + 1e-6), 0.0, 1.0)
    base_score = 0.6 * ssim_score + 0.4 * nmse_score
    
    # Stability bonus (reward consistent performance across domains)
    ssim_stability = 1.0 - np.clip(ssim_std / 0.2, 0.0, 1.0)  # Normalize by reasonable std
    nmse_stability = 1.0 - np.clip(nmse_std / (avg_nmse + 1e-6), 0.0, 1.0)  # Relative std
    stability_bonus = 0.1 * (0.6 * ssim_stability + 0.4 * nmse_stability)
    
    # Catastrophic forgetting penalty (penalize very poor worst-case performance)
    cf_penalty = 0.0
    if min_ssim < 0.5:  # If any domain performs very poorly
        cf_penalty = 0.15 * (0.5 - min_ssim)  # Up to 0.075 penalty
    if max_nmse > avg_nmse * 3:  # If any domain has 3x worse NMSE
        cf_penalty += 0.05  # Additional penalty
    
    # Training stability (get from training metrics if available)
    training_penalty = 0.0
    
    # Use best model metrics if available, otherwise fall back to final metrics
    best_mean_val_loss = wandb_run.summary.get("final_best_mean_val_loss", None)
    best_epoch = wandb_run.summary.get("final_best_epoch", None)
    best_task = wandb_run.summary.get("final_best_task", None)
    
    # Fallback to final metrics if best not available
    final_train_loss = wandb_run.summary.get("final_train_loss", None)
    final_val_loss = wandb_run.summary.get("final_val_loss", None)
    
    # Check for gradient explosion indicators
    max_grad_norm = wandb_run.summary.get("max_gradient_norm", None)
    if max_grad_norm is not None and max_grad_norm > 100:  # Very large gradients
        training_penalty += 0.1  # Penalize gradient explosion
    
    if final_train_loss is not None and final_val_loss is not None:
        # Overfitting penalty
        train_val_gap = abs(final_val_loss - final_train_loss)
        avg_loss = (final_train_loss + final_val_loss) / 2
        relative_gap = train_val_gap / (avg_loss + 1e-8)
        overfitting_penalty = 0.05 * (1 - np.exp(-2 * relative_gap))
        
        # Late convergence penalty (only if we have epoch info)
        convergence_penalty = 0.0
        if best_epoch is not None:
            convergence_penalty = 0.05 * (1 - np.exp(-0.05 * max(0, best_epoch - 20)))
        
        training_penalty = overfitting_penalty + convergence_penalty
    
    # Final score
    final_score = base_score + stability_bonus - cf_penalty - training_penalty
    
    print(f"  Score Breakdown:")
    print(f"    Base Score: {base_score:.4f} (SSIM: {0.6 * ssim_score:.3f}, NMSE: {0.4 * nmse_score:.3f})")
    print(f"    Stability Bonus: +{stability_bonus:.4f}")
    print(f"    Catastrophic Forgetting Penalty: -{cf_penalty:.4f}")
    print(f"    Training Stability Penalty: -{training_penalty:.4f}")
    if best_mean_val_loss is not None:
        print(f"    Best Model: Task {best_task}, Epoch {best_epoch}, Val Loss {best_mean_val_loss:.6f}")
    print(f"    FINAL SCORE: {final_score:.4f}")
    
    return final_score

def create_lora_rank_suggestions(trial: optuna.Trial, num_domains: int = 9, 
                                target_domain_override: int = None) -> Dict[str, int]:
    """
    SINGLE-DOMAIN rank testing.
    
    Strategy: Train ONLY on target domain with different ranks.
    Each trial is independent single-domain training, not continual learning.
    
    Args:
        trial: Optuna trial
        num_domains: Total number of domains  
        target_domain_override: If provided, always test this domain (for chunk-based testing)
    
    Trial mapping (when target_domain_override is None):
    - Trials 0-4: Train only on Domain 0 with ranks [2,4,8,12,16]
    - Trials 5-9: Train only on Domain 1 with ranks [2,4,8,12,16]
    - etc.
    
    When target_domain_override is provided:
    - Trials 0-4: Train only on target_domain with ranks [2,4,8,12,16]
    """
    
    # Define rank options
    rank_options = [2, 4, 8, 12, 16]  # Respects r << 18 constraint
    
    # Determine target domain and rank
    if target_domain_override is not None:
        # Chunk-based testing: test specific domain with trial.number determining rank
        target_domain = target_domain_override
        rank_index = trial.number % len(rank_options)
        target_rank = rank_options[rank_index]
        
        if trial.number >= len(rank_options):
            raise optuna.exceptions.TrialPruned(
                f"Trial {trial.number} exceeds planned ranks for domain {target_domain}. "
                f"Expected max: {len(rank_options) - 1}"
            )
    else:
        # Full systematic testing: trial number determines both domain and rank
        target_domain = trial.number // len(rank_options)
        rank_index = trial.number % len(rank_options)
        target_rank = rank_options[rank_index]
        
        if target_domain >= num_domains:
            raise optuna.exceptions.TrialPruned(
                f"Trial {trial.number} exceeds planned domain-rank combinations. "
                f"Expected max: {num_domains * len(rank_options) - 1}"
            )
    
    # Build domain rank configuration - ONLY for the target domain
    # Other domains won't be trained, so their ranks don't matter
    domain_ranks = {str(target_domain): target_rank}
    
    mode = "CHUNK" if target_domain_override is not None else "FULL SYSTEMATIC"
    print(f"   {mode} SINGLE-DOMAIN TRAINING:")
    print(f"     Trial {trial.number}: Train ONLY on Domain {target_domain} with rank {target_rank}")
    print(f"     Progress: Rank {rank_index+1}/{len(rank_options)} for Domain {target_domain}")
    print(f"     Configuration: {domain_ranks}")
    
    return domain_ranks

def create_lora_alpha_suggestions(trial: optuna.Trial, num_domains: int = 9, 
                                  domain_ranks: Dict[str, int] = None) -> Dict[str, int]:
    """
    Suggest LoRA alpha values for target domain only.
    
    For single-domain training, we only need alpha for the domain being trained.
    Uses alpha = rank strategy based on LoRA paper recommendation.
    """
    
    if domain_ranks is None:
        raise ValueError("domain_ranks must be provided when using alpha = rank strategy")
    
    # Implement alpha = rank strategy (recommended by LoRA paper)
    # Only for the domain(s) we're actually training
    domain_alphas = {domain_id: rank for domain_id, rank in domain_ranks.items()}
    
    print(f"   Using ALPHA = RANK strategy for target domain: {domain_alphas}")
    
    return domain_alphas

def suggest_lora_params_from_sweep(trial: optuna.Trial, sweep_params_space: dict, num_domains: int = 9):
    """
    Enhanced parameter suggestion that handles LoRA-specific parameters.
    """
    suggested_params = {}
    
    # Handle standard parameters first
    for param_name, param_details in sweep_params_space.items():
        # Skip metadata and LoRA-specific parameters (we'll handle those separately)
        if param_name in ['train_config_path', 'wandb.project', 'wandb.entity', 
                         'optuna_runner_wandb.project', 'optuna_runner_wandb.entity',
                         'task_lora_ranks', 'task_lora_alphas']:
            continue
            
        if 'value' in param_details:
            suggested_params[param_name] = param_details['value']
            print(f"  Fixed parameter: {param_name} = {param_details['value']}")
        elif 'values' in param_details:
            suggested_params[param_name] = trial.suggest_categorical(param_name, param_details['values'])
        elif 'distribution' in param_details:
            dist = param_details['distribution']
            min_val_orig = param_details.get('min')
            max_val_orig = param_details.get('max')
            
            try:
                if dist in ['uniform', 'log_uniform', 'log_uniform_values']:
                    min_val_f = float(min_val_orig)
                    max_val_f = float(max_val_orig)
                    suggested_params[param_name] = trial.suggest_float(
                        param_name, min_val_f, max_val_f, 
                        log=(dist != 'uniform')
                    )
                elif dist == 'int_uniform':
                    min_val_i = int(float(min_val_orig))
                    max_val_i = int(float(max_val_orig))
                    suggested_params[param_name] = trial.suggest_int(param_name, min_val_i, max_val_i)
                elif dist == 'q_uniform':
                    min_q, max_q, q_val = float(min_val_orig), float(max_val_orig), float(param_details['q'])
                    suggested_params[param_name] = trial.suggest_float(param_name, min_q, max_q, step=q_val)
            except (ValueError, TypeError) as e:
                print(f"Warning: Could not process param '{param_name}': {e}")
                continue
    
    # Add LoRA-specific parameters
    # Check if we have a target domain override from command line
    target_domain_override = getattr(trial.study, '_target_domain_override', None)
    lora_ranks = create_lora_rank_suggestions(trial, num_domains, target_domain_override=target_domain_override)
    lora_alphas = create_lora_alpha_suggestions(trial, num_domains, domain_ranks=lora_ranks)
    
    suggested_params['task_lora_ranks'] = lora_ranks
    suggested_params['task_lora_alphas'] = lora_alphas
    
    print(f"  LoRA Ranks: {lora_ranks}")
    print(f"  LoRA Alphas: {lora_alphas}")
    print(f"   Alpha/Rank Ratios: {', '.join([f'D{k}: {v}/{lora_ranks[k]}={v/lora_ranks[k]:.1f}' for k, v in lora_alphas.items()])}")
    
    return suggested_params



def parse_sweep_config(sweep_config_path_str: str):
    """Parse the LoRA-specific sweep YAML configuration."""
    sweep_config_path = Path(sweep_config_path_str)
    with open(sweep_config_path, 'r') as f:
        sweep_config = yaml.safe_load(f)
    
    if 'parameters' not in sweep_config:
        raise ValueError(f"Key 'parameters' not found in sweep config: {sweep_config_path}")

    base_train_config_rel_path = sweep_config['parameters'].get('train_config_path', {}).get('value')
    if not base_train_config_rel_path:
        raise ValueError(f"Base training config path not found in sweep parameters")

    metric_info = sweep_config.get('metric', {'name': 'multi_domain_composite_score', 'goal': 'maximize'})
    metric_to_optimize = metric_info['name']
    optimization_goal = metric_info['goal']
    
    wandb_project_sweep = sweep_config['parameters'].get('wandb.project', {}).get('value')
    wandb_entity_sweep = sweep_config['parameters'].get('wandb.entity', {}).get('value')

    return sweep_config['parameters'], base_train_config_rel_path, metric_to_optimize, optimization_goal, wandb_project_sweep, wandb_entity_sweep

def update_config_with_trial_params(base_config: dict, trial_params: dict):
    """Update the base config with LoRA-specific trial parameters for single-domain training."""
    updated_config = yaml.safe_load(yaml.safe_dump(base_config))
    
    for key, value in trial_params.items():
        # Handle special LoRA parameters
        if key == 'task_lora_ranks':
            # Convert string keys to integers for the config
            updated_config.setdefault('model', {}).setdefault('params', {})['task_lora_ranks'] = {
                int(k): v for k, v in value.items()
            }
            
            # For single-domain training, set the data sequence to only the target domain
            target_domain = int(list(value.keys())[0])  # Get the single domain we're training
            updated_config.setdefault('data', {})['sequence'] = [target_domain]
            print(f"   Modified config to train only on domain {target_domain}")
            
        elif key == 'task_lora_alphas':
            # Convert string keys to integers for the config
            updated_config.setdefault('model', {}).setdefault('params', {})['task_lora_alphas'] = {
                int(k): v for k, v in value.items()
            }
        else:
            # Handle regular dot-notation parameters
            parts = key.split('.')
            d = updated_config
            for part in parts[:-1]:
                if part not in d or not isinstance(d[part], dict):
                    d[part] = {}
                d = d[part]
            d[parts[-1]] = value
    
    return updated_config

def objective(trial: optuna.Trial, sweep_config_yaml_path_str: str, 
              results_root_dir: Path, domain_ids: List[str]):
    """Optuna objective function for LoRA continual learning."""
    current_trial_full_config = {}

    try:
        # Parse sweep configuration
        sweep_params_space, base_train_cfg_rel_path, metric_name, metric_goal, \
            wandb_proj_sweep, wandb_entity_sweep = parse_sweep_config(sweep_config_yaml_path_str)

        # Generate trial hyperparameters with LoRA-specific handling
        trial_hyperparams = suggest_lora_params_from_sweep(trial, sweep_params_space, len(domain_ids))

        # Load base configuration
        actual_base_train_cfg_path = (Path(sweep_config_yaml_path_str).parent / base_train_cfg_rel_path).resolve()
        if not actual_base_train_cfg_path.exists():
            raise optuna.exceptions.TrialPruned(f"Base train config not found: {actual_base_train_cfg_path}")
        
        with open(actual_base_train_cfg_path, 'r') as f:
            base_cfg_content = yaml.safe_load(f)
        
        current_trial_full_config = update_config_with_trial_params(base_cfg_content, trial_hyperparams)
        
        # Setup trial directory
        trial_results_dir = results_root_dir / f"trial_{trial.number}_lora"
        trial_checkpoints_dir = trial_results_dir / "checkpoints" / "lora"
        trial_checkpoints_dir.mkdir(parents=True, exist_ok=True)
        
        # Update config with checkpoint directory
        current_trial_full_config.setdefault('logging', {})['checkpoint_dir'] = str(trial_checkpoints_dir.parent)

        # Debug: Print actual LoRA config that will be used
        if 'model' in current_trial_full_config and 'params' in current_trial_full_config['model']:
            lora_ranks = current_trial_full_config['model']['params'].get('task_lora_ranks', {})
            lora_alphas = current_trial_full_config['model']['params'].get('task_lora_alphas', {})
            print(f"[Optuna T{trial.number}] DEBUG - Config LoRA Ranks: {lora_ranks}")
            print(f"[Optuna T{trial.number}] DEBUG - Config LoRA Alphas: {lora_alphas}")

        # Save trial configuration
        temp_cfg_file = trial_results_dir / f"trial_{trial.number}_config.yaml"
        with open(temp_cfg_file, 'w') as f:
            yaml.dump(current_trial_full_config, f)

        # Setup W&B for the training run
        timestamp = int(time.time())
        train_py_wandb_run_id = f"lora_trial_{trial.number}_{timestamp}"
        train_py_wandb_project = current_trial_full_config.get('wandb',{}).get('project', wandb_proj_sweep)
        train_py_wandb_entity = current_trial_full_config.get('wandb',{}).get('entity', wandb_entity_sweep)
        
        print(f"[Optuna T{trial.number}] LoRA Trial Configuration:")
        print(f"  Project: {train_py_wandb_project}")
        print(f"  Entity: {train_py_wandb_entity}")
        print(f"  Run ID: {train_py_wandb_run_id}")
        print(f"  Trial hyperparams: {trial_hyperparams}")
        
        # Setup results file path for evaluation metrics
        results_file = trial_results_dir / f"trial_{trial.number}_eval_results.json"
        
        # Import and call training function directly (no subprocess)
        print(f"\n[Optuna T{trial.number}] Running LoRA training directly...")
        print(f"[Optuna T{trial.number}] Config: {temp_cfg_file}")
        print(f"[Optuna T{trial.number}] Checkpoints: {trial_checkpoints_dir}")
        
        try:
            # Import the training function
            sys.path.insert(0, str(SCRIPT_DIR))
            from train import main as train_main
            
            # Save current working directory and change to training script directory
            # (to match the subprocess behavior for relative path resolution)
            original_cwd = os.getcwd()
            os.chdir(SCRIPT_DIR)
            
            try:
                # Call training function directly and get evaluation results
                eval_results = train_main(
                    config_path=str(temp_cfg_file),
                    wandb_run_id=train_py_wandb_run_id,
                    wandb_project=train_py_wandb_project,
                    wandb_entity=train_py_wandb_entity,
                    eval_results_file=str(results_file)
                )
            finally:
                # Always restore the original working directory
                os.chdir(original_cwd)
            
            print(f"[Optuna T{trial.number}] LoRA training completed successfully")
            
            # Calculate score from single-domain results
            if eval_results and 'domain_results' in eval_results:
                # For single-domain training, we get performance on just the target domain
                target_domain_results = list(eval_results['domain_results'].values())[0]
                
                # Simple single-domain score: SSIM - log penalty for NMSE
                ssim_score = target_domain_results['ssim']
                nmse_score = target_domain_results['nmse']
                
                # Score to maximize (higher SSIM, lower NMSE)
                lambda_log = 0.4
                composite_score = ssim_score - lambda_log * np.log10(nmse_score + 1e-6)
                
                target_domain_id = list(eval_results['domain_results'].keys())[0]
                print(f"[Optuna T{trial.number}] COMPLETED - Domain {target_domain_id}: SSIM={ssim_score:.4f}, NMSE={nmse_score:.6f}")
                print(f"[Optuna T{trial.number}] Single-domain score: {composite_score:.6f}")
                return composite_score
            else:
                error_msg = "Training completed but no evaluation results returned"
                print(f"[Optuna T{trial.number}] FATAL: {error_msg}")
                raise optuna.exceptions.TrialPruned(error_msg)
                
        except Exception as e_train:
            error_msg = f"LoRA training failed: {e_train}"
            print(f"[Optuna T{trial.number}] FATAL: {error_msg}")
            import traceback
            traceback.print_exc()
            raise optuna.exceptions.TrialPruned(error_msg)

    except optuna.exceptions.TrialPruned as e_prune:
        print(f"[Optuna T{trial.number}] Trial pruned: {e_prune}")
        raise
    except Exception as e_obj:
        print(f"[Optuna T{trial.number}] UNEXPECTED EXCEPTION: {e_obj}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        # Clean up temporary config file
        try:
            if 'temp_cfg_file' in locals() and temp_cfg_file.exists():
                temp_cfg_file.unlink()
        except Exception:
            pass  # Don't fail the trial just because we couldn't clean up

def analyze_systematic_domain_results(study: optuna.Study) -> Dict[int, int]:
    """
    Analyze results from systematic single-domain testing.
    
    Returns:
        Dictionary mapping domain_id -> best_rank
    """
    print("\n" + "="*80)
    print(" SINGLE-DOMAIN RANK ANALYSIS")
    print("="*80)
    
    # Group trials by domain
    domain_results = {}
    rank_options = [2, 4, 8, 12, 16]
    
    # Check if we have a target domain override (chunk mode)
    target_domain_override = getattr(study, '_target_domain_override', None)
    
    for trial in study.trials:
        if trial.state != optuna.trial.TrialState.COMPLETE:
            continue
            
        # Calculate domain and rank from trial number
        if target_domain_override is not None:
            # Chunk mode: all trials are for the same domain
            target_domain = target_domain_override
            rank_index = trial.number % len(rank_options)
        else:
            # Full systematic mode
            target_domain = trial.number // len(rank_options)
            rank_index = trial.number % len(rank_options)
            
        target_rank = rank_options[rank_index]
        
        if target_domain not in domain_results:
            domain_results[target_domain] = []
        
        domain_results[target_domain].append({
            'trial_number': trial.number,
            'rank': target_rank,
            'score': trial.value,
            'trial': trial
        })
    
    # Find best rank for each domain
    best_ranks = {}
    print(f"\n SINGLE-DOMAIN RESULTS:")
    
    for domain in sorted(domain_results.keys()):
        domain_trials = sorted(domain_results[domain], key=lambda x: x['score'], reverse=True)
        
        if domain_trials:
            best_trial = domain_trials[0]
            best_ranks[domain] = best_trial['rank']
            
            print(f"\n Domain {domain} (Single-Domain Training):")
            for i, trial_data in enumerate(domain_trials):
                status = "" if i == 0 else "  "
                print(f"  {status} Rank {trial_data['rank']:2d}: Score {trial_data['score']:.6f} (Trial {trial_data['trial_number']})")
    
    # Summary
    print(f"\n" + "="*80)
    print(" OPTIMAL RANKS PER DOMAIN (Single-Domain Training):")
    print("="*80)
    
    optimal_config = {}
    for domain, rank in sorted(best_ranks.items()):
        optimal_config[domain] = rank
        print(f"Domain {domain}: Rank {rank}")
    
    print(f"\n Configuration for continual learning:")
    print(f"task_lora_ranks: {optimal_config}")
    print(f"task_lora_alphas: {optimal_config}  # alpha = rank strategy")
    
    print(f"\n Next steps:")
    print(f"1. Use these optimal ranks in your continual learning training")
    print(f"2. Each domain was optimized independently")
    print(f"3. Test the combined configuration on continual learning")
    
    return optimal_config

def main_lora_optuna():
    """Main function for LoRA continual learning Optuna optimization."""
    parser = argparse.ArgumentParser(description="Optuna optimization for LoRA continual learning")
    parser.add_argument("--sweep_config", type=str, required=True, help="Path to LoRA sweep YAML config")
    parser.add_argument("--n_trials", type=int, default=20, help="Number of Optuna trials")
    parser.add_argument("--study_name", type=str, default=None, help="Optuna study name")
    parser.add_argument("--storage", type=str, default=None, help="Optuna storage URL")
    parser.add_argument("--results_root_dir", type=str, default="./main_algorithm_v2/offline/optuna_lora_results", 
                       help="Root directory for trial outputs")
    parser.add_argument("--domain_ids", type=str, nargs="+", default=["0", "1", "2", "3", "4", "5", "6", "7", "8"], 
                       help="Domain IDs to evaluate")
    parser.add_argument("--target_domain", type=int, default=None,
                       help="Target domain for chunk-based testing (0-8). If provided, only test this domain.")
    
    # W&B arguments for the study
    parser.add_argument("--wandb_study_project", type=str, default=None, help="W&B project for study logging")
    parser.add_argument("--wandb_study_entity", type=str, default=None, help="W&B entity for study logging")
    parser.add_argument("--wandb_study_tags", type=str, nargs="+", default=["lora_optuna_study"], 
                       help="Tags for study W&B run")

    args = parser.parse_args()

    results_root = Path(args.results_root_dir).resolve()
    results_root.mkdir(parents=True, exist_ok=True)

    sweep_cfg_path_obj = Path(args.sweep_config)
    study_name = args.study_name if args.study_name else f"lora_optuna_{sweep_cfg_path_obj.stem}"
    
    print("\n" + "="*80)
    print("OPTUNA OPTIMIZATION FOR LORA CONTINUAL LEARNING")
    print("="*80)
    print(f"Study: {study_name}")
    print(f"Trials: {args.n_trials}")
    print(f"Domains: {args.domain_ids}")
    print(f"Results: {results_root}")
    print("="*80 + "\n")

    # W&B study logging
    study_summary_wandb_run = None
    if args.wandb_study_project:
        try:
            study_summary_wandb_run = wandb.init(
                project=args.wandb_study_project,
                entity=args.wandb_study_entity,
                name=f"LoRAOptunaStudy_{study_name}",
                group="LoRA_Optuna_Studies",
                tags=args.wandb_study_tags + [study_name],
                config=vars(args),
                notes="LoRA continual learning hyperparameter optimization",
                job_type="lora_optuna_study"
            )
            print(f"W&B Study Log: {study_summary_wandb_run.name} (ID: {study_summary_wandb_run.id})")
        except Exception as e:
            print(f"Warning: Failed to init W&B for study: {e}")

    # Create and run Optuna study
    study = optuna.create_study(
        study_name=study_name, 
        storage=args.storage, 
        direction='maximize',  # Maximizing composite score
        load_if_exists=True
    )
    
    # Set target domain override if provided
    if args.target_domain is not None:
        study._target_domain_override = args.target_domain
        print(f" CHUNK MODE: Only testing Domain {args.target_domain}")
    else:
        print(f" FULL MODE: Testing all domains systematically")
    
    def study_callback(study_obj: optuna.Study, trial_result: optuna.trial.FrozenTrial):
        if study_summary_wandb_run:
            log_data = {
                "trial_count": trial_result.number + 1,
                f"trial_{trial_result.number}_score": trial_result.value,
                f"trial_{trial_result.number}_state": str(trial_result.state),
            }
            
            # Log best value if available
            try:
                if len(study_obj.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE])) > 0:
                    log_data["best_score_so_far"] = study_obj.best_value
            except Exception:
                pass
            
            # Log trial parameters
            for param_n, param_v in trial_result.params.items():
                log_data[f"trial_{trial_result.number}_param_{param_n.replace('.', '_')}"] = param_v
            
            study_summary_wandb_run.log(log_data)

    # Run optimization
    study.optimize(
        lambda trial_instance: objective(
            trial_instance, args.sweep_config, 
            results_root, args.domain_ids
        ),
        n_trials=args.n_trials,
        callbacks=[study_callback] if study_summary_wandb_run else None
    )

    # Results summary
    print("\n" + "="*80)
    print("LORA SYSTEMATIC DOMAIN-BY-DOMAIN OPTIMIZATION COMPLETE")
    print("="*80)
    print(f"Study: {study.study_name}")
    print(f"Completed Trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")
    
    # Analyze systematic results
    try:
        optimal_ranks = analyze_systematic_domain_results(study)
        
        # Also show traditional "best trial" info
        best_trial = study.best_trial
        if best_trial:
            # Calculate which domain/rank this best trial was testing
            rank_options = [2, 4, 8, 12, 16]
            best_domain = best_trial.number // len(rank_options)
            best_rank_idx = best_trial.number % len(rank_options)
            best_rank = rank_options[best_rank_idx]
            
            print(f"\n BEST OVERALL TRIAL:")
            print(f"Trial #{best_trial.number}: Domain {best_domain} with Rank {best_rank}")
            print(f"Score: {study.best_value:.6f}")
            
            if study_summary_wandb_run:
                study_summary_wandb_run.summary["best_trial_number"] = best_trial.number
                study_summary_wandb_run.summary["best_trial_score"] = study.best_value
                study_summary_wandb_run.summary["best_trial_domain"] = best_domain
                study_summary_wandb_run.summary["best_trial_rank"] = best_rank
                
                # Log optimal configuration
                for domain, rank in optimal_ranks.items():
                    study_summary_wandb_run.summary[f"optimal_domain_{domain}_rank"] = rank
        else:
            print("No successful trials.")
            
    except Exception as e:
        print(f"Error in systematic analysis: {e}")
        # Fallback to traditional analysis
        try:
            best_trial = study.best_trial
            if best_trial:
                print(f"Best Trial: #{best_trial.number}")
                print(f"Best Score: {study.best_value:.6f}")
        except ValueError:
            print("No successful trials.")

    if study_summary_wandb_run:
        study_summary_wandb_run.finish()
    
    print("="*80)

if __name__ == "__main__":
    main_lora_optuna()