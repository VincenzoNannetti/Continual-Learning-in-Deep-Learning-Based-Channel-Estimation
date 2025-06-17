import optuna
import yaml
import subprocess
import sys
import os
import argparse
import wandb
import threading
import queue
import numpy as np  # Added for composite scoring
from pathlib import Path

# Ensure the script can find standard_training_2 modules
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))
if str(SCRIPT_DIR.parent) not in sys.path:
    sys.path.append(str(SCRIPT_DIR.parent))

def calculate_composite_score(wandb_run, trial_number):
    """
    Calculate a composite score that:
    1. Maximizes SSIM (0-1, higher better)
    2. Minimizes NMSE (>0, lower better) with log-scaled normalization
    3. Penalizes overfitting (train/val gap) with exponential curve
    4. Prefers early convergence with exponential curve
    
    Returns a score to be MAXIMIZED by Optuna.
    """
    
    # Fetch core metrics
    eval_ssim = wandb_run.summary.get("eval_ssim", None)
    eval_nmse = wandb_run.summary.get("eval_nmse", None)
    
    # Fetch overfitting indicators
    final_train_loss = wandb_run.summary.get("final_train_loss", None)
    final_val_loss = wandb_run.summary.get("final_val_loss", None)
    best_epoch = wandb_run.summary.get("best_epoch", 50)
    
    if eval_ssim is None or eval_nmse is None:
        raise optuna.exceptions.TrialPruned("Missing core evaluation metrics (SSIM/NMSE)")
    
    print(f"[Optuna T{trial_number}] Core Metrics:")
    print(f"  SSIM: {eval_ssim:.6f} (target: maximize)")
    print(f"  NMSE: {eval_nmse:.6f} (target: minimize)")
    
    # === COMPOSITE SCORE CALCULATION ===
    
    # 1. SSIM component (0-1, higher better) - use directly
    ssim_score = float(eval_ssim)
    
    # 2. NMSE component with log-scaled normalization (guards against extremely low NMSE)
    nmse_raw = float(eval_nmse)
    # Log-scaled transform to avoid artificially perfect scores for ultra-low NMSE
    nmse_score = 1.0 - np.clip(np.log10(nmse_raw + 1e-6) / np.log10(0.5 + 1e-6), 0.0, 1.0)
    
    # 3. Base composite (weighted combination)
    alpha = 0.6  # Weight for SSIM 
    beta = 0.4   # Weight for NMSE
    base_score = alpha * ssim_score + beta * nmse_score
    
    print(f"  Base Score: {base_score:.6f}")
    print(f"    SSIM contribution: {alpha} × {ssim_score:.4f} = {alpha * ssim_score:.4f}")
    print(f"    NMSE contribution: {beta} × {nmse_score:.4f} = {beta * nmse_score:.4f}")
    
    # === EXPONENTIAL OVERFITTING PENALTY ===
    overfitting_penalty = 0.0
    if final_train_loss is not None and final_val_loss is not None:
        # Calculate relative overfitting gap
        train_val_gap = abs(final_val_loss - final_train_loss)
        avg_loss = (final_train_loss + final_val_loss) / 2
        relative_gap = train_val_gap / (avg_loss + 1e-8)
        
        # Exponential penalty: small gaps slide by, large gaps penalized rapidly
        overfitting_penalty = 0.1 * (1 - np.exp(-3 * relative_gap))  # Max ~0.1
        
        print(f"  Overfitting Analysis:")
        print(f"    Train Loss: {final_train_loss:.6f}")
        print(f"    Val Loss: {final_val_loss:.6f}")
        print(f"    Relative Gap: {relative_gap:.4f}")
        print(f"    Exponential Penalty: {overfitting_penalty:.6f}")
    
    # === EXPONENTIAL CONVERGENCE PENALTY ===
    # Diminishing penalty curve after epoch 30, maxing around 0.1
    convergence_penalty = 0.1 * (1 - np.exp(-0.03 * max(0, best_epoch - 30)))
    
    print(f"  Convergence: Best epoch {best_epoch}, Exp Penalty: {convergence_penalty:.6f}")
    
    # === FINAL SCORE ===
    final_score = base_score - overfitting_penalty - convergence_penalty
    
    print(f"  FINAL COMPOSITE SCORE: {final_score:.6f}")
    print(f"    = {base_score:.4f} - {overfitting_penalty:.4f} - {convergence_penalty:.4f}")
    
    return final_score

def parse_sweep_config(sweep_config_path_str: str):
    """Parses the W&B sweep YAML to extract parameters for Optuna."""
    sweep_config_path = Path(sweep_config_path_str)
    with open(sweep_config_path, 'r') as f:
        sweep_config = yaml.safe_load(f)
    
    if 'parameters' not in sweep_config:
        raise ValueError(f"Key 'parameters' not found in sweep config: {sweep_config_path}")

    base_train_config_rel_path = sweep_config['parameters'].get('train_config_path', {}).get('value')
    if not base_train_config_rel_path:
        raise ValueError(f"Base training config path key ('train_config_path.value') not found in sweep parameters: {sweep_config_path}")

    metric_info = sweep_config.get('metric')
    if not metric_info or 'name' not in metric_info or 'goal' not in metric_info:
        raise ValueError(f"Metric name and goal not found in sweep config: {sweep_config_path}")

    metric_to_optimize = metric_info['name']
    optimization_goal = metric_info['goal']
    
    wandb_project_sweep = sweep_config['parameters'].get('wandb.project', {}).get('value')
    wandb_entity_sweep = sweep_config['parameters'].get('wandb.entity', {}).get('value')

    return sweep_config['parameters'], base_train_config_rel_path, metric_to_optimize, optimization_goal, wandb_project_sweep, wandb_entity_sweep

def suggest_params_from_sweep(trial: optuna.Trial, sweep_params_space: dict):
    """Suggests hyperparameters based on the W&B sweep_params_space structure for an Optuna trial."""
    suggested_params = {}
    for param_name, param_details in sweep_params_space.items():
        # Skip metadata parameters that aren't actual model/training parameters
        if param_name in ['train_config_path', 'wandb.project', 'wandb.entity', 'optuna_runner_wandb.project', 'optuna_runner_wandb.entity']: 
            continue
            
        if 'value' in param_details:
            # Fixed values should be included to override base config
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
                    if dist == 'uniform':
                         suggested_params[param_name] = trial.suggest_float(param_name, min_val_f, max_val_f, log=False)
                    else:
                         suggested_params[param_name] = trial.suggest_float(param_name, min_val_f, max_val_f, log=True)
                elif dist == 'int_uniform':
                    min_val_i = int(float(min_val_orig))
                    max_val_i = int(float(max_val_orig))
                    suggested_params[param_name] = trial.suggest_int(param_name, min_val_i, max_val_i)
                elif dist == 'q_uniform':
                    min_q, max_q, q_val = float(min_val_orig), float(max_val_orig), float(param_details['q'])
                    suggested_params[param_name] = trial.suggest_float(param_name, min_q, max_q, step=q_val)
                else:
                    print(f"Warning: Unsupported distribution '{dist}' for param '{param_name}'. Skipping.")
                    continue
            except (ValueError, TypeError) as e:
                print(f"Warning: Could not convert min/max/q for param '{param_name}' with dist '{dist}'. Error: {e}. Skipping.")
                continue
        else:
            print(f"Warning: Param '{param_name}' in sweep has no 'value', 'values', or 'distribution'. Skipping suggestion.")
    return suggested_params

def update_config_with_trial_params(base_config: dict, trial_params: dict):
    """Updates the base config with parameters suggested by Optuna trial."""
    updated_config = yaml.safe_load(yaml.safe_dump(base_config))
    for key, value in trial_params.items():
        parts = key.split('.')
        d = updated_config
        for part in parts[:-1]:
            if part not in d or not isinstance(d[part], dict):
                d[part] = {}
            d = d[part]
        d[parts[-1]] = value
    return updated_config

def stream_subprocess_output(process: subprocess.Popen, trial_number: int, log_prefix: str = "[Train]"):
    """Stream subprocess output in real-time with timestamps and prefix."""
    def read_stream(stream, stream_name: str, q_stream: queue.Queue):
        try:
            for line_bytes in iter(stream.readline, b''):
                if not line_bytes:
                    break
                q_stream.put((stream_name, line_bytes.decode('utf-8', errors='replace').rstrip()))
        except Exception as e_stream:
            q_stream.put((stream_name, f"Error reading stream {stream_name}: {e_stream}"))
        finally:
            stream.close()

    output_q = queue.Queue()
    stdout_thread = threading.Thread(target=read_stream, args=(process.stdout, 'STDOUT', output_q))
    stderr_thread = threading.Thread(target=read_stream, args=(process.stderr, 'STDERR', output_q))
    stdout_thread.daemon = True
    stderr_thread.daemon = True
    stdout_thread.start()
    stderr_thread.start()
    
    print(f"{log_prefix} Trial {trial_number}: Subprocess PID {process.pid} started. Streaming output...")
    
    all_output_lines = []
    try:
        while process.poll() is None:
            try:
                stream_name, line = output_q.get(timeout=0.5) 
                print(f"{log_prefix} Trial {trial_number} [{stream_name}]: {line}")
                all_output_lines.append(f"[{stream_name}]: {line}")
            except queue.Empty:
                continue
        
        while not output_q.empty():
            try:
                stream_name, line = output_q.get_nowait()
                print(f"{log_prefix} Trial {trial_number} [{stream_name}]: {line}")
                all_output_lines.append(f"[{stream_name}]: {line}")
            except queue.Empty:
                break
    finally:
        stdout_thread.join(timeout=5)
        stderr_thread.join(timeout=5)

    return_code = process.returncode
    print(f"{log_prefix} Trial {trial_number}: Subprocess finished with return code {return_code}.")
    return return_code, "\n".join(all_output_lines)

def objective(trial: optuna.Trial, sweep_config_yaml_path_str: str, python_exe: str, results_root_dir: Path, all_data: bool = False):
    """Optuna objective function for a single trial."""
    current_trial_full_config = {}

    try:
        sweep_params_space, base_train_cfg_rel_path, metric_name, metric_goal, \
            wandb_proj_sweep, wandb_entity_sweep = parse_sweep_config(sweep_config_yaml_path_str)

        trial_hyperparams = suggest_params_from_sweep(trial, sweep_params_space)

        actual_base_train_cfg_path = (Path(sweep_config_yaml_path_str).parent / base_train_cfg_rel_path).resolve()
        if not actual_base_train_cfg_path.exists():
            raise optuna.exceptions.TrialPruned(f"Base train config not found: {actual_base_train_cfg_path}")
        
        with open(actual_base_train_cfg_path, 'r') as f:
            base_cfg_content = yaml.safe_load(f)
        current_trial_full_config = update_config_with_trial_params(base_cfg_content, trial_hyperparams)
        
        # --- Determine model name for directory path ---
        model_name_for_path = "unknown_model"
        try:
            model_name_for_path = current_trial_full_config.get('model', {}).get('name', 'unknown_model')
            # Sanitize model_name for path (e.g., replace slashes or invalid chars if any)
            model_name_for_path = model_name_for_path.replace('/', '_').replace('\\', '_')
        except Exception:
            pass # Keep default 'unknown_model'

        trial_results_dir = results_root_dir / f"trial_{trial.number}_{model_name_for_path}"
        trial_checkpoints_dir = trial_results_dir / "checkpoints"
        trial_checkpoints_dir.mkdir(parents=True, exist_ok=True)
        current_trial_full_config.setdefault('logging', {})['checkpoint_dir'] = str(trial_checkpoints_dir)

        temp_cfg_file = trial_results_dir / f"trial_{trial.number}_train_config.yaml"
        with open(temp_cfg_file, 'w') as f:
            yaml.dump(current_trial_full_config, f)

        # W&B details for the subprocess (train.py) - this will be the ONLY run per trial
        import time
        timestamp = int(time.time())  # Unix timestamp for uniqueness
        train_py_wandb_run_id = f"trial_{trial.number}_{timestamp}"
        train_py_wandb_project = current_trial_full_config.get('wandb',{}).get('project', wandb_proj_sweep)
        train_py_wandb_entity = current_trial_full_config.get('wandb',{}).get('entity', wandb_entity_sweep)
        train_py_wandb_log_freq = current_trial_full_config.get('wandb',{}).get('log_freq_train', 1)
        
        # Debug: Print W&B configuration being used
        print(f"[Optuna T{trial.number}] W&B config for train.py:")
        print(f"  Project: {train_py_wandb_project}")
        print(f"  Entity: {train_py_wandb_entity}")
        print(f"  Run ID: {train_py_wandb_run_id}")
        print(f"  Log Freq: {train_py_wandb_log_freq}")
        
        cmd = [
            python_exe,
            str(SCRIPT_DIR / "train.py"),
            "--config_path", str(temp_cfg_file),
            "--wandb_run_id", train_py_wandb_run_id,
        ]
        if train_py_wandb_project:
            cmd.extend(["--wandb_project", train_py_wandb_project])
        if train_py_wandb_entity:
            cmd.extend(["--wandb_entity", train_py_wandb_entity])
        cmd.extend(["--wandb_log_freq", str(train_py_wandb_log_freq)])
        if all_data:
            cmd.append("--all_data")
        
        print(f"\n[Optuna T{trial.number}] Executing: {' '.join(cmd)}")
        print(f"[Optuna T{trial.number}] Trial hyperparams: {trial_hyperparams}")
        print(f"[Optuna T{trial.number}] Full config at: {temp_cfg_file}")
        print(f"[Optuna T{trial.number}] Checkpoints in: {trial_checkpoints_dir}")

        # Set environment to ensure subprocess can find modules and handle Unicode properly
        env = os.environ.copy()
        project_root = str(SCRIPT_DIR.parent)
        if 'PYTHONPATH' in env:
            env['PYTHONPATH'] = f"{project_root}{os.pathsep}{env['PYTHONPATH']}"
        else:
            env['PYTHONPATH'] = project_root
        
        # Set encoding environment variables to handle Unicode characters properly
        env['PYTHONIOENCODING'] = 'utf-8'
        env['PYTHONLEGACYWINDOWSFSENCODING'] = '0'  # Disable legacy encoding on Windows

        subprocess_cwd = SCRIPT_DIR.parent

        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                                 cwd=subprocess_cwd, env=env)
        return_code, train_log = stream_subprocess_output(process, trial.number, log_prefix=f"[Optuna T{trial.number} Train]")
        
        trial.set_user_attr("training_log_tail", train_log[-2000:])

        if return_code != 0:
            err_msg = f"Training script failed (return code {return_code})."
            print(f"[Optuna T{trial.number}] {err_msg}")
            raise optuna.exceptions.TrialPruned(err_msg)
        
        print(f"[Optuna T{trial.number}] Training completed successfully")

        # Fetch evaluation metrics from the training W&B run using API
        if train_py_wandb_project and train_py_wandb_run_id:
            try:
                print(f"[Optuna T{trial.number}] Fetching evaluation metrics from W&B run: {train_py_wandb_run_id}")
                
                # Use W&B API to fetch metrics from the training run
                api = wandb.Api(timeout=30)
                
                # Construct run path
                run_path = f"{train_py_wandb_project}/{train_py_wandb_run_id}"
                if train_py_wandb_entity:
                    run_path = f"{train_py_wandb_entity}/{run_path}"
                
                print(f"[Optuna T{trial.number}] Fetching from run path: {run_path}")
                
                # Retry logic for fetching the run
                retries = 5
                base_delay = 3
                wandb_run = None
                
                for i in range(retries):
                    try:
                        delay = base_delay * (2 ** min(i, 2))  # Exponential backoff
                        if i > 0:
                            print(f"[Optuna T{trial.number}] Waiting {delay} seconds before retry {i+1}/{retries}...")
                            time.sleep(delay)
                        
                        wandb_run = api.run(run_path)
                        print(f"[Optuna T{trial.number}] Successfully fetched W&B run")
                        break
                        
                    except Exception as e:
                        print(f"[Optuna T{trial.number}] Error fetching W&B run (Attempt {i+1}/{retries}): {e}")
                        if i == retries - 1:
                            raise
                
                if wandb_run is None:
                    raise Exception("Failed to fetch W&B run after all retries")
                
                # Get the evaluation metrics from the run summary
                available_metrics = list(wandb_run.summary.keys())
                print(f"[Optuna T{trial.number}] Available metrics in W&B run: {available_metrics}")
                
                # === COMPOSITE SCORING APPROACH ===
                # Use composite scoring that combines SSIM + NMSE with overfitting penalties
                try:
                    composite_score = calculate_composite_score(wandb_run, trial.number)
                    print(f"[Optuna T{trial.number}] COMPLETED with composite score: {composite_score:.6f}")
                    return composite_score
                    
                except Exception as e_composite:
                    print(f"[Optuna T{trial.number}] ERROR in composite scoring: {e_composite}")
                    # Fallback to original single metric approach if composite fails
                    print(f"[Optuna T{trial.number}] Falling back to single metric approach...")
                    
                    # Try to get evaluation metrics - first try the exact metric name, then try common patterns
                    if metric_name in wandb_run.summary:
                        target_metric_val = wandb_run.summary[metric_name]
                        print(f"[Optuna T{trial.number}] Found fallback metric '{metric_name}': {target_metric_val}")
                        return float(target_metric_val)
                    else:
                        # Try common evaluation metric patterns
                        possible_names = [metric_name, f"eval_{metric_name}", f"test_{metric_name}", f"final_{metric_name}"]
                        target_metric_val = None
                        
                        for possible_name in possible_names:
                            if possible_name in wandb_run.summary:
                                target_metric_val = wandb_run.summary[possible_name]
                                print(f"[Optuna T{trial.number}] Found fallback metric as '{possible_name}': {target_metric_val}")
                                return float(target_metric_val)
                        
                        # If we reach here, no metrics were found
                        err_msg = f"Both composite scoring and fallback failed. Target metric '{metric_name}' (or variants) not found in W&B summary. Available: {available_metrics}"
                        print(f"[Optuna T{trial.number}] ERROR: {err_msg}")
                        raise optuna.exceptions.TrialPruned(err_msg)
                
            except Exception as e_wandb_fetch:
                print(f"[Optuna T{trial.number}] ERROR: Could not fetch metrics from W&B: {e_wandb_fetch}")
                err_msg = f"Failed to fetch target metric '{metric_name}' from W&B run"
                print(f"[Optuna T{trial.number}] {err_msg}")
                raise optuna.exceptions.TrialPruned(err_msg)
        else:
            # No W&B configured - this is an error since we need metrics from somewhere
            err_msg = f"No W&B configuration provided, cannot fetch evaluation metrics"
            print(f"[Optuna T{trial.number}] ERROR: {err_msg}")
            raise optuna.exceptions.TrialPruned(err_msg)

    except optuna.exceptions.TrialPruned as e_prune:
        print(f"[Optuna T{trial.number}] Trial pruned: {e_prune}")
        raise
    except Exception as e_obj:
        print(f"[Optuna T{trial.number}] UNEXPECTED EXCEPTION in objective: {e_obj}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        if 'temp_cfg_file' in locals() and temp_cfg_file.exists():
            temp_cfg_file.unlink()

def main_optuna_runner():
    """Main function for running Optuna hyperparameter optimization."""
    parser = argparse.ArgumentParser(description="Optuna Hyperparameter Tuning for standard_training_2 models")
    parser.add_argument("--sweep_config", type=str, required=True, help="Path to W&B-style sweep YAML for Optuna.")
    parser.add_argument("--n_trials", type=int, default=10, help="Number of Optuna trials.")
    parser.add_argument("--study_name", type=str, default=None, help="Optuna study name. Defaults based on sweep config.")
    parser.add_argument("--storage", type=str, default=None, help="Optuna storage URL. In-memory if None.")
    parser.add_argument("--python_executable", type=str, default=sys.executable, help="Python executable for train.py.")
    parser.add_argument("--results_root_dir", type=str, default="./standard_training_2/optuna_study_results", help="Root directory for all trial outputs.")
    
    # W&B arguments for the main Optuna study logger run
    parser.add_argument("--wandb_study_project", type=str, default=None, help="W&B Project for the main Optuna study log run.")
    parser.add_argument("--wandb_study_entity", type=str, default=None, help="W&B Entity for the main Optuna study log run.")
    parser.add_argument("--wandb_study_tags", type=str, nargs="+", default=["optuna_study_log"], help="Tags for Optuna study's W&B run.")
    parser.add_argument("--all_data", action="store_true", help="Pass --all_data flag to training scripts to use combined datasets.")

    args = parser.parse_args()

    results_root = Path(args.results_root_dir).resolve()
    results_root.mkdir(parents=True, exist_ok=True)

    sweep_cfg_path_obj = Path(args.sweep_config)
    study_name = args.study_name if args.study_name else f"st2_optuna_{sweep_cfg_path_obj.stem}"
    
    _, _, metric_to_opt, metric_goal_str, _, _ = parse_sweep_config(args.sweep_config)
    optuna_direction = 'maximize' if metric_goal_str.lower() == 'maximize' else 'minimize'

    print("\n" + "="*80 + "\nOPTUNA HYPERPARAMETER OPTIMIZATION for Standard Training 2.0" + "\n" + "="*80)
    print(f"Optuna Study Name: {study_name}, Direction: {optuna_direction} '{metric_to_opt}'")
    print(f"Trials: {args.n_trials}, Storage: {args.storage if args.storage else 'In-memory'}")
    print(f"Python for train.py: {args.python_executable}, Results Root: {results_root}")
    
    # W&B Run for the entire Optuna Study
    study_summary_wandb_run = None
    if args.wandb_study_project:
        try:
            study_summary_wandb_run = wandb.init(
                project=args.wandb_study_project,
                entity=args.wandb_study_entity,
                name=f"OptunaStudyLog_{study_name}",
                group="Optuna_Studies_Logs",
                tags=args.wandb_study_tags + [study_name, sweep_cfg_path_obj.name],
                config=vars(args),
                notes="Main logger for an Optuna study.",
                job_type="optuna_study_manager"
            )
            print(f"W&B Main Study Log run: {study_summary_wandb_run.name} (ID: {study_summary_wandb_run.id})")
            study_summary_wandb_run.config.update({
                "optuna_study_name_actual": study_name, 
                "metric_to_optimize": metric_to_opt, 
                "optuna_direction": optuna_direction
            })
        except Exception as e_study_wandb:
            print(f"Warning: Failed to init W&B for Optuna Study Log: {e_study_wandb}")
            study_summary_wandb_run = None
    print("="*80 + "\n")

    study = optuna.create_study(study_name=study_name, storage=args.storage, direction=optuna_direction, load_if_exists=True)
    
    def study_callback(study_obj: optuna.Study, trial_result: optuna.trial.FrozenTrial):
        if study_summary_wandb_run:
            log_data = {
                "optuna_study_trial_count": trial_result.number + 1,
                f"trial_{trial_result.number}_value": trial_result.value,
                f"trial_{trial_result.number}_state": str(trial_result.state),
            }
            
            # Only log best value if there are completed trials
            try:
                if len(study_obj.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE])) > 0:
                    log_data["optuna_study_best_value_so_far"] = study_obj.best_value
            except Exception:
                # Skip logging best value if it's not available
                pass
            
            for param_n, param_v in trial_result.params.items():
                 log_data[f"trial_{trial_result.number}_param_{param_n.replace('.', '_')}"] = param_v
            
            for attr_n, attr_v in trial_result.user_attrs.items():
                log_data[f"trial_{trial_result.number}_attr_{attr_n.replace('.', '_')}"] = attr_v

            study_summary_wandb_run.log(log_data)

    study.optimize(
        lambda trial_instance: objective(trial_instance, args.sweep_config, args.python_executable, results_root, args.all_data),
        n_trials=args.n_trials,
        callbacks=[study_callback] if study_summary_wandb_run else None
    )

    print("\n" + "="*80 + "\nOPTUNA OPTIMIZATION COMPLETE" + "\n" + "="*80)
    print(f"Study: {study.study_name}")
    
    # Safely check for best trial
    try:
        best_trial = study.best_trial
        if best_trial:
            print(f"Best Trial: #{best_trial.number}, Value ({metric_to_opt}): {study.best_value}")
            print("Best Params:")
            for k, v_ in best_trial.params.items():
                print(f"  {k}: {v_}")
            if study_summary_wandb_run:
                study_summary_wandb_run.summary["best_trial_number"] = best_trial.number
                study_summary_wandb_run.summary["best_trial_value"] = study.best_value
                study_summary_wandb_run.summary.update({f"best_param_{k.replace('.', '_')}": v_ for k,v_ in best_trial.params.items()})
        else:
            print("No trials completed successfully.")
            if study_summary_wandb_run:
                study_summary_wandb_run.summary["status"] = "No_successful_trials"
    except ValueError as e:
        print("No trials completed successfully.")
        if study_summary_wandb_run:
            study_summary_wandb_run.summary["status"] = "No_successful_trials"

    if study_summary_wandb_run:
        study_summary_wandb_run.finish()
        print("W&B Main Study Log run finished.")
    print("="*80)

if __name__ == "__main__":
    main_optuna_runner() 