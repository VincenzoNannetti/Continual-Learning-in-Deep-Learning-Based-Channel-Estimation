# run_continual.ps1 - PowerShell script to run continual learning training/evaluation workflows
# Author: Vincenzo Nannetti

# Example usage:
# Train and evaluate (default mode)
# .\scripts\run_continual.ps1 -Config configs/supermask_config.yaml

# Train only with dataset A
# .\scripts\run_continual.ps1 -Config configs/supermask_config.yaml -Mode train -UseDataset a -Suffix _dataset_a

# Evaluate only with a specific checkpoint dir
# .\scripts\run_continual.ps1 -Config configs/supermask_config.yaml -Mode eval -CheckpointDir checkpoints/experiment-timestamp

# Training with custom task sequence
# .\scripts\run_continual.ps1 -Config configs/supermask_config.yaml -Sequence "0,1,2"

param (
    [Parameter(Mandatory=$true, HelpMessage="Path to configuration file")]
    [string]$Config,

    [Parameter(HelpMessage="Workflow mode: train-eval, train, eval")]
    [ValidateSet("train-eval", "train", "eval")]
    [string]$Mode = "train-eval",

    [Parameter(HelpMessage="Path to checkpoint directory (for eval mode)")]
    [string]$CheckpointDir = "",

    [Parameter(HelpMessage="Experiment name suffix")]
    [string]$Suffix = "",

    [Parameter(HelpMessage="Which dataset to use (a or b)")]
    [ValidateSet("a", "b", "")]
    [string]$UseDataset = "",

    [Parameter(HelpMessage="Supermask task sequence, comma-separated")]
    [string]$Sequence = "",

    [Parameter(HelpMessage="Task ID for evaluation (if omitted, evaluates all tasks)")]
    [int]$TaskId = -1,
    
    [Parameter(HelpMessage="Disable Weights & Biases logging")]
    [switch]$NoWandb,

    [Parameter(HelpMessage="Path to Python executable")]
    [string]$Python = "python"
)

# Show usage information
function Show-Usage {
    Write-Host "Usage: .\scripts\run_continual.ps1 -Config <config_file> [OPTIONS]" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Continual Learning Training/Evaluation Workflow" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Options:" -ForegroundColor Yellow
    Write-Host "  -Config <file>         Path to configuration file (required)"
    Write-Host "  -Mode <mode>           Workflow mode: train-eval, train, eval (default: train-eval)"
    Write-Host "  -CheckpointDir <path>  Path to checkpoint directory (for eval mode)"
    Write-Host "  -Suffix <suffix>       Experiment name suffix"
    Write-Host "  -UseDataset {a,b}      Which dataset to use (a or b)"
    Write-Host "  -Sequence <seq>        Supermask task sequence, comma-separated (e.g., '0,1,2' or 'a,b,c')"
    Write-Host "  -TaskId <id>           Task ID for evaluation (if omitted, evaluates all tasks)"
    Write-Host "  -NoWandb               Disable Weights & Biases logging"
    Write-Host "  -Python <path>         Path to Python executable (default: python)"
    Write-Host ""
    Write-Host "Examples:" -ForegroundColor Green
    Write-Host "  # Train and evaluate" -ForegroundColor Gray
    Write-Host "  .\scripts\run_continual.ps1 -Config configs/supermask_config.yaml" -ForegroundColor Gray
    Write-Host ""
    Write-Host "  # Train only with dataset A and specific sequence" -ForegroundColor Gray
    Write-Host "  .\scripts\run_continual.ps1 -Config configs/supermask_config.yaml -Mode train -UseDataset a -Sequence '0,2,1'" -ForegroundColor Gray
    Write-Host ""
    Write-Host "  # Evaluate only task 0 with a specific checkpoint directory" -ForegroundColor Gray
    Write-Host "  .\scripts\run_continual.ps1 -Config configs/supermask_config.yaml -Mode eval -CheckpointDir results/checkpoints/my_run -TaskId 0" -ForegroundColor Gray
}

# Validate parameters
if (-not (Test-Path $Config -PathType Leaf)) {
    Write-Host "Error: Config file not found: $Config" -ForegroundColor Red
    Show-Usage
    exit 1
}

if ($Mode -eq "eval" -and $CheckpointDir -ne "" -and -not (Test-Path $CheckpointDir -PathType Container)) {
    Write-Host "Error: Checkpoint directory not found: $CheckpointDir" -ForegroundColor Red
    exit 1
}

# Build command arguments
$CommonArgs = "--config `"$Config`""

if ($Suffix -ne "") {
    $CommonArgs += " --experiment_suffix `"$Suffix`""
}

if ($NoWandb) {
    $CommonArgs += " --no_wandb"
}

$TrainArgs = $CommonArgs
if ($UseDataset -ne "") {
    $TrainArgs += " --dataset_to_use $UseDataset"
}
if ($Sequence -ne "") {
    $TrainArgs += " --supermask_sequence `"$Sequence`""
}

$EvalArgs = $CommonArgs
if ($CheckpointDir -ne "") {
    $EvalArgs += " --checkpoint_dir `"$CheckpointDir`""
}
if ($TaskId -ge 0) {
    $EvalArgs += " --task_id $TaskId"
}

# Print workflow information
Write-Host "========== Continual Learning Workflow ==========" -ForegroundColor Cyan
Write-Host "Mode: $Mode" -ForegroundColor Yellow
Write-Host "Config: $Config" -ForegroundColor Yellow
if ($Suffix -ne "") {
    Write-Host "Experiment suffix: $Suffix" -ForegroundColor Yellow
}
if ($UseDataset -ne "") {
    Write-Host "Using dataset: $UseDataset" -ForegroundColor Yellow
}
if ($Sequence -ne "") {
    Write-Host "Task sequence: $Sequence" -ForegroundColor Yellow
}
if ($Mode -eq "eval" -and $TaskId -ge 0) {
    Write-Host "Evaluating task ID: $TaskId" -ForegroundColor Yellow
}
Write-Host "========================================" -ForegroundColor Cyan

# Execute the workflow based on mode
try {
    switch ($Mode) {
        "train-eval" {
            Write-Host "Running training with evaluation..." -ForegroundColor Green
            $TrainCmd = "$Python -m continual_learning.train $TrainArgs"
            Write-Host "Running: $TrainCmd" -ForegroundColor Gray
            Invoke-Expression $TrainCmd
            
            if ($LASTEXITCODE -ne 0) {
                Write-Host "Error: Training with evaluation failed with exit code $LASTEXITCODE" -ForegroundColor Red
                exit $LASTEXITCODE
            }
        }
        "train" {
            Write-Host "Running training only..." -ForegroundColor Green
            $TrainCmd = "$Python -m continual_learning.train $TrainArgs --no_eval"
            Write-Host "Running: $TrainCmd" -ForegroundColor Gray
            Invoke-Expression $TrainCmd
            
            if ($LASTEXITCODE -ne 0) {
                Write-Host "Error: Training failed with exit code $LASTEXITCODE" -ForegroundColor Red
                exit $LASTEXITCODE
            }
        }
        "eval" {
            Write-Host "Running evaluation only..." -ForegroundColor Green
            $EvalCmd = "$Python -m continual_learning.evaluate $EvalArgs"
            Write-Host "Running: $EvalCmd" -ForegroundColor Gray
            Invoke-Expression $EvalCmd
            
            if ($LASTEXITCODE -ne 0) {
                Write-Host "Error: Evaluation failed with exit code $LASTEXITCODE" -ForegroundColor Red
                exit $LASTEXITCODE
            }
        }
    }
    
    Write-Host "Workflow completed successfully!" -ForegroundColor Green
}
catch {
    Write-Host "Error: $_" -ForegroundColor Red
    exit 1
}
