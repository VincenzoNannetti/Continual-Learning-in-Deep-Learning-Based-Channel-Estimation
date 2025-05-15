# run_standard.ps1 - PowerShell script to run standard ML training/evaluation workflows
# Author: Vincenzo Nannetti

# Example usage:
# Train and evaluate (default mode)
# .\scripts\run_standard.ps1 -Config configs/my_config.yaml

# Train only with dataset A
# .\scripts\run_standard.ps1 -Config configs/my_config.yaml -Mode train -UseDataset a -Suffix _dataset_a

# Evaluate only with a specific checkpoint
# .\scripts\run_standard.ps1 -Config configs/my_config.yaml -Mode eval -Checkpoint checkpoints/best_model.pth

# Training with W&B disabled
# .\scripts\run_standard.ps1 -Config configs/my_config.yaml -NoWandb

param (
    [Parameter(Mandatory=$true, HelpMessage="Path to configuration file")]
    [string]$Config,

    [Parameter(HelpMessage="Workflow mode: train-eval, train, eval")]
    [ValidateSet("train-eval", "train", "eval")]
    [string]$Mode = "train-eval",

    [Parameter(HelpMessage="Path to model checkpoint (for eval mode)")]
    [string]$Checkpoint = "",

    [Parameter(HelpMessage="Path to data configuration override (for eval mode)")]
    [string]$DataConfig = "",

    [Parameter(HelpMessage="Experiment name suffix")]
    [string]$Suffix = "",

    [Parameter(HelpMessage="Which dataset to use (a or b)")]
    [ValidateSet("a", "b", "")]
    [string]$UseDataset = "",

    [Parameter(HelpMessage="Disable Weights & Biases logging")]
    [switch]$NoWandb,

    [Parameter(HelpMessage="Path to Python executable")]
    [string]$Python = "python"
)

# Show usage information
function Show-Usage {
    Write-Host "Usage: .\scripts\run_standard.ps1 -Config <config_file> [OPTIONS]" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Standard ML Training/Evaluation Workflow" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Options:" -ForegroundColor Yellow
    Write-Host "  -Config <file>         Path to configuration file (required)"
    Write-Host "  -Mode <mode>           Workflow mode: train-eval, train, eval (default: train-eval)"
    Write-Host "  -Checkpoint <path>     Path to model checkpoint (for eval mode)"
    Write-Host "  -DataConfig <file>     Path to data configuration override (for eval mode)"
    Write-Host "  -Suffix <suffix>       Experiment name suffix"
    Write-Host "  -UseDataset {a,b}      Which dataset to use (a or b)"
    Write-Host "  -NoWandb               Disable Weights & Biases logging"
    Write-Host "  -Python <path>         Path to Python executable (default: python)"
    Write-Host ""
    Write-Host "Examples:" -ForegroundColor Green
    Write-Host "  # Train and evaluate" -ForegroundColor Gray
    Write-Host "  .\scripts\run_standard.ps1 -Config configs/my_config.yaml" -ForegroundColor Gray
    Write-Host ""
    Write-Host "  # Train only with dataset A" -ForegroundColor Gray
    Write-Host "  .\scripts\run_standard.ps1 -Config configs/my_config.yaml -Mode train -UseDataset a -Suffix _dataset_a" -ForegroundColor Gray
    Write-Host ""
    Write-Host "  # Evaluate only with a specific checkpoint" -ForegroundColor Gray
    Write-Host "  .\scripts\run_standard.ps1 -Config configs/my_config.yaml -Mode eval -Checkpoint checkpoints/best_model.pth" -ForegroundColor Gray
}

# Validate parameters
if (-not (Test-Path $Config -PathType Leaf)) {
    Write-Host "Error: Config file not found: $Config" -ForegroundColor Red
    Show-Usage
    exit 1
}

if ($Mode -eq "eval" -and $Checkpoint -ne "" -and -not (Test-Path $Checkpoint -PathType Leaf)) {
    Write-Host "Error: Checkpoint file not found: $Checkpoint" -ForegroundColor Red
    exit 1
}

if ($DataConfig -ne "" -and -not (Test-Path $DataConfig -PathType Leaf)) {
    Write-Host "Error: Data config file not found: $DataConfig" -ForegroundColor Red
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

$EvalArgs = $CommonArgs
if ($Checkpoint -ne "") {
    $EvalArgs += " --checkpoint `"$Checkpoint`""
}
if ($DataConfig -ne "") {
    $EvalArgs += " --data_config `"$DataConfig`""
}

# Print workflow information
Write-Host "========== Standard ML Workflow ==========" -ForegroundColor Cyan
Write-Host "Mode: $Mode" -ForegroundColor Yellow
Write-Host "Config: $Config" -ForegroundColor Yellow
if ($Suffix -ne "") {
    Write-Host "Experiment suffix: $Suffix" -ForegroundColor Yellow
}
if ($UseDataset -ne "") {
    Write-Host "Using dataset: $UseDataset" -ForegroundColor Yellow
}
Write-Host "========================================" -ForegroundColor Cyan

# Execute the workflow based on mode
try {
    switch ($Mode) {
        "train-eval" {
            Write-Host "Running training..." -ForegroundColor Green
            $TrainCmd = "$Python -m standard_training.train $TrainArgs"
            Write-Host "Running: $TrainCmd" -ForegroundColor Gray
            Invoke-Expression $TrainCmd
            
            if ($LASTEXITCODE -ne 0) {
                Write-Host "Error: Training failed with exit code $LASTEXITCODE" -ForegroundColor Red
                exit $LASTEXITCODE
            }
        }
        "train" {
            Write-Host "Running training only..." -ForegroundColor Green
            $TrainCmd = "$Python -m standard_training.train $TrainArgs --no_eval"
            Write-Host "Running: $TrainCmd" -ForegroundColor Gray
            Invoke-Expression $TrainCmd
            
            if ($LASTEXITCODE -ne 0) {
                Write-Host "Error: Training failed with exit code $LASTEXITCODE" -ForegroundColor Red
                exit $LASTEXITCODE
            }
        }
        "eval" {
            Write-Host "Running evaluation only..." -ForegroundColor Green
            $EvalCmd = "$Python -m standard_training.evaluate $EvalArgs"
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