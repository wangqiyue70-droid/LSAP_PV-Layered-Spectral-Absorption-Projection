param(
    [string]$Device = 'cuda:0',
    [ValidateSet('feedback', 'align')][string]$DinoMode = 'feedback',
    [ValidateRange(1, 1000)][int]$SampleSteps = 20,
    [switch]$Amp,
    [switch]$ActivationCheckpoint,
    [string]$RunName = ('dino_smoke_' + (Get-Date -Format 'yyyyMMdd_HHmmss'))
)
$ErrorActionPreference = 'Stop'
if ($RunName -notmatch '^[a-zA-Z0-9_-]+$') { throw 'RunName must contain only letters, numbers, underscores and hyphens.' }
$taskPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
$taskSource = Join-Path $PSScriptRoot '.downloads\dinov2\hubconf.py'
$taskWeights = Join-Path $PSScriptRoot '.downloads\dinov2_vits14_pretrain.pth'
if (-not (Test-Path -LiteralPath $taskPython)) { throw 'Prepare the project .venv first.' }
if (-not (Test-Path -LiteralPath $taskSource) -or -not (Test-Path -LiteralPath $taskWeights)) {
    throw 'Run setup_dino.ps1 to prepare the official frozen DINOv2 teacher first.'
}
$env:PYTHONIOENCODING = 'utf-8'
Push-Location (Join-Path $PSScriptRoot 'LSAP-PV')
try {
    Write-Output 'Synthetic software test with a real frozen DINO teacher. This is not real palm-vein training.'
    if (-not (Test-Path -LiteralPath 'smoke_conditions\pv_pattern_results\palmvein')) {
        & $taskPython generate_condition_single.py
        if ($LASTEXITCODE -ne 0) { throw 'Condition generation failed.' }
    }
    $taskArguments = @('train_flow.py', '--smoke-test', '--dino-mode', $DinoMode,
                       '--steps', '2', '--out-dir', "runs/$RunName", '--device', $Device)
    if ($Amp) { $taskArguments += '--amp' }
    if ($ActivationCheckpoint) { $taskArguments += '--activation-checkpoint' }
    & $taskPython @taskArguments
    if ($LASTEXITCODE -ne 0) { throw 'DINO Flow training smoke test failed.' }
    & $taskPython train_flow.py --resume "runs/$RunName/flow_step_0000002.pt" `
        --steps 3 --out-dir "runs/$RunName" --device $Device
    if ($LASTEXITCODE -ne 0) { throw 'DINO Flow resume smoke test failed.' }
    & $taskPython sample_flow.py --model-path "runs/$RunName/flow_step_0000003.pt" `
        --cond-path smoke_conditions/pv_pattern_results/palmvein --steps $SampleSteps `
        --output "results_flow/$RunName.png" --device $Device
    if ($LASTEXITCODE -ne 0) { throw 'DINO Flow sampling smoke test failed.' }
}
finally { Pop-Location }
