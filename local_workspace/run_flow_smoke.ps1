param(
    [string]$Device = 'cuda:0',
    [ValidateRange(1, 1000)][int]$SampleSteps = 20,
    [string]$RunName = ('flow_smoke_' + (Get-Date -Format 'yyyyMMdd_HHmmss'))
)
$ErrorActionPreference = 'Stop'
if ($RunName -notmatch '^[a-zA-Z0-9_-]+$') { throw 'RunName must contain only letters, numbers, underscores and hyphens.' }
$taskPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $taskPython)) { throw 'Prepare the project .venv first.' }
$env:PYTHONIOENCODING = 'utf-8'
Push-Location (Join-Path $PSScriptRoot 'LSAP-PV')
try {
    Write-Output 'Synthetic software test only: this does not train a palm-vein research model.'
    if (-not (Test-Path -LiteralPath 'smoke_conditions\pv_pattern_results\palmvein')) {
        & $taskPython generate_condition_single.py
        if ($LASTEXITCODE -ne 0) { throw 'Condition generation failed.' }
    }
    & $taskPython train_flow.py --smoke-test --steps 2 --out-dir "runs/$RunName" --device $Device
    if ($LASTEXITCODE -ne 0) { throw 'Flow training smoke test failed.' }
    & $taskPython train_flow.py --resume "runs/$RunName/flow_step_0000002.pt" `
        --steps 3 --out-dir "runs/$RunName" --device $Device
    if ($LASTEXITCODE -ne 0) { throw 'Flow resume smoke test failed.' }
    & $taskPython sample_flow.py --model-path "runs/$RunName/flow_step_0000003.pt" `
        --cond-path smoke_conditions/pv_pattern_results/palmvein --steps $SampleSteps `
        --output "results_flow/$RunName.png" --device $Device
    if ($LASTEXITCODE -ne 0) { throw 'Flow sampling smoke test failed.' }
}
finally { Pop-Location }
