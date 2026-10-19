param(
    [string]$Device = 'cuda:0',
    [int]$Seed = 0,
    [ValidateRange(2, 1000)][int]$Steps = 20,
    [string]$Output = ('results_smoke/sample_' + (Get-Date -Format 'yyyyMMdd_HHmmss') + '.png')
)
$ErrorActionPreference = 'Stop'
$taskPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
$taskRepo = Join-Path $PSScriptRoot 'LSAP-PV'
if (-not (Test-Path -LiteralPath $taskPython)) {
    throw 'Run setup_local.ps1 to prepare the environment first.'
}
$env:PYTHONIOENCODING = 'utf-8'
$env:OMP_NUM_THREADS = '4'
$env:MKL_NUM_THREADS = '4'
Push-Location $taskRepo
try {
    if (-not (Test-Path -LiteralPath 'smoke_conditions\pv_pattern_results\palmvein')) {
        & $taskPython generate_condition_single.py
        if ($LASTEXITCODE -ne 0) { throw 'Condition generation failed.' }
    }
    & $taskPython sample_single.py `
        --model_path checkpoint/vein_sty_concat_sigma_cosine_vpem/ema_0.9999_070000.pt `
        --cond_path smoke_conditions/pv_pattern_results/palmvein `
        --batch_size 1 --device $Device --seed $Seed --steps $Steps --output $Output
    if ($LASTEXITCODE -ne 0) { throw 'Sampling failed.' }
}
finally { Pop-Location }
