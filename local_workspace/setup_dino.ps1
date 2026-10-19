param([string]$Proxy)
$ErrorActionPreference = 'Stop'
$taskPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $taskPython)) { throw 'Prepare the project .venv first.' }
$env:PYTHONIOENCODING = 'utf-8'
Push-Location $PSScriptRoot
try {
    $taskArguments = @('research/setup_dino_teacher.py')
    if ($Proxy) { $taskArguments += @('--proxy', $Proxy) }
    & $taskPython @taskArguments
    if ($LASTEXITCODE -ne 0) { throw 'Official DINO teacher preparation failed.' }
}
finally { Pop-Location }
