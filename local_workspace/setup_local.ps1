$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
try {
    if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
        python -m venv .venv
        if ($LASTEXITCODE -ne 0) { throw 'Virtual environment creation failed.' }
    }
    $taskPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
    & $taskPython -m pip install -r requirements-local.txt
    if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
    & $taskPython research/download_torch.py
    if ($LASTEXITCODE -ne 0) { throw 'Official CUDA wheel download failed.' }
    & $taskPython -m pip install `
        '.downloads/torch-2.8.0+cu128-cp312-cp312-win_amd64.whl' `
        '.downloads/torchvision-0.23.0+cu128-cp312-cp312-win_amd64.whl'
    if ($LASTEXITCODE -ne 0) { throw 'CUDA wheel installation failed.' }
    & $taskPython research/download_checkpoint.py
    if ($LASTEXITCODE -ne 0) { throw 'Author checkpoint download failed.' }
    & $taskPython -m pip check
    if ($LASTEXITCODE -ne 0) { throw 'Python dependency consistency check failed.' }
    & $taskPython -c "import torch; print('PyTorch:',torch.__version__); print('CUDA:',torch.cuda.is_available()); assert torch.cuda.is_available(); print('GPU:',torch.cuda.get_device_name(0))"
    if ($LASTEXITCODE -ne 0) { throw 'CUDA verification failed.' }
}
finally { Pop-Location }
