$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
try {
    if (-not (Test-Path '.venv\Scripts\python.exe')) {
        python -m venv .venv
        if ($LASTEXITCODE -ne 0) { throw 'Could not create .venv (Python >= 3.10 required).' }
    }
    $requirements = if (Test-Path 'requirements-lock.txt') { 'requirements-lock.txt' } else { 'requirements.txt' }
    & '.\.venv\Scripts\python.exe' -m pip install --cache-dir '.cache\pip' -r $requirements
    if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
    & '.\.venv\Scripts\python.exe' -m unittest discover -s tests -v
    if ($LASTEXITCODE -ne 0) { throw 'Checks failed.' }
    Write-Host 'Ready. Offline: .\.venv\Scripts\python.exe run.py --mock --method both'
} finally {
    Pop-Location
}
