param(
    [switch]$InstallDeps
)

$projectRoot = Split-Path -Parent $PSScriptRoot
$backendDir = Join-Path $projectRoot "backend"
$frontendDir = Join-Path $projectRoot "frontend"

if ($InstallDeps) {
    Write-Host "Installing backend dependencies..."
    Push-Location $backendDir
    if (!(Test-Path ".\.venv")) {
        python -m venv .venv
    }
    & ".\.venv\Scripts\python.exe" -m pip install --upgrade pip
    & ".\.venv\Scripts\python.exe" -m pip install -r requirements.txt
    Pop-Location

    Write-Host "Installing frontend dependencies..."
    Push-Location $frontendDir
    npm install
    Pop-Location
}

$backendCmd = "cd '$backendDir'; if (Test-Path '.\.venv\Scripts\Activate.ps1') { . '.\.venv\Scripts\Activate.ps1' }; python -m uvicorn app.main:app --reload --port 8000"
$frontendCmd = "cd '$frontendDir'; npm run dev -- --host 0.0.0.0 --port 5173"

Start-Process powershell -ArgumentList "-NoExit", "-Command", $backendCmd
Start-Process powershell -ArgumentList "-NoExit", "-Command", $frontendCmd

Write-Host "DeepShield dev servers started:"
Write-Host "Backend:  http://localhost:8000"
Write-Host "Frontend: http://localhost:5173"
