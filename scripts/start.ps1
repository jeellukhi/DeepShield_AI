param(
    [switch]$InstallDeps,
    [switch]$StopMongo
)

$projectRoot = Split-Path -Parent $PSScriptRoot
$backendDir  = Join-Path $projectRoot "backend"
$frontendDir = Join-Path $projectRoot "frontend"

# Stop MongoDB container if requested
if ($StopMongo) {
    Write-Host "Stopping MongoDB container..." -ForegroundColor Yellow
    docker stop deepshield-mongodb 2>$null
    docker rm   deepshield-mongodb 2>$null
    Write-Host "MongoDB stopped." -ForegroundColor Green
    exit 0
}

# Step 1: Start MongoDB via Docker
Write-Host ""
Write-Host "=== Step 1: Starting MongoDB via Docker ===" -ForegroundColor Cyan

$mongoRunning = docker ps --filter "name=deepshield-mongodb" --format "{{.Names}}" 2>$null
if ($mongoRunning -eq "deepshield-mongodb") {
    Write-Host "MongoDB is already running." -ForegroundColor Green
} else {
    Write-Host "Starting MongoDB container..." -ForegroundColor Yellow
    docker run -d `
        --name deepshield-mongodb `
        --restart unless-stopped `
        -p 27017:27017 `
        mongo:7 2>&1 | Out-Null

    Write-Host "Waiting for MongoDB to be ready..." -ForegroundColor Yellow
    Start-Sleep -Seconds 4
    Write-Host "MongoDB started on port 27017." -ForegroundColor Green
}

# Step 2: Install deps if -InstallDeps flag is passed
if ($InstallDeps) {
    Write-Host ""
    Write-Host "=== Installing Backend Dependencies ===" -ForegroundColor Cyan
    Push-Location $backendDir
    if (!(Test-Path ".\.venv")) {
        python -m venv .venv
    }
    & ".\.venv\Scripts\python.exe" -m pip install --upgrade pip -q
    & ".\.venv\Scripts\python.exe" -m pip install -r requirements.txt -q
    Pop-Location

    Write-Host ""
    Write-Host "=== Installing Frontend Dependencies ===" -ForegroundColor Cyan
    Push-Location $frontendDir
    npm install --silent
    Pop-Location
}

# Step 3: Start Backend
Write-Host ""
Write-Host "=== Step 2: Starting Backend (FastAPI) ===" -ForegroundColor Cyan
$backendCmd = "cd '$backendDir'; Write-Host 'Activating Python venv...' -ForegroundColor Yellow; . '.\.venv\Scripts\Activate.ps1'; Write-Host 'Starting FastAPI on http://localhost:8000' -ForegroundColor Green; python -m uvicorn app.main:app --reload --port 8000"
Start-Process powershell -ArgumentList "-NoExit", "-Command", $backendCmd

Start-Sleep -Seconds 2

# Step 4: Start Frontend
Write-Host "=== Step 3: Starting Frontend (React/Vite) ===" -ForegroundColor Cyan
$frontendCmd = "cd '$frontendDir'; Write-Host 'Starting React on http://localhost:5173' -ForegroundColor Green; npm run dev -- --host 0.0.0.0 --port 5173"
Start-Process powershell -ArgumentList "-NoExit", "-Command", $frontendCmd

# Summary
Write-Host ""
Write-Host "================================================" -ForegroundColor Green
Write-Host "  DeepShield AI is starting up!" -ForegroundColor Green
Write-Host "================================================" -ForegroundColor Green
Write-Host ""
Write-Host "  MongoDB  : mongodb://localhost:27017" -ForegroundColor White
Write-Host "  Backend  : http://localhost:8000"     -ForegroundColor White
Write-Host "  Frontend : http://localhost:5173"     -ForegroundColor White
Write-Host "  API Docs : http://localhost:8000/docs" -ForegroundColor White
Write-Host ""
Write-Host "  Wait ~5 seconds then open: http://localhost:5173" -ForegroundColor Yellow
Write-Host ""
Write-Host "  To stop MongoDB later, run:" -ForegroundColor Gray
Write-Host "  .\scripts\start.ps1 -StopMongo" -ForegroundColor Gray
Write-Host ""
