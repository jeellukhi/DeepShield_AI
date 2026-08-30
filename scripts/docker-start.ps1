param(
    [switch]$Build
)

$projectRoot = Split-Path -Parent $PSScriptRoot
$composeFile = Join-Path $projectRoot "docker\docker-compose.yml"

if ($Build) {
    docker compose -f $composeFile up -d --build
} else {
    docker compose -f $composeFile up -d
}

Write-Host "DeepShield Docker stack started."
Write-Host "Frontend: http://localhost:5173"
Write-Host "Backend:  http://localhost:8000"
