$projectRoot = Split-Path -Parent $PSScriptRoot
$composeFile = Join-Path $projectRoot "docker\docker-compose.yml"

docker compose -f $composeFile down

Write-Host "DeepShield Docker stack stopped."
