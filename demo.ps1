[CmdletBinding()]
param(
    [switch]$Rebuild
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSCommandPath
$pythonPath = Join-Path $projectRoot ".venv\Scripts\python.exe"
$artifactPath = Join-Path $projectRoot "artifacts\store_sales_forecast_v1.pkl"
$historyPath = Join-Path $projectRoot "artifacts\store_sales_forecast_v1_history.csv.gz"
$imageName = "retail-sales-forecast-api:v1"
$containerName = "retail-sales-forecast-api"
$dashboardUrl = "http://127.0.0.1:8000/demo"

Set-Location -LiteralPath $projectRoot

foreach ($requiredPath in @($pythonPath, $artifactPath, $historyPath)) {
    if (-not (Test-Path -LiteralPath $requiredPath -PathType Leaf)) {
        throw "Required demo file is missing: $requiredPath"
    }
}

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    throw "Docker CLI is not available. Start Docker Desktop and try again."
}

& docker info *> $null
if ($LASTEXITCODE -ne 0) {
    throw "Docker Desktop is not ready. Start it and try again."
}

& docker image inspect $imageName *> $null
$imageExists = $LASTEXITCODE -eq 0
if ($Rebuild -or -not $imageExists) {
    Write-Host "Building the runtime image..." -ForegroundColor Cyan
    & docker build --target runtime --tag $imageName .
    if ($LASTEXITCODE -ne 0) {
        throw "Docker image build failed."
    }
}

$existingContainer = [string](& docker ps --all --quiet --filter "name=^/$containerName$")
if ($null -eq $existingContainer) {
    $existingContainer = ""
} else {
    $existingContainer = $existingContainer.Trim()
}
if ($existingContainer) {
    Write-Host "Restarting the existing demo container..." -ForegroundColor Cyan
    & docker rm --force $containerName *> $null
    if ($LASTEXITCODE -ne 0) {
        throw "The existing demo container could not be removed."
    }
}

$previousApiKey = [Environment]::GetEnvironmentVariable(
    "RETAIL_FORECAST_API_KEY",
    "Process"
)
$containerStarted = $false
$demoReady = $false

try {
    $apiKeyOutput = @(& $pythonPath -c "import secrets; print(secrets.token_urlsafe(32))" 2>&1)
    $apiKeyExitCode = $LASTEXITCODE
    $apiKey = [string]($apiKeyOutput -join [Environment]::NewLine)
    if ($null -eq $apiKey) {
        $apiKey = ""
    } else {
        $apiKey = $apiKey.Trim()
    }
    if ($apiKeyExitCode -ne 0 -or $apiKey.Length -lt 32) {
        throw "A temporary API key could not be generated."
    }
    $env:RETAIL_FORECAST_API_KEY = $apiKey

    $runArguments = @(
        "run",
        "--rm",
        "--detach",
        "--name", $containerName,
        "--publish", "127.0.0.1:8000:8000",
        "--env", "RETAIL_FORECAST_API_KEY",
        "--env", "RETAIL_FORECAST_DEMO_MODE=1",
        "--env", "RETAIL_FORECAST_ARTIFACT_PATH=/app/private/store_sales_forecast_v1.pkl",
        "--env", "RETAIL_FORECAST_HISTORY_PATH=/app/private/store_sales_forecast_v1_history.csv.gz",
        "--mount", "type=bind,source=$projectRoot\artifacts,target=/app/private,readonly",
        $imageName
    )

    Write-Host "Starting the forecasting API..." -ForegroundColor Cyan
    $containerOutput = @(& docker @runArguments 2>&1)
    $containerExitCode = $LASTEXITCODE
    $containerId = [string]($containerOutput -join [Environment]::NewLine)
    if ($null -eq $containerId) {
        $containerId = ""
    } else {
        $containerId = $containerId.Trim()
    }
    if ($containerExitCode -ne 0 -or -not $containerId) {
        $dockerMessage = [string]($containerOutput -join " ")
        throw "The demo container could not be started. Docker reported: $dockerMessage"
    }
    $containerStarted = $true

    $healthStatus = "starting"
    for ($attempt = 1; $attempt -le 60; $attempt++) {
        $healthOutput = @(& docker inspect --format "{{.State.Health.Status}}" $containerName 2>&1)
        if ($LASTEXITCODE -ne 0) {
            $dockerMessage = [string]($healthOutput -join " ")
            throw "The demo container health could not be inspected. Docker reported: $dockerMessage"
        }
        $healthStatus = [string]($healthOutput -join [Environment]::NewLine)
        if ($null -eq $healthStatus) {
            $healthStatus = ""
        } else {
            $healthStatus = $healthStatus.Trim()
        }
        if ($healthStatus -eq "healthy") {
            break
        }
        if ($healthStatus -eq "unhealthy") {
            & docker logs $containerName
            throw "The demo container reported an unhealthy status."
        }
        Start-Sleep -Seconds 2
    }

    if ($healthStatus -ne "healthy") {
        & docker logs $containerName
        throw "The demo container did not become healthy within 120 seconds."
    }

    Write-Host "`nRunning the complete live forecast verification..." -ForegroundColor Cyan
    & $pythonPath "verify_api.py" --record-demo-verification
    if ($LASTEXITCODE -ne 0) {
        throw "Live API verification failed."
    }

    $metricsRequest = @{
        Uri = "http://127.0.0.1:8000/metrics"
        Headers = @{"X-API-Key" = $apiKey}
    }
    $metrics = Invoke-RestMethod @metricsRequest

    Start-Process $dashboardUrl
    $demoReady = $true

    Write-Host "`nDEMO READY" -ForegroundColor Green
    Write-Host "API status: healthy"
    Write-Host "Successful forecast batches: $($metrics.forecast.success_total)"
    Write-Host "Schema rejections: $($metrics.forecast.schema_rejections_total)"
    Write-Host "Contract rejections: $($metrics.forecast.contract_rejections_total)"
    Write-Host "Authentication rejections: $($metrics.forecast.authentication_rejections_total)"
    Write-Host "Model errors: $($metrics.forecast.model_errors_total)"
    Write-Host "`nThe interview dashboard is open:" -ForegroundColor Cyan
    Write-Host $dashboardUrl
    Write-Host "`nWhen the demo is finished, stop it with:"
    Write-Host "docker stop $containerName" -ForegroundColor Yellow
} catch {
    if ($containerStarted -and -not $demoReady) {
        & docker rm --force $containerName *> $null
    }
    throw
} finally {
    [Environment]::SetEnvironmentVariable(
        "RETAIL_FORECAST_API_KEY",
        $previousApiKey,
        "Process"
    )
}
