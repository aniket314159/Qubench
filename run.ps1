<#
    run.ps1 — Windows equivalent of run.sh.

        .\run.ps1            backend only   (http://localhost:8000)
        .\run.ps1 web        frontend only  (http://localhost:5173)
        .\run.ps1 all        both, in two new windows

    --reload-dir backend is not optional. A bare --reload watches the whole
    working directory, which here means thousands of files in .venv plus
    node_modules; startup crawls and spurious reloads fire.
#>

param([string]$Target = "api")

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

if (-not (Test-Path ".venv")) {
    Write-Host "No .venv at the repo root. Create it:" -ForegroundColor Yellow
    Write-Host "  python -m venv .venv"
    Write-Host "  .\.venv\Scripts\Activate.ps1"
    Write-Host "  pip install torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cpu"
    Write-Host "  pip install -r backend\requirements.txt"
    exit 1
}

function Start-Api {
    & ".\.venv\Scripts\python.exe" -m uvicorn backend.main:app `
        --reload --reload-dir backend --port 8000
}

function Start-Web {
    if (-not (Test-Path "frontend\node_modules")) {
        Push-Location frontend; npm install; Pop-Location
    }
    Push-Location frontend; npm run dev; Pop-Location
}

switch ($Target) {
    { $_ -in "api", "backend", "" } { Start-Api }
    { $_ -in "web", "frontend" }    { Start-Web }
    { $_ -in "all", "both" } {
        # Two windows rather than background jobs: uvicorn's reloader and
        # Vite both want a live console, and Ctrl-C in a job does not reach
        # them. Close the windows to stop.
        Start-Process powershell -ArgumentList `
            "-NoExit", "-Command", "Set-Location '$PSScriptRoot'; .\run.ps1 api"
        Start-Sleep -Seconds 6     # let the API bind before Vite serves a page
        Start-Process powershell -ArgumentList `
            "-NoExit", "-Command", "Set-Location '$PSScriptRoot'; .\run.ps1 web"
        Write-Host ""
        Write-Host "  API  http://localhost:8000/docs"
        Write-Host "  App  http://localhost:5173"
        Write-Host "  wired: curl.exe -s localhost:8000/health"
        Write-Host ""
    }
    default { Write-Host "usage: .\run.ps1 [api|web|all]"; exit 1 }
}
