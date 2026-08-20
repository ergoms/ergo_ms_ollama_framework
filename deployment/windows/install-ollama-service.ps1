#Requires -RunAsAdministrator
<#
.SYNOPSIS
    Устанавливает Windows-службу ergo-ollama для модуля ollama_framework.
#>

$ErrorActionPreference = 'Stop'

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ModuleRoot = (Resolve-Path (Join-Path $ScriptDir '..\..')).Path
$ProjectRoot = (Resolve-Path (Join-Path $ModuleRoot '..\..')).Path

$coreDeployment = Join-Path $ProjectRoot 'core\deployment\windows\lib'
. (Join-Path $coreDeployment 'core.ps1')
. (Join-Path $coreDeployment 'nssm.ps1')
. (Join-Path $coreDeployment 'services.ps1')

function New-OllamaServiceWrapper {
    param([string]$Root)

    $apiPath = Join-Path $Root 'core\api'
    $venvActivate = Join-Path $Root 'virtual_env\python\Scripts\activate.bat'
    $wrapperDir = Get-ProjectWrappersDir -ProjectRoot $Root
    $wrapperPath = Join-Path $wrapperDir 'start_ollama.bat'

    New-Item -ItemType Directory -Path $wrapperDir -Force | Out-Null

    $ergomsCmd = Get-Command ergoms -ErrorAction SilentlyContinue
    if (-not $ergomsCmd) {
        throw (
            'Команда ergoms не найдена в PATH. ' +
            'Добавьте core\deployment\bin в PATH или установите CLI проекта, ' +
            'затем повторите установку службы.'
        )
    }

    $content = @(
        '@echo off',
        'chcp 65001 >nul',
        'set PYTHONIOENCODING=utf-8',
        'set PYTHONUTF8=1',
        "set PYTHONPATH=$Root",
        "cd /d `"$Root`"",
        'ergoms ollama_framework:start-ollama'
    ) -join "`r`n"

    Set-Content -Path $wrapperPath -Value $content -Encoding ASCII
    return $wrapperPath
}

function Install-OllamaService {
    param([string]$Root)

    $serviceName = 'ergo-ollama'
    $nssmExe = Install-NSSM
    $logsDir = Get-ProjectLogsDir -ProjectRoot $Root
    New-Item -ItemType Directory -Path $logsDir -Force | Out-Null

    $existingService = Get-Service -Name $serviceName -ErrorAction SilentlyContinue
    if ($existingService) {
        Write-ColorOutput "-> Service $serviceName already exists, reinstalling..." Yellow
        if ($existingService.Status -eq 'Running') {
            & $nssmExe stop $serviceName 2>$null
            Start-Sleep -Seconds 2
        }
        & $nssmExe remove $serviceName confirm 2>$null
        Start-Sleep -Seconds 2
    }

    $wrapperPath = New-OllamaServiceWrapper -Root $Root
    Write-ColorOutput "-> Installing service: $serviceName" Cyan

    & $nssmExe install $serviceName $wrapperPath
    & $nssmExe set $serviceName DisplayName "Ergo MS - $serviceName"
    & $nssmExe set $serviceName Description "Ergo Management System - Ollama Server"
    & $nssmExe set $serviceName AppDirectory (Join-Path $Root 'core')
    $singleLog = Join-Path $logsDir 'ollama-serve.log'
    $pythonExe = Join-Path $Root 'virtual_env\python\Scripts\python.exe'
    $logEnvScript = Join-Path $Root 'core\deployment\scripts\log_env.py'
    if ((Test-Path -LiteralPath $pythonExe) -and (Test-Path -LiteralPath $logEnvScript)) {
        $resolved = & $pythonExe $logEnvScript path OLLAMA $Root 2>$null
        if (-not [string]::IsNullOrWhiteSpace($resolved)) {
            $singleLog = $resolved.Trim()
        }
    }
    $logParent = Split-Path -Parent $singleLog
    if ($logParent) {
        New-Item -ItemType Directory -Path $logParent -Force | Out-Null
    }
    & $nssmExe set $serviceName AppStdout $singleLog
    & $nssmExe set $serviceName AppStderr $singleLog
    & $nssmExe set $serviceName AppEnvironmentExtra "PYTHONIOENCODING=UTF-8" "PYTHONUTF8=1" "PYTHONUNBUFFERED=1"
    & $nssmExe set $serviceName Start SERVICE_AUTO_START
    & $nssmExe set $serviceName AppExit Default Restart
    & $nssmExe set $serviceName AppRestartDelay 5000

    Write-ColorOutput "[OK] Service $serviceName installed" Green
}

Write-ColorOutput "-> Installing Ollama service for: $ProjectRoot" Cyan
Install-OllamaService -Root $ProjectRoot
Start-Service -Name 'ergo-ollama'
Write-ColorOutput "`n[OK] Ollama service installed and started!" Green
Show-ServicesStatus -ProjectRoot $ProjectRoot
