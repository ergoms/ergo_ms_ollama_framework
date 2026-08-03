#Requires -RunAsAdministrator
<#
.SYNOPSIS
    Удаляет Windows-службу ergo-ollama модуля ollama_framework.
#>

$ErrorActionPreference = 'Stop'

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ModuleRoot = (Resolve-Path (Join-Path $ScriptDir '..\..')).Path
$ProjectRoot = (Resolve-Path (Join-Path $ModuleRoot '..\..')).Path

$coreDeployment = Join-Path $ProjectRoot 'core\deployment\windows\lib'
. (Join-Path $coreDeployment 'core.ps1')
. (Join-Path $coreDeployment 'nssm.ps1')

function Uninstall-OllamaService {
    param([string]$Root)

    $serviceName = 'ergo-ollama'
    $service = Get-Service -Name $serviceName -ErrorAction SilentlyContinue
    if (-not $service) {
        Write-ColorOutput "[SKIP] Service $serviceName not found" Gray
        return
    }

    $nssmDir = Get-NssmDir -Root $Root
    $nssmExe = Join-Path $nssmDir 'nssm.exe'
    if (-not (Test-Path -LiteralPath $nssmExe)) {
        try {
            $nssmExe = Install-NSSM -Root $Root
        } catch {
            $nssmExe = $null
        }
    }

    Write-ColorOutput "-> Removing service: $serviceName" Cyan
    if ($service.Status -ne 'Stopped') {
        if ($nssmExe -and (Test-Path -LiteralPath $nssmExe)) {
            & $nssmExe stop $serviceName 2>$null
        }
        Stop-Service -Name $serviceName -Force -ErrorAction SilentlyContinue
        Start-Sleep -Seconds 2
    }

    if ($nssmExe -and (Test-Path -LiteralPath $nssmExe)) {
        & $nssmExe remove $serviceName confirm 2>$null
    }
    if (Get-Service -Name $serviceName -ErrorAction SilentlyContinue) {
        sc.exe delete $serviceName 2>$null
    }

    $wrapperPath = Join-Path (Get-ProjectWrappersDir -ProjectRoot $Root) 'start_ollama.bat'
    if (Test-Path -LiteralPath $wrapperPath) {
        Remove-Item -LiteralPath $wrapperPath -Force -ErrorAction SilentlyContinue
    }

    Write-ColorOutput "[OK] Service $serviceName removed" Green
}

Write-ColorOutput "-> Uninstalling Ollama service for: $ProjectRoot" Cyan
Uninstall-OllamaService -Root $ProjectRoot
