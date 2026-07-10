[CmdletBinding()]
param(
    [switch]$LocalFilesOnly,
    [switch]$DryRun,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Phase3Args
)

$ErrorActionPreference = "Stop"

function Get-ActivePowerScheme {
    $output = (& powercfg.exe /getactivescheme | Out-String)
    $match = [regex]::Match(
        $output,
        "[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}"
    )
    if (-not $match.Success) {
        throw "Could not determine the active Windows power scheme."
    }
    return $match.Value
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if ($repoRoot -notmatch "^(?<drive>[A-Za-z]):\\(?<rest>.*)$") {
    throw "The Windows wrapper requires a repository on a mounted drive."
}
$wslRepoRoot = "/mnt/$($Matches.drive.ToLower())/$($Matches.rest.Replace('\', '/'))"

$originalScheme = Get-ActivePowerScheme
$argsForPython = @("scripts/run_phase3.py")
if ($LocalFilesOnly) {
    $argsForPython += "--local-files-only"
}
if ($Phase3Args) {
    $argsForPython += $Phase3Args
}
if ($wslRepoRoot.Contains("'") -or ($argsForPython | Where-Object { $_ -and $_.Contains("'") })) {
    throw "Single quotes are not supported in the WSL wrapper arguments."
}
$quotedArgs = $argsForPython | ForEach-Object { "'$_'" }
$bashCommand = "cd '$wslRepoRoot' && /home/akifliu/.venvs/llm-quant-profiler/bin/python " + ($quotedArgs -join " ")

if ($DryRun) {
    Write-Output "Original power scheme: $originalScheme"
    Write-Output "Canonical command: wsl.exe bash -lc $bashCommand"
    exit 0
}

try {
    & powercfg.exe /setactive SCHEME_MIN
    if ($LASTEXITCODE -ne 0) {
        throw "Could not activate the Windows High performance power scheme."
    }
    & wsl.exe bash -lc $bashCommand
    if ($LASTEXITCODE -ne 0) {
        throw "Canonical workflow failed with exit code $LASTEXITCODE."
    }
}
finally {
    & powercfg.exe /setactive $originalScheme
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "Could not restore the original Windows power scheme: $originalScheme"
    }
}
