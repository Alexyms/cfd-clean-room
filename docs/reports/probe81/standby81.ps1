# Prompt 81: the machine's Modern Standby intervals since the queue's runner started, from the
# System event log (Kernel-Power 506 entering, 507 exiting), written to
# results/builder81/standby.json for tables81.py, which takes them out of the walls. This is
# prompt 79's standby79.ps1 with its output directory moved; the runner itself holds the system
# awake with SetThreadExecutionState, which Modern Standby does not always honour.
#
#   powershell -NoProfile -File docs/reports/probe81/standby81.ps1 -Since 2026-11-14T20:00:00
param([Parameter(Mandatory = $true)][string]$Since)
$events = Get-WinEvent -FilterHashtable @{
    LogName = 'System'; ProviderName = 'Microsoft-Windows-Kernel-Power'; Id = 506, 507
    StartTime = [datetime]$Since
} -ErrorAction SilentlyContinue | Sort-Object TimeCreated
$intervals = @()
$entered = $null
foreach ($e in $events) {
    if ($e.Id -eq 506) { $entered = $e.TimeCreated }
    elseif ($e.Id -eq 507 -and $null -ne $entered) {
        $intervals += [ordered]@{
            entered = $entered.ToString("yyyy-MM-ddTHH:mm:ss")
            exited = $e.TimeCreated.ToString("yyyy-MM-ddTHH:mm:ss")
        }
        $entered = $null
    }
}
$target = Join-Path (Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))) "results\builder81\standby.json"
ConvertTo-Json -InputObject @($intervals) | Set-Content -Path $target -Encoding utf8
Get-Content $target
