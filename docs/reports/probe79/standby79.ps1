# Prompt 79: the machine's Modern Standby intervals since the queue's runner started, from the
# System event log (Kernel-Power 506 entering, 507 exiting), written to
# results/builder79/standby.json for tables79.py, which takes them out of the walls.
#
#   powershell -NoProfile -File docs/reports/probe79/standby79.ps1
param([string]$Since = "2026-11-12T22:09:00")
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
$target = Join-Path (Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))) "results\builder79\standby.json"
ConvertTo-Json -InputObject @($intervals) | Set-Content -Path $target -Encoding utf8
Get-Content $target
