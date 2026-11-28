# Prompt 97: keep the machine out of idle sleep while the runner works (prompt 79's
# keepawake.ps1). Holds SetThreadExecutionState (system required) until the runner's
# wsl.exe exits, then releases it. No power setting changes.
param([int]$RunnerPid, [string]$Log)
Add-Type -Namespace Win32 -Name Power -MemberDefinition @"
[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);
"@
$ES_CONTINUOUS = [uint32]"0x80000000"
$ES_SYSTEM_REQUIRED = [uint32]"0x00000001"
"$(Get-Date -Format s) hold for runner pid $RunnerPid" | Add-Content $Log
[Win32.Power]::SetThreadExecutionState($ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED) | Out-Null
while (Get-Process -Id $RunnerPid -ErrorAction SilentlyContinue) { Start-Sleep -Seconds 60 }
[Win32.Power]::SetThreadExecutionState($ES_CONTINUOUS) | Out-Null
"$(Get-Date -Format s) released: runner pid $RunnerPid exited" | Add-Content $Log
