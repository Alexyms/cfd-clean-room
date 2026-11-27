# Prompt 97: start the runner detached from any console or session (prompt 79's lesson:
# a process made by Win32_Process.Create has no window to close), then the keep-awake
# holder on its pid. The runner resumes where its records end, so this may be run again
# after a stop; it refuses while results/builder97/HALT exists. "-Queue queue97b" starts
# prompt 97b's runner (sweeps97.py), which runs prompt 97's remaining jobs after its own.
param([ValidateSet("queue", "queue97b")][string]$Queue = "queue")
$root = "C:\Users\tomat\Projects\cfd_clean_room"
$log = Join-Path $root "results\builder97\launch.log"
$hidden = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ShowWindow = [uint16]0}
$runner = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
    CommandLine = "wsl.exe -d Ubuntu-24.04 -- bash /mnt/c/Users/tomat/Projects/cfd_clean_room/docs/reports/probe97/run97.sh $Queue"
    ProcessStartupInformation = $hidden
}
"$(Get-Date -Format s) runner: return $($runner.ReturnValue) pid $($runner.ProcessId)" | Add-Content $log
$keep = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
    CommandLine = "powershell.exe -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$root\docs\reports\probe97\keepawake97.ps1`" -RunnerPid $($runner.ProcessId) -Log `"$root\results\builder97\keepawake.log`""
    ProcessStartupInformation = $hidden
}
"$(Get-Date -Format s) keepawake: return $($keep.ReturnValue) pid $($keep.ProcessId)" | Add-Content $log
Get-Content $log -Tail 2
