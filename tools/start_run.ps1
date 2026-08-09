# Start the pretraining run so it survives this console closing.
#
#   powershell -ExecutionPolicy Bypass -File tools\start_run.ps1
#
# Two things have to be true for a long run to stay up, and neither is the notebook's doing.
#
# 1. The utility VM must not idle out. `.wslconfig` handles that with
#    `vmIdleTimeout=2147483647`; without it WSL2 shuts the VM down 60 seconds after the last
#    session exits and takes the distro with it.
#
# 2. Some wsl.exe session must stay open. WSL kills the process tree belonging to a
#    non-interactive `wsl -- ...` invocation once that invocation returns -- about 8 seconds
#    later, with no signal recorded anywhere, and it reaches even a systemd transient unit.
#    Measured: every session-less launch died at launch+8s at exactly the same log line, while a
#    launch made from a console that stayed open ran for an hour. The keep-alive below is a single
#    idle `sleep infinity`; it holds the session and nothing else.
#
# The run itself still goes through systemd (`wsl_pretrain_bg.sh`), so it is detached from *this*
# console and gets journald logging. The keep-alive is what stops WSL from reaping it.

$ErrorActionPreference = 'Stop'
$Repo = '/mnt/c/Users/9700X-5070/Downloads/github/aletheia-nvfp4'

$alive = Get-CimInstance Win32_Process -Filter "Name='wsl.exe'" |
         Where-Object { $_.CommandLine -like '*sleep*infinity*' }

if ($alive) {
    Write-Host "keep-alive : already running (pid $($alive.ProcessId))"
} else {
    Start-Process wsl -ArgumentList '-d','Ubuntu','-u','root','--','sleep','infinity' `
                      -WindowStyle Hidden
    Start-Sleep -Seconds 2
    $alive = Get-CimInstance Win32_Process -Filter "Name='wsl.exe'" |
             Where-Object { $_.CommandLine -like '*sleep*infinity*' }
    if (-not $alive) { throw "keep-alive session failed to start" }
    Write-Host "keep-alive : started (pid $($alive.ProcessId))"
}

wsl -d Ubuntu -u root -- bash "$Repo/tools/wsl_pretrain_bg.sh"

Write-Host ""
Write-Host "watch      : Get-Content '\\wsl.localhost\Ubuntu\root\aletheia-run\logs\pretrain.log' -Wait -Tail 40"
Write-Host "status     : wsl -d Ubuntu -u root -- /opt/ale/bin/python $Repo/tools/train_status.py /root/aletheia-run/aletheia_nvfp4"
