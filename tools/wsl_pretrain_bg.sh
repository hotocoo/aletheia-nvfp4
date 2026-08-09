#!/usr/bin/env bash
# Start the pretraining run detached from whatever launched it.
#
#   wsl -d Ubuntu -u root -- bash /mnt/c/.../tools/wsl_pretrain_bg.sh
#
# Why this exists rather than the Start-Process recipe in RUNNING.md: launching through a hidden
# host PowerShell window ties the run's lifetime to a wsl.exe session on the Windows side. When
# that session's state goes bad -- the WSL service can start answering `Wsl/Service/E_UNEXPECTED`
# while the distro itself is still up -- the launcher dies and takes papermill with it. That is
# not hypothetical; it is how the first attempt at this run ended, about an hour in.
#
# `setsid nohup ... &` is not enough on its own either. WSL reaps the process tree of a
# non-interactive `wsl.exe -- cmd` invocation when that session exits, so a backgrounded child
# dies within seconds of the launcher returning. Handing the job to systemd instead puts it under
# PID 1, where nothing on the Windows side can reach it. The setsid path below is kept only as a
# fallback for distros without systemd (`systemd=true` under `[boot]` in /etc/wsl.conf).
set -eu

HERE=$(cd "$(dirname "$0")" && pwd)
WORK=/root/aletheia-run
UNIT=aletheia-pretrain
mkdir -p "$WORK/logs"

if systemctl is-active --quiet "$UNIT" 2>/dev/null; then
    echo "already running as $UNIT.service"
    systemctl status "$UNIT" --no-pager --lines=0 || true
    exit 0
fi
if pgrep -f "papermill .*Aletheia_NVFP4_Pretrain" >/dev/null 2>&1; then
    echo "already running:"
    pgrep -af "papermill .*Aletheia_NVFP4_Pretrain"
    echo "stop it first if you meant to restart"
    exit 0
fi

if command -v systemd-run >/dev/null 2>&1 && systemctl is-system-running --quiet 2>/dev/null \
   || command -v systemd-run >/dev/null 2>&1 && [ -d /run/systemd/system ]; then
    systemctl reset-failed "$UNIT" 2>/dev/null || true
    systemd-run --unit="$UNIT" --collect --same-dir \
        --property=Type=simple \
        --property=KillMode=mixed \
        --property=TimeoutStopSec=120 \
        bash "$HERE/wsl_pretrain.sh"
    echo "started as $UNIT.service"
    echo "log    : $WORK/logs/pretrain.log"
    echo "journal: journalctl -u $UNIT -f"
    echo "stop   : systemctl stop $UNIT"
else
    setsid nohup bash "$HERE/wsl_pretrain.sh" </dev/null >>"$WORK/logs/launcher.log" 2>&1 &
    disown || true
    echo "started, detached (no systemd -- this dies if the wsl.exe session is reaped)"
    echo "log    : $WORK/logs/pretrain.log"
fi
