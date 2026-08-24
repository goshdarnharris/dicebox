#!/bin/bash
# Diagnostic: log DSI screen power transitions plus a process snapshot at
# each change, to help figure out what keeps blanking the touchscreen.
# Usage: ./screenmon.sh [output]   (default output: DSI-2)
# Logs to ~/screenmon.log. Runs forever.
export XDG_RUNTIME_DIR=/run/user/1000 WAYLAND_DISPLAY=wayland-0

OUTPUT=${1:-DSI-2}
LOG="$HOME/screenmon.log"

echo "=== monitor start $(date) ===" >> "$LOG"
last=""
while true; do
    state=$(wlopm 2>/dev/null | awk -v o="$OUTPUT" '$1==o {print $2}')
    if [ "$state" != "$last" ]; then
        {
            echo "$(date '+%F %T') $OUTPUT -> ${state:-unknown}"
            ps aux --sort=-%cpu | head -8
            echo "---"
        } >> "$LOG"
        last="$state"
    fi
    sleep 2
done
