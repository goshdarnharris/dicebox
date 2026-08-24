#!/bin/bash
# Workaround for flaky DSI screen power management on the dicebox:
# poll the touchscreen's power state and force it back on if it drops.
# Usage: ./screen-keepalive.sh [output]   (default output: DSI-2)
# Logs transitions to ~/screen-keepalive.log. Runs forever.
export XDG_RUNTIME_DIR=/run/user/1000 WAYLAND_DISPLAY=wayland-0

OUTPUT=${1:-DSI-2}
LOG="$HOME/screen-keepalive.log"
INTERVAL=5

echo "=== keepalive start $(date) ===" >> "$LOG"
while true; do
    state=$(wlopm 2>/dev/null | awk -v o="$OUTPUT" '$1==o {print $2}')
    if [ "$state" = "off" ]; then
        echo "$(date '+%F %T') $OUTPUT was off - turning on" >> "$LOG"
        wlopm --on "$OUTPUT" 2>> "$LOG"
    fi
    sleep "$INTERVAL"
done
