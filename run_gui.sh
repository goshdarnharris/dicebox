#!/bin/bash
# Launch the DiceBox GUI. Used by the desktop icon and session autostart.
cd /home/user/dicebox
exec python3 GUI.py >> /home/user/dicebox/gui.log 2>&1
