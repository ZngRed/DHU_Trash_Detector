#!/bin/bash

# export QT_DEBUG_PLUGINS=1

echo password | sudo -S chmod 777 /dev/ttyUSB0

gnome-terminal -t "trash" -x bash -c "python3 /home/dhu/TrashDetector/main.py;exec bash"

exit 0