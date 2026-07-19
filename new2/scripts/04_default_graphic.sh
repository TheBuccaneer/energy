#!/bin/bash
set -euo pipefail
echo "Switching back to graphical mode."
sudo systemctl isolate graphical.target
