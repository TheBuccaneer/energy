#!/bin/bash
set -euo pipefail
echo "Switching to text mode. The graphical session will close now."
sudo systemctl isolate multi-user.target
