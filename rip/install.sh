#!/bin/bash
# install.sh — deploy rip scripts from the MediaManagement repo to /opt/rip/
# Run on panda after pulling the repo:
#
#   sudo bash /path/to/MediaManagement/rip/install.sh
#
# Or from your local machine:
#   ssh panda "cd /opt/MediaManagement && sudo git pull && sudo bash rip/install.sh"

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="/opt/rip"

echo "Installing rip scripts from $SCRIPT_DIR to $DEST..."
mkdir -p "$DEST"

for script in rip-video.sh rip-cd.sh; do
  cp "$SCRIPT_DIR/$script" "$DEST/$script"
  chmod +x "$DEST/$script"
  echo "  Installed $script"
done

echo "Done."
