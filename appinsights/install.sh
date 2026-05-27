#!/bin/bash
# Installs App Insights helpers and deploys updated rip scripts.
# Run as root: sudo bash install.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

echo "==> Installing App Insights helpers to /opt/appinsights/"
mkdir -p /opt/appinsights
cp "$SCRIPT_DIR/ai-track.sh" /opt/appinsights/ai-track.sh
cp "$SCRIPT_DIR/ai-track.py" /opt/appinsights/ai-track.py
chmod +x /opt/appinsights/ai-track.sh /opt/appinsights/ai-track.py
chown -R root:root /opt/appinsights

echo "==> Deploying updated rip-video.sh"
cp "$REPO_ROOT/rip-video.sh" /opt/rip/rip-video.sh
chmod +x /opt/rip/rip-video.sh

echo "==> Deploying updated rip-cd.sh"
cp "$REPO_ROOT/rip-cd.sh" /opt/rip/rip-cd.sh
chmod +x /opt/rip/rip-cd.sh

echo "==> Sending test event to App Insights..."
python3 /opt/appinsights/ai-track.py \
  --role "panda-server" \
  --event "AppInsightsInstalled" \
  --props '{"host":"panda","version":"1.0"}' \
  && echo "    Test event sent OK" \
  || echo "    WARNING: test event failed — check network / connection string"

echo ""
echo "Done. Next: rebuild the Jenkins container to pick up opencensus-ext-azure."
echo "  cd /opt/jenkins && sudo docker compose build && sudo docker compose up -d"
