#!/bin/bash
DRIVE="$1"
LOG="/var/log/rip-cd.log"

source /opt/appinsights/ai-track.sh

echo "[$(date)] ===== DISC DETECTED in $DRIVE =====" >> "$LOG"
ai_event "rip-cd" "DiscDetected" "drive=$DRIVE"

echo "[$(date)] Starting CD rip..." >> "$LOG"
ai_event "rip-cd" "RipStarted" "drive=$DRIVE"

abcde -N -d "$DRIVE" -c /etc/abcde-rip.conf >> "$LOG" 2>&1
STATUS=$?

if [ $STATUS -eq 0 ]; then
  echo "[$(date)] Rip completed successfully" >> "$LOG"
  ai_event "rip-cd" "RipCompleted" "drive=$DRIVE"
else
  echo "[$(date)] ERROR: Rip failed with status $STATUS" >> "$LOG"
  ai_trace "rip-cd" "Error" "CD rip failed" "drive=$DRIVE" "exit_status=$STATUS"
fi

echo "[$(date)] Ejecting $DRIVE..." >> "$LOG"
eject "$DRIVE"
EJECT_STATUS=$?
if [ $EJECT_STATUS -eq 0 ]; then
  echo "[$(date)] Disc ejected successfully" >> "$LOG"
  ai_event "rip-cd" "DiscEjected" "drive=$DRIVE"
else
  echo "[$(date)] WARNING: Eject failed with status $EJECT_STATUS" >> "$LOG"
  ai_trace "rip-cd" "Warning" "Eject failed" "drive=$DRIVE" "eject_status=$EJECT_STATUS"
fi

echo "[$(date)] ===== DONE =====" >> "$LOG"
