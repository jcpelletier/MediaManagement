#!/bin/bash
DRIVE="$1"
LOG="/var/log/rip-cd.log"

source /opt/appinsights/ai-track.sh

echo "[$(date)] ===== DISC DETECTED in $DRIVE =====" >> "$LOG"
ai_event "rip-cd" "DiscDetected" "drive=$DRIVE"

echo "[$(date)] Starting CD rip..." >> "$LOG"
ai_event "rip-cd" "RipStarted" "drive=$DRIVE"

# Timestamp reference: any directory newer than this was created by this rip
TIMESTAMP_REF=$(mktemp)

abcde -N -d "$DRIVE" -c /etc/abcde-rip.conf >> "$LOG" 2>&1
STATUS=$?

if [ $STATUS -eq 0 ]; then
  # Find the album directory created during this rip
  ALBUM_DIR=$(find /mnt/media/Media/Music -mindepth 2 -maxdepth 2 -type d -newer "$TIMESTAMP_REF" 2>/dev/null | head -1)
  if [ -n "$ALBUM_DIR" ]; then
    ALBUM=$(basename "$ALBUM_DIR")
    ARTIST=$(basename "$(dirname "$ALBUM_DIR")")
    TRACK_COUNT=$(find "$ALBUM_DIR" -name "*.flac" | wc -l)
    echo "[$(date)] Rip completed: $ARTIST / $ALBUM ($TRACK_COUNT tracks)" >> "$LOG"
    ai_event "rip-cd" "RipCompleted" "drive=$DRIVE" "artist=$ARTIST" "album=$ALBUM" "track_count=$TRACK_COUNT"
    /opt/discord-bot/notify-discord.sh "rip-cd" "SUCCESS" "0" "" \
      "🎵 Ripped **$ARTIST / $ALBUM** ($TRACK_COUNT track(s))"
  else
    echo "[$(date)] Rip completed (output directory not found)" >> "$LOG"
    ai_event "rip-cd" "RipCompleted" "drive=$DRIVE"
    /opt/discord-bot/notify-discord.sh "rip-cd" "SUCCESS" "0" "" \
      "🎵 CD rip completed (album directory not found)"
  fi
else
  echo "[$(date)] ERROR: Rip failed with exit status $STATUS" >> "$LOG"
  ai_trace "rip-cd" "Error" "CD rip failed" "drive=$DRIVE" "exit_status=$STATUS"
  /opt/discord-bot/notify-discord.sh "rip-cd" "FAILURE" "0" "" \
    "❌ CD rip failed (exit status $STATUS)"
fi

rm -f "$TIMESTAMP_REF"

echo "[$(date)] Ejecting $DRIVE..." >> "$LOG"
eject "$DRIVE"
EJECT_STATUS=$?
if [ $EJECT_STATUS -eq 0 ]; then
  echo "[$(date)] Disc ejected" >> "$LOG"
  ai_event "rip-cd" "DiscEjected" "drive=$DRIVE"
else
  echo "[$(date)] WARNING: Eject failed with status $EJECT_STATUS" >> "$LOG"
  ai_trace "rip-cd" "Warning" "Eject failed" "drive=$DRIVE" "eject_status=$EJECT_STATUS"
fi

echo "[$(date)] ===== DONE =====" >> "$LOG"
