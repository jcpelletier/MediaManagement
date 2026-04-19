#!/bin/bash
DRIVE="$1"
STAGING="/mnt/media/Video"
LOG="/var/log/rip-video.log"

source /opt/appinsights/ai-track.sh

echo "[$(date)] ===== DISC DETECTED in $DRIVE =====" >> "$LOG"
ai_event "rip-video" "DiscDetected" "drive=$DRIVE"

# Wait for disc to spin up and become readable
echo "[$(date)] Waiting for disc to spin up..." >> "$LOG"
sleep 10

# Get disc info (retry up to 3 times in case drive is still spinning up)
for attempt in 1 2 3; do
  INFO=$(/snap/bin/makemkvcon -r info dev:"$DRIVE" 2>/dev/null)
  if echo "$INFO" | grep -q "^DRV:"; then
    break
  fi
  echo "[$(date)] Disc not ready yet, retrying ($attempt/3)..." >> "$LOG"
  sleep 5
done

# Extract disc title
DISC_TITLE=$(echo "$INFO" | grep "^DRV:" | grep "$DRIVE" | cut -d'"' -f4)
if [ -z "$DISC_TITLE" ]; then
  DISC_TITLE="Unknown_$(date +%Y%m%d_%H%M%S)"
  echo "[$(date)] WARNING: Could not read disc title, using $DISC_TITLE" >> "$LOG"
  ai_trace "rip-video" "Warning" "Could not read disc title" "drive=$DRIVE" "fallback_title=$DISC_TITLE"
else
  echo "[$(date)] Disc title: $DISC_TITLE" >> "$LOG"
  ai_event "rip-video" "DiscIdentified" "drive=$DRIVE" "disc_title=$DISC_TITLE"
fi

# Get total expected output size (sum of all title sizes, attribute 11)
# TINFO format: TINFO:<title>,<attr>,<int_val>,"<str_val>" — size is in $4 (quoted string)
TOTAL_SIZE=$(echo "$INFO" | grep "^TINFO:" | awk -F',' '$2==11 {gsub(/"/, "", $4); sum+=$4} END {print sum}')
if [ -z "$TOTAL_SIZE" ] || [ "$TOTAL_SIZE" -eq 0 ]; then
  echo "[$(date)] WARNING: Could not determine disc size, progress reporting disabled" >> "$LOG"
  ai_trace "rip-video" "Warning" "Could not determine disc size" "drive=$DRIVE" "disc_title=$DISC_TITLE"
  TOTAL_SIZE=0
else
  TOTAL_HUMAN=$(numfmt --to=iec $TOTAL_SIZE)
  echo "[$(date)] Expected output size: $TOTAL_HUMAN" >> "$LOG"
fi

OUTPUT_DIR="$STAGING/$DISC_TITLE"
mkdir -p "$OUTPUT_DIR"
echo "[$(date)] Output directory: $OUTPUT_DIR" >> "$LOG"
echo "[$(date)] Starting rip..." >> "$LOG"
ai_event "rip-video" "RipStarted" "drive=$DRIVE" "disc_title=$DISC_TITLE" "expected_size=${TOTAL_HUMAN:-unknown}"

# Capture MakeMKV output to a per-rip temp file so we can scan for corruption after
RIP_LOG=$(mktemp /tmp/rip-video-XXXXXX.log)

# Start rip in background
/snap/bin/makemkvcon mkv dev:"$DRIVE" all "$OUTPUT_DIR" >> "$RIP_LOG" 2>&1 &
RIP_PID=$!

# Monitor progress, log each 10% milestone
LAST_MILESTONE=0
while kill -0 $RIP_PID 2>/dev/null; do
  sleep 30
  if [ "$TOTAL_SIZE" -gt 0 ]; then
    CURRENT_SIZE=$(du -sb "$OUTPUT_DIR" 2>/dev/null | cut -f1)
    if [ -n "$CURRENT_SIZE" ] && [ "$CURRENT_SIZE" -gt 0 ]; then
      PCT=$(($CURRENT_SIZE * 100 / $TOTAL_SIZE))
      MILESTONE=$(($PCT / 10 * 10))
      if [ "$MILESTONE" -gt "$LAST_MILESTONE" ]; then
        CURRENT_HUMAN=$(numfmt --to=iec $CURRENT_SIZE)
        TOTAL_HUMAN=$(numfmt --to=iec $TOTAL_SIZE)
        echo "[$(date)] Progress: ~${MILESTONE}% ($CURRENT_HUMAN / $TOTAL_HUMAN)" >> "$LOG"
        ai_event "rip-video" "RipProgress" "disc_title=$DISC_TITLE" "percent=${MILESTONE}" "current=$CURRENT_HUMAN" "total=$TOTAL_HUMAN"
        LAST_MILESTONE=$MILESTONE
      fi
    fi
  fi
done

wait $RIP_PID
STATUS=$?

# Append per-rip log to main log
cat "$RIP_LOG" >> "$LOG"

if [ $STATUS -eq 0 ]; then
  FINAL_SIZE=$(du -sb "$OUTPUT_DIR" 2>/dev/null | cut -f1)
  FINAL_HUMAN=$(numfmt --to=iec $FINAL_SIZE)
  echo "[$(date)] Rip completed successfully -- final size: $FINAL_HUMAN" >> "$LOG"
  ai_event "rip-video" "RipCompleted" "drive=$DRIVE" "disc_title=$DISC_TITLE" "final_size=$FINAL_HUMAN"
else
  echo "[$(date)] ERROR: Rip failed with status $STATUS" >> "$LOG"
  ai_trace "rip-video" "Error" "Rip failed" "drive=$DRIVE" "disc_title=$DISC_TITLE" "exit_status=$STATUS"
fi

# ── Corruption scan ───────────────────────────────────────────────────────────
# Check per-rip log for MakeMKV corruption/error indicators.
CORRUPT_LINES=$(grep -iE \
  "(corrupt|damaged vob|dvdfab|mactheripper|error reading|hash check failed|read error)" \
  "$RIP_LOG" | head -20)
CORRUPT_COUNT=$(grep -ciE \
  "(corrupt|damaged vob|dvdfab|mactheripper|error reading|hash check failed|read error)" \
  "$RIP_LOG" || true)

if [ -n "$CORRUPT_LINES" ]; then
  ai_trace "rip-video" "Warning" "Disc corruption detected" "drive=$DRIVE" "disc_title=$DISC_TITLE" "error_count=$CORRUPT_COUNT"

  # Build a short Discord message
  DISCORD_MSG=$(python3 - "$DISC_TITLE" "$CORRUPT_COUNT" "$CORRUPT_LINES" <<'PYEOF'
import sys
title = sys.argv[1]
count = sys.argv[2]
lines = sys.argv[3]
# First 3 unique lines as examples
examples = list(dict.fromkeys(lines.strip().splitlines()))[:3]
ex_text = "\n".join(f"  • {l.strip()}" for l in examples if l.strip())
print(f"⚠️ Disc rip completed with {count} corruption warning(s) in **{title}**.\nThe video may have glitches where MakeMKV skipped damaged sectors.\n{ex_text}")
PYEOF
)

  /opt/discord-bot/notify-discord.sh \
    "rip-video" "WARNING" "0" "" "$DISCORD_MSG" \
    || echo "[$(date)] WARNING: Could not send Discord corruption alert" >> "$LOG"
fi

rm -f "$RIP_LOG"
# ─────────────────────────────────────────────────────────────────────────────

echo "[$(date)] Ejecting $DRIVE..." >> "$LOG"
eject "$DRIVE"
EJECT_STATUS=$?
if [ $EJECT_STATUS -eq 0 ]; then
  echo "[$(date)] Disc ejected successfully" >> "$LOG"
  ai_event "rip-video" "DiscEjected" "drive=$DRIVE" "disc_title=$DISC_TITLE"
else
  echo "[$(date)] WARNING: Eject failed with status $EJECT_STATUS" >> "$LOG"
  ai_trace "rip-video" "Warning" "Eject failed" "drive=$DRIVE" "eject_status=$EJECT_STATUS"
fi

echo "[$(date)] ===== DONE: $DISC_TITLE =====" >> "$LOG"
