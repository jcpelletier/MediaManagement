#!/bin/bash
DRIVE="$1"
STAGING="/mnt/media/Video"
LOG="/var/log/rip-video.log"
# Durable archive of rip manifests (raw "what each title was named" provenance).
# Kept outside the staging tree so it survives sorting + cleanup; this is the pool
# we later hand-label a sample of for the sort-accuracy tests.
MANIFEST_ARCHIVE="/mnt/media/rip_manifests"

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

# Start rip in background
/snap/bin/makemkvcon mkv dev:"$DRIVE" all "$OUTPUT_DIR" >> "$LOG" 2>&1 &
RIP_PID=$!

# Monitor progress, log each 10% milestone
LAST_MILESTONE=0
while kill -0 $RIP_PID 2>/dev/null; do
  sleep 30
  if [ "$TOTAL_SIZE" -gt 0 ]; then
    CURRENT_SIZE=$(du -sb "$OUTPUT_DIR" 2>/dev/null | cut -f1)
    if [ -n "$CURRENT_SIZE" ] && [ "$CURRENT_SIZE" -gt 0 ]; then
      PCT=$((CURRENT_SIZE * 100 / TOTAL_SIZE))
      MILESTONE=$((PCT / 10 * 10))
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

if [ $STATUS -eq 0 ]; then
  FINAL_SIZE=$(du -sb "$OUTPUT_DIR" 2>/dev/null | cut -f1)
  FINAL_HUMAN=$(numfmt --to=iec $FINAL_SIZE)
  echo "[$(date)] Rip completed successfully -- final size: $FINAL_HUMAN" >> "$LOG"
  ai_event "rip-video" "RipCompleted" "drive=$DRIVE" "disc_title=$DISC_TITLE" "final_size=$FINAL_HUMAN"
else
  echo "[$(date)] ERROR: Rip failed with status $STATUS" >> "$LOG"
  ai_trace "rip-video" "Error" "Rip failed" "drive=$DRIVE" "disc_title=$DISC_TITLE" "exit_status=$STATUS"
fi

# Record raw provenance for the sort-accuracy dataset on EVERY rip (success or
# partial). This is only the pre-sort input the sorter will see: the disc title
# plus each title's output filename, duration, and size (from the makemkvcon info
# blob in $INFO). It contains NO labels; humans hand-label a sample of these
# later for the accuracy tests. A copy lives next to the rip and a durable copy
# goes to $MANIFEST_ARCHIVE so it survives sorting + cleanup. Best-effort: never
# let manifest writing fail the rip.
if command -v python3 >/dev/null 2>&1; then
  mkdir -p "$MANIFEST_ARCHIVE" 2>/dev/null
  INFO_TMP=$(mktemp)
  printf '%s' "$INFO" > "$INFO_TMP"
  python3 - "$INFO_TMP" "$OUTPUT_DIR" "$DISC_TITLE" "$DRIVE" "$MANIFEST_ARCHIVE" <<'PYEOF' >> "$LOG" 2>&1 || echo "[$(date)] WARNING: rip_manifest.json not written" >> "$LOG"
import json, os, re, sys
from datetime import date

info_path, out_dir, disc_title, drive, archive_dir = sys.argv[1:6]

# makemkvcon -r info TINFO format: TINFO:<title>,<attr>,<code>,"<value>"
# attr 9 = duration (H:MM:SS), 11 = size in bytes, 27 = output file name.
ATTR_DURATION, ATTR_SIZE, ATTR_FILENAME = 9, 11, 27
row = re.compile(r'^TINFO:(\d+),(\d+),\d+,"(.*)"\s*$')

titles = {}
with open(info_path, encoding="utf-8", errors="ignore") as fh:
    for line in fh:
        m = row.match(line.strip())
        if not m:
            continue
        tid, attr, val = int(m.group(1)), int(m.group(2)), m.group(3)
        titles.setdefault(tid, {})[attr] = val

def dur_to_seconds(s):
    parts = (s or "").split(":")
    try:
        parts = [int(p) for p in parts]
    except ValueError:
        return None
    sec = 0
    for p in parts:
        sec = sec * 60 + p
    return sec if sec > 0 else None

out_titles = []
for tid in sorted(titles):
    attrs = titles[tid]
    src = attrs.get(ATTR_FILENAME)
    info_size = attrs.get(ATTR_SIZE)
    info_size = int(info_size) if (info_size or "").isdigit() else None
    # A title can fail to save (copy-protected / bad sector). Reflect what is
    # actually on disk so a fixture only contains files the sorter really sees,
    # and prefer the real on-disk size when present.
    path = os.path.join(out_dir, src) if src else None
    saved = bool(path and os.path.isfile(path))
    size_bytes = os.path.getsize(path) if saved else info_size
    out_titles.append({
        "index": tid,
        "src": src,
        "duration_s": dur_to_seconds(attrs.get(ATTR_DURATION)),
        "size_bytes": size_bytes,
        "saved": saved,
    })

manifest = {
    "schema_version": 1,
    "disc_title": disc_title,
    "ripped_at": date.today().isoformat(),
    "source": "makemkv",
    "drive": drive,
    "titles": out_titles,
}

payload = json.dumps(manifest, indent=2)
with open(os.path.join(out_dir, "rip_manifest.json"), "w", encoding="utf-8") as fh:
    fh.write(payload)

# Durable archive copy, keyed by disc title + date so re-rips do not clobber.
safe = re.sub(r"[^A-Za-z0-9._-]+", "_", disc_title).strip("_") or "disc"
archive_name = f"{safe}__{date.today().isoformat()}.json"
try:
    with open(os.path.join(archive_dir, archive_name), "w", encoding="utf-8") as fh:
        fh.write(payload)
except OSError as e:
    print(f"WARNING: could not archive manifest: {e}")

saved_n = sum(1 for t in out_titles if t["saved"])
print(f"rip_manifest.json written: {len(out_titles)} titles ({saved_n} saved)")
PYEOF
  rm -f "$INFO_TMP"
fi

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
