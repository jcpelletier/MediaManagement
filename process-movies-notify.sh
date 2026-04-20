#!/bin/bash
# process-movies-notify.sh — post-build Discord notification for Process_Movies.
# Reports movies that were added and folders that could not be identified.
#
# Usage (Jenkins post-build shell step):
#   bash "$REPO_DIR/process-movies-notify.sh" \
#       "$JOB_NAME" "$BUILD_RESULT" "$BUILD_NUMBER" "$BUILD_URL" \
#       /tmp/process_movies_summary.json

JOB_NAME="${1:-Process_Movies}"
STATUS="${2:-UNKNOWN}"
BUILD_NUMBER="${3:-0}"
BUILD_URL="${4:-}"
SUMMARY_JSON="${5:-/tmp/process_movies_summary.json}"

if [ ! -f "$SUMMARY_JSON" ]; then
  echo "process-movies-notify.sh: no summary at $SUMMARY_JSON, skipping"
  exit 0
fi

MESSAGE=$(python3 - "$SUMMARY_JSON" "$STATUS" <<'PYEOF'
import json, sys

path, status = sys.argv[1], sys.argv[2]
try:
    with open(path) as f:
        s = json.load(f)
except Exception:
    sys.exit(0)

moved    = s.get("moved_movies", [])
skipped  = s.get("skipped_folders", [])
dry_run  = s.get("dry_run", False)
total    = s.get("total", 0)

lines = []

if moved:
    label = "Would add" if dry_run else "Added"
    lines.append(f"🎬 {label} {len(moved)}/{total} movie(s):")
    for m in moved:
        lines.append(f"  • {m}")

if skipped:
    lines.append(f"\n⚠️ {len(skipped)} folder(s) could not be identified:")
    for item in skipped:
        lines.append(f"  • {item['folder']} — {item['reason']}")

if not lines:
    sys.exit(0)

print("\n".join(lines))
PYEOF
)

if [ -z "$MESSAGE" ]; then
  exit 0
fi

/opt/discord-bot/notify-discord.sh \
  "$JOB_NAME" "$STATUS" "$BUILD_NUMBER" "$BUILD_URL" "$MESSAGE"
