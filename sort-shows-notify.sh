#!/bin/bash
# sort-shows-notify.sh — post-build Discord notification for Sort_Shows.
# Reports added episodes and any files that could not be identified.
#
# Usage (Jenkins post-build "Execute shell" step):
#   bash "$WORKSPACE/sort-shows-notify.sh" \
#       "$JOB_NAME" "$BUILD_RESULT" "$BUILD_NUMBER" "$BUILD_URL" \
#       /tmp/sort_shows_summary.json

JOB_NAME="${1:-Sort_Shows}"
STATUS="${2:-UNKNOWN}"
BUILD_NUMBER="${3:-0}"
BUILD_URL="${4:-}"
SUMMARY_JSON="${5:-/tmp/sort_shows_summary.json}"

if [ ! -f "$SUMMARY_JSON" ]; then
  echo "sort-shows-notify.sh: no summary at $SUMMARY_JSON, skipping"
  exit 0
fi

MESSAGE=$(python3 - "$SUMMARY_JSON" "$STATUS" <<'PYEOF'
import json, sys

path, status = sys.argv[1], sys.argv[2]
try:
    with open(path) as f:
        s = json.load(f)
except Exception as e:
    sys.exit(0)

renamed   = s.get("renamed_episodes", [])
skipped   = s.get("skipped_files", [])
extras    = s.get("extras_moved", [])
dry_run   = s.get("dry_run", False)
total     = s.get("total", 0)

lines = []

if renamed:
    label = "Would add" if dry_run else "Added"
    lines.append(f"✅ {label} {len(renamed)}/{total} episode(s):")
    for ep in renamed:
        lines.append(f"  • {ep}")

if extras:
    label = "Would move" if dry_run else "Moved"
    lines.append(f"\n📦 {label} {len(extras)} file(s) to Extras:")
    for ex in extras:
        lines.append(f"  • {ex}")

if skipped:
    lines.append(f"\n⚠️ {len(skipped)} file(s) could not be handled:")
    for item in skipped:
        lines.append(f"  • {item['file']} — {item['reason']}")

if not lines:
    # Nothing to report (all already named, or truly empty run)
    sys.exit(0)

print("\n".join(lines))
PYEOF
)

if [ -z "$MESSAGE" ]; then
  exit 0
fi

/opt/discord-bot/notify-discord.sh \
  "$JOB_NAME" "$STATUS" "$BUILD_NUMBER" "$BUILD_URL" "$MESSAGE"
