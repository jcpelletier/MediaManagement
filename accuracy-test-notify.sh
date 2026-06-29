#!/bin/bash
# accuracy-test-notify.sh - post-build Discord summary for the Sort accuracy test.
# Reports per-index identification accuracy for Sort_Rips and Sort_TV.
#
# Usage (Jenkins post-build "Execute shell" step):
#   bash "$WORKSPACE/accuracy-test-notify.sh" \
#       "$JOB_NAME" "$BUILD_RESULT" "$BUILD_NUMBER" "$BUILD_URL" \
#       /tmp/accuracy_test_summary.json

JOB_NAME="${1:-Sort_Accuracy_Test}"
STATUS="${2:-UNKNOWN}"
BUILD_NUMBER="${3:-0}"
BUILD_URL="${4:-}"
SUMMARY_JSON="${5:-/tmp/accuracy_test_summary.json}"

if [ ! -f "$SUMMARY_JSON" ]; then
  echo "accuracy-test-notify.sh: no summary at $SUMMARY_JSON, skipping"
  exit 0
fi

MESSAGE=$(python3 - "$SUMMARY_JSON" <<'PYEOF'
import json, sys

path = sys.argv[1]
try:
    with open(path) as f:
        s = json.load(f)
except Exception:
    sys.exit(0)

results = s.get("results", {})
limit = s.get("limit")
scope = "ALL" if limit is None else str(limit)


def pct(num, den):
    return f"{(100.0 * num / den):.1f}%" if den else "n/a"


# Title-hint indexes mix cohorts of differing label quality; report the span
# (worst..best) so a 100% realistic cohort is not hidden behind a degraded one.
COHORT_NAMES = {"real": "real label", "cohortA": "full-title", "cohortB": "initials"}


def cohort_breakdown(items):
    bd = {}
    for it in items:
        s = it.get("label_source") or "?"
        agg = bd.setdefault(s, [0, 0])
        agg[1] += 1
        if it.get("correct_title"):
            agg[0] += 1
    return {s: (c, n) for s, (c, n) in bd.items() if s in COHORT_NAMES and n}


lines = [f"Accuracy test (scope: {scope} per script/index)"]

rips = results.get("rips", {})
if rips:
    lines.append("\n🎬 Sort_Rips:")
    for idx in sorted(rips):
        r = rips[idx]
        t = r.get("total", 0)
        label = r.get("label", f"Index {idx}")
        hint = cohort_breakdown(r.get("items", []))
        if len(hint) >= 2:
            order = [s for s in ("real", "cohortA", "cohortB") if s in hint]
            pcts = [100.0 * hint[s][0] / hint[s][1] for s in order]
            detail = ", ".join(f"{COHORT_NAMES[s]} {hint[s][0]}/{hint[s][1]}" for s in order)
            lines.append(f"  • {label}: {min(pcts):.0f}%–{max(pcts):.0f}%  ({detail})")
        else:
            lines.append(
                f"  • {label}: title {r.get('title_correct',0)}/{t} "
                f"({pct(r.get('title_correct',0), t)}), "
                f"+year {r.get('title_year_correct',0)}/{t}, miss {r.get('miss',0)}"
            )

tv = results.get("tv", {})
if tv:
    lines.append("\n📺 Sort_TV:")
    for idx in sorted(tv):
        r = tv[idx]
        t = r.get("total", 0)
        label = r.get("label", f"Index {idx}")
        lines.append(
            f"  • {label}: episode {r.get('episode_correct',0)}/{t} "
            f"({pct(r.get('episode_correct',0), t)}), "
            f"season {r.get('season_correct',0)}/{t}, "
            f"extras {r.get('extras',0)}, miss {r.get('miss',0)}"
        )

if len(lines) <= 1:
    sys.exit(0)

print("\n".join(lines))
PYEOF
)

if [ -z "$MESSAGE" ]; then
  exit 0
fi

/opt/discord-bot/notify-discord.sh \
  "$JOB_NAME" "$STATUS" "$BUILD_NUMBER" "$BUILD_URL" "$MESSAGE"
