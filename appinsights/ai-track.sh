#!/bin/bash
# /opt/appinsights/ai-track.sh
# Source this file in bash scripts to get ai_event and ai_trace functions.
# All calls are fire-and-forget (background) — they won't block or fail the parent script.
#
# Usage:
#   source /opt/appinsights/ai-track.sh
#   ai_event  "rip-video" "RipStarted"    disc_title="The Matrix" expected_size="25GB"
#   ai_trace  "rip-video" "Warning"       "Could not read disc title" drive=/dev/sr0
#
# Severity levels for ai_trace: Verbose, Information, Warning, Error, Critical

AI_IKEY="0e6fc7c0-a8ec-491c-beb4-9114e950f261"
AI_ENDPOINT="https://eastus-8.in.applicationinsights.azure.com/v2/track"

# ai_event ROLE EVENT_NAME [key=value ...]
ai_event() {
  local role="$1" event_name="$2"
  shift 2
  python3 - "$role" "$event_name" "$@" <<'PYEOF' &
import sys, json, urllib.request, datetime

role       = sys.argv[1]
event_name = sys.argv[2]
props      = dict(kv.split("=", 1) for kv in sys.argv[3:] if "=" in kv)

AI_IKEY     = "0e6fc7c0-a8ec-491c-beb4-9114e950f261"
AI_ENDPOINT = "https://eastus-8.in.applicationinsights.azure.com/v2/track"

payload = [{
    "name": "Microsoft.ApplicationInsights.Event",
    "time": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.000Z"),
    "iKey": AI_IKEY,
    "tags": {"ai.cloud.roleName": role, "ai.device.type": "Other"},
    "data": {
        "baseType": "EventData",
        "baseData": {"ver": 2, "name": event_name, "properties": props}
    }
}]

try:
    req = urllib.request.Request(
        AI_ENDPOINT,
        json.dumps(payload).encode(),
        {"Content-Type": "application/json"}
    )
    urllib.request.urlopen(req, timeout=5)
except Exception:
    pass  # never fail the parent script
PYEOF
}

# ai_trace ROLE SEVERITY MESSAGE [key=value ...]
# SEVERITY: Verbose | Information | Warning | Error | Critical
ai_trace() {
  local role="$1" severity="$2" message="$3"
  shift 3
  python3 - "$role" "$severity" "$message" "$@" <<'PYEOF' &
import sys, json, urllib.request, datetime

role     = sys.argv[1]
severity = sys.argv[2].lower()
message  = sys.argv[3]
props    = dict(kv.split("=", 1) for kv in sys.argv[4:] if "=" in kv)

AI_IKEY     = "0e6fc7c0-a8ec-491c-beb4-9114e950f261"
AI_ENDPOINT = "https://eastus-8.in.applicationinsights.azure.com/v2/track"

level_map = {"verbose": 0, "information": 1, "warning": 2, "error": 3, "critical": 4}
level_int = level_map.get(severity, 1)

payload = [{
    "name": "Microsoft.ApplicationInsights.Message",
    "time": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.000Z"),
    "iKey": AI_IKEY,
    "tags": {"ai.cloud.roleName": role, "ai.device.type": "Other"},
    "data": {
        "baseType": "MessageData",
        "baseData": {
            "ver": 2,
            "message": message,
            "severityLevel": level_int,
            "properties": props
        }
    }
}]

try:
    req = urllib.request.Request(
        AI_ENDPOINT,
        json.dumps(payload).encode(),
        {"Content-Type": "application/json"}
    )
    urllib.request.urlopen(req, timeout=5)
except Exception:
    pass
PYEOF
}
