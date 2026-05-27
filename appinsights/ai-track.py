"""
/opt/appinsights/ai-track.py

Thin Application Insights client for Jenkins jobs and Python scripts.
Uses only stdlib — no SDK dependency — so it works inside any venv.

As a module:
    from ai_track import AiTrack
    ai = AiTrack("jenkins-process-movies")
    ai.event("JobStarted", source_dir="/mnt/media/Video", job=os.environ.get("JOB_NAME",""))
    ai.trace("Information", "Processing 12 files")
    ai.trace("Error", "ffmpeg failed", file="movie.mkv", exit_code="1")

As a CLI (from shell / Jenkins post-build step):
    python3 /opt/appinsights/ai-track.py \\
        --role jenkins --event BuildCompleted \\
        --props '{"job_name":"Login_Test","result":"FAILURE","build_number":"42"}'
"""

import json
import sys
import urllib.request
import datetime
import argparse

AI_IKEY     = "0e6fc7c0-a8ec-491c-beb4-9114e950f261"
AI_ENDPOINT = "https://eastus-8.in.applicationinsights.azure.com/v2/track"

SEVERITY_MAP = {
    "verbose":     0,
    "information": 1,
    "warning":     2,
    "error":       3,
    "critical":    4,
}


def _post(payload: list) -> None:
    """Fire-and-forget POST to App Insights. Never raises."""
    try:
        req = urllib.request.Request(
            AI_ENDPOINT,
            json.dumps(payload).encode(),
            {"Content-Type": "application/json"},
        )
        urllib.request.urlopen(req, timeout=5)
    except Exception:
        pass


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")


class AiTrack:
    """Lightweight App Insights client bound to a role name (source)."""

    def __init__(self, role: str):
        self.role = role

    def _tags(self) -> dict:
        return {"ai.cloud.roleName": self.role, "ai.device.type": "Other"}

    def event(self, name: str, **props) -> None:
        """Send a custom event with optional string properties."""
        _post([{
            "name": "Microsoft.ApplicationInsights.Event",
            "time": _now(),
            "iKey": AI_IKEY,
            "tags": self._tags(),
            "data": {
                "baseType": "EventData",
                "baseData": {
                    "ver": 2,
                    "name": name,
                    "properties": {k: str(v) for k, v in props.items()},
                },
            },
        }])

    def trace(self, severity: str, message: str, **props) -> None:
        """Send a trace (log message). severity: Verbose|Information|Warning|Error|Critical"""
        level = SEVERITY_MAP.get(severity.lower(), 1)
        _post([{
            "name": "Microsoft.ApplicationInsights.Message",
            "time": _now(),
            "iKey": AI_IKEY,
            "tags": self._tags(),
            "data": {
                "baseType": "MessageData",
                "baseData": {
                    "ver": 2,
                    "message": message,
                    "severityLevel": level,
                    "properties": {k: str(v) for k, v in props.items()},
                },
            },
        }])

    def exception(self, message: str, **props) -> None:
        """Send an exception / error event."""
        _post([{
            "name": "Microsoft.ApplicationInsights.Exception",
            "time": _now(),
            "iKey": AI_IKEY,
            "tags": self._tags(),
            "data": {
                "baseType": "ExceptionData",
                "baseData": {
                    "ver": 2,
                    "exceptions": [{
                        "typeName": "ScriptError",
                        "message": message,
                        "hasFullStack": False,
                    }],
                    "properties": {k: str(v) for k, v in props.items()},
                },
            },
        }])


# ---------------------------------------------------------------------------
# CLI entry point — used by Jenkins post-build shell steps
# ---------------------------------------------------------------------------
# Examples:
#   python3 /opt/appinsights/ai-track.py --role jenkins --event BuildCompleted \
#       --props '{"job_name":"Login_Test","result":"FAILURE","build_number":"42"}'
#
#   python3 /opt/appinsights/ai-track.py --role jenkins --trace Warning \
#       --message "Build unstable" --props '{"job_name":"Process_Movies"}'

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Send a telemetry item to App Insights")
    parser.add_argument("--role",    required=True, help="Source role name (e.g. jenkins, rip-video)")
    parser.add_argument("--event",   help="Custom event name")
    parser.add_argument("--trace",   help="Trace severity (Verbose|Information|Warning|Error|Critical)")
    parser.add_argument("--message", help="Message text (for --trace)")
    parser.add_argument("--props",   default="{}", help="JSON object of custom properties")
    args = parser.parse_args()

    try:
        props = json.loads(args.props)
    except json.JSONDecodeError as e:
        print(f"ERROR: --props is not valid JSON: {e}", file=sys.stderr)
        sys.exit(1)

    ai = AiTrack(args.role)

    if args.event:
        ai.event(args.event, **props)
        print(f"Sent event '{args.event}' for role '{args.role}'")
    elif args.trace:
        if not args.message:
            print("ERROR: --message is required with --trace", file=sys.stderr)
            sys.exit(1)
        ai.trace(args.trace, args.message, **props)
        print(f"Sent trace [{args.trace}] for role '{args.role}'")
    else:
        print("ERROR: specify --event or --trace", file=sys.stderr)
        sys.exit(1)
