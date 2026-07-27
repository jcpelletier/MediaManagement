# AGENTS.md — MediaManagement

AI coding agent instructions for this repository.

## What this repo is

Jenkins pipeline scripts for the panda home server media library
(`jcpelletier/MediaManagement`). Python scripts that run inside Jenkins jobs to sort,
convert, and manage the media library on the server.

## Architecture

| Script | Jenkins job | What it does |
|---|---|---|
| `Sort_Rips.py` | Process_Movies (midnight) | Moves staged video to library, TMDB identification |
| `Extract_Subs.py` | Nightly_Convert (3am) | Extracts subtitles from MKV files |
| `Convert_Video.py` | Nightly_Convert (3am) | Re-encodes to h264_nvenc |
| `Sort_TV.py` | Process_Movies | TV episode sorting |
| `Fetch_Subs.py` | Nightly_Convert | Fetches subtitle files |
| `AudioSync_Subs.py` | Nightly_Convert | Subtitle audio sync |
| `accuracy_test.py` | Sort_Accuracy_Test (manual) | Measures Sort_Rips/Sort_TV identification accuracy against the library via obfuscated hardlink staging |

## Key rules

**TMDB API version** — `Sort_Rips.py` uses TMDB **v3** API. Authentication is via
`api_key=` query param — do NOT use a Bearer token (v4 JWT). This is a known gotcha;
wrong auth returns 401.

**No Discord or bot logic** — these are standalone pipeline scripts. They should not
import discord.py or pandabot_core.

**GPU transcoding** — `Convert_Video.py` uses `h264_nvenc` (NVIDIA hardware encoder on
the GTX 970). Do not change the encoder to software (`libx264`) — the Jenkins container
has GPU passthrough specifically for this.

**Media paths** — staging area is `/mnt/media/Video`; sorted library is under
`/mnt/media/Media/` (Movies, Shows, Music subdirs). Do not hardcode other paths.

**Jenkins notification** — jobs call `jenkins-notify.sh` post-build for App Insights and
Discord failure alerts. Do not remove or break this call.

## Running tests

No automated test suite. Changes are validated by triggering Jenkins jobs manually
or watching the next scheduled run.

## Files never to modify

- Any `.env` or credential files (not in repo)
- Jenkins job configurations (managed via Jenkins UI, not this repo)

## Deployment

PRs target `main`. Jenkins jobs pull from `main` at build time. The server also has a
clone at `/opt/MediaManagement/` — run `sudo git pull origin main` there after merging
if you need scripts outside Jenkins to pick up changes immediately.

Full context: see `CLAUDE.md` in the parent PandaMigration repo.
