# Disc-routing ground-truth fixtures

Each JSON here is one ripped disc: its pre-sort title list (filename, size,
duration) plus the verified movie-vs-TV routing answer. `tests/test_routing.py`
replays these through `Sort_Rips.classify_disc_routing` (no media, deterministic),
so the movie-vs-TV decision is guarded on every commit. This is the layer that
caught the Paw Patrol compilation disc being misfiled as a movie.

For full **identification** accuracy (which episode / which movie, end to end
against the real library), use `accuracy_test.py` and the `Sort_Accuracy_Test`
Jenkins job — that is the project's identification-accuracy harness.

## Schema (`schema_version: 1`)

```jsonc
{
  "schema_version": 1,
  "disc_title": "PAW PATROL_ PUPS SAVE PUPLANTIS#9B2B", // raw disc/folder name the sorter sees
  "verified": true,                  // only verified fixtures are scored
  "expected_routing": "tv",          // "tv" | "movie" — the looks_like_tv_disc decision
  "titles": [
    { "src": "B2_t01.mkv", "size_bytes": 521223393, "duration_s": 738 }
  ]
}
```

Only `expected_routing` plus each title's `size_bytes` / `duration_s` are needed
for the routing test. Other fields (episode/movie labels, notes) are
informational and unused by the test.

## Adding a fixture

Record a disc's title list (sizes + durations via `ffprobe`) and the correct
routing, set `verified: true`, and commit. `rip-video.sh` also writes a
`rip_manifest.json` per rip (next to the rip and to `/mnt/media/rip_manifests`)
capturing this raw title list automatically — copy from there.
