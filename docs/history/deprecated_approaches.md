# Deprecated / archived files

This document tracks files that were active at some point but are now
superseded. They are preserved (not deleted) under `archive/` so that
the historical state of the project — runs, results, scripts that
generated specific figures — can still be reconstructed.

## Launchers

### `archive/launch_brev_jvkq0dvug.sh`

The launch script for Brev instance `jvkq0dvug` (Azure DGX). It
hardcoded `/home/nvidia/haiwen/antonia/struct2seq_bidir_rl` as
`SCRIPT_DIR` and was carried forward through the
SIGABRT/preemption-resume episodes in early May 2026.

**Why archived:** `launch_brev.sh` does everything this script does
(plus auto-discovery of `top2M.csv` in either the project root or
`$HOME`), without the instance-specific path. The flags this script
added (`--skip-play-episode0`, `--save-every-steps`) are also
accepted by `run.py` directly and can be passed to `launch_brev.sh`
through `EXTRA_ARGS`.

**When you might still want it:** to reproduce a specific resume
command from the 2026-05-03 → 2026-05-08 sequence of preemption
restarts (logs under `lab_notebook/2026-05-{03,08}.md`).

## Reserved sections

Add new entries here as more files are archived. Use the format:

```
### `archive/<path>`

What it was. When it was current. Why it was superseded. How to find
the replacement.
```
