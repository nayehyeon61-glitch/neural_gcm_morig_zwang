#!/usr/bin/env bash
# Kernel cgroup limit, in decimal bytes. No silent fallback to an advisory cap.
set -euo pipefail
if [ "$#" -eq 0 ]; then
  printf '%s\n' 'Usage: scripts/run_96gb.sh .venv/bin/ngcm-mz train ...' >&2
  exit 2
fi
if ! command -v systemd-run >/dev/null 2>&1; then
  printf '%s\n' 'systemd-run is required for this hard 96GB cap. Use a job scheduler/container with a 96000000000-byte memory cgroup instead.' >&2
  exit 2
fi
exec systemd-run --user --scope --quiet -p MemoryMax=96000000000 -p MemorySwapMax=0 -- "$@"
