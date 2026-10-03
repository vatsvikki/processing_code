#!/usr/bin/env bash
# Depth image -> time (depth_to_time_marimo.py) on the remote machine -> http://localhost:2736
#   ./launch_depth_to_time.sh                 the notebook on seismic-unix (file paths are paths ON THE REMOTE machine)
#   ./launch_depth_to_time.sh other-host run  other ssh host, app mode (see launch_remote.sh)
cd "$(dirname "$0")"
NOTEBOOK=depth_to_time_marimo.py PORT="${PORT:-2736}" exec ./launch_remote.sh "$@"
