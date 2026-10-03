#!/usr/bin/env bash
# Migration velocity converter (migration_velocity_marimo.py) on the remote machine -> http://localhost:2734
#   ./launch_velocity.sh                 the notebook on seismic-unix (type the velocity file path - a path ON THE REMOTE machine)
#   ./launch_velocity.sh other-host run  other ssh host, app mode (see launch_remote.sh)
cd "$(dirname "$0")"
NOTEBOOK=migration_velocity_marimo.py PORT="${PORT:-2734}" exec ./launch_remote.sh "$@"
