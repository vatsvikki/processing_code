#!/usr/bin/env bash
# CDP / NMO stacking notebook (cdp_nmo_stack_marimo.py) on the remote machine -> http://localhost:2732
#   CDP_INPUT_FILE=/home/ubuntu/data/.../file.sgy ./launch_stack.sh    (file path ON THE REMOTE machine)
#   ./launch_stack.sh other-host run                                   other ssh host, app mode (see launch_remote.sh)
cd "$(dirname "$0")"
NOTEBOOK=cdp_nmo_stack_marimo.py PORT="${PORT:-2732}" exec ./launch_remote.sh "$@"
