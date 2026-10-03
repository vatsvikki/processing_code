#!/usr/bin/env bash
# CDP recalculation notebook (plot_cdp_from_header_marimo.py) on the remote machine -> http://localhost:2730
#   CDP_INPUT_FILE=/home/ubuntu/data/.../file.sgy ./launch_cdp.sh      (file path ON THE REMOTE machine)
#   ./launch_cdp.sh other-host run                                     other ssh host, app mode (see launch_remote.sh)
cd "$(dirname "$0")"
NOTEBOOK=plot_cdp_from_header_marimo.py PORT="${PORT:-2730}" exec ./launch_remote.sh "$@"
