#!/usr/bin/env bash
# Shared launcher: run one notebook of this folder on a remote machine over SSH and use it in THIS machine's browser.
# Normally called through launch_cdp.sh / launch_stack.sh, which set NOTEBOOK and PORT; it can also be used directly:
#   NOTEBOOK=cdp_nmo_stack_marimo.py PORT=2732 ./launch_remote.sh   any notebook of this folder, on its own port
#   ./launch_remote.sh processing-us                               another ssh host / alias (default "seismic-unix")
#   ./launch_remote.sh seismic-unix run                            app mode (code hidden) instead of the notebook editor
#   CDP_INPUT_FILE=/data/survey.sgy ./launch_remote.sh             file shown in the path box - a path ON THE REMOTE machine
#   REMOTE_DIR=~/qc LOCAL_PORT=8080 ./launch_remote.sh             other remote folder / other port on this machine
#   NO_SYNC=1 ./launch_remote.sh                                   skip copying the code (already up to date there)
# What it does: copy the notebook + sc_scaling_pipeline.py to ~/indvidual_codes on the host -> if no Python there has
# the packages, build a .venv from requirements.txt (internet needed once) -> start marimo on the host's localhost
# only (nothing exposed) -> forward the port through SSH -> open http://localhost:LOCAL_PORT. Ctrl+C stops both.
# Inputs saved in a notebook (cdp_from_header_settings.json, ilxl_corner_table.json) stay on the remote machine.
set -euo pipefail
cd "$(dirname "$0")"

HOST_ALIAS="${1:-${REMOTE_HOST:-seismic-unix}}"
MODE="${2:-edit}"
REMOTE_DIR="${REMOTE_DIR:-indvidual_codes}"      # relative to the remote home unless it starts with /
PORT="${PORT:-2730}"                             # port marimo uses on the remote machine
LOCAL_PORT="${LOCAL_PORT:-$PORT}"                # port you open on this machine
NOTEBOOK="${NOTEBOOK:-plot_cdp_from_header_marimo.py}"
[ -f "$NOTEBOOK" ] || { echo "No notebook '$NOTEBOOK' in $(pwd)" >&2; exit 1; }

if (echo > "/dev/tcp/127.0.0.1/$LOCAL_PORT") >/dev/null 2>&1; then
    echo "Local port $LOCAL_PORT is already in use - pick another: LOCAL_PORT=xxxx $0 $*" >&2; exit 1
fi

SSH=(ssh -x -o ServerAliveInterval=60 -o ExitOnForwardFailure=yes)
echo ">> Host: $HOST_ALIAS   remote folder: $REMOTE_DIR   port: $PORT -> http://localhost:$LOCAL_PORT"
"${SSH[@]}" -o BatchMode=yes -o ConnectTimeout=10 "$HOST_ALIAS" true \
    || { echo "Cannot ssh to '$HOST_ALIAS' (check ~/.ssh/config, key, network)." >&2; exit 1; }

# ---- 1. copy the code (only what the notebook needs; never the local settings file) -------------------------
if [ "${NO_SYNC:-0}" != 1 ]; then
    echo ">> Copying code ..."
    "${SSH[@]}" "$HOST_ALIAS" "mkdir -p $(printf '%q' "$REMOTE_DIR")"
    rsync -az -e "ssh -x" "$NOTEBOOK" sc_scaling_pipeline.py requirements.txt "$HOST_ALIAS:$REMOTE_DIR/"
    # saved IL/XL corner table (stacking notebook): only seeds the remote once, never overwrites what was saved there
    if [ -f ilxl_corner_table.json ]; then
        rsync -az --ignore-existing -e "ssh -x" ilxl_corner_table.json "$HOST_ALIAS:$REMOTE_DIR/"
    fi
fi

# ---- 2. on the remote: find / build a Python with the packages, then start marimo ---------------------------
_q() { printf '%q' "$1"; }
REMOTE_SCRIPT=$(cat <<EOF
set -e
cd $(_q "$REMOTE_DIR")
DEPS="import marimo, numpy, pandas, plotly, matplotlib, scipy"
PY=""
for c in .venv/bin/python python3 python; do
    p="\$(command -v "\$c" 2>/dev/null || true)"
    if [ -n "\$p" ] && "\$p" -c "\$DEPS" >/dev/null 2>&1; then PY="\$p"; break; fi
done
if [ -z "\$PY" ]; then
    echo ">> No Python with the packages found - creating .venv from requirements.txt ..."
    python3 -m venv .venv || { echo "venv failed - install python3 (3.10+) with venv support" >&2; exit 1; }
    .venv/bin/python -m pip install --quiet --upgrade pip
    .venv/bin/python -m pip install --quiet -r requirements.txt
    PY=.venv/bin/python
fi
echo ">> Using \$PY"
# a copy of this notebook left on this port by a dropped connection (tunnel gone, marimo still running): stop it
# (anchored on "python ... -m marimo" so it never matches this shell, whose own command line contains the same words)
if pkill -f "^[^ ]*python[^ ]* -m marimo (edit|run) $NOTEBOOK --host 127.0.0.1 --port $(_q "$PORT") "; then
    echo ">> Stopped the copy left running on port $(_q "$PORT") by an earlier session"; sleep 1
fi
export CDP_INPUT_FILE=$(_q "${CDP_INPUT_FILE:-}")
exec "\$PY" -m marimo $(_q "$MODE") $NOTEBOOK --host 127.0.0.1 --port $(_q "$PORT") --headless --no-token
EOF
)

( # open the page once the tunnel has had time to come up
    sleep 8
    URL="http://localhost:$LOCAL_PORT"
    if command -v open >/dev/null 2>&1; then open "$URL"; elif command -v xdg-open >/dev/null 2>&1; then xdg-open "$URL" >/dev/null 2>&1; fi
) &

echo ">> Starting marimo on $HOST_ALIAS (Ctrl+C here stops it) ..."
exec "${SSH[@]}" -t -L "$LOCAL_PORT:127.0.0.1:$PORT" "$HOST_ALIAS" "bash -c $(_q "$REMOTE_SCRIPT")"
