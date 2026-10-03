#!/usr/bin/env bash
# Run the Trace QC app on a remote machine over SSH and use it in THIS machine's browser.
#   ./launch_remote.sh                                   host "seismic-unix" (from ~/.ssh/config), code in ~/processing
#   ./launch_remote.sh other-host                        another ssh host / alias
#   SEGY_FILE=/data/survey.sgy ./launch_remote.sh        file to open first - a path ON THE REMOTE machine
#   REMOTE_DIR=~/qc PORT=2718 LOCAL_PORT=8080 ./launch_remote.sh
#   ./launch_remote.sh host edit                         notebook editor instead of app mode
#   NO_SYNC=1 ./launch_remote.sh                         skip copying the code (already up to date there)
# What it does: rsync this folder to the host -> start ./run_app.sh there (localhost only, nothing exposed)
# -> forward the port through the SSH connection -> open http://localhost:LOCAL_PORT. Ctrl+C stops app and tunnel.
# First run on the host builds a .venv from requirements.txt if needed (Ubuntu: `sudo apt install python3-venv`).
set -euo pipefail
cd "$(dirname "$0")"

HOST_ALIAS="${1:-${REMOTE_HOST:-seismic-unix}}"
MODE="${2:-run}"
REMOTE_DIR="${REMOTE_DIR:-processing}"          # relative to the remote home unless it starts with /
PORT="${PORT:-2718}"                            # port the app uses on the remote machine
LOCAL_PORT="${LOCAL_PORT:-$PORT}"               # port you open on this machine

# an earlier launch of this app still holding the port (e.g. left open in another terminal): stop it - this launch
# replaces it. Anything else on the port is left alone (pick another port then).
_old=""
for _p in $(lsof -nP -t -iTCP:"$LOCAL_PORT" -sTCP:LISTEN 2>/dev/null || true); do
    _c="$(ps -o command= -p "$_p" 2>/dev/null || true)"
    if printf '%s' "$_c" | grep -q -- "-L $LOCAL_PORT:127.0.0.1:.*run_app.sh"; then
        _old="$_old $_p"
    fi
done
if [ -n "$_old" ]; then
    echo ">> Stopping the earlier session of this app on port $LOCAL_PORT (process$_old)"
    kill $_old 2>/dev/null || true
    for _ in 1 2 3 4 5 6 7 8 9 10; do (echo > "/dev/tcp/127.0.0.1/$LOCAL_PORT") >/dev/null 2>&1 || break; sleep 1; done
    sleep 2                                         # let the app on the remote shut down too
fi
if (echo > "/dev/tcp/127.0.0.1/$LOCAL_PORT") >/dev/null 2>&1; then
    echo "Local port $LOCAL_PORT is already in use - pick another: LOCAL_PORT=xxxx $0 $*" >&2; exit 1
fi
# notebook tools (🛠 Tools menu) run on the next port, on both sides: the app links to <its own port> + 1
if (echo > "/dev/tcp/127.0.0.1/$((LOCAL_PORT + 1))") >/dev/null 2>&1; then
    echo "Local port $((LOCAL_PORT + 1)) (notebook tools) is already in use - pick another: LOCAL_PORT=xxxx $0 $*" >&2; exit 1
fi

SSH=(ssh -x -o ServerAliveInterval=60 -o ExitOnForwardFailure=yes)
echo ">> Host: $HOST_ALIAS   remote folder: $REMOTE_DIR   app port: $PORT -> http://localhost:$LOCAL_PORT"
"${SSH[@]}" -o BatchMode=yes -o ConnectTimeout=10 "$HOST_ALIAS" true \
    || { echo "Cannot ssh to '$HOST_ALIAS' (check ~/.ssh/config, key, network)." >&2; exit 1; }

# ---- 1. copy the code (not machine-specific files: .env, .venv, output, caches) ---------------------
if [ "${NO_SYNC:-0}" != 1 ]; then
    echo ">> Syncing code ..."
    rsync -az --delete -e "ssh -x" \
        --exclude '.env' --exclude '.venv/' --exclude 'output/' --exclude '__pycache__/' \
        --exclude '.DS_Store' --exclude '*.pyc' --exclude '__marimo__/' \
        --exclude 'notebooks/qc_plots/' --exclude 'notebooks/ilxl_corner_table.json' \
        ./ "$HOST_ALIAS:$REMOTE_DIR/"
fi

# ---- 2. start the app on the remote, tunnel the port, open the browser -------------------------------
_q() { printf '%q' "$1"; }
REMOTE_CMD="cd $(_q "$REMOTE_DIR") && chmod +x run_app.sh && HEADLESS=1 PORT=$(_q "$PORT") SEGY_FILE=$(_q "${SEGY_FILE:-}") ./run_app.sh $(_q "$MODE")"

( # open the page once the tunnel has had time to come up
    sleep 6
    URL="http://localhost:$LOCAL_PORT"
    if command -v open >/dev/null 2>&1; then open "$URL"; elif command -v xdg-open >/dev/null 2>&1; then xdg-open "$URL" >/dev/null 2>&1; fi
) &

echo ">> Starting the app on $HOST_ALIAS (Ctrl+C here stops it) ..."
exec "${SSH[@]}" -t -L "$LOCAL_PORT:127.0.0.1:$PORT" -L "$((LOCAL_PORT + 1)):127.0.0.1:$((PORT + 1))" \
    "$HOST_ALIAS" "$REMOTE_CMD"
