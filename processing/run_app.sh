#!/usr/bin/env bash
# Launch the Trace QC GUI on any machine (Mac / Linux / WSL).
#   ./run_app.sh                       app mode (code hidden)   : marimo run,  http://localhost:2718
#   ./run_app.sh edit                  notebook editor          : marimo edit
#   HOST=0.0.0.0 ./run_app.sh          reachable from other machines (login token switched on automatically)
#   PORT=8080 SEGY_FILE=/data/a.sgy ./run_app.sh
#   TOKEN_PASSWORD=secret HOST=0.0.0.0 ./run_app.sh    fixed password instead of a random token
# Settings can also live in a `.env` file next to this script (see .env.example). Shell variables win.
# First run on a new machine: if no Python with marimo/numpy/matplotlib is found, a `.venv` is created here
# and requirements.txt is installed into it (needs internet once).
set -euo pipefail
cd "$(dirname "$0")"

# ---- settings: shell environment first, then .env ---------------------------------------------------
_keep="$(env)"                                   # remember what the caller exported
if [ -f .env ]; then set -a; . ./.env; set +a; fi
for _v in HOST PORT PYTHON SEGY_FILE TOKEN_PASSWORD MARIMO; do   # re-apply caller's values over .env
    _line="$(printf '%s\n' "$_keep" | grep -E "^${_v}=" || true)"
    if [ -n "$_line" ]; then export "$_v=${_line#*=}"; fi
done

MODE="${1:-run}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-2718}"
DEPS="import marimo, numpy, matplotlib"

# ---- find (or build) a Python that has the packages -------------------------------------------------
PY=""
for _c in "${PYTHON:-}" ".venv/bin/python" python3 python; do
    [ -n "$_c" ] || continue
    _p="$(command -v "$_c" 2>/dev/null || true)"
    if [ -n "$_p" ] && "$_p" -c "$DEPS" >/dev/null 2>&1; then PY="$_p"; break; fi
done
if [ -z "$PY" ]; then
    _base="$(command -v python3 || command -v python || true)"
    [ -n "$_base" ] || { echo "No Python found. Install Python 3.9+ and run again." >&2; exit 1; }
    echo ">> No Python with marimo/numpy/matplotlib found - creating .venv from requirements.txt ..."
    "$_base" -m venv .venv || { echo "venv failed - on Ubuntu/Debian: sudo apt install python3-venv python3-pip" >&2; rm -rf .venv; exit 1; }
    .venv/bin/python -m pip install --quiet --upgrade pip
    .venv/bin/python -m pip install --quiet -r requirements.txt
    PY=".venv/bin/python"
fi
export SEGY_FILE="${SEGY_FILE:-}"                # read by app_marimo.py

# ---- network / auth ---------------------------------------------------------------------------------
ARGS=(--host "$HOST" --port "$PORT")
case "$HOST" in
    127.0.0.1|localhost|::1) LOCAL=1 ;;
    *)                       LOCAL=0 ;;
esac
[ "$LOCAL" = 1 ] || ARGS+=(--headless)           # a server has no browser to open
[ "${HEADLESS:-0}" = 1 ] && ARGS+=(--headless)

TOKEN_FILE=""
if [ -n "${TOKEN_PASSWORD:-}" ]; then
    TOKEN_FILE="$(mktemp)"; chmod 600 "$TOKEN_FILE"
    printf '%s' "$TOKEN_PASSWORD" > "$TOKEN_FILE"
    ARGS+=(--token-password-file "$TOKEN_FILE")
elif [ "$LOCAL" = 0 ]; then
    ARGS+=(--token)                              # random token, printed in the URL marimo shows
fi
unset TOKEN_PASSWORD

if [ "$LOCAL" = 0 ]; then
    _ip="$( (hostname -I 2>/dev/null | awk '{print $1}') || true)"
    [ -n "$_ip" ] || _ip="$( (ipconfig getifaddr en0 2>/dev/null) || true)"
    echo ">> Serving on all interfaces. From another machine open:  http://${_ip:-<this-machine-ip>}:${PORT}"
    echo ">> (If it does not connect: allow port ${PORT} in the firewall, or use an SSH tunnel - see README.)"
fi

# ---- notebook tools (🛠 Tools menu, e.g. Shot Geometry QC): a second marimo server for notebooks/, on PORT + 1 --
# Each notebook runs there as a normal marimo page (its own progress bars, cell by cell); the app links to it.
# NOTEBOOK_TOOLS=1 tells the app this server is running (without it the app embeds the notebook in its own page).
NB_PORT=$((PORT + 1))
NB_PID=""
if [ -d notebooks ] && [ "${NOTEBOOK_SERVER:-1}" = 1 ]; then
    if "$PY" -c "import socket,sys; s=socket.socket(); s.bind(('$HOST' if '$HOST' != 'localhost' else '127.0.0.1', $NB_PORT))" 2>/dev/null; then
        NB_ARGS=(--host "$HOST" --port "$NB_PORT" --headless)
        if [ -n "$TOKEN_FILE" ]; then NB_ARGS+=(--token-password-file "$TOKEN_FILE"); elif [ "$LOCAL" = 0 ]; then NB_ARGS+=(--token); fi
        mkdir -p output
        "$PY" -m marimo run notebooks "${NB_ARGS[@]}" > output/notebooks_server.log 2>&1 &
        NB_PID=$!
        export NOTEBOOK_TOOLS=1
        echo ">> Notebook tools on port $NB_PORT (log: output/notebooks_server.log)"
    else
        echo ">> Port $NB_PORT is busy - notebook tools open inside the app page instead (no progress bars there)."
    fi
fi
trap '[ -n "$NB_PID" ] && kill "$NB_PID" 2>/dev/null; [ -n "$TOKEN_FILE" ] && rm -f "$TOKEN_FILE"' EXIT

"$PY" -m marimo "$MODE" app_marimo.py "${ARGS[@]}"
