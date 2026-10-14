#!/usr/bin/env bash
# Dev launcher.
#
#   ./run.sh            backend only   (http://localhost:8000)
#   ./run.sh web        frontend only  (http://localhost:5173)
#   ./run.sh all        both, Ctrl-C stops both
#
# --reload-dir backend is not optional. A bare --reload watches the whole
# working directory, which here means ~10,000 .py files in .venv plus
# node_modules. That makes startup slow and fires spurious reloads.

set -euo pipefail
cd "$(dirname "$0")"

if [[ ! -d .venv ]]; then
  echo "No .venv at the repo root. Create it:"
  echo "  python3 -m venv .venv && source .venv/bin/activate"
  echo "  pip install torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cpu"
  echo "  pip install -r backend/requirements.txt"
  exit 1
fi

start_api() {
  # shellcheck disable=SC1091
  source .venv/bin/activate
  exec uvicorn backend.main:app --reload --reload-dir backend --port 8000
}

start_web() {
  [[ -d frontend/node_modules ]] || (cd frontend && npm install)
  cd frontend && exec npm run dev
}

case "${1:-api}" in
  api|backend|"")
    start_api
    ;;
  web|frontend)
    start_web
    ;;
  all|both)
    trap 'kill 0' EXIT INT TERM      # Ctrl-C takes down both children
    ( start_api ) &
    ( start_web ) &
    echo
    echo "  API  http://localhost:8000/docs"
    echo "  App  http://localhost:5173"
    echo "  wired: curl -s localhost:8000/health"
    echo
    wait
    ;;
  *)
    echo "usage: ./run.sh [api|web|all]" >&2
    exit 1
    ;;
esac
