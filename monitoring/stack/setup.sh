#!/usr/bin/env bash
# One-command monitoring for LTX: Prometheus + Grafana + node/GPU exporters, with the LTX dashboard
# pre-loaded. You choose the Grafana login on first run; it is stored only in ./.env (git-ignored).
#
#   ./setup.sh                  # first run: asks for a Grafana user + password, then starts the stack
#   ./setup.sh --password       # change the Grafana admin password later
#   ./setup.sh --down           # stop the stack (data is kept in Docker volumes)
#
# Non-interactive: GRAFANA_ADMIN_PASSWORD=... ./setup.sh   (optionally GRAFANA_ADMIN_USER=...)
set -euo pipefail
cd "$(dirname "$0")"
ENV=.env

compose() { docker compose --env-file "$ENV" "$@"; }

ask_password() {  # prints the chosen password on stdout; prompts go to the terminal
  local p1 p2
  while true; do
    read -rsp "Grafana admin password (min 8 chars): " p1 </dev/tty; echo >/dev/tty
    [ ${#p1} -ge 8 ] || { echo "Too short." >/dev/tty; continue; }
    read -rsp "Repeat password: " p2 </dev/tty; echo >/dev/tty
    [ "$p1" = "$p2" ] && break
    echo "Passwords don't match." >/dev/tty
  done
  printf '%s' "$p1"
}

set_env() {  # set_env KEY VALUE: replace or append one line in .env, keeping it private
  local tmp; tmp=$(mktemp)
  [ -f "$ENV" ] && grep -v "^$1=" "$ENV" >"$tmp" || true
  printf '%s=%s\n' "$1" "$2" >>"$tmp"
  install -m 600 "$tmp" "$ENV"; rm -f "$tmp"
}

case "${1:-}" in
  --down)
    [ -f "$ENV" ] || { echo "Nothing to stop: no $ENV, so the stack was never set up."; exit 0; }
    compose down; exit 0 ;;
  --password)
    [ -f "$ENV" ] || { echo "No $ENV yet: run ./setup.sh first."; exit 1; }
    pw=$(ask_password)
    # Grafana keeps the password in its own database (the .env value only seeds a fresh install),
    # so change it inside the running container. Reading stdin keeps it out of ps and shell history.
    compose up -d grafana >/dev/null
    for _ in $(seq 30); do docker exec ltx-grafana grafana cli --version >/dev/null 2>&1 && break; sleep 1; done
    printf '%s' "$pw" | docker exec -i ltx-grafana grafana cli admin reset-admin-password --password-from-stdin >/dev/null
    set_env GRAFANA_ADMIN_PASSWORD "$pw"
    echo "Grafana admin password updated."; exit 0 ;;
  "") ;;
  *) sed -n '2,10p' "$0"; exit 1 ;;
esac

command -v docker >/dev/null || { echo "Docker is required: https://docs.docker.com/engine/install/"; exit 1; }
docker compose version >/dev/null 2>&1 || { echo "The Docker Compose plugin is required."; exit 1; }

[ -f "$ENV" ] || install -m 600 .env.example "$ENV"
if ! grep -q '^GRAFANA_ADMIN_PASSWORD=.\+' "$ENV"; then
  if [ -n "${GRAFANA_ADMIN_PASSWORD:-}" ]; then
    pw=$GRAFANA_ADMIN_PASSWORD
  else
    echo "Choose the Grafana login for this machine (stored only in monitoring/stack/.env)."
    read -rp "Grafana admin user [admin]: " user </dev/tty
    set_env GRAFANA_ADMIN_USER "${GRAFANA_ADMIN_USER:-${user:-admin}}"
    pw=$(ask_password)
  fi
  set_env GRAFANA_ADMIN_PASSWORD "$pw"
fi

compose up -d
port=$(grep -oP '^GRAFANA_PORT=\K.*' "$ENV" || true)
echo
echo "Grafana:    http://localhost:${port:-3001}/d/ltx-video   (log in with the user/password you chose)"
echo "Prometheus: http://localhost:$(grep -oP '^PROMETHEUS_PORT=\K.*' "$ENV" || echo 9090)"
echo "The LTX panels fill in once the exporter runs: ./setup-services.sh (in the repo root)."
echo "Change the password later with: monitoring/stack/setup.sh --password"
