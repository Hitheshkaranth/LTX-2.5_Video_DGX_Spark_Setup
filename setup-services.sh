#!/usr/bin/env bash
# Install the LTX Video Studio web UI and the Prometheus exporter as systemd --user services,
# and optionally publish the UI to your tailnet.
#
#   ./setup-services.sh            # services only (UI on http://127.0.0.1:8090, metrics on :9092)
#   ./setup-services.sh --tailnet  # also `tailscale serve` the UI over HTTPS, tailnet-only
#   LTX_WORKERS=2 ./setup-services.sh   # render up to 2 jobs in parallel (default 1)
#   EXPORTER_HOST=172.17.0.1 ./setup-services.sh  # keep metrics off the LAN (Docker bridge only)
set -euo pipefail
ROOT=$(cd "$(dirname "$0")" && pwd)
UNITS="$HOME/.config/systemd/user"
mkdir -p "$UNITS" "$ROOT/logs" "$ROOT/outputs"

TS_LISTEN_LINE=""
if [ "${1:-}" = "--tailnet" ]; then
  command -v tailscale >/dev/null 2>&1 || { echo "tailscale not on PATH" >&2; exit 1; }
  TS_IP=$(tailscale ip -4)
  # Direct listener on the Tailscale IP for clients without MagicDNS (serve routes by hostname).
  TS_LISTEN_LINE="Environment=LTX_WEBUI_TS_LISTEN=$TS_IP:8091
Environment=TAILSCALE_BIN=$(command -v tailscale)"
fi

cat > "$UNITS/ltx-webui.service" <<EOF
[Unit]
Description=LTX Video Studio web UI
After=network.target

[Service]
Type=simple
Environment=LTX_WEBUI_HOST=127.0.0.1
Environment=LTX_WEBUI_PORT=8090
Environment=LTX_WORKERS=${LTX_WORKERS:-1}
Environment=LLM_CONTAINER=${LLM_CONTAINER:-}
$TS_LISTEN_LINE
ExecStart=/usr/bin/python3 $ROOT/webui/server.py
Restart=always
RestartSec=5
# Renders outlive a studio restart; the new server adopts them (logs/webui_queue.json).
KillMode=process

[Install]
WantedBy=default.target
EOF

cat > "$UNITS/ltx-exporter.service" <<EOF
[Unit]
Description=Prometheus exporter for LTX-2.5 video generation jobs
After=network.target

[Service]
Type=simple
Environment=LTX_LOGS=$ROOT/logs
Environment=EXPORTER_PORT=9092
Environment=EXPORTER_HOST=${EXPORTER_HOST:-0.0.0.0}
ExecStart=/usr/bin/python3 $ROOT/monitoring/ltx_exporter.py
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
EOF

# Keep user services running after you log out (otherwise a headless box stops the studio at SSH logout).
if [ "$(loginctl show-user "$USER" -p Linger --value 2>/dev/null)" != "yes" ]; then
  loginctl enable-linger "$USER" 2>/dev/null || sudo loginctl enable-linger "$USER"
fi

systemctl --user daemon-reload
systemctl --user enable --now ltx-webui ltx-exporter
systemctl --user restart ltx-webui ltx-exporter
echo "==> ltx-webui: $(systemctl --user is-active ltx-webui) (http://127.0.0.1:8090)"
echo "==> ltx-exporter: $(systemctl --user is-active ltx-exporter) (http://127.0.0.1:9092/metrics)"

if [ "${1:-}" = "--tailnet" ]; then
  # Needs Serve enabled for the tailnet (admin console) and `sudo tailscale set --operator=$USER` once.
  tailscale serve --bg http://127.0.0.1:8090
  tailscale serve --bg --http 8090 http://127.0.0.1:8090
  tailscale serve status
fi

cat <<EOF

Grafana dashboard: run monitoring/stack/setup.sh (starts Prometheus + Grafana in Docker; you choose the login).
Already run Prometheus? Add monitoring/prometheus-scrape.yml to it and import monitoring/grafana-dashboard.json.
EOF
