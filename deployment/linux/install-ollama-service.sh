#!/usr/bin/env bash
# Установка systemd-службы ergo-ollama

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODULE_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
ERGO_ROOT="$(cd "$MODULE_ROOT/../.." && pwd)"

# shellcheck source=../../../../core/deployment/linux/lib/core.sh
source "$ERGO_ROOT/core/deployment/linux/lib/core.sh"
# shellcheck source=../../../../core/deployment/linux/lib/systemd.sh
source "$ERGO_ROOT/core/deployment/linux/lib/systemd.sh"
# shellcheck source=../../../../core/deployment/linux/lib/services.sh
source "$ERGO_ROOT/core/deployment/linux/lib/services.sh"

set_service_project_root "$ERGO_ROOT"
write_env_file "$ERGO_ROOT"

_ollama_log_path() {
  local root="$1"
  local py="$root/virtual_env/python/bin/python"
  local script="$root/core/deployment/scripts/log_env.py"
  if [[ -x "$py" && -f "$script" ]]; then
    "$py" "$script" path OLLAMA "$root"
    return 0
  fi
  echo "$root/logs/ollama-serve.log"
}

OLLAMA_LOG_PATH="$(_ollama_log_path "$ERGO_ROOT")"
mkdir -p "$(dirname "$OLLAMA_LOG_PATH")"
: >>"$OLLAMA_LOG_PATH"

OLLAMA_UNIT=$(cat <<UNIT
[Unit]
Description=Ergo Ollama Server
After=network.target

[Service]
Type=simple
EnvironmentFile=__ERGO_MS_ENV__
ExecStart=/bin/bash -lc 'cd "\$ERGO_ROOT" && ergoms ollama_framework:start-ollama'
Restart=always
RestartSec=5
Environment=PYTHONUNBUFFERED=1
Environment=HOME=$ERGO_ROOT/virtual_env/cache/ollama/home
Environment=OLLAMA_NO_CLOUD=1
StandardOutput=append:$OLLAMA_LOG_PATH
StandardError=append:$OLLAMA_LOG_PATH

[Install]
WantedBy=multi-user.target
UNIT
)

echo ""
echo "=== Installing Ollama Service ==="
echo ""

install_unit "ergo-ollama" "$OLLAMA_UNIT" "$ERGO_ROOT"
daemon_reload
enable_and_start ergo-ollama.service

echo ""
echo "=== Ollama Service Installed and Started ==="
echo ""
