#!/usr/bin/env bash
# Удаление systemd-службы ergo-ollama

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODULE_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
ERGO_ROOT="$(cd "$MODULE_ROOT/../.." && pwd)"

# shellcheck source=../../../../core/deployment/linux/lib/core.sh
source "$ERGO_ROOT/core/deployment/linux/lib/core.sh"
# shellcheck source=../../../../core/deployment/linux/lib/systemd.sh
source "$ERGO_ROOT/core/deployment/linux/lib/systemd.sh"

UNIT_NAME="ergo-ollama.service"
PROJECT_UNIT="$ERGO_ROOT/core/deployment/wrappers/systemd/${UNIT_NAME}"
LEGACY_UNIT="/etc/systemd/system/${UNIT_NAME}"

echo ""
echo "=== Uninstalling Ollama Service ==="
echo ""

if [[ ! -f "$PROJECT_UNIT" ]] && [[ ! -e "$LEGACY_UNIT" ]] \
  && ! systemctl list-unit-files --type=service --no-legend 2>/dev/null | grep -q '^ergo-ollama\.service'; then
  echo "[SKIP] Service ergo-ollama not found"
  exit 0
fi

systemctl_do stop "$UNIT_NAME" 2>/dev/null || true
systemctl_do disable "$UNIT_NAME" 2>/dev/null || true

if [[ -e "$LEGACY_UNIT" ]] || [[ -L "$LEGACY_UNIT" ]]; then
  if [[ $(id -u) -eq 0 ]]; then
    rm -f "$LEGACY_UNIT"
  else
    sudo rm -f "$LEGACY_UNIT"
  fi
fi

if [[ -f "$PROJECT_UNIT" ]]; then
  rm -f "$PROJECT_UNIT"
fi

daemon_reload

echo ""
echo "[OK] Ollama service removed"
echo ""
