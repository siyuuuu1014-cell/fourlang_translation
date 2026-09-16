#!/usr/bin/env bash
# Manage the FourLang translation API via supervisord.
set -euo pipefail

ROOT=/root/autodl-tmp/fourlang_translation
CONF="$ROOT/service/supervisord_api.conf"
SUPERVISORCTL="/usr/bin/supervisorctl"

case "${1:-status}" in
  start)
    # Stop any previous bare-nohup instance so supervisord owns the process.
    pkill -f "scripts/service/run_api.py" 2>/dev/null || true
    sleep 1
    supervisord -c "$CONF"
    sleep 3
    "$SUPERVISORCTL" -c "$CONF" status
    ;;
  stop)
    "$SUPERVISORCTL" -c "$CONF" stop fourlang_api 2>/dev/null || true
    "$SUPERVISORCTL" -c "$CONF" shutdown 2>/dev/null || true
    ;;
  restart)
    "$SUPERVISORCTL" -c "$CONF" restart fourlang_api
    ;;
  status)
    "$SUPERVISORCTL" -c "$CONF" status
    ;;
  *)
    echo "usage: $0 {start|stop|restart|status}"
    exit 1
    ;;
esac
