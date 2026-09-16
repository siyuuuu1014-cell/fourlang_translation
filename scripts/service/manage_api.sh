#!/usr/bin/env bash
# Manage the FourLang translation API (supervisord daemon; supervisorctl is not installed).
set -euo pipefail

ROOT=/root/autodl-tmp/fourlang_translation
CONF="$ROOT/service/supervisord_api.conf"

case "${1:-status}" in
  start)
    pkill -f "scripts/service/run_api.py" 2>/dev/null || true
    pkill -f "supervisord_api.conf" 2>/dev/null || true
    sleep 1
    nohup supervisord -c "$CONF" >/dev/null 2>&1 &
    sleep 4
    ps -eo pid,etime,cmd | grep "run_api.py" | grep -v grep || echo "run_api not up"
    ;;
  stop)
    pkill -f "supervisord_api.conf" 2>/dev/null || true
    pkill -f "scripts/service/run_api.py" 2>/dev/null || true
    echo "stopped"
    ;;
  restart)
    pkill -f "scripts/service/run_api.py" 2>/dev/null || true
    sleep 6
    ps -eo pid,etime,cmd | grep "run_api.py" | grep -v grep || echo "run_api not up"
    ;;
  status)
    ps -eo pid,etime,cmd | grep -E "run_api.py|supervisord_api.conf" | grep -v grep || echo "not running"
    curl -s -m 3 http://127.0.0.1:8000/health >/dev/null 2>&1 && echo "health: ok" || echo "health: unreachable"
    ;;
  *)
    echo "usage: $0 {start|stop|restart|status}"
    exit 1
    ;;
esac
