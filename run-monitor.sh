#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
exec "$HOME/.local/share/fr5/monitor-venv/bin/python" -m fr5_runtime.monitor
