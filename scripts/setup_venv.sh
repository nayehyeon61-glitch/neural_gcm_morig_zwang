#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
NGCM_PYTHON="${NGCM_PYTHON:-python3.12}"
"$NGCM_PYTHON" -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -c requirements-tested.txt -e '.[neuralgcm,test]'
.venv/bin/python -m pip check
printf '%s\n' 'CPU environment ready. Run: source .venv/bin/activate'
