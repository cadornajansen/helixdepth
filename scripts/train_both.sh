#!/usr/bin/env bash
# Explicitly launched by the user; survives SSH disconnect through nohup.
set -euo pipefail
cd "$(dirname "$0")/.."
test ! -e artifacts/main_started || { echo 'Existing run preserved; follow recovery instructions.'; exit 1; }
.venv/bin/python scripts/verify_transfer.py > /tmp/helixdepth-main-verification.json
mkdir -p artifacts
date -u > artifacts/main_started
trap 'code=$?; printf "%s\n" "$code" > artifacts/main_exit_code; date -u > artifacts/main_finished' EXIT
.venv/bin/python -u scripts/run_experiment.py --mode recovery --output artifacts/main_recovery
.venv/bin/python -u scripts/run_experiment.py --mode run --variant baseline --output artifacts/main_baseline
.venv/bin/python -u scripts/run_experiment.py --mode run --variant helixdepth --output artifacts/main_helixdepth
echo 'BOTH MODELS FINISHED. Download checkpoints and stop the Pod. Evaluation is a separate command.'
