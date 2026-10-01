#!/usr/bin/env bash
# Run only INSIDE an already-authorized Linux x86_64 CUDA Pod.
set -euo pipefail
cd "$(dirname "$0")/.."
test "$(uname -m)" = x86_64
python3 scripts/verify_transfer.py
nvidia-smi
test ! -e .venv || { echo 'Existing .venv preserved; use a fresh extraction.' >&2; exit 1; }
curl -LsSf https://astral.sh/uv/0.11.28/install.sh -o /tmp/helixdepth-uv-install.sh
sh /tmp/helixdepth-uv-install.sh
export PATH="$HOME/.local/bin:$PATH"
uv --version
uv python install 3.12.13
uv venv --python 3.12.13 .venv
uv pip sync --python .venv/bin/python --require-hashes --only-binary :all: \
  --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cu128 \
  --index-strategy unsafe-best-match requirements/runpod.lock
uv pip install --python .venv/bin/python --no-deps --no-build-isolation -e .
mkdir -p artifacts/runpod_results
uv pip freeze --python .venv/bin/python > artifacts/runpod_results/environment.txt
.venv/bin/python - <<'PY'
import sys
import torch
assert sys.version_info[:3] == (3, 12, 13)
assert str(torch.__version__) == '2.8.0+cu128'
assert torch.cuda.is_available(), 'CUDA unavailable: setup stops before profiling'
print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name())
PY
.venv/bin/python scripts/verify_transfer.py > artifacts/runpod_results/verification.json
.venv/bin/python count_params.py --output artifacts/runpod_results/parameter_counts.json
.venv/bin/python -m pytest -q
echo 'Setup complete. Profiling is a separate explicit command in docs/runpod.md.'
