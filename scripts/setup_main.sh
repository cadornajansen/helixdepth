#!/usr/bin/env bash
# Reuse the working Pod environment; do not reinstall CUDA/PyTorch.
set -euo pipefail
cd "$(dirname "$0")/.."
python3 scripts/verify_transfer.py
test ! -e .venv || { echo 'Existing environment preserved. Choose a fresh extraction directory.'; exit 1; }
test -x /workspace/helixdepth/.venv/bin/python || { echo 'Original Pod environment missing; use setup_runpod.sh instead.'; exit 1; }
ln -s /workspace/helixdepth/.venv .venv
export PATH="$HOME/.local/bin:$PATH"
uv pip install --python .venv/bin/python --no-deps --no-build-isolation -e .
.venv/bin/python - <<'PY'
import sys
import torch
from pathlib import Path
import helixdepth
assert sys.version_info[:3] == (3, 12, 13), sys.version
assert str(torch.__version__) == '2.8.0+cu128', torch.__version__
assert torch.cuda.is_available(), 'CUDA unavailable'
assert Path(helixdepth.__file__).resolve().is_relative_to(Path.cwd()), helixdepth.__file__
print('CUDA ready:', torch.cuda.get_device_name())
PY
.venv/bin/python -m pytest -q
echo 'Ready. Launch scripts/train_both.sh with nohup.'
