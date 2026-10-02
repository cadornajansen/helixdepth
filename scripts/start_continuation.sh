#!/usr/bin/env bash
# Run from the repository root. Keep the Pod running after this stage.
set -euo pipefail
exec 9>artifacts/continuation.lock
flock -n 9 || { echo 'Another continuation launcher is active'; exit 1; }
test ! -e artifacts/helixdepth_additional_100m_v1
.venv/bin/python - <<'PY'
import hashlib
import json
from pathlib import Path

root = Path('artifacts/corpus_fresh_100m_v1')
manifest = json.loads((root / 'manifest.json').read_text())
assert manifest['fresh_verified']
for name, identity in manifest['files'].items():
    with (root / name).open('rb') as stream:
        assert hashlib.file_digest(stream, 'sha256').hexdigest() == identity['sha256'], name
gate = json.loads(Path('artifacts/continuation_gpu_check_v1/result.json').read_text())
assert gate['passed'] and gate['fresh_process_exact_resume']
with Path('artifacts/main_helixdepth/latest.pt').open('rb') as stream:
    assert hashlib.file_digest(stream, 'sha256').hexdigest() == gate['parent_sha256']
print('Fresh file hashes and GPU recovery gate independently verified', flush=True)
PY
deadline=$(date -u -d '+4 hours' '+%Y-%m-%dT%H:%M:%S+00:00')
printf '%s\n' "$deadline" > artifacts/continuation_deadline.txt
timeout --signal=TERM --kill-after=30s 4h .venv/bin/python -u scripts/run_continuation.py \
  --initialize-from artifacts/main_helixdepth/latest.pt \
  --data artifacts/corpus_fresh_100m_v1 \
  --output artifacts/helixdepth_additional_100m_v1 \
  --device cuda --deadline "$deadline"
