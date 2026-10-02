#!/usr/bin/env bash
# Same fresh corpus and optimization settings as the completed HelixDepth stage.
set -euo pipefail
if [[ ${1:-} != '' && ${1:-} != '--check-only' ]]; then
  echo 'Usage: bash scripts/start_baseline_continuation.sh [--check-only]'
  exit 1
fi
exec 9>artifacts/continuation.lock
flock -n 9 || { echo 'Another continuation launcher is active'; exit 1; }
test ! -e artifacts/baseline_additional_100m_v1
.venv/bin/python - <<'PY'
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from helixdepth.training import TrainingConfig

root = Path('artifacts/corpus_fresh_100m_v1')
manifest = json.loads((root / 'manifest.json').read_text())
assert manifest['fresh_verified']
for name, identity in manifest['files'].items():
    with (root / name).open('rb') as stream:
        assert hashlib.file_digest(stream, 'sha256').hexdigest() == identity['sha256'], name
parent = Path('artifacts/main_baseline')
original = json.loads((parent / 'metrics.json').read_text())
assert original['variant'] == 'baseline' and original['complete_schedule']
with (parent / 'latest.pt').open('rb') as stream:
    assert hashlib.file_digest(stream, 'sha256').hexdigest() == original['files']['latest.pt']['sha256']
helix = json.loads(Path('artifacts/helixdepth_additional_100m_v1/metrics.json').read_text())
assert helix['stop_reason'] == 'schedule_complete'
with (root / 'manifest.json').open('rb') as stream:
    assert hashlib.file_digest(stream, 'sha256').hexdigest() == helix['identity']['data_manifest_sha256']
steps = (manifest['splits']['train']['tokens'] - 1) // 8192
config = TrainingConfig(context=512, batch_size=16, learning_rate=1e-4,
                        schedule_steps=steps, warmup_steps=min(244, steps // 10),
                        checkpoint_every=500, validation_batches=64)
assert asdict(config) == helix['identity']['config']
assert original['target_tokens'] == helix['identity']['parent_targets']
assert helix['stage_targets'] == steps * 8192
print('BASELINE READY: original checkpoint verified; fresh data and settings match HelixDepth', flush=True)
PY
if [[ ${1:-} == '--check-only' ]]; then exit 0; fi
deadline=$(date -u -d '+4 hours' '+%Y-%m-%dT%H:%M:%S+00:00')
printf '%s\n' "$deadline" > artifacts/baseline_continuation_deadline.txt
timeout --signal=TERM --kill-after=30s 4h .venv/bin/python -u scripts/run_continuation.py \
  --initialize-from artifacts/main_baseline/latest.pt \
  --data artifacts/corpus_fresh_100m_v1 \
  --output artifacts/baseline_additional_100m_v1 \
  --device cuda --deadline "$deadline"
