#!/usr/bin/env bash
# One further fresh-data stage per model; verified local backup is mandatory.
set -euo pipefail
exec 9>artifacts/continuation.lock
flock -n 9 || { echo 'Another continuation launcher is active'; exit 1; }
.venv/bin/python - <<'PY'
import hashlib
import json
from pathlib import Path

root = Path('artifacts')
receipt = json.loads((root / 'continuation_200m_backup_verified.json').read_text())
assert receipt['verified']
reports = []
for variant in ('baseline', 'helixdepth'):
    previous = root / f'{variant}_additional_100m_v1'
    report = json.loads((previous / 'metrics.json').read_text())
    assert report['stop_reason'] == 'schedule_complete'
    assert report['selected']['stage_step'] == report['step']
    assert not (root / f'{variant}_additional_100m_v2').exists()
    for name, identity in report['files'].items():
        with (previous / name).open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        assert digest == identity['sha256'] == receipt['models'][variant][name], name
    reports.append(report)
assert reports[0]['identity']['config'] == reports[1]['identity']['config']
assert reports[0]['cumulative_targets'] == reports[1]['cumulative_targets']
data = root / 'corpus_fresh_100m_v2'
manifest = json.loads((data / 'manifest.json').read_text())
assert manifest['fresh_verified'] and manifest['splits']['train']['tokens'] >= 99_000_000
with (root / 'corpus_fresh_100m_v1/manifest.json').open('rb') as stream:
    assert hashlib.file_digest(stream, 'sha256').hexdigest() == manifest['previous_manifest_sha256']
for name, identity in manifest['files'].items():
    with (data / name).open('rb') as stream:
        assert hashlib.file_digest(stream, 'sha256').hexdigest() == identity['sha256'], name
print('BACKUPS AND NEXT FRESH CORPUS VERIFIED; starting matched stage', flush=True)
PY
deadline=$(date -u -d '+4 hours' '+%Y-%m-%dT%H:%M:%S+00:00')
printf '%s\n' "$deadline" > artifacts/matched_continuation_v2_deadline.txt
for variant in baseline helixdepth; do
  echo "START $variant additional stage 2"
  remaining=$(( $(date -d "$deadline" '+%s') - $(date '+%s') ))
  test "$remaining" -gt 120
  timeout --signal=TERM --kill-after=30s "${remaining}s" .venv/bin/python -u scripts/run_continuation.py \
    --initialize-from "artifacts/${variant}_additional_100m_v1/latest.pt" \
    --parent-report "artifacts/${variant}_additional_100m_v1/metrics.json" \
    --data artifacts/corpus_fresh_100m_v2 \
    --output "artifacts/${variant}_additional_100m_v2" \
    --device cuda --deadline "$deadline"
done
echo 'BOTH MATCHED STAGES FINISHED; Pod remains running'
