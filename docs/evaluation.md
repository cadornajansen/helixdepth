# Evaluation protocol and verified results

The latest completed comparison is the matched **300M-target** evaluation at
the end of this document. The original recovery command and CP6 results below
are preserved for the historical 100M models.

Run from `/workspace/helixdepth-main-v2` on the existing Pod:

```bash
nohup .venv/bin/python -u scripts/evaluate_suite.py > artifacts/evaluation_recovery.log 2>&1 < /dev/null &
tail -f artifacts/evaluation_recovery.log
```

Run this once. Ctrl+C exits the log viewer; the background worker continues.
If the supervisor exits before completion, rerun the same Python command:
verified completed jobs are skipped. Preserve the previous log with a different
redirection filename when restarting. The OS lock prevents concurrent suite
launches in this checkout; it does not control legacy manually launched workers.

The supervisor runs ten jobs sequentially: four official zero-shot benchmarks
and the frozen 262144-token WikiText-103 test slice for each trained model.
It loads each benchmark's unchanged official task directory using
TaskManager(include_defaults=False), avoiding full-catalogue discovery.
Defaults: batch16, context512, pinned harness0.4.13, unchanged seeds/scoring.
Each worker has a20-minute timeout and at most two attempts per invocation.
A failed job is recorded and the remaining jobs still run. Batch size, example
count and scoring settings are never silently reduced on failure.

Every attempt has a separate folder/log under artifacts/evaluation_recovery/.
Result JSON and completion summaries use temporary files followed by atomic
replacement. Completion requires hashes, model/tokenizer identity, context,
zero-shot settings, finite accuracy/perplexity and full expected sample counts.
Expected benchmark counts: HellaSwag10042, ARC-Easy2376, PIQA1838, WinoGrande1267.
Changing model/tokenizer/source/package/settings requires a new --output.
Failed/interrupted attempts and earlier eval_v2 reports are preserved. Legacy
reports are not silently imported because they lack this runner's identity
manifest. HellaSwag is rerun as part of the new fully verified set.

The main log reports a heartbeat every30seconds and points to the worker log.
Only `ALL 10 EVALUATIONS VERIFIED` and suite.json complete=true establish full
completion. Nonzero exit and complete=false mean results are incomplete.
Inspect suite.json for the paths to final reports and logs. Download the whole
evaluation_recovery folder afterward. All generated outputs stay Gitignored.

Local verification:9 focused tests passed, including simulated native-crash
exit/retry, resume without relaunch, damaged results, bounded retries, timeout
termination, process locking, changed-input rejection, atomic writes, restricted
task discovery, and the real harness adapter on a two-question offline fixture.
The tests include multiple assertions per test. No actual CUDA evaluation or
benchmark score is established by these fixtures. Evidence:
docs/results/evaluation_recovery_tests.xml. Restricted discovery of all four
official task directories took approximately0.016seconds locally.

No software can guarantee immunity to hardware/network/native-library crashes.
This change removes the observed repeated catalogue scan and contains failures;
it does not establish the original segmentation fault's precise cause. On Linux,
workers inherit the suite lock so a surviving worker still blocks a duplicate
supervisor after an abrupt parent death. Windows lock tests passed; Linux/CUDA
execution subsequently completed in the user-operated Pod run documented below.

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_evaluation_runner.py
```

## Completed CP6 evaluation — October 2, 2026

All ten jobs completed on their first attempt. The downloaded suite reports
complete=true. Local verification rechecked every completion receipt and result
hash with cached_result, all source hashes, the tokenizer hash, and both actual
backed-up model files against run.json. All matched.

The full-precision report is [evaluation_summary.json](results/evaluation_summary.json).
It preserves raw and length-normalized accuracy, standard errors, WikiText slice
identity, package/source/model hashes, and raw-result paths/hashes. Raw reports
remain in ignored artifacts/downloaded_evaluation_recovery.

HelixDepth perplexity is 114.355971 versus 117.685474 for baseline, a 2.829153%
reduction. Raw accuracy changes (percentage points) are HellaSwag -0.0199,
ARC-Easy 0, PIQA +0.8161, and WinoGrande +0.3946. These small single-run differences
do not establish statistical significance or broad reasoning superiority.
Higher training time and peak memory accompany the perplexity gain. PASSED means
evaluation output verified, not a high benchmark score.

Recheck downloaded evidence from the repository root in PowerShell:

```powershell
@'
import json
from pathlib import Path
from scripts.evaluate_suite import cached_result, sha256, TASKS
root = Path('artifacts/downloaded_evaluation_recovery')
run = json.loads((root / 'run.json').read_text())
assert json.loads((root / 'suite.json').read_text())['complete']
for name, digest in run['sources'].items():
    assert sha256(Path(name)) == digest, name
assert sha256(Path('artifacts/cp2/tokenizer.json')) == run['tokenizer_sha256']
for variant in ('baseline', 'helixdepth'):
    assert sha256(Path(f'artifacts/main_backups/main_{variant}/model.pt')) == run['models'][variant]
    identity = dict(run, model_sha256=run['models'][variant])
    for task in TASKS:
        assert cached_result(root / f'{variant}_{task}', task, identity) is not None
print('All ten results and local inputs verified')
'@ | .\.venv\Scripts\python.exe
```

This checks evidence without launching training or evaluation. If source code
changes later, use the recorded source version when rechecking identity.

## Completed matched 300M evaluation — October 2, 2026

All ten jobs passed on their first attempt. The supervisor launched at
06:12:05 UTC (2:12 PM Manila); its completed suite was written at 06:24:20 UTC
(2:24 PM Manila). The roughly 12-minute interval includes process startup,
loading, scoring and result checks; worker scoring timers alone are shorter.
Both 300M model backups had passed local checksum verification before launch.

No evaluation code, package versions or scoring settings changed from the
original verified suite. Source hashes in both run manifests matched. Only the
selected model paths and fresh output directory changed. Staging symlinks under
`artifacts/evaluation_models_300m/main_{variant}/model.pt` pointed to the respective
`artifacts/{variant}_additional_100m_v2/model.pt` exports; no checkpoint was edited.

Exact evaluator invocation, launched as a detached process on the Pod:

```bash
.venv/bin/python -u scripts/evaluate_suite.py \
  --model-root artifacts/evaluation_models_300m \
  --output artifacts/evaluation_300m \
  --tokenizer artifacts/cp2/tokenizer.json \
  --device cuda --batch-size 16 --attempts 2 --timeout-seconds 1200 \
  --wikitext-revision b08601e04326c79dfdd32d625aee71d232d685c3
```

This command documents the completed evaluation; rerunning against intact
matching receipts skips completed jobs. The supervisor log and launch record
are preserved with the downloaded evidence. No training command is involved.

| Raw zero-shot accuracy / token perplexity | Baseline | HelixDepth | Helix minus baseline |
|---|---:|---:|---:|
| HellaSwag | 26.49% | 26.49% | 0.00 percentage points |
| ARC-Easy | 37.79% | 36.99% | -0.80 percentage points |
| PIQA | 56.64% | 57.07% | +0.44 percentage points |
| WinoGrande | 47.51% | 49.25% | +1.74 percentage points |
| WikiText-103 perplexity | 86.511235 | 85.418300 | -1.092935 (1.26% lower) |

Both models use the complete expected benchmark sample counts and the same
262,144-token WikiText slice as before. The full-precision
[300M summary](results/evaluation_summary_300m.json) retains raw and normalized
accuracy, standard errors, result hashes, model identities and three-stage
training accounting. The original [100M summary](results/evaluation_summary.json)
is preserved separately. There was no external 200M benchmark evaluation.

For HelixDepth, 100M to 300M reduced WikiText perplexity by 25.30%; raw ARC-Easy,
PIQA and HellaSwag increased by 3.20, 0.60 and 0.10 percentage points, while
WinoGrande decreased by 1.10 points. Baseline's perplexity improved by 26.49%.
These are additional-training improvements, not evidence that the architecture
caused all of the gains. HelixDepth's relative perplexity advantage over baseline
narrowed from 2.83% to 1.26%.

Individual accuracy standard errors are approximately 0.44 points for HellaSwag,
0.99 for ARC-Easy, 1.15–1.16 for PIQA and 1.40–1.41 for WinoGrande. They do not
establish significance of the paired differences or variation across training
seeds. Normalized PIQA is 55.71% for baseline versus 55.60% for HelixDepth,
reversing the small raw-accuracy advantage. The headline metric remains raw
accuracy consistently across both evaluation stages.

### Download and evidence verification

Direct SSH/SCP intermittently timed out after evaluation. The authenticated
RunPod SSH gateway remained available. The evidence was archived, transferred
through that gateway, checked against its remote byte count and SHA-256, and
extracted locally without overwriting earlier results. The archive contains
reports and logs, not datasets or new model weights.

The remote manifest covers all **45 evidence files / 129,497,109 bytes**.
Local verification checked that complete inventory, every file size/hash, each
job's completion receipt, full sample counts, scoring settings, source hashes,
the tokenizer, actual backed-up models and the unchanged WikiText slice.
Training records additionally passed parent-checkpoint, cumulative-target and
full-validation checks. All ten accepted outputs are `attempt-1`.

Raw evidence: `artifacts/downloaded_evaluation_300m/`.
Model/tokenizer backup: `artifacts/backups/continuation_300m/`.
Generated checkpoints, archives and raw evidence remain Gitignored.

From the local repository root, one read-only command rechecks every downloaded
evaluation file against the transfer manifest:

```powershell
.venv/Scripts/python.exe scripts/verify_transfer.py --root artifacts/downloaded_evaluation_300m --manifest artifacts/downloaded_evaluation_300m/transfer_manifest.json
```

Expected: `passed: true`, `files_verified: 45`. The local operational report
builder `artifacts/build_300m_summary.py` performs the deeper receipt/model/data
checks and regenerates `docs/results/evaluation_summary_300m.json`; it is saved
alongside ignored local working artifacts. No further training was launched.
