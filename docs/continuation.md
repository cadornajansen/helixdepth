# Continued pretraining: matched stages

Both architectures have now completed matched continuation stages, reaching
299,982,848 targets each. Each continued from its own weights originally trained
from random initialization. The protocol sections below preserve the sequence
of preparation decisions; the completed-stage records supersede their earlier
pending statuses. No 400M stage has started.

## Protocol

- Original checkpoints, tokenizer, validation set and evaluation outputs stay intact.
- Fresh corpus: 115,077 documents, 99,999,958 packed tokens; original source
  revision and filtering policy retained. Every saved document was re-encoded
  and checked against packed bytes and indexes.
- Exact text hashes and source IDs from the original corpus/audit are excluded.
  Near duplicates are not exhaustively excluded; domain filters do not prove
  benchmark decontamination.
- Initialize model weights from `artifacts/main_helixdepth/latest.pt`, with a
  fresh AdamW optimizer and learning-rate schedule. This is a new stage, not
  an exact resume of the old optimization trajectory.
- Context 512, batch 16, float32, peak learning rate 0.0001, 244 warmup updates,
  12,207 updates / 99,999,744 next-token targets. No automatic extra stage.
- Full frozen validation before training, every 2,000 updates and at completion.
  `best.pt` retains the lowest validation loss, including the original weights
  as the step-zero candidate. Final `model.pt` exports that selected candidate.
- `latest.pt` saves resumable stage state every 500 updates and on normal exit.
- No 10AM cutoff. A four-hour ceiling begins at launch; the runner reserves
  two minutes for saving, with an external timeout as a fallback. A hard kill
  may require recovery from the last saved checkpoint.
- The Pod remains running after completion, as requested, and remains billable.

## Execution flow

`prepare_continuation_data.py` reads the pinned source after the old scan,
filters previously seen documents, tokenizes fresh text, verifies every packed
record and writes a manifest. `start_continuation.sh` independently checks file
hashes and the GPU recovery result, takes an exclusive lock and starts
`run_continuation.py`. The runner loads old weights, performs training and full
validation, and saves both recovery state and the selected model separately.

The data builder reported a native abort during interpreter shutdown after
writing its completed verification manifest. A separate process subsequently
verified every manifest file hash successfully before launch. This is recorded
as a preparation failure after output production, not a clean builder exit.

## Verification

Local tests cover fresh-data exclusion, initialization, exact resume, identity
guards, deadline rejection and the production runner's best-model export.
A discarded four-update RTX 4090 fixture matched uninterrupted training to a
two-update run resumed in a fresh process, including complete checkpoint state.
It also verified the original checkpoint hash was unchanged.

```powershell
.venv/Scripts/python.exe -m pytest -q tests/test_continuation.py
```

To inspect the live stage in the Pod terminal:

```bash
cd /workspace/helixdepth-main-v2
tail -n 20 artifacts/continuation.log
```

Do not relaunch an active stage. The launch script deliberately rejects an
existing stage directory. Recovery uses the original saved deadline and
`run_continuation.py --resume`; it must preserve the recorded runner, inputs
and configuration identities.

Corpus and GPU evidence are in `docs/results/continuation_corpus.json` and
`docs/results/continuation_gpu_check.json`. New benchmark results remain pending.

## Matched baseline continuation

HelixDepth completed all 12,207 updates, selecting the final step with full
validation loss 3.7680845545993455. Baseline continuation is prepared separately
on exactly the same fresh corpus, optimization settings and token budget.
Its launcher verifies the original baseline recovery checkpoint checksum and
compares the proposed configuration and corpus identity to HelixDepth's saved
stage report. It writes to `artifacts/baseline_additional_100m_v1`.

Preparation check only (does not train):

```bash
cd /workspace/helixdepth-main-v2
bash scripts/start_baseline_continuation.sh --check-only
```

Start the prepared stage when ready:

```bash
cd /workspace/helixdepth-main-v2
nohup bash scripts/start_baseline_continuation.sh > artifacts/baseline_continuation.log 2>&1 < /dev/null &
tail -f artifacts/baseline_continuation.log
```

The four-hour ceiling begins when this launcher starts. The exclusive lock
prevents concurrent launches. The old HelixDepth runner is preserved on the Pod
as `artifacts/helixdepth_additional_100m_v1/run_continuation_used.py` because
its recorded script hash predates the model-specific report-label correction.
Five local tests pass, including full production-runner resume/export tests
for both architectures. Baseline training has not started during preparation.

## Completed 200M-target stages and next matched stage

Both additional stages subsequently finished. Baseline selected full validation
loss 3.822446083484463; HelixDepth selected 3.7680845545993455. Both selected
step 12,207 and reached 199,991,296 cumulative next-token targets. Stage wall
times were 22.29 and 46.90 minutes respectively. These are validation results;
new external benchmark results are still pending.

Both model exports, best candidates and recovery checkpoints were downloaded
to `artifacts/backups/continuation_200m/` and verified against saved SHA-256
hashes and byte sizes. The transfer was retried after an SSH disconnect.
The receipt is `docs/results/continuation_200m_backup_verified.json`.

The next corpus contains 113,703 documents / 99,999,461 tokens. It begins after
source index 249,697 and excludes exact hashes/source IDs in the original
corpus and first fresh-corpus audit. Every saved document was re-encoded and
compared with packed bytes/indexes. Its builder again aborted during native
interpreter shutdown after writing verified output. The launcher independently
checks all saved hashes before training. Near-duplicate limitations remain.

`start_matched_continuation_v2.sh` checks the backup receipt and completed
parents, then runs baseline followed by HelixDepth with one common four-hour
ceiling. Both use the same fresh corpus and unchanged optimization recipe.
Each stage adds 12,206 updates / 99,991,552 targets, reaching 299,982,848 total.
New outputs are `artifacts/{variant}_additional_100m_v2/`; completed runs are
preserved. `--parent-report` binds cumulative accounting to the prior verified
recovery checkpoint. The previous stage's completed weights initialize each new stage, while
the optimizer and learning-rate schedule restart.

To inspect the live sequence:

```bash
cd /workspace/helixdepth-main-v2
tail -n 20 artifacts/matched_continuation_v2.log
```

Evidence: `docs/results/continuation_corpus_v2.json`,
`baseline_continuation_200m.json`, `helixdepth_continuation_200m.json` and the
backup receipt. Five continuation tests passed after extending prior-corpus
exclusion and cumulative accounting; a further backup corruption test passed.
The Pod stays running after both stages finish.

## Completed 300M-target stages and verified backups — October 2, 2026

Both v2 stages completed all 12,206 updates and selected their final checkpoint.
Each model has processed 299,982,848 targets across 36,619 total updates. Both
used the same fresh v2 corpus, context 512, batch 16, float32 and seed 2026.
Each continuation carries learned weights forward but starts a fresh optimizer
and learning-rate schedule; this is not a single uninterrupted 300M schedule.

| Measurement | Baseline | HelixDepth |
|---|---:|---:|
| Full validation loss at 100M | 3.972635 | 3.918152 |
| Full validation loss at 200M | 3.822446 | 3.768085 |
| Full validation loss at 300M | 3.747606 | 3.693470 |
| Latest stage wall time | 22.18 min | 46.70 min |
| Cumulative recorded training wall time | 65.33 min | 137.56 min |
| Maximum peak allocated GPU memory across stages | 5.51 GiB | 9.02 GiB |

Full validation scores the same frozen 385,549 next-token targets in 754 batches.
This loss is separate from WikiText-103 perplexity. Wall time includes validation
and checkpoint work inside the training timers, and excludes setup, profiling,
data preparation, external evaluation and idle time. Memory is the maximum
stage peak, not a sum. Equal targets and nearly equal parameters do not imply
equal compute: HelixDepth used about 2.11 times the recorded training wall time.

The new local backup `artifacts/backups/continuation_300m/` preserves both model
exports, best checkpoints, optimizer/recovery checkpoints, stage configurations,
metrics, progress logs, tokenizer, corpus manifest and shared launcher log.
All 16 manifest-listed files (1,651,852,647 bytes) passed independent remote and
local size/SHA-256 checks. The earlier 200M backup remains intact. No corpus
dataset was downloaded for this backup.

Evidence: [backup receipt](results/continuation_300m_backup_verified.json),
[baseline report](results/baseline_continuation_300m.json), and
[HelixDepth report](results/helixdepth_continuation_300m.json).

Recheck the backup from the local repository root:

```powershell
.venv/Scripts/python.exe scripts/verify_transfer.py --root artifacts/backups/continuation_300m --manifest artifacts/backups/continuation_300m/continuation_300m_backup_manifest.json
```

This verifies saved bytes; it does not train. Further training requires the
project owner's review of the evaluation results and remaining budget.
