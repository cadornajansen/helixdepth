# RunPod transfer and profiling only

This package was prepared locally. The user has since deployed an RTX 4090
Pod and completed the first batch-1 profile; see docs/results/gpu_profile_b1_summary.json.
Main training is not authorized. Your current Intel UHD/CPU-only
PyTorch environment cannot run these CUDA commands. Final token budget is open.

## 1. Build on local Windows (PowerShell)

```powershell
cd C:\Development\personal\helixdepth
$bundle = 'C:\Development\personal\helixdepth\artifacts\runpod_ready\helixdepth-runpod.tar.gz'
$expected = ((Get-Content "$bundle.sha256") -split '\s+')[0]
if ((Get-FileHash $bundle -Algorithm SHA256).Hash.ToLower() -ne $expected) { throw 'Archive checksum mismatch' }
```

The package already exists: the block above verifies it without rebuilding.
To build a new snapshot, use a new directory:

```powershell
.venv/Scripts/python.exe scripts/package_runpod.py --output artifacts/runpod_repeat
```

Use that new directory's archive/sidecar for transfer. Do not overwrite existing evidence.
The archive includes source/tests/docs, both full YAML models, training/data
settings, the dependency lock, tokenizer JSON/fitting IDs/source provenance,
both packed binaries/document indexes and their manifest. It excludes `.venv`,
`.git`, raw dataset caches, and smoke checkpoints. Binary/artifact outputs stay
under Git-ignored `artifacts/`; evidence JSON/XML and lock files are reviewable.
Each payload file has a SHA256/size in `transfer_manifest.json`; the archive has
an external SHA256 sidecar. SHA256 detects changes; it is not a signature from
an independent authority. Keep the locally generated sidecar as the reference.

## 2. Transfer after separate GPU authorization (local Windows)

Use an authorized **Linux x86_64 NVIDIA CUDA Pod**, with a driver compatible
with PyTorch's CUDA 12.8 wheel, sufficient disk for the environment, and exposed
TCP SSH port 22. No automatic rental script is provided. The NVIDIA driver comes
from the host; installing a wheel does not create a GPU or replace that driver.
Use the Pod's **Connect ÃƒÂ¢Ã¢â‚¬Â Ã¢â‚¬â„¢ SSH over exposed TCP** values. RunPod's basic SSH proxy
is not the SCP connection. Fill in these three values from your connection:

```powershell
$podHost = Read-Host 'Pod public IP from SSH over exposed TCP'
$podPort = [int](Read-Host 'Mapped public SSH port')
$keyPath = Read-Host 'Absolute path to your existing private SSH key'
scp -P $podPort -i $keyPath $bundle "$bundle.sha256" "root@${podHost}:/workspace/"
ssh -p $podPort -i $keyPath "root@$podHost"
```

Source: [RunPod SSH](https://docs.runpod.io/pods/configuration/use-ssh) and
[file transfer](https://docs.runpod.io/pods/storage/transfer-files).

## 3. Verify and set up INSIDE the future Pod (Linux Bash)

```bash
cd /workspace
sha256sum -c helixdepth-runpod.tar.gz.sha256
test ! -e helixdepth || { echo 'Existing project preserved; use a fresh directory'; exit 1; }
mkdir helixdepth
tar -xzf helixdepth-runpod.tar.gz -C helixdepth
cd /workspace/helixdepth
python3 scripts/verify_transfer.py
bash scripts/setup_runpod.sh
```

The base image needs Bash, curl, tar, Python 3.11+ (for dependency-free checksum
verification), and `nvidia-smi`. Setup downloads pinned uv 0.11.28 and managed
Python **3.12.13**, creates a fresh `.venv`, installs the **63-package hash-pinned
Linux lock**, then installs this source without resolving extra dependencies.
It checks `torch==2.8.0+cu128`, CUDA availability and runs the regression suite.
It does not run profiling or training. Internet is required to install wheels;
this is not an offline environment archive. Local CPU dependencies are untouched.

Lock reproduction (local Windows; do not re-resolve on the Pod):

```powershell
uv pip compile requirements/runpod.in --python-version 3.12.13 --python-platform x86_64-unknown-linux-gnu --generate-hashes --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match --output-file requirements/runpod.lock
```

The lock contains transitive versions and hashes, including NVIDIA libraries and
Triton. Download hashes are checked during installation; changing the lock means
a new package/evidence version. Python's managed runtime and the uv installer
are version pinned but are downloaded from Astral, not included as offline files.
Sources: [uv cross-platform resolution](https://docs.astral.sh/uv/concepts/resolution/),
[uv requirements](https://docs.astral.sh/uv/pip/compile/),
[PyTorch 2.8.0 wheel commands](https://pytorch.org/get-started/previous-versions/).

## 4. Profile INSIDE the future Pod

```bash
cd /workspace/helixdepth
.venv/bin/python scripts/train.py --mode profile --device cuda --model-size full --variant both --batch-sizes 1 2 4 8 16 32 --warmup 20 --profile-steps 100 --profile-windows 3 --output artifacts/runpod_results/profile.json
```

Both full configs use vocabulary 16,384, width 640, FFN 1,792, ten heads and
context **512**. Baseline has 40,968,320 parameters and six stages; HelixDepth
has 41,184,432 parameters and twelve stages reusing its six stored blocks.
These counts were measured by construction on the meta device, not GPU execution.

Each model starts at batch 1, then 2/4/8/16/32. Each case creates fresh random
weights with seed 2026 and starts at data offset zero; both models consume the
same targets for the same batch. Warmup is 20 optimizer updates. Three separate
100-update windows are timed with CUDA synchronization before/after each window.
Throughput is total **prediction targets** divided by total measured seconds.
The timing includes data retrieval/transfer, forward/backward/loss, finite checks,
clipping, AdamW/scheduler and step metric collection. Model/data initialization,
warmup, validation, checkpoint writes, report writes and the final parameter
finite scan are excluded. **The profiler never validates or writes checkpoints.**

Each result records window rates, aggregate tokens/sec, peak allocated/reserved
PyTorch bytes, config/runtime/data identities, offsets and numerical status.
CUDA allocation counters include resident model/optimizer memory after warmup;
they are not total device/process memory. TF32 is disabled; float32/default SDPA
and nondeterministic algorithms are recorded. This does not estimate BF16/FP16.

A result is timing-stable only with at least three windows, each >=5 seconds,
and throughput coefficient of variation (standard deviation/mean) <=10%.
An unstable result is labeled; do not call it a stable speed estimate. To repeat
one unstable case with longer windows while staying within the preparation data:

```bash
# Example batch 1 only; change variant/batch to the reported unstable case.
.venv/bin/python scripts/train.py --mode profile --device cuda --model-size full --variant baseline --batch-size 1 --warmup 20 --profile-steps 200 --profile-windows 3 --output artifacts/runpod_results/profile_baseline_b1_repeat.json
```

Keep `(warmup + steps*windows)*batch*512 + 1 <= 7,323,284`. Default batch 32 uses
5,242,880 targets including warmup and fits one pass. A 200-step window at batch
32 does not fit; use a bounded different plan rather than repeating the corpus.
The cap is 32, not a claim of the largest possible GPU batch. If 32 fits, a future
higher-batch probe needs shorter windows or expanded unique data and its own
recorded timing check.

Reports are saved after every case. OOM/non-finite loss or gradient/parameter
failures record their phase and error; larger batches for that model are skipped,
then the other model is tested. Failed cases have no valid throughput estimate.
Compare only stable, successful common batch sizes. Normal completion of the
sweep does **not** mean all cases passed: inspect each `status` and `timing_stable`.
A process kill can lose only the current unfinished case; prior results remain.

## 5. Download reports to local Windows

```powershell
scp -P $podPort -i $keyPath -r "root@${podHost}:/workspace/helixdepth/artifacts/runpod_results" 'C:\Development\personal\helixdepth\artifacts\downloaded_profile'
```

The first batch-1 GPU measurements are now reviewed and recorded in repository
docs and Notion. Baseline
now passes after a longer-window repeat; batches 2/4/8/16/32 have been measured and all timing repeats now pass. The raw report is preserved under ignored artifacts/.

## 6. Checkpoint backup and recovery (for a later authorized diagnostic/run)

Profiling creates no checkpoint. These commands apply only when a separately
authorized training run has written a complete `latest.pt`. They are recovery
instructions, not an instruction to start main training.

Inside the Pod, substitute the existing run's checkpoint path:

```bash
.venv/bin/python scripts/backup_checkpoint.py --checkpoint artifacts/YOUR_EXISTING_RUN/latest.pt --output artifacts/backups/recovery_001
cd artifacts/backups
sha256sum -c recovery_001.tar.gz.sha256
cd /workspace/helixdepth
```

The helper freezes the atomic completed checkpoint, reads it with
`weights_only=True`, saves exact model/training/runtime/data metadata and the
dependency lock, then creates an inner file manifest plus archive checksum.
The original transfer manifest is included when available. Never recover `.tmp`.
Do not modify the frozen backup or reuse its output name. Keep the original
transfer archive alongside it: recovery needs the same source, data and tokenizer.

Download BOTH files to local Windows before terminating a Pod:

```powershell
New-Item -ItemType Directory -Force 'C:\Development\personal\helixdepth\artifacts\downloads' | Out-Null
scp -P $podPort -i $keyPath "root@${podHost}:/workspace/helixdepth/artifacts/backups/recovery_001.tar.gz" "root@${podHost}:/workspace/helixdepth/artifacts/backups/recovery_001.tar.gz.sha256" 'C:\Development\personal\helixdepth\artifacts\downloads\'
$backupFile = 'C:\Development\personal\helixdepth\artifacts\downloads\recovery_001.tar.gz'
$backupHash = ((Get-Content "$backupFile.sha256") -split '\s+')[0]
if ((Get-FileHash $backupFile -Algorithm SHA256).Hash.ToLower() -ne $backupHash) { throw 'Backup checksum mismatch' }
```

On a recovery Pod, first restore/verify/setup the original transfer package.
Upload the backup/sidecar to `/workspace/helixdepth/artifacts/backups/`, then:

```bash
cd /workspace/helixdepth/artifacts/backups
sha256sum -c recovery_001.tar.gz.sha256
test ! -e restored_001 || { echo 'Existing restore preserved'; exit 1; }
mkdir restored_001
tar -xzf recovery_001.tar.gz -C restored_001
cd /workspace/helixdepth
.venv/bin/python scripts/verify_transfer.py --root artifacts/backups/restored_001 --manifest artifacts/backups/restored_001/checkpoint_manifest.json
# Only after separate authorization to continue the existing run:
# Replace baseline if model_used.json says helixdepth. Use a NEW output directory.
.venv/bin/python scripts/train.py --mode run --device cuda --model-size full --variant baseline --config artifacts/backups/restored_001/training_used.json --data artifacts/cp3_data --resume artifacts/backups/restored_001/latest.pt --output artifacts/recovered_001
```

Use the saved schedule horizon: do not increase it on resume. A completed horizon
cannot continue. Model/data/settings/runtime mismatches are deliberately rejected.
Exact resume has been verified for reduced CPU models only; GPU and cross-machine
bitwise equivalence remain unvalidated. Use the same GPU type/runtime for any
deterministic GPU claim and test it before relying on it.

`/workspace` volume storage survives Pod stops/restarts but is deleted on Pod
termination; container disk is more temporary. Download and verify backups first.
Independent network volumes involve separate storage choices/costs and are not
provisioned here. Source: [RunPod storage lifecycle](https://docs.runpod.io/pods/storage/types).

## Completed profiling backup

All twelve model/batch combinations (two architectures times six batch sizes)
have passing numerical and timing checks after baseline repeats. See
[combined evidence](results/gpu_profiling_complete.json). Seven files from
artifacts/runpod_results were downloaded to local ignored
artifacts/downloaded_runpod_results; every SHA256 and size matches the remote
copy. This includes four profiling reports plus environment, parameter counts
and package verification. Profiling created no trained checkpoints to recover.

The user can now Stop the Pod to end GPU charges. Volume storage still accrues
about USD0.014/hour at the shown configuration. Termination removes its volume
and ends that storage charge; do so only after confirming no additional remote
files need saving. No stop/termination has been performed by the assistant.
The original source/data/tokenizer archive and all profiling evidence are local.
