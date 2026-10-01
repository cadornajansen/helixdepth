# Matched main experiment

Authorized October 1, 2026; organizer-confirmed deadline October 2, 23:59 Manila.
No top-ups or new GPU rental. Use existing Pod within remaining USD13.72;
observed balance at preparation USD13.71. No public submission is automatic.

Both random-initialized full models use configs/training_main.yaml: context512,
batch16, seed2026, AdamW(0.9,0.95), learning rate0.0003, weight decay0.1,
gradient norm limit1, 244 warmup updates, cosine decay to10% learning rate,
12,206 updates =99,991,552 scored targets. One source-order corpus pass; no
reshuffling or epoch repetition. The final7,593 potential targets are unused.
Frozen CP2 tokenizer; unchanged443-document FineWeb-Edu validation split.

Float32, TF32 disabled, deterministic algorithms and math attention are enabled
for exact recovery. Prior default-attention profiling is a planning estimate,
not measured throughput for this deterministic configuration.

The recovery gate runs each full model for4 updates uninterrupted, then2 updates
plus2 resumed in a fresh process. Every saved checkpoint field must match
exactly, including weights, optimizer, schedule, random states and data position.
Main training starts independently from the original seed after this gate.

Each run logs every update, saves an atomic recovery checkpoint every500 updates,
checks64 validation batches initially/every1000 updates, and scores the entire
validation split at the end. Training-step timing excludes validation and writes;
wall timing includes setup and reports. A three-hour process cap saves recovery
state and stops the run; it does not itself stop Pod billing. Stop the Pod after
downloads and verification. Preserve old outputs; resume into a new directory.

On the Pod, from the new extracted project with its Linux environment:

```bash
.venv/bin/python -u scripts/run_experiment.py --mode recovery --output artifacts/main_recovery
.venv/bin/python -u scripts/run_experiment.py --mode run --variant baseline --output artifacts/main_baseline
.venv/bin/python -u scripts/run_experiment.py --mode run --variant helixdepth --output artifacts/main_helixdepth
```

For recovery after interruption, keep the original config/data/software and add
`--resume artifacts/main_baseline/latest.pt` with a new output path. Do not change
the schedule horizon to the remaining steps. Download latest.pt, model.pt,
metrics.json and progress.jsonl from both variants. Verify files against hashes
in metrics.json using scripts/verify_transfer.py with --root and --manifest.
The model.pt export is for evaluation; latest.pt also includes optimizer/RNG/data.

Completion requires measured recovery, two complete schedules, matching data
positions, local checksum-verified backups and the required benchmark reports.
Prepared commands alone are not completion evidence.
