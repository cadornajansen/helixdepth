# CP3: reliable training before GPU rental

## Local commands (PowerShell, repository root)

```powershell
.venv/Scripts/python.exe scripts/pack_data.py
.venv/Scripts/python.exe -m pytest -q -p no:cacheprovider --junitxml=docs/results/cp3_correctness.xml
.venv/Scripts/python.exe scripts/train.py --mode smoke
```

Packing and run directories are never silently overwritten. After the saved run
exists, reproduce smoke checks in a new directory:

```powershell
.venv/Scripts/python.exe scripts/train.py --mode smoke --output artifacts/cp3_smoke_repeat --report docs/results/cp3_smoke_repeat.json
```

The default mode is a reduced CPU smoke run. Main training is not started by the
default command. `configs/training.yaml` is an eight-step diagnostic protocol,
not the final experiment schedule or token budget. Full-size run mode is explicit;
do not begin main training until corpus/budget/readiness are resolved.

## Packing and target flow

The packer first checks CP2 document, token, tokenizer, manifest and fitting-record
hashes against verified evidence. It concatenates the saved token rows in their
existing deterministic order, including each document's EOS. Separate training
and validation files never mix. It writes little-endian int32 binary files and
document indexes (ID, starting token offset, token length), recording hashes,
token counts, tokenizer hash and source revision in a new manifest.

`PackedStream` memory-maps these files using `torch.from_file`; only the batch is
converted to int64 token IDs. A memory map reads file-backed data without loading
the whole expanded corpus into Python lists. Binary indexes preserve document
provenance for later inspection; the training loader reads the token stream.

At offset s, for context 512:

- Inputs: tokens s through s+511.
- Targets: tokens s+1 through s+512.
- Next sequence starts at s+512. Its first input was the preceding final target;
  each **target** is scored once. A batch contains consecutive such sequences.

The model receives exactly 512 tokens. The trainer compares all 512 logits with
the explicit shifted targets, rather than calling `LanguageModel.loss` and
shifting a second time. The model implementation is unchanged.

Attention is causal across the packed sequence, including document boundaries;
EOS separates documents but does not reset attention inside a sequence. Each
sequence resets RoPE positions to zero. This is a deliberate shared policy for
both models; document-isolated attention is not implemented. Token EOS-to-next-
document transitions are included in loss.

Training uses full batches and stops at corpus exhaustion or the configured
schedule horizon. It never automatically restarts an epoch. Any final incomplete
training batch is unused and can be calculated from manifest counts; no padding
is introduced. At batch 1/context 512, the preparation corpus supports 14,303
full training sequences (7,323,136 prediction targets), leaving 147 tail targets.
Validation includes its final short sequence and computes a token-weighted mean.
Diagnostic runs limit validation to two sequences (1,024 targets) and explicitly
report `complete_split=false`; this is not full-corpus perplexity or a benchmark.

## Optimizer, validation and recovery

Float32 only, AdamW betas (0.9,0.95), learning rate 0.0003, weight decay 0.1,
gradient-norm limit 1.0. The diagnostic schedule linearly warms up for two steps
then follows cosine decay to a 0.1 learning-rate fraction over its eight-step
horizon. Schedule settings are fixed across continuation; changing them causes
checkpoint rejection. The logged learning rate is the rate used for that update.
`gradient_norm_before_clip` records the norm before limiting it.

A training step reads the next batch without changing its data position,
clears gradients, computes finite loss, backpropagates, clips finite gradients,
updates optimizer then scheduler, and advances the data position after completion.
No gradient accumulation or mixed-precision scaler is used. Validation enters
evaluation/no-gradient mode, reads only validation tokens from its own beginning,
sums cross entropy over scored tokens and restores the previous model mode.
It does not advance training data or consume randomness.

Recovery checkpoints contain model/config, AdamW moments and counters, scheduler
state/config, completed step count, loss/LR/gradient/data-offset history, processed
tokens, training data identity and position, validation identity, Python random
state, CPU PyTorch random state, CUDA random states when applicable, runtime
version/device type/precision/deterministic settings. NumPy is not used in this
training path, so it has no relevant RNG state to restore.

The file is flushed and synchronized under `latest.pt.tmp` before atomic rename
to `latest.pt`. An incomplete write leaves the previous complete checkpoint
usable. Temporary files are not loaded. Checkpoints are written every two
completed steps and at successful run completion. Interrupting mid-step does
not save partially updated state: resume the last complete checkpoint, replaying
at most the work since it. Disk/process failures before the first checkpoint have
no recovery state. Existing model-only CP1 save/load files remain supported by
their original API; these training checkpoints are a separate format.

Restore rebuilds the tied/shared model, optimizer and scheduler, validates data
hashes/settings/runtime, restores scheduler then optimizer, then restores random
states and the next data position. Gradients are cleared because checkpoints are
at optimizer-step boundaries. Resume requires the same protocol and corpus;
an expanded/reordered corpus cannot be silently substituted.

Example manual interruption and continuation using reduced models:

```powershell
.venv/Scripts/python.exe scripts/train.py --mode run --variant baseline --model-size smoke --stop-after 4 --output artifacts/manual_first
.venv/Scripts/python.exe scripts/train.py --mode run --variant baseline --model-size smoke --resume artifacts/manual_first/latest.pt --output artifacts/manual_resumed
```

Both commands retain the same eight-step scheduler horizon. The second starts
from completed step 4 and ends at 8; its output directory must be new.

## Deterministic reliability evidence

CPU determinism uses seed 2026, one thread, deterministic algorithms, math SDPA,
float32 and TF32 disabled. For each architecture, smoke checks compare an
uninterrupted eight-step run with a four-step run resumed in a fresh Python
process. Every final model/optimizer/scheduler tensor and scalar, history, RNG
state and data offset is checked exactly. Next Python/PyTorch draws and the next
batch also match. Both models consume the same packed offsets.

Smoke configurations retain vocabulary 16,384, context 512, six stored blocks,
six/twelve stages and the same projection/sharing implementation. Width 32,
FFN 64, four heads, rank 4, depth width 8 and hypernetwork width 16 keep CPU tests
short. They are not the full 40M-parameter models. Both full-size models have since completed a short float32 RTX 4090 profile
at batch 1/context 512; mixed precision, sustained main training and deterministic
GPU resume remain unvalidated.

## Prepared GPU profiling â€” not run locally

See [RunPod transfer, setup, profiling and recovery](runpod.md) for the current
hash-pinned Linux package and explicit future GPU commands. Profiling now uses
20 warmup updates and three 100-update windows, supports ascending batch sweeps,
labels timing stability, and saves OOM/numerical failures without losing previous
cases. No validation/checkpoint writes/setup are included in measured throughput.
The local CPU-only environment cannot execute this command. No GPU was rented.

## Expand unique training text while keeping the tokenizer fixed

The 7,323,284 training tokens are preparation data. A 500M-token run would imply
about 68.3 passes over it, increasing repetition/memorization risk. Final training
tokens remain **undecided**. Do not cycle the preparation stream to reach a target.

Before the main experiment:

1. Keep the exact tokenizer JSON/hash and current validation document IDs/content
   hashes frozen. Save expansion data under a new versioned manifest/directory;
   preserve CP2 artifacts and their provenance.
2. Read additional documents from the pinned source using an explicitly bounded,
   recorded selection rule, source row/shard positions and seed. Deduplicate
   canonical text against both existing splits and all new documents.
3. Apply the same hash split rule. Existing validation hashes and source IDs,
   plus new validation-assigned groups, remain barred from training. New held-out
   groups can be recorded separately; do not change the original validation set
   for the primary comparison. Near-duplicate/benchmark filtering requires
   additional measured checks before stronger contamination claims.
4. Encode eligible new training text with the existing tokenizer, append one EOS
   per document, save IDs/hashes/provenance/counts and verify split separation.
   **Do not refit BPE.** Build a new fixed packed stream and manifest shared by
   both model runs. Report unique documents/tokens separately from tokens processed.
5. Start the main models from random initialization against this final immutable
   corpus. Smoke weights are discarded; their checkpoints cannot resume against
   different data. Choose the matched token budget only after GPU profiling.

Expansion is documented here, not executed in CP3. Current rules/deadline still
need verification before main training/submission. No paid compute authorized.

Context7 official PyTorch documentation informed checkpoint restore ordering,
deterministic SDPA, scheduling, RNG state and CUDA synchronization for this work.

## First actual GPU profile — 2026-10-01

Both full architectures passed 320 temporary optimizer updates (20 warmup plus
three 100-update windows) at context 512/batch 1 on RTX 4090, torch 2.8.0+cu128,
float32, default SDPA, deterministic=False, TF32=False. Both processed identical
offsets and 153,600 measured prediction targets. No checkpoint/main run was made.

Baseline: 10,790.49 targets/sec, peak allocated 850,584,064 bytes, reserved
929,038,336 bytes. Its timing flag is false solely because all windows lasted
4.70–4.79 seconds, below five seconds; throughput CV is 0.758%, within the limit.
Repeat baseline with three 200-update windows before treating speed as a stable
estimate. HelixDepth: 4,300.58 targets/sec, peak allocated 1,038,280,192 bytes,
reserved 1,145,044,992 bytes; windows 11.89–11.92 seconds and CV 0.0767% pass.
No loss/gradient/parameter non-finite failure was reported. Larger batches and
cost per token remain to measure; these are not language quality results.

See [measured summary](results/gpu_profile_b1_summary.json). The complete raw
report is preserved under ignored artifacts/downloaded_profile/profile_b1.json.

## Baseline timing repeat — 2026-10-01

Baseline now passes the timing rule using 20 warmup plus three 200-update
windows: 10,674.36 targets/sec, windows 9.565818/9.577533/9.635887 seconds,
throughput CV 0.3188%, same peak allocated/reserved bytes as the first run.
620 temporary updates completed, 307,200 measured targets; all numerical checks
passed. Both architectures now have successful stable batch-1 measurements.
See [repeat evidence](results/gpu_profile_b1_repeat_summary.json). Next measure
batches 2/4/8/16/32 with the existing profiler; no final budget or main run chosen.

## Larger-batch GPU profile — 2026-10-01

Both full models passed float32/context512 cases at batches 2,4,8,16,32 on RTX4090,
20warmup and 3x100 measured updates each. Every same-batch pair used identical
corpus identity and offsets. No OOM/non-finite loss, gradients or final parameters
was reported. Actual raw reports are ignored; [summary](results/gpu_profile_larger_batches_summary.json)
records the complete per-case evidence.

| Batch | Baseline targets/sec | HelixDepth targets/sec | Baseline allocated GiB | HelixDepth allocated GiB |
|---|---:|---:|---:|---:|
| 2 | 21,149.68 | 8,875.24 | 1.05 | 1.38 |
| 4 | 41,660.86 | 17,190.23 | 1.56 | 2.21 |
| 8 | 79,024.08 | 34,392.07 | 2.55 | 3.84 |
| 16 | 100,788.89 | 46,843.99 | 4.58 | 7.15 |
| 32 | 101,396.68 | 46,702.62 | 8.62 | 13.76 |

Both variants at 8/16/32 and all HelixDepth cases pass the timing rule. Baseline
2/4 remain provisional because all windows are shorter than 5seconds, with low
CV (0.156%/0.419%). Repeat those two cases using 3x200steps before labeling the
whole sweep timing-stable. Recommend batch16 for a future shared configuration:
batch32 changed speed +0.603% baseline/-0.302% HelixDepth while allocated memory
increased about1.9x. Such small speed differences are not evidence of a robust
batch32 gain or regression. Batch16 reserved memory: baseline5,163,188,224bytes,
HelixDepth8,095,006,720bytes. These counters are PyTorch allocations, not total
device memory. No mixed precision, main training or benchmark evaluation.

No larger-than-32 search is needed to make the current batch16 recommendation.
Do not extrapolate main-run cost without separately allowing for validation,
checkpoint writes, corpus loading/setup and reserves. Final budget remains open.

## Profiling timing checks complete — 2026-10-01

Longer baseline repeats now pass at batch2 (21,543.17targets/sec) and batch4
(41,670.58targets/sec). Each used20warmup+3x200measured updates, windows
9.43–10.00seconds; throughput CV0.674%/1.204%. Peak memory unchanged from
original cases. All12 model/batch combinations at1/2/4/8/16/32 now have numerical
and timing evidence. Batch16 remains the recommended common experiment setting;
no additional GPU probes were started. [Combined evidence](results/gpu_profiling_complete.json)
selects the stable repeats and preserves original reports separately.

Plain-English meaning: the complete models fit the GPU and can perform short
learning runs without reported numerical failures. We know their speed and
memory demand. This does not demonstrate good English generation or sustained
training reliability. Batch16 means16 sequences of512 tokens, or8192 prediction
targets per update. The tokenizer/data remain unchanged; larger unique training
text and final matched token budget still need preparation before main training.
