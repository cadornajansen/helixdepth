# HelixDepth

Depth-conditioned weight synthesis for parameter-efficient language models.
Six stored Transformer blocks reused across twelve stages, with small
depth-specific low-rank modifications, compared with a six-block baseline.
Both models start from random initialization.

Checkpoint 1 implements architecture and CPU correctness checks. Checkpoint 2
prepares a fixed English corpus and an own byte-level BPE tokenizer. Checkpoint 3
adds packed training, validation and verified deterministic CPU resumption.
Full-model GPU profiling and an expanded ~100M-token corpus are verified.
On October 2, the user launched the CP4/CP5 background pipeline on a replacement
Pod. The supplied terminal screenshot confirms 40 setup tests passed and the
baseline four-step GPU recovery run executed. Recovery completion and main-run
progress are not yet confirmed. Benchmark scores remain pending.
See [current training status](docs/results/training_launch.json),
[main protocol](docs/main-experiment.md) and [launch/recovery commands](docs/start-training.md).

## Reproduce checkpoint 1 (PowerShell)

```powershell
uv venv --python 3.12 .venv
uv pip install --python .venv/Scripts/python.exe -e . --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match
.venv/Scripts/python.exe count_params.py --output docs/results/parameter_counts.json
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe scripts/tiny_overfit.py --output docs/results/tiny_overfit.json
```

Counts use full model shapes on the meta device. Computation checks run reduced
models on CPU. CP1 scripts use CPU and download no pretrained weights.
Direct dependencies are pinned in `pyproject.toml`.

## Verified checkpoint 1 results

| Model | Total/trainable parameters | Stages | Stored blocks |
|---|---:|---:|---:|
| Baseline | 40,968,320 | 6 | 6 |
| HelixDepth | 41,184,432 | 12 | 6 |

HelixDepth has 216,112 more parameters (approximately 0.5275%); these are
approximately parameter-matched models. Both remain below 50M.
Eight correctness tests passed. In the 100-step synthetic CPU wiring check,
baseline loss fell 3.490663 -> 0.001904 and HelixDepth 3.488209 -> 0.001999.
These are not language-quality evaluations. See machine-readable
[counts](docs/results/parameter_counts.json) and [overfit history](docs/results/tiny_overfit.json).

See [architecture and execution flow](docs/architecture.md),
[records](docs/records.md), and [configs](configs).

## Verified checkpoint 2 results

FineWeb-Edu `sample-10BT` is pinned to revision
`87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`. A bounded 20,000-document source
prefix, seed 2026, canonical-text duplicate grouping and a 32 MiB text cap produce:

| Split | Documents | Text tokens | Tokens including document EOS |
|---|---:|---:|---:|
| Training | 8,458 | 7,314,826 | 7,323,284 |
| Validation | 443 | 385,107 | 385,550 |

Tokenizer vocabulary is exactly **16,384**, including four special tokens.
Splits were assigned before fitting; only training documents entered the trainer.
Content hashes and source IDs are disjoint. Every saved token row and text
round trip passed verification. An independent source-read/fit rerun produced
identical document, tokenizer and token-file hashes. The complete suite has
13 passing tests. Benchmark-domain exclusions are heuristic; exhaustive
benchmark overlap and near-duplicate checks are not implemented.

```powershell
.venv/Scripts/python.exe scripts/prepare_cp2.py --verify-only
```

Generated documents, manifests and tokenizer live in `artifacts/cp2/` (ignored
by Git). See [data preparation and provenance](docs/data.md),
[measured evidence](docs/results/cp2.json), and
[reproducibility evidence](docs/results/cp2_reproducibility.json).

## Checkpoint 3 local training reliability

Verified context-512 packing, explicit next-token targets, shared data order,
AdamW warmup/cosine scheduling, gradient clipping, partial token-weighted
validation, and atomic resumable checkpoints. For both reduced CPU models,
eight uninterrupted updates exactly match four updates followed by recovery
in a fresh Python process: model/optimizer/scheduler/history/RNG/data states
all agree. Both consume the same 4,096 targets. The original CP3 suite had 23 passing
tests; transfer/profiling checks bring the current suite to 28. See [training flow and corpus expansion](docs/training.md),
[CPU smoke evidence](docs/results/cp3_smoke.json), and
[packing evidence](docs/results/cp3_packing.json).

```powershell
.venv/Scripts/python.exe scripts/pack_data.py
.venv/Scripts/python.exe scripts/train.py --mode smoke
```

Existing packed/run directories are preserved; use a new output directory for
a repeat run as documented. The CP2 tokenizer remains fixed. Its 7.32M training
tokens are preparation data: 500M processed tokens would require about 68.3
passes. The stream does not cycle automatically. A larger frozen-tokenizer
collection is now verified; see the CP4 preparation section below.
Final token budget is open. Both full models completed an RTX 4090 float32
batch-1/context-512 profile; both now pass timing checks after a longer baseline repeat. GPU resumption,
mixed precision and sustained main training remain unvalidated. See
[measured batch-1 evidence](docs/results/gpu_profile_b1_summary.json).

AI assistance: OpenAI Codex implemented checkpoints 1-3 locally and ran checks, using
Context7 for PyTorch, Datasets and Tokenizers documentation. No subagents were used. The project owner
still needs to review and understand the code.

## RunPod package and completed GPU profiling

See [exact Windows transfer and Linux setup/profiling/recovery commands](docs/runpod.md).
Build a checksum-verified archive with `.venv/Scripts/python.exe scripts/package_runpod.py`.
Generated packages, data, checkpoints and GPU reports remain under ignored `artifacts/`.
The CUDA environment uses the 63-package hash-pinned `requirements/runpod.lock`.
Both full models pass batches 1/2/4/8/16/32 at context 512 on RTX 4090. Batch 16 is the
recommended common setting; every tested batch now passes numerical and timing checks.
Main training remains pending.

Measured [larger-batch GPU evidence](docs/results/gpu_profile_larger_batches_summary.json)
shows throughput plateauing at batch 16; batch 32 nearly doubles memory without
meaningful speed improvement in this run. Final training budget, competition-rule
verification and GPU recovery checks remain unresolved.

[Completed GPU profiling evidence](docs/results/gpu_profiling_complete.json)
combines the original runs with successful longer-window baseline repeats.
Seven remote result/environment files were downloaded and checksum-verified.
The GPU can now be stopped; main-run preparation is a separate next task.

## CP4 preparation: expanded training corpus

Local expansion is verified: **113,907 training documents / 99,999,146 tokens**
including EOS, compared with the original 7,323,284 tokens. The original
443-document validation set and 16,384-entry tokenizer remain unchanged.
Every saved text/token/index and selection decision passed verification.
All **38 tests passed**. No main training has started; 100M is preparation
capacity, not the final agreed training budget.

See [selection policy, execution flow and exact commands](docs/corpus-expansion.md),
[measured corpus evidence](docs/results/corpus_expansion.json) and
[context-512/batch-16 loader checks](docs/results/expanded_loader.json).
Generated files are in ignored `artifacts/corpus_100m_v1/`; existing CP2 data
and the earlier RunPod transfer archive are preserved. The expanded corpus
is not included in that earlier archive.

```powershell
.venv/Scripts/python.exe scripts/expand_corpus.py --verify-only
```
