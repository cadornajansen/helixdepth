# HelixDepth

Depth-conditioned weight synthesis for parameter-efficient language models.
Six stored Transformer blocks reused across twelve stages, with small
depth-specific low-rank modifications, compared with a six-block baseline.
Both models start from random initialization.

Checkpoint 1 implements architecture and CPU correctness checks. Checkpoint 2
prepares a fixed English corpus and an own byte-level BPE tokenizer. Checkpoint 3
adds packed training, validation and verified deterministic CPU resumption.
Full-model GPU profiling and an expanded ~100M-token corpus are verified.
**300M comparison complete (October 2, 2026):** both models finished matched
continued-pretraining stages, and all ten external evaluations passed on their
first attempt. Both models, recovery checkpoints and downloaded evaluation
files passed checksum verification. No further training has started. Earlier
checkpoint sections below describe historical preparation stages.
See [verified 300M evaluation results](docs/results/evaluation_summary_300m.json),
[main protocol](docs/main-experiment.md) and [launch/recovery commands](docs/start-training.md).

## Main experiment results

Each model has processed **299,982,848 next-token targets in 36,619 updates**,
starting from random weights and continuing through two fresh-data stages.
Both use one RTX 4090, float32, batch 16, context 512 and seed 2026. Each
continuation preserves its own learned weights but restarts AdamW and its
learning-rate schedule. See the [continuation protocol](docs/continuation.md).

### Current matched 300M comparison

| Measurement | Baseline | HelixDepth |
|---|---:|---:|
| Trainable parameters, including tied embedding/output weights once | 40,968,320 | 41,184,432 |
| HellaSwag accuracy (10,042 examples) | 26.49% | 26.49% |
| ARC-Easy accuracy (2,376 examples) | 37.79% | 36.99% |
| PIQA accuracy (1,838 examples) | 56.64% | 57.07% |
| WinoGrande accuracy (1,267 examples) | 47.51% | 49.25% |
| WikiText-103 token perplexity (lower is better) | 86.51 | 85.42 |
| Full frozen training-validation loss (385,549 targets) | 3.7476 | 3.6935 |
| Cumulative recorded training wall time | 65.33 minutes | 137.56 minutes |
| Maximum peak allocated GPU memory across stages | 5.51 GiB | 9.02 GiB |

All accuracy headlines consistently use raw zero-shot accuracy from
lm-evaluation-harness 0.4.13. The [full-precision summary](docs/results/evaluation_summary_300m.json)
also retains length-normalized scores, standard errors, sample counts and
checkpoint/source hashes. Normalized PIQA slightly favors baseline, unlike
raw PIQA; we do not select a different headline metric to favor either model.
WikiText uses the same pinned test slice, tokenizer and context as the 100M run:
the first 262,144 own-tokenizer tokens, not the entire test split.

Compared with its original 100M result, HelixDepth's perplexity fell **25.30%**
(114.36 to 85.42). ARC-Easy rose **3.20 percentage points**, PIQA rose **0.60**,
HellaSwag rose **0.10**, and WinoGrande fell **1.10**. Language prediction
improved substantially; reasoning results are mixed. Baseline also improved,
including a 26.49% perplexity reduction and a 4.00-point ARC-Easy increase.

At the matched 300M budget, HelixDepth's perplexity is **1.26% lower** than
baseline's, down from a 2.83% relative advantage at 100M. It ties HellaSwag,
scores lower on ARC-Easy and higher on raw PIQA and WinoGrande. These small
differences do not establish broad reasoning superiority or statistical
significance. There is one seed per architecture, and the reported individual
standard errors do not replace a paired significance analysis or repeated runs.

This is an equal-data, approximately equal-parameter comparison, **not an
equal-compute comparison**. HelixDepth used **2.11x** recorded training wall time
and more memory. The combined three-stage training timers total **3.38 GPU-hours**
(about $2.54 at the observed $0.751/hour rate). Timers include validation and
checkpoint work within each run; they exclude setup, profiling, data preparation,
external evaluation and idle time. This estimate is not the total RunPod bill.

The experiment also cannot isolate the contribution of depth conditioning from
extra execution depth and low-rank modifications without further ablations.
Exact-document/source exclusions and domain filtering do not establish exhaustive
benchmark or near-duplicate decontamination. Perplexity depends on the tokenizer,
so these numbers are not directly comparable with another tokenizer's scores.

Both selected exports, best/recovery checkpoints, tokenizer and training evidence
are backed up locally: **16 files / 1,651,852,647 bytes**, plus the manifest and
verification receipt. All **45 evaluation evidence files** passed an independent
transfer-manifest check and all ten completion receipts were reverified against
the actual backed-up models. Earlier 100M evidence and 200M backups remain intact.
See the [backup receipt](docs/results/continuation_300m_backup_verified.json),
[baseline training report](docs/results/baseline_continuation_300m.json),
[HelixDepth training report](docs/results/helixdepth_continuation_300m.json),
and [evaluation commands and verification](docs/evaluation.md).

### Text-generation demonstration

The [generation guide](docs/generation-demo.md) provides a local CPU command
using the backed-up checkpoint and tokenizer. The standalone script does not
train or modify model files. A bounded RTX 4090 run captured three identical
prompts per model with temperature 0.8, top-k 40 and fixed seeds; all six
[unedited outputs and timings](docs/results/generation_demo_300m.json) are saved.
Outputs show repetition, incoherence and factual mistakes. These are base-model
text continuations, not a reliable conversational assistant or reasoning proof.
Local CPU replay has also passed a short actual-checkpoint smoke check.

For recording, use the [three-minute video guide](docs/video-guide.md) and
[updated 300M architecture/results diagram](docs/visuals/helixdepth-explained.svg).
The [model card](docs/model-card.md) explains the custom checkpoint format,
training provenance, reproduction commands and intended use.

### Original matched 100M comparison (historical)

Each model trained from random initialization on 99,991,552 next-token targets
in 12,206 updates, using one RTX 4090, float32, batch 16 and context 512.

| Measurement | Baseline | HelixDepth |
|---|---:|---:|
| HellaSwag accuracy (10,042 examples) | 26.41% | 26.39% |
| ARC-Easy accuracy (2,376 examples) | 33.80% | 33.80% |
| PIQA accuracy (1,838 examples) | 55.66% | 56.47% |
| WinoGrande accuracy (1,267 examples) | 49.96% | 50.36% |
| WikiText-103 token perplexity (lower is better) | 117.69 | 114.36 |
| Main-run wall time | 20.86 minutes | 43.95 minutes |
| Peak allocated GPU memory | 5.51 GiB | 9.02 GiB |

Accuracy uses the raw zero-shot metric from lm-evaluation-harness 0.4.13.
The full-precision JSON also includes length-normalized accuracy and standard
errors. WikiText uses the same tokenizer and first 262,144 tokens of the pinned
WikiText-103 test split for both models, with context 512.

HelixDepth reduced perplexity by **2.83%**, but reasoning accuracy was mostly
unchanged. This gain cost approximately **2.1x main-run wall time** and more
peak memory. One seed per architecture does not establish statistical
significance, broad reasoning superiority or a training-efficiency advantage.
Perplexity is not directly comparable across different tokenizers. Main-run
time totals about **1.08 GPU-hours**, excluding setup, profiling, recovery checks
and evaluation; it is not total billed usage.

See [evaluation protocol and verification](docs/evaluation.md),
[baseline training evidence](docs/results/main_baseline.json) and
[HelixDepth training evidence](docs/results/main_helixdepth.json).

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
At CP3, the final token budget was open. Both full models completed an RTX 4090
float32 batch-1/context-512 profile and passed timing checks after a longer
baseline repeat. GPU recovery and sustained training were verified in subsequent
stages; mixed precision was not used. See
[measured batch-1 evidence](docs/results/gpu_profile_b1_summary.json).

AI assistance: OpenAI Codex assisted with architecture implementation, data and
training scripts, tests, RunPod operations, evaluation recovery and documentation.
Context7 supplied PyTorch, Datasets and Tokenizers documentation. Later reporting
work used review and evidence-summary subagents. The project owner directed the
experiment and training decisions; AI assistance is not evidence of model quality.

## RunPod package and completed GPU profiling

See [exact Windows transfer and Linux setup/profiling/recovery commands](docs/runpod.md).
Build a checksum-verified archive with `.venv/Scripts/python.exe scripts/package_runpod.py`.
Generated packages, data, checkpoints and GPU reports remain under ignored `artifacts/`.
The CUDA environment uses the 63-package hash-pinned `requirements/runpod.lock`.
Both full models pass batches 1/2/4/8/16/32 at context 512 on RTX 4090. Batch 16 is the
recommended common setting; every tested batch now passes numerical and timing checks.
These profiling runs preceded the completed training reported above.

Measured [larger-batch GPU evidence](docs/results/gpu_profile_larger_batches_summary.json)
shows throughput plateauing at batch 16; batch 32 nearly doubles memory without
meaningful speed improvement in this run. Subsequent experiments retained
float32 and batch 16, with separate recovery checks and measured training budgets.

[Completed GPU profiling evidence](docs/results/gpu_profiling_complete.json)
combines the original runs with successful longer-window baseline repeats.
Seven remote result/environment files were downloaded and checksum-verified.
This profiling milestone preceded main-run preparation.

## CP4 preparation: expanded training corpus

Local expansion is verified: **113,907 training documents / 99,999,146 tokens**
including EOS, compared with the original 7,323,284 tokens. The original
443-document validation set and 16,384-entry tokenizer remain unchanged.
Every saved text/token/index and selection decision passed verification.
All **38 tests passed** at this preparation milestone, before main training.
The 100M figure here describes the original corpus capacity; later stages used
additional fresh corpora with the same tokenizer and frozen validation set.

See [selection policy, execution flow and exact commands](docs/corpus-expansion.md),
[measured corpus evidence](docs/results/corpus_expansion.json) and
[context-512/batch-16 loader checks](docs/results/expanded_loader.json).
Generated files are in ignored `artifacts/corpus_100m_v1/`; existing CP2 data
and the earlier RunPod transfer archive are preserved. The expanded corpus
is not included in that earlier archive.

```powershell
.venv/Scripts/python.exe scripts/expand_corpus.py --verify-only
```
