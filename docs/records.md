# Experiment records

## 2026-10-01 15:09 Asia/Manila â€” CP1 implementation and verification

Experiment: CP1-CPU-20261001. Read the complete
`C:/Users/DDCic/Downloads/HELIXDEPTH_CODEX_HANDOFF.txt`. Inspected
`C:/Development/personal/helixdepth`: clean main working tree, origin
https://github.com/cadornajansen/helixdepth, only a title README, no applicable
on-disk AGENTS.md in the inspected ancestors/project. Followed the user-supplied
mentoring and documentation instructions. No changes were made to Abstruck.

Implemented the handoff architecture without changing its dimensions or schedule.
README was UTF-16 and converted to UTF-8. Used Context7 for PyTorch APIs.
Selected Python 3.12 and PyTorch 2.8.0 CPU locally rather than the system Python
3.14, which had no PyTorch. No GPU was rented, contacted, or used; no pretrained
weights or datasets were downloaded. OpenAI Codex wrote the code and ran checks;
no subagents were used. Work remains local, uncommitted and unpushed.

### Exact commands (PowerShell, repository root)

```powershell
uv venv --python 3.12 .venv
uv pip install --python .venv/Scripts/python.exe -e . --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match
.venv/Scripts/python.exe count_params.py --output docs/results/parameter_counts.json
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe scripts/tiny_overfit.py --output docs/results/tiny_overfit.json
.venv/Scripts/python.exe count_params.py --output docs/results/parameter_counts.json
.venv/Scripts/python.exe -m pytest -q --junitxml=docs/results/correctness.xml
git diff --check
```

The second count split the depth embedding and hypernetwork breakdown for clarity.
The second pytest run records machine-readable evidence and includes completed
test-function type annotations. No model behavior changed between runs.

### Observed results

| Component | Baseline | HelixDepth |
|---|---:|---:|
| Tied embedding/output | 10,485,760 | 10,485,760 |
| Six stored projection blocks | 30,474,240 | 30,474,240 |
| Stage norms | 7,680 | 15,360 |
| Final norm | 640 | 640 |
| Global low-rank bases | 0 | 198,656 |
| Depth embeddings | 0 | 384 |
| Hypernetwork | 0 | 9,392 |
| **Total = trainable** | **40,968,320** | **41,184,432** |

Actual instantiated meta models match both analytical predictions exactly. Both
are below 50M. Difference: 216,112 parameters, approximately 0.5275% of baseline.
Alias checks exclude the repeated head reference (10,485,760 parameters) exactly
once; schedule and block identity assertions pass. Evidence:
`results/parameter_counts.json`.

Correctness: initial run **8 passed, 1 warning in 5.91s**; final evidence run
**8 passed, 1 warning in 3.62s**. Checks cover both models' logit shape, explicit
one-token target shift, future-token isolation, prefix consistency, finite loss
and gradients, low-rank matrix orientation, every stage/projection's generated
coefficient gradients, nonzero gradients and actual updates for depth embeddings,
all hypernetwork parameters and all A/B factors over three optimizer steps,
distinct stage coefficients, exact save/load predictions and config, tied weights,
actual reused-block invocation counts, separate stage norms, RoPE norm/position
behavior, and invalid input/config rejection. Evidence: `results/correctness.xml`.

Synthetic overfit: seed 2026, one CPU thread, float32, two fixed length-16 token
sequences, tiny configs saved in JSON, AdamW lr=0.01, weight_decay=0, gradient
clip=1.0, 100 steps. Threshold declared before execution: final loss <0.2 and
<10% of initial loss. Baseline **3.4906632900 -> 0.0019038008**; HelixDepth
**3.4882090092 -> 0.0019992695**. Both passed. Evidence:
`results/tiny_overfit.json`, including loss every 20 steps.

### Environment and limitations

Python 3.12.13; Windows-11-10.0.26200-SP0; uv 0.11.28; torch 2.8.0+cpu;
pytest 8.4.2; PyYAML 6.0.2. Installed supporting packages: colorama 0.4.6,
filelock 4.0.8, fsspec 2026.9.0, iniconfig 2.3.0, jinja2 3.1.6,
markupsafe 3.0.3, mpmath 1.3.0, networkx 3.7, packaging 26.3,
pluggy 1.6.0, pygments 2.21.0, setuptools 84.0.0, sympy 1.14.0,
typing-extensions 4.16.0. Direct dependencies are pinned; no transitive lockfile
is provided yet. The environment emits a PyTorch warning because optional NumPy
is absent; all checks use tensors directly and passed without NumPy.

Full-size models were counted on meta only; execution, backward, and save/load
were validated on reduced CPU models. No full-size forward/training stability,
mixed precision, GPU memory/throughput, benchmark quality, inference performance,
or deployment file-size measurements exist. Synthetic overfit is only a wiring
check. Model checkpoints do not yet include training resumption state. Notion was
not edited; synchronization remains pending. Official rules/deadline were not
reverified in CP1; they must be checked before training/submission.

### Continuation

CP1 is verified within the stated CPU scope. Stop here. The next requested action
is CP2 data preparation: select and pin a FineWeb/FineWeb-Edu subset and provenance,
split held-out documents before fitting an own byte-level BPE tokenizer, fit only
training documents, and produce exactly 16,384 tokens including special tokens.
Keep required evaluation data out of corpus/tokenizer fitting. No contamination
removal claim is verified. No corpus download or tokenizer work has started.

Later comparisons must use the same tokenizer, document order, token budget,
context and optimizer/schedule/hardware settings; twelve vs six stages are not
compute-matched. Required eventual evaluations are HellaSwag, ARC-Easy, PIQA,
WinoGrande and WikiText-103 perplexity, subject to official-rule verification.
All-in budget remains PHP 800; no paid compute is authorized. Verify live costs
and profile before choosing equal training budgets. No benchmark gains, smaller
checkpoint than a parameter-matched baseline, or phone readiness are established.

## 2026-10-01 15:29 Asia/Manila â€” CP2 source pin and Notion synchronization

Retrospective record: the source file was created earlier in this chat; its exact
completion time was not logged. Confirmed the workspace remains
`C:/Development/personal/helixdepth`. Preserved all existing local changes.

Created `configs/data.yaml` with dataset `HuggingFaceFW/fineweb-edu`, configuration
`sample-10BT`, revision `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`, upstream split
`train`, and text column `text`. Queried Hugging Face dataset metadata with
`Invoke-RestMethod https://huggingface.co/api/datasets/HuggingFaceFW/fineweb-edu`;
the response supplied this revision and the `odc-by` license label. Consulted the
official dataset card and Context7 Datasets documentation for revision pinning
and streaming. Detailed provenance and selection rules remain pending.

Validation used the existing environment and PyYAML `safe_load`: exactly the
five expected keys, all values strings, and a 40-character revision; passed.
An exact command to inspect the saved configuration is:

```powershell
.venv/Scripts/python.exe -c "from pathlib import Path; import yaml; print(yaml.safe_load(Path('configs/data.yaml').read_text(encoding='utf-8')))"
```

No dataset was downloaded; no tokenizer was fitted or dependencies installed.
The upstream 10BT pool is not our selected corpus size or training-token budget.
The configuration has no data consumer yet. No training or evaluation result
was produced. CP2 is in progress, not complete.

User requested continuing Notion documentation after every completed change.
Read the existing HelixDepth Notion page and enhanced-Markdown documentation,
then updated its current checkpoint, CP1 checklist and measured count summary.
Added a retrospective record covering CP1 files, architecture, environment,
correctness and overfit evidence, workspace verification, virtual-environment
guidance, CP2 source selection/configuration, validation, limitations and next
action. Preserved the original 14:40 planning record and clarified the older
page's checkpoint numbering relative to the saved handoff. Training resumption,
GPU profiling and later checkpoints remain incomplete.

Synchronization succeeded and a subsequent page read confirmed CP2 status,
parameter counts, pinned revision, new record and preserved planning history.
Notion: https://app.notion.com/p/3ecd08af02fb81f29b20de9cd85bfc83
The earlier CP1 note that synchronization was pending describes historical
state; that backlog is now synchronized. Future synchronization failures must
be reported honestly.

Next action: define a bounded, reproducible document selection and held-out
document split before tokenizer fitting. Keep required evaluation material out
of training/tokenizer fitting; no contamination-removal claim is verified.
Proceed to CP3 training readiness after CP2 data/tokenizer artifacts and checks
are complete. No GPU rental/spend, push, publication or submission occurred.

## 2026-10-01 15:45 Asia/Manila â€” CP2 bounded corpus and own tokenizer verified

Experiment: CP2-CPU-20261001. Workspace confirmed
`C:/Development/personal/helixdepth`, main; existing changes preserved.
User authorized CP2 corpus selection, splitting, tokenizer fitting, evidence and
Notion synchronization. No subagents, GPU rental/use, paid compute, push,
publication or submission.

### Selection and provenance

Pinned FineWeb-Edu `sample-10BT` revision
`87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`, upstream train, text column text.
Read the first 20,000 source rows through Datasets streaming. Seed 2026.
Canonicalize Unicode NFC, CRLF/CR to LF, strip outer whitespace. Keep 200â€“20,000
UTF-8 bytes; exclude wikipedia.org, huggingface.co and github.com plus subdomains.
Group by canonical-text SHA-256, retaining original source IDs/URLs/dumps/row
indices and raw-text hashes. Order groups by SHA256(seed:selection:text_sha256),
tie-break by content hash. Greedily select at most 10,000 groups and 33,554,432
text bytes. A group that exceeds the remaining byte budget is skipped.
This is random ranking within a bounded prefix, not uniform whole-dataset sampling.

Assign the whole group before fitting using
int(SHA256(seed:split:text_sha256),16) % 1000 < 50 for validation, else training.
Duplicate text cannot cross splits. Grouping covers exact text and the stated
normalization variants; near-duplicate detection is not implemented.

Saved pinned source card and SHA-256. Dataset-level license: ODC-By 1.0;
Common Crawl terms also apply per the card. Underlying web-page copyright is not
claimed uniform. Educational filtering uses a classifier trained using Llama3
annotations, as described by upstream; no pretrained model/tokenizer downloaded.
Domain exclusions are heuristic. No evaluation datasets were loaded/added, but
absence of benchmark text republished elsewhere is not established.

### Measured results

Scanned 20,000 rows: 570 length exclusions, 140 domain exclusions, one duplicate
grouped, 19,289 eligible unique texts. Selected 8,901 documents totaling
33,554,189 UTF-8 text bytes. The duplicate group was outside the selected subset;
synthetic unit tests verify grouping behavior with exact and normalized duplicates.

| Split | Documents/source IDs | UTF-8 text bytes | Text tokens | Tokens including EOS |
|---|---:|---:|---:|---:|
| Training | 8,458 | 31,860,667 | 7,314,826 | 7,323,284 |
| Validation | 443 | 1,693,522 | 385,107 | 385,550 |

Own byte-level BPE fitted exclusively from saved training text, all 256 byte
symbols available initially, minimum pair frequency 2, one tokenizer worker.
Vocabulary exactly 16,384 including PAD=0, BOS=1, EOS=2, UNK=3. No tokenizer
normalizer or automatic post-processor. One EOS appended per saved document;
BOS/PAD reserved. IDs actually consumed by the fitting iterator recorded and
verified against the entire training split. Both splits encoded with the same
saved/reloaded tokenizer.

All document hashes and seeded assignments, source-ID/content separation,
bounds, fitted IDs, vocabulary/contiguous IDs/byte alphabet/special IDs, every
saved token sequence, all document decode round trips, no UNK and Unicode probes
passed. Verification-only rerun matched the saved artifact hashes.
Final complete suite: **13 passed in 2.26s**, no warnings (8 model + 5 data tests).

A full independent source read, selection and tokenizer fit in
`artifacts/cp2_repeat` matched seven artifact hashes byte-for-byte: train.jsonl,
validation.jsonl, tokenizer.json, both token JSONL files, tokenizer_training.json,
source_card.md. Counts also match. Manifest files differ in output settings and
code provenance; no identical-manifest claim is made. The first development run
was refitted from the unchanged verified splits after fitting-ID auditing was
added; final refit and independent rerun agree. Original preparation-code hash
remains in the first manifest; final fitting-code hash is in tokenizer provenance.

### Files and environment

Modified: configs/data.yaml (bounds/split/tokenizer settings), pyproject.toml
(pinned data dependencies), .gitignore (artifacts), README.md (measured CP2 state),
docs/records.md (this appended record). Created: src/helixdepth/data.py,
scripts/prepare_cp2.py, tests/test_data.py, docs/data.md,
docs/results/cp2.json, docs/results/cp2_reproducibility.json,
docs/results/cp2_correctness.xml, docs/results/cp2_environment.txt.
No model architecture/config files changed; no files deleted.

Generated ignored files in artifacts/cp2 and artifacts/cp2_repeat:
train.jsonl, validation.jsonl, train.tokens.jsonl, validation.tokens.jsonl,
tokenizer.json, tokenizer_training.json, manifest.json, source_card.md.
Also generated artifacts/cp2_reproduction.yaml and artifacts/cp2_repeat/report.json.
All remain local; preserve these ignored artifacts before any future cleanup.
The primary corpus files include source IDs and canonical/raw content hashes;
the manifest records source revision, rules, selection counts, file hashes,
versions and code hashes. cp2.json records hashes/byte sizes of all primary files.

Installed datasets==5.0.1 and tokenizers==0.23.2 into the existing Python 3.12.13
environment; torch remains 2.8.0+cpu. Their dependencies installed NumPy 2.5.3 and
changed fsspec from 2026.9.0 to 2026.6.0. Model regression tests passed. Full observed
environment is saved in cp2_environment.txt; direct dependencies are pinned.
Consulted Context7 official Datasets revision/streaming and Tokenizers
byte-level BPE trainer/save/load documentation, and the pinned dataset card.
Preparation refuses to overwrite existing directories. --fit-existing permits
an explicit refit on unchanged checksum-verified splits; --verify-only checks
saved artifacts against previous evidence.

### Exact commands run (PowerShell, repository root)

```powershell
uv pip install --python .venv/Scripts/python.exe 'datasets==5.0.1' 'tokenizers==0.23.2'
.venv/Scripts/python.exe -m pytest -q -p no:cacheprovider --junitxml=docs/results/cp2_correctness.xml
.venv/Scripts/python.exe scripts/prepare_cp2.py
.venv/Scripts/python.exe scripts/prepare_cp2.py --fit-existing
.venv/Scripts/python.exe scripts/prepare_cp2.py --verify-only
.venv/Scripts/python.exe -c "from pathlib import Path; import yaml; config = yaml.safe_load(Path('configs/data.yaml').read_text(encoding='utf-8')); config['output_dir'] = 'artifacts/cp2_repeat'; Path('artifacts/cp2_reproduction.yaml').write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')"
.venv/Scripts/python.exe scripts/prepare_cp2.py --config artifacts/cp2_reproduction.yaml --report artifacts/cp2_repeat/report.json
uv pip freeze --python .venv/Scripts/python.exe | Set-Content -Encoding utf8 docs/results/cp2_environment.txt
git diff --check
```

The test command ran first with 11 passing tests (3 new data tests), then after
adding fitting-isolation/tamper tests with the final 13 passing tests. Independent
rerun comparison checked split counts and the seven SHA-256 values in both reports
and saved cp2_reproducibility.json. See docs/data.md for clean reproduction setup.

### Completion and remaining limits

**CP2 requested completion criteria pass:** bounded deterministic selection,
saved splits/IDs/provenance/hashes, duplicate-safe pre-fit split, training-only
own tokenizer with exactly 16,384 entries, verified shared token artifacts,
measured counts and reproducible commands/evidence.
The small prepared corpus does not establish an eventual training budget or
language quality. Packing/masking, resumable training state, full-size stability,
GPU profiling and required benchmark evaluation remain for later checkpoints.
Official competition rules/deadline still require rechecking before model
training/submission. Total budget remains PHP 800; paid compute spend is zero.

Notion synchronization succeeded. A subsequent read confirmed CP2 completion,
the timestamped record, measured counts, exact vocabulary and preservation of
the original planning and CP1/source-pin history. Project page:
https://app.notion.com/p/3ecd08af02fb81f29b20de9cd85bfc83

Next action after the user requests it: CP3 training readiness, starting locally
with token-file loading/packing and resumable training state. Stop at CP2 now.

## 2026-10-01 16:06 Asia/Manila â€” CP3 pre-rental training reliability verified

Experiment: CP3-CPU-20261001. Confirmed C:/Development/personal/helixdepth,
main; preserved all existing changes. User authorized CP3 packing, training
recovery, short CPU diagnostics, a prepared GPU profiling command, and repository/
Notion evidence. No subagents, dependency installations, GPU rental/use, paid
compute, main training, push, publication or submission.

### Implementation and data flow

Checksum-verified CP2 token/doc/tokenizer/manifest/fitting artifacts before
packing. Fixed tokenizer SHA-256 remains
`a98dbd3bb95de3ea9823b963e09c9069b25f2cf56bb1c5bb4f60a23cb7542bb2`.
Packed 7,323,284 training and 385,550 validation tokens into separate little-endian
int32 files, retaining EOS boundaries and exact CP2 document order. Document
indexes save content IDs, offsets and token lengths; packed manifest stores
source revision, source-token hashes, binary/index hashes and tokenizer identity.

Memory-mapped token streams produce [batch,512] inputs and targets shifted one
position. Windows advance by 512, overlapping one token as input/preceding target;
prediction targets are not duplicated. Trainer uses explicit target cross entropy,
without calling the model's internally shifting loss API. Cross-document causal
attention and EOS-to-next-document transitions are deliberately included within
packed windows. RoPE resets for each sequence. No padding or document-isolated
attention masks. Both architectures use identical order. No implicit epoch cycling:
the one-pass loader fails on exhaustion. Full training batches only; at batch 1
this corpus supports 14,303 sequences and leaves 147 tail targets unused.

Added float32 AdamW (betas 0.9/0.95, weight decay 0.1, base lr 0.0003), two-step
warmup then cosine decay to 0.1 LR fraction over an eight-step diagnostic horizon,
finite-loss/gradient rejection and norm clipping at 1.0. Schedule is a smoke
protocol, not the final experiment budget. Logged loss, applied LR, unclipped
gradient norm, step, target count and before/after data positions. Validation uses
only its own stream, includes a final short window, and reports token-weighted
loss with scored-token count and full/partial flag. It restores model mode and
does not alter training position or RNG.

Resumable files include model/config, optimizer moments/counters, scheduler/config,
step/history, processed tokens, data identity/position (epoch zero), validation
identity, Python/CPU PyTorch/CUDA RNG states as applicable and runtime settings.
NumPy is not used in the training path. Restore validates corpus/config/runtime,
loads scheduler then optimizer after construction, restores RNG and data state,
and clears gradients at the optimizer-step boundary. Writes flush/fsync a
temporary file then atomically replace the complete checkpoint. Periodic saves
every two steps. Interrupted mid-step resumes from the previous saved boundary;
work after it is replayed. Incomplete write test confirms prior file survives.

### Measured CPU evidence

Reduced models preserve vocabulary 16,384, context 512, six stored blocks,
six/twelve stages and architecture code. Width 32, heads 4, FFN 64, rank 4,
depth width 8, hyper width 16. Parameters: baseline 586,144; HelixDepth 589,420.
Seed 2026, one thread, deterministic algorithms, math SDPA, float32, TF32 disabled.

| Run | First/last train loss | Initial/final partial validation loss | Training-loop seconds |
|---|---|---|---:|
| Baseline | 9.7118797302 / 9.7016086578 | 9.7038607597 / 9.6966905594 | 1.1095486 |
| HelixDepth | 9.7118682861 / 9.6894750595 | 9.7036700249 / 9.6893329620 | 1.8021293 |

Each uninterrupted run performed eight updates / 4,096 prediction targets.
Validation scored the same first 1,024 validation targets at each endpoint,
`complete_split=false`. These short random-initialization losses and CPU times
are diagnostics, not quality/performance comparisons of full-size models.

For each architecture, also ran four steps, saved state, launched a fresh Python
process and resumed to eight. Entire final model/optimizer/scheduler/history/RNG/
data states matched exactly, as did the next random draws and next data batch.
Both architectures' data states match, next offset 4,096. Serialized checkpoint
bytes/sizes need not match; equivalence compares loaded logical state/tensors.

Final suite: **23 passed in 3.22s**, no warnings. Eight original model checks,
five CP2 data checks and ten CP3 cases cover shift/order/exhaustion, checksum
tampering, explicit loss targets, exact resume for both variants, validation tail/
mode/RNG/data preservation, changed-data/protocol rejection, unavailable-CUDA
guard, interrupted checkpoint-write preservation and actual clipping limit.
An earlier 21-test run passed before adding the final write/clipping checks.

### GPU profiling and corpus expansion

Prepared, did not execute:
```powershell
python scripts/train.py --mode profile --device cuda --model-size full --variant both --batch-size 1 --warmup 3 --profile-steps 10 --output docs/results/gpu_profile.json
```
Uses both full documented models, equal data order/context/batch, float32, default
SDPA and deterministic algorithms disabled (recorded). Initializes optimizer in
warmup, synchronizes CUDA, resets peak stats, times ten complete training updates,
then synchronizes. Reports prediction target tokens/sec, peak allocated/reserved
bytes, runtime/hardware/config and data offsets. Includes transfers, forward/
backward, clipping and optimizer/scheduler; excludes validation/checkpoint I/O.
CUDA absence and missing explicit CUDA/full-model flags reject profiling. No
GPU measurement is claimed; full-size compute, mixed precision, GPU exact resume
and GPU cost/time remain unvalidated.

Documented expansion in docs/training.md: freeze tokenizer and existing validation
IDs/content hashes; read more unique pinned-source documents under a new bounded,
versioned manifest; deduplicate against old/new corpus; keep existing and new
validation-assigned hash groups/source IDs out of training; encode with fixed
tokenizer, do not refit; verify separation and make one immutable expanded stream
shared by both main models. New held-out groups remain separate from the original
primary validation set. Near-duplicate/benchmark-overlap removal remains unverified.
No expansion ran in CP3. Preparation corpus implies 68.2754 passes for 500M
tokens; loader does not automatically cycle. Main models must start from random
initialization, not smoke weights, after expansion and profiling. Final budget
is explicitly null/open. Official rules/deadline still need rechecking first.

### Files and exact commands

Created: configs/training.yaml, src/helixdepth/packing.py,
src/helixdepth/training.py, scripts/pack_data.py, scripts/train.py,
tests/test_training.py, docs/training.md, docs/results/cp3_packing.json,
docs/results/cp3_smoke.json, docs/results/cp3_correctness.xml,
docs/results/cp3_readiness.json. Modified README.md and docs/records.md.
No files deleted. Model code/configs and CP2 corpus/tokenizer remain unchanged.

Generated ignored local artifacts: artifacts/cp3_data/manifest.json,
train.bin, validation.bin, train.index.jsonl, validation.index.jsonl;
artifacts/cp3_smoke/training_used.json; for each of baseline and helixdepth,
uninterrupted/interrupted/resumed directories each contain latest.pt and
metrics.json. Temporary pytest fixtures were managed by pytest, not saved project
artifacts. Preserve ignored recovery files before cleanup. cp3_readiness.json
records source/config/evidence hashes and explicit unvalidated/pending items.

```powershell
.venv/Scripts/python.exe scripts/pack_data.py
.venv/Scripts/python.exe -m pytest -q -p no:cacheprovider --junitxml=docs/results/cp3_correctness.xml
.venv/Scripts/python.exe scripts/train.py --mode smoke
.venv/Scripts/python.exe scripts/train.py --help
git diff --check
```

Smoke checks launch resumed subprocesses with the saved identical training config;
exact child arguments and paths are in scripts/train.py and saved run metrics.
Manual resume and fresh-directory repeat commands are in docs/training.md.
Context7 official PyTorch 2.8 documentation supplied scheduler/optimizer restore
ordering, RNG state, deterministic SDPA and CUDA timing guidance. No new packages.

### Outcome and next action

**Requested pre-rental CP3 scope passes.** GPU profiling is prepared, not executed.
Do not infer full-size/GPU stability or a final training budget from reduced CPU
checks. Notion synchronization succeeded. A subsequent page read confirmed
current CP3 status, the new record, 23 tests, exact fresh-process resume, open
budget and preserved planning/CP1/CP2 history. Project page:
https://app.notion.com/p/3ecd08af02fb81f29b20de9cd85bfc83

Next requested action: expand unique training corpus with the fixed tokenizer,
then perform separately authorized GPU profiling before choosing main-run token
budgets. Stop now; no rental or main training has begun.

## 2026-10-01 16:15 Asia/Manila â€” Local CUDA profiling guard explained

User screenshot shows CP2 verification output and a user-run test result of
23 passed in 5.11s, followed by the prepared full-model CUDA profile command
raising the expected availability guard. Screenshot evidence is not an agent-run
GPU profile or new agent-run test result.

Confirmed the workspace is HelixDepth. Queried the project interpreter:
`.venv/Scripts/python.exe -c "import sys, torch; print(sys.executable); print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"`.
Observed Python in the project .venv, PyTorch 2.8.0+cpu, CUDA build None and
CUDA available False. Inspected scripts/train.py: the guard raises before
model construction or profiling. Context7 official PyTorch docs consulted.
The screenshot used bare `python`; its interpreter was not independently
identified, while the project .venv result above was verified directly.

Explained that the profiling command is for a later authorized GPU machine
with CUDA-enabled PyTorch. Current local CPU checks remain passing; this guard
is not a model/test failure. No code/dependencies changed, no GPU profiling or
main training started, no rental/spend, and no push/publication. Only this local
record and its Notion counterpart were added. Next action remains unique-corpus
expansion with the fixed tokenizer, then separately authorized GPU profiling.

## 2026-10-01 16:19 Asia/Manila â€” Hardware cause of local CUDA failure identified

Investigated the user's request to make the CUDA profiling command work.
`Get-CimInstance Win32_VideoController | Select-Object Name, DriverVersion`
reports Intel(R) UHD Graphics, driver 32.0.101.7076. No nvidia-smi command was
found. Windows reports no usable NVIDIA graphics device in this inspection.
The installed CPU-only PyTorch is also incompatible with CUDA, but replacing
that package alone cannot make Intel UHD execute CUDA. Consulted Context7
PyTorch documentation for CUDA platform prerequisites.

Explained directly that the exact CUDA profiling command requires a compatible
GPU machine; this local machine can run the existing CPU smoke mode. Provided
an explicit fresh-output CPU command rather than suggesting that CPU smoke
produces GPU profiling measurements. No software or code changed, no rental
or GPU computation performed. Updated records and Notion only.

## 2026-10-01 16:39 Asia/Manila â€” RunPod transfer and profiling preparation

Scope: prepare a future authorized GPU environment, transfer/checksum package,
full-model batch profiling and checkpoint recovery. No GPU rental, spending,
GPU execution, main training, publishing or push occurred. Working directory
was verified as C:/Development/personal/helixdepth; existing main/uncommitted
changes were preserved. Model architecture and fixed tokenizer are unchanged.

Created requirements/runpod.in and requirements/runpod.lock with uv 0.11.28,
targeting Linux x86_64 / Python 3.12.13. Resolution pinned 63 packages with
artifact hashes, including torch 2.8.0+cu128, PyYAML 6.0.2, pytest 8.4.2,
datasets 5.0.1, tokenizers 0.23.2 and build tools. Local CPU environment was
not modified. Consulted Context7 for PyTorch/uv/RunPod and official RunPod
storage/SSH plus PyTorch wheel documentation. Linux installation itself is
unvalidated until an authorized Pod runs the provided setup script.

Profiling now delegates to src/helixdepth/profiling.py. Both full model YAML
configs remain included, with context 512 and known meta-device parameter
counts 40,968,320 / 41,184,432. The prepared batch sweep is 1,2,4,8,16,32;
each fresh case resets seed/data position, warms up 20 updates and measures
three windows of 100 updates. Timed work is training-step data transfer,
forward/loss/backward, finite checks, clipping, optimizer/scheduler and metrics.
Setup/warmup, validation, checkpoint/report writes and post-measurement finite
parameter scan are excluded. CUDA synchronization brackets each window.
Stability requires three windows each >=5 seconds and throughput CV <=10%.
Results save after each case; OOM/numerical failures record phase/error and
skip larger batches for that architecture, while preserving other results.
GPU tokens/sec, peak memory, actual numerical status remain NOT MEASURED.

Transfer builder uses an explicit source/data allowlist and creates ignored
artifacts/runpod_ready/helixdepth-runpod.tar.gz plus external SHA256 sidecar. An inner
manifest hashes every payload file. The archive contains code/tests/docs,
locked dependencies, model/training/data configs, exact CP2 tokenizer/fitting
IDs/source card/revision/selection manifest, and packed train/validation data
with document indexes. It excludes .venv/.git/raw caches/smoke weights. Extracted
round-trip verification checks every payload hash/size, tokenizer vocabulary
16,384 and original CP2 hashes, fitting IDs against training IDs, zero document
split overlap, source revision and packed/index counts/hashes. Measured archive
size, checksum and complete payload inventory are in docs/results/runpod_package.json
(the report is excluded from its own archive to avoid a checksum cycle).

Created scripts/verify_transfer.py, scripts/package_runpod.py,
scripts/setup_runpod.sh, scripts/backup_checkpoint.py, tests/test_runpod.py,
docs/runpod.md and the requirements files. Modified scripts/train.py,
README.md, docs/training.md and this record. Added measured evidence
runpod_correctness.xml, runpod_readiness.json, runpod_package.json under
docs/results/. No files deleted. New binary archives/data/checkpoint copies
remain under the existing Git-ignored artifacts/ directory.

Verification: 28 tests passed (2.93 seconds in the recorded final pre-package
run); tests cover timed boundaries/counts, OOM retention/other-model continuation,
non-finite failure reporting and corrupt/missing/unsafe file rejection, using
explicit test doubles rather than fabricated GPU measurements. Bash syntax
check passed; the setup script was not executed. Created a backup of an existing
reduced CPU baseline checkpoint at step 4 / 2,048 targets, extracted it, checked
six payload files and identical checkpoint SHA256, successfully loaded it into
the trainer and verified the next batch shape [1,512]. Backup/recovery archive
hash and load evidence are in docs/results/runpod_readiness.json. GPU recovery
bitwise equivalence remains unvalidated; exact matching runtime/settings/data
are required by the existing loader.

Exact future Windows upload/SSH and Linux setup/profile/download/recovery
commands are in docs/runpod.md. Profiling writes reports, never checkpoints;
recovery examples apply only to a later separately authorized existing run.
/workspace survives stops but volume data is deleted on termination: download
and verify backups first. Main-run corpus expansion with the fixed tokenizer,
competition rules/deadline checks and final token budget remain pending. The
7.32M-token corpus is preparation data and is never silently cycled. Notion
receives this record and the verified bundle checksum after local verification.

## 2026-10-01 19:57 Asia/Manila — RunPod deployment guidance

User requested deployment guidance from a console screenshot. Screenshot shows
RunPod PyTorch 2.8.0, one RTX 4090 / 24GB, GPU rate USD 0.74/hour and a warning
that nothing is mounted at /workspace. These are screenshot observations, not
an API-confirmed deployment or final quote. Workspace remains HelixDepth.

Consulted Context7 RunPod documentation plus official manage-pods/storage/SSH
pages. Recommended one on-demand GPU, existing PyTorch template, pod name
helixdepth-profile, current 30GB container disk, 50GB volume disk mounted at
/workspace (following the shown template recommendation), and SSH terminal
access/TCP22 with a public key registered under Credentials. Final displayed
cost must include storage. Guidance explains that Deploy starts paid compute,
that this session is setup/profiling only, and that direct SSH over exposed TCP
is needed for the prepared SCP transfer. No Pod was deployed by the assistant,
no keys created, no rental/spend performed, no main training started, no GPU
results claimed. Next user action: configure storage/SSH, then obtain the
running Pod's Connect details for transfer. Original transfer archive unchanged.

## 2026-10-01 20:22 Asia/Manila — Storage choice clarified

Screenshots show Network volume selected and Volume disk (Legacy) available
in the top-right card. Clarified the recommendation: select Volume disk (Legacy),
50GB, mounted at /workspace for the profiling Pod. It survives stops/restarts
but is deleted on termination; download results first. Confirmed with Context7
RunPod storage docs. No storage created or paid operation performed by assistant.
Notion synchronization remains pending from the previous server error.

## 2026-10-01 20:38 Asia/Manila — USD 15 deployment budget checked

User clarified the available budget is USD 15. Checked official RunPod pricing
via Context7 and https://docs.runpod.io/pods/pricing: running volume/container
storage USD 0.10/GB/month, stopped volume USD 0.20/GB/month; prorated billing,
not a full-month upfront storage charge. Screenshot GPU quote USD 0.74/hour.
Using a 30-day month for approximate arithmetic: 50GB running volume costs
USD 0.006944/hour, 30GB running container disk USD 0.004167/hour, combined with
GPU approximately USD 0.751111/hour. Final console quote is authoritative.
50GB is conservative setup headroom following template guidance, not measured
minimum disk consumption. Retained this storage recommendation; GPU uptime is
the dominant cost. Recommended an initial two-hour setup/profiling budget target
(approximately USD 1.50), keeping roughly USD 13.50 for later work; this is not
an automatic stop or a promise that profiling will complete within two hours.
Full experiment affordability remains unknown until actual profiling results.
Download/verify results before termination; stopped storage continues billing.
No Pod/storage deployment, spending, training or GPU measurement by assistant.
Notion sync remains pending due to the previously observed service errors.

Notion service recovered during this budget check: deployment guidance, storage choice and USD 15 cost calculations were synchronized and read back successfully.

## 2026-10-01 20:42 Asia/Manila — Deployment summary visually checked

Screenshot shows official RunPod PyTorch 2.8.0 CUDA12.8.1 image, one RTX4090
24GB, total disk80GB, GPU USD0.74/hour, container USD0.004/hour, volume
USD0.007/hour, displayed total USD0.75/hour and stopped storage USD0.014/hour.
This matches the intended30GB container plus50GB volume configuration and
budget estimate. Screenshot does not show SSH settings/account public key or
On-Demand pricing selection; those remain to check before deployment. Suggested
optional name helixdepth-profile; current generated name is also valid. Next
step after checks: user deploys setup/profiling Pod and supplies Connect/SSH over
exposed TCP details. No assistant deployment, spend or training. Consulted
Context7 RunPod deployment/SSH documentation. Original transfer archive unchanged.

## 2026-10-01 20:54 Asia/Manila — RTX 3090 recommended for available profiling capacity

User reports other GPUs unavailable; screenshot shows RTX3090 with24GB at
USD0.50/hour, while RTX4090 remains selected/unavailable. Verified NVIDIA
RTX3090 official Ampere/24GB specifications and consulted Context7 PyTorch2.8
CUDA notes. Existing project uses41M-or-less full models/context512, float32,
CUDA12.8 build; no model/dependency change is needed solely for this GPU choice.
Expected to fit at batch1, but full GPU execution/memory and maximum fitting
batch remain unvalidated. Profiling measures each model separately, starts1 and
records OOM at larger batches. Recommend selecting one RTX3090/On-Demand,
retaining storage/template/SSH settings, and checking sidebar updates to3090.
At unchanged storage quotes USD0.004+0.007/hour, projected total USD0.511/hour,
approximately USD1.02 for two hours. These are estimates from screenshot prices,
not measured run costs. Cheaper hourly rate does not prove cheaper cost per token;
actual throughput still needs profiling. No assistant rental/spend/deployment
or main training. Original transfer archive unchanged.

## 2026-10-01 21:00 Asia/Manila — User deployed Pod; SSH registration next

User reports deployment on RTX4090. Connect screenshot shows running Pod
helixdepth-data (msj4y79sp099wi), Jupyter ready, SSH key not configured, and
direct TCP SSH section partially obscured; mapped port cannot be read safely.
Verified workspace HelixDepth and found existing local id_ed25519.pub. No keys
created/overwritten. Recommend displaying existing public key in local PowerShell,
pasting it in RunPod Title/Key form, and obtaining full SSH over exposed TCP
connection command. Existing running Pod may need authorized_keys injection if
account key is not automatically applied, per Context7 docs. User deployment
is observed/reported; assistant did not rent or start training. GPU billing is
now running; setup/profiling only, main training remains off. Original transfer
archive unchanged.

## 2026-10-01 21:30 Asia/Manila — Exact Pod transfer command prepared

User supplied SSH root@47.47.180.47 port11420 using id_ed25519. Verified local
HelixDepth workspace, private key file existence (contents not read) and final
transfer archive checksum ba8c463551f16d90d87f2ecd27a057dd775aed28abf4dc01ad291d18fbc3d31b,
13,394,801 bytes. Prepared PowerShell SCP command sending archive and sidecar
to /workspace using this exact IP/port/key, followed by SSH login. Consulted
Context7 RunPod SCP syntax. Remote transfer/login not performed by assistant
or verified yet. User should see both uploads finish before login. Setup and
profiling follow after transfer confirmation; main training remains off.

## 2026-10-01 21:34 Asia/Manila — SSH password fallback diagnosed

SCP screenshot prompts root server password and fails. Noninteractive direct
SSH diagnostic with BatchMode=yes/IdentitiesOnly=yes reached47.47.180.47:11420,
matched known host and offered intended local Ed25519 key, but server rejected
it. Public-key fingerprint from .pub matches the offered key; private key
contents were not inspected. This is server-password fallback, not a key
passphrase prompt. Account key was registered after Pod started; missing Pod
authorized_keys entry is likely (not yet remotely inspected). Context7 confirms
running Pods may require explicit public-key injection. Prepared commands to
append public key without overwriting other keys in existing root Pod terminal,
set .ssh700/authorized_keys600, then retry SCP with password fallback disabled.
No remote key file modification/transfer performed or success claimed yet.
No Pod restart, main training or additional rental. Original archive unchanged.

## 2026-10-01 21:52 Asia/Manila — Remote upload checksum verified

User reports upload complete. Read-only SSH using the configured key now succeeds
on47.47.180.47:11420 with BatchMode=yes/IdentitiesOnly=yes. Executed remote
cd /workspace && sha256sum -c helixdepth-runpod.tar.gz.sha256; returned
helixdepth-runpod.tar.gz: OK, exit0. This confirms archive bytes match uploaded
sidecar (local archive/sidecar previously independently checked). Existing key
authentication issue is resolved. Inspected existing setup_runpod.sh before
preparing guarded extraction, internal payload verification and pinned environment
setup commands. Setup installs dependencies, verifies CUDA/counts and runs tests;
no profiling/main training. Extraction/install not executed or claimed completed
by assistant. No source changes; docs record updated. Original archive unchanged.

## 2026-10-01 22:00 Asia/Manila — RunPod pinned setup verified; batch1 profile next

User screenshots show successful initial setup:28 tests passed in3.27seconds and
Setup complete message. Second screenshot is a rerun refused by Existing .venv
preserved guard; this is expected preservation, not a failed initial setup.
Read-only SSH verified Python3.12.13, torch2.8.0+cu128, CUDA runtime12.8,
torch.cuda.is_available()==True and NVIDIA GeForce RTX4090. Existing profile.json
and profile_b1.json are absent. Screenshot driver580.159.04 reports supported
CUDA13.0; installed torch runtime12.8 is confirmed separately. Parameter count
output uses meta device; full-model GPU execution still requires the profile.

Recommended next one bounded run: both full models, context512, batch1,
20warmup updates and three100-update windows, output ignored
artifacts/runpod_results/profile_b1.json. Profiling temporarily updates fresh
weights to measure training speed; it saves no trained model/checkpoint and
is not main training. Saves rates/memory/numerical status/stability. Larger
batch sweep follows only after inspecting first actual results. No profile
started by assistant; no GPU throughput/memory result fabricated. Consulted
Context7 PyTorch CUDA timing docs. No source changes or environment replacement.

## 2026-10-01 22:08 Asia/Manila — First measured full-model GPU profile

Downloaded actual profile_b1.json via SCP to ignored artifacts/downloaded_profile/;
SHA256 ac901b076884bfe0bf6d188ccb95768bf195ce2d7cd6ebdd24e7efd5b735b160.
RTX4090, torch2.8.0+cu128, float32/defaultSDPA/TF32off, context512/batch1,
20warmup plus3x100measured updates, 320completed eachmodel, 153600measured
prediction targets. Full-model constructed parameters40968320/41184432.
Both statuses passed with finite loss/gradient metrics; passed path also verifies
finite final model parameters. Same data identities and offsets10240→163840.
No checkpoints or main training. GPU execution is now measured, not inferred.

Baseline10790.488529targets/sec, peak allocated850584064bytes/reserved929038336;
three windows4.744844/4.700893/4.789021seconds, CV0.00758258. stableFalse is
solely the >=5second window minimum; CV is low and numerical checks pass.
HelixDepth4300.579932targets/sec, peak allocated1038280192/reserved1145044992;
windows11.893688/11.906437/11.915988seconds, CV0.000767384, stableTrue.
Baseline speed remains provisional under declared timing rule. Recommended
baseline-only repeat with200steps/window; larger batch sweep after review.
No actual-cost estimate or language-quality claim from this short profile.

Created docs/results/gpu_profile_b1_summary.json; updated README.md,
docs/training.md, docs/runpod.md and docs/records.md to reflect actual execution.
No code/model/dependency/tokenizer changes. Raw downloaded report ignored by
Git; original transfer archive unchanged. No assistant-run GPU training/profile;
this run was executed by user. GPU resume, sustained training and final budget
remain unvalidated/open. Local working directory verified HelixDepth.

## 2026-10-01 22:15 Asia/Manila — Stable baseline batch1 timing confirmed

Downloaded actual baseline repeat profile via SCP; raw report ignored under
artifacts/downloaded_profile/profile_baseline_b1_repeat.json, SHA256
c95ce60f04c1559345c54f963b9ebc8d019626bb53fb102e22ace7c195dffdf5.
Full baseline40968320parameters, RTX4090/float32/context512/batch1,
20warmup plus3x200measured updates, completed620, measured307200targets.
Speed10674.361529targets/sec; windows9.565818/9.577533/9.635887seconds,
CV0.00318807, passed/stableTrue, numerical checks passed. Peak allocated
850584064bytes/reserved929038336bytes; offsets10240→317440.
Both models now have stable batch1 measurements. GPU quality, sustained
training and GPU resume are not established by throughput profiling.

Created docs/results/gpu_profile_b1_repeat_summary.json; updated README.md,
docs/training.md, docs/runpod.md and this record. No code/dependency/model or
tokenizer changes; raw report ignored; original transfer archive unchanged.
Next recommended command: both full variants, batch sizes2,4,8,16,32,
20warmup+3x100measured steps, outputprofile_larger_batches.json. Largest case
uses5242880targets including warmup and fits current single-pass preparation
corpus; memory limit and numerical stability are measured per case, not assumed.
No larger-batch run started by assistant; main training remains off.

## 2026-10-01 22:26 Asia/Manila — Larger-batch GPU sweep measured

Downloaded user-run profile_larger_batches.json, raw ignored, SHA256
0ff84572d2eeb75e5cddd9004ce3695fd46d6dcbd4c1718aefde03d231e7a590.
Both full variants passed all2/4/8/16/32 cases on RTX4090/float32/context512,
20warmup+3x100steps. Verified identical data identity/offsets per batch.
No OOM or numerical failure. Speedsbaseline21149.68/41660.86/79024.08/
100788.89/101396.68; Helix8875.24/17190.23/34392.07/46843.99/46702.62targets/sec.
Baseline2/4 timing flags false solely windowsbelow5seconds; allothercasesstable.
One final baseline2/4 repeat with3x200steps recommended; no runstartedbyassistant.

Batch16 peak allocated4912581120bytes baseline/7676415488Helix; reserved
5163188224/8095006720. Batch32 allocated9250507264/14770064896; speeds only
+0.603%/-0.302% relative16. Recommend commonbatch16 for later reviewed experiment,
not final trainingauthorization. No needtoprobehigherbatchesforcurrentchoice.
Created gpu_profile_larger_batches_summary.json and updatedREADME,training/RunPod
docs andrecords. No code,model,tokenizer,dependencychanges; originalarchive
unchanged. Raw data/reportsstayignored. Mainrun,expandeduniquecorpus,finaltoken
budget,rules/deadlinechecks,GPUresumeunvalidated. Recommend finishshorttiming
repeat,downloadallreports/environment,thenstopGPUbilling. NoactualPodstopdone.

## 2026-10-01 22:34 Asia/Manila — Profiling checks complete; remote evidence backed up

User ran finalbaseline2/4 repeats. Downloaded entire remote runpod_results directory
to ignored artifacts/downloaded_runpod_results. Retrieved remoteSHA256/byte-size
inventory via read-onlySSH and compared every local file: all7 match (environment,
parameter counts, package verification and4profilingJSON reports). Original
source/data/tokenizer and initial transfer archive remain local. No trained
checkpoint exists from profiling; all models were temporary and discarded.

Actual finalrepeatrawSHA256 b4a2f26978ba9bb6156c9edf9f473894644edaee59f2ef1f45c2deab4a0db715.
Baseline2:21543.174237targets/sec, stableTrue, CV0.00673655;
baseline4:41670.577487targets/sec, stableTrue, CV0.01203514.
Windows9.43–10.00seconds; numericalchecks pass. Peakallocated2:1130379776bytes,
4:1670441472bytes; reserved1153433600/1744830464. Both used20warmup+3x200measured
steps. Combined originalplusrepeats: all12model/batch cases numeric/timingpass.
Commonbatch16 remains recommended (100788.89baseline/46843.99Helix targets/sec;
allocated4.58/7.15GiB). Recommendation does not alter trainingconfiguration.

Created docs/results/gpu_profiling_complete.json with complete chosen stablecases
and7filechecksuminventory. UpdatedREADME.md,docs/training.md,docs/runpod.md and
thisrecord. No source/model/tokenizer/dependencychanges. Verified backups, not
unobserved actualcharges or mainquality. Raw files remain Gitignored; no deletedfiles.

GPUprofiling portion nowcomplete. Told user they can StopPod to endGPUbilling;
stoppedstorage costs aboutUSD0.014/hour. Stop/termination not executed byassistant.
Termination is only after user confirms nootherremote filesneed saving. Finalmain
trainingtokenbudget,unique-corpusexpansion,competitionrules/deadlinechecks andGPU
resume remain unresolved. No newfeature or maintraining started. Concepts explained:
profiling validates operations/speed/memory, not goodEnglish; checksums verify
that downloadedfiles are identical; batch16 processes8192targets per update.

## 2026-10-01 — Authorized GitHub publication preparation

The project owner authorized pushing the existing implementation to GitHub.
Verified the workspace is C:/Development/personal/helixdepth, branch main, with
origin https://github.com/cadornajansen/helixdepth. Fetched origin; both local
and remote started at 5adcf23081d49d228b7499b23e3417620fe8d84b. Preserved all
existing implementation and measured evidence.

Current checkpoint: CP3 training readiness, with deterministic CPU recovery
verified and all twelve full-model GPU profiling cases passing after repeats.
GPU checkpoint recovery remains unvalidated. CP4 is baseline training; unique
corpus expansion, the final matched token budget and official rules/deadline
verification remain pending. Main training has not started.

Pre-publication test: `.venv/Scripts/python.exe -m pytest -q -p no:cacheprovider`
passed all 28 tests in 16.34 seconds. A source-tree credential-pattern scan found
no matches. Corrected an outdated README GPU heading, damaged punctuation and
a duplicated profiling sentence. Generated artifacts, virtual environments and
model checkpoints remain excluded by .gitignore. The publication commit contains
source, configs, pinned requirements, tests, documentation and compact measured
evidence; push completion and commit identity are recorded separately in Notion.

## 2026-10-01 23:02 Asia/Manila — Local corpus expansion implementation

The owner requested completing the next preparation checkpoint in one turn.
Scope: expand unique training text locally with the frozen CP2 tokenizer and
validation collection, verify packed data and record evidence. This is CP4
preparation, not authorization to restart a GPU or start main training. The
provided RunPod screenshot shows a stopped Pod (Start Pod action), USD 13.72
balance and displayed storage cost USD 0.01/hour; no live billing audit was done.

Verified the correct HelixDepth workspace and clean main at 061468f before edits.
Read existing data selection, packing, training guide, architecture, handoff
and applicable user instructions. Context7 official Datasets documentation
confirmed pinned revision and streaming iteration. No dependencies changed.

Added configs/expansion.yaml, src/helixdepth/expansion.py,
scripts/expand_corpus.py and tests/test_expansion.py. Added
docs/corpus-expansion.md and linked it from data/training guides. Preserve
original train order, then append first eligible occurrences in pinned source
order. Seed 2026 split assignments and exact canonical text/source-ID exclusions
keep frozen and new held-out groups out of training. All scanned decisions are
audited. Reused IDs with conflicting text fail closed. No BPE fitting occurs.

Bounds: at most 500,000 source rows; 100,000,000 total training tokens including
EOS; 200,000 training documents; 512 MiB canonical training text. Stop before
first whole document exceeding a cap. This is capacity, not a selected final
training budget. Output artifacts/corpus_100m_v1 remains Gitignored. Preserve
existing outputs; interrupted builds require a new output directory.

Command: .venv/Scripts/python.exe -m pytest -q -p no:cacheprovider
--junitxml=docs/results/expansion_correctness.xml
Result: 38 passed in 19.82 seconds (28 existing + 10 new cases). Tests cover frozen
artifacts, deterministic fixture rebuilds, canonical duplicates, validation
exclusion, source-ID conflicts, all capacity bounds, context 512 loader targets,
packed corruption and source-audit order corruption.

Started .venv/Scripts/python.exe -u scripts/expand_corpus.py locally. Build
and full saved-data verification are in progress at this record; final measured
counts and outcome follow separately. No GPU operation, main training, commit
or push performed for this preparation change.

## 2026-10-01 23:21 Asia/Manila — Corpus preparation complete

The local command `.venv/Scripts/python.exe -u scripts/expand_corpus.py`
completed successfully. Actual collection: 113,907 canonical-unique training
documents, 99,999,146 tokens including EOS, 433,153,254 text bytes. Added
105,449 documents and 92,675,862 tokens; retained all original training documents
in their original order. Validation remains 443 documents / 385,550 tokens.

Scanned 124,371 source rows from pinned revision
87f09149ef4734204d70ed1d046ddc9ca3f2b8f9. Audit: 105,450 eligible candidates
(one final candidate exceeded capacity and was not retained), 8,920 existing
or duplicate occurrences, 5,882 reserved validation groups, 3,234 length
exclusions and 885 domain exclusions. No tokenizer fit or document truncation.

Full verification re-encoded all documents, checked exact round trips, every
packed token/index, disjoint content/source IDs, source audit decisions/order,
original training prefix and unchanged frozen artifacts. All checks passed.
Tokenizer SHA256 a98dbd3bb95de3ea9823b963e09c9069b25f2cf56bb1c5bb4f60a23cb7542bb2.
Train binary SHA256 6b51a55e42ef2da46a7f5f0f87bfa323026bc899d94c24633c199b544c5c34b4.
Manifest SHA256 84a8c807eae3d387c4d09247d83b1870e0ffffb9eb64ec6e55432af6011c2757.
Validation binary SHA256 f9b852858604564ef70895796677886d9a24d174dd34082319d0cddb5706a9b8,
identical to original CP3 validation. Current preparation module/data helper
hashes match the manifest. Fixture builds reproduce byte-for-byte; a second
complete source download/rebuild was not run and is not claimed.

Independent actual loader check: context 512 / batch 16, offsets 0 / 7315456 / 99983360,
two loaders match and inputs/targets match direct int32 binary reads with
exact one-token shift. Original-corpus recovery state correctly rejected.
12,206 complete batches score 99,991,552 targets; 7,593 tail targets unused.
Evidence: docs/results/corpus_expansion.json, expanded_loader.json and
expansion_correctness.xml (38 tests passed in 19.82 seconds). Full verification
command: `.venv/Scripts/python.exe scripts/expand_corpus.py --verify-only`.

Created: configs/expansion.yaml; src/helixdepth/expansion.py;
scripts/expand_corpus.py; tests/test_expansion.py; docs/corpus-expansion.md;
docs/results/corpus_expansion.json; docs/results/expanded_loader.json;
docs/results/expansion_correctness.xml. Modified: README.md, docs/data.md,
docs/training.md, docs/records.md. No files deleted; model/trainer/dependency
files unchanged. New ignored artifacts/corpus_100m_v1 contains tokenizer.json,
tokenizer_training.json, source_card.md, train.jsonl, validation.jsonl,
train.bin, validation.bin, train.index.jsonl, validation.index.jsonl,
selection_audit.jsonl and manifest.json. Payload excluding manifest 968,326,372
bytes. Existing CP2, packed CP3 and RunPod archive preserved. Expanded corpus
requires separate transfer packaging; old archive still contains CP2 data.

This completes local corpus preparation for CP4, not CP4 baseline training.
The 100M capacity is not a final experiment token budget. No GPU restart,
paid compute, main training, commit or push performed. Exact/canonical
deduplication only; near-duplicate and exhaustive benchmark contamination
checks remain unimplemented. Final budget, GPU recovery verification and
official competition rules/deadline review remain pending.

Notion synchronized and fetched again to verify the new checkpoint status and
completion record: https://app.notion.com/p/3ecd08af02fb81f29b20de9cd85bfc83 .
Repository whitespace check passed; generated expanded data remains Gitignored.

## 2026-10-01 23:35 Asia/Manila — Official competition review

Read the official overview, rules and organizer extension announcement.
Created docs/competition-rules.md with source links, a concise requirement
mapping and project-specific readiness gaps. Submission extension is confirmed;
the older build-period clause remains inconsistent and was not silently resolved.
No organizer message was sent. No participant-eligibility certification made.

Both measured architectures are within the parameter cap. Our own validation
is not a substitute for the named benchmark evaluation suite. Recommended
next protocol: one matched pass,99,991,552 scored targets each,context512,batch16;
recommendation only, not a configured or started training run. Existing local
corpus changes preserved. No code/data changes, GPU actions, commit or push.
Documentation-only verification: checked cited official pages and Git whitespace.
Notion synchronization follows this record.

Notion update completed and fetched again: current status and official-review record verified. No code tests rerun for this documentation-only change.

## 2026-10-02 00:00 Manila — User-operated main training package ready

Accepted organizer Discord clarification: October 2,23:59 Manila deadline.
Latest user instruction: user runs every remote command; assistant prepares only.
No Pod start, paid GPU execution, main training, benchmark run, commit or push.
Observed stopped Pod balance USD13.71; existing files preserved.

New protocol: configs/training_main.yaml; 12206 updates, batch16, context512,
99991552 targets per model,244 warmup updates, seed2026,float32,deterministic
math attention, AdamW3e-4, decay0.1, norm cap1. Prior throughput used default
attention and is not a measurement of these deterministic main settings.

Files created in this preparation:
- configs/training_main.yaml: matched fixed main schedule.
- scripts/run_experiment.py: training reports, full final validation, atomic
  periodic recovery checkpoints, model export, full-model fresh-process gate.
- scripts/setup_main.sh: verify and reuse existing Pod Python/CUDA environment.
- scripts/train_both.sh: background-compatible sequential recovery/baseline/Helix
  launcher; exits on failure, records exit code; preserves existing runs.
- src/helixdepth/evaluation.py: causal continuation scoring with correct shift,
  right padding and windowing, frozen-tokenizer boundary handling.
- scripts/evaluate_model.py: prepared optional zero-shot harness adapter and
  held-out WikiText103 slice perplexity; not yet run against trained models.
- requirements/evaluation.in and evaluation.lock: optional pinned harness0.4.13
  with Linux/CUDA dependencies; not needed to start training.
- tests/test_main_experiment.py: score alignment/batching and main runner export,
  full held-out validation, token position and output-preservation checks.
- docs/main-experiment.md: protocol, recovery and limitations.
- docs/start-training.md: exact Windows upload, Linux launch, backup commands.
- docs/results/main_preparation.xml:40 tests passed in18.60seconds.
- docs/results/main_package_v2.json: successful archive verification evidence.
Modified scripts/package_runpod.py and scripts/verify_transfer.py to package the
expanded corpus and require original fitting IDs be training-only, with frozen
CP2 provenance and verified expanded-manifest hash. Faster gzip level1 packaging.
Modified docs/competition-rules.md to accept organizer clarification and record
training authorization. This record added to docs/records.md.

An intermediate artifacts/runpod_main_v1 archive failed its round-trip check
because source changed while packaging. Do not use it. Final runpod_main_v2
passed all86 packaged file checks,16384 vocabulary,zero split overlap,8458
original tokenizer-fitting IDs contained only in expanded training documents.
Final archive188347545bytes SHA256:
18eadeb8c9af26fbddc61e34b29ba5ce7d35c28023478579fe52fab8f63791fa
Archive and generated corpus stay Gitignored. No GPU recovery/training/evaluation
results claimed. Three-hour cap per main process does not stop Pod billing.

Exact local test: .venv/Scripts/python.exe -m pytest -q
Execution flow: verify transfer -> environment/tests -> fresh-process GPU recovery
for both models -> baseline main -> HelixDepth main -> user downloads/checksums.
Notion current checkpoint and preparation record synchronized. Main training is
ready for user launch; CP4/CP5 are not complete until measured runs finish.

## 2026-10-02 — Replacement Pod setup correction

User deleted the unavailable original Pod and created a new one. Screenshot
shows new direct TCP endpoint213.173.99.46:20000 and balanceUSD13.70.
Updated docs/start-training.md to use that endpoint and setup_runpod.sh for a
fresh pinned environment, replacing setup_main.sh reuse instructions. Both
scripts already exist in verified runpod_main_v2; no archive rebuild or code
change required. Reviewed actual fresh setup script and RunPod Context7 SSH/SCP
documentation. No remote commands executed by assistant; user remains operator.
Notion current checkpoint updated. Main training not yet reported as started.

## 2026-10-02 — User launch evidence and authorized repository publication

User terminal screenshot confirms replacement Pod setup:40 tests passed in45.99s.
Background train_both.sh started; observed baseline step4/4,loss9.7973 and14194
training targets/sec during the recovery check. This is real gradient training
for the recovery gate, not proof that the12206-step baseline main run began.
CP4 is in progress at its recovery gate; CP5 is queued by the launcher. Exact
GPU resume comparison completion, main completion and benchmark scores remain
unconfirmed. No remote access/commands executed by assistant; user is operator.

Created docs/results/training_launch.json from screenshot evidence, explicitly
labeling it as reported evidence rather than downloaded logs. Updated README.md,
docs/training.md and this record. Notion is synchronized with this same status.
User authorized repushing the repository. Publish source/configuration/docs/tests
and small measured reports; artifacts, datasets,tokenizer binaries,transfer
archives,.venv and checkpoints stay ignored. Do not modify/redeploy the currently
running Pod or its checksum-verified package as part of this publication.

No competition outcome can be guaranteed. Required next evidence is completed
matched runs, checksum-verified backups, required benchmark results, and an
accurate explanation/demo. Recovery loss alone does not establish model quality.
Full changed-file inventory is available in the publication commit; preparation
file purposes are recorded in the preceding main-package entry.
