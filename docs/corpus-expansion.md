# Frozen-tokenizer corpus expansion

This is preparation for CP4, not baseline training. `configs/expansion.yaml`
sets a **100,000,000-token capacity**, including document EOS tokens. It does
not set the eventual matched training budget. Both models must later use the
same corpus, order and number of prediction targets.

## Reproduce or verify

Run these commands from the repository root in PowerShell:

```powershell
# Build once; needs the verified CP2 artifacts and network access to the source.
.venv/Scripts/python.exe scripts/expand_corpus.py

# Verify saved output locally without another source download.
.venv/Scripts/python.exe scripts/expand_corpus.py --verify-only

# Independent rebuild, without overwriting the original expanded collection.
.venv/Scripts/python.exe scripts/expand_corpus.py --output artifacts/corpus_100m_repeat --report artifacts/corpus_100m_repeat_evidence.json
```

An existing output directory is never overwritten. A failed/interrupted build
keeps its incomplete directory for inspection; it is not a verified corpus.
Use a new `--output` directory to rebuild. There is no mid-download resume.
A final `manifest.json` is written only after all output files are complete;
the separate evidence report is written only after verification passes.

The artifact directory is ignored by Git. It contains the frozen tokenizer and
its original fitting record, source card, training and validation text JSONL,
packed int32 token streams, document indexes, selection audit and manifest.
The compact `docs/results/corpus_expansion.json` belongs in Git. Preserve the
artifact directory separately: cloning GitHub does not restore these data files.
The earlier RunPod transfer archive still contains the original preparation
corpus; it is not silently replaced by this build.

## Exact selection rule

1. Verify every original CP2 artifact against its saved evidence. Inherit the
   exact dataset ID, configuration, revision, normalization, seed 2026, hash
   split rule, size filters and excluded domains from that evidence.
2. Preserve every original CP2 training document in its existing order. Keep
   the original 443-document validation collection byte-for-byte unchanged.
3. Stream at most 500,000 rows from the beginning of the pinned source. Source
   order is used without shuffling. This also finds eligible documents omitted
   from the original 20,000-row candidate pool by its smaller byte cap.
4. Normalize text to NFC, normalize line endings, trim outer whitespace and
   compute SHA-256. Exclude hashes and source IDs already encountered in either
   original split or earlier source rows. A reused ID with different content
   fails the build rather than risking evaluation leakage.
5. Apply the original seeded hash split rule. All validation-assigned groups
   remain excluded from training, including newly encountered ones. Record
   these groups in the audit; do not enlarge the primary validation set.
6. Apply the original 200–20,000-byte and domain filters. Retain the first
   eligible occurrence of each new training text in source order. All scanned
   rows, including rejected rows and duplicate aliases, retain source IDs,
   indices, raw-text hashes, canonical hashes and decisions in the audit.
7. Encode with the existing tokenizer, append one EOS, and stop before the
   first whole document that would exceed any cap: 100M total training tokens,
   200,000 training documents, or 512 MiB of canonical training text. These caps
   include the original training documents. A capacity-stopping audit contains
   one final eligible document that was inspected but not retained. If the
   source/candidate limit comes first, report the actual smaller collection.

The fixed seed controls split assignment; it is not a promise of random source
sampling. The expanded training order deliberately preserves the original
prefix and then follows source order. It is not a uniform sample of the full
dataset. Neither near-duplicate removal nor complete benchmark contamination
removal is established by these checks.

## How the data flows

`scripts/expand_corpus.py` reads the config and opens the pinned source using
Datasets streaming. `new_documents` filters one document at a time.
`write_split` encodes retained text with the frozen tokenizer and writes:

- `train.jsonl`: document text and provenance.
- `train.bin`: consecutive little-endian int32 token IDs with EOS boundaries.
- `train.index.jsonl`: each document's ID, starting offset and token count.

Validation is encoded separately and its original text file is copied unchanged.
The existing `PackedStream` can read the resulting directory without changes.
At context 512, inputs are 512 consecutive IDs and targets are the next 512 IDs,
shifted by one. At batch 16, each complete update scores 8,192 prediction targets.
The final incomplete batch is not used; usable targets are calculated from
`floor((saved_tokens - 1) / 8192) * 8192`.

Verification checks file hashes, frozen tokenizer/fitting/validation identities,
the original training prefix, disjoint source IDs and canonical text hashes,
seeded assignments, source audit decisions and retained order. It re-encodes
every saved document, checks exact text round trips and compares every packed
ID and document offset. This independent re-encoding rebuilds the binary-stream
hash; it is more than trusting the initially written manifest.

## What this establishes

Passing checks establish a larger reproducible preparation collection with
unchanged token meanings and primary validation. They do not demonstrate model
quality or make 100M tokens the chosen experiment budget. CPU fixture tests
cover duplicate/validation exclusion, conflicting IDs, reproducible builds,
capacity boundaries, preserved files, corrupted packed data and loader behavior.

Main training, GPU recovery verification, final training-budget selection,
competition-rule/deadline verification and updated GPU transfer packaging remain
separate work. No GPU restart is needed for this local preparation checkpoint.

API reference: Context7 supplied the official Datasets
[pinned-revision loading](https://github.com/huggingface/datasets/blob/main/docs/source/loading.mdx)
and [streaming iteration](https://github.com/huggingface/datasets/blob/main/docs/source/about_mapstyle_vs_iterable.mdx)
documentation. The existing dependency pins are unchanged.

## Actual loader check on the expanded files

The saved loader evidence in `results/expanded_loader.json` checks two independent
loaders at context 512 / batch 16 against direct binary reads at three positions:
the beginning, the original/new-text boundary, and the final complete batch.
Inputs and one-token-shifted targets must match exactly. It also checks that
a recovery state for the original smaller corpus is rejected and that the
validation binary is unchanged. These are data checks, not model training.

A compact reproduction after building the expanded corpus:

```python
from pathlib import Path
from helixdepth.packing import PackedStream

data = Path("artifacts/corpus_100m_v1")
a = PackedStream(data, "train", context=512, batch_size=16)
b = PackedStream(data, "train", context=512, batch_size=16)
for offset in (0, 7315456, 99983360):
    a.offset = b.offset = offset
    x, y = a.peek()
    other_x, other_y = b.peek()
    assert x.equal(other_x) and y.equal(other_y)
    assert x.flatten()[1:].equal(y.flatten()[:-1])
```

Those offsets refer to this version's measured collection; a different capacity
requires recalculating the boundary and final full batch from the manifests.

## Measured completion — 2026-10-01 23:21 Asia/Manila

All full-corpus verification checks passed.

| Collection | Documents | Tokens including EOS | Canonical text bytes |
|---|---:|---:|---:|
| Original training | 8,458 | 7,323,284 | 31,860,667 |
| Expanded training | 113,907 | 99,999,146 | 433,153,254 |
| Frozen validation | 443 | 385,550 | 1,693,522 |

Added **105,449 training documents and 92,675,862 tokens**. The collection has
13.65 times the original token count, from canonical-unique documents. This
does not mean every word/token is unique or that near duplicates were removed.

Scanned 124,371 pinned-source rows: 105,450 eligible training candidates,
8,920 existing/duplicate occurrences, 5,882 reserved held-out groups,
3,234 length exclusions and 885 domain exclusions. The final eligible candidate
would exceed capacity and was not saved. No document was truncated.

At context 512 / batch 16, the stream provides 12,206 complete batches and
99,991,552 prediction targets, leaving 7,593 unused tail targets. The first
stream token has no preceding input within this stream. Both independent
loaders match at the beginning, original/new-document boundary and final batch;
explicit shifted targets match direct binary reads. Old-corpus recovery state
is rejected. The validation binary matches the original CP3 binary exactly.

Tokenizer SHA-256: `a98dbd3bb95de3ea9823b963e09c9069b25f2cf56bb1c5bb4f60a23cb7542bb2`.
Training binary SHA-256: `6b51a55e42ef2da46a7f5f0f87bfa323026bc899d94c24633c199b544c5c34b4`.
Expanded manifest SHA-256: `84a8c807eae3d387c4d09247d83b1870e0ffffb9eb64ec6e55432af6011c2757`.
All 38 tests passed in 19.82 seconds. The full actual collection was independently
re-encoded and compared token-for-token; all checks passed. Fixture builds were
repeated byte-for-byte. A second complete remote-source rebuild was not run;
the reproducible command above is available for that stronger check.

Evidence: [corpus report](results/corpus_expansion.json),
[loader check](results/expanded_loader.json), and
[test results](results/expansion_correctness.xml). Generated payload is
968,326,372 bytes excluding the manifest; it remains outside Git.
