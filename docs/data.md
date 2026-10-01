# CP2 data and tokenizer

For the later frozen-tokenizer expansion, see [corpus expansion](corpus-expansion.md).
CP2's original artifacts and the historical results below remain unchanged.

## Reproduce locally (PowerShell, repository root)

```powershell
uv pip install --python .venv/Scripts/python.exe -e . --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe scripts/prepare_cp2.py
.venv/Scripts/python.exe scripts/prepare_cp2.py --verify-only
```

Preparation refuses to overwrite an existing artifact directory. To refit a
tokenizer after an interrupted fit, use `--fit-existing`; this reuses verified
saved document splits and replaces tokenizer/token files only. To reproduce the
whole selection separately, copy the config, change only `output_dir`, then run
with `--config path/to/copy.yaml --report path/to/second-report.json`.

## Source and bounded selection

Source: HuggingFaceFW/fineweb-edu, `sample-10BT`, upstream `train`, revision
`87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`. The upstream split is a source pool;
our held-out validation is constructed independently. Text comes from the `text`
column. Source configuration and selection settings are in `configs/data.yaml`.

Read the first 20,000 rows in the pinned source order with streaming enabled.
Canonicalize Unicode to NFC, line endings to LF, and strip outer whitespace.
Keep documents with 200–20,000 UTF-8 bytes. Exclude Wikipedia, Hugging Face and
GitHub source domains, including subdomains. Group identical canonical text by
SHA-256; retain all distinct source IDs and original-text hashes for each group.
This deduplicates exact text and the stated normalization variants, not near
duplicates. Rank groups by SHA256(`2026:selection:text_sha256`), with content hash
as tie-breaker. Greedily retain at most 10,000 groups and 33,554,432 text bytes;
skip a group if it does not fit the remaining byte budget. Source-prefix bias
remains; this is not uniform sampling across the entire 10BT pool.

Before tokenizer fitting, compute
`int(SHA256("2026:split:" + text_sha256), 16) % 1000`. Buckets below 50 go to
validation; the others go to training. This targets 5% validation by document,
without forcing an exact proportion. Each duplicate group has one assignment.

## Provenance and licensing

The pinned dataset card is saved as `artifacts/cp2/source_card.md`, with its URL
and SHA-256 recorded in the manifest. It describes Common Crawl English text
filtered for educational value; filtering used a classifier trained using
Llama3 annotations. These are existing web documents; no model weights are
downloaded for this experiment. Dataset-level license: ODC-By 1.0; the dataset
card also specifies Common Crawl's Terms of Use. This does not establish a
uniform copyright license for every underlying web page.

- [Pinned dataset card](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/blob/87f09149ef4734204d70ed1d046ddc9ca3f2b8f9/README.md)
- [ODC-By 1.0](https://opendatacommons.org/licenses/by/1-0/)
- [Common Crawl terms](https://commoncrawl.org/terms-of-use)

No evaluation dataset is loaded or intentionally added to the corpus. Domain
exclusion helps remove identifiable Wikipedia and benchmark-hosting sources;
it cannot establish absence of benchmark text republished elsewhere. Exhaustive
benchmark overlap and near-duplicate detection are not implemented. No verified
benchmark-decontamination claim is made.

## Tokenizer and execution flow

1. Save `train.jsonl` and `validation.jsonl`, including canonical text, content
   hashes, source IDs/URLs/dumps/raw-text hashes, source row indices and splits.
2. Save `manifest.json`: exact settings, counts, file hashes, source-card hash,
   dependency versions and preparation-code hashes. Verify document separation.
3. Fit an own BPE tokenizer from **only** `train.jsonl`. Byte-level
   pre-tokenization starts with all 256 byte symbols; BPE learns frequent symbol
   combinations. It uses no pretrained tokenizer, model or vocabulary.
4. Request 16,384 entries **including** `<pad>`, `<bos>`, `<eos>`, `<unk>`
   (IDs 0–3). Minimum pair frequency is 2; tokenizer workers is 1. No tokenizer
   normalizer or automatic post-processing is used; byte decoding preserves the
   saved canonical text, including Unicode. Fail if actual vocabulary is smaller.
5. Save `tokenizer.json` and `tokenizer_training.json` with the training-file
   hash, the IDs actually consumed by the fitting iterator, and training settings.
   Encode both splits using this one tokenizer.
   Each `*.tokens.jsonl` row retains a document ID and appends exactly one EOS.
   BOS/PAD are reserved but not automatically inserted. Documents remain separate;
   training packing and document attention-mask policy belong to later work.
6. Reload the tokenizer, verify every content hash and seeded split assignment,
   disjoint source IDs/content, exact vocabulary/contiguous IDs/special tokens,
   every saved token sequence, no UNK, and every text round trip. Check additional
   Unicode probes. Save document/text-byte/token counts and artifact checksums to
   `docs/results/cp2.json`. Token counts distinguish text-only and EOS-inclusive.

Generated artifacts are ignored by Git; preserve them locally. Machine-readable
evidence and reproducible code/config remain in the repository. Subsequent
models must read the same token files. Full language-model training, corpus
size suitability for a measured GPU budget, and quality are outside CP2.

Context7 supplied official Datasets revision/streaming documentation and
Tokenizers byte-level BPE trainer/save/load documentation for this implementation.

## Measured completion evidence

Scanned 20,000 source rows: 570 excluded by length, 140 by domain, one duplicate
grouped, leaving 19,289 eligible unique texts. The byte cap selected 8,901 unique
documents totaling 33,554,189 UTF-8 bytes. The observed duplicate group was not
selected. Duplicate-handling behavior is exercised explicitly by unit tests.

Training: 8,458 documents, 31,860,667 text bytes, 7,314,826 text tokens,
7,323,284 tokens including EOS. Validation: 443 documents, 1,693,522 text bytes,
385,107 text tokens, 385,550 including EOS. Actual vocabulary is 16,384;
special IDs are PAD=0, BOS=1, EOS=2, UNK=3. All saved documents round-trip and
produce no UNK IDs. The fitting iterator consumed exactly the training IDs.

The final test suite passed 13 tests in 2.26 seconds, with no warnings. A full
independent source read and fit in `artifacts/cp2_repeat/` matched seven files
byte-for-byte: both document splits, both token splits, tokenizer JSON,
tokenizer fitting provenance and pinned source card. Manifests differ only in
output-directory settings and preparation-code provenance as applicable; they
are not claimed byte-identical. Evidence: `docs/results/cp2.json`,
`cp2_reproducibility.json`, `cp2_correctness.xml`, and `cp2_environment.txt`.

CP2's bounded-data, split-separation, training-only tokenizer, exact-vocabulary,
token-count and reproducibility checks pass. This is a small prepared corpus,
not a claim that it meets the eventual training-token budget. No model was
trained, no GPU was rented, and no quality result was produced.
