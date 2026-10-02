---
language:
- en
tags:
- pytorch
- text-generation
- trained-from-scratch
- research
datasets:
- HuggingFaceFW/fineweb-edu
---

# HelixDepth 300M-target experiment

Two custom PyTorch base language models, each trained from random initialization
on **299,982,848 next-token targets**. The 300M label describes training targets,
not parameter count. This release includes HelixDepth and its conventional baseline.

Source, dependency versions and executable scripts:
[cadornajansen/helixdepth](https://github.com/cadornajansen/helixdepth).
The release manifest records exact file sizes and SHA-256 checksums.

## Models and architecture

| Property | Baseline | HelixDepth |
|---|---:|---:|
| Unique trainable parameters, tied output counted once | 40,968,320 | 41,184,432 |
| Stored Transformer blocks | 6 | 6 |
| Executed stages | 6 | 12 |
| Width / attention heads | 640 / 10 | 640 / 10 |
| Vocabulary / maximum context | 16,384 / 512 | 16,384 / 512 |

Both use RoPE, pre-RMSNorm, SwiGLU and tied input/output embeddings. HelixDepth
reuses six blocks twice, with separate stage norms and a small depth-conditioned
network controlling shared rank-16 weight modifications. The experiment changes
both execution depth and conditioning; no ablation isolates their contributions.

## Training and data

English FineWeb-Edu sample-10BT, revision
`87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`, supplied the training text.
The source dataset identifies the ODC-By-1.0 license; source provenance and
filtering are documented in the code repository. This release contains no corpus.
Our byte-level BPE tokenizer was fitted on training text and then frozen.

Each model uses seed 2026, float32, context 512, batch 16, and one RTX 4090.
An initial approximately 100M-target run was followed by two fresh-data stages;
each continuation retains its own learned weights and restarts AdamW and the
learning-rate schedule. No third-party pretrained weights, instruction tuning or
teacher-model distillation was used. Training and validation are separate;
exact-document/source exclusions do not establish exhaustive benchmark or
near-duplicate decontamination.

## Evaluation

All ten jobs completed using lm-evaluation-harness 0.4.13, zero-shot, batch 16,
and the same tokenizer and settings. Accuracy below is raw accuracy throughout.

| Measurement | Baseline | HelixDepth |
|---|---:|---:|
| HellaSwag, 10,042 examples | 26.49% | 26.49% |
| ARC-Easy, 2,376 examples | 37.79% | 36.99% |
| PIQA, 1,838 examples | 56.64% | 57.07% |
| WinoGrande, 1,267 examples | 47.51% | 49.25% |
| WikiText-103 token perplexity | 86.51 | 85.42 |
| Frozen validation loss | 3.7476 | 3.6935 |
| Recorded training wall time across three stages | 65.33 min | 137.56 min |
| Maximum peak allocated training GPU memory | 5.51 GiB | 9.02 GiB |

WikiText uses the first 262,144 own-tokenizer tokens of the wikitext-103-raw-v1
test split, revision `b08601e04326c79dfdd32d625aee71d232d685c3`.
It is a held-out slice, not the entire test split. Frozen validation is a
different dataset; its loss is not the WikiText result.

HelixDepth has 1.26% lower WikiText perplexity at approximately the same parameter
count and equal training-target budget, with 2.11 times the recorded training
wall time and more allocated memory. Reasoning results are mixed; normalized
PIQA slightly favors baseline. One seed per model and no paired statistical
analysis mean these differences do not establish general reasoning superiority.
Perplexity is not directly comparable across different tokenizers.

Combined training timers total 3.38 GPU-hours. These include validation and
checkpoint work inside runs, but exclude setup, profiling, preparation, external
evaluation and idle time. They are not total rental usage or billed cost.

## Files and use

- `helixdepth/model.pt` and `baseline/model.pt`: model-only PyTorch exports with
  embedded architecture configuration and state dictionary, float32.
- `tokenizer.json`: shared frozen tokenizer.
- `configs/`: human-readable model configurations.
- `evaluation_summary_300m.json`: full scores, settings, identities and limitations.
- `generation_demo_300m.json`: all six unedited sampled examples.
- `release_manifest.json`: file sizes, SHA-256 hashes and source revision.

These are custom models, **not Transformers AutoModel checkpoints**. Use the
linked source repository and its `LanguageModel.load`, which uses
`torch.load(..., weights_only=True)`. No hosted inference API is needed.

After installing the repository dependencies, download the model release:

```powershell
.venv/Scripts/hf.exe download buildwithjansen/helixdepth-300m --local-dir artifacts/hf_300m
.venv/Scripts/python.exe scripts/generate_demo.py --model artifacts/hf_300m/helixdepth/model.pt --tokenizer artifacts/hf_300m/tokenizer.json --output artifacts/hf_demo.json --device cpu --prompt "Plants need sunlight because" --max-new-tokens 32 --temperature 0
```

Run from the source repository root. The output path must be new. Use the
baseline model path for the comparator. For archival reproduction, pin the Hub
revision reported in the source repository's release receipt.

## Intended use and limitations

Research, education and reproducible architecture comparison. These are base
text-completion models, not reliable assistants. Recorded generations repeat,
lose coherence and make factual errors. They have no demonstrated safety tuning,
instruction-following reliability, robust arithmetic, or broad reasoning ability.
Do not use their outputs for consequential decisions. Pretraining data may contain
biases and inappropriate material. Illustrative generation timings are not an
optimized serving benchmark; generation currently has no KV cache.

## Credits and release terms

Project by buildwithjansen / cadornajansen. OpenAI Codex assisted with architecture
implementation, training/evaluation tooling, debugging, documentation and checks.
Credit PyTorch, Hugging Face Datasets/Tokenizers/Hub, FineWeb-Edu, EleutherAI's
lm-evaluation-harness, the benchmark authors and RunPod. See the source README
and data documentation for details. No model license has been selected in this
release; public availability alone is not a grant of an open-source license.
