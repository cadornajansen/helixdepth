# Generate text from the saved 300M models

These are base language models trained on text completion. They were not trained
as chat assistants. The demo illustrates their actual behavior; it is not a
reasoning benchmark or a claim that generated statements are correct.

## Local replay (PowerShell, from the repository root)

The checkpoint and tokenizer are already in the verified local backup:

```powershell
.venv/Scripts/python.exe scripts/generate_demo.py --model artifacts/backups/continuation_300m/helixdepth_additional_100m_v2/model.pt --tokenizer artifacts/backups/continuation_300m/tokenizer.json --output artifacts/demo_local_helixdepth.json --device cpu
```

Use `baseline_additional_100m_v2/model.pt` and a different output filename to
replay baseline. Existing output files are preserved; choose a new filename
for another run. No network access, datasets, hosted model or GPU is required
for local CPU generation. It uses the repository's existing dependencies.

Defaults were chosen before inspecting either model's generated text:

- Three prompts: "Plants need sunlight because", "When rain falls on a
  mountain,", and "To solve a difficult problem, it helps to".
- At most 96 new tokens per prompt, ending early on document-end token 2.
- Temperature 0.8, top-k 40, seeds 2026, 2027 and 2028 in prompt order.
- Float32, one prompt at a time; non-EOS special tokens are excluded.
- Every example is saved. No repetition penalty or post-editing is applied.

To try your own text, add `--prompt "Your opening text"`; repeat that option
for multiple prompts. Set `--temperature 0` for greedy decoding. The prompt
and requested continuation must fit within 512 tokens, with at most 256 new
tokens allowed. A fixed seed makes the sampling choice repeatable within a
matching environment; identical text across devices or library versions is
not guaranteed.

## What the script does

1. Loads the saved model through the existing `LanguageModel.load` method and
   loads our frozen tokenizer. It checks the vocabulary size and hashes inputs.
2. Converts a prompt into token IDs: the model's numeric representation of text.
3. Reads the model's final-position scores for the next token. With sampling,
   top-k restricts candidates to the 40 highest scores and temperature controls
   how concentrated their probabilities are. Greedy mode takes the highest.
4. Appends the chosen token and repeats until EOS or the output limit.
5. Decodes the generated IDs into text and saves all prompts, outputs, token IDs,
   settings, hashes, device information and timing to a new JSON file.

`model.eval()` sets evaluation behavior; `torch.inference_mode()` disables
gradient tracking. Neither performs an optimizer update. Input files are
rehashed afterward. There is no training or checkpoint-writing path.

Timing starts after an untimed warmup and excludes checkpoint loading and text
encoding/decoding. CUDA is synchronized around the timed loop, so unfinished
GPU work is included. The current architecture has no key/value cache: each
next-token prediction recomputes the accumulated context. These are illustrative
single-run timings, not an optimized serving-throughput benchmark.

## Bounded GPU capture

For the recorded capture, both models use the same script and defaults on the
existing RTX 4090. A separate `timeout` wrapper limits the complete two-model
sequence to five minutes. The model paths are
`artifacts/{variant}_additional_100m_v2/model.pt`, and outputs/logs are written
under a new `artifacts/demo_300m/` directory. Checkpoints are verified against
the 300M backup receipt before launch.

Small CPU checks cover a repeated seeded output, unchanged model weights,
absence of gradients, restoration of the caller's model mode, rejection of an
oversized context and stopping on EOS in greedy mode.

In the video, show the prompt, actual output and decoding settings together.
Keep benchmark scores separate from these illustrations. Preserve all recorded
examples even if the video only has time to show one.

## Recorded result — October 2, 2026

Both RTX 4090 runs completed, producing all six fixed-prompt completions at
96 new tokens each. The unedited outputs and timing are saved in
[generation_demo_300m.json](results/generation_demo_300m.json). Three raw files
(both reports and the shared log) were downloaded and checked against a remote
SHA-256/size manifest. Model hashes match the verified 300M checkpoints;
tokenizer and uploaded script hashes match the local files.

Observed limitations are material: the models repeat phrases, lose coherence
and produce false explanations. For example, the HelixDepth sunlight example
does not correctly explain why plants need sunlight. The problem-solving
completion uses list-like educational language but does not demonstrate solving
a problem. Present the project as an architecture experiment, not a reliable
chatbot. All examples are preserved rather than selecting only a favorable one.

Local CPU replay was also verified on the actual backed-up HelixDepth model
using a four-token greedy continuation. It produced " they are not able" in
about 0.18 seconds of timed generation, excluding loading and warmup. This is
a smoke check of offline operation, not an assessment of text quality or a
general CPU speed claim. The Pod is unnecessary for replaying the saved demo.
