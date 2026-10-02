# Record HelixDepth in about three minutes

Use this as speaking notes, not a script you need to memorize. The project is
a small-model architecture experiment with measured benefits and costs.

## 0:00–0:25 — Why you built it

Show yourself and the project name.

“This is my first hackathon, and I built HelixDepth solo with AI coding assistance.
I started without a machine-learning background. With one rented GPU and a small
budget, I wanted to investigate a concrete question: can reusing the same model
blocks improve language prediction without adding many parameters?”

Use this introduction only if it accurately reflects your experience. Describe
decisions you can explain; do not imply you wrote all code without assistance.

## 0:25–1:05 — Show the architectural idea

Open [the diagram](visuals/helixdepth-explained.html), zoom into section 02.

“A parameter is a learned number. Our baseline stores six Transformer blocks and
uses each once. HelixDepth also stores six blocks, but runs through them twice.
A small conditioning network uses the stage number to control small changes to
the calculations. That gives twelve processing stages with about 41.18 million
parameters, compared with 40.97 million for the baseline. Extra processing still
costs time and memory.”

The proposal combines block reuse, extra depth and depth-conditioned low-rank
changes. We have not isolated which component causes the difference. Do not claim
this is the first use of shared weights or hypernetworks.

## 1:05–1:35 — Explain the experiment

Show section 03, then the README results table.

“Both models started with random weights and learned from the same text, using
our own fixed tokenizer. Each processed almost 300 million next-token targets.
One update contains 16 sequences, each with 512 predictions: 8,192 supervised
targets. We measured prediction error, computed gradients, and used an optimizer
to adjust the weights. Separate validation text measured progress.”

300M is the number of training targets, not the number of model parameters.
We continued our own from-scratch runs; no pretrained model or teacher model was used.

## 1:35–2:10 — Show real text generation

From the repository root in PowerShell, run:

```powershell
$demoOutput = "artifacts/demo_video_$(Get-Date -Format 'yyyyMMdd_HHmmss').json"
.venv/Scripts/python.exe scripts/generate_demo.py --model artifacts/backups/continuation_300m/helixdepth_additional_100m_v2/model.pt --tokenizer artifacts/backups/continuation_300m/tokenizer.json --output $demoOutput --device cpu --prompt "Plants need sunlight because" --max-new-tokens 32 --temperature 0
```

“This loads the actual saved model on my CPU. It predicts one token, appends it,
and repeats. There is no hosted language-model API. This is a base text model:
it can be repetitive and factually wrong. The demo shows its real limitations.”

Keep the prompt, actual output and settings visible. This short greedy capture
uses different settings from the preserved 96-token sampled examples; do not
present it as the same capture. If you show saved output instead, label it recorded.

## 2:10–2:45 — Explain results and the tradeoff

Show the [README comparison](../README.md#current-matched-300m-comparison).

“All ten required evaluation jobs completed. On the same WikiText slice,
HelixDepth reached 85.42 perplexity versus 86.51: 1.26 percent lower.
Commonsense benchmark results were mixed. This is not proof of better reasoning
across the board. HelixDepth also took about 2.1 times the training time and more
GPU memory. The evidence supports a modest prediction-quality gain at nearly the
same parameter count, with a real compute cost.”

Perplexity measures predictive uncertainty; lower is better. It is not a
percentage accuracy. Compare models using the same tokenizer and test protocol.

## 2:45–3:10 — Close with what you learned

“The most important thing I learned was how to test an idea fairly. I preserved
the baseline, used the same data, verified the checkpoints, and reported where
the idea helped and where it did not. More experiments would be needed to separate
the effect of reusing blocks from the small depth-specific changes.”

Show the public source URL and only model-release links that have been verified.
Disclose OpenAI Codex assistance in the description and Built With section.

## Before recording

- Open the updated diagram and final README, and run the short command once.
- Hide account balances, SSH details, tokens and unrelated browser tabs.
- Keep narration in English or add English subtitles; target 2–5 minutes.
- Save three clear screenshots: architecture, final result table, actual inference.
- After recording, check sound, readable text and public video access while signed out.

Remaining submission work: upload the video, complete project/team fields,
verify public links and save the submission confirmation. Recording does not
require the RunPod to remain active.
