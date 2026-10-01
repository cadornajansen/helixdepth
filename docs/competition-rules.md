# GIBC V2 rules review

Checked 2026-10-01 23:35 Asia/Manila.

## Deadline evidence

The [organizer extension announcement](https://gibc-v2.devpost.com/updates/46335-deadline-extended-you-now-have-until-oct-2-to-submit-your-project)
sets submission to October 2, 11:59 PM GMT+8, matching Manila time. Use
2026-10-02T23:59:00+08:00 as the announced submission deadline.

The user supplied the organizer's Discord clarification on October 1 at 23:35:
Icy confirms that the deadline has moved to October 2 according to the update
notice. We accept October 2, 23:59 GMT+8 as the working deadline. The stale
October 1 text on the rules page is not a blocker to the authorized work.

## Track 01 requirements

The [official rules](https://gibc-v2.devpost.com/rules) require: at most 50M
trainable parameters including embeddings/head; random initialization without
pretraining, fine-tuning or distillation; count/config evidence; hardware, time
and compute reporting; HellaSwag, ARC-Easy, PIQA and WinoGrande via
lm-evaluation-harness, plus held-out WikiText-103 perplexity and evaluation code.
AI assistance requires disclosure and author understanding. Students aged 13+
may enter, solo or in teams up to six. No minimum training-token count was found.

## Submission materials

The [overview](https://gibc-v2.devpost.com/) lists six components: description;
public runnable source with README; accessible 2–5-minute English/subtitled demo;
Built With credits; real team identities/accounts; at least three screenshots.
Training curves and pipeline screenshots can satisfy the image requirement.

## HelixDepth implications and current evidence

- Actual baseline count: 40,968,320. Actual HelixDepth count: 41,184,432.
  Both are below the cap; source and count evidence are already available.
- The implementation initializes weights randomly; no pretrained weights were
  downloaded. The tokenizer was fitted locally and is now frozen.
- Corpus preparation is verified, but main training and the named benchmark
  evaluations are not complete. Our FineWeb-Edu validation loss cannot be
  presented as WikiText-103 perplexity or as a benchmark accuracy score.
- Current code assistance is disclosed in the repository; that disclosure must
  remain accurate as more work is added. Final publication, trained artifacts,
  demonstration, team information and submission access still need verification.
- Recommendation, not a rule: use one matched pass over the expanded corpus,
  99,991,552 scored targets per model at context 512 / batch 16. This leaves
  time and budget for evaluation and packaging. The user subsequently authorized
  this protocol, existing Pod usage, training, recovery and evaluation within the
  remaining USD13.72. The protocol is saved in configs/training_main.yaml.
- GPU recovery verification, refreshed transfer packaging and a final shared
  training protocol remain next engineering work. No additional corpus expansion
  is required for the proposed single-pass comparison.

## Scope of this review

This review establishes the published requirements and the deadline discrepancy;
it does not certify participant eligibility, settle organizer intent, prove
benchmark decontamination or claim the final submission is complete.
No code, data, model, GPU state, commits or public submission changed.
