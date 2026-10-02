# Publication inventory — October 2, 2026

This release publishes previously local training/evaluation work plus final
300M evidence and recording material. Checkpoints, datasets, credentials, raw
download archives and virtual environments are excluded from Git. No files deleted.

## Published file inventory

| File | Purpose |
|---|---|
| `README.md` | Documentation and reproduction instructions |
| `docs/continuation.md` | Documentation and reproduction instructions |
| `docs/evaluation.md` | Documentation and reproduction instructions |
| `docs/generation-demo.md` | Documentation and reproduction instructions |
| `docs/model-card.md` | Documentation and reproduction instructions |
| `docs/records.md` | Documentation and reproduction instructions |
| `docs/results/baseline_continuation_200m.json` | Measured report or verification evidence |
| `docs/results/baseline_continuation_300m.json` | Measured report or verification evidence |
| `docs/results/baseline_continuation_v2_stage.json` | Measured report or verification evidence |
| `docs/results/continuation_200m_backup_verified.json` | Measured report or verification evidence |
| `docs/results/continuation_300m_backup_verified.json` | Measured report or verification evidence |
| `docs/results/continuation_corpus.json` | Measured report or verification evidence |
| `docs/results/continuation_corpus_v2.json` | Measured report or verification evidence |
| `docs/results/continuation_gpu_check.json` | Measured report or verification evidence |
| `docs/results/continuation_stage.json` | Measured report or verification evidence |
| `docs/results/evaluation_recovery_tests.xml` | Measured report or verification evidence |
| `docs/results/evaluation_summary.json` | Measured report or verification evidence |
| `docs/results/evaluation_summary_300m.json` | Measured report or verification evidence |
| `docs/results/generation_demo_300m.json` | Measured report or verification evidence |
| `docs/results/helixdepth_continuation_200m.json` | Measured report or verification evidence |
| `docs/results/helixdepth_continuation_300m.json` | Measured report or verification evidence |
| `docs/results/main_baseline.json` | Measured report or verification evidence |
| `docs/results/main_helixdepth.json` | Measured report or verification evidence |
| `docs/video-guide.md` | Documentation and reproduction instructions |
| `docs/visuals/build_explainer.py` | Architecture and results visual |
| `docs/visuals/helixdepth-explained.html` | Architecture and results visual |
| `docs/visuals/helixdepth-explained.png` | Architecture and results visual |
| `docs/visuals/helixdepth-explained.svg` | Architecture and results visual |
| `scripts/check_continuation_gpu.py` | Training, evaluation, generation or verification tool |
| `scripts/evaluate_model.py` | Training, evaluation, generation or verification tool |
| `scripts/evaluate_suite.py` | Training, evaluation, generation or verification tool |
| `scripts/generate_demo.py` | Training, evaluation, generation or verification tool |
| `scripts/prepare_continuation_data.py` | Training, evaluation, generation or verification tool |
| `scripts/run_continuation.py` | Training, evaluation, generation or verification tool |
| `scripts/start_baseline_continuation.sh` | Training, evaluation, generation or verification tool |
| `scripts/start_continuation.sh` | Training, evaluation, generation or verification tool |
| `scripts/start_matched_continuation_v2.sh` | Training, evaluation, generation or verification tool |
| `scripts/verify_continuation_backup.py` | Training, evaluation, generation or verification tool |
| `tests/test_continuation.py` | Training/evaluation checks |
| `tests/test_evaluation_runner.py` | Training/evaluation checks |
| `docs/publication.md` | This complete publication inventory and checks |
| `docs/results/model_release_verified.json` | Public model release identity and integrity receipt |

## Verification

- Full repository suite: 55 tests passed in 60.69 seconds.
- All ten final evaluation receipts and both local model backups reverified.
- Final diagram rendered and inspected; old 200M snapshot replaced with 300M.
- Exact 32-token CPU command in the video guide ran successfully.
- Staged files checked for accidental model/dataset archives and credential patterns.

## Recording handoff

Use [video-guide.md](video-guide.md) for narration and a local generation command.
The model outputs are repetitive and sometimes false; preserve this limitation.
The final benchmark comparison is an equal-data, nearly equal-parameter experiment,
not proof of a compute-efficiency advantage or broad reasoning superiority.

Final GitHub/Hugging Face/Notion publication outcomes are recorded in records.md.
The Pod remains running under the owner's explicit instruction.
