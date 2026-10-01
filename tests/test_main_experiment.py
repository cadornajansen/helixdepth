import json
from array import array
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

from helixdepth.config import tiny_config
from helixdepth.evaluation import score_tokens
from helixdepth.model import LanguageModel
from helixdepth.packing import sha256_file
from helixdepth.training import TrainingConfig
from scripts.run_experiment import train_run


def test_likelihood_shift_padding_and_long_continuations() -> None:
    torch.set_num_threads(1)
    model = LanguageModel(tiny_config("baseline"))
    context = [4, 5, 6]
    continuation = [7, 8, 9, 10]
    expected = F.log_softmax(model(torch.tensor([context + continuation[:-1]])), -1)
    expected = expected[0, 2:6].gather(1, torch.tensor(continuation)[:, None]).sum().item()
    pairs = [(context, continuation), ([2], [3]), ([2], [4] * (model.config.max_context + 3))]
    singles = score_tokens(model, pairs, 1)
    batched = score_tokens(model, pairs, 3)
    assert batched[0][0] == pytest.approx(expected, abs=1e-5)
    for single, batch in zip(singles, batched):
        assert single[0] == pytest.approx(batch[0], abs=1e-4)
        assert single[1] == batch[1]
    assert model.training


def test_main_run_report_checkpoint_and_full_validation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data = tmp_path / "data"
    data.mkdir()
    manifest = {"format": "little-endian-int32", "tokenizer_sha256": "fixture", "splits": {}}
    for split, count in (("train", 65), ("validation", 19)):
        path = data / f"{split}.bin"
        with path.open("wb") as stream:
            array("i", [i % 32 for i in range(count)]).tofile(stream)
        manifest["splits"][split] = {"tokens": count, "bin_sha256": sha256_file(path)}
    (data / "manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr("scripts.run_experiment.ModelConfig.from_yaml", lambda path: tiny_config("baseline"))
    config = TrainingConfig(context=8, batch_size=2, warmup_steps=1, schedule_steps=4)
    report = train_run("baseline", config, data, tmp_path / "run", 4, None, "cpu")
    assert report["target_tokens"] == 64
    assert report["final_validation_full"]["scored_tokens"] == 18
    assert report["complete_schedule"]
    checkpoint = torch.load(tmp_path / "run/latest.pt", weights_only=True)
    assert checkpoint["tokens_processed"] == 64
    assert len(LanguageModel.load(tmp_path / "run/model.pt").state_dict()) > 0
    with pytest.raises(FileExistsError):
        train_run("baseline", config, data, tmp_path / "run", 4, None, "cpu")
