from pathlib import Path
from collections.abc import Callable
from typing import Literal

import pytest
import torch
from torch.nn import functional as F

from helixdepth.config import ModelConfig, tiny_config
from helixdepth.hypernetwork import LowRankBasis
from helixdepth.model import LanguageModel, apply_rope


@pytest.fixture(autouse=True)
def deterministic_cpu() -> None:
    torch.manual_seed(2026)
    torch.set_num_threads(1)


@pytest.mark.parametrize("variant", ["baseline", "helixdepth"])
def test_shape_loss_causality_backward(variant: Literal["baseline", "helixdepth"]) -> None:
    model = LanguageModel(tiny_config(variant)).eval()
    tokens = torch.randint(0, 32, (2, 12))
    logits = model(tokens)
    assert logits.shape == (2, 12, 32)
    expected = F.cross_entropy(logits[:, :-1].reshape(-1, 32), tokens[:, 1:].reshape(-1))
    torch.testing.assert_close(model.loss(tokens), expected)
    future = tokens.clone()
    future[:, 6:] = (future[:, 6:] + 1) % 32
    torch.testing.assert_close(model(future)[:, :6], logits[:, :6], rtol=0, atol=1e-6)
    torch.testing.assert_close(model(tokens[:, :6]), logits[:, :6], rtol=0, atol=1e-6)
    loss = model.loss(tokens)
    loss.backward()
    assert torch.isfinite(loss)
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name


def test_adapter_orientation() -> None:
    basis = LowRankBasis(8, 12, 3)
    linear = torch.nn.Linear(8, 12, bias=False)
    x, coefficients = torch.randn(2, 5, 8), torch.randn(3)
    actual = linear(x) + basis(x, coefficients)
    dense_weight = linear.weight + (basis.A @ torch.diag(coefficients) @ basis.B).T
    torch.testing.assert_close(actual, F.linear(x, dense_weight))


def test_depth_path_learns() -> None:
    model = LanguageModel(tiny_config("helixdepth"))
    assert model.conditioner is not None
    watched = {name: p for name, p in model.named_parameters()
               if name.startswith(("bases.", "conditioner."))}
    before = {name: p.detach().clone() for name, p in watched.items()}
    coefficients_before = model.conditioner().detach().clone()
    captured: list[torch.Tensor] = []

    def retain_coefficients(module: torch.nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        output.retain_grad()
        captured.append(output)

    handle = model.conditioner.register_forward_hook(retain_coefficients)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.0)
    tokens = torch.randint(0, 32, (2, 12))
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        model.loss(tokens).backward()
        gradients = captured[-1].grad
        assert gradients is not None and torch.isfinite(gradients).all()
        assert (gradients.abs().sum(-1) > 0).all(), "Every stage/projection needs coefficient gradients"
        for name, parameter in watched.items():
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
            assert parameter.grad.abs().sum() > 0, name
        optimizer.step()
    handle.remove()
    for name, parameter in watched.items():
        assert not torch.equal(before[name], parameter), name
    assert not torch.equal(coefficients_before, model.conditioner())
    assert not torch.equal(model.conditioner()[0], model.conditioner()[1])


@pytest.mark.parametrize("variant", ["baseline", "helixdepth"])
def test_round_trip_and_sharing(variant: Literal["baseline", "helixdepth"], tmp_path: Path) -> None:
    model = LanguageModel(tiny_config(variant)).eval()
    tokens = torch.randint(0, 32, (2, 8))
    path = tmp_path / "model.pt"
    model.save(path)
    restored = LanguageModel.load(path).eval()
    assert restored.config == model.config
    assert restored.head.weight is restored.embedding.weight
    assert restored.schedule == ((0, 1) if variant == "baseline" else (0, 1, 0, 1))
    assert len({id(block) for block in restored.blocks}) == 2
    assert len({id(norm.weight) for pair in restored.stage_norms for norm in pair}) == 2 * model.config.stages
    calls = [0, 0]

    def counter(index: int) -> Callable[[torch.nn.Module, tuple[object, ...], torch.Tensor], None]:
        def hook(module: torch.nn.Module, inputs: tuple[object, ...], output: torch.Tensor) -> None:
            calls[index] += 1
        return hook

    handles = [block.register_forward_hook(counter(i)) for i, block in enumerate(restored.blocks)]
    torch.testing.assert_close(restored(tokens), model(tokens), rtol=0, atol=0)
    assert calls == ([1, 1] if variant == "baseline" else [2, 2])
    for handle in handles:
        handle.remove()


def test_rope_positions_and_norm() -> None:
    x = torch.randn(2, 4, 8, 8)
    rotated = apply_rope(x, 10000.0)
    torch.testing.assert_close(rotated[:, :, 0], x[:, :, 0])
    torch.testing.assert_close(rotated.square().sum(-1), x.square().sum(-1))
    assert not torch.equal(rotated[:, :, 1:], x[:, :, 1:])


def test_input_limits_and_config() -> None:
    model = LanguageModel(tiny_config("baseline"))
    with pytest.raises(ValueError):
        model(torch.zeros(1, 17, dtype=torch.long))
    with pytest.raises(ValueError):
        model.loss(torch.zeros(1, 1, dtype=torch.long))
    with pytest.raises(ValueError):
        ModelConfig(width=30, heads=10)
    with pytest.raises(ValueError):
        ModelConfig(stages=12)
