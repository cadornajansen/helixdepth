"""Float32 training with explicit targets and atomic, complete recovery state."""
import math
import os
import random
import sys
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import Tensor
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from .config import ModelConfig
from .model import LanguageModel
from .packing import PackedStream


@dataclass(frozen=True)
class TrainingConfig:
    context: int = 512
    batch_size: int = 1
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    gradient_clip: float = 1.0
    warmup_steps: int = 2
    schedule_steps: int = 8
    minimum_lr_fraction: float = 0.1
    seed: int = 2026
    deterministic: bool = True
    cpu_threads: int = 1
    validation_batches: int = 2
    checkpoint_every: int = 2
    precision: str = "float32"

    def __post_init__(self) -> None:
        for name in ("context", "batch_size", "schedule_steps", "cpu_threads", "validation_batches", "checkpoint_every"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not 0 <= self.warmup_steps < self.schedule_steps:
            raise ValueError("Warmup must be shorter than the schedule")
        if self.learning_rate <= 0 or self.gradient_clip <= 0 or self.weight_decay < 0:
            raise ValueError("Invalid optimizer settings")
        if not 0 <= self.minimum_lr_fraction <= 1 or self.precision != "float32":
            raise ValueError("Only float32 training is implemented")

    @classmethod
    def from_yaml(cls, path: Path) -> "TrainingConfig":
        return cls(**yaml.safe_load(path.read_text(encoding="utf-8")))


def seed_runtime(config: TrainingConfig, device: torch.device) -> None:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.set_num_threads(config.cpu_threads)
    torch.use_deterministic_algorithms(config.deterministic)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = config.deterministic
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)


def shifted_loss(model: LanguageModel, inputs: Tensor, targets: Tensor, reduction: str = "mean") -> Tensor:
    """Packing already shifts targets by one; do not call model.loss and shift twice."""
    if inputs.shape != targets.shape:
        raise ValueError("Inputs and targets must have matching shapes")
    logits = model(inputs)
    return F.cross_entropy(logits.reshape(-1, model.config.vocab_size), targets.reshape(-1), reduction=reduction)


class Trainer:
    def __init__(self, model_config: ModelConfig, config: TrainingConfig,
                 data_dir: Path, device: str = "cpu") -> None:
        self.device = torch.device(device)
        if self.device.type not in ("cpu", "cuda"):
            raise ValueError("Supported devices are cpu and cuda")
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable. Profiling requires an authorized GPU and a CUDA-enabled PyTorch build.")
        if config.context > model_config.max_context:
            raise ValueError("Training context exceeds the model context")
        self.config = config
        seed_runtime(config, self.device)
        self.train_data = PackedStream(data_dir, "train", config.context, config.batch_size)
        self.validation_data = PackedStream(data_dir, "validation", config.context, 1)
        self.model = LanguageModel(model_config).to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.learning_rate,
                                          weight_decay=config.weight_decay, betas=(0.9, 0.95),
                                          foreach=False, fused=False)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, self.lr_multiplier)
        self.step = 0
        self.history: list[dict[str, int | float]] = []

    def lr_multiplier(self, completed_steps: int) -> float:
        next_step = completed_steps + 1
        if next_step <= self.config.warmup_steps:
            return next_step / self.config.warmup_steps
        progress = min(1.0, (next_step - self.config.warmup_steps) /
                       (self.config.schedule_steps - self.config.warmup_steps))
        floor = self.config.minimum_lr_fraction
        return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress))

    def attention_context(self) -> Any:
        return sdpa_kernel(SDPBackend.MATH) if self.config.deterministic else nullcontext()

    def train_step(self) -> dict[str, int | float]:
        if self.step >= self.config.schedule_steps:
            raise ValueError("Configured schedule exhausted; final experiment budget is not set")
        inputs, targets = self.train_data.peek()
        offset = self.train_data.offset
        learning_rate = self.optimizer.param_groups[0]["lr"]
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        with self.attention_context():
            loss = shifted_loss(self.model, inputs.to(self.device), targets.to(self.device))
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss")
            loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip,
                                              error_if_nonfinite=True, foreach=False)
        self.optimizer.step()
        self.scheduler.step()
        self.train_data.advance()
        self.step += 1
        metric = {"step": self.step, "loss": loss.item(), "learning_rate": learning_rate,
                  "gradient_norm_before_clip": norm.item(), "data_offset_before": offset,
                  "data_offset_after": self.train_data.offset,
                  "target_tokens": targets.numel()}
        self.history.append(metric)
        return metric

    def validate(self, max_batches: int | None = None) -> dict[str, int | float | bool]:
        prior_mode = self.model.training
        total_loss = 0.0
        scored_tokens = 0
        batches = 0
        data = self.validation_data
        offset = 0
        self.model.eval()
        try:
            with torch.no_grad(), self.attention_context():
                while offset < len(data.tokens) - 1 and (max_batches is None or batches < max_batches):
                    length = min(data.context, len(data.tokens) - offset - 1)
                    inputs = data.tokens[offset:offset + length].long().unsqueeze(0).to(self.device)
                    targets = data.tokens[offset + 1:offset + length + 1].long().unsqueeze(0).to(self.device)
                    loss = shifted_loss(self.model, inputs, targets, reduction="sum")
                    if not torch.isfinite(loss):
                        raise ValueError("Non-finite validation loss")
                    total_loss += loss.item()
                    scored_tokens += length
                    offset += length
                    batches += 1
        finally:
            self.model.train(prior_mode)
        if not scored_tokens:
            raise ValueError("Validation needs at least two tokens")
        return {"loss": total_loss / scored_tokens, "scored_tokens": scored_tokens,
                "batches": batches, "complete_split": scored_tokens == len(data.tokens) - 1}

    def runtime_identity(self) -> dict[str, Any]:
        return {"torch": str(torch.__version__), "python": sys.version,
                "device_type": self.device.type, "precision": self.config.precision,
                "deterministic": self.config.deterministic, "cpu_threads": self.config.cpu_threads,
                "attention": "math" if self.config.deterministic else "default"}

    def save_checkpoint(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "format_version": 1, "model_config": self.model.config.to_dict(),
            "training_config": asdict(self.config), "runtime": self.runtime_identity(),
            "model": self.model.state_dict(), "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(), "step": self.step, "history": self.history,
            "data": self.train_data.state_dict(), "validation_identity": self.validation_data.identity,
            "rng": {"python": random.getstate(), "torch_cpu": torch.get_rng_state(),
                    "torch_cuda": torch.cuda.get_rng_state_all() if self.device.type == "cuda" else []},
            "tokens_processed": self.step * self.config.batch_size * self.config.context,
            "boundary": "After complete optimizer/scheduler step; no gradient accumulation",
        }
        temporary = path.with_name(path.name + ".tmp")
        with temporary.open("wb") as stream:
            torch.save(checkpoint, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)

    def load_checkpoint(self, path: Path) -> None:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if checkpoint["format_version"] != 1 or checkpoint["model_config"] != self.model.config.to_dict():
            raise ValueError("Checkpoint model configuration mismatch")
        if checkpoint["training_config"] != asdict(self.config) or checkpoint["runtime"] != self.runtime_identity():
            raise ValueError("Checkpoint training/runtime mismatch; deterministic continuation needs the same settings")
        if checkpoint["validation_identity"] != self.validation_data.identity:
            raise ValueError("Checkpoint validation corpus mismatch")
        self.train_data.load_state_dict(checkpoint["data"])
        expected_tokens = checkpoint["step"] * self.config.batch_size * self.config.context
        if checkpoint["tokens_processed"] != expected_tokens or self.train_data.offset != expected_tokens:
            raise ValueError("Checkpoint step and data position disagree")
        self.model.load_state_dict(checkpoint["model"])
        # Initialize both first, then restore scheduler and optimizer in this order.
        self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.step = checkpoint["step"]
        self.history = checkpoint["history"]
        self.optimizer.zero_grad(set_to_none=True)
        random.setstate(checkpoint["rng"]["python"])
        torch.set_rng_state(checkpoint["rng"]["torch_cpu"])
        if self.device.type == "cuda":
            torch.cuda.set_rng_state_all(checkpoint["rng"]["torch_cuda"])
