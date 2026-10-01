"""Causal continuation scoring for the frozen tokenizer; no pretrained assets."""
from typing import Sequence

import torch
from tokenizers import Tokenizer

from .model import LanguageModel


def score_tokens(model: LanguageModel, pairs: Sequence[tuple[list[int], list[int]]],
                 batch_size: int = 16) -> list[tuple[float, bool]]:
    """Score every continuation token once, left-truncating context as needed."""
    if batch_size < 1:
        raise ValueError("Batch size must be positive")
    context_size = model.config.max_context
    device = next(model.parameters()).device
    scores = [0.0] * len(pairs)
    greedy = [True] * len(pairs)
    windows: list[tuple[int, list[int], list[int]]] = []
    for index, (context, continuation) in enumerate(pairs):
        if not context:
            raise ValueError("An explicit prefix token is required")
        for offset in range(0, len(continuation), context_size):
            targets = continuation[offset:offset + context_size]
            prefix = context + continuation[:offset]
            inputs = (prefix + targets[:-1])[-context_size:]
            windows.append((index, inputs, targets))
    # Similar lengths reduce padding while final outputs retain request order.
    windows.sort(key=lambda item: len(item[1]), reverse=True)
    prior_mode = model.training
    model.eval()
    try:
        with torch.inference_mode():
            for offset in range(0, len(windows), batch_size):
                batch = windows[offset:offset + batch_size]
                tokens = torch.zeros((len(batch), max(len(row[1]) for row in batch)), dtype=torch.long, device=device)
                for row, (_, inputs, _) in enumerate(batch):
                    tokens[row, :len(inputs)] = torch.tensor(inputs, device=device)
                logits = model(tokens)
                for row, (index, inputs, targets) in enumerate(batch):
                    selected = logits[row, len(inputs) - len(targets):len(inputs)].float()
                    target = torch.tensor(targets, device=device)
                    values = selected.log_softmax(-1).gather(-1, target[:, None]).squeeze(-1)
                    if not torch.isfinite(values).all():
                        raise ValueError("Non-finite evaluation log probabilities")
                    scores[index] += values.sum().item()
                    greedy[index] = greedy[index] and bool((selected.argmax(-1) == target).all().item())
    finally:
        model.train(prior_mode)
    return list(zip(scores, greedy))


def encode_pair(tokenizer: Tokenizer, context: str, continuation: str,
                prefix_id: int = 2) -> tuple[list[int], list[int]]:
    """Move trailing spaces into the answer, following harness causal scoring."""
    trimmed = context.rstrip()
    continuation = context[len(trimmed):] + continuation
    context = trimmed
    context_ids = tokenizer.encode(context, add_special_tokens=False).ids
    joined = tokenizer.encode(context + continuation, add_special_tokens=False).ids
    # A BPE merge crossing the boundary belongs to the scored continuation.
    shared = 0
    while shared < min(len(context_ids), len(joined)) and context_ids[shared] == joined[shared]:
        shared += 1
    return joined[:shared] or [prefix_id], joined[shared:]
