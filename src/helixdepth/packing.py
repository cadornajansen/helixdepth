"""Checksum-verified, single-pass token streams shared by both architectures."""
import hashlib
import json
import os
import sys
from array import array
from pathlib import Path
from typing import Any

import torch
from torch import Tensor


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def pack_cp2(source: Path, evidence: Path, destination: Path) -> dict[str, Any]:
    if destination.exists():
        raise FileExistsError(f"Preserving existing packed corpus: {destination}")
    if sys.byteorder != "little" or array("i").itemsize != 4:
        raise RuntimeError("Packed format requires little-endian 32-bit integers")
    report = json.loads(evidence.read_text(encoding="utf-8"))
    if not report["passed"] or report["vocab_size_including_special_tokens"] != 16384:
        raise ValueError("CP2 evidence must pass with vocabulary 16,384")
    required = ("tokenizer.json", "manifest.json", "train.jsonl", "validation.jsonl",
                "train.tokens.jsonl", "validation.tokens.jsonl", "tokenizer_training.json")
    for name in required:
        if sha256_file(source / name) != report["files"][name]["sha256"]:
            raise ValueError(f"CP2 artifact checksum mismatch: {name}")
    destination.mkdir(parents=True)
    manifest: dict[str, Any] = {
        "format": "little-endian-int32", "vocab_size": 16384, "eos_id": 2,
        "tokenizer_sha256": report["files"]["tokenizer.json"]["sha256"],
        "source_revision": report["settings"]["revision"], "splits": {},
        "order": "Saved CP2 JSONL row order; no shuffle, no implicit epochs",
        "packing": "Concatenate document IDs including one EOS each; windows use stride=context with one-token input/target shift",
    }
    for split in ("train", "validation"):
        count = 0
        document_count = 0
        with (source / f"{split}.tokens.jsonl").open(encoding="utf-8") as rows:
            with (destination / f"{split}.bin").open("wb") as output:
                with (destination / f"{split}.index.jsonl").open("w", encoding="utf-8", newline="\n") as index:
                    for line in rows:
                        row = json.loads(line)
                        ids = row["input_ids"]
                        if not ids or ids[-1] != 2 or any(type(value) is not int or not 0 <= value < 16384 for value in ids):
                            raise ValueError("Invalid CP2 token row")
                        index.write(json.dumps({"document_id": row["document_id"], "offset": count,
                                                "tokens": len(ids)}) + "\n")
                        array("i", ids).tofile(output)
                        count += len(ids)
                        document_count += 1
                    output.flush()
                    os.fsync(output.fileno())
        if count != report["splits"][split]["tokens_including_eos"] or document_count != report["splits"][split]["documents"]:
            raise ValueError("Packed counts differ from CP2 evidence")
        manifest["splits"][split] = {
            "tokens": count, "documents": document_count,
            "source_tokens_sha256": report["files"][f"{split}.tokens.jsonl"]["sha256"],
            "bin_sha256": sha256_file(destination / f"{split}.bin"),
            "index_sha256": sha256_file(destination / f"{split}.index.jsonl"),
        }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


class PackedStream:
    def __init__(self, directory: Path, split: str, context: int, batch_size: int) -> None:
        if context < 1 or batch_size < 1 or split not in ("train", "validation"):
            raise ValueError("Invalid packed stream settings")
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if manifest["format"] != "little-endian-int32" or sys.byteorder != "little":
            raise ValueError("Unsupported packed stream format")
        path = directory / f"{split}.bin"
        split_info = manifest["splits"][split]
        if path.stat().st_size != split_info["tokens"] * 4 or sha256_file(path) != split_info["bin_sha256"]:
            raise ValueError("Packed token file differs from manifest")
        self.tokens = torch.from_file(str(path.resolve()), shared=False, size=split_info["tokens"], dtype=torch.int32)
        self.context = context
        self.batch_size = batch_size
        self.offset = 0
        self.identity = {"bin_sha256": split_info["bin_sha256"], "split": split,
                         "tokenizer_sha256": manifest["tokenizer_sha256"], "tokens": split_info["tokens"]}

    def peek(self) -> tuple[Tensor, Tensor]:
        needed = self.batch_size * self.context
        if self.offset + needed + 1 > len(self.tokens):
            raise StopIteration("Single-pass corpus exhausted; expand unique data before longer training")
        inputs = self.tokens[self.offset:self.offset + needed].reshape(self.batch_size, self.context).long()
        targets = self.tokens[self.offset + 1:self.offset + needed + 1].reshape(self.batch_size, self.context).long()
        return inputs, targets

    def advance(self) -> None:
        self.offset += self.batch_size * self.context

    def state_dict(self) -> dict[str, Any]:
        return {"identity": self.identity, "offset": self.offset, "context": self.context,
                "batch_size": self.batch_size, "epoch": 0}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        expected = self.state_dict()
        if any(state[key] != expected[key] for key in ("identity", "context", "batch_size", "epoch")):
            raise ValueError("Checkpoint data/settings mismatch")
        offset = state["offset"]
        if type(offset) is not int or offset < 0 or offset % (self.context * self.batch_size) or offset > len(self.tokens) - 1:
            raise ValueError("Invalid checkpoint data position")
        self.offset = offset
