"""Load a locally prepared Primitive Operation Painter model package.

The module accepts either a local package directory or a Hugging Face repo ID.
Using a repo ID delegates the download to ``huggingface_hub``; normal local
usage never makes a network request.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_model

from model_gpt import GeometrizeGPT


CONFIG_FILE = "config.json"
WEIGHTS_FILE = "model.safetensors"
REQUIRED_ARCHITECTURE_FIELDS = (
    "vocab_size",
    "d_model",
    "n_layer",
    "n_head",
)
REQUIRED_SEQUENCE_FIELDS = (
    "prefix_steps",
    "prediction_steps",
    "context_steps",
    "tokens_per_step",
    "position_embedding_steps",
)


def resolve_model_directory(model_dir_or_repo: str | Path) -> Path:
    """Return a local package directory, downloading only when given a repo ID."""
    candidate = Path(model_dir_or_repo).expanduser()
    if candidate.is_dir():
        return candidate.resolve()

    # Treat a non-directory value as a Hub repository ID only on explicit use.
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "Model directory does not exist. Install huggingface_hub to load a "
            "remote repository, or supply an existing local model directory."
        ) from exc

    return Path(snapshot_download(repo_id=str(model_dir_or_repo))).resolve()


def _require_positive_int(mapping: dict[str, Any], key: str) -> int:
    value = mapping.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"config.json field '{key}' must be a positive integer")
    return value


def load_release_config(model_dir_or_repo: str | Path) -> tuple[Path, dict[str, Any]]:
    """Read and validate the model package configuration."""
    model_dir = resolve_model_directory(model_dir_or_repo)
    config_path = model_dir / CONFIG_FILE
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing release configuration: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    if not isinstance(config, dict) or config.get("model_type") != "geometrize_gpt":
        raise ValueError("config.json is not a Primitive Operation Painter release config")
    if config.get("format_version") != 1:
        raise ValueError("Unsupported config.json format_version")

    architecture = config.get("architecture")
    sequence = config.get("sequence")
    token_layout = config.get("token_layout")
    weights = config.get("weights")
    if not isinstance(architecture, dict) or architecture.get("name") != "GeometrizeGPT":
        raise ValueError("config.json has an invalid architecture section")
    if not isinstance(sequence, dict) or not isinstance(token_layout, dict):
        raise ValueError("config.json requires sequence and token_layout sections")
    if not isinstance(weights, dict) or weights.get("file") != WEIGHTS_FILE:
        raise ValueError(f"config.json must reference {WEIGHTS_FILE}")

    for key in REQUIRED_ARCHITECTURE_FIELDS:
        _require_positive_int(architecture, key)
    for key in REQUIRED_SEQUENCE_FIELDS:
        _require_positive_int(sequence, key)

    context_steps = sequence["context_steps"]
    prefix_steps = sequence["prefix_steps"]
    prediction_steps = sequence["prediction_steps"]
    tokens_per_step = sequence["tokens_per_step"]
    position_steps = sequence["position_embedding_steps"]
    if prefix_steps + prediction_steps != context_steps:
        raise ValueError("prefix_steps + prediction_steps must equal context_steps")
    if context_steps > position_steps:
        raise ValueError("context_steps cannot exceed position_embedding_steps")
    if architecture["vocab_size"] != token_layout.get("vocab_size"):
        raise ValueError("architecture vocab_size does not match token_layout vocab_size")
    if not isinstance(token_layout.get("version"), str):
        raise ValueError("token_layout.version must be a string")

    return model_dir, config


def build_model_from_config(config: dict[str, Any]) -> GeometrizeGPT:
    """Build an uninitialized model using the exact release configuration."""
    architecture = config["architecture"]
    sequence = config["sequence"]
    tokens_per_step = sequence["tokens_per_step"]
    return GeometrizeGPT(
        vocab_size=architecture["vocab_size"],
        d_model=architecture["d_model"],
        n_layer=architecture["n_layer"],
        n_head=architecture["n_head"],
        max_context_len=sequence["context_steps"] * tokens_per_step,
        max_position_embeddings=sequence["position_embedding_steps"] * tokens_per_step,
    )


def load_pretrained(
    model_dir_or_repo: str | Path,
    device: str | torch.device = "cpu",
) -> tuple[GeometrizeGPT, dict[str, Any], Path]:
    """Strictly load a release package and return ``(model, config, directory)``."""
    model_dir, config = load_release_config(model_dir_or_repo)
    weights_path = model_dir / WEIGHTS_FILE
    if not weights_path.is_file():
        raise FileNotFoundError(f"Missing model weights: {weights_path}")

    model = build_model_from_config(config)
    load_model(model, str(weights_path), strict=True, device="cpu")
    if model.wte.weight.data_ptr() != model.lm_head.weight.data_ptr():
        raise RuntimeError("Loaded model no longer has tied input/output embeddings")
    model.to(device)
    model.eval()
    return model, config, model_dir
