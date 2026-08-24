"""Render a six-row comparison from the bundled 11-step example sequences.

The example CSV stores one background operation followed by ten accepted
primitives for each image. Those eleven steps are the model prompt. The
released 144-step model then autoregressively completes the remaining 133
steps; the output figure compares the short input with the full completion.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from pretrained import load_pretrained
from token_layout import TOKEN_LAYOUT
from visualize import (
    CSV_DTYPE,
    decode_tokens_to_render_data,
    encode_image_group,
    generate,
    render_single_image,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CSV_PATH = PROJECT_ROOT / "example" / "sequences" / "v1" / "data_part_1.csv"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "example" / "example_inference.png"
EXAMPLE_SEQUENCE_STEPS = 11
COMPLETION_STEPS = 144
EXAMPLE_COUNT = 6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render six ground-truth/prediction pairs from the bundled "
            "11-step Primitive Operation Painter example sequences."
        )
    )
    parser.add_argument(
        "--model-dir",
        required=True,
        type=Path,
        help="Local model package directory containing config.json and model.safetensors.",
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=DEFAULT_CSV_PATH,
        help=f"Example CSV path (default: {DEFAULT_CSV_PATH}).",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Comparison image path (default: {DEFAULT_OUTPUT_PATH}).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.4,
        help="Positive sampling temperature for the 144-step completion (default: 0.4).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Torch sampling seed (default: 0).",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Inference device (default: auto).",
    )
    return parser.parse_args()


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is not available.")
    return torch.device(device_name)


def load_example_groups(csv_path: Path) -> list[tuple[str, pd.DataFrame]]:
    csv_path = csv_path.expanduser().resolve()
    if not csv_path.is_file():
        raise FileNotFoundError(f"Example CSV not found: {csv_path}")

    frame = pd.read_csv(csv_path, dtype=CSV_DTYPE, engine="c")
    expected_columns = list(CSV_DTYPE)
    if list(frame.columns) != expected_columns:
        raise ValueError(
            f"Unexpected CSV columns in {csv_path}; expected {expected_columns}, "
            f"got {list(frame.columns)}"
        )
    if frame.empty:
        raise ValueError(f"Example CSV is empty: {csv_path}")
    if frame["image_name"].isna().any():
        raise ValueError(f"Example CSV contains an empty image_name: {csv_path}")

    image_names = frame["image_name"].to_numpy(copy=False)
    starts = np.concatenate(
        (
            np.array([0], dtype=np.int64),
            np.flatnonzero(image_names[1:] != image_names[:-1]).astype(np.int64) + 1,
        )
    )
    ends = np.concatenate((starts[1:], np.array([len(frame)], dtype=np.int64)))
    groups: list[tuple[str, pd.DataFrame]] = []
    seen_names: set[str] = set()
    for start, end in zip(starts, ends, strict=True):
        group = frame.iloc[start:end].copy()
        image_name = str(group["image_name"].iloc[0])
        if image_name in seen_names:
            raise ValueError(
                f"Example CSV must keep each image_name contiguous; {image_name!r} reappears."
            )
        seen_names.add(image_name)
        if len(group) != EXAMPLE_SEQUENCE_STEPS:
            raise ValueError(
                f"Example sequence {image_name!r} has {len(group)} rows; "
                f"expected exactly {EXAMPLE_SEQUENCE_STEPS}."
            )
        if int(group.iloc[0]["shape_type"]) != -1:
            raise ValueError(f"Example sequence {image_name!r} must begin with a background row.")
        if (group.iloc[1:]["shape_type"] < 0).any():
            raise ValueError(
                f"Example sequence {image_name!r} contains a non-background negative shape_type."
            )
        groups.append((image_name, group))

    if len(groups) != EXAMPLE_COUNT:
        raise ValueError(f"Example CSV contains {len(groups)} sequences; expected {EXAMPLE_COUNT}.")
    return groups


def validate_release_config(release_config: dict) -> int:
    sequence = release_config["sequence"]
    if release_config["token_layout"]["version"] != TOKEN_LAYOUT.version:
        raise ValueError("Release token layout does not match this source checkout.")
    if sequence["prefix_steps"] != EXAMPLE_SEQUENCE_STEPS - 1:
        raise ValueError(
            f"This example requires prefix_steps={EXAMPLE_SEQUENCE_STEPS - 1}; "
            f"the model package declares {sequence['prefix_steps']}."
        )
    if sequence["tokens_per_step"] != 9:
        raise ValueError(
            f"This example requires nine tokens per step; the model package declares "
            f"{sequence['tokens_per_step']}."
        )
    if sequence["context_steps"] != COMPLETION_STEPS:
        raise ValueError(
            f"This example requires a {COMPLETION_STEPS}-step model; the model "
            f"package declares {sequence['context_steps']} context steps."
        )
    return sequence["tokens_per_step"]


def main() -> None:
    args = parse_args()
    if args.temperature <= 0:
        raise ValueError("--temperature must be positive.")

    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    groups = load_example_groups(args.csv_path)
    model, release_config, model_dir = load_pretrained(args.model_dir, device=device)
    tokens_per_step = validate_release_config(release_config)
    canvas_size = release_config["canvas"]["canvas_size"]

    ground_truth = torch.stack(
        [encode_image_group(group, EXAMPLE_SEQUENCE_STEPS) for _, group in groups]
    )
    input_tokens = EXAMPLE_SEQUENCE_STEPS * tokens_per_step
    target_tokens = COMPLETION_STEPS * tokens_per_step
    predicted = generate(
        model,
        ground_truth[:, :input_tokens].to(device),
        target_tokens,
        args.temperature,
    )

    input_render_data = [
        decode_tokens_to_render_data(tokens) for tokens in ground_truth
    ]
    completion_render_data = [
        decode_tokens_to_render_data(tokens) for tokens in predicted
    ]
    for (image_name, _), render_data in zip(groups, input_render_data, strict=True):
        if len(render_data) != EXAMPLE_SEQUENCE_STEPS:
            raise RuntimeError(
                f"Input sequence {image_name!r} decoded to {len(render_data)} steps; "
                f"expected {EXAMPLE_SEQUENCE_STEPS}."
            )
    for (image_name, _), render_data in zip(groups, completion_render_data, strict=True):
        if len(render_data) != COMPLETION_STEPS:
            raise RuntimeError(
                f"Model completion for {image_name!r} decoded to {len(render_data)} steps; "
                f"expected {COMPLETION_STEPS}. Try a different seed or temperature."
            )

    figure, axes = plt.subplots(
        EXAMPLE_COUNT,
        2,
        figsize=(7, 3.5 * EXAMPLE_COUNT),
        squeeze=False,
    )
    for row_index, (image_name, _) in enumerate(groups):
        render_single_image(input_render_data[row_index], axes[row_index, 0], canvas_size)
        render_single_image(
            completion_render_data[row_index], axes[row_index, 1], canvas_size
        )
        axes[row_index, 0].set_ylabel(f"Sequence {image_name}", rotation=90, labelpad=12)
        if row_index == 0:
            axes[row_index, 0].set_title("Input: 11 true steps")
            axes[row_index, 1].set_title("Model completion: 144 steps")

    figure.suptitle("Primitive Operation Painter example inference", y=0.995)
    figure.tight_layout()
    output_path = args.output_path.expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    print(f"Model package: {model_dir}")
    print(f"Example CSV: {args.csv_path.expanduser().resolve()}")
    print(f"Saved comparison: {output_path.resolve()}")


if __name__ == "__main__":
    main()
