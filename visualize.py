"""Visualize local inference from a prepared Primitive Operation Painter model.

This script intentionally requires explicit model and data locations.  It does
not select a local checkpoint, create output directories during import, or use
personal absolute-path fallbacks.
"""

from __future__ import annotations

import argparse
import math
import os
from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from pretrained import load_pretrained
from token_layout import (
    ANGLE_BINS_PER_DEGREE,
    SIZE_BINS_PER_PIXEL,
    TOKEN_LAYOUT,
    XY_BINS_PER_PIXEL,
)


CSV_DTYPE = {
    "image_name": "string",
    "cx": "float32",
    "cy": "float32",
    "w": "float32",
    "h": "float32",
    "shape_type": "int8",
    "theta": "float32",
    "r": "uint8",
    "g": "uint8",
    "b": "uint8",
}


# Each pair is (a, b) and is evaluated as a * one-based shape-step index + b.
# top_k is floored because categorical sampling requires an integer count.
GPT_SAMPLING_CONFIG = {
    "cx": {"temperature": (0.003, 0.5), "top_k": (0.0, 50)},
    "cy": {"temperature": (0.0, 0.3), "top_k": (0.0, 30)},
    "theta": {"temperature": (0.003, 1.0), "top_k": (0.0, 75)},
    "w": {"temperature": (0.0, 1.0), "top_k": (0.0, 50)},
    "h": {"temperature": (0.0, 0.3), "top_k": (0.0, 30)},
    "shape_type": {"temperature": (0.0, 0.3), "top_k": (0.0, 50)},
    "r": {"temperature": (0.003, 0.7), "top_k": (0.5, 50)},
    "g": {"temperature": (0.0, 0.3), "top_k": (0.5, 30)},
    "b": {"temperature": (0.0, 0.3), "top_k": (0.5, 30)},
}
TOKEN_FIELD_ORDER = ("cx", "cy", "theta", "w", "h", "shape_type", "r", "g", "b")
TOKEN_LAYOUT_FIELD_NAMES = {
    "cx": "x",
    "cy": "y",
    "theta": "angle",
    "w": "width",
    "h": "height",
    "shape_type": "shape",
    "r": "red",
    "g": "green",
    "b": "blue",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render local primitive-operation predictions from a release package."
    )
    parser.add_argument(
        "--model-dir",
        required=True,
        type=Path,
        help="Local directory containing config.json and model.safetensors.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Root containing v1/data_part_1.csv; defaults to ANIME_PAINTER_DATA_DIR.",
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=None,
        help="Explicit CSV path, overriding --data-dir.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("visualizations"))
    parser.add_argument("--num-tests", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def resolve_csv_path(args: argparse.Namespace) -> Path:
    if args.csv_path is not None:
        candidate = args.csv_path.expanduser()
    else:
        data_dir = args.data_dir
        if data_dir is None:
            configured = os.environ.get("ANIME_PAINTER_DATA_DIR")
            if not configured:
                raise ValueError(
                    "Supply --data-dir or --csv-path, or set ANIME_PAINTER_DATA_DIR. "
                    "No personal-path fallback is available."
                )
            data_dir = Path(configured)
        candidate = data_dir.expanduser() / "v1" / "data_part_1.csv"
    if not candidate.is_file():
        raise FileNotFoundError(f"Input CSV not found: {candidate}")
    return candidate.resolve()


def iter_complete_image_groups(csv_path: Path, chunk_rows: int):
    """Yield complete contiguous image groups without loading the full CSV."""
    carry = None
    for chunk in pd.read_csv(csv_path, dtype=CSV_DTYPE, engine="c", chunksize=chunk_rows):
        if carry is not None:
            chunk = pd.concat([carry, chunk], ignore_index=True)
        if chunk.empty:
            continue
        last_image_name = chunk["image_name"].iloc[-1]
        complete_rows = chunk[chunk["image_name"].ne(last_image_name)]
        carry = chunk[chunk["image_name"].eq(last_image_name)]
        if not complete_rows.empty:
            yield from complete_rows.groupby("image_name", sort=False)
    if carry is not None and not carry.empty:
        yield from carry.groupby("image_name", sort=False)


def encode_image_group(group_df: pd.DataFrame, target_steps: int) -> torch.Tensor:
    """Encode the leading primitive operations of one ground-truth image."""
    sequence: list[int] = []
    for row in group_df.iloc[:target_steps].itertuples(index=False):
        cx, cy, width, height = row.cx, row.cy, row.w, row.h
        theta_degrees = (row.theta * 180.0 / math.pi) % 180.0
        if theta_degrees >= 90.0:
            theta_degrees -= 90.0
            width, height = height, width
        sequence.extend(
            [
                int(np.clip(cx * XY_BINS_PER_PIXEL, 0, TOKEN_LAYOUT.x_bins - 1))
                + TOKEN_LAYOUT.x_offset,
                int(np.clip(cy * XY_BINS_PER_PIXEL, 0, TOKEN_LAYOUT.y_bins - 1))
                + TOKEN_LAYOUT.y_offset,
                int(
                    np.clip(
                        theta_degrees * ANGLE_BINS_PER_DEGREE,
                        0,
                        TOKEN_LAYOUT.angle_bins - 1,
                    )
                )
                + TOKEN_LAYOUT.angle_offset,
                int(
                    np.clip(width * SIZE_BINS_PER_PIXEL, 0, TOKEN_LAYOUT.width_bins - 1)
                )
                + TOKEN_LAYOUT.width_offset,
                int(
                    np.clip(height * SIZE_BINS_PER_PIXEL, 0, TOKEN_LAYOUT.height_bins - 1)
                )
                + TOKEN_LAYOUT.height_offset,
                int(np.clip(row.shape_type + 1, 0, TOKEN_LAYOUT.shape_bins - 1))
                + TOKEN_LAYOUT.shape_offset,
                int(np.clip(row.r // 2, 0, TOKEN_LAYOUT.color_bins - 1))
                + TOKEN_LAYOUT.red_offset,
                int(np.clip(row.g // 2, 0, TOKEN_LAYOUT.color_bins - 1))
                + TOKEN_LAYOUT.green_offset,
                int(np.clip(row.b // 2, 0, TOKEN_LAYOUT.color_bins - 1))
                + TOKEN_LAYOUT.blue_offset,
            ]
        )
    return torch.tensor(sequence, dtype=torch.long)


def load_test_samples(csv_path: Path, num_samples: int, target_steps: int) -> list[torch.Tensor]:
    samples: list[torch.Tensor] = []
    for _, group_df in iter_complete_image_groups(csv_path, chunk_rows=2_048):
        if len(group_df) < target_steps:
            continue
        samples.append(encode_image_group(group_df, target_steps))
        if len(samples) == num_samples:
            break
    if len(samples) < num_samples:
        raise RuntimeError(
            f"CSV contains only {len(samples)} complete samples; {num_samples} are required."
        )
    return samples


def decode_tokens_to_render_data(tokens: torch.Tensor) -> np.ndarray:
    rows: list[list[float]] = []
    for offset in range(0, tokens.numel() - 8, 9):
        values = tokens[offset : offset + 9].cpu().tolist()
        if values[0] >= TOKEN_LAYOUT.special_offset:
            break
        rows.append(
            [
                (values[0] - TOKEN_LAYOUT.x_offset) / XY_BINS_PER_PIXEL,
                (values[1] - TOKEN_LAYOUT.y_offset) / XY_BINS_PER_PIXEL,
                (values[3] - TOKEN_LAYOUT.width_offset) / SIZE_BINS_PER_PIXEL,
                (values[4] - TOKEN_LAYOUT.height_offset) / SIZE_BINS_PER_PIXEL,
                values[5] - TOKEN_LAYOUT.shape_offset - 1,
                ((values[2] - TOKEN_LAYOUT.angle_offset) / ANGLE_BINS_PER_DEGREE)
                * math.pi
                / 180.0,
                (values[6] - TOKEN_LAYOUT.red_offset) * 2,
                (values[7] - TOKEN_LAYOUT.green_offset) * 2,
                (values[8] - TOKEN_LAYOUT.blue_offset) * 2,
            ]
        )
    return np.asarray(rows, dtype=np.float32)


def render_single_image(
    render_data: np.ndarray,
    axis,
    canvas_size: int,
    final_primitive_alpha: float = 1.0,
) -> None:
    """Render one operation sequence without patch outlines.

    ``final_primitive_alpha`` is used by the README animation to fade in the
    newest complete primitive.  Normal visualizations retain the established
    50% primitive opacity by using its default of one.
    """
    axis.set_xlim(0, canvas_size)
    axis.set_ylim(canvas_size, 0)
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_aspect("equal")
    for spine in axis.spines.values():
        spine.set_visible(False)
    if len(render_data) == 0:
        return

    background = render_data[0]
    axis.set_facecolor(tuple(np.clip(background[6:9] / 255.0, 0.0, 1.0)))
    from matplotlib.patches import Ellipse, Rectangle

    final_primitive_alpha = float(np.clip(final_primitive_alpha, 0.0, 1.0))
    primitive_rows = render_data[1:]
    for primitive_index, (
        cx,
        cy,
        width,
        height,
        shape_type,
        theta,
        red,
        green,
        blue,
    ) in enumerate(primitive_rows):
        alpha = 0.5
        if primitive_index == len(primitive_rows) - 1:
            alpha *= final_primitive_alpha
        colour = (*np.clip([red, green, blue], 0.0, 255.0) / 255.0, alpha)
        if int(shape_type) == 1:
            primitive = Ellipse(
                (cx, cy), max(0.1, width), max(0.1, height),
                angle=np.degrees(theta), facecolor=colour,
                edgecolor="none", linewidth=0,
            )
        else:
            primitive = Rectangle(
                (cx - width / 2, cy - height / 2), max(0.1, width), max(0.1, height),
                angle=np.degrees(theta), rotation_point="center", facecolor=colour,
                edgecolor="none", linewidth=0,
            )
        axis.add_patch(primitive)


def sampling_parameters_for_token_position(token_position: int) -> tuple[str, int, float, int, slice]:
    """Return field-aware sampling values for the next absolute token position."""
    if token_position < 0:
        raise ValueError("token_position must be non-negative")

    field_name = TOKEN_FIELD_ORDER[token_position % len(TOKEN_FIELD_ORDER)]
    shape_step = token_position // len(TOKEN_FIELD_ORDER) + 1
    config = GPT_SAMPLING_CONFIG[field_name]
    temperature_a, temperature_b = config["temperature"]
    top_k_a, top_k_b = config["top_k"]
    temperature = temperature_a * shape_step + temperature_b
    raw_top_k = top_k_a * shape_step + top_k_b
    token_slice = TOKEN_LAYOUT.field_slice(TOKEN_LAYOUT_FIELD_NAMES[field_name])
    field_vocab_size = token_slice.stop - token_slice.start
    top_k = min(field_vocab_size, max(1, math.floor(raw_top_k)))
    return field_name, shape_step, temperature, top_k, token_slice


def sample_next_token(logits: torch.Tensor, next_token_position: int) -> torch.Tensor:
    """Sample one token using the scheduled field temperature and top-k."""
    _, _, temperature, top_k, token_slice = sampling_parameters_for_token_position(
        next_token_position
    )
    field_logits = logits[:, token_slice]
    top_logits, top_indices = torch.topk(field_logits, k=top_k, dim=-1)
    if temperature <= 0.0:
        selected_relative_index = top_indices[:, :1]
    else:
        probabilities = F.softmax(top_logits / temperature, dim=-1)
        selected_top_index = torch.multinomial(probabilities, num_samples=1)
        selected_relative_index = top_indices.gather(1, selected_top_index)
    return selected_relative_index + token_slice.start


@torch.inference_mode()
def generate(
    model,
    prompt: torch.Tensor,
    target_tokens: int,
    on_completed_shape: Callable[[torch.Tensor], None] | None = None,
) -> torch.Tensor:
    """Autoregressively sample tokens and optionally report completed shape steps.

    ``on_completed_shape`` receives the full current token tensor after every
    completed nine-token drawing operation generated beyond the prompt. The
    callback must copy the tensor itself if it needs to retain a snapshot.
    """
    if prompt.size(1) > target_tokens:
        raise ValueError("prompt length cannot exceed target_tokens")
    logits, cache = model(prompt, use_cache=True)
    current = prompt
    while current.size(1) < target_tokens:
        next_tokens = sample_next_token(logits[:, -1, :], current.size(1))
        current = torch.cat([current, next_tokens], dim=1)
        if on_completed_shape is not None and current.size(1) % len(TOKEN_FIELD_ORDER) == 0:
            on_completed_shape(current)
        if current.size(1) < target_tokens:
            logits, cache = model(next_tokens, past_key_values=cache, use_cache=True)
    return current


def main() -> None:
    args = parse_args()
    if args.num_tests <= 0 or args.batch_size <= 0:
        raise ValueError("--num-tests and --batch-size must be positive")
    torch.manual_seed(args.seed)

    csv_path = resolve_csv_path(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, release_config, model_dir = load_pretrained(args.model_dir, device=device)
    sequence = release_config["sequence"]
    canvas = release_config["canvas"]
    if release_config["token_layout"]["version"] != TOKEN_LAYOUT.version:
        raise ValueError("Release token layout does not match this source checkout")

    context_steps = sequence["context_steps"]
    prefix_steps = sequence["prefix_steps"]
    tokens_per_step = sequence["tokens_per_step"]
    samples = load_test_samples(csv_path, args.num_tests, context_steps)
    target_tokens = context_steps * tokens_per_step

    args.output_dir.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(
        args.num_tests,
        1 + args.batch_size,
        figsize=(3 * (1 + args.batch_size), 3 * args.num_tests),
    )
    axes = np.atleast_2d(axes)

    for row_index, ground_truth in enumerate(samples):
        prompt = ground_truth[: prefix_steps * tokens_per_step].unsqueeze(0)
        prompt = prompt.repeat(args.batch_size, 1).to(device)
        predicted = generate(model, prompt, target_tokens)
        render_single_image(
            decode_tokens_to_render_data(ground_truth[:target_tokens]),
            axes[row_index, 0],
            canvas["canvas_size"],
        )
        if row_index == 0:
            axes[row_index, 0].set_title("Ground truth")
        for batch_index in range(args.batch_size):
            render_single_image(
                decode_tokens_to_render_data(predicted[batch_index]),
                axes[row_index, batch_index + 1],
                canvas["canvas_size"],
            )
            if row_index == 0:
                axes[row_index, batch_index + 1].set_title(f"Prediction {batch_index + 1}")

    figure.tight_layout()
    output_path = args.output_dir / "full_sequence_batch_inference_test.png"
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    print(f"Model package: {model_dir}")
    print(f"CSV input: {csv_path}")
    print(f"Saved visualization: {output_path.resolve()}")


if __name__ == "__main__":
    main()
