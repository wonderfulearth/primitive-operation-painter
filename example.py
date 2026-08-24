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
from PIL import Image

from pretrained import load_pretrained
from token_layout import TOKEN_LAYOUT
from visualize import (
    CSV_DTYPE,
    GPT_SAMPLING_CONFIG,
    decode_tokens_to_render_data,
    encode_image_group,
    generate,
    render_single_image,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = PROJECT_ROOT / "model"
DEFAULT_CSV_PATH = PROJECT_ROOT / "example" / "sequences" / "v1" / "data_part_1.csv"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "example" / "example_inference.png"
MODEL_CONFIG_FILE = "config.json"
MODEL_WEIGHTS_FILE = "model.safetensors"
EXAMPLE_SEQUENCE_STEPS = 11
COMPLETION_STEPS = 144
EXAMPLE_COUNT = 6
GIF_INITIAL_FRAME_DURATION_MS = 3_000
GIF_MAX_BYTES = 10 * 1024 * 1024
GIF_ENCODING_LEVELS = ((720, 128), (640, 96), (560, 64))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render six ground-truth/prediction pairs from the bundled "
            "11-step Primitive Operation Painter example sequences."
        )
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help=(
            "Local model package directory containing config.json and "
            f"model.safetensors (default: {DEFAULT_MODEL_DIR})."
        ),
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
        "--gif-output",
        type=Path,
        default=None,
        help=(
            "Optional autoregressive GIF path. When supplied, keep the input frame "
            "for three seconds and add one frame for every predicted shape."
        ),
    )
    parser.add_argument(
        "--gif-fps",
        type=int,
        default=10,
        help="Frames per second after the three-second input hold (default: 10).",
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


def require_model_package(model_dir: Path) -> Path:
    """Validate a local package before the shared loader can attempt any download."""
    model_dir = model_dir.expanduser()
    expected_files = (MODEL_CONFIG_FILE, MODEL_WEIGHTS_FILE)
    if not model_dir.is_dir():
        raise FileNotFoundError(
            f"Model package directory not found: {model_dir}.\n"
            f"Place the downloaded model package at: {DEFAULT_MODEL_DIR}\n"
            "It must contain config.json and model.safetensors, or supply "
            "another local package directory with --model-dir."
        )
    missing_files = [name for name in expected_files if not (model_dir / name).is_file()]
    if missing_files:
        raise FileNotFoundError(
            f"Model package is incomplete: {model_dir}. Missing: {', '.join(missing_files)}.\n"
            "Expected files: config.json and model.safetensors."
        )
    return model_dir.resolve()


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
    if len(GPT_SAMPLING_CONFIG) != 9:
        raise RuntimeError("The shared GPT sampling configuration must cover all nine fields.")
    return sequence["tokens_per_step"]


def render_animation_frame(
    groups: list[tuple[str, pd.DataFrame]],
    render_data_by_sample: list[np.ndarray],
    step_count: int,
    canvas_size: int,
) -> Image.Image:
    """Render one 3x2 grid frame for a shared autoregressive step count."""
    figure, axes = plt.subplots(2, 3, figsize=(7.2, 5.1), dpi=100, squeeze=False)
    for axis, (image_name, _), render_data in zip(
        axes.flat, groups, render_data_by_sample, strict=True
    ):
        render_single_image(render_data, axis, canvas_size)
        axis.set_title(f"Example {image_name}", fontsize=9)

    if step_count == EXAMPLE_SEQUENCE_STEPS:
        title = "Input: 11 true steps"
    else:
        title = f"Autoregressive completion: {step_count} / {COMPLETION_STEPS} steps"
    figure.suptitle(title, y=0.995)
    figure.tight_layout(rect=(0, 0, 1, 0.96), pad=0.35, w_pad=0.25, h_pad=0.5)
    figure.canvas.draw()
    rgba = np.asarray(figure.canvas.buffer_rgba()).copy()
    plt.close(figure)
    return Image.fromarray(rgba[:, :, :3], mode="RGB")


def render_autoregressive_frames(
    groups: list[tuple[str, pd.DataFrame]],
    token_states: list[torch.Tensor],
    tokens_per_step: int,
    canvas_size: int,
) -> list[Image.Image]:
    """Decode and render the input state plus one frame for each completed shape."""
    expected_frame_count = 1 + COMPLETION_STEPS - EXAMPLE_SEQUENCE_STEPS
    if len(token_states) != expected_frame_count:
        raise RuntimeError(
            f"Expected {expected_frame_count} animation states, got {len(token_states)}."
        )

    frames: list[Image.Image] = []
    for state in token_states:
        if state.ndim != 2 or state.size(0) != EXAMPLE_COUNT:
            raise RuntimeError(f"Invalid animation state shape: {tuple(state.shape)}")
        step_count, remainder = divmod(state.size(1), tokens_per_step)
        if remainder or not EXAMPLE_SEQUENCE_STEPS <= step_count <= COMPLETION_STEPS:
            raise RuntimeError(
                f"Animation state has invalid token length: {state.size(1)} tokens."
            )
        render_data = [decode_tokens_to_render_data(tokens) for tokens in state]
        if any(len(data) != step_count for data in render_data):
            raise RuntimeError(
                f"Animation state for step {step_count} could not be decoded completely."
            )
        frames.append(render_animation_frame(groups, render_data, step_count, canvas_size))
    return frames


def write_optimized_gif(
    frames: list[Image.Image], output_path: Path, fps: int
) -> tuple[int, int, int]:
    """Write a README-friendly GIF and return (width, palette_size, byte_size)."""
    if fps <= 0:
        raise ValueError("--gif-fps must be positive.")
    if len(frames) != 1 + COMPLETION_STEPS - EXAMPLE_SEQUENCE_STEPS:
        raise ValueError("GIF export received an unexpected frame count.")

    output_path = output_path.expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    durations = [GIF_INITIAL_FRAME_DURATION_MS] + [round(1_000 / fps)] * (len(frames) - 1)
    for max_width, palette_size in GIF_ENCODING_LEVELS:
        scale = min(1.0, max_width / frames[0].width)
        output_size = (
            max(1, round(frames[0].width * scale)),
            max(1, round(frames[0].height * scale)),
        )
        encoded_frames = []
        for frame in frames:
            resized = frame.resize(output_size, Image.Resampling.LANCZOS)
            encoded_frames.append(
                resized.quantize(colors=palette_size, method=Image.Quantize.MEDIANCUT)
            )
        encoded_frames[0].save(
            output_path,
            format="GIF",
            save_all=True,
            append_images=encoded_frames[1:],
            duration=durations,
            loop=0,
            disposal=2,
            optimize=True,
        )
        byte_size = output_path.stat().st_size
        if byte_size <= GIF_MAX_BYTES:
            return output_size[0], palette_size, byte_size

    output_path.unlink(missing_ok=True)
    raise RuntimeError(
        f"GIF could not be compressed below {GIF_MAX_BYTES / 1024 / 1024:.0f} MiB. "
        "Reduce the animation dimensions or palette settings."
    )


def main() -> None:
    args = parse_args()
    if args.gif_fps <= 0:
        raise ValueError("--gif-fps must be positive.")

    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    try:
        model_dir = require_model_package(args.model_dir)
    except FileNotFoundError as exc:
        raise SystemExit(f"❌ {exc}") from None
    groups = load_example_groups(args.csv_path)
    model, release_config, model_dir = load_pretrained(model_dir, device=device)
    tokens_per_step = validate_release_config(release_config)
    canvas_size = release_config["canvas"]["canvas_size"]

    ground_truth = torch.stack(
        [encode_image_group(group, EXAMPLE_SEQUENCE_STEPS) for _, group in groups]
    )
    input_tokens = EXAMPLE_SEQUENCE_STEPS * tokens_per_step
    target_tokens = COMPLETION_STEPS * tokens_per_step
    animation_states: list[torch.Tensor] | None = None
    on_completed_shape = None
    if args.gif_output is not None:
        animation_states = [ground_truth[:, :input_tokens].detach().cpu().clone()]

        def record_completed_shape(tokens: torch.Tensor) -> None:
            animation_states.append(tokens.detach().cpu().clone())

        on_completed_shape = record_completed_shape
    predicted = generate(
        model,
        ground_truth[:, :input_tokens].to(device),
        target_tokens,
        on_completed_shape=on_completed_shape,
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
    if animation_states is not None:
        animation_frames = render_autoregressive_frames(
            groups, animation_states, tokens_per_step, canvas_size
        )
        gif_width, palette_size, gif_byte_size = write_optimized_gif(
            animation_frames, args.gif_output, args.gif_fps
        )
        print(
            f"Saved GIF: {args.gif_output.expanduser().resolve()} | "
            f"frames={len(animation_frames)} | width={gif_width}px | "
            f"palette={palette_size} | size={gif_byte_size / 1024 / 1024:.2f} MiB"
        )
    print(f"Model package: {model_dir}")
    print(f"Example CSV: {args.csv_path.expanduser().resolve()}")
    print(f"Saved comparison: {output_path.resolve()}")


if __name__ == "__main__":
    main()
