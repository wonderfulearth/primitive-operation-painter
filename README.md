# Primitive Operation Painter

Primitive Operation Painter is a custom PyTorch autoregressive model for
predicting sequences of primitive drawing operations. Each drawing step is
encoded as nine discrete tokens: x, y, angle, width, height, shape type, and
RGB colour.

This repository contains only the files required to use and continue training
the released **144-step** model: the model definition, token layout,
full-sequence data encoder, training entry point, and visualization script. It
deliberately does **not** contain training data, model weights, resumable
checkpoints, historical migration scripts, or generated results.  The only
small bundled data artifact is the six-sequence inference example described
below.

## License and data boundary

The code and the released EMA inference weights are intended to be licensed
under [MIT](LICENSE). Training data is not included or redistributed. Anyone
using this project is responsible for confirming they have the necessary
rights for their own input and training data.

## Setup

Use Python 3.10 or later and install a PyTorch build suitable for your system,
then install the remaining dependencies:

```powershell
python -m pip install -r requirements.txt
```

Set the training-data root with `ANIME_PAINTER_DATA_DIR`, or generate it with
the bundled `fast_shape_render` converter. Its default output directory is
`data/output_256`, which is also the Python programs' default data location.
The data root must contain versioned CSV directories such as `v1/*.csv`.

```powershell
$env:ANIME_PAINTER_DATA_DIR = 'D:\datasets\output_256'
python train_gpt_pretrain.py
```

## Convert an image dataset to training sequences

`fast_shape_render` is a separate Rust/WGPU GPU converter. It resizes each
input image to 256×256, approximates it with primitive drawing operations, and
writes the CSV sequence layout consumed by the Python dataset loader.

Install the Rust toolchain, then set the input directory before running it.
The converter requires a hardware GPU: DirectX 12 on Windows, Vulkan on Linux,
and Metal on macOS. Its default output is `data/output_256`; no personal path
is embedded in the converter.

```powershell
$env:SHAPE_RENDERER_INPUT_DIR = 'D:\images\faces_256'

Push-Location fast_shape_render
cargo run --release
Pop-Location
```

Optional environment variables:

- `SHAPE_RENDERER_OUTPUT_DIR` — use a different CSV output root.
- `SHAPE_RENDERER_NUM_VERSIONS` — number of output variants; defaults to 10.
- `SHAPE_RENDERER_MAX_OUTPUT_STEPS` — optionally retain a fixed number of
  CSV rows per image, including its background row. When set, conversion fails
  if an image has too few accepted primitives to fill the requested sequence.
- `SHAPE_RENDERER_IMAGES_PER_GPU_BATCH` and
  `SHAPE_RENDERER_HISTORY_READBACK_IMAGES` — reduce these if GPU memory is
  limited.

After conversion, either use the default `data/output_256` location or point
training and visualization to a custom output root with
`ANIME_PAINTER_DATA_DIR` or `--data-dir`.

## Six-sequence inference example

The repository includes six compact example sequences under
`example/sequences/v1/data_part_1.csv`. Each one has eleven
steps: a background operation followed by ten primitives. They are a quick
inference demonstration, not training data: the released trainer requires
complete 144-step sequences.

Point `example.py` at a local model package that contains `config.json` and
`model.safetensors`. It renders a two-column, six-row PNG. The left side shows
the eleven input operations; the right side starts from those same eleven
operations and samples the remaining 133 steps to make a complete 144-step
sequence.

```powershell
python example.py `
  --model-dir path\to\primitive-operation-painter-weight
```

By default the result is written to
`example/example_inference.png`. Use `--csv-path`, `--output-path`, `--seed`,
or `--device` to override the example inputs and inference settings.

Both `example.py` and `visualize.py` use the same field-aware sampling
schedule. For each generated token, temperature and `top_k` are evaluated as
`a * step + b` from `GPT_SAMPLING_CONFIG`; `step` is the one-based drawing
operation index, including the background operation. `top_k` is rounded down
to an integer and clipped to the current field's valid token range.

## Local visualization

`visualize.py` requires a local model package and a CSV data root. It never
falls back to a personal absolute path or silently selects a checkpoint.

```powershell
python visualize.py `
  --model-dir path\to\primitive-operation-painter-weight `
  --data-dir D:\datasets\output_256 `
  --num-tests 8
```

The planned public code repository name is `primitive-operation-painter`; the
planned model repository name is `primitive-operation-painter-weight`. This
local preparation does not create or upload either remote repository.

## Continue training the released weight

Place or download the model package in a local directory containing
`config.json` and `model.safetensors`, then point the trainer at your own CSV
dataset and that directory:

```powershell
$env:ANIME_PAINTER_DATA_DIR = 'D:\datasets\output_256'
python train_gpt_pretrain.py `
  --initial-model-dir path\to\primitive-operation-painter-weight
```

The trainer validates that the package is the compatible 144-step model before
loading its EMA weights. New resumable training checkpoints are written to
`checkpoints_gpt_fullseq_144ctx_256reso/`. On later runs, omit
`--initial-model-dir` to resume from the newest local training checkpoint.
