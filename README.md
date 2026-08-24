# Primitive Operation Painter

Primitive Operation Painter is a custom PyTorch autoregressive model for
predicting sequences of primitive drawing operations. Each drawing step is
encoded as nine discrete tokens: x, y, angle, width, height, shape type, and
RGB colour.

This repository contains only the files required to use and continue training
the released **144-step** model: the model definition, token layout,
full-sequence data encoder, training entry point, and visualization script. It
deliberately does **not** contain training data, model weights, resumable
checkpoints, historical migration scripts, or generated results.

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

Set the training-data root with `ANIME_PAINTER_DATA_DIR`, or place it at the
project-relative path `../High_reso_dataset/fast_shape_renderer/output_256`.
The data root must contain the expected versioned CSV directories such as
`v1/*.csv`.

```powershell
$env:ANIME_PAINTER_DATA_DIR = 'D:\datasets\output_256'
python train_gpt_pretrain.py
```

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
