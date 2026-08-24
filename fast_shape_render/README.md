# fast_shape_render

`fast_shape_render` is the GPU image-to-sequence converter used by Primitive
Operation Painter. It recursively reads common image formats, resizes each
image to 256×256, fits a sequence of rectangles and ellipses on the GPU, and
writes CSV files for the Python training and visualization programs.

## Requirements

- Stable Rust toolchain and Cargo.
- A hardware GPU supported by WGPU: DirectX 12 on Windows, Vulkan on Linux, or
  Metal on macOS.

## Run locally

Set the source image directory, then run the release build from this folder:

```powershell
$env:SHAPE_RENDERER_INPUT_DIR = 'D:\images\faces_256'
cargo run --release
```

`SHAPE_RENDERER_INPUT_DIR` is required; the converter never uses a hard-coded
personal input location.

By default, output is written to the parent project's `data/output_256`
directory. The output is organized as `v1/` through `v10/`, with files named
`data_part_*.csv` and the following header:

```text
image_name,cx,cy,w,h,shape_type,theta,r,g,b
```

## Optional environment variables

- `SHAPE_RENDERER_OUTPUT_DIR`: override the output directory.
- `SHAPE_RENDERER_NUM_VERSIONS`: number of generated variants; default `10`.
- `SHAPE_RENDERER_MAX_OUTPUT_STEPS`: optionally retain a fixed number of CSV
  rows per image, including the background row. The GPU still performs all
  275 search steps. If an image has too few accepted primitives to reach this
  count, conversion stops with an error rather than writing a partial group.
- `SHAPE_RENDERER_IMAGES_PER_GPU_BATCH`: per-GPU input batch size; default
  `1000`.
- `SHAPE_RENDERER_HISTORY_READBACK_IMAGES`: images per GPU-to-CPU history
  readback chunk; default `128`.

The normal generated CSV directory is intentionally ignored by Git. Use it
directly with the parent project's `train_gpt_pretrain.py` or `visualize.py`.

## Bundled short example

The parent project includes a compact six-sequence demonstration at
`../example/sequences/v1/data_part_1.csv`. It was generated from six local images
with `SHAPE_RENDERER_NUM_VERSIONS=1` and
`SHAPE_RENDERER_MAX_OUTPUT_STEPS=11`, then the source images were removed.
Each group contains one background row and ten accepted primitives. It uses
the same CSV schema as training data, but it is intentionally too short for
the 144-step trainer and is only for `../example.py` inference visualization.
That script uses all eleven rows as its prompt, then samples the remaining 133
steps to render a 144-step completion beside the short input.
