# Models

Four ONNX models are downloaded by `scripts/fetch-models.sh` into `public/models/`
(gitignored — not committed). ONNX Runtime Web's WASM/JS assets are self-hosted
into `public/ort/` from `node_modules/onnxruntime-web/dist` (also gitignored).

Contracts below were captured by running `npm run probe` (`scripts/probe-models.ts`)
against the actual downloaded files. Later pipeline tasks depend on these
*contracts* (input/output shapes and element counts), not on the exact
tensor/output names.

## Source URLs

- Detector models (`open-image-models` release `assets`):
  - `https://github.com/ankandrew/open-image-models/releases/download/assets/yolo-v9-t-384-license-plates-end2end.onnx`
  - `https://github.com/ankandrew/open-image-models/releases/download/assets/yolo-v9-t-512-license-plates-end2end.onnx`
- OCR models (`cnn-ocr-lp` release `arg-plates`):
  - `https://github.com/ankandrew/cnn-ocr-lp/releases/download/arg-plates/cct_xs_v2_global.onnx`
  - `https://github.com/ankandrew/cnn-ocr-lp/releases/download/arg-plates/cct_s_v2_global.onnx`

## Downloaded file sizes (`npm run fetch-models`)

```
7771218  public/models/yolo-v9-t-384-license-plates-end2end.onnx   (≈7.8 MB)
7799480  public/models/yolo-v9-t-512-license-plates-end2end.onnx   (≈7.8 MB)
3344292  public/models/cct_xs_v2_global.onnx                       (≈3.3 MB)
5262230  public/models/cct_s_v2_global.onnx                        (≈5.3 MB)
```

All four passed the fetch script's >1MB sanity check, and `public/ort/` contains
the copied `onnxruntime-web` `.wasm`/`.mjs` assets (self-hosted, no CDN).

## Probe output (`npm run probe`)

Verbatim output from running the probe against the downloaded models, fed with
zero-filled tensors of the shape each model expects:

```
=== public/models/yolo-v9-t-384-license-plates-end2end.onnx
inputs : images
outputs: output0
output output0: dims=[1,7] type=float32

=== public/models/yolo-v9-t-512-license-plates-end2end.onnx
inputs : images
outputs: output0
output output0: dims=[0,7] type=float32

=== public/models/cct_xs_v2_global.onnx
inputs : input
outputs: plate, region
output plate: dims=[1,10,37] type=float32
output region: dims=[1,66] type=float32

=== public/models/cct_s_v2_global.onnx
inputs : input
outputs: plate, region
output plate: dims=[1,10,37] type=float32
output region: dims=[1,66] type=float32

All model contracts verified.
```

## Contract summary

### Detector: `yolo-v9-t-384-license-plates-end2end.onnx`, `yolo-v9-t-512-license-plates-end2end.onnx`

- Input: `images`, float32, `[1, 3, size, size]` (size = 384 or 512), CHW, presumably normalized RGB.
- Output: `output0`, float32, `[N, 7]` where N is the number of detected rows
  (0 on a blank/zero image — as seen above for the 512 model; the 384 model
  returned 1 spurious row of `7` values on an all-zero input, which is expected
  noise from a zero-filled probe input, not a real detection).
- The end2end YOLO export bakes NMS in, so each output row is presumed to already
  be a single detection: `[x1, y1, x2, y2, score, class, ...]`-style 7 columns
  (later tasks must verify exact column semantics against real images).

### OCR: `cct_xs_v2_global.onnx`, `cct_s_v2_global.onnx`

- Input: `input`, uint8, `[1, 64, 128, 3]` (NHWC, likely RGB crop of a plate).
- Outputs:
  - `plate`: float32, `[1, 10, 37]` — 10 character slots x 37-symbol alphabet
    (370 elements total), i.e. per-slot logits/probabilities over the alphabet.
  - `region`: float32, `[1, 66]` — 66-way region/country classification head.
- Both the `xs` (small/fast) and `s` (larger) variants expose the identical
  input/output contract; they differ only in internal capacity/accuracy.
