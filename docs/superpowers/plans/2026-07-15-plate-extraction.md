# Plate Extraction PWA Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A PWA that reads German license plates from a phone photo fully on-device, validated against the 24 sample photos in `attachments/` (target ≥ 31/34 labeled plates found and read exactly; all detected plates have equal priority — no "blocked car" guessing).

**Architecture:** One pure-TypeScript pipeline (`src/pipeline/`) shared verbatim between the browser app and a Node eval CLI. Two ONNX models run it: a YOLOv9-tiny plate detector (384px letterbox input, built-in NMS) and a CCT-XS v2 plate OCR (64×128 RGB input, 10 slots × 37-char alphabet + country region head). Platform code only decodes images (browser canvas / Node sharp) and adapts ONNX Runtime sessions. Validation is a pluggable rule engine — German rules score and correct reads but never reject them.

**Tech Stack:** Vite + TypeScript (strict) + Vitest; `onnxruntime-web` (WASM backend, self-hosted) in browser; `onnxruntime-node` + `sharp` (dev-only) for the eval CLI; `tsx` to run TS scripts.

## Global Constraints

- **Privacy:** No network request may ever contain image data. All inference is local. ONNX Runtime WASM assets are self-hosted same-origin via Vite `?url` imports (never a CDN, never public/-dir module imports).
- **Gitignored (privacy + binaries):** `attachments/`, `eval/expected.json`, `eval/out/`, `public/models/`, `public/ort/`, `node_modules/`, `dist/`.
- **Never reject a read:** a failed German-rule match returns the raw read flagged `formatValid: false` — it must never be dropped (spec requirement, keeps EU plates usable).
- **Purity:** files in `src/pipeline/` must not import DOM or Node APIs. Only `src/web/` touches the DOM; only `src/node/` and `scripts/` touch Node APIs.
- **Environment:** Node v24.11.0 / npm 11 (verified present). All commands run from repo root `<repo-root>`.
- Commit at the end of every task.

## Verified model contracts (do not re-derive; verified 2026-07-15 against upstream source)

**Detector** `yolo-v9-t-384-license-plates-end2end.onnx`
(`https://github.com/ankandrew/open-image-models/releases/download/assets/`; also `-512-` variant)
- Input: `[1, 3, S, S]` float32, **RGB**, values 0–1, letterboxed with gray fill `(114,114,114)`. S = 384 (or 512). Letterbox: `r = min(S/h, S/w)`, `newW = round(w·r)`, `newH = round(h·r)`, `dw = (S-newW)/2`, `dh = (S-newH)/2`, pad left/top = `round(dw-0.1)` / `round(dh-0.1)`.
- Output: single tensor, flat rows of 7 floats `[batchIdx, x1, y1, x2, y2, classId, score]`, already NMS-suppressed, variable row count. Un-letterbox: `x_orig = (x - dw) / r`.

**OCR** `cct_xs_v2_global.onnx` (and `cct_s_v2_global.onnx`)
(`https://github.com/ankandrew/cnn-ocr-lp/releases/download/arg-plates/`)
- Input: `[1, 64, 128, 3]` **uint8 NHWC RGB**, plain stretch-resize (no aspect preservation), linear interpolation. Model normalizes internally.
- Outputs (identify by element count per batch, names vary): plate head `10 slots × 37` = **370** softmax probabilities; region head = **66** softmax probabilities.
- Alphabet: `'0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_'`, pad `'_'` (strip trailing pads).

---

### Task 1: Project scaffold

**Files:**
- Create: `package.json`, `tsconfig.json`, `vite.config.ts`, `index.html`, `src/main.ts`, `src/pipeline/smoke.test.ts`, `README.md`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: nothing.
- Produces: working `npm run dev`, `npm test`, `npm run build`; npm scripts `fetch-models`, `probe`, `eval` wired (scripts themselves come in Task 2/9).

- [ ] **Step 1: Init package and install dependencies**

```bash
npm init -y
npm pkg set type=module name=anpr version=0.1.0 private=true
npm install onnxruntime-web
npm install -D vite typescript vitest tsx onnxruntime-node sharp @types/node
npm pkg set scripts.dev="vite --host" scripts.build="vite build" scripts.preview="vite preview --host" scripts.test="vitest run" scripts.test:watch="vitest" scripts.fetch-models="bash scripts/fetch-models.sh" scripts.probe="tsx scripts/probe-models.ts" scripts.eval="tsx scripts/eval.ts"
```

- [ ] **Step 2: Write config files**

`tsconfig.json`:
```json
{
  "compilerOptions": {
    "target": "ES2022",
    "module": "ESNext",
    "moduleResolution": "bundler",
    "strict": true,
    "noUncheckedIndexedAccess": true,
    "skipLibCheck": true,
    "lib": ["ES2022", "DOM", "DOM.Iterable"],
    "types": ["vite/client", "node"],
    "noEmit": true
  },
  "include": ["src", "scripts", "vite.config.ts"]
}
```

`vite.config.ts` (COOP/COEP headers enable multithreaded WASM; dev middleware for eval data comes in Task 10):
```ts
import { defineConfig } from 'vite'

export default defineConfig({
  optimizeDeps: { exclude: ['onnxruntime-web'] },
  server: {
    headers: {
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    },
  },
})
```

`index.html`:
```html
<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Plate Reader</title>
  </head>
  <body>
    <div id="app">Plate Reader — scaffold OK</div>
    <script type="module" src="/src/main.ts"></script>
  </body>
</html>
```

`src/main.ts`:
```ts
console.log('plate-reader scaffold')
```

`src/pipeline/smoke.test.ts` (placeholder; deleted in Task 3):
```ts
import { describe, it, expect } from 'vitest'

describe('scaffold', () => {
  it('runs vitest', () => {
    expect(1 + 1).toBe(2)
  })
})
```

Append to `.gitignore` (keep existing lines):
```
eval/expected.json
eval/out/
public/models/
public/ort/
```

`README.md`:
```markdown
# ANPR — plate extraction (stage 1)

Reads German license plates from phone photos, fully on-device (PWA, ONNX in WebAssembly).
Photos never leave the device. Spec: docs/superpowers/specs/2026-07-15-plate-extraction-design.md

## Setup

    npm install
    npm run fetch-models   # downloads ONNX models + copies ORT wasm (once)
    npm test               # unit tests
    npm run eval           # accuracy eval over attachments/ (needs eval/expected.json)
    npm run dev            # dev server (LAN-exposed; open on phone via Mac's IP)

## Privacy

attachments/ (colleague car photos) and eval/expected.json (their plate numbers)
are gitignored — never commit them.
```

- [ ] **Step 3: Verify test runner and dev server**

Run: `npm test`
Expected: 1 passed.

Run: `npm run build`
Expected: `vite build` completes without error.

- [ ] **Step 4: Commit**

```bash
git add package.json package-lock.json tsconfig.json vite.config.ts index.html src README.md .gitignore
git commit -m "feat: scaffold Vite+TS+Vitest project"
```

---

### Task 2: Model download + contract probe

**Files:**
- Create: `scripts/fetch-models.sh`, `scripts/probe-models.ts`, `docs/models.md`

**Interfaces:**
- Consumes: nothing.
- Produces: `public/models/{yolo-v9-t-384-license-plates-end2end.onnx, yolo-v9-t-512-license-plates-end2end.onnx, cct_xs_v2_global.onnx, cct_s_v2_global.onnx}`, `public/ort/*.wasm|*.mjs`; `docs/models.md` recording actual input/output names + dims (later tasks rely on the *contracts* above, not on names).

- [ ] **Step 1: Write `scripts/fetch-models.sh`**

```bash
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p public/models public/ort

OIM="https://github.com/ankandrew/open-image-models/releases/download/assets"
OCR="https://github.com/ankandrew/cnn-ocr-lp/releases/download/arg-plates"

fetch() { # url dest
  if [ ! -s "$2" ]; then
    echo "downloading $(basename "$2")..."
    curl -fL --retry 3 -o "$2" "$1"
  else
    echo "already present: $(basename "$2")"
  fi
  # sanity: model files must be > 1MB
  [ "$(wc -c < "$2")" -gt 1000000 ] || { echo "ERROR: $2 too small"; exit 1; }
}

fetch "$OIM/yolo-v9-t-384-license-plates-end2end.onnx" public/models/yolo-v9-t-384-license-plates-end2end.onnx
fetch "$OIM/yolo-v9-t-512-license-plates-end2end.onnx" public/models/yolo-v9-t-512-license-plates-end2end.onnx
fetch "$OCR/cct_xs_v2_global.onnx" public/models/cct_xs_v2_global.onnx
fetch "$OCR/cct_s_v2_global.onnx" public/models/cct_s_v2_global.onnx

# ONNX Runtime web assets are NOT copied to public/ — the app imports them
# with Vite `?url` imports (see src/web/ort-web.ts), which serves them in dev
# and bundles them into dist/ for prod, same-origin (never a CDN).
echo "OK"
```

- [ ] **Step 2: Run it**

Run: `chmod +x scripts/fetch-models.sh && npm run fetch-models`
Expected: four "downloading …" lines then `OK`. `ls -la public/models` shows 4 files (≈7.8 MB, 7.8 MB, 3.3 MB, 5.3 MB).

- [ ] **Step 3: Write `scripts/probe-models.ts`** — executable verification of the model contracts

```ts
import * as ort from 'onnxruntime-node'

async function probe(path: string, feedsFactory: (inputName: string) => Record<string, ort.Tensor>) {
  const s = await ort.InferenceSession.create(path)
  console.log(`\n=== ${path}`)
  console.log('inputs :', s.inputNames.join(', '))
  console.log('outputs:', s.outputNames.join(', '))
  const inputName = s.inputNames[0]
  if (!inputName) throw new Error('model has no inputs')
  const res = await s.run(feedsFactory(inputName))
  for (const [name, t] of Object.entries(res)) {
    console.log(`output ${name}: dims=[${t.dims.join(',')}] type=${t.type}`)
  }
  return res
}

// Detector: expect output rows of 7 (possibly 0 rows on a blank image)
for (const size of [384, 512]) {
  const res = await probe(
    `public/models/yolo-v9-t-${size}-license-plates-end2end.onnx`,
    (name) => ({ [name]: new ort.Tensor('float32', new Float32Array(1 * 3 * size * size), [1, 3, size, size]) }),
  )
  const out = Object.values(res)[0]!
  const cols = out.dims[out.dims.length - 1]
  if (cols !== 7 || (out.data as Float32Array).length % 7 !== 0)
    throw new Error(`detector ${size}: expected rows of 7, got dims [${out.dims.join(',')}]`)
}

// OCR: expect one output with 370 elems (10 slots x 37 alphabet) and one with 66 (regions)
for (const m of ['cct_xs_v2_global', 'cct_s_v2_global']) {
  const res = await probe(`public/models/${m}.onnx`, (name) => ({
    [name]: new ort.Tensor('uint8', new Uint8Array(1 * 64 * 128 * 3), [1, 64, 128, 3]),
  }))
  const sizes = Object.values(res).map((t) => t.dims.reduce((a, b) => a * b, 1))
  if (!sizes.includes(370)) throw new Error(`${m}: no 370-element plate head (got ${sizes.join(',')})`)
  if (!sizes.includes(66)) throw new Error(`${m}: no 66-element region head (got ${sizes.join(',')})`)
}
console.log('\nAll model contracts verified.')
```

- [ ] **Step 4: Run probe and record results**

Run: `npm run probe`
Expected: per model, input/output names and dims printed; final line `All model contracts verified.` If an assertion throws, STOP — the contract section of this plan is wrong and must be fixed before proceeding.

Create `docs/models.md` and paste the probe output into it, under a heading per model, plus the source URLs from `fetch-models.sh`.

- [ ] **Step 5: Commit**

```bash
git add scripts/fetch-models.sh scripts/probe-models.ts docs/models.md
git commit -m "feat: model fetch script and contract probe"
```

---

### Task 3: Pipeline types + pure image ops

**Files:**
- Create: `src/pipeline/types.ts`, `src/pipeline/image.ts`, `src/pipeline/image.test.ts`
- Delete: `src/pipeline/smoke.test.ts`

**Interfaces:**
- Consumes: nothing.
- Produces (used by all later tasks):
  - `ImageDataLike { data: Uint8ClampedArray; width: number; height: number }` (RGBA, browser-`ImageData`-compatible)
  - `Box { x1: number; y1: number; x2: number; y2: number; score: number }`
  - `TensorLike { type: 'float32' | 'uint8'; data: Float32Array | Uint8Array; dims: number[] }`
  - `OrtSessionLike { inputNames: readonly string[]; outputNames: readonly string[]; run(feeds: Record<string, TensorLike>): Promise<Record<string, TensorLike>> }`
  - `resizeBilinear(src: ImageDataLike, dstW: number, dstH: number): ImageDataLike`
  - `crop(src: ImageDataLike, box: Box): ImageDataLike`
  - `expandBox(box: Box, margin: number, imgW: number, imgH: number): Box`
  - `cropResize(src: ImageDataLike, box: Box, dstW: number, dstH: number): ImageDataLike`
  - `letterbox(src: ImageDataLike, size: number): { image: ImageDataLike; ratio: number; dw: number; dh: number }`

- [ ] **Step 1: Write `src/pipeline/types.ts`**

```ts
/** RGBA pixel buffer; structurally compatible with the DOM's ImageData. */
export interface ImageDataLike {
  data: Uint8ClampedArray
  width: number
  height: number
}

export interface Box {
  x1: number
  y1: number
  x2: number
  y2: number
  score: number
}

export interface TensorLike {
  type: 'float32' | 'uint8'
  data: Float32Array | Uint8Array
  dims: number[]
}

/** Minimal structural interface satisfied by both onnxruntime-web and -node adapters. */
export interface OrtSessionLike {
  inputNames: readonly string[]
  outputNames: readonly string[]
  run(feeds: Record<string, TensorLike>): Promise<Record<string, TensorLike>>
}

export interface OcrRead {
  text: string
  charProbs: number[]
  region?: string
  regionProb?: number
}

export interface Correction {
  pos: number
  from: string
  to: string
}

export interface PlateValidation {
  /** Normalized raw OCR text (A-Z0-9 only). */
  raw: string
  /** Best plate string after corrections (== raw when no rule matched). */
  plate: string
  /** Human display form, e.g. "BN CR 788" (== plate when no rule matched). */
  display: string
  formatValid: boolean
  corrections: Correction[]
  /** Full plate with umlaut district applied, e.g. "TÖLAB123" (suggestion only). */
  umlautSuggestion?: string
  /** Matching rule id, e.g. "DE", or null. */
  rule: string | null
  /** Mean per-char OCR probability, +0.05 if a rule matched (clamped to 1). */
  confidence: number
}

export interface PlateCandidate {
  box: Box
  read: OcrRead
  validation: PlateValidation
}
```

- [ ] **Step 2: Write failing tests `src/pipeline/image.test.ts`** (and delete `smoke.test.ts`)

```ts
import { describe, it, expect } from 'vitest'
import { resizeBilinear, crop, expandBox, cropResize, letterbox } from './image'
import type { ImageDataLike } from './types'

/** Build an image from per-pixel RGB rows (alpha forced to 255). */
function img(width: number, height: number, rgb: number[][]): ImageDataLike {
  const data = new Uint8ClampedArray(width * height * 4)
  rgb.forEach((px, i) => {
    data[i * 4] = px[0]!
    data[i * 4 + 1] = px[1]!
    data[i * 4 + 2] = px[2]!
    data[i * 4 + 3] = 255
  })
  return { data, width, height }
}

const px = (im: ImageDataLike, x: number, y: number) => {
  const o = (y * im.width + x) * 4
  return [im.data[o], im.data[o + 1], im.data[o + 2]]
}

describe('resizeBilinear', () => {
  it('identity resize returns identical pixels', () => {
    const src = img(2, 2, [[10, 20, 30], [40, 50, 60], [70, 80, 90], [100, 110, 120]])
    const out = resizeBilinear(src, 2, 2)
    expect([...out.data]).toEqual([...src.data])
  })

  it('2x downscale averages a uniform 2x2 block', () => {
    // 2x2 image of values 0,0,0 / 100,100,100 / 100,100,100 / 200,200,200 -> 1x1 = mean 100
    const src = img(2, 2, [[0, 0, 0], [100, 100, 100], [100, 100, 100], [200, 200, 200]])
    const out = resizeBilinear(src, 1, 1)
    expect(px(out, 0, 0)).toEqual([100, 100, 100])
  })
})

describe('crop / expandBox / cropResize', () => {
  const src = img(4, 4, Array.from({ length: 16 }, (_, i) => [i * 10, i * 10, i * 10]))

  it('crop extracts the exact region', () => {
    const c = crop(src, { x1: 1, y1: 1, x2: 3, y2: 3, score: 1 })
    expect(c.width).toBe(2)
    expect(c.height).toBe(2)
    expect(px(c, 0, 0)).toEqual([50, 50, 50]) // src pixel (1,1) = index 5
  })

  it('crop clamps out-of-bounds boxes', () => {
    const c = crop(src, { x1: -5, y1: -5, x2: 2, y2: 2, score: 1 })
    expect(c.width).toBe(2)
    expect(c.height).toBe(2)
    expect(px(c, 0, 0)).toEqual([0, 0, 0]) // src pixel (0,0)
  })

  it('expandBox grows by margin fraction of box size and clamps to image', () => {
    const b = expandBox({ x1: 10, y1: 10, x2: 20, y2: 20, score: 1 }, 0.1, 100, 100)
    expect(b).toMatchObject({ x1: 9, y1: 9, x2: 21, y2: 21 })
    const edge = expandBox({ x1: 0, y1: 0, x2: 10, y2: 10, score: 1 }, 0.5, 100, 100)
    expect(edge.x1).toBe(0)
    expect(edge.y1).toBe(0)
  })

  it('cropResize returns requested dimensions', () => {
    const out = cropResize(src, { x1: 0, y1: 0, x2: 4, y2: 4, score: 1 }, 8, 2)
    expect(out.width).toBe(8)
    expect(out.height).toBe(2)
  })
})

describe('letterbox', () => {
  it('100x50 -> 64: ratio 0.64, horizontal fit, vertical padding', () => {
    const src = img(100, 50, Array.from({ length: 5000 }, () => [255, 0, 0]))
    const lb = letterbox(src, 64)
    expect(lb.ratio).toBeCloseTo(0.64)
    expect(lb.dw).toBeCloseTo(0)
    expect(lb.dh).toBeCloseTo(16)
    expect(lb.image.width).toBe(64)
    expect(lb.image.height).toBe(64)
    // top padding row is gray 114
    expect(px(lb.image, 32, 0)).toEqual([114, 114, 114])
    // center is red content
    expect(px(lb.image, 32, 32)).toEqual([255, 0, 0])
  })
})
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `rm src/pipeline/smoke.test.ts && npm test`
Expected: FAIL — `Cannot find module './image'` (or equivalent).

- [ ] **Step 4: Write `src/pipeline/image.ts`**

```ts
import type { Box, ImageDataLike } from './types'

/** Center-aligned bilinear resize (matches cv2.INTER_LINEAR mapping). */
export function resizeBilinear(src: ImageDataLike, dstW: number, dstH: number): ImageDataLike {
  if (dstW === src.width && dstH === src.height) {
    return { data: new Uint8ClampedArray(src.data), width: src.width, height: src.height }
  }
  const dst = new Uint8ClampedArray(dstW * dstH * 4)
  const sx = src.width / dstW
  const sy = src.height / dstH
  for (let y = 0; y < dstH; y++) {
    const fy = Math.min(Math.max((y + 0.5) * sy - 0.5, 0), src.height - 1)
    const y0 = Math.floor(fy)
    const y1 = Math.min(y0 + 1, src.height - 1)
    const wy = fy - y0
    for (let x = 0; x < dstW; x++) {
      const fx = Math.min(Math.max((x + 0.5) * sx - 0.5, 0), src.width - 1)
      const x0 = Math.floor(fx)
      const x1 = Math.min(x0 + 1, src.width - 1)
      const wx = fx - x0
      const i00 = (y0 * src.width + x0) * 4
      const i10 = (y0 * src.width + x1) * 4
      const i01 = (y1 * src.width + x0) * 4
      const i11 = (y1 * src.width + x1) * 4
      const o = (y * dstW + x) * 4
      for (let c = 0; c < 4; c++) {
        const top = src.data[i00 + c]! * (1 - wx) + src.data[i10 + c]! * wx
        const bot = src.data[i01 + c]! * (1 - wx) + src.data[i11 + c]! * wx
        dst[o + c] = Math.round(top * (1 - wy) + bot * wy)
      }
    }
  }
  return { data: dst, width: dstW, height: dstH }
}

/** Crop box region (rounded, clamped, min 1px). */
export function crop(src: ImageDataLike, box: Box): ImageDataLike {
  const x1 = Math.min(Math.max(Math.round(box.x1), 0), src.width - 1)
  const y1 = Math.min(Math.max(Math.round(box.y1), 0), src.height - 1)
  const x2 = Math.min(Math.max(Math.round(box.x2), x1 + 1), src.width)
  const y2 = Math.min(Math.max(Math.round(box.y2), y1 + 1), src.height)
  const w = x2 - x1
  const h = y2 - y1
  const data = new Uint8ClampedArray(w * h * 4)
  for (let y = 0; y < h; y++) {
    const srcOff = ((y1 + y) * src.width + x1) * 4
    data.set(src.data.subarray(srcOff, srcOff + w * 4), y * w * 4)
  }
  return { data, width: w, height: h }
}

/** Expand each side by margin × that dimension's box size, clamped to the image. */
export function expandBox(box: Box, margin: number, imgW: number, imgH: number): Box {
  const mw = (box.x2 - box.x1) * margin
  const mh = (box.y2 - box.y1) * margin
  return {
    x1: Math.max(0, box.x1 - mw),
    y1: Math.max(0, box.y1 - mh),
    x2: Math.min(imgW, box.x2 + mw),
    y2: Math.min(imgH, box.y2 + mh),
    score: box.score,
  }
}

export function cropResize(src: ImageDataLike, box: Box, dstW: number, dstH: number): ImageDataLike {
  return resizeBilinear(crop(src, box), dstW, dstH)
}

export interface Letterboxed {
  image: ImageDataLike
  ratio: number
  dw: number
  dh: number
}

/** YOLOv9-style letterbox: aspect-preserving resize into size×size with gray(114) padding. */
export function letterbox(src: ImageDataLike, size: number): Letterboxed {
  const r = Math.min(size / src.height, size / src.width)
  const newW = Math.round(src.width * r)
  const newH = Math.round(src.height * r)
  const resized = resizeBilinear(src, newW, newH)
  const dw = (size - newW) / 2
  const dh = (size - newH) / 2
  const left = Math.round(dw - 0.1)
  const top = Math.round(dh - 0.1)
  const data = new Uint8ClampedArray(size * size * 4)
  for (let i = 0; i < size * size; i++) {
    data[i * 4] = 114
    data[i * 4 + 1] = 114
    data[i * 4 + 2] = 114
    data[i * 4 + 3] = 255
  }
  for (let y = 0; y < newH; y++) {
    const dstOff = ((top + y) * size + left) * 4
    data.set(resized.data.subarray(y * newW * 4, (y + 1) * newW * 4), dstOff)
  }
  return { image: { data, width: size, height: size }, ratio: r, dw, dh }
}
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `npm test`
Expected: all image tests PASS.

- [ ] **Step 6: Commit**

```bash
git add src/pipeline
git commit -m "feat: pipeline types and pure image ops (bilinear, crop, letterbox)"
```

---

### Task 4: Detector module

**Files:**
- Create: `src/pipeline/detector.ts`, `src/pipeline/detector.test.ts`

**Interfaces:**
- Consumes: `ImageDataLike`, `TensorLike`, `Box`, `Letterboxed` from Task 3.
- Produces:
  - `toDetectorTensor(lb: ImageDataLike): TensorLike` — CHW RGB float32/255, dims `[1,3,H,W]`
  - `decodeDetections(out: TensorLike, geom: { ratio: number; dw: number; dh: number }, imgW: number, imgH: number, confThresh?: number): Box[]` (default confThresh 0.25; filters rows below threshold and degenerate boxes < 4px; clamps to image bounds; coords in ORIGINAL image space)

- [ ] **Step 1: Write failing tests `src/pipeline/detector.test.ts`**

```ts
import { describe, it, expect } from 'vitest'
import { toDetectorTensor, decodeDetections } from './detector'
import type { ImageDataLike, TensorLike } from './types'

describe('toDetectorTensor', () => {
  it('produces planar CHW RGB scaled to 0-1', () => {
    // 2x1 image: pixel0 = (255, 0, 51), pixel1 = (0, 102, 255)
    const im: ImageDataLike = {
      data: new Uint8ClampedArray([255, 0, 51, 255, 0, 102, 255, 255]),
      width: 2,
      height: 1,
    }
    const t = toDetectorTensor(im)
    expect(t.dims).toEqual([1, 3, 1, 2])
    expect(t.type).toBe('float32')
    const d = t.data as Float32Array
    // R plane
    expect(d[0]).toBeCloseTo(1.0)
    expect(d[1]).toBeCloseTo(0.0)
    // G plane
    expect(d[2]).toBeCloseTo(0.0)
    expect(d[3]).toBeCloseTo(0.4)
    // B plane
    expect(d[4]).toBeCloseTo(0.2)
    expect(d[5]).toBeCloseTo(1.0)
  })
})

describe('decodeDetections', () => {
  const row = (x1: number, y1: number, x2: number, y2: number, score: number) => [0, x1, y1, x2, y2, 0, score]

  it('un-letterboxes coordinates back to original image space', () => {
    // original 768x384 -> letterbox 384: ratio 0.5, dw 0, dh 96
    const out: TensorLike = {
      type: 'float32',
      data: new Float32Array(row(100, 146, 200, 196, 0.9)),
      dims: [1, 7],
    }
    const boxes = decodeDetections(out, { ratio: 0.5, dw: 0, dh: 96 }, 768, 384)
    expect(boxes).toHaveLength(1)
    expect(boxes[0]!.x1).toBeCloseTo(200) // (100-0)/0.5
    expect(boxes[0]!.y1).toBeCloseTo(100) // (146-96)/0.5
    expect(boxes[0]!.x2).toBeCloseTo(400)
    expect(boxes[0]!.y2).toBeCloseTo(200)
    expect(boxes[0]!.score).toBeCloseTo(0.9)
  })

  it('filters below confidence threshold and degenerate boxes, clamps to bounds', () => {
    const out: TensorLike = {
      type: 'float32',
      data: new Float32Array([
        ...row(10, 10, 100, 50, 0.1), // below threshold
        ...row(10, 10, 12, 11, 0.9), // degenerate (<4px after unletterbox at ratio 1)
        ...row(-20, 10, 100, 50, 0.8), // clamped to x1=0
      ]),
      dims: [3, 7],
    }
    const boxes = decodeDetections(out, { ratio: 1, dw: 0, dh: 0 }, 200, 200)
    expect(boxes).toHaveLength(1)
    expect(boxes[0]!.x1).toBe(0)
  })

  it('returns empty array for empty output', () => {
    const out: TensorLike = { type: 'float32', data: new Float32Array(0), dims: [0, 7] }
    expect(decodeDetections(out, { ratio: 1, dw: 0, dh: 0 }, 100, 100)).toEqual([])
  })
})
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `npm test`
Expected: FAIL — `Cannot find module './detector'`.

- [ ] **Step 3: Write `src/pipeline/detector.ts`**

```ts
import type { Box, ImageDataLike, TensorLike } from './types'

export const DETECTOR_DEFAULT_SIZE = 384

/** RGBA letterboxed image -> [1,3,H,W] float32 RGB in 0-1. */
export function toDetectorTensor(lb: ImageDataLike): TensorLike {
  const n = lb.width * lb.height
  const data = new Float32Array(n * 3)
  for (let i = 0; i < n; i++) {
    data[i] = lb.data[i * 4]! / 255
    data[n + i] = lb.data[i * 4 + 1]! / 255
    data[2 * n + i] = lb.data[i * 4 + 2]! / 255
  }
  return { type: 'float32', data, dims: [1, 3, lb.height, lb.width] }
}

/**
 * Decode end2end YOLOv9 output rows [batchIdx, x1, y1, x2, y2, classId, score]
 * back to original-image coordinates.
 */
export function decodeDetections(
  out: TensorLike,
  geom: { ratio: number; dw: number; dh: number },
  imgW: number,
  imgH: number,
  confThresh = 0.25,
): Box[] {
  const d = out.data
  const rows = Math.floor(d.length / 7)
  const boxes: Box[] = []
  for (let r = 0; r < rows; r++) {
    const base = r * 7
    const score = d[base + 6]!
    if (score < confThresh) continue
    const x1 = Math.min(Math.max((d[base + 1]! - geom.dw) / geom.ratio, 0), imgW)
    const y1 = Math.min(Math.max((d[base + 2]! - geom.dh) / geom.ratio, 0), imgH)
    const x2 = Math.min(Math.max((d[base + 3]! - geom.dw) / geom.ratio, 0), imgW)
    const y2 = Math.min(Math.max((d[base + 4]! - geom.dh) / geom.ratio, 0), imgH)
    if (x2 - x1 < 4 || y2 - y1 < 4) continue
    boxes.push({ x1, y1, x2, y2, score })
  }
  return boxes
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `npm test`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/pipeline/detector.ts src/pipeline/detector.test.ts
git commit -m "feat: detector tensor conversion and output decoding"
```

---

### Task 5: OCR module

**Files:**
- Create: `src/pipeline/ocr.ts`, `src/pipeline/ocr.test.ts`

**Interfaces:**
- Consumes: `ImageDataLike`, `TensorLike`, `OcrRead` from Task 3.
- Produces:
  - Constants `OCR_WIDTH = 128`, `OCR_HEIGHT = 64`, `OCR_SLOTS = 10`, `OCR_ALPHABET` (37 chars, `_` = pad), `OCR_REGIONS` (66 labels)
  - `toOcrTensor(crop: ImageDataLike): TensorLike` — uint8 NHWC RGB `[1,64,128,3]`; throws if crop is not 128×64
  - `decodeOcr(outputs: Record<string, TensorLike>): OcrRead` — identifies plate head (370 elems) and region head (66 elems) by size
  - Test helper (exported from the test file for reuse in Task 8): `fakeOcrOutputs(text: string, prob?: number): Record<string, TensorLike>`

- [ ] **Step 1: Write failing tests `src/pipeline/ocr.test.ts`**

```ts
import { describe, it, expect } from 'vitest'
import { toOcrTensor, decodeOcr, OCR_ALPHABET, OCR_SLOTS, OCR_REGIONS, OCR_WIDTH, OCR_HEIGHT } from './ocr'
import type { ImageDataLike, TensorLike } from './types'

/** Build fake OCR model outputs that decode to `text` with per-char prob `prob`. */
export function fakeOcrOutputs(text: string, prob = 0.95): Record<string, TensorLike> {
  const plate = new Float32Array(OCR_SLOTS * OCR_ALPHABET.length)
  const rest = (1 - prob) / (OCR_ALPHABET.length - 1)
  for (let slot = 0; slot < OCR_SLOTS; slot++) {
    const ch = slot < text.length ? text[slot]! : '_'
    const idx = OCR_ALPHABET.indexOf(ch)
    for (let a = 0; a < OCR_ALPHABET.length; a++) {
      plate[slot * OCR_ALPHABET.length + a] = a === idx ? prob : rest
    }
  }
  const region = new Float32Array(OCR_REGIONS.length)
  region[OCR_REGIONS.indexOf('Germany')] = 0.8
  return {
    plateHead: { type: 'float32', data: plate, dims: [1, OCR_SLOTS * OCR_ALPHABET.length] },
    regionHead: { type: 'float32', data: region, dims: [1, OCR_REGIONS.length] },
  }
}

describe('toOcrTensor', () => {
  it('produces uint8 NHWC RGB and drops alpha', () => {
    const data = new Uint8ClampedArray(OCR_WIDTH * OCR_HEIGHT * 4)
    data[0] = 11
    data[1] = 22
    data[2] = 33
    data[3] = 255
    const im: ImageDataLike = { data, width: OCR_WIDTH, height: OCR_HEIGHT }
    const t = toOcrTensor(im)
    expect(t.dims).toEqual([1, OCR_HEIGHT, OCR_WIDTH, 3])
    expect(t.type).toBe('uint8')
    expect([t.data[0], t.data[1], t.data[2]]).toEqual([11, 22, 33])
    expect(t.data.length).toBe(OCR_HEIGHT * OCR_WIDTH * 3)
  })

  it('throws on wrong crop size', () => {
    const im: ImageDataLike = { data: new Uint8ClampedArray(4), width: 1, height: 1 }
    expect(() => toOcrTensor(im)).toThrow()
  })
})

describe('decodeOcr', () => {
  it('decodes text, strips trailing pads, reports char probs and region', () => {
    const read = decodeOcr(fakeOcrOutputs('KRLM144', 0.9))
    expect(read.text).toBe('KRLM144')
    expect(read.charProbs).toHaveLength(7)
    expect(read.charProbs[0]).toBeCloseTo(0.9)
    expect(read.region).toBe('Germany')
    expect(read.regionProb).toBeCloseTo(0.8)
  })

  it('works without a region head', () => {
    const outputs = fakeOcrOutputs('BNCR788')
    delete outputs.regionHead
    const read = decodeOcr(outputs)
    expect(read.text).toBe('BNCR788')
    expect(read.region).toBeUndefined()
  })
})
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `npm test`
Expected: FAIL — `Cannot find module './ocr'`.

- [ ] **Step 3: Write `src/pipeline/ocr.ts`**

```ts
import type { ImageDataLike, OcrRead, TensorLike } from './types'

export const OCR_WIDTH = 128
export const OCR_HEIGHT = 64
export const OCR_SLOTS = 10
export const OCR_ALPHABET = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_'
export const OCR_PAD = '_'

/** Region labels of cct v2 global models (verbatim from upstream plate config). */
export const OCR_REGIONS = [
  'Albania', 'Andorra', 'Argentina', 'Armenia', 'Australia', 'Austria', 'Azerbaijan', 'Bahrain',
  'Belarus', 'Belgium', 'Bosnia and Herzegovina', 'Brazil', 'Bulgaria', 'Cambodia', 'Canada', 'Croatia',
  'Cyprus', 'Czech Republic', 'Denmark', 'Estonia', 'Finland', 'France', 'Georgia', 'Germany',
  'Gibraltar', 'Greece', 'Guernsey', 'Hungary', 'Iceland', 'Indonesia', 'Ireland', 'Israel', 'Italy',
  'Latvia', 'Liechtenstein', 'Lithuania', 'Luxembourg', 'Malaysia', 'Malta', 'Mexico', 'Moldova',
  'Monaco', 'Montenegro', 'Netherlands', 'New Zealand', 'North Macedonia', 'Norway', 'Poland',
  'Portugal', 'Qatar', 'Romania', 'San Marino', 'Serbia', 'Singapore', 'Slovakia', 'Slovenia', 'Spain',
  'Sweden', 'Switzerland', 'Thailand', 'Turkey', 'United States', 'Ukraine', 'United Kingdom', 'Vietnam',
  'Unknown',
] as const

/** 128x64 RGBA crop -> [1,64,128,3] uint8 NHWC RGB. Model normalizes internally. */
export function toOcrTensor(cropped: ImageDataLike): TensorLike {
  if (cropped.width !== OCR_WIDTH || cropped.height !== OCR_HEIGHT) {
    throw new Error(`OCR crop must be ${OCR_WIDTH}x${OCR_HEIGHT}, got ${cropped.width}x${cropped.height}`)
  }
  const n = OCR_WIDTH * OCR_HEIGHT
  const data = new Uint8Array(n * 3)
  for (let i = 0; i < n; i++) {
    data[i * 3] = cropped.data[i * 4]!
    data[i * 3 + 1] = cropped.data[i * 4 + 1]!
    data[i * 3 + 2] = cropped.data[i * 4 + 2]!
  }
  return { type: 'uint8', data, dims: [1, OCR_HEIGHT, OCR_WIDTH, 3] }
}

const elemCount = (t: TensorLike) => t.dims.reduce((a, b) => a * b, 1)

/** Decode OCR outputs; heads are identified by element count (names vary between exports). */
export function decodeOcr(outputs: Record<string, TensorLike>): OcrRead {
  let plate: TensorLike | undefined
  let region: TensorLike | undefined
  for (const t of Object.values(outputs)) {
    const n = elemCount(t)
    if (n === OCR_SLOTS * OCR_ALPHABET.length) {
      if (plate) throw new Error('multiple OCR outputs match plate head size (370)')
      plate = t
    } else if (n === OCR_REGIONS.length) {
      if (region) throw new Error('multiple OCR outputs match region head size (66)')
      region = t
    }
  }
  if (!plate) throw new Error('OCR output missing plate head (370 elements)')

  const probs = plate.data as Float32Array
  let text = ''
  const charProbs: number[] = []
  for (let slot = 0; slot < OCR_SLOTS; slot++) {
    let best = 0
    let bestP = -Infinity
    for (let a = 0; a < OCR_ALPHABET.length; a++) {
      const p = probs[slot * OCR_ALPHABET.length + a]!
      if (p > bestP) {
        bestP = p
        best = a
      }
    }
    text += OCR_ALPHABET[best]!
    charProbs.push(bestP)
  }
  // strip trailing pad chars (and their probs)
  while (text.endsWith(OCR_PAD)) {
    text = text.slice(0, -1)
    charProbs.pop()
  }

  const read: OcrRead = { text, charProbs }
  if (region) {
    const rd = region.data as Float32Array
    let best = 0
    for (let i = 1; i < rd.length; i++) if (rd[i]! > rd[best]!) best = i
    read.region = OCR_REGIONS[best]!
    read.regionProb = rd[best]!
  }
  return read
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `npm test`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/pipeline/ocr.ts src/pipeline/ocr.test.ts
git commit -m "feat: OCR tensor conversion and output decoding with region head"
```

---

### Task 6: Validator (pluggable rules, German rule set)

**Files:**
- Create: `src/pipeline/rules/de.ts`, `src/pipeline/validate.ts`, `src/pipeline/validate.test.ts`

**Interfaces:**
- Consumes: `Correction`, `PlateValidation` from types.
- Produces:
  - `normalizePlateText(s: string): string` — uppercase, strip all non-`A-Z0-9`
  - `matchGerman(raw: string): RuleMatch | null` where `RuleMatch { plate: string; display: string; corrections: Correction[]; parts: { district: string; letters: string; digits: string; suffix: string } }`
  - `validate(rawText: string, charProbs: number[]): PlateValidation`
  - Behavior contract: **never rejects** — no rule match ⇒ passthrough with `formatValid: false, rule: null`.

**German plate format:** `district (1–3 letters) + letters (1–2) + digits (1–4, no leading zero) + optional suffix E or H`, total ≤ 8 chars. OCR confusion corrections (only when they complete a pattern match, max 2): digit→letter `0→O, 1→I, 8→B, 5→S, 2→Z` in letter positions; letter→digit `O→0, I→1, B→8, S→5, Z→2` in digit positions. Umlaut districts can't be produced by OCR (alphabet has no umlauts) ⇒ suggestion only via lookup.

- [ ] **Step 1: Write failing tests `src/pipeline/validate.test.ts`**

```ts
import { describe, it, expect } from 'vitest'
import { normalizePlateText, validate } from './validate'
import { matchGerman } from './rules/de'

describe('normalizePlateText', () => {
  it('uppercases and strips separators', () => {
    expect(normalizePlateText('bn-cr 788')).toBe('BNCR788')
    expect(normalizePlateText('KR•LM 144')).toBe('KRLM144')
  })
})

describe('matchGerman', () => {
  it('matches a clean plate with zero corrections', () => {
    const m = matchGerman('BNCR788')!
    expect(m.plate).toBe('BNCR788')
    expect(m.display).toBe('BN CR 788')
    expect(m.corrections).toEqual([])
    expect(m.parts).toEqual({ district: 'BN', letters: 'CR', digits: '788', suffix: '' })
  })

  it('prefers the segmentation with fewest corrections', () => {
    // TOLAB123: TOL|AB|123 needs 0 swaps; TO|LA|B123 would need 1 (B->8)
    const m = matchGerman('TOLAB123')!
    expect(m.corrections).toEqual([])
    expect(m.parts.district).toBe('TOL')
  })

  it('corrects digit-lookalike in district: 0KXY226 -> OK XY 226', () => {
    const m = matchGerman('0KXY226')!
    expect(m.plate).toBe('OKXY226')
    expect(m.corrections).toEqual([{ pos: 0, from: '0', to: 'O' }])
  })

  it('corrects letter-lookalike in digits: BNCR7B8 -> BN CR 788', () => {
    const m = matchGerman('BNCR7B8')!
    expect(m.plate).toBe('BNCR788')
    expect(m.corrections).toEqual([{ pos: 5, from: 'B', to: '8' }])
  })

  it('handles E/H suffix', () => {
    const m = matchGerman('MXY123E')!
    expect(m.parts).toEqual({ district: 'M', letters: 'XY', digits: '123', suffix: 'E' })
    expect(m.display).toBe('M XY 123E')
  })

  it('rejects leading zero in digits when no segmentation can rescue it', () => {
    // ABCDE012: only fitting segmentation is ABC|DE|012 (leading zero); 'E' blocks all others
    expect(matchGerman('ABCDE012')).toBeNull()
    // NOTE: strings like BNCR0788 legitimately match via another segmentation
    // (BNC|RO|788 with 0->O) — that is correct behavior, not a bug.
  })

  it('caps corrections at 2 (protects EU plates from mangling)', () => {
    // XR25GB (NL-style): any segmentation needs >2 swaps or hits a non-correctable char
    expect(matchGerman('XR25GB')).toBeNull()
  })

  it('rejects too-long and too-short strings', () => {
    expect(matchGerman('BNCRX12345')).toBeNull() // 10 chars
    expect(matchGerman('D1')).toBeNull()
  })
})

describe('validate', () => {
  it('valid German read: formatValid, display, rule DE, confidence = mean + 0.05', () => {
    const v = validate('BNCR788', [0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9])
    expect(v.formatValid).toBe(true)
    expect(v.rule).toBe('DE')
    expect(v.plate).toBe('BNCR788')
    expect(v.display).toBe('BN CR 788')
    expect(v.confidence).toBeCloseTo(0.95)
  })

  it('unrecognized format passes through unchanged (never rejected)', () => {
    const v = validate('AB12CD', [0.8, 0.8, 0.8, 0.8, 0.8, 0.8])
    expect(v.formatValid).toBe(false)
    expect(v.rule).toBeNull()
    expect(v.plate).toBe('AB12CD')
    expect(v.display).toBe('AB12CD')
    expect(v.confidence).toBeCloseTo(0.8)
  })

  it('suggests umlaut district', () => {
    const v = validate('TOLAB123', [0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9])
    expect(v.umlautSuggestion).toBe('TÖLAB123')
  })

  it('no umlaut suggestion for regular districts', () => {
    const v = validate('BNCR788', [0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9])
    expect(v.umlautSuggestion).toBeUndefined()
  })
})
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `npm test`
Expected: FAIL — modules not found.

- [ ] **Step 3: Write `src/pipeline/rules/de.ts`**

```ts
import type { Correction } from '../types'

export interface RuleMatch {
  plate: string
  display: string
  corrections: Correction[]
  parts: { district: string; letters: string; digits: string; suffix: string }
}

/** digit -> letter lookalike (applied in letter positions) */
const D2L: Record<string, string> = { '0': 'O', '1': 'I', '8': 'B', '5': 'S', '2': 'Z' }
/** letter -> digit lookalike (applied in digit positions) */
const L2D: Record<string, string> = { O: '0', I: '1', B: '8', S: '5', Z: '2' }

const MAX_CORRECTIONS = 2
const MAX_TOTAL_LEN = 8

/**
 * ASCII district -> umlaut district. Suggestion only — extend freely; an entry
 * must only exist when the ASCII form is NOT itself an issued district code
 * (verify against the official registry before adding).
 * Deliberately absent: TU (= Tuttlingen), MU (= Landkreis München since 2026)
 * — both are real issued codes, so suggesting TÜ/MÜ for them would be wrong.
 */
export const UMLAUT_DISTRICTS: Record<string, string> = {
  TOL: 'TÖL', FU: 'FÜ', GO: 'GÖ', LO: 'LÖ', BUS: 'BÜS',
  SOM: 'SÖM', DUW: 'DÜW', KUN: 'KÜN', SUW: 'SÜW', RUD: 'RÜD', RUG: 'RÜG',
  PLO: 'PLÖ', JUL: 'JÜL', HMU: 'HMÜ', FUS: 'FÜS', MUR: 'MÜR', BUD: 'BÜD',
  FLO: 'FLÖ', UB: 'ÜB', NO: 'NÖ',
}

const isLetter = (c: string) => c >= 'A' && c <= 'Z'
const isDigit = (c: string) => c >= '0' && c <= '9'

/** Force chars in [start,end) to the given class, correcting lookalikes. Returns null if impossible. */
function forceClass(
  raw: string,
  start: number,
  end: number,
  cls: 'letter' | 'digit',
  corrections: Correction[],
): string | null {
  let out = ''
  for (let i = start; i < end; i++) {
    const c = raw[i]!
    if (cls === 'letter' ? isLetter(c) : isDigit(c)) {
      out += c
    } else {
      const sub = cls === 'letter' ? D2L[c] : L2D[c]
      if (!sub) return null
      corrections.push({ pos: i, from: c, to: sub })
      out += sub
    }
  }
  return out
}

/**
 * Try to interpret `raw` (normalized A-Z0-9) as a German plate.
 * Tries every segmentation; returns the one needing the fewest corrections (max 2), else null.
 */
export function matchGerman(raw: string): RuleMatch | null {
  const n = raw.length
  if (n < 3 || n > MAX_TOTAL_LEN) return null

  let best: RuleMatch | null = null
  for (let a = 1; a <= 3; a++) {
    for (let b = 1; b <= 2; b++) {
      for (const withSuffix of [false, true]) {
        const suffix = withSuffix ? raw[n - 1]! : ''
        if (withSuffix && suffix !== 'E' && suffix !== 'H') continue
        const digitsEnd = withSuffix ? n - 1 : n
        const digitsLen = digitsEnd - a - b
        if (digitsLen < 1 || digitsLen > 4) continue

        const corrections: Correction[] = []
        const district = forceClass(raw, 0, a, 'letter', corrections)
        if (district === null) continue
        const letters = forceClass(raw, a, a + b, 'letter', corrections)
        if (letters === null) continue
        const digits = forceClass(raw, a + b, digitsEnd, 'digit', corrections)
        if (digits === null) continue
        if (digits[0] === '0') continue
        if (corrections.length > MAX_CORRECTIONS) continue

        // Tie-break policy: at equal correction count, the FIRST valid
        // segmentation in iteration order (a asc, b asc, no-suffix first)
        // wins — i.e. the shortest district. plate/display are identical
        // across such ties; only parts.district (and thus umlaut
        // suggestions) depends on it. Deterministic by construction.
        if (!best || corrections.length < best.corrections.length) {
          const plate = district + letters + digits + suffix
          best = {
            plate,
            display: `${district} ${letters} ${digits}${suffix}`,
            corrections,
            parts: { district, letters, digits, suffix },
          }
        }
      }
    }
  }
  return best
}
```

- [ ] **Step 4: Write `src/pipeline/validate.ts`**

```ts
import type { PlateValidation } from './types'
import { matchGerman, UMLAUT_DISTRICTS } from './rules/de'

export function normalizePlateText(s: string): string {
  return s.toUpperCase().replace(/[^A-Z0-9]/g, '')
}

const clamp01 = (x: number) => Math.min(1, Math.max(0, x))

/**
 * Validate an OCR read. Rules score and refine; they NEVER reject.
 * Only rule set today: German ("DE"). EU support = add rule sets here.
 */
export function validate(rawText: string, charProbs: number[]): PlateValidation {
  const raw = normalizePlateText(rawText)
  const meanProb = charProbs.length ? charProbs.reduce((a, b) => a + b, 0) / charProbs.length : 0

  const m = matchGerman(raw)
  if (m) {
    const umlaut = UMLAUT_DISTRICTS[m.parts.district]
    return {
      raw,
      plate: m.plate,
      display: m.display,
      formatValid: true,
      corrections: m.corrections,
      ...(umlaut ? { umlautSuggestion: umlaut + m.plate.slice(m.parts.district.length) } : {}),
      rule: 'DE',
      confidence: clamp01(meanProb + 0.05),
    }
  }
  return {
    raw,
    plate: raw,
    display: raw,
    formatValid: false,
    corrections: [],
    rule: null,
    confidence: clamp01(meanProb),
  }
}
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `npm test`
Expected: PASS. If `XR25GB` unexpectedly matches, walk the segmentations by hand and tighten the test comment — do not weaken the cap.

- [ ] **Step 6: Commit**

```bash
git add src/pipeline/rules src/pipeline/validate.ts src/pipeline/validate.test.ts
git commit -m "feat: pluggable plate validation with German rule set"
```

---

### Task 7: Candidate ranking — REMOVED (design change 2026-07-15)

**User decision:** all detected plates have equal priority; the app must not
guess which car is "the blocked one". No `rank.ts`, no centrality/area
scoring. The pipeline (Task 8) orders candidates by raw detector confidence
purely for stable output — the order carries no meaning and the UI presents
all plates as equals.

---

### Task 8: Pipeline orchestrator

**Files:**
- Create: `src/pipeline/pipeline.ts`, `src/pipeline/pipeline.test.ts`

**Interfaces:**
- Consumes: everything from Tasks 3–6; `fakeOcrOutputs` from `./ocr.test`. (No ranking module — design change: candidates ordered by detector confidence only, order carries no meaning.)
- Produces (the single API both frontends use):
  - `PipelineSessions { detector: OrtSessionLike; ocr: OrtSessionLike }`
  - `PipelineOptions { detectorSize?: number (default 384); confThresh?: number (0.25); cropMargin?: number (0.0); maxCandidates?: number (5) }`
  - `PipelineResult { candidates: PlateCandidate[]; timings: { detectMs: number; ocrMs: number; totalMs: number } }`
  - `extractPlates(image: ImageDataLike, sessions: PipelineSessions, opts?: PipelineOptions): Promise<PipelineResult>`
  - `ocrRegion(image: ImageDataLike, box: Box, sessions: PipelineSessions): Promise<PlateCandidate>` — OCR-only path for manual crop

- [ ] **Step 1: Write failing tests `src/pipeline/pipeline.test.ts`**

```ts
import { describe, it, expect } from 'vitest'
import { extractPlates, ocrRegion } from './pipeline'
import { fakeOcrOutputs } from './ocr.test'
import type { ImageDataLike, OrtSessionLike, TensorLike } from './types'

const blank = (w: number, h: number): ImageDataLike => {
  const data = new Uint8ClampedArray(w * h * 4)
  for (let i = 3; i < data.length; i += 4) data[i] = 255
  return { data, width: w, height: h }
}

/** Fake detector returning fixed rows regardless of input. */
const fakeDetector = (rows: number[][]): OrtSessionLike => ({
  inputNames: ['images'],
  outputNames: ['output'],
  run: async () => ({
    output: { type: 'float32', data: new Float32Array(rows.flat()), dims: [rows.length, 7] } as TensorLike,
  }),
})

const fakeOcr = (text: string): OrtSessionLike => ({
  inputNames: ['input'],
  outputNames: ['plateHead', 'regionHead'],
  run: async (feeds) => {
    const t = Object.values(feeds)[0]!
    if (t.dims.join(',') !== '1,64,128,3' || t.type !== 'uint8') throw new Error('bad OCR feed')
    return fakeOcrOutputs(text)
  },
})

describe('extractPlates', () => {
  // 384x384 image -> letterbox is identity (ratio 1, dw=dh=0): detector coords == image coords
  const image = blank(384, 384)

  it('returns a validated candidate with timings', async () => {
    const sessions = {
      detector: fakeDetector([[0, 100, 200, 160, 224, 0, 0.9]]),
      ocr: fakeOcr('BNCR788'),
    }
    const res = await extractPlates(image, sessions)
    expect(res.candidates).toHaveLength(1)
    const c = res.candidates[0]!
    expect(c.validation.plate).toBe('BNCR788')
    expect(c.validation.display).toBe('BN CR 788')
    expect(c.read.region).toBe('Germany')
    expect(c.box.x1).toBeCloseTo(100)
    expect(res.timings.totalMs).toBeGreaterThan(0)
  })

  it('returns empty candidates when nothing detected', async () => {
    const sessions = { detector: fakeDetector([]), ocr: fakeOcr('BNCR788') }
    const res = await extractPlates(image, sessions)
    expect(res.candidates).toEqual([])
  })

  it('caps candidates at maxCandidates in detector-confidence order', async () => {
    const rows = [
      [0, 10, 10, 40, 20, 0, 0.55],
      [0, 120, 170, 280, 220, 0, 0.95], // highest confidence -> first (stable order only, no semantics)
      [0, 300, 10, 340, 25, 0, 0.75],
    ]
    const sessions = { detector: fakeDetector(rows), ocr: fakeOcr('KRLM144') }
    const res = await extractPlates(image, sessions, { maxCandidates: 2 })
    expect(res.candidates).toHaveLength(2)
    expect(res.candidates[0]!.box.x1).toBeCloseTo(120)
    expect(res.candidates[1]!.box.x1).toBeCloseTo(300)
  })
})

describe('ocrRegion', () => {
  it('runs OCR on a manual box without detection', async () => {
    const sessions = { detector: fakeDetector([]), ocr: fakeOcr('OKXY226') }
    const c = await ocrRegion(blank(384, 384), { x1: 10, y1: 10, x2: 200, y2: 60, score: 1 }, sessions)
    expect(c.validation.plate).toBe('OKXY226')
  })
})
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `npm test`
Expected: FAIL — `Cannot find module './pipeline'`.

- [ ] **Step 3: Write `src/pipeline/pipeline.ts`**

```ts
import type { Box, ImageDataLike, PlateCandidate } from './types'
import type { OrtSessionLike } from './types'
import { letterbox, cropResize, expandBox } from './image'
import { toDetectorTensor, decodeDetections, DETECTOR_DEFAULT_SIZE } from './detector'
import { toOcrTensor, decodeOcr, OCR_WIDTH, OCR_HEIGHT } from './ocr'
import { validate } from './validate'

export interface PipelineSessions {
  detector: OrtSessionLike
  ocr: OrtSessionLike
}

export interface PipelineOptions {
  detectorSize?: number
  confThresh?: number
  cropMargin?: number
  maxCandidates?: number
}

export interface PipelineResult {
  candidates: PlateCandidate[]
  timings: { detectMs: number; ocrMs: number; totalMs: number }
}

const now = () => performance.now()

async function readBox(
  image: ImageDataLike,
  box: Box,
  margin: number,
  sessions: PipelineSessions,
): Promise<PlateCandidate> {
  const expanded = expandBox(box, margin, image.width, image.height)
  const cropped = cropResize(image, expanded, OCR_WIDTH, OCR_HEIGHT)
  const feed = toOcrTensor(cropped)
  const inputName = sessions.ocr.inputNames[0]!
  const outputs = await sessions.ocr.run({ [inputName]: feed })
  const read = decodeOcr(outputs)
  return { box, read, validation: validate(read.text, read.charProbs) }
}

/** Full pipeline: detect plates, then OCR + validate each candidate. */
export async function extractPlates(
  image: ImageDataLike,
  sessions: PipelineSessions,
  opts: PipelineOptions = {},
): Promise<PipelineResult> {
  const size = opts.detectorSize ?? DETECTOR_DEFAULT_SIZE
  const confThresh = opts.confThresh ?? 0.25
  const margin = opts.cropMargin ?? 0.0
  const maxCandidates = opts.maxCandidates ?? 5

  const t0 = now()
  const lb = letterbox(image, size)
  const feed = toDetectorTensor(lb.image)
  const inputName = sessions.detector.inputNames[0]!
  const outputs = await sessions.detector.run({ [inputName]: feed })
  const first = outputs[sessions.detector.outputNames[0]!] ?? Object.values(outputs)[0]!
  const boxes = decodeDetections(first, lb, image.width, image.height, confThresh)
  const t1 = now()

  // stable order by detector confidence only — carries NO meaning (all plates equal per design)
  const ordered = [...boxes].sort((a, b) => b.score - a.score).slice(0, maxCandidates)
  const candidates: PlateCandidate[] = []
  for (const box of ordered) {
    try {
      candidates.push(await readBox(image, box, margin, sessions))
    } catch {
      // one unreadable candidate must not cost the others (all plates equal):
      // skip it and continue with the remaining boxes
    }
  }
  const t2 = now()

  return {
    candidates,
    timings: { detectMs: t1 - t0, ocrMs: t2 - t1, totalMs: t2 - t0 },
  }
}

/** OCR-only path for a user-drawn region (no detection). */
export async function ocrRegion(
  image: ImageDataLike,
  box: Box,
  sessions: PipelineSessions,
): Promise<PlateCandidate> {
  return readBox(image, box, 0.0, sessions)
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `npm test`
Expected: all suites PASS (~image, detector, ocr, validate, rank, pipeline).

- [ ] **Step 5: Commit**

```bash
git add src/pipeline/pipeline.ts src/pipeline/pipeline.test.ts
git commit -m "feat: pipeline orchestrator with ranking and manual-region path"
```

---

### Task 9: Ground truth + Node eval CLI

**NOTE:** The labeling step (Step 1) must be done in the **main session** (it requires viewing the private photos with the multimodal Read tool and a user sanity-check). If executing via subagents, the orchestrator does Step 1 itself before dispatching the rest of this task.

**Files:**
- Create: `src/node/ort-node.ts`, `src/node/decode.ts`, `scripts/eval.ts`, `eval/expected.example.json`, `eval/expected.json` (gitignored)

**Interfaces:**
- Consumes: `extractPlates`, `PipelineOptions` from Task 8; `normalizePlateText` from Task 6.
- Produces:
  - `loadNodeSession(path: string): Promise<OrtSessionLike>`
  - `decodeImageFile(path: string): Promise<ImageDataLike>` (sharp: EXIF-rotate, ensureAlpha, raw RGBA)
  - `npm run eval [-- --detector 384|512 --ocr xs|s --margin 0.05]` → per-image table + summary; failure crops dumped to `eval/out/`

- [ ] **Step 1: Produce ground-truth labels (main session + user)**

For each of the 24 files in `attachments/`: create a ≤1024px preview with `sips -Z 1024` into the scratchpad, Read it, and record the plate of the photographed (blocked) car — the dominant plate. Five are already known from brainstorming: `20250218_091355.jpg: BNCR788`, `20250319_101557.jpg: KRLM144`, `20250626_090702.jpg: OKXY226`, `20260417_205500.jpg: plate-I`, `20260611_121655.jpg: plate-E`.

Write `eval/expected.json` mapping filename → normalized plate(s) (no spaces). The value is a string (single readable plate) or an array of ALL clearly readable plates in the photo — unordered, all equal priority (user decision: no "blocked car" guessing; the pipeline must surface every readable plate and the human picks). Plates that are cut off or not confidently human-readable are omitted:
```json
{
  "20250218_091355.jpg": "BNCR788",
  "20260203_085449.jpg": ["plate-A", "plate-C", "plate-B"]
}
```
(…all 24 entries.)

Also write committed `eval/expected.example.json`:
```json
{
  "single_plate_photo.jpg": "MXY123",
  "multi_plate_photo.jpg": ["MXY123", "KAB1234"]
}
```

**Gate:** show the user the full filename → plate table and ask them to sanity-check before proceeding.

- [ ] **Step 2: Write `src/node/ort-node.ts`**

```ts
import * as ort from 'onnxruntime-node'
import type { OrtSessionLike, TensorLike } from '../pipeline/types'

export async function loadNodeSession(path: string): Promise<OrtSessionLike> {
  const s = await ort.InferenceSession.create(path)
  return {
    inputNames: s.inputNames,
    outputNames: s.outputNames,
    async run(feeds: Record<string, TensorLike>) {
      const ortFeeds: Record<string, ort.Tensor> = {}
      for (const [k, t] of Object.entries(feeds)) ortFeeds[k] = new ort.Tensor(t.type, t.data, t.dims)
      const res = await s.run(ortFeeds)
      const out: Record<string, TensorLike> = {}
      for (const [k, t] of Object.entries(res)) {
        out[k] = { type: t.type as TensorLike['type'], data: t.data as Float32Array, dims: [...t.dims] }
      }
      return out
    },
  }
}
```

- [ ] **Step 3: Write `src/node/decode.ts`**

```ts
import sharp from 'sharp'
import type { ImageDataLike } from '../pipeline/types'

/** Decode an image file to RGBA, applying EXIF orientation. */
export async function decodeImageFile(path: string): Promise<ImageDataLike> {
  const { data, info } = await sharp(path).rotate().ensureAlpha().raw().toBuffer({ resolveWithObject: true })
  return { data: new Uint8ClampedArray(data.buffer, data.byteOffset, data.byteLength), width: info.width, height: info.height }
}

/** Save an RGBA region as PNG (debugging failed reads). */
export async function saveRegionPng(image: ImageDataLike, path: string): Promise<void> {
  await sharp(Buffer.from(image.data.buffer, image.data.byteOffset, image.data.byteLength), {
    raw: { width: image.width, height: image.height, channels: 4 },
  })
    .png()
    .toFile(path)
}
```

- [ ] **Step 4: Write `scripts/eval.ts`**

```ts
import fs from 'node:fs'
import path from 'node:path'
import { loadNodeSession } from '../src/node/ort-node'
import { decodeImageFile, saveRegionPng } from '../src/node/decode'
import { extractPlates } from '../src/pipeline/pipeline'
import { normalizePlateText } from '../src/pipeline/validate'
import { crop, expandBox } from '../src/pipeline/image'

function arg(name: string, def: string): string {
  const i = process.argv.indexOf(`--${name}`)
  return i >= 0 && process.argv[i + 1] ? process.argv[i + 1]! : def
}

const detectorSize = Number(arg('detector', '384'))
const ocrName = arg('ocr', 'xs') === 's' ? 'cct_s_v2_global' : 'cct_xs_v2_global'
const cropMargin = Number(arg('margin', '0'))

const expectedPath = 'eval/expected.json'
if (!fs.existsSync(expectedPath)) {
  console.error('eval/expected.json missing — see eval/expected.example.json')
  process.exit(1)
}
// value: one plate or an unordered array of ALL readable plates (equal priority)
const expected: Record<string, string | string[]> = JSON.parse(fs.readFileSync(expectedPath, 'utf8'))

const detector = await loadNodeSession(`public/models/yolo-v9-t-${detectorSize}-license-plates-end2end.onnx`)
const ocr = await loadNodeSession(`public/models/${ocrName}.onnx`)
fs.mkdirSync('eval/out', { recursive: true })

let platesExpected = 0
let platesFound = 0
let photosFull = 0
let totalMs = 0
const results: object[] = []

console.log(`config: detector=${detectorSize} ocr=${ocrName} margin=${cropMargin}\n`)
for (const [file, want] of Object.entries(expected)) {
  const plates = (Array.isArray(want) ? want : [want]).map(normalizePlateText)
  const image = await decodeImageFile(path.join('attachments', file))
  const res = await extractPlates(image, { detector, ocr }, { detectorSize, cropMargin })
  totalMs += res.timings.totalMs
  const got = res.candidates.map((c) => c.validation.plate)
  const missed = plates.filter((p) => !got.includes(p))
  const extras = got.filter((g) => !plates.includes(g))
  platesExpected += plates.length
  platesFound += plates.length - missed.length

  let status: string
  if (missed.length === 0) {
    status = 'PASS'
    photosFull++
  } else if (res.candidates.length === 0) {
    status = 'NONE'
  } else {
    status = `MISS ${missed.join(',')}`
  }
  if (missed.length && res.candidates[0]) {
    const b = expandBox(res.candidates[0].box, 0.1, image.width, image.height)
    await saveRegionPng(crop(image, b), `eval/out/${file.replace('.jpg', '')}-crop.png`)
  }
  results.push({ file, want: plates, got, status, ms: Math.round(res.timings.totalMs) })
  console.log(
    `${status.padEnd(28)} ${file}  found=${plates.length - missed.length}/${plates.length}${extras.length ? `  extra=${extras.join(',')}` : ''}  ${Math.round(res.timings.totalMs)}ms`,
  )
}

const n = Object.keys(expected).length
console.log(
  `\nplates found: ${platesFound}/${platesExpected}   photos fully covered: ${photosFull}/${n}   avg ${Math.round(totalMs / n)}ms/image`,
)
// "extra" reads are informational, not failures: often a real plate we deemed
// unreadable during labeling, or a misread of one — inspect, don't panic.
fs.writeFileSync('eval/out/results.json', JSON.stringify({ config: { detectorSize, ocrName, cropMargin }, results }, null, 2))
```

- [ ] **Step 5: Run the eval — baseline measurement**

Run: `npm run eval`
Expected: a table of 24 rows, a summary line like `top-1: N/24`, and `eval/out/results.json` written. **Record the baseline plates-found score in the commit message.** Any score is acceptable for this task — improving it is Task 12. If the script crashes (not low accuracy — crashes), debug before committing: most likely tensor name/shape mismatches; re-run `npm run probe` and compare.

- [ ] **Step 6: Commit**

```bash
git add src/node scripts/eval.ts eval/expected.example.json
git commit -m "feat: node eval CLI over labeled samples (baseline: <N>/34 plates)"
```

---

### Task 10: Browser app — core flow

**Files:**
- Create: `src/web/ort-web.ts`, `src/web/decode.ts`, `src/web/app.ts`, `src/web/ui.css`, `public/manifest.webmanifest`, `scripts/make-icon.ts`
- Modify: `index.html`, `src/main.ts`, `vite.config.ts`

**Interfaces:**
- Consumes: `extractPlates`, `PipelineSessions` (Task 8), image helpers (Task 3).
- Produces:
  - `loadWebSession(url: string): Promise<OrtSessionLike>` (WASM from `/ort/`, threads when crossOriginIsolated)
  - `fileToImageData(file: Blob): Promise<ImageDataLike>`
  - `cropToDataUrl(image: ImageDataLike, box: Box): string` (thumbnail for Task 11's menu)
  - Working page: photo/gallery buttons → result card (display text, badges, editable field, latency); state hooks for Task 11 (`renderCandidates` stub called with full candidate list).

- [ ] **Step 1: Write `src/web/ort-web.ts`**

```ts
// wasm-only build (we never use webgpu/jsep) + Vite ?url imports so the runtime
// files are served in dev and bundled in prod from OUR origin (never a CDN).
// NOTE: do NOT copy ort files into public/ and point wasmPaths there — Vite
// refuses dynamic import() of modules under public/ (500), which breaks ort
// with "no available backend" in every real browser.
import * as ort from 'onnxruntime-web/wasm'
import ortWasmUrl from 'onnxruntime-web/dist/ort-wasm-simd-threaded.wasm?url'
import ortMjsUrl from 'onnxruntime-web/dist/ort-wasm-simd-threaded.mjs?url'
import type { OrtSessionLike, TensorLike } from '../pipeline/types'

let configured = false

export async function loadWebSession(url: string): Promise<OrtSessionLike> {
  if (!configured) {
    ort.env.wasm.wasmPaths = { wasm: ortWasmUrl, mjs: ortMjsUrl }
    ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1
    configured = true
  }
  const s = await ort.InferenceSession.create(url, { executionProviders: ['wasm'] })
  return {
    inputNames: s.inputNames,
    outputNames: s.outputNames,
    async run(feeds: Record<string, TensorLike>) {
      const ortFeeds: Record<string, ort.Tensor> = {}
      for (const [k, t] of Object.entries(feeds)) ortFeeds[k] = new ort.Tensor(t.type, t.data, t.dims)
      const res = await s.run(ortFeeds)
      const out: Record<string, TensorLike> = {}
      for (const [k, t] of Object.entries(res)) {
        out[k] = { type: t.type as TensorLike['type'], data: t.data as Float32Array, dims: [...t.dims] }
      }
      return out
    },
  }
}
```

- [ ] **Step 2: Write `src/web/decode.ts`**

```ts
import type { Box, ImageDataLike } from '../pipeline/types'
import { crop, expandBox } from '../pipeline/image'

/** Decode a photo file to RGBA ImageData, honoring EXIF orientation. */
export async function fileToImageData(file: Blob): Promise<ImageDataLike> {
  let bmp: ImageBitmap
  try {
    bmp = await createImageBitmap(file, { imageOrientation: 'from-image' })
  } catch {
    bmp = await createImageBitmap(file) // older Safari: option object unsupported
  }
  const canvas = document.createElement('canvas')
  canvas.width = bmp.width
  canvas.height = bmp.height
  const ctx = canvas.getContext('2d', { willReadFrequently: true })!
  ctx.drawImage(bmp, 0, 0)
  bmp.close()
  return ctx.getImageData(0, 0, canvas.width, canvas.height)
}

/** JPEG data-url thumbnail of a box region (for the candidate menu). */
export function cropToDataUrl(image: ImageDataLike, box: Box): string {
  const c = crop(image, expandBox(box, 0.15, image.width, image.height))
  const canvas = document.createElement('canvas')
  canvas.width = c.width
  canvas.height = c.height
  canvas.getContext('2d')!.putImageData(new ImageData(c.data, c.width, c.height), 0, 0)
  return canvas.toDataURL('image/jpeg', 0.8)
}
```

- [ ] **Step 3: Write `src/web/app.ts`** (core flow; candidate menu + manual crop are Task 11 — `renderCandidates` is a stub here)

```ts
import type { ImageDataLike, PlateCandidate } from '../pipeline/types'
import { extractPlates, type PipelineSessions, type PipelineResult } from '../pipeline/pipeline'
import { loadWebSession } from './ort-web'
import { fileToImageData } from './decode'

const $ = <T extends HTMLElement>(sel: string) => document.querySelector(sel) as T

let sessions: PipelineSessions | null = null
let currentImage: ImageDataLike | null = null
let lastResult: PipelineResult | null = null

async function ensureSessions(): Promise<PipelineSessions> {
  if (sessions) return sessions
  setStatus('Loading models… (first time only)')
  const [detector, ocr] = await Promise.all([
    loadWebSession('/models/yolo-v9-t-384-license-plates-end2end.onnx'),
    loadWebSession('/models/cct_xs_v2_global.onnx'),
  ])
  sessions = { detector, ocr }
  setStatus('')
  return sessions
}

function setStatus(msg: string) {
  $('#status').textContent = msg
}

/** Render plate text with low-confidence characters flagged (spec: flag uncertain chars). */
function renderPlateText(el: HTMLElement, display: string, charProbs: number[]) {
  el.innerHTML = ''
  let idx = 0 // corrections replace chars in place, so probs align with non-space display chars
  for (const ch of display) {
    if (ch === ' ') {
      el.appendChild(document.createTextNode(' '))
      continue
    }
    const span = document.createElement('span')
    span.textContent = ch
    if ((charProbs[idx] ?? 1) < 0.5) span.className = 'lowconf'
    el.appendChild(span)
    idx++
  }
}

export function showCandidate(c: PlateCandidate) {
  $('#result').hidden = false
  renderPlateText($('#plate-text'), c.validation.display, c.read.charProbs)
  ;($('#plate-edit') as HTMLInputElement).value = c.validation.plate
  const badges: string[] = []
  if (c.validation.formatValid) badges.push('✓ German format')
  else badges.push('⚠ unrecognized format')
  if (c.validation.corrections.length) badges.push(`✏ ${c.validation.corrections.length} corrected`)
  if (c.read.region) badges.push(`🌍 ${c.read.region}`)
  $('#badges').innerHTML = badges.map((b) => `<span class="badge">${b}</span>`).join('')
  const uml = $('#umlaut')
  if (c.validation.umlautSuggestion) {
    uml.hidden = false
    uml.textContent = `District with umlaut? ${c.validation.umlautSuggestion}`
  } else {
    uml.hidden = true
  }
}

// Stub — Task 11 replaces this with the selection menu + manual crop entry point.
export function renderCandidates(_image: ImageDataLike, result: PipelineResult) {
  const first = result.candidates[0]
  if (first) showCandidate(first)
}

async function handleFile(file: File) {
  // reset stale state up front so a failure never shows a previous photo's read
  $('#result').hidden = true
  $('#no-plate').hidden = true
  $('#timings').textContent = ''
  try {
    const s = await ensureSessions()
    setStatus('Reading photo…')
    currentImage = await fileToImageData(file)
    setStatus('Looking for plates…')
    const result = await extractPlates(currentImage, s)
    lastResult = result
    setStatus('')
    $('#timings').textContent =
      `detect ${Math.round(result.timings.detectMs)} ms · read ${Math.round(result.timings.ocrMs)} ms`
    $('#no-plate').hidden = result.candidates.length > 0
    $('#result').hidden = result.candidates.length === 0
    if (result.candidates.length > 0) renderCandidates(currentImage, result)
  } catch (err) {
    setStatus(`Error: ${err instanceof Error ? err.message : String(err)}`)
  }
}

export function getCurrentImage() {
  return currentImage
}
export function getLastResult() {
  return lastResult
}
export function getSessions() {
  return sessions
}

for (const id of ['camera-input', 'gallery-input']) {
  $(`#${id}`).addEventListener('change', (e) => {
    const file = (e.target as HTMLInputElement).files?.[0]
    if (file) void handleFile(file)
    ;(e.target as HTMLInputElement).value = ''
  })
}
```

- [ ] **Step 4: Rewrite `index.html`, `src/main.ts`, add `src/web/ui.css`**

`index.html`:
```html
<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <meta name="theme-color" content="#111418" />
    <link rel="manifest" href="/manifest.webmanifest" />
    <link rel="icon" href="/icon-192.png" />
    <title>Plate Reader</title>
  </head>
  <body>
    <main id="app">
      <h1>Plate Reader</h1>
      <p class="hint">Take a photo of the car — plates are read on your phone; nothing is uploaded.</p>
      <div class="actions">
        <label class="btn primary">
          📷 Take photo
          <input id="camera-input" type="file" accept="image/*" capture="environment" hidden />
        </label>
        <label class="btn">
          🖼 From gallery
          <input id="gallery-input" type="file" accept="image/*" hidden />
        </label>
      </div>
      <p id="status" role="status"></p>
      <section id="result" hidden>
        <div id="plate-text" class="plate"></div>
        <div id="badges"></div>
        <button id="umlaut" class="chip" hidden></button>
        <label class="edit-label">Fix if needed:
          <input id="plate-edit" autocapitalize="characters" autocomplete="off" />
        </label>
        <div id="candidates"></div>
      </section>
      <section id="no-plate" hidden>
        <p>No plate found. Try again closer, or draw a box around the plate:</p>
        <canvas id="crop-canvas"></canvas>
      </section>
      <p id="timings" class="muted"></p>
    </main>
    <script type="module" src="/src/main.ts"></script>
  </body>
</html>
```

`src/main.ts`:
```ts
import './web/ui.css'
import './web/app'
```

`src/web/ui.css`:
```css
:root { color-scheme: dark light; font-family: system-ui, sans-serif; }
body { margin: 0; background: #111418; color: #e8eaed; }
#app { max-width: 480px; margin: 0 auto; padding: 16px; }
h1 { font-size: 1.3rem; }
.hint, .muted { color: #9aa0a6; font-size: 0.9rem; }
.actions { display: flex; gap: 12px; margin: 16px 0; }
.btn { flex: 1; text-align: center; padding: 14px; border-radius: 12px; background: #2a2f36; cursor: pointer; user-select: none; }
.btn.primary { background: #1a73e8; }
.plate { font-family: ui-monospace, monospace; font-size: 2.2rem; letter-spacing: 0.12em; text-align: center;
  background: #fff; color: #111; border: 3px solid #111; border-radius: 8px; padding: 10px 6px; margin: 12px 0; }
.plate .lowconf { color: #d93025; text-decoration: underline wavy #d93025; }
.badge { display: inline-block; background: #2a2f36; border-radius: 999px; padding: 4px 10px; margin: 2px 4px 2px 0; font-size: 0.85rem; }
.chip { background: #103d2e; color: #7ee2b8; border: none; border-radius: 999px; padding: 6px 12px; margin: 6px 0; cursor: pointer; }
.edit-label { display: block; margin: 10px 0; font-size: 0.9rem; color: #9aa0a6; }
#plate-edit { display: block; width: 100%; box-sizing: border-box; font-family: ui-monospace, monospace; font-size: 1.2rem;
  padding: 8px; margin-top: 4px; border-radius: 8px; border: 1px solid #3c4043; background: #1b1f24; color: #e8eaed; }
#candidates { margin-top: 10px; }
.cand { display: flex; align-items: center; gap: 10px; width: 100%; padding: 8px; margin: 6px 0; border-radius: 10px;
  border: 1px solid #3c4043; background: #1b1f24; color: #e8eaed; cursor: pointer; font-family: ui-monospace, monospace; font-size: 1.05rem; }
.cand img { height: 36px; border-radius: 4px; }
.cand.selected { border-color: #1a73e8; }
#crop-canvas { width: 100%; border-radius: 8px; touch-action: none; }
```

- [ ] **Step 5: PWA manifest + icon + dev middleware**

`public/manifest.webmanifest`:
```json
{
  "name": "Plate Reader",
  "short_name": "Plates",
  "start_url": "/",
  "display": "standalone",
  "background_color": "#111418",
  "theme_color": "#111418",
  "icons": [
    { "src": "/icon-192.png", "sizes": "192x192", "type": "image/png" },
    { "src": "/icon-512.png", "sizes": "512x512", "type": "image/png" }
  ]
}
```

`scripts/make-icon.ts`:
```ts
import sharp from 'sharp'

const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="512" height="512">
  <rect width="512" height="512" rx="96" fill="#1a73e8"/>
  <rect x="66" y="196" width="380" height="120" rx="16" fill="#fff" stroke="#111" stroke-width="10"/>
  <text x="256" y="282" font-family="monospace" font-size="72" font-weight="bold" fill="#111" text-anchor="middle">BN·CR 7</text>
</svg>`
for (const size of [192, 512]) {
  await sharp(Buffer.from(svg)).resize(size, size).png().toFile(`public/icon-${size}.png`)
}
console.log('icons written')
```

Run: `tsx scripts/make-icon.ts`
Expected: `public/icon-192.png`, `public/icon-512.png` created.

Modify `vite.config.ts` — add the dev-only middleware serving `attachments/` and `eval/` (for the Task 11 eval page), and the eval.html build input:

```ts
import { defineConfig, type Plugin } from 'vite'
import fs from 'node:fs'
import path from 'node:path'

/** Dev-only: serve sample photos + ground truth for /eval.html (both are gitignored, never bundled). */
function serveLocalData(): Plugin {
  return {
    name: 'serve-local-data',
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        const url = (req.url ?? '').split('?')[0]!
        const map: [string, string][] = [
          ['/attachments/', 'attachments'],
          ['/eval-data/', 'eval'],
        ]
        for (const [prefix, dir] of map) {
          if (!url.startsWith(prefix)) continue
          const root = path.join(process.cwd(), dir)
          const p = path.join(root, decodeURIComponent(url.slice(prefix.length)))
          // anchored guard: plain startsWith(root) would pass /attachments-evil/*
          if (p !== root && !p.startsWith(root + path.sep)) {
            res.statusCode = 403
            return res.end()
          }
          if (fs.existsSync(p) && fs.statSync(p).isFile()) {
            res.setHeader('Content-Type', p.endsWith('.json') ? 'application/json' : 'image/jpeg')
            fs.createReadStream(p).pipe(res)
            return
          }
          res.statusCode = 404
          return res.end()
        }
        next()
      })
    },
  }
}

export default defineConfig({
  optimizeDeps: { exclude: ['onnxruntime-web'] },
  plugins: [serveLocalData()],
  server: {
    headers: {
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    },
  },
})
```

(No `build.rollupOptions.input` yet — `eval.html` does not exist until Task 11, which adds the two-page input.)

- [ ] **Step 6: Manual verification in desktop browser**

Run: `npm run dev` and open `http://localhost:5173`.
- Pick a sample photo from `attachments/` via "From gallery".
- Expected: status transitions, then a plate result card with a plausible read, badges, and a latency line. Console free of errors.
- If session creation fails with a fetch/MIME error on `.wasm`: confirm `public/ort/` contains wasm files and `wasmPaths` is `/ort/`.

- [ ] **Step 7: Check types compile and tests still pass**

Run: `npx tsc --noEmit && npm test`
Expected: no type errors; all tests pass.

- [ ] **Step 8: Commit**

```bash
git add index.html src/main.ts src/web public/manifest.webmanifest public/icon-192.png public/icon-512.png scripts/make-icon.ts vite.config.ts
git commit -m "feat: browser app core flow with on-device inference"
```

---

### Task 11: Candidate menu, manual crop, browser eval page

**Files:**
- Create: `eval.html`, `src/web/eval.ts`, `src/web/manual-crop.ts`
- Modify: `src/web/app.ts` (replace `renderCandidates` stub, wire manual crop), `vite.config.ts` (two-page build input, per Task 10 note)

**Interfaces:**
- Consumes: `ocrRegion` (Task 8), `cropToDataUrl` (Task 10), app state getters (Task 10).
- Produces: multi-plate pop-up selection menu; drag-a-box manual crop fallback; `/eval.html` dev scoreboard (works on the phone → real on-device latency numbers).

- [ ] **Step 1: Replace `renderCandidates` in `src/web/app.ts`**

Replace the stub with:

```ts
import { cropToDataUrl } from './decode' // add to existing imports

export function renderCandidates(image: ImageDataLike, result: PipelineResult) {
  const container = $('#candidates')
  container.innerHTML = ''
  if (result.candidates.length === 1) {
    showCandidate(result.candidates[0]!)
    return
  }
  // multiple plates: ALL equal priority (design decision) — list them, human picks
  $('#result').hidden = false
  $('#plate-text').textContent = ''
  // reset single-candidate leftovers so a prior photo's state can't leak in
  $('#badges').innerHTML = ''
  const uml = $('#umlaut') as HTMLButtonElement
  uml.hidden = true
  uml.onclick = null
  ;($('#plate-edit') as HTMLInputElement).value = ''
  const label = document.createElement('p')
  label.className = 'muted'
  label.textContent = `${result.candidates.length} plates found — tap the one you need:`
  container.appendChild(label)
  result.candidates.forEach((c) => {
    const btn = document.createElement('button')
    btn.className = 'cand'
    const img = document.createElement('img')
    img.src = cropToDataUrl(image, c.box)
    img.alt = ''
    btn.appendChild(img)
    btn.appendChild(document.createTextNode(c.validation.display))
    btn.addEventListener('click', () => {
      container.querySelectorAll('.cand').forEach((el) => el.classList.remove('selected'))
      btn.classList.add('selected')
      showCandidate(c)
    })
    container.appendChild(btn)
  })
}
```

- [ ] **Step 2: Write `src/web/manual-crop.ts`**

```ts
import type { Box, ImageDataLike } from '../pipeline/types'

/**
 * Let the user drag a rectangle on the photo; resolves with the box in image coordinates.
 * Renders the image into the given canvas (downscaled to fit) and tracks one pointer drag.
 */
export function pickRegion(canvas: HTMLCanvasElement, image: ImageDataLike, onPick: (box: Box) => void): void {
  const maxW = Math.min(image.width, 1000)
  const scale = maxW / image.width
  canvas.width = Math.round(image.width * scale)
  canvas.height = Math.round(image.height * scale)
  const ctx = canvas.getContext('2d')!

  const full = document.createElement('canvas')
  full.width = image.width
  full.height = image.height
  full.getContext('2d')!.putImageData(new ImageData(image.data, image.width, image.height), 0, 0)
  const redraw = () => ctx.drawImage(full, 0, 0, canvas.width, canvas.height)
  redraw()

  let start: { x: number; y: number } | null = null

  const pos = (e: PointerEvent) => {
    const r = canvas.getBoundingClientRect()
    return {
      x: ((e.clientX - r.left) / r.width) * canvas.width,
      y: ((e.clientY - r.top) / r.height) * canvas.height,
    }
  }

  canvas.onpointerdown = (e) => {
    canvas.setPointerCapture(e.pointerId)
    start = pos(e)
  }
  canvas.onpointermove = (e) => {
    if (!start) return
    const p = pos(e)
    redraw()
    ctx.strokeStyle = '#1a73e8'
    ctx.lineWidth = 3
    ctx.strokeRect(start.x, start.y, p.x - start.x, p.y - start.y)
  }
  canvas.onpointercancel = () => {
    start = null
    redraw()
  }
  canvas.onpointerup = (e) => {
    if (!start) return
    const p = pos(e)
    const box: Box = {
      x1: Math.min(start.x, p.x) / scale,
      y1: Math.min(start.y, p.y) / scale,
      x2: Math.max(start.x, p.x) / scale,
      y2: Math.max(start.y, p.y) / scale,
      score: 1,
    }
    start = null
    if (box.x2 - box.x1 > 8 && box.y2 - box.y1 > 4) onPick(box)
  }
}
```

- [ ] **Step 3: Wire manual crop into `src/web/app.ts`**

In `handleFile`, inside the `result.candidates.length === 0` branch (i.e. after `$('#no-plate').hidden = ...` lines), add:

```ts
if (result.candidates.length === 0 && currentImage) {
  const canvas = $('#crop-canvas') as HTMLCanvasElement
  pickRegion(canvas, currentImage, (box) => {
    void (async () => {
      const c = await ocrRegion(currentImage!, box, s)
      $('#no-plate').hidden = true
      $('#result').hidden = false
      $('#candidates').innerHTML = ''
      showCandidate(c)
    })()
  })
}
```

Add imports: `import { ocrRegion } from '../pipeline/pipeline'` and `import { pickRegion } from './manual-crop'`.
Also wire the umlaut chip (in `showCandidate`, after setting its text): clicking `#umlaut` sets `($('#plate-edit') as HTMLInputElement).value = c.validation.umlautSuggestion!`.

- [ ] **Step 4: Write `eval.html` + `src/web/eval.ts`, switch vite input to two pages**

`eval.html`:
```html
<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Eval — Plate Reader</title>
    <style>
      body { font-family: system-ui; background: #111418; color: #e8eaed; margin: 16px; }
      #summary { font-size: 1.2rem; margin-bottom: 12px; }
      .card { display: flex; gap: 10px; align-items: center; border-bottom: 1px solid #2a2f36; padding: 6px 0; }
      .card img { height: 48px; border-radius: 4px; }
      .pass { color: #7ee2b8; } .fail { color: #f28b82; }
      .mono { font-family: ui-monospace, monospace; }
    </style>
  </head>
  <body>
    <h1>Pipeline eval</h1>
    <div id="summary">loading models…</div>
    <div id="cards"></div>
    <script type="module" src="/src/web/eval.ts"></script>
  </body>
</html>
```

`src/web/eval.ts`:
```ts
import { extractPlates } from '../pipeline/pipeline'
import { normalizePlateText } from '../pipeline/validate'
import { loadWebSession } from './ort-web'
import { fileToImageData, cropToDataUrl } from './decode'

const summary = document.getElementById('summary')!
const cards = document.getElementById('cards')!

const [detector, ocr] = await Promise.all([
  loadWebSession('/models/yolo-v9-t-384-license-plates-end2end.onnx'),
  loadWebSession('/models/cct_xs_v2_global.onnx'),
])
const expected: Record<string, string> = await (await fetch('/eval-data/expected.json')).json()

let done = 0
let top1 = 0
let totalMs = 0
const n = Object.keys(expected).length

for (const [file, want] of Object.entries(expected)) {
  const blob = await (await fetch(`/attachments/${file}`)).blob()
  const image = await fileToImageData(blob)
  const res = await extractPlates(image, { detector, ocr })
  const got = res.candidates[0]?.validation.plate ?? '—'
  const pass = got === normalizePlateText(want)
  done++
  if (pass) top1++
  totalMs += res.timings.totalMs

  const card = document.createElement('div')
  card.className = 'card'
  const thumb = res.candidates[0] ? `<img src="${cropToDataUrl(image, res.candidates[0].box)}" alt="">` : ''
  card.innerHTML = `${thumb}<span class="${pass ? 'pass' : 'fail'}">${pass ? 'PASS' : 'FAIL'}</span>
    <span class="mono">${file}</span> <span class="mono">want ${want} · got ${got}</span>
    <span>${Math.round(res.timings.totalMs)}ms</span>`
  cards.appendChild(card)
  summary.textContent = `top-1 ${top1}/${done} of ${n} · avg ${Math.round(totalMs / done)}ms`
}
```

In `vite.config.ts`, set the two-page input (per the Task 10 note):
```ts
build: { rollupOptions: { input: { main: 'index.html', eval: 'eval.html' } } },
```

- [ ] **Step 5: Manual verification**

Run: `npm run dev`
- Main page with a multi-car sample (e.g. `20250218_091355.jpg`): candidate menu appears with thumbnails; tapping switches the result.
- `http://localhost:5173/eval.html`: scoreboard fills, final score matches `npm run eval` (same models/config; minor per-image ms differences are fine, mismatched reads are NOT — same pipeline code must give same reads).
- Manual crop: pick a photo where nothing is detected (or temporarily raise `confThresh` to 0.95 to force it), drag a box over a plate, verify a read appears. Revert any temporary change.

- [ ] **Step 6: Types + tests still green**

Run: `npx tsc --noEmit && npm test`
Expected: clean.

- [ ] **Step 7: Commit**

```bash
git add src/web eval.html vite.config.ts
git commit -m "feat: candidate selection menu, manual crop fallback, browser eval page"
```

---

### Task 12: Accuracy iteration to target

**Files:**
- Create: `docs/eval-results.md`
- Modify (only if a lever wins): `src/pipeline/pipeline.ts` defaults / `src/web/app.ts` + `src/web/eval.ts` model URLs

**Interfaces:**
- Consumes: `npm run eval` flags from Task 9.
- Produces: chosen default config (detector size, OCR model, crop margin) applied consistently in `pipeline.ts` defaults, `app.ts`, and `eval.ts`; `docs/eval-results.md` with the matrix and per-failure analysis.

- [ ] **Step 1: Run the config matrix**

```bash
for det in 384 512; do for ocr in xs s; do for m in 0 0.05 0.1; do
  echo "=== det=$det ocr=$ocr margin=$m"
  npm run eval -- --detector $det --ocr $ocr --margin $m | tail -1
done; done; done
```

Record all 12 summary lines in `docs/eval-results.md`.

- [ ] **Step 2: Adopt the winning config**

Apply the best config as the defaults: `DETECTOR_DEFAULT_SIZE` (detector.ts) and/or `cropMargin` default (pipeline.ts), and the model URLs in `app.ts`/`eval.ts` if `s` beats `xs` (weigh: `s` is +2 MB download and ~2× OCR latency — prefer `xs` unless it wins by ≥2 images). Re-run `npm run eval` to confirm the recorded score reproduces with defaults.

- [ ] **Step 3: Analyze remaining failures**

For each photo with missed plates: open `eval/out/<name>-crop.png`. Classify: (a) detector missed the plate entirely (fewer candidates than expected plates), (c) OCR misread characters (plate detected, wrong text). Write the classification into `docs/eval-results.md`.

- [ ] **Step 4: If plates found < 31/34, apply targeted levers (one at a time, re-run eval after each)**

In order of expected payoff, matched to failure class. Each lever is an opt-in `PipelineOptions` flag evaluated via a new `--flag` in `scripts/eval.ts`; adopt as default only if it wins ≥2 plates without losing any.

- Class (c) small/blurry crops: use a small margin only for small boxes — in `readBox`, `const m = box.x2 - box.x1 < 64 ? Math.max(margin, 0.05) : margin`.
- Class (c) low contrast (night shots): per-channel linear contrast stretch of the OCR crop, `PipelineOptions.normalizeCrop`. Add to `image.ts`:

```ts
/** Per-channel linear min→max contrast stretch (alpha untouched). */
export function stretchContrast(im: ImageDataLike): ImageDataLike {
  const data = new Uint8ClampedArray(im.data)
  for (let c = 0; c < 3; c++) {
    let min = 255
    let max = 0
    for (let i = c; i < data.length; i += 4) {
      const v = data[i]!
      if (v < min) min = v
      if (v > max) max = v
    }
    const range = max - min
    if (range < 10) continue
    for (let i = c; i < data.length; i += 4) data[i] = ((data[i]! - min) * 255) / range
  }
  return { data, width: im.width, height: im.height }
}
```

Apply in `readBox` after `cropResize`, guarded by the option. Unit-test with a synthetic low-contrast image (values 100–150 stretch to 0–255).

- Class (c) tilted plates (the spec's "deskew if tilted"): rotation sweep, `PipelineOptions.rotationSweep: number[]` (e.g. `[-10, -5, 0, 5, 10]`). OCR the crop at each angle and keep the read with the highest mean charProb. Add to `image.ts`:

```ts
/** Rotate around center by deg (bilinear, gray 114 fill). */
export function rotateImage(im: ImageDataLike, deg: number): ImageDataLike {
  const rad = (-deg * Math.PI) / 180
  const cos = Math.cos(rad)
  const sin = Math.sin(rad)
  const cx = im.width / 2
  const cy = im.height / 2
  const data = new Uint8ClampedArray(im.data.length)
  for (let y = 0; y < im.height; y++) {
    for (let x = 0; x < im.width; x++) {
      const sx = cos * (x - cx) - sin * (y - cy) + cx
      const sy = sin * (x - cx) + cos * (y - cy) + cy
      const o = (y * im.width + x) * 4
      if (sx < 0 || sy < 0 || sx > im.width - 1 || sy > im.height - 1) {
        data[o] = 114
        data[o + 1] = 114
        data[o + 2] = 114
        data[o + 3] = 255
        continue
      }
      const x0 = Math.floor(sx)
      const y0 = Math.floor(sy)
      const x1 = Math.min(x0 + 1, im.width - 1)
      const y1 = Math.min(y0 + 1, im.height - 1)
      const wx = sx - x0
      const wy = sy - y0
      for (let c = 0; c < 4; c++) {
        const top = im.data[(y0 * im.width + x0) * 4 + c]! * (1 - wx) + im.data[(y0 * im.width + x1) * 4 + c]! * wx
        const bot = im.data[(y1 * im.width + x0) * 4 + c]! * (1 - wx) + im.data[(y1 * im.width + x1) * 4 + c]! * wx
        data[o + c] = Math.round(top * (1 - wy) + bot * wy)
      }
    }
  }
  return { data, width: im.width, height: im.height }
}
```

In `readBox`: when the option is set, crop once at `expandBox(box, Math.max(margin, 0.15), …)` at native resolution (`crop`, not `cropResize`), then for each angle: rotate → `resizeBilinear` to 128×64 → OCR; keep the best mean charProb. Costs one OCR run per angle (~each is fast); eval decides if it pays.

- Class (a) detector missed: try `--detector 512`; adopt if it wins (same download size, ~1.7× detect latency — fine for single-shot).

Stop when plates found ≥ 31/34 or all levers are exhausted; document the end state either way. Do NOT chase 24/24 with image-specific hacks — a photo can legitimately be unreadable; note it and move on.

- [ ] **Step 5: Commit**

```bash
git add docs/eval-results.md src/pipeline src/web
git commit -m "feat: tune pipeline config from eval matrix (<final>/34 plates)"
```

---

### Task 13: On-phone smoke test (with user)

**Files:**
- Modify: `README.md` (record results)

**Interfaces:**
- Consumes: everything.
- Produces: verified on-phone behavior + real latency numbers; punch list for follow-ups.

- [ ] **Step 1: Start the LAN dev server and get the URL**

```bash
npm run dev
ipconfig getifaddr en0
```
Give the user the URL `http://<that-ip>:5173` (phone must be on the same Wi-Fi).

- [ ] **Step 2: User walks through on their phone (both iOS and Android if available)**

Checklist to relay:
1. Open the URL → app loads, models load (watch first-load time).
2. "Take photo" opens the camera; photograph any parked car → plate result appears; note the latency line.
3. "From gallery" with an existing photo works.
4. Multi-car photo → candidate menu appears, taps work.
5. `http://<ip>:5173/eval.html` → full-set score + **on-phone avg ms** (the real performance answer).
6. iOS Safari: Share → Add to Home Screen → opens standalone with icon.

- [ ] **Step 3: Record results in README under a "Status" heading**

Note: date, phone models tested, eval score on device, avg latency, any UI glitches found (as a punch list, not fixes).

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "docs: on-phone smoke test results"
```

---

### Task 14: Result UI redesign — minimal (user decisions from first on-phone test)

**User decisions:** app must stay minimal — photo in, plates out, nothing else.
Photo shown with detection rectangles; plates as German-style cards; country
shown; edit field stays; umlauts auto-applied (not suggested); everything else
removed; models preload silently at startup; no manual steps ever.

**Files:**
- Modify: `src/web/app.ts`, `src/web/ui.css`, `index.html`, `src/pipeline/validate.ts`, `src/pipeline/validate.test.ts`, `src/pipeline/types.ts`
- Create: `src/web/photo-view.ts` (photo + rectangles overlay)
- Delete: `src/web/manual-crop.ts`; `ocrRegion` in `src/pipeline/pipeline.ts` + its test (no consumer)

**Requirements (acceptance criteria):**
1. After every read: show the decoded photo (EXIF-corrected ImageData,
   downscaled to display width ≤ 1000px) with a rounded rectangle around
   EVERY detected plate, numbered when more than one.
2. Below the photo, one card per plate — uniform for 1 or N. German plate
   style matching a real plate's anatomy and proportions (user-supplied
   reference, revised):
   - exact aspect-ratio 520/110; blue EU band (#003399) left with star
     circle + white "D";
   - after the district letters: TWO GENERIC SEAL CIRCLES stacked
     vertically (top: green disc, bottom: silver/grey disc with a faint
     generic emblem tint — no detailed artwork), each ≈ 32% of plate
     height, small gap, the stack ≈ one character width;
   - FE-Schrift characters ≈ 65-72% of plate height, tight tracking,
     modest group gaps — the text must dominate the plate face like the
     real thing, not float small inside it; (sanctioned exception:
     8-glyph plates drop to ≈58% via a shrink tier — the single bundled
     font can't fit 8 glyphs + seals at full size, mirroring real plates'
     switch to Engschrift);
   - while the in-place input is focused (editing), the seal stack may
     hide so the input spans the face; it returns on blur.
   Tapping a card (or its rectangle) selects it (highlight both).
3. Selected plate shows the edit field (kept). Country handling (user
   decision, revised): plates matching the German format (rule 'DE') render
   in the German plate style and may show "Country: Germany"; any other
   read renders as a PLAIN text card (no EU band, no plate styling) with NO
   country line — the app never guesses a country for non-German reads.
   NOTHING else: no badges, no umlaut chip, no uncertainty marks, no
   latency line, no model-loading copy.
4. Umlauts auto-apply (validator change): when the matched district has an
   entry in UMLAUT_DISTRICTS (which post-TU/MU-fix contains only
   collision-free codes), `plate` and `display` use the umlaut form
   automatically; `raw` keeps the literal read. Remove the
   `umlautSuggestion` field from PlateValidation; update tests: the
   TOLAB123 case asserts plate `TÖLAB123` / display `TÖL AB 123`; the TU/MU
   test asserts plate stays `TUXY1234`/`MUAB123` (no mapping entries).
5. Manual crop fully removed (files, element, wiring, pipeline export,
   test). "No plate found — try again closer." is the complete no-result UX.
6. Models preload silently at app startup via the shared single-flight
   promise; `handleFile` awaits it under the normal "Reading photo…" status;
   a failed preload retries on next use (never cache a rejection).
7. `npx tsc --noEmit` clean; `npm test` green (counts change with removed
   ocrRegion test + updated validator tests); `npm run build` clean;
   headless check (.superpowers/sdd/browser-repro.mjs) still passes.

8. Plate authenticity (user feedback on first render): the card must look
   like a REAL German plate:
   - Bundle an FE-Schrift digitization (the actual German plate typeface) as
     a self-hosted font file (committed asset + @font-face; never a CDN).
     License due diligence REQUIRED: only bundle a file whose license
     permits redistribution; document source + license in the report and a
     LICENSES.md entry. If no cleanly-licensed FE-Schrift is found, fall
     back to a free DIN 1451 Mittelschrift (the pre-1994 plate font, e.g.
     Peter Wiegel's free digitization) and say so.
   - Proportions near the real 520:110; blue EU band with circle of stars
     (tiny inline SVG) above white "D".
9. In-plate editing (user request): remove the separate edit field — the
   selected plate card's text IS the editable input (transparent input or
   contenteditable styled with the plate font, uppercase, grouped). Tap the
   plate, edit in place. Normalization (uppercase, charset) still applies.

10. App-like layout, no page scroll (user requirement): viewport-locked
    shell — `body { overflow: hidden; height: 100dvh; overscroll-behavior:
    none }`. Zones: photo panel (top, fixed share of height), plate cards
    area (fills middle, scrolls INTERNALLY only if plates overflow), action
    bar with the camera/gallery buttons pinned at the bottom, always
    visible. No whole-page scrolling in any state, including "no plate
    found" and while models preload. Test on narrow viewport (headless
    check may set viewport 390x844) that documentElement scrollHeight ≤
    innerHeight.

11. Photo always shown (user requirement): the photo panel renders the
    decoded photo in EVERY outcome, including zero detections — the
    "No plate found — try again closer." message appears in the plates
    area below the (rectangle-free) photo, never instead of it.

**Step commit:** "feat: minimal result UI — photo overlay + authentic plate cards with in-place edit; app-shell layout; auto-umlaut; remove manual crop"

---

### Task 15: Service worker — offline PWA (user request)

**Files:**
- Create: `public/sw.js` (or vite-plugin-pwa if judged cleaner — decide and
  document), `src/web/sw-register.ts`
- Modify: `index.html` or `src/main.ts` (registration), `vite.config.ts` if
  plugin route chosen

**Requirements:**
1. After first successful visit, the app fully works offline: app shell,
   JS/CSS, manifest, icons, ort wasm/mjs assets, and BOTH model files are
   cached on device (~11 MB + app).
2. Cache-first strategy with versioned cache name; new deploy invalidates
   old cache (activate handler cleans stale caches).
3. The dev middleware routes (/attachments/, /eval-data/) and eval.html are
   NOT precached (dev-only surface).
4. No external requests of any kind (constraint unchanged).
5. Verification: build + preview, headless browser loads page, then a second
   load with network interception (playwright route abort) still renders the
   app shell and creates the OCR session from cache. Document the check.
6. App-feel (user requirement: "not something that opens in a browser"):
   - index.html gains Apple standalone tags: `apple-mobile-web-app-capable`,
     `apple-mobile-web-app-status-bar-style` (black-translucent), and
     `apple-touch-icon` pointing at icon-192.png; manifest keeps
     `display: standalone`.
   - Verify manifest is valid (icons, start_url, scope) so Android shows the
     install prompt and iOS launches chrome-less from the home-screen icon.
7. HTTPS reality (service workers require a secure context):
   - Document in README: offline/install testing needs HTTPS; provide the
     mkcert recipe (mkcert -install; mkcert <lan-ip>; vite server.https with
     the generated cert; phone installs the mkcert root profile once).
   - Deployment note: for the team, serve dist/ from any company-internal
     HTTPS host — static files only.
   - iOS caveat documented: long-unused installed PWAs may have their cache
     evicted; one online visit restores it.
8. Note in README: offline works after first visit; server only needed for
   updates.

9. GitHub Pages deployment (user decision: public repo `plate-reader` on
   their personal account, Pages as the HTTPS install source):
   - vite `base` configurable via env (`BASE_PATH`, default '/'), so the
     app works at https://<user>.github.io/plate-reader/. SW + manifest +
     asset URLs must respect the base (relative paths).
   - `.github/workflows/deploy.yml`: on push to main — checkout, setup
     Node 24, `npm ci`, `bash scripts/fetch-models.sh` (models come from
     upstream GitHub releases at build time; never committed),
     `BASE_PATH=/plate-reader/ npm run build`, upload dist/ via
     actions/upload-pages-artifact, deploy via actions/deploy-pages.
   - eval.html/eval-data are dev-only: exclude eval.html from the deployed
     build (build.rollupOptions.input drops it when BASE_PATH is set, or
     delete from dist in the workflow step) — the public app ships without
     the dev scoreboard.

**Step commit:** "feat: service worker + installable app-feel + GitHub Pages deploy"
