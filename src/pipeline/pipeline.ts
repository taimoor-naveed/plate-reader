import type { Box, ImageDataLike, OcrRead, PlateCandidate } from './types'
import type { OrtSessionLike } from './types'
import { letterbox, cropResize, crop, expandBox, resizeBilinear, stretchContrast, rotateImage } from './image'
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
  /** Class (c) small/blurry crops: widen the margin (to at least 0.05) for boxes narrower than 64px. */
  smallBoxMargin?: boolean
  /** Class (c) low contrast (night shots): per-channel linear contrast stretch of the OCR crop. */
  normalizeCrop?: boolean
  /** Class (c) tilted plates: OCR the crop at each angle (degrees) and keep the highest mean charProb. */
  rotationSweep?: number[]
}

const SMALL_BOX_WIDTH_PX = 64

/** Widen the margin for small boxes (class-c small/blurry crops); a no-op unless smallBoxMargin is set. */
export function resolveMargin(box: Box, margin: number, smallBoxMargin?: boolean): number {
  if (!smallBoxMargin) return margin
  return box.x2 - box.x1 < SMALL_BOX_WIDTH_PX ? Math.max(margin, 0.05) : margin
}

const meanProb = (probs: number[]) => (probs.length ? probs.reduce((a, b) => a + b, 0) / probs.length : 0)

export interface PipelineResult {
  candidates: PlateCandidate[]
  timings: { detectMs: number; ocrMs: number; totalMs: number }
}

const now = () => performance.now()

/** Rotation-sweep OCR: crop once at native resolution, OCR at each angle, keep the highest mean charProb. */
async function readBoxWithRotationSweep(
  image: ImageDataLike,
  box: Box,
  margin: number,
  sessions: PipelineSessions,
  angles: number[],
  normalizeCrop?: boolean,
): Promise<PlateCandidate> {
  const expanded = expandBox(box, Math.max(margin, 0.15), image.width, image.height)
  const native = crop(image, expanded)
  const inputName = sessions.ocr.inputNames[0]!
  let best: OcrRead | null = null
  let bestScore = -Infinity
  for (const deg of angles) {
    const rotated = deg === 0 ? native : rotateImage(native, deg)
    let resized = resizeBilinear(rotated, OCR_WIDTH, OCR_HEIGHT)
    if (normalizeCrop) resized = stretchContrast(resized)
    const feed = toOcrTensor(resized)
    const outputs = await sessions.ocr.run({ [inputName]: feed })
    const read = decodeOcr(outputs)
    const score = meanProb(read.charProbs)
    if (score > bestScore) {
      bestScore = score
      best = read
    }
  }
  const read = best!
  return { box, read, validation: validate(read.text, read.charProbs) }
}

async function readBox(
  image: ImageDataLike,
  box: Box,
  margin: number,
  sessions: PipelineSessions,
  opts: Pick<PipelineOptions, 'smallBoxMargin' | 'normalizeCrop' | 'rotationSweep'> = {},
): Promise<PlateCandidate> {
  const effMargin = resolveMargin(box, margin, opts.smallBoxMargin)
  if (opts.rotationSweep && opts.rotationSweep.length > 0) {
    return readBoxWithRotationSweep(image, box, effMargin, sessions, opts.rotationSweep, opts.normalizeCrop)
  }
  const expanded = expandBox(box, effMargin, image.width, image.height)
  let cropped = cropResize(image, expanded, OCR_WIDTH, OCR_HEIGHT)
  if (opts.normalizeCrop) cropped = stretchContrast(cropped)
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
      candidates.push(await readBox(image, box, margin, sessions, opts))
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
