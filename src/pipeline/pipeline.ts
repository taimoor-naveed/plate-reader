import type { Box, ImageDataLike, OcrRead, PlateCandidate } from './types'
import type { OrtSessionLike } from './types'
import { letterbox, cropResize, crop, expandBox, resizeBilinear, stretchContrast, rotateImage } from './image'
import { toDetectorTensor, decodeDetections, DETECTOR_DEFAULT_SIZE, touchesEdge } from './detector'
import { toOcrTensor, decodeOcr, OCR_WIDTH, OCR_HEIGHT } from './ocr'
import { validate } from './validate'
import { isCertain } from './certainty'

export interface PipelineSessions {
  detector: OrtSessionLike
  ocr: OrtSessionLike
  /**
   * Optional second OCR model: reads that fail the certainty gate get one
   * retry with it (see PipelineOptions.escalate). Unused when absent.
   */
  ocrFallback?: OrtSessionLike
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
  /**
   * Deskew fallback (2026-08-10): for reads that fail the certainty gate on
   * a tilt-suspicious box (h/w > 0.7 — an upright plate is ~4.7:1), pick ONE
   * rotation angle by geometry (the angle at which the re-detected box looks
   * most plate-shaped) and OCR once. Never a confidence argmax across angles
   * — that manufactures wrong-at-1.00 reads (docs/eval-results.md Addendum 1).
   * The deskewed read is adopted only if it passes the certainty gate.
   */
  deskew?: boolean
  /**
   * OCR escalation (2026-08-10): reads that fail the certainty gate get one
   * retry with sessions.ocrFallback; adopted only if certain. Never touches
   * a read that already passed the gate.
   */
  escalate?: boolean
}

const SMALL_BOX_WIDTH_PX = 64
/** Boxes with h/w above this are tilt-suspicious (upright plates are wide and flat). */
const TILT_MIN_HW = 0.7
/** Fallbacks only run when the primary read has at least this many chars (skip junk boxes). */
const FALLBACK_MIN_READ = 4
/** Deskew works on a crop downscaled to this max dimension — plenty for a 128x64 OCR input. */
const DESKEW_MAX_DIM = 640

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

/** One detector pass over an image (letterbox -> run -> decode to image coords). */
async function detectImage(image: ImageDataLike, detector: OrtSessionLike, size: number, confThresh: number): Promise<Box[]> {
  const lb = letterbox(image, size)
  const feed = toDetectorTensor(lb.image)
  const inputName = detector.inputNames[0]!
  const outputs = await detector.run({ [inputName]: feed })
  const first = outputs[detector.outputNames[0]!] ?? Object.values(outputs)[0]!
  return decodeDetections(first, lb, image.width, image.height, confThresh)
}

/** Plate-likeness of the best box at one rotation: w/h (capped at 5) x detector score. */
function plateLikeness(boxes: Box[]): { score: number; box: Box | null } {
  let best: Box | null = null
  let bestS = 0
  for (const b of boxes) {
    const wh = Math.min((b.x2 - b.x1) / Math.max(1, b.y2 - b.y1), 5)
    const s = wh * b.score
    if (s > bestS) {
      bestS = s
      best = b
    }
  }
  return { score: bestS, box: best }
}

/** Margin-0 OCR of a box within an image (the shared primitive of all read paths). */
async function ocrCrop(image: ImageDataLike, box: Box, ocr: OrtSessionLike, normalizeCrop?: boolean): Promise<OcrRead> {
  let cropped = cropResize(image, expandBox(box, 0, image.width, image.height), OCR_WIDTH, OCR_HEIGHT)
  if (normalizeCrop) cropped = stretchContrast(cropped)
  const inputName = ocr.inputNames[0]!
  const outputs = await ocr.run({ [inputName]: toOcrTensor(cropped) })
  return decodeOcr(outputs)
}

/**
 * Geometric deskew: downscale the crop (rotation is per-pixel JS — 640px max
 * keeps phones fast without hurting a 128x64 OCR input), sweep coarse angles
 * scoring only re-detected box GEOMETRY, refine +-10 in 5deg steps, then
 * rotate once and OCR the tight re-detected box once. Returns the read, or
 * null when no rotation makes the region look like a plate.
 */
async function deskewAttempt(
  image: ImageDataLike,
  box: Box,
  sessions: PipelineSessions,
  size: number,
  confThresh: number,
): Promise<OcrRead | null> {
  let native = crop(image, expandBox(box, 0.4, image.width, image.height))
  const maxDim = Math.max(native.width, native.height)
  if (maxDim > DESKEW_MAX_DIM) {
    const s = DESKEW_MAX_DIM / maxDim
    native = resizeBilinear(native, Math.max(1, Math.round(native.width * s)), Math.max(1, Math.round(native.height * s)))
  }
  let bestAngle = 0
  let bestScore = 0
  for (let a = -75; a <= 75; a += 15) {
    const { score } = plateLikeness(await detectImage(rotateImage(native, a), sessions.detector, size, confThresh))
    if (score > bestScore) {
      bestScore = score
      bestAngle = a
    }
  }
  if (bestScore === 0) return null
  for (const a of [bestAngle - 10, bestAngle - 5, bestAngle + 5, bestAngle + 10]) {
    if (Math.abs(a) > 80) continue
    const { score } = plateLikeness(await detectImage(rotateImage(native, a), sessions.detector, size, confThresh))
    if (score > bestScore) {
      bestScore = score
      bestAngle = a
    }
  }
  const rotated = rotateImage(native, bestAngle)
  const { box: tight } = plateLikeness(await detectImage(rotated, sessions.detector, size, confThresh))
  // tight box grazing the rotated crop's own border means the plate partially
  // left the working crop — the read would be unreliable, give up instead
  if (!tight || touchesEdge(tight, rotated.width, rotated.height)) return null
  return ocrCrop(rotated, tight, sessions.ocr)
}

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

/**
 * Read one box: primary read, then (only when the primary fails the certainty
 * gate) the opt-in fallbacks. A read that already passed the gate is NEVER
 * touched — fallbacks can only turn a failing read into a passing one, so
 * the levers cannot regress today's correct results.
 */
async function readCandidate(
  image: ImageDataLike,
  box: Box,
  margin: number,
  sessions: PipelineSessions,
  opts: PipelineOptions,
): Promise<PlateCandidate> {
  const size = opts.detectorSize ?? DETECTOR_DEFAULT_SIZE
  const confThresh = opts.confThresh ?? 0.25

  let cand = await readBox(image, box, margin, sessions, opts)

  if (!isCertain(cand) && cand.read.text.length >= FALLBACK_MIN_READ) {
    const w = box.x2 - box.x1
    const h = box.y2 - box.y1
    if (opts.deskew && h / w > TILT_MIN_HW) {
      try {
        const read = await deskewAttempt(image, box, sessions, size, confThresh)
        if (read) {
          const deskewed: PlateCandidate = { box, read, validation: validate(read.text, read.charProbs) }
          if (isCertain(deskewed)) cand = deskewed
        }
      } catch {
        // deskew failed on this crop — keep the primary read
      }
    }
    if (!isCertain(cand) && opts.escalate && sessions.ocrFallback) {
      try {
        const read = await ocrCrop(image, box, sessions.ocrFallback, opts.normalizeCrop)
        const escalated: PlateCandidate = { box, read, validation: validate(read.text, read.charProbs) }
        if (isCertain(escalated)) cand = escalated
      } catch {
        // fallback model failed on this crop — keep the primary read
      }
    }
  }
  return cand
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
  const boxes = await detectImage(image, sessions.detector, size, confThresh)
  const t1 = now()

  // stable order by detector confidence only — carries NO meaning (all plates equal per design)
  const ordered = [...boxes].sort((a, b) => b.score - a.score).slice(0, maxCandidates)
  const candidates: PlateCandidate[] = []
  for (const box of ordered) {
    try {
      candidates.push(await readCandidate(image, box, margin, sessions, opts))
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
