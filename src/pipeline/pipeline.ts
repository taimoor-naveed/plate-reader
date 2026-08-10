import type { Box, ImageDataLike, OcrRead, PlateCandidate } from './types'
import type { OrtSessionLike } from './types'
import { letterbox, cropResize, crop, expandBox, resizeBilinear, stretchContrast, rotateImage } from './image'
import { toDetectorTensor, decodeDetections, DETECTOR_DEFAULT_SIZE, iou, nms, tileGrid, touchesEdge } from './detector'
import { toOcrTensor, decodeOcr, OCR_WIDTH, OCR_HEIGHT } from './ocr'
import { validate } from './validate'
import { isCertain } from './certainty'

export interface PipelineSessions {
  detector: OrtSessionLike
  ocr: OrtSessionLike
  /**
   * Larger OCR model (cct_s), used only when a read fails the certainty gate
   * (escalation) and to corroborate tile-pass candidates. Optional: without
   * it escalation is skipped and tile candidates stay uncorroborated (found,
   * boxed, but never certain).
   */
  ocrFallback?: OrtSessionLike
}

export interface TilingOptions {
  /** Tile side in source px (default 1024). */
  tile?: number
  /**
   * Fraction of tile shared between neighbors (default 0.2 -> 205px). Must
   * exceed the largest plate the tile pass must catch — see tileGrid().
   */
  overlap?: number
  /** Max candidates the tile pass may ADD on top of full-frame ones (default 3). */
  addCap?: number
  /** Added boxes wider than this are dropped — full-frame territory (default 400). */
  maxAddedWidth?: number
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
   * Tiled detection (2026-08-10): also run the detector over overlapping
   * native-resolution tiles so small/distant plates survive the 384px
   * letterbox (a 150px plate in a 4000px photo is ~14px in the detector
   * input — invisible). STRICTLY ADDITIVE: full-frame boxes always win; tile
   * boxes are only added where they overlap no full-frame box.
   */
  tiling?: boolean | TilingOptions
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
   * retry with sessions.ocrFallback (cct_s); adopted only if certain. Never
   * touches a read that already passed the gate.
   */
  escalate?: boolean
}

const SMALL_BOX_WIDTH_PX = 64
const TILE_SIZE = 1024
const TILE_OVERLAP = 0.2
const TILE_ADD_CAP = 3
const TILE_MAX_ADDED_WIDTH = 400
/** Detections within this many px of an INTERIOR tile edge are discarded (seam clips). */
const TILE_BORDER_SKIP = 6
/** A tile box overlapping any full-frame box at or above this IoU is a duplicate, not an addition. */
const TILE_ADD_IOU = 0.3
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

/**
 * Tile pass: detect over overlapping native-res tiles, discard seam-clipped
 * detections (the overlap guarantees a whole copy exists in a neighbor tile),
 * NMS-merge, then keep only boxes that ADD to the full-frame result: no
 * overlap with any full-frame box, not touching the photo border (a clipped
 * plate can never be certain — don't spend a slot on it), and small enough
 * to be genuine tile-pass territory.
 */
async function detectAddedTileBoxes(
  image: ImageDataLike,
  detector: OrtSessionLike,
  size: number,
  confThresh: number,
  fullFrame: Box[],
  topts: TilingOptions,
): Promise<Box[]> {
  const tile = topts.tile ?? TILE_SIZE
  const overlap = topts.overlap ?? TILE_OVERLAP
  const tiles = tileGrid(image.width, image.height, tile, overlap)
  let tileBoxes: Box[] = []
  for (const t of tiles) {
    const src = crop(image, t)
    const ox = Math.round(Math.max(0, Math.min(t.x1, image.width - 1)))
    const oy = Math.round(Math.max(0, Math.min(t.y1, image.height - 1)))
    for (const b of await detectImage(src, detector, size, confThresh)) {
      // seam-clipped? (interior tile edges only — the photo border is a real boundary)
      const clipL = b.x1 < TILE_BORDER_SKIP && ox > 0
      const clipT = b.y1 < TILE_BORDER_SKIP && oy > 0
      const clipR = b.x2 > src.width - TILE_BORDER_SKIP && ox + src.width < image.width
      const clipB = b.y2 > src.height - TILE_BORDER_SKIP && oy + src.height < image.height
      if (clipL || clipT || clipR || clipB) continue
      tileBoxes.push({ x1: b.x1 + ox, y1: b.y1 + oy, x2: b.x2 + ox, y2: b.y2 + oy, score: b.score })
    }
  }
  return nms(tileBoxes)
    .filter((t) => fullFrame.every((f) => iou(f, t) < TILE_ADD_IOU))
    .filter((t) => !touchesEdge(t, image.width, image.height))
    .filter((t) => t.x2 - t.x1 <= (topts.maxAddedWidth ?? TILE_MAX_ADDED_WIDTH))
    .slice(0, topts.addCap ?? TILE_ADD_CAP)
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
 * rotate once and OCR the tight re-detected box once. Returns the read and
 * the crop it came from (for corroboration), or null when no rotation makes
 * the region look like a plate.
 */
async function deskewAttempt(
  image: ImageDataLike,
  box: Box,
  sessions: PipelineSessions,
  size: number,
  confThresh: number,
): Promise<{ read: OcrRead; cropImage: ImageDataLike; cropBox: Box } | null> {
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
  if (!tight || touchesEdge(tight, rotated.width, rotated.height)) return null
  const read = await ocrCrop(rotated, tight, sessions.ocr)
  return { read, cropImage: rotated, cropBox: tight }
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
 * the levers cannot regress today's correct results. Tile-pass candidates
 * that end up certain additionally need the other OCR model to corroborate
 * the text on the same crop (same length, <=1 char apart) — see certainty.ts.
 */
async function readCandidate(
  image: ImageDataLike,
  box: Box,
  margin: number,
  sessions: PipelineSessions,
  opts: PipelineOptions,
  fromTile: boolean,
): Promise<PlateCandidate> {
  const size = opts.detectorSize ?? DETECTOR_DEFAULT_SIZE
  const confThresh = opts.confThresh ?? 0.25
  const frameEdge = touchesEdge(box, image.width, image.height)

  let cand = await readBox(image, box, margin, sessions, opts)
  cand.frameEdge = frameEdge
  // the crop behind cand's read, for corroboration (margin-0 default path)
  let cropImage = image
  let cropBox = box
  let usedFallbackModel = false

  const fallbackEligible = !frameEdge && cand.read.text.length >= FALLBACK_MIN_READ
  if (!isCertain(cand) && fallbackEligible) {
    const w = box.x2 - box.x1
    const h = box.y2 - box.y1
    let deskewCrop: { image: ImageDataLike; box: Box } | null = null
    if (opts.deskew && h / w > TILT_MIN_HW) {
      const r = await deskewAttempt(image, box, sessions, size, confThresh)
      if (r) {
        deskewCrop = { image: r.cropImage, box: r.cropBox }
        const deskewed: PlateCandidate = { box, read: r.read, validation: validate(r.read.text, r.read.charProbs), frameEdge }
        if (isCertain(deskewed)) {
          cand = deskewed
          cropImage = r.cropImage
          cropBox = r.cropBox
        }
      }
    }
    if (!isCertain(cand) && opts.escalate && sessions.ocrFallback) {
      // best crop available: the deskewed one if the sweep found a plate, else the original
      const src = deskewCrop ?? { image, box }
      try {
        const read = await ocrCrop(src.image, src.box, sessions.ocrFallback, opts.normalizeCrop)
        const escalated: PlateCandidate = { box, read, validation: validate(read.text, read.charProbs), frameEdge }
        if (isCertain(escalated)) {
          cand = escalated
          cropImage = src.image
          cropBox = src.box
          usedFallbackModel = true
        }
      } catch {
        // fallback model failed on this crop — keep the primary read
      }
    }
  }

  if (fromTile && isCertain(cand)) {
    cand.uncorroborated = !(await corroborate(cand, cropImage, cropBox, usedFallbackModel, sessions, opts))
  }
  return cand
}

/** Second-model read of the same crop; corroborates iff same length and <=1 char differs. */
async function corroborate(
  cand: PlateCandidate,
  cropImage: ImageDataLike,
  cropBox: Box,
  usedFallbackModel: boolean,
  sessions: PipelineSessions,
  opts: PipelineOptions,
): Promise<boolean> {
  const partner = usedFallbackModel ? sessions.ocr : sessions.ocrFallback
  if (!partner) return false
  try {
    const p = await ocrCrop(cropImage, cropBox, partner, opts.normalizeCrop)
    const a = cand.read.text
    const b = p.text
    if (a.length !== b.length) return false
    let diff = 0
    for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) diff++
    return diff <= 1
  } catch {
    return false
  }
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
  // stable order by detector confidence only — carries NO meaning (all plates equal per design)
  const ordered = [...boxes].sort((a, b) => b.score - a.score).slice(0, maxCandidates)

  let addedBoxes: Box[] = []
  if (opts.tiling) {
    const topts = typeof opts.tiling === 'object' ? opts.tiling : {}
    addedBoxes = await detectAddedTileBoxes(image, sessions.detector, size, confThresh, ordered, topts)
  }
  const t1 = now()

  const candidates: PlateCandidate[] = []
  for (const [i, box] of [...ordered, ...addedBoxes].entries()) {
    try {
      candidates.push(await readCandidate(image, box, margin, sessions, opts, i >= ordered.length))
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
