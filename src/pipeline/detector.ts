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

export function iou(a: Box, b: Box): number {
  const ix = Math.max(0, Math.min(a.x2, b.x2) - Math.max(a.x1, b.x1))
  const iy = Math.max(0, Math.min(a.y2, b.y2) - Math.max(a.y1, b.y1))
  const inter = ix * iy
  const union = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter
  return union > 0 ? inter / union : 0
}

/** Greedy IoU NMS, highest score first. */
export function nms(boxes: Box[], thresh = 0.45): Box[] {
  const sorted = [...boxes].sort((x, y) => y.score - x.score)
  const kept: Box[] = []
  for (const b of sorted) if (kept.every((k) => iou(k, b) < thresh)) kept.push(b)
  return kept
}

/**
 * Overlapping tile grid for native-resolution detection of small plates.
 * The overlap (tile x overlap fraction) must exceed the largest plate the
 * tile pass is responsible for: any object smaller than the overlap that one
 * tile clips falls entirely inside a neighboring tile, so seam-clipped
 * detections can simply be DISCARDED (a half plate can read as a plausible
 * truncated plate — never repair, always rely on the whole copy next door).
 * Returns [] when the image already fits in one tile.
 */
export function tileGrid(w: number, h: number, tile: number, overlap: number): Box[] {
  if (w <= tile && h <= tile) return []
  const step = tile * (1 - overlap)
  const nx = Math.max(1, Math.ceil((w - tile) / step) + 1)
  const ny = Math.max(1, Math.ceil((h - tile) / step) + 1)
  const tiles: Box[] = []
  for (let iy = 0; iy < ny; iy++) {
    for (let ix = 0; ix < nx; ix++) {
      const x1 = Math.min(ix * step, Math.max(0, w - tile))
      const y1 = Math.min(iy * step, Math.max(0, h - tile))
      tiles.push({ x1, y1, x2: Math.min(x1 + tile, w), y2: Math.min(y1 + tile, h), score: 0 })
    }
  }
  return tiles
}

/** Box lies within `px` of any image border. */
export function touchesEdge(b: Box, imgW: number, imgH: number, px = 2): boolean {
  return b.x1 <= px || b.y1 <= px || b.x2 >= imgW - px || b.y2 >= imgH - px
}
