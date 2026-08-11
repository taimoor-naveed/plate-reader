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

/** Box lies within `px` of any image border. */
export function touchesEdge(b: Box, imgW: number, imgH: number, px = 2): boolean {
  return b.x1 <= px || b.y1 <= px || b.x2 >= imgW - px || b.y2 >= imgH - px
}
