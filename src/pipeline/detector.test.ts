import { describe, it, expect } from 'vitest'
import { toDetectorTensor, decodeDetections, iou, nms, tileGrid, touchesEdge } from './detector'
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

describe('iou / nms', () => {
  const box = (x1: number, y1: number, x2: number, y2: number, score = 1) => ({ x1, y1, x2, y2, score })

  it('iou: identical boxes -> 1, disjoint -> 0, half overlap computed', () => {
    expect(iou(box(0, 0, 10, 10), box(0, 0, 10, 10))).toBeCloseTo(1)
    expect(iou(box(0, 0, 10, 10), box(20, 20, 30, 30))).toBe(0)
    // [0,10]x[0,10] vs [5,15]x[0,10]: inter 50, union 150
    expect(iou(box(0, 0, 10, 10), box(5, 0, 15, 10))).toBeCloseTo(1 / 3)
  })

  it('nms keeps the higher-score box of an overlapping pair, both of a disjoint pair', () => {
    const kept = nms([box(0, 0, 10, 10, 0.5), box(1, 0, 11, 10, 0.9), box(50, 50, 60, 60, 0.3)])
    expect(kept).toHaveLength(2)
    expect(kept[0]!.score).toBe(0.9)
    expect(kept.map((b) => b.x1).sort((a, b) => a - b)).toEqual([1, 50])
  })
})

describe('tileGrid', () => {
  it('returns no tiles when the image fits in one tile', () => {
    expect(tileGrid(1024, 768, 1024, 0.2)).toEqual([])
  })

  it('covers the full image with overlapping, in-bounds tiles', () => {
    const tiles = tileGrid(4000, 2252, 1024, 0.2)
    expect(tiles.length).toBeGreaterThan(0)
    for (const t of tiles) {
      expect(t.x1).toBeGreaterThanOrEqual(0)
      expect(t.y1).toBeGreaterThanOrEqual(0)
      expect(t.x2).toBeLessThanOrEqual(4000)
      expect(t.y2).toBeLessThanOrEqual(2252)
      expect(t.x2 - t.x1).toBeLessThanOrEqual(1024)
      expect(t.y2 - t.y1).toBeLessThanOrEqual(1024)
    }
    // full coverage: the corners of the image are inside some tile
    expect(Math.min(...tiles.map((t) => t.x1))).toBe(0)
    expect(Math.min(...tiles.map((t) => t.y1))).toBe(0)
    expect(Math.max(...tiles.map((t) => t.x2))).toBe(4000)
    expect(Math.max(...tiles.map((t) => t.y2))).toBe(2252)
    // seam-safety: consecutive column starts advance by at most tile - overlap*tile,
    // so any object smaller than the overlap lands whole in some tile
    const xs = [...new Set(tiles.map((t) => t.x1))].sort((a, b) => a - b)
    for (let i = 1; i < xs.length; i++) expect(xs[i]! - xs[i - 1]!).toBeLessThanOrEqual(1024 * 0.8 + 1e-9)
  })
})

describe('touchesEdge', () => {
  it('detects boxes within 2px of any border, passes interior boxes', () => {
    const box = (x1: number, y1: number, x2: number, y2: number) => ({ x1, y1, x2, y2, score: 1 })
    expect(touchesEdge(box(1, 100, 200, 150), 4000, 3000)).toBe(true) // left
    expect(touchesEdge(box(3900, 100, 3999, 150), 4000, 3000)).toBe(true) // right (within 2px)
    expect(touchesEdge(box(100, 0, 200, 50), 4000, 3000)).toBe(true) // top
    expect(touchesEdge(box(100, 2900, 200, 2999), 4000, 3000)).toBe(true) // bottom
    expect(touchesEdge(box(100, 100, 200, 150), 4000, 3000)).toBe(false)
  })
})
