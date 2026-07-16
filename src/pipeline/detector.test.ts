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
