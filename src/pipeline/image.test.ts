import { describe, it, expect } from 'vitest'
import { resizeBilinear, crop, expandBox, cropResize, letterbox, stretchContrast, rotateImage } from './image'
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

describe('stretchContrast', () => {
  it('stretches a 100-150 low-contrast range to 0-255 per channel', () => {
    const im = img(2, 1, [
      [100, 100, 100],
      [150, 150, 150],
    ])
    const out = stretchContrast(im)
    expect(px(out, 0, 0)).toEqual([0, 0, 0])
    expect(px(out, 1, 0)).toEqual([255, 255, 255])
  })

  it('leaves a near-flat channel unchanged (range < 10)', () => {
    const im = img(2, 1, [
      [120, 50, 200],
      [125, 55, 205],
    ])
    const out = stretchContrast(im)
    expect([...out.data]).toEqual([...im.data])
  })

  it('does not touch alpha', () => {
    const im = img(2, 1, [
      [100, 100, 100],
      [150, 150, 150],
    ])
    const out = stretchContrast(im)
    expect(out.data[3]).toBe(255)
    expect(out.data[7]).toBe(255)
  })
})

describe('rotateImage', () => {
  it('0 degrees is the identity', () => {
    const src = img(
      4,
      4,
      Array.from({ length: 16 }, (_, i) => [i * 10, i * 10, i * 10]),
    )
    const out = rotateImage(src, 0)
    expect([...out.data]).toEqual([...src.data])
  })

  it('rotates a marker pixel to the expected location and fills exposed corners gray', () => {
    // 6x6, all background 50 except a marker at (x=4, y=3)
    const rgb = Array.from({ length: 36 }, () => [50, 50, 50])
    rgb[3 * 6 + 4] = [200, 200, 200]
    const src = img(6, 6, rgb)
    const out = rotateImage(src, 90)
    // derived from the rotation formula: dst(3,4) samples src(4,3) exactly (marker)
    expect(px(out, 3, 4)).toEqual([200, 200, 200])
    // dst(0,0) samples src(0,6) which is out of bounds -> gray fill
    expect(px(out, 0, 0)).toEqual([114, 114, 114])
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
