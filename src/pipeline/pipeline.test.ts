import { describe, it, expect } from 'vitest'
import { extractPlates, resolveMargin } from './pipeline'
import { fakeOcrOutputs } from './ocr.test'
import type { Box, ImageDataLike, OrtSessionLike, TensorLike } from './types'

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

  it('skips candidates whose OCR fails instead of failing the whole run', async () => {
    let calls = 0
    const flaky: OrtSessionLike = {
      inputNames: ['input'],
      outputNames: ['plateHead', 'regionHead'],
      run: async () => {
        if (calls++ === 0) throw new Error('boom')
        return fakeOcrOutputs('BNCR788')
      },
    }
    const rows = [
      [0, 10, 10, 60, 30, 0, 0.9],
      [0, 100, 100, 220, 160, 0, 0.8],
    ]
    const sessions = { detector: fakeDetector(rows), ocr: flaky }
    const res = await extractPlates(image, sessions)
    expect(res.candidates).toHaveLength(1)
    expect(res.candidates[0]!.validation.plate).toBe('BNCR788')
  })
})

describe('resolveMargin', () => {
  const small: Box = { x1: 0, y1: 0, x2: 50, y2: 20, score: 1 } // width 50 < 64
  const large: Box = { x1: 0, y1: 0, x2: 100, y2: 20, score: 1 } // width 100 >= 64

  it('is a no-op when smallBoxMargin is unset', () => {
    expect(resolveMargin(small, 0, false)).toBe(0)
    expect(resolveMargin(small, 0, undefined)).toBe(0)
  })

  it('bumps the margin to at least 0.05 for boxes narrower than 64px', () => {
    expect(resolveMargin(small, 0, true)).toBe(0.05)
    expect(resolveMargin(small, 0.1, true)).toBe(0.1) // already bigger, unchanged
  })

  it('leaves boxes >= 64px wide untouched', () => {
    expect(resolveMargin(large, 0, true)).toBe(0)
  })
})

describe('extractPlates with normalizeCrop', () => {
  it('contrast-stretches the OCR crop when opts.normalizeCrop is set', async () => {
    // low-contrast image: every channel confined to [100,150]
    const w = 384
    const h = 384
    const data = new Uint8ClampedArray(w * h * 4)
    for (let i = 0; i < w * h; i++) {
      const v = 100 + (i % 2) * 50
      data[i * 4] = v
      data[i * 4 + 1] = v
      data[i * 4 + 2] = v
      data[i * 4 + 3] = 255
    }
    const lowContrastImage: ImageDataLike = { data, width: w, height: h }
    let seenMin = 255
    let seenMax = 0
    const ocr: OrtSessionLike = {
      inputNames: ['input'],
      outputNames: ['plateHead', 'regionHead'],
      run: async (feeds) => {
        for (const v of Object.values(feeds)[0]!.data) {
          if (v < seenMin) seenMin = v
          if (v > seenMax) seenMax = v
        }
        return fakeOcrOutputs('BNCR788')
      },
    }
    const sessions = { detector: fakeDetector([[0, 50, 50, 300, 150, 0, 0.9]]), ocr }
    await extractPlates(lowContrastImage, sessions, { normalizeCrop: true })
    expect(seenMin).toBe(0)
    expect(seenMax).toBe(255)
  })
})

describe('extractPlates with rotationSweep', () => {
  it('OCRs the crop at every angle and keeps the read with the highest mean charProb', async () => {
    const image = blank(384, 384)
    let calls = 0
    const ocr: OrtSessionLike = {
      inputNames: ['input'],
      outputNames: ['plateHead', 'regionHead'],
      run: async () => {
        calls++
        // only the 3rd angle probed (index 2, i.e. 0 deg) is a confident read
        return calls === 3 ? fakeOcrOutputs('BNCR788', 0.99) : fakeOcrOutputs('WRONG12', 0.4)
      },
    }
    const sessions = { detector: fakeDetector([[0, 100, 100, 220, 160, 0, 0.9]]), ocr }
    const res = await extractPlates(image, sessions, { rotationSweep: [-10, -5, 0, 5, 10] })
    expect(calls).toBe(5)
    expect(res.candidates).toHaveLength(1)
    expect(res.candidates[0]!.validation.plate).toBe('BNCR788')
  })
})
