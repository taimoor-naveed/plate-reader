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

  it('throws when two outputs match the plate head size', () => {
    const outputs = fakeOcrOutputs('BNCR788')
    outputs.duplicate = { ...outputs.plateHead! }
    expect(() => decodeOcr(outputs)).toThrow(/multiple OCR outputs/)
  })
})
