import type { ImageDataLike, OcrRead, TensorLike } from './types'

export const OCR_WIDTH = 128
export const OCR_HEIGHT = 64
export const OCR_SLOTS = 10
export const OCR_ALPHABET = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_'
export const OCR_PAD = '_'

/** Region labels of cct v2 global models (verbatim from upstream plate config). */
export const OCR_REGIONS = [
  'Albania', 'Andorra', 'Argentina', 'Armenia', 'Australia', 'Austria', 'Azerbaijan', 'Bahrain',
  'Belarus', 'Belgium', 'Bosnia and Herzegovina', 'Brazil', 'Bulgaria', 'Cambodia', 'Canada', 'Croatia',
  'Cyprus', 'Czech Republic', 'Denmark', 'Estonia', 'Finland', 'France', 'Georgia', 'Germany',
  'Gibraltar', 'Greece', 'Guernsey', 'Hungary', 'Iceland', 'Indonesia', 'Ireland', 'Israel', 'Italy',
  'Latvia', 'Liechtenstein', 'Lithuania', 'Luxembourg', 'Malaysia', 'Malta', 'Mexico', 'Moldova',
  'Monaco', 'Montenegro', 'Netherlands', 'New Zealand', 'North Macedonia', 'Norway', 'Poland',
  'Portugal', 'Qatar', 'Romania', 'San Marino', 'Serbia', 'Singapore', 'Slovakia', 'Slovenia', 'Spain',
  'Sweden', 'Switzerland', 'Thailand', 'Turkey', 'United States', 'Ukraine', 'United Kingdom', 'Vietnam',
  'Unknown',
] as const

/** 128x64 RGBA crop -> [1,64,128,3] uint8 NHWC RGB. Model normalizes internally. */
export function toOcrTensor(cropped: ImageDataLike): TensorLike {
  if (cropped.width !== OCR_WIDTH || cropped.height !== OCR_HEIGHT) {
    throw new Error(`OCR crop must be ${OCR_WIDTH}x${OCR_HEIGHT}, got ${cropped.width}x${cropped.height}`)
  }
  const n = OCR_WIDTH * OCR_HEIGHT
  const data = new Uint8Array(n * 3)
  for (let i = 0; i < n; i++) {
    data[i * 3] = cropped.data[i * 4]!
    data[i * 3 + 1] = cropped.data[i * 4 + 1]!
    data[i * 3 + 2] = cropped.data[i * 4 + 2]!
  }
  return { type: 'uint8', data, dims: [1, OCR_HEIGHT, OCR_WIDTH, 3] }
}

const elemCount = (t: TensorLike) => t.dims.reduce((a, b) => a * b, 1)

/** Decode OCR outputs; heads are identified by element count (names vary between exports). */
export function decodeOcr(outputs: Record<string, TensorLike>): OcrRead {
  let plate: TensorLike | undefined
  let region: TensorLike | undefined
  for (const t of Object.values(outputs)) {
    const n = elemCount(t)
    if (n === OCR_SLOTS * OCR_ALPHABET.length) {
      if (plate) throw new Error('multiple OCR outputs match plate head size (370)')
      plate = t
    } else if (n === OCR_REGIONS.length) {
      if (region) throw new Error('multiple OCR outputs match region head size (66)')
      region = t
    }
  }
  if (!plate) throw new Error('OCR output missing plate head (370 elements)')

  const probs = plate.data as Float32Array
  let text = ''
  const charProbs: number[] = []
  for (let slot = 0; slot < OCR_SLOTS; slot++) {
    let best = 0
    let bestP = -Infinity
    for (let a = 0; a < OCR_ALPHABET.length; a++) {
      const p = probs[slot * OCR_ALPHABET.length + a]!
      if (p > bestP) {
        bestP = p
        best = a
      }
    }
    text += OCR_ALPHABET[best]!
    charProbs.push(bestP)
  }
  // strip trailing pad chars (and their probs)
  while (text.endsWith(OCR_PAD)) {
    text = text.slice(0, -1)
    charProbs.pop()
  }

  const read: OcrRead = { text, charProbs }
  if (region) {
    const rd = region.data as Float32Array
    let best = 0
    for (let i = 1; i < rd.length; i++) if (rd[i]! > rd[best]!) best = i
    read.region = OCR_REGIONS[best]!
    read.regionProb = rd[best]!
  }
  return read
}
