/** RGBA pixel buffer; structurally compatible with the DOM's ImageData. */
export interface ImageDataLike {
  data: Uint8ClampedArray
  width: number
  height: number
}

export interface Box {
  x1: number
  y1: number
  x2: number
  y2: number
  score: number
}

export interface TensorLike {
  type: 'float32' | 'uint8'
  data: Float32Array | Uint8Array
  dims: number[]
}

/** Minimal structural interface satisfied by both onnxruntime-web and -node adapters. */
export interface OrtSessionLike {
  inputNames: readonly string[]
  outputNames: readonly string[]
  run(feeds: Record<string, TensorLike>): Promise<Record<string, TensorLike>>
}

export interface OcrRead {
  text: string
  charProbs: number[]
  region?: string
  regionProb?: number
}

export interface Correction {
  pos: number
  from: string
  to: string
}

export interface PlateValidation {
  /** Normalized raw OCR text (A-Z0-9 only). */
  raw: string
  /** Best plate string after corrections (== raw when no rule matched). */
  plate: string
  /** Human display form, e.g. "BN CR 788" (== plate when no rule matched). */
  display: string
  formatValid: boolean
  corrections: Correction[]
  /** Matching rule id, e.g. "DE", or null. */
  rule: string | null
  /** Mean per-char OCR probability, +0.05 if a rule matched (clamped to 1). */
  confidence: number
}

export interface PlateCandidate {
  box: Box
  read: OcrRead
  validation: PlateValidation
}
