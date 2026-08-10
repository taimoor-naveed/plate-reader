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
  /**
   * True when equally-plausible issued-district segmentations tie (see
   * RuleMatch.ambiguous in rules/de.ts). Always false when no rule matched.
   */
  ambiguous: boolean
  /**
   * Whether the matched district code is actually issued (umlaut mapping
   * counts: TOL is issued via TÖL). Always false when no rule matched.
   */
  districtIssued: boolean
  /** Matching rule id, e.g. "DE", or null. */
  rule: string | null
  /** Mean per-char OCR probability, +0.05 if a rule matched (clamped to 1). */
  confidence: number
}

export interface PlateCandidate {
  box: Box
  read: OcrRead
  validation: PlateValidation
  /**
   * Box touches the photo border (within a couple of px): the plate may
   * physically continue beyond the frame, so a truncated read can look
   * complete AND format-valid. Certainty treats these reads more strictly.
   */
  frameEdge?: boolean
  /**
   * Tile-pass candidate whose corroboration read (the other OCR model on the
   * same crop) disagreed — or could not run. Never certain.
   */
  uncorroborated?: boolean
}
