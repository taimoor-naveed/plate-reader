import type { PlateValidation } from './types'
import { matchGerman, UMLAUT_DISTRICTS } from './rules/de'

export function normalizePlateText(s: string): string {
  return s.toUpperCase().replace(/[^A-Z0-9]/g, '')
}

const clamp01 = (x: number) => Math.min(1, Math.max(0, x))

/**
 * Validate an OCR read. Rules score and refine; they NEVER reject.
 * Only rule set today: German ("DE"). EU support = add rule sets here.
 */
export function validate(rawText: string, charProbs: number[]): PlateValidation {
  const raw = normalizePlateText(rawText)
  const meanProb = charProbs.length ? charProbs.reduce((a, b) => a + b, 0) / charProbs.length : 0

  const m = matchGerman(raw)
  if (m) {
    // Auto-apply the umlaut district when it's a known collision-free mapping
    // (UMLAUT_DISTRICTS excludes codes that are themselves real issued districts,
    // e.g. TU/MU — see rules/de.ts). raw is untouched: it is the literal OCR read.
    const district = UMLAUT_DISTRICTS[m.parts.district] ?? m.parts.district
    return {
      raw,
      plate: district + m.parts.letters + m.parts.digits + m.parts.suffix,
      // authority matches (Behördenkennzeichen) have no middle letter group
      display: m.parts.letters
        ? `${district} ${m.parts.letters} ${m.parts.digits}${m.parts.suffix}`
        : `${district} ${m.parts.digits}${m.parts.suffix}`,
      formatValid: true,
      corrections: m.corrections,
      ambiguous: m.ambiguous,
      districtIssued: m.districtIssued,
      rule: 'DE',
      confidence: clamp01(meanProb + 0.05),
    }
  }
  return {
    raw,
    plate: raw,
    display: raw,
    formatValid: false,
    corrections: [],
    ambiguous: false,
    districtIssued: false,
    rule: null,
    confidence: clamp01(meanProb),
  }
}
