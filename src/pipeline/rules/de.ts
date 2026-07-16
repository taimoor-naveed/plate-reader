import type { Correction } from '../types'

export interface RuleMatch {
  plate: string
  display: string
  corrections: Correction[]
  parts: { district: string; letters: string; digits: string; suffix: string }
}

/** digit -> letter lookalike (applied in letter positions) */
const D2L: Record<string, string> = { '0': 'O', '1': 'I', '8': 'B', '5': 'S', '2': 'Z' }
/** letter -> digit lookalike (applied in digit positions) */
const L2D: Record<string, string> = { O: '0', I: '1', B: '8', S: '5', Z: '2' }

const MAX_CORRECTIONS = 2
const MAX_TOTAL_LEN = 8

/**
 * ASCII district -> umlaut district. Suggestion only — extend freely; an entry
 * must only exist when the ASCII form is NOT itself an issued district code
 * (verify against the official registry before adding).
 * Deliberately absent: TU (= Tuttlingen), MU (= Landkreis München since 2026)
 * — both are real issued codes, so suggesting TÜ/MÜ for them would be wrong.
 */
export const UMLAUT_DISTRICTS: Record<string, string> = {
  TOL: 'TÖL', FU: 'FÜ', GO: 'GÖ', LO: 'LÖ', BUS: 'BÜS',
  SOM: 'SÖM', DUW: 'DÜW', KUN: 'KÜN', SUW: 'SÜW', RUD: 'RÜD', RUG: 'RÜG',
  PLO: 'PLÖ', JUL: 'JÜL', HMU: 'HMÜ', FUS: 'FÜS', MUR: 'MÜR', BUD: 'BÜD',
  FLO: 'FLÖ', UB: 'ÜB', NO: 'NÖ',
}

const isLetter = (c: string) => c >= 'A' && c <= 'Z'
const isDigit = (c: string) => c >= '0' && c <= '9'

/** Force chars in [start,end) to the given class, correcting lookalikes. Returns null if impossible. */
function forceClass(
  raw: string,
  start: number,
  end: number,
  cls: 'letter' | 'digit',
  corrections: Correction[],
): string | null {
  let out = ''
  for (let i = start; i < end; i++) {
    const c = raw[i]!
    if (cls === 'letter' ? isLetter(c) : isDigit(c)) {
      out += c
    } else {
      const sub = cls === 'letter' ? D2L[c] : L2D[c]
      if (!sub) return null
      corrections.push({ pos: i, from: c, to: sub })
      out += sub
    }
  }
  return out
}

/**
 * Try to interpret `raw` (normalized A-Z0-9) as a German plate.
 * Tries every segmentation; returns the one needing the fewest corrections (max 2), else null.
 */
export function matchGerman(raw: string): RuleMatch | null {
  const n = raw.length
  if (n < 3 || n > MAX_TOTAL_LEN) return null

  let best: RuleMatch | null = null
  for (let a = 1; a <= 3; a++) {
    for (let b = 1; b <= 2; b++) {
      for (const withSuffix of [false, true]) {
        const suffix = withSuffix ? raw[n - 1]! : ''
        if (withSuffix && suffix !== 'E' && suffix !== 'H') continue
        const digitsEnd = withSuffix ? n - 1 : n
        const digitsLen = digitsEnd - a - b
        if (digitsLen < 1 || digitsLen > 4) continue

        const corrections: Correction[] = []
        const district = forceClass(raw, 0, a, 'letter', corrections)
        if (district === null) continue
        const letters = forceClass(raw, a, a + b, 'letter', corrections)
        if (letters === null) continue
        const digits = forceClass(raw, a + b, digitsEnd, 'digit', corrections)
        if (digits === null) continue
        if (digits[0] === '0') continue
        if (corrections.length > MAX_CORRECTIONS) continue

        // Tie-break policy: at equal correction count, the FIRST valid
        // segmentation in iteration order (a asc, b asc, no-suffix first)
        // wins — i.e. the shortest district. plate and corrections are
        // identical across such ties; display spacing and parts
        // (district/letters) depend on the chosen split. Deterministic
        // by construction.
        if (!best || corrections.length < best.corrections.length) {
          const plate = district + letters + digits + suffix
          best = {
            plate,
            display: `${district} ${letters} ${digits}${suffix}`,
            corrections,
            parts: { district, letters, digits, suffix },
          }
        }
      }
    }
  }
  return best
}
