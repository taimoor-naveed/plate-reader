import type { Correction } from '../types'
import { ISSUED_DISTRICTS } from './de-districts'

export interface RuleMatch {
  plate: string
  display: string
  corrections: Correction[]
  parts: { district: string; letters: string; digits: string; suffix: string }
  /**
   * True when another issued-district segmentation ties the winner at equal
   * corrections with a different reading (BLA1234: B LA 1234 vs BL A 1234).
   * The text alone cannot decide such ties — only the seal position on the
   * physical plate could — so the certainty gate must treat the read as
   * uncertain even though the returned split is the documented best guess.
   */
  ambiguous: boolean
}

/** digit -> letter lookalike (applied in letter positions) */
const D2L: Record<string, string> = { '0': 'O', '1': 'I', '8': 'B', '5': 'S', '2': 'Z' }
/** letter -> digit lookalike (applied in digit positions) */
const L2D: Record<string, string> = { O: '0', I: '1', B: '8', S: '5', Z: '2' }

const MAX_CORRECTIONS = 2
const MAX_TOTAL_LEN = 8

/**
 * OCR reads umlauts in their ASCII base form; Ä/Ö/Ü never reach the matcher.
 * Also exported for ground-truth comparisons (eval): expected values may be
 * written in either form, while validated plates carry real umlauts.
 */
export const foldUmlauts = (s: string) => s.replace(/Ä/g, 'A').replace(/Ö/g, 'O').replace(/Ü/g, 'U')

/**
 * ASCII district -> umlaut district, derived from the issued registry: every
 * umlaut code maps from its ASCII transliteration UNLESS that ASCII form is
 * itself an issued code — then the OCR read is genuinely that other district
 * and rewriting it would be wrong. Exactly two codes are excluded this way:
 * BO (Bochum, so no BÖ suggestion for Börde) and MU (Landkreis München since
 * 2026, so no MÜ suggestion for München-Stadt's old code). Note TU is NOT
 * issued (Tuttlingen is TUT) — TU therefore maps to TÜ (Tübingen).
 */
export const UMLAUT_DISTRICTS: Readonly<Record<string, string>> = (() => {
  const map: Record<string, string> = {}
  for (const code of ISSUED_DISTRICTS) {
    const ascii = foldUmlauts(code)
    if (ascii !== code && !ISSUED_DISTRICTS.has(ascii)) map[ascii] = code
  }
  return map
})()

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

/** Issued check for an ASCII district as read by OCR (TOL counts via TÖL). */
const issuedDistrict = (d: string) => ISSUED_DISTRICTS.has(d) || ISSUED_DISTRICTS.has(UMLAUT_DISTRICTS[d] ?? '')

interface Candidate {
  display: string
  corrections: Correction[]
  parts: { district: string; letters: string; digits: string; suffix: string }
  issued: boolean
  /** Authority form (Behördenkennzeichen): district + digits, no middle letters. */
  authority: boolean
}

/**
 * Preference at equal correction count (the characters are identical across
 * ties; only the segmentation differs):
 * 1. fewer corrections always wins;
 * 2. a district that is actually issued beats one that is not
 *    (fixes BXY123 -> B XY 123: "BX" is not a code);
 * 3. a standard match beats an authority match (authority plates are rare —
 *    they only win on strictly fewer corrections, e.g. M230 -> M 230 over
 *    the corrected M Z 30);
 * 4. when BOTH splits are issued codes (DAP151: D=Düsseldorf, DA=Darmstadt)
 *    the longer district is the documented default — but such ties are
 *    reported via RuleMatch.ambiguous, because the text alone cannot decide;
 * 5. when neither is issued there is no signal: keep the first
 *    (shortest-district) split, as before the registry existed.
 */
function betterThan(x: Candidate, y: Candidate): boolean {
  if (x.corrections.length !== y.corrections.length) return x.corrections.length < y.corrections.length
  if (x.issued !== y.issued) return x.issued
  if (x.authority !== y.authority) return !x.authority
  return x.issued && x.parts.district.length > y.parts.district.length
}

/**
 * Try to interpret `raw` (normalized A-Z0-9) as a German plate.
 * Grammars: standard (district + 1-2 letters + 1-4 digits, no leading zero)
 * and authority/Behörde (district + 1-6 digits, leading zeros allowed),
 * each with an optional E/H suffix. Tries every segmentation; returns the
 * one needing the fewest corrections (max 2), else null.
 */
export function matchGerman(raw: string): RuleMatch | null {
  const n = raw.length
  if (n < 3 || n > MAX_TOTAL_LEN) return null

  const candidates: Candidate[] = []
  for (let a = 1; a <= 3; a++) {
    for (let b = 0; b <= 2; b++) {
      for (const withSuffix of [false, true]) {
        const suffix = withSuffix ? raw[n - 1]! : ''
        if (withSuffix && suffix !== 'E' && suffix !== 'H') continue
        const authority = b === 0
        const digitsEnd = withSuffix ? n - 1 : n
        const digitsLen = digitsEnd - a - b
        if (digitsLen < 1 || digitsLen > (authority ? 6 : 4)) continue

        const corrections: Correction[] = []
        const district = forceClass(raw, 0, a, 'letter', corrections)
        if (district === null) continue
        const letters = forceClass(raw, a, a + b, 'letter', corrections)
        if (letters === null) continue
        const digits = forceClass(raw, a + b, digitsEnd, 'digit', corrections)
        if (digits === null) continue
        // Standard numbers never lead with zero; authority numbers may (HH 07194).
        if (!authority && digits[0] === '0') continue
        if (corrections.length > MAX_CORRECTIONS) continue

        candidates.push({
          display: authority ? `${district} ${digits}${suffix}` : `${district} ${letters} ${digits}${suffix}`,
          corrections,
          parts: { district, letters, digits, suffix },
          issued: issuedDistrict(district),
          authority,
        })
      }
    }
  }

  let best: Candidate | null = null
  for (const c of candidates) if (!best || betterThan(c, best)) best = c
  if (!best) return null

  // Ambiguous iff the winner's equivalence class BEFORE the arbitrary
  // longer-district tie-break (same corrections, issued, same grammar)
  // contains more than one distinct reading — and the districts are issued,
  // so both readings are real possibilities (non-issued ties carry no signal
  // worth surfacing).
  const chosen = best
  const tied = candidates.filter(
    (c) => c.corrections.length === chosen.corrections.length && c.issued === chosen.issued && c.authority === chosen.authority,
  )
  const ambiguous = chosen.issued && new Set(tied.map((c) => c.display)).size > 1

  const { district, letters, digits, suffix } = chosen.parts
  return {
    plate: district + letters + digits + suffix,
    display: chosen.display,
    corrections: chosen.corrections,
    parts: chosen.parts,
    ambiguous,
  }
}
