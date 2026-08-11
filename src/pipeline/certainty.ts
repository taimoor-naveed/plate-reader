import type { PlateCandidate } from './types'

/**
 * The certainty gate: only candidates passing this are shown to the user at
 * all — boxes and cards alike. A plate the pipeline can't read with certainty
 * is treated as not read; the user retakes the photo instead of
 * second-guessing a maybe-read. The web app, the eval harnesses and the tests
 * all consume this one gate.
 *
 * What "certain" guarantees (user decision 2026-08-11): the CHARACTER STRING
 * is right. It deliberately does NOT guarantee the district grouping: a read
 * like a 3-letter prefix that could split district|letters two ways (DA·H
 * vs D·AH) is still shown — the characters are identical either way, only
 * the displayed spacing embodies a guess. Earlier policy (2026-08-04) hid
 * such ties, plus reads whose district was never issued; that hid perfectly
 * readable plates and was reverted. The matcher still reports `ambiguous`
 * and `districtIssued` in the validation for display/debugging — they just
 * no longer gate.
 *
 * The conditions, and why each exists:
 *
 * - rule 'DE': only German-format reads are ever shown.
 *
 * - zero corrections: validate.ts grants the +0.05 format bonus to
 *   lookalike-corrected matches too, so a corrected read can still reach the
 *   confidence bar — but a read that needed a character swapped is by
 *   definition not a certain read of what's on the plate.
 *
 * - confidence ≥ 0.995, not 1.0: confidence is a mean of per-char
 *   probabilities (+0.05 format bonus, clamped), so 0.995 = "rounds to 100%"
 *   — a strict 1.0 arbitrarily hid a correct 8-char read with a single 94%
 *   character.
 *
 * - region Germany: a second, independent signal — the OCR's country head
 *   must also classify the plate as German. Structure alone can be fooled:
 *   foreign plates one lookalike-correction away from a valid German pattern
 *   read as "certain German" (7 of 108 public mixed-EU scenes; the country
 *   head flagged every one as non-German at prob 1.00). Costs zero correct
 *   reads on the local eval set.
 */
export function isCertain(c: PlateCandidate): boolean {
  return (
    c.validation.rule === 'DE' &&
    c.validation.corrections.length === 0 &&
    c.validation.confidence >= 0.995 &&
    c.read.region === 'Germany'
  )
}
