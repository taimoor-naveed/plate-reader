import type { PlateCandidate } from './types'

/**
 * The certainty gate (user decision 2026-07-27, tightened 2026-08-04): only
 * candidates passing this are shown to the user at all — boxes and cards
 * alike. A plate the pipeline can't read with certainty is treated as not
 * read; the user retakes the photo instead of second-guessing a maybe-read.
 * The web app, the eval harness, and the tests all consume this one gate.
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
 * - not ambiguous: when several issued-district segmentations tie at equal
 *   corrections — within a grammar (BLA1234: B LA 1234 vs BL A 1234) or
 *   across grammars (AB123: standard A B 123 vs authority AB 123) — the
 *   text alone cannot decide which district is on the plate — only the seal
 *   position could. The matcher still returns its documented best guess for
 *   display/editing flows, but a guess between real alternatives is not a
 *   certain read.
 *
 * - issued district: a reading whose district code was never issued cannot
 *   be what's on a real German plate, no matter how confident the OCR is —
 *   a "certain" read of a nonexistent district is a misread by definition.
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
 *
 * - frame-edge boxes get no format bonus (2026-08-10): a plate cut off by the
 *   photo border yields a truncated read that is itself format-valid (think
 *   "BN CR 788" clipped to a confident "BN CR 7") — structure cannot vouch
 *   for completeness, so for boxes touching the border the RAW mean charProb
 *   must clear the bar alone. A complete plate that merely sits flush against
 *   the border still passes (its raw confidence is ~1.0); the observed
 *   clipped reads all hovered at 0.86-0.96 on the cut character and only
 *   reached the bar via the +0.05 bonus. Residual known hole: a plate clipped
 *   exactly at a character boundary can read 1.0-raw; nothing image-side can
 *   detect that.
 *
 * - corroboration (2026-08-10): tile-pass candidates (small/distant plates
 *   the full-frame detector missed) are a newer, riskier channel — in stress
 *   tests a tile read with a dropped final digit passed the gate at raw
 *   0.958 + bonus. Such candidates carry uncorroborated=true unless a second
 *   OCR model read the same crop and agreed (same length, <=1 char apart);
 *   uncorroborated reads are never certain.
 */
const rawMean = (probs: number[]) => (probs.length ? probs.reduce((a, b) => a + b, 0) / probs.length : 0)

export function isCertain(c: PlateCandidate): boolean {
  if (c.uncorroborated) return false
  if (c.frameEdge && rawMean(c.read.charProbs) < 0.995) return false
  return (
    c.validation.rule === 'DE' &&
    c.validation.corrections.length === 0 &&
    !c.validation.ambiguous &&
    c.validation.districtIssued &&
    c.validation.confidence >= 0.995 &&
    c.read.region === 'Germany'
  )
}
