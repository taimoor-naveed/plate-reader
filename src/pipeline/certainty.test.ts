import { describe, it, expect } from 'vitest'
import { isCertain } from './certainty'
import { validate } from './validate'
import type { PlateCandidate } from './types'

const BOX = { x1: 0, y1: 0, x2: 10, y2: 10, score: 1 }

/** Candidate as the pipeline would build it: validation derived from the read. */
function candidate(text: string, opts: { probs?: number[]; region?: string | null } = {}): PlateCandidate {
  const charProbs = opts.probs ?? Array.from(text, () => 1)
  // region: null = "the model has no region head" (OcrRead.region stays unset)
  const region = opts.region === undefined ? 'Germany' : (opts.region ?? undefined)
  return {
    box: BOX,
    read: { text, charProbs, region },
    validation: validate(text, charProbs),
  }
}

describe('isCertain', () => {
  it('passes a clean full-confidence German read', () => {
    expect(isCertain(candidate('BNCR788'))).toBe(true)
  })

  it('passes a clean authority read (HH 07194, M 230)', () => {
    expect(isCertain(candidate('HH07194'))).toBe(true)
    expect(isCertain(candidate('M230'))).toBe(true)
  })

  it('rejects a corrected read even at clamped confidence 1.0', () => {
    // BNCR7B8 needs B->8; the +0.05 format bonus still lifts it to 1.0,
    // which is exactly why confidence alone was an insufficient gate.
    const c = candidate('BNCR7B8')
    expect(c.validation.corrections).toHaveLength(1)
    expect(c.validation.confidence).toBe(1)
    expect(isCertain(c)).toBe(false)
  })

  it('shows ambiguous issued-district ties — the string is certain, only the grouping is a guess (2026-08-11)', () => {
    for (const text of ['BLA1234', 'MAB123', 'DAP151']) {
      const c = candidate(text)
      expect(c.validation.ambiguous).toBe(true) // still reported, no longer gates
      expect(isCertain(c)).toBe(true)
    }
  })

  it('shows the cross-grammar tie AB123 — same characters under either reading (2026-08-11)', () => {
    const c = candidate('AB123')
    expect(c.validation.ambiguous).toBe(true) // still reported, no longer gates
    expect(isCertain(c)).toBe(true)
  })

  it('shows a clean full-confidence read even of a non-issued district (2026-08-11)', () => {
    // user decision: a wrong district guess is acceptable, hiding a
    // confidently-read plate is not — districtIssued is informational now
    const c = candidate('QQ1234')
    expect(c.validation.districtIssued).toBe(false)
    expect(isCertain(c)).toBe(true)
  })

  it('rejects reads below the 0.995 confidence bar', () => {
    expect(isCertain(candidate('BNCR788', { probs: [0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9] }))).toBe(false)
  })

  it('accepts confidence in [0.995, 1): the bar is "rounds to 100%", not a strict 1.0', () => {
    const c = candidate('M230', { probs: [1, 1, 1, 0.79] }) // mean 0.9475 + 0.05 bonus = 0.9975
    expect(c.validation.confidence).toBeGreaterThanOrEqual(0.995)
    expect(c.validation.confidence).toBeLessThan(1)
    expect(isCertain(c)).toBe(true)
  })

  it('rejects reads the country head does not classify as German', () => {
    expect(isCertain(candidate('BNCR788', { region: 'Poland' }))).toBe(false)
    expect(isCertain(candidate('BNCR788', { region: null }))).toBe(false)
  })

  it('rejects reads with no matching rule', () => {
    expect(isCertain(candidate('XR25GB'))).toBe(false)
  })
})
