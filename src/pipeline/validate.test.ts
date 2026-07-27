import { describe, it, expect } from 'vitest'
import { normalizePlateText, validate } from './validate'
import { matchGerman } from './rules/de'

describe('normalizePlateText', () => {
  it('uppercases and strips separators', () => {
    expect(normalizePlateText('bn-cr 788')).toBe('BNCR788')
    expect(normalizePlateText('KR•LM 144')).toBe('KRLM144')
  })
})

describe('matchGerman', () => {
  it('matches a clean plate with zero corrections', () => {
    const m = matchGerman('BNCR788')!
    expect(m.plate).toBe('BNCR788')
    expect(m.display).toBe('BN CR 788')
    expect(m.corrections).toEqual([])
    expect(m.parts).toEqual({ district: 'BN', letters: 'CR', digits: '788', suffix: '' })
  })

  it('prefers the segmentation with fewest corrections', () => {
    // TOLAB123: TOL|AB|123 needs 0 swaps; TO|LA|B123 would need 1 (B->8)
    const m = matchGerman('TOLAB123')!
    expect(m.corrections).toEqual([])
    expect(m.parts.district).toBe('TOL')
  })

  it('splits by issued district: BXY123 -> B XY 123 (no district "BX")', () => {
    const m = matchGerman('BXY123')!
    expect(m.display).toBe('B XY 123')
    expect(m.parts.district).toBe('B')
  })

  it('prefers the longer district when both splits are issued: DAP151 -> DA P 151', () => {
    // D (Düsseldorf) and DA (Darmstadt) are both issued; the text alone cannot
    // decide, and the longer district is the documented default.
    const m = matchGerman('DAP151')!
    expect(m.display).toBe('DA P 151')
    expect(m.parts).toEqual({ district: 'DA', letters: 'P', digits: '151', suffix: '' })
  })

  it('corrects digit-lookalike in district: 0KXY226 -> OK XY 226', () => {
    const m = matchGerman('0KXY226')!
    expect(m.plate).toBe('OKXY226')
    expect(m.corrections).toEqual([{ pos: 0, from: '0', to: 'O' }])
  })

  it('corrects letter-lookalike in digits: BNCR7B8 -> BN CR 788', () => {
    const m = matchGerman('BNCR7B8')!
    expect(m.plate).toBe('BNCR788')
    expect(m.corrections).toEqual([{ pos: 5, from: 'B', to: '8' }])
  })

  it('handles E/H suffix', () => {
    const m = matchGerman('MXY123E')!
    expect(m.parts).toEqual({ district: 'M', letters: 'XY', digits: '123', suffix: 'E' })
    expect(m.display).toBe('M XY 123E')
  })

  it('rejects leading zero in digits when no segmentation can rescue it', () => {
    // ABCDE012: only fitting segmentation is ABC|DE|012 (leading zero); 'E' blocks all others
    expect(matchGerman('ABCDE012')).toBeNull()
    // NOTE: strings like BNCR0788 legitimately match via another segmentation
    // (BNC|RO|788 with 0->O) — that is correct behavior, not a bug.
  })

  it('caps corrections at 2 (protects EU plates from mangling)', () => {
    // XR25GB (NL-style): any segmentation needs >2 swaps or hits a non-correctable char
    expect(matchGerman('XR25GB')).toBeNull()
  })

  it('rejects too-long and too-short strings', () => {
    expect(matchGerman('BNCRX12345')).toBeNull() // 10 chars
    expect(matchGerman('D1')).toBeNull()
  })

  it('tie-break: equal-correction segmentations keep the shortest district', () => {
    const m = matchGerman('TUXY1234')!
    expect(m.parts.district).toBe('TU')
    expect(m.corrections).toEqual([])
  })
})

describe('validate', () => {
  it('valid German read: formatValid, display, rule DE, confidence = mean + 0.05', () => {
    const v = validate('BNCR788', [0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9])
    expect(v.formatValid).toBe(true)
    expect(v.rule).toBe('DE')
    expect(v.plate).toBe('BNCR788')
    expect(v.display).toBe('BN CR 788')
    expect(v.confidence).toBeCloseTo(0.95)
  })

  it('unrecognized format passes through unchanged (never rejected)', () => {
    const v = validate('AB12CD', [0.8, 0.8, 0.8, 0.8, 0.8, 0.8])
    expect(v.formatValid).toBe(false)
    expect(v.rule).toBeNull()
    expect(v.plate).toBe('AB12CD')
    expect(v.display).toBe('AB12CD')
    expect(v.confidence).toBeCloseTo(0.8)
  })

  it('auto-applies the umlaut district: plate and display use it, raw stays literal', () => {
    const v = validate('TOLAB123', [0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9])
    expect(v.raw).toBe('TOLAB123')
    expect(v.plate).toBe('TÖLAB123')
    expect(v.display).toBe('TÖL AB 123')
  })

  it('leaves plate/display unchanged for districts with no umlaut mapping', () => {
    const v = validate('BNCR788', [0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9])
    expect(v.plate).toBe('BNCR788')
    expect(v.display).toBe('BN CR 788')
  })

  it('does not umlaut-map real issued codes TU/MU', () => {
    expect(validate('TUXY1234', [0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9]).plate).toBe('TUXY1234')
    expect(validate('MUAB123', [0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9]).plate).toBe('MUAB123')
  })
})
