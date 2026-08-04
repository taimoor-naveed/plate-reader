import { describe, it, expect } from 'vitest'
import { matchGerman, UMLAUT_DISTRICTS, foldUmlauts } from './de'
import { ISSUED_DISTRICTS } from './de-districts'

describe('authority plates (Behördenkennzeichen: district + digits, no middle letters)', () => {
  it('HH07194 -> HH 07194 with zero corrections (regression: was "HH O 7194" via 0->O)', () => {
    const m = matchGerman('HH07194')!
    expect(m.display).toBe('HH 07194')
    expect(m.parts).toEqual({ district: 'HH', letters: '', digits: '07194', suffix: '' })
    expect(m.corrections).toEqual([])
    expect(m.ambiguous).toBe(false)
  })

  it('M230 -> M 230 with zero corrections (regression: was "M Z 30" via 2->Z)', () => {
    const m = matchGerman('M230')!
    expect(m.display).toBe('M 230')
    expect(m.parts).toEqual({ district: 'M', letters: '', digits: '230', suffix: '' })
    expect(m.corrections).toEqual([])
    expect(m.ambiguous).toBe(false)
  })

  it('allows leading zeros in the authority number only', () => {
    // standard grammar still rejects a leading zero: no segmentation of
    // ABCDE012 survives, exactly as before
    expect(matchGerman('ABCDE012')).toBeNull()
  })

  it('loses the tie to an uncorrected standard match: AB123 -> A B 123', () => {
    // A|B|123 (standard) and AB|123 (authority, AB = Aschaffenburg) both need
    // zero corrections; standard plates are vastly more common, so authority
    // only wins on strictly fewer corrections — and the resolved tie is not
    // reported as ambiguous.
    const m = matchGerman('AB123')!
    expect(m.display).toBe('A B 123')
    expect(m.ambiguous).toBe(false)
  })

  it('an issued authority district beats non-issued standard splits: QLB123 -> QLB 123', () => {
    // QLB (Quedlinburg) is issued; neither Q nor QL is — issued-ness outranks
    // the standard-over-authority preference.
    const m = matchGerman('QLB123')!
    expect(m.display).toBe('QLB 123')
    expect(m.parts.district).toBe('QLB')
    expect(m.corrections).toEqual([])
  })
})

describe('ambiguous issued-district ties (RuleMatch.ambiguous)', () => {
  it('flags BLA1234 (B LA 1234 vs BL A 1234)', () => {
    const m = matchGerman('BLA1234')!
    expect(m.display).toBe('BL A 1234') // longer district is still the returned default
    expect(m.ambiguous).toBe(true)
  })

  it('flags MAB123 (M AB 123 vs MA B 123)', () => {
    const m = matchGerman('MAB123')!
    expect(m.display).toBe('MA B 123')
    expect(m.ambiguous).toBe(true)
  })

  it('flags DAP151 (D AP 151 vs DA P 151)', () => {
    const m = matchGerman('DAP151')!
    expect(m.display).toBe('DA P 151')
    expect(m.ambiguous).toBe(true)
  })

  it('does not flag reads where only one split has an issued district', () => {
    expect(matchGerman('BXY123')!.ambiguous).toBe(false) // "BX" is not a code
    expect(matchGerman('BNCR788')!.ambiguous).toBe(false)
    expect(matchGerman('TUXY1234')!.ambiguous).toBe(false) // TU (via TÜ) issued, TUX not
  })
})

describe('UMLAUT_DISTRICTS derivation from the issued registry', () => {
  it('maps exactly the umlaut codes whose ASCII form is not itself issued', () => {
    expect(UMLAUT_DISTRICTS).toEqual({
      AO: 'AÖ', BUD: 'BÜD', BUR: 'BÜR', BUS: 'BÜS', BUZ: 'BÜZ',
      DUW: 'DÜW', FLO: 'FLÖ', FU: 'FÜ', FUS: 'FÜS', GO: 'GÖ',
      GU: 'GÜ', HMU: 'HMÜ', HOS: 'HÖS', JUL: 'JÜL', KON: 'KÖN',
      KOT: 'KÖT', KOZ: 'KÖZ', KUN: 'KÜN', LO: 'LÖ', LOB: 'LÖB',
      LUN: 'LÜN', MUB: 'MÜB', MUL: 'MÜL', MUR: 'MÜR', NO: 'NÖ',
      OHR: 'ÖHR', PLO: 'PLÖ', PRU: 'PRÜ', RUD: 'RÜD', RUG: 'RÜG',
      SAK: 'SÄK', SLU: 'SLÜ', SMU: 'SMÜ', SOM: 'SÖM', SUW: 'SÜW',
      TOL: 'TÖL', TU: 'TÜ', UB: 'ÜB', WU: 'WÜ', WUM: 'WÜM',
    })
  })

  it('excludes exactly the codes whose ASCII form is itself issued (BO, MU)', () => {
    expect(UMLAUT_DISTRICTS).not.toHaveProperty('BO') // BO = Bochum, so never BÖ
    expect(UMLAUT_DISTRICTS).not.toHaveProperty('MU') // MU = Lkr. München, so never MÜ
    expect(ISSUED_DISTRICTS.has('BO')).toBe(true)
    expect(ISSUED_DISTRICTS.has('MU')).toBe(true)
  })

  it('covers every umlaut code in the registry (mapped, or excluded for a real ASCII collision)', () => {
    for (const code of ISSUED_DISTRICTS) {
      const ascii = foldUmlauts(code)
      if (ascii === code) continue
      if (ISSUED_DISTRICTS.has(ascii)) expect(UMLAUT_DISTRICTS).not.toHaveProperty(ascii)
      else expect(UMLAUT_DISTRICTS[ascii]).toBe(code)
    }
  })
})
