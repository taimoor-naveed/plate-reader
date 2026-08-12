import { describe, it, expect } from 'vitest'
import {
  parsePlatesFile,
  parseCachedRecord,
  isStale,
  listState,
  plateKey,
  buildIndex,
  lookupOwner,
  matrixToUrl,
  classifyFetchError,
  refreshMessage,
  listInfo,
  MAX_AGE_MS,
  type PlatesFile,
} from './registry'

const file: PlatesFile = {
  version: 1,
  people: [
    { name: 'Jane Doe', matrixId: '@jane:matrix.example.com', plates: ['BN CR 788', 'TÖL AB 123'] },
    { name: 'John NoChat', plates: ['M X 1'] },
  ],
}

describe('parsePlatesFile', () => {
  it('accepts the documented schema', () => {
    expect(parsePlatesFile(JSON.parse(JSON.stringify(file)))).toEqual(file)
  })
  it('tolerates unknown extra fields', () => {
    const json = { version: 1, extra: true, people: [{ name: 'A', plates: [], note: 'blue Golf' }] }
    expect(parsePlatesFile(json)).toEqual({ version: 1, people: [{ name: 'A', plates: [] }] })
  })
  it('trims name and matrixId', () => {
    const parsed = parsePlatesFile({ version: 1, people: [{ name: ' A ', matrixId: ' @a:b.c ', plates: [] }] })
    expect(parsed?.people[0]).toEqual({ name: 'A', matrixId: '@a:b.c', plates: [] })
  })
  it.each([
    ['not an object', 'hi'],
    ['null', null],
    ['wrong version', { version: 2, people: [] }],
    ['missing people', { version: 1 }],
    ['people not array', { version: 1, people: {} }],
    ['person without name', { version: 1, people: [{ plates: [] }] }],
    ['empty name', { version: 1, people: [{ name: '  ', plates: [] }] }],
    ['missing plates', { version: 1, people: [{ name: 'A' }] }],
    ['non-string plate', { version: 1, people: [{ name: 'A', plates: [7] }] }],
    ['non-string matrixId', { version: 1, people: [{ name: 'A', matrixId: 7, plates: [] }] }],
    ['empty matrixId', { version: 1, people: [{ name: 'A', matrixId: ' ', plates: [] }] }],
  ])('rejects %s', (_label, json) => {
    expect(parsePlatesFile(json)).toBeNull()
  })
})

describe('plateKey / buildIndex', () => {
  it('folds umlauts BEFORE stripping (TÖL AB 123 -> TOLAB123, not TLAB123)', () => {
    expect(plateKey('TÖL AB 123')).toBe('TOLAB123')
  })
  it('uppercases before folding so hand-written lowercase umlauts survive', () => {
    expect(plateKey('tölab123')).toBe('TOLAB123')
  })
  it('strips spaces, dashes, dots', () => {
    expect(plateKey('bn-cr.788')).toBe('BNCR788')
  })
  it('maps every plate of a person to the same Person reference', () => {
    const index = buildIndex(file)
    expect(index.get('BNCR788')).toBe(index.get('TOLAB123'))
    expect(index.size).toBe(3)
  })
  it('first person wins on duplicate plates', () => {
    const dup: PlatesFile = {
      version: 1,
      people: [
        { name: 'First', plates: ['B AA 1'] },
        { name: 'Second', plates: ['B-AA-1'] },
      ],
    }
    expect(buildIndex(dup).get('BAA1')?.name).toBe('First')
  })
  it('skips entries that normalize to nothing', () => {
    const junk: PlatesFile = { version: 1, people: [{ name: 'A', plates: ['???', ' '] }] }
    expect(buildIndex(junk).size).toBe(0)
  })
  it('empty file -> empty index', () => {
    expect(buildIndex({ version: 1, people: [] }).size).toBe(0)
  })
})

describe('lookupOwner', () => {
  const index = buildIndex(file)
  it('hits from the compact validated plate (real umlaut)', () => {
    expect(lookupOwner(index, 'TÖLAB123')?.name).toBe('Jane Doe')
  })
  it('hits from an edited display-form value', () => {
    expect(lookupOwner(index, 'TÖL AB 123')?.name).toBe('Jane Doe')
  })
  it('misses on unknown plate', () => {
    expect(lookupOwner(index, 'X YZ 999')).toBeUndefined()
  })
  it('no fuzzy match for a one-char-off plate', () => {
    expect(lookupOwner(index, 'BN CR 789')).toBeUndefined()
  })
})

describe('isStale / listState', () => {
  const now = 1_800_000_000_000
  it('no timestamp -> stale', () => {
    expect(isStale(undefined, now)).toBe(true)
  })
  it('6d23h old -> fresh', () => {
    expect(isStale(now - (MAX_AGE_MS - 60 * 60 * 1000), now)).toBe(false)
  })
  it('exactly 7d old -> still fresh (boundary uses >)', () => {
    expect(isStale(now - MAX_AGE_MS, now)).toBe(false)
  })
  it('7d + 1ms -> stale', () => {
    expect(isStale(now - MAX_AGE_MS - 1, now)).toBe(true)
  })
  it('future timestamp (clock skew) -> fresh', () => {
    expect(isStale(now + 1000, now)).toBe(false)
  })
  it('states: missing, stale, ok', () => {
    expect(listState(false, undefined, now)).toBe('missing')
    expect(listState(true, now - MAX_AGE_MS - 1, now)).toBe('stale')
    expect(listState(true, now, now)).toBe('ok')
  })
})

describe('parseCachedRecord', () => {
  it('roundtrips a valid record', () => {
    const raw = JSON.stringify({ fetchedAt: 123, file })
    expect(parseCachedRecord(raw)).toEqual({ fetchedAt: 123, file })
  })
  it.each([
    ['null (no record)', null],
    ['garbage', '{garbage'],
    ['non-object', '"hi"'],
    ['fetchedAt wrong type', JSON.stringify({ fetchedAt: 'yesterday', file })],
    ['embedded file invalid', JSON.stringify({ fetchedAt: 123, file: { version: 2, people: [] } })],
  ])('null on %s', (_label, raw) => {
    expect(parseCachedRecord(raw)).toBeNull()
  })
})

describe('classifyFetchError', () => {
  it.each([
    ['timeout', new DOMException('timed out', 'TimeoutError')],
    ['abort', new DOMException('aborted', 'AbortError')],
    ['network/CORS', new TypeError('Failed to fetch')],
    ['unknown', 42],
  ])('%s -> unreachable', (_label, err) => {
    expect(classifyFetchError(err)).toEqual({ kind: 'unreachable' })
  })
})

describe('refreshMessage', () => {
  // constructed from local date parts -> formatting is TZ-independent
  const cachedAt = new Date(2026, 7, 10).getTime()
  it('updated, plural and singular', () => {
    expect(refreshMessage({ kind: 'updated', count: 42 }, cachedAt)).toBe('Plates list updated — 42 plates.')
    expect(refreshMessage({ kind: 'updated', count: 1 }, undefined)).toBe('Plates list updated — 1 plate.')
  })
  it('no-url points at settings', () => {
    expect(refreshMessage({ kind: 'no-url' }, undefined)).toBe('No plates list configured — set the URL in settings.')
  })
  it('unreachable with cache names the VPN and the cached date', () => {
    const msg = refreshMessage({ kind: 'unreachable' }, cachedAt)
    expect(msg).toContain('company network/VPN')
    expect(msg).toContain('Using the list from Aug 10, 2026.')
  })
  it('unreachable without cache says no list yet', () => {
    expect(refreshMessage({ kind: 'unreachable' }, undefined)).toContain('No list available yet.')
  })
  it('http-error and invalid always state whether the old list is in use', () => {
    expect(refreshMessage({ kind: 'http-error', status: 500 }, cachedAt)).toBe(
      'Plates server error (HTTP 500). Using the list from Aug 10, 2026.',
    )
    expect(refreshMessage({ kind: 'invalid' }, undefined)).toBe(
      'Plates server sent an invalid list. No list available yet.',
    )
  })
})

describe('listInfo', () => {
  const now = new Date(2026, 7, 12).getTime()
  it('no list yet', () => {
    expect(listInfo(undefined, undefined, now)).toBe('No list loaded yet.')
  })
  it('fresh list', () => {
    expect(listInfo(42, new Date(2026, 7, 10).getTime(), now)).toBe('42 plates · updated Aug 10, 2026')
  })
  it('stale list appends age', () => {
    expect(listInfo(1, new Date(2026, 6, 31).getTime(), now)).toBe(
      '1 plate · updated Jul 31, 2026 — 12 days old, update recommended.',
    )
  })
})

describe('matrixToUrl', () => {
  it('percent-encodes the id', () => {
    expect(matrixToUrl('@jane:matrix.example.com')).toBe('https://matrix.to/#/%40jane%3Amatrix.example.com')
  })
})
