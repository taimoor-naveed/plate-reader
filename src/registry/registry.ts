import { normalizePlateText } from '../pipeline/validate'
import { foldUmlauts } from '../pipeline/rules/de'

/**
 * Owner registry: matches recognized plates against a people list fetched
 * from a user-configured server (the list itself is company data and never
 * ships with this repo — see README "Privacy").
 *
 * Everything here is pure (no DOM, fetch, storage, or clocks) so it runs
 * under the node-only vitest setup; the impure glue lives in
 * src/web/registry-client.ts.
 */

export interface Person {
  name: string
  /** Matrix user id (@user:server) — optional; without it the owner is shown without a message link. */
  matrixId?: string
  plates: string[]
}

export interface PlatesFile {
  version: 1
  people: Person[]
}

/** localStorage record under `plate-reader.platesCache`. */
export interface CachedRegistry {
  /** ms epoch of the successful fetch */
  fetchedAt: number
  file: PlatesFile
}

export type RefreshOutcome =
  | { kind: 'updated'; count: number }
  | { kind: 'no-url' }
  | { kind: 'unreachable' } // network error, DNS, CORS, timeout
  | { kind: 'http-error'; status: number }
  | { kind: 'invalid' } // non-JSON body or schema violation

/**
 * Strict whole-file validation; null on any violation. The file is
 * hand-maintained, so reject loudly rather than salvage partially — the app
 * then keeps its last good list.
 */
export function parsePlatesFile(json: unknown): PlatesFile | null {
  if (typeof json !== 'object' || json === null) return null
  const obj = json as Record<string, unknown>
  if (obj.version !== 1) return null
  if (!Array.isArray(obj.people)) return null
  const people: Person[] = []
  for (const entry of obj.people) {
    if (typeof entry !== 'object' || entry === null) return null
    const p = entry as Record<string, unknown>
    if (typeof p.name !== 'string' || p.name.trim() === '') return null
    if (p.matrixId !== undefined && (typeof p.matrixId !== 'string' || p.matrixId.trim() === '')) return null
    if (!Array.isArray(p.plates) || p.plates.some((x) => typeof x !== 'string')) return null
    const person: Person = { name: p.name.trim(), plates: p.plates as string[] }
    if (typeof p.matrixId === 'string') person.matrixId = p.matrixId.trim()
    people.push(person)
  }
  return { version: 1, people }
}

/** Guarded parse of the localStorage record; null on any corruption. */
export function parseCachedRecord(raw: string | null): CachedRegistry | null {
  if (raw === null) return null
  let json: unknown
  try {
    json = JSON.parse(raw)
  } catch {
    return null
  }
  if (typeof json !== 'object' || json === null) return null
  const obj = json as Record<string, unknown>
  if (typeof obj.fetchedAt !== 'number' || !Number.isFinite(obj.fetchedAt)) return null
  const file = parsePlatesFile(obj.file)
  if (!file) return null
  return { fetchedAt: obj.fetchedAt, file }
}

export const MAX_AGE_MS = 7 * 24 * 60 * 60 * 1000

/** True when there is no cache or it is older than 7 days (exactly 7 days is still fresh). */
export function isStale(fetchedAt: number | undefined, now: number): boolean {
  return fetchedAt === undefined || now - fetchedAt > MAX_AGE_MS
}

/** Input for the gear warning badge: shown unless the state is 'ok'. */
export function listState(hasList: boolean, fetchedAt: number | undefined, now: number): 'ok' | 'stale' | 'missing' {
  if (!hasList) return 'missing'
  return isStale(fetchedAt, now) ? 'stale' : 'ok'
}

/**
 * Canonical comparison key, both sides (same recipe as the eval harness,
 * scripts/eval.ts): uppercase first (foldUmlauts only knows Ä/Ö/Ü — a
 * hand-written "töl…" would otherwise reach normalizePlateText as "Ö" and be
 * stripped), fold umlauts, then drop everything outside A-Z0-9.
 */
export function plateKey(s: string): string {
  return normalizePlateText(foldUmlauts(s.toUpperCase()))
}

export type PlateIndex = Map<string, Person[]>

/**
 * Entries normalizing to '' are skipped. A plate listed for several people
 * (rare, but one car can belong to multiple owners) maps to ALL of them,
 * in file order.
 */
export function buildIndex(file: PlatesFile): PlateIndex {
  const index: PlateIndex = new Map()
  for (const person of file.people) {
    for (const plate of person.plates) {
      const key = plateKey(plate)
      if (key === '') continue
      const owners = index.get(key)
      if (!owners) index.set(key, [person])
      else if (!owners.includes(person)) owners.push(person)
    }
  }
  return index
}

/**
 * Exact match only, deliberately: the UI only shows zero-correction reads
 * (isCertain gate), and a fuzzy hit would name the wrong colleague.
 * Accepts both validation.plate (compact, real umlauts) and user-edited
 * display text ("TÖL AB 123") — both converge on the same key.
 */
export function lookupOwners(index: PlateIndex, text: string): Person[] {
  return index.get(plateKey(text)) ?? []
}

/** matrix.to hands off to the Element app, or shows its own fallback page when Element isn't installed. */
export function matrixToUrl(matrixId: string): string {
  return `https://matrix.to/#/${encodeURIComponent(matrixId)}`
}

/**
 * Every failure between "fetch threw" and "bad status" collapses to
 * 'unreachable': timeout (TimeoutError), abort, network/DNS/CORS (TypeError)
 * are indistinguishable to the user, and the dominant real cause is the same
 * — the server is only reachable on the company network/VPN.
 */
export function classifyFetchError(_e: unknown): RefreshOutcome {
  return { kind: 'unreachable' }
}

const formatDate = (ms: number) =>
  new Date(ms).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })

/**
 * The one place that owns every user-facing refresh message.
 * cachedAt = fetchedAt of the list still in use, or undefined when there is none.
 */
export function refreshMessage(outcome: RefreshOutcome, cachedAt: number | undefined): string {
  const fallback = cachedAt === undefined ? 'No list available yet.' : `Using the list from ${formatDate(cachedAt)}.`
  switch (outcome.kind) {
    case 'updated':
      return `Plates list updated — ${outcome.count} plate${outcome.count === 1 ? '' : 's'}.`
    case 'no-url':
      return 'No plates list configured — set the URL in settings.'
    case 'unreachable':
      return `Couldn't reach the plates server — it's only available on the company network/VPN. ${fallback}`
    case 'http-error':
      return `Plates server error (HTTP ${outcome.status}). ${fallback}`
    case 'invalid':
      return `Plates server sent an invalid list. ${fallback}`
  }
}

/** Status line for the settings panel (#list-info). */
export function listInfo(plateCount: number | undefined, fetchedAt: number | undefined, now: number): string {
  if (plateCount === undefined || fetchedAt === undefined) return 'No list loaded yet.'
  const base = `${plateCount} plate${plateCount === 1 ? '' : 's'} · updated ${formatDate(fetchedAt)}`
  if (!isStale(fetchedAt, now)) return base
  const days = Math.floor((now - fetchedAt) / (24 * 60 * 60 * 1000))
  return `${base} — ${days} days old, update recommended.`
}
