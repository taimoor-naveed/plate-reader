import {
  buildIndex,
  classifyFetchError,
  isStale,
  parseCachedRecord,
  parsePlatesFile,
  type CachedRegistry,
  type PlateIndex,
  type PlatesFile,
  type RefreshOutcome,
} from '../registry/registry'

/**
 * Impure shell around src/registry/registry.ts: localStorage + fetch +
 * single-flight refresh. The keys are prefixed because GitHub Pages serves
 * every project page from one origin (same reasoning as the SW cache name,
 * public/sw.js).
 */

const URL_KEY = 'plate-reader.platesUrl'
const CACHE_KEY = 'plate-reader.platesCache'
const FETCH_TIMEOUT_MS = 8000

let index: PlateIndex | null = null
let cachedAt: number | undefined
let inFlight: Promise<RefreshOutcome> | null = null
let onUpdated: (() => void) | null = null

// Hydrate from localStorage at import. Any corruption (or storage access
// being blocked entirely) simply behaves as "no cache yet".
try {
  const record = parseCachedRecord(localStorage.getItem(CACHE_KEY))
  if (record) {
    index = buildIndex(record.file)
    cachedAt = record.fetchedAt
  }
} catch {
  /* storage blocked -> no cache */
}

export function getIndex(): PlateIndex | null {
  return index
}

export function getCachedAt(): number | undefined {
  return cachedAt
}

// In-memory fallback so a blocked localStorage write (quota, private mode)
// degrades to session-only behavior instead of "No plates list configured"
// while the URL sits visibly in the input.
let urlMem: string | null = null

export function getUrl(): string {
  if (urlMem !== null) return urlMem
  try {
    return localStorage.getItem(URL_KEY) ?? ''
  } catch {
    return ''
  }
}

export function setUrl(value: string) {
  const trimmed = value.trim()
  urlMem = trimmed
  try {
    if (trimmed === '') localStorage.removeItem(URL_KEY)
    else localStorage.setItem(URL_KEY, trimmed)
  } catch (e) {
    console.warn('could not persist plates URL', e)
  }
  // Clearing the URL means "remove the list": names and plates must not
  // keep being served from cache after the user withdrew the source.
  if (trimmed === '') {
    index = null
    cachedAt = undefined
    try {
      localStorage.removeItem(CACHE_KEY)
    } catch {
      /* nothing to remove */
    }
    onUpdated?.()
  }
}

/** Called after every successful refresh — app.ts re-renders owner rows in place. */
export function setOnUpdated(cb: () => void) {
  onUpdated = cb
}

/** One fetch of one URL; returns the outcome plus the parsed file on success. */
async function fetchList(url: string): Promise<{ outcome: RefreshOutcome; file?: PlatesFile }> {
  let res: Response
  try {
    res = await fetch(url, { mode: 'cors', cache: 'no-store', signal: AbortSignal.timeout(FETCH_TIMEOUT_MS) })
  } catch (e) {
    return { outcome: classifyFetchError(e) }
  }
  if (!res.ok) return { outcome: { kind: 'http-error', status: res.status } }
  let json: unknown
  try {
    json = await res.json()
  } catch (e) {
    // the shared timeout can fire during the body read — that's the network,
    // not the server's data; only a non-abort failure means a bad body
    // (HTML error page, captive portal, truncated JSON …)
    return { outcome: e instanceof DOMException ? classifyFetchError(e) : { kind: 'invalid' } }
  }
  const file = parsePlatesFile(json)
  if (!file) return { outcome: { kind: 'invalid' } }
  return { outcome: { kind: 'updated', count: 0 }, file } // count filled in on commit
}

async function doRefresh(): Promise<RefreshOutcome> {
  const url = getUrl()
  if (!url) return { kind: 'no-url' }
  const { outcome, file } = await fetchList(url)
  // the configured URL changed while this fetch was in flight: neither the
  // data nor the outcome belongs to the current config — start over
  if (getUrl() !== url) return doRefresh()
  if (!file) return outcome

  index = buildIndex(file)
  cachedAt = Date.now()
  try {
    const record: CachedRegistry = { fetchedAt: cachedAt, file }
    localStorage.setItem(CACHE_KEY, JSON.stringify(record))
  } catch (e) {
    // quota/private mode: the in-memory index still serves this session
    console.warn('could not persist plates cache', e)
  }
  onUpdated?.()
  return { kind: 'updated', count: index.size }
}

/**
 * Single-flight (same pattern as ensureSessions in app.ts): concurrent
 * callers share one in-flight promise. Outcomes are values, never
 * rejections, so the slot is cleared on settle either way.
 */
export function refreshRegistry(): Promise<RefreshOutcome> {
  if (!inFlight) {
    inFlight = doRefresh().finally(() => {
      inFlight = null
    })
  }
  return inFlight
}

/**
 * Launch-time auto-update: null (no fetch, stay silent) while the cached
 * list is fresh; otherwise kicks a refresh — including the no-list and
 * no-URL-yet cases, whose outcomes drive the first-launch notice.
 */
export function maybeAutoRefresh(): Promise<RefreshOutcome> | null {
  if (index && !isStale(cachedAt, Date.now())) return null
  return refreshRegistry()
}
