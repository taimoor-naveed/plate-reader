// Offline service worker: cache-first for same-origin GETs, versioned cache
// name, stale-cache cleanup on activate. Registered in production builds
// only (see src/web/sw-register.ts) — this file is never registered in dev,
// so it never interferes with Vite's dev server / HMR.
//
// The two __PLACEHOLDER__ constants below are stamped by scripts/stamp-sw.mjs
// after every `npm run build` (the build script runs it automatically):
//
// - PRECACHE_MANIFEST becomes the literal list of every same-origin file the
//   built app can request (shell document, hashed JS/CSS/wasm/font under
//   assets/, both ONNX models, manifest, icons), as URLs relative to this
//   script's location. Precaching the complete stamped manifest at install
//   is what makes the app fully offline-capable after a SINGLE online visit:
//   the first page load's own JS/CSS fetches happen before this SW controls
//   the page (so runtime caching alone can never catch them on visit one),
//   but install runs during that same visit and fetches everything itself.
//
// - CACHE_NAME embeds a hash of the *contents* of every precached file, so
//   this file's bytes change exactly when any shipped asset changes. That
//   byte change is what drives updates: the browser re-fetches sw.js on
//   navigation, sees the diff, runs install (precache the new build into the
//   new cache) and activate (delete every older plate-reader-* cache). With
//   a static name neither would ever fire again after the first deploy.
//
// An unstamped copy of this file fails install loudly (the placeholder URL
// 404s in cache.addAll) rather than silently serving a broken cache.
const PRECACHE_MANIFEST = ['__PRECACHE_MANIFEST__']
const CACHE_NAME = 'plate-reader-__BUILD_HASH__'

// Resolve relative to this script's own location so the same file works
// unmodified whether served from '/' (local preview) or '/plate-reader/'
// (GitHub Pages) — never hardcode an absolute base path here. The '' entry
// in the manifest is the app shell document itself (navigations request the
// scope root, not /index.html).
const BASE = new URL('./', self.location.href).pathname
const PRECACHE_URLS = PRECACHE_MANIFEST.map((p) => BASE + p)

// ignoreVary: the onnxruntime-web threaded wasm build re-requests its own
// .mjs/.wasm bootstrap files from multiple contexts (the initial module
// import, then again from each pthread Worker it spins up) with different
// request shapes — some carry an `Origin` header, some don't. Vite's dev/
// preview server sends `Vary: Origin` on every response, so the Cache API's
// default (Vary-aware) matching treats those as different entries and misses
// on the worker-issued fetches even though the URL was already cached. The
// SW only ever caches same-origin requests (the fetch guard below), so
// Vary-based negotiation is never meaningful here — ignoreVary makes matching
// purely URL/method-based, which is what we actually want.
const MATCH_OPTS = { ignoreVary: true }

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches
      .open(CACHE_NAME)
      .then((cache) => cache.addAll(PRECACHE_URLS))
      .then(() => self.skipWaiting()),
  )
})

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches
      .keys()
      // Only reap this app's own caches: GitHub Pages serves every project page
      // from the same origin, so an unscoped cleanup would delete sibling apps'
      // caches too.
      .then((keys) => Promise.all(keys.filter((key) => key.startsWith('plate-reader-') && key !== CACHE_NAME).map((key) => caches.delete(key))))
      .then(() => self.clients.claim()),
  )
})

self.addEventListener('fetch', (event) => {
  const req = event.request
  // Same-origin GET only. The app's single cross-origin request — the
  // user-configured plates-list fetch (src/web/registry-client.ts) — is
  // deliberately excluded: its offline story is localStorage, not the SW
  // cache, and the SW must never start or cache cross-origin requests on the
  // app's behalf. Anything non-same-origin passes through untouched (default
  // browser handling applies).
  if (req.method !== 'GET' || new URL(req.url).origin !== self.location.origin) return

  event.respondWith(
    caches.match(req, MATCH_OPTS).then((cached) => {
      if (cached) return cached
      return fetch(req)
        .then((res) => {
          if (res.ok) {
            const copy = res.clone()
            caches.open(CACHE_NAME).then((cache) => cache.put(req, copy))
          }
          return res
        })
        .catch(() => {
          // Offline and not already cached: for a navigation request, fall
          // back to the precached app shell rather than a hard network error.
          if (req.mode === 'navigate') return caches.match(BASE, MATCH_OPTS)
          return Response.error()
        })
    }),
  )
})
