#!/usr/bin/env node
// Post-build service worker stamping — runs automatically as part of
// `npm run build` (see package.json). Plain Node, no dependencies.
//
// Why this exists (two launch-blocking defects it fixes):
// 1. Offline-after-ONE-visit: the first page load's own JS/CSS/wasm fetches
//    happen before the SW controls the page, so runtime caching can never
//    capture them on visit one. Only an install-time precache of the FULL
//    asset list (including Vite's content-hashed filenames, unknowable until
//    after the build) covers a single-visit install.
// 2. Cache invalidation: browsers only re-run a SW's install/activate when
//    sw.js BYTES change. public/sw.js is copied verbatim, so with a static
//    cache name a new deploy would never invalidate old caches — users would
//    be stuck on the first version forever. Stamping a content hash into
//    CACHE_NAME makes sw.js bytes change exactly when any shipped asset does.
import { createHash } from 'node:crypto'
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const dist = path.join(path.dirname(fileURLToPath(import.meta.url)), '..', 'dist')
if (!fs.existsSync(path.join(dist, 'sw.js'))) {
  console.error('stamp-sw: dist/sw.js not found — run `vite build` first')
  process.exit(1)
}

// Trim the deploy: these ship in public/models/ for the local eval/probe CLIs
// but the built app never requests them — dead weight otherwise. Since
// 2026-08-11 the app loads exactly TWO models: the 384 detector and cct_s
// (the single OCR model). Keep this list in sync with src/web/app.ts —
// trimming a model the app requests breaks the deployed app at startup
// ("failed to load external data", 2026-08-11 incident).
const UNUSED_IN_APP = [
  'models/yolo-v9-t-512-license-plates-end2end.onnx',
  'models/cct_xs_v2_global.onnx',
]
for (const rel of UNUSED_IN_APP) {
  const p = path.join(dist, rel)
  if (fs.existsSync(p)) fs.rmSync(p)
}

// Collect every file in dist/ as URLs relative to sw.js (= dist root).
const files = []
;(function walk(dir) {
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const p = path.join(dir, entry.name)
    if (entry.isDirectory()) walk(p)
    else files.push(path.relative(dist, p).split(path.sep).join('/'))
  }
})(dist)

// Excluded from precache: sw.js itself; eval.html + its chunk (dev-only
// scoreboard — needs the dev middleware's gitignored data, and is absent
// entirely from BASE_PATH builds); index.html is represented by the ''
// entry instead (navigations request the scope root, not /index.html).
const EXCLUDE = /^(sw\.js|eval\.html|assets\/eval-[^/]+\.js|index\.html)$/
const manifest = ['', ...files.filter((f) => !EXCLUDE.test(f)).sort()]

// Version hash covers the CONTENTS of every precached file, not just the URL
// list: models/icons/manifest keep stable (un-hashed) filenames, so a
// URL-only hash would miss content changes to them.
const h = createHash('sha256')
for (const rel of manifest) {
  h.update(rel).update('\0')
  h.update(fs.readFileSync(path.join(dist, rel === '' ? 'index.html' : rel)))
}
const version = h.digest('hex').slice(0, 12)

const swPath = path.join(dist, 'sw.js')
let sw = fs.readFileSync(swPath, 'utf8')
sw = sw.replace("['__PRECACHE_MANIFEST__']", JSON.stringify(manifest))
sw = sw.replace('__BUILD_HASH__', version)
if (sw.includes('__PRECACHE_MANIFEST__') || sw.includes('__BUILD_HASH__')) {
  console.error('stamp-sw: failed to replace placeholders in dist/sw.js')
  process.exit(1)
}
fs.writeFileSync(swPath, sw)
console.log(`stamp-sw: ${manifest.length} precache entries, cache plate-reader-${version}`)
