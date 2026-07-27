import { defineConfig, type Plugin } from 'vite'
import fs from 'node:fs'
import path from 'node:path'
import { execSync } from 'node:child_process'

// Build identity shown small in the app header: short commit + build date.
// Exists to answer "which build is this phone actually running" — installed
// PWAs lag one launch behind a deploy. 'dev' outside a git checkout (CI and
// local dev both have one).
let commit = 'dev'
try {
  commit = execSync('git rev-parse --short HEAD', { stdio: ['ignore', 'pipe', 'ignore'] }).toString().trim()
} catch {
  /* not a git checkout */
}
const APP_BUILD = `${commit} · ${new Date().toISOString().slice(0, 10)}`

/** Dev-only: serve sample photos + ground truth for /eval.html (both are gitignored, never bundled). */
function serveLocalData(): Plugin {
  return {
    name: 'serve-local-data',
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        const url = (req.url ?? '').split('?')[0]!
        const map: [string, string][] = [
          ['/attachments/', 'attachments'],
          ['/eval-data/', 'eval'],
        ]
        for (const [prefix, dir] of map) {
          if (!url.startsWith(prefix)) continue
          const root = path.join(process.cwd(), dir)
          const p = path.join(root, decodeURIComponent(url.slice(prefix.length)))
          // anchored guard: plain startsWith(root) would pass /attachments-evil/*
          if (p !== root && !p.startsWith(root + path.sep)) {
            res.statusCode = 403
            return res.end()
          }
          if (fs.existsSync(p) && fs.statSync(p).isFile()) {
            res.setHeader('Content-Type', p.endsWith('.json') ? 'application/json' : 'image/jpeg')
            fs.createReadStream(p).pipe(res)
            return
          }
          res.statusCode = 404
          return res.end()
        }
        next()
      })
    },
  }
}

// Configurable base path so the same build serves correctly at the site root
// (local dev/preview) and under a subpath (GitHub Pages project site, e.g.
// https://<user>.github.io/plate-reader/). The deploy workflow sets
// BASE_PATH=/plate-reader/; everything else defaults to '/'.
const BASE_PATH = process.env.BASE_PATH || '/'

// eval.html is the dev-only accuracy scoreboard (reads gitignored
// attachments/ + eval/expected.json via the dev middleware above — neither
// exists in a deployed build). It is only included as a build entry for the
// default root build (used locally); the GitHub Pages build sets BASE_PATH,
// which drops it so the public app ships without the dev scoreboard.
const buildInput: Record<string, string> =
  BASE_PATH === '/' ? { main: 'index.html', eval: 'eval.html' } : { main: 'index.html' }

export default defineConfig({
  base: BASE_PATH,
  define: { __APP_BUILD__: JSON.stringify(APP_BUILD) },
  optimizeDeps: { exclude: ['onnxruntime-web'] },
  plugins: [serveLocalData()],
  server: {
    headers: {
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    },
  },
  build: { rollupOptions: { input: buildInput } },
})
