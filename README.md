# ANPR — plate extraction (stage 1)

Reads German license plates from phone photos, fully on-device (PWA, ONNX in WebAssembly).
Photos never leave the device. Spec: docs/superpowers/specs/2026-07-15-plate-extraction-design.md

## Setup

    npm install
    npm run fetch-models   # downloads ONNX models + copies ORT wasm (once)
    npm test               # unit tests
    npm run eval           # accuracy eval over attachments/ (needs eval/expected.json)
    npm run dev            # dev server (LAN-exposed; open on phone via Mac's IP)

## Install & offline

The deployed app lives at https://taimoor-naveed.github.io/plate-reader/
(GitHub Pages — pushes to `main` build and deploy automatically via
`.github/workflows/deploy.yml`; one-time prerequisite: the repo's
Settings → Pages must have **Source: GitHub Actions**). It's a PWA: open
that URL once with a network connection, then:

- **Install it** so it feels like a real app, not a browser tab:
  - **iOS (Safari):** Share → "Add to Home Screen". Launches full-screen,
    no browser chrome (via the `apple-mobile-web-app-capable` /
    `apple-touch-icon` tags in `index.html`).
  - **Android (Chrome):** you'll get an install prompt ("Add to Home
    screen" / "Install app"), driven by `public/manifest.webmanifest`
    (`display: standalone`).
- **Offline after the first visit:** a service worker (`public/sw.js`,
  stamped at build time by `scripts/stamp-sw.mjs` with the complete asset
  list of that exact build) precaches everything the app needs — shell,
  JS/CSS/wasm, both ONNX models (~11 MB), manifest, icons — during the
  first visit. From then on the app works fully offline; a server is only
  needed to pick up updates.
- **Updates:** a new deploy changes `sw.js` (its cache name embeds a hash
  of the build's contents). The update is fetched on the next load *with
  network* and becomes active on the launch after that; old caches are
  cleaned up automatically. Offline launches keep using the cached build
  indefinitely — there is no expiry.
- **iOS cache-eviction caveat:** iOS may evict an installed PWA's cache if
  it goes unused for an extended period (roughly a couple of weeks). If
  offline launch ever fails to load photos/models after a long gap, open
  it once online to restore the cache.

### HTTPS for local install/offline testing

Service workers (and the install prompt) require a secure context — plain
`http://<lan-ip>` on your phone won't do. To test install/offline from
your own machine before relying on the Pages deployment:

    brew install mkcert
    mkcert -install                  # once, trusts a local root CA on this Mac
    mkcert <your-lan-ip>             # e.g. mkcert 192.168.1.23 — writes a cert+key pair

Point Vite's dev/preview server at the generated cert (e.g. via
`server.https` / `preview.https` in `vite.config.ts` with the mkcert
`.pem`/`-key.pem` paths), then on your phone: open `chrome://flags` (or
Safari settings) once to trust mkcert's root CA profile — usually done by
visiting `http://<lan-ip>:<port>/` isn't enough; install the mkcert root
CA on the phone (AirDrop/email the `rootCA.pem` from
`mkcert -CAROOT`, then Settings → General → VPN & Device Management →
install profile → enable full trust for the mkcert cert in Certificate
Trust Settings). After that, `https://<lan-ip>:<port>` on the phone is a
secure context and behaves exactly like the deployed Pages URL.

For a team deployment instead of GitHub Pages: serve `dist/` (static
files only) from any company-internal HTTPS host — no server-side logic
required.

## Privacy

attachments/ (colleague car photos) and eval/expected.json (their plate numbers)
are gitignored — never commit them.

## Status (MVP, 2026-07-16)

Verified on the user's phone over LAN (dev server): model preload, live camera
reads, multi-plate photos, tap-to-select, in-place editing. On-device latency
well within budget. UI iterated live to the minimal app-shell design
(photo + rectangles, authentic plate cards, no page scroll). Remaining known
misses (4/34 labeled plates) are documented in docs/eval-results.md; retake
is the designed recourse.
